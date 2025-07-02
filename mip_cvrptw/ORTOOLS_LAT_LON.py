from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    
    # Load location and vehicle data
    orders_df = pd.read_csv("C:\\Users\\Acer\\Downloads\\order_data_lat_lon_100.csv")
    vehicles_df = pd.read_csv("C:\\Users\\Acer\\Downloads\\VEHICLE_DATA_LAT_LON_100.csv")
    
    num_locations = len(orders_df) + 1  # Including depot
    num_vehicles = len(vehicles_df)
    
    # Distance matrix
    distance_matrix = np.zeros((num_locations, num_locations))
    time_matrix = np.zeros((num_locations, num_locations))
    
    # Depot at index 0
    locations = [(52.506885, -1.728302)] + list(zip(orders_df['lat'], orders_df['long']))
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return int(R * c)

    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                distance_matrix[i][j] = haversine(locations[i][0], locations[i][1], locations[j][0], locations[j][1])
                time_matrix[i][j] = distance_matrix[i][j] / 30 * 60  # Assuming 30 km/h speed
    
    data['distance_matrix'] = distance_matrix.astype(int).tolist()
    data['time_matrix'] = time_matrix.astype(int).tolist()
    
    data['vehicle_capacities'] = vehicles_df['Max Weight Kg'].tolist()
    data['variable_costs'] = vehicles_df['Per Km Cost'].tolist()
    data['fixed_costs'] = vehicles_df['Fixed Cost'].tolist()
    data['num_vehicles'] = num_vehicles
    data['depot'] = 0
    
    orders_df['start_time'] = pd.to_datetime(orders_df['start_time'], format="%H:%M").dt.hour * 60
    orders_df['end_time'] = pd.to_datetime(orders_df['end_time'], format="%H:%M").dt.hour * 60
    
    data['demands'] = [0] + orders_df['weight_kg'].tolist()
    data['service_times'] = [0] + [0 for _ in range(len(orders_df))]
    data['time_windows'] = [(600, 1380)] + list(zip(orders_df['start_time'], orders_df['end_time']))
    
    return data

def solve_cvrptw():
    """Solves the CVRPTW using OR-Tools."""
    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.AddDimension(transit_callback_index, 0, 10000, True, "Distance")
    
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')
    
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node] + data['service_times'][from_node]
    
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_callback_index, 300, 1380, False, 'Time')
    time_dimension = routing.GetDimensionOrDie('Time')
    
    for node_idx, (start, end) in enumerate(data['time_windows']):
        index = manager.NodeToIndex(node_idx)
        time_dimension.CumulVar(index).SetRange(start, end)
    
    def cost_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        vehicle_var = routing.VehicleVar(from_index)
        if not vehicle_var.Bound():
            return 100000  # Large penalty for unbound vehicle
        vehicle_id = vehicle_var.Min()
        return data['fixed_costs'][vehicle_id] + (data['variable_costs'][vehicle_id] * data['distance_matrix'][from_node][to_node])
    
    cost_callback_index = routing.RegisterTransitCallback(cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_callback_index)
    
    for node in range(len(data['distance_matrix'])):
        routing.solver().Add(routing.VehicleVar(node) >= 0)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 10
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        print_solution(data, manager, routing, solution)
    else:
        print("No solution found.")

def print_solution(data, manager, routing, solution):
    """Prints the solution with correct cost calculation and excludes unused vehicles."""
    total_distance, total_cost = 0, 0
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_distance, vehicle_cost,route_time = 0, 0, 600

        route = []
        delivery_times = []
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += data['distance_matrix'][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            route_time += data['time_matrix'][manager.IndexToNode(previous_index)][manager.IndexToNode(index)] + data['service_times'][node_index]
            delivery_times.append(route_time)
        if len(route) > 2:  # Ensure the vehicle has a delivery assignment
            route.append(manager.IndexToNode(index))
            vehicle_cost += route_distance * data['variable_costs'][vehicle_id] + data['fixed_costs'][vehicle_id]
            total_distance += route_distance
            total_cost += vehicle_cost
            
            print(f'Route for vehicle {vehicle_id}: {" -> ".join(map(str, route))}')
            print(f'Delivery times: {delivery_times}')
            print(f'Distance: {route_distance} km | Fixed Cost: {data["fixed_costs"][vehicle_id]} | Variable Cost: {route_distance * data["variable_costs"][vehicle_id]:.2f} | Total Cost: {vehicle_cost:.2f}\n')
    
    print(f'Total distance: {total_distance} km, Total cost: {total_cost:.2f}')


if __name__ == '__main__':
    solve_cvrptw()
