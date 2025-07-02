from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import pandas as pd
import numpy as np
import csv

# Read distance matrix
file_path = 'C:\\Users\\Acer\\Downloads\\JSON_extracted_distance_matrix.csv'
distance_matrix_1 = []
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        distance_matrix_1.append([float(cell) for cell in row])

# Read time matrix
file_path_time = 'C:\\Users\\Acer\\Downloads\\JSON_extracted_time_matrix.csv'
time_matrix_1 = []
with open(file_path_time, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        time_matrix_1.append([float(cell) for cell in row])

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    
    # Load location and vehicle data
    orders_df = pd.read_csv("C:\\Users\\Acer\\Downloads\\Test_cases\\customer_uae_data.csv")
    vehicles_df = pd.read_csv("C:\\Users\\Acer\\Downloads\\Test_cases\\trucks_updated.csv")
    
    num_locations = len(orders_df)
    num_vehicles = len(vehicles_df)
    
    # Distance and time matrices
    data['distance_matrix'] = distance_matrix_1
    data['time_matrix'] = time_matrix_1
    
    # Vehicle capacities and costs
    data['vehicle_capacities'] = vehicles_df['capacity_kg'].tolist()
    data['vehicle_volume'] = vehicles_df['max_volume'].tolist()
    data['variable_costs'] = vehicles_df['Per Km Cost'].tolist()
    data['fixed_costs'] = vehicles_df['Fixed Cost'].tolist()
    data['num_vehicles'] = num_vehicles
    
    # Depot handling - depots at indices 30, 31, 32
    data['depots'] = [30, 31, 32]
    
    # Assign vehicles to depots in round-robin fashion
    data['starts'] = [30, 31, 32] * ((num_vehicles // 3) + 1)
    data['starts'] = data['starts'][:num_vehicles]
    data['ends'] = data['starts']  # Vehicles return to same depot
    
    # Customer demands and time windows
    data['demands_weight'] = orders_df['weight'].tolist()
    data['demands_volume'] = orders_df['volume'].tolist()
    data['time_windows'] = list(zip(orders_df['tw_early'], orders_df['tw_late']))
    
    return data

def solve_cvrptw():
    """Solves the CVRPTW using OR-Tools."""
    data = create_data_model()
    
    # Create routing index manager
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']),
        data['num_vehicles'],
        data['starts'],
        data['ends']
    )
    
    routing = pywrapcp.RoutingModel(manager)
    
    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # Add distance dimension
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        100000,  # maximum distance
        True,  # start cumul to zero
        'Distance')
    
    # Weight capacity constraint
    def weight_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands_weight'][from_node]
    
    weight_callback_index = routing.RegisterUnaryTransitCallback(weight_callback)
    routing.AddDimensionWithVehicleCapacity(
        weight_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],
        True,  # start cumul to zero
        'Weight')
    
    # Volume capacity constraint
    def volume_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands_volume'][from_node]
    
    volume_callback_index = routing.RegisterUnaryTransitCallback(volume_callback)
    routing.AddDimensionWithVehicleCapacity(
        volume_callback_index,
        0,  # null capacity slack
        data['vehicle_volume'],
        True,  # start cumul to zero
        'Volume')
    
    # Time window constraint
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]
    
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        30,  
        14400,  
        False,  
        'Time')
    
    time_dimension = routing.GetDimensionOrDie('Time')
    for location_idx, (start, end) in enumerate(data['time_windows']):
        print(f"Location {location_idx}: Time window ({start}, {end})")
        if location_idx not in data['depots']:  # Skip depots
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(start, end)

# Adjust the time dimension parameters (values should match your problem scale)
    routing.AddDimension(
        time_callback_index,
        60,  # allow some slack (adjust based on your needs)
        86400,  # maximum route duration (24 hours in seconds)
        False,  # don't force start cumul to zero
        'Time')
    
    
    # Set cost function
    def cost_callback(from_index, to_index):
        """Returns the cost of traversing the arc between two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        distance = data['distance_matrix'][from_node][to_node]
        
        # Get vehicle index
        vehicle_var = routing.VehicleVar(from_index)
        if vehicle_var.Bound():
            vehicle_idx = vehicle_var.Value()
            return int(data['fixed_costs'][vehicle_idx]) + data['variable_costs'][vehicle_idx] * distance
        return 100000  # Large penalty for unassigned
    
    cost_callback_index = routing.RegisterTransitCallback(cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_callback_index)
    
    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
    search_parameters.time_limit.seconds = 300
    search_parameters.log_search = True
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        print_solution(data, manager, routing, solution)
    else:
        print("No solution found.")

def print_solution(data, manager, routing, solution):
    """Prints solution on console with correct distance and time calculations."""
    total_distance = 0
    total_cost = 0
    total_time = 0
    time_dimension = routing.GetDimensionOrDie('Time')
    distance_dimension = routing.GetDimensionOrDie('Distance')
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'Route for vehicle {vehicle_id} (Depot {manager.IndexToNode(index)}):\n'
        route_distance = 0
        route_load_weight = 0
        route_load_volume = 0
        route_time = 0
        first_node = True
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            # Get actual distance traveled between nodes
            segment_distance = data['distance_matrix'][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            route_distance += segment_distance
            
            # Get actual time taken between nodes
            segment_time = data['time_matrix'][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            route_time += segment_time
            plan_output += (
                f'{node_index} '
                f'Time({route_time}) '
                f'Weight({data["demands_weight"][node_index]}) '
                f'Volume({data["demands_volume"][node_index]}) -> ')
            if not first_node:  # Don't count depot's demand
                route_load_weight += data['demands_weight'][node_index]
                route_load_volume += data['demands_volume'][node_index]
            first_node = False
        
        # Add the last node (depot return)
        node_index = manager.IndexToNode(index)
        plan_output += (
            f'{node_index} ')
        
        
        plan_output += (
            f'Distance: {route_distance:.2f}m '
            f'Weight: {route_load_weight:.2f}kg '
            f'Volume: {route_load_volume:.2f}m³ '
            f'Time: {route_time:.2f}min\n')
        
        # Calculate route cost
        route_cost = data['fixed_costs'][vehicle_id] + (route_distance * data['variable_costs'][vehicle_id])
        plan_output += f'Route cost: {route_cost:.2f}\n'
        
        # Only count vehicles that actually have deliveries
        if route_load_weight > 0 or route_load_volume > 0:
            total_distance += route_distance
            total_cost += route_cost
            total_time += route_time
            print(plan_output)
    
    print(f'\nSummary:')
    print(f'Total distance for all vehicles: {total_distance:.2f}km')
    print(f'Total time for all vehicles: {total_time:.2f}min')
    print(f'Total cost for all vehicles: {total_cost:.2f}')
if __name__ == '__main__':
    solve_cvrptw()


