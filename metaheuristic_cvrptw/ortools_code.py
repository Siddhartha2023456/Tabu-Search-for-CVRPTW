from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np

# Load and preprocess data
locations_df = pd.read_csv("C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/locations.csv")
order_list_df = pd.read_excel('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/order_list_1.xlsx')
travel_matrix_df = pd.read_csv('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/travel_matrix.csv')
trucks_df = pd.read_csv('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/trucks.csv')

# Preprocess data
dest1 = list(set(order_list_df['Destination Code']))
dest = [str(i) for i in dest1]
Q = sorted(list(set(trucks_df['truck_max_weight'])))
Q1 = [Q[0]] * 5 + [Q[1]] * 1 + [Q[2]] * 2 + [Q[3]] * 7 + [Q[4]] * 4
vcost = [24, 35, 45, 56, 90]
var_cost = [vcost[0]] * 5 + [vcost[1]] * 1 + [vcost[2]] * 2 + [vcost[3]] * 7 + [vcost[4]] * 4
fixed_cost = Q1
demands = [0] + list(order_list_df.groupby('Destination Code').sum("Total Weight")['Total Weight'])
dest2 = ['A123'] + sorted(dest)
dest3 = {dest2[i]: i for i in range(len(dest2))}

travel_matrix_df = travel_matrix_df[
    (travel_matrix_df['source_location_code'].isin(dest + ['A123'])) &
    (travel_matrix_df['destination_location_code'].isin(dest + ['A123']))
]
travel_matrix_df['mapped_source'] = travel_matrix_df['source_location_code'].map(dest3)
travel_matrix_df['mapped_destination'] = travel_matrix_df['destination_location_code'].map(dest3)
num_nodes = len(set(travel_matrix_df['mapped_source']).union(set(travel_matrix_df['mapped_destination'])))

# Create data model
def create_data_model():
    data = {
        'distance_matrix': np.zeros((num_nodes, num_nodes), dtype=int),
        'time_matrix': np.zeros((num_nodes, num_nodes), dtype=int),
        'time_windows': [
            (
                int(row['location_loading_unloading_window_start'].split(':')[0]) * 60 +
                int(row['location_loading_unloading_window_start'].split(':')[1]),
                int(row['location_loading_unloading_window_end'].split(':')[0]) * 60 +
                int(row['location_loading_unloading_window_end'].split(':')[1])
            )
            for _, row in locations_df.iterrows()
        ],
        'demands': demands,
        'vehicle_capacities': Q1,
        'variable_costs': var_cost,
        'fixed_costs': fixed_cost,
        'num_vehicles': len(Q1),
        'depot': 0
    }
    for i in travel_matrix_df.index:
        data['distance_matrix'][travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i]] = \
            travel_matrix_df['travel_distance_in_km'][i]
        data['time_matrix'][travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i]] = \
            travel_matrix_df['travel_time_in_min'][i]

    return data

def print_solution(data, manager, routing, solution):
    total_cost = solution.ObjectiveValue()
    print(f'Total cost: {total_cost}')

    # Display dropped nodes.
    dropped_nodes = []
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if solution.Value(routing.NextVar(node)) == node:
            dropped_nodes.append(manager.IndexToNode(node))

    if dropped_nodes:
        print(f"Dropped nodes: {dropped_nodes}")

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:"
        route_distance = 0
        route_load = 0
        route_nodes = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += f" {node_index} Load({route_load}) ->"
            route_nodes.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        route_nodes.append(manager.IndexToNode(index))
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        if route_load > data['vehicle_capacities'][vehicle_id]:
            plan_output += "WARNING: Route exceeds vehicle capacity!\n"
        print(plan_output)

def main():
    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def cost_callback(from_index, to_index, vehicle_id):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_distance = data['distance_matrix'][from_node][to_node]
        variable_cost = data['variable_costs'][vehicle_id]
        return travel_distance * variable_cost

    transit_callback_index = routing.RegisterTransitCallback(cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        data['vehicle_capacities'],
        True,
        'Capacity'
    )

    # Add time constraints
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        1440,  # Allow waiting time
        1440,  # Maximum travel time
        False,
        'Time'
    )

    # Adjust search strategy to balance routes better
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 30
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print_solution(data, manager, routing, solution)
    else:
        print("No solution found!")

if __name__ == '__main__':
    main()
