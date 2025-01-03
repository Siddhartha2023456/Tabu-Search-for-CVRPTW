from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import numpy as np

# Load and preprocess data
locations_df = pd.read_csv("C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/locations.csv")
order_list_df = pd.read_excel('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/order_list_1.xlsx')
travel_matrix_df = pd.read_csv('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/travel_matrix.csv')
trucks_df = pd.read_csv('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/trucks.csv')

dest1 = list(set(order_list_df['Destination Code']))
dest = [str(i) for i in dest1]

# Preprocess data
Q = sorted(list(set(trucks_df['truck_max_weight'])))
Q1 = [Q[0]] * 5 + [Q[1]] * 1 + [Q[2]] * 2 + [Q[3]] * 7 + [Q[4]] * 4
vcost = [24, 35, 45, 56, 90]
var_cost = [vcost[0]] * 5 + [vcost[1]] * 1 + [vcost[2]] * 2 + [vcost[3]] * 7 + [vcost[4]] * 4
fixed_cost = Q1
demands = [0] + list(order_list_df.groupby('Destination Code').sum("Total Weight")['Total Weight'])
# Map travel matrix
dest2 = ['A123'] + sorted(dest)
dest3 = {dest2[i]: i for i in range(len(dest2))}

travel_matrix_df = travel_matrix_df[(travel_matrix_df['source_location_code'].isin(dest + ['A123'])) & (
    travel_matrix_df['destination_location_code'].isin(dest + ['A123']))]
travel_matrix_df['mapped_source'] = travel_matrix_df['source_location_code'].map(dest3)
travel_matrix_df['mapped_destination'] = travel_matrix_df['destination_location_code'].map(dest3)
num_nodes = len(set(travel_matrix_df['mapped_source']).union(set(travel_matrix_df['mapped_destination'])))
dist_matrix= np.zeros((num_nodes, num_nodes), dtype=int)
for i in travel_matrix_df.index:
    dist_matrix[travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i]] = \
        travel_matrix_df['travel_distance_in_km'][i]


# Update the create_data_model function to include variable costs
# def create_data_model():
#     """Stores the data for the problem."""
#     data = {}
#     data['distance_matrix'] = np.zeros((num_nodes, num_nodes), dtype=int)
#     for i in travel_matrix_df.index:
#         data['distance_matrix'][travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i]] = \
#             travel_matrix_df['travel_distance_in_km'][i]

#     data['time_matrix'] = np.zeros((num_nodes, num_nodes), dtype=int)
#     for i in travel_matrix_df.index:
#         data['time_matrix'][travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i]] = \
#             travel_matrix_df['travel_time_in_min'][i]

#     # Convert time strings to minutes in create_data_model
#     def time_to_minutes(time_str):
#         hours, minutes = map(int, time_str.split(':'))
#         return hours * 60 + minutes

#     data['time_windows'] = [
#         (
#             time_to_minutes(row['location_loading_unloading_window_start']),
#             time_to_minutes(row['location_loading_unloading_window_end'])
#         )
#         for _, row in locations_df.iterrows()
#     ]
#     data['demands'] = demands
#     data['vehicle_capacities'] = Q1
#     data['variable_costs'] = var_cost
#     data['fixed_costs'] = fixed_cost
#     data['num_vehicles'] = len(Q1)
#     data['depot'] = 0
#     print(data['distance_matrix'])
#     return data
def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = dist_matrix
    data["demands"] = demands
    data["vehicle_capacities"] = Q1
    data["num_vehicles"] = len(Q1)
    data["depot"] = 0
    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print(f"Total distance of all routes: {total_distance}m")
    print(f"Total load of all routes: {total_load}")


def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfVehicle(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)


if __name__ == "__main__":
    main()