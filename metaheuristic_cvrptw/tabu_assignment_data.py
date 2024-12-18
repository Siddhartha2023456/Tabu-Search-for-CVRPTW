import copy
import time
import pandas as pd
import numpy as np
import itertools
import cProfile

# Start measuring time
starting_time = time.time()


def initialize_solution(nodes, vehicles, dist_matrix, demands_w, max_capacity_w):
    """
    Generate an initial solution where each vehicle starts from the depot
    and routes are built using the path-cheapest arc strategy.
    Ensures each customer is visited at least once.
    """
    s_t = time.time()
    solution = {v: [] for v in vehicles}
    remaining_demand_w = copy.deepcopy(demands_w)
    unvisited = set(nodes[1:])  # Exclude depot

    for v in vehicles:
        current_node = 0  # Start at the depot
        current_capacity_w = max_capacity_w[v]
        route = [current_node]

        while unvisited:
            # Find the nearest node that satisfies all constraints
            nearest_node = None
            nearest_distance = float("inf")
            for n in unvisited:
                # Check if it satisfies the capacity constraint
                if remaining_demand_w[n] <= current_capacity_w and dist_matrix[current_node, n] < nearest_distance:
                    nearest_node = n
                    nearest_distance = dist_matrix[current_node, n]

            if nearest_node is None:
                break  # No valid node found for this vehicle, end its route

            # Add the nearest node to the route
            route.append(nearest_node)
            current_capacity_w -= remaining_demand_w[nearest_node]
            remaining_demand_w[nearest_node] = 0
            unvisited.remove(nearest_node)
            current_node = nearest_node

        route.append(0)  # Return to depot
        solution[v] = route

        # Stop if all customers have been visited
        if not unvisited:
            break

    # Assign remaining unvisited nodes to any vehicle with capacity left
    for n in unvisited:
        for v in vehicles:
            if remaining_demand_w[n] <= max_capacity_w[v]:
                solution[v].insert(-1, n)  # Add before returning to depot
                break
    e_t = time.time()
    run_time = e_t - s_t
    print(f"Time for initial solution:{run_time} seconds")
    return solution


def is_valid_route(route, demands_w, max_capacity_w):
    total_weight = 0
    for node in route:
        if node != 0:  # Exclude depot
            total_weight += demands_w[node]
            if total_weight > max_capacity_w:
                return False
    return True


def generate_neighbors(solution, vehicles, nodes, tabu_list, max_capacity_w, demands_w, dist_matrix):
    """
    Generate neighbors for the given solution using relocation, swap, and 2-opt moves.
    Optimized for reduced runtime.
    """
    s_t = time.time()
    neighbors = []

    # Cache current weights for each vehicle
    route_weights = {
        v: sum(demands_w[node] for node in solution[v] if node != 0) for v in vehicles
    }

    # Relocation: Move a customer from one vehicle to another
    for v1, v2 in itertools.permutations(vehicles, 2):
        for i in range(1, len(solution[v1]) - 1):  # Exclude depot
            node = solution[v1][i]
            for j in range(1, len(solution[v2])):  # Allow insertions in v2
                # Modify routes incrementally
                route_v1 = solution[v1][:]
                route_v2 = solution[v2][:]
                route_v1.remove(node)
                route_v2.insert(j, node)

                # Incremental weight checks
                new_weight_v1 = route_weights[v1] - demands_w[node]
                new_weight_v2 = route_weights[v2] + demands_w[node]

                if new_weight_v1 <= max_capacity_w[v1] and new_weight_v2 <= max_capacity_w[v2]:
                    new_solution = solution.copy()
                    new_solution[v1] = route_v1
                    new_solution[v2] = route_v2
                    if new_solution not in tabu_list:
                        neighbors.append(new_solution)

    # Swap: Swap two customers between two different vehicles
    for v1, v2 in itertools.permutations(vehicles, 2):
        for i in range(1, len(solution[v1]) - 1):
            for j in range(1, len(solution[v2]) - 1):
                node1, node2 = solution[v1][i], solution[v2][j]

                # Modify routes incrementally
                route_v1 = solution[v1][:]
                route_v2 = solution[v2][:]
                route_v1[i], route_v2[j] = node2, node1

                # Incremental weight checks
                new_weight_v1 = route_weights[v1] - demands_w[node1] + demands_w[node2]
                new_weight_v2 = route_weights[v2] - demands_w[node2] + demands_w[node1]

                if new_weight_v1 <= max_capacity_w[v1] and new_weight_v2 <= max_capacity_w[v2]:
                    new_solution = solution.copy()
                    new_solution[v1] = route_v1
                    new_solution[v2] = route_v2
                    if new_solution not in tabu_list:
                        neighbors.append(new_solution)

    # 2-Opt: Reverse a subsequence in a single route
    for v in vehicles:
        route = solution[v]
        for i in range(1, len(route) - 2):  # Exclude depot
            for j in range(i + 1, len(route) - 1):  # Ensure valid subsequence
                new_route = route[:]
                new_route[i:j + 1] = reversed(new_route[i:j + 1])

                # Incremental weight check (no weight change for 2-opt)
                if route_weights[v] <= max_capacity_w[v]:
                    new_solution = solution.copy()
                    new_solution[v] = new_route
                    if new_solution not in tabu_list:
                        neighbors.append(new_solution)

    e_t = time.time()
    run_time = e_t - s_t
    print(f"Time for generating neighbors: {run_time:.4f} seconds")
    return neighbors


def calculate_total_distance(solution, dist_matrix):
    """
    Calculate the total distance for a given solution.
    """
    total_distance = 0
    for route in solution.values():  # Assuming solution is a dictionary
        if len(route) > 1:  # Skip empty or single-point routes
            total_distance += sum(
                dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)
            )
    return total_distance


# def calculate_total_cost(solution, dist_matrix, Q1, var_cost, fixed_cost):
#     total_fixed_cost = 0
#     total_variable_cost = 0

#     for vehicle, route in solution.items():
#         route_length = len(route)
#         if route_length > 1:  # Skip unused vehicles (routes with only the depot)
#             # Add fixed cost for the vehicle
#             total_fixed_cost += fixed_cost[vehicle]

#             # Calculate total distance using a loop (optimized)
#             total_distance = 0
#             for i in range(len(route) - 1):
#                 total_distance += dist_matrix[route[i], route[i + 1]]

#             # Add variable cost (distance-based)
#             total_variable_cost += total_distance * var_cost[vehicle]
#     total_cost = total_fixed_cost + total_variable_cost
#     return total_cost
def calculate_total_cost(solution, dist_matrix, Q1, var_cost, fixed_cost):
    # Store the previous solution and costs for incremental updates
    if not hasattr(calculate_total_cost, "previous_solution"):
        calculate_total_cost.previous_solution = {}
        calculate_total_cost.previous_cost = {}
        calculate_total_cost.total_cost = 0

    total_cost = 0

    for vehicle, route in solution.items():
        previous_route = calculate_total_cost.previous_solution.get(vehicle, [])

        # Recalculate cost only if the route has changed
        if route != previous_route:
            if len(route) > 1:  # Skip unused vehicles (routes with only the depot)
                # Fixed cost for the vehicle
                fixed_cost_vehicle = fixed_cost[vehicle]

                # Calculate total distance using a loop
                total_distance = 0
                for i in range(len(route) - 1):
                    total_distance += dist_matrix[route[i], route[i + 1]]

                # Variable cost (distance-based)
                variable_cost_vehicle = total_distance * var_cost[vehicle]

                # Update the previous cost for this vehicle
                calculate_total_cost.previous_cost[vehicle] = fixed_cost_vehicle + variable_cost_vehicle
            else:
                # No cost for unused vehicles
                calculate_total_cost.previous_cost[vehicle] = 0

        # Ensure the vehicle has an entry in previous_cost to avoid KeyError
        if vehicle not in calculate_total_cost.previous_cost:
            calculate_total_cost.previous_cost[vehicle] = 0

        # Add the cost of this vehicle (either recalculated or from previous)
        total_cost += calculate_total_cost.previous_cost[vehicle]

    # Update the previous solution
    calculate_total_cost.previous_solution = solution.copy()

    # Store the total cost
    calculate_total_cost.total_cost = total_cost

    return total_cost


def tabu_search(
        nodes, vehicles, dist_matrix, demands_w, max_capacity_w, Q1, var_cost, fixed_cost, max_iter, tabu_tenure
):
    """
    Tabu Search for minimizing total cost (fixed + variable) in a CVRPTW problem.
    Includes a stopping criterion: if no improvement for 3 iterations, terminate the search early.
    """
    st_time = time.time()

    # Initialize
    current_solution = initialize_solution(nodes, vehicles, dist_matrix, demands_w, max_capacity_w)
    best_solution = current_solution
    best_cost = calculate_total_cost(current_solution, dist_matrix, Q1, var_cost, fixed_cost)
    tabu_list = []
    tabu_queue = []
    current_costs = []  # To store the current cost in each iteration
    no_improvement_count = 0  # Counter for iterations without improvement

    for iteration in range(max_iter):
        s_t = time.time()

        # Generate neighbors
        neighbors = generate_neighbors(
            current_solution, vehicles, nodes, tabu_list, max_capacity_w, demands_w, dist_matrix
        )

        # Evaluate neighbors based on total cost
        best_neighbor = None
        best_neighbor_cost = float("inf")
        for neighbor in neighbors:
            neighbor_cost = calculate_total_cost(neighbor, dist_matrix, Q1, var_cost, fixed_cost)
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost

        # Update current solution if a better neighbor is found
        if best_neighbor and best_neighbor_cost < best_cost:
            current_solution = best_neighbor
            best_cost = best_neighbor_cost
            best_solution = current_solution
            no_improvement_count = 0  # Reset the no improvement counter
        else:
            no_improvement_count += 1

        # Early stopping condition
        if no_improvement_count >= 3:
            print(f"Stopping early: No improvement in the last 3 iterations.")
            break

        # Update tabu list
        tabu_list.append(current_solution)
        if len(tabu_queue) >= tabu_tenure:
            tabu_list.remove(tabu_queue.pop(0))
        tabu_queue.append(current_solution)

        # Record current cost
        current_costs.append(best_cost)

        e_t = time.time()
        run_time = e_t - s_t
        print(f"Iteration {iteration + 1}, Current Cost: {best_cost}, Time for this iteration: {run_time:.2f} seconds")

    en_time = time.time()
    total_time = en_time - st_time
    print(f"Total time in Tabu Search: {total_time:.2f} seconds")

    return best_solution, best_cost, current_costs


# Load data
locations_df = pd.read_csv("../inputs/locations.csv")
order_list_df = pd.read_excel('../inputs/order_list_1.xlsx')
travel_matrix_df = pd.read_csv('../inputs/travel_matrix.csv')
trucks_df = pd.read_csv('../inputs/trucks.csv')
Q = sorted(list(set(trucks_df['truck_max_weight'])))
Q1 = [Q[0]] * 5 + [Q[1]] * 1 + [Q[2]] * 2 + [Q[3]] * 7 + [Q[4]] * 4
vcost = [24, 35, 45, 56, 90]
var_cost = [vcost[0]] * 5 + [vcost[1]] * 1 + [vcost[2]] * 2 + [vcost[3]] * 7 + [vcost[4]] * 4
fixed_cost = Q1
dest1 = list(set(order_list_df['Destination Code']))
dest = [str(i) for i in dest1]
order_list_df = order_list_df[order_list_df['Destination Code'].isin(dest1)]
order_list_df1 = order_list_df.sort_values(by='Destination Code').groupby('Destination Code').sum(
    "Total Weight").reset_index()
sum(order_list_df1["Total Weight"])
locations_df = locations_df[locations_df['location_code'].isin(dest + ['A123'])]
# Convert loading/unloading windows to minutes with explicit format
locations_df['start_minutes'] = pd.to_datetime(locations_df['location_loading_unloading_window_start'],
                                               format='%H:%M').dt.hour * 60 + pd.to_datetime(
    locations_df['location_loading_unloading_window_start'], format='%H:%M').dt.minute
locations_df['end_minutes'] = pd.to_datetime(locations_df['location_loading_unloading_window_end'],
                                             format='%H:%M').dt.hour * 60 + pd.to_datetime(
    locations_df['location_loading_unloading_window_end'], format='%H:%M').dt.minute
customers = locations_df.sort_values(by='location_code').iloc[:len(order_list_df1), :]
locations_df2 = locations_df.sort_values(by='location_code')
cap_df = dict(zip(trucks_df['truck_type'], trucks_df['truck_max_weight']))
max_veh_access = []
for i in locations_df2.index:
    max_veh_access.append(cap_df[eval(locations_df2['trucks_allowed'][i])[-1]])
max_veh_access = max_veh_access[len(order_list_df1):] + max_veh_access[:len(order_list_df1)]
depot = locations_df.sort_values(by='location_code').iloc[len(order_list_df1):, :]
Nodes = pd.concat([depot, customers], ignore_index=True)
vehicles = [k for k in range(0, len(Q1))]
customers = [i for i in range(1, len(Nodes))]
nodes = [i for i in range(0, len(Nodes))]
demands_w = [0] + list(order_list_df1['Total Weight'])
start_time = list(Nodes['start_minutes'])
finish_time = list(Nodes['end_minutes'])
dest2 = ['A123'] + sorted(dest)
dest3 = {}
for i in range(len(dest2)):
    dest3[dest2[i]] = i
travel_matrix_df = travel_matrix_df[(travel_matrix_df['source_location_code'].isin(dest + ['A123'])) & (
    travel_matrix_df['destination_location_code'].isin(dest + ['A123']))]
travel_matrix_df['mapped_source'] = travel_matrix_df['source_location_code'].map(dest3)
travel_matrix_df['mapped_destination'] = travel_matrix_df['destination_location_code'].map(dest3)
dist_matrix = {}
time_matrix = {}
for i in travel_matrix_df.index:
    dist_matrix[(travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i])] = \
    travel_matrix_df['travel_distance_in_km'][i]
    time_matrix[(travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i])] = \
    travel_matrix_df['travel_time_in_min'][i]
max_capacity_w = {v: Q1[v] for v in range(len(Q1))}
best_solution, best_cost, cost_progress = tabu_search(
    nodes, vehicles, dist_matrix, demands_w, max_capacity_w, Q1=Q1, var_cost=var_cost, fixed_cost=fixed_cost,
    max_iter=45, tabu_tenure=10
)

# End measuring time
end_time = time.time()

# Calculate and print runtime
runtime = end_time - starting_time
print(f"Runtime: {runtime:.2f} seconds")
print('*'*50)
# Display results
print("Best Solution:")
distance = []
for v, route in best_solution.items():
    route_distance = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    distance.append(route_distance)
    print(f"Vehicle {v}: Route: {route}, Distance: {route_distance:.2f}")
# print(f"Total Distance: {best_distance}")
print(f"Total cost = {best_cost}")
print(f"Total distance = {sum(distance)}")
print('-'*75)
# Create a profile object
profiler = cProfile.Profile()

# Profile the code block
profiler.enable()
tabu_search(nodes, vehicles, dist_matrix, demands_w, max_capacity_w, Q1=Q1, var_cost=var_cost, fixed_cost=fixed_cost,
            max_iter=45, tabu_tenure=10)  # Call your connected functions
profiler.disable()

# Print profiling results
profiler.print_stats(sort='time')
