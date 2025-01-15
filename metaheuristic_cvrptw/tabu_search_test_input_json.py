import json
import pandas as pd
import copy
import time
import math
import cProfile
import itertools
# Start measuring time
starting_time = time.time()
# def initialize_solution(nodes, vehicles, dist_matrix, demands_w, demands_v, max_capacity_w, max_capacity_v):
#     """
#     Generate an initial solution where each vehicle starts from the depot
#     and routes are built using the path-cheapest arc strategy.
#     """
#     solution = {v: [] for v in vehicles}
#     remaining_demand_w = copy.deepcopy(demands_w)
#     remaining_demand_v = copy.deepcopy(demands_v)
#     unvisited = set(nodes[1:])  # Exclude depot

#     for v in vehicles:
#         current_node = 0  # Start at the depot
#         current_capacity_w = max_capacity_w[v]
#         current_capacity_v = max_capacity_v[v]
#         route = [current_node]

#         while unvisited:
#             # Find the nearest node that satisfies capacity constraints
#             nearest_node = None
#             nearest_distance = float("inf")
#             for n in unvisited:
#                 if (
#                     remaining_demand_w[n] <= current_capacity_w and
#                     remaining_demand_v[n] <= current_capacity_v and
#                     dist_matrix[current_node, n] < nearest_distance
#                 ):
#                     nearest_node = n
#                     nearest_distance = dist_matrix[current_node, n]

#             if nearest_node is None:
#                 break  # No valid node found, stop this route

#             # Add the nearest node to the route
#             route.append(nearest_node)
#             current_capacity_w -= remaining_demand_w[nearest_node]
#             current_capacity_v -= remaining_demand_v[nearest_node]
#             remaining_demand_w[nearest_node] = 0
#             remaining_demand_v[nearest_node] = 0
#             unvisited.remove(nearest_node)
#             current_node = nearest_node

#         route.append(0)  # Return to depot
#         solution[v] = route

#     return solution

import copy

def initialize_solution(nodes, vehicles, dist_matrix, demands_w, demands_v, max_capacity_w, max_capacity_v):
    """
    Generate an initial solution where each vehicle starts from the depot
    and routes are built using the path-cheapest arc strategy while satisfying:
    1. Maximum number of customers in a route = 2
    2. Maximum distance between consecutive customers (excluding depot) = 100 Km
    """
    # max_customers_per_route = 3  # Constraint 1
    # max_distance_between_customers = 100  # Constraint 2

    solution = {v: [] for v in vehicles}
    remaining_demand_w = copy.deepcopy(demands_w)
    remaining_demand_v = copy.deepcopy(demands_v)
    unvisited = set(nodes[1:])  # Exclude depot

    for v in vehicles:
        current_node = 0  # Start at the depot
        current_capacity_w = max_capacity_w[v]
        current_capacity_v = max_capacity_v[v]
        route = [current_node]
        customer_count = 0

        while unvisited:
            # Find the nearest node that satisfies all constraints
            nearest_node = None
            nearest_distance = float("inf")
            for n in unvisited:
                # Check constraints
                if (
                    remaining_demand_w[n] <= current_capacity_w and
                    remaining_demand_v[n] <= current_capacity_v and
                    dist_matrix[current_node, n] < nearest_distance
                ):
                    
                    nearest_node = n
                    nearest_distance = dist_matrix[current_node, n]

            if nearest_node is None:
                break  # No valid node found, stop this route

            # Add the nearest node to the route
            route.append(nearest_node)
            customer_count += 1
            current_capacity_w -= remaining_demand_w[nearest_node]
            current_capacity_v -= remaining_demand_v[nearest_node]
            remaining_demand_w[nearest_node] = 0
            remaining_demand_v[nearest_node] = 0
            unvisited.remove(nearest_node)
            current_node = nearest_node

        route.append(0)  # Return to depot
        solution[v] = route

    return solution

# def generate_neighbors(solution, vehicles, nodes, tabu_list, max_capacity_w, max_capacity_v, demands_w, demands_v, dist_matrix):
#     neighbors = []

#     # Relocation: Move a node from one vehicle to another
#     for v1 in vehicles:
#         for v2 in vehicles:
#             if v1 == v2:
#                 continue

#             for i in range(1, len(solution[v1]) - 1):  # Exclude depot
#                 node = solution[v1][i]

#                 # Check if moving this node to v2 violates constraints
#                 if demands_w[node] > max_capacity_w[v2] or demands_v[node] > max_capacity_v[v2]:
#                     continue

#                 for j in range(1, len(solution[v2])):  # Exclude depot
#                     new_solution = copy.deepcopy(solution)
#                     new_solution[v1].remove(node)
#                     new_solution[v2].insert(j, node)

#                     # Validate cumulative capacities for both routes
#                     if (
#                         is_valid_route(new_solution[v1], demands_w, demands_v, max_capacity_w[v1], max_capacity_v[v1]) and
#                         is_valid_route(new_solution[v2], demands_w, demands_v, max_capacity_w[v2], max_capacity_v[v2])
#                     ):
#                         if new_solution not in tabu_list:
#                             neighbors.append(new_solution)

#     # Swap: Swap two nodes between routes
#     for v1 in vehicles:
#         for v2 in vehicles:
#             if v1 == v2:
#                 continue

#             for i in range(1, len(solution[v1]) - 1):  # Exclude depot
#                 for j in range(1, len(solution[v2]) - 1):  # Exclude depot
#                     node1 = solution[v1][i]
#                     node2 = solution[v2][j]

#                     new_solution = copy.deepcopy(solution)
#                     new_solution[v1][i], new_solution[v2][j] = node2, node1

#                     # Validate cumulative capacities for both routes
#                     if (
#                         is_valid_route(new_solution[v1], demands_w, demands_v, max_capacity_w[v1], max_capacity_v[v1]) and
#                         is_valid_route(new_solution[v2], demands_w, demands_v, max_capacity_w[v2], max_capacity_v[v2])
#                     ):
#                         if new_solution not in tabu_list:
#                             neighbors.append(new_solution)

#     # 2-Opt: Reverse a subsequence in a single route
#     for v in vehicles:
#         route = solution[v]
#         for i in range(1, len(route) - 2):  # Exclude depot
#             for j in range(i + 1, len(route) - 1):  # Ensure valid subsequence
#                 new_solution = copy.deepcopy(solution)
#                 new_solution[v] = route[:i] + route[i:j+1][::-1] + route[j+1:]

#                 # Validate route capacities
#                 if is_valid_route(new_solution[v], demands_w, demands_v, max_capacity_w[v], max_capacity_v[v]):
#                     if new_solution not in tabu_list:
#                         neighbors.append(new_solution)

#     return neighbors


# def is_valid_route(route, demands_w, demands_v, max_capacity_w, max_capacity_v):
#     """
#     Helper function to check if a route satisfies capacity constraints.
#     """
#     total_weight = sum(demands_w[node] for node in route if node != 0)  # Exclude depot
#     total_volume = sum(demands_v[node] for node in route if node != 0)
#     return total_weight <= max_capacity_w and total_volume <= max_capacity_v

def generate_neighbors(solution, vehicles, nodes, tabu_list, max_capacity_w, max_capacity_v, demands_w, demands_v, dist_matrix):
    """
    Generate neighbors for the given solution using relocation, swap, 2-opt moves, 
    and merging routes of smaller vehicles into a larger vehicle.
    Optimized for reduced runtime, considering max_capacity_v and demands_v.
    """
    s_t = time.time()
    neighbors = []

    def is_valid_route(route, demands_w, demands_v, max_capacity_w, max_capacity_v):
        """
        Helper function to check if a route satisfies capacity constraints.
        """
        total_weight = sum(demands_w[node] for node in route if node != 0)  # Exclude depot
        total_volume = sum(demands_v[node] for node in route if node != 0)
        return total_weight <= max_capacity_w and total_volume <= max_capacity_v

    # Cache current weights and volumes for each vehicle
    route_weights = {
        v: sum(demands_w[node] for node in solution[v] if node != 0) for v in vehicles
    }
    route_volumes = {
        v: sum(demands_v[node] for node in solution[v] if node != 0) for v in vehicles
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

                # Incremental checks for weight and volume
                new_weight_v1 = route_weights[v1] - demands_w[node]
                new_weight_v2 = route_weights[v2] + demands_w[node]
                new_volume_v1 = route_volumes[v1] - demands_v[node]
                new_volume_v2 = route_volumes[v2] + demands_v[node]

                if (new_weight_v1 <= max_capacity_w[v1] and new_weight_v2 <= max_capacity_w[v2] and
                    new_volume_v1 <= max_capacity_v[v1] and new_volume_v2 <= max_capacity_v[v2]):
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

                # Incremental checks for weight and volume
                new_weight_v1 = route_weights[v1] - demands_w[node1] + demands_w[node2]
                new_weight_v2 = route_weights[v2] - demands_w[node2] + demands_w[node1]
                new_volume_v1 = route_volumes[v1] - demands_v[node1] + demands_v[node2]
                new_volume_v2 = route_volumes[v2] - demands_v[node2] + demands_v[node1]

                if (new_weight_v1 <= max_capacity_w[v1] and new_weight_v2 <= max_capacity_w[v2] and
                    new_volume_v1 <= max_capacity_v[v1] and new_volume_v2 <= max_capacity_v[v2]):
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

                # Check weight and volume (no change in total for 2-opt)
                if route_weights[v] <= max_capacity_w[v] and route_volumes[v] <= max_capacity_v[v]:
                    new_solution = solution.copy()
                    new_solution[v] = new_route
                    if new_solution not in tabu_list:
                        neighbors.append(new_solution)

    # Merge routes of two smaller vehicles into a larger vehicle
    for v1, v2, v_large in itertools.permutations(vehicles, 3):
        if len(solution[v1]) > 2 and len(solution[v2]) > 2:
            if max_capacity_w[v_large] >= (max_capacity_w[v1] + max_capacity_w[v2]) and \
               max_capacity_v[v_large] >= (max_capacity_v[v1] + max_capacity_v[v2]):
                combined_route = solution[v1][1:-1] + solution[v2][1:-1]  # Exclude depots
                combined_weight = route_weights[v1] + route_weights[v2]
                combined_volume = route_volumes[v1] + route_volumes[v2]

                if combined_weight <= max_capacity_w[v_large] and combined_volume <= max_capacity_v[v_large]:
                    new_solution = solution.copy()
                    new_solution[v1] = [0, 0]  # Empty route
                    new_solution[v2] = [0, 0]  # Empty route
                    new_solution[v_large] = [0] + combined_route + [0]
                    visited_nodes = {node for route in new_solution.values() for node in route if node != 0}
                    if len(visited_nodes) == 108 and new_solution not in tabu_list:
                        neighbors.append(new_solution)

    # Split a route of a larger vehicle into two smaller vehicles
    for v_large, v1, v2 in itertools.permutations(vehicles, 3):
        if len(solution[v_large]) > 2 and max_capacity_w[v_large] > max_capacity_w[v1] and max_capacity_w[v_large] > max_capacity_w[v2]:
            route_large = solution[v_large][1:-1]  # Exclude depots
            for split_point in range(1, len(route_large)):
                route_v1 = route_large[:split_point]
                route_v2 = route_large[split_point:]

                weight_v1 = sum(demands_w[node] for node in route_v1)
                weight_v2 = sum(demands_w[node] for node in route_v2)
                volume_v1 = sum(demands_v[node] for node in route_v1)
                volume_v2 = sum(demands_v[node] for node in route_v2)

                if (weight_v1 <= max_capacity_w[v1] and weight_v2 <= max_capacity_w[v2] and
                    volume_v1 <= max_capacity_v[v1] and volume_v2 <= max_capacity_v[v2]):
                    new_solution = solution.copy()
                    new_solution[v_large] = []  # Empty route
                    new_solution[v1] = [0] + route_v1 + [0]
                    new_solution[v2] = [0] + route_v2 + [0]
                    visited_nodes = {node for route in new_solution.values() for node in route if node != 0}
                    if visited_nodes == set(nodes) and new_solution not in tabu_list:
                        neighbors.append(new_solution)

    # Validate all neighbors
    valid_neighbors = []
    for neighbor in neighbors:
        if all(is_valid_route(neighbor[v], demands_w, demands_v, max_capacity_w[v], max_capacity_v[v]) for v in vehicles):
            valid_neighbors.append(neighbor)

    e_t = time.time()
    run_time = e_t - s_t
    print(f"Time for generating neighbors: {run_time:.4f} seconds")
    return valid_neighbors


def calculate_total_cost(solution, dist_matrix, fixed_cost, variable_cost):
    total_fixed_cost = 0
    total_variable_cost = 0

    for v, route in solution.items():
        if len(route) > 2:  # Ignore empty routes
            total_fixed_cost += fixed_cost[v]
            total_variable_cost += sum(
                dist_matrix[route[i], route[i + 1]] * variable_cost[v]
                for i in range(len(route) - 1)
            )
    total_cost = total_fixed_cost + total_variable_cost
    return total_cost,total_fixed_cost,total_variable_cost



def calculate_total_distance(solution, dist_matrix):
    total_distance = 0
    for route in solution.values():
        if len(route) > 1:  # Skip empty routes
            total_distance += sum(
                dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)
            )
    return total_distance



import time

def tabu_search(nodes, vehicles, dist_matrix, demands_w, demands_v, max_capacity_w, max_capacity_v, fixed_cost, variable_cost, max_iter, tabu_tenure, no_improvement_limit, time_limit):
    # Initialize
    current_solution = initialize_solution(nodes, vehicles, dist_matrix, demands_w, demands_v, max_capacity_w, max_capacity_v)
    best_solution = current_solution
    best_cost, best_fixed_cost, best_variable_cost = calculate_total_cost(current_solution, dist_matrix, fixed_cost, variable_cost)
    tabu_list = []
    tabu_queue = []
    current_costs = []  # To store the current cost in each iteration

    no_improvement_count = 0  # Track consecutive no-improvement iterations
    start_time = time.time()  # Track start time

    for iteration in range(max_iter):
        # Check time limit
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            print(f"Terminating due to time limit: {elapsed_time:.2f}s")
            break

        # Generate neighbors
        neighbors = generate_neighbors(
            current_solution, vehicles, nodes, tabu_list, max_capacity_w, max_capacity_v, demands_w, demands_v, dist_matrix
        )

        # Evaluate neighbors
        best_neighbor = None
        best_neighbor_cost = float("inf")
        for neighbor in neighbors:
            cost, _, _ = calculate_total_cost(neighbor, dist_matrix, fixed_cost, variable_cost)
            if cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = cost

        # Update current solution
        if best_neighbor and best_neighbor_cost < best_cost:
            current_solution = best_neighbor
            best_cost, best_fixed_cost, best_variable_cost = calculate_total_cost(current_solution, dist_matrix, fixed_cost, variable_cost)
            best_solution = current_solution
            no_improvement_count = 0  # Reset no-improvement counter
        else:
            no_improvement_count += 1

        # Check no-improvement termination
        if no_improvement_count >= no_improvement_limit:
            print(f"Terminating due to no improvement in {no_improvement_count} consecutive iterations.")
            break

        # Update tabu list
        tabu_list.append(current_solution)
        if len(tabu_queue) >= tabu_tenure:
            tabu_list.remove(tabu_queue.pop(0))
        tabu_queue.append(current_solution)

        # Record current cost
        current_costs.append(best_cost)
        print(f"Iteration {iteration + 1}, Current Cost: {best_cost}, Time Elapsed: {elapsed_time:.2f}s")

    # Calculate best distance
    best_distance = calculate_total_distance(best_solution, dist_matrix)

    return best_solution, best_cost, best_fixed_cost, best_variable_cost, best_distance, current_costs

# Example Usage
file_path = "inputs/ncubate_request.json"
with open(file_path, 'r') as file:
    data = json.load(file)
# DATA ANALYSIS
loc_id_mapping = {loc_id: idx for idx, loc_id in enumerate(data["loc_ids"])}
mapped_location_matrix = [loc_id_mapping[loc] for loc in data["location_matrix"]]
df_orders = pd.DataFrame([data["weight_matrix"], data["volume_matrix"], mapped_location_matrix]).transpose()
df_orders.columns=["order_weight","order_volume","order_loc"]
df_orders_f = df_orders.groupby("order_loc").sum().reset_index()
distance_matrix = data["costs"]
duration_matrix = data["durations"]
max_veh_weight = data["max_weight"]
max_veh_volume = data["max_volume"]
time_windows = data["timeWindows"]
fixed_cost_list = data["max_weight"]
total_order_weights = sum(df_orders_f["order_weight"])
total_order_volume = sum(df_orders_f["order_volume"])
copy_max_wt = []
copy_max_vol = []
copy_fixed_cost_list = []
per_km_cost_list = data["perKmCostPerVehicle"]
copy_per_km_cost_list = []
for i in range(len(max_veh_weight)):
    num_veh_weight = math.ceil(total_order_weights / max_veh_weight[i])
    num_veh_volume = math.ceil(total_order_volume / max_veh_volume[i])
    num_veh = max(num_veh_weight, num_veh_volume) * 2
    for j in range(num_veh):
        copy_max_wt.append(max_veh_weight[i])
        copy_max_vol.append(max_veh_volume[i])
        copy_fixed_cost_list.append(int(fixed_cost_list[i]))
        copy_per_km_cost_list.append(per_km_cost_list[i])
start_time = [i for i, j in time_windows]
finish_time = [j for i, j in time_windows]
nodes = list(loc_id_mapping.values())
depot = 0
customers = nodes[1:]
for i in range(len(distance_matrix)):
    for j in range(len(distance_matrix[i])):
        if i ==0 or j == 0:
            distance_matrix[i][j] = 0
dist_matrix = {
    (i, j): distance_matrix[i][j]
    for i in range(len(distance_matrix))
    for j in range(len(distance_matrix[i]))
}
time_matrix = {
    (i, j): duration_matrix[i][j]
    for i in range(len(duration_matrix))
    for j in range(len(duration_matrix[i]))
}

df_vehicle = pd.DataFrame([max_veh_weight,max_veh_volume]).transpose().reset_index()
df_vehicle.columns = ["v_id","max_weight","max_volume"]
vehicles = [i for i in range(len(copy_max_wt))]
demand_w = list(df_orders_f["order_weight"])
demand_v = list(df_orders_f["order_volume"])
max_vehw =list(df_vehicle["max_weight"])
max_vehv =list(df_vehicle["max_volume"])
variable_cost = list(data["perKmCostPerVehicle"])
# demands = demand_w
max_capacity_w = {v: copy_max_wt[v] for v in vehicles}
max_capacity_v = {v: copy_max_vol[v] for v in vehicles}

# Run Tabu Search
best_solution, best_cost, best_fixed_cost, best_variable_cost, best_distance, current_costs = tabu_search(
    nodes=nodes,
    vehicles=vehicles,
    dist_matrix=dist_matrix,
    demands_w=demand_w,
    demands_v=demand_v,
    max_capacity_w=max_capacity_w,
    max_capacity_v=max_capacity_v,
    fixed_cost=copy_fixed_cost_list,
    variable_cost=copy_per_km_cost_list,
    max_iter=30,
    tabu_tenure=10, no_improvement_limit=3, time_limit=70
)
unique_costs = sorted(set(fixed_cost_list))

# Map each unique cost to an index (1-based)
cost_to_index = {cost: idx for idx, cost in enumerate(unique_costs)}
index_to_cost = {idx: cost for cost, idx in cost_to_index.items()}
print(cost_to_index)
# Generate the list of indices corresponding to the fixed costs
indices = [cost_to_index[cost] for cost in copy_fixed_cost_list]
# End measuring time
end_time = time.time()
print(f"Number of nodes: {len(nodes)}")
# Calculate and print runtime
runtime = end_time - starting_time
print(f"Runtime: {runtime:.2f} seconds")
# Display results
print("Best Solution:")
i = 0
for v, route in best_solution.items():
    route_distance = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    if len(route)>2:
        i += 1
        t = len(route)
        print(f"Route: {i},Vehicle Type: {indices[v]},\nLocation Sequence: {route[1:t-1]},\nDistance: {route_distance}, Fixed Cost: {copy_fixed_cost_list[v]}, Per Km Cost: {route_distance * copy_per_km_cost_list[v]}\nRoute Cost: {route_distance * copy_per_km_cost_list[v] + copy_fixed_cost_list[v]}")
        print("-" * 50)
print(f"Best Total Cost: {best_cost}")
print(f"Fixed Cost: {best_fixed_cost}")
print(f"Variable Cost: {best_variable_cost}")
print(f"Total Distance: {best_distance}")
print(f"Vehicle Type with fixed cost: {index_to_cost}")

# profiler = cProfile.Profile()

# # Profile the code block
# profiler.enable()
# tabu_search(
#     nodes=nodes,
#     vehicles=vehicles,
#     dist_matrix=dist_matrix,
#     demands_w=demand_w,
#     demands_v=demand_v,
#     max_capacity_w=max_capacity_w,
#     max_capacity_v=max_capacity_v,
#     fixed_cost=copy_fixed_cost_list,
#     variable_cost=copy_per_km_cost_list,
#     max_iter=30,
#     tabu_tenure=10,
# )
#  # Call your connected functions
# profiler.disable()

# # Print profiling results
# profiler.print_stats(sort='time')
