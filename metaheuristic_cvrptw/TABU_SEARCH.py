import json
import pandas as pd
import copy
import time

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
    max_customers_per_route = 3  # Constraint 1
    max_distance_between_customers = 100  # Constraint 2

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
                    dist_matrix[current_node, n] < nearest_distance and
                    customer_count < max_customers_per_route
                ):
                    # Apply distance constraint only for customer-to-customer moves
                    if len(route) > 1 and dist_matrix[route[-1], n] > max_distance_between_customers:
                        continue
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

def generate_neighbors(solution, vehicles, nodes, tabu_list, max_capacity_w, max_capacity_v, demands_w, demands_v, dist_matrix):
    neighbors = []

    # Relocation: Move a node from one vehicle to another
    for v1 in vehicles:
        for v2 in vehicles:
            if v1 == v2:
                continue

            for i in range(1, len(solution[v1]) - 1):  # Exclude depot
                node = solution[v1][i]

                # Check if moving this node to v2 violates constraints
                if demands_w[node] > max_capacity_w[v2] or demands_v[node] > max_capacity_v[v2]:
                    continue

                for j in range(1, len(solution[v2])):  # Exclude depot
                    new_solution = copy.deepcopy(solution)
                    new_solution[v1].remove(node)
                    new_solution[v2].insert(j, node)

                    # Validate cumulative capacities for both routes
                    if (
                        is_valid_route(new_solution[v1], demands_w, demands_v, max_capacity_w[v1], max_capacity_v[v1]) and
                        is_valid_route(new_solution[v2], demands_w, demands_v, max_capacity_w[v2], max_capacity_v[v2])
                    ):
                        if new_solution not in tabu_list:
                            neighbors.append(new_solution)

    # Swap: Swap two nodes between routes
    for v1 in vehicles:
        for v2 in vehicles:
            if v1 == v2:
                continue

            for i in range(1, len(solution[v1]) - 1):  # Exclude depot
                for j in range(1, len(solution[v2]) - 1):  # Exclude depot
                    node1 = solution[v1][i]
                    node2 = solution[v2][j]

                    new_solution = copy.deepcopy(solution)
                    new_solution[v1][i], new_solution[v2][j] = node2, node1

                    # Validate cumulative capacities for both routes
                    if (
                        is_valid_route(new_solution[v1], demands_w, demands_v, max_capacity_w[v1], max_capacity_v[v1]) and
                        is_valid_route(new_solution[v2], demands_w, demands_v, max_capacity_w[v2], max_capacity_v[v2])
                    ):
                        if new_solution not in tabu_list:
                            neighbors.append(new_solution)

    # 2-Opt: Reverse a subsequence in a single route
    for v in vehicles:
        route = solution[v]
        for i in range(1, len(route) - 2):  # Exclude depot
            for j in range(i + 1, len(route) - 1):  # Ensure valid subsequence
                new_solution = copy.deepcopy(solution)
                new_solution[v] = route[:i] + route[i:j+1][::-1] + route[j+1:]

                # Validate route capacities
                if is_valid_route(new_solution[v], demands_w, demands_v, max_capacity_w[v], max_capacity_v[v]):
                    if new_solution not in tabu_list:
                        neighbors.append(new_solution)

    return neighbors


def is_valid_route(route, demands_w, demands_v, max_capacity_w, max_capacity_v):
    """
    Helper function to check if a route satisfies capacity constraints.
    """
    total_weight = sum(demands_w[node] for node in route if node != 0)  # Exclude depot
    total_volume = sum(demands_v[node] for node in route if node != 0)
    return total_weight <= max_capacity_w and total_volume <= max_capacity_v



def calculate_total_cost(solution, dist_matrix, fixed_cost, variable_cost):
    total_fixed_cost = 0
    total_variable_cost = 0

    for v, route in solution.items():
        if len(route) > 1:  # Ignore empty routes
            for i in range(len(route)-1):
                total_fixed_cost += fixed_cost[route[i], v]  # Fixed cost for the vehicle

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


# Tabu Search Algorithm
# def tabu_search(nodes, vehicles, dist_matrix, demands_w, demands_v, max_capacity_w, max_capacity_v, fixed_cost, variable_cost, max_iter, tabu_tenure):
#     # Initialize
#     current_solution = initialize_solution(nodes, vehicles, dist_matrix, demands_w, demands_v, max_capacity_w, max_capacity_v)
#     best_solution = current_solution
#     best_cost = calculate_total_cost(current_solution, dist_matrix, fixed_cost, variable_cost)
#     tabu_list = []
#     tabu_queue = []
#     current_costs = []  # To store the current cost in each iteration

#     for iteration in range(max_iter):
#         # Generate neighbors
#         neighbors = generate_neighbors(
#             current_solution, vehicles, nodes, tabu_list, max_capacity_w, max_capacity_v, demands_w, demands_v, dist_matrix
#         )

#         # Evaluate neighbors
#         best_neighbor = None
#         best_neighbor_cost = float("inf")
#         for neighbor in neighbors:
#             cost = calculate_total_cost(neighbor, dist_matrix, fixed_cost, variable_cost)
#             if cost < best_neighbor_cost:
#                 best_neighbor = neighbor
#                 best_neighbor_cost = cost

#         # Update current solution
#         if best_neighbor and best_neighbor_cost < best_cost:
#             current_solution = best_neighbor
#             best_cost = best_neighbor_cost
#             best_solution = current_solution

#         # Update tabu list
#         tabu_list.append(current_solution)
#         if len(tabu_queue) >= tabu_tenure:
#             tabu_list.remove(tabu_queue.pop(0))
#         tabu_queue.append(current_solution)

#         # Record current cost
#         current_costs.append(best_cost)
#         print(f"Iteration {iteration + 1}, Current Cost: {best_cost}")

#     # Calculate best distance
#     best_distance = calculate_total_distance(best_solution, dist_matrix)

#     return best_solution, best_cost, best_distance, current_costs

def tabu_search(nodes, vehicles, dist_matrix, demands_w, demands_v, max_capacity_w, max_capacity_v, fixed_cost, variable_cost, max_iter, tabu_tenure):
    # Initialize
    current_solution = initialize_solution(nodes, vehicles, dist_matrix, demands_w, demands_v, max_capacity_w, max_capacity_v)
    best_solution = current_solution
    best_cost, best_fixed_cost, best_variable_cost = calculate_total_cost(current_solution, dist_matrix, fixed_cost, variable_cost)
    tabu_list = []
    tabu_queue = []
    current_costs = []  # To store the current cost in each iteration

    for iteration in range(max_iter):
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

        # Update tabu list
        tabu_list.append(current_solution)
        if len(tabu_queue) >= tabu_tenure:
            tabu_list.remove(tabu_queue.pop(0))
        tabu_queue.append(current_solution)

        # Record current cost
        current_costs.append(best_cost)
        print(f"Iteration {iteration + 1}, Current Cost: {best_cost}")

    # Calculate best distance
    best_distance = calculate_total_distance(best_solution, dist_matrix)

    return best_solution, best_cost, best_fixed_cost, best_variable_cost, best_distance, current_costs



# Example Usage
file_path = 'inputs/solver_params_26.json'
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
fixed_cost_matrix = data["vehicle_base_fare_matrix"]
start_time = [i for i, j in time_windows]
finish_time = [j for i, j in time_windows]
nodes = list(loc_id_mapping.values())
depot = 0
customers = nodes[1:]
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
fixed_cost = {
    (i, j): fixed_cost_matrix[i][j]
    for i in range(len(fixed_cost_matrix))
    for j in range(len(fixed_cost_matrix[i]))
}
df_vehicle = pd.DataFrame([max_veh_weight,max_veh_volume]).transpose().reset_index()
df_vehicle.columns = ["v_id","max_weight","max_volume"]
vehicles = list(df_vehicle["v_id"])
demand_w = list(df_orders_f["order_weight"])
demand_v = list(df_orders_f["order_volume"])
max_vehw =list(df_vehicle["max_weight"])
max_vehv =list(df_vehicle["max_volume"])
variable_cost = list(data["perKmCostPerVehicle"])
variable_cost
# demands = demand_w
max_capacity_w = {v: max_vehw[v] for v in vehicles}
max_capacity_v = {v: max_vehv[v] for v in vehicles}

# # Run Tabu Search
# best_solution, best_cost,best_distance, current_costs = tabu_search(
#     nodes=nodes,
#     vehicles=vehicles,
#     dist_matrix=dist_matrix,
#     demands_w=demand_w,
#     demands_v=demand_v,
#     max_capacity_w=max_capacity_w,
#     max_capacity_v=max_capacity_v,
#     fixed_cost=fixed_cost,
#     variable_cost=variable_cost,
#     max_iter=100,
#     tabu_tenure=10,
# )

# # Display results
# print("Best Solution:")
# for v, route in best_solution.items():
#     print(f"Vehicle {v}: {route}")
# print(f"Best Cost: {best_cost}")
# print(f"Best Distance:{best_distance}")
# Run Tabu Search
best_solution, best_cost, best_fixed_cost, best_variable_cost, best_distance, current_costs = tabu_search(
    nodes=nodes,
    vehicles=vehicles,
    dist_matrix=dist_matrix,
    demands_w=demand_w,
    demands_v=demand_v,
    max_capacity_w=max_capacity_w,
    max_capacity_v=max_capacity_v,
    fixed_cost=fixed_cost,
    variable_cost=variable_cost,
    max_iter=10,
    tabu_tenure=10,
)

# End measuring time
end_time = time.time()

# Calculate and print runtime
runtime = end_time - starting_time
print(f"Runtime: {runtime:.2f} seconds")
# Display results
print("Best Solution:")
for v, route in best_solution.items():
    route_distance = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    print(f"Vehicle {v}: Route: {route}, Distance: {route_distance:.2f}")
print(f"Best Total Cost: {best_cost}")
print(f"Fixed Cost: {best_fixed_cost}")
print(f"Variable Cost: {best_variable_cost}")
print(f"Total Distance: {best_distance}")

