import copy
import time
import pandas as pd
import numpy as np
import itertools
import cProfile

# Start measuring time
starting_time = time.time()

def is_valid_capacity(route, demands_w, max_capacity_w):
    total_weight = 0
    for node in route:
        if node != 0:  # Exclude depot
            total_weight += demands_w[node]
            if total_weight > int(max_capacity_w):
                return False
    return True

def is_valid_time_window(route, time_matrix, start_time, finish_time,service_time):
    current_time = 0  # Start at time 0
    for i in range(len(route) - 1):  # Traverse the route using indices
        node = route[i]
        next_node = route[i + 1]

        # Update the current time to simulate travel and waiting
        current_time += time_matrix[node, next_node]
        if current_time < start_time[next_node]:  # Wait for time window
            current_time = start_time[next_node]

        # Check if we are within the time window
        if current_time + service_time > finish_time[next_node]:
            return False
    return True

def is_valid_route(route, demands_w, max_capacity_w, time_matrix, start_time, finish_time,service_time):
    if not is_valid_capacity(route, demands_w, max_capacity_w):
        return False

    for i in range(len(route) - 1):
        next_node = route[i + 1]
        if (finish_time[next_node] - start_time[next_node]) < 560:
            if not is_valid_time_window(route, time_matrix, start_time, finish_time,service_time):
                return False

    return True


def initialize_solution(nodes, vehicles, dist_matrix, demands_w, max_capacity_w, time_matrix, start_time, finish_time):
    s_t = time.time()
    solution = {v: [] for v in vehicles}
    remaining_demand_w = copy.deepcopy(demands_w)
    unvisited = set(nodes[1:])  # Exclude depot

    for v in vehicles:
        current_node = 0  # Start at the depot
        current_capacity_w = max_capacity_w[v]
        current_time = 0  # Start at time 0
        route = [current_node]

        while unvisited:
            # Find the nearest feasible node
            nearest_node = None
            nearest_distance = float("inf")
            for n in unvisited:
                # Check capacity and time window feasibility
                if remaining_demand_w[n] <= current_capacity_w and dist_matrix[current_node, n] < nearest_distance:
                    # Simulate arrival time
                    arrival_time = current_time + time_matrix[current_node, n]
                    if arrival_time <= finish_time[n]:
                        nearest_node = n
                        nearest_distance = dist_matrix[current_node, n]

            if nearest_node is None:
                break  # No valid node found for this vehicle, end its route

            # Add the nearest node to the route
            route.append(nearest_node)
            current_capacity_w -= remaining_demand_w[nearest_node]
            remaining_demand_w[nearest_node] = 0
            unvisited.remove(nearest_node)
            current_time += time_matrix[current_node, nearest_node]
            if current_time < start_time[nearest_node]:  # Wait for the time window to open
                current_time = start_time[nearest_node]
            current_node = nearest_node

        route.append(0)  # Return to depot
        solution[v] = route

        # Stop if all customers have been visited
        if not unvisited:
            break

    
    e_t = time.time()
    run_time = e_t - s_t
    print(f"Time for initial solution: {run_time:.2f} seconds")
    return solution

# SAVINGS ALGO
# def initialize_solution(nodes, vehicles, dist_matrix, demands_w, max_capacity_w, time_matrix, start_time, finish_time):
#     s_t = time.time()
#     solution = {v: [0, 0] for v in vehicles}  # Initialize each vehicle route with depot start and end
#     remaining_demand_w = copy.deepcopy(demands_w)
#     unvisited = set(nodes[1:])  # Exclude depot

#     # Step 1: Calculate savings for all pairs of customers
#     savings = []
#     for i in unvisited:
#         for j in unvisited:
#             if i != j:
#                 saving = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
#                 savings.append((saving, i, j))
#     savings.sort(reverse=True, key=lambda x: x[0])  # Sort by savings in descending order

#     # Step 2: Build routes using the savings
#     routes = {n: [n] for n in unvisited}  # Initially, each customer is its own route
#     capacities = {n: demands_w[n] for n in unvisited}

#     for saving, i, j in savings:
#         if i in routes and j in routes and routes[i] != routes[j]:
#             # Check if merging routes is feasible
#             if capacities[i] + capacities[j] <= max_capacity_w[max(vehicles, key=lambda v: max_capacity_w[v])]:
#                 # Merge routes
#                 if routes[i][-1] == i and routes[j][0] == j:
#                     new_route = routes[i] + routes[j]
#                 elif routes[j][-1] == j and routes[i][0] == i:
#                     new_route = routes[j] + routes[i]
#                 else:
#                     continue

#                 # Update routes and capacities
#                 for node in new_route:
#                     routes[node] = new_route
#                 capacities[i] += capacities[j]
#                 capacities[j] = capacities[i]

#     # Step 3: Assign merged routes to vehicles
#     vehicle_index = 0
#     for route in set(tuple(r) for r in routes.values()):
#         if vehicle_index >= len(vehicles):
#             break

#         # Ensure the route starts and ends at the depot
#         full_route = [0] + list(route) + [0]

#         # Check time window feasibility
#         current_time = 0
#         feasible = True
#         for k in range(len(full_route) - 1):
#             current_time += time_matrix[full_route[k], full_route[k + 1]]
#             if current_time < start_time[full_route[k + 1]]:
#                 current_time = start_time[full_route[k + 1]]
#             if current_time > finish_time[full_route[k + 1]]:
#                 feasible = False
#                 break

#         if feasible:
#             solution[vehicles[vehicle_index]] = full_route
#             vehicle_index += 1

#     # Step 4: Assign remaining unvisited nodes to any vehicle with capacity left
#     for n in unvisited:
#         for v in vehicles:
#             if sum(demands_w[node] for node in solution[v] if node != 0) + demands_w[n] <= max_capacity_w[v]:
#                 solution[v].insert(-1, n)  # Add before returning to depot
#                 break

#     e_t = time.time()
#     run_time = e_t - s_t
#     print(f"Time for initial solution with savings algorithm: {run_time:.2f} seconds")
#     return solution


# def generate_neighbors(solution, vehicles, nodes, tabu_list, max_capacity_w, demands_w, dist_matrix):
#     """
#     Generate neighbors for the given solution using relocation, swap, and 2-opt moves.
#     Optimized for reduced runtime.
#     """
#     s_t = time.time()
#     neighbors = []

#     # Cache current weights for each vehicle
#     route_weights = {
#         v: sum(demands_w[node] for node in solution[v] if node != 0) for v in vehicles
#     }
    
#     # Relocation: Move a customer from one vehicle to another
#     for v1, v2 in itertools.permutations(vehicles, 2):
#         for i in range(1, len(solution[v1]) - 1):  # Exclude depot
#             node = solution[v1][i]
#             for j in range(1, len(solution[v2])):  # Allow insertions in v2
#                 # Modify routes incrementally
#                 route_v1 = solution[v1][:]
#                 route_v2 = solution[v2][:]
#                 route_v1.remove(node)
#                 route_v2.insert(j, node)

#                 # Incremental weight checks
#                 new_weight_v1 = route_weights[v1] - demands_w[node]
#                 new_weight_v2 = route_weights[v2] + demands_w[node]

#                 if new_weight_v1 <= max_capacity_w[v1] and new_weight_v2 <= max_capacity_w[v2]:
#                     new_solution = solution.copy()
#                     new_solution[v1] = route_v1
#                     new_solution[v2] = route_v2
#                     if new_solution not in tabu_list:
#                         neighbors.append(new_solution)

#     # Swap: Swap two customers between two different vehicles
#     for v1, v2 in itertools.permutations(vehicles, 2):
#         for i in range(1, len(solution[v1]) - 1):
#             for j in range(1, len(solution[v2]) - 1):
#                 node1, node2 = solution[v1][i], solution[v2][j]

#                 # Modify routes incrementally
#                 route_v1 = solution[v1][:]
#                 route_v2 = solution[v2][:]
#                 route_v1[i], route_v2[j] = node2, node1

#                 # Incremental weight checks
#                 new_weight_v1 = route_weights[v1] - demands_w[node1] + demands_w[node2]
#                 new_weight_v2 = route_weights[v2] - demands_w[node2] + demands_w[node1]

#                 if new_weight_v1 <= max_capacity_w[v1] and new_weight_v2 <= max_capacity_w[v2]:
#                     new_solution = solution.copy()
#                     new_solution[v1] = route_v1
#                     new_solution[v2] = route_v2
#                     if new_solution not in tabu_list:
#                         neighbors.append(new_solution)

#     # 2-Opt: Reverse a subsequence in a single route
#     for v in vehicles:
#         route = solution[v]
#         for i in range(1, len(route) - 2):  # Exclude depot
#             for j in range(i + 1, len(route) - 1):  # Ensure valid subsequence
#                 new_route = route[:]
#                 new_route[i:j + 1] = reversed(new_route[i:j + 1])

#                 # Incremental weight check (no weight change for 2-opt)
#                 if route_weights[v] <= max_capacity_w[v]:
#                     new_solution = solution.copy()
#                     new_solution[v] = new_route
#                     if new_solution not in tabu_list:
#                         neighbors.append(new_solution)

#     e_t = time.time()
#     run_time = e_t - s_t
#     print(f"Time for generating neighbors: {run_time:.4f} seconds")
#     return neighbors

def generate_neighbors(solution, vehicles, nodes, tabu_list, max_capacity_w, demands_w, dist_matrix):
    """
    Generate neighbors for the given solution using relocation, swap, 2-opt moves, 
    and merging routes of smaller vehicles into a larger vehicle.
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

    # Merge routes of two smaller vehicles into a larger vehicle
    for v1, v2, v_large in itertools.permutations(vehicles, 3):
        if len(solution[v1]) > 2 and len(solution[v2]) > 2:
            if max_capacity_w[v_large] >= (max_capacity_w[v1] + max_capacity_w[v2]):
                combined_route = solution[v1][1:-1] + solution[v2][1:-1]  # Exclude depots
                combined_weight = route_weights[v1] + route_weights[v2]

                if combined_weight <= max_capacity_w[v_large]:
                    new_solution = solution.copy()
                    new_solution[v1] = [0, 0]  # Empty route
                    new_solution[v2] = [0, 0]  # Empty route
                    new_solution[v_large] = [0] + combined_route + [0]
                    # Ensure all nodes are visited
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

                if weight_v1 <= max_capacity_w[v1] and weight_v2 <= max_capacity_w[v2]:
                    new_solution = solution.copy()
                    new_solution[v_large] = []  # Empty route
                    new_solution[v1] = [0] + route_v1 + [0]
                    new_solution[v2] = [0] + route_v2 + [0]
                    # Ensure all nodes are visited
                    visited_nodes = {node for route in new_solution.values() for node in route if node != 0}
                    if visited_nodes == set(nodes) and new_solution not in tabu_list:
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
        if len(route) > 2:  # Skip empty or single-point routes
            total_distance += sum(
                dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)
            )
    return total_distance

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
            if len(route) > 2:  # Skip unused vehicles (routes with only the depot)
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
        nodes, vehicles, dist_matrix, demands_w, max_capacity_w, Q1, var_cost, fixed_cost, max_iter, tabu_tenure,
        time_matrix, start_time, finish_time, time_limit
):
    """
    Tabu Search for minimizing total cost (fixed + variable) in a CVRPTW problem.
    Includes stopping criteria: no improvement for 3 iterations or exceeding the time limit.
    """
    st_time = time.time()

    # Initialize
    current_solution = initialize_solution(nodes, vehicles, dist_matrix, demands_w, max_capacity_w, 
                                           time_matrix, start_time, finish_time)
    best_solution = current_solution
    best_cost = calculate_total_cost(current_solution, dist_matrix, Q1, var_cost, fixed_cost)
    tabu_list = []
    tabu_queue = []
    current_costs = []  # To store the current cost in each iteration
    no_improvement_count = 0  # Counter for iterations without improvement

    for iteration in range(max_iter):
        s_t = time.time()

        # Check time limit
        if time.time() - st_time > time_limit:
            print(f"Stopping early: Exceeded the time limit of {time_limit} seconds.")
            break

        # Generate neighbors
        neighbors = generate_neighbors(
            current_solution, vehicles, nodes, tabu_list, max_capacity_w, demands_w, dist_matrix
        )

        # Evaluate neighbors based on total cost
        best_neighbor = None
        best_neighbor_cost = float("inf")
        for neighbor in neighbors:
            # Validate neighbor feasibility with time window constraints
            feasible = all(
                is_valid_route(neighbor[v], demands_w, max_capacity_w[v], time_matrix, start_time, finish_time)
                for v in vehicles
            )

            if feasible:
                neighbor_cost = calculate_total_cost(neighbor, dist_matrix, Q1, var_cost, fixed_cost)
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost

        # Update current solution if a better neighbor is found
        if best_neighbor and best_neighbor_cost < best_cost:
            current_solution = best_neighbor
            best_cost = best_neighbor_cost
            best_solution = current_solution
            no_improvement_count = 0 
        else:
            no_improvement_count += 1

        # Early stopping condition for no improvement
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
locations_df = pd.read_csv("C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/locations.csv")
order_list_df = pd.read_excel('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/order_list_1.xlsx')
travel_matrix_df = pd.read_csv('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/travel_matrix.csv')
trucks_df = pd.read_csv('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/trucks.csv')
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
print(len(nodes))
print(len(demands_w))
print(max_capacity_w)
print(Q1)
print(len(start_time))
best_solution, best_cost, cost_progress = tabu_search(
    nodes, vehicles, dist_matrix, demands_w, max_capacity_w, Q1=Q1, var_cost=var_cost, fixed_cost=fixed_cost,
    max_iter=100, tabu_tenure=10, time_matrix=time_matrix, start_time=start_time, finish_time=finish_time,time_limit=60
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
fcost = 0
for v, route in best_solution.items():
    route_distance = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    route_time = sum(time_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    distance.append(route_distance)
    print(f"Vehicle {v}: Route: {route}, Distance: {route_distance:.2f}, Time: {route_time:.2f}, fixed cost:{fixed_cost[v]}")
    if len(route)>2:    
        fcost += fixed_cost[v]
# print(f"Total Distance: {best_distance}")
print(f"Total cost = {best_cost}")
print(f"Total distance = {sum(distance)}")
print('-'*75)
print(f" Fixed Cost :{fcost}")
print(f" Variable Cost :{best_cost - fcost}")

# profiler = cProfile.Profile()

# # Profile the code block
# profiler.enable()
# tabu_search(
#     nodes, vehicles, dist_matrix, demands_w, max_capacity_w, Q1=Q1, var_cost=var_cost, fixed_cost=fixed_cost,
#     max_iter=100, tabu_tenure=10, time_matrix=time_matrix, start_time=start_time, finish_time=finish_time
# )
#  # Call your connected functions
# profiler.disable()

# # Print profiling results
# profiler.print_stats(sort='time')
# print("Best Solution:")
# distance = []
# fcost = 0
# for v, route in best_solution.items():
#     route_distance = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
#     route_time = sum(time_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
#     distance.append(route_distance)
#     print(f"Vehicle {v}:")
#     current_time = 0  # Start time for the vehicle
#     for i in range(len(route) - 1):
#         location = route[i]
#         next_location = route[i + 1]
#         travel_time = time_matrix[location, next_location]
#         current_time += travel_time
#         if current_time < start_time[next_location]:
#             current_time = start_time[next_location]  # Wait for time window to open
#         print(f"  Location {location}: Arrival Time: {current_time} minutes")
#     print(f"  Route: {route}, Distance: {route_distance:.2f}, Time: {route_time:.2f}, Fixed Cost: {fixed_cost[v]}")
#     if len(route) > 2:
#         fcost += fixed_cost[v]
# print(f"Total cost = {best_cost}")
# print(f"Total distance = {sum(distance)}")
# print('-' * 75)
# print(f" Fixed Cost :{fcost}")
# print(f" Variable Cost :{best_cost - fcost}")
