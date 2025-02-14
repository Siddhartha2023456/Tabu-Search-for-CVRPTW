from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import itertools
import copy
import time
import numpy as np
dataset = pd.read_csv("inputs\\order_data_lat_lon_2000.csv")
vehicle_data = pd.read_csv("inputs\\VEHICLE_DATA_LAT_LON_2000.csv")
loc_data = dataset[['lat','long']]
depot = pd.DataFrame({'lat': [52.506885], 'long': [-1.728302]})
loc_data_with_depot = pd.concat([depot, loc_data], ignore_index=True)   
loc_data_with_depot = loc_data_with_depot.reset_index()
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance_km = R * c
    return int(distance_km)

# Convert distance to driving time at 30 km/h
def calculate_driving_time(distance_km):
    speed_kmh = 30  # Average speed in km/h
    time_hours = distance_km / speed_kmh
    return int(time_hours * 60)  # Convert hours to seconds
locations = loc_data_with_depot.set_index("index")[["lat", "long"]].to_dict("index")

# Initialize matrices as dictionaries
dist_matrix = {}
time_matrix = {}

# Compute distance and time matrices
for (idx1, coord1), (idx2, coord2) in itertools.combinations(locations.items(), 2):
    dist = haversine(coord1["lat"], coord1["long"], coord2["lat"], coord2["long"])
    time = calculate_driving_time(dist)
    
    dist_matrix[(idx1, idx2)] = dist
    dist_matrix[(idx2, idx1)] = dist  # Symmetric
    time_matrix[(idx1, idx2)] = time
    time_matrix[(idx2, idx1)] = time  # Symmetric

# Include self-distance as 0
for idx in locations.keys():
    dist_matrix[(idx, idx)] = 0.0
    time_matrix[(idx, idx)] = 0

# # Print matrices
# print("Distance Matrix:", dist_matrix)
# print("\nTime Matrix:", time_matrix)
nodes = list(loc_data_with_depot['index'])
vehicles = [i for i in range(len(vehicle_data))]
demands_w=[0]+dataset["weight_kg"].tolist()
Q1 = vehicle_data["Max Weight Kg"].tolist()
max_capacity_w = {i: Q1[i] for i in range(len(vehicles))}
var_cost = vehicle_data["Per Km Cost"].tolist()
fixed_cost = vehicle_data["Fixed Cost"].tolist()
start_time = [0] + (pd.to_datetime(dataset["start_time"],
                                             format='%H:%M').dt.hour * 60 + pd.to_datetime(
    dataset["start_time"],format='%H:%M').dt.minute).tolist()

finish_time = [1440] + (pd.to_datetime(dataset["end_time"],
                                             format='%H:%M').dt.hour * 60 + pd.to_datetime(
    dataset["end_time"],format='%H:%M').dt.minute).tolist()
service_time = [0] + [0 for i in range(len(dataset))]
# service_time = [0 for i in range(len(service_time))]



def is_valid_capacity(route, demands_w, max_capacity_w):
    total_weight = 0
    for node in route:
        if node != 0:  # Exclude depot
            total_weight += demands_w[node]
            if total_weight > int(max_capacity_w):
                return False
    return True

def is_valid_time_window(route, time_matrix, start_time, finish_time, service_time):
    current_time = 600  # Start at time 0
    
    for i in range(len(route) - 1):  # Traverse the route using indices
        node = route[i]
        next_node = route[i + 1]

        # Add service time at the current node before moving to the next node
        current_time += service_time[node]  

        # Travel to the next node
        current_time += time_matrix[node, next_node]

        # Wait if arriving before the start of the time window
        if current_time < start_time[next_node]:  
            current_time = start_time[next_node]

        # Check if we exceed the finish time window
        if current_time > finish_time[next_node]:
            return False
        
    return True


def is_valid_route(route, demands_w, max_capacity_w, time_matrix, start_time, finish_time,service_time):
    if not is_valid_capacity(route, demands_w, max_capacity_w):
        return False

    for i in range(len(route) - 1):
        if not is_valid_time_window(route, time_matrix, start_time, finish_time,service_time):
            return False

    return True


# def initialize_solution(nodes, vehicles, dist_matrix, demands_w, max_capacity_w, time_matrix, start_time, finish_time, service_time):
#     start = time.time()
#     solution = {v: [] for v in vehicles}
#     remaining_demand_w = copy.deepcopy(demands_w)
#     unvisited = set(nodes[1:])  # Exclude depot

#     total_cost = 0
#     total_time = 0

#     for v in vehicles:
#         current_node = 0  # Start at the depot
#         current_capacity_w = max_capacity_w[v]
#         current_time = 0  # Start at time 600
#         route = [current_node]

#         print(f"Vehicle {v} starts at depot with time {current_time}")

#         while unvisited:
#             nearest_node = None
#             nearest_distance = float("inf")

#             for n in unvisited:
#                 if remaining_demand_w[n] <= current_capacity_w and dist_matrix[current_node, n] < nearest_distance:
#                     arrival_time = current_time + time_matrix[current_node, n]
#                     if arrival_time <= finish_time[n]:
#                         nearest_node = n
#                         nearest_distance = dist_matrix[current_node, n]

#             if nearest_node is None:
#                 break  # No valid node found for this vehicle, end its route

#             # Move to the nearest node
#             arrival_time = current_time + time_matrix[current_node, nearest_node]
#             if arrival_time < start_time[nearest_node]:  # Wait for time window
#                 arrival_time = start_time[nearest_node]
#             current_time = arrival_time + service_time[nearest_node]  # Add service time

#             print(f"Vehicle {v} arrives at node {nearest_node} at time {arrival_time}, departs at {current_time}")

#             route.append(nearest_node)
#             current_capacity_w -= remaining_demand_w[nearest_node]
#             remaining_demand_w[nearest_node] = 0
#             unvisited.remove(nearest_node)
#             total_cost += dist_matrix[current_node, nearest_node]
#             total_time += time_matrix[current_node, nearest_node]
#             current_node = nearest_node

#             # Check if the vehicle can return to the depot within its time window
#             return_time = current_time + time_matrix[current_node, 0]
#             if return_time > finish_time[0]:  
#                 # Remove the last node and add it back to unvisited
#                 removed_node = route.pop()
#                 unvisited.add(removed_node)
#                 remaining_demand_w[removed_node] = demands_w[removed_node]
#                 print(f"Vehicle {v} removes node {removed_node} and returns earlier to depot.")

#                 # Adjust time and capacity after removing the last node
#                 current_node = route[-1]
#                 current_capacity_w += demands_w[removed_node]
#                 current_time -= (service_time[removed_node] + time_matrix[current_node, removed_node])

#                 break  # End route for this vehicle

#         route.append(0)  # Return to depot
#         total_cost += dist_matrix[current_node, 0]
#         total_time += time_matrix[current_node, 0]
#         solution[v] = route
#         print(f"Vehicle {v} returns to depot at time {current_time + time_matrix[current_node, 0]}")

#         if not unvisited:
#             break  # All nodes have been visited
#     f1cost = 0
#     variable1cost = 0
#     for v, route in solution.items():
#         if len(route) >2:
#             route_distance = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
#             f1cost += fixed_cost[v]
#             variable1cost += route_distance * var_cost[v]
#     end = time.time()
#     print(f"Total cost of initial solution: {f1cost + variable1cost}")
#     # print(f"Total time of initial solution: {end - start:.2f} seconds")

#     return solution

def initialize_solution(nodes, vehicles, dist_matrix, demands_w, max_capacity_w, time_matrix, start_time, finish_time, service_time):
    start = time.time()
    solution = {v: [] for v in vehicles}
    remaining_demand_w = copy.deepcopy(demands_w)
    unvisited = set(nodes[1:])  # Exclude depot
    total_cost = 0
    total_time = 0

    for v in vehicles:
        current_node = 0  # Start at the depot
        current_capacity_w = max_capacity_w[v]
        current_time = 0  # Start at time 600
        route = [current_node]

        while unvisited:
            nearest_node = None
            best_score = float("inf")  # Used for weighted selection

            for n in unvisited:
                if remaining_demand_w[n] <= current_capacity_w:
                    arrival_time = current_time + time_matrix[current_node, n]
                    if arrival_time <= finish_time[n]:
                        # Weighted score = Distance + Slack Time (to favor nodes with tighter constraints)
                        score = dist_matrix[current_node, n] + (finish_time[n] - arrival_time) * 0.1
                        if score < best_score:
                            best_score = score
                            nearest_node = n

            if nearest_node is None:
                break  # No valid node found for this vehicle, end its route

            # Move to the nearest node
            arrival_time = current_time + time_matrix[current_node, nearest_node]
            if arrival_time < start_time[nearest_node]:  # Wait for time window
                arrival_time = start_time[nearest_node]
            current_time = arrival_time + service_time[nearest_node]  # Add service time
            
            route.append(nearest_node)
            current_capacity_w -= remaining_demand_w[nearest_node]
            remaining_demand_w[nearest_node] = 0
            unvisited.remove(nearest_node)
            total_cost += dist_matrix[current_node, nearest_node]
            total_time += time_matrix[current_node, nearest_node]
            current_node = nearest_node

            # Check if the vehicle can return to the depot within its time window
            return_time = current_time + time_matrix[current_node, 0]
            if return_time > finish_time[0]:
                # Backtrack: Remove last few nodes until depot return is feasible
                while route and return_time > finish_time[0]:
                    removed_node = route.pop()
                    unvisited.add(removed_node)
                    remaining_demand_w[removed_node] = demands_w[removed_node]
                    current_node = route[-1] if route else 0
                    current_capacity_w += demands_w[removed_node]
                    current_time -= (service_time[removed_node] + time_matrix[current_node, removed_node])
                    return_time = current_time + time_matrix[current_node, 0]
                break

        route.append(0)  # Return to depot
        total_cost += dist_matrix[current_node, 0]
        total_time += time_matrix[current_node, 0]
        solution[v] = route

        if not unvisited:
            break  # All nodes have been visited
    
    # Cost Calculation
    f1cost = 0
    variable1cost = 0
    for v, route in solution.items():
        if len(route) > 2:
            route_distance = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
            f1cost += fixed_cost[v]
            variable1cost += route_distance * var_cost[v]
    
    end = time.time()
    print(f"Total cost of initial solution: {f1cost + variable1cost}")
    
    return solution

import numpy as np
from sklearn.cluster import AgglomerativeClustering

def initialize_solution_with_clustering(nodes, vehicles, dist_matrix, demands_w, max_capacity_w, 
                                        time_matrix, start_time, finish_time, service_time, n_clusters):
    start = time.time()
    
    # Extract coordinates for clustering (excluding depot)
    coords = np.array([[locations[i]['lat'], locations[i]['long']] for i in nodes if i != 0])
    
    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    cluster_labels = clustering.fit_predict(coords)
    
    # Create clusters and include the depot (node 0) in each
    clusters = {i: [0] for i in range(n_clusters)}
    for i, label in enumerate(cluster_labels):
        clusters[label].append(nodes[i + 1])  # Shift index since nodes[0] is the depot
    
    # Divide vehicles among clusters
    vehicles_per_cluster = np.array_split(vehicles, n_clusters)
    
    # Solve CVRP for each cluster
    solutions = {}
    for i in range(n_clusters):
        solutions[i] = initialize_solution(
            clusters[i], list(vehicles_per_cluster[i]), dist_matrix, demands_w, max_capacity_w, 
            time_matrix, start_time, finish_time, service_time
        )
    
    # Merge all solutions
    final_solution = {}
    for sol in solutions.values():
        final_solution.update(sol)
    
    end = time.time()
    print(f"Total cost of initial solution: {calculate_total_cost(final_solution, dist_matrix, max_capacity_w, var_cost, fixed_cost)}")
    # print(f"Total time for initialization: {end - start:.2f} seconds")
    
    return final_solution


import itertools

def generate_neighbors(solution, vehicles, nodes, tabu_list, max_capacity_w, demands_w, dist_matrix, time_matrix, start_time, finish_time, service_time):
    """
    Generate neighbors for the given solution using relocation, swap, 2-opt, merge, and split moves.
    Now includes time window feasibility checks.
    """
    neighbors = []
    route_weights = {
        v: sum(demands_w[node] for node in solution[v] if node != 0) for v in vehicles
    }
    
    def is_feasible(route):
        return is_valid_time_window(route, time_matrix, start_time, finish_time, service_time)
    
    # Relocation
    for v1, v2 in itertools.permutations(vehicles, 2):
        for i in range(1, len(solution[v1]) - 1):  # Exclude depot
            node = solution[v1][i]
            for j in range(1, len(solution[v2])):  # Allow insertions in v2
                route_v1 = solution[v1][:]
                route_v2 = solution[v2][:]
                route_v1.remove(node)
                route_v2.insert(j, node)

                new_weight_v1 = route_weights[v1] - demands_w[node]
                new_weight_v2 = route_weights[v2] + demands_w[node]

                if new_weight_v1 <= max_capacity_w[v1] and new_weight_v2 <= max_capacity_w[v2]:
                    if is_feasible(route_v1) and is_feasible(route_v2):
                        new_solution = solution.copy()
                        new_solution[v1] = route_v1
                        new_solution[v2] = route_v2
                        if new_solution not in tabu_list:
                            neighbors.append(new_solution)

    # Swap
    for v1, v2 in itertools.permutations(vehicles, 2):
        for i in range(1, len(solution[v1]) - 1):
            for j in range(1, len(solution[v2]) - 1):
                node1, node2 = solution[v1][i], solution[v2][j]
                route_v1 = solution[v1][:]
                route_v2 = solution[v2][:]
                route_v1[i], route_v2[j] = node2, node1

                new_weight_v1 = route_weights[v1] - demands_w[node1] + demands_w[node2]
                new_weight_v2 = route_weights[v2] - demands_w[node2] + demands_w[node1]

                if new_weight_v1 <= max_capacity_w[v1] and new_weight_v2 <= max_capacity_w[v2]:
                    if is_feasible(route_v1) and is_feasible(route_v2):
                        new_solution = solution.copy()
                        new_solution[v1] = route_v1
                        new_solution[v2] = route_v2
                        if new_solution not in tabu_list:
                            neighbors.append(new_solution)

    # 2-Opt
    for v in vehicles:
        route = solution[v]
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = route[:]
                new_route[i:j + 1] = reversed(new_route[i:j + 1])
                
                if is_feasible(new_route):
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
                    if is_feasible(combined_route):
                        new_solution = solution.copy()
                        new_solution[v1] = [0, 0]  # Empty route
                        new_solution[v2] = [0, 0]  # Empty route
                        new_solution[v_large] = [0] + combined_route + [0]
                        # Ensure all nodes are visited
                        visited_nodes = {node for route in new_solution.values() for node in route if node != 0}
                        if visited_nodes == set(nodes) and new_solution not in tabu_list:
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
                    if is_feasible(route_v1) and is_feasible(route_v2):
                        new_solution = solution.copy()
                        new_solution[v_large] = []  # Empty route
                        new_solution[v1] = [0] + route_v1 + [0]
                        new_solution[v2] = [0] + route_v2 + [0]
                        # Ensure all nodes are visited
                        visited_nodes = {node for route in new_solution.values() for node in route if node != 0}
                        if visited_nodes == set(nodes) and new_solution not in tabu_list:
                            neighbors.append(new_solution)
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
            if len(route) > 2: # Skip unused vehicles (routes with only the depot)
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

import time

def tabu_search(
        nodes, vehicles, dist_matrix, demands_w, max_capacity_w, Q1, var_cost, fixed_cost, max_iter, tabu_tenure,
        time_matrix, start_time, finish_time, service_time, time_limit
):
    """
    Tabu Search for minimizing total cost (fixed + variable) in a CVRPTW problem.
    Includes stopping criteria: no improvement for 3 iterations or exceeding the time limit.
    """
    
    start_time_exec = time.time()  # Track the start time

    # Initialize
    current_solution = initialize_solution(nodes, vehicles, dist_matrix, demands_w, max_capacity_w, 
                                           time_matrix, start_time, finish_time, service_time)
    best_solution = current_solution
    best_cost = calculate_total_cost(current_solution, dist_matrix, Q1, var_cost, fixed_cost)
    tabu_list = []
    tabu_queue = []
    current_costs = []  # To store the current cost in each iteration
    no_improvement_count = 0  # Counter for iterations without improvement

    for iteration in range(max_iter):
        iteration_start_time = time.time()

        # Check time limit
        if time.time() - start_time_exec > time_limit:
            print(f"Stopping early: Exceeded the time limit of {time_limit} seconds.")
            break

        # Generate neighbors
        neighbors = generate_neighbors(
            current_solution, vehicles, nodes, tabu_list, max_capacity_w, demands_w, dist_matrix,
            time_matrix=time_matrix, start_time=start_time, finish_time=finish_time, service_time=service_time
        )

        # Evaluate neighbors based on total cost
        best_neighbor = None
        best_neighbor_cost = float("inf")
        for neighbor in neighbors:
            # Validate neighbor feasibility with time window constraints
            feasible = all(
                is_valid_route(neighbor[v], demands_w, max_capacity_w[v], time_matrix, start_time, finish_time, service_time=service_time)
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
            print("Stopping early: No improvement in the last 10 iterations.")
            break

        # Update tabu list
        tabu_list.append(current_solution)
        if len(tabu_queue) >= tabu_tenure:
            tabu_list.remove(tabu_queue.pop(0))
        tabu_queue.append(current_solution)

        # Record current cost
        current_costs.append(best_cost)

        iteration_time = time.time() - iteration_start_time
        print(f"Iteration {iteration + 1}, Current Cost: {best_cost}, Time for this iteration: {iteration_time:.2f} seconds")

    total_time = time.time() - start_time_exec
    print(f"Total time in Tabu Search: {total_time:.2f} seconds")

    return best_solution, best_cost, current_costs
import time

# Start measuring total runtime
start_time_total = time.time()

best_solution, best_cost, cost_progress = tabu_search(nodes, vehicles, dist_matrix, demands_w, max_capacity_w, Q1=Q1, var_cost=var_cost, fixed_cost=fixed_cost,
    max_iter=100, tabu_tenure=10, time_matrix=time_matrix, start_time=start_time, finish_time=finish_time,service_time=service_time,time_limit=3600)
# End measuring total runtime
end_time_total = time.time()

total_runtime = end_time_total - start_time_total
print(f"Total Tabu Search Runtime: {total_runtime:.2f} seconds")
print('*' * 50)
print("Best Solution:")
distance = []
fcost = 0
variable_cost = 0
route_t = []
service_t = []

# Store vehicle arrival times
vehicle_arrival_times = {}

for v, route in best_solution.items():
    if len(route) >2:
        
    
        route_distance = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
        route_time = 0  # Track cumulative time
        serv_time = 0  # Track service time
        vehicle_arrival_times[v] = {}
        
        current_time = 600  # Start time (10:00 AM in minutes)
        
        for i in range(len(route) - 1):
            node = route[i]
            next_node = route[i + 1]
            
            # Travel time to next node
            route_time += time_matrix[node, next_node]
            current_time += time_matrix[node, next_node]
            
            # If arriving before the time window, wait
            if current_time < start_time[next_node]:
                current_time = start_time[next_node]
            
            # Store arrival time
            vehicle_arrival_times[v][next_node] = current_time
            
            # Add service time
            current_time += service_time[next_node]
            serv_time += service_time[next_node]
        
        distance.append(route_distance)
        route_t.append(route_time)
        service_t.append(serv_time)
        
        print(f"Vehicle {v}: Route: {route}, Distance: {route_distance:.2f}, Travel Time: {route_time:.2f} minutes, Service Time: {serv_time:.2f} minutes, Fixed Cost: {fixed_cost[v]}, Variable Cost: {route_distance * var_cost[v]}")
        fcost += fixed_cost[v]
        variable_cost += route_distance * var_cost[v]

print(f"Total Cost = {fcost + variable_cost}")
print(f"Total Distance = {sum(distance)}")
print(f"Total Travel Time = {sum(route_t)}")
print(f"Total Service Time = {sum(service_t)}")
print('-' * 75)
print(f"Fixed Cost: {fcost}")
print(f"Variable Cost: {variable_cost}")

# # Print arrival times for each vehicle
# print("\nVehicle Arrival Times:")
# for v, arrival_times in vehicle_arrival_times.items():
#     print(f"Vehicle {v}:")
#     for loc, time in arrival_times.items():
#         print(f"  Arrives at location {loc} at time {time} minutes")