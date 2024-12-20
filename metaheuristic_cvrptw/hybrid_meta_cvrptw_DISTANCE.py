import json
import pandas as pd
import random
# Path Cheapest Arc Initialization
def path_cheapest_arc(nodes, depot, vehicles, dist_matrix, demand_w, demand_v, max_vehw, max_vehv, start_time, finish_time):
    vehicle_routes = {k: [(depot, depot)] for k in vehicles}  # Start and end at depot
    remaining_capacity_w = {k: max_vehw[k] for k in vehicles}
    remaining_capacity_v = {k: max_vehv[k] for k in vehicles}
    remaining_time = {k: 0 for k in vehicles}

    unvisited_customers = set(nodes) - {depot}

    for vehicle in vehicles:
        current_node = depot
        while unvisited_customers:
            # Find the cheapest feasible arc
            next_node = None
            min_cost = float('inf')
            for customer in unvisited_customers:
                if (
                    dist_matrix[current_node, customer] < min_cost
                    and demand_w[customer] <= remaining_capacity_w[vehicle]
                    and demand_v[customer] <= remaining_capacity_v[vehicle]
                    and remaining_time[vehicle] + dist_matrix[current_node, customer] <= finish_time[customer]
                ):
                    min_cost = dist_matrix[current_node, customer]
                    next_node = customer

            if next_node is None:
                break  # No feasible customers left for this vehicle

            # Update vehicle route and constraints
            vehicle_routes[vehicle].append((current_node, next_node))
            remaining_capacity_w[vehicle] -= demand_w[next_node]
            remaining_capacity_v[vehicle] -= demand_v[next_node]
            remaining_time[vehicle] += dist_matrix[current_node, next_node]
            current_node = next_node
            unvisited_customers.remove(next_node)

        # Return to the depot if the route is not already at the depot
        if vehicle_routes[vehicle][-1][1] != depot:
            vehicle_routes[vehicle].append((current_node, depot))

    return vehicle_routes
# from scipy.sparse.csgraph import minimum_spanning_tree
# import networkx as nx
# from itertools import combinations

# import numpy as np
# from scipy.sparse.csgraph import minimum_spanning_tree
# import networkx as nx

# # Convert distance dictionary to 2D array
# def dist_dict_to_matrix(dist_dict, num_nodes):
#     dist_matrix = np.full((num_nodes, num_nodes), np.inf)  # Initialize with large values (infinity)
#     for (i, j), dist in dist_dict.items():
#         dist_matrix[i][j] = dist
#     return dist_matrix

# import networkx as nx
# import numpy as np
# from scipy.sparse.csgraph import minimum_spanning_tree

# def christofides(nodes, depot, vehicles, dist_matrix_dict, demand_w, demand_v, max_vehw, max_vehv):
#     num_nodes = len(nodes)
    
#     # Convert dictionary to 2D distance matrix
#     dist_matrix = dist_dict_to_matrix(dist_matrix_dict, num_nodes)
    
#     # Step 1: Solve MST using the distance matrix
#     mst = minimum_spanning_tree(dist_matrix).toarray().astype(float)

#     # Step 2: Find odd degree vertices in MST
#     G = nx.from_numpy_array(mst)
#     odd_degree_nodes = [node for node, degree in G.degree if degree % 2 == 1]
#     print(f"Odd-degree nodes before matching: {odd_degree_nodes}")

#     # Ensure there is an even number of odd-degree nodes
#     if len(odd_degree_nodes) % 2 != 0:
#         raise ValueError("Odd-degree nodes count is not even. Debug required.")

#     # Step 3: Manually adjust the matching of odd-degree nodes
#     matching = {}
#     while odd_degree_nodes:
#         node1 = odd_degree_nodes.pop(0)
#         closest_node = min(odd_degree_nodes, key=lambda node: dist_matrix[node1, node])
#         matching[node1] = closest_node
#         matching[closest_node] = node1
#         odd_degree_nodes.remove(closest_node)

#     print(f"Manual Matching: {matching}")

#     # Step 4: Add the matching edges to the MST graph (ensure bidirectional updates)
#     for u, v in matching.items():
#         G.add_edge(u, v, weight=dist_matrix[u, v])
#         G.add_edge(v, u, weight=dist_matrix[v, u])  # Add the reverse direction as well

#     # Check degrees after adding matching edges
#     print("Graph degrees after matching:", dict(G.degree))

#     # Validate the graph is Eulerian
#     if not nx.is_eulerian(G):
#         print("Graph is not Eulerian after matching.")
#         print(f"Graph degrees: {dict(G.degree)}")
#         raise ValueError("Graph is not Eulerian after matching step. Debug required.")

#     # Step 5: Compute Eulerian Circuit
#     eulerian_circuit = list(nx.eulerian_circuit(G))

#     # Step 6: Convert Eulerian Circuit to Hamiltonian Path (remove repeated nodes)
#     path = []
#     visited = set()
#     for u, v in eulerian_circuit:
#         if u not in visited:
#             path.append(u)
#             visited.add(u)
#     path.append(depot)  # Ensure it ends at the depot

#     # Step 7: Split path into vehicle routes based on capacities
#     vehicle_routes = {vehicle: [] for vehicle in vehicles}
#     current_route = []
#     remaining_capacity_w = max_vehw[vehicles[0]]
#     remaining_capacity_v = max_vehv[vehicles[0]]
#     vehicle_index = 0

#     for node in path:
#         if (
#             node != depot
#             and (demand_w[node] > remaining_capacity_w or demand_v[node] > remaining_capacity_v)
#         ):
#             # Assign current route to a vehicle
#             vehicle_routes[vehicles[vehicle_index]] = current_route + [(current_route[-1], depot)]
#             vehicle_index += 1
#             if vehicle_index >= len(vehicles):
#                 raise ValueError("Not enough vehicles to satisfy demand.")
#             current_route = [(depot, node)]
#             remaining_capacity_w = max_vehw[vehicles[vehicle_index]] - demand_w[node]
#             remaining_capacity_v = max_vehv[vehicles[vehicle_index]] - demand_v[node]
#         else:
#             if current_route:
#                 current_route.append((current_route[-1][1], node))
#             else:
#                 current_route.append((depot, node))
#             remaining_capacity_w -= demand_w[node]
#             remaining_capacity_v -= demand_v[node]

#     # Assign last route to vehicle
#     if current_route:
#         vehicle_routes[vehicles[vehicle_index]] = current_route + [(current_route[-1][1], depot)]

#     return vehicle_routes








# Calculate Total Distance
def calculate_total_distance(routes, dist_matrix):
    total_distance = 0
    for vehicle, route in routes.items():
        total_distance += sum(dist_matrix[i, j] for i, j in route)
    return total_distance

# # Tabu Search
# def tabu_search(initial_routes, dist_matrix, demand_w, demand_v, max_vehw, max_vehv, iterations=100, tabu_tenure=10):
#     current_solution = initial_routes
#     best_solution = initial_routes
#     tabu_list = set()
#     best_cost = calculate_total_distance(current_solution, dist_matrix)

#     for it in range(iterations):
#         neighborhood = generate_neighbors(current_solution, demand_w, demand_v, max_vehw, max_vehv, dist_matrix)
#         best_neighbor = None
#         best_neighbor_cost = float('inf')

#         for neighbor in neighborhood:
#             move = neighbor["move"]
#             if move not in tabu_list:
#                 cost = calculate_total_distance(neighbor["routes"], dist_matrix)
#                 if cost < best_neighbor_cost:
#                     best_neighbor = neighbor
#                     best_neighbor_cost = cost

#         if best_neighbor:
#             current_solution = best_neighbor["routes"]
#             tabu_list.add(best_neighbor["move"])
#             if len(tabu_list) > tabu_tenure:
#                 tabu_list.pop()

#             if best_neighbor_cost < best_cost:
#                 best_solution = current_solution
#                 best_cost = best_neighbor_cost

#         print(f"Iteration {it + 1}: Best Cost = {best_cost}")

#     return best_solution, best_cost
# Enhanced Tabu Search
def enhanced_tabu_search(
    initial_routes, dist_matrix, demand_w, demand_v, max_vehw, max_vehv, 
    iterations=10, tabu_tenure=10, intensify_threshold=20, diversify_threshold=50
):
    current_solution = initial_routes
    best_solution = initial_routes
    tabu_list = set()
    best_cost = calculate_total_distance(current_solution, dist_matrix)
    current_cost = best_cost
    stagnation_counter = 0

    for it in range(iterations):
        neighborhood = generate_neighbors(
            current_solution, demand_w, demand_v, max_vehw, max_vehv, dist_matrix
        )
        best_neighbor = None
        best_neighbor_cost = float('inf')
        diversify_moves = []

        for neighbor in neighborhood:
            move = neighbor["move"]
            cost = calculate_total_distance(neighbor["routes"], dist_matrix)

            # Check aspiration criteria: Allow a tabu move if it improves the best solution
            if move in tabu_list and cost >= best_cost:
                continue

            if cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = cost

            # Track moves for diversification
            if cost > current_cost and move not in tabu_list:
                diversify_moves.append(neighbor)

        # If no valid neighbor is found, select a move to diversify
        if best_neighbor is None and diversify_moves:
            best_neighbor = random.choice(diversify_moves)
            best_neighbor_cost = calculate_total_distance(best_neighbor["routes"], dist_matrix)

        if best_neighbor:
            current_solution = best_neighbor["routes"]
            tabu_list.add(best_neighbor["move"])
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop()

            current_cost = best_neighbor_cost
            stagnation_counter = stagnation_counter + 1 if current_cost >= best_cost else 0

            # Update best solution
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
                stagnation_counter = 0

        # Intensification: Restart search from best solution if no improvement for some iterations
        if stagnation_counter >= intensify_threshold:
            current_solution = best_solution
            stagnation_counter = 0
            print(f"Iteration {it + 1}: Intensification triggered")

        # Diversification: Randomize solution if stagnation persists beyond diversify threshold
        if stagnation_counter >= diversify_threshold:
            current_solution = randomize_solution(best_solution, demand_w, demand_v, max_vehw, max_vehv)
            stagnation_counter = 0
            print(f"Iteration {it + 1}: Diversification triggered")

        print(f"Iteration {it + 1}: Current Cost = {current_cost}, Best Cost = {best_cost}")

    return best_solution, best_cost

# Randomize Solution for Diversification
def randomize_solution(solution, demand_w, demand_v, max_vehw, max_vehv):
    randomized_solution = solution.copy()
    vehicles = list(randomized_solution.keys())

    for vehicle in vehicles:
        if len(randomized_solution[vehicle]) > 2:  # Avoid empty or single-node routes
            route = randomized_solution[vehicle]
            random.shuffle(route[1:-1])  # Shuffle nodes, keeping depot at start and end
            randomized_solution[vehicle] = route

    return randomized_solution


# Generate Neighbors
def generate_neighbors(routes, demand_w, demand_v, max_vehw, max_vehv, dist_matrix):
    neighbors = []
    vehicles = list(routes.keys())
    for v1 in vehicles:
        for v2 in vehicles:
            for i in range(len(routes[v1])):
                for j in range(len(routes[v2])):
                    # Intra-route or Inter-route swap
                    if v1 == v2:  # Intra-route
                        new_routes = intra_route_swap(routes, v1, i, j)
                    # else:  # Inter-route
                    #     new_routes = inter_route_swap(routes, v1, v2, i, j, demand_w, demand_v, max_vehw, max_vehv)

                    if new_routes:
                        neighbors.append({"routes": new_routes, "move": (v1, v2, i, j)})
    return neighbors

# Intra-route Swap
def intra_route_swap(routes, vehicle, i, j):
    if i >= j:  # Ensure i < j for valid swap
        return None
    new_routes = routes.copy()
    route = routes[vehicle].copy()
    route[i], route[j] = route[j], route[i]
    new_routes[vehicle] = route
    return new_routes

# Inter-route Swap
# def inter_route_swap(routes, v1, v2, i, j, demand_w, demand_v, max_vehw, max_vehv):
#     route1 = routes[v1].copy()
#     route2 = routes[v2].copy()
#     if i >= len(route1) or j >= len(route2):
#         return None

#     node1 = route1[i][1]
#     node2 = route2[j][1]

#     # Avoid self-loops and ensure depot constraints are met
#     if node1 == node2 or node1 == 0 or node2 == 0:
#         return None

#     # Check feasibility
#     if (
#         demand_w[node2] - demand_w[node1] > max_vehw[v1]
#         or demand_v[node2] - demand_v[node1] > max_vehv[v1]
#         or demand_w[node1] - demand_w[node2] > max_vehw[v2]
#         or demand_v[node1] - demand_v[node2] > max_vehv[v2]
#     ):
#         return None

#     # Perform swap
#     route1[i] = (route1[i][0], node2)
#     route2[j] = (route2[j][0], node1)
#     new_routes = routes.copy()
#     new_routes[v1] = route1
#     new_routes[v2] = route2

#     # Ensure routes still start and end at the depot
#     if new_routes[v1][0][0] != 0 or new_routes[v2][0][0] != 0:
#         return None
#     if new_routes[v1][-1][1] != 0 or new_routes[v2][-1][1] != 0:
#         return None

#     return new_routes


# Main Execution
if __name__ == "__main__":
    file_path = 'inputs/solver_params_26.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
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
    start_time = {idx: tw[0] for idx, tw in enumerate(time_windows)}
    finish_time = {idx: tw[1] for idx, tw in enumerate(time_windows)}
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
    df_vehicle = pd.DataFrame([max_veh_weight,max_veh_volume]).transpose().reset_index()
    df_vehicle.columns = ["v_id","max_weight","max_volume"]
    vehicles = list(df_vehicle["v_id"])
    demand_w = dict(df_orders_f["order_weight"])
    demand_v = dict(df_orders_f["order_volume"])
    max_vehw =dict(df_vehicle["max_weight"])
    max_vehv =dict(df_vehicle["max_volume"])
    # Step 1: Generate Initial Solution
    initial_routes = path_cheapest_arc(nodes, depot, vehicles, dist_matrix, demand_w, demand_v, max_vehw, max_vehv,start_time,finish_time)

    # Step 2: Optimize Using Tabu Search
    best_solution, best_cost = enhanced_tabu_search(initial_routes, dist_matrix, demand_w, demand_v, max_vehw, max_vehv)

    print("\nFinal Solution:")
    print(best_solution)
    print(f"Best Cost: {best_cost}")
