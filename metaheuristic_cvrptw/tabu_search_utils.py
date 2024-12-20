import copy
import time
import pandas as pd
import numpy as np
import itertools
from config import PARAMETERS
def initialize_solution(nodes, vehicles, dist_matrix, demands_w, max_capacity_w):

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
    Tabu Search with advanced LNS to escape local optima for minimizing total cost (fixed + variable) in a CVRPTW problem.
    """
    def large_neighborhood_search(solution):
        """
        Perform an advanced perturbation on the solution by removing and reinserting customers.
        """
        perturbed_solution = copy.deepcopy(solution)

        # Remove a subset of customers randomly
        all_customers = [node for v in vehicles for node in solution[v] if node != 0]
        num_to_remove = max(1, int(len(all_customers)/(10*PARAMETERS["removal_fraction"])))  # Remove 20% of customers
        removed_customers = set(np.random.choice(all_customers, num_to_remove, replace=False))

        # Remove from routes
        for v in vehicles:
            perturbed_solution[v] = [node for node in perturbed_solution[v] if node not in removed_customers]

        # Reinsert customers using a greedy approach
        for customer in removed_customers:
            best_vehicle, best_position, best_cost = None, None, float("inf")
            for v in vehicles:
                for pos in range(1, len(perturbed_solution[v])):  # Insert between any two nodes
                    new_route = perturbed_solution[v][:]
                    new_route.insert(pos, customer)

                    # Check feasibility
                    if is_valid_route(new_route, demands_w, max_capacity_w[v]):
                        new_cost = calculate_total_distance({v: new_route}, dist_matrix)
                        if new_cost < best_cost:
                            best_vehicle, best_position, best_cost = v, pos, new_cost

            if best_vehicle is not None:
                perturbed_solution[best_vehicle].insert(best_position, customer)

        return perturbed_solution

    def run_tabu_search(initial_solution, max_iter):
        """Run the core Tabu Search logic."""
        current_solution = initial_solution
        best_solution = current_solution
        best_cost = calculate_total_cost(current_solution, dist_matrix, Q1, var_cost, fixed_cost)
        tabu_list = []
        tabu_queue = []
        no_improvement_count = 0
        current_costs = []

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
                no_improvement_count = 0 
            else:
                no_improvement_count += 1

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

            # Stop if no improvement for 3 iterations
            if no_improvement_count >= PARAMETERS["stoppage criteria_no_improvement_count"]:
                break

        return best_solution, best_cost, current_costs

    st_time = time.time()

    # Initialize
    current_solution = initialize_solution(nodes, vehicles, dist_matrix, demands_w, max_capacity_w)
    best_solution = current_solution
    best_cost = calculate_total_cost(current_solution, dist_matrix, Q1, var_cost, fixed_cost)
    all_costs = []

    # Repeat the logic 3 times
    for phase in range(PARAMETERS["Diversification_logic_run_count"]):
        print(f"Starting Phase {phase + 1}...")

        # Run Tabu Search for max_iter
        current_solution, best_cost, costs = run_tabu_search(current_solution, max_iter)
        all_costs.extend(costs)

        # Apply LNS if no improvement for 3 iterations
        if phase < 2:  # Avoid LNS in the last phase
            print(f"Applying advanced LNS after Phase {phase + 1}.")
            perturbed_solution = large_neighborhood_search(current_solution)
            perturbed_cost = calculate_total_cost(perturbed_solution, dist_matrix, Q1, var_cost, fixed_cost)
            print(f"Cost after LNS: {perturbed_cost}")

            if perturbed_cost < best_cost:
                current_solution = perturbed_solution
                best_cost = perturbed_cost
                best_solution = current_solution

        # Use the best solution as the initial solution and run for 40 iterations
        current_solution, best_cost, costs = run_tabu_search(best_solution, 100)
        all_costs.extend(costs)

    en_time = time.time()
    total_time = en_time - st_time
    print(f"Total time in Tabu Search: {total_time:.2f} seconds")

    return best_solution, best_cost, all_costs