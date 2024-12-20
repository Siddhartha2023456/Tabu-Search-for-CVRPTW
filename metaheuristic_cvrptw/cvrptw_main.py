from config import PARAMETERS,load_data,preprocess_data
from tabu_search_utils import tabu_search

# Example usage:
locations_df, order_list_df, travel_matrix_df, trucks_df = load_data()
nodes, vehicles, dist_matrix, demands_w, max_capacity_w, Q1, var_cost, fixed_cost = preprocess_data(
    locations_df, order_list_df, travel_matrix_df, trucks_df)

# Call the Tabu Search function
best_solution, best_cost, cost_progress = tabu_search(
    nodes, vehicles, dist_matrix, demands_w, max_capacity_w, Q1=Q1, var_cost=var_cost, fixed_cost=fixed_cost,
    max_iter=PARAMETERS["max_iter"], tabu_tenure=PARAMETERS["tabu_tenure"],
)
print("Best Solution:")
distance = []
fcost = 0
for v, route in best_solution.items():
    route_distance = sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    distance.append(route_distance)
    print(f"Vehicle {v}: Route: {route}, Distance: {route_distance:.2f},fixed cost:{fixed_cost[v]}")
    if route:    
        fcost += fixed_cost[v]
# print(f"Total Distance: {best_distance}")
print(f"Total cost = {best_cost}")
print(f"Total distance = {sum(distance)}")
print('-'*75)
print(f" Fixed Cost :{fcost}")
print(f" Variable Cost :{best_cost - fcost}")
