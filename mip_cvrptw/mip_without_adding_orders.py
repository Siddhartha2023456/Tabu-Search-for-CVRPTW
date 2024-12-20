# Rewritten code from cell 1
# import json

# # Specify the path to the JSON file
# file_path = 'inputs/solver_params_26.json'
    

# # Open and load the JSON file
# with open(file_path, 'r') as file:
#     data = json.load(file)

# # Print the loaded data
# print(data)

data = {'numNodes': 8, 'loc_ids': ['PLC-9e64068d-0315-40f2-86b8-f1127a31981a', 'PLC-d22fba1b-b955-4030-b995-94fd0658ddd1', 'PLC-b4276938-dc67-4d61-a977-88aa5fc512e4', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-7b034c4e-257b-4d2a-814f-8567c6370f64', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e'], 'costs': [[0, 68.262, 58.493, 234.233, 157.275, 230.117, 656.132, 108.679], [68.262, 0, 24.302, 166.294, 95.279, 162.127, 703.504, 68.121], [58.493, 24.302, 0, 178.583, 98.81, 174.631, 679.801, 55.353], [234.233, 166.294, 178.583, 0, 90.373, 4.425, 819.063, 153.952], [157.275, 95.279, 98.81, 90.373, 0, 87.391, 731.915, 63.876], [230.117, 162.127, 174.631, 4.425, 87.391, 0, 816.933, 150.779], [656.132, 703.504, 679.801, 819.063, 731.915, 816.933, 0, 673.805], [108.679, 68.121, 55.353, 153.952, 63.876, 150.779, 673.805, 0]], 'durations': [[0, 164, 140, 562, 377, 552, 1575, 261], [164, 0, 58, 399, 229, 389, 1688, 163], [140, 58, 0, 429, 237, 419, 1632, 133], [562, 399, 429, 0, 217, 11, 1966, 369], [377, 229, 237, 217, 0, 210, 1757, 153], [552, 389, 419, 11, 210, 0, 1961, 362], [1575, 1688, 1632, 1966, 1757, 1961, 0, 1617], [261, 163, 133, 369, 153, 362, 1617, 0]], 'timeWindows': [[0, 144000], [0, 144000], [0, 144000], [0, 144000], [0, 144000], [0, 144000], [0, 144000], [0, 144000]], 'demands': [[0, 0, 0, 0, 0, 0, 0, 0], [270.17499999999995, 270.17499999999995, 270.17499999999995, 270.17499999999995, 270.17499999999995, 270.17499999999995, 270.17499999999995, 270.17499999999995], [88.474, 88.474, 88.474, 88.474, 88.474, 88.474, 88.474, 88.474], [567.91, 567.91, 567.91, 567.91, 567.91, 567.91, 567.91, 567.91], [59.510999999999996, 59.510999999999996, 59.510999999999996, 59.510999999999996, 59.510999999999996, 59.510999999999996, 59.510999999999996, 59.510999999999996], [123.723, 123.723, 123.723, 123.723, 123.723, 123.723, 123.723, 123.723], [543.998, 543.998, 543.998, 543.998, 543.998, 543.998, 543.998, 543.998], [149.78599999999997, 149.78599999999997, 149.78599999999997, 149.78599999999997, 149.78599999999997, 149.78599999999997, 149.78599999999997, 149.78599999999997]], 'max_weight': [5500, 3500, 4500, 600, 1200, 5000], 'max_volume': [777, 425, 635, 106, 265, 706], 'hop_fare': [1000, 1000, 1000, 1000, 1000, 1000], 'perKmCostPerVehicle': [32, 22, 25, 13, 19, 28], 'weight_matrix': [0, 0.005, 0.005, 0.008, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.002, 0.002, 0.023, 0.003, 0.005, 0.004, 0.002, 0.003, 0.003, 0.008, 0.003, 0.005, 0.008, 0.003, 0.005, 0.028, 0.005, 0.003, 0.009000000000000001, 0.005, 0.009000000000000001, 0.009000000000000001, 0.005, 0.005, 0.003, 0.005, 0.007, 0.003, 0.005, 0.025, 0.02, 0.015, 0.005, 0.015, 0.05, 0.001, 0.009000000000000001, 0.005, 0.035, 0.01, 0.01, 0.015, 0.01, 0.01, 0.02, 0.02, 0.03, 0.01, 0.005, 0.025, 0.02, 0.005, 0.02, 0.005, 0.005, 0.02, 0.003, 0.04, 0.03, 0.02, 0.025, 0.02, 0.06, 0.005], 'volume_matrix': [0, 19.955000000000002, 15.184999999999999, 14.696, 18.365000000000002, 14.835, 8.3, 8.655000000000001, 14.125, 6.18, 7.945, 7.42, 14.125, 2.332, 5.934, 28.428, 4.452, 7.065, 11.3, 4.308, 5.511, 3.4979999999999998, 14.696, 8.901, 8.3, 13.848, 8.475000000000001, 7.945, 34.608, 6.535, 7.098000000000001, 16.533, 7.595000000000001, 15.579, 11.124, 7.945, 8.475, 7.098000000000001, 14.125, 27.937, 11.973, 9.185, 45.925, 73.46000000000001, 44.505, 8.3, 25.965, 167.75, 2.825, 25.425, 7.945, 43.26, 16.95, 14.13, 42.375, 23.66, 18.37, 34.620000000000005, 31.78, 37.08, 16.95, 15.89, 35.325, 56.5, 11.83, 36.74, 9.004999999999999, 13.420000000000002, 34.620000000000005, 5.1930000000000005, 114.44000000000001, 84.75, 24.72, 42.375, 63.56, 89.03999999999999, 14.305000000000001], 'location_matrix': ['PLC-9e64068d-0315-40f2-86b8-f1127a31981a', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e', 'PLC-2c7b97d7-0d89-45d3-98ff-dc0bf9004e4e', 'PLC-7b034c4e-257b-4d2a-814f-8567c6370f64', 'PLC-7b034c4e-257b-4d2a-814f-8567c6370f64', 'PLC-7b034c4e-257b-4d2a-814f-8567c6370f64', 'PLC-7b034c4e-257b-4d2a-814f-8567c6370f64', 'PLC-7b034c4e-257b-4d2a-814f-8567c6370f64', 'PLC-7b034c4e-257b-4d2a-814f-8567c6370f64', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-964323a4-11f7-4992-85af-d4b9bdd96001', 'PLC-b4276938-dc67-4d61-a977-88aa5fc512e4', 'PLC-b4276938-dc67-4d61-a977-88aa5fc512e4', 'PLC-b4276938-dc67-4d61-a977-88aa5fc512e4', 'PLC-b4276938-dc67-4d61-a977-88aa5fc512e4', 'PLC-b4276938-dc67-4d61-a977-88aa5fc512e4', 'PLC-b4276938-dc67-4d61-a977-88aa5fc512e4', 'PLC-b4276938-dc67-4d61-a977-88aa5fc512e4', 'PLC-b4276938-dc67-4d61-a977-88aa5fc512e4', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-bdb8661a-7b36-4c7b-9b04-6bb507470fda', 'PLC-d22fba1b-b955-4030-b995-94fd0658ddd1', 'PLC-d22fba1b-b955-4030-b995-94fd0658ddd1', 'PLC-d22fba1b-b955-4030-b995-94fd0658ddd1', 'PLC-d22fba1b-b955-4030-b995-94fd0658ddd1', 'PLC-d22fba1b-b955-4030-b995-94fd0658ddd1', 'PLC-d22fba1b-b955-4030-b995-94fd0658ddd1', 'PLC-d22fba1b-b955-4030-b995-94fd0658ddd1', 'PLC-d22fba1b-b955-4030-b995-94fd0658ddd1', 'PLC-d22fba1b-b955-4030-b995-94fd0658ddd1', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-f336a43b-345c-4f8d-a009-637024141c88', 'PLC-f336a43b-345c-4f8d-a009-637024141c88'], 'vehicle_base_fare_matrix': [[0, 0, 0, 0, 0, 0], [2500, 2000, 2000, 1000, 1500, 2000], [2000, 1500, 1500, 1000, 1500, 2000], [7500, 5500, 6000, 3500, 4500, 7000], [5500, 3500, 4000, 2500, 3000, 4500], [7500, 5500, 6000, 3000, 4500, 6500], [21000, 14500, 16500, 9000, 12500, 18500], [3500, 2500, 3000, 1500, 2500, 3500]], 'optimization_option': 'dist'}


# Rewritten code from cell 2
data["loc_ids"]

# Rewritten code from cell 3
loc_id_mapping = {loc_id: idx for idx, loc_id in enumerate(data["loc_ids"])}


# Rewritten code from cell 4
mapped_location_matrix = [loc_id_mapping[loc] for loc in data["location_matrix"]]


# Rewritten code from cell 5
import pandas as pd
df_orders = pd.DataFrame([data["weight_matrix"], data["volume_matrix"], mapped_location_matrix]).transpose()
df_orders.columns=["order_weight","order_volume","order_loc"]
df_orders["order_id"] = range(len(df_orders))



# Rewritten code from cell 7
df_orders_f = df_orders.groupby("order_loc").sum().reset_index()


# Rewritten code from cell 8
distance_matrix = data["costs"]
duration_matrix = data["durations"]
max_veh_weight = data["max_weight"]
max_veh_volume = data["max_volume"]
time_windows = data["timeWindows"]
fixed_cost_matrix = data["vehicle_base_fare_matrix"]
fixed_cost_matrix

# Rewritten code from cell 9
start_time = [i for i, j in time_windows]
finish_time = [j for i, j in time_windows]
nodes = list(loc_id_mapping.values())
depot = 0
# customers = mapped_location_matrix
customers = nodes[1:]

# Rewritten code from cell 10


# Rewritten code from cell 11
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




# Rewritten code from cell 13
df_vehicle = pd.DataFrame([max_veh_weight,max_veh_volume]).transpose().reset_index()
df_vehicle.columns = ["v_id","max_weight","max_volume"]
vehicles = list(df_vehicle["v_id"])
demand_w = list(df_orders["order_weight"])
demand_dict_w = {(i, mapped_location_matrix[i]): demand_w[i] for i in range(len(demand_w))}
demand_v = list(df_orders["order_volume"])
demand_dict_v = {(i, mapped_location_matrix[i]): demand_v[i] for i in range(len(demand_v))}
max_vehw =list(df_vehicle["max_weight"])
max_vehv =list(df_vehicle["max_volume"])
variable_cost = list(data["perKmCostPerVehicle"])
variable_cost
orders = list(df_orders["order_id"])
order_cust_pair = list(zip(orders,mapped_location_matrix))


# Rewritten code from cell 14
order_cust_pairs=order_cust_pair[1:]


# Rewritten code from cell 15
import gurobipy as gp
from gurobipy import *

# Rewritten code from cell 16
my_model=gp.Model('CVRPTW')
xijk=my_model.addVars(nodes, nodes, vehicles, vtype=GRB.BINARY, name='xijk')
sik = my_model.addVars(nodes, vehicles, vtype = GRB.CONTINUOUS, name = 'sik')
w = my_model.addVars(order_cust_pairs, vehicles, vtype = GRB.CONTINUOUS, name = 'wojk')
v = my_model.addVars(order_cust_pairs, vehicles, vtype = GRB.CONTINUOUS, name = 'vojk')
I = my_model.addVars(vehicles, vtype = GRB.BINARY, name = 'I')
# Rewritten code from cell 17
#Objective function
# obj_fn = (gp.quicksum(dist_matrix[i,j]*(variable_cost[k] for k in vehicles)*gp.quicksum(xijk[i,j,k] for k in vehicles) for i in nodes for j in nodes))
obj_fn = (
    gp.quicksum(dist_matrix[i, j] * variable_cost[k] * xijk[i, j, k] for i in nodes for j in nodes for k in vehicles)
    + gp.quicksum(fixed_cost[i, k] * I[k] for k in vehicles for i in nodes)
)

my_model.setObjective(obj_fn, GRB.MINIMIZE)


# Rewritten code from cell 18
#Source to sink constraints
my_model.addConstrs(gp.quicksum(xijk[0,j,k] for j in customers)<=1 for k in vehicles)
my_model.addConstrs(gp.quicksum(xijk[i,0,k] for i in customers)<=1 for k in vehicles);
# my_model.addConstrs(
#     gp.quicksum(xijk[0, j, k] for j in customers) == gp.quicksum(xijk[i, 0, k] for i in customers) 
#     for k in vehicles
# );


# Rewritten code from cell 19
my_model.addConstrs(gp.quicksum(xijk[i,h,k] for i in nodes)- gp.quicksum(xijk[h,j,k] for j in nodes)==0 
                    for h in customers for k in vehicles);
# my_model.addConstrs(
#     (gp.quicksum(xijk[i, h, k] for i in nodes if i != h) - gp.quicksum(xijk[h, j, k] for j in nodes if j != h) == 0
#      for h in customers for k in vehicles),
#     name="visit_once_constraint"
# )
# my_model.addConstrs(gp.quicksum(xijk[i, j, k] for j in nodes for k in vehicles) == 1 for i in customers);
# my_model.addConstrs(gp.quicksum(xijk[j, i, k] for j in customers for k in vehicles if i != j) == 1 for i in customers);
my_model.addConstrs(gp.quicksum(xijk[i, j, k]  for j in nodes for k in vehicles if i == j) == 0 for i in customers);

# Rewritten code from cell 20
#capacity constraints
my_model.addConstrs(gp.quicksum(w[o,j,k] for k in vehicles) == demand_dict_w[o,j] 
                    for (o,j) in order_cust_pairs)
my_model.addConstrs(gp.quicksum(w[o,j,k] for (o,j) in order_cust_pairs) <= max_vehw[k] for k in vehicles)
my_model.addConstrs(gp.quicksum(v[o,j,k] for k in vehicles) == demand_dict_v[o,j] for (o,j) in order_cust_pairs)
my_model.addConstrs(gp.quicksum(v[o,j,k] for (o,j) in order_cust_pairs) <= max_vehv[k] for k in vehicles);

# Rewritten code from cell 21
bigM = 1000000
my_model.addConstrs(w[o,j,k] <= bigM * gp.quicksum(xijk[i,j,k] for i in nodes) for (o,j) in order_cust_pairs for k in vehicles);
my_model.addConstrs(v[o,j,k] <= bigM * gp.quicksum(xijk[i,j,k] for i in nodes) for (o,j) in order_cust_pairs for k in vehicles);

# Rewritten code from cell 22
my_model.addConstrs(sik[i,k] + time_matrix[i,j] - sik[j,k] <= (1-xijk[i,j,k]) *bigM 
                    for i in customers 
                    for j in customers for k in vehicles);

# Rewritten code from cell 23
my_model.addConstrs(sik[i,k] <= finish_time[i] for i in nodes for k in vehicles)
                    
my_model.addConstrs(sik[i,k] >= start_time[i] for i in nodes for k in vehicles); 

# Rewritten code from cell 24
# my_model.setParam("MIPGap",0.6)

# Rewritten code from cell 25
my_model.optimize()

# Rewritten code from cell 26
# import csv

# # Define the output file
# output_file = "vehicle_routes_and_orders.csv"

# # Open the file in write mode
# with open(output_file, mode="w", newline="") as file:
#     writer = csv.writer(file)
    
#     # Write the header
#     writer.writerow(["Vehicle", "Route", "Order", "Customer", "Weight"])
    
#     # Write the data
#     for k in vehicles:
#         route = []
        
#         # Extract the routes from xijk variable
#         for i in nodes:
#             for j in nodes:
#                 if xijk[i, j, k].X > 0.5:  # Binary variable, thresholded for precision
#                     route.append((i, j))
        
#         # Format the route as a string
#         route_str = " -> ".join(f"{i} to {j}" for i, j in route) if route else "No route assigned"
        
#         # Extract served orders from w variable
#         served_orders = []
#         for o, j in order_cust_pairs:
#             if w[o, j, k].X > 0:  # Continuous variable, check positive value
#                 served_orders.append((o, j, w[o, j, k].X))
        
#         # Write data to the CSV
#         if served_orders:
#             for o, j, weight in served_orders:
#                 writer.writerow([f"Vehicle {k}", route_str, f"Order {o}", f"Customer {j}", weight])
#         else:
#             # Write vehicle with no served orders
#             writer.writerow([f"Vehicle {k}", route_str, "No orders served", "", ""])

# print(f"Data written to {output_file}")



# Rewritten code from cell 27
# import csv

# # Define the file name
# output_file = "vehicle_assignments.csv"

# # Open the file in write mode
# with open(output_file, mode="w", newline="") as file:
#     # Create a CSV writer object
#     writer = csv.writer(file)
    
#     # Write the header row
#     writer.writerow(["Vehicle", "Customer", "Order", "Weight"])
    
#     # Write data rows
#     for o, j in order_cust_pairs:
#         for k in vehicles:
#             if w[o, j, k].X > 0:
#                 writer.writerow([f"vehicle{k}", f"customer{j}", f"order{o}", w[o, j, k].X])

# print(f"Data written to {output_file}")



# Rewritten code from cell 28
# # Calculate and display the distance traveled by each vehicle
# vehicle_distances = {}

# for vehicle, route in vehicle_routes.items():
#     total_distance = sum(dist_matrix[i, j] for i, j in route)
#     vehicle_distances[vehicle] = total_distance
#     print(f"Vehicle {vehicle} Distance Traveled: {total_distance}")

# # Optionally, print a summary
# total_distance_all_vehicles = sum(vehicle_distances.values())
# print(f"Total Distance Traveled by All Vehicles: {total_distance_all_vehicles}")
# Initialize a dictionary to store total distance for each vehicle
total_distance = {k: 0 for k in vehicles}

# Calculate the total distance for each vehicle
for k in vehicles:
    for i in nodes:
        for j in nodes:
            if xijk[i, j, k].X > 0.5:  # Binary variable, thresholded for precision
                total_distance[k] += dist_matrix[i, j]

# Print the results
for k, distance in total_distance.items():
    print(f"Total distance traveled by Vehicle {k}: {distance}")

