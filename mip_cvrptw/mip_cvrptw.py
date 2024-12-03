#IMPORT INPUTS
import json
import pandas as pd
import gurobipy as gp
from gurobipy import *
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
df_vehicle = pd.DataFrame([max_veh_weight,max_veh_volume]).transpose().reset_index()
df_vehicle.columns = ["v_id","max_weight","max_volume"]
vehicles = list(df_vehicle["v_id"])
demand_w = list(df_orders_f["order_weight"])
demand_v = list(df_orders_f["order_volume"])
max_vehw =list(df_vehicle["max_weight"])
max_vehv =list(df_vehicle["max_volume"])
# CVRPTW FORMULATION
my_model=gp.Model('CVRPTW')

# DECISION VARIABLES
xijk=my_model.addVars(nodes, nodes, vehicles, vtype=GRB.BINARY, name='xijk')
sik = my_model.addVars(nodes, vehicles, vtype = GRB.CONTINUOUS, name = 'sik')
wjk = my_model.addVars(customers, vehicles, vtype = GRB.CONTINUOUS, name = 'wjk')
vjk = my_model.addVars(customers, vehicles, vtype = GRB.CONTINUOUS, name = 'vjk')

# OBJECTIVE FUNCTION
obj_fn = (gp.quicksum(dist_matrix[i,j]* gp.quicksum(xijk[i,j,k] for k in vehicles) for i in nodes for j in nodes))
my_model.setObjective(obj_fn, GRB.MINIMIZE)

# CONSTRAINTS
#Source to sink constraints
my_model.addConstrs(gp.quicksum(xijk[0,j,k] for j in customers)==1 for k in vehicles)
my_model.addConstrs(gp.quicksum(xijk[i,0,k] for i in customers)==1 for k in vehicles);

my_model.addConstrs(gp.quicksum(xijk[i,h,k] for i in nodes)- gp.quicksum(xijk[h,j,k] for j in nodes)==0 
                    for h in customers for k in vehicles);

my_model.addConstrs(gp.quicksum(xijk[i, j, k] for j in nodes for k in vehicles) == 1 for i in customers);
my_model.addConstrs(gp.quicksum(xijk[i, j, k] for j in nodes for k in vehicles if i == j) == 0 for i in customers);

#capacity constraints
my_model.addConstrs(gp.quicksum(wjk[j,k] for k in vehicles) == demand_w[j] 
                    for j in customers)
my_model.addConstrs(gp.quicksum(wjk[j,k] for j in customers) <= max_vehw[k] for k in vehicles)
my_model.addConstrs(gp.quicksum(vjk[j,k] for k in vehicles) == demand_v[j] 
                    for j in customers)
my_model.addConstrs(gp.quicksum(vjk[j,k] for j in customers) <= max_vehv[k] for k in vehicles);

bigM = 1000000
my_model.addConstrs(wjk[j,k] <= bigM * gp.quicksum(xijk[i,j,k] for i in nodes) for j in customers for k in vehicles);
my_model.addConstrs(vjk[j,k] <= bigM * gp.quicksum(xijk[i,j,k] for i in nodes) for j in customers for k in vehicles);

my_model.addConstrs(sik[i,k] + time_matrix[i,j] - sik[j,k] <= (1-xijk[i,j,k]) *bigM 
                    for i in customers 
                    for j in customers for k in vehicles);

my_model.addConstrs(sik[i,k] <= finish_time[i] for i in nodes for k in vehicles)
                    
my_model.addConstrs(sik[i,k] >= start_time[i] for i in nodes for k in vehicles); 
my_model.optimize()

# Create a new dictionary to store non-zero values
non_zero_xijk = {}

# Iterate over all keys and values in xijk
for (i, j, k), value in xijk.items():
    # Check if the value is a Gurobi variable
    if hasattr(value, "X"):  # If it's a Gurobi variable, use .X to get the optimized value
        if value.X > 0.5:  # Filter for non-zero values
            non_zero_xijk[(i, j, k)] = value.X
    else:  # Otherwise, assume it's a numeric value
        if value > 0.5:  # Filter for non-zero values
            non_zero_xijk[(i, j, k)] = value

# Print the non-zero values


# Create a dictionary to store routes for each vehicle
vehicle_routes = {}

# Iterate over the non_zero_xijk dictionary to create routes
for (i, j, k), value in non_zero_xijk.items():
    if value == 1.0:
        if k not in vehicle_routes:
            vehicle_routes[k] = []  # Create a list for each vehicle if not already created
        vehicle_routes[k].append((i, j))
        
        

# Display the routes for each vehicle
for vehicle, route in vehicle_routes.items():
    print(f"Vehicle {vehicle} Route: {route}")

# Calculate and display the distance traveled by each vehicle
vehicle_distances = {}

for vehicle, route in vehicle_routes.items():
    total_distance = sum(dist_matrix[i, j] for i, j in route)
    vehicle_distances[vehicle] = total_distance
    print(f"Vehicle {vehicle} Distance Traveled: {total_distance}")

# Optionally, print a summary
total_distance_all_vehicles = sum(vehicle_distances.values())
print(f"Total Distance Traveled by All Vehicles: {total_distance_all_vehicles}")