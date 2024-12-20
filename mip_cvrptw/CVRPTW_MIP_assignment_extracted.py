import gurobipy as gp
from gurobipy import *
import numpy as np
import pandas as pd

bigM = 1000000

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

order_list_df1 = order_list_df.sort_values(by = 'Destination Code').groupby('Destination Code').sum("Total Weight").reset_index()

locations_df = locations_df[locations_df['location_code'].isin(dest + ['A123'])]

# Convert loading/unloading windows to minutes with explicit format
locations_df['start_minutes'] = pd.to_datetime(locations_df['location_loading_unloading_window_start'], format='%H:%M').dt.hour * 60 + pd.to_datetime(locations_df['location_loading_unloading_window_start'], format='%H:%M').dt.minute
locations_df['end_minutes'] = pd.to_datetime(locations_df['location_loading_unloading_window_end'], format='%H:%M').dt.hour * 60 + pd.to_datetime(locations_df['location_loading_unloading_window_end'], format='%H:%M').dt.minute

customers = locations_df.sort_values(by = 'location_code').iloc[:len(order_list_df1),:]

locations_df2 = locations_df.sort_values(by = 'location_code')

cap_df = dict(zip(trucks_df['truck_type'], trucks_df['truck_max_weight']))

max_veh_access = []
for i in locations_df2.index:
    max_veh_access.append(cap_df[eval(locations_df2['trucks_allowed'][i])[-1]])

max_veh_access = max_veh_access[len(order_list_df1):] + max_veh_access[:len(order_list_df1)]

depot = locations_df.sort_values(by = 'location_code').iloc[len(order_list_df1):,:]

Nodes = pd.concat([depot,customers], ignore_index = True)

vehicles = [ k for k in range(0,len(Q1))]
customers = [ i for i in range(1,len(Nodes))]
nodes = [ i for i in range(0, len(Nodes))]

demand = [0] + list(order_list_df1['Total Weight'])


start_time = list(Nodes['start_minutes'])
finish_time = list(Nodes['end_minutes'])

# Constants
service_time_customer = 20  
service_time_depot = 60 

dest2 = ['A123'] + sorted(dest)

dest3 = {} 
for i in range(len(dest2)):
    dest3[dest2[i]] = i

print(dest3)

travel_matrix_df = travel_matrix_df[(travel_matrix_df['source_location_code'].isin(dest + ['A123'])) & (travel_matrix_df['destination_location_code'].isin(dest + ['A123']))]

travel_matrix_df['mapped_source'] = travel_matrix_df['source_location_code'].map(dest3)
travel_matrix_df['mapped_destination'] = travel_matrix_df['destination_location_code'].map(dest3)

dist_matrix = {}
time_matrix = {}
for i in travel_matrix_df.index:
    dist_matrix[(travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i])] = travel_matrix_df['travel_distance_in_km'][i]
    time_matrix[(travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i])] = travel_matrix_df['travel_time_in_min'][i]


#Decision Variable

my_model=gp.Model('CVRPTW')

xijk=my_model.addVars(nodes, nodes, vehicles, vtype=GRB.BINARY, name='xijk')
sik = my_model.addVars(nodes, vehicles, vtype = GRB.CONTINUOUS, name = 'sik',lb = 0)
q = my_model.addVars(customers, vehicles, vtype = GRB.CONTINUOUS, name = 'qjk', lb = 0)
I = my_model.addVars(vehicles, vtype = GRB.BINARY, name = 'I')
# Objective function: Minimize total cost (fixed cost + variable cost)
obj_fn = (
    gp.quicksum(
        dist_matrix[i, j] * var_cost[k] * xijk[i, j, k]
        for i in nodes for j in nodes for k in vehicles
    ) +
    gp.quicksum(fixed_cost[k] * I[k] for k in vehicles)
)



my_model.setObjective(obj_fn, GRB.MINIMIZE)



#Source to sink constraints
my_model.addConstrs(gp.quicksum(xijk[0,j,k] for j in customers)<=1 for k in vehicles);

my_model.addConstrs(gp.quicksum(xijk[i,0,k] for i in customers)<=1 for k in vehicles);

my_model.addConstrs(gp.quicksum(xijk[i,j,k] for i in nodes)- gp.quicksum(xijk[j,i,k] for i in nodes)==0 
                    for j in nodes for k in vehicles);

#capacity constraints
my_model.addConstrs(gp.quicksum(q[j,k] for k in vehicles) == demand[j] 
                    for j in customers); 
my_model.addConstrs(gp.quicksum(q[j,k] for j in customers) <= Q1[k] for k in vehicles);

my_model.addConstrs(q[j,k] <= Q1[k] * gp.quicksum(xijk[i,j,k] for i in nodes) for j in customers for k in vehicles);

my_model.addConstrs(sik[i,k] + time_matrix[i,j] + service_time_customer
                    - sik[j,k] <= (1-xijk[i,j,k]) *10000
                    for i in customers 
                    for j in customers for k in vehicles);

my_model.addConstrs(sik[i,k] <= finish_time[i] for i in nodes for k in vehicles)
                    
my_model.addConstrs(sik[i,k] >= start_time[i] for i in nodes for k in vehicles);                                   

# for i in nodes:
#     for j in nodes:
#         for k in vehicles:
#             if Q[k] <= max_veh_access[j] and Q[k] <= max_veh_access[i]:
#                 my_model.addConstr(xijk[i,j,k] <= 1);
#             else:
#                 my_model.addConstr(xijk[i,j,k] == 0);

my_model.addConstrs(gp.quicksum(xijk[0,j,k] for j in customers) <= I[k] for k in vehicles);
# Parse the initial solution into a format suitable for warm start
initial_routes = {
    0: [0, 76, 0],
    1: [0, 80, 45, 100, 12, 29, 25, 0],
    2: [0, 99, 62, 0],
    3: [0, 58, 50, 63, 1, 35, 54, 0],
    4: [0, 47, 95, 26, 17, 23, 64, 0],
    5: [0, 77, 83, 61, 48, 0],
    6: [0, 105, 71, 73, 51, 82, 0],
    7: [0, 36, 30, 69, 41, 14, 24, 22, 92, 31, 0],
    8: [0, 49, 70, 38, 67, 98, 56, 42, 107, 108, 57, 33, 43, 34, 88, 106, 0],
    9: [0, 94, 74, 6, 102, 9, 13, 87, 0],
    10: [0, 53, 2, 7, 8, 5, 78, 60, 20, 101, 16, 3, 46, 79, 32, 59, 0],
    11: [0, 68, 103, 104, 86, 84, 11, 52, 97, 89, 96, 15, 90, 18, 0],
    12: [0, 55, 40, 37, 44, 81, 4, 39, 66, 72, 75, 65, 93, 19, 0],
    13: [0, 27, 28, 91, 10, 85, 21, 0],
    14: [],
    15: [],
    16: [],
    17: [],
    18: []
}

# Reverse mapping of dest3 to get the key corresponding to a value
reverse_dest3 = {v: k for k, v in dest3.items()}

# Remap the initial route locations to their respective keys in dest3
remapped_routes = {
    route: [reverse_dest3[location] for location in locations if location in reverse_dest3]
    for route, locations in initial_routes.items()
}

# Print the remapped routes
print(remapped_routes)

for vehicle, route in remapped_routes.items():
    if vehicle in I:  # Ensure the vehicle exists in I
        if route:  # If the vehicle has a route
            I[vehicle].start = 1.0
        else:
            I[vehicle].start = 0.0  # No route assigned to this vehicle
    else:
        print(f"Warning: Vehicle {vehicle} not found in I")

    # Set route variables (xijk)
    for idx in range(len(route) - 1):
        i = dest3[route[idx]]
        j = dest3[route[idx + 1]]
        if (i, j, vehicle) in xijk:
            xijk[i, j, vehicle].start = 1.0

    # Set service start time (sik) for each node in the route
    for idx, location in enumerate(route):
        if idx < len(route):
            i = dest3[location]
            sik[i, vehicle].start = start_time[i]  # Assign start time
        if location in demand:
            q[dest3[location], vehicle].start = demand[dest3[location]]  # Allocate demand



my_model.setParam('Heuristics', 0.6)  
my_model.setParam('MIPGap', 0.35)  


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
print("Non-zero values of x:")
for (i, j, k), value in non_zero_xijk.items():
    print(f"x_{i}_{j}_{k} = {value}")


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


# Calculate the variable cost
variable_cost = sum(dist_matrix[i, j] * var_cost[k] * xijk[i, j, k].X
                    for i in nodes for j in nodes for k in vehicles if xijk[i, j, k].X > 0)

# Calculate the fixed cost (only if xijk[i, j, k] > 0 for any k)
fixed_cost_total = sum(fixed_cost[k] for k in vehicles if any(xijk[i, j, k].X > 0 for i in nodes for j in nodes))

# Print the results
print(f"Variable Cost: {variable_cost}")
print(f"Fixed Cost: {fixed_cost_total}")
