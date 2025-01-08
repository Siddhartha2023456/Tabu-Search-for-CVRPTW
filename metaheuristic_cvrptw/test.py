import pandas as pd
excel_data = pd.ExcelFile('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/DATASET.xlsx')
locations_df = excel_data.parse(sheet_name="LOCATIONS")
order_list_df = excel_data.parse(sheet_name="ORDERS")
travel_matrix_df = excel_data.parse(sheet_name="TRAVEL MATRIX")
trucks_df = excel_data.parse(sheet_name="TRUCKS")
# locations_df = pd.read_csv("C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/locations.csv")
# order_list_df = pd.read_excel('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/order_list_1.xlsx')
# travel_matrix_df = pd.read_csv('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/travel_matrix.csv')
# trucks_df = pd.read_csv('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/trucks.csv')
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
locations_df['location_code'] = locations_df['location_code'].astype(str)
locations_df1 = locations_df[locations_df['location_code'].isin(dest + ['A123'])]
locations_df1['start_minutes'] = pd.to_datetime(locations_df['location_loading_unloading_window_start'], format='%H:%M:%S').dt.hour * 60 + pd.to_datetime(locations_df['location_loading_unloading_window_start'], format='%H:%M:%S').dt.minute
locations_df1['end_minutes'] = pd.to_datetime(locations_df['location_loading_unloading_window_end'], format='%H:%M:%S').dt.hour * 60 + pd.to_datetime(locations_df['location_loading_unloading_window_end'], format='%H:%M:%S').dt.minute
customers = locations_df1.sort_values(by='location_code').iloc[:len(order_list_df1), :]
locations_df2 = locations_df1.sort_values(by='location_code')
cap_df = dict(zip(trucks_df['truck_type'], trucks_df['truck_max_weight']))
max_veh_access = []
for i in locations_df2.index:
    max_veh_access.append(cap_df[eval(locations_df2['trucks_allowed'][i])[-1]])
max_veh_access = max_veh_access[len(order_list_df1):] + max_veh_access[:len(order_list_df1)]
depot = locations_df1.sort_values(by='location_code').iloc[len(order_list_df1):, :]
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
travel_matrix_df1 = travel_matrix_df.copy()
print(travel_matrix_df['source_location_code'].dtypes)
travel_matrix_df1['source_location_code'] = travel_matrix_df['source_location_code'].astype(str)
print(travel_matrix_df1['source_location_code'].dtypes)
travel_matrix_df['destination_location_code'] = travel_matrix_df['destination_location_code'].astype(str)
travel_matrix_df = travel_matrix_df[(travel_matrix_df['source_location_code'].isin(dest + ['A123'])) & (
    travel_matrix_df['destination_location_code'].isin(dest + ['A123']))]
travel_matrix_df['mapped_source'] = travel_matrix_df['source_location_code'].map(dest3)
travel_matrix_df['mapped_destination'] = travel_matrix_df['destination_location_code'].map(dest3)
print(travel_matrix_df.head())
dist_matrix = {}
time_matrix = {}
for i in travel_matrix_df.index:
    dist_matrix[(travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i])] = \
    travel_matrix_df['travel_distance_in_km'][i]
    time_matrix[(travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i])] = \
    travel_matrix_df['travel_time_in_min'][i]
max_capacity_w = {v: Q1[v] for v in range(len(Q1))}
print(f"dest1:{len(dest1)}")
print(f"dest:{len(dest)}")
print(f"order_list_df:{len(order_list_df)}")
print(f"order_list_df1:{len(order_list_df1)}")
print(f"locations_df:{len(locations_df)}")
print(f"locations_df1:{len(locations_df1)}")
print(f"locations_df2:{len(locations_df2)}")
print(len(Nodes))
print(vehicles)
print(len(customers))
print(len(nodes))
print(len(demands_w))
print(f"dist:{len(dist_matrix)}")