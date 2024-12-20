import pandas as pd
PARAMETERS = {
    "max_iter": 100,
    "tabu_tenure": 10,
    "removal_fraction": 0.2, 
    "stoppage criteria_no_improvement_count": 3,
    "Diversification_logic_run_count": 3,



}


def load_data():
    """Load data from specified file paths."""
    locations_df = pd.read_csv("C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/locations.csv")
    order_list_df = pd.read_excel('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/order_list_1.xlsx')
    travel_matrix_df = pd.read_csv('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/travel_matrix.csv')
    trucks_df = pd.read_csv('C:/Users/Acer/Documents/GitHub/Tabu-Search-for-CVRPTW/inputs/trucks.csv')
    return locations_df, order_list_df, travel_matrix_df, trucks_df

def preprocess_data(locations_df, order_list_df, travel_matrix_df, trucks_df):
    """Preprocess the data for Tabu Search."""
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
    
    locations_df = locations_df[locations_df['location_code'].isin(dest + ['A123'])]
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
    dest3 = {dest2[i]: i for i in range(len(dest2))}
    
    travel_matrix_df = travel_matrix_df[(travel_matrix_df['source_location_code'].isin(dest + ['A123'])) &
                                        (travel_matrix_df['destination_location_code'].isin(dest + ['A123']))]
    travel_matrix_df['mapped_source'] = travel_matrix_df['source_location_code'].map(dest3)
    travel_matrix_df['mapped_destination'] = travel_matrix_df['destination_location_code'].map(dest3)
    
    dist_matrix = {(travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i]): 
                   travel_matrix_df['travel_distance_in_km'][i] for i in travel_matrix_df.index}
    time_matrix = {(travel_matrix_df['mapped_source'][i], travel_matrix_df['mapped_destination'][i]): 
                   travel_matrix_df['travel_time_in_min'][i] for i in travel_matrix_df.index}
    
    max_capacity_w = {v: Q1[v] for v in range(len(Q1))}
    
    return nodes, vehicles, dist_matrix, demands_w, max_capacity_w, Q1, var_cost, fixed_cost


