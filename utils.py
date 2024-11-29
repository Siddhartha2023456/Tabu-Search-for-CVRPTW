import math
import random
import numpy as np


random.seed(1503)




def prep_data(solver_data):
    fixed_cost_list = solver_data['max_weight']
    per_km_cost_list = solver_data['perKmCostPerVehicle']
    max_weight_list = solver_data['max_weight']
    max_volume_list = solver_data['max_volume']
    hop_fare = solver_data['hop_fare']
    max_weight_gram = [math.ceil(1000 * max_weight_list[i]) for i in range(len(max_weight_list))]
    max_volume_inch = [math.ceil(1728 * max_volume_list[i]) for i in range(len(max_volume_list))]
    fixed_cost_list = [math.ceil(fixed_cost_list[i]) for i in range(len(fixed_cost_list))]
    per_km_cost_list = [math.ceil(per_km_cost_list[i]) for i in range(len(per_km_cost_list))]

    vehicle_base_fare_matrix = solver_data['vehicle_base_fare_matrix']
    base_fare_dict = {}

    # Iterate over each location (row)
    for loc_index, row in enumerate(vehicle_base_fare_matrix):
        # Iterate over each vehicle type (column)
        for veh_index, fare in enumerate(row):
            base_fare_dict[(loc_index, veh_index)] = fare

    loc_ids = solver_data['loc_ids']
    dist_matrix = solver_data['costs']
    travel_time_matrix = solver_data['durations']
    dist_matrix = [[0] + row[1:] for row in dist_matrix]
    travel_time_matrix = [[0] + row[1:] for row in travel_time_matrix]

    location_time_windows = solver_data['timeWindows']
    order_weights = solver_data['weight_matrix']
    order_volumes = solver_data['volume_matrix']
    order_loc_ids = solver_data['location_matrix']
    order_loc_index = [loc_ids.index(order_loc_ids[i]) for i in range(len(order_loc_ids))]
    service_times = [0 for _ in range(len(order_weights))]

    order_weights_gram = [round(weight * 1000) for weight in order_weights]
    order_volumes_inch = [round(volume * 1728) for volume in order_volumes]
    order_time_windows = [location_time_windows[loc_ids.index(order_loc_ids[i])] for i in range(len(order_weights))]

    # Creating a mapping from location ID to index
    loc_id_to_index = {loc_id: index for index, loc_id in enumerate(loc_ids)}

    # Initialize an empty distance matrix for orders
    order_count = len(order_loc_ids)
    order_distance_matrix = np.zeros((order_count, order_count))
    order_time_matrix = np.zeros((order_count, order_count))

    # Fill the order distance matrix
    for i in range(order_count):
        for j in range(order_count):
            loc_i_index = loc_id_to_index[order_loc_ids[i]]
            loc_j_index = loc_id_to_index[order_loc_ids[j]]
            order_distance_matrix[i, j] = dist_matrix[loc_i_index][loc_j_index]
            order_time_matrix[i, j] = travel_time_matrix[loc_i_index][loc_j_index]

    total_order_weights = sum(order_weights_gram)
    total_order_volume = sum(order_volumes_inch)
    copy_max_weight_gram = []
    copy_max_volume_inch = []
    copy_fixed_cost_list = []
    copy_per_km_cost_list = []
    copy_veh_index_list = []
    for i in range(len(max_weight_gram)):
        num_veh_weight = math.ceil(total_order_weights / max_weight_gram[i])
        num_veh_volume = math.ceil(total_order_volume / max_volume_inch[i])
        num_veh = max(num_veh_weight, num_veh_volume)
        for j in range(num_veh):
            copy_max_weight_gram.append(max_weight_gram[i])
            copy_max_volume_inch.append(max_volume_inch[i])
            copy_fixed_cost_list.append(int(fixed_cost_list[i]))
            copy_per_km_cost_list.append(per_km_cost_list[i])
            copy_veh_index_list.append(i)

    data = {}
    data['distance_matrix'] = np.round(order_distance_matrix).astype(int)
    data['time_matrix'] = np.round(order_time_matrix).astype(int)
    data['weight_demands'] = order_weights_gram
    data['volume_demands'] = order_volumes_inch
    data['service_times'] = service_times
    data['time_windows'] = order_time_windows
    data['order_loc_ids'] = order_loc_ids
    data['order_loc_index'] = order_loc_index
    data['num_vehicles'] = len(copy_max_weight_gram)
    data['vehicle_weight_capacities'] = copy_max_weight_gram
    data['vehicle_volume_capacities'] = copy_max_volume_inch
    data['vehicle_time_capacities'] = [99999999 for _ in range(len(copy_max_weight_gram))]
    data['fixed_costs'] = copy_fixed_cost_list
    data['cost_per_km'] = copy_per_km_cost_list
    data['cost_per_ton'] = [1 for _ in range(len(copy_per_km_cost_list))]
    data['veh_type_id'] = copy_veh_index_list
    data['depot'] = 0
    data['optimization_option'] = solver_data['optimization_option']
    data['base_fare'] = base_fare_dict
    data['hop_fare'] = hop_fare

    return data


def create_data_model(input_data):
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = input_data['distance_matrix']
    data["time_matrix"] = input_data['time_matrix']
    data["order_loc_ids"] = input_data['order_loc_ids']
    data['order_loc_index'] = input_data['order_loc_index']
    data["demands"] = input_data['volume_demands']
    data["weights"] = input_data['weight_demands']
    data["volumes"] = input_data['volume_demands']
    data["vehicle_capacities"] = input_data['vehicle_volume_capacities']
    data["veh_max_weight"] = input_data['vehicle_weight_capacities']
    data["veh_max_volume"] = input_data['vehicle_volume_capacities']
    data["fixed_cost"] = input_data['fixed_costs']
    data["cost_per_km"] = input_data['cost_per_km']
    data["cost_per_ton"] = input_data['cost_per_ton']
    data['veh_type_id'] = input_data['veh_type_id']
    data["num_vehicles"] = len(input_data['vehicle_volume_capacities'])
    data["depot"] = 0
    data['optimization_option'] = input_data.get('optimization_option', 'dist')
    data['base_fare'] = input_data['base_fare']
    data['hop_fare'] = input_data['hop_fare']
    return data