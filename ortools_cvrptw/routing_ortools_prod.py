import json
import math
import random
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from itertools import groupby

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


def get_solution(data, manager, routing, assignment):
    total_cost = assignment.ObjectiveValue()
    # Display dropped nodes.
    dropped_nodes = "Dropped nodes:"
    dropped_nodes_list = []
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes += f" {manager.IndexToNode(node)}"
            dropped_nodes_list.append(manager.IndexToNode(node))

    # Display routes
    routes_dict = {}
    routes_dict_list = []
    routes_list = []
    total_distance = 0
    total_load = 0
    route_count = 0
    for vehicle_id in range(data["num_vehicles"]):
        route_dict = {}
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        route_nodes = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            route_nodes.append(node_index)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        route_nodes.append(manager.IndexToNode(index))
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        # print(plan_output)
        route_weight = round(sum([data['weights'][route_nodes[i]] for i in range(len(route_nodes))]) / 1000, 2)
        route_volume = round(sum([data['volumes'][route_nodes[i]] for i in range(len(route_nodes))]) / 1728, 2)
        routes_list.append(route_nodes)
        total_distance += route_distance
        total_load += route_load
        if len(route_nodes) > 2:
            route_count += 1
            node_seq = route_nodes[1:-1]
            loc_index_seq = [data['order_loc_index'][node_seq[i]] for i in range(len(node_seq))]
            loc_id_seq = [data['order_loc_ids'][node_seq[i]] for i in range(len(node_seq))]
            route_dict.update({
                'route_id': route_count,
                'order_index_seq': node_seq,
                'drop_loc_index_seq': loc_index_seq,
                'drop_loc_id_seq': loc_id_seq,
                'unique_drop_loc_id_seq': [key for key, _ in groupby(loc_index_seq)],
                'route_weight_kg': route_weight,
                'route_volume_cft': route_volume,
                'vehicle_type': data['veh_type_id'][vehicle_id],
                'vehicle_max_weight_kg': round(data['veh_max_weight'][vehicle_id] / 1000),
                'vehicle_max_volume_cft': round(data['veh_max_volume'][vehicle_id] / 1728),
                'vehicle_fixed_cost': data['fixed_cost'][vehicle_id],
                'vehicle_per_km_cost': data['cost_per_km'][vehicle_id]
            })
            routes_dict_list.append(route_dict)
            # print('-'*75)
            # print(routes_dict)
    routes_json = json.dumps(routes_dict_list, indent=4)

    return routes_json


def get_best_routes(data):
    
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # -----------------------------------------------------------------------------------------
    # Objective Function
    # -----------------------------------------------------------------------------------------
    hop_cost = 10000

    def vehicle_cost_callback(veh_id, from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        from_loc_id = data['order_loc_ids'][from_node]
        to_loc_id = data['order_loc_ids'][to_node]
        hop_cost_val = 0
        distance = math.ceil(data['distance_matrix'][from_node][to_node])

        if from_node != 0:
            if from_loc_id != to_loc_id:
                hop_cost_val += hop_cost

        return int(data['cost_per_km'][veh_id] * 50 * distance) + hop_cost_val

    def travel_time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["time_matrix"][from_node][to_node]

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    for vehicle_id in range(data['num_vehicles']):
        vehicle_cost_callback_index = routing.RegisterTransitCallback(
            lambda from_index, to_index, vehicle_id=vehicle_id: vehicle_cost_callback(vehicle_id, from_index,
                                                                                      to_index))
        routing.SetArcCostEvaluatorOfVehicle(vehicle_cost_callback_index, vehicle_id)
        veh_fixed_cost = math.ceil(data['fixed_cost'][vehicle_id])
        routing.SetFixedCostOfVehicle(veh_fixed_cost, vehicle_id)
        # print(f'{vehicle_id}: [{data['veh_type_id'][vehicle_id]}, {data['veh_max_weight'][vehicle_id]}, {data['veh_max_volume'][vehicle_id]}]: FC= {veh_fixed_cost}; cost_per_km= {data['cost_per_km'][vehicle_id]}')

    # if data['optimization_option'] == 'dist':
    #     transit_distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    #     routing.SetArcCostEvaluatorOfAllVehicles(transit_distance_callback_index)
    # elif data['optimization_option'] == 'time':
    #     transit_travel_time_callback_index = routing.RegisterTransitCallback(travel_time_callback)
    #     routing.SetArcCostEvaluatorOfAllVehicles(transit_travel_time_callback_index)

    # -----------------------------------------------------------------------------------------
    # Constraints
    # -----------------------------------------------------------------------------------------
    # Add Capacity constraints
    def weight_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data["weights"][from_node]

    weight_callback_index = routing.RegisterUnaryTransitCallback(weight_callback)
    routing.AddDimensionWithVehicleCapacity(
        weight_callback_index,
        0,  # null capacity slack
        data["veh_max_weight"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Weight_Capacity",
    )

    def volume_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data["volumes"][from_node]

    volume_callback_index = routing.RegisterUnaryTransitCallback(volume_callback)
    routing.AddDimensionWithVehicleCapacity(
        volume_callback_index,
        0,  # null capacity slack
        data["veh_max_volume"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Volume_Capacity",
    )

    # # Allow to drop nodes.
    # penalty = 9999999
    # for node in range(1, len(data["distance_matrix"])):
    #     routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # -----------------------------------------------------------------------------------------
    # Solve
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(20)
    search_parameters.log_search = False

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    if assignment:
        routes_json = get_solution(data, manager, routing, assignment)
        return routes_json

    return []
