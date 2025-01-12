import json
from routing_ortools_prod import get_best_routes
# from routing_ortools_cost import get_best_routes_true_cost
# from routing_ortools_multi_pickup import get_best_routes
# from deployed_routing_ortools import deployed_get_best_routes


def calculate_actual_route_cost(input_data, veh_type, seq):
    route_cost = 0
    route_dist = 0
    for i in range(len(seq) - 1):
        route_dist += input_data['costs'][seq[i]][seq[i + 1]]
    per_km_cost = route_dist * input_data['perKmCostPerVehicle'][veh_type]
    route_cost += per_km_cost
    base_fare = input_data['max_weight'][veh_type]
    route_cost += base_fare
    print(f"Dist: {route_dist} km; Per km Cost= {per_km_cost}, Fixed Cost= {base_fare}")

    return route_cost,route_dist


def call_main():
    # new_search_params QA_test_case_solver_params
    # solver_params99 prod_test1 solver_params_26 pickupdrop_request
    with open('inputs/QA_test_case_solver_params (1).json', 'r') as file:
        solver_params = json.load(file)
        op_json, dropped_nodes = get_best_routes(solver_params)
        print(f"Dropped nodes: {len(set(dropped_nodes))} {dropped_nodes}")
        route_dicts = json.loads(op_json)
        total_routing_cost = 0
        print('*' * 100)
        loc_ids_list = []
        distance = 0
        for route in route_dicts:
            print('-' * 50)
            print(f'Route: {route['route_id']}; Vehicle Type: {route['vehicle_type']}')
            print(f"Location Sequence: {route['unique_drop_loc_id_seq']}")
            loc_ids_list += route['unique_drop_loc_id_seq']
            # print(f"Route: Weight= {route['route_weight_kg']} kg; Volume= {route['route_volume_cft']} cft")
            # print(
            #     f"Vehicle: Max Weight= {route['vehicle_max_weight_kg']} kg; Max Volume= {route['vehicle_max_volume_cft']} cft")
            route_cost,route_dist = calculate_actual_route_cost(solver_params, route['vehicle_type'],
                                                     route['unique_drop_loc_id_seq'])
            print(f'Route cost= {route_cost}')
            distance += route_dist
            total_routing_cost += route_cost
        prod_cost = total_routing_cost
        print('-' * 75)
        print(f"Total cost: {total_routing_cost}")
        print(f"Total distance: {distance} km")
        loc_ids_list = sorted(loc_ids_list)
        print(f"Locations: {loc_ids_list}; {len(loc_ids_list)} ")
        # print('*' * 100)
        # print(f"Deployed Outputs:")
        # op_json, dropped_nodes = deployed_get_best_routes(solver_params)
        # route_dicts = json.loads(op_json)
        # total_routing_cost = 0
        # print('*' * 100)
        # loc_ids_list = []
        # for route in route_dicts:
        #     print('-' * 50)
        #     print(f'Route: {route['route_id']}; Vehicle Type: {route['vehicle_type']}')
        #     print(f"unique_drop_loc_id_seq: {route['unique_drop_loc_id_seq']}")
        #     loc_ids_list += route['unique_drop_loc_id_seq']
        #     print(f"Route: Weight= {route['route_weight_kg']} kg; Volume= {route['route_volume_cft']} cft")
        #     print(
        #         f"Vehicle: Max Weight= {route['vehicle_max_weight_kg']} kg; Max Volume= {route['vehicle_max_volume_cft']} cft")
        #     route_cost = calculate_actual_route_cost(solver_params, route['vehicle_type'],
        #                                              route['unique_drop_loc_id_seq'])
        #     print(f'Route cost= {route_cost}')
        #     total_routing_cost += route_cost
        # print('-' * 75)
        # loc_ids_list = sorted(loc_ids_list)
        # print(f"Prod deployed locations: {loc_ids_list}; {len(loc_ids_list)}")
        # print(f"Total cost: {total_routing_cost}; prod deployed cost: {prod_cost}; diff: {prod_cost-total_routing_cost}")
        print('*' * 100)


if __name__ == "__main__":
    call_main()
