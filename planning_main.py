import json
from ortools_cvrptw.routing_ortools_prod import get_best_routes
from utils import prep_data, create_data_model


def calculate_actual_route_cost(input_data, veh_type, seq):
    route_cost = 0
    route_dist = 0
    for i in range(len(seq) - 1):
        route_dist += input_data['costs'][seq[i]][seq[i + 1]]
    per_km_cost = route_dist * input_data['perKmCostPerVehicle'][veh_type]
    route_cost += per_km_cost
    base_fare = input_data['vehicle_base_fare_matrix'][seq[0]][veh_type]
    route_cost += base_fare
    hop_cost = (len(seq) - 1) * input_data['hop_fare'][veh_type]
    route_cost += hop_cost
    print(f"Dist: {route_dist} km; dist_cost= {per_km_cost}, base fare= {base_fare}, hop_cost= {hop_cost}")

    return route_cost


def call_main():
    # new_search_params QA_test_case_solver_params
    # solver_params99 prod_test1 solver_params_26 pickupdrop_request
    with open('inputs/solver_params_26.json', 'r') as file:
        solver_params = json.load(file)
        model_data = prep_data(solver_params)

        # Instantiate the data problem.
        data = create_data_model(model_data)
        op_json = get_best_routes(data)
        route_dicts = json.loads(op_json)
        total_routing_cost = 0
        print('*' * 100)
        for route in route_dicts:
            print('-' * 50)
            print(f'Route: {route['route_id']}; Vehicle Type: {route['vehicle_type']}')
            print(route['unique_drop_loc_id_seq'])
            print(f"Route: Weight= {route['route_weight_kg']} kg; Volume= {route['route_volume_cft']} cft")
            print(
                f"Vehicle: Max Weight= {route['vehicle_max_weight_kg']} kg; Max Volume= {route['vehicle_max_volume_cft']} cft")
            route_cost = calculate_actual_route_cost(solver_params, route['vehicle_type'],
                                                     route['unique_drop_loc_id_seq'])
            print(f'Route cost= {route_cost}')
            total_routing_cost += route_cost
        prod_cost = total_routing_cost
        # print('-' * 75)
        # print(f"Total cost: {total_routing_cost}")
        # print('*' * 100)
        # op_json = get_best_routes_true_cost(solver_params)
        # route_dicts = json.loads(op_json)
        # total_routing_cost = 0
        # print('*' * 100)
        # for route in route_dicts:
        #     print('-' * 50)
        #     print(f'Route: {route['route_id']}; Vehicle Type: {route['vehicle_type']}')
        #     print(route['unique_drop_loc_id_seq'])
        #     print(f"Route: Weight= {route['route_weight_kg']} kg; Volume= {route['route_volume_cft']} cft")
        #     print(
        #         f"Vehicle: Max Weight= {route['vehicle_max_weight_kg']} kg; Max Volume= {route['vehicle_max_volume_cft']} cft")
        #     route_cost = calculate_actual_route_cost(solver_params, route['vehicle_type'],
        #                                              route['unique_drop_loc_id_seq'])
        #     print(f'Route cost= {route_cost}')
        #     total_routing_cost += route_cost
        # print('-' * 75)
        # print(f"Total cost: {total_routing_cost}; prod deployed cost: {prod_cost}; diff: {prod_cost-total_routing_cost}")
        print('*' * 100)


if __name__ == "__main__":
    call_main()
