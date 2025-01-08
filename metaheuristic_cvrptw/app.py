import streamlit as st
import pandas as pd
import numpy as np
import time
from tabu_assignment_data import tabu_search

# Streamlit app
st.title("Tabu Search Optimization")

# Sidebar for inputs
st.sidebar.header("Input Parameters")
uploaded_file = st.sidebar.file_uploader("Upload Data File (Excel with 4 Sheets)", type="xlsx")
max_iterations = st.sidebar.number_input("Max Iterations", min_value=1, max_value=10000, value=100)
no_improvement_count = st.sidebar.number_input("No Improvement Count", min_value=1, max_value=100, value=3)
tabu_tenure = st.sidebar.number_input("Tabu Tenure", min_value=1, max_value=100, value=10)

if uploaded_file:
    try:
        # Load the Excel file
        excel_data = pd.ExcelFile(uploaded_file)
        locations_df = excel_data.parse(sheet_name="LOCATIONS")
        order_list_df = excel_data.parse(sheet_name="ORDERS")
        travel_matrix_df = excel_data.parse(sheet_name="TRAVEL MATRIX")
        trucks_df = excel_data.parse(sheet_name="TRUCKS")

        # Display dataset previews
        st.write("Locations Dataset Preview:")
        st.dataframe(locations_df.head())

        st.write("Orders Dataset Preview:")
        st.dataframe(order_list_df.head())

        st.write("Travel Matrix Dataset Preview:")
        st.dataframe(travel_matrix_df.head())

        st.write("Trucks Dataset Preview:")
        st.dataframe(trucks_df.head())

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
        locations_df = locations_df[locations_df['location_code'].isin(dest + ['A123'])]
        locations_df['start_minutes'] = pd.to_datetime(locations_df['location_loading_unloading_window_start'], format='%H:%M:%S').dt.hour * 60 + pd.to_datetime(locations_df['location_loading_unloading_window_start'], format='%H:%M:%S').dt.minute
        locations_df['end_minutes'] = pd.to_datetime(locations_df['location_loading_unloading_window_end'], format='%H:%M:%S').dt.hour * 60 + pd.to_datetime(locations_df['location_loading_unloading_window_end'], format='%H:%M:%S').dt.minute
        st.write("halfway reached")
        customers = locations_df.sort_values(by='location_code').iloc[:len(order_list_df1), :]
        locations_df2 = locations_df.sort_values(by='location_code')
        cap_df = dict(zip(trucks_df['truck_type'], trucks_df['truck_max_weight']))
        depot = locations_df.sort_values(by='location_code').iloc[len(order_list_df1):, :]
        Nodes = pd.concat([depot, customers], ignore_index=True)
        vehicles = [k for k in range(0, len(Q1))]
        customers = [i for i in range(1, len(Nodes))]
        nodes = [i for i in range(0, len(Nodes))]
        demands_w = [0] + list(order_list_df1['Total Weight'])
        demands_w = [int(d) for d in demands_w]
        start_time = list(Nodes['start_minutes'])
        finish_time = list(Nodes['end_minutes'])
        dest2 = ['A123'] + sorted(dest)
        dest3 = {}
        for i in range(len(dest2)):
            dest3[dest2[i]] = i

        # Convert source and destination codes to strings
        travel_matrix_df['source_location_code'] = travel_matrix_df['source_location_code'].astype(str)
        travel_matrix_df['destination_location_code'] = travel_matrix_df['destination_location_code'].astype(str)

        # Filter rows where source and destination codes are in the 'dest' list or 'A123'
        valid_codes = set(dest + ['A123'])
        filtered_df = travel_matrix_df[
            travel_matrix_df['source_location_code'].isin(valid_codes) &
            travel_matrix_df['destination_location_code'].isin(valid_codes)
        ]

        # Map the source and destination codes using 'dest3' dictionary
        filtered_df['mapped_source'] = filtered_df['source_location_code'].map(dest3)
        filtered_df['mapped_destination'] = filtered_df['destination_location_code'].map(dest3)

        # Initialize distance and time matrices
        dist_matrix = {}
        time_matrix = {}

        # Iterate over the rows of filtered_df
        for _, row in filtered_df.iterrows():
            source = row['mapped_source']
            destination = row['mapped_destination']
            dist_matrix[(source, destination)] = row['travel_distance_in_km']
            time_matrix[(source, destination)] = row['travel_time_in_min']


        max_capacity_w = {k: Q1[k] for k in range(len(Q1))}
        st.write(travel_matrix_df.head())
        st.write(filtered_df.head())
        # st.text(dist_matrix)
        # st.text(time_matrix)

        # Run optimization
        if st.sidebar.button("Run Optimization"):
            
            # Ensure all demands and nodes align
            assert len(demands_w) == len(nodes), "Mismatch between demands_w and nodes!"

            # Ensure all vehicles have a max capacity defined
            assert len(vehicles) == len(max_capacity_w), "Mismatch between vehicles and max_capacity_w!"

            # Ensure travel matrix contains all required distances
            for i in nodes:
                for j in nodes:
                    if (i, j) not in dist_matrix:
                        st.warning(f"Missing distance for nodes {i} and {j}") # Assign a large distance for missing
            st.write("All dataset is correct.")
            with st.spinner("Running Tabu Search..."):
                start_time = time.time()
                best_solution, best_cost, cost_progress = tabu_search(nodes, vehicles, dist_matrix, demands_w, max_capacity_w, Q1=Q1, var_cost=var_cost, fixed_cost=fixed_cost,max_iter=max_iterations, tabu_tenure=10)
                end_time = time.time()

                # Display Results
                st.success(f"Optimization completed in {end_time - start_time:.2f} seconds")

                # Parse and display the best solution in a human-readable format
                st.subheader("Best Solution: Vehicle Routes")
                for vehicle, route in best_solution.items():
                    if len(route) > 2:
                        route_str = " → ".join(map(str, route))  # Convert route to a readable format
                        st.write(f"**Vehicle {vehicle + 1}:** Depot → {route_str[4:-4]} → Depot")

                st.write(f"Best Cost: {best_cost}")
                st.line_chart(cost_progress)
                
    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.sidebar.warning("Please upload the required Excel file.")
