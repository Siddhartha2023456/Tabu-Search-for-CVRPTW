using System;
using System.Collections.Generic;
using System.Data;
using System.Globalization;
using System.IO;
using System.Linq;
using CsvHelper;
using CsvHelper.Configuration;
using System.Collections.Concurrent;
using ExcelDataReader;

public class Program
{
    // Constants
    private const int service_time_customer = 20;
    private const int service_time_depot = 60;
    private const string depot = "A123";
    private const int max_iter = 1000;
    private const int tabu_tenure = 10;

    private static Dictionary<string, Dictionary<string, double>> travelMatrix;
    private static List<Dictionary<string, object>> orders;
    private static List<Dictionary<string, object>> trucks;
    private static List<string> locations;
    private static ConcurrentQueue<Dictionary<string, List<string>>> tabuList = new ConcurrentQueue<Dictionary<string, List<string>>>();

    public static void Main()
    {
        System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);

        // Load data files
        var locationsData = LoadCsv("C:/Users/Acer/Downloads/locations.csv");
        var orderListData = LoadExcel("C:/Users/Acer/Downloads/order_list_1.xlsx");
        var travelMatrixData = LoadCsv("C:/Users/Acer/Downloads/travel_matrix.csv");
        var trucksData = LoadCsv("C:/Users/Acer/Downloads/trucks.csv");

        // Initialize data
        locations = locationsData.AsEnumerable().Select(r => r["location_code"].ToString()).ToList();
        orders = ConvertToListOfDictionaries(orderListData);
        travelMatrix = ConvertToNestedDictionary(travelMatrixData, "source_location_code", "destination_location_code");
        trucks = ConvertToListOfDictionaries(trucksData);

        // Run the Tabu Search algorithm
        RunTabuSearch();
    }

    // Utility Functions
    public static DataTable LoadCsv(string filePath)
    {
        using var reader = new StreamReader(filePath);
        using var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture) { HasHeaderRecord = true });
        using var dataTable = new DataTable();
        using var dr = new CsvDataReader(csv);
        dataTable.Load(dr);
        return dataTable;
    }

    public static DataTable LoadExcel(string filePath)
    {
        using var stream = File.Open(filePath, FileMode.Open, FileAccess.Read);
        using var reader = ExcelReaderFactory.CreateReader(stream);
        var result = reader.AsDataSet();
        return result.Tables[0];
    }

    public static List<Dictionary<string, object>> ConvertToListOfDictionaries(DataTable table)
    {
        return table.AsEnumerable().Select(row => table.Columns.Cast<DataColumn>().ToDictionary(col => col.ColumnName, col => row[col])).ToList();
    }

    public static Dictionary<string, Dictionary<string, double>> ConvertToNestedDictionary(DataTable table, string key1, string key2)
    {
        return table.AsEnumerable()
                    .GroupBy(row => row[key1].ToString())
                    .ToDictionary(
                        g => g.Key,
                        g => g.ToDictionary(
                            row => row[key2].ToString(),
                            row => Convert.ToDouble(row["travel_distance_in_km"]) // Ensure travel distance is double
                        )
                    );
    }

    public static int CalculateCost(Dictionary<string, List<string>> solution)
    {
        int totalCost = 0;
        foreach (var route in solution.Values)
        {
            for (int i = 0; i < route.Count - 1; i++)
            {
                if (travelMatrix.TryGetValue(route[i], out var destinations) &&
                    destinations.TryGetValue(route[i + 1], out var travelDistance))
                {
                    totalCost += (int)travelDistance;
                }
            }
        }
        return totalCost;
    }

    // Generate an initial feasible solution
    public static Dictionary<string, List<string>> GenerateInitialSolution()
    {
        var random = new Random();
        var initialSolution = trucks.ToDictionary(truck => truck["truck_id"].ToString(), truck => new List<string> { depot });

        foreach (var order in orders)
        {
            if (order.TryGetValue("Column2", out var destinationCode))
            {
                var assignedTruck = trucks[random.Next(trucks.Count)];
                string truckId = assignedTruck["truck_id"].ToString();
                initialSolution[truckId].Add(destinationCode.ToString());
            }
            else
            {
                Console.WriteLine("Warning: 'Destination Code' key not found in order. Skipping this order.");
            }
        }

        foreach (var route in initialSolution.Values)
        {
            route.Add(depot);
        }

        return initialSolution;
    }

    public static void RunTabuSearch()
    {
        var bestSolution = GenerateInitialSolution();
        int bestCost = CalculateCost(bestSolution);
        var currentSolution = new Dictionary<string, List<string>>(bestSolution);
        int currentCost = bestCost;

        for (int iteration = 0; iteration < max_iter; iteration++)
        {
            var neighbors = GenerateNeighbors(currentSolution);
            Dictionary<string, List<string>> bestNeighbor = null;
            int bestNeighborCost = int.MaxValue;

            foreach (var neighbor in neighbors)
            {
                int neighborCost = CalculateCost(neighbor);
                if (neighborCost < bestNeighborCost && !IsTabu(neighbor))
                {
                    bestNeighbor = neighbor;
                    bestNeighborCost = neighborCost;
                }
            }

            if (bestNeighbor != null && bestNeighborCost < currentCost)
            {
                currentSolution = new Dictionary<string, List<string>>(bestNeighbor);
                currentCost = bestNeighborCost;
                tabuList.Enqueue(currentSolution);

                if (tabuList.Count > tabu_tenure)
                {
                    tabuList.TryDequeue(out _);
                }

                if (currentCost < bestCost)
                {
                    bestSolution = new Dictionary<string, List<string>>(currentSolution);
                    bestCost = currentCost;
                }
            }

            Console.WriteLine($"Iteration {iteration + 1}: Best Cost = {bestCost}");
        }

        Console.WriteLine("Optimal Routes:");
        foreach (var truck in bestSolution)
        {
            Console.WriteLine($"Truck {truck.Key}: Route - {string.Join(" -> ", truck.Value)}");
        }
        Console.WriteLine($"Minimum Cost: {bestCost}");
    }

    public static bool IsTabu(Dictionary<string, List<string>> solution)
    {
        return tabuList.Any(tabuSolution =>
            tabuSolution.Count == solution.Count &&
            tabuSolution.All(tabuRoute =>
                solution.TryGetValue(tabuRoute.Key, out var solutionRoute) &&
                tabuRoute.Value.SequenceEqual(solutionRoute)
            )
        );
    }

    public static List<Dictionary<string, List<string>>> GenerateNeighbors(Dictionary<string, List<string>> currentSolution)
    {
        var neighbors = new List<Dictionary<string, List<string>>>();
        foreach (var truck in currentSolution)
        {
            var route = truck.Value;
            for (int i = 1; i < route.Count - 2; i++)
            {
                for (int j = i + 1; j < route.Count - 1; j++)
                {
                    var newRoute = new List<string>(route);
                    (newRoute[i], newRoute[j]) = (newRoute[j], newRoute[i]);

                    var neighbor = new Dictionary<string, List<string>>(currentSolution)
                    {
                        [truck.Key] = newRoute
                    };
                    neighbors.Add(neighbor);
                }
            }
        }
        return neighbors;
    }
}
