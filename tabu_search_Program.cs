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
    private const int max_iter = 100;
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
    public static Dictionary<string, List<string>> GetProvidedInitialSolution()
    {
        Console.WriteLine("Using provided initial solution...");
        return new Dictionary<string, List<string>>
    {
        {"T3_1", new List<string> {"A123", "10364446", "10208593", "10219625", "10208746", "10208681", "12715142", "12715142", "12585577", "12345850", "11956776", "11957296", "10208730", "12854098", "10208776", "A123"}},
        {"T3_2", new List<string> {"A123", "12742843", "12854117", "12854117", "12420540", "10208641", "12854149", "10208614", "11956776", "12854137", "12854121", "12474916", "12854192", "12475665", "A123"}},
        {"T3_3", new List<string> {"A123", "12577977", "12854137", "10208702", "12854164", "10208607", "12475516", "10208698", "12995436", "12854121", "12854159", "10208589", "12475360", "10208760", "10364446", "12044475", "12854148", "10208605", "10208681", "12489344", "12854113", "A123"}},
        {"T3_4", new List<string> {"A123", "13005196", "11950159", "10208698", "12854126", "12231175", "10208582", "11956776", "10279711", "10212334", "12585577", "A123"}},
        {"T3_5", new List<string> {"A123", "12854184", "12742843", "10208117", "12854139", "10208659", "12725145", "12854149", "12854197", "10208762", "10212202", "13013290", "11539696", "10364446", "12854130", "A123"}},
        {"T3_6", new List<string> {"A123", "12854133", "12838671", "11346601", "12740556", "10208675", "10218045", "12026582", "10210421", "10208768", "12750741", "12854127", "10364922", "A123"}},
        {"T7_1", new List<string> {"A123", "10366979", "10208776", "12721000", "10208661", "12854147", "10310757", "10208605", "10218045", "12854108", "10208730", "12750741", "10208675", "A123"}},
        {"T7_2", new List<string> {"A123", "11539696", "13067405", "13067657", "12474916", "10218045", "10218045", "A123"}},
        {"T10_1", new List<string> {"A123", "12740556", "10208639", "12854133", "12854195", "12750741", "11957296", "10208698", "12854140", "10214844", "A123"}},
        {"T10_2", new List<string> {"A123", "12044475", "10212258", "12656560", "12475623", "12750741", "10208686", "11956776", "11956776", "12854147", "10208701", "12420540", "A123"}},
        {"T10_3", new List<string> {"A123", "10364896", "12854138", "10364446", "10214844", "12715142", "10208499", "10208499", "12854172", "10208696", "10208707", "12854198", "11956776", "12750741", "10208659", "10208659", "10208719", "A123"}},
        {"T10_4", new List<string> {"A123", "10393503", "12854173", "10310590", "A123"}},
        {"T10_5", new List<string> {"A123", "12708761", "12171444", "10208712", "12724371", "12546176", "13005196", "10364446", "12219936", "A123"}},
        {"T10_6", new List<string> {"A123", "12738762", "10210421", "12860572", "10208120", "10364922", "12854153", "10366955", "12854171", "12502170", "13015040", "10208602", "10208602", "10208596", "10208595", "12854154", "A123"}},
        {"T10_7", new List<string> {"A123", "12475523", "10208764", "10208575", "12769191", "10364896", "12779571", "A123"}},
        {"T40_1", new List<string> {"A123", "10208694", "10208746", "13005196", "12475628", "10208670", "10364446", "10210421", "12860572", "11957296", "10208595", "10208119", "10219625", "10208651", "12854172", "12194109", "A123"}},
        {"T40_2", new List<string> {"A123", "10218045", "10208747", "12860572", "10208768", "12983715", "10279711", "10279711", "10279711", "12854166", "10208717", "12021664", "12750741", "11646201", "A123"}},
        {"T40_3", new List<string> {"A123", "10364446", "12854198", "10208629", "12171479", "10208731", "13045617", "10208651", "11539696", "10208701", "12750741", "12750741", "13005196", "A123"}},
        {"T40_4", new List<string> {"A123", "12194109", "10279711", "10212373", "12475666", "12628669", "12854192", "12690815", "12750741", "10208694", "A123"}}
    };
    }

    public static void RunTabuSearch()
    {
        int tabu_tenure = 20;
        var frequencyMap = new Dictionary<string, int>();

        var bestSolution = GetProvidedInitialSolution(); // Initial solution
        int bestCost = CalculateCost(bestSolution);
        var currentSolution = new Dictionary<string, List<string>>(bestSolution);
        int currentCost = bestCost;

        Console.WriteLine($"Starting Tabu Search with initial cost: {bestCost}");

        int iteration = 0;
        while (iteration < 100) // Stopping criteria set to 100 iterations
        {
            // Dynamically adjust tabu tenure
            if (iteration % 5 == 0)
            {
                tabu_tenure = bestCost - currentCost > 0
                    ? Math.Max(tabu_tenure - 1, 5)
                    : Math.Min(tabu_tenure + 1, 20);
            }

            var neighbors = GenerateNeighbors(currentSolution);
            Dictionary<string, List<string>> bestNeighbor = null;
            int bestNeighborCost = int.MaxValue;

            // Select the best neighbor with controlled randomness
            if (neighbors.Any())
            {
                bestNeighbor = SelectBestNeighbor(neighbors, frequencyMap, 5, iteration);
                bestNeighborCost = CalculateCost(bestNeighbor);
            }

            if (bestNeighbor != null && !IsTabu(bestNeighbor))
            {
                currentSolution = new Dictionary<string, List<string>>(bestNeighbor);
                currentCost = bestNeighborCost;

                // Update tabu list
                tabuList.Enqueue(currentSolution);
                if (tabuList.Count > tabu_tenure)
                {
                    tabuList.TryDequeue(out _);
                }

                // Update the best solution
                if (currentCost < bestCost)
                {
                    bestSolution = new Dictionary<string, List<string>>(currentSolution);
                    bestCost = currentCost;
                }
            }
            else
            {
                Console.WriteLine($"Iteration {iteration + 1}: No low cost neighbors found.");
            }

            // Update frequency map
            UpdateFrequency(frequencyMap, currentSolution);

            // Logging
            Console.WriteLine($"Iteration {iteration + 1}: Best Cost = {bestCost}, Tabu Tenure = {tabu_tenure}");

            // Increment iteration
            iteration++;
        }

        // Output the best solution
        Console.WriteLine("Optimal Routes:");
        foreach (var truck in bestSolution)
        {
            Console.WriteLine($"Truck {truck.Key}: Route - {string.Join(" -> ", truck.Value)}");
        }
        Console.WriteLine($"Minimum Cost: {bestCost}");
    }
    public static int CalculateDiversifiedCost(Dictionary<string, List<string>> solution, Dictionary<string, int> frequencyMap, int iteration)
    {
        int cost = CalculateCost(solution);
        int penaltyWeight = Math.Min(10 + iteration / 5, 50); // Gradually increase penalty weight
        int penalty = solution.Values.Sum(route => route.Sum(location => frequencyMap.GetValueOrDefault(location, 0) * penaltyWeight));
        return cost + penalty;
    }

    public static void UpdateFrequency(Dictionary<string, int> frequencyMap, Dictionary<string, List<string>> solution)
    {
        foreach (var route in solution.Values)
        {
            foreach (var location in route)
            {
                if (location != depot)
                {
                    if (!frequencyMap.ContainsKey(location))
                        frequencyMap[location] = 0;
                    frequencyMap[location]++;
                }
            }
        }
    }

    public static bool IsTabu(Dictionary<string, List<string>> solution)
    {
        return tabuList.Any(tabuSolution =>
            tabuSolution.Count == solution.Count &&
            tabuSolution.All(tabuRoute =>
                solution.TryGetValue(tabuRoute.Key, out var solutionRoute) &&
                CalculateRouteSimilarity(tabuRoute.Value, solutionRoute) > 0.5 // Allow 10% difference
            )
        );
    }

    public static double CalculateRouteSimilarity(List<string> route1, List<string> route2)
    {
        if (route1.Count != route2.Count) return 0;

        int matchCount = route1.Where((location, index) => location == route2[index]).Count();
        return (double)matchCount / route1.Count;
    }

    public static Dictionary<string, List<string>> SelectBestNeighbor(List<Dictionary<string, List<string>>> neighbors, Dictionary<string, int> frequencyMap, int k, int iteration)
    {
        var scoredNeighbors = neighbors
            .Select(neighbor => new { Solution = neighbor, Cost = CalculateDiversifiedCost(neighbor, frequencyMap, iteration) })
            .OrderBy(n => n.Cost)
            .Take(k)
            .ToList();

        Random random = new Random();
        return scoredNeighbors[random.Next(scoredNeighbors.Count)].Solution;
    }

    public static List<Dictionary<string, List<string>>> GenerateNeighbors(Dictionary<string, List<string>> currentSolution)
    {
        var neighbors = new List<Dictionary<string, List<string>>>();

        foreach (var truck in currentSolution.Keys.ToList())
        {
            var route = currentSolution[truck];

            // Perform intra-route swaps
            for (int i = 1; i < route.Count - 1; i++)
            {
                for (int j = i + 1; j < route.Count - 1; j++)
                {
                    var newRoute = new List<string>(route);
                    (newRoute[i], newRoute[j]) = (newRoute[j], newRoute[i]);

                    var neighbor = new Dictionary<string, List<string>>(currentSolution)
                    {
                        [truck] = newRoute
                    };
                    neighbors.Add(neighbor);
                }
            }

            // Perform inter-route swaps
            foreach (var otherTruck in currentSolution.Keys.Where(k => k != truck))
            {
                var otherRoute = currentSolution[otherTruck];
                for (int i = 1; i < route.Count - 1; i++)
                {
                    for (int j = 1; j < otherRoute.Count - 1; j++)
                    {
                        var newRoute1 = new List<string>(route);
                        var newRoute2 = new List<string>(otherRoute);

                        (newRoute1[i], newRoute2[j]) = (newRoute2[j], newRoute1[i]);

                        var neighbor = new Dictionary<string, List<string>>(currentSolution)
                        {
                            [truck] = newRoute1,
                            [otherTruck] = newRoute2
                        };
                        neighbors.Add(neighbor);
                    }
                }
            }
        }

        return neighbors;
    }
}



