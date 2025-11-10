/*
Licensed under the MIT License.
Copyright 2023-2025 Daniel Lidstrom
*/

using System.Globalization;
using System.Text.Json;
using Neural;
using static Neural.Logical;

Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
if (args.FirstOrDefault() == "--logical")
{
    var trainingData = Enumerable.Range(0, 2)
        .SelectMany(x => Enumerable.Range(0, 2), (l, r) => (l, r))
        .Select(p => new { Input = new double[] { p.l, p.r }, Output = new double[] { Xor(p), Xnor(p), Or(p), And(p), Nor(p), Nand(p) } })
        .ToArray();

    Trainer trainer = Trainer.Create(2, 2, 6, Neural.Random.Rand);
    double lr = 1.0;
    int ITERS = 4000;
    for (int e = 0; e < ITERS; e++)
    {
        var sample = trainingData[e % trainingData.Length];
        trainer.Train(sample.Input, sample.Output, lr);
    }

    Network network = trainer.Network;
    Console.WriteLine($"Result after {ITERS} iterations");
    Console.WriteLine("        XOR   XNOR    OR   AND   NOR   NAND");
    foreach (var sample in trainingData)
    {
        double[] pred = network.Predict(sample.Input);
        Console.WriteLine(
            "{0:N0},{1:N0} = {2:N3}  {3:N3} {4:N3} {5:N3} {6:N3}  {7:N3}",
            sample.Input[0],
            sample.Input[1],
            pred[0],
            pred[1],
            pred[2],
            pred[3],
            pred[4],
            pred[5]);
    }

    var networkVals = new
    {
        network.WeightsHidden,
        network.BiasesHidden,
        network.WeightsOutput,
        network.BiasesOutput
    };
    Console.WriteLine($"network: {networkVals.ToJson()}");
}
else if (args.FirstOrDefault() == "--semeion")
{
    // --semeion <file> hiddens epochs lr
    const int inputCount = 16  * 16;
    const int outputCount = 10;
    int hiddenCount = int.Parse(args[2]);
    int EPOCHS = int.Parse(args[3]);
    double LR = double.Parse(args[4]);
    DataItem[] allData = File.ReadAllLines(args[1])
        .Select(line => line.Split(" ").Select(x => double.Parse(x, CultureInfo.InvariantCulture)))
        .Select(number => new DataItem(
           Input: number.Take(inputCount).ToArray(),
            Output: number.Skip(inputCount).Take(outputCount).ToArray()))
        .ToArray();
    Trainer trainer = Trainer.Create(inputCount, hiddenCount, outputCount, Neural.Random.Rand);
    for (int e = 0; e < EPOCHS; e++)
    {
        allData.Shuffle(Neural.Random.Rand);
        foreach (DataItem dataItem in allData)
        {
            trainer.Train(dataItem.Input, dataItem.Output, LR);
        }

        // compute accuracy
        (double confidence, int digit)[] predictions =
            allData.Select(x => trainer.Network.Predict(x.Input).Select((n, i) => (n, i)).Max())
            .ToArray();
        int[] actual =
            allData.Select(x => x.Output.Select((n, i) => (n, i)).Max().i)
            .ToArray();
        double correct = predictions.Zip(actual, (l, r) => l.digit == r ? 1.0 : 0.0).Sum();
        double accuracy = correct / allData.Length;

        // compute confidencies
        double averageConfidence = predictions.Select(x => x.confidence).Average();
        Console.WriteLine($"accuracy: {accuracy:P3} ({correct:G0}/{allData.Length}), avg confidence: {averageConfidence:P3}");
    }

    // predict an item
    DataItem predictItem = allData[10];
    (double, int)[] preds = trainer.Network.Predict(predictItem.Input).Select((n, i) => (n, i)).ToArray();
    int maxPred = preds.Max().Item2;
    string pixels = string.Join(
        "",
        predictItem.Input.Select(x => x.ToString("N0")))
        .Replace("1", "*").Replace("0", " ");
    Console.WriteLine(
        string.Join(
            Environment.NewLine,
            pixels.Chunk(16).Select(x => new string(x))));
    Console.WriteLine("Prediction (output from network for the above input):");
    foreach ((double conf, int n) in preds)
    {
        Console.Write($"{n}: {conf,8:P3}");
        if (n == maxPred)
        {
            Console.Write(" <-- best prediction");
        }

        Console.WriteLine();
    }
}
else {
  Console.WriteLine("Specify --logical or --semeion <file>");
}

namespace Neural
{
    public static class Extensions
    {
        public static string ToJson(this object o) => JsonSerializer.Serialize(o, new JsonSerializerOptions() { WriteIndented = true });
        public static void Shuffle<T>(this T[] list, Func<double> random)
        {
            for (var i = 0; i < list.Length; i++)
            {
                var j = (int)Math.Floor(random.Invoke() * list.Length);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }
    }

    public static class Random
    {
        private static readonly uint P = 2147483647;
        private static readonly uint A = 16807;
        private static uint current = 1;
        public static double Rand()
        {
            current = current * A % P;
            double result = (double)current / P;
            return result;
        }
    }

    public static class Logical
    {
        public static int Xor((int a, int b) p) => p.a ^ p.b;
        public static int Xnor((int a, int b) p) => 1 - Xor(p);
        public static int Or((int a, int b) p) => p.a | p.b;
        public static int And((int a, int b) p) => p.a & p.b;
        public static int Nand((int a, int b) p) => 1 - And(p);
        public static int Nor((int a, int b) p) => 1 - Or(p);
    }

    public record DataItem(double[] Input, double[] Output);
}
