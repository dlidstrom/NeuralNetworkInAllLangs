﻿/*
Licensed under the MIT License given below.
Copyright 2023 Daniel Lidstrom
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

using System.Globalization;
using System.Text.Json;
using Neural;
using static Neural.Logical;

Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
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

namespace Neural
{
    public static class Extensions
    {
        public static string ToJson(this object o) => JsonSerializer.Serialize(o, new JsonSerializerOptions() { WriteIndented = true });
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
}