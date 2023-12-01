/*
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

namespace Neural;

public static class ActivationFunctions
{
    public static double Sigmoid(double f)
    {
        return 1.0 / (1.0 + Math.Exp(-f));
    }

    public static double SigmoidPrim(double f)
    {
        return f * (1.0 - f);
    }
}

public record Network(
    int HiddenCount,
    int OutputCount,
    double[][] WeightsHidden,
    double[] BiasesHidden,
    double[][] WeightsOutput,
    double[] BiasesOutput)
{
    public double[] Predict(double[] input)
    {
        double[] y_hidden = new double[HiddenCount];
        double[] y_output = new double[OutputCount];
        return Predict(input, y_hidden, y_output);
    }

    public double[] Predict(double[] input, double[] y_hidden, double[] y_output)
    {
        for (int c = 0; c < WeightsHidden[0].Length; c++)
        {
            double sum = 0.0;
            for (int r = 0; r < WeightsHidden.Length; r++)
            {
                sum += input[r] * WeightsHidden[r][c];
                y_hidden[c] = ActivationFunctions.Sigmoid(sum + BiasesHidden[c]);
            }
        }

        for (int c = 0; c < WeightsOutput[0].Length; c++)
        {
            double sum = 0.0;
            for (int r = 0; r < WeightsOutput.Length; r++)
            {
                sum += y_hidden[r] * WeightsOutput[r][c];
            }

            y_output[c] = ActivationFunctions.Sigmoid(sum + BiasesOutput[c]);
        }

        return y_output;
    }
}

public record Trainer(
    Network Network,
    double[] Hidden,
    double[] Output,
    double[] GradHidden,
    double[] GradOutput)
{
    public static Trainer Create(Network network, int hiddenCount, int outputCount)
    {
        double[] hidden = new double[hiddenCount];
        double[] output = new double[hiddenCount];
        double[] gradHidden = new double[hiddenCount];
        double[] gradOutput = new double[outputCount];
        return new Trainer(network, hidden, output, gradHidden, gradOutput);
    }

    public static Trainer Create(int inputCount, int hiddenCount, int outputCount, Func<double> rand)
    {
        double[] hidden = new double[hiddenCount];
        double[] output = new double[outputCount];
        double[] gradHidden = new double[hiddenCount];
        double[] gradOutput = new double[outputCount];
        double[][] weightsHidden = Enumerable
            .Range(1, inputCount)
            .Select(_ => Array.ConvertAll(new double[hiddenCount], v => rand.Invoke() - 0.5))
            .ToArray();
        double[] biasesHidden = new double[hiddenCount];
        double[][] weightsOutput = Enumerable
            .Range(1, hiddenCount)
            .Select(_ => Array.ConvertAll(new double[outputCount], v => rand.Invoke() - 0.5))
            .ToArray();
        double[] biasesOutput = new double[outputCount];
        Network network = new(hiddenCount, outputCount, weightsHidden, biasesHidden, weightsOutput, biasesOutput);
        return new Trainer(network, hidden, output, gradHidden, gradOutput);
    }

    public void Train (double[] input, double[] y, double lr)
    {
        _ = Network.Predict(input, Hidden, Output);
        for (int c = 0; c < Output.Length; c++)
        {
            GradOutput[c] = (Output[c] - y[c]) * ActivationFunctions.SigmoidPrim(Output[c]);
        }

        for (int r = 0; r < Network.WeightsOutput.Length; r++)
        {
            double sum = 0.0;
            for (int c = 0; c < Network.WeightsOutput[0].Length; c++)
            {
                sum += GradOutput[c] * Network.WeightsOutput[r][c];
            }

            GradHidden[r] = sum * ActivationFunctions.SigmoidPrim(Hidden[r]);
        }

        for (int r = 0; r < Network.WeightsOutput.Length; r++)
        {
            for (int c = 0; c < Network.WeightsOutput[0].Length; c++)
            {
                Network.WeightsOutput[r][c] -= lr * GradOutput[c] * Hidden[r];
            }
        }

        for (int r = 0; r < Network.WeightsHidden.Length; r++)
        {
            for (int c = 0; c < Network.WeightsHidden[0].Length; c++)
            {
                Network.WeightsHidden[r][c] -= lr * GradHidden[c] * input[r];
            }
        }

        for (int c = 0; c < GradOutput.Length; c++)
        {
            Network.BiasesOutput[c] -= lr * GradOutput[c];
        }

        for (int c = 0; c  < GradHidden.Length; c++)
        {
            Network.BiasesHidden[c] -= lr * GradHidden[c];
        }
    }
}
