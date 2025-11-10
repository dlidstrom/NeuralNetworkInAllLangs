/*
Licensed under the MIT License.
Copyright 2023-2025 Daniel Lidstrom
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
    int InputCount,
    int HiddenCount,
    int OutputCount,
    double[] WeightsHidden,
    double[] BiasesHidden,
    double[] WeightsOutput,
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
        for (int c = 0; c < HiddenCount; c++)
        {
            double sum = 0.0;
            for (int r = 0; r < InputCount; r++)
            {
                sum += input[r] * WeightsHidden[r * HiddenCount + c];
            }

            y_hidden[c] = ActivationFunctions.Sigmoid(sum + BiasesHidden[c]);
        }

        for (int c = 0; c < OutputCount; c++)
        {
            double sum = 0.0;
            for (int r = 0; r < HiddenCount; r++)
            {
                sum += y_hidden[r] * WeightsOutput[r * OutputCount + c];
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
        double[] output = new double[outputCount];
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
        double[] weightsHidden = Array.ConvertAll(new double[inputCount * hiddenCount], v => rand.Invoke() - 0.5);
        double[] biasesHidden = new double[hiddenCount];
        double[] weightsOutput = Array.ConvertAll(new double[hiddenCount * outputCount], v => rand.Invoke() - 0.5);
        double[] biasesOutput = new double[outputCount];
        Network network = new(inputCount, hiddenCount, outputCount, weightsHidden, biasesHidden, weightsOutput, biasesOutput);
        return new Trainer(network, hidden, output, gradHidden, gradOutput);
    }

    public void Train (double[] input, double[] y, double lr)
    {
        _ = Network.Predict(input, Hidden, Output);
        for (int c = 0; c < Network.OutputCount; c++)
        {
            GradOutput[c] = (Output[c] - y[c]) * ActivationFunctions.SigmoidPrim(Output[c]);
        }

        for (int r = 0; r < Network.HiddenCount; r++)
        {
            double sum = 0.0;
            for (int c = 0; c < Network.OutputCount; c++)
            {
                sum += GradOutput[c] * Network.WeightsOutput[r * Network.OutputCount + c];
            }

            GradHidden[r] = sum * ActivationFunctions.SigmoidPrim(Hidden[r]);
        }

        for (int r = 0; r < Network.HiddenCount; r++)
        {
            for (int c = 0; c < Network.OutputCount; c++)
            {
                Network.WeightsOutput[r * Network.OutputCount + c] -= lr * GradOutput[c] * Hidden[r];
            }
        }

        for (int r = 0; r < Network.InputCount; r++)
        {
            for (int c = 0; c < Network.HiddenCount; c++)
            {
                Network.WeightsHidden[r * Network.HiddenCount + c] -= lr * GradHidden[c] * input[r];
            }
        }

        for (int c = 0; c < Network.OutputCount; c++)
        {
            Network.BiasesOutput[c] -= lr * GradOutput[c];
        }

        for (int c = 0; c < Network.HiddenCount; c++)
        {
            Network.BiasesHidden[c] -= lr * GradHidden[c];
        }
    }
}
