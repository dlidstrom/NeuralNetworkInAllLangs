public record Trainer(
  Network Network,
  double[] Hidden,
  double[] Output,
  double[] GradHidden,
  double[] GradOutput)
{
  static Trainer Create(Network network, int hiddenCount, int outputCount) {
    double[] hidden = new double[hiddenCount];
    double[] output = new double[hiddenCount];
    double[] gradHidden = new double[hiddenCount];
    double[] gradOutput = new double[outputCount];
    return new Trainer(network, hidden, output, gradHidden, gradOutput);
  }

  public static Trainer Create(int inputCount, int hiddenCount, int outputCount, Func<double> rand) {
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

      for (int c = 0; c < GradHidden.Length; c++)
      {
          Network.BiasesHidden[c] -= lr * GradHidden[c];
      }
  }
}
