public record Neural(
  int hiddenCount,
  int outputCount,
  double[][] weightsHidden,
  double[] biasesHidden,
  double[][] weightsOutput,
  double[] biasesOutput) {
  static class ActivationFunctions {
    static double sigmoid(double f) {
      return 1.0 / (1.0 + Math.exp(-f));
    }
  }

  static void print() {
    System.out.println("From Neural");
  }

  double[] Predict(double[] input) {
    double[] y_hidden = new double[hiddenCount];
    double[] y_output = new double[outputCount];
    return Predict(input, y_hidden, y_output);
  }

  public double[] Predict(double[] input, double[] y_hidden, double[] y_output) {
    for (int c = 0; c < weightsHidden[0].length; c++) {
      double sum = 0.0;
      for (int r = 0; r < weightsHidden.length; r++) {
        sum += input[r] * weightsHidden[r][c];
        y_hidden[c] = ActivationFunctions.sigmoid(sum + biasesHidden[c]);
      }
    }

    for (int c = 0; c < weightsOutput[0].length; c++) {
      double sum = 0.0;
      for (int r = 0; r < weightsOutput.length; r++) {
        sum += y_hidden[r] * weightsOutput[r][c];
      }

      y_output[c] = ActivationFunctions.sigmoid(sum + biasesOutput[c]);
    }

      return y_output;
  }
}
