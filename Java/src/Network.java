/*
Licensed under the MIT License given below.
Copyright 2024 Daniel Lidstrom
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

public class Network {
    int inputCount;
    int hiddenCount;
    int outputCount;
    double[] weightsHidden;
    double[] biasesHidden;
    double[] weightsOutput;
    double[] biasesOutput;

    public Network(
        int inputCount,
        int hiddenCount,
        int outputCount,
        double[] weightsHidden,
        double[] biasesHidden,
        double[] weightsOutput,
        double[] biasesOutput) {
        this.inputCount = inputCount;
        this.hiddenCount = hiddenCount;
        this.outputCount = outputCount;
        this.weightsHidden = weightsHidden;
        this.biasesHidden = biasesHidden;
        this.weightsOutput = weightsOutput;
        this.biasesOutput = biasesOutput;
    }

    public int inputCount() {
        return inputCount;
    }

    public int hiddenCount() {
        return hiddenCount;
    }

    public int outputCount() {
        return outputCount;
    }

    public double[] weightsHidden() {
        return weightsHidden;
    }

    public double[] biasesHidden() {
        return biasesHidden;
    }

    public double[] weightsOutput() {
       return weightsOutput;
    }

    public double[] biasesOutput() {
        return biasesOutput;
    }

    public double[] predict(double[] input) {
        double[] yHidden = new double[hiddenCount];
        double[] yOutput = new double[outputCount];
        return predict(input, yHidden, yOutput);
    }

    public double[] predict(double[] input, double[] yHidden, double[] yOutput) {
        for (int c = 0; c < hiddenCount; c++) {
            double sum = 0.0;
            for (int r = 0; r < inputCount; r++) {
                sum += input[r] * weightsHidden[r * hiddenCount + c];
            }

            yHidden[c] = ActivationFunctions.sigmoid(sum + biasesHidden[c]);
        }

        for (int c = 0; c < outputCount; c++) {
            double sum = 0.0;
            for (int r = 0; r < hiddenCount; r++) {
                sum += yHidden[r] * weightsOutput[r * outputCount + c];
            }

            yOutput[c] = ActivationFunctions.sigmoid(sum + biasesOutput[c]);
        }

        return yOutput;
    }

    public static class ActivationFunctions {
        public static double sigmoid(double f) {
            return 1.0 / (1.0 + Math.exp(-f));
        }

        public static double sigmoidPrim(double f) {
            return f * (1.0 - f);
        }
    }
}
