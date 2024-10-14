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

import java.util.function.Supplier;

public class Trainer {
    Network network;
    double[] hidden;
    double[] output;
    double[] gradHidden;
    double[] gradOutput;

    public Trainer(
        Network network,
        double[] hidden,
        double[] output,
        double[] gradHidden,
        double[] gradOutput) {
        this.network = network;
        this.hidden = hidden;
        this.output = output;
        this.gradHidden = gradHidden;
        this.gradOutput = gradOutput;
    }

    public Network network() {
        return this.network;
    }

    public static Trainer create(Network network, int hiddenCount, int outputCount) {
        double[] hidden = new double[hiddenCount];
        double[] output = new double[outputCount];
        double[] gradHidden = new double[hiddenCount];
        double[] gradOutput = new double[outputCount];
        return new Trainer(network, hidden, output, gradHidden, gradOutput);
    }

    public static Trainer create(int inputCount, int hiddenCount, int outputCount, Supplier<Double> rand) {
        double[] hidden = new double[hiddenCount];
        double[] output = new double[outputCount];
        double[] gradHidden = new double[hiddenCount];
        double[] gradOutput = new double[outputCount];
        double[] weightsHidden = new double[inputCount * hiddenCount];
        for (int i = 0; i < weightsHidden.length; i++) {
            weightsHidden[i] = rand.get() - 0.5;
        }
        double[] biasesHidden = new double[hiddenCount];
        double[] weightsOutput = new double[hiddenCount * outputCount];
        for (int i = 0; i < weightsOutput.length; i++) {
            weightsOutput[i] = rand.get() - 0.5;
        }
        double[] biasesOutput = new double[outputCount];
        Network network = new Network(inputCount, hiddenCount, outputCount, weightsHidden, biasesHidden, weightsOutput, biasesOutput);
        return new Trainer(network, hidden, output, gradHidden, gradOutput);
    }

    public void train(double[] input, double[] y, double lr) {
        network.predict(input, hidden, output);
        for (int c = 0; c < network.outputCount(); c++) {
            gradOutput[c] = (output[c] - y[c]) * Network.ActivationFunctions.sigmoidPrim(output[c]);
        }

        for (int r = 0; r < network.hiddenCount(); r++) {
            double sum = 0.0;
            for (int c = 0; c < network.outputCount(); c++) {
                sum += gradOutput[c] * network.weightsOutput()[r * network.outputCount() + c];
            }

            gradHidden[r] = sum * Network.ActivationFunctions.sigmoidPrim(hidden[r]);
        }

        for (int r = 0; r < network.hiddenCount(); r++) {
            for (int c = 0; c < network.outputCount(); c++) {
                network.weightsOutput()[r * network.outputCount() + c] -= lr * gradOutput[c] * hidden[r];
            }
        }

        for (int r = 0; r < network.inputCount(); r++) {
            for (int c = 0; c < network.hiddenCount(); c++) {
                network.weightsHidden()[r * network.hiddenCount() + c] -= lr * gradHidden[c] * input[r];
            }
        }

        for (int c = 0; c < network.outputCount(); c++) {
            network.biasesOutput()[c] -= lr * gradOutput[c];
        }

        for (int c = 0; c < network.hiddenCount(); c++) {
            network.biasesHidden()[c] -= lr * gradHidden[c];
        }
    }
}
