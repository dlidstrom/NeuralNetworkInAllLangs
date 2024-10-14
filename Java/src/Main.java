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

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import java.util.function.Supplier;

public class Main {
    public static void main(String[] args) {
        CustomRandom random = new CustomRandom();
        Supplier<Double> rand = () -> random.get();
        var trainingData = Arrays.asList(
            new DataItem(new double[]{0, 0}, new double[]{Logical.xor(0, 0), Logical.xnor(0, 0), Logical.or(0, 0), Logical.and(0, 0), Logical.nor(0, 0), Logical.nand(0, 0)}),
            new DataItem(new double[]{0, 1}, new double[]{Logical.xor(0, 1), Logical.xnor(0, 1), Logical.or(0, 1), Logical.and(0, 1), Logical.nor(0, 1), Logical.nand(0, 1)}),
            new DataItem(new double[]{1, 0}, new double[]{Logical.xor(1, 0), Logical.xnor(1, 0), Logical.or(1, 0), Logical.and(1, 0), Logical.nor(1, 0), Logical.nand(1, 0)}),
            new DataItem(new double[]{1, 1}, new double[]{Logical.xor(1, 1), Logical.xnor(1, 1), Logical.or(1, 1), Logical.and(1, 1), Logical.nor(1, 1), Logical.nand(1, 1)})
        ).toArray(new DataItem[0]);

        Trainer trainer = Trainer.create(2, 2, 6, rand);
        double lr = 1.0;
        int ITERS = 4000;
        for (int e = 0; e < ITERS; e++) {
            var sample = trainingData[e % trainingData.length];
            trainer.train(sample.input(), sample.output(), lr);
        }

        Network network = trainer.network();
        System.out.println("Result after " + ITERS + " iterations");
        System.out.println("        XOR   XNOR    OR   AND   NOR   NAND");
        for (var sample : trainingData) {
            double[] pred = network.predict(sample.input());
            System.out.printf(
                Locale.ROOT,
                "%d,%d = %.3f  %.3f %.3f %.3f %.3f  %.3f%n",
                (int) sample.input()[0], (int) sample.input()[1],
                pred[0], pred[1], pred[2], pred[3], pred[4], pred[5]);
        }

        System.out.println("weights hidden:");
        for (int i = 0; i < network.inputCount(); i++) {
            for (int j = 0; j < network.hiddenCount(); j++) {
                System.out.printf(Locale.ROOT, " %9.6f", network.weightsHidden()[network.inputCount() * i + j]);
            }

            System.out.printf("\n");
        }

        System.out.printf("biases hidden:\n");
        for (int i = 0; i < network.hiddenCount(); i++) {
            System.out.printf(Locale.ROOT, " %9.6f", network.biasesHidden()[i]);
        }

        System.out.printf("\n");

        System.out.printf("weights output:\n");
        for (int i = 0; i < network.hiddenCount(); i++) {
            for (int j = 0; j < network.outputCount(); j++) {
                System.out.printf(Locale.ROOT, " %9.6f", network.weightsOutput()[i * network.outputCount() + j]);
            }

            System.out.printf("\n");
        }

        System.out.printf("biases output:\n");
        for (int i = 0; i < network.outputCount(); i++) {
            System.out.printf(Locale.ROOT, " %9.6f", network.biasesOutput()[i]);
        }

        System.out.printf("\n");
    }

    public static class DataItem {
        private final double[] input;
        private final double[] output;

        public DataItem(double[] input, double[] output) {
            this.input = input;
            this.output = output;
        }

        public double[] input() {
            return input;
        }

        public double[] output() {
            return output;
        }
    }

    public static class Logical {
        public static int xor(int a, int b) {
            return a ^ b;
        }

        public static int xnor(int a, int b) {
            return 1 - xor(a, b);
        }

        public static int or(int a, int b) {
            return a | b;
        }

        public static int and(int a, int b) {
            return a & b;
        }

        public static int nand(int a, int b) {
            return 1 - and(a, b);
        }

        public static int nor(int a, int b) {
            return 1 - or(a, b);
        }
    }
}
