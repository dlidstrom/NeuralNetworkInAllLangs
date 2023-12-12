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

#include "neural.h"
#include <cmath>

using namespace Neural;

namespace {
    double sigmoid(double f) { return 1.0 / (1.0 + exp(-f)); }
    double sigmoid_prim(double f) { return f * (1.0 - f); }
}

/* network */

Vector Network::Predict(const Vector& input) const {
    Vector y_hidden = Vector(hiddenCount);
    Vector y_output = Vector(outputCount);
    return Predict(input, y_hidden, y_output);
}

Vector Network::Predict(const Vector& input, Vector& hidden, Vector& output) const {
    for (std::size_t c = 0; c < hiddenCount; c++) {
        double sum = 0.0;
        for (size_t r = 0; r < input.size(); r++) {
            sum += input[r] * weightsHidden[r * hiddenCount + c];
        }

        hidden[c] = sigmoid(sum + biasesHidden[c]);
    }

    for (size_t c = 0; c < outputCount; c++) {
        double sum = 0.0;
        for (size_t r = 0; r < hiddenCount; r++) {
            sum += hidden[r] * weightsOutput[r * outputCount + c];
        }

        output[c] = sigmoid(sum + biasesOutput[c]);
    }

    return output;
}

/* trainer */

Trainer Trainer::Create(Neural::Network&& network, size_t hiddenCount, size_t outputCount) {
    Vector hidden = Vector(hiddenCount);
    Vector output = Vector(hiddenCount);
    Vector gradHidden = Vector(hiddenCount);
    Vector gradOutput = Vector(outputCount);
    return Trainer {
        network,
        hidden,
        output,
        gradHidden,
        gradOutput
    };
}

Trainer Trainer::Create(size_t inputCount, size_t hiddenCount, size_t outputCount, std::function<double()> rand) {
    Vector hidden = Vector(hiddenCount);
    Vector output = Vector(outputCount);
    Vector gradHidden = Vector(hiddenCount);
    Vector gradOutput = Vector(outputCount);
    Vector weightsHidden = Vector();
    for (size_t i = 0; i < inputCount * hiddenCount; i++) {
        weightsHidden.push_back(rand() - 0.5);
    }

    Vector biasesHidden = Vector(hiddenCount);
    Vector weightsOutput = Vector();
    for (size_t i = 0; i < hiddenCount * outputCount; i++) {
        weightsOutput.push_back(rand() - 0.5);
    }

    Vector biasesOutput = Vector(outputCount);
    Neural::Network network = {
        inputCount,
        hiddenCount,
        outputCount,
        std::move(weightsHidden),
        std::move(biasesHidden),
        std::move(weightsOutput),
        std::move(biasesOutput)
    };
    return Trainer {
        network,
        hidden,
        output,
        gradHidden,
        gradOutput
    };
}

void Trainer::Train(const Vector& input, const Vector& y, double lr) {
    network.Predict(input, hidden, output);
    for (size_t c = 0; c < output.size(); c++) {
        gradOutput[c] = (output[c] - y[c]) * sigmoid_prim(output[c]);
    }

    for (size_t r = 0; r < network.hiddenCount; r++) {
        double sum = 0.0;
        for (size_t c = 0; c < network.outputCount; c++) {
            sum += gradOutput[c] * network.weightsOutput[r * network.outputCount + c];
        }

        gradHidden[r] = sum * sigmoid_prim(hidden[r]);
    }

    for (size_t r = 0; r < network.hiddenCount; r++) {
        for (size_t c = 0; c < network.outputCount; c++) {
            network.weightsOutput[r * network.outputCount + c] -= lr * gradOutput[c] * hidden[r];
        }
    }

    for (size_t r = 0; r < network.hiddenCount; r++) {
        for (size_t c = 0; c < network.outputCount; c++) {
            network.weightsHidden[r * network.hiddenCount + c] -= lr * gradHidden[c] * input[r];
        }
    }

    for (size_t c = 0; c < network.outputCount; c++) {
        network.biasesOutput[c] -= lr * gradOutput[c];
    }

    for (size_t c = 0; c < network.hiddenCount; c++) {
        network.biasesHidden[c] -= lr * gradHidden[c];
    }
}
