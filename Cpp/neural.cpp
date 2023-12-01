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
    double sigmoid_prim(double f) { return 1.0 * (1.0 - f); }
}

/* network */
Network::Network(
    int hiddenCount,
    int outputCount,
    Matrix&& weightsHidden,
    Vector&& biasesHidden,
    Matrix&& weightsOutput,
    Vector&& biasesOutput)
    : hiddenCount(hiddenCount)
    , outputCount(outputCount)
    , weightsHidden(weightsHidden)
    , biasesHidden(biasesHidden)
    , weightsOutput(weightsOutput)
    , biasesOutput(biasesOutput) {
}

int Network::HiddenCount() const {
    return hiddenCount;
}

int Network::OutputCount() const {
    return outputCount;
}

Matrix Network::WeightsHidden() const {
    return weightsHidden;
}

Vector Network::BiasesHidden() const {
    return biasesHidden;
}

Matrix Network::WeightsOutput() const {
    return weightsOutput;
}

Vector Network::BiasesOutput() const {
    return biasesOutput;
}

Vector Network::Predict(const Vector& input) const {
    Vector y_hidden = Vector(hiddenCount);
    Vector y_output = Vector(outputCount);
    return Predict(input, y_hidden, y_output);
}

Vector Network::Predict(const Vector& input, Vector& hidden, Vector& output) const {
    for (int c = 0; c < weightsHidden[0].size(); c++)
    {
        double sum = 0.0;
        for (int r = 0; r < weightsHidden.size(); r++)
        {
            sum += input[r] * weightsHidden[r][c];
            hidden[c] = sigmoid(sum + biasesHidden[c]);
        }
    }

    for (int c = 0; c < weightsOutput[0].size(); c++)
    {
        double sum = 0.0;
        for (int r = 0; r < weightsOutput.size(); r++)
        {
            sum += hidden[r] * weightsOutput[r][c];
        }

        output[c] = sigmoid(sum + biasesOutput[c]);
    }

    return output;
}

/* trainer */

Trainer::Trainer(
    Neural::Network&& network,
    Vector&& hidden,
    Vector&& output,
    Vector&& gradHidden,
    Vector&& gradOutput)
    : network(network)
    , hidden(hidden)
    , output(output)
    , gradHidden(gradHidden)
    , gradOutput(gradOutput) {
}

const Network& Trainer::Network() const {
    return network;
}

const Vector& Trainer::Hidden() const {
    return hidden;
}

const Vector& Trainer::Output() const {
    return output;
}

const Vector& Trainer::GradHidden() const {
    return gradHidden;
}

const Vector& Trainer::GradOutput() const {
    return gradOutput;
}

Trainer Trainer::Create(Neural::Network&& network, int hiddenCount, int outputCount)
{
    Vector hidden = Vector(hiddenCount);
    Vector output = Vector(hiddenCount);
    Vector gradHidden = Vector(hiddenCount);
    Vector gradOutput = Vector(outputCount);
    return Trainer(
        std::move(network),
        std::move(hidden),
        std::move(output),
        std::move(gradHidden),
        std::move(gradOutput));
}

Trainer Trainer::Create(int inputCount, int hiddenCount, int outputCount, std::function<double()> rand)
{
    Vector hidden = Vector(hiddenCount);
    Vector output = Vector(outputCount);
    Vector gradHidden = Vector(hiddenCount);
    Vector gradOutput = Vector(outputCount);
    Matrix weightsHidden = Matrix();
    for (int i = 0; i < inputCount; i++) {
        Vector v;
        std::generate_n(
            std::back_inserter(v),
            hiddenCount,
            [rand] { return rand() - 0.5; });
        weightsHidden.push_back(v);
    }

    Vector biasesHidden = Vector(hiddenCount);
    Matrix weightsOutput = Matrix();
    for (int i = 0; i < hiddenCount; i++) {
        Vector v;
        std::generate_n(
            std::back_inserter(v),
            outputCount,
            [rand] { return rand() - 0.5; });
        weightsOutput.push_back(v);
    }

    Vector biasesOutput = Vector(outputCount);
    Neural::Network network = Neural::Network(
        hiddenCount,
        outputCount,
        std::move(weightsHidden),
        std::move(biasesHidden),
        std::move(weightsOutput),
        std::move(biasesOutput));
    return Trainer(
        std::move(network),
        std::move(hidden),
        std::move(output),
        std::move(gradHidden),
        std::move(gradOutput));
}

void Trainer::Train(const Vector& input, const Vector& y, double lr) {
    network.Predict(input, hidden, output);
    for (int c = 0; c < output.size(); c++)
    {
        gradOutput[c] = (output[c] - y[c]) * sigmoid_prim(output[c]);
    }

    for (int r = 0; r < network.WeightsOutput().size(); r++)
    {
        double sum = 0.0;
        for (int c = 0; c < network.WeightsOutput()[0].size(); c++)
        {
            sum += gradOutput[c] * network.WeightsOutput()[r][c];
        }

        gradHidden[r] = sum * sigmoid_prim(hidden[r]);
    }

    for (int r = 0; r < network.WeightsOutput().size(); r++)
    {
        for (int c = 0; c < network.WeightsOutput()[0].size(); c++)
        {
            network.WeightsOutput()[r][c] -= lr * gradOutput[c] * hidden[r];
        }
    }

    for (int r = 0; r < network.WeightsHidden().size(); r++)
    {
        for (int c = 0; c < network.WeightsHidden()[0].size(); c++)
        {
            network.WeightsHidden()[r][c] -= lr * gradHidden[c] * input[r];
        }
    }

    for (int c = 0; c < gradOutput.size(); c++)
    {
        network.BiasesOutput()[c] -= lr * gradOutput[c];
    }

    for (int c = 0; c < gradHidden.size(); c++)
    {
        network.BiasesHidden()[c] -= lr * gradHidden[c];
    }
}
