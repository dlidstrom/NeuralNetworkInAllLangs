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
#include <stdlib.h>
#include <math.h>

static double sigmoid(double f) { return 1.0 / (1.0 + exp(-f)); }
static double sigmoid_prim(double f) { return f * (1.0 - f); }

Network network_create(uint32_t n_inputs, uint32_t n_hidden, uint32_t n_outputs, RandFcn rand) {
    Network network = {0};
    network.n_inputs = n_inputs;
    network.n_hidden = n_hidden;
    network.n_outputs = n_outputs;

    network.weights_hidden = calloc(
        n_inputs * n_outputs,
        sizeof(*network.weights_hidden));
    network.biases_hidden = calloc(
        n_hidden,
        sizeof(*network.biases_hidden));
    network.weights_output = calloc(
        n_hidden * n_outputs,
        sizeof(*network.weights_output));
    network.biases_output = calloc(
        n_outputs,
        sizeof(*network.biases_output));
    network.hidden = calloc(
        n_hidden,
        sizeof(*network.hidden));
    network.output = calloc(
        n_outputs,
        sizeof(*network.output));

    // initialize everything but the biases
    for (size_t i = 0; i < n_inputs * n_hidden; i++) {
        network.weights_hidden[i] = rand() - 0.5;
    }

    for (size_t i = 0; i < n_hidden * n_outputs; i++) {
        network.weights_output[i] = rand() - 0.5;
    }

    return network;
}

void network_free(Network* network) {
    free(network->weights_hidden);
    free(network->biases_hidden);
    free(network->weights_output);
    free(network->biases_output);
    free(network->hidden);
    free(network->output);
}

void network_predict(Network* network, double* input) {
    for (uint32_t c = 0; c < network->n_hidden; c++) {
        double sum = 0;
        for (uint32_t r = 0; r < network->n_inputs; r++) {
            sum += input[r] * network->weights_hidden[r * network->n_hidden + c];
        }

        network->hidden[c] = sigmoid(sum + network->biases_hidden[c]);
    }

    for (uint32_t c = 0; c < network->n_outputs; c++) {
        double sum = 0;
        for (uint32_t r = 0; r < network->n_hidden; r++) {
            sum += network->hidden[r] * network->weights_output[r * network->n_outputs + c];
        }

        network->output[c] = sigmoid(sum + network->biases_output[c]);
    }
}

/* trainer */

Trainer trainer_create(Network* network) {
    Trainer trainer = {0};
    trainer.grad_hidden = calloc(network->n_hidden, sizeof(*trainer.grad_hidden));
    trainer.grad_output = calloc(network->n_outputs, sizeof(*trainer.grad_output));
    return trainer;
}

void trainer_train(Trainer* trainer, Network* network, double* input, double* y, double lr) {
    network_predict(network, input);
    for (uint32_t c = 0; c < network->n_outputs; c++) {
        trainer->grad_output[c] = (network->output[c] - y[c]) * sigmoid_prim(network->output[c]);
    }

    for (uint32_t r = 0; r < network->n_hidden; r++) {
        double sum = 0.0;
        for (uint32_t c = 0; c < network->n_outputs; c++) {
            sum += trainer->grad_output[c] * network->weights_output[r * network->n_outputs + c];
        }

        trainer->grad_hidden[r] = sum * sigmoid_prim(network->hidden[r]);
    }

    for (size_t r = 0; r < network->n_hidden; r++) {
        for (size_t c = 0; c < network->n_outputs; c++) {
            network->weights_output[r * network->n_outputs + c] -= lr * trainer->grad_output[c] * network->hidden[r];
        }
    }

    for (size_t r = 0; r < network->n_inputs; r++) {
        for (size_t c = 0; c < network->n_hidden; c++) {
            network->weights_hidden[r * network->n_hidden + c] -= lr * trainer->grad_hidden[c] * input[r];
        }
    }

    for (size_t c = 0; c < network->n_outputs; c++) {
        network->biases_output[c] -= lr * trainer->grad_output[c];
    }

    for (size_t c = 0; c < network->n_hidden; c++) {
        network->biases_hidden[c] -= lr * trainer->grad_hidden[c];
    }
}

void trainer_free(Trainer* trainer) {
    free(trainer->grad_hidden);
    free(trainer->grad_output);
}
