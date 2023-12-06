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

#if !defined(NEURAL_H)
#define NEURAL_H

#include <stdint.h>

typedef struct Network {
    double* weights_hidden;
    double* biases_hidden;
    double* weights_output;
    double* biases_output;
    double* hidden;
    double* output;
    uint32_t n_inputs;
    uint32_t n_hidden;
    uint32_t n_outputs;
} Network;

typedef double (*RandFcn)();
Network network_create(uint32_t n_inputs, uint32_t n_hidden, uint32_t n_outputs, RandFcn rand);
void network_free(Network* network);
void network_predict(Network* network, double* input);

typedef struct Trainer {
    double* grad_hidden;
    double* grad_output;
} Trainer;

Trainer trainer_create(Network* network);
void trainer_train(Trainer* trainer, Network* network, double* input, double* output, double lr);
void trainer_free(Trainer* trainer);

#endif
