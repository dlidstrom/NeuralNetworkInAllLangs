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
#include <stdio.h>
#include <stdlib.h>

uint32_t P = 2147483647;
uint32_t A = 16807;
uint32_t current = 1;

double Rand(void) {
  current = current * A % P;
  double result = (double)current / P;
  return result;
}

void print_network(const Network *network);

static uint32_t xor (uint32_t i, uint32_t j) { return i ^ j; } static uint32_t
    xnor(uint32_t i, uint32_t j) {
  return 1 - xor(i, j);
}
static uint32_t or (uint32_t i, uint32_t j) { return i | j; }
static uint32_t and (uint32_t i, uint32_t j) { return i & j; }
static uint32_t nor(uint32_t i, uint32_t j) { return 1 - or (i, j); }
static uint32_t nand(uint32_t i, uint32_t j) { return 1 - and(i, j); }

const int ITERS = 4000;

int main(void) {
  Network network = {0};
  network_init(&network, 2, 2, 6, Rand);
  Trainer trainer = {0};
  trainer_init(&trainer, &network);
  double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double outputs[4][6] = {
      {xor(0, 0), xnor(0, 0), or (0, 0), and(0, 0), nor(0, 0), nand(0, 0)},
      {xor(0, 1), xnor(0, 1), or (0, 1), and(0, 1), nor(0, 1), nand(0, 1)},
      {xor(1, 0), xnor(1, 0), or (1, 0), and(1, 0), nor(1, 0), nand(1, 0)},
      {xor(1, 1), xnor(1, 1), or (1, 1), and(1, 1), nor(1, 1), nand(1, 1)}};

  for (size_t i = 0; i < ITERS; i++) {
    double *input = inputs[i % 4];
    double *output = outputs[i % 4];

    trainer_train(&trainer, &network, input, output, 1.0);
  }

  printf(
      "Result after %d iterations\n        XOR  XNOR    OR   AND   NOR  NAND\n",
      ITERS);
  for (size_t i = 0; i < 4; i++) {
    double *input = inputs[i % 4];
    network_predict(&network, input);
    printf("%.0f,%.0f = %.3f %.3f %.3f %.3f %.3f %.3f\n", input[0], input[1],
           network.output[0], network.output[1], network.output[2],
           network.output[3], network.output[4], network.output[5]);
  }

  print_network(&network);
  trainer_free(&trainer);
  network_free(&network);
  return 0;
}

void print_network(const Network *network) {
  printf("weights hidden:\n");
  for (size_t i = 0; i < network->n_inputs; i++) {
    for (size_t j = 0; j < network->n_hidden; j++) {
      printf(" %9.6f", network->weights_hidden[network->n_inputs * i + j]);
    }

    printf("\n");
  }

  printf("biases hidden:\n");
  for (size_t i = 0; i < network->n_hidden; i++) {
    printf(" %9.6f", network->biases_hidden[i]);
  }

  printf("\n");

  printf("weights output:\n");
  for (size_t i = 0; i < network->n_hidden; i++) {
    for (size_t j = 0; j < network->n_outputs; j++) {
      printf(" %9.6f", network->weights_output[i * network->n_outputs + j]);
    }

    printf("\n");
  }

  printf("biases output:\n");
  for (size_t i = 0; i < network->n_outputs; i++) {
    printf(" %9.6f", network->biases_output[i]);
  }

  printf("\n");
}
