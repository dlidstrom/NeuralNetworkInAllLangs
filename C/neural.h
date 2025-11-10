/*
Licensed under the MIT License.
Copyright 2023-2025 Daniel Lidstrom
*/

#if !defined(NEURAL_H)
#define NEURAL_H

#include <stdint.h>

typedef struct Network {
  double *weights_hidden;
  double *biases_hidden;
  double *weights_output;
  double *biases_output;
  double *hidden;
  double *output;
  uint32_t n_inputs;
  uint32_t n_hidden;
  uint32_t n_outputs;
} Network;

typedef double (*RandFcn)(void);
Network *network_init(Network *network, uint32_t n_inputs, uint32_t n_hidden,
                      uint32_t n_outputs, RandFcn rand);
void network_free(Network *network);
void network_predict(Network *network, const double *input);

typedef struct Trainer {
  double *grad_hidden;
  double *grad_output;
} Trainer;

Trainer *trainer_init(Trainer *trainer, Network *network);
void trainer_train(Trainer *trainer, Network *network, const double *input,
                   const double *output, double lr);
void trainer_free(Trainer *trainer);

#endif
