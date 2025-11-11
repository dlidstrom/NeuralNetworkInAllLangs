/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#if !defined(NEURAL_IO_H)
#define NEURAL_IO_H

#include "neural.h"
#include <string>

namespace Neural {

bool SaveNetwork(const Network &network, const std::string &filename);
bool LoadNetwork(Network &network, const std::string &filename);

} // namespace Neural

#endif
