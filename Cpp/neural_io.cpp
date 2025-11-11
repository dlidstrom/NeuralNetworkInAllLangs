/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#include "neural_io.h"
#include <fstream>
#include <iostream>

namespace Neural {

bool SaveNetwork(const Network &network, const std::string &filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << filename << "\n";
    return false;
  }

  // Write network dimensions
  file.write(reinterpret_cast<const char *>(&network.inputCount),
             sizeof(size_t));
  file.write(reinterpret_cast<const char *>(&network.hiddenCount),
             sizeof(size_t));
  file.write(reinterpret_cast<const char *>(&network.outputCount),
             sizeof(size_t));

  // Write weightsHidden
  size_t size = network.weightsHidden.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
  file.write(reinterpret_cast<const char *>(network.weightsHidden.data()),
             size * sizeof(double));

  // Write biasesHidden
  size = network.biasesHidden.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
  file.write(reinterpret_cast<const char *>(network.biasesHidden.data()),
             size * sizeof(double));

  // Write weightsOutput
  size = network.weightsOutput.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
  file.write(reinterpret_cast<const char *>(network.weightsOutput.data()),
             size * sizeof(double));

  // Write biasesOutput
  size = network.biasesOutput.size();
  file.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
  file.write(reinterpret_cast<const char *>(network.biasesOutput.data()),
             size * sizeof(double));

  file.close();
  return true;
}

bool LoadNetwork(Network &network, const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file for reading: " << filename << "\n";
    return false;
  }

  // Read network dimensions
  file.read(reinterpret_cast<char *>(&network.inputCount), sizeof(size_t));
  file.read(reinterpret_cast<char *>(&network.hiddenCount), sizeof(size_t));
  file.read(reinterpret_cast<char *>(&network.outputCount), sizeof(size_t));

  // Read weightsHidden
  size_t size;
  file.read(reinterpret_cast<char *>(&size), sizeof(size_t));
  network.weightsHidden.resize(size);
  file.read(reinterpret_cast<char *>(network.weightsHidden.data()),
            size * sizeof(double));

  // Read biasesHidden
  file.read(reinterpret_cast<char *>(&size), sizeof(size_t));
  network.biasesHidden.resize(size);
  file.read(reinterpret_cast<char *>(network.biasesHidden.data()),
            size * sizeof(double));

  // Read weightsOutput
  file.read(reinterpret_cast<char *>(&size), sizeof(size_t));
  network.weightsOutput.resize(size);
  file.read(reinterpret_cast<char *>(network.weightsOutput.data()),
            size * sizeof(double));

  // Read biasesOutput
  file.read(reinterpret_cast<char *>(&size), sizeof(size_t));
  network.biasesOutput.resize(size);
  file.read(reinterpret_cast<char *>(network.biasesOutput.data()),
            size * sizeof(double));

  file.close();
  return true;
}

} // namespace Neural
