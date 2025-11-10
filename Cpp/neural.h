/*
Licensed under the MIT License.
Copyright 2023-2025 Daniel Lidstrom
*/

#if !defined(NEURAL_H)
#define NEURAL_H

#include <functional>
#include <vector>

namespace Neural {
typedef std::vector<double> Vector;
typedef std::vector<Vector> Matrix;
typedef std::function<double()> RandFcn;
struct Network {
  size_t inputCount;
  size_t hiddenCount;
  size_t outputCount;
  Vector weightsHidden;
  Vector biasesHidden;
  Vector weightsOutput;
  Vector biasesOutput;
  Vector Predict(const Vector &input) const;
  void Predict(const Vector &input, Vector &hidden, Vector &output) const;
};

struct Trainer {
  Network network;
  Vector hidden;
  Vector output;
  Vector gradHidden;
  Vector gradOutput;
  static Trainer Create(Neural::Network network);
  static Trainer Create(size_t inputCount, size_t hiddenCount,
                        size_t outputCount, RandFcn rand);
  void Train(const Vector &input, const Vector &output, double lr);
};
} // namespace Neural

#endif
