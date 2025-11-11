/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#ifndef CONNECTFOUR_EVALUATOR_H
#define CONNECTFOUR_EVALUATOR_H

#include "board.h"
#include <vector>
#include <utility>

namespace ConnectFour {

// Abstract evaluator interface for MCTS
class Evaluator {
public:
  virtual ~Evaluator() = default;

  // Evaluate a position and return (policy priors, value estimate)
  // policy: vector of size COLS with move probabilities
  // value: position evaluation from player's perspective [-1, 1]
  virtual std::pair<std::vector<double>, double> Evaluate(
      const Board& board, Player player) = 0;
};

} // namespace ConnectFour

#endif // CONNECTFOUR_EVALUATOR_H
