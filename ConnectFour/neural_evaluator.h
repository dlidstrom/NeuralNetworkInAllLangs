/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#ifndef CONNECTFOUR_NEURAL_EVALUATOR_H
#define CONNECTFOUR_NEURAL_EVALUATOR_H

#include "evaluator.h"
#include "../Cpp/neural.h"

namespace ConnectFour {

// Neural network evaluator
class NeuralEvaluator : public Evaluator {
public:
  explicit NeuralEvaluator(Neural::Network network) 
      : network(std::move(network)) {}

  std::pair<std::vector<double>, double> Evaluate(
      const Board& board, Player player) override;

private:
  Neural::Network network;

  static Player GetOpponent(Player p) {
    return (p == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
  }
};

} // namespace ConnectFour

#endif // CONNECTFOUR_NEURAL_EVALUATOR_H
