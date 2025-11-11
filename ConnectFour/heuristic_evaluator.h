/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#ifndef CONNECTFOUR_HEURISTIC_EVALUATOR_H
#define CONNECTFOUR_HEURISTIC_EVALUATOR_H

#include "evaluator.h"

namespace ConnectFour {

// Heuristic evaluator based on minimax-style evaluation
class HeuristicEvaluator : public Evaluator {
public:
  HeuristicEvaluator() = default;

  std::pair<std::vector<double>, double> Evaluate(
      const Board& board, Player player) override;

private:
  // Count threats (N pieces in a row with rest empty)
  int CountThreats(const Board& board, Player player, int length);

  // Heuristic position evaluation
  double EvaluatePosition(const Board& board, Player player);

  // Helper to get opponent
  static Player GetOpponent(Player p) {
    return (p == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
  }
};

} // namespace ConnectFour

#endif // CONNECTFOUR_HEURISTIC_EVALUATOR_H
