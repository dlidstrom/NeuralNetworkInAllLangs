/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#ifndef CONNECTFOUR_MINIMAX_H
#define CONNECTFOUR_MINIMAX_H

#include "board.h"
#include <limits>

namespace ConnectFour {

class MinimaxAI {
public:
  explicit MinimaxAI(int depth = 7) : maxDepth(depth) {}

  // Get the best move using minimax with alpha-beta pruning
  int SelectMove(const Board& board, Player player);

  // Get the number of nodes evaluated in last search
  int GetNodesEvaluated() const { return nodesEvaluated; }

private:
  int maxDepth;
  int nodesEvaluated;

  // Minimax with alpha-beta pruning
  double Minimax(Board& board, int depth, double alpha, double beta,
                 Player player, Player maximizingPlayer);

  // Evaluate board position heuristically
  double EvaluatePosition(const Board& board, Player player);

  // Count potential winning lines
  int CountThreats(const Board& board, Player player, int length);
};

} // namespace ConnectFour

#endif // CONNECTFOUR_MINIMAX_H
