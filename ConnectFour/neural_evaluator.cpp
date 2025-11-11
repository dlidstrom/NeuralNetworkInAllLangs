/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#include "neural_evaluator.h"
#include <algorithm>

namespace ConnectFour {

std::pair<std::vector<double>, double> NeuralEvaluator::Evaluate(
    const Board& board, Player player) {
  
  // Get normalized input with mirroring
  bool wasMirrored;
  std::vector<double> input = board.GetNormalizedInput(player, wasMirrored);

  // Get network prediction
  std::vector<double> output = network.Predict(input);

  // Output has COLS values for policy
  std::vector<int> validMoves = board.GetValidMoves();
  std::vector<double> priors(COLS, 0.0);

  // Check for immediate wins or losses and boost priors
  for (int col : validMoves) {
    Board testBoard = board;
    testBoard.MakeMove(col, player);

    // If this move wins immediately, heavily boost its prior
    if (testBoard.CheckWinner() == player) {
      priors[col] = 100.0;
    }
    // Check if opponent can win on their next move if we don't play here
    else {
      Board opponentTest = board;
      opponentTest.MakeMove(col, GetOpponent(player));
      if (opponentTest.CheckWinner() == GetOpponent(player)) {
        priors[col] = 50.0;
      }
    }
  }

  // Fill in network priors for non-critical moves
  double totalProb = 0.0;
  bool hasCriticalMove = false;

  for (int col : validMoves) {
    if (priors[col] > 0.0) {
      hasCriticalMove = true;
      totalProb += priors[col];
    } else {
      int actualCol = wasMirrored ? Board::MirrorColumn(col) : col;
      if (actualCol >= 0 && actualCol < static_cast<int>(output.size())) {
        priors[col] = std::max(0.01, output[actualCol]);
        totalProb += priors[col];
      }
    }
  }

  // Normalize priors
  if (totalProb > 0.0) {
    for (int col : validMoves) {
      priors[col] /= totalProb;
    }
  } else {
    // Uniform distribution if network outputs are bad
    double uniform = 1.0 / validMoves.size();
    for (int col : validMoves) {
      priors[col] = uniform;
    }
  }

  // For untrained networks, value estimation is unreliable
  // Return neutral value
  double value = 0.0;

  return {priors, value};
}

} // namespace ConnectFour
