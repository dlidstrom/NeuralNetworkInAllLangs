/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#include "ai_player.h"
#include <limits>

namespace ConnectFour {

AIPlayer::AIPlayer(Neural::Network network, double explorationRate)
    : network(std::move(network)), explorationRate(explorationRate) {
  std::random_device rd;
  rng.seed(rd());
}

std::vector<double> AIPlayer::GetMoveProbabilities(const Board &board,
                                                   Player player) {
  std::vector<int> validMoves = board.GetValidMoves();
  std::vector<double> probabilities(COLS, -1.0);

  if (validMoves.empty()) {
    return probabilities;
  }

  // Get neural network input
  std::vector<double> input = board.ToNeuralInput(player);

  // Get network output
  std::vector<double> output = network.Predict(input);

  // Map output to valid moves only
  for (int col : validMoves) {
    probabilities[col] = output[col];
  }

  return probabilities;
}

int AIPlayer::SelectMove(const Board &board, Player player, bool explore) {
  std::vector<int> validMoves = board.GetValidMoves();

  if (validMoves.empty()) {
    return -1;
  }

  // Exploration: random move
  if (explore &&
      std::uniform_real_distribution<>(0.0, 1.0)(rng) < explorationRate) {
    std::uniform_int_distribution<> dist(0, validMoves.size() - 1);
    return validMoves[dist(rng)];
  }

  // Exploitation: best move according to network
  std::vector<double> probabilities = GetMoveProbabilities(board, player);

  int bestMove = -1;
  double bestValue = -std::numeric_limits<double>::infinity();

  for (int col : validMoves) {
    if (probabilities[col] > bestValue) {
      bestValue = probabilities[col];
      bestMove = col;
    }
  }

  return bestMove;
}

} // namespace ConnectFour
