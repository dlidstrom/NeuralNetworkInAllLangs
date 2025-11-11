/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#if !defined(AI_PLAYER_H)
#define AI_PLAYER_H

#include "board.h"
#include "neural.h"
#include <random>

namespace ConnectFour {

class AIPlayer {
public:
  AIPlayer(Neural::Network network, double explorationRate = 0.1);

  // Get the best move according to the neural network
  int SelectMove(const Board &board, Player player, bool explore = false);

  // Get move probabilities for all valid moves
  std::vector<double> GetMoveProbabilities(const Board &board, Player player);

  Neural::Network &GetNetwork() { return network; }
  const Neural::Network &GetNetwork() const { return network; }

private:
  Neural::Network network;
  double explorationRate;
  std::mt19937 rng;
};

} // namespace ConnectFour

#endif
