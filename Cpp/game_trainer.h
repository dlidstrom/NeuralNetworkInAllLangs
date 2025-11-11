/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#if !defined(TRAINER_GAME_H)
#define TRAINER_GAME_H

#include "board.h"
#include "neural.h"
#include <random>
#include <vector>

namespace ConnectFour {

struct GameRecord {
  std::vector<std::vector<double>> states;
  std::vector<int> moves;
  std::vector<Player> players;
  Player winner;
};

class GameTrainer {
public:
  GameTrainer(Neural::Trainer trainer, double explorationRate = 0.2);

  // Play one self-play game and collect experience
  GameRecord PlaySelfPlayGame();

  // Train the network on a game record
  void TrainOnGame(const GameRecord &record, double learningRate);

  // Run multiple training iterations
  void Train(int numGames, double learningRate, int printEvery = 100);

  Neural::Network &GetNetwork() { return trainer.network; }
  const Neural::Network &GetNetwork() const { return trainer.network; }

private:
  Neural::Trainer trainer;
  double explorationRate;
  std::mt19937 rng;

  int SelectMove(const Board &board, Player player, bool explore);
  double GetReward(Player player, Player winner);
};

} // namespace ConnectFour

#endif
