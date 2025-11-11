/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#ifndef CONNECTFOUR_TRAINER_H
#define CONNECTFOUR_TRAINER_H

#include "board.h"
#include "mcts.h"
#include "minimax.h"
#include "../Cpp/neural.h"
#include <vector>
#include <random>

namespace ConnectFour {

// Training example from self-play
struct TrainingExample {
  std::vector<double> state;
  std::vector<double> policy; // Target policy from MCTS
  double value; // Outcome: 1 for win, -1 for loss, 0 for draw
};

// Result of an evaluation match
struct EvaluationResult {
  int wins;
  int losses;
  int draws;

  double WinRate() const {
    int total = wins + losses + draws;
    return total > 0 ? static_cast<double>(wins) / total : 0.0;
  }
};

class Trainer {
public:
  Trainer(Neural::Trainer neuralTrainer, int mctsSimulations = 800,
          double mctsExplorationConstant = 1.414);

  // Play one self-play game using MCTS
  std::vector<TrainingExample> PlaySelfPlayGame(double temperature = 1.0);

  // Train network on collected examples
  void TrainOnExamples(const std::vector<TrainingExample>& examples, double learningRate);

  // Evaluate current network against minimax
  EvaluationResult EvaluateAgainstMinimax(int numGames, int minimaxDepth = 6);

  // Main training loop
  void Train(int numIterations, int gamesPerIteration, int evalEvery,
             double learningRate, int evalGames = 20);

  Neural::Network& GetNetwork() { return neuralTrainer.network; }
  const Neural::Network& GetNetwork() const { return neuralTrainer.network; }

private:
  Neural::Trainer neuralTrainer;
  int mctsSimulations;
  double mctsExplorationConstant;
  std::mt19937 rng;

  // Temperature schedule for move selection
  double GetTemperature(int moveCount) const;

  // Play one evaluation game against minimax
  // Returns 1 if NN wins, -1 if minimax wins, 0 for draw
  int PlayEvaluationGame(MinimaxAI& minimax, bool nnGoesFirst);
};

} // namespace ConnectFour

#endif // CONNECTFOUR_TRAINER_H
