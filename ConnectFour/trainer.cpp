/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#include "trainer.h"
#include "neural_evaluator.h"
#include <iostream>
#include <algorithm>

namespace ConnectFour {

Trainer::Trainer(Neural::Trainer neuralTrainer, int mctsSimulations,
                 double mctsExplorationConstant)
    : neuralTrainer(std::move(neuralTrainer)),
      mctsSimulations(mctsSimulations),
      mctsExplorationConstant(mctsExplorationConstant) {
  std::random_device rd;
  rng.seed(rd());
}

double Trainer::GetTemperature(int moveCount) const {
  // Use higher temperature early in game for exploration
  // Lower temperature later for more deterministic play
  if (moveCount < 10) {
    return 1.0;
  } else if (moveCount < 20) {
    return 0.5;
  } else {
    return 0.1;
  }
}

std::vector<TrainingExample> Trainer::PlaySelfPlayGame(double /* temperature */) {
  std::vector<TrainingExample> examples;
  Board board;
  Player currentPlayer = Player::PLAYER1;
  int moveCount = 0;

  while (!board.IsGameOver()) {
    // Run MCTS with neural evaluator
    auto evaluator = std::make_unique<NeuralEvaluator>(neuralTrainer.network);
    MCTS mcts(std::move(evaluator), mctsExplorationConstant);
    mcts.SearchSimulations(board, currentPlayer, mctsSimulations);

    // Get policy from MCTS visit counts
    std::vector<double> policy = mcts.GetMoveProbabilities();

    // Store training example
    bool wasMirrored;
    std::vector<double> state = board.GetNormalizedInput(currentPlayer, wasMirrored);

    // Adjust policy if board was mirrored
    std::vector<double> adjustedPolicy = policy;
    if (wasMirrored) {
      for (int col = 0; col < COLS; col++) {
        adjustedPolicy[col] = policy[Board::MirrorColumn(col)];
      }
    }

    TrainingExample example;
    example.state = state;
    example.policy = adjustedPolicy;
    example.value = 0.0; // Will be filled in at end of game

    examples.push_back(example);

    // Select move using softmax with temperature
    double temp = GetTemperature(moveCount);
    int move = mcts.SelectMoveSoftmax(temp, rng);

    // Make move
    board.MakeMove(move, currentPlayer);

    // Switch player
    currentPlayer = (currentPlayer == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
    moveCount++;
  }

  // Fill in outcome values
  Player winner = board.CheckWinner();
  for (size_t i = 0; i < examples.size(); i++) {
    // Determine which player made this move
    Player movePlayer = (i % 2 == 0) ? Player::PLAYER1 : Player::PLAYER2;

    if (winner == Player::NONE) {
      examples[i].value = 0.0; // Draw
    } else if (winner == movePlayer) {
      examples[i].value = 1.0; // This player won
    } else {
      examples[i].value = -1.0; // This player lost
    }
  }

  return examples;
}

void Trainer::TrainOnExamples(const std::vector<TrainingExample>& examples, double learningRate) {
  for (const auto& example : examples) {
    // Train the network to predict MCTS visit distribution
    // Scale the policy to emphasize moves based on game outcome
    std::vector<double> target = example.policy;

    // If we won, boost the probabilities; if we lost, they're still correct
    // (because MCTS explores even bad positions during learning)
    // The key insight: good moves in winning games should be reinforced more
    double outcomeBoost = 0.0;
    if (example.value > 0.0) {
      // We won - boost the selected moves
      outcomeBoost = 0.2 * example.value;
    } else if (example.value < 0.0) {
      // We lost - slightly suppress these moves
      outcomeBoost = 0.1 * example.value;
    }

    for (int i = 0; i < COLS; i++) {
      if (target[i] > 0.0) {
        target[i] = std::max(0.01, std::min(1.0, target[i] + outcomeBoost));
      }
    }

    neuralTrainer.Train(example.state, target, learningRate);
  }
}

int Trainer::PlayEvaluationGame(MinimaxAI& minimax, bool nnGoesFirst) {
  Board board;
  Player nnPlayer = nnGoesFirst ? Player::PLAYER1 : Player::PLAYER2;
  Player currentPlayer = Player::PLAYER1;

  while (!board.IsGameOver()) {
    int move;

    if (currentPlayer == nnPlayer) {
      // NN player using MCTS
      auto evaluator = std::make_unique<NeuralEvaluator>(neuralTrainer.network);
      MCTS mcts(std::move(evaluator), mctsExplorationConstant);
      mcts.SearchSimulations(board, currentPlayer, mctsSimulations / 2); // Use fewer sims for eval
      move = mcts.SelectBestMove();
    } else {
      // Minimax player
      move = minimax.SelectMove(board, currentPlayer);
    }

    if (move < 0 || !board.MakeMove(move, currentPlayer)) {
      // Invalid move - should not happen
      std::cerr << "Invalid move selected!\n";
      return 0;
    }

    currentPlayer = (currentPlayer == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
  }

  Player winner = board.CheckWinner();

  if (winner == Player::NONE) {
    return 0; // Draw
  } else if (winner == nnPlayer) {
    return 1; // NN wins
  } else {
    return -1; // Minimax wins
  }
}

EvaluationResult Trainer::EvaluateAgainstMinimax(int numGames, int minimaxDepth) {
  MinimaxAI minimax(minimaxDepth);
  EvaluationResult result{0, 0, 0};

  std::cout << "Evaluating against minimax (depth " << minimaxDepth << ")...\n";

  for (int i = 0; i < numGames; i++) {
    bool nnGoesFirst = (i % 2 == 0);
    int outcome = PlayEvaluationGame(minimax, nnGoesFirst);

    if (outcome == 1) {
      result.wins++;
    } else if (outcome == -1) {
      result.losses++;
    } else {
      result.draws++;
    }

    if ((i + 1) % 5 == 0) {
      std::cout << "  Game " << (i + 1) << "/" << numGames
                << " - W:" << result.wins << " L:" << result.losses
                << " D:" << result.draws << "\n";
    }
  }

  std::cout << "Evaluation complete: Win rate = " << (result.WinRate() * 100.0) << "%\n";

  return result;
}

void Trainer::Train(int numIterations, int gamesPerIteration, int evalEvery,
                    double learningRate, int evalGames) {
  std::cout << "Starting training...\n";
  std::cout << "Iterations: " << numIterations << "\n";
  std::cout << "Games per iteration: " << gamesPerIteration << "\n";
  std::cout << "MCTS simulations: " << mctsSimulations << "\n";
  std::cout << "Learning rate: " << learningRate << "\n\n";

  for (int iter = 0; iter < numIterations; iter++) {
    std::cout << "\n=== Iteration " << (iter + 1) << "/" << numIterations << " ===\n";

    // Collect training data from self-play
    std::vector<TrainingExample> allExamples;

    for (int game = 0; game < gamesPerIteration; game++) {
      std::vector<TrainingExample> examples = PlaySelfPlayGame();
      allExamples.insert(allExamples.end(), examples.begin(), examples.end());

      if ((game + 1) % 10 == 0) {
        std::cout << "  Self-play game " << (game + 1) << "/" << gamesPerIteration
                  << " complete (" << allExamples.size() << " examples)\n";
      }
    }

    std::cout << "Collected " << allExamples.size() << " training examples\n";

    // Shuffle examples
    std::shuffle(allExamples.begin(), allExamples.end(), rng);

    // Train on examples
    std::cout << "Training network...\n";
    TrainOnExamples(allExamples, learningRate);

    // Evaluate against minimax periodically
    if ((iter + 1) % evalEvery == 0) {
      std::cout << "\n--- Evaluation ---\n";
      EvaluationResult result = EvaluateAgainstMinimax(evalGames);
      std::cout << "Win: " << result.wins << " Loss: " << result.losses
                << " Draw: " << result.draws << "\n";
      std::cout << "Win rate: " << (result.WinRate() * 100.0) << "%\n";
    }
  }

  std::cout << "\n=== Training Complete ===\n";
}

} // namespace ConnectFour
