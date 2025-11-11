/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#include "game_trainer.h"
#include <iostream>
#include <limits>

namespace ConnectFour {

GameTrainer::GameTrainer(Neural::Trainer trainer, double explorationRate)
    : trainer(std::move(trainer)), explorationRate(explorationRate) {
  std::random_device rd;
  rng.seed(rd());
}

int GameTrainer::SelectMove(const Board &board, Player player, bool explore) {
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
  std::vector<double> input = board.ToNeuralInput(player);
  std::vector<double> output = trainer.network.Predict(input);

  int bestMove = -1;
  double bestValue = -std::numeric_limits<double>::infinity();

  for (int col : validMoves) {
    if (output[col] > bestValue) {
      bestValue = output[col];
      bestMove = col;
    }
  }

  return bestMove;
}

GameRecord GameTrainer::PlaySelfPlayGame() {
  GameRecord record;
  Board board;

  Player currentPlayer = Player::PLAYER1;

  while (!board.IsGameOver()) {
    // Record state before move
    record.states.push_back(board.ToNeuralInput(currentPlayer));
    record.players.push_back(currentPlayer);

    // Select and make move
    int move = SelectMove(board, currentPlayer, true);
    record.moves.push_back(move);

    board.MakeMove(move, currentPlayer);

    // Switch player
    currentPlayer =
        (currentPlayer == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
  }

  record.winner = board.CheckWinner();
  return record;
}

double GameTrainer::GetReward(Player player, Player winner) {
  if (winner == Player::NONE) {
    return 0.0; // Draw
  } else if (winner == player) {
    return 1.0; // Win
  } else {
    return -1.0; // Loss
  }
}

void GameTrainer::TrainOnGame(const GameRecord &record, double learningRate) {
  // Train on each position in the game
  for (size_t i = 0; i < record.states.size(); i++) {
    const auto &state = record.states[i];
    int move = record.moves[i];
    Player player = record.players[i];

    // Calculate reward from this player's perspective
    double reward = GetReward(player, record.winner);

    // Discount reward based on how far from the end
    double discountFactor = 0.95;
    double discountedReward =
        reward * std::pow(discountFactor, record.states.size() - i - 1);

    // Create target output
    // Start with current network prediction
    std::vector<double> target = trainer.network.Predict(state);

    // Adjust the value for the move that was made
    // We want to push the network's output towards 1.0 for winning moves
    // and towards 0.0 for losing moves
    target[move] = 0.5 + discountedReward * 0.5;

    // Train the network
    trainer.Train(state, target, learningRate);
  }
}

void GameTrainer::Train(int numGames, double learningRate, int printEvery) {
  int player1Wins = 0;
  int player2Wins = 0;
  int draws = 0;

  std::cout << "Starting training for " << numGames << " games...\n";

  for (int game = 0; game < numGames; game++) {
    GameRecord record = PlaySelfPlayGame();

    // Update statistics
    if (record.winner == Player::PLAYER1) {
      player1Wins++;
    } else if (record.winner == Player::PLAYER2) {
      player2Wins++;
    } else {
      draws++;
    }

    // Train on this game
    TrainOnGame(record, learningRate);

    // Print progress
    if ((game + 1) % printEvery == 0) {
      std::cout << "Game " << (game + 1) << "/" << numGames
                << " - P1: " << player1Wins << ", P2: " << player2Wins
                << ", Draw: " << draws << "\n";
    }
  }

  std::cout << "\nTraining complete!\n";
  std::cout << "Final stats - P1: " << player1Wins << ", P2: " << player2Wins
            << ", Draw: " << draws << "\n";
}

} // namespace ConnectFour
