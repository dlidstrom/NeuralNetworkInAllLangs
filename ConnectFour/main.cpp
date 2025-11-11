/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#include "board.h"
#include "mcts.h"
#include "minimax.h"
#include "trainer.h"
#include "../Cpp/neural.h"
#include "../Cpp/neural_io.h"
#include <iostream>
#include <random>
#include <string>

using namespace ConnectFour;
using namespace Neural;

const std::string WEIGHTS_FILE = "connectfour_mcts_weights.bin";
const int INPUT_SIZE = BOARD_SIZE * 3; // 3 values per cell
const int HIDDEN_SIZE = 256; // Larger network for MCTS
const int OUTPUT_SIZE = COLS;

void DisplayMenu() {
  std::cout << "\n=== Connect Four MCTS AI ===\n";
  std::cout << "1. Train new network with self-play\n";
  std::cout << "2. Continue training existing network\n";
  std::cout << "3. Play against AI (with MCTS search)\n";
  std::cout << "4. Watch AI vs Minimax\n";
  std::cout << "5. Evaluate AI against Minimax\n";
  std::cout << "6. Exit\n";
  std::cout << "Choose option: ";
}

Network CreateNewNetwork() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  auto rand = [&]() { return dis(gen); };

  Neural::Trainer trainer = Neural::Trainer::Create(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, rand);
  return trainer.network;
}

void TrainNetwork(bool loadExisting) {
  Network network;

  if (loadExisting && LoadNetwork(network, WEIGHTS_FILE)) {
    std::cout << "Loaded existing network from " << WEIGHTS_FILE << "\n";
  } else {
    std::cout << "Creating new network...\n";
    network = CreateNewNetwork();
  }

  int numIterations, gamesPerIteration, evalEvery;
  double learningRate;

  std::cout << "Enter number of training iterations: ";
  std::cin >> numIterations;

  std::cout << "Enter games per iteration: ";
  std::cin >> gamesPerIteration;

  std::cout << "Enter evaluation frequency (every N iterations): ";
  std::cin >> evalEvery;

  std::cout << "Enter learning rate (e.g., 0.001): ";
  std::cin >> learningRate;

  Neural::Trainer neuralTrainer = Neural::Trainer::Create(std::move(network));
  ConnectFour::Trainer gameTrainer(std::move(neuralTrainer), 800, 1.414);

  gameTrainer.Train(numIterations, gamesPerIteration, evalEvery, learningRate, 20);

  if (SaveNetwork(gameTrainer.GetNetwork(), WEIGHTS_FILE)) {
    std::cout << "Network saved to " << WEIGHTS_FILE << "\n";
  }
}

void PlayAgainstAI() {
  Network network;

  if (!LoadNetwork(network, WEIGHTS_FILE)) {
    std::cout << "No trained network found. Please train first.\n";
    return;
  }

  Board board;
  MCTS mcts(std::move(network), 1.414);

  std::cout << "\nYou are X, AI is O.\n";
  std::cout << "Enter column number (0-6) to make a move.\n";

  Player humanPlayer = Player::PLAYER1;
  Player aiPlayer = Player::PLAYER2;

  std::cout << "Do you want to go first? (y/n): ";
  char choice;
  std::cin >> choice;

  if (choice == 'n' || choice == 'N') {
    std::swap(humanPlayer, aiPlayer);
    std::cout << "You are O, AI is X.\n";
  }

  Player currentPlayer = Player::PLAYER1;

  while (!board.IsGameOver()) {
    board.Display();

    if (currentPlayer == humanPlayer) {
      std::cout << "Your turn. ";
      std::vector<int> validMoves = board.GetValidMoves();
      std::cout << "Valid moves: ";
      for (int col : validMoves) {
        std::cout << col << " ";
      }
      std::cout << "\nEnter column: ";

      int col;
      std::cin >> col;

      if (!board.IsValidMove(col)) {
        std::cout << "Invalid move! Try again.\n";
        continue;
      }

      board.MakeMove(col, currentPlayer);
    } else {
      std::cout << "AI is thinking (running MCTS for 2 seconds)...\n";

      // Run MCTS search with time limit
      mcts.SearchTime(board, currentPlayer, 2.0);
      int col = mcts.SelectBestMove();

      std::cout << "AI plays column " << col << "\n";
      std::cout << "Position value: " << mcts.GetRootValue() << "\n";

      board.MakeMove(col, currentPlayer);
    }

    currentPlayer = (currentPlayer == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
  }

  board.Display();

  Player winner = board.CheckWinner();
  if (winner == Player::NONE) {
    std::cout << "\nGame ended in a draw!\n";
  } else if (winner == humanPlayer) {
    std::cout << "\nCongratulations! You won!\n";
  } else {
    std::cout << "\nAI wins!\n";
  }
}

void WatchAIVsMinimax() {
  Network network;

  if (!LoadNetwork(network, WEIGHTS_FILE)) {
    std::cout << "No trained network found. Please train first.\n";
    return;
  }

  std::cout << "Enter minimax depth (e.g., 6): ";
  int depth;
  std::cin >> depth;

  MinimaxAI minimax(depth);
  Board board;

  Player nnPlayer = Player::PLAYER1;
  Player currentPlayer = Player::PLAYER1;

  std::cout << "\nNeural Network (X) vs Minimax depth " << depth << " (O)\n";
  std::cout << "Press Enter to see next move...\n";

  while (!board.IsGameOver()) {
    board.Display();

    std::cout << "\nPlayer " << (currentPlayer == Player::PLAYER1 ? "X" : "O")
              << "'s turn...\n";
    std::cin.ignore();
    std::cin.get();

    int move;

    if (currentPlayer == nnPlayer) {
      MCTS mcts(network, 1.414);
      std::cout << "Running MCTS...\n";
      mcts.SearchSimulations(board, currentPlayer, 400);
      move = mcts.SelectBestMove();
      std::cout << "NN plays column " << move << " (value: " << mcts.GetRootValue() << ")\n";
    } else {
      move = minimax.SelectMove(board, currentPlayer);
      std::cout << "Minimax plays column " << move
                << " (nodes: " << minimax.GetNodesEvaluated() << ")\n";
    }

    board.MakeMove(move, currentPlayer);
    currentPlayer = (currentPlayer == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
  }

  board.Display();

  Player winner = board.CheckWinner();
  if (winner == Player::NONE) {
    std::cout << "\nGame ended in a draw!\n";
  } else if (winner == nnPlayer) {
    std::cout << "\nNeural Network wins!\n";
  } else {
    std::cout << "\nMinimax wins!\n";
  }
}

void EvaluateAI() {
  Network network;

  if (!LoadNetwork(network, WEIGHTS_FILE)) {
    std::cout << "No trained network found. Please train first.\n";
    return;
  }

  std::cout << "Enter number of evaluation games: ";
  int numGames;
  std::cin >> numGames;

  std::cout << "Enter minimax depth (e.g., 6): ";
  int depth;
  std::cin >> depth;

  Neural::Trainer neuralTrainer = Neural::Trainer::Create(std::move(network));
  ConnectFour::Trainer gameTrainer(std::move(neuralTrainer), 400, 1.414);

  EvaluationResult result = gameTrainer.EvaluateAgainstMinimax(numGames, depth);

  std::cout << "\n=== Evaluation Results ===\n";
  std::cout << "Wins: " << result.wins << "\n";
  std::cout << "Losses: " << result.losses << "\n";
  std::cout << "Draws: " << result.draws << "\n";
  std::cout << "Win rate: " << (result.WinRate() * 100.0) << "%\n";
}

int main() {
  while (true) {
    DisplayMenu();

    int choice;
    std::cin >> choice;

    switch (choice) {
    case 1:
      TrainNetwork(false);
      break;
    case 2:
      TrainNetwork(true);
      break;
    case 3:
      PlayAgainstAI();
      break;
    case 4:
      WatchAIVsMinimax();
      break;
    case 5:
      EvaluateAI();
      break;
    case 6:
      std::cout << "Goodbye!\n";
      return 0;
    default:
      std::cout << "Invalid option. Try again.\n";
    }
  }

  return 0;
}
