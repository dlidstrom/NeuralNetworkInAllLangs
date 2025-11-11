/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#include "ai_player.h"
#include "neural.h"
#include "neural_io.h"
#include "game_trainer.h"
#include <iostream>
#include <random>
#include <string>

using namespace ConnectFour;
using namespace Neural;

const std::string WEIGHTS_FILE = "connect_four_weights.bin";
const int INPUT_SIZE =
    BOARD_SIZE * 3; // 3 values per cell (my piece, opponent piece, empty)
const int HIDDEN_SIZE = 128;
const int OUTPUT_SIZE = COLS; // One output per column

void DisplayMenu() {
  std::cout << "\n=== Connect Four AI ===\n";
  std::cout << "1. Train new network\n";
  std::cout << "2. Continue training existing network\n";
  std::cout << "3. Play against AI\n";
  std::cout << "4. Watch AI play against itself\n";
  std::cout << "5. Exit\n";
  std::cout << "Choose option: ";
}

Network CreateNewNetwork() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  auto rand = [&]() { return dis(gen); };

  Trainer trainer = Trainer::Create(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, rand);
  return trainer.network;
}

void TrainNetwork(int numGames, bool loadExisting) {
  Network network;

  if (loadExisting && LoadNetwork(network, WEIGHTS_FILE)) {
    std::cout << "Loaded existing network from " << WEIGHTS_FILE << "\n";
  } else {
    std::cout << "Creating new network...\n";
    network = CreateNewNetwork();
  }

  std::cout << "Enter learning rate (e.g., 0.01): ";
  double learningRate;
  std::cin >> learningRate;

  Trainer trainer = Trainer::Create(std::move(network));
  GameTrainer gameTrainer(std::move(trainer), 0.2);

  gameTrainer.Train(numGames, learningRate, 100);

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

  AIPlayer ai(std::move(network), 0.0); // No exploration during play
  Board board;

  std::cout << "\nYou are X, AI is O.\n";
  std::cout << "Enter column number (0-6) to make a move.\n";

  Player humanPlayer = Player::PLAYER1;
  Player aiPlayer = Player::PLAYER2;

  // Ask who goes first
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
      std::cout << "AI is thinking...\n";
      int col = ai.SelectMove(board, currentPlayer, false);
      std::cout << "AI plays column " << col << "\n";
      board.MakeMove(col, currentPlayer);
    }

    currentPlayer =
        (currentPlayer == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
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

void WatchAIPlay() {
  Network network;

  if (!LoadNetwork(network, WEIGHTS_FILE)) {
    std::cout << "No trained network found. Please train first.\n";
    return;
  }

  AIPlayer ai(std::move(network), 0.0);
  Board board;

  Player currentPlayer = Player::PLAYER1;

  std::cout << "\nWatching AI play against itself...\n";
  std::cout << "Press Enter to see next move...\n";

  while (!board.IsGameOver()) {
    board.Display();

    std::cout << "\nPlayer " << (currentPlayer == Player::PLAYER1 ? "X" : "O")
              << "'s turn...\n";
    std::cin.ignore();
    std::cin.get();

    int col = ai.SelectMove(board, currentPlayer, false);
    std::cout << "AI plays column " << col << "\n";

    board.MakeMove(col, currentPlayer);
    currentPlayer =
        (currentPlayer == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
  }

  board.Display();

  Player winner = board.CheckWinner();
  if (winner == Player::NONE) {
    std::cout << "\nGame ended in a draw!\n";
  } else {
    std::cout << "\nPlayer " << (winner == Player::PLAYER1 ? "X" : "O")
              << " wins!\n";
  }
}

int main() {
  while (true) {
    DisplayMenu();

    int choice;
    std::cin >> choice;

    switch (choice) {
    case 1: {
      std::cout << "Enter number of training games: ";
      int numGames;
      std::cin >> numGames;
      TrainNetwork(numGames, false);
      break;
    }
    case 2: {
      std::cout << "Enter number of additional training games: ";
      int numGames;
      std::cin >> numGames;
      TrainNetwork(numGames, true);
      break;
    }
    case 3:
      PlayAgainstAI();
      break;
    case 4:
      WatchAIPlay();
      break;
    case 5:
      std::cout << "Goodbye!\n";
      return 0;
    default:
      std::cout << "Invalid option. Try again.\n";
    }
  }

  return 0;
}
