/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#include "tests.h"
#include "board.h"
#include "minimax.h"
#include "mcts.h"
#include "heuristic_evaluator.h"
#include "neural_evaluator.h"
#include "../Cpp/neural.h"
#include <iostream>
#include <random>

namespace ConnectFour {

void TestRunner::AddTest(const std::string& name, std::function<TestResult()> test) {
  tests.push_back({name, test});
}

void TestRunner::RunAll() {
  std::cout << "\n=== Running Tests ===\n\n";

  int passed = 0;
  failureCount = 0;

  for (const auto& [name, test] : tests) {
    std::cout << "Running: " << name << "... ";
    TestResult result = test();

    if (result.passed) {
      std::cout << "PASSED";
      passed++;
    } else {
      std::cout << "FAILED";
      failureCount++;
    }

    if (!result.message.empty()) {
      std::cout << " - " << result.message;
    }
    std::cout << "\n";
  }

  std::cout << "\n=== Test Summary ===\n";
  std::cout << "Passed: " << passed << "/" << tests.size() << "\n";
  std::cout << "Failed: " << failureCount << "/" << tests.size() << "\n";
}

TestResult TestBoardBasics() {
  Board board;

  // Test initial state
  if (board.GetValidMoves().size() != COLS) {
    return {"TestBoardBasics", false, "Initial board should have all columns valid"};
  }

  // Test making moves
  if (!board.MakeMove(3, Player::PLAYER1)) {
    return {"TestBoardBasics", false, "Should be able to make move in column 3"};
  }

  if (board.GetCell(0, 3) != Player::PLAYER1) {
    return {"TestBoardBasics", false, "Piece should be at bottom of column"};
  }

  // Test win detection - horizontal
  Board winBoard;
  for (int col = 0; col < 4; col++) {
    winBoard.MakeMove(col, Player::PLAYER1);
  }

  if (winBoard.CheckWinner() != Player::PLAYER1) {
    return {"TestBoardBasics", false, "Should detect horizontal win"};
  }

  return {"TestBoardBasics", true, ""};
}

TestResult TestMinimaxBasics() {
  MinimaxAI minimax(4);

  // Test that minimax finds obvious win
  Board board;
  // Set up board where PLAYER1 can win by playing column 3
  // X X X _ _ _ _
  board.MakeMove(0, Player::PLAYER1);
  board.MakeMove(1, Player::PLAYER1);
  board.MakeMove(2, Player::PLAYER1);

  int move = minimax.SelectMove(board, Player::PLAYER1);

  if (move != 3) {
    return {"TestMinimaxBasics", false, "Minimax should find winning move (col 3), got " + std::to_string(move)};
  }

  return {"TestMinimaxBasics", true, ""};
}

TestResult TestMCTSBasics() {
  // Create heuristic evaluator
  auto evaluator = std::make_unique<HeuristicEvaluator>();

  MCTS mcts(std::move(evaluator), 1.414);

  Board board;

  // Run MCTS
  mcts.SearchSimulations(board, Player::PLAYER1, 100);

  int move = mcts.SelectBestMove();

  if (move < 0 || move >= COLS) {
    return {"TestMCTSBasics", false, "MCTS should return valid move"};
  }

  return {"TestMCTSBasics", true, "Selected move: " + std::to_string(move)};
}

TestResult TestMCTSFindsWinInOne() {
  // Create heuristic evaluator
  auto evaluator = std::make_unique<HeuristicEvaluator>();

  // Set up board where PLAYER1 can win by playing column 3
  Board board;
  board.MakeMove(0, Player::PLAYER1);
  board.MakeMove(1, Player::PLAYER1);
  board.MakeMove(2, Player::PLAYER1);

  MCTS mcts(std::move(evaluator), 1.414);
  mcts.SearchSimulations(board, Player::PLAYER1, 400);

  int move = mcts.SelectBestMove();

  if (move != 3) {
    return {"TestMCTSFindsWinInOne", false,
            "MCTS should find immediate win (col 3), got " + std::to_string(move)};
  }

  return {"TestMCTSFindsWinInOne", true, ""};
}

TestResult TestMCTSBlocksLossInOne() {
  // Create heuristic evaluator
  auto evaluator = std::make_unique<HeuristicEvaluator>();

  // Set up board where PLAYER2 has three in a row, PLAYER1 must block
  Board board;
  board.MakeMove(0, Player::PLAYER2);
  board.MakeMove(1, Player::PLAYER2);
  board.MakeMove(2, Player::PLAYER2);
  board.MakeMove(4, Player::PLAYER1); // Random move

  MCTS mcts(std::move(evaluator), 1.414);
  mcts.SearchSimulations(board, Player::PLAYER1, 400);

  int move = mcts.SelectBestMove();

  if (move != 3) {
    return {"TestMCTSBlocksLossInOne", false,
            "MCTS should block opponent win (col 3), got " + std::to_string(move)};
  }

  return {"TestMCTSBlocksLossInOne", true, ""};
}

int PlayTestGame(MinimaxAI& minimax, std::unique_ptr<Evaluator> evaluator,
                 bool mctsGoesFirst, int mctsSimulations) {
  Board board;
  Player mctsPlayer = mctsGoesFirst ? Player::PLAYER1 : Player::PLAYER2;
  Player currentPlayer = Player::PLAYER1;

  int moveCount = 0;
  const int MAX_MOVES = 42;

  while (!board.IsGameOver() && moveCount < MAX_MOVES) {
    int move;

    if (currentPlayer == mctsPlayer) {
      // Create a fresh evaluator for this move (since MCTS takes ownership)
      auto moveEvaluator = std::make_unique<HeuristicEvaluator>();
      MCTS mcts(std::move(moveEvaluator), 1.414);
      mcts.SearchSimulations(board, currentPlayer, mctsSimulations);
      move = mcts.SelectBestMove();
    } else {
      move = minimax.SelectMove(board, currentPlayer);
    }

    if (move < 0 || !board.MakeMove(move, currentPlayer)) {
      // Invalid move
      return (currentPlayer == mctsPlayer) ? -1 : 1;
    }

    currentPlayer = (currentPlayer == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
    moveCount++;
  }

  Player winner = board.CheckWinner();

  if (winner == Player::NONE) {
    return 0; // Draw
  } else if (winner == mctsPlayer) {
    return 1; // MCTS wins
  } else {
    return -1; // Minimax wins
  }
}

TestResult TestMCTSVsMinimax2Ply() {
  MinimaxAI minimax(2);

  // Play a few test games
  int mctsWins = 0;
  int minimaxWins = 0;
  int draws = 0;
  const int NUM_GAMES = 6;

  std::cout << "\n  Playing " << NUM_GAMES << " test games... ";

  for (int i = 0; i < NUM_GAMES; i++) {
    bool mctsFirst = (i % 2 == 0);
    auto evaluator = std::make_unique<HeuristicEvaluator>();
    int result = PlayTestGame(minimax, std::move(evaluator), mctsFirst, 800);

    if (result == 1) {
      mctsWins++;
    } else if (result == -1) {
      minimaxWins++;
    } else {
      draws++;
    }
  }

  std::cout << "\n  Results: MCTS " << mctsWins << " - Minimax " << minimaxWins
            << " - Draws " << draws << "\n  ";

  // MCTS with heuristic evaluation should beat minimax depth 2
  if (mctsWins < NUM_GAMES * 0.5) {
    return {"TestMCTSVsMinimax2Ply", false,
            "MCTS should win at least 50% of games against minimax depth 2"};
  }

  return {"TestMCTSVsMinimax2Ply", true,
          "MCTS won " + std::to_string(mctsWins) + "/" + std::to_string(NUM_GAMES) + " games"};
}

TestResult TestUntrainedMCTSVsMinimax2() {
  // Test MCTS with heuristic evaluation vs minimax depth 2
  MinimaxAI minimax(2);

  int mctsWins = 0;
  int minimaxWins = 0;
  int draws = 0;
  const int NUM_GAMES = 4;  // Reduced for speed
  const int MCTS_SIMS = 800; // With heuristic, we don't need as many simulations

  std::cout << "\n  Playing " << NUM_GAMES << " games (MCTS: " << MCTS_SIMS
            << " sims vs Minimax depth 2)...\n  ";

  for (int i = 0; i < NUM_GAMES; i++) {
    bool mctsFirst = (i % 2 == 0);
    auto evaluator = std::make_unique<HeuristicEvaluator>();
    int result = PlayTestGame(minimax, std::move(evaluator), mctsFirst, MCTS_SIMS);

    if (result == 1) {
      mctsWins++;
      std::cout << "W";
    } else if (result == -1) {
      minimaxWins++;
      std::cout << "L";
    } else {
      draws++;
      std::cout << "D";
    }
    std::cout.flush();
  }

  std::cout << "\n  Final: MCTS " << mctsWins << " - " << minimaxWins
            << " Minimax (Draws: " << draws << ")\n  ";

  double winRate = static_cast<double>(mctsWins) / NUM_GAMES;

  // We expect MCTS with heuristic to beat minimax depth 2 consistently
  if (winRate >= 0.75) {
    return {"TestUntrainedMCTSVsMinimax2", true,
            "MCTS achieved " + std::to_string(static_cast<int>(winRate * 100)) + "% win rate"};
  } else if (winRate >= 0.5) {
    return {"TestUntrainedMCTSVsMinimax2", true,
            "MCTS competitive with " + std::to_string(static_cast<int>(winRate * 100)) +
            "% win rate (acceptable)"};
  } else {
    return {"TestUntrainedMCTSVsMinimax2", false,
            "MCTS only achieved " + std::to_string(static_cast<int>(winRate * 100)) +
            "% win rate (expected >50%)"};
  }
}

} // namespace ConnectFour
