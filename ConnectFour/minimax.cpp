/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#include "minimax.h"
#include <algorithm>

namespace ConnectFour {

int MinimaxAI::SelectMove(const Board& board, Player player) {
  nodesEvaluated = 0;

  std::vector<int> validMoves = board.GetValidMoves();
  if (validMoves.empty()) {
    return -1;
  }

  int bestMove = validMoves[0];
  double bestValue = -std::numeric_limits<double>::infinity();

  // Make a mutable copy
  Board mutableBoard = board;

  for (int col : validMoves) {
    mutableBoard.MakeMove(col, player);

    // Get opponent
    Player opponent = (player == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;

    // Run minimax
    double value = Minimax(mutableBoard, maxDepth - 1,
                          -std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::infinity(),
                          opponent, player);

    mutableBoard.UndoMove(col);

    if (value > bestValue) {
      bestValue = value;
      bestMove = col;
    }
  }

  return bestMove;
}

double MinimaxAI::Minimax(Board& board, int depth, double alpha, double beta,
                          Player player, Player maximizingPlayer) {
  nodesEvaluated++;

  // Check terminal conditions
  Player winner = board.CheckWinner();
  if (winner != Player::NONE) {
    // Game over - return large value
    if (winner == maximizingPlayer) {
      return 10000.0 + depth; // Prefer faster wins
    } else {
      return -10000.0 - depth; // Prefer slower losses
    }
  }

  if (board.IsFull()) {
    return 0.0; // Draw
  }

  if (depth == 0) {
    // Leaf node - evaluate heuristically
    return EvaluatePosition(board, maximizingPlayer);
  }

  std::vector<int> validMoves = board.GetValidMoves();
  Player opponent = (player == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;

  if (player == maximizingPlayer) {
    // Maximizing player
    double maxEval = -std::numeric_limits<double>::infinity();

    for (int col : validMoves) {
      board.MakeMove(col, player);
      double eval = Minimax(board, depth - 1, alpha, beta, opponent, maximizingPlayer);
      board.UndoMove(col);

      maxEval = std::max(maxEval, eval);
      alpha = std::max(alpha, eval);

      if (beta <= alpha) {
        break; // Beta cutoff
      }
    }

    return maxEval;
  } else {
    // Minimizing player
    double minEval = std::numeric_limits<double>::infinity();

    for (int col : validMoves) {
      board.MakeMove(col, player);
      double eval = Minimax(board, depth - 1, alpha, beta, opponent, maximizingPlayer);
      board.UndoMove(col);

      minEval = std::min(minEval, eval);
      beta = std::min(beta, eval);

      if (beta <= alpha) {
        break; // Alpha cutoff
      }
    }

    return minEval;
  }
}

int MinimaxAI::CountThreats(const Board& board, Player player, int length) {
  int count = 0;

  // Check all possible lines of WIN_LENGTH
  for (int row = 0; row < ROWS; row++) {
    for (int col = 0; col < COLS; col++) {
      // Check horizontal
      if (col <= COLS - WIN_LENGTH) {
        int myPieces = 0;
        int empty = 0;
        for (int i = 0; i < WIN_LENGTH; i++) {
          Player cell = board.GetCell(row, col + i);
          if (cell == player) myPieces++;
          else if (cell == Player::NONE) empty++;
        }
        if (myPieces == length && empty == WIN_LENGTH - length) {
          count++;
        }
      }

      // Check vertical
      if (row <= ROWS - WIN_LENGTH) {
        int myPieces = 0;
        int empty = 0;
        for (int i = 0; i < WIN_LENGTH; i++) {
          Player cell = board.GetCell(row + i, col);
          if (cell == player) myPieces++;
          else if (cell == Player::NONE) empty++;
        }
        if (myPieces == length && empty == WIN_LENGTH - length) {
          count++;
        }
      }

      // Check diagonal (down-right)
      if (row <= ROWS - WIN_LENGTH && col <= COLS - WIN_LENGTH) {
        int myPieces = 0;
        int empty = 0;
        for (int i = 0; i < WIN_LENGTH; i++) {
          Player cell = board.GetCell(row + i, col + i);
          if (cell == player) myPieces++;
          else if (cell == Player::NONE) empty++;
        }
        if (myPieces == length && empty == WIN_LENGTH - length) {
          count++;
        }
      }

      // Check diagonal (down-left)
      if (row <= ROWS - WIN_LENGTH && col >= WIN_LENGTH - 1) {
        int myPieces = 0;
        int empty = 0;
        for (int i = 0; i < WIN_LENGTH; i++) {
          Player cell = board.GetCell(row + i, col - i);
          if (cell == player) myPieces++;
          else if (cell == Player::NONE) empty++;
        }
        if (myPieces == length && empty == WIN_LENGTH - length) {
          count++;
        }
      }
    }
  }

  return count;
}

double MinimaxAI::EvaluatePosition(const Board& board, Player player) {
  Player opponent = (player == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;

  double score = 0.0;

  // Count threats of different lengths
  // Three in a row is very valuable
  score += CountThreats(board, player, 3) * 100.0;
  score -= CountThreats(board, opponent, 3) * 100.0;

  // Two in a row is somewhat valuable
  score += CountThreats(board, player, 2) * 10.0;
  score -= CountThreats(board, opponent, 2) * 10.0;

  // Control of center is valuable
  int centerCol = COLS / 2;
  for (int row = 0; row < ROWS; row++) {
    if (board.GetCell(row, centerCol) == player) {
      score += 3.0;
    } else if (board.GetCell(row, centerCol) == opponent) {
      score -= 3.0;
    }
  }

  return score;
}

} // namespace ConnectFour
