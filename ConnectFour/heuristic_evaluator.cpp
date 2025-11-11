/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#include "heuristic_evaluator.h"
#include <algorithm>
#include <cmath>

namespace ConnectFour {

std::pair<std::vector<double>, double> HeuristicEvaluator::Evaluate(
    const Board& board, Player player) {
  
  std::vector<int> validMoves = board.GetValidMoves();
  std::vector<double> priors(COLS, 0.0);

  if (validMoves.empty()) {
    return {priors, 0.0};
  }

  // Check for immediate wins/blocks and evaluate each move
  std::vector<double> moveScores(COLS, -1e9);
  Player opponent = GetOpponent(player);

  for (int col : validMoves) {
    Board testBoard = board;
    testBoard.MakeMove(col, player);

    // Check if this move wins immediately
    if (testBoard.CheckWinner() == player) {
      moveScores[col] = 1e6; // Winning move
      continue;
    }

    // Check if we must block opponent win
    Board opponentBoard = board;
    opponentBoard.MakeMove(col, opponent);
    if (opponentBoard.CheckWinner() == opponent) {
      moveScores[col] = 1e5; // Blocking move
      continue;
    }

    // Otherwise, evaluate the resulting position
    moveScores[col] = EvaluatePosition(testBoard, player);
  }

  // Find min and max scores for normalization
  double minScore = *std::min_element(moveScores.begin(), moveScores.end());
  double maxScore = *std::max_element(moveScores.begin(), moveScores.end());

  // Convert scores to probabilities using softmax-like transformation
  std::vector<double> expScores(COLS, 0.0);
  double sumExp = 0.0;

  for (int col : validMoves) {
    // Normalize to [0, 1] range, then apply exponential
    double normalized = (maxScore > minScore) ? 
        (moveScores[col] - minScore) / (maxScore - minScore) : 0.5;
    expScores[col] = std::exp(normalized * 5.0); // Temperature = 0.2
    sumExp += expScores[col];
  }

  // Normalize to probabilities
  if (sumExp > 0.0) {
    for (int col : validMoves) {
      priors[col] = expScores[col] / sumExp;
    }
  } else {
    // Uniform distribution if something went wrong
    double uniform = 1.0 / validMoves.size();
    for (int col : validMoves) {
      priors[col] = uniform;
    }
  }

  // Value estimate: evaluate current position
  double value = EvaluatePosition(board, player);
  
  // Normalize value to [-1, 1] range
  // A score of 100 (one 3-in-a-row) is significant but not terminal
  value = std::tanh(value / 200.0);

  return {priors, value};
}

int HeuristicEvaluator::CountThreats(const Board& board, Player player, int length) {
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

double HeuristicEvaluator::EvaluatePosition(const Board& board, Player player) {
  Player opponent = GetOpponent(player);

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
