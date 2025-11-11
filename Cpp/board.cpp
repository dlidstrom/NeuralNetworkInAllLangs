/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#include "board.h"
#include <iostream>

namespace ConnectFour {

Board::Board() { Reset(); }

void Board::Reset() {
  cells.fill(Player::NONE);
  heights.fill(0);
}

Player Board::GetCell(int row, int col) const {
  return cells[row * COLS + col];
}

bool Board::IsValidMove(int col) const {
  return col >= 0 && col < COLS && heights[col] < ROWS;
}

std::vector<int> Board::GetValidMoves() const {
  std::vector<int> moves;
  for (int col = 0; col < COLS; col++) {
    if (IsValidMove(col)) {
      moves.push_back(col);
    }
  }
  return moves;
}

bool Board::MakeMove(int col, Player player) {
  if (!IsValidMove(col)) {
    return false;
  }

  int row = heights[col];
  cells[row * COLS + col] = player;
  heights[col]++;
  return true;
}

bool Board::CheckLine(int startRow, int startCol, int dRow, int dCol) const {
  for (int i = 0; i < WIN_LENGTH; i++) {
    int row = startRow + i * dRow;
    int col = startCol + i * dCol;

    if (row < 0 || row >= ROWS || col < 0 || col >= COLS) {
      return false;
    }

    Player cell = GetCell(row, col);
    if (cell == Player::NONE ||
        (i > 0 && cell != GetCell(startRow, startCol))) {
      return false;
    }
  }

  return GetCell(startRow, startCol) != Player::NONE;
}

Player Board::CheckWinner() const {
  // Check horizontal
  for (int row = 0; row < ROWS; row++) {
    for (int col = 0; col <= COLS - WIN_LENGTH; col++) {
      if (CheckLine(row, col, 0, 1)) {
        return GetCell(row, col);
      }
    }
  }

  // Check vertical
  for (int row = 0; row <= ROWS - WIN_LENGTH; row++) {
    for (int col = 0; col < COLS; col++) {
      if (CheckLine(row, col, 1, 0)) {
        return GetCell(row, col);
      }
    }
  }

  // Check diagonal (down-right)
  for (int row = 0; row <= ROWS - WIN_LENGTH; row++) {
    for (int col = 0; col <= COLS - WIN_LENGTH; col++) {
      if (CheckLine(row, col, 1, 1)) {
        return GetCell(row, col);
      }
    }
  }

  // Check diagonal (down-left)
  for (int row = 0; row <= ROWS - WIN_LENGTH; row++) {
    for (int col = WIN_LENGTH - 1; col < COLS; col++) {
      if (CheckLine(row, col, 1, -1)) {
        return GetCell(row, col);
      }
    }
  }

  return Player::NONE;
}

bool Board::IsFull() const {
  for (int col = 0; col < COLS; col++) {
    if (heights[col] < ROWS) {
      return false;
    }
  }
  return true;
}

bool Board::IsGameOver() const {
  return CheckWinner() != Player::NONE || IsFull();
}

void Board::Display() const {
  std::cout << "\n  ";
  for (int col = 0; col < COLS; col++) {
    std::cout << col << " ";
  }
  std::cout << "\n +" << std::string(COLS * 2 - 1, '-') << "+\n";

  for (int row = ROWS - 1; row >= 0; row--) {
    std::cout << " |";
    for (int col = 0; col < COLS; col++) {
      Player cell = GetCell(row, col);
      char c = ' ';
      if (cell == Player::PLAYER1)
        c = 'X';
      else if (cell == Player::PLAYER2)
        c = 'O';
      std::cout << c << '|';
    }
    std::cout << "\n";
  }
  std::cout << " +" << std::string(COLS * 2 - 1, '-') << "+\n";
}

std::vector<double> Board::ToNeuralInput(Player perspective) const {
  // Encode board as: [my pieces, opponent pieces, empty]
  // Each position gets 3 values (one-hot encoding)
  std::vector<double> input(BOARD_SIZE * 3, 0.0);

  Player opponent =
      (perspective == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;

  for (int i = 0; i < BOARD_SIZE; i++) {
    if (cells[i] == perspective) {
      input[i * 3] = 1.0;
    } else if (cells[i] == opponent) {
      input[i * 3 + 1] = 1.0;
    } else {
      input[i * 3 + 2] = 1.0;
    }
  }

  return input;
}

} // namespace ConnectFour
