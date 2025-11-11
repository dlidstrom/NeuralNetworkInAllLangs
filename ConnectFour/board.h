/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#ifndef CONNECTFOUR_BOARD_H
#define CONNECTFOUR_BOARD_H

#include <array>
#include <vector>
#include <string>

namespace ConnectFour {

constexpr int ROWS = 6;
constexpr int COLS = 7;
constexpr int BOARD_SIZE = ROWS * COLS;
constexpr int WIN_LENGTH = 4;

enum class Player { NONE = 0, PLAYER1 = 1, PLAYER2 = 2 };

class Board {
public:
  Board();

  // Game logic
  bool MakeMove(int col, Player player);
  void UndoMove(int col);
  bool IsValidMove(int col) const;
  std::vector<int> GetValidMoves() const;
  Player CheckWinner() const;
  bool IsFull() const;
  bool IsGameOver() const;

  // Board state
  Player GetCell(int row, int col) const;
  void Reset();
  void Display() const;

  // Neural network interface with normalization and mirroring
  std::vector<double> ToNeuralInput(Player perspective) const;

  // Get normalized input - always from current player's perspective
  // Returns the smaller of normal and mirrored board
  std::vector<double> GetNormalizedInput(Player currentPlayer, bool& wasMirrored) const;

  // Mirror a column index (for use with mirrored boards)
  static int MirrorColumn(int col) { return COLS - 1 - col; }

private:
  std::array<Player, BOARD_SIZE> cells;
  std::array<int, COLS> heights;

  bool CheckLine(int startRow, int startCol, int dRow, int dCol) const;
  Board GetMirroredBoard() const;
  std::vector<double> BoardToInput(Player perspective) const;
};

} // namespace ConnectFour

#endif // CONNECTFOUR_BOARD_H
