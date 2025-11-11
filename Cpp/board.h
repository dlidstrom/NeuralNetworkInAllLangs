/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#if !defined(CONNECT_FOUR_H)
#define CONNECT_FOUR_H

#include <array>
#include <optional>
#include <vector>

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
  bool IsValidMove(int col) const;
  std::vector<int> GetValidMoves() const;
  Player CheckWinner() const;
  bool IsFull() const;
  bool IsGameOver() const;

  // Board state
  Player GetCell(int row, int col) const;
  void Reset();
  void Display() const;

  // Neural network interface
  std::vector<double> ToNeuralInput(Player perspective) const;

private:
  std::array<Player, BOARD_SIZE> cells;
  std::array<int, COLS> heights;

  bool CheckLine(int startRow, int startCol, int dRow, int dCol) const;
};

} // namespace ConnectFour

#endif
