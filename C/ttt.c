/*
Licensed under the MIT License given below.
Copyright 2023 Daniel Lidstrom
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "ttt.h"
#include "neural.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Board {
  char squares[9];
  char turn;
} Board;

char EMPTY = ' ';
char X = 'X';
char O = 'O';
const char* X_WON = "X won";
const char* O_WON = "O won";
const char* DRAW = "Draw";
const int N_INPUTS = 18;

void board_init(Board* board) {
  memset(&board->squares, EMPTY, 9);
  board->turn = X;
}

void board_set_input_vector(Board* board, double* output) {
  // X's point of view
  for (int i = 0; i < 9; i++) {
    output[i] = board->squares[i] == board->turn;
    output[i + 9] = board->squares[i] != board->turn && board->squares[i] != EMPTY;
  }
}

/**
 * |---|---|---|
 * | X | X | O |   1   2   3
 * |---|---|---|
 * | X | X | O |   4   5   6
 * |---|---|---|
 * | X | X | O |   7   8   9
 * |---|---|---|
 */
void board_print(Board* board) {
  double input_vector[N_INPUTS] = {0};
  board_set_input_vector(board, input_vector);

  printf(
    "|---|---|---|\n"
    "| %c | %c | %c |   1   2   3\n"
    "|---|---|---|\n"
    "| %c | %c | %c |   4   5   6\n"
    "|---|---|---|\n"
    "| %c | %c | %c |   7   8   9\n"
    "|---|---|---|\n"
    "Turn to move: %c\n"
    "Network input: %.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f %.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f\n",
    board->squares[0], board->squares[1], board->squares[2],
    board->squares[3], board->squares[4], board->squares[5],
    board->squares[6], board->squares[7], board->squares[8],
    board->turn,
    input_vector[0], input_vector[1], input_vector[2],
    input_vector[3], input_vector[4], input_vector[5],
    input_vector[6], input_vector[7], input_vector[8],
    input_vector[9], input_vector[10], input_vector[11],
    input_vector[12], input_vector[13], input_vector[14],
    input_vector[15], input_vector[16], input_vector[17]);
}

const char* board_game_over(Board* board) {
  int filled = 1;
  for (int i = 0; i < 9; i++) {
    if (board->squares[i] == EMPTY) {
      filled = 0;
      break;
    }
  }

  // check diagonals
  if ((board->squares[0] == X && board->squares[4] == X && board->squares[8] == X)
      || (board->squares[2] == X && board->squares[4] == X && board->squares[6] == X)) {
    return X_WON;
  }
  if ((board->squares[0] == O && board->squares[4] == O && board->squares[8] == O)
      || (board->squares[2] == O && board->squares[4] == O && board->squares[6] == O)) {
    return O_WON;
  }

  for (int i = 0; i < 3; i++) {
    // check rows
    if (board->squares[i] == X && board->squares[i + 1] == X && board->squares[i + 2] == X) {
      return X_WON;
    }
    if (board->squares[i] == O && board->squares[i + 1] == O && board->squares[i + 2] == O) {
      return O_WON;
    }
    // check cols
    if (board->squares[i] == X && board->squares[i + 3] == X && board->squares[i + 6] == X) {
      return X_WON;
    }
    if (board->squares[i] == O && board->squares[i + 3] == O && board->squares[i + 6] == O) {
      return O_WON;
    }
  }

  if (filled) return DRAW;
  return NULL;
}

int board_valid_move(Board* board, int square) {
  return board->squares[square] == EMPTY;
}

void board_play(Board* board, int square) {
  board->squares[square] = board->turn;
  if (board->turn == X)
    board->turn = O;
  else
    board->turn = X;
}

// ask for input, returns 0-8
int read_input(void) {
  char buffer[10];
  while (1) {
    printf("Enter a number: ");
    if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
      fputs("error: invalid input\n", stderr);
      continue;
    }

    if (strlen(buffer) != 2 || buffer[0] < '1' || buffer[0] > '9') {
      printf("Please enter a single digit 1-9.\n");
      continue;
    }

    break;
  }

  return buffer[0] - '1';
}

typedef struct Sample {
  double input_vector[N_INPUTS];
} Sample;

void train_network(Network* network) {
  Trainer trainer = {0};
  trainer_init(&trainer, network);

  // run some games and learn the results
  // we will collect last 1000 board positions seen
  const int RUNS = 4000;
  const int N = 1000;
  Sample* samples = calloc(1000 * N_INPUTS, sizeof(*samples));
  Board board = {0};

  // play game until finished, collect input vectors along the way
  // once we have 1000 samples then start training network
  // repeat play for how long?
  board_init(&board);
  const char* status = board_game_over(&board);
  while (status == 0) {
    status = board_game_over(&board);
  }

  free(samples);
}

void tic_tac_toe(void) {
  Network network = {0};
  network_init(&network, 2, 2, 6, Rand);

  printf("Run tic-tac-toe\n");
  Board board = {0};
  board_init(&board);

  printf("Play until board is filled. Enter the number of the square.\n");
  board_print(&board);
  while (board_game_over(&board) == 0) {
    int input = read_input();
    if (board_valid_move(&board, input) == 0) {
      printf("Invalid move. Pick an empty square, please.\n");
      continue;
    }

    board_play(&board, input);
    board_print(&board);
  }

  printf("%s\n", board_game_over(&board));
}
