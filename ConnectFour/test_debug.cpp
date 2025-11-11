#define DEBUG_EXPANSION
#include "mcts.h"
#include "../Cpp/neural.h"
#include <iostream>
#include <random>

using namespace ConnectFour;

int main() {
    #ifdef DEBUG_EXPANSION
    std::cout << "DEBUG_EXPANSION is defined\n";
    #else
    std::cout << "DEBUG_EXPANSION is NOT defined\n";
    #endif

    // Test position just before a forced win
    Board board;
    // Set up: O has 3 in a row, X must block
    board.MakeMove(0, Player::PLAYER2); // O
    board.MakeMove(0, Player::PLAYER1); // X blocks
    board.MakeMove(1, Player::PLAYER2); // O
    board.MakeMove(1, Player::PLAYER1); // X blocks
    board.MakeMove(2, Player::PLAYER2); // O
    board.MakeMove(2, Player::PLAYER1); // X blocks

    std::cout << "Test position (O threatens to win in column 3):\n";
    board.Display();

    std::cout << "\nManual verification:\n";
    std::cout << "Valid moves: ";
    for (int col : board.GetValidMoves()) {
        std::cout << col << " ";
    }
    std::cout << "\n";

    // Test each column to see if opponent wins
    for (int col = 0; col < COLS; col++) {
        if (board.IsValidMove(col)) {
            Board testBoard = board;
            testBoard.MakeMove(col, Player::PLAYER2);
            Player winner = testBoard.CheckWinner();
            std::cout << "If O plays col " << col << ": ";
            if (winner == Player::PLAYER2) {
                std::cout << "O WINS!\n";
            } else {
                std::cout << "No winner\n";
            }
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    auto rand = [&]() { return dis(gen); };
    Neural::Trainer trainer = Neural::Trainer::Create(BOARD_SIZE * 3, 256, COLS, rand);

    std::cout << "\nRunning MCTS with 100 simulations for X (with debug)...\n";
    MCTS mcts(trainer.network, 1.414);
    mcts.SearchSimulations(board, Player::PLAYER1, 100);

    std::vector<int> visits = mcts.GetVisitCounts();

    std::cout << "\nVisit distribution:\n";
    for (int i = 0; i < COLS; i++) {
        if (board.IsValidMove(i)) {
            std::cout << "  Col " << i << ": " << visits[i] << " visits\n";
        }
    }

    int bestMove = mcts.SelectBestMove();
    std::cout << "\nBest move: " << bestMove << " (should be 3 to block)\n";

    return 0;
}
