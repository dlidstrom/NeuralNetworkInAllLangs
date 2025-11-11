#include "mcts.h"
#include "../Cpp/neural.h"
#include <iostream>
#include <random>

using namespace ConnectFour;

int main() {
    // Test position just before a forced win
    Board board;
    // Set up: O has 3 in a row, X must block
    board.MakeMove(0, Player::PLAYER2); // O
    board.MakeMove(0, Player::PLAYER1); // X blocks
    board.MakeMove(1, Player::PLAYER2); // O  
    board.MakeMove(1, Player::PLAYER1); // X blocks
    board.MakeMove(2, Player::PLAYER2); // O
    board.MakeMove(2, Player::PLAYER1); // X blocks
    
    // Now if X doesn't play column 3, O wins next turn
    
    std::cout << "Test position (O threatens to win in column 3):\n";
    board.Display();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    auto rand = [&]() { return dis(gen); };
    Neural::Trainer trainer = Neural::Trainer::Create(BOARD_SIZE * 3, 256, COLS, rand);
    
    std::cout << "\nRunning MCTS with 2000 simulations for X...\n";
    MCTS mcts(trainer.network, 1.414);
    mcts.SearchSimulations(board, Player::PLAYER1, 2000);
    
    std::vector<int> visits = mcts.GetVisitCounts();
    
    std::cout << "\nVisit distribution:\n";
    for (int i = 0; i < COLS; i++) {
        if (board.IsValidMove(i)) {
            std::cout << "  Col " << i << ": " << visits[i] << " visits\n";
        }
    }
    
    int bestMove = mcts.SelectBestMove();
    std::cout << "\nBest move: " << bestMove << " (should be 3 to block)\n";
    
    // Now let's test what happens if X plays the WRONG move
    Board testBoard = board;
    testBoard.MakeMove(4, Player::PLAYER1); // X plays elsewhere
    testBoard.MakeMove(3, Player::PLAYER2); // O wins
    
    std::cout << "\nIf X plays column 4 instead:\n";
    testBoard.Display();
    std::cout << "Winner: " << (testBoard.CheckWinner() == Player::PLAYER2 ? "O wins!" : "No winner yet") << "\n";
    
    return 0;
}
