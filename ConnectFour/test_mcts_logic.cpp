#include "mcts.h"
#include "../Cpp/neural.h"
#include <iostream>
#include <random>

using namespace ConnectFour;

int main() {
    // Create simple position where X can win immediately in column 3
    Board board;
    board.MakeMove(0, Player::PLAYER1); // X
    board.MakeMove(0, Player::PLAYER2); // O on top
    board.MakeMove(1, Player::PLAYER1); // X
    board.MakeMove(1, Player::PLAYER2); // O on top
    board.MakeMove(2, Player::PLAYER1); // X
    board.MakeMove(2, Player::PLAYER2); // O on top
    // Now column 3 is a winning move for X
    
    std::cout << "Board state (X can win in column 3):\n";
    board.Display();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    auto rand = [&]() { return dis(gen); };
    Neural::Trainer trainer = Neural::Trainer::Create(BOARD_SIZE * 3, 256, COLS, rand);
    
    std::cout << "\nRunning MCTS with 400 simulations...\n";
    MCTS mcts(trainer.network, 1.414);
    mcts.SearchSimulations(board, Player::PLAYER1, 400);
    
    std::vector<int> visits = mcts.GetVisitCounts();
    std::vector<double> probs = mcts.GetMoveProbabilities();
    
    std::cout << "\nResults:\n";
    for (int i = 0; i < COLS; i++) {
        if (board.IsValidMove(i)) {
            std::cout << "  Col " << i << ": " << visits[i] << " visits, "
                      << (probs[i] * 100) << "% probability\n";
        }
    }
    
    int bestMove = mcts.SelectBestMove();
    std::cout << "\nBest move selected: " << bestMove;
    std::cout << " (should be 3)\n";
    
    if (bestMove == 3) {
        std::cout << "SUCCESS: MCTS found the winning move!\n";
    } else {
        std::cout << "FAILURE: MCTS missed the obvious win!\n";
    }
    
    return 0;
}
