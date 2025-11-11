#include "mcts.h"
#include "minimax.h"
#include "../Cpp/neural.h"
#include <iostream>
#include <random>

using namespace ConnectFour;

int main() {
    // Create a position where the right move requires 2-ply thinking
    Board board;
    // Let's create a position where one move leads to a forced win next turn
    board.MakeMove(3, Player::PLAYER1); // X in center
    board.MakeMove(2, Player::PLAYER2); // O
    board.MakeMove(3, Player::PLAYER1); // X on top in center
    board.MakeMove(4, Player::PLAYER2); // O
    
    std::cout << "Board state:\n";
    board.Display();
    std::cout << "\nPlayer X to move\n";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    auto rand = [&]() { return dis(gen); };
    Neural::Trainer trainer = Neural::Trainer::Create(BOARD_SIZE * 3, 256, COLS, rand);
    
    // What does minimax think?
    MinimaxAI minimax(2);
    int minimaxMove = minimax.SelectMove(board, Player::PLAYER1);
    std::cout << "\nMinimax depth 2 selects: column " << minimaxMove << "\n";
    
    // What does MCTS think with various simulation counts?
    for (int sims : {100, 500, 2000, 5000}) {
        MCTS mcts(trainer.network, 1.414);
        mcts.SearchSimulations(board, Player::PLAYER1, sims);
        
        int mctsMove = mcts.SelectBestMove();
        std::vector<int> visits = mcts.GetVisitCounts();
        
        std::cout << "\nMCTS with " << sims << " sims selects: column " << mctsMove << "\n";
        std::cout << "  Visit distribution: ";
        for (int i = 0; i < COLS; i++) {
            if (visits[i] > 0) std::cout << i << ":" << visits[i] << " ";
        }
        std::cout << "\n";
        
        if (mctsMove == minimaxMove) {
            std::cout << "  ✓ Matches minimax\n";
        } else {
            std::cout << "  ✗ Disagrees with minimax\n";
        }
    }
    
    return 0;
}
