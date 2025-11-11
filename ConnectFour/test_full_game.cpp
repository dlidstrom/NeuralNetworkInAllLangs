#include "mcts.h"
#include "minimax.h"
#include "../Cpp/neural.h"
#include <iostream>
#include <random>

using namespace ConnectFour;

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    auto rand = [&]() { return dis(gen); };
    Neural::Trainer trainer = Neural::Trainer::Create(BOARD_SIZE * 3, 256, COLS, rand);
    
    MinimaxAI minimax(2);
    Board board;
    Player mctsPlayer = Player::PLAYER1;
    Player currentPlayer = Player::PLAYER1;
    
    std::cout << "Game: MCTS (5000 sims) vs Minimax depth 2\n\n";
    
    int move = 0;
    while (!board.IsGameOver() && move < 42) {
        board.Display();
        
        int col;
        if (currentPlayer == mctsPlayer) {
            MCTS mcts(trainer.network, 1.414);
            mcts.SearchSimulations(board, currentPlayer, 5000);
            col = mcts.SelectBestMove();
            double value = mcts.GetRootValue();
            std::cout << "MCTS plays " << col << " (eval: " << value << ")\n";
        } else {
            col = minimax.SelectMove(board, currentPlayer);
            std::cout << "Minimax plays " << col << "\n";
        }
        
        board.MakeMove(col, currentPlayer);
        currentPlayer = (currentPlayer == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
        move++;
        
        if (move > 10) break; // Just show first 10 moves
    }
    
    board.Display();
    std::cout << "\n(Stopped after 10 moves for analysis)\n";
    
    return 0;
}
