#include "mcts.h"
#include "minimax.h"
#include "../Cpp/neural.h"
#include <iostream>
#include <random>

using namespace ConnectFour;

int PlayOut(Board board, Player startPlayer, int firstMove, MinimaxAI& minimax) {
    board.MakeMove(firstMove, startPlayer);
    Player currentPlayer = (startPlayer == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
    
    int moves = 1;
    while (!board.IsGameOver() && moves < 20) {
        int move = minimax.SelectMove(board, currentPlayer);
        if (move < 0) break;
        board.MakeMove(move, currentPlayer);
        currentPlayer = (currentPlayer == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
        moves++;
    }
    
    Player winner = board.CheckWinner();
    if (winner == startPlayer) return 1;
    if (winner == Player::NONE) return 0;
    return -1;
}

int main() {
    Board board;
    board.MakeMove(3, Player::PLAYER1);
    board.MakeMove(2, Player::PLAYER2);
    board.MakeMove(3, Player::PLAYER1);
    board.MakeMove(4, Player::PLAYER2);
    
    MinimaxAI minimax(4); // Deeper minimax to resolve
    
    std::cout << "Testing different first moves for X:\n\n";
    
    for (int col : {2, 4}) {
        if (!board.IsValidMove(col)) continue;
        
        std::cout << "If X plays column " << col << ":\n";
        int result = PlayOut(board, Player::PLAYER1, col, minimax);
        
        if (result == 1) {
            std::cout << "  X wins with best play\n";
        } else if (result == -1) {
            std::cout << "  O wins with best play\n";
        } else {
            std::cout << "  Draw with best play\n";
        }
    }
    
    return 0;
}
