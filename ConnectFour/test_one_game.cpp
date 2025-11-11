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
    Player mctsPlayer = Player::PLAYER1;  // MCTS goes first
    Player currentPlayer = Player::PLAYER1;

    std::cout << "Playing one game: MCTS (X) vs Minimax depth 2 (O)\n\n";

    int moveNum = 0;
    const int MCTS_SIMS = 5000;

    while (!board.IsGameOver() && moveNum < 42) {
        std::cout << "Move " << (moveNum + 1) << " - Player " 
                  << (currentPlayer == Player::PLAYER1 ? "X" : "O") << "\n";

        int move;

        if (currentPlayer == mctsPlayer) {
            MCTS mcts(trainer.network, 1.414);
            mcts.SearchSimulations(board, currentPlayer, MCTS_SIMS);
            
            std::vector<int> visits = mcts.GetVisitCounts();
            std::cout << "  MCTS visit counts: ";
            for (int i = 0; i < COLS; i++) {
                if (visits[i] > 0) {
                    std::cout << i << ":" << visits[i] << " ";
                }
            }
            std::cout << "\n";
            std::cout << "  Root value: " << mcts.GetRootValue() << "\n";
            
            move = mcts.SelectBestMove();
        } else {
            move = minimax.SelectMove(board, currentPlayer);
            std::cout << "  Minimax chose: " << move << "\n";
        }

        std::cout << "  Playing column " << move << "\n";
        board.MakeMove(move, currentPlayer);
        board.Display();
        std::cout << "\n";

        currentPlayer = (currentPlayer == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
        moveNum++;
    }

    Player winner = board.CheckWinner();

    if (winner == Player::NONE) {
        std::cout << "Game ended in a DRAW\n";
    } else if (winner == mctsPlayer) {
        std::cout << "MCTS WINS!\n";
    } else {
        std::cout << "Minimax WINS!\n";
    }

    return 0;
}
