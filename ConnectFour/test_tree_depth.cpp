#include "mcts.h"
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
    
    Board board;
    MCTS mcts(trainer.network, 1.414);
    
    std::cout << "Running MCTS with 1600 simulations...\n";
    mcts.SearchSimulations(board, Player::PLAYER1, 1600);
    
    std::vector<int> visits = mcts.GetVisitCounts();
    std::cout << "\nRoot children visit counts:\n";
    int total = 0;
    for (int i = 0; i < COLS; i++) {
        std::cout << "  Col " << i << ": " << visits[i] << "\n";
        total += visits[i];
    }
    std::cout << "Total: " << total << " (should be ~1600)\n";
    
    return 0;
}
