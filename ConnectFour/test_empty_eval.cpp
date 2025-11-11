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

    Board emptyBoard;

    std::cout << "Testing MCTS evaluation from empty board\n\n";

    for (int sims : {100, 1000, 5000}) {
        MCTS mcts(trainer.network, 1.414);
        mcts.SearchSimulations(emptyBoard, Player::PLAYER1, sims);

        double rootValue = mcts.GetRootValue();
        std::vector<int> visits = mcts.GetVisitCounts();

        int totalVisits = 0;
        for (int v : visits) {
            totalVisits += v;
        }

        std::cout << "Simulations: " << sims << "\n";
        std::cout << "  Root value: " << rootValue << "\n";
        std::cout << "  Total child visits: " << totalVisits << "\n";
        std::cout << "  Expected value: ~0.10 (from 55% win rate)\n\n";
    }

    return 0;
}
