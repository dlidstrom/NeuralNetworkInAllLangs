#include "board.h"
#include <iostream>
#include <random>

using namespace ConnectFour;

inline Player GetOpponent(Player p) {
    return (p == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
}

// Forward declaration
void TestSubtlePosition();

// Run N random rollouts from a position and return win rate
double TestRolloutQuality(const Board& startBoard, Player player, int numRollouts) {
    std::mt19937 rng(std::random_device{}());
    int wins = 0;
    int losses = 0;
    int draws = 0;

    for (int i = 0; i < numRollouts; i++) {
        Board rolloutBoard = startBoard;
        Player currentPlayer = player;

        while (!rolloutBoard.IsGameOver()) {
            std::vector<int> validMoves = rolloutBoard.GetValidMoves();
            if (validMoves.empty()) break;

            std::uniform_int_distribution<> dist(0, validMoves.size() - 1);
            int move = validMoves[dist(rng)];

            rolloutBoard.MakeMove(move, currentPlayer);
            currentPlayer = GetOpponent(currentPlayer);
        }

        Player winner = rolloutBoard.CheckWinner();
        if (winner == player) {
            wins++;
        } else if (winner == GetOpponent(player)) {
            losses++;
        } else {
            draws++;
        }
    }

    double winRate = (wins + 0.5 * draws) / numRollouts;
    return winRate;
}

int main() {
    std::cout << "Testing random rollout quality in Connect Four\n\n";

    // Test 1: Empty board (neutral position)
    Board emptyBoard;
    double emptyWinRate = TestRolloutQuality(emptyBoard, Player::PLAYER1, 10000);
    std::cout << "Empty board win rate for X: " << (emptyWinRate * 100) << "%\n";

    // Test 2: Position where X has clear advantage (3 in a row with open ends)
    Board advantageBoard;
    advantageBoard.MakeMove(2, Player::PLAYER1);
    advantageBoard.MakeMove(0, Player::PLAYER2);
    advantageBoard.MakeMove(3, Player::PLAYER1);
    advantageBoard.MakeMove(1, Player::PLAYER2);
    advantageBoard.MakeMove(4, Player::PLAYER1);
    // X has three in a row at bottom: _ _ X X X _ _
    // O has pieces at 0 and 1
    
    std::cout << "\nPosition with X having 3 in a row:\n";
    advantageBoard.Display();
    
    double advantageWinRate = TestRolloutQuality(advantageBoard, Player::PLAYER1, 10000);
    std::cout << "Win rate for X with advantage: " << (advantageWinRate * 100) << "%\n";

    // Test 3: Position where X is about to lose (O has 3 in a row)
    Board disadvantageBoard;
    disadvantageBoard.MakeMove(0, Player::PLAYER2);
    disadvantageBoard.MakeMove(0, Player::PLAYER1);
    disadvantageBoard.MakeMove(1, Player::PLAYER2);
    disadvantageBoard.MakeMove(1, Player::PLAYER1);
    disadvantageBoard.MakeMove(2, Player::PLAYER2);
    // O has three in a row: O O O _ _ _ _
    // X has pieces blocking on top
    
    std::cout << "\nPosition with O having 3 in a row (X's turn):\n";
    disadvantageBoard.Display();
    
    double disadvantageWinRate = TestRolloutQuality(disadvantageBoard, Player::PLAYER1, 10000);
    std::cout << "Win rate for X when threatened: " << (disadvantageWinRate * 100) << "%\n";

    std::cout << "\n=== Analysis ===\n";
    std::cout << "Empty board should be ~50% (neutral)\n";
    std::cout << "Advantage position should be significantly >50%\n";
    std::cout << "Disadvantage position should be significantly <50%\n";
    std::cout << "\nIf random rollouts are high quality, we'd see large differences.\n";
    std::cout << "If random rollouts are noisy, all positions will look similar (~50%).\n";

    TestSubtlePosition();

    return 0;
}

void TestSubtlePosition() {
    std::cout << "\n=== Testing Subtle Positional Advantages ===\n";
    
    // Position 1: X controls center (generally good in Connect Four)
    Board centerControl;
    centerControl.MakeMove(3, Player::PLAYER1); // X center
    centerControl.MakeMove(0, Player::PLAYER2); // O side
    centerControl.MakeMove(3, Player::PLAYER1); // X center again
    centerControl.MakeMove(6, Player::PLAYER2); // O far side
    
    std::cout << "\nX controls center:\n";
    centerControl.Display();
    double centerWinRate = TestRolloutQuality(centerControl, Player::PLAYER1, 10000);
    std::cout << "Win rate for X: " << (centerWinRate * 100) << "%\n";
    
    // Position 2: After just 1 move each
    Board earlyGame;
    earlyGame.MakeMove(3, Player::PLAYER1); // X plays center
    earlyGame.MakeMove(5, Player::PLAYER2); // O plays off-center
    
    std::cout << "\nEarly game (X played center, O played col 5):\n";
    earlyGame.Display();
    double earlyWinRate = TestRolloutQuality(earlyGame, Player::PLAYER1, 10000);
    std::cout << "Win rate for X: " << (earlyWinRate * 100) << "%\n";
    
    std::cout << "\nConclusion: If these subtle positions show little difference from 50%,\n";
    std::cout << "then random rollouts can't distinguish good from bad until threats appear.\n";
}
