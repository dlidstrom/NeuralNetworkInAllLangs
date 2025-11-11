/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#ifndef CONNECTFOUR_MCTS_H
#define CONNECTFOUR_MCTS_H

#include "board.h"
#include "../Cpp/neural.h"
#include <memory>
#include <vector>
#include <map>
#include <random>

namespace ConnectFour {

// MCTS Node
struct MCTSNode {
  Board board;
  Player player;
  int move; // Move that led to this node (-1 for root)
  bool wasMirrored; // Whether the board was mirrored for NN input
  double prior; // Prior probability for this move from parent's NN evaluation

  int visits;
  double totalValue;
  std::vector<std::unique_ptr<MCTSNode>> children;
  MCTSNode* parent;

  // Prior probability from neural network
  std::vector<double> priorProbabilities;

  MCTSNode(const Board& b, Player p, int m, MCTSNode* par, double pr = 1.0)
      : board(b), player(p), move(m), wasMirrored(false), prior(pr),
        visits(0), totalValue(0.0), parent(par) {}

  double GetQValue() const {
    if (visits == 0) return 0.0;
    return totalValue / visits;
  }

  double GetUCB(double explorationConstant, int parentVisits) const;
  bool IsLeaf() const { return children.empty(); }
  bool IsFullyExpanded() const;
};

class MCTS {
public:
  MCTS(Neural::Network network, double explorationConstant = 1.414);

  // Search for a number of simulations
  void SearchSimulations(const Board& rootBoard, Player rootPlayer, int numSimulations);

  // Search for a time limit (in seconds)
  void SearchTime(const Board& rootBoard, Player rootPlayer, double timeLimit);

  // Get visit counts for each move (for training)
  std::vector<int> GetVisitCounts() const;

  // Get move probabilities based on visit counts
  std::vector<double> GetMoveProbabilities() const;

  // Select best move (highest visit count)
  int SelectBestMove() const;

  // Select move using softmax with temperature
  int SelectMoveSoftmax(double temperature, std::mt19937& rng);

  // Get value estimate of root position
  double GetRootValue() const;

private:
  Neural::Network network;
  double explorationConstant;
  std::unique_ptr<MCTSNode> root;

  // MCTS phases
  MCTSNode* Selection(MCTSNode* node);
  void Expansion(MCTSNode* node);
  double Simulation(MCTSNode* node);
  void Backpropagation(MCTSNode* node, double value);

  // Get neural network evaluation
  std::pair<std::vector<double>, double> EvaluatePosition(const Board& board, Player player);

  // Helper to get opponent
  static Player GetOpponent(Player player) {
    return (player == Player::PLAYER1) ? Player::PLAYER2 : Player::PLAYER1;
  }
};

// Softmax function
std::vector<double> Softmax(const std::vector<double>& values, double temperature = 1.0);

} // namespace ConnectFour

#endif // CONNECTFOUR_MCTS_H
