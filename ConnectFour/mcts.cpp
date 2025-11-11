/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#include "mcts.h"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace ConnectFour {

std::vector<double> Softmax(const std::vector<double>& values, double temperature) {
  std::vector<double> result(values.size());

  // Find max for numerical stability
  double maxVal = *std::max_element(values.begin(), values.end());

  // Compute exp(x/T)
  double sum = 0.0;
  for (size_t i = 0; i < values.size(); i++) {
    result[i] = std::exp((values[i] - maxVal) / temperature);
    sum += result[i];
  }

  // Normalize
  for (size_t i = 0; i < values.size(); i++) {
    result[i] /= sum;
  }

  return result;
}

double MCTSNode::GetUCB(double explorationConstant, int parentVisits) const {
  // AlphaZero-style PUCT formula
  double exploitation = GetQValue();

  // Use prior probability to guide exploration
  double exploration = explorationConstant * prior *
                      std::sqrt(parentVisits) / (1.0 + visits);

  return exploitation + exploration;
}

bool MCTSNode::IsFullyExpanded() const {
  if (board.IsGameOver()) {
    return true;
  }

  return !children.empty() && children.size() == board.GetValidMoves().size();
}

MCTS::MCTS(std::unique_ptr<Evaluator> evaluator, double explorationConstant)
    : evaluator(std::move(evaluator)), explorationConstant(explorationConstant) {}

void MCTS::SearchSimulations(const Board& rootBoard, Player rootPlayer, int numSimulations) {
  // Create root node
  root = std::make_unique<MCTSNode>(rootBoard, rootPlayer, -1, nullptr);

  // Expand root immediately
  Expansion(root.get());

  for (int i = 0; i < numSimulations; i++) {
    // Selection
    MCTSNode* node = Selection(root.get());

    // Expansion (if not terminal and visited)
    if (!node->board.IsGameOver() && node->visits > 0) {
      Expansion(node);
      // Select best child according to priors for first simulation
      if (!node->children.empty()) {
        double bestPrior = -1.0;
        MCTSNode* bestChild = node->children[0].get();
        for (const auto& child : node->children) {
          if (child->prior > bestPrior) {
            bestPrior = child->prior;
            bestChild = child.get();
          }
        }
        node = bestChild;
      }
    }

    // Simulation (actually just NN evaluation or terminal check)
    double value = Simulation(node);

    // Backpropagation
    Backpropagation(node, value);
  }
}

void MCTS::SearchTime(const Board& rootBoard, Player rootPlayer, double timeLimit) {
  root = std::make_unique<MCTSNode>(rootBoard, rootPlayer, -1, nullptr);
  Expansion(root.get());

  auto startTime = std::chrono::steady_clock::now();
  int simulations = 0;

  while (true) {
    auto currentTime = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(currentTime - startTime).count();

    if (elapsed >= timeLimit) {
      break;
    }

    // Selection
    MCTSNode* node = Selection(root.get());

    // Expansion
    if (!node->board.IsGameOver() && node->visits > 0) {
      Expansion(node);
      if (!node->children.empty()) {
        double bestPrior = -1.0;
        MCTSNode* bestChild = node->children[0].get();
        for (const auto& child : node->children) {
          if (child->prior > bestPrior) {
            bestPrior = child->prior;
            bestChild = child.get();
          }
        }
        node = bestChild;
      }
    }

    // Simulation
    double value = Simulation(node);

    // Backpropagation
    Backpropagation(node, value);

    simulations++;
  }

  std::cout << "MCTS completed " << simulations << " simulations in " << timeLimit << " seconds\n";
}

MCTSNode* MCTS::Selection(MCTSNode* node) {
  while (node->IsFullyExpanded() && !node->board.IsGameOver()) {
    // Select child with highest UCB
    MCTSNode* bestChild = nullptr;
    double bestUCB = -std::numeric_limits<double>::infinity();

    for (const auto& child : node->children) {
      double ucb = child->GetUCB(explorationConstant, node->visits);
      if (ucb > bestUCB) {
        bestUCB = ucb;
        bestChild = child.get();
      }
    }

    node = bestChild;
  }

  return node;
}

void MCTS::Expansion(MCTSNode* node) {
  if (node->board.IsGameOver()) {
    return;
  }

  #ifdef DEBUG_EXPANSION
  std::cout << "Expanding node for player " << (node->player == Player::PLAYER1 ? "X" : "O") << "\n";
  #endif

  // Get evaluation from evaluator
  auto [priors, value] = evaluator->Evaluate(node->board, node->player);
  node->priorProbabilities = priors;

  // Create child nodes for all valid moves
  std::vector<int> validMoves = node->board.GetValidMoves();

  for (int col : validMoves) {
    Board childBoard = node->board;
    childBoard.MakeMove(col, node->player);

    // Get prior probability for this move
    double prior = priors[col];

    auto child = std::make_unique<MCTSNode>(
        childBoard,
        GetOpponent(node->player),
        col,
        node,
        prior
    );

    node->children.push_back(std::move(child));
  }
}

double MCTS::Simulation(MCTSNode* node) {
  // Check terminal state
  Player winner = node->board.CheckWinner();
  if (winner != Player::NONE) {
    // Game is over
    // Return value from root player's perspective
    MCTSNode* current = node;
    while (current->parent != nullptr) {
      current = current->parent;
    }
    Player rootPlayer = current->player;

    if (winner == rootPlayer) {
      return 1.0; // Win for root player
    } else {
      return -1.0; // Loss for root player
    }
  }

  if (node->board.IsFull()) {
    return 0.0; // Draw
  }

  // For non-terminal positions, use the evaluator's value estimate
  auto [priors, value] = evaluator->Evaluate(node->board, node->player);

  // Get root player for value perspective
  MCTSNode* current = node;
  int depth = 0;
  while (current->parent != nullptr) {
    current = current->parent;
    depth++;
  }
  Player rootPlayer = current->player;

  // Value is from node->player perspective, convert to root perspective
  // Flip value for each level up the tree
  if (depth % 2 == 1) {
    value = -value;
  }

  return value;
}

void MCTS::Backpropagation(MCTSNode* node, double value) {
  while (node != nullptr) {
    node->visits++;
    node->totalValue += value;

    // Flip value for parent (opponent's perspective)
    value = -value;
    node = node->parent;
  }
}

std::vector<int> MCTS::GetVisitCounts() const {
  std::vector<int> counts(COLS, 0);

  if (!root) {
    return counts;
  }

  for (const auto& child : root->children) {
    counts[child->move] = child->visits;
  }

  return counts;
}

std::vector<double> MCTS::GetMoveProbabilities() const {
  std::vector<int> counts = GetVisitCounts();
  std::vector<double> probs(COLS, 0.0);

  int totalVisits = 0;
  for (int count : counts) {
    totalVisits += count;
  }

  if (totalVisits > 0) {
    for (int i = 0; i < COLS; i++) {
      probs[i] = static_cast<double>(counts[i]) / totalVisits;
    }
  }

  return probs;
}

int MCTS::SelectBestMove() const {
  if (!root || root->children.empty()) {
    return -1;
  }

  int bestMove = -1;
  int maxVisits = -1;

  for (const auto& child : root->children) {
    if (child->visits > maxVisits) {
      maxVisits = child->visits;
      bestMove = child->move;
    }
  }

  return bestMove;
}

int MCTS::SelectMoveSoftmax(double temperature, std::mt19937& rng) {
  std::vector<int> counts = GetVisitCounts();

  // Convert to doubles for softmax
  std::vector<double> values(COLS);
  for (int i = 0; i < COLS; i++) {
    values[i] = static_cast<double>(counts[i]);
  }

  std::vector<double> probs = Softmax(values, temperature);

  // Sample from distribution
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return dist(rng);
}

double MCTS::GetRootValue() const {
  if (!root) {
    return 0.0;
  }
  return root->GetQValue();
}

} // namespace ConnectFour
