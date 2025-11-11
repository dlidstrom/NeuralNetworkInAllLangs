# Connect Four with MCTS and Neural Network

A sophisticated Connect Four AI that learns through self-play using Monte Carlo Tree Search (MCTS) combined with a neural network.

## Features

### 1. **Board State Normalization & Mirroring**

- Always normalizes board from current player's perspective
- Implements horizontal mirroring to reduce state space by ~2x
- Chooses lexicographically smaller representation for consistent learning

### 2. **Minimax AI Benchmark**

- Traditional alpha-beta pruning minimax search
- Configurable depth (default: 7 plies)
- Heuristic evaluation function based on threats and center control
- Used as a baseline opponent for evaluation

### 3. **MCTS with Neural Network**

- Monte Carlo Tree Search guided by neural network predictions
- Neural network predicts move probabilities for each position
- Two search modes:
  - **Simulation-based**: Fixed number of simulations (800 for training)
  - **Time-based**: Search for specified seconds (2s for human play)
- Softmax temperature-based move selection for exploration

### 4. **Self-Play Training**

- Generates training data through self-play using MCTS
- Stores (state, MCTS policy, outcome) tuples
- Temperature schedule: higher early game, lower late game
- Training targets combine MCTS visit distributions with game outcomes
- Periodic evaluation against minimax to track progress

### 5. **Game Modes**

1. **Train new network**: Start fresh with random weights
2. **Continue training**: Load and improve existing network
3. **Play against AI**: Human vs AI with full MCTS search
4. **Watch AI vs Minimax**: Observe strategy differences
5. **Evaluate**: Benchmark AI performance against minimax

## Architecture

### Neural Network

- **Input**: 126 values (42 cells � 3 features per cell)
  - One-hot encoding: [my pieces, opponent pieces, empty]
- **Hidden layer**: 256 neurons with sigmoid activation
- **Output**: 7 values (move probabilities for each column)

### MCTS Algorithm

1. **Selection**: UCB-based tree traversal
2. **Expansion**: Create child nodes for legal moves
3. **Simulation**: Neural network evaluation (no rollouts)
4. **Backpropagation**: Update visit counts and values

### Training Process

- Each iteration:
  1. Play N self-play games with MCTS
  2. Collect training examples
  3. Train neural network on examples
  4. Periodically evaluate vs minimax
- Learning rate: 0.001 (recommended)
- MCTS simulations: 800 per move (training)

## Building

```bash
make
```

This will compile all source files and create the `connectfour` executable.

## Usage

Run the program:

```bash
./connectfour
```

Or use:

```bash
make run
```

### Training Example

1. Choose option 1 (Train new network)
2. Enter training parameters:
   - Iterations: 10
   - Games per iteration: 20
   - Evaluation frequency: 2 (evaluate every 2 iterations)
   - Learning rate: 0.001

The system will:

- Play 20 self-play games per iteration
- Train on ~400-800 examples per iteration
- Evaluate against minimax every 2 iterations
- Save the trained network to `connectfour_mcts_weights.bin`

### Playing Against the AI

1. Train or load a network first
2. Choose option 3 (Play against AI)
3. Decide who goes first
4. Enter column numbers (0-6) to play
5. AI will think for 2 seconds using MCTS

## Files

- `board.h/cpp`: Board representation with normalization and mirroring
- `minimax.h/cpp`: Minimax AI with alpha-beta pruning
- `mcts.h/cpp`: MCTS implementation with neural network integration
- `trainer.h/cpp`: Self-play training loop with evaluation
- `main.cpp`: Main program with user interface
- `Makefile`: Build configuration

## Dependencies

- C++17 compiler (g++ or clang++)
- Neural network implementation from `../Cpp/neural.h`
- No external libraries required

## Training Tips

1. **Start small**: Train for 5-10 iterations first to verify it works
2. **Learning rate**: Start with 0.001, decrease if training is unstable
3. **Evaluation**: Check win rate vs minimax depth 6
4. **Iterations**: Plan for 50-100+ iterations for strong play
5. **Time**: Each iteration with 20 games takes ~5-10 minutes

## Performance Expectations

After training:

- **10 iterations**: Should beat random play
- **25 iterations**: ~20-30% win rate vs minimax depth 6
- **50 iterations**: ~40-50% win rate vs minimax depth 6
- **100+ iterations**: Can match or exceed minimax depth 6

## Technical Details

### State Space Reduction

The mirroring technique reduces the effective state space:

- Without mirroring: ~4.5 � 10^12 possible positions
- With mirroring: ~2.25 � 10^12 possible positions
- Normalized perspective: Always from current player's view

### MCTS Enhancements

- UCB exploration constant: 1.414 (2)
- Temperature schedule for move selection
- Visit count-based training targets
- Value estimation from game outcomes

### Minimax Configuration

- Default depth: 7 plies
- Evaluation features:
  - Three-in-a-row: +100 per occurrence
  - Two-in-a-row: +10 per occurrence
  - Center control: +3 per piece
  - Terminal states: �10000

## Cleaning

Remove compiled files:

```bash
make clean
```

Remove trained weights only:

```bash
make clean-weights
```

## License

Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
