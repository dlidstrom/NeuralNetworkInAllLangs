# Connect Four AI Game

A Connect Four game implementation with a neural network AI that learns through self-play reinforcement learning.

## Features

- **7x6 Connect Four board** - More strategic than traditional 3x3 tic-tac-toe
- **Neural network AI** - Uses the existing neural network implementation from this repository
- **Training mode** - Self-play reinforcement learning to improve the AI
- **Game mode** - Play against the trained AI
- **Watch mode** - Observe the AI play against itself
- **Save/Load weights** - Persist trained networks between sessions

## Building

```bash
make game
```

This will create the executable at `build/connect_four_game`.

## Usage

Run the game:

```bash
./build/connect_four_game
```

### Menu Options

1. **Train new network** - Start training from scratch
   - Creates a new random neural network
   - Trains through self-play
   - Saves weights to `connect_four_weights.bin`

2. **Continue training existing network** - Resume training
   - Loads existing weights from `connect_four_weights.bin`
   - Continues training with additional games
   - Updates saved weights

3. **Play against AI** - Interactive game mode
   - Loads trained AI from `connect_four_weights.bin`
   - Choose to play as X or O
   - Enter column numbers (0-6) to make moves

4. **Watch AI play against itself** - Demonstration mode
   - Observe the trained AI's strategy
   - Press Enter to advance moves
   - See how the network plays both sides

5. **Exit** - Quit the program

## Training Recommendations

### Initial Training

For your first training session:

- **Games**: Start with 1000-5000 games
- **Learning rate**: Try 0.01 or 0.005
- Training takes a few minutes depending on game count

Example training session:

```txt
Choose option: 1
Enter number of training games: 2000
Enter learning rate (e.g., 0.01): 0.01
```

### Continued Training

After initial training, you can improve the AI:

- **Games**: Add 1000-2000 more games
- **Learning rate**: Use same or slightly lower (0.005)
- The AI improves gradually with more training

### Training Tips

- More games = better play, but diminishing returns after ~10,000 games
- Lower learning rates (0.001-0.01) work better than high rates
- The AI learns by playing against itself, so it discovers strategies over time
- Training statistics show Player 1 vs Player 2 wins - balanced is good!

## How It Works

### Neural Network Architecture

- **Input**: 126 neurons (7x6 board × 3 values per cell)
  - Each cell encoded as: [my piece, opponent piece, empty]
- **Hidden**: 128 neurons with sigmoid activation
- **Output**: 7 neurons (one per column)
  - Higher values = better moves

### Training Algorithm

- **Self-play**: Network plays against itself
- **Exploration**: 20% random moves during training
- **Reward signal**: +1 for wins, -1 for losses, 0 for draws
- **Temporal credit**: Rewards discounted by 0.95 per move
- **Backpropagation**: Updates weights based on game outcomes

### Game Rules

- Players alternate dropping pieces in columns
- Pieces fall to the lowest available row
- Win by connecting 4 pieces horizontally, vertically, or diagonally
- Game ends in draw if board fills without a winner

## File Structure

- `connect_four.h/cpp` - Game board logic and rules
- `ai_player.h/cpp` - AI player using neural network
- `trainer_game.h/cpp` - Self-play training system
- `neural_io.h/cpp` - Network save/load functionality
- `connect_four_game.cpp` - Main program with menu
- `neural.h/cpp` - Core neural network (from parent repo)

## Saved Data

- `connect_four_weights.bin` - Trained network weights (binary format)
- Created automatically after training
- Can be copied/shared to preserve trained models

## Technical Details

The neural network uses:

- Feedforward architecture with one hidden layer
- Sigmoid activation functions
- Gradient descent with backpropagation
- Policy-based reinforcement learning

Connect Four state space:

- ~4.5 trillion possible positions
- Average game length: 36 moves
- First player advantage exists (but small with good play)

## Tips for Playing

- The AI's strength depends entirely on training
- With 5000+ training games, it plays quite well
- Center columns (3,4) are strategically important
- Watch mode helps understand AI's strategy
- Don't expect perfect play - it's a learning AI!

## Cleaning Up

To remove built files and saved weights:

```bash
make clean
```

This removes:

- Build directory and object files
- The compiled executable
- Saved weight files (*.bin)
