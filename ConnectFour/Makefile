# Makefile for Connect Four MCTS AI
# Licensed under the MIT License.
# Copyright 2025 Daniel Lidstrom

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -I..
LDFLAGS =

# Source files
SOURCES = main.cpp board.cpp minimax.cpp mcts.cpp trainer.cpp \
          ../Cpp/neural.cpp ../Cpp/neural_io.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Executable
TARGET = connectfour

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJECTS) $(TARGET) connectfour_mcts_weights.bin

# Clean weights only
clean-weights:
	rm -f connectfour_mcts_weights.bin

# Run
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean clean-weights run
