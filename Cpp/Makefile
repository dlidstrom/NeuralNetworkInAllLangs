# Licensed under the MIT License.
# Copyright 2023-2025 Daniel Lidstrom

CXX = clang++
CXX_FLAGS = -std=c++17 -Wfatal-errors -Wall -Wextra -Wpedantic -Wconversion -Wshadow

# Final binary
BIN = cpp
GAME_BIN = connect_four_game

# Put all auto generated stuff to this build dir.
BUILD_DIR = ./build

# List of all .cpp source files.
CPP = main.cpp neural.cpp
GAME_CPP = neural.cpp neural_io.cpp board.cpp ai_player.cpp game_trainer.cpp connect_four_game.cpp

# All .o files go to build dir.
OBJ = $(CPP:%.cpp=$(BUILD_DIR)/%.o)
GAME_OBJ = $(GAME_CPP:%.cpp=$(BUILD_DIR)/%.o)

# Gcc/Clang will create these .d files containing dependencies.
DEP = $(OBJ:%.o=%.d)
GAME_DEP = $(GAME_OBJ:%.o=%.d)

# Default target named after the binary.
$(BIN) : $(BUILD_DIR)/$(BIN)

# Game target
game: $(BUILD_DIR)/$(GAME_BIN)

# Actual target of the binary - depends on all .o files.
$(BUILD_DIR)/$(BIN) : $(OBJ)
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) $^ -o $@

# Game binary target
$(BUILD_DIR)/$(GAME_BIN) : $(GAME_OBJ)
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) $^ -o $@

# Include all .d files
-include $(DEP)
-include $(GAME_DEP)

# Build target for every single object file.
# The potential dependency on header files is covered
# by calling `-include $(DEP)`.
$(BUILD_DIR)/%.o : %.cpp
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -MMD -c $< -o $@

.PHONY : clean
clean :
	-rm $(BUILD_DIR)/$(BIN) $(BUILD_DIR)/$(GAME_BIN) $(OBJ) $(GAME_OBJ) $(DEP) $(GAME_DEP) *.bin
