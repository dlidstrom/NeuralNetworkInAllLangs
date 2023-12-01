# Licensed under the MIT License given below.
# Copyright 2023 Daniel Lidstrom
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the “Software”), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#CXX=g++
#FLAGS=-std=c++17

#all: cpp

#cpp: main.o neural.o
#	$(CXX) $(FLAGS) -o cpp main.o neural.o

#main.o: main.cpp
#	$(CXX) $(FLAGS) -c main.cpp

#neural.o: neural.cpp
#	$(CXX) $(FLAGS) -c neural.cpp

CXX = clang++
CXX_FLAGS = -std=c++17 -Wfatal-errors -Wall -Wextra -Wpedantic -Wconversion -Wshadow

# Final binary
BIN = cpp

# Put all auto generated stuff to this build dir.
BUILD_DIR = ./build

# List of all .cpp source files.
CPP = main.cpp $(wildcard *.cpp)

# All .o files go to build dir.
OBJ = $(CPP:%.cpp=$(BUILD_DIR)/%.o)

# Gcc/Clang will create these .d files containing dependencies.
DEP = $(OBJ:%.o=%.d)

# Default target named after the binary.
$(BIN) : $(BUILD_DIR)/$(BIN)

# Actual target of the binary - depends on all .o files.
$(BUILD_DIR)/$(BIN) : $(OBJ)
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) $^ -o $@

# Include all .d files
-include $(DEP)

# Build target for every single object file.
# The potential dependency on header files is covered
# by calling `-include $(DEP)`.
$(BUILD_DIR)/%.o : %.cpp
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -MMD -c $< -o $@

.PHONY : clean
clean :
	-rm $(BUILD_DIR)/$(BIN) $(OBJ) $(DEP)
