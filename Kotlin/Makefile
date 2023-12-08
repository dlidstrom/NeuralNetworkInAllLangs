# Makefile for Kotlin project

# Environment variables
KOTLIN_COMPILER = kotlinc
KOTLIN_FLAGS = -include-runtime -d
JAVA_RUNTIME = java

##
# source directory
##
SRC_DIR := src

##
# output directory
##
OUT_DIR := out

##
# sources
##
SRCS := $(wildcard $(SRC_DIR)/*.kt)

# Default target
all: build

# Clean build artifacts
clean:
	rm -rf build

# Build the project
build: $(SRCS)
	$(KOTLIN_COMPILER) $(KOTLIN_FLAGS) build/neural.jar src/*.kt

# Run the project
run:
	$(JAVA_RUNTIME) -jar build/neural.jar
