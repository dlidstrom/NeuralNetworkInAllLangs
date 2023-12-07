# Makefile for Kotlin project

# Environment variables
KOTLIN_COMPILER = kotlinc
KOTLIN_FLAGS = -include-runtime -d
JAVA_RUNTIME = java

# Default target
all: build

# Clean build artifacts
clean:
	rm -rf build

# Build the project
build:
	$(KOTLIN_COMPILER) $(KOTLIN_FLAGS) build/neural.jar src/*.kt

# Run the project
run:
	$(JAVA_RUNTIME) -jar build/neural.jar
