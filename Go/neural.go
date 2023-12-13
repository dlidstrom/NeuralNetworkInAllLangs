/*
Licensed under the MIT License given below.
Copyright 2023 Daniel Lidstrom
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

package main

import (
	"fmt"
	"math"
)

type Network struct {
	inputCount    uint32
	hiddenCount   uint32
	outputCount   uint32
	weightsHidden []float64
	biasesHidden  []float64
	weightsOutput []float64
	biasesOutput  []float64
}

func sigmoid(d float64) float64 {
	return 1.0 / (1.0 + math.Exp(-d))
}

func sigmoid_prim(d float64) float64 {
	return d * (1.0 - d)
}

func (network Network) Predict(input []float64) []float64 {
	hidden := make([]float64, network.hiddenCount)
	output := make([]float64, network.outputCount)
	return network.PredictInplace(input, hidden, output)
}

func (network Network) PredictInplace(input []float64, hidden []float64, output []float64) ([]float64) {
	for c := uint32(0); c < network.hiddenCount; c++ {
		sum := 0.0
		for r := uint32(0); r < network.inputCount; r++ {
			sum += input[r] * network.weightsHidden[r * network.hiddenCount + c]
		}

		hidden[c] = sigmoid(sum + network.biasesHidden[c])
	}

	for c := uint32(0); c < network.outputCount; c++ {
		sum := 0.0
		for r := uint32(0); r < network.hiddenCount; r++ {
			sum += hidden[r] * network.weightsOutput[r * network.outputCount + c]
		}

		output[c] = sigmoid(sum + network.biasesOutput[c])
	}

	return output
}

func (network Network) Print() {
	fmt.Println("weightsHidden:")
	for i := 0; i < len(network.weightsHidden); i++ {
		fmt.Printf("%.6f ", network.weightsHidden[i])
	}

	fmt.Println()
	fmt.Println("biasesHidden:")
	for i := 0; i < len(network.biasesHidden); i++ {
		fmt.Printf("%.6f ", network.biasesHidden[i])
	}

	fmt.Println()
	fmt.Println("weightsOutput:")
	for i := 0; i < len(network.weightsOutput); i++ {
		fmt.Printf("%.6f ", network.weightsOutput[i])
	}

	fmt.Println()
	fmt.Println("biasesOutput:")
	for i := 0; i < len(network.biasesOutput); i++ {
		fmt.Printf("%.6f ", network.biasesOutput[i])
	}

	fmt.Println()
}

type Trainer struct {
	network Network
	hidden []float64
	output []float64
	gradHidden []float64
	gradOutput []float64
}

type RandomGenerator func() float64

func NewTrainer(
	inputCount uint32,
	hiddenCount uint32,
	outputCount uint32,
	rand RandomGenerator,
) (*Trainer) {
	weightsHidden := make([]float64, inputCount * hiddenCount)
	for i := 0; i < len(weightsHidden); i++ { weightsHidden[i] = rand() - 0.5 }
	biasesHidden := make([]float64, hiddenCount)
	weightsOutput := make([]float64, hiddenCount * outputCount)
	for i := 0; i < len(weightsOutput); i++ { weightsOutput[i] = rand() - 0.5 }
	biasesOutput := make([]float64, outputCount)
	network := Network { inputCount, hiddenCount, outputCount, weightsHidden, biasesHidden, weightsOutput, biasesOutput }
	hidden := make([]float64, hiddenCount)
	output := make([]float64, outputCount)
	gradHidden := make([]float64, hiddenCount)
	gradOutput := make([]float64, outputCount)
	return &Trainer { network, hidden, output, gradHidden, gradOutput }
}

func (trainer *Trainer) Train(input []float64, y []float64, lr float64) {
	trainer.network.PredictInplace(input, trainer.hidden, trainer.output)
	for c := uint32(0); c < trainer.network.outputCount; c++ {
		trainer.gradOutput[c] = (trainer.output[c] - y[c]) * sigmoid_prim(trainer.output[c])
	}

	for r := uint32(0); r < trainer.network.hiddenCount; r++ {
		sum := 0.0
		for c := uint32(0); c < trainer.network.outputCount; c++ {
			sum += trainer.gradOutput[c] * trainer.network.weightsOutput[r * trainer.network.outputCount + c]
		}

		trainer.gradHidden[r] = sum * sigmoid_prim(trainer.hidden[r])
	}

	for r := uint32(0); r < trainer.network.hiddenCount; r++ {
		for c := uint32(0); c < trainer.network.outputCount; c++ {
			trainer.network.weightsOutput[r * trainer.network.outputCount + c] -= lr * trainer.gradOutput[c] * trainer.hidden[r]
		}
	}

	for r := uint32(0); r < trainer.network.inputCount; r++ {
		for c := uint32(0); c < trainer.network.hiddenCount; c++ {
			trainer.network.weightsHidden[r * trainer.network.hiddenCount + c] -= lr * trainer.gradHidden[c] * input[r]
		}
	}

	for c := uint32(0); c < trainer.network.outputCount; c++ {
		trainer.network.biasesOutput[c] -= lr * trainer.gradOutput[c]
	}

	for c := uint32(0); c < trainer.network.hiddenCount; c++ {
		trainer.network.biasesHidden[c] -= lr * trainer.gradHidden[c]
	}
}
