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

import kotlin.math.exp

fun interface Random {
  fun generate(): Double
}

class Network(
  val inputCount: Int,
  val hiddenCount: Int,
  val outputCount: Int,
  val weightsHidden: DoubleArray,
  val biasesHidden: DoubleArray,
  val weightsOutput: DoubleArray,
  val biasesOutput: DoubleArray) {
  companion object {
    fun sigmoid(f: Double): Double = 1.0 / (1.0 + exp(-f))
  }

  fun predict(input: DoubleArray): DoubleArray {
    val hidden = DoubleArray(hiddenCount)
    val output = DoubleArray(outputCount)
    return predict(input, hidden, output)
  }

  fun predict(input: DoubleArray, hidden: DoubleArray, output: DoubleArray): DoubleArray {
    for (c in 0..<hiddenCount) {
      var sum = 0.0
      for (r in 0..<inputCount) {
        sum += input[r] * weightsHidden[r * hiddenCount + c]
      }

      hidden[c] = sigmoid(sum + biasesHidden[c])
    }

    for (c in 0..<outputCount) {
      var sum = 0.0
      for (r in 0..<hiddenCount) {
        sum += hidden[r] * weightsOutput[r * outputCount + c]
      }

      output[c] = sigmoid(sum + biasesOutput[c])
    }

    return output
  }
}

class Trainer(
  val network: Network,
  val hidden: DoubleArray,
  val output: DoubleArray,
  val gradHidden: DoubleArray,
  val gradOutput: DoubleArray) {
  companion object {
    fun create(
      inputCount: Int,
      hiddenCount: Int,
      outputCount: Int,
      rand: Random): Trainer {
      val weightsHidden = DoubleArray(inputCount * hiddenCount) { _ -> rand.generate() - 0.5 }
      var biasesHidden = DoubleArray(hiddenCount)
      val weightsOutput = DoubleArray(hiddenCount * outputCount) { _ -> rand.generate() - 0.5 }
      val biasesOutput = DoubleArray(outputCount)
      val network = Network(inputCount, hiddenCount, outputCount, weightsHidden, biasesHidden, weightsOutput, biasesOutput)
      val hidden = DoubleArray(hiddenCount)
      val output = DoubleArray(outputCount)
      val gradHidden = DoubleArray(hiddenCount)
      val gradOutput = DoubleArray(outputCount)
      return Trainer(network, hidden, output, gradHidden, gradOutput)
    }

    fun sigmoidPrim(f: Double) = f * (1.0 - f)
  }

  fun train(input: DoubleArray, y: DoubleArray, lr: Double) {
    network.predict(input, hidden, output)
    for (c in 0..<network.outputCount) {
      gradOutput[c] = (output[c] - y[c]) * sigmoidPrim(output[c])
    }

    for (r in 0..<network.hiddenCount) {
      var sum = 0.0
      for (c in 0..<network.outputCount) {
        sum += gradOutput[c] * network.weightsOutput[r * network.outputCount + c]
      }

      gradHidden[r] = sum * sigmoidPrim(hidden[r])
    }

    for (r in 0..<network.hiddenCount) {
      for (c in 0..<network.outputCount) {
        network.weightsOutput[r * network.outputCount + c] -= lr * gradOutput[c] * hidden[r]
      }
    }

    for (r in 0..<network.inputCount) {
      for (c in 0..<network.hiddenCount) {
        network.weightsHidden[r * network.hiddenCount + c] -= lr * gradHidden[c] * input[r]
      }
    }

    for (c in 0..<network.outputCount) {
      network.biasesOutput[c] -= lr * gradOutput[c]
    }

    for (c in 0..<network.hiddenCount) {
      network.biasesHidden[c] -= lr * gradHidden[c]
    }
  }
}
