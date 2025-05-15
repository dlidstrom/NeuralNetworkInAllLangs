(*
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
*)

module Neural

open System

type Activations = {
  activate: float -> float
  derivative: float -> float
}
module Activations =
  let sigmoid = {
    activate = fun x -> 1.0 / (1.0 + Math.Exp(-x))
    derivative = fun x -> x * (1.0 - x)
  }
  let tanh = {
    activate = fun x -> Math.Tanh x
    derivative = fun x -> 1.0 - Math.Pow(Math.Tanh x, 2.0)
  }
  let relu = {
    activate = fun x -> if x < 0.0 then 0.0 else x
    derivative = fun x -> if x < 0.0 then 0.0 else 1.0
  }

type Vector = float []
type Matrix = Vector []

type Network(
  n_inputs,
  n_hidden,
  n_outputs,
  weightsHidden: Vector,
  biasesHidden: Vector,
  weightsOutput: Vector,
  biasesOutput: Vector,
  activations) =
  member val WeightsHidden = weightsHidden
  member val BiasesHidden = biasesHidden
  member val WeightsOutput = weightsOutput
  member val BiasesOutput = biasesOutput

  member this.Predict (input: Vector) =
    let y_hidden = Array.zeroCreate n_hidden
    let y_output = Array.zeroCreate n_outputs
    this.Predict(input, y_hidden, y_output)

  member _.Predict (input: Vector, y_hidden, y_output) =
    for c = 0 to n_hidden - 1 do
      let mutable sum = 0.
      for r = 0 to n_inputs - 1 do
        sum <- sum + input[r] * weightsHidden[r * n_hidden + c]
      y_hidden[c] <- activations.activate (sum + biasesHidden[c])

    for c = 0 to n_outputs - 1 do
      let mutable sum = 0.
      for r = 0 to n_hidden - 1 do
        sum <- sum + y_hidden[r] * weightsOutput[r * n_outputs + c]
      y_output[c] <- activations.activate (sum + biasesOutput[c])

    y_output

type Trainer(network, n_inputs, n_hidden, n_outputs, activations) =
  let y_hidden = Array.zeroCreate n_hidden
  let y_output = Array.zeroCreate n_outputs
  let grad_hidden = Array.zeroCreate n_hidden
  let grad_output = Array.zeroCreate n_outputs
  new(n_inputs, n_hidden, n_outputs, activations, randFloat) =
    let weightsHidden = Array.init (n_inputs * n_hidden) (fun _ -> randFloat() - 0.5)
    let biasesHidden = Array.zeroCreate n_hidden
    let weightsOutput = Array.init (n_hidden * n_outputs) (fun _ -> randFloat() - 0.5)
    let biasesOutput = Array.zeroCreate n_outputs
    let network = Network(
      n_inputs,
      n_hidden,
      n_outputs,
      weightsHidden,
      biasesHidden,
      weightsOutput,
      biasesOutput,
      activations)
    Trainer(network, n_inputs, n_hidden, n_outputs, activations)
  member val Network = network

  member _.Train (input: Vector, y: Vector, lr) =
    network.Predict(input, y_hidden, y_output) |> ignore<Vector>
    for c = 0 to n_outputs - 1 do
      grad_output[c] <- (y_output[c] - y[c]) * activations.derivative y_output[c]

    for r = 0 to n_hidden - 1 do
      let mutable sum = 0.
      for c = 0 to n_outputs - 1 do
        sum <- sum + grad_output[c] * network.WeightsOutput[r * n_outputs + c]
      grad_hidden[r] <- sum * activations.derivative y_hidden[r]

    for r = 0 to n_hidden - 1 do
      for c = 0 to n_outputs - 1 do
        network.WeightsOutput[r * n_outputs + c] <- network.WeightsOutput[r * n_outputs + c] - lr * grad_output[c] * y_hidden[r]

    for r = 0 to n_inputs - 1 do
      for c = 0 to n_hidden - 1 do
        network.WeightsHidden[r * n_hidden + c] <- network.WeightsHidden[r * n_hidden + c] - lr * grad_hidden[c] * input[r]

    for c = 0 to n_outputs - 1 do
      network.BiasesOutput[c] <- network.BiasesOutput[c] - lr * grad_output[c]

    for c = 0 to n_hidden - 1 do
      network.BiasesHidden[c] <- network.BiasesHidden[c] - lr * grad_hidden[c]
