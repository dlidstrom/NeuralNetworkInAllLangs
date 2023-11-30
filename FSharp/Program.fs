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

open Neural

let randFloat =
  let P = 2147483647u
  let A = 16807u;
  let mutable current = 1u
  let inner() =
    current <- current * A % P;
    let result = float current / float P
    result
  inner
let xor a b = a ^^^ b
let orf (a: int) b = a ||| b
let andf (a: int) b = a &&& b
let inv = (-) 1
let xnor a b = xor a b |> inv
let nand a b = andf a b |> inv
let nor a b = orf a b |> inv

let trainingData = [|
  for i = 0 to 1 do
    for j = 0 to 1 do
      [| float i; j |],
      [| xor i j |> float; xnor i j; orf i j; andf i j; nor i j; nand i j |]
|]

let trainer = Trainer(2, 2, 6, randFloat)
let lr = 1.0
let ITERS = 4000
for e = 0 to ITERS - 1 do
  let input, y = trainingData[e % trainingData.Length]
  trainer.Train(input, y, lr)

let network = trainer.Network
printfn "Result after %d iterations" ITERS
printfn "        XOR   XNOR    OR   AND   NOR   NAND"
for i, _ in trainingData do
  let pred = network.Predict(i)
  printfn
    "%.0f,%.0f = %.3f  %.3f %.3f %.3f %.3f  %.3f"
    i[0]
    i[1]
    pred[0]
    pred[1]
    pred[2]
    pred[3]
    pred[4]
    pred[5]

let networkVals = {|
  WeightsHidden = network.WeightsHidden
  BiasesHidden = network.BiasesHidden
  WeightsOutput = network.WeightsOutput
  BiasesOutput = network.BiasesOutput
|}
printfn $"network: %A{networkVals}"
