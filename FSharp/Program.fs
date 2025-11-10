(*
Licensed under the MIT License.
Copyright 2023-2025 Daniel Lidstrom
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
let xnor a b = 1 - xor a b
let nand a b = 1 - andf a b
let nor a b = 1 - orf a b

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
