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
let nInputs = 26             // e.g. 13 card ranks: one-hot or count-based
let nHidden = 20
let nPolicyOutputs = 13       // e.g. 5 possible legal moves
let learningRate = 1.0

let trainingData =
  [|
    for _ in 1..100 ->
      // Random binary input vector (e.g., presence of card ranks)
      let input = Array.init nInputs (fun _ -> if randFloat() > 0.5 then 1.0 else 0.0)

      // Simulate a one-hot policy label (e.g., "move 2 is the best")
      let bestMove = int (randFloat() * float nPolicyOutputs)
      let policy = Array.init nPolicyOutputs (fun i -> if i = bestMove then 1.0 else 0.0)

      // Simulate win probability target (e.g., from self-play outcome)
      let value = if randFloat() > 0.5 then 1.0 else 0.0

      input, Array.concat [policy;  [| value |] ]
  |]
let trainer = Trainer(nInputs, nHidden, nPolicyOutputs + 1, randFloat)
let lr = 0.1
let ITERS = 40000
for e = 0 to ITERS - 1 do
  let input, y = trainingData[e % trainingData.Length]
  trainer.Train(input, y, lr)

let network = trainer.Network
let input, output = trainingData[1]
let policyOut = network.Predict input
printfn "\nTest Prediction:"
printfn "Input    : %A" input
printfn "Expected : %s" (output |> Array.map (sprintf "%0.2f") |> String.concat ", ")
printfn "Predicted: %s" (policyOut |> Array.map (sprintf "%0.2f") |> String.concat ", ")
