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

// ReLU-friendly bump function
let bumpFunction x =
    max 0.0 (x - 1.0) - max 0.0 (x - 3.0)

// Generate training data: 101 points from 0.0 to 5.0
let trainingData =
    [|
        for i in 0 .. 100 ->
            let x = float i / 20.0  // [0.0 .. 5.0] in 0.05 steps
            [| x |], [| bumpFunction x |]
    |]

let randFloat =
  let P = 2147483647u
  let A = 16807u;
  let mutable current = 1u
  let inner() =
    current <- current * A % P;
    let result = float current / float P
    result
  inner

let trainer = Trainer(1, 4, 1, Activations.relu, randFloat)
let lr = 0.01
let ITERS = 5000
for i in 0 .. ITERS - 1 do
    let x, y = trainingData[i % trainingData.Length]
    trainer.Train(x, y, lr)

// Predict and print results
let net = trainer.Network

let y_hidden = Array.zeroCreate 4
let y_output = Array.zeroCreate 1
let z_hidden = Array.zeroCreate 4
let z_output = Array.zeroCreate 1

printfn "x     target    predicted    z_output  hidden activations"
[0.0 .. 0.25 .. 5.0]
|> List.iter (fun x ->
    net.Predict([| x |], z_hidden, z_output, y_hidden, y_output) |> ignore
    let target = bumpFunction x
    let output = y_output[0]
    printfn "%.2f   %.4f    %.4f   %A  %A" x target output z_output y_hidden
)

printfn "x     target    predicted"
[0.0 .. 0.25 .. 5.0]
|> List.iter (fun x ->
    let pred = net.Predict([| x |])
    let target = bumpFunction x
    printfn "%.2f   %.4f    %.4f" x target pred[0])
