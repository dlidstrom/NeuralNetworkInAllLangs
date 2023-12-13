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
)

const P uint32 = 2147483647
const A uint32 = 16807

var current uint32 = 1

func rand() float64 {
	current = current * A % P
	var result float64 = float64(current) / float64(P)
	return result
}

func xor(i uint32, j uint32) uint32 {
	return i ^ j
}

func xnor(i uint32, j uint32) uint32 {
	return 1 - xor(i, j)
}

func or(i uint32, j uint32) uint32 {
	return i | j
}

func and(i uint32, j uint32) uint32 {
	return i & j
}

func nor(i uint32, j uint32) uint32 {
	return 1 - or(i, j)
}

func nand(i uint32, j uint32) uint32 {
	return 1 - and(i, j)
}

type DataItem struct {
	input  []float64
	output []float64
}

func main() {
	var allData []DataItem
	for i := uint32(0); i < 2; i++ {
		for j := uint32(0); j < 2; j++ {
			d := DataItem{
				input: []float64{float64(i), float64(j)},
				output: []float64{
					float64(xor(i, j)),
					float64(xnor(i, j)),
					float64(or(i, j)),
					float64(and(i, j)),
					float64(nor(i, j)),
					float64(nand(i, j)),
				},
			}

			allData = append(allData, d)
		}
	}

	trainer := NewTrainer(2, 2, 6, rand)
	ITERS := 4000
	lr := 1.0
	for i := 0; i < ITERS; i++ {
		dataItem := allData[i % 4]
		trainer.Train(dataItem.input, dataItem.output, lr)
	}

	fmt.Printf("Result after %d iterations\n", ITERS)
	fmt.Println("        XOR  XNOR    OR   AND   NOR  NAND")
	for i := 0; i < len(allData); i++ {
		data := allData[i]
		pred := trainer.network.Predict(data.input)
		fmt.Printf(
			"%.0f,%.0f = %.3f %.3f %.3f %.3f %.3f %.3f\n",
			data.input[0],
			data.input[1],
			pred[0],
			pred[1],
			pred[2],
			pred[3],
			pred[4],
			pred[5],
		)
	}

	trainer.network.Print()
}
