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

fun main() {
  val trainer = Trainer.create(2, 2, 6, object: Random {
    val P: UInt = 2147483647u
    val A: UInt = 16807u
    var current: UInt = 1u
    override fun generate(): Double {
      current = current * A % P
      val result = current.toDouble() / P.toDouble()
      return result;
    }
  })

  val xor: (Int, Int) -> Int = Int::xor
  val and: (Int, Int) -> Int = Int::and
  val or: (Int, Int) -> Int = Int::or
  val inputs = arrayOf(arrayOf(0, 0), arrayOf(0, 1), arrayOf(1, 0), arrayOf(1, 1))
  val trainingData = inputs.map {
    Pair(
      arrayOf(it[0].toDouble(), it[1].toDouble()).toDoubleArray(),
      arrayOf(
        xor(it[0], it[1]).toDouble(),
        1 - xor(it[0], it[1]).toDouble(),
        or(it[0], it[1]).toDouble(),
        and(it[0], it[1]).toDouble(),
        1 - or(it[0], it[1]).toDouble(),
        1 - and(it[0], it[1]).toDouble()).toDoubleArray())
  }

  val lr = 1.0
  val ITERS = 4000
  for (i in 1..ITERS) {
    for (it in trainingData) {
      trainer.train(it.first, it.second, lr)
    }
  }

  println("Result after ${ITERS} iterations")
  println("        XOR  XNOR    OR   AND   NOR  NAND")
  for (it in trainingData) {
    val pred = trainer.network.predict(it.first)
    println("%.0f,%.0f = %.3f %.3f %.3f %.3f %.3f %.3f".format(it.first[0], it.first[1], pred[0], pred[1], pred[2], pred[3], pred[4], pred[5]))
  }

  println("weights hidden:")
  for (it in trainer.network.weightsHidden) println("%.6f".format(it))
  println("biases hidden:")
  for (it in trainer.network.biasesHidden) println("%.6f".format(it))
  println("weights output:")
  for (it in trainer.network.weightsOutput) println("%.6f".format(it))
  println("biases output:")
  for (it in trainer.network.biasesOutput) println("%.6f".format(it))
}
