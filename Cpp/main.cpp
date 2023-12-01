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

#include "neural.h"
#include <iostream>

using namespace Neural;

namespace {
    int ITERS = 4000;
    u_int32_t P = 2147483647;
    u_int32_t A = 16807;
    u_int32_t current = 1;
    double Rand() {
        current = current * A % P;
        double result = (double)current / P;
        return result;
    }

    int Xor(int i, int j) { return i ^ j; }
    int Xnor(int i, int j) { return 1 - Xor(i, j); }
    int Or(int i, int j) { return i | j; }
    int And(int i, int j) { return i & j; }
    int Nor(int i, int j) { return 1 - Or(i, j); }
    int Nand(int i, int j) { return 1 - And(i, j); }
}

int main() {
    Trainer trainer = Trainer::Create(2, 2, 6, Rand);
    Matrix inputs = Matrix();
    Matrix outputs = Matrix();
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            inputs.push_back({ (double)i, (double)j});
            outputs.push_back({
                (double)Xor(i, j),
                (double)Xnor(i, j),
                (double)Or(i, j),
                (double)And(i, j),
                (double)Nor(i, j),
                (double)Nand(i, j)
            });
        }
    }

    double lr = 1.0;
    for (size_t i = 0; i < ITERS; i++) {
        Vector input = inputs[i % inputs.size()];
        Vector output = outputs[i % outputs.size()];
        trainer.Train(input, output, lr);
    }

    std::cout << "Result after " << ITERS << " iterations" << std::endl;
    for (auto r : trainer.Network().WeightsHidden()) {
        for (auto c : r) {
            std::cout << c << ' ';
        }
        std::cout << std::endl;
    }

    return 0;
}
