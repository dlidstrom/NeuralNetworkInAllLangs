/*
Licensed under the MIT License.
Copyright 2023-2025 Daniel Lidstrom
*/

#include "neural.h"
#include <iomanip>
#include <iostream>

using namespace Neural;

namespace {
    const int ITERS = 4000;
    const double lr = 1.0;
    u_int32_t P = 2147483647;
    u_int32_t A = 16807;
    u_int32_t current = 1;
    double Rand() {
        current = current * A % P;
        double result = (double)current / P;
        return result;
    }

    size_t Xor(size_t i, size_t j) { return i ^ j; }
    size_t Xnor(size_t i, size_t j) { return 1 - Xor(i, j); }
    size_t Or(size_t i, size_t j) { return i | j; }
    size_t And(size_t i, size_t j) { return i & j; }
    size_t Nor(size_t i, size_t j) { return 1 - Or(i, j); }
    size_t Nand(size_t i, size_t j) { return 1 - And(i, j); }
}

void show_weights(const Network& network);

int main() {
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

    Trainer trainer = Trainer::Create(2, 2, 6, Rand);
    for (size_t i = 0; i < ITERS; i++) {
        const Vector& input = inputs[i % inputs.size()];
        const Vector& output = outputs[i % outputs.size()];
        trainer.Train(input, output, lr);
    }

    std::cout
        << "Result after "
        << ITERS
        << " iterations\n"
        << "        XOR   XNOR    OR   AND   NOR   NAND\n";
    const Network& network = trainer.network;
    for (size_t i = 0; i < inputs.size(); i++) {
        const Vector& input = inputs[i];
        Vector pred = network.Predict(input);
        std::cout
            << std::fixed
            << std::setprecision(0)
            << input[0]
            << ','
            << input[1]
            << " = "
            << std::setprecision(3)
            << pred[0]
            << "  "
            << pred[1]
            << " "
            << pred[2]
            << " "
            << pred[3]
            << " "
            << pred[4]
            << "  "
            << pred[5]
            << '\n';
    }

    show_weights(trainer.network);

    return 0;
}

void show_weights(const Network& network) {
    std::cout << "WeightsHidden:\n" << std::setprecision(6);
    for (size_t i = 0; i < network.inputCount; i++) {
        for (size_t j = 0; j < network.hiddenCount; j++) {
            std::cout << network.weightsHidden[network.inputCount * i + j] << ' ';
        }

        std::cout << '\n';
    }

    std::cout << "BiasesHidden:\n";
    for (auto c : network.biasesHidden) {
        std::cout << c << ' ';
    }

    std::cout << "\nWeightsOutput:\n";
    for (size_t i = 0; i < network.hiddenCount; i++) {
        for (size_t j = 0; j < network.outputCount; j++) {
            std::cout << network.weightsOutput[network.outputCount * i + j] << ' ';
        }

        std::cout << '\n';
    }

    std::cout << "BiasesOutput:\n";
    for (auto c : network.biasesOutput) {
        std::cout << c << ' ';
    }

    std::cout << '\n';
}
