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

#if !defined(NEURAL_H)
#define NEURAL_H

#include <functional>
#include <vector>

namespace Neural {
    typedef std::vector<double> Vector;
    typedef std::vector<Vector> Matrix;
    class Network {
        int hiddenCount;
        int outputCount;
        Matrix weightsHidden;
        Vector biasesHidden;
        Matrix weightsOutput;
        Vector biasesOutput;
    public:
        Network(
            int hiddenCount,
            int outputCount,
            Matrix&& weightsHidden,
            Vector&& biasesHidden,
            Matrix&& weightsOutput,
            Vector&& biasesOutput);
        int HiddenCount() const;
        int OutputCount() const;
        Matrix WeightsHidden() const;
        Vector BiasesHidden() const;
        Matrix WeightsOutput() const;
        Vector BiasesOutput() const;
        Vector Predict(const Vector& input) const;
        Vector Predict(const Vector& input, Vector& hidden, Vector& output) const;
    };

    class Trainer {
        Network network;
        Vector hidden;
        Vector output;
        Vector gradHidden;
        Vector gradOutput;
        Trainer(
            Neural::Network&& network,
            Neural::Vector&& hidden,
            Neural::Vector&& output,
            Neural::Vector&& gradHidden,
            Neural::Vector&& gradOutput);
    public:
        const Network& Network() const;
        const Vector& Hidden() const;
        const Vector& Output() const;
        const Vector& GradHidden() const;
        const Vector& GradOutput() const;
        static Trainer Create(Neural::Network&& network, int hiddenCount, int outputCount);
        static Trainer Create(int inputCount, int hiddenCount, int outputCount, std::function<double()> rand);
        void Train(const Vector& input, const Vector& output, double lr);
    };
}

#endif
