# Neural Network in All Langs <!-- omit in toc -->

- [1. Training](#1-training)
  - [1.1. Logical Functions](#11-logical-functions)
  - [1.2. Hand Written Digits](#12-hand-written-digits)
- [2. Learning](#2-learning)
- [3. Implementation Goals](#3-implementation-goals)
  - [3.1. Simple Random Number Generator](#31-simple-random-number-generator)
  - [3.2. License](#32-license)
- [4. Reference Implementation](#4-reference-implementation)
  - [4.1. Inputs and Randomized Starting Weights](#41-inputs-and-randomized-starting-weights)
  - [4.2. Forward Propagation](#42-forward-propagation)
  - [4.3. Backpropagation](#43-backpropagation)
  - [4.4. Weight Updates](#44-weight-updates)
- [5. Using this in your own solution](#5-using-this-in-your-own-solution)
- [6. References](#6-references)

This repository aims to implement a vanilla neural network in all major
programming languages. It is the "hello world" of ai programming. We will
implement a fully connected network with a single hidden layer using the sigmoid
activation function for both the hidden and the output layer. This kind of
network can be used to do hand writing recognition, or other kinds of pattern
recognitions, categorizations, or predictions. This is intended as your entry
level into ai programming, i.e. for the enthusiast or hobby programmer. Any more
advanced use cases should look elsewhere as there are infinitely more powerful
methods available for the professional.

We do not aim to go through the math involved (see [1] if you're interested). We
prefer to focus on the code itself and will happily copy a solution from one
programming language to another without worrying about the theoretical
background.

## 1. Training

For training we will use two datasets.

### 1.1. Logical Functions

The first is simple and will be these logical functions: xor, xnor, or, nor,
and, and nand. This truth table represents the values that the network will
learn, given two inputs; $i_1$ and $i_2$:

$$\begin{array}{rcl}
i_1 & i_2 & xor & xnor & or & nor & and & nand \\
0  & 0  & 0 & 1  &  0 & 1 & 0 & 1 \\
0  & 1  & 1 & 0  &  1 & 0 & 0 & 1 \\
1  & 0  & 1 & 0  &  1 & 0 & 0 & 1 \\
1  & 1  & 0 & 1  &  1 & 0 & 1 & 0
\end{array}$$

This test is interesting as it shows how flexible a simple neural network can
be. There are two inputs, 6 outputs, and it is sufficient to have two hidden
neurons. Such a network consists of a total of 24 weights:

- 4 hidden weights (2 inputs * 2 hidden)
- 2 hidden biases (one for each hidden neuron)
- 12 output weights (2 hidden * 6 outputs)
- 6 output biases (one for each output neuron)

### 1.2. Hand Written Digits

The second dataset consists of thousands of hand written digits. This is
actually also a "toy" dataset but training a network to recognize all digits
correctly is still a bit of a challenge. This dataset was originally downloaded
from <https://archive.ics.uci.edu/dataset/178/semeion+handwritten+digit>.

Each line consists of 256 inputs (16x16 pixels) corresponding to one
hand written digit. At the end of the line are 10 digits which signify
the handwritten digit:

```txt
0: 1 0 0 0 0 0 0 0 0 0
1: 0 1 0 0 0 0 0 0 0 0
2: 0 0 1 0 0 0 0 0 0 0
3: 0 0 0 1 0 0 0 0 0 0
4: 0 0 0 0 1 0 0 0 0 0
...
9: 0 0 0 0 0 0 0 0 0 1
```

Parsing this dataset needs to be implemented for each language.

## 2. Learning

Our code will perform backpropagation to learn the weights. We update
the weights after each input. This is called stochastic learning, as
opposed to batch learning where multiple inputs are presented before
updating weights. Stochastic learning is generally preferred [2]. Note
that inputs need to be shuffled for effective learning.

## 3. Implementation Goals

One of our goals is to have as few or no dependencies. These implementations
should be easy to integrate and that requires dependency free code. Another goal
is to implement fast code. Nifty, one-liners which look good but have bad
performance should be avoided. It is fine to use for loops for matrix
multiplication, as an example (i.e. no fancy linear algebra libraries are needed
unless this is available in the standard library of the programming language).

We strive for:

- code that is easy to copy/paste for reuse
- dependency-free code
- adequate performance in favour of nifty one-liners
- making it easy to serialize weights for storing and loading, but leave it for
  the users own preference
- implementations in all major languages
- simple tests that verify our implementations and secure them for the future
- having fun exploring neural networks!

### 3.1. Simple Random Number Generator

Now, a note about random number generation. Training a neural network requires
that the initial weights are randomly assigned. We will specify a simple random
number generator algorithm that should be used in all implementations. We
actually want each implementation to learn the same weights. This makes it
easier to verify the implementation. Of course, whoever wants to integrate into
their own solution is free to pick another random number generator.

```csharp
uint p = 2147483647;
uint a = 16807;
uint current = 1;
uint Rand()
{
    current = a * current % p;
    return current;
}

double Random()
{
    return (double)Rand() / p;
}
```

The first few random numbers are:

```txt
7,82636925942561E-06
0,131537788143166
0,755604293083588
0,44134794289309
0,734872750814479
0,00631718803491313
0,172979253424788
0,262310957192588
```

### 3.2. License

> All code must be licensed under the permissive MIT license.
> No GPL!

## 4. Reference Implementation

For reference you can use this Python implementation which uses NumPy,
but should be fairly easy to understand. Why Python? Because Python
has become the *lingua franca* of ai programming. It is also easy to
modify and fast to re-run, thus ideal for experiments.

We will now go through the reference implementation and include some
math diagrams for those that want to know what's going on. You'll see
the *how* but not the *why* (see references section for that).

Here, one forward and one backward propagation is shown. You can use
these values to verify your own calculations. The example is the logical
functions shown earlier with the inputs being both `1`, i.e. `1 1`. There
are 3 hidden neurons and 6 outputs (xor, xnor, and, nand, or, nor).

### 4.1. Inputs and Randomized Starting Weights

2 inputs, 3 hidden neurons, 6 outputs. These are the initial values for the
input layer and the hidden layer. $w$ is the weights, $b$ is the biases. Note
that we are showing randomized biases here to help understand the calculations.
For the implementation we will initialize biases to 0 per the recommendation
here [3].

$$\begin{array}{rcl}
input & = &
\begin{bmatrix}
1 & 1
\end{bmatrix} \\
w_{hidden} & = &
\begin{bmatrix}
0.375 & 0.951 & 0.732 \\
0.599 & 0.156 & 0.156
\end{bmatrix} \\
b_{hidden} & = &
\begin{bmatrix}
0.058 & 0.866 & 0.601
\end{bmatrix} \\
w_{output} & = &
\begin{bmatrix}
0.708 & 0.021 & 0.970 & 0.832 & 0.212 & 0.182 \\
0.183 & 0.304 & 0.525 & 0.432 & 0.291 & 0.612 \\
0.139 & 0.292 & 0.366 & 0.456 & 0.785 & 0.200
\end{bmatrix} \\
b_{output} & = &
\begin{bmatrix}
0.514 & 0.592 & 0.046 & 0.608 & 0.171 & 0.065
\end{bmatrix} \\
\end{array}$$

### 4.2. Forward Propagation

First we show forward propagation for the hidden layer.

$$\begin{array}{rcl}
sigmoid(x) & = & \frac{1}{1 + e^{-x}} \\
y_{hidden} & = & sigmoid(np.dot(input, w_{hidden}) + b_{hidden}) \\
y_{hidden} & = & sigmoid(\begin{bmatrix}
1 \cdot 0.375 + 1 \cdot 0.599 & 1 \cdot 0.951 + 1 \cdot 0.156 & 1 \cdot 0.732 + 1 \cdot 0.156
\end{bmatrix} + b_{hidden}) \\
y_{hidden} & = & sigmoid(\begin{bmatrix}
0.974 & 1.107 & 0.888
\end{bmatrix} + \begin{bmatrix}
0.058 & 0.866 & 0.601
\end{bmatrix}
) \\
y_{hidden} & = & sigmoid(\begin{bmatrix}
1.032 & 1.973 & 1.489
\end{bmatrix}
) \\
y_{hidden} & = &
\begin{bmatrix}
0.737 & 0.878 & 0.816
\end{bmatrix} \\
\end{array}$$

Now to forward propagation for the output layer. This is the actual prediction
of the network.

$$\begin{array}{rcl}
y_{hidden} & = &
\begin{bmatrix}
0.737 & 0.878 & 0.816
\end{bmatrix} \\
y_{output} & = & sigmoid(np.dot(y_{hidden}, w_{output}) + b_{output}) \\
y_{output} & = & sigmoid(np.dot( \\
& & \begin{bmatrix}
0.737 & 0.878 & 0.816
\end{bmatrix}, \\
& & \begin{bmatrix}
0.708 & 0.021 & 0.97 & 0.832 & 0.212 & 0.182 \\
0.183 & 0.304 & 0.525 & 0.432 & 0.291 & 0.612 \\
0.139 & 0.292 & 0.366 & 0.456 & 0.785 & 0.200
\end{bmatrix}) \\
& & + \begin{bmatrix}
0.514 & 0.592 & 0.046 & 0.608 & 0.171 & 0.065
\end{bmatrix}) \\
y_{output} & = & sigmoid(\begin{bmatrix}
0.797 & 0.521 & 1.475 & 1.365 & 1.053 & 0.834
\end{bmatrix} + \\
& & \begin{bmatrix}
0.514 & 0.592 & 0.046 & 0.608 & 0.171 & 0.065
\end{bmatrix}) \\
y_{output} & = & sigmoid(\begin{bmatrix}
1.311 & 1.113 & 1.521 & 1.973 & 1.223 & 0.899
\end{bmatrix}) \\
y_{output} & = & \begin{bmatrix}
0.788 & 0.753 & 0.821 & 0.878 & 0.773 & 0.711
\end{bmatrix} \\
\end{array}$$

### 4.3. Backpropagation

Now we have calculated output. These are off according to the expected output
and the purpose of the next step, backpropagation, is to correct the weights for
a slightly improved prediction in the next iteration. First step of
backpropagation is to compute the error gradient ($\nabla$) of the output layer.

$$\begin{array}{rcl}
sigmoid'(x) & = & x \cdot (1 - x) \\
y & = & \begin{bmatrix}
\underset{xor}{0}
& \underset{xnor}{1}
& \underset{or}{1}
& \underset{and}{1}
& \underset{nor}{0}
& \underset{nand}{0}
\end{bmatrix} \text{the expected value for the given inputs} \\
\nabla_{output} & = & error'(y, y_{output}) \cdot sigmoid'(y_{output}) \\
\nabla_{output} & = & (y_{output} - y) \cdot sigmoid'(y_{output}) \\
& & \text{note that these are performed element-wise} \\
\nabla_{output} & = & (\begin{bmatrix}
0.788-0 & 0.753-1 & 0.821-1 & 0.878-1 & 0.773-0 & 0.711-0
\end{bmatrix} \\
& & \cdot (\begin{bmatrix}
0.788 & 0.753 & 0.821 & 0.878 & 0.773 & 0.711
\end{bmatrix}) \\
& & \cdot \begin{bmatrix}
1-0.788 & 1-0.753 & 1-0.821 & 1-0.878 & 1-0.773 & 1-0.711
\end{bmatrix} \\
\nabla_{output} & = &
\begin{bmatrix}
0.788 & -0.247 & -0.179 & -0.122 & 0.773 & 0.711 \\
{}\cdot0.788 & {}\cdot0.753 & {}\cdot0.821 & {}\cdot0.878 & {}\cdot0.773 & {}\cdot0.711 \\
{}\cdot0.212 & {}\cdot0.247 & {}\cdot0.179 & {}\cdot0.122 & {}\cdot0.227 & {}\cdot0.289
\end{bmatrix} \\
\nabla_{output} & = &
\begin{bmatrix}
0.132 & -0.046 & -0.026 & -0.013 & 0.136 & 0.146
\end{bmatrix} \\
\end{array}$$

Now compute the error gradient of the hidden layer.

$$\begin{array}{rcl}
\nabla_{hidden} & = & \nabla_{output} \cdot w_{output}^T \cdot sigmoid'(y_{hidden}) \\
\nabla_{hidden} & = & \begin{bmatrix}
0.132 & -0.046 & -0.026 & -0.013 & 0.136 & 0.146
\end{bmatrix} \\
& & \cdot \begin{bmatrix}
0.708 & 0.183 & 0.139 \\
0.021 & 0.304 & 0.292 \\
0.970 & 0.525 & 0.366 \\
0.832 & 0.432 & 0.456 \\
0.212 & 0.291 & 0.785 \\
0.182 & 0.612 & 0.200
\end{bmatrix} \\
& & \cdot sigmoid'(\begin{bmatrix}
0.737 & 0.878 & 0.816
\end{bmatrix}) \\
\nabla_{hidden} & = & \begin{bmatrix}
0.112 & 0.120 & 0.125
\end{bmatrix} \cdot
\begin{bmatrix}
0.194 & 0.107 & 0.150
\end{bmatrix} \\
\nabla_{hidden} & = & \begin{bmatrix}
0.022 & 0.013 & 0.019
\end{bmatrix}
\end{array}$$

### 4.4. Weight Updates

Finally we can apply weight updates. $\alpha$ is the learning rate which here
will be $1$. First update weights and biases for the output layer.

$$\begin{array}{rcl}
w_{output} & = & w_{output} - \alpha \cdot y_{hidden}^T \cdot \nabla_{output} \\
w_{output} & = & \begin{bmatrix}
0.611 & 0.055 & 0.989 & 0.842 & 0.112 & 0.074 \\
0.068 & 0.345 & 0.548 & 0.443 & 0.172 & 0.484 \\
0.032 & 0.33  & 0.388 & 0.467 & 0.674 & 0.080
\end{bmatrix} \\
b_{output} & = & b_{output} - \alpha \cdot \nabla_{output} \\
b_{output} & = &
\begin{bmatrix}
0.383 & 0.638 & 0.073 & 0.621 & 0.035 & -0.081
\end{bmatrix}
\end{array}$$

Now update weights and biases for the input layer.

$$\begin{array}{rcl}
w_{hidden} & = & w_{hidden} - \alpha \cdot x^T \cdot \nabla_{hidden} \\
w_{hidden} & = &
\begin{bmatrix}
0.353 & 0.938 & 0.713 \\
0.577 & 0.143 & 0.137
\end{bmatrix} \\
b_{hidden} & = & b_{hidden} - \alpha \cdot \nabla_{hidden} \\
b_{hidden} & = &
\begin{bmatrix}
0.037 & 0.853 & 0.582
\end{bmatrix}
\end{array}$$

## 5. Using this in your own solution

If you do use any of these implementations in your own solution, then here
are some things to keep in mind for good results:

- shuffle inputs
- try to have about the same number of samples for each output to avoid
  "drowning out" a sample
- try different learning rates (0.1 to 0.5 seems to work well for many problems)
- you may try "annealing" the learning rate, meaning start high (0.5) and slowly
  decrease over the epochs

## 6. References

[1] <http://neuralnetworksanddeeplearning.com/>
[2] <https://leon.bottou.org/publications/pdf/tricks-1998.pdf>
[3] <https://cs231n.github.io/neural-networks-2/>
