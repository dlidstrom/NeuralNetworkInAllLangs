# Neural Network in All Languages <!-- omit in toc -->

<img src="https://github.com/dlidstrom/NeuralNetworkInAllLangs/raw/main/doc/networks.png" width="400px">

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![F#](https://img.shields.io/badge/f%23-%23239120.svg?style=for-the-badge&logo=f-sharp)
![C#](https://img.shields.io/badge/c%23-%23239120.svg?style=for-the-badge&logo=c-sharp&logoColor=white)
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![C](https://img.shields.io/badge/c-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white)
![Kotlin](https://img.shields.io/badge/kotlin-%237F52FF.svg?style=for-the-badge&logo=kotlin&logoColor=white)

[![Run Tests](https://github.com/dlidstrom/NeuralNetworkInAllLangs/actions/workflows/ci.yaml/badge.svg)](https://github.com/dlidstrom/NeuralNetworkInAllLangs/actions/workflows/ci.yaml)

- [1. Introduction](#1-introduction)
- [2. Training](#2-training)
  - [2.1. Logical Functions](#21-logical-functions)
    - [2.1.1. Lithmus Test](#211-lithmus-test)
  - [2.2. Hand Written Digits](#22-hand-written-digits)
- [3. Learning](#3-learning)
- [4. Implementation Goals](#4-implementation-goals)
  - [4.1. Simple Random Number Generator](#41-simple-random-number-generator)
  - [4.2. License](#42-license)
  - [4.3. Implementations](#43-implementations)
    - [4.3.1. Sample Output](#431-sample-output)
- [5. Reference Implementation](#5-reference-implementation)
  - [5.1. Inputs and Randomized Starting Weights](#51-inputs-and-randomized-starting-weights)
  - [5.2. Forward Propagation](#52-forward-propagation)
  - [5.3. Backpropagation](#53-backpropagation)
  - [5.4. Weight Updates](#54-weight-updates)
- [6. Using this in your own solution](#6-using-this-in-your-own-solution)
- [7. References](#7-references)

## 1. Introduction

This repository aims to implement a vanilla neural network in all major
programming languages. It is the "hello world" of ai programming. We will
implement a fully connected network with a single hidden layer using the sigmoid
activation function for both the hidden and the output layer. This kind of
network can be used to do hand writing recognition, or other kinds of pattern
recognitions, categorizations, or predictions. This is intended as your entry
level into ai programming, i.e. for the enthusiast or hobby programmer (you and
me). More advanced use cases should look elsewhere as there are infinitely more
powerful methods available for the professional.

> Disclaimer! Do not expect blazing fast performance. If you have such
> requirements or expectations then you should definitely look elsewhere. Stay
> here if you want to learn more about implementing a neural network!

We do not aim to justify the math involved (see [1] if you're interested). We
prefer to focus on the code itself and will happily copy a solution from one
programming language to another without worrying about the theoretical
background.

## 2. Training

For training and verifying our implementations we will use two datasets.

### 2.1. Logical Functions

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

- 4 hidden weights (2 inputs * 2 hidden neurons)
- 2 hidden biases (one for each hidden neuron)
- 12 output weights (2 hidden neurons * 6 output neurons)
- 6 output biases (one for each output neuron)

<img src="https://github.com/dlidstrom/NeuralNetworkInAllLangs/raw/main/doc/nn.png" width="400px">

> 💯 We expect each implementation to learn exactly the same network weights!

#### 2.1.1. Lithmus Test

The logical functions example can be used as a "lithmus test" of neural network
implementations. A proper implementation will be able to learn the 6 functions
using the 24 weights as detailed above. An improper implementation (one that
doesn't implement biases correctly, for example) likely will need more hidden
nodes to learn successfully (if at all). A larger network means more
mathematical operations so keep this in mind when you evaluate other
implementations. You don't want to waste cpu cycles unnecessarily.

### 2.2. Hand Written Digits

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

## 3. Learning

Our code will perform backpropagation to learn the weights. We update
the weights after each input. This is called stochastic learning, as
opposed to batch learning where multiple inputs are presented before
updating weights. Stochastic learning is generally preferred [2]. Note
that inputs need to be shuffled for effective learning.

## 4. Implementation Goals

One of our goals is to have as few or no dependencies. These implementations
should be easy to integrate and that requires dependency-free code. Another goal
is to implement fast code. Nifty, one-liners which look good but have bad
performance should be avoided. It is fine to use for loops for matrix
multiplication, as an example (i.e. no fancy linear algebra libraries are needed
unless this is available in the standard library of the programming language).

We strive for:

- code that is easy to copy/paste for reuse
- dependency-free code
- straight forward code, no excessive object orientation which makes the code
  look like an OOAD excercise from the 90s
- adequate performance in favour of nifty (but slow) one-liners
- making it easy to serialize weights for storing and loading, but leave it for
  the users own preference
- implementations in all major languages
- simple tests that verify our implementations and secure them for the future
- having fun exploring neural networks!

### 4.1. Simple Random Number Generator

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

> This was chosen to avoid any complexity! There are widely used algorithms for
> better random number generation but it isn't important in this case. We simply
> need some starting values and they don't have to be very random as long as
> they are all different. We might've just used the current microseconds!
>
> The code samples all contain an extension point where you can plug in your own
> implementation, should you wish to do so (or just hardcode your choice!).

### 4.2. License

All code *in this repository* is licensed under MIT license.
This is a **permissive** license and you can use this code in your
personal projects, or commercial as well, without needing to share
anything back. MIT license is the most common license on GitHub.

If you would like to contribute to this repository, for example
by adding an implemention in another programming language,
then you must also accept your code with MIT license.

> All code in this repo must be licensed under the permissive MIT license.
> Please add license header to every source file. No GPL allowed!

### 4.3. Implementations

This is the current status of the implementations available. We follow a maturity model based on these criteria:

- Level 0: implement logical functions network
- Level 1: use modules/files to make implementation easy to reuse by copy/paste
- Level 2: implement a unit test to verify level 0 and make the code future safe
- Level 3: implement digit recognition with the Semeion dataset
- Level 4: implement a unit test to verify level 3 and make the code future safe

| Language | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 | Contributor |
|-|-|-|-|-|-|-|
| C# | ⭐️ | ⭐️ | ⭐️ | ⭐️ | ⭐️ | [@dlidstrom](https://github.com/dlidstrom) |
| Rust | ⭐️ | ⭐️ | ⭐️ | | | [@dlidstrom](https://github.com/dlidstrom) |
| F# | ⭐️ | ⭐️ | ⭐️ | | | [@dlidstrom](https://github.com/dlidstrom) |
| C++ | ⭐️ | ⭐️ | ⭐️ | | | [@dlidstrom](https://github.com/dlidstrom) |
| C | ⭐️ | ⭐️ | ⭐️ | | | [@dlidstrom](https://github.com/dlidstrom) |
| Kotlin | ⭐️ | ⭐️ | | | | [@dlidstrom](https://github.com/dlidstrom) |
| Python | ⭐️ | | | | | [@dlidstrom](https://github.com/dlidstrom) |

> Note! The Python implementation is only here as a reference. If you are using Python you already
> have access to all ai tools and libraries you need.

#### 4.3.1. Sample Output

Digit recognition is done using only 14 hidden neurons, 10 learning epochs (an
epoch is a run through the entire dataset), and a learning rate of 0.5. Using
these hyper parameters we are able to recognize 99.1% of the Semeion digits
accurately. You may be able to improve by adding more hidden neurons, doing more
epochs, and annealing the learning rate (decrease slowly). However we are also
at risk of over learning which decreases our network's ability to generalize (it
learns too specific, i.e. the noise in the data set).

This output shows accuracy in predicting the correct digit, and average
confidence i.e. score of the largest output value:

```bash
~/CSharp $ dotnet run --semeion ../semeion.data 14 10 0.5
accuracy: 85.876 % (1368/1593), avg confidence: 68.060 %
accuracy: 91.965 % (1465/1593), avg confidence: 78.090 %
accuracy: 95.041 % (1514/1593), avg confidence: 84.804 %
accuracy: 96.673 % (1540/1593), avg confidence: 86.184 %
accuracy: 97.552 % (1554/1593), avg confidence: 88.259 %
accuracy: 98.242 % (1565/1593), avg confidence: 90.609 %
accuracy: 98.745 % (1573/1593), avg confidence: 92.303 %
accuracy: 98.870 % (1575/1593), avg confidence: 93.385 %
accuracy: 98.870 % (1575/1593), avg confidence: 93.261 %
accuracy: 99.121 % (1579/1593), avg confidence: 94.304 %
        *******
     ****** ***
  ******     **
 *****      ****
****      *****
***       ***
**      *****
**** **** ***
 ******* ***
         ***
        ***
       ***
      ****
     ***
  ******
  ***
Prediction (output from network for the above input):
0:  0.252 %
1:  0.253 %
2:  0.010 %
3:  0.028 %
4:  0.005 %
5:  4.867 %
6:  0.000 %
7:  2.864 %
8:  7.070 %
9: 94.103 % <-- best prediction
```

Looks good, doesn't it?

## 5. Reference Implementation

For reference we have [a Python implementation](./Python/Xor.py) which uses NumPy,
and should be fairly easy to understand. Why Python? Because Python
has become the *lingua franca* of ai programming. It is also easy to
modify and fast to re-run, thus ideal for experiments.

We will now go through the reference implementation and include some math
diagrams for those that want to know what's going on. You'll see the *how* but
not the *why* (see references section for that).

Here, one forward and one backward propagation is shown. You can use these
values to verify your own calculations. The example is the logical functions
shown earlier with the inputs being both `1`, i.e. `1 1`. We will use 3 hidden
neurons and 6 outputs (xor, xnor, and, nand, or, nor).

### 5.1. Inputs and Randomized Starting Weights

These are the initial values for the input layer and the hidden layer. $w$ is
the weights, $b$ is the biases. Note that we are showing randomized biases here
to help understand the calculations. For the implementation we will initialize
biases to 0 per the recommendation here [3].

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

### 5.2. Forward Propagation

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

### 5.3. Backpropagation

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

### 5.4. Weight Updates

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

## 6. Using this in your own solution

If you do use any of these implementations in your own solution, then here
are some things to keep in mind for good results:

- shuffle inputs
- try to have about the same number of samples for each output to avoid
  "drowning out" a sample
- try different learning rates (0.1 to 0.5 seems to work well for many problems)
- you may try "annealing" the learning rate, meaning start high (0.5) and slowly
  decrease over the epochs

## 7. References

[1] <http://neuralnetworksanddeeplearning.com/> <br>
[2] <https://leon.bottou.org/publications/pdf/tricks-1998.pdf> <br>
[3] <https://cs231n.github.io/neural-networks-2/>
