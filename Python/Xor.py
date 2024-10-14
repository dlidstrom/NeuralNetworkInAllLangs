# https://mlnotebook.github.io/post/nn-in-python/
# https://flipdazed.github.io/blog/python%20tutorial/introduction-to-neural-networks-in-python-using-XOR

import numpy as np

# true means run a single epoch and output intermediate values; set to false to run all epochs
debug = False

np.set_printoptions(precision=3, suppress=True)

np.random.seed(42) # this makes sure you get the same results as me

def xor(x1, x2):
    return bool(x1) != bool(x2)

def xnor(x1, x2):
    return 1 - xor(x1, x2)

def orf(x1, x2):
    return bool(x1) or bool(x2)

def andf(x1, x2):
    return bool(x1) and bool(x2)

def nand(x1, x2):
    return 1 - andf(x1, x2)

def nor(x1, x2):
    return 1 - orf(x1, x2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_result):
    return sigmoid_result * (1 - sigmoid_result)

def error(target, prediction):
    return .5 * (target - prediction)**2

def error_derivative(target, prediction):
    return - target + prediction

xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
ys = np.array([
    [xor(*i), xnor(*i), orf(*i), andf(*i), nor(*i), nand(*i)]
        for i in xs
    ],
    dtype=int)
print(ys)
alpha = 1
n_neurons_input, n_neurons_hidden, n_neurons_output = 2, 2, 6

w_hidden = np.random.random(size=(n_neurons_input, n_neurons_hidden))
if debug: print("w_hidden", w_hidden)
b_hidden = np.random.random(size=(1, n_neurons_hidden))
if debug: print("b_hidden", b_hidden)

w_output = np.random.random(size=(n_neurons_hidden, n_neurons_output))
if debug: print("w_output", w_output)
b_output = np.random.random(size=(1, n_neurons_output))
if debug: print("b_output", b_output)

for i in range(4000):
    ix = i % len(xs)
    x = xs[ix:ix+1]
    y = ys[ix:ix+1]

    # forward prop
    if debug: print("x", x)
    y_hidden = sigmoid(np.dot(x, w_hidden) + b_hidden)
    if debug: print("y_hidden = sigmoid(np.dot(x, w_hidden) + b_hidden)", y_hidden)
    if debug: print("np.dot(y_hidden, w_output)", np.dot(y_hidden, w_output))
    if debug: print("np.dot(y_hidden, w_output) + b_output", np.dot(y_hidden, w_output) + b_output)
    y_output = sigmoid(np.dot(y_hidden, w_output) + b_output)
    if debug: print("y_output = sigmoid(np.dot(y_hidden, w_output) + b_output)", y_output)

    # back prop
    grad_output = error_derivative(y, y_output) * sigmoid_derivative(y_output)
    if debug: print("grad_output = error_derivative(y, y_output) * sigmoid_derivative(y_output)", grad_output)
    if debug: print("grad_output", grad_output)
    if debug: print("w_output.T", w_output.T)
    if debug: print("grad_output.dot(w_output.T)", grad_output.dot(w_output.T))
    grad_hidden = grad_output.dot(w_output.T) * sigmoid_derivative(y_hidden)
    if debug: print("sigmoid_derivative(y_hidden)", sigmoid_derivative(y_hidden))
    if debug: print("grad_hidden = grad_output.dot(w_output.T) * sigmoid_derivative(y_hidden)", grad_hidden)

    # update parameters
    if debug: print("y_hidden.T.dot(grad_output)", y_hidden.T.dot(grad_output))
    w_output -= alpha * y_hidden.T.dot(grad_output)
    if debug: print("w_output -= alpha * y_hidden.T.dot(grad_output)", w_output)
    w_hidden -= alpha * x.T.dot(grad_hidden)
    if debug: print("w_hidden -= alpha * x.T.dot(grad_hidden)", w_hidden)

    if debug: print("np.sum(grad_output)", np.sum(grad_output))
    b_output -= alpha * grad_output
    if debug: print("b_output -= alpha * np.sum(grad_output)", b_output)
    if debug: print("np.sum(grad_hidden)", np.sum(grad_hidden))
    b_hidden -= alpha * grad_hidden
    if debug: print("b_hidden -= alpha * np.sum(grad_hidden)", b_hidden)
    if debug: break

ys_hidden = sigmoid(np.dot(xs, w_hidden) + b_hidden)
ys_output = sigmoid(np.dot(ys_hidden, w_output) + b_output)
print(ys_output)
