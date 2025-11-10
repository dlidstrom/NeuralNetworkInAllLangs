/*
    Licensed under the MIT License.
    Copyright 2023-2025 Daniel Lidstrom
*/

#[path= "rnd.rs"] pub mod rnd;

use rnd::Rnd;

fn sigmoid(f: f64) -> f64 {
    1.0 / (1.0 + (-f).exp())
}

fn sigmoid_prim(f: f64) -> f64 {
    f * (1.0 - f)
}

pub type Vector = Vec<f64>;
pub type Matrix = Vec<Vector>;

#[derive(Debug)]
pub struct Network {
    pub weights_hidden: Vector,
    pub biases_hidden: Vector,
    pub weights_output: Vector,
    pub biases_output: Vector,
    n_inputs: usize,
    n_hidden: usize,
    n_outputs: usize
}
impl Network {
    pub fn new(
        n_inputs: usize,
        n_hidden: usize,
        n_outputs: usize,
        weights_hidden: Vector,
        biases_hidden: Vector,
        weights_output: Vector,
        biases_output: Vector) -> Network {
        Network {
            n_inputs,
            n_hidden,
            n_outputs,
            weights_hidden,
            biases_hidden,
            weights_output,
            biases_output
        }
    }

    pub fn predict(&self, input: &Vector) -> Vector {
        let mut y_hidden = vec![0.0; self.n_hidden];
        let mut y_output = vec![0.0; self.n_outputs];
        self.predict_inplace(&input, &mut y_hidden, &mut y_output);
        y_output
    }

    pub fn predict_inplace(&self, input: &Vector, y_hidden: &mut Vector, y_output: &mut Vector) -> () {
        for c in 0..self.n_hidden {
            let mut sum = 0.0;
            for r in 0..self.n_inputs {
                sum += input[r] * self.weights_hidden[r * self.n_hidden + c]
            }

            y_hidden[c] = sigmoid(sum + self.biases_hidden[c]);
        }

        for c in 0..self.n_outputs {
            let mut sum = 0.0;
            for r in 0..self.n_hidden {
                sum += y_hidden[r] * self.weights_output[r * self.n_outputs + c]
            }

            y_output[c] = sigmoid(sum + self.biases_output[c])
        }
    }
}

pub struct Trainer {
    pub network: Network,
    y_hidden: Vector,
    y_output: Vector,
    grad_hidden: Vector,
    grad_output: Vector
}
impl Trainer {
    pub fn load(
        network: Network,
        n_hidden: usize,
        n_outputs: usize) -> Trainer {
        let y_hidden = vec![0.0; n_hidden];
        let y_output = vec![0.0; n_outputs];
        let grad_hidden = vec![0.0; n_hidden];
        let grad_output = vec![0.0; n_outputs];
        Trainer { network, y_hidden, y_output, grad_hidden, grad_output }
    }

    pub fn new(n_inputs: usize, n_hidden: usize, n_outputs: usize, rnd: &mut Rnd) -> Trainer {
        let mut weights_hidden = Vector::new();
        for _ in 0..n_inputs * n_hidden {
            weights_hidden.push(rnd.next_float() - 0.5);
        }

        let biases_hidden = vec![0.0; n_hidden];
        let mut weights_output = Vector::new();
        for _ in 0..n_hidden * n_outputs {
            weights_output.push(rnd.next_float() - 0.5);
        }

        let biases_output = vec![0.0; n_outputs];
        let network = Network::new(
            n_inputs,
            n_hidden,
            n_outputs,
            weights_hidden,
            biases_hidden,
            weights_output,
            biases_output);
        Trainer::load(network, n_hidden, n_outputs)
    }

    pub fn train(&mut self, input: &Vector, y: &Vector, lr: f64) -> () {
        self.network.predict_inplace(&input, &mut self.y_hidden, &mut self.y_output);
        for c in 0..self.network.n_outputs {
            self.grad_output[c] = (self.y_output[c] - y[c]) * sigmoid_prim(self.y_output[c]);
        }

        for r in 0..self.network.n_hidden {
            let mut sum = 0.0;
            for c in 0..self.network.n_outputs {
                sum += self.grad_output[c] * self.network.weights_output[r * self.network.n_outputs + c];
            }

            self.grad_hidden[r] = sum * sigmoid_prim(self.y_hidden[r]);
        }

        for r in 0..self.network.n_hidden {
            for c in 0..self.network.n_outputs {
                self.network.weights_output[r * self.network.n_outputs + c] -= lr * self.grad_output[c] * self.y_hidden[r];
            }
        }

        for r in 0..self.network.n_inputs {
            for c in 0..self.network.n_hidden {
                self.network.weights_hidden[r * self.network.n_hidden + c] -= lr * self.grad_hidden[c] * input[r];
            }
        }

        for c in 0..self.network.n_outputs {
            self.network.biases_output[c] -= lr * self.grad_output[c];
        }

        for c in 0..self.network.n_hidden {
            self.network.biases_hidden[c] -= lr * self.grad_hidden[c];
        }
    }
}
