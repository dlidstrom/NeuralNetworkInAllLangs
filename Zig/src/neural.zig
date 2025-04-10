const std = @import("std");

const RuntimeError: type = error{
    InputSizeMismatch,
    HiddenSizeMismatch,
    OutputSizeMismatch,
};

fn sigmoid(x: f64) f64 {
    return 1 / (1 + std.math.exp(-x));
}

fn sigmoidPrime(d: f64) f64 {
    return d * (1 - d);
}

pub const Network = struct {
    const Self = @This();

    input_count: usize,
    hidden_count: usize,
    output_count: usize,
    weights_hidden: []f64,
    biases_hidden: []f64,
    weights_output: []f64,
    biases_output: []f64,

    pub fn predict(
        self: *Self,
        allocator: std.mem.Allocator,
        input: []f64,
    ) ![]f64 {
        const hidden = try allocator.alloc(f64, self.hidden_count);
        defer allocator.free(hidden);

        const output = try allocator.alloc(f64, self.output_count);

        return self.predictInplace(input, hidden, output);
    }

    pub fn predictInplace(
        self: *Self,
        input: []f64,
        hidden: []f64,
        output: []f64,
    ) ![]f64 {
        if (input.len != self.input_count) {
            return RuntimeError.InputSizeMismatch;
        }

        if (hidden.len != self.hidden_count) {
            return RuntimeError.InputSizeMismatch;
        }

        if (output.len != self.output_count) {
            return RuntimeError.InputSizeMismatch;
        }

        for (0..hidden.len) |h| {
            var sum: f64 = 0;
            for (0..input.len) |i| {
                sum += input[i] * self.weights_hidden[i * hidden.len + h];
            }

            hidden[h] = sigmoid(sum + self.biases_hidden[h]);
        }

        for (0..output.len) |o| {
            var sum: f64 = 0;
            for (0..hidden.len) |h| {
                sum += hidden[h] * self.weights_output[h * output.len + o];
            }

            output[o] = sigmoid(sum + self.biases_output[o]);
        }

        return output;
    }

    pub fn print(self: *Self) void {
        std.debug.print("weightsHidden:\n", .{});
        for (0..self.weights_hidden.len) |i| {
            std.debug.print("{d:.6} ", .{self.weights_hidden[i]});
        }
        std.debug.print("\n", .{});

        std.debug.print("biasesHidden:\n", .{});
        for (0..self.biases_hidden.len) |i| {
            std.debug.print("{d:.6} ", .{self.biases_hidden[i]});
        }
        std.debug.print("\n", .{});

        std.debug.print("weightsOutput:\n", .{});
        for (0..self.weights_output.len) |i| {
            std.debug.print("{d:.6} ", .{self.weights_output[i]});
        }
        std.debug.print("\n", .{});

        std.debug.print("biasesOutput:\n", .{});
        for (0..self.biases_output.len) |i| {
            std.debug.print("{d:.6} ", .{self.biases_output[i]});
        }
        std.debug.print("\n", .{});
    }
};

pub const Trainer = struct {
    const Self = @This();

    network: Network,
    hidden: []f64,
    output: []f64,
    grad_hidden: []f64,
    grad_output: []f64,

    pub fn init(
        allocator: std.mem.Allocator,
        input_count: usize,
        hidden_count: usize,
        output_count: usize,
        rand: fn () f64,
    ) !Self {
        const network = Network{
            .input_count = input_count,
            .hidden_count = hidden_count,
            .output_count = output_count,

            .weights_hidden = try allocator.alloc(f64, input_count * hidden_count),
            .biases_hidden = try allocator.alloc(f64, hidden_count),

            .weights_output = try allocator.alloc(f64, hidden_count * output_count),
            .biases_output = try allocator.alloc(f64, output_count),
        };

        for (0..network.weights_hidden.len) |i| {
            network.weights_hidden[i] = rand() - 0.5;
        }
        for (0..network.weights_output.len) |i| {
            network.weights_output[i] = rand() - 0.5;
        }
        for (0..network.biases_hidden.len) |i| {
            network.biases_hidden[i] = 0;
        }
        for (0..network.biases_output.len) |i| {
            network.biases_output[i] = 0;
        }

        const self = Self{
            .network = network,
            .hidden = try allocator.alloc(f64, hidden_count),
            .output = try allocator.alloc(f64, output_count),
            .grad_hidden = try allocator.alloc(f64, hidden_count),
            .grad_output = try allocator.alloc(f64, output_count),
        };

        for (0..self.hidden.len) |i| {
            self.hidden[i] = 0;
        }
        for (0..self.output.len) |i| {
            self.output[i] = 0;
        }
        for (0..self.grad_hidden.len) |i| {
            self.grad_hidden[i] = 0;
        }
        for (0..self.grad_output.len) |i| {
            self.grad_output[i] = 0;
        }

        return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.network.biases_hidden);
        allocator.free(self.network.biases_output);
        allocator.free(self.network.weights_hidden);
        allocator.free(self.network.weights_output);

        allocator.free(self.hidden);
        allocator.free(self.output);
        allocator.free(self.grad_hidden);
        allocator.free(self.grad_output);
    }

    pub fn train(self: *Self, input: []f64, y: []f64, lr: f64) !void {
        if (y.len != self.output.len) {
            return RuntimeError.OutputSizeMismatch;
        }

        _ = try self.network.predictInplace(
            input,
            self.hidden,
            self.output,
        );

        for (0..self.grad_output.len) |o| {
            self.grad_output[o] = (self.output[o] - y[o]) * sigmoidPrime(self.output[o]);
        }

        // ∂L/∂z0 = [(a1 - y) * a1 * (1 - a1)] * w1 * a0 * (1 - a0)
        for (0..self.grad_hidden.len) |h| {
            var sum: f64 = 0;
            for (0..self.grad_output.len) |o| {
                sum += self.grad_output[o] * self.network.weights_output[h * self.output.len + o];
            }

            self.grad_hidden[h] = sum * sigmoidPrime(self.hidden[h]);
        }

        for (0..self.hidden.len) |h| {
            for (0..self.output.len) |o| {
                self.network.weights_output[h * self.output.len + o] -= lr * self.grad_output[o] * self.hidden[h];
            }
        }

        for (0..input.len) |i| {
            for (0..self.hidden.len) |h| {
                self.network.weights_hidden[i * self.hidden.len + h] -= lr * self.grad_hidden[h] * input[i];
            }
        }

        for (0..self.network.biases_output.len) |o| {
            self.network.biases_output[o] -= lr * self.grad_output[o];
        }

        for (0..self.network.biases_hidden.len) |h| {
            self.network.biases_hidden[h] -= lr * self.grad_hidden[h];
        }
    }
};
