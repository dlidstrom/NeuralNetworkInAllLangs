const std = @import("std");
const Trainer = @import("neural.zig").Trainer;

fn rand() f64 {
    const P: u32 = 2147483647;
    const A: u32 = 16807;
    const S = struct {
        var current: u32 = 1;
    };

    S.current = S.current *% A % P;

    return @as(f64, @floatFromInt(S.current)) / @as(f64, @floatFromInt(P));
}

fn _xor(i: u32, j: u32) u32 {
    return i ^ j;
}

fn _xnor(i: u32, j: u32) u32 {
    return 1 - _xor(i, j);
}

fn _or(i: u32, j: u32) u32 {
    return i | j;
}

fn _and(i: u32, j: u32) u32 {
    return i & j;
}

fn _nor(i: u32, j: u32) u32 {
    return 1 - _or(i, j);
}

fn _nand(i: u32, j: u32) u32 {
    return 1 - _and(i, j);
}

fn DataItem(comptime I: usize, comptime O: usize) type {
    return struct {
        input: [I]f64,
        output: [O]f64,
    };
}

pub fn main() !void {
    var dba = std.heap.DebugAllocator(.{}){};
    defer std.debug.assert(dba.deinit() == .ok);
    const allocator = dba.allocator();

    var all_data = try allocator.alloc(DataItem(2, 6), 4);
    defer allocator.free(all_data);

    for ([_]u32{ 0, 1 }) |i| {
        for ([_]u32{ 0, 1 }) |j| {
            all_data[i * 2 + j] = .{
                .input = .{ @floatFromInt(i), @floatFromInt(j) },
                .output = .{
                    @floatFromInt(_xor(i, j)),
                    @floatFromInt(_xnor(i, j)),
                    @floatFromInt(_or(i, j)),
                    @floatFromInt(_and(i, j)),
                    @floatFromInt(_nor(i, j)),
                    @floatFromInt(_nand(i, j)),
                },
            };
        }
    }

    var trainer = try Trainer.init(allocator, 2, 2, 6, rand);
    defer trainer.deinit(allocator);

    const steps = 4000;
    const lr: f64 = 1.0;

    for (0..steps) |i| {
        var example = all_data[i % 4];
        try trainer.train(&example.input, &example.output, lr);
    }

    std.debug.print("Result after {d} iterations\n", .{steps});
    std.debug.print("        XOR  XNOR    OR   AND   NOR  NAND\n", .{});
    for (0..all_data.len) |i| {
        var example = all_data[i];
        const pred = try trainer.network.predict(allocator, &example.input);
        defer allocator.free(pred);
        std.debug.print(
            "{d:.0},{d:.0} = {d:.3} {d:.3} {d:.3} {d:.3} {d:.3} {d:.3}\n",
            .{
                example.input[0],
                example.input[1],
                pred[0],
                pred[1],
                pred[2],
                pred[3],
                pred[4],
                pred[5],
            },
        );
    }

    trainer.network.print();
}
