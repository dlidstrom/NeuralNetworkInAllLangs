/*
    Licensed under the MIT License.
    Copyright 2023-2025 Daniel Lidstrom
*/

const P: u32 = 2147483647;
const A: u32 = 16807;
pub struct Rnd {
    current: u32
}
impl Rnd {
    pub fn new() -> Rnd {
        Rnd { current: 1 }
    }
    pub fn next(&mut self) -> u32 {
        self.current = self.current.wrapping_mul(A) % P;
        self.current
    }
    pub fn next_float(&mut self) -> f64 {
        let u = self.next();
        let result = (u) as f64 / P as f64;
        result
    }
}
