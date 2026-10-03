//! Tiny xorshift64* PRNG. Replaces the template's `rand` dependency: the only
//! randomness here is visual (matrix rain columns, rain glyph jitter), and a
//! dependency-free generator keeps the on-box self-hosted build trivial.
//!
//! Seeded from CLOCK_REALTIME nanoseconds xor'd with a stack address, which is
//! plenty for falling glyphs.

use std::time::{SystemTime, UNIX_EPOCH};

pub struct Rng(u64);

impl Rng {
    pub fn new() -> Self {
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15);
        let stack = &t as *const _ as u64;
        Rng(t ^ stack.rotate_left(17) ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `0..n` (n > 0).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// True with probability `pct` percent.
    pub fn pct(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
}
