//! Tiny xorshift64* PRNG. Replaces the template's `rand` dependency: the only
//! randomness here is visual (matrix rain columns, rain glyph jitter), and a
//! dependency-free generator keeps the on-box self-hosted build trivial.
//!
//! Everything on the render path seeds through [`Rng::with_seed`], never from
//! the wall clock or addresses: the selftest checksums are the M3 acceptance
//! test (`docs/fbdev-wgpu-plan.md` — the wgpu path must produce the same
//! frames as `softrender`), so rendering must be a pure function of the
//! scene and the frame time. The rain still looks random because the seeds
//! differ per scene and the frame time never repeats.

pub struct Rng(u64);

impl Rng {
    /// Deterministic constructor. The seed is run through the splitmix64
    /// finalizer so any input — a constant, frame-time bits, `0` — lands on
    /// a healthy xorshift state (state `0` would stall the generator).
    pub fn with_seed(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Rng((z ^ (z >> 31)) | 1)
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
