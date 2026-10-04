//! The lowered shader program: a flat register machine over 32-bit words.
//!
//! `compile.rs` produces it from naga IR; `vm.rs` interprets it; `jit.rs`
//! translates it to x86-64. Every value is one scalar in one 32-bit register
//! (f32/i32/u32 bit patterns, bool as 0/1); vectors, matrices, structs and
//! arrays are scalarized away by the compiler. Register 0 is always zero.
//!
//! Registers written by exactly one `Const` are *hoisted* into
//! `Program::init` and never appear in `code`: an invocation context
//! initializes its register file from `init` once and thereafter only
//! rewrites what the code writes.
//!
//! Transcendentals and the few ops whose semantics differ between Rust and
//! the CPU's native instructions (NaN-aware min/max, saturating float→int
//! casts, integer division by zero, ...) go through `Fun` helpers: plain
//! `extern "C" fn(u32, u32) -> u32` taking and returning raw bits. The VM and
//! the JIT call the *same* helper, so they agree bit for bit by construction,
//! and the helpers are the same Rust `f32` methods the interpreter uses.

pub type R = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    FEq, FNe, FLt, FLe, FGt, FGe,
    IEq, INe,
    SLt, SLe, SGt, SGe,
    ULt, ULe, UGt, UGe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fun {
    Sin, Cos, Tan, Sinh, Cosh, Tanh, Asin, Acos, Atan, Asinh, Acosh, Atanh,
    Atan2, Pow, Exp, Exp2, Ln, Log2,
    Ceil, Floor, Round, Trunc, Sign, FMin, FMax, FRem, Clamp01, IsInf,
    F2I, F2U,
    IAbs, ISign, IDivS, IRemS, IDivU, IRemU,
}

#[derive(Clone, Copy, Debug)]
pub enum Inst {
    Mov { d: R, s: R },
    /// a write-per-invocation constant (local-variable zero init); SSA
    /// constants are hoisted into `Program::init` instead
    Const { d: R, v: u32 },
    FAdd { d: R, a: R, b: R },
    FSub { d: R, a: R, b: R },
    FMul { d: R, a: R, b: R },
    FDiv { d: R, a: R, b: R },
    FNeg { d: R, a: R },
    FAbs { d: R, a: R },
    Sqrt { d: R, a: R },
    IAdd { d: R, a: R, b: R },
    ISub { d: R, a: R, b: R },
    IMul { d: R, a: R, b: R },
    And { d: R, a: R, b: R },
    Or { d: R, a: R, b: R },
    Xor { d: R, a: R, b: R },
    Not { d: R, a: R },
    Shl { d: R, a: R, b: R },
    ShrS { d: R, a: R, b: R },
    ShrU { d: R, a: R, b: R },
    Cmp { d: R, a: R, b: R, c: Cmp },
    /// d = (c != 0) ? a : b
    Select { d: R, c: R, a: R, b: R },
    I2F { d: R, a: R },
    U2F { d: R, a: R },
    Call { d: R, a: R, b: R, f: Fun },
    /// `Call` with a one-entry result cache in four *persistent* registers
    /// `c..c+4` = (valid, key a, key b, result). The register file is only
    /// initialized once per draw, so the cache survives across invocations:
    /// shaders whose neighbouring pixels/vertices evaluate the same
    /// transcendental on the same arguments (flat cell colours, per-frame
    /// uniforms) skip libm almost every time.
    CallC { d: R, a: R, b: R, f: Fun, c: R },
    /// d = u32 at byte (regs[off] + imm) of buffer slot `buf`, 0 if out of range
    LoadBuf { d: R, buf: u32, off: R, imm: u32 },
    /// texture fetch / sample / size: `Program::tex_ops[op]`
    Tex { op: u32 },
    Jmp { t: u32 },
    Jz { c: R, t: u32 },
    Jnz { c: R, t: u32 },
    /// Start of memoized region `Program::memos[m]`: if the region's live-in
    /// registers match a cached entry, copy the cached live-out registers
    /// and jump to `t` (just past the matching `MemoPut`); else fall through
    /// and run the region.
    MemoGet { m: u32, t: u32 },
    /// End of region `m`: remember (live-ins -> live-outs) for the next `MemoGet`.
    MemoPut { m: u32 },
    Kill,
    Ret,
}

/// How a fragment `@location` input varies across a primitive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Flat,
    Linear,
    Perspective,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    VertexIndex,
    InstanceIndex,
    /// fragment `@builtin(position)` component
    Position(u8),
    /// `@location(l)` component: vertex attribute (vertex stage) or varying
    /// (fragment stage)
    Location(u32, u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dst {
    Position(u8),
    /// vertex: varying; fragment: color target `l`
    Location(u32, u8),
}

#[derive(Debug, Clone)]
pub struct Program {
    pub code: Vec<Inst>,
    pub nregs: u32,
    /// register file contents at invocation start (hoisted constants)
    pub init: Vec<u32>,
    /// (group, binding) per buffer slot used by `LoadBuf`
    pub bufs: Vec<(u32, u32)>,
    pub inputs: Vec<(R, Src)>,
    pub outputs: Vec<(Dst, R)>,
    /// (location, interpolation) of each `@location` input (fragment stage)
    pub interp: Vec<(u32, Interp)>,
    /// (group, binding) per texture slot / sampler slot used by `Tex`
    pub texs: Vec<(u32, u32)>,
    pub smps: Vec<(u32, u32)>,
    /// operands of `Inst::Tex`. Heap-stable: the JIT embeds element addresses,
    /// so this Vec must never be grown after `jit::compile`.
    pub tex_ops: Vec<super::texture::TexOp>,
    /// memoized regions (see `memo.rs`). Heap-stable like `tex_ops`.
    pub memos: Vec<MemoInfo>,
    /// where the result is constant along rows / columns of pixels (see `runs.rs`)
    pub runs: super::runs::RunInfo,
}

/// A pure region of the program whose result depends only on a few registers:
/// a direct-mapped cache in persistent registers maps those (`ins`) to the
/// region's results (`outs`). The cache survives across invocations (the
/// register file is initialized once per draw) and is private to the
/// invoker, i.e. to one thread.
#[derive(Debug, Clone)]
pub struct MemoInfo {
    pub ins: Vec<R>,
    pub outs: Vec<R>,
    /// first register of the table: `1 << bits` entries of `1 + ins + outs`
    /// registers: [valid, keys.., results..]
    pub base: R,
    pub bits: u32,
    /// register holding the entry `MemoGet` selected, for `MemoPut`
    pub slot: R,
}

impl MemoInfo {
    pub fn entry_words(&self) -> u32 {
        1 + (self.ins.len() + self.outs.len()) as u32
    }
    pub fn table_regs(&self) -> u32 {
        self.entry_words() << self.bits
    }
}

/// `slot` value: this invocation must not update the cache
const NO_SLOT: u32 = u32::MAX;

fn memo_hash(m: &MemoInfo, key: impl Fn(usize) -> u32) -> u32 {
    // Fibonacci hashing: for one key, consecutive values (cell indices) land in
    // distinct slots
    let mut h = 0u32;
    for i in 0..m.ins.len() {
        h = (h.rotate_left(5) ^ key(i)).wrapping_mul(0x9E37_79B1);
    }
    h >> (32 - m.bits)
}

/// `MemoGet`: returns 1 on a hit (the live-outs are already written). `regs`
/// is the register file with `stride` words per register (1 scalar, 4 wide:
/// register r, lane l at r*stride + l); cache entries only use lane 0, and a
/// wide batch whose lanes disagree on a key neither hits nor updates the cache.
///
/// # Safety
/// `regs` must point to a register file holding every register `m` names.
pub unsafe extern "C" fn memo_get(regs: *mut u32, m: *const MemoInfo, stride: u32) -> u32 {
    let st = stride as usize;
    unsafe {
        let m = &*m;
        let at = |r: R, lane: usize| regs.add(r as usize * st + lane);
        let slot = at(m.slot, 0);
        for &k in &m.ins {
            let v = *at(k, 0);
            for lane in 1..st {
                if *at(k, lane) != v {
                    *slot = NO_SLOT;
                    return 0;
                }
            }
        }
        let e = m.base + memo_hash(m, |i| *at(m.ins[i], 0)) * m.entry_words();
        *slot = e;
        if *at(e, 0) == 0 {
            return 0;
        }
        for (i, &k) in m.ins.iter().enumerate() {
            if *at(e + 1 + i as u32, 0) != *at(k, 0) {
                return 0;
            }
        }
        let o = e + 1 + m.ins.len() as u32;
        for (j, &r) in m.outs.iter().enumerate() {
            let v = *at(o + j as u32, 0);
            if st == 4 {
                // one 16-byte store: the caller reloads the register as a
                // vector, and four 4-byte stores would defeat store forwarding
                (at(r, 0) as *mut [u32; 4]).write_unaligned([v; 4]);
            } else {
                for lane in 0..st {
                    *at(r, lane) = v;
                }
            }
        }
        1
    }
}

/// `MemoPut`
///
/// # Safety
/// As for `memo_get`.
pub unsafe extern "C" fn memo_put(regs: *mut u32, m: *const MemoInfo, stride: u32) {
    let st = stride as usize;
    unsafe {
        let m = &*m;
        let at = |r: R| regs.add(r as usize * st);
        let e = *at(m.slot);
        if e == NO_SLOT {
            return;
        }
        *at(e) = 1;
        for (i, &k) in m.ins.iter().enumerate() {
            *at(e + 1 + i as u32) = *at(k);
        }
        let o = e + 1 + m.ins.len() as u32;
        for (j, &r) in m.outs.iter().enumerate() {
            *at(o + j as u32) = *at(r);
        }
    }
}

impl Inst {
    /// the register this instruction writes, if it writes exactly one
    /// (`Tex` writes several: see `Program::tex_ops`; `CallC` also updates its
    /// cache registers)
    pub fn dst(&self) -> Option<R> {
        use Inst::*;
        match *self {
            Mov { d, .. } | Const { d, .. } | FAdd { d, .. } | FSub { d, .. } | FMul { d, .. }
            | FDiv { d, .. } | FNeg { d, .. } | FAbs { d, .. } | Sqrt { d, .. } | IAdd { d, .. }
            | ISub { d, .. } | IMul { d, .. } | And { d, .. } | Or { d, .. } | Xor { d, .. }
            | Not { d, .. } | Shl { d, .. } | ShrS { d, .. } | ShrU { d, .. } | Cmp { d, .. }
            | Select { d, .. } | I2F { d, .. } | U2F { d, .. } | Call { d, .. } | CallC { d, .. }
            | LoadBuf { d, .. } => Some(d),
            Tex { .. } | Jmp { .. } | Jz { .. } | Jnz { .. } | MemoGet { .. } | MemoPut { .. }
            | Kill | Ret => None,
        }
    }

    /// redirect the single destination register (no-op for instructions without one)
    pub fn set_dst(&mut self, r: R) {
        use Inst::*;
        match self {
            Mov { d, .. } | Const { d, .. } | FAdd { d, .. } | FSub { d, .. } | FMul { d, .. }
            | FDiv { d, .. } | FNeg { d, .. } | FAbs { d, .. } | Sqrt { d, .. } | IAdd { d, .. }
            | ISub { d, .. } | IMul { d, .. } | And { d, .. } | Or { d, .. } | Xor { d, .. }
            | Not { d, .. } | Shl { d, .. } | ShrS { d, .. } | ShrU { d, .. } | Cmp { d, .. }
            | Select { d, .. } | I2F { d, .. } | U2F { d, .. } | Call { d, .. } | CallC { d, .. }
            | LoadBuf { d, .. } => *d = r,
            Tex { .. } | Jmp { .. } | Jz { .. } | Jnz { .. } | MemoGet { .. } | MemoPut { .. } | Kill | Ret => {}
        }
    }

    /// rewrite every register operand this instruction reads (not a `CallC`'s
    /// cache registers, and not texture coordinates, which live in `tex_ops`)
    pub fn map_reads(&mut self, mut f: impl FnMut(R) -> R) {
        use Inst::*;
        match self {
            Mov { s, .. } => *s = f(*s),
            Const { .. } | Jmp { .. } | Kill | Ret | Tex { .. } | MemoGet { .. } | MemoPut { .. } => {}
            FAdd { a, b, .. } | FSub { a, b, .. } | FMul { a, b, .. } | FDiv { a, b, .. }
            | IAdd { a, b, .. } | ISub { a, b, .. } | IMul { a, b, .. } | And { a, b, .. }
            | Or { a, b, .. } | Xor { a, b, .. } | Shl { a, b, .. } | ShrS { a, b, .. }
            | ShrU { a, b, .. } | Cmp { a, b, .. } | Call { a, b, .. } | CallC { a, b, .. } => {
                *a = f(*a);
                *b = f(*b);
            }
            FNeg { a, .. } | FAbs { a, .. } | Sqrt { a, .. } | Not { a, .. } | I2F { a, .. }
            | U2F { a, .. } => *a = f(*a),
            Select { c, a, b, .. } => {
                *c = f(*c);
                *a = f(*a);
                *b = f(*b);
            }
            LoadBuf { off, .. } => *off = f(*off),
            Jz { c, .. } | Jnz { c, .. } => *c = f(*c),
        }
    }

    /// call `f` with every register this instruction reads
    pub fn reads(&self, mut f: impl FnMut(R)) {
        use Inst::*;
        match *self {
            Mov { s, .. } => f(s),
            Const { .. } | Jmp { .. } | Kill | Ret => {}
            FAdd { a, b, .. } | FSub { a, b, .. } | FMul { a, b, .. } | FDiv { a, b, .. }
            | IAdd { a, b, .. } | ISub { a, b, .. } | IMul { a, b, .. } | And { a, b, .. }
            | Or { a, b, .. } | Xor { a, b, .. } | Shl { a, b, .. } | ShrS { a, b, .. }
            | ShrU { a, b, .. } | Cmp { a, b, .. } | Call { a, b, .. } => {
                f(a);
                f(b);
            }
            FNeg { a, .. } | FAbs { a, .. } | Sqrt { a, .. } | Not { a, .. } | I2F { a, .. }
            | U2F { a, .. } => f(a),
            Select { c, a, b, .. } => {
                f(c);
                f(a);
                f(b);
            }
            // the cache registers are read and written by the instruction itself
            CallC { a, b, c, .. } => {
                f(a);
                f(b);
                for i in 0..4 {
                    f(c + i);
                }
            }
            LoadBuf { off, .. } => f(off),
            // texture ops read their coordinates (Size reads none); memo
            // instructions touch registers named by `Program::memos`
            Tex { .. } | MemoGet { .. } | MemoPut { .. } => {}
            Jz { c, .. } | Jnz { c, .. } => f(c),
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

pub type Helper = extern "C" fn(u32, u32) -> u32;

#[inline]
fn f(x: u32) -> f32 {
    f32::from_bits(x)
}
#[inline]
fn b(x: f32) -> u32 {
    x.to_bits()
}

macro_rules! h1 {
    ($name:ident, |$a:ident| $e:expr) => {
        extern "C" fn $name(a: u32, _b: u32) -> u32 {
            let $a = f(a);
            b($e)
        }
    };
}
macro_rules! h2 {
    ($name:ident, |$a:ident, $b2:ident| $e:expr) => {
        extern "C" fn $name(a: u32, bb: u32) -> u32 {
            let $a = f(a);
            let $b2 = f(bb);
            b($e)
        }
    };
}

h1!(h_sin, |x| x.sin());
h1!(h_cos, |x| x.cos());
h1!(h_tan, |x| x.tan());
h1!(h_sinh, |x| x.sinh());
h1!(h_cosh, |x| x.cosh());
h1!(h_tanh, |x| x.tanh());
h1!(h_asin, |x| x.asin());
h1!(h_acos, |x| x.acos());
h1!(h_atan, |x| x.atan());
h1!(h_asinh, |x| x.asinh());
h1!(h_acosh, |x| x.acosh());
h1!(h_atanh, |x| x.atanh());
h2!(h_atan2, |x, y| x.atan2(y));
h2!(h_pow, |x, y| x.powf(y));
h1!(h_exp, |x| x.exp());
h1!(h_exp2, |x| x.exp2());
h1!(h_ln, |x| x.ln());
h1!(h_log2, |x| x.log2());
h1!(h_ceil, |x| x.ceil());
h1!(h_floor, |x| x.floor());
h1!(h_round, |x| x.round());
h1!(h_trunc, |x| x.trunc());
h1!(h_sign, |x| match x.partial_cmp(&0.0) {
    Some(std::cmp::Ordering::Greater) => 1.0,
    Some(std::cmp::Ordering::Less) => -1.0,
    _ => 0.0,
});
h2!(h_fmin, |x, y| x.min(y));
h2!(h_fmax, |x, y| x.max(y));
h2!(h_frem, |x, y| x % y);
// smoothstep's clamp(t, 0, 1): NaN stays NaN (unlike min/max chains)
h1!(h_clamp01, |x| x.clamp(0.0, 1.0));
extern "C" fn h_isinf(a: u32, _b: u32) -> u32 {
    f(a).is_infinite() as u32
}
extern "C" fn h_f2i(a: u32, _b: u32) -> u32 {
    (f(a) as i32) as u32
}
extern "C" fn h_f2u(a: u32, _b: u32) -> u32 {
    f(a) as u32
}
extern "C" fn h_iabs(a: u32, _b: u32) -> u32 {
    (a as i32).wrapping_abs() as u32
}
extern "C" fn h_isign(a: u32, _b: u32) -> u32 {
    (a as i32).signum() as u32
}
// WGSL leaves division by zero (and INT_MIN / -1) indeterminate; we return
// the dividend / 0 instead of trapping, like most GPUs.
extern "C" fn h_idivs(a: u32, bb: u32) -> u32 {
    if bb == 0 { a } else { (a as i32).wrapping_div(bb as i32) as u32 }
}
extern "C" fn h_irems(a: u32, bb: u32) -> u32 {
    if bb == 0 { 0 } else { (a as i32).wrapping_rem(bb as i32) as u32 }
}
extern "C" fn h_idivu(a: u32, bb: u32) -> u32 {
    if bb == 0 { a } else { a / bb }
}
extern "C" fn h_iremu(a: u32, bb: u32) -> u32 {
    if bb == 0 { 0 } else { a % bb }
}

impl Fun {
    /// worth a result cache: libm-class cost (tens of ns), pure in (a, b)
    pub fn cacheable(self) -> bool {
        use Fun::*;
        matches!(
            self,
            Sin | Cos | Tan | Sinh | Cosh | Tanh | Asin | Acos | Atan | Asinh | Acosh | Atanh
                | Atan2 | Pow | Exp | Exp2 | Ln | Log2
        )
    }

    pub fn helper(self) -> Helper {
        use Fun::*;
        match self {
            Sin => h_sin, Cos => h_cos, Tan => h_tan, Sinh => h_sinh, Cosh => h_cosh,
            Tanh => h_tanh, Asin => h_asin, Acos => h_acos, Atan => h_atan,
            Asinh => h_asinh, Acosh => h_acosh, Atanh => h_atanh, Atan2 => h_atan2,
            Pow => h_pow, Exp => h_exp, Exp2 => h_exp2, Ln => h_ln, Log2 => h_log2,
            Ceil => h_ceil, Floor => h_floor, Round => h_round, Trunc => h_trunc,
            Sign => h_sign, FMin => h_fmin, FMax => h_fmax, FRem => h_frem,
            Clamp01 => h_clamp01, IsInf => h_isinf, F2I => h_f2i, F2U => h_f2u,
            IAbs => h_iabs, ISign => h_isign, IDivS => h_idivs, IRemS => h_irems,
            IDivU => h_idivu, IRemU => h_iremu,
        }
    }
}

/// Evaluate a comparison on raw words (shared by the VM and by constant
/// folding; the JIT emits the equivalent flag sequences).
#[inline]
pub fn cmp(c: Cmp, a: u32, bb: u32) -> bool {
    match c {
        Cmp::FEq => f(a) == f(bb),
        Cmp::FNe => f(a) != f(bb),
        Cmp::FLt => f(a) < f(bb),
        Cmp::FLe => f(a) <= f(bb),
        Cmp::FGt => f(a) > f(bb),
        Cmp::FGe => f(a) >= f(bb),
        Cmp::IEq => a == bb,
        Cmp::INe => a != bb,
        Cmp::SLt => (a as i32) < (bb as i32),
        Cmp::SLe => (a as i32) <= (bb as i32),
        Cmp::SGt => (a as i32) > (bb as i32),
        Cmp::SGe => (a as i32) >= (bb as i32),
        Cmp::ULt => a < bb,
        Cmp::ULe => a <= bb,
        Cmp::UGt => a > bb,
        Cmp::UGe => a >= bb,
    }
}
