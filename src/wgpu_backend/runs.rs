//! Run detection for fragment programs whose result varies with the pixel
//! position only through quantization.
//!
//! A terminal's cell-background shader computes `floor((x - pad) / cell_w)`
//! and looks the colour up from that: the result is constant over each
//! cell's 16x32 pixels. If a program's dependence on `@builtin(position).x`
//! (and, separately, `.y`) provably flows only through such "step" values, the
//! rasterizer need not shade every pixel: after shading one it can ask how far
//! the step values stay unchanged and replicate the result over that run
//! (along x), and over whole rows (along y).
//!
//! The analysis is conservative. Per axis, every register is classified:
//!
//! * `Clean` — does not depend on that position component;
//! * `Aff`   — an affine-looking function of it, built from `+ - * /` and
//!   negation with `Clean` operands (kept as a chain, so it can be re-evaluated
//!   with exactly the same f32 operations the program uses);
//! * `Step`  — a quantization (`floor`/`ceil`/`round`/`trunc`/float->int) of an
//!   `Aff` value, or anything computed only from `Step` and `Clean` values;
//! * `Bad`   — anything else that touches the position.
//!
//! The program qualifies for an axis only if its outputs, branch conditions
//! and texture coordinates are never `Aff` or `Bad`. Each distinct
//! quantization is a *site*; a run ends where any site's value changes. All
//! f32 operations are monotone, so "same value at both ends" holds in
//! between, and the exact end of a run is found by evaluating the chain at a
//! handful of positions (exponential + binary search). `AKUMA_RUNS=0` disables
//! the whole mechanism.

use std::sync::atomic::{AtomicBool, Ordering};

use super::program::{Fun, Inst, Interp, Program, Src, R};
use super::texture::TexKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cls {
    Clean,
    Step,
    Aff,
    Bad,
}

#[derive(Clone, Copy, Debug)]
pub enum ChainKind {
    /// v + c
    Add,
    /// v - c
    Sub,
    /// c - v
    RSub,
    Mul,
    /// v / c
    Div,
    Neg,
}

#[derive(Clone, Copy, Debug)]
pub struct ChainOp {
    pub kind: ChainKind,
    /// the clean operand
    pub r: R,
}

#[derive(Clone, Debug)]
pub struct Site {
    pub chain: Vec<ChainOp>,
    pub f: Fun,
}

#[derive(Clone, Debug, Default)]
pub struct Axis {
    pub sites: Vec<Site>,
}

#[derive(Clone, Debug, Default)]
pub struct RunInfo {
    pub x: Option<Axis>,
    pub y: Option<Axis>,
}

/// tests flip this to compare the same draw with and without runs
pub static ENABLED: AtomicBool = AtomicBool::new(true);

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

impl Site {
    /// the site's step value at position `pos` (None if the chain leaves the finite numbers)
    fn id(&self, reg: &dyn Fn(R) -> u32, pos: f32) -> Option<u32> {
        let mut v = pos;
        for op in &self.chain {
            let c = f32::from_bits(reg(op.r));
            v = match op.kind {
                ChainKind::Add => v + c,
                ChainKind::Sub => v - c,
                ChainKind::RSub => c - v,
                ChainKind::Mul => v * c,
                ChainKind::Div => v / c,
                ChainKind::Neg => -v,
            };
        }
        if !v.is_finite() {
            return None;
        }
        Some(match self.f {
            Fun::Floor => v.floor().to_bits(),
            Fun::Ceil => v.ceil().to_bits(),
            Fun::Round => v.round().to_bits(),
            Fun::Trunc => v.trunc().to_bits(),
            Fun::F2I => (v as i32) as u32,
            _ => v as u32, // F2U
        })
    }
}

impl Axis {
    /// How many positions after `pos` (stepping by 1.0) keep every site's step
    /// value unchanged, up to `max`. `reg` reads the (position-independent)
    /// chain operands.
    pub fn extent(&self, reg: &dyn Fn(R) -> u32, pos: f32, max: usize) -> usize {
        let mut n = max;
        for s in &self.sites {
            if n == 0 {
                return 0;
            }
            let Some(id0) = s.id(reg, pos) else { return 0 };
            let same = |j: usize| s.id(reg, pos + j as f32) == Some(id0);
            if same(n) {
                continue; // monotone: constant all the way
            }
            let (mut lo, mut hi) = (0usize, n);
            let mut step = 1usize;
            loop {
                let t = lo + step;
                if t >= hi {
                    break;
                }
                if same(t) {
                    lo = t;
                    step *= 2;
                } else {
                    hi = t;
                    break;
                }
            }
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if same(mid) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            n = lo;
        }
        n
    }
}

fn site_fn(f: Fun) -> bool {
    matches!(f, Fun::Floor | Fun::Ceil | Fun::Round | Fun::Trunc | Fun::F2I | Fun::F2U)
}

struct Analysis {
    cls: Vec<Option<Cls>>,
    ok: bool,
    /// (instruction index, site)
    sites: Vec<(usize, Site)>,
}

fn join(a: Option<Cls>, b: Cls) -> Cls {
    match a {
        None => b,
        Some(x) if x == b => b,
        Some(Cls::Clean) if b == Cls::Step => Cls::Step,
        Some(Cls::Step) if b == Cls::Clean => Cls::Step,
        _ => Cls::Bad,
    }
}

fn classify(p: &Program, axis: u8, nw: &[u32]) -> Analysis {
    let nr = p.nregs as usize;
    let mut cls: Vec<Option<Cls>> = vec![None; nr];
    let mut chain: Vec<Vec<ChainOp>> = vec![Vec::new(); nr];
    let mut is_input = vec![false; nr];
    for &(r, src) in &p.inputs {
        is_input[r as usize] = true;
        cls[r as usize] = Some(match src {
            Src::Position(a) if a == axis => Cls::Aff,
            Src::Position(a) if a < 2 => Cls::Clean,
            Src::Location(l, _) => match p.interp.iter().find(|(x, _)| *x == l) {
                Some((_, Interp::Flat)) | None => Cls::Clean,
                _ => Cls::Bad,
            },
            _ => Cls::Bad,
        });
    }
    for r in 0..nr {
        if nw[r] == 0 && !is_input[r] {
            cls[r] = Some(Cls::Clean);
        }
    }
    let mut sites: Vec<(usize, Site)> = Vec::new();
    for _ in 0..10 {
        let mut changed = false;
        sites.clear();
        for (idx, inst) in p.code.iter().enumerate() {
            let c = |r: R| cls[r as usize];
            // (class, chain) of the destination, when all operands are known
            let new: Option<(Cls, Option<Vec<ChainOp>>)> = match *inst {
                Inst::Const { .. } => Some((Cls::Clean, None)),
                Inst::Mov { s, .. } => c(s).map(|k| (k, (k == Cls::Aff).then(|| chain[s as usize].clone()))),
                Inst::FAdd { a, b, .. } | Inst::FSub { a, b, .. } | Inst::FMul { a, b, .. } | Inst::FDiv { a, b, .. } => {
                    match (c(a), c(b)) {
                        (Some(x), Some(y)) => {
                            let kind = |flip: bool| match (inst, flip) {
                                (Inst::FAdd { .. }, _) => Some(ChainKind::Add),
                                (Inst::FMul { .. }, _) => Some(ChainKind::Mul),
                                (Inst::FSub { .. }, false) => Some(ChainKind::Sub),
                                (Inst::FSub { .. }, true) => Some(ChainKind::RSub),
                                (Inst::FDiv { .. }, false) => Some(ChainKind::Div),
                                _ => None,
                            };
                            match (x, y) {
                                (Cls::Aff, Cls::Clean) => match kind(false) {
                                    Some(k) => {
                                        let mut ch = chain[a as usize].clone();
                                        ch.push(ChainOp { kind: k, r: b });
                                        Some((Cls::Aff, Some(ch)))
                                    }
                                    None => Some((Cls::Bad, None)),
                                },
                                (Cls::Clean, Cls::Aff) => match kind(true) {
                                    Some(k) => {
                                        let mut ch = chain[b as usize].clone();
                                        ch.push(ChainOp { kind: k, r: a });
                                        Some((Cls::Aff, Some(ch)))
                                    }
                                    None => Some((Cls::Bad, None)),
                                },
                                _ => Some((generic(&[x, y]), None)),
                            }
                        }
                        _ => None,
                    }
                }
                Inst::FNeg { a, .. } => c(a).map(|k| {
                    if k == Cls::Aff {
                        let mut ch = chain[a as usize].clone();
                        ch.push(ChainOp { kind: ChainKind::Neg, r: 0 });
                        (Cls::Aff, Some(ch))
                    } else {
                        (generic(&[k]), None)
                    }
                }),
                Inst::Call { a, f, .. } if site_fn(f) => c(a).map(|k| {
                    if k == Cls::Aff {
                        sites.push((idx, Site { chain: chain[a as usize].clone(), f }));
                        (Cls::Step, None)
                    } else {
                        (generic(&[k]), None)
                    }
                }),
                Inst::Call { a, b, .. } | Inst::CallC { a, b, .. } => match (c(a), c(b)) {
                    (Some(x), Some(y)) => Some((generic(&[x, y]), None)),
                    _ => None,
                },
                Inst::Tex { op } => {
                    let t = &p.tex_ops[op as usize];
                    let ks = if t.kind == TexKind::Size { Some(vec![]) } else { c(t.x).zip(c(t.y)).map(|(x, y)| vec![x, y]) };
                    ks.map(|ks| (generic(&ks), None))
                }
                Inst::Select { c: cc, a, b, .. } => match (c(cc), c(a), c(b)) {
                    (Some(x), Some(y), Some(z)) => Some((generic(&[x, y, z]), None)),
                    _ => None,
                },
                _ => {
                    // every other single-destination instruction: generic over its operands
                    let mut ks = Vec::new();
                    let mut all = true;
                    inst.reads(|r| match c(r) {
                        Some(k) => ks.push(k),
                        None => all = false,
                    });
                    if inst.dst().is_some() && all { Some((generic(&ks), None)) } else { None }
                }
            };
            let Some((k, ch)) = new else { continue };
            // destinations
            let mut dsts: Vec<R> = Vec::new();
            super::memo::writes(p, inst, |d| dsts.push(d));
            if let Inst::CallC { d, .. } = *inst {
                if dsts.is_empty() {
                    dsts.push(d);
                }
            }
            for d in dsts {
                let du = d as usize;
                let mut kk = k;
                if kk == Cls::Aff && nw[du] != 1 {
                    kk = Cls::Bad; // chains only for single-assignment registers
                }
                let nk = if nw[du] == 1 { kk } else { join(cls[du], kk) };
                if cls[du] != Some(nk) {
                    cls[du] = Some(nk);
                    changed = true;
                }
                if nk == Cls::Aff {
                    let ch = ch.clone().unwrap_or_default();
                    if chain[du].len() != ch.len() {
                        chain[du] = ch;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // validity: nothing position-dependent beyond steps reaches an output, a
    // branch or a texture coordinate
    let mut ok = true;
    let good = |r: R| matches!(cls[r as usize], None | Some(Cls::Clean) | Some(Cls::Step));
    for &(_, r) in &p.outputs {
        ok &= good(r);
    }
    for inst in &p.code {
        match *inst {
            Inst::Jz { c, .. } | Inst::Jnz { c, .. } => ok &= good(c),
            Inst::Tex { op } => {
                let t = &p.tex_ops[op as usize];
                if t.kind != TexKind::Size {
                    ok &= good(t.x) && good(t.y);
                }
            }
            _ => {}
        }
    }
    Analysis { cls, ok, sites }
}

fn generic(ks: &[Cls]) -> Cls {
    if ks.iter().any(|k| matches!(k, Cls::Aff | Cls::Bad)) {
        Cls::Bad
    } else if ks.contains(&Cls::Step) {
        Cls::Step
    } else {
        Cls::Clean
    }
}

pub fn analyze(p: &Program) -> RunInfo {
    if std::env::var("AKUMA_RUNS").as_deref() == Ok("0") {
        return RunInfo::default();
    }
    let nr = p.nregs as usize;
    let mut nw = vec![0u32; nr];
    for i in &p.code {
        super::memo::writes(p, i, |r| nw[r as usize] += 1);
        if let Inst::CallC { c, .. } = *i {
            for k in 0..4 {
                nw[(c + k) as usize] += 2;
            }
        }
    }
    let ax = classify(p, 0, &nw);
    let ay = classify(p, 1, &nw);
    // first instruction index at or after which control flow may skip code
    let first_branch = p
        .code
        .iter()
        .position(|i| matches!(i, Inst::Jmp { .. } | Inst::Jz { .. } | Inst::Jnz { .. } | Inst::Kill | Inst::Ret | Inst::MemoGet { .. }))
        .unwrap_or(p.code.len());
    let def_idx = |r: R| -> Option<usize> {
        let mut at = None;
        for (k, i) in p.code.iter().enumerate() {
            super::memo::writes(p, i, |d| {
                if d == r {
                    at = Some(k)
                }
            });
        }
        at
    };
    let build = |a: &Analysis, other: &Analysis| -> Option<Axis> {
        if !a.ok {
            return None;
        }
        let mut sites = Vec::new();
        for (idx, s) in &a.sites {
            // the site must run on every path
            if *idx >= first_branch {
                return None;
            }
            for op in &s.chain {
                if matches!(op.kind, ChainKind::Neg) {
                    continue;
                }
                let r = op.r as usize;
                // chain operands: position-independent in *both* axes, and
                // computed before the site (or never written: a constant)
                if a.cls[r] != Some(Cls::Clean) || other.cls[r] != Some(Cls::Clean) || nw[r] > 1 {
                    return None;
                }
                if let Some(d) = def_idx(op.r) {
                    if d >= *idx {
                        return None;
                    }
                }
            }
            sites.push(s.clone());
        }
        Some(Axis { sites })
    };
    RunInfo { x: build(&ax, &ay), y: build(&ay, &ax) }
}
