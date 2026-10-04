//! Region memoization for fragment programs.
//!
//! Many real fragment shaders do their expensive work (colour-space maths,
//! transcendentals, texture decode) as a pure function of one or two values
//! that are constant over many neighbouring pixels: sugarloaf's cell
//! background shader computes its colour from the 32-bit cell word it loaded,
//! and that word is the same for the 16x32 pixels of a cell. This pass finds
//! such a region statically and brackets it:
//!
//! ```text
//!   MemoGet m, past      ; live-ins match a cached entry? copy results, jump
//!   <region>
//!   MemoPut m            ; remember (live-ins -> live-outs)
//! past:
//!   Ret
//! ```
//!
//! The region is the tail of the program (it ends at the final `Ret`), so its
//! live-outs are the stage's outputs. A region is accepted only if it is
//! single-entry / single-exit, free of loops, `Kill` and `Ret`, its live-ins
//! (registers read before the region writes them, apart from hoisted
//! constants) are few, never written inside the region, and are not stage
//! inputs (those differ per pixel, so a key made of them would never hit),
//! and every output is written on every path through it. Anything else is
//! left alone: the pass can only ever decline.
//!
//! The cache is a small direct-mapped table in persistent registers (see
//! `MemoInfo`); executors share `program::memo_get` / `memo_put`, so the VM
//! and both JITs agree by construction. `AKUMA_MEMO=0` turns the pass off.

use super::program::{Inst, MemoInfo, Program, R};
use super::texture::TexKind;

/// at most this many key registers
const MAX_KEYS: usize = 2;
/// table size: 1 << BITS entries
const BITS: u32 = 10;
/// a region must be worth at least this many (weighted) instructions
const MIN_WEIGHT: u32 = 60;

fn weight(i: &Inst) -> u32 {
    match i {
        Inst::Call { .. } | Inst::CallC { .. } | Inst::Tex { .. } => 20,
        Inst::LoadBuf { .. } => 3,
        Inst::FDiv { .. } | Inst::Sqrt { .. } => 3,
        _ => 1,
    }
}

/// registers `i` reads, excluding a `CallC`'s own cache (pure state)
pub(super) fn reads(p: &Program, i: &Inst, mut f: impl FnMut(R)) {
    match *i {
        Inst::CallC { a, b, .. } => {
            f(a);
            f(b);
        }
        Inst::Tex { op } => {
            let t = &p.tex_ops[op as usize];
            f(t.x);
            f(t.y);
        }
        _ => i.reads(f),
    }
}

/// registers `i` writes, excluding a `CallC`'s cache
pub(super) fn writes(p: &Program, i: &Inst, mut f: impl FnMut(R)) {
    if let Inst::Tex { op } = *i {
        let t = &p.tex_ops[op as usize];
        for k in 0..if t.kind == TexKind::Size { 2 } else { 4 } {
            f(t.d + k);
        }
    } else if let Some(d) = i.dst() {
        f(d);
    }
}

fn jump_target(i: &Inst) -> Option<u32> {
    match *i {
        Inst::Jmp { t } | Inst::Jz { t, .. } | Inst::Jnz { t, .. } | Inst::MemoGet { t, .. } => Some(t),
        _ => None,
    }
}

struct Pick {
    s: usize,
    e: usize,
    ins: Vec<R>,
    outs: Vec<R>,
}

/// Memoize the best region that starts at or after instruction `min_start`
/// (code before it — e.g. what run detection needs — must always execute).
pub fn apply(p: &mut Program, min_start: usize) {
    if std::env::var("AKUMA_MEMO").as_deref() == Ok("0") {
        return;
    }
    if !p.memos.is_empty() {
        return;
    }
    if let Some(pick) = select(p, min_start) {
        if std::env::var_os("AKUMA_EXEC_VERBOSE").is_some() {
            eprintln!(
                "[memo] region {}..{} of {}: keys {:?}, results {:?}",
                pick.s, pick.e, p.code.len(), pick.ins, pick.outs
            );
        }
        insert(p, pick);
    }
}

fn select(p: &Program, min_start: usize) -> Option<Pick> {
    let n = p.code.len();
    if n < 8 || !matches!(p.code[n - 1], Inst::Ret) {
        return None;
    }
    let e = n - 1;
    let nregs = p.nregs as usize;

    // registers the code (or a texture op) ever writes; the rest are the
    // hoisted constants (and inputs, which the caller writes)
    let mut written = vec![false; nregs];
    for i in &p.code {
        writes(p, i, |r| written[r as usize] = true);
    }
    let is_input = |r: R| p.inputs.iter().any(|&(x, _)| x == r);
    // forward jumps only
    for (k, i) in p.code.iter().enumerate() {
        if let Some(t) = jump_target(i) {
            if (t as usize) <= k {
                return None;
            }
        }
    }
    let out_regs: Vec<R> = {
        let mut v: Vec<R> = p.outputs.iter().map(|&(_, r)| r).collect();
        v.sort_unstable();
        v.dedup();
        v
    };

    let mut best: Option<(i64, Pick)> = None;
    for s in min_start.max(1)..e {
        let Some(pick) = try_region(p, s, e, &written, &is_input, &out_regs) else { continue };
        let weight_sum: u32 = p.code[s..e].iter().map(weight).sum();
        let overhead = 25 + 15 * pick.ins.len() as u32 + 4 * pick.outs.len() as u32;
        if weight_sum < MIN_WEIGHT || weight_sum <= overhead {
            continue;
        }
        // keys loaded from memory (a cell word, a glyph id) take few distinct
        // values, unlike keys computed from the pixel position (a cell
        // index): prefer them even for a somewhat smaller region
        let data_keys = pick.ins.iter().all(|&k| {
            p.code[..s]
                .iter()
                .rev()
                .find(|i| i.dst() == Some(k))
                .is_some_and(|i| matches!(i, Inst::LoadBuf { .. }))
        });
        let score = weight_sum as i64 - overhead as i64 + if data_keys { 60 } else { 0 };
        if best.as_ref().is_none_or(|(b, _)| score >= *b) {
            best = Some((score, pick));
        }
    }
    best.map(|(_, b)| b)
}

fn try_region(
    p: &Program,
    s: usize,
    e: usize,
    written: &[bool],
    is_input: &dyn Fn(R) -> bool,
    out_regs: &[R],
) -> Option<Pick> {
    let code = &p.code;
    // single entry: nothing outside jumps into (s, e)
    for (k, i) in code.iter().enumerate() {
        if let Some(t) = jump_target(i) {
            let t = t as usize;
            let inside = k >= s && k < e;
            if inside {
                // single exit: inside jumps stay inside the region or land on e
                if t <= k || t > e {
                    return None;
                }
            } else if t > s && t < e {
                return None;
            }
        }
    }
    // no early exits
    if code[s..e].iter().any(|i| matches!(i, Inst::Kill | Inst::Ret | Inst::MemoGet { .. } | Inst::MemoPut { .. })) {
        return None;
    }

    // jumps inside the region, (position, target), for the dominance test
    let jumps: Vec<(usize, usize)> = (s..e)
        .filter_map(|k| jump_target(&code[k]).map(|t| (k, t as usize)))
        .collect();
    // last write to each register before position i that is on every path
    // from s to i: a write at j covers i unless some jump before j skips
    // past j and lands at or before i
    let covers = |j: usize, i: usize| !jumps.iter().any(|&(k, t)| k < j && t > j && t <= i);

    // per register: positions of writes inside the region
    let mut first_cover: std::collections::HashMap<R, Vec<usize>> = std::collections::HashMap::new();
    let mut live_in: Vec<R> = Vec::new();
    for i in s..e {
        let inst = &code[i];
        let mut bad = false;
        reads(p, inst, |r| {
            if r == 0 || live_in.contains(&r) {
                return;
            }
            let r_written = written[r as usize];
            if !r_written && !is_input(r) {
                return; // hoisted constant
            }
            let covered = first_cover
                .get(&r)
                .is_some_and(|ws| ws.iter().any(|&j| covers(j, i)));
            if !covered {
                live_in.push(r);
                if live_in.len() > MAX_KEYS {
                    bad = true;
                }
            }
        });
        if bad {
            return None;
        }
        writes(p, inst, |r| first_cover.entry(r).or_default().push(i));
    }
    // keys must be stable values: not stage inputs, not rewritten in the region
    for &k in &live_in {
        if is_input(k) || first_cover.contains_key(&k) {
            return None;
        }
    }
    // results: outputs written in the region, each on every path
    let mut outs = Vec::new();
    for &o in out_regs {
        if let Some(ws) = first_cover.get(&o) {
            if !ws.iter().any(|&j| covers(j, e)) {
                return None;
            }
            outs.push(o);
        }
    }
    if outs.is_empty() || outs.len() > 6 {
        return None;
    }
    // a region register read after the region (only `Ret` follows) -> none
    Some(Pick { s, e, ins: live_in, outs })
}

fn insert(p: &mut Program, pick: Pick) {
    let Pick { s, e, ins, outs } = pick;
    let bits = std::env::var("AKUMA_MEMO_BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(BITS).clamp(1, 16);
    let m = MemoInfo { ins, outs, base: p.nregs + 1, bits, slot: p.nregs };
    let total = m.table_regs();
    p.nregs += 1 + total;
    p.init.resize(p.nregs as usize, 0);

    // new layout: [0..s) | MemoGet | [s..e) | MemoPut | Ret
    let remap_inside = |t: u32| -> u32 { (t as usize + 1) as u32 }; // s < t <= e (e lands on MemoPut)
    let ret_new = (e + 2) as u32;
    let mut code = Vec::with_capacity(p.code.len() + 2);
    for (k, inst) in p.code.iter().enumerate() {
        if k == s {
            code.push(Inst::MemoGet { m: 0, t: ret_new });
        }
        if k == e {
            code.push(Inst::MemoPut { m: 0 });
        }
        let mut inst = *inst;
        let inside = k >= s && k < e;
        if let Some(t) = jump_target(&inst) {
            let t = t as usize;
            let nt = if t < s || (t == s) {
                t as u32 // before the region, or its MemoGet
            } else if inside && t <= e {
                remap_inside(t as u32)
            } else if t == e {
                ret_new // an outside jump to the final Ret skips MemoPut
            } else {
                (t + 2) as u32
            };
            match &mut inst {
                Inst::Jmp { t } | Inst::Jz { t, .. } | Inst::Jnz { t, .. } => *t = nt,
                _ => {}
            }
        }
        code.push(inst);
    }
    p.code = code;
    p.memos.push(m);
}
