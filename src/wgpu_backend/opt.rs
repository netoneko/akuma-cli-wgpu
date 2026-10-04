//! Program-to-program optimizer: constant and copy propagation, constant
//! folding (including loads from buffers whose contents are known for the
//! draw), constant-branch folding, unreachable-code removal, jump threading
//! and dead-code elimination by liveness.
//!
//! Two uses:
//!
//! * at pipeline creation, with no buffer contents (`optimize(p, None)`):
//!   strips the copies and dead stores the lowering leaves behind;
//! * per draw, `optimize(p, Some(bufs))`: every load at a constant offset
//!   from a bound buffer (uniforms, mostly: padding, cell size, colour-space
//!   flags, ...) becomes a constant, which folds the branches and arithmetic
//!   that depend only on them. The loads it folded are reported so a caller
//!   can cache the specialized program and reuse it while those words stay
//!   unchanged.
//!
//! Constant folding evaluates instructions with `vm::run` itself, so folded
//! values are bit-identical to what any executor would have computed.
//!
//! Soundness leans on two properties of the lowering: a register written by
//! exactly one instruction is SSA (its definition dominates every use), and
//! every use of a multi-write local is preceded, on every path, by a write
//! in the same invocation. Knowledge about multi-write registers is only
//! kept inside straight-line blocks (reset at every jump target).

use super::memo::{reads, writes};
use super::program::{Inst, Program, R};
use super::vm;

/// (buffer slot, byte address, value) of a load folded into a constant
pub type Folded = (u32, u32, u32);

pub fn optimize(p: &mut Program, bufs: Option<&[&[u8]]>) -> Vec<Folded> {
    let mut folded = Vec::new();
    if std::env::var("AKUMA_OPT").as_deref() == Ok("0") {
        return folded;
    }
    for _ in 0..12 {
        let mut ch = fold(p, bufs, &mut folded);
        ch |= prune(p);
        ch |= ifconv(p);
        ch |= dce(p);
        if !ch {
            break;
        }
    }
    // inputs nothing reads any more need not be written per invocation
    let mut read = vec![false; p.nregs as usize];
    for i in &p.code {
        reads(p, i, |r| read[r as usize] = true);
    }
    for (_, r) in &p.outputs {
        read[*r as usize] = true;
    }
    let keep: Vec<(R, super::program::Src)> =
        p.inputs.iter().copied().filter(|&(r, _)| read[r as usize]).collect();
    p.inputs = keep;
    folded.sort_unstable();
    folded.dedup();
    folded
}

fn jump_target(i: &Inst) -> Option<u32> {
    match *i {
        Inst::Jmp { t } | Inst::Jz { t, .. } | Inst::Jnz { t, .. } | Inst::MemoGet { t, .. } => Some(t),
        _ => None,
    }
}

fn set_target(i: &mut Inst, nt: u32) {
    match i {
        Inst::Jmp { t } | Inst::Jz { t, .. } | Inst::Jnz { t, .. } | Inst::MemoGet { t, .. } => *t = nt,
        _ => {}
    }
}

/// how many instructions write each register (cache registers count double:
/// they must never look like single-assignment constants)
fn write_counts(p: &Program) -> Vec<u32> {
    let mut nw = vec![0u32; p.nregs as usize];
    for i in &p.code {
        writes(p, i, |r| nw[r as usize] += 1);
        if let Inst::CallC { c, .. } = *i {
            for k in 0..4 {
                nw[(c + k) as usize] += 2;
            }
        }
    }
    nw
}

/// 0-or-1 destination instructions without side effects
fn pure_dst(i: &Inst) -> Option<R> {
    match *i {
        Inst::Tex { .. } | Inst::MemoGet { .. } | Inst::MemoPut { .. } => None,
        _ => i.dst(),
    }
}

fn load_word(buf: &[u8], addr: u32) -> u32 {
    let a = addr as usize;
    match buf.get(a..a.wrapping_add(4)) {
        Some(s) => u32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        None => 0,
    }
}

fn fold(p: &mut Program, bufs: Option<&[&[u8]]>, folded: &mut Vec<Folded>) -> bool {
    let n = p.code.len();
    let nr = p.nregs as usize;
    let nw = write_counts(p);
    let mut is_input = vec![false; nr];
    for &(r, _) in &p.inputs {
        is_input[r as usize] = true;
    }
    // registers that never change: hoisted constants
    let mut kval: Vec<Option<u32>> = (0..nr)
        .map(|r| if nw[r] == 0 && !is_input[r] { Some(p.init[r]) } else { None })
        .collect();
    kval[0] = Some(0);
    let mut gcopy: Vec<Option<R>> = vec![None; nr];
    let mut label = vec![false; n + 1];
    for i in &p.code {
        if let Some(t) = jump_target(i) {
            label[t as usize] = true;
        }
    }
    // block-local knowledge about multi-write registers
    let mut lval: Vec<Option<u32>> = vec![None; nr];
    let mut lcopy: Vec<Option<R>> = vec![None; nr];
    let mut dirty: Vec<R> = Vec::new();
    let mut keep = vec![true; n];
    let mut changed = false;
    let mut scratch = vec![0u32; nr];
    let mut new_code: Vec<Inst> = p.code.clone();

    for i in 0..n {
        if label[i] {
            for r in dirty.drain(..) {
                lval[r as usize] = None;
                lcopy[r as usize] = None;
            }
        }
        let orig = p.code[i];
        let mut inst = orig;
        // forward copies
        {
            let (lc, gc, nwr) = (&lcopy, &gcopy, &nw);
            inst.map_reads(|mut r| {
                for _ in 0..8 {
                    if let Some(s) = lc[r as usize] {
                        r = s;
                    } else if nwr[r as usize] == 1 && gc[r as usize].is_some() {
                        r = gc[r as usize].unwrap();
                    } else {
                        break;
                    }
                }
                r
            });
        }
        let cv = |r: R, kval: &Vec<Option<u32>>, lval: &Vec<Option<u32>>| kval[r as usize].or(lval[r as usize]);

        // constant branches
        match inst {
            Inst::Jz { c, t } => {
                if let Some(v) = cv(c, &kval, &lval) {
                    if v == 0 {
                        inst = Inst::Jmp { t };
                    } else {
                        keep[i] = false;
                    }
                }
            }
            Inst::Jnz { c, t } => {
                if let Some(v) = cv(c, &kval, &lval) {
                    if v != 0 {
                        inst = Inst::Jmp { t };
                    } else {
                        keep[i] = false;
                    }
                }
            }
            _ => {}
        }
        if !keep[i] {
            changed = true;
            continue;
        }

        if let Some(d) = pure_dst(&inst) {
            // select with a known condition / identical arms is a move
            if let Inst::Select { d, c, a, b } = inst {
                if let Some(cc) = cv(c, &kval, &lval) {
                    inst = Inst::Mov { d, s: if cc != 0 { a } else { b } };
                } else if a == b {
                    inst = Inst::Mov { d, s: a };
                }
            }
            // evaluate when every operand is known
            let mut v: Option<u32> = None;
            match inst {
                Inst::Const { v: c, .. } => v = Some(c),
                Inst::CallC { .. } | Inst::LoadBuf { .. } | Inst::Mov { .. } | _ => {
                    let mut all = true;
                    let mut ops: Vec<(R, u32)> = Vec::new();
                    reads(p, &inst, |r| match cv(r, &kval, &lval) {
                        Some(x) => ops.push((r, x)),
                        None => all = false,
                    });
                    let loadable = !matches!(inst, Inst::LoadBuf { .. }) || bufs.is_some();
                    if all && loadable {
                        for &(r, x) in &ops {
                            scratch[r as usize] = x;
                        }
                        let ev = match inst {
                            Inst::CallC { d, a, b, f, .. } => Inst::Call { d, a, b, f },
                            other => other,
                        };
                        let b: &[&[u8]] = bufs.unwrap_or(&[]);
                        vm::run(&[ev, Inst::Ret], &mut scratch, b, &[], &[], &[], &[]);
                        v = Some(scratch[d as usize]);
                        if let Inst::LoadBuf { buf, off, imm, .. } = inst {
                            let addr = cv(off, &kval, &lval).unwrap().wrapping_add(imm);
                            folded.push((buf, addr, load_word(b[buf as usize], addr)));
                        }
                    }
                }
            }
            if let Some(v) = v {
                if nw[d as usize] == 1 {
                    // single assignment: the register becomes a hoisted constant
                    kval[d as usize] = Some(v);
                    p.init[d as usize] = v;
                    keep[i] = false;
                    changed = true;
                    continue;
                }
                inst = Inst::Const { d, v };
                kill(d, &mut lval, &mut lcopy, &dirty);
                lval[d as usize] = Some(v);
                dirty.push(d);
            } else if let Inst::Mov { d, s } = inst {
                if nw[d as usize] == 1 && (nw[s as usize] <= 1) {
                    gcopy[d as usize] = Some(s);
                } else {
                    kill(d, &mut lval, &mut lcopy, &dirty);
                    if s != d {
                        lcopy[d as usize] = Some(s);
                        dirty.push(d);
                    }
                }
            } else {
                kill(d, &mut lval, &mut lcopy, &dirty);
            }
        } else if let Inst::Tex { .. } = inst {
            writes(p, &inst, |d| kill(d, &mut lval, &mut lcopy, &dirty));
        }
        new_code[i] = inst;
        if format!("{inst:?}") != format!("{orig:?}") {
            changed = true;
        }
    }
    if keep.iter().any(|k| !k) {
        let code: Vec<Inst> = new_code;
        p.code = code;
        compact(p, &keep);
    } else {
        p.code = new_code;
    }
    changed
}

/// forget everything block-local about register `d` (it is being redefined)
fn kill(d: R, lval: &mut [Option<u32>], lcopy: &mut [Option<R>], dirty: &[R]) {
    lval[d as usize] = None;
    lcopy[d as usize] = None;
    for &r in dirty {
        if lcopy[r as usize] == Some(d) {
            lcopy[r as usize] = None;
        }
    }
}

/// drop the instructions with `keep[i] == false`, retargeting jumps to the
/// first surviving instruction at or after their old target
fn compact(p: &mut Program, keep: &[bool]) {
    let n = p.code.len();
    let mut newidx = vec![0u32; n + 1];
    let mut c = 0u32;
    for i in 0..n {
        newidx[i] = c;
        if keep[i] {
            c += 1;
        }
    }
    newidx[n] = c;
    let mut out = Vec::with_capacity(c as usize);
    for (i, inst) in p.code.iter().enumerate() {
        if !keep[i] {
            continue;
        }
        let mut inst = *inst;
        if let Some(t) = jump_target(&inst) {
            set_target(&mut inst, newidx[t as usize]);
        }
        out.push(inst);
    }
    p.code = out;
}

fn successors(code: &[Inst], i: usize, mut f: impl FnMut(usize)) {
    match code[i] {
        Inst::Jmp { t } => f(t as usize),
        Inst::Jz { t, .. } | Inst::Jnz { t, .. } | Inst::MemoGet { t, .. } => {
            f(t as usize);
            f(i + 1);
        }
        Inst::Kill | Inst::Ret => {}
        _ => f(i + 1),
    }
}

/// unreachable code, jump-to-jump chains, jumps to the next instruction
fn prune(p: &mut Program) -> bool {
    let n = p.code.len();
    let mut changed = false;
    // thread jumps through jumps
    for i in 0..n {
        if let Some(mut t) = jump_target(&p.code[i]) {
            let t0 = t;
            for _ in 0..16 {
                match p.code[t as usize] {
                    Inst::Jmp { t: u } if u != t => t = u,
                    _ => break,
                }
            }
            if t != t0 {
                set_target(&mut p.code[i], t);
                changed = true;
            }
        }
    }
    // reachability
    let mut reach = vec![false; n];
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        if i >= n || reach[i] {
            continue;
        }
        reach[i] = true;
        successors(&p.code, i, |s| stack.push(s));
    }
    let mut keep = reach;
    // a jump to the next surviving instruction does nothing
    loop {
        let mut again = false;
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            if let Inst::Jmp { t } = p.code[i] {
                let next = (i + 1..n).find(|&j| keep[j]);
                let tgt = (t as usize..n).find(|&j| keep[j]);
                if next.is_some() && next == tgt {
                    keep[i] = false;
                    again = true;
                }
            }
        }
        if !again {
            break;
        }
    }
    if keep.iter().any(|k| !k) {
        compact(p, &keep);
        changed = true;
    }
    changed
}

/// dead code by backward liveness over the control-flow graph
fn dce(p: &mut Program) -> bool {
    let n = p.code.len();
    let nr = p.nregs as usize;
    let words = nr.div_ceil(64);
    let mut live_in = vec![0u64; n * words];
    let mut out_mask = vec![0u64; words];
    for &(_, r) in &p.outputs {
        out_mask[r as usize / 64] |= 1 << (r % 64);
    }
    let mut changed_any = false;
    loop {
        // iterate to a fixpoint
        let mut iter_changed = true;
        while iter_changed {
            iter_changed = false;
            for i in (0..n).rev() {
                let mut live = vec![0u64; words];
                match p.code[i] {
                    Inst::Ret => live.copy_from_slice(&out_mask),
                    _ => successors(&p.code, i, |s| {
                        if s < n {
                            for w in 0..words {
                                live[w] |= live_in[s * words + w];
                            }
                        }
                    }),
                }
                writes(p, &p.code[i], |d| live[d as usize / 64] &= !(1 << (d % 64)));
                reads(p, &p.code[i], |r| live[r as usize / 64] |= 1 << (r % 64));
                if let Inst::CallC { c, .. } = p.code[i] {
                    for k in 0..4 {
                        let r = c + k;
                        live[r as usize / 64] |= 1 << (r % 64);
                    }
                }
                if live_in[i * words..(i + 1) * words] != live[..] {
                    live_in[i * words..(i + 1) * words].copy_from_slice(&live);
                    iter_changed = true;
                }
            }
        }
        // remove pure instructions whose results are not live afterwards
        let mut keep = vec![true; n];
        for i in 0..n {
            let removable = match p.code[i] {
                Inst::Tex { .. } => true,
                ref other => pure_dst(other).is_some(),
            };
            if !removable {
                continue;
            }
            let mut live_after = vec![0u64; words];
            match p.code[i] {
                Inst::Ret => live_after.copy_from_slice(&out_mask),
                _ => successors(&p.code, i, |s| {
                    if s < n {
                        for w in 0..words {
                            live_after[w] |= live_in[s * words + w];
                        }
                    }
                }),
            }
            let mut any_live = false;
            writes(p, &p.code[i], |d| any_live |= live_after[d as usize / 64] >> (d % 64) & 1 != 0);
            if !any_live {
                keep[i] = false;
            }
        }
        if keep.iter().all(|k| *k) {
            break;
        }
        compact(p, &keep);
        changed_any = true;
        // liveness is stale after compaction: restart with fresh arrays
        return changed_any | dce(p);
    }
    changed_any
}


/// Calls cheap enough to run speculatively.
fn cheap_call(f: super::program::Fun) -> bool {
    use super::program::Fun::*;
    matches!(f, Ceil | Floor | Round | Trunc | Sign | FMin | FMax | Clamp01 | IsInf | F2I | F2U | IAbs | ISign)
}

/// If-conversion. A small `if` / `if else` whose arms are straight-line pure
/// code becomes both arms executed unconditionally plus a `Select` per
/// register they assign. A conditional branch is a liability for the wide
/// JIT — a branch whose lanes disagree throws the whole batch back to
/// one-lane-at-a-time execution — and WGSL's short-circuit `||` / `&&` (which
/// the front end turns into a branch and a temporary) disagree all the time:
/// every vertex batch of sugarloaf's glyph shader used to diverge on
/// `vid == 1u || vid == 3u`.
///
/// Arms may only assign registers through single-destination pure
/// instructions (no texture fetches, no result-cached libm calls), so
/// speculating them is safe: out-of-range buffer loads yield 0 and nothing
/// traps. Registers written once in total (arm-local temporaries) are kept as
/// they are; any other register an arm assigns gets a fresh temporary per arm
/// and a `Select` after the arms.
fn ifconv(p: &mut Program) -> bool {
    let mut any = false;
    for _ in 0..256 {
        if !ifconv_once(p) {
            break;
        }
        any = true;
    }
    any
}

fn ifconv_once(p: &mut Program) -> bool {
    const MAX_ARM: usize = 16;
    let n = p.code.len();
    let mut tcount = vec![0u32; n + 1];
    for i in &p.code {
        if let Some(t) = jump_target(i) {
            tcount[t as usize] += 1;
        }
    }
    let nw = write_counts(p);
    for i in 0..n {
        let (c, e, jz) = match p.code[i] {
            Inst::Jz { c, t } => (c, t as usize, true),
            Inst::Jnz { c, t } => (c, t as usize, false),
            _ => continue,
        };
        if e <= i + 1 || e >= n {
            continue;
        }
        // diamond (then-arm ends in a jump over the else-arm) or triangle
        let (then_end, else_arm, m) = match p.code[e - 1] {
            Inst::Jmp { t } if (t as usize) > e && (t as usize) < n => (e - 1, Some(e..t as usize), t as usize),
            _ => (e, None, e),
        };
        if else_arm.is_some() && tcount[e] != 1 {
            continue;
        }
        // nothing else may enter the arms
        if (i + 1..m).any(|k| k != e && tcount[k] > 0) {
            continue;
        }
        let arm_ok = |r: std::ops::Range<usize>| -> bool {
            r.len() <= MAX_ARM
                && p.code[r.clone()].iter().all(|x| match *x {
                    Inst::Call { f, .. } => cheap_call(f),
                    Inst::CallC { .. } | Inst::Tex { .. } => false,
                    ref other => pure_dst(other).is_some(),
                })
        };
        let then_r = i + 1..then_end;
        let else_r = else_arm.clone().unwrap_or(e..e);
        if !arm_ok(then_r.clone()) || !arm_ok(else_r.clone()) {
            continue;
        }
        // the condition must survive the arms
        let mut writes_c = false;
        for k in then_r.clone().chain(else_r.clone()) {
            writes(p, &p.code[k], |d| writes_c |= d == c);
        }
        if writes_c {
            continue;
        }

        // rename the multiply-assigned registers each arm writes
        let mut next = p.nregs;
        let mut rename = |arm: std::ops::Range<usize>| -> (Vec<Inst>, Vec<(R, R)>) {
            let mut map: Vec<(R, R)> = Vec::new();
            let mut out = Vec::new();
            for k in arm {
                let mut inst = p.code[k];
                inst.map_reads(|r| map.iter().find(|(o, _)| *o == r).map_or(r, |(_, t)| *t));
                if let Some(d) = pure_dst(&inst) {
                    if nw[d as usize] > 1 {
                        let t = match map.iter().find(|(o, _)| *o == d) {
                            Some((_, t)) => *t,
                            None => {
                                let t = next;
                                next += 1;
                                map.push((d, t));
                                t
                            }
                        };
                        inst.set_dst(t);
                    }
                }
                out.push(inst);
            }
            (out, map)
        };
        let (then_code, then_map) = rename(then_r);
        let (else_code, else_map) = rename(else_r);
        let mut regs: Vec<R> = then_map.iter().chain(else_map.iter()).map(|(d, _)| *d).collect();
        regs.sort_unstable();
        regs.dedup();
        let mut new_code: Vec<Inst> = Vec::new();
        new_code.extend(then_code);
        new_code.extend(else_code);
        for d in regs {
            let tv = then_map.iter().find(|(o, _)| *o == d).map_or(d, |(_, t)| *t);
            let ev = else_map.iter().find(|(o, _)| *o == d).map_or(d, |(_, t)| *t);
            // Jz: c != 0 runs the then-arm; Jnz: c != 0 jumps to the else label
            let (a, b) = if jz { (tv, ev) } else { (ev, tv) };
            new_code.push(Inst::Select { d, c, a, b });
        }
        // splice [i, m) := new_code
        let len = new_code.len();
        let old = m - i;
        let mut code: Vec<Inst> = Vec::with_capacity(n + len);
        code.extend_from_slice(&p.code[..i]);
        code.extend_from_slice(&new_code);
        code.extend_from_slice(&p.code[m..]);
        for (k, inst) in code.iter_mut().enumerate() {
            if k >= i && k < i + len {
                continue;
            }
            if let Some(t) = jump_target(inst) {
                let t = t as usize;
                let nt = if t >= m { t + len - old } else { t };
                set_target(inst, nt as u32);
            }
        }
        p.code = code;
        let extra = (next - p.nregs) as usize;
        p.nregs = next;
        p.init.resize(p.init.len() + extra, 0);
        return true; // indices moved: the caller's next round starts afresh
    }
    false
}
