//! The bytecode executor for `Program` (see `program.rs`).
//!
//! The portable fast path: no executable memory, runs anywhere the crate
//! builds. It is also the oracle the JIT is diffed against.

use super::program::{cmp, memo_get, memo_put, Inst, MemoInfo};
use super::texture::{SmpRef, TexOp, TexRef};

/// Run `code` over `regs` (already holding inputs and hoisted constants).
/// Returns true if the invocation was killed (`discard`).
pub fn run(
    code: &[Inst],
    regs: &mut [u32],
    bufs: &[&[u8]],
    texs: &[TexRef],
    smps: &[SmpRef],
    tex_ops: &[TexOp],
    memos: &[MemoInfo],
) -> bool {
    let mut pc = 0usize;
    macro_rules! fop {
        ($d:expr, $a:expr, $b:expr, $op:tt) => {
            regs[$d as usize] =
                (f32::from_bits(regs[$a as usize]) $op f32::from_bits(regs[$b as usize])).to_bits()
        };
    }
    loop {
        match code[pc] {
            Inst::Mov { d, s } => regs[d as usize] = regs[s as usize],
            Inst::Const { d, v } => regs[d as usize] = v,
            Inst::FAdd { d, a, b } => fop!(d, a, b, +),
            Inst::FSub { d, a, b } => fop!(d, a, b, -),
            Inst::FMul { d, a, b } => fop!(d, a, b, *),
            Inst::FDiv { d, a, b } => fop!(d, a, b, /),
            Inst::FNeg { d, a } => regs[d as usize] = regs[a as usize] ^ 0x8000_0000,
            Inst::FAbs { d, a } => regs[d as usize] = regs[a as usize] & 0x7fff_ffff,
            Inst::Sqrt { d, a } => {
                regs[d as usize] = f32::from_bits(regs[a as usize]).sqrt().to_bits()
            }
            Inst::IAdd { d, a, b } => {
                regs[d as usize] = regs[a as usize].wrapping_add(regs[b as usize])
            }
            Inst::ISub { d, a, b } => {
                regs[d as usize] = regs[a as usize].wrapping_sub(regs[b as usize])
            }
            Inst::IMul { d, a, b } => {
                regs[d as usize] = regs[a as usize].wrapping_mul(regs[b as usize])
            }
            Inst::And { d, a, b } => regs[d as usize] = regs[a as usize] & regs[b as usize],
            Inst::Or { d, a, b } => regs[d as usize] = regs[a as usize] | regs[b as usize],
            Inst::Xor { d, a, b } => regs[d as usize] = regs[a as usize] ^ regs[b as usize],
            Inst::Not { d, a } => regs[d as usize] = !regs[a as usize],
            Inst::Shl { d, a, b } => {
                regs[d as usize] = regs[a as usize].wrapping_shl(regs[b as usize] & 31)
            }
            Inst::ShrS { d, a, b } => {
                regs[d as usize] =
                    (regs[a as usize] as i32).wrapping_shr(regs[b as usize] & 31) as u32
            }
            Inst::ShrU { d, a, b } => {
                regs[d as usize] = regs[a as usize].wrapping_shr(regs[b as usize] & 31)
            }
            Inst::Cmp { d, a, b, c } => {
                regs[d as usize] = cmp(c, regs[a as usize], regs[b as usize]) as u32
            }
            Inst::Select { d, c, a, b } => {
                regs[d as usize] = if regs[c as usize] != 0 { regs[a as usize] } else { regs[b as usize] }
            }
            Inst::I2F { d, a } => regs[d as usize] = (regs[a as usize] as i32 as f32).to_bits(),
            Inst::U2F { d, a } => regs[d as usize] = (regs[a as usize] as f32).to_bits(),
            Inst::Call { d, a, b, f } => {
                regs[d as usize] = (f.helper())(regs[a as usize], regs[b as usize])
            }
            Inst::CallC { d, a, b, f, c } => {
                let (ka, kb) = (regs[a as usize], regs[b as usize]);
                let c = c as usize;
                if regs[c] != 0 && regs[c + 1] == ka && regs[c + 2] == kb {
                    regs[d as usize] = regs[c + 3];
                } else {
                    let r = (f.helper())(ka, kb);
                    regs[c] = 1;
                    regs[c + 1] = ka;
                    regs[c + 2] = kb;
                    regs[c + 3] = r;
                    regs[d as usize] = r;
                }
            }
            Inst::LoadBuf { d, buf, off, imm } => {
                let addr = regs[off as usize].wrapping_add(imm) as usize;
                let bytes = bufs[buf as usize];
                regs[d as usize] = match bytes.get(addr..addr + 4) {
                    Some(s) => u32::from_le_bytes([s[0], s[1], s[2], s[3]]),
                    None => 0,
                };
            }
            Inst::Tex { op } => unsafe {
                super::texture::tex_helper(
                    regs.as_mut_ptr(),
                    texs.as_ptr(),
                    smps.as_ptr(),
                    &tex_ops[op as usize],
                    1,
                );
            },
            Inst::Jmp { t } => {
                pc = t as usize;
                continue;
            }
            Inst::Jz { c, t } => {
                if regs[c as usize] == 0 {
                    pc = t as usize;
                    continue;
                }
            }
            Inst::Jnz { c, t } => {
                if regs[c as usize] != 0 {
                    pc = t as usize;
                    continue;
                }
            }
            Inst::MemoGet { m, t } => {
                // SAFETY: the memo's registers are inside `regs` by construction
                if unsafe { memo_get(regs.as_mut_ptr(), &memos[m as usize], 1) } != 0 {
                    pc = t as usize;
                    continue;
                }
            }
            Inst::MemoPut { m } => unsafe { memo_put(regs.as_mut_ptr(), &memos[m as usize], 1) },
            Inst::Kill => return true,
            Inst::Ret => return false,
        }
        pc += 1;
    }
}
