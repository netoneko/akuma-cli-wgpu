//! x86-64 machine-code emitter for `Program` (see `program.rs`).
//!
//! A deliberately simple template JIT: every virtual register lives in memory
//! (`[rbx + 4*r]`), each instruction loads its operands into eax/ecx/xmm0,
//! operates, and stores the result back. No register allocation, no
//! scheduling — that already removes all dispatch, decoding and bounds
//! checks, which is where the VM spends its time.
//!
//! W^X: the buffer is `mmap`ed read-write, filled, then flipped to read+exec
//! with `mprotect`. No page is ever writable and executable at once, which is
//! exactly what the Akuma kernel's policy allows (verified on the trashcan by
//! `userspace/jitprobe` in the kernel repo, 2026-10-04).
//!
//! Calling convention of the emitted function (System V):
//!   `extern "C" fn(regs: *mut u32, bufs: *const BufRef) -> u32`
//! returning 1 if the invocation was killed, else 0. rbx = regs, r12 = bufs
//! (both callee-saved, so helper calls leave them intact).
//!
//! Anything the emitter cannot encode returns `Err` and the stage stays on
//! the VM. The VM and the JIT call the very same `Fun` helpers, and the
//! inline sequences are IEEE-exact, so they agree bit for bit.

use super::program::{Cmp, Inst, Program};
use super::texture::{SmpRef, TexRef};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BufRef {
    pub ptr: *const u8,
    pub len: usize,
}

type Entry = extern "C" fn(*mut u32, *const BufRef, *const TexRef, *const SmpRef) -> u32;

pub struct Jit {
    mem: *mut u8,
    len: usize,
    entry: Entry,
}

// the code is immutable once flipped to R+X
unsafe impl Send for Jit {}
unsafe impl Sync for Jit {}

impl std::fmt::Debug for Jit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Jit({} bytes)", self.len)
    }
}

impl Drop for Jit {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.mem as *mut libc::c_void, self.len);
        }
    }
}

impl Jit {
    /// Run one invocation; true = killed.
    #[inline]
    pub fn run(&self, regs: &mut [u32], bufs: &[BufRef], texs: &[TexRef], smps: &[SmpRef]) -> bool {
        (self.entry)(regs.as_mut_ptr(), bufs.as_ptr(), texs.as_ptr(), smps.as_ptr()) != 0
    }

    pub fn code_len(&self) -> usize {
        self.len
    }
}

struct Asm {
    b: Vec<u8>,
}

impl Asm {
    fn u8(&mut self, v: u8) {
        self.b.push(v);
    }
    fn bytes(&mut self, v: &[u8]) {
        self.b.extend_from_slice(v);
    }
    fn u32(&mut self, v: u32) {
        self.b.extend_from_slice(&v.to_le_bytes());
    }
    /// `op [rbx + 4*r]` with reg field `reg`: opcode bytes, ModRM(mod=10,rm=rbx), disp32
    fn rm(&mut self, op: &[u8], reg: u8, r: u32) {
        self.bytes(op);
        self.u8(0x80 | (reg << 3) | 3);
        self.u32(r * 4);
    }
    fn load_eax(&mut self, r: u32) {
        self.rm(&[0x8B], 0, r);
    }
    fn load_ecx(&mut self, r: u32) {
        self.rm(&[0x8B], 1, r);
    }
    fn store_eax(&mut self, r: u32) {
        self.rm(&[0x89], 0, r);
    }
    /// short conditional jump with a placeholder; returns the patch position
    fn jcc8(&mut self, cc: u8) -> usize {
        self.bytes(&[cc, 0]);
        self.b.len() - 1
    }
    fn jmp8(&mut self) -> usize {
        self.bytes(&[0xEB, 0]);
        self.b.len() - 1
    }
    /// point the short jump at `pos` to the current position
    fn patch8(&mut self, pos: usize) {
        let rel = self.b.len() - (pos + 1);
        assert!(rel < 128, "jit: short jump out of range");
        self.b[pos] = rel as u8;
    }
    fn movss_load(&mut self, xmm: u8, r: u32) {
        self.rm(&[0xF3, 0x0F, 0x10], xmm, r);
    }
    fn movss_store(&mut self, r: u32) {
        self.rm(&[0xF3, 0x0F, 0x11], 0, r);
    }
    fn epilogue(&mut self, killed: bool) {
        if killed {
            self.bytes(&[0xB8, 1, 0, 0, 0]); // mov eax,1
        } else {
            self.bytes(&[0x31, 0xC0]); // xor eax,eax
        }
        self.bytes(&[0x48, 0x83, 0xC4, 0x08]); // add rsp,8
        self.bytes(&[0x41, 0x5E]); // pop r14
        self.bytes(&[0x41, 0x5D]); // pop r13
        self.bytes(&[0x41, 0x5C]); // pop r12
        self.u8(0x5B); // pop rbx
        self.u8(0xC3); // ret
    }
}

#[derive(Clone, Copy)]
enum Jk {
    Jmp,
    Jz,
    Jnz,
}

pub fn compile(p: &Program) -> Result<Jit, String> {
    let mut a = Asm { b: Vec::with_capacity(p.code.len() * 24 + 64) };
    // prologue
    a.u8(0x53); // push rbx
    a.bytes(&[0x41, 0x54]); // push r12
    a.bytes(&[0x41, 0x55]); // push r13
    a.bytes(&[0x41, 0x56]); // push r14
    a.bytes(&[0x48, 0x83, 0xEC, 0x08]); // sub rsp,8   (rsp 16-aligned at calls)
    a.bytes(&[0x48, 0x89, 0xFB]); // mov rbx,rdi   regs
    a.bytes(&[0x49, 0x89, 0xF4]); // mov r12,rsi   bufs
    a.bytes(&[0x49, 0x89, 0xD5]); // mov r13,rdx   texs
    a.bytes(&[0x49, 0x89, 0xCE]); // mov r14,rcx   smps

    let mut starts = Vec::with_capacity(p.code.len() + 1);
    let mut fixups: Vec<(usize, u32)> = Vec::new(); // (rel32 position, target inst)

    for inst in &p.code {
        starts.push(a.b.len());
        match *inst {
            Inst::Mov { d, s } => {
                a.load_eax(s);
                a.store_eax(d);
            }
            Inst::Const { d, v } => {
                a.rm(&[0xC7], 0, d);
                a.u32(v);
            }
            Inst::FAdd { d, a: x, b } => fbin(&mut a, 0x58, d, x, b),
            Inst::FSub { d, a: x, b } => fbin(&mut a, 0x5C, d, x, b),
            Inst::FMul { d, a: x, b } => fbin(&mut a, 0x59, d, x, b),
            Inst::FDiv { d, a: x, b } => fbin(&mut a, 0x5E, d, x, b),
            Inst::FNeg { d, a: x } => {
                a.load_eax(x);
                a.u8(0x35); // xor eax, imm32
                a.u32(0x8000_0000);
                a.store_eax(d);
            }
            Inst::FAbs { d, a: x } => {
                a.load_eax(x);
                a.u8(0x25); // and eax, imm32
                a.u32(0x7fff_ffff);
                a.store_eax(d);
            }
            Inst::Sqrt { d, a: x } => {
                a.rm(&[0xF3, 0x0F, 0x51], 0, x); // sqrtss xmm0,[x]
                a.movss_store(d);
            }
            Inst::IAdd { d, a: x, b } => ibin(&mut a, 0x03, d, x, b),
            Inst::ISub { d, a: x, b } => ibin(&mut a, 0x2B, d, x, b),
            Inst::IMul { d, a: x, b } => {
                a.load_eax(x);
                a.rm(&[0x0F, 0xAF], 0, b);
                a.store_eax(d);
            }
            Inst::And { d, a: x, b } => ibin(&mut a, 0x23, d, x, b),
            Inst::Or { d, a: x, b } => ibin(&mut a, 0x0B, d, x, b),
            Inst::Xor { d, a: x, b } => ibin(&mut a, 0x33, d, x, b),
            Inst::Not { d, a: x } => {
                a.load_eax(x);
                a.bytes(&[0xF7, 0xD0]);
                a.store_eax(d);
            }
            Inst::Shl { d, a: x, b } => shift(&mut a, &[0xD3, 0xE0], d, x, b),
            Inst::ShrS { d, a: x, b } => shift(&mut a, &[0xD3, 0xF8], d, x, b),
            Inst::ShrU { d, a: x, b } => shift(&mut a, &[0xD3, 0xE8], d, x, b),
            Inst::Cmp { d, a: x, b, c } => cmp(&mut a, c, d, x, b),
            Inst::Select { d, c, a: x, b } => {
                a.load_eax(b);
                a.load_ecx(c);
                a.bytes(&[0x85, 0xC9]); // test ecx,ecx
                a.rm(&[0x0F, 0x45], 0, x); // cmovne eax,[x]
                a.store_eax(d);
            }
            Inst::I2F { d, a: x } => {
                a.rm(&[0xF3, 0x0F, 0x2A], 0, x); // cvtsi2ss xmm0, dword [x]
                a.movss_store(d);
            }
            Inst::U2F { d, a: x } => {
                a.load_eax(x); // zero-extends into rax
                a.bytes(&[0xF3, 0x48, 0x0F, 0x2A, 0xC0]); // cvtsi2ss xmm0, rax
                a.movss_store(d);
            }
            Inst::Call { d, a: x, b, f } => {
                a.rm(&[0x8B], 7, x); // mov edi,[x]
                a.rm(&[0x8B], 6, b); // mov esi,[b]
                a.bytes(&[0x48, 0xB8]); // mov rax, imm64
                a.bytes(&(f.helper() as usize as u64).to_le_bytes());
                a.bytes(&[0xFF, 0xD0]); // call rax
                a.store_eax(d);
            }
            Inst::CallC { d, a: x, b, f, c } => {
                a.rm(&[0x83], 7, c); // cmp dword [valid], 0
                a.u8(0);
                let j_miss0 = a.jcc8(0x74); // je miss
                a.load_eax(x);
                a.rm(&[0x3B], 0, c + 1); // cmp eax,[key a]
                let j_miss1 = a.jcc8(0x75); // jne miss
                a.load_ecx(b);
                a.rm(&[0x3B], 1, c + 2); // cmp ecx,[key b]
                let j_miss2 = a.jcc8(0x75); // jne miss
                a.rm(&[0x8B], 0, c + 3); // mov eax,[result]
                let j_done = a.jmp8();
                // miss: call the helper, refill the cache
                a.patch8(j_miss0);
                a.patch8(j_miss1);
                a.patch8(j_miss2);
                a.rm(&[0x8B], 7, x); // mov edi,[a]
                a.rm(&[0x8B], 6, b); // mov esi,[b]
                a.bytes(&[0x48, 0xB8]); // mov rax, imm64
                a.bytes(&(f.helper() as usize as u64).to_le_bytes());
                a.bytes(&[0xFF, 0xD0]); // call rax
                a.store_eax(c + 3); // result
                // edi/esi are caller-saved: reload the keys from the registers
                a.load_ecx(x);
                a.rm(&[0x89], 1, c + 1);
                a.load_ecx(b);
                a.rm(&[0x89], 1, c + 2);
                a.rm(&[0xC7], 0, c); // mov dword [valid], 1
                a.u32(1);
                a.patch8(j_done);
                a.store_eax(d);
            }
            Inst::LoadBuf { d, buf, off, imm } => {
                a.load_eax(off);
                a.u8(0x05); // add eax, imm32 (wraps at 32 bits, like the VM)
                a.u32(imm);
                a.bytes(&[0x48, 0x8D, 0x48, 0x04]); // lea rcx,[rax+4]
                a.bytes(&[0x49, 0x3B, 0x8C, 0x24]); // cmp rcx,[r12+16*buf+8]
                a.u32(buf * 16 + 8);
                a.bytes(&[0x77, 0x0D]); // ja zero (skip 13 bytes)
                a.bytes(&[0x49, 0x8B, 0x94, 0x24]); // mov rdx,[r12+16*buf]
                a.u32(buf * 16);
                a.bytes(&[0x8B, 0x04, 0x02]); // mov eax,[rdx+rax]
                a.bytes(&[0xEB, 0x02]); // jmp done
                a.bytes(&[0x31, 0xC0]); // zero: xor eax,eax
                a.store_eax(d); // done:
            }
            Inst::Tex { op } => {
                // tex_helper(regs, texs, smps, &tex_ops[op])
                a.bytes(&[0x48, 0x89, 0xDF]); // mov rdi,rbx
                a.bytes(&[0x4C, 0x89, 0xEE]); // mov rsi,r13
                a.bytes(&[0x4C, 0x89, 0xF2]); // mov rdx,r14
                let opref = p.tex_ops.get(op as usize).ok_or("tex op out of range")?;
                a.bytes(&[0x48, 0xB9]); // mov rcx, imm64
                a.bytes(&(opref as *const _ as usize as u64).to_le_bytes());
                a.bytes(&[0x48, 0xB8]); // mov rax, imm64
                a.bytes(&(super::texture::tex_helper as usize as u64).to_le_bytes());
                a.bytes(&[0xFF, 0xD0]); // call rax
            }
            Inst::Jmp { t } => {
                a.u8(0xE9);
                fixups.push((a.b.len(), t));
                a.u32(0);
            }
            Inst::Jz { c, t } => {
                a.load_eax(c);
                a.bytes(&[0x85, 0xC0, 0x0F, 0x84]);
                fixups.push((a.b.len(), t));
                a.u32(0);
            }
            Inst::Jnz { c, t } => {
                a.load_eax(c);
                a.bytes(&[0x85, 0xC0, 0x0F, 0x85]);
                fixups.push((a.b.len(), t));
                a.u32(0);
            }
            Inst::Kill => a.epilogue(true),
            Inst::Ret => a.epilogue(false),
        }
    }
    starts.push(a.b.len());
    let _ = Jk::Jmp; // (kept for symmetry with the fixup kinds above)
    let _ = (Jk::Jz, Jk::Jnz);

    for (pos, t) in fixups {
        let target = *starts.get(t as usize).ok_or("jump target out of range")?;
        let rel = target as i64 - (pos as i64 + 4);
        let rel = i32::try_from(rel).map_err(|_| "jump too far")?;
        a.b[pos..pos + 4].copy_from_slice(&rel.to_le_bytes());
    }

    finalize(&a.b)
}

fn fbin(a: &mut Asm, op: u8, d: u32, x: u32, b: u32) {
    a.movss_load(0, x);
    a.rm(&[0xF3, 0x0F, op], 0, b);
    a.movss_store(d);
}

fn ibin(a: &mut Asm, op: u8, d: u32, x: u32, b: u32) {
    a.load_eax(x);
    a.rm(&[op], 0, b);
    a.store_eax(d);
}

fn shift(a: &mut Asm, op: &[u8], d: u32, x: u32, b: u32) {
    a.load_eax(x);
    a.load_ecx(b);
    a.bytes(op); // shl/sar/shr eax,cl — the CPU masks cl to 5 bits, like the VM
    a.store_eax(d);
}

fn cmp(a: &mut Asm, c: Cmp, d: u32, x: u32, b: u32) {
    use Cmp::*;
    match c {
        IEq | INe | SLt | SLe | SGt | SGe | ULt | ULe | UGt | UGe => {
            a.load_eax(x);
            a.rm(&[0x3B], 0, b); // cmp eax,[b]
            let cc = match c {
                IEq => 0x94,
                INe => 0x95,
                SLt => 0x9C,
                SLe => 0x9E,
                SGt => 0x9F,
                SGe => 0x9D,
                ULt => 0x92,
                ULe => 0x96,
                UGt => 0x97,
                _ => 0x93,
            };
            a.bytes(&[0x0F, cc, 0xC0, 0x0F, 0xB6, 0xC0]); // setcc al; movzx eax,al
        }
        // a < b  <=>  b > a : compare (b, a) and use "above" so that unordered
        // (CF=ZF=1) is false
        FLt | FLe => {
            a.movss_load(0, b);
            a.rm(&[0x0F, 0x2E], 0, x); // ucomiss xmm0,[x]
            let cc = if c == FLt { 0x97 } else { 0x93 };
            a.bytes(&[0x0F, cc, 0xC0, 0x0F, 0xB6, 0xC0]);
        }
        FGt | FGe => {
            a.movss_load(0, x);
            a.rm(&[0x0F, 0x2E], 0, b); // ucomiss xmm0,[b]
            let cc = if c == FGt { 0x97 } else { 0x93 };
            a.bytes(&[0x0F, cc, 0xC0, 0x0F, 0xB6, 0xC0]);
        }
        FEq => {
            a.movss_load(0, x);
            a.rm(&[0x0F, 0x2E], 0, b);
            // equal and ordered
            a.bytes(&[0x0F, 0x94, 0xC0, 0x0F, 0x9B, 0xC1, 0x20, 0xC8, 0x0F, 0xB6, 0xC0]);
        }
        FNe => {
            a.movss_load(0, x);
            a.rm(&[0x0F, 0x2E], 0, b);
            // not equal or unordered
            a.bytes(&[0x0F, 0x95, 0xC0, 0x0F, 0x9A, 0xC1, 0x08, 0xC8, 0x0F, 0xB6, 0xC0]);
        }
    }
    a.store_eax(d);
}

/// mmap RW, copy, flip to R+X. No page is ever W+X.
fn finalize(code: &[u8]) -> Result<Jit, String> {
    unsafe {
        let page = 4096usize;
        let len = code.len().div_ceil(page).max(1) * page;
        let mem = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        if mem == libc::MAP_FAILED {
            return Err(format!("jit mmap failed: errno {}", *libc::__errno_location()));
        }
        std::ptr::copy_nonoverlapping(code.as_ptr(), mem as *mut u8, code.len());
        if libc::mprotect(mem, len, libc::PROT_READ | libc::PROT_EXEC) != 0 {
            let e = *libc::__errno_location();
            libc::munmap(mem, len);
            return Err(format!("jit mprotect(R+X) failed: errno {e}"));
        }
        Ok(Jit {
            mem: mem as *mut u8,
            len,
            entry: std::mem::transmute::<*mut libc::c_void, Entry>(mem),
        })
    }
}
