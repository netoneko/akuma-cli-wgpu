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

use super::program::{Cmp, Fun, Inst, Program};
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
    pub wide: bool,
}

/// returned by wide entry points besides 0 (ok) and 1 (killed, all lanes)
pub const STATUS_DIVERGED: u32 = 2;

/// one register's 4 lanes; 16-aligned so the vector templates can use `movaps`
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct L4(pub [u32; 4]);

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

    /// Run one batch of up to 4 lanes (wide code only). 0 = ok, 1 = killed,
    /// `STATUS_DIVERGED` = the lanes branched differently; nothing about the
    /// outputs may be trusted in that case.
    #[inline]
    pub fn run_wide(&self, regs: &mut [L4], bufs: &[BufRef], texs: &[TexRef], smps: &[SmpRef]) -> u32 {
        debug_assert!(self.wide);
        (self.entry)(regs.as_mut_ptr() as *mut u32, bufs.as_ptr(), texs.as_ptr(), smps.as_ptr())
    }

    pub fn code_len(&self) -> usize {
        self.len
    }
}

struct Asm {
    b: Vec<u8>,
    /// register file layout: register r, lane l lives at (r*stride + l)*4.
    /// Scalar code: stride 1, lane 0. Wide code: stride 4; the per-lane
    /// scalar fallback templates just run with lane = 0..4.
    stride: u32,
    lane: u32,
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
        self.u32((r * self.stride + self.lane) * 4);
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
        self.epilogue_code(killed as u32);
    }
    fn epilogue_code(&mut self, code: u32) {
        if code == 0 {
            self.bytes(&[0x31, 0xC0]); // xor eax,eax
        } else {
            self.u8(0xB8); // mov eax, imm32
            self.u32(code);
        }
        self.bytes(&[0x48, 0x83, 0xC4, 0x08]); // add rsp,8
        self.bytes(&[0x41, 0x5E]); // pop r14
        self.bytes(&[0x41, 0x5D]); // pop r13
        self.bytes(&[0x41, 0x5C]); // pop r12
        self.u8(0x5B); // pop rbx
        self.u8(0xC3); // ret
    }
}


/// One instruction, scalar form, for the register layout `a` is set up for
/// (stride/lane). Used directly by the scalar JIT, and once per lane by the
/// wide JIT for instructions that are not worth vectorizing.
fn emit_scalar(
    a: &mut Asm,
    p: &Program,
    inst: &Inst,
    fixups: &mut Vec<(usize, u32)>,
) -> Result<(), String> {
        match *inst {
            Inst::Mov { d, s } => {
                a.load_eax(s);
                a.store_eax(d);
            }
            Inst::Const { d, v } => {
                a.rm(&[0xC7], 0, d);
                a.u32(v);
            }
            Inst::FAdd { d, a: x, b } => fbin(a, 0x58, d, x, b),
            Inst::FSub { d, a: x, b } => fbin(a, 0x5C, d, x, b),
            Inst::FMul { d, a: x, b } => fbin(a, 0x59, d, x, b),
            Inst::FDiv { d, a: x, b } => fbin(a, 0x5E, d, x, b),
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
            Inst::IAdd { d, a: x, b } => ibin(a, 0x03, d, x, b),
            Inst::ISub { d, a: x, b } => ibin(a, 0x2B, d, x, b),
            Inst::IMul { d, a: x, b } => {
                a.load_eax(x);
                a.rm(&[0x0F, 0xAF], 0, b);
                a.store_eax(d);
            }
            Inst::And { d, a: x, b } => ibin(a, 0x23, d, x, b),
            Inst::Or { d, a: x, b } => ibin(a, 0x0B, d, x, b),
            Inst::Xor { d, a: x, b } => ibin(a, 0x33, d, x, b),
            Inst::Not { d, a: x } => {
                a.load_eax(x);
                a.bytes(&[0xF7, 0xD0]);
                a.store_eax(d);
            }
            Inst::Shl { d, a: x, b } => shift(a, &[0xD3, 0xE0], d, x, b),
            Inst::ShrS { d, a: x, b } => shift(a, &[0xD3, 0xF8], d, x, b),
            Inst::ShrU { d, a: x, b } => shift(a, &[0xD3, 0xE8], d, x, b),
            Inst::Cmp { d, a: x, b, c } => cmp(a, c, d, x, b),
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
                // regs pointer for this lane, and the register stride, so the
                // helper finds register r at r*stride (scalar: stride 1)
                a.bytes(&[0x48, 0x8D, 0xBB]); // lea rdi,[rbx+disp32]
                a.u32(a.lane * 4);
                a.bytes(&[0x41, 0xB8]); // mov r8d, imm32
                a.u32(a.stride);
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
    Ok(())
}

pub fn compile(p: &Program) -> Result<Jit, String> {
    compile_impl(p, false)
}

/// 4 lanes per instruction with SSE4.1 (needs `pmulld`/`blendvps`). Control flow
/// stays uniform across lanes: a branch whose lanes disagree makes the whole
/// batch return `STATUS_DIVERGED` so the caller can run the lanes one by one.
pub fn compile_wide(p: &Program) -> Result<Jit, String> {
    if !is_x86_feature_detected!("sse4.1") {
        return Err("no SSE4.1".into());
    }
    compile_impl(p, true)
}

fn compile_impl(p: &Program, wide: bool) -> Result<Jit, String> {
    let mut a = Asm {
        b: Vec::with_capacity(p.code.len() * if wide { 64 } else { 24 } + 64),
        stride: if wide { 4 } else { 1 },
        lane: 0,
    };
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
    let mut bail_jumps: Vec<usize> = Vec::new(); // rel32 positions to point at the bail stub

    for inst in &p.code {
        starts.push(a.b.len());
        if wide {
            emit_wide(&mut a, p, inst, &mut fixups, &mut bail_jumps)?;
        } else {
            emit_scalar(&mut a, p, inst, &mut fixups)?;
        }
    }
    starts.push(a.b.len());
    if wide {
        // bail stub: lanes disagreed at a branch
        let stub = a.b.len();
        a.epilogue_code(2);
        for pos in bail_jumps {
            let rel = (stub as i64 - (pos as i64 + 4)) as i32;
            a.b[pos..pos + 4].copy_from_slice(&rel.to_le_bytes());
        }
    }

    for (pos, t) in fixups {
        let target = *starts.get(t as usize).ok_or("jump target out of range")?;
        let rel = target as i64 - (pos as i64 + 4);
        let rel = i32::try_from(rel).map_err(|_| "jump too far")?;
        a.b[pos..pos + 4].copy_from_slice(&rel.to_le_bytes());
    }

    let mut j = finalize(&a.b)?;
    j.wide = wide;
    Ok(j)
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
            wide: false,
        })
    }
}

// ---------------------------------------------------------------------------
// wide (4-lane SSE4.1) emission
// ---------------------------------------------------------------------------

impl Asm {
    /// `cmp dword [valid], 0` for a wide cache's lane-0 valid word
    fn rm_valid_cmp(&mut self, c: u32) {
        self.rm(&[0x83], 7, c);
        self.u8(0);
    }
    /// SSE op with a memory operand: [prefix] 0F op /r, ModRM(mod=10, rm=rbx)
    /// addressing register `r`'s 16-byte lane block
    fn vm(&mut self, prefix: &[u8], op: &[u8], xmm: u8, r: u32) {
        self.bytes(prefix);
        self.u8(0x0F);
        self.bytes(op);
        self.u8(0x80 | (xmm << 3) | 3);
        self.u32(r * 16);
    }
    /// SSE op, register-register
    fn vr(&mut self, prefix: &[u8], op: &[u8], dst: u8, src: u8) {
        self.bytes(prefix);
        self.u8(0x0F);
        self.bytes(op);
        self.u8(0xC0 | (dst << 3) | src);
    }
    fn vload(&mut self, xmm: u8, r: u32) {
        self.vm(&[], &[0x28], xmm, r); // movaps xmm,[r]
    }
    fn vstore(&mut self, xmm: u8, r: u32) {
        self.vm(&[], &[0x29], xmm, r); // movaps [r],xmm
    }
    /// xmm = all ones
    fn v_ones(&mut self, xmm: u8) {
        self.vr(&[0x66], &[0x76], xmm, xmm); // pcmpeqd xmm,xmm
    }
    /// xmm = 0x80000000 in every lane
    fn v_signbit(&mut self, xmm: u8) {
        self.v_ones(xmm);
        self.bytes(&[0x66, 0x0F, 0x72, 0xF0 | xmm, 31]); // pslld xmm,31
    }
}

/// the register holds a hoisted constant (never written by code): its value
fn const_reg(p: &Program, r: u32) -> Option<u32> {
    if r == 0 {
        return Some(0);
    }
    // a register is constant iff nothing in the code writes it and the init
    // value is the one it always has; cheap scan, memoized by the caller's
    // program size being small relative to JIT cost
    let written = p.code.iter().any(|i| match *i {
        Inst::Mov { d, .. } | Inst::Const { d, .. } | Inst::FAdd { d, .. } | Inst::FSub { d, .. }
        | Inst::FMul { d, .. } | Inst::FDiv { d, .. } | Inst::FNeg { d, .. } | Inst::FAbs { d, .. }
        | Inst::Sqrt { d, .. } | Inst::IAdd { d, .. } | Inst::ISub { d, .. } | Inst::IMul { d, .. }
        | Inst::And { d, .. } | Inst::Or { d, .. } | Inst::Xor { d, .. } | Inst::Not { d, .. }
        | Inst::Shl { d, .. } | Inst::ShrS { d, .. } | Inst::ShrU { d, .. } | Inst::Cmp { d, .. }
        | Inst::Select { d, .. } | Inst::I2F { d, .. } | Inst::U2F { d, .. } | Inst::Call { d, .. }
        | Inst::CallC { d, .. } | Inst::LoadBuf { d, .. } => d == r,
        _ => false,
    });
    // registers written by Tex ops (4 consecutive from `d`, or 2) are not constants either
    let tex = p.tex_ops.iter().any(|t| r >= t.d && r < t.d + 4);
    if written || tex {
        None
    } else {
        Some(p.init[r as usize])
    }
}

fn emit_wide(
    a: &mut Asm,
    p: &Program,
    inst: &Inst,
    fixups: &mut Vec<(usize, u32)>,
    bail: &mut Vec<usize>,
) -> Result<(), String> {
    match *inst {
        Inst::Mov { d, s } => {
            a.vload(0, s);
            a.vstore(0, d);
        }
        Inst::Const { d, v } => {
            if v == 0 {
                a.vr(&[], &[0x57], 0, 0); // xorps xmm0,xmm0
                a.vstore(0, d);
            } else {
                for k in 0..4 {
                    a.rm(&[0xC7], 0, d); // mov dword [d lane k], imm32 (stride 4 layout)
                    a.u32(v);
                    // rm() used lane 0: patch the displacement for lane k
                    let at = a.b.len() - 8;
                    let disp = (d * 16 + k * 4) as i32;
                    a.b[at..at + 4].copy_from_slice(&disp.to_le_bytes());
                }
            }
        }
        Inst::FAdd { d, a: x, b } => vbin(a, &[], 0x58, d, x, b),
        Inst::FSub { d, a: x, b } => vbin(a, &[], 0x5C, d, x, b),
        Inst::FMul { d, a: x, b } => vbin(a, &[], 0x59, d, x, b),
        Inst::FDiv { d, a: x, b } => vbin(a, &[], 0x5E, d, x, b),
        Inst::Sqrt { d, a: x } => {
            a.vm(&[], &[0x51], 0, x); // sqrtps xmm0,[x]
            a.vstore(0, d);
        }
        Inst::FNeg { d, a: x } => {
            a.vload(0, x);
            a.v_signbit(1);
            a.vr(&[], &[0x57], 0, 1); // xorps xmm0,xmm1
            a.vstore(0, d);
        }
        Inst::FAbs { d, a: x } => {
            a.vload(0, x);
            a.v_ones(1);
            a.bytes(&[0x66, 0x0F, 0x72, 0xD1, 1]); // psrld xmm1,1
            a.vr(&[], &[0x54], 0, 1); // andps xmm0,xmm1
            a.vstore(0, d);
        }
        Inst::IAdd { d, a: x, b } => vbin(a, &[0x66], 0xFE, d, x, b),
        Inst::ISub { d, a: x, b } => vbin(a, &[0x66], 0xFA, d, x, b),
        Inst::IMul { d, a: x, b } => {
            a.vload(0, x);
            a.bytes(&[0x66, 0x0F, 0x38, 0x40, 0x83]); // pmulld xmm0,[b]
            a.u32(b * 16);
            a.vstore(0, d);
        }
        Inst::And { d, a: x, b } => vbin(a, &[0x66], 0xDB, d, x, b),
        Inst::Or { d, a: x, b } => vbin(a, &[0x66], 0xEB, d, x, b),
        Inst::Xor { d, a: x, b } => vbin(a, &[0x66], 0xEF, d, x, b),
        Inst::Not { d, a: x } => {
            a.vload(0, x);
            a.v_ones(1);
            a.vr(&[0x66], &[0xEF], 0, 1); // pxor xmm0,xmm1
            a.vstore(0, d);
        }
        // shifts by a constant count: one vector op; variable counts need AVX2,
        // so those go lane by lane
        Inst::Shl { d, a: x, b } | Inst::ShrS { d, a: x, b } | Inst::ShrU { d, a: x, b }
            if const_reg(p, b).is_some() =>
        {
            let n = const_reg(p, b).unwrap() & 31;
            let op = match inst {
                Inst::Shl { .. } => 0xF0u8,  // /6 pslld
                Inst::ShrS { .. } => 0xE0,   // /4 psrad
                _ => 0xD0,                   // /2 psrld
            };
            a.vload(0, x);
            a.bytes(&[0x66, 0x0F, 0x72, op, n as u8]);
            a.vstore(0, d);
        }
        Inst::Cmp { d, a: x, b, c } => vcmp(a, c, d, x, b),
        Inst::Select { d, c, a: x, b } => {
            // mask = 0 - c  (c is 0/1 per lane)  ->  xmm0 (implicit blendvps mask)
            a.vr(&[], &[0x57], 0, 0); // xorps xmm0,xmm0
            a.vm(&[0x66], &[0xFA], 0, c); // psubd xmm0,[c]
            a.vload(1, b);
            a.bytes(&[0x66, 0x0F, 0x38, 0x14, 0x8B]); // blendvps xmm1,[x]
            a.u32(x * 16);
            a.vstore(1, d);
        }
        Inst::I2F { d, a: x } => {
            a.vm(&[], &[0x5B], 0, x); // cvtdq2ps xmm0,[x]
            a.vstore(0, d);
        }
        // a load at a constant offset is the same word in every lane: load
        // once and broadcast
        Inst::LoadBuf { d, buf, off: 0, imm } => {
            a.lane = 0;
            let saved = a.stride;
            a.stride = 4;
            emit_scalar(a, p, &Inst::LoadBuf { d, buf, off: 0, imm }, fixups)?;
            // emit_scalar stored the loaded word into lane 0 of d; broadcast it
            a.stride = saved;
            a.vload(0, d);
            a.bytes(&[0x66, 0x0F, 0x70, 0xC0, 0x00]); // pshufd xmm0,xmm0,0
            a.vstore(0, d);
        }
        // rounding: one vector op (SSE4.1 roundps)
        Inst::Call { d, a: x, f: f @ (Fun::Floor | Fun::Ceil | Fun::Trunc), .. } => {
            let mode = match f {
                Fun::Floor => 0x09,
                Fun::Ceil => 0x0A,
                _ => 0x0B,
            };
            a.bytes(&[0x66, 0x0F, 0x3A, 0x08, 0x83]); // roundps xmm0,[x],mode
            a.u32(x * 16);
            a.u8(mode);
            a.vstore(0, d);
        }
        // f32 -> i32: cvttps2dq is exact except for NaN / out of range, where it
        // yields 0x80000000; Rust saturates (NaN -> 0), so those rare batches
        // rerun lane by lane through the helper
        Inst::Call { d, a: x, b, f: Fun::F2I } => {
            a.vm(&[0xF3], &[0x5B], 0, x); // cvttps2dq xmm0,[x]
            a.vstore(0, d);
            a.v_signbit(1);
            a.vr(&[0x66], &[0x76], 1, 0); // pcmpeqd xmm1,xmm0
            a.vr(&[], &[0x50], 0, 1); // movmskps eax,xmm1
            a.bytes(&[0x85, 0xC0, 0x0F, 0x84]); // test eax,eax; jz done
            let skip = a.b.len();
            a.u32(0);
            for lane in 0..4 {
                a.lane = lane;
                emit_scalar(a, p, &Inst::Call { d, a: x, b, f: Fun::F2I }, fixups)?;
            }
            a.lane = 0;
            let rel = (a.b.len() - (skip + 4)) as i32;
            a.b[skip..skip + 4].copy_from_slice(&rel.to_le_bytes());
        }
        // u32 -> f32: cvtdq2ps is right while bit 31 is clear in every lane
        Inst::U2F { d, a: x } => {
            a.vload(0, x);
            a.vr(&[], &[0x50], 0, 0); // movmskps eax,xmm0 (sign bits = bit 31)
            a.bytes(&[0x85, 0xC0, 0x0F, 0x85]); // test eax,eax; jnz slow
            let to_slow = a.b.len();
            a.u32(0);
            a.vm(&[], &[0x5B], 0, x); // cvtdq2ps xmm0,[x]
            a.vstore(0, d);
            a.u8(0xE9); // jmp done
            let to_done = a.b.len();
            a.u32(0);
            let rel = (a.b.len() - (to_slow + 4)) as i32;
            a.b[to_slow..to_slow + 4].copy_from_slice(&rel.to_le_bytes());
            for lane in 0..4 {
                a.lane = lane;
                emit_scalar(a, p, &Inst::U2F { d, a: x }, fixups)?;
            }
            a.lane = 0;
            let rel = (a.b.len() - (to_done + 4)) as i32;
            a.b[to_done..to_done + 4].copy_from_slice(&rel.to_le_bytes());
        }
        // cached libm call, vector-wide: all four lanes' arguments equal to the
        // cached ones (typical for a flat cell) -> one vector load; otherwise
        // the helper per lane, then refill the cache
        Inst::CallC { d, a: x, b, f, c } => {
            a.rm_valid_cmp(c); // cmp dword [valid lane 0],0
            a.bytes(&[0x0F, 0x84]); // je miss
            let m0 = a.b.len();
            a.u32(0);
            a.vload(0, x);
            a.vm(&[0x66], &[0x76], 0, c + 1); // pcmpeqd xmm0,[key a]
            a.vload(1, b);
            a.vm(&[0x66], &[0x76], 1, c + 2); // pcmpeqd xmm1,[key b]
            a.vr(&[0x66], &[0xDB], 0, 1); // pand xmm0,xmm1
            a.vr(&[], &[0x50], 0, 0); // movmskps eax,xmm0
            a.bytes(&[0x83, 0xF8, 0x0F, 0x0F, 0x85]); // cmp eax,15; jne miss
            let m1 = a.b.len();
            a.u32(0);
            a.vload(0, c + 3);
            a.vstore(0, d);
            a.u8(0xE9); // jmp done
            let done = a.b.len();
            a.u32(0);
            // miss
            let rel = (a.b.len() - (m0 + 4)) as i32;
            a.b[m0..m0 + 4].copy_from_slice(&rel.to_le_bytes());
            let rel = (a.b.len() - (m1 + 4)) as i32;
            a.b[m1..m1 + 4].copy_from_slice(&rel.to_le_bytes());
            for lane in 0..4 {
                a.lane = lane;
                emit_scalar(a, p, &Inst::Call { d, a: x, b, f }, fixups)?;
            }
            a.lane = 0;
            a.vload(0, x);
            a.vstore(0, c + 1);
            a.vload(0, b);
            a.vstore(0, c + 2);
            a.vload(0, d);
            a.vstore(0, c + 3);
            a.stride = 4;
            a.rm(&[0xC7], 0, c); // mov dword [valid lane 0], 1
            a.u32(1);
            let rel = (a.b.len() - (done + 4)) as i32;
            a.b[done..done + 4].copy_from_slice(&rel.to_le_bytes());
        }
        Inst::Jmp { t } => {
            a.u8(0xE9);
            fixups.push((a.b.len(), t));
            a.u32(0);
        }
        Inst::Jz { c, t } | Inst::Jnz { c, t } => {
            let jz = matches!(inst, Inst::Jz { .. });
            a.vload(0, c);
            a.bytes(&[0x66, 0x0F, 0x72, 0xF0, 31]); // pslld xmm0,31
            a.vr(&[], &[0x50], 0, 0); // movmskps eax,xmm0: bit k = lane k true
            a.bytes(&[0x85, 0xC0]); // test eax,eax
            if jz {
                // all lanes false -> jump; all true -> fall through; mixed -> bail
                a.bytes(&[0x0F, 0x84]); // je t
                fixups.push((a.b.len(), t));
                a.u32(0);
                a.bytes(&[0x83, 0xF8, 0x0F]); // cmp eax,15
                a.bytes(&[0x0F, 0x85]); // jne bail
                bail.push(a.b.len());
                a.u32(0);
            } else {
                // all lanes false -> fall through; all true -> jump; mixed -> bail
                a.bytes(&[0x74, 14]); // jz +14 (over: cmp 3 + jne 6 + jmp 5)
                a.bytes(&[0x83, 0xF8, 0x0F]); // cmp eax,15
                a.bytes(&[0x0F, 0x85]); // jne bail
                bail.push(a.b.len());
                a.u32(0);
                a.u8(0xE9); // jmp t
                fixups.push((a.b.len(), t));
                a.u32(0);
            }
        }
        Inst::Kill => a.epilogue(true),
        Inst::Ret => a.epilogue(false),
        // everything else: the scalar template, once per lane
        _ => {
            for lane in 0..4 {
                a.lane = lane;
                emit_scalar(a, p, inst, fixups)?;
            }
            a.lane = 0;
        }
    }
    Ok(())
}

fn vbin(a: &mut Asm, prefix: &[u8], op: u8, d: u32, x: u32, b: u32) {
    a.vload(0, x);
    a.vm(prefix, &[op], 0, b);
    a.vstore(0, d);
}

/// lane-wise compare, result 0/1 per lane
fn vcmp(a: &mut Asm, c: Cmp, d: u32, x: u32, b: u32) {
    use Cmp::*;
    match c {
        FEq | FNe | FLt | FLe => {
            let pred = match c {
                FEq => 0,
                FLt => 1,
                FLe => 2,
                _ => 4, // not-equal, true for unordered like Rust's !=
            };
            a.vload(0, x);
            a.vm(&[], &[0xC2], 0, b); // cmpps xmm0,[b],pred
            a.u8(pred);
        }
        FGt | FGe => {
            // a > b  <=>  b < a
            a.vload(0, b);
            a.vm(&[], &[0xC2], 0, x);
            a.u8(if c == FGt { 1 } else { 2 });
        }
        _ => {
            let unsigned = matches!(c, ULt | ULe | UGt | UGe);
            a.vload(0, x);
            a.vload(1, b);
            if unsigned {
                // flip the sign bit of both so the signed compare orders them
                a.v_signbit(2);
                a.vr(&[0x66], &[0xEF], 0, 2); // pxor xmm0,xmm2
                a.vr(&[0x66], &[0xEF], 1, 2); // pxor xmm1,xmm2
            }
            // result lands in xmm0
            match c {
                IEq => a.vr(&[0x66], &[0x76], 0, 1), // pcmpeqd xmm0,xmm1
                INe => {
                    a.vr(&[0x66], &[0x76], 0, 1);
                    a.v_ones(3);
                    a.vr(&[0x66], &[0xEF], 0, 3); // invert
                }
                SGt | UGt => a.vr(&[0x66], &[0x66], 0, 1), // pcmpgtd xmm0,xmm1  (a > b)
                SLe | ULe => {
                    a.vr(&[0x66], &[0x66], 0, 1);
                    a.v_ones(3);
                    a.vr(&[0x66], &[0xEF], 0, 3);
                }
                SLt | ULt => {
                    a.vr(&[0x66], &[0x66], 1, 0); // pcmpgtd xmm1,xmm0  (b > a)
                    a.vr(&[], &[0x28], 0, 1); // movaps xmm0,xmm1
                }
                _ => {
                    // SGe | UGe: not (a < b)
                    a.vr(&[0x66], &[0x66], 1, 0);
                    a.v_ones(3);
                    a.vr(&[0x66], &[0xEF], 1, 3);
                    a.vr(&[], &[0x28], 0, 1);
                }
            }
        }
    }
    // mask (all ones / zero) -> 1 / 0
    a.bytes(&[0x66, 0x0F, 0x72, 0xD0, 31]); // psrld xmm0,31
    a.vstore(0, d);
}
