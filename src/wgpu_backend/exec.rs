//! Shader stage execution behind one interface.
//!
//! A pipeline stage is run by one of three executors, all producing the same
//! `RawVertex` / fragment words so the fixed-function rasterizer never knows
//! which one ran:
//!
//! * `Interp`   — the naga-IR tree walker (`interp.rs`). Handles everything the
//!   backend supports; slow. It is the fallback for any shader the compiler
//!   below declines.
//! * `Compiled` — `compile.rs` lowers the naga IR once, at pipeline creation,
//!   into a flat scalar register program (`Program`). It is then run by the
//!   bytecode VM (`vm.rs`) or, on x86-64 with the kernel's RW→RX `mprotect`
//!   path available, by machine code emitted from the same program (`jit.rs`).
//!
//! `AKUMA_EXEC=interp|vm|jit` forces an executor (default: best available) so
//! the three can be diffed against each other (`akuma-wgpu exec-selftest`).

use std::sync::Arc;

use super::compile;
use super::interp::{self, Resources, Shader, Value};
use super::program::{Dst, Interp, Program, Src};
use super::texture::{SmpRef, TexRef};
use super::vm;

/// How many `@location` slots a stage interface may use (rio's shaders use
/// at most 3; flat 4-component slots, raw 32-bit words).
pub const MAX_LOC: usize = 8;

/// One vertex-stage result in executor-neutral form.
#[derive(Clone, Copy, Debug, Default)]
pub struct RawVertex {
    pub position: [f32; 4],
    /// raw bits per (location, component): f32/u32/i32 reinterpreted as u32
    pub varyings: [[u32; 4]; MAX_LOC],
}

pub type Varyings = [[u32; 4]; MAX_LOC];

#[derive(Debug)]
pub enum Stage {
    Interp(InterpStage),
    Compiled(CompiledStage),
}

#[derive(Debug)]
pub struct CompiledStage {
    pub prog: Program,
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    pub jit: Option<super::jit::Jit>,
}

#[derive(Debug)]
pub struct InterpStage {
    pub shader: Arc<Shader>,
    pub entry: usize,
    /// (location, type) of each fragment `@location` input, to rebuild
    /// interpreter `Value`s from raw varying words
    frag_inputs: Vec<(u32, naga::Handle<naga::Type>)>,
}

impl Stage {
    /// Pick the best executor for a stage: compiled if the lowering accepts
    /// the shader, the interpreter otherwise. `AKUMA_EXEC=interp` forces the
    /// interpreter; `AKUMA_EXEC_VERBOSE=1` reports each decision.
    pub fn new(shader: Arc<Shader>, entry: usize) -> Stage {
        let force = std::env::var("AKUMA_EXEC").unwrap_or_default();
        Stage::build(shader, entry, &force).unwrap_or_else(|e| panic!("exec: {e}"))
    }

    /// `force`: "" (best available), "interp", "vm" or "jit" (an error if
    /// the shader cannot be compiled / jitted).
    pub fn build(shader: Arc<Shader>, entry: usize, force: &str) -> Result<Stage, String> {
        let strict = force == "vm" || force == "jit";
        let verbose = std::env::var_os("AKUMA_EXEC_VERBOSE").is_some();
        let name = shader.module.entry_points[entry].name.clone();
        if force != "interp" {
            match compile::compile(&shader, entry) {
                Ok(prog) => {
                    if verbose {
                        eprintln!(
                            "[exec] {name}: compiled, {} insts, {} regs",
                            prog.code.len(),
                            prog.nregs
                        );
                    }
                    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                    let jit = if force == "vm" {
                        None
                    } else {
                        match super::jit::compile(&prog) {
                            Ok(j) => {
                                if verbose {
                                    eprintln!("[exec] {name}: jit, {} bytes of x86-64", j.code_len());
                                }
                                Some(j)
                            }
                            Err(e) => {
                                if verbose {
                                    eprintln!("[exec] {name}: jit declined ({e}), using vm");
                                }
                                None
                            }
                        }
                    };
                    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                    if force == "jit" && jit.is_none() {
                        return Err("jit unavailable for this shader".into());
                    }
                    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
                    if force == "jit" {
                        return Err("jit unavailable on this target".into());
                    }
                    return Ok(Stage::Compiled(CompiledStage {
                        prog,
                        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                        jit,
                    }));
                }
                Err(e) => {
                    let uses_handles = shader
                        .module
                        .global_variables
                        .iter()
                        .any(|(_, g)| g.space == naga::AddressSpace::Handle);
                    if strict || uses_handles {
                        return Err(format!(
                            "{name}: cannot compile{}: {e}",
                            if uses_handles { " (and the interpreter cannot sample textures)" } else { "" }
                        ));
                    }
                    if verbose {
                        eprintln!("[exec] {name}: interpreter ({e})");
                    }
                }
            }
        }
        Ok(Stage::Interp(InterpStage::new(shader, entry)))
    }

    /// (location, interpolation) of the stage's `@location` inputs
    pub fn frag_interp(&self) -> Vec<(u32, Interp)> {
        match self {
            Stage::Compiled(c) => c.prog.interp.clone(),
            Stage::Interp(s) => s
                .frag_inputs
                .iter()
                .map(|(l, ty)| {
                    let float = matches!(
                        &s.shader.module.types[*ty].inner,
                        naga::TypeInner::Scalar(sc) | naga::TypeInner::Vector { scalar: sc, .. }
                            if sc.kind == naga::ScalarKind::Float
                    );
                    (*l, if float { Interp::Perspective } else { Interp::Flat })
                })
                .collect(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Stage::Interp(_) => "interp",
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Stage::Compiled(c) if c.jit.is_some() => "jit",
            Stage::Compiled(_) => "vm",
        }
    }

    /// Bind the draw's buffers; the returned invoker runs invocations.
    pub fn begin<'a>(&'a self, res: &'a Resources<'a>) -> Invoker<'a> {
        match self {
            Stage::Interp(s) => Invoker::Interp { s, res },
            Stage::Compiled(c) => {
                let bufs: Vec<&[u8]> = c
                    .prog
                    .bufs
                    .iter()
                    .map(|k| res.bufs.get(k).copied().unwrap_or(&[]))
                    .collect();
                let texs: Vec<TexRef> = c
                    .prog
                    .texs
                    .iter()
                    .map(|k| res.texs.get(k).copied().unwrap_or(TexRef::EMPTY))
                    .collect();
                let smps: Vec<SmpRef> = c
                    .prog
                    .smps
                    .iter()
                    .map(|k| res.smps.get(k).copied().unwrap_or(SmpRef::DEFAULT))
                    .collect();
                #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                if let Some(j) = &c.jit {
                    let refs = bufs
                        .iter()
                        .map(|b| super::jit::BufRef { ptr: b.as_ptr(), len: b.len() })
                        .collect();
                    return Invoker::Jit { p: &c.prog, j, regs: c.prog.init.clone(), refs, texs, smps };
                }
                Invoker::Vm { p: &c.prog, regs: c.prog.init.clone(), bufs, texs, smps }
            }
        }
    }
}

pub enum Invoker<'a> {
    Interp { s: &'a InterpStage, res: &'a Resources<'a> },
    Vm {
        p: &'a Program,
        regs: Vec<u32>,
        bufs: Vec<&'a [u8]>,
        texs: Vec<TexRef>,
        smps: Vec<SmpRef>,
    },
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    Jit {
        p: &'a Program,
        j: &'a super::jit::Jit,
        regs: Vec<u32>,
        refs: Vec<super::jit::BufRef>,
        texs: Vec<TexRef>,
        smps: Vec<SmpRef>,
    },
}

impl Invoker<'_> {
    /// `attrs`: the vertex-buffer attributes for this invocation, by location
    pub fn run_vertex(&mut self, vertex_index: u32, instance_index: u32, attrs: &Varyings) -> RawVertex {
        match self {
            Invoker::Interp { s, res } => s.run_vertex(res, vertex_index, instance_index),
            Invoker::Vm { p, regs, bufs, texs, smps } => {
                vertex_in(p, regs, vertex_index, instance_index, attrs);
                vm::run(&p.code, regs, bufs, texs, smps, &p.tex_ops);
                vertex_out(p, regs)
            }
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Invoker::Jit { p, j, regs, refs, texs, smps } => {
                vertex_in(p, regs, vertex_index, instance_index, attrs);
                j.run(regs, refs, texs, smps);
                vertex_out(p, regs)
            }
        }
    }

    /// True when every fragment of a primitive produces the same result: the
    /// stage reads no position and only flat varyings (buffers/textures are
    /// constant for the draw). The rasterizer then runs it once per triangle.
    pub fn constant_per_primitive(&self) -> bool {
        let prog = match self {
            Invoker::Vm { p, .. } => p,
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Invoker::Jit { p, .. } => p,
            Invoker::Interp { .. } => return false,
        };
        prog.interp.iter().all(|(_, m)| *m == Interp::Flat)
            && !prog.inputs.iter().any(|(_, s)| matches!(s, Src::Position(_)))
    }

    pub fn run_fragment(&mut self, varyings: &Varyings, frag_pos: [f32; 4]) -> Option<[u32; 4]> {
        match self {
            Invoker::Interp { s, res } => s.run_fragment(res, varyings, frag_pos),
            Invoker::Vm { p, regs, bufs, texs, smps } => {
                frag_in(p, regs, varyings, frag_pos);
                if vm::run(&p.code, regs, bufs, texs, smps, &p.tex_ops) {
                    return None;
                }
                Some(frag_out(p, regs))
            }
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Invoker::Jit { p, j, regs, refs, texs, smps } => {
                frag_in(p, regs, varyings, frag_pos);
                if j.run(regs, refs, texs, smps) {
                    return None;
                }
                Some(frag_out(p, regs))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// interpreter adapter
// ---------------------------------------------------------------------------

impl InterpStage {
    fn new(shader: Arc<Shader>, entry: usize) -> InterpStage {
        let module = &shader.module;
        let f = &module.entry_points[entry].function;
        let mut frag_inputs = Vec::new();
        for a in &f.arguments {
            match &a.binding {
                Some(naga::Binding::Location { location, .. }) => frag_inputs.push((*location, a.ty)),
                None => {
                    if let naga::TypeInner::Struct { members, .. } = &module.types[a.ty].inner {
                        for m in members {
                            if let Some(naga::Binding::Location { location, .. }) = &m.binding {
                                frag_inputs.push((*location, m.ty));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        InterpStage { shader, entry, frag_inputs }
    }

    fn run_vertex(&self, res: &Resources<'_>, vi: u32, inst: u32) -> RawVertex {
        let vo = interp::run_vertex(&self.shader, self.entry, res, vi, inst);
        let mut out = RawVertex { position: vo.position, ..Default::default() };
        for (loc, v) in &vo.varyings {
            let mut bits = [0u32; 4];
            value_bits(v, &mut bits);
            out.varyings[*loc as usize] = bits;
        }
        out
    }

    fn run_fragment(&self, res: &Resources<'_>, varyings: &Varyings, pos: [f32; 4]) -> Option<[u32; 4]> {
        let vals: Vec<(u32, Value)> = self
            .frag_inputs
            .iter()
            .map(|(loc, ty)| (*loc, bits_value(&self.shader, *ty, &varyings[*loc as usize])))
            .collect();
        interp::run_fragment(&self.shader, self.entry, res, &vals, pos)
    }
}

fn scalar_bits(v: &Value) -> u32 {
    match v {
        Value::F32(x) => x.to_bits(),
        Value::I32(x) => *x as u32,
        Value::U32(x) => *x,
        Value::Bool(b) => *b as u32,
        other => panic!("exec: varying component {other:?}"),
    }
}

fn value_bits(v: &Value, out: &mut [u32; 4]) {
    match v {
        Value::Vec(vs) => {
            for (i, c) in vs.iter().take(4).enumerate() {
                out[i] = scalar_bits(c);
            }
        }
        scalar => out[0] = scalar_bits(scalar),
    }
}

fn bits_value(sh: &Shader, ty: naga::Handle<naga::Type>, bits: &[u32; 4]) -> Value {
    let mk = |s: naga::Scalar, w: u32| match s.kind {
        naga::ScalarKind::Float => Value::F32(f32::from_bits(w)),
        naga::ScalarKind::Sint => Value::I32(w as i32),
        naga::ScalarKind::Uint => Value::U32(w),
        naga::ScalarKind::Bool => Value::Bool(w != 0),
        k => panic!("exec: varying kind {k:?}"),
    };
    match &sh.module.types[ty].inner {
        naga::TypeInner::Scalar(s) => mk(*s, bits[0]),
        naga::TypeInner::Vector { size, scalar } => {
            let n = match size {
                naga::VectorSize::Bi => 2,
                naga::VectorSize::Tri => 3,
                naga::VectorSize::Quad => 4,
            };
            Value::Vec((0..n).map(|i| mk(*scalar, bits[i])).collect())
        }
        other => panic!("exec: varying type {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// compiled-stage input/output marshalling (shared by VM and JIT)
// ---------------------------------------------------------------------------

#[inline]
fn vertex_in(p: &Program, regs: &mut [u32], vertex_index: u32, instance_index: u32, attrs: &Varyings) {
    for &(r, src) in &p.inputs {
        regs[r as usize] = match src {
            Src::VertexIndex => vertex_index,
            Src::InstanceIndex => instance_index,
            Src::Location(l, c) => attrs[l as usize][c as usize],
            Src::Position(_) => 0,
        };
    }
}

#[inline]
fn vertex_out(p: &Program, regs: &[u32]) -> RawVertex {
    let mut out = RawVertex::default();
    for &(d, r) in &p.outputs {
        let bits = regs[r as usize];
        match d {
            Dst::Position(i) => out.position[i as usize] = f32::from_bits(bits),
            Dst::Location(l, c) => out.varyings[l as usize][c as usize] = bits,
        }
    }
    out
}

#[inline]
fn frag_in(p: &Program, regs: &mut [u32], varyings: &Varyings, frag_pos: [f32; 4]) {
    for &(r, src) in &p.inputs {
        regs[r as usize] = match src {
            Src::Position(c) => frag_pos[c as usize].to_bits(),
            Src::Location(l, c) => varyings[l as usize][c as usize],
            _ => 0,
        };
    }
}

#[inline]
fn frag_out(p: &Program, regs: &[u32]) -> [u32; 4] {
    let mut color = [0u32; 4];
    for &(d, r) in &p.outputs {
        if let Dst::Location(0, c) = d {
            color[c as usize] = regs[r as usize];
        }
    }
    color
}
