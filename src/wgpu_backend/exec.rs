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

use super::interp::{self, Resources, Shader, Value};

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
    pub fn new(shader: Arc<Shader>, entry: usize) -> Stage {
        Stage::Interp(InterpStage::new(shader, entry))
    }

    pub fn name(&self) -> &'static str {
        match self {
            Stage::Interp(_) => "interp",
        }
    }

    /// Bind the draw's buffers; the returned invoker runs invocations.
    pub fn begin<'a>(&'a self, res: &'a Resources<'a>) -> Invoker<'a> {
        match self {
            Stage::Interp(s) => Invoker::Interp { s, res },
        }
    }
}

pub enum Invoker<'a> {
    Interp { s: &'a InterpStage, res: &'a Resources<'a> },
}

impl Invoker<'_> {
    pub fn run_vertex(&mut self, vertex_index: u32, instance_index: u32) -> RawVertex {
        match self {
            Invoker::Interp { s, res } => s.run_vertex(res, vertex_index, instance_index),
        }
    }

    pub fn run_fragment(&mut self, varyings: &Varyings, frag_pos: [f32; 4]) -> Option<[u32; 4]> {
        match self {
            Invoker::Interp { s, res } => s.run_fragment(res, varyings, frag_pos),
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
