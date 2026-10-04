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
    /// 4-lane SSE4.1 code, when available (always alongside `jit`, which
    /// handles batches whose lanes diverge)
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    pub wide: Option<super::jit::Jit>,
    /// the optimized program before memoization: what per-draw
    /// specialization starts from
    template: Program,
    /// which executor was asked for (`AKUMA_EXEC`), to build specializations alike
    force: String,
    /// recent specializations, newest last
    spec_cache: std::sync::Mutex<Vec<Arc<Spec>>>,
}

/// A program specialized to the constant words of one draw's buffers, with
/// its machine code. Valid for any draw whose buffers still hold the words it
/// folded (`folded`).
#[derive(Debug)]
pub struct Spec {
    pub prog: Program,
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    pub jit: Option<super::jit::Jit>,
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    pub wide: Option<super::jit::Jit>,
    folded: Vec<super::opt::Folded>,
}

/// Tests set this to specialize even tiny draws.
pub static FORCE_SPEC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// What `Stage::plan` decided for a draw: run the generic program (`None`)
/// or a specialized one.
pub type Plan = Option<Arc<Spec>>;

/// machine code for `prog`, per the executor policy `force`
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn backends(
    prog: &Program,
    force: &str,
    name: &str,
    verbose: bool,
) -> Result<(Option<super::jit::Jit>, Option<super::jit::Jit>), String> {
    let jit = if force == "vm" {
        None
    } else {
        match super::jit::compile(prog) {
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
    if force == "jit" && jit.is_none() {
        return Err("jit unavailable for this shader".into());
    }
    let wide = if jit.is_some() && force != "jit" && std::env::var("AKUMA_WIDE").as_deref() != Ok("0") {
        match super::jit::compile_wide(prog) {
            Ok(w) => {
                if verbose {
                    eprintln!("[exec] {name}: wide jit, {} bytes", w.code_len());
                }
                Some(w)
            }
            Err(e) => {
                if force == "jitw" {
                    return Err(format!("{name}: wide jit declined: {e}"));
                }
                None
            }
        }
    } else {
        None
    };
    if force == "jitw" && wide.is_none() {
        return Err("wide jit unavailable".into());
    }
    Ok((jit, wide))
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
        let strict = force == "vm" || force == "jit" || force == "jitw";
        let verbose = std::env::var_os("AKUMA_EXEC_VERBOSE").is_some();
        let name = shader.module.entry_points[entry].name.clone();
        if force != "interp" {
            match compile::compile_with_template(&shader, entry) {
                Ok((prog, template)) => {
                    if verbose {
                        eprintln!(
                            "[exec] {name}: compiled, {} insts, {} regs",
                            prog.code.len(),
                            prog.nregs
                        );
                    }
                    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                    let (jit, wide) = backends(&prog, force, &name, verbose)?;
                    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
                    if force == "jit" {
                        return Err("jit unavailable on this target".into());
                    }
                    return Ok(Stage::Compiled(CompiledStage {
                        prog,
                        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                        jit,
                        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                        wide,
                        template,
                        force: force.to_string(),
                        spec_cache: std::sync::Mutex::new(Vec::new()),
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
            Stage::Compiled(c) if c.wide.is_some() => "jitw",
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Stage::Compiled(c) if c.jit.is_some() => "jit",
            Stage::Compiled(_) => "vm",
        }
    }

    /// Bind the draw's buffers; the returned invoker runs invocations.
    pub fn begin<'a>(&'a self, res: &'a Resources<'a>) -> Invoker<'a> {
        self.begin_with(res, &None)
    }

    /// Decide, once per draw, whether to run a program specialized to this
    /// draw's constant buffer words. `work` is the draw's estimated number of
    /// invocations of this stage; small draws are not worth a compile.
    pub fn plan(&self, res: &Resources<'_>, work: u64) -> Plan {
        let Stage::Compiled(c) = self else { return None };
        let forced = FORCE_SPEC.load(std::sync::atomic::Ordering::Relaxed);
        if std::env::var("AKUMA_SPEC").as_deref() == Ok("0")
            || (!forced && work.saturating_mul(c.prog.code.len() as u64) < 20_000_000)
        {
            return None;
        }
        let bufs: Vec<&[u8]> = c.prog.bufs.iter().map(|k| res.bufs.get(k).copied().unwrap_or(&[])).collect();
        let valid = |s: &Spec| {
            s.folded.iter().all(|&(b, addr, v)| {
                let a = addr as usize;
                let w = bufs[b as usize].get(a..a.wrapping_add(4)).map_or(0, |x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]));
                w == v
            })
        };
        let mut cache = c.spec_cache.lock().unwrap();
        if let Some(i) = cache.iter().position(|s| valid(s)) {
            let s = cache.remove(i);
            cache.push(s.clone());
            return Some(s);
        }
        let verbose = std::env::var_os("AKUMA_EXEC_VERBOSE").is_some();
        let t0 = crate::clock::monotonic();
        let mut prog = c.template.clone();
        let folded = super::opt::optimize(&mut prog, Some(&bufs));
        if prog.code.len() >= c.template.code.len() && folded.is_empty() {
            return None;
        }
        let was = prog.code.len();
        if self.is_fragment_program(&prog) {
            let pre = super::runs::analyze(&prog);
            super::memo::apply(&mut prog, pre.needs_until);
            prog.runs = super::runs::analyze(&prog);
        }
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        let (jit, wide) = match backends(&prog, &c.force, "specialized", false) {
            Ok(x) => x,
            Err(_) => return None,
        };
        if std::env::var_os("AKUMA_DUMP_SPEC").is_some() {
            for (k, i) in prog.code.iter().enumerate() {
                eprintln!("  {k:4}: {i:?}");
            }
            eprintln!("  memos {:?}", prog.memos.iter().map(|m| (&m.ins, &m.outs)).collect::<Vec<_>>());
        }
        if verbose {
            eprintln!(
                "[exec] specialized: {} -> {} insts ({} folded loads) in {:.2} ms",
                c.template.code.len(),
                was,
                folded.len(),
                (crate::clock::monotonic() - t0) * 1000.0
            );
        }
        let s = Arc::new(Spec {
            prog,
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            jit,
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            wide,
            folded,
        });
        cache.push(s.clone());
        if cache.len() > 6 {
            cache.remove(0);
        }
        Some(s)
    }

    /// fragment programs write colour target 0 (vertex programs write position)
    fn is_fragment_program(&self, p: &Program) -> bool {
        p.outputs.iter().any(|(d, _)| matches!(d, Dst::Location(0, _)))
            && !p.outputs.iter().any(|(d, _)| matches!(d, Dst::Position(_)))
    }

    pub fn begin_with<'a>(&'a self, res: &'a Resources<'a>, plan: &Plan) -> Invoker<'a> {
        match self {
            Stage::Interp(s) => Invoker::Interp { s, res },
            Stage::Compiled(c) => {
                // SAFETY: the specialization (if any) is kept alive by `hold`
                // for as long as the invoker, and its heap contents never move
                let prog: &'a Program = match plan {
                    Some(sp) => unsafe { &*(&sp.prog as *const Program) },
                    None => &c.prog,
                };
                #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                let (jit, wide): (Option<&'a super::jit::Jit>, Option<&'a super::jit::Jit>) = match plan {
                    Some(sp) => unsafe {
                        (
                            sp.jit.as_ref().map(|j| &*(j as *const super::jit::Jit)),
                            sp.wide.as_ref().map(|j| &*(j as *const super::jit::Jit)),
                        )
                    },
                    None => (c.jit.as_ref(), c.wide.as_ref()),
                };
                let hold = plan.clone();
                let bufs: Vec<&[u8]> = prog
                    .bufs
                    .iter()
                    .map(|k| res.bufs.get(k).copied().unwrap_or(&[]))
                    .collect();
                let texs: Vec<TexRef> = prog
                    .texs
                    .iter()
                    .map(|k| res.texs.get(k).copied().unwrap_or(TexRef::EMPTY))
                    .collect();
                let smps: Vec<SmpRef> = prog
                    .smps
                    .iter()
                    .map(|k| res.smps.get(k).copied().unwrap_or(SmpRef::DEFAULT))
                    .collect();
                #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                if let Some(j) = jit {
                    let refs: Vec<super::jit::BufRef> = bufs
                        .iter()
                        .map(|b| super::jit::BufRef { ptr: b.as_ptr(), len: b.len() })
                        .collect();
                    if let Some(w) = wide {
                        let scalar = Box::new(Invoker::Jit {
                            p: prog,
                            j,
                            regs: prog.init.clone(),
                            refs: refs.clone(),
                            texs: texs.clone(),
                            smps: smps.clone(),
                            hold: hold.clone(),
                        });
                        return Invoker::Wide {
                            p: prog,
                            w,
                            regs: prog.init.iter().map(|&v| super::jit::L4([v; 4])).collect(),
                            refs,
                            texs,
                            smps,
                            scalar,
                            span: SpanRegs::new(prog),
                            hold,
                        };
                    }
                    return Invoker::Jit { p: prog, j, regs: prog.init.clone(), refs, texs, smps, hold };
                }
                Invoker::Vm { p: prog, regs: prog.init.clone(), bufs, texs, smps, hold }
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
        hold: Plan,
    },
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    Jit {
        p: &'a Program,
        j: &'a super::jit::Jit,
        regs: Vec<u32>,
        refs: Vec<super::jit::BufRef>,
        texs: Vec<TexRef>,
        smps: Vec<SmpRef>,
        hold: Plan,
    },
    /// 4 invocations per call; `scalar` re-runs a batch whose lanes diverged
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    Wide {
        p: &'a Program,
        w: &'a super::jit::Jit,
        regs: Vec<super::jit::L4>,
        refs: Vec<super::jit::BufRef>,
        texs: Vec<TexRef>,
        smps: Vec<SmpRef>,
        scalar: Box<Invoker<'a>>,
        span: SpanRegs,
        hold: Plan,
    },
}

/// Where a wide stage's inputs and colour outputs live, resolved once per
/// draw for `Invoker::shade4`.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub struct SpanRegs {
    /// registers of @builtin(position).x / .y
    pos_x: Vec<u32>,
    pos_y: Vec<u32>,
    /// (register, location, component) varying inputs
    locs: Vec<(u32, u32, u32)>,
    /// registers of colour target 0's r, g, b, a (0 = the always-zero register)
    out: [u32; 4],
    /// the code overwrites an input register: re-set the inputs every batch
    inputs_clobbered: bool,
    /// indices into `locs` of the varyings that differ between the pixels of a
    /// batch (the rest are set once per span)
    dyn_idx: [u8; 32],
    n_dyn: usize,
}

/// What `Invoker::shade4` produced for up to 4 consecutive pixels.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub enum Shade4 {
    /// r, g, b, a of the 4 lanes (f32 bits)
    Colors([super::jit::L4; 4]),
    /// the lanes disagreed at a branch: run them one by one (`run_fragment`)
    Diverged,
    /// every lane was discarded
    Killed,
}

impl Invoker<'_> {
    /// `attrs`: the vertex-buffer attributes for this invocation, by location
    pub fn run_vertex(&mut self, vertex_index: u32, instance_index: u32, attrs: &Varyings) -> RawVertex {
        match self {
            Invoker::Interp { s, res } => s.run_vertex(res, vertex_index, instance_index),
            Invoker::Vm { p, regs, bufs, texs, smps, .. } => {
                vertex_in(p, regs, vertex_index, instance_index, attrs);
                vm::run(&p.code, regs, bufs, texs, smps, &p.tex_ops, &p.memos);
                vertex_out(p, regs)
            }
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Invoker::Jit { p, j, regs, refs, texs, smps, .. } => {
                vertex_in(p, regs, vertex_index, instance_index, attrs);
                j.run(regs, refs, texs, smps);
                vertex_out(p, regs)
            }
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Invoker::Wide { scalar, .. } => scalar.run_vertex(vertex_index, instance_index, attrs),
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
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Invoker::Wide { p, .. } => p,
            Invoker::Interp { .. } => return false,
        };
        prog.interp.iter().all(|(_, m)| *m == Interp::Flat)
            && !prog.inputs.iter().any(|(_, s)| matches!(s, Src::Position(_)))
    }

    pub fn run_fragment(&mut self, varyings: &Varyings, frag_pos: [f32; 4]) -> Option<[u32; 4]> {
        match self {
            Invoker::Interp { s, res } => s.run_fragment(res, varyings, frag_pos),
            Invoker::Vm { p, regs, bufs, texs, smps, .. } => {
                frag_in(p, regs, varyings, frag_pos);
                if vm::run(&p.code, regs, bufs, texs, smps, &p.tex_ops, &p.memos) {
                    return None;
                }
                Some(frag_out(p, regs))
            }
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Invoker::Jit { p, j, regs, refs, texs, smps, .. } => {
                frag_in(p, regs, varyings, frag_pos);
                if j.run(regs, refs, texs, smps) {
                    return None;
                }
                Some(frag_out(p, regs))
            }
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Invoker::Wide { scalar, .. } => scalar.run_fragment(varyings, frag_pos),
        }
    }

    /// Does the stage read `@builtin(position)` z or w (the depth / 1/w
    /// components, which cost a barycentric evaluation to produce)? x and y
    /// are free.
    pub fn uses_position_zw(&self) -> bool {
        let prog = match self {
            Invoker::Vm { p, .. } => p,
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Invoker::Jit { p, .. } => p,
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Invoker::Wide { p, .. } => p,
            Invoker::Interp { .. } => return true,
        };
        prog.inputs.iter().any(|(_, s)| matches!(s, Src::Position(2) | Src::Position(3)))
    }

    /// Wide stages: the (location, component) of each varying input the code
    /// reads, in the order `shade4_with` indexes them.
    pub fn span_locs(&self, out: &mut [(u32, u32); 32]) -> usize {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        if let Invoker::Wide { span, .. } = self {
            let mut n = 0;
            for &(_, l, c) in span.locs.iter().take(32) {
                out[n] = (l, c);
                n += 1;
            }
            return n;
        }
        let _ = out;
        0
    }

    /// Wide stages: (x runs, y runs) are usable — see `runs.rs`.
    pub fn has_runs(&self) -> (bool, bool) {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        if let Invoker::Wide { p, .. } = self {
            if super::runs::enabled() {
                return (p.runs.x.is_some(), p.runs.y.is_some());
            }
        }
        (false, false)
    }

    /// After a batch ran: how many pixels past pixel column `last_px` (up to
    /// `max`) provably produce the same result as it does.
    pub fn x_extent(&self, last_px: i64, max: usize) -> usize {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        if let Invoker::Wide { p, regs, .. } = self {
            if let Some(ax) = &p.runs.x {
                return ax.extent(&|r| regs[r as usize].0[0], last_px as f32 + 0.5, max);
            }
        }
        let _ = (last_px, max);
        0
    }

    /// As `x_extent`, in rows: how many rows after row `py` produce, for the
    /// same column, the same result.
    pub fn y_extent(&self, py: i64, max: usize) -> usize {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        if let Invoker::Wide { p, regs, .. } = self {
            if let Some(ay) = &p.runs.y {
                return ay.extent(&|r| regs[r as usize].0[0], py as f32 + 0.5, max);
            }
        }
        let _ = (py, max);
        0
    }

    /// Wide stages only: set the inputs that stay fixed along a run of
    /// pixels on one row of a flat-varying primitive (the varyings and the
    /// row's y). Returns false for any other executor.
    pub fn span_begin(&mut self, var: &Varyings, py: f32, dyn_locs: &[u8]) -> bool {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        if let Invoker::Wide { regs, span, .. } = self {
            span.n_dyn = dyn_locs.len().min(32);
            span.dyn_idx[..span.n_dyn].copy_from_slice(&dyn_locs[..span.n_dyn]);
            for &(r, l, c) in &span.locs {
                regs[r as usize].0 = [var[l as usize][c as usize]; 4];
            }
            for &r in &span.pos_y {
                regs[r as usize].0 = [py.to_bits(); 4];
            }
            return true;
        }
        let _ = (var, py, dyn_locs);
        false
    }

    /// Shade the `n` (1..=4) pixels at x = `x0 + 0.5 ..` of the row `span_begin`
    /// was given (flat varyings). Lanes past `n` repeat the last pixel so they
    /// cannot diverge.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    #[inline]
    pub fn shade4(&mut self, var: &Varyings, x0: f32, py: f32, n: usize) -> Shade4 {
        let Invoker::Wide { span, .. } = self else { unreachable!() };
        // (location, component) per input, in `span_locs` order
        let idx: [(u32, u32); 32] = {
            let mut a = [(0, 0); 32];
            for (i, &(_, l, c)) in span.locs.iter().enumerate().take(32) {
                a[i] = (l, c);
            }
            a
        };
        self.shade4_with(x0, py, n, |_, i| var[idx[i].0 as usize][idx[i].1 as usize])
    }

    /// As `shade4`, with the varying inputs supplied per lane: `f(lane, location,
    /// component)` gives the f32 bits.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    #[inline]
    pub fn shade4_with(&mut self, x0: f32, py: f32, n: usize, mut f: impl FnMut(usize, usize) -> u32) -> Shade4 {
        let Invoker::Wide { w, regs, refs, texs, smps, span, .. } = self else {
            unreachable!("shade4 on a non-wide invoker")
        };
        let mut xs = [0u32; 4];
        for (k, x) in xs.iter_mut().enumerate() {
            *x = (x0 + k.min(n - 1) as f32 + 0.5).to_bits();
        }
        for &r in &span.pos_x {
            regs[r as usize].0 = xs;
        }
        // varyings that differ per lane (or inputs the code overwrites) are
        // written for every batch
        if span.inputs_clobbered {
            for &r in &span.pos_y {
                regs[r as usize].0 = [py.to_bits(); 4];
            }
            for (i, &(r, _, _)) in span.locs.iter().enumerate() {
                let mut v = [0u32; 4];
                for (k, x) in v.iter_mut().enumerate() {
                    *x = f(k.min(n - 1), i);
                }
                regs[r as usize].0 = v;
            }
        } else {
            for j in 0..span.n_dyn {
                let i = span.dyn_idx[j] as usize;
                let mut v = [0u32; 4];
                for (k, x) in v.iter_mut().enumerate() {
                    *x = f(k.min(n - 1), i);
                }
                regs[span.locs[i].0 as usize].0 = v;
            }
        }
        let status = w.run_wide(regs, refs, texs, smps);
        if super::prof::enabled() {
            super::prof::inc(11, 1);
            if status == super::jit::STATUS_DIVERGED {
                super::prof::inc(12, 1);
            }
        }
        match status {
            0 => Shade4::Colors([
                regs[span.out[0] as usize],
                regs[span.out[1] as usize],
                regs[span.out[2] as usize],
                regs[span.out[3] as usize],
            ]),
            1 => Shade4::Killed,
            _ => Shade4::Diverged,
        }
    }

    /// invocations this invoker prefers to be handed at once
    pub fn lanes(&self) -> usize {
        match self {
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Invoker::Wide { .. } => 4,
            _ => 1,
        }
    }

    /// Run several vertex invocations (`ids` = (vertex index, instance), with
    /// matching `attrs`), appending results in order. Wide invokers run 4 at
    /// a time; everyone else loops.
    pub fn run_vertex_batch(&mut self, ids: &[(u32, u32)], attrs: &[Varyings], out: &mut Vec<RawVertex>) {
        debug_assert_eq!(ids.len(), attrs.len());
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        if let Invoker::Wide { p, w, regs, refs, texs, smps, scalar, .. } = self {
            for (ci, chunk) in ids.chunks(4).enumerate() {
                let base = ci * 4;
                for &(r, src) in &p.inputs {
                    let mut lanes = [0u32; 4];
                    for (k, l) in lanes.iter_mut().enumerate() {
                        let i = k.min(chunk.len() - 1);
                        *l = match src {
                            Src::VertexIndex => chunk[i].0,
                            Src::InstanceIndex => chunk[i].1,
                            Src::Location(loc, c) => attrs[base + i][loc as usize][c as usize],
                            Src::Position(_) => 0,
                        };
                    }
                    regs[r as usize].0 = lanes;
                }
                match w.run_wide(regs, refs, texs, smps) {
                    0 | 1 => {
                        for k in 0..chunk.len() {
                            let mut v = RawVertex::default();
                            for &(d, r) in &p.outputs {
                                let bits = regs[r as usize].0[k];
                                match d {
                                    Dst::Position(i) => v.position[i as usize] = f32::from_bits(bits),
                                    Dst::Location(l, c) => v.varyings[l as usize][c as usize] = bits,
                                }
                            }
                            out.push(v);
                        }
                    }
                    _ => {
                        // lanes diverged: one by one
                        for (k, &(vi, ii)) in chunk.iter().enumerate() {
                            out.push(scalar.run_vertex(vi, ii, &attrs[base + k]));
                        }
                    }
                }
            }
            return;
        }
        for (i, &(vi, ii)) in ids.iter().enumerate() {
            out.push(self.run_vertex(vi, ii, &attrs[i]));
        }
    }

    /// Fragment counterpart of `run_vertex_batch`: (frag position, varyings)
    /// per invocation, results into `out` (same length).
    pub fn run_fragment_batch(&mut self, ins: &[([f32; 4], &Varyings)], out: &mut [Option<[u32; 4]>]) {
        debug_assert_eq!(ins.len(), out.len());
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        if let Invoker::Wide { p, w, regs, refs, texs, smps, scalar, .. } = self {
            for (ci, chunk) in ins.chunks(4).enumerate() {
                let base = ci * 4;
                for &(r, src) in &p.inputs {
                    let mut lanes = [0u32; 4];
                    for (k, l) in lanes.iter_mut().enumerate() {
                        let i = k.min(chunk.len() - 1);
                        *l = match src {
                            Src::Position(c) => chunk[i].0[c as usize].to_bits(),
                            Src::Location(loc, c) => chunk[i].1[loc as usize][c as usize],
                            _ => 0,
                        };
                    }
                    regs[r as usize].0 = lanes;
                }
                let status = w.run_wide(regs, refs, texs, smps);
                if super::prof::enabled() {
                    super::prof::inc(11, 1);
                    if status == super::jit::STATUS_DIVERGED {
                        super::prof::inc(12, 1);
                    }
                }
                match status {
                    0 => {
                        for k in 0..chunk.len() {
                            let mut color = [0u32; 4];
                            for &(d, r) in &p.outputs {
                                if let Dst::Location(0, c) = d {
                                    color[c as usize] = regs[r as usize].0[k];
                                }
                            }
                            out[base + k] = Some(color);
                        }
                    }
                    1 => {
                        for k in 0..chunk.len() {
                            out[base + k] = None;
                        }
                    }
                    _ => {
                        for (k, (pos, var)) in chunk.iter().enumerate() {
                            out[base + k] = scalar.run_fragment(var, *pos);
                        }
                    }
                }
            }
            return;
        }
        for (i, (pos, var)) in ins.iter().enumerate() {
            out[i] = self.run_fragment(var, *pos);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
impl SpanRegs {
    fn new(p: &Program) -> SpanRegs {
        let mut s = SpanRegs {
            pos_x: vec![],
            pos_y: vec![],
            locs: vec![],
            out: [0; 4],
            dyn_idx: [0; 32],
            n_dyn: 0,
            inputs_clobbered: p.code.iter().any(|i| {
                i.dst().is_some_and(|d| p.inputs.iter().any(|&(r, _)| r == d))
            }) || p.tex_ops.iter().any(|t| p.inputs.iter().any(|&(r, _)| r >= t.d && r < t.d + 4)),
        };
        for &(r, src) in &p.inputs {
            match src {
                Src::Position(0) => s.pos_x.push(r),
                Src::Position(1) => s.pos_y.push(r),
                Src::Location(l, c) => s.locs.push((r, l, c as u32)),
                _ => {}
            }
        }
        for &(d, r) in &p.outputs {
            if let Dst::Location(0, c) = d {
                s.out[c as usize] = r;
            }
        }
        s
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
