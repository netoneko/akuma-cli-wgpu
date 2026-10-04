//! naga IR → `Program` lowering (see `program.rs`).
//!
//! Runs once per pipeline stage, at pipeline creation. Anything this pass
//! does not understand returns `Err`, and the stage silently stays on the
//! tree-walking interpreter — so this file may grow feature by feature
//! without ever being a correctness gate.
//!
//! How values are represented while lowering:
//!
//! * a `Lv` is a tree whose leaves are scalar registers (`S(reg, kind)`);
//!   vectors, matrices (columns of rows), arrays and structs are `A(children)`.
//!   No aggregate exists at run time.
//! * a local `var` owns *mutable* registers; every `Load` copies them into
//!   fresh ones, so a later `Store` cannot change a value already read.
//!   Everything else is single-assignment.
//! * expressions are evaluated where naga's `Emit` statements put them
//!   (pre-emit kinds — literals, constants, arguments, variable references —
//!   are evaluated once at function entry, so control flow never leaves one
//!   half-defined).
//! * user functions are inlined at their call sites.
//!
//! Arithmetic mirrors `interp.rs` operation for operation (same order, same
//! accumulator seeds in dot/length/normalize) so that all executors agree
//! bit for bit on the shaders that matter.

use std::collections::HashMap;

use naga::{
    AddressSpace, BinaryOperator as B, BuiltIn, Expression, Function, Handle, Literal,
    MathFunction as M, ScalarKind, TypeInner, UnaryOperator,
};

use super::interp::{
    align_of, array_stride, member_offsets, size_of, vec_stride, vsize, w, Shader,
};
use super::program::{Cmp, Dst, Fun, Inst, Interp, Program, Src, R};
use super::texture::{Elem, TexKind, TexOp};

type Res<T> = Result<T, String>;

macro_rules! bail {
    ($($t:tt)*) => { return Err(format!($($t)*)) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum K {
    F,
    I,
    U,
    B,
}

#[derive(Clone, Debug)]
enum Lv {
    S(R, K),
    A(Vec<Lv>),
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Ty(Handle<naga::Type>),
    Scalar(naga::Scalar),
    Vec(u32, naga::Scalar),
}

#[derive(Clone, Debug)]
enum Ptr {
    /// a sub-tree of a local variable's registers
    Local(Lv),
    /// a local aggregate indexed by a run-time value: the candidates, each
    /// with the register holding "this is the selected one"
    LocalDyn(Vec<(R, Lv)>),
    Buf { buf: u32, imm: u32, dynr: Option<R>, shape: Shape },
}

#[derive(Clone, Debug)]
enum Val {
    V(Lv),
    P(Ptr),
    /// a bound texture (slot, element type) / sampler (slot)
    Tex(u32, Elem),
    Smp(u32),
}

struct Ctx<'m> {
    fun: &'m Function,
    memo: Vec<Option<Val>>,
    locals: Vec<Lv>,
    args: Vec<Val>,
    ret: Option<Lv>,
    end: u32,
    breaks: Vec<u32>,
    conts: Vec<u32>,
}

const MAX_INLINE_DEPTH: u32 = 16;
const MAX_CODE: usize = 200_000;

struct Lowerer<'m> {
    m: &'m naga::Module,
    code: Vec<Inst>,
    nregs: u32,
    init: Vec<(R, u32)>,
    cmap: HashMap<u32, R>,
    kconst: HashMap<R, u32>,
    bufs: Vec<(u32, u32)>,
    inputs: Vec<(R, Src)>,
    interp: Vec<(u32, Interp)>,
    texs: Vec<(u32, u32)>,
    smps: Vec<(u32, u32)>,
    tex_ops: Vec<TexOp>,
    labels: Vec<Option<u32>>,
    gmemo: Vec<Option<Lv>>,
}

pub fn compile(sh: &Shader, entry: usize) -> Res<Program> {
    compile_with_template(sh, entry).map(|(p, _)| p)
}

/// The program, and the same program before memoization (the starting point
/// for per-draw specialization, which must memoize after it has folded).
pub fn compile_with_template(sh: &Shader, entry: usize) -> Res<(Program, Program)> {
    let m = &sh.module;
    let mut lw = Lowerer {
        m,
        code: Vec::new(),
        nregs: 1, // r0 = 0
        init: Vec::new(),
        cmap: HashMap::new(),
        kconst: HashMap::new(),
        bufs: Vec::new(),
        inputs: Vec::new(),
        interp: Vec::new(),
        texs: Vec::new(),
        smps: Vec::new(),
        tex_ops: Vec::new(),
        labels: Vec::new(),
        gmemo: vec![None; m.global_expressions.len()],
    };
    lw.cmap.insert(0, 0);
    lw.kconst.insert(0, 0);
    let ep = &m.entry_points[entry];
    let f = &ep.function;

    // entry arguments
    let mut args = Vec::new();
    for a in &f.arguments {
        args.push(Val::V(lw.make_input(a.ty, &a.binding)?));
    }
    let ret = lw.lower_function(f, args, 0)?;

    let mut outputs = Vec::new();
    if let (Some(res), Some(lv)) = (&f.result, &ret) {
        lw.collect_outputs(res.ty, &res.binding, lv, &mut outputs)?;
    }
    lw.push(Inst::Ret);
    lw.resolve_labels()?;

    // drop inputs nothing reads (e.g. an unused `@builtin(position)` member of
    // a fragment input struct): they need not be written per invocation, and
    // "does this stage depend on position?" becomes answerable
    {
        let mut read = vec![false; lw.nregs as usize];
        for i in &lw.code {
            i.reads(|r| read[r as usize] = true);
        }
        for op in &lw.tex_ops {
            read[op.x as usize] = true;
            read[op.y as usize] = true;
        }
        for (_, r) in &outputs {
            read[*r as usize] = true;
        }
        lw.inputs.retain(|(r, _)| read[*r as usize]);
    }
    let mut init = vec![0u32; lw.nregs as usize];
    for (r, v) in &lw.init {
        init[*r as usize] = *v;
    }
    let mut prog = Program {
        code: lw.code,
        nregs: lw.nregs,
        init,
        bufs: lw.bufs,
        inputs: lw.inputs,
        outputs,
        interp: lw.interp,
        texs: lw.texs,
        smps: lw.smps,
        tex_ops: lw.tex_ops,
        memos: Vec::new(),
    };
    super::opt::optimize(&mut prog, None);
    let template = prog.clone();
    if ep.stage == naga::ShaderStage::Fragment {
        super::memo::apply(&mut prog);
    }
    Ok((prog, template))
}

fn scalar_k(s: naga::Scalar) -> Res<K> {
    Ok(match (s.kind, s.width) {
        (ScalarKind::Float, 4) => K::F,
        (ScalarKind::Sint, 4) => K::I,
        (ScalarKind::Uint, 4) => K::U,
        (ScalarKind::Bool, _) => K::B,
        (k, wd) => bail!("scalar {k:?} x{wd}"),
    })
}

fn kind_k(k: ScalarKind) -> Res<K> {
    Ok(match k {
        ScalarKind::Float => K::F,
        ScalarKind::Sint => K::I,
        ScalarKind::Uint => K::U,
        ScalarKind::Bool => K::B,
        k => bail!("kind {k:?}"),
    })
}

fn leaves(lv: &Lv, out: &mut Vec<(R, K)>) {
    match lv {
        Lv::S(r, k) => out.push((*r, *k)),
        Lv::A(v) => v.iter().for_each(|c| leaves(c, out)),
    }
}

fn is_mat(lv: &Lv) -> bool {
    matches!(lv, Lv::A(items) if matches!(items.first(), Some(Lv::A(_))))
}

impl<'m> Lowerer<'m> {
    // -- emission helpers ---------------------------------------------------

    fn push(&mut self, i: Inst) {
        self.code.push(i);
    }

    fn nr(&mut self) -> R {
        self.nregs += 1;
        self.nregs - 1
    }

    fn konst(&mut self, bits: u32, k: K) -> Lv {
        if let Some(&r) = self.cmap.get(&bits) {
            return Lv::S(r, k);
        }
        let r = self.nr();
        self.init.push((r, bits));
        self.cmap.insert(bits, r);
        self.kconst.insert(r, bits);
        Lv::S(r, k)
    }

    fn fconst(&mut self, x: f32) -> (R, K) {
        match self.konst(x.to_bits(), K::F) {
            Lv::S(r, k) => (r, k),
            _ => unreachable!(),
        }
    }

    fn new_label(&mut self) -> u32 {
        self.labels.push(None);
        (self.labels.len() - 1) as u32
    }

    fn place(&mut self, l: u32) {
        self.labels[l as usize] = Some(self.code.len() as u32);
    }

    fn resolve_labels(&mut self) -> Res<()> {
        for i in self.code.iter_mut() {
            let t = match i {
                Inst::Jmp { t } | Inst::Jz { t, .. } | Inst::Jnz { t, .. } => t,
                _ => continue,
            };
            *t = self.labels[*t as usize].ok_or("unplaced label")?;
        }
        Ok(())
    }

    fn check_size(&self) -> Res<()> {
        if self.code.len() > MAX_CODE {
            bail!("program too large")
        }
        Ok(())
    }

    // -- types ---------------------------------------------------------------

    fn alloc(&mut self, ty: Handle<naga::Type>) -> Res<Lv> {
        Ok(match &self.m.types[ty].inner {
            TypeInner::Scalar(s) => Lv::S(self.nr(), scalar_k(*s)?),
            TypeInner::Vector { size, scalar } => {
                let k = scalar_k(*scalar)?;
                Lv::A((0..vsize(*size)).map(|_| Lv::S(self.nr(), k)).collect())
            }
            TypeInner::Matrix { columns, rows, scalar } => {
                let k = scalar_k(*scalar)?;
                Lv::A(
                    (0..vsize(*columns))
                        .map(|_| Lv::A((0..vsize(*rows)).map(|_| Lv::S(self.nr(), k)).collect()))
                        .collect(),
                )
            }
            TypeInner::Array { base, size, .. } => {
                let n = match size {
                    naga::ArraySize::Constant(c) => c.get(),
                    _ => bail!("unsized array value"),
                };
                let base = *base;
                let mut v = Vec::new();
                for _ in 0..n {
                    v.push(self.alloc(base)?);
                }
                Lv::A(v)
            }
            TypeInner::Struct { members, .. } => {
                let tys: Vec<_> = members.iter().map(|mm| mm.ty).collect();
                let mut v = Vec::new();
                for t in tys {
                    v.push(self.alloc(t)?);
                }
                Lv::A(v)
            }
            other => bail!("alloc type {other:?}"),
        })
    }

    fn zeros_like(&mut self, ty: Handle<naga::Type>) -> Res<Lv> {
        let lv = self.alloc(ty)?;
        let mut ls = Vec::new();
        leaves(&lv, &mut ls);
        // freshly allocated regs: make them hoisted zeros by aliasing r0
        fn map(lv: &Lv) -> Lv {
            match lv {
                Lv::S(_, k) => Lv::S(0, *k),
                Lv::A(v) => Lv::A(v.iter().map(map).collect()),
            }
        }
        Ok(map(&lv))
    }

    // -- interface -------------------------------------------------------------

    fn make_input(&mut self, ty: Handle<naga::Type>, binding: &Option<naga::Binding>) -> Res<Lv> {
        match binding {
            None => match &self.m.types[ty].inner {
                TypeInner::Struct { members, .. } => {
                    let members = members.clone();
                    let mut v = Vec::new();
                    for mm in &members {
                        v.push(self.make_input(mm.ty, &mm.binding)?);
                    }
                    Ok(Lv::A(v))
                }
                other => bail!("unbound entry argument {other:?}"),
            },
            Some(b) => {
                let lv = self.alloc(ty)?;
                let mut ls = Vec::new();
                leaves(&lv, &mut ls);
                if let naga::Binding::Location { location, interpolation, .. } = b {
                    // integers cannot be interpolated; floats default to perspective
                    let int_typed = ls.first().is_some_and(|(_, k)| *k != K::F);
                    let mode = match interpolation {
                        _ if int_typed => Interp::Flat,
                        Some(naga::Interpolation::Flat) => Interp::Flat,
                        Some(naga::Interpolation::Linear) => Interp::Linear,
                        _ => Interp::Perspective,
                    };
                    self.interp.push((*location, mode));
                }
                for (i, (r, _)) in ls.iter().enumerate() {
                    let src = match b {
                        naga::Binding::BuiltIn(BuiltIn::VertexIndex) => Src::VertexIndex,
                        naga::Binding::BuiltIn(BuiltIn::InstanceIndex) => Src::InstanceIndex,
                        naga::Binding::BuiltIn(BuiltIn::Position { .. }) => Src::Position(i as u8),
                        naga::Binding::Location { location, .. } => Src::Location(*location, i as u8),
                        other => bail!("entry input binding {other:?}"),
                    };
                    self.inputs.push((*r, src));
                }
                Ok(lv)
            }
        }
    }

    fn collect_outputs(
        &mut self,
        ty: Handle<naga::Type>,
        binding: &Option<naga::Binding>,
        lv: &Lv,
        out: &mut Vec<(Dst, R)>,
    ) -> Res<()> {
        match binding {
            None => {
                let members = match &self.m.types[ty].inner {
                    TypeInner::Struct { members, .. } => members.clone(),
                    other => bail!("unbound entry result {other:?}"),
                };
                let Lv::A(items) = lv else { bail!("struct result shape") };
                for (mm, item) in members.iter().zip(items) {
                    self.collect_outputs(mm.ty, &mm.binding, item, out)?;
                }
                Ok(())
            }
            Some(b) => {
                let mut ls = Vec::new();
                leaves(lv, &mut ls);
                for (i, (r, _)) in ls.iter().enumerate() {
                    let d = match b {
                        naga::Binding::BuiltIn(BuiltIn::Position { .. }) => Dst::Position(i as u8),
                        naga::Binding::Location { location, .. } => Dst::Location(*location, i as u8),
                        other => bail!("entry output binding {other:?}"),
                    };
                    out.push((d, *r));
                }
                Ok(())
            }
        }
    }

    // -- functions ------------------------------------------------------------

    fn lower_function(&mut self, f: &'m Function, args: Vec<Val>, depth: u32) -> Res<Option<Lv>> {
        if depth > MAX_INLINE_DEPTH {
            bail!("inline depth")
        }
        let end = self.new_label();
        let mut ctx = Ctx {
            fun: f,
            memo: vec![None; f.expressions.len()],
            locals: Vec::new(),
            args,
            ret: None,
            end,
            breaks: Vec::new(),
            conts: Vec::new(),
        };
        // local variables: mutable registers, zero-initialized per invocation
        for (_, lv) in f.local_variables.iter() {
            let l = self.alloc(lv.ty)?;
            let mut ls = Vec::new();
            leaves(&l, &mut ls);
            for (r, _) in ls {
                self.push(Inst::Const { d: r, v: 0 });
            }
            ctx.locals.push(l);
        }
        if let Some(res) = &f.result {
            ctx.ret = Some(self.alloc(res.ty)?);
        }
        // Expressions naga never `Emit`s: literals/constants, argument and
        // variable references, and anything built purely from constants
        // (`vec4(0.0)`, `array(1.0, 2.0)`, `2.0 * 3.0`, ...). All are pure, so
        // evaluate them once here, before any control flow.
        let mut is_const = vec![false; f.expressions.len()];
        for (h, e) in f.expressions.iter() {
            let c = |x: &Handle<Expression>| is_const[x.index()];
            let k = match e {
                Expression::Literal(_) | Expression::Constant(_) | Expression::ZeroValue(_) => true,
                Expression::Compose { components, .. } => components.iter().all(c),
                Expression::Splat { value, .. } => c(value),
                Expression::Swizzle { vector, .. } => c(vector),
                Expression::Access { base, index } => c(base) && c(index),
                Expression::AccessIndex { base, .. } => c(base),
                Expression::Unary { expr, .. } | Expression::As { expr, .. } => c(expr),
                Expression::Binary { left, right, .. } => c(left) && c(right),
                Expression::Select { condition, accept, reject } => {
                    c(condition) && c(accept) && c(reject)
                }
                Expression::Math { arg, arg1, arg2, arg3, .. } => {
                    c(arg) && [arg1, arg2, arg3].into_iter().flatten().all(c)
                }
                _ => false,
            };
            is_const[h.index()] = k;
            if k || matches!(
                e,
                Expression::FunctionArgument(_)
                    | Expression::GlobalVariable(_)
                    | Expression::LocalVariable(_)
            ) {
                let v = self.expr(&mut ctx, h)?;
                ctx.memo[h.index()] = Some(v);
            }
        }
        for (h, lv) in f.local_variables.iter() {
            if let Some(init) = lv.init {
                let v = self.val(&mut ctx, init)?;
                let dst = ctx.locals[h.index()].clone();
                self.store_lv(&dst, &v)?;
            }
        }
        self.block(&mut ctx, &f.body)?;
        self.place(end);
        Ok(ctx.ret)
    }

    fn val(&mut self, ctx: &mut Ctx<'m>, h: Handle<Expression>) -> Res<Lv> {
        match self.get(ctx, h)? {
            Val::V(l) => Ok(l),
            _ => bail!("pointer/handle where a value was expected"),
        }
    }

    /// The value captured at the expression's Emit/Call. naga does not Emit
    /// pointer chains (`a.x = ..`, `arr[i]`), so those are rebuilt at each use
    /// and deliberately not memoized — a dynamic index must be re-read, and
    /// whatever code it emits belongs to the path that uses it. Anything else
    /// that was never emitted is a lowering bug / unsupported shape.
    fn get(&mut self, ctx: &mut Ctx<'m>, h: Handle<Expression>) -> Res<Val> {
        if let Some(v) = &ctx.memo[h.index()] {
            return Ok(v.clone());
        }
        match &ctx.fun.expressions[h] {
            Expression::Access { .. } | Expression::AccessIndex { .. } => self.expr(ctx, h),
            other => Err(format!(
                "expression {} ({:?}) used before its Emit",
                h.index(),
                std::mem::discriminant(other)
            )),
        }
    }

    // -- statements -------------------------------------------------------------

    fn block(&mut self, ctx: &mut Ctx<'m>, b: &'m naga::Block) -> Res<()> {
        for s in b.iter() {
            self.stmt(ctx, s)?;
            self.check_size()?;
        }
        Ok(())
    }

    fn cond_reg(lv: &Lv) -> Res<R> {
        match lv {
            Lv::S(r, K::B) => Ok(*r),
            other => bail!("condition {other:?}"),
        }
    }

    fn stmt(&mut self, ctx: &mut Ctx<'m>, s: &'m naga::Statement) -> Res<()> {
        use naga::Statement as S;
        match s {
            S::Emit(range) => {
                for h in range.clone() {
                    let v = self.expr(ctx, h)?;
                    ctx.memo[h.index()] = Some(v);
                }
            }
            S::Block(b) => self.block(ctx, b)?,
            S::If { condition, accept, reject } => {
                let c = Self::cond_reg(&self.val(ctx, *condition)?)?;
                let l_else = self.new_label();
                let l_end = self.new_label();
                self.push(Inst::Jz { c, t: l_else });
                self.block(ctx, accept)?;
                if !reject.is_empty() {
                    self.push(Inst::Jmp { t: l_end });
                }
                self.place(l_else);
                if !reject.is_empty() {
                    self.block(ctx, reject)?;
                    self.place(l_end);
                }
            }
            S::Loop { body, continuing, break_if } => {
                let top = self.new_label();
                let cont = self.new_label();
                let end = self.new_label();
                self.place(top);
                ctx.breaks.push(end);
                ctx.conts.push(cont);
                self.block(ctx, body)?;
                self.place(cont);
                self.block(ctx, continuing)?;
                if let Some(bi) = break_if {
                    let c = Self::cond_reg(&self.val(ctx, *bi)?)?;
                    self.push(Inst::Jnz { c, t: end });
                }
                self.push(Inst::Jmp { t: top });
                ctx.breaks.pop();
                ctx.conts.pop();
                self.place(end);
            }
            S::Switch { selector, cases } => {
                let Lv::S(sel, _) = self.val(ctx, *selector)? else { bail!("switch selector") };
                let end = self.new_label();
                let mut body_labels = Vec::new();
                let mut default = None;
                for c in cases.iter() {
                    if c.fall_through && !c.body.is_empty() {
                        bail!("switch fall-through")
                    }
                    let l = self.new_label();
                    body_labels.push(l);
                    match c.value {
                        naga::SwitchValue::Default => default = Some(l),
                        naga::SwitchValue::I32(v) => {
                            let Lv::S(k, _) = self.konst(v as u32, K::I) else { unreachable!() };
                            let t = self.nr();
                            self.push(Inst::Cmp { d: t, a: sel, b: k, c: Cmp::IEq });
                            self.push(Inst::Jnz { c: t, t: l });
                        }
                        naga::SwitchValue::U32(v) => {
                            let Lv::S(k, _) = self.konst(v, K::U) else { unreachable!() };
                            let t = self.nr();
                            self.push(Inst::Cmp { d: t, a: sel, b: k, c: Cmp::IEq });
                            self.push(Inst::Jnz { c: t, t: l });
                        }
                    }
                }
                self.push(Inst::Jmp { t: default.unwrap_or(end) });
                ctx.breaks.push(end);
                for (c, l) in cases.iter().zip(body_labels) {
                    self.place(l);
                    self.block(ctx, &c.body)?;
                    self.push(Inst::Jmp { t: end });
                }
                ctx.breaks.pop();
                self.place(end);
            }
            S::Break => {
                let t = *ctx.breaks.last().ok_or("break outside loop")?;
                self.push(Inst::Jmp { t });
            }
            S::Continue => {
                let t = *ctx.conts.last().ok_or("continue outside loop")?;
                self.push(Inst::Jmp { t });
            }
            S::Return { value } => {
                if let Some(v) = value {
                    let v = self.val(ctx, *v)?;
                    let dst = ctx.ret.clone().ok_or("return value without result type")?;
                    self.store_lv(&dst, &v)?;
                }
                self.push(Inst::Jmp { t: ctx.end });
            }
            S::Kill => self.push(Inst::Kill),
            S::Store { pointer, value } => {
                let v = self.val(ctx, *value)?;
                match self.get(ctx, *pointer)? {
                    Val::P(Ptr::Local(dst)) => self.store_lv(&dst, &v)?,
                    Val::P(Ptr::LocalDyn(cands)) => {
                        // conditional store into each candidate:
                        // dst = flag ? value : dst
                        for (flag, dst) in &cands {
                            self.store_lv_if(*flag, dst, &v)?;
                        }
                    }
                    other => bail!("store through {other:?}"),
                }
            }
            S::Call { function, arguments, result } => {
                let mut args = Vec::new();
                for a in arguments {
                    args.push(self.get(ctx, *a)?);
                }
                let f = &self.m.functions[*function];
                let ret = self.lower_function(f, args, 1)?;
                if let (Some(slot), Some(r)) = (result, ret) {
                    ctx.memo[slot.index()] = Some(Val::V(r));
                }
            }
            other => bail!("statement {:?}", std::mem::discriminant(other)),
        }
        Ok(())
    }

    fn store_lv(&mut self, dst: &Lv, src: &Lv) -> Res<()> {
        match (dst, src) {
            (Lv::S(d, _), Lv::S(s, _)) => {
                if d != s {
                    self.push(Inst::Mov { d: *d, s: *s });
                }
                Ok(())
            }
            (Lv::A(a), Lv::A(b)) if a.len() == b.len() => {
                for (x, y) in a.iter().zip(b) {
                    self.store_lv(x, y)?;
                }
                Ok(())
            }
            _ => bail!("store shape mismatch"),
        }
    }

    fn store_lv_if(&mut self, flag: R, dst: &Lv, src: &Lv) -> Res<()> {
        match (dst, src) {
            (Lv::S(d, _), Lv::S(s, _)) => {
                self.push(Inst::Select { d: *d, c: flag, a: *s, b: *d });
                Ok(())
            }
            (Lv::A(a), Lv::A(b)) if a.len() == b.len() => {
                for (x, y) in a.iter().zip(b) {
                    self.store_lv_if(flag, x, y)?;
                }
                Ok(())
            }
            _ => bail!("store shape mismatch"),
        }
    }

    fn copy_lv(&mut self, lv: &Lv) -> Lv {
        match lv {
            Lv::S(r, k) => {
                let d = self.nr();
                self.push(Inst::Mov { d, s: *r });
                Lv::S(d, *k)
            }
            Lv::A(v) => Lv::A(v.iter().map(|c| self.copy_lv(c)).collect()),
        }
    }

    // -- expressions ------------------------------------------------------------

    fn expr(&mut self, ctx: &mut Ctx<'m>, h: Handle<Expression>) -> Res<Val> {
        let e = &ctx.fun.expressions[h];
        Ok(match e {
            Expression::Literal(l) => Val::V(self.literal(*l)?),
            Expression::Constant(c) => {
                let init = self.m.constants[*c].init;
                Val::V(self.gexpr(init)?)
            }
            Expression::ZeroValue(ty) => Val::V(self.zeros_like(*ty)?),
            Expression::FunctionArgument(i) => ctx.args[*i as usize].clone(),
            Expression::GlobalVariable(g) => {
                let gv = &self.m.global_variables[*g];
                match gv.space {
                    AddressSpace::Uniform | AddressSpace::Storage { access: _ } => {
                        let b = gv.binding.as_ref().ok_or("resource without binding")?;
                        let slot = match self.bufs.iter().position(|&x| x == (b.group, b.binding)) {
                            Some(i) => i,
                            None => {
                                self.bufs.push((b.group, b.binding));
                                self.bufs.len() - 1
                            }
                        };
                        Val::P(Ptr::Buf {
                            buf: slot as u32,
                            imm: 0,
                            dynr: None,
                            shape: Shape::Ty(gv.ty),
                        })
                    }
                    AddressSpace::Handle => {
                        let b = gv.binding.as_ref().ok_or("resource without binding")?;
                        let key = (b.group, b.binding);
                        match &self.m.types[gv.ty].inner {
                            TypeInner::Image { dim: naga::ImageDimension::D2, arrayed: false, class, .. } => {
                                let elem = match class {
                                    naga::ImageClass::Sampled { kind, multi: false } => match kind {
                                        ScalarKind::Float => Elem::F,
                                        ScalarKind::Uint => Elem::U,
                                        ScalarKind::Sint => Elem::I,
                                        k => bail!("texture of {k:?}"),
                                    },
                                    other => bail!("image class {other:?}"),
                                };
                                let slot = match self.texs.iter().position(|&x| x == key) {
                                    Some(i) => i,
                                    None => {
                                        self.texs.push(key);
                                        self.texs.len() - 1
                                    }
                                };
                                Val::Tex(slot as u32, elem)
                            }
                            TypeInner::Sampler { comparison: false } => {
                                let slot = match self.smps.iter().position(|&x| x == key) {
                                    Some(i) => i,
                                    None => {
                                        self.smps.push(key);
                                        self.smps.len() - 1
                                    }
                                };
                                Val::Smp(slot as u32)
                            }
                            other => bail!("handle global of type {other:?}"),
                        }
                    }
                    other => bail!("global in {other:?}"),
                }
            }
            Expression::LocalVariable(l) => Val::P(Ptr::Local(ctx.locals[l.index()].clone())),
            Expression::Load { pointer } => match self.get(ctx, *pointer)? {
                Val::P(Ptr::Local(lv)) => Val::V(self.copy_lv(&lv)),
                Val::P(Ptr::LocalDyn(cands)) => {
                    // select chain over the candidates (the first is the default)
                    let mut acc = self.copy_lv(&cands[0].1);
                    for (flag, lv) in &cands[1..] {
                        acc = self.select(&Lv::S(*flag, K::B), lv, &acc)?;
                    }
                    Val::V(acc)
                }
                Val::P(Ptr::Buf { buf, imm, dynr, shape }) => {
                    Val::V(self.load_buf(buf, imm, dynr, shape)?)
                }
                Val::V(_) | Val::Tex(..) | Val::Smp(..) => bail!("load of a value / handle"),
            },
            Expression::Compose { ty, components } => {
                let mut comps = Vec::new();
                for c in components {
                    comps.push(self.val(ctx, *c)?);
                }
                Val::V(self.compose(*ty, comps)?)
            }
            Expression::Splat { size, value } => {
                let v = self.val(ctx, *value)?;
                Val::V(Lv::A(vec![v; vsize(*size) as usize]))
            }
            Expression::Swizzle { size, vector, pattern } => {
                let Lv::A(items) = self.val(ctx, *vector)? else { bail!("swizzle of scalar") };
                let n = vsize(*size) as usize;
                Val::V(Lv::A(
                    (0..n)
                        .map(|i| {
                            items[match pattern[i] {
                                naga::SwizzleComponent::X => 0,
                                naga::SwizzleComponent::Y => 1,
                                naga::SwizzleComponent::Z => 2,
                                naga::SwizzleComponent::W => 3,
                            }]
                            .clone()
                        })
                        .collect(),
                ))
            }
            Expression::AccessIndex { base, index } => {
                let b = self.get(ctx, *base)?;
                self.access_static(b, *index)?
            }
            Expression::Access { base, index } => {
                let b = self.get(ctx, *base)?;
                let i = self.val(ctx, *index)?;
                self.access_dyn(b, i)?
            }
            Expression::Unary { op, expr } => {
                let v = self.val(ctx, *expr)?;
                Val::V(self.unary(*op, &v)?)
            }
            Expression::Binary { op, left, right } => {
                let l = self.val(ctx, *left)?;
                let r = self.val(ctx, *right)?;
                Val::V(self.binary(*op, &l, &r)?)
            }
            Expression::Select { condition, accept, reject } => {
                let c = self.val(ctx, *condition)?;
                let a = self.val(ctx, *accept)?;
                let r = self.val(ctx, *reject)?;
                Val::V(self.select(&c, &a, &r)?)
            }
            Expression::Relational { fun, argument } => {
                let v = self.val(ctx, *argument)?;
                Val::V(self.relational(*fun, &v)?)
            }
            Expression::Math { fun, arg, arg1, arg2, arg3 } => {
                let mut args = vec![self.val(ctx, *arg)?];
                for a in [arg1, arg2, arg3].into_iter().flatten() {
                    args.push(self.val(ctx, *a)?);
                }
                Val::V(self.math(*fun, args)?)
            }
            Expression::As { expr, kind, convert } => {
                let v = self.val(ctx, *expr)?;
                Val::V(self.cast(&v, *kind, *convert)?)
            }
            Expression::ImageSample {
                image,
                sampler,
                gather: None,
                coordinate,
                array_index: None,
                offset: None,
                level,
                depth_ref: None,
                clamp_to_edge: _,
            } => {
                // one mip level on this GPU: every sample level is level 0
                let _ = level;
                let Val::Tex(tex, elem) = self.get(ctx, *image)? else { bail!("sample of a non-texture") };
                if elem != Elem::F {
                    bail!("sampling an integer texture")
                }
                let Val::Smp(smp) = self.get(ctx, *sampler)? else { bail!("sample with a non-sampler") };
                let Lv::A(c) = self.val(ctx, *coordinate)? else { bail!("sample coordinate") };
                let (Lv::S(x, _), Lv::S(y, _)) = (&c[0], &c[1]) else { bail!("sample coordinate") };
                Val::V(self.tex_op(TexKind::Sample, tex, smp, *x, *y, Elem::F, false))
            }
            Expression::ImageLoad { image, coordinate, array_index: None, sample: None, level, .. } => {
                let Val::Tex(tex, elem) = self.get(ctx, *image)? else { bail!("load of a non-texture") };
                if let Some(l) = level {
                    // only level 0 exists
                    let Lv::S(r, _) = self.val(ctx, *l)? else { bail!("load level") };
                    if self.kconst.get(&r) != Some(&0) {
                        bail!("textureLoad at a non-zero / dynamic mip level")
                    }
                }
                let Lv::A(c) = self.val(ctx, *coordinate)? else { bail!("load coordinate") };
                let (Lv::S(x, kx), Lv::S(y, _)) = (&c[0], &c[1]) else { bail!("load coordinate") };
                Val::V(self.tex_op(TexKind::Load, tex, u32::MAX, *x, *y, elem, *kx == K::I))
            }
            Expression::ImageQuery { image, query } => {
                let Val::Tex(tex, _) = self.get(ctx, *image)? else { bail!("query of a non-texture") };
                match query {
                    naga::ImageQuery::Size { .. } => {
                        let d = self.nregs;
                        self.nregs += 2;
                        self.tex_ops.push(TexOp {
                            kind: TexKind::Size,
                            tex,
                            smp: u32::MAX,
                            x: 0,
                            y: 0,
                            d,
                            elem: Elem::U,
                            signed: false,
                        });
                        self.push(Inst::Tex { op: (self.tex_ops.len() - 1) as u32 });
                        Val::V(Lv::A(vec![Lv::S(d, K::U), Lv::S(d + 1, K::U)]))
                    }
                    naga::ImageQuery::NumLevels | naga::ImageQuery::NumLayers | naga::ImageQuery::NumSamples => {
                        Val::V(self.konst(1, K::U))
                    }
                }
            }
            Expression::CallResult(_) => {
                // filled by the Call statement; evaluated by Emit only if naga
                // lists it there (it does not), so a hit here is a bug
                return ctx.memo[h.index()].clone().ok_or_else(|| "CallResult before Call".into());
            }
            other => bail!("expression {:?}", std::mem::discriminant(other)),
        })
    }

    /// emit a texture op producing 4 fresh consecutive registers
    #[allow(clippy::too_many_arguments)]
    fn tex_op(&mut self, kind: TexKind, tex: u32, smp: u32, x: R, y: R, elem: Elem, signed: bool) -> Lv {
        let d = self.nregs;
        self.nregs += 4;
        self.tex_ops.push(TexOp { kind, tex, smp, x, y, d, elem, signed });
        self.push(Inst::Tex { op: (self.tex_ops.len() - 1) as u32 });
        let k = match elem {
            Elem::F => K::F,
            Elem::U => K::U,
            Elem::I => K::I,
        };
        Lv::A((0..4).map(|i| Lv::S(d + i, k)).collect())
    }

    fn literal(&mut self, l: Literal) -> Res<Lv> {
        Ok(match l {
            Literal::F32(x) => self.konst(x.to_bits(), K::F),
            Literal::F64(x) => self.konst((x as f32).to_bits(), K::F),
            Literal::AbstractFloat(x) => self.konst((x as f32).to_bits(), K::F),
            Literal::U32(x) => self.konst(x, K::U),
            Literal::I32(x) => self.konst(x as u32, K::I),
            Literal::AbstractInt(x) => self.konst(x as i32 as u32, K::I),
            Literal::U64(x) => self.konst(x as u32, K::U),
            Literal::I64(x) => self.konst(x as i32 as u32, K::I),
            Literal::Bool(b) => self.konst(b as u32, K::B),
            other => bail!("literal {other:?}"),
        })
    }

    /// module-level constant expressions (memoized; only ever reached from the
    /// function-entry pre-pass, so the registers are hoisted constants)
    fn gexpr(&mut self, h: Handle<Expression>) -> Res<Lv> {
        if let Some(v) = &self.gmemo[h.index()] {
            return Ok(v.clone());
        }
        let m = self.m;
        let code_before = self.code.len();
        let v = match &m.global_expressions[h] {
            Expression::Literal(l) => self.literal(*l)?,
            Expression::Constant(c) => self.gexpr(m.constants[*c].init)?,
            Expression::ZeroValue(ty) => self.zeros_like(*ty)?,
            Expression::Compose { ty, components } => {
                let mut comps = Vec::new();
                for c in components {
                    comps.push(self.gexpr(*c)?);
                }
                self.compose(*ty, comps)?
            }
            Expression::Splat { size, value } => {
                let v = self.gexpr(*value)?;
                Lv::A(vec![v; vsize(*size) as usize])
            }
            Expression::AccessIndex { base, index } => {
                let bv = self.gexpr(*base)?;
                match self.access_static(Val::V(bv), *index)? {
                    Val::V(l) => l,
                    _ => bail!("const access"),
                }
            }
            Expression::Unary { op, expr } => {
                let v = self.gexpr(*expr)?;
                self.unary(*op, &v)?
            }
            Expression::Binary { op, left, right } => {
                let l = self.gexpr(*left)?;
                let r = self.gexpr(*right)?;
                self.binary(*op, &l, &r)?
            }
            Expression::As { expr, kind, convert } => {
                let v = self.gexpr(*expr)?;
                self.cast(&v, *kind, *convert)?
            }
            other => bail!("const-expression {:?}", std::mem::discriminant(other)),
        };
        // memoize only if it emitted no instructions: a register assigned by
        // code inside one branch must not be reused on another path
        if self.code.len() == code_before {
            self.gmemo[h.index()] = Some(v.clone());
        }
        Ok(v)
    }

    fn compose(&mut self, ty: Handle<naga::Type>, comps: Vec<Lv>) -> Res<Lv> {
        match &self.m.types[ty].inner {
            TypeInner::Vector { .. } => {
                let mut flat = Vec::new();
                for c in &comps {
                    match c {
                        Lv::S(..) => flat.push(c.clone()),
                        Lv::A(v) => flat.extend(v.iter().cloned()),
                    }
                }
                Ok(Lv::A(flat))
            }
            TypeInner::Matrix { columns, rows, .. } => {
                let (nc, nr) = (vsize(*columns) as usize, vsize(*rows) as usize);
                if comps.len() == nc {
                    Ok(Lv::A(comps))
                } else if comps.len() == nc * nr {
                    Ok(Lv::A(comps.chunks(nr).map(|c| Lv::A(c.to_vec())).collect()))
                } else {
                    bail!("matrix compose with {} components", comps.len())
                }
            }
            _ => Ok(Lv::A(comps)),
        }
    }

    // -- access chains ----------------------------------------------------------

    fn access_static(&mut self, base: Val, index: u32) -> Res<Val> {
        match base {
            Val::Tex(..) | Val::Smp(..) => bail!("index into a texture/sampler handle"),
            Val::V(Lv::A(items)) => items
                .get(index as usize)
                .cloned()
                .map(Val::V)
                .ok_or_else(|| "static index out of range".to_string()),
            Val::V(Lv::S(..)) => bail!("index into scalar"),
            Val::P(Ptr::Local(Lv::A(items))) => items
                .get(index as usize)
                .cloned()
                .map(|l| Val::P(Ptr::Local(l)))
                .ok_or_else(|| "static index out of range".to_string()),
            Val::P(Ptr::Local(_)) => bail!("index into scalar pointer"),
            Val::P(Ptr::LocalDyn(cands)) => {
                let mut out = Vec::new();
                for (flag, lv) in cands {
                    match lv {
                        Lv::A(items) => out.push((
                            flag,
                            items.get(index as usize).cloned().ok_or("static index out of range")?,
                        )),
                        Lv::S(..) => bail!("index into scalar pointer"),
                    }
                }
                Ok(Val::P(Ptr::LocalDyn(out)))
            }
            Val::P(Ptr::Buf { buf, imm, dynr, shape }) => {
                let (off, shape) = self.buf_step(shape, index)?;
                Ok(Val::P(Ptr::Buf { buf, imm: imm.wrapping_add(off), dynr, shape }))
            }
        }
    }

    /// one static step into a buffer shape: byte offset + the new shape
    fn buf_step(&self, shape: Shape, i: u32) -> Res<(u32, Shape)> {
        let m = self.m;
        Ok(match shape {
            Shape::Ty(ty) => match &m.types[ty].inner {
                TypeInner::Struct { members, .. } => {
                    let offs = member_offsets(m, members);
                    (offs[i as usize], Shape::Ty(members[i as usize].ty))
                }
                TypeInner::Array { base, .. } => (i * array_stride(m, *base), Shape::Ty(*base)),
                TypeInner::Vector { scalar, .. } => (i * w(*scalar), Shape::Scalar(*scalar)),
                TypeInner::Matrix { rows, scalar, .. } => (
                    i * vec_stride(vsize(*rows), *scalar),
                    Shape::Vec(vsize(*rows), *scalar),
                ),
                other => bail!("buffer step into {other:?}"),
            },
            Shape::Vec(_, s) => (i * w(s), Shape::Scalar(s)),
            Shape::Scalar(_) => bail!("buffer step into scalar"),
        })
    }

    fn access_dyn(&mut self, base: Val, index: Lv) -> Res<Val> {
        let Lv::S(ir, _) = index else { bail!("vector index") };
        if let Some(&k) = self.kconst.get(&ir) {
            return self.access_static(base, k);
        }
        match base {
            Val::V(Lv::A(items)) => {
                // dynamic index into a value aggregate: select chain
                let mut acc = items[0].clone();
                for (k, it) in items.iter().enumerate().skip(1) {
                    let Lv::S(kc, _) = self.konst(k as u32, K::U) else { unreachable!() };
                    let t = self.nr();
                    self.push(Inst::Cmp { d: t, a: ir, b: kc, c: Cmp::IEq });
                    acc = self.select(&Lv::S(t, K::B), it, &acc)?;
                }
                Ok(Val::V(acc))
            }
            Val::P(Ptr::Buf { buf, imm, dynr, shape }) => {
                let (stride, shape) = match shape {
                    Shape::Ty(ty) => match &self.m.types[ty].inner {
                        TypeInner::Array { base, .. } => (array_stride(self.m, *base), Shape::Ty(*base)),
                        TypeInner::Vector { scalar, .. } => (w(*scalar), Shape::Scalar(*scalar)),
                        TypeInner::Matrix { rows, scalar, .. } => (
                            vec_stride(vsize(*rows), *scalar),
                            Shape::Vec(vsize(*rows), *scalar),
                        ),
                        other => bail!("dynamic buffer index into {other:?}"),
                    },
                    Shape::Vec(_, s) => (w(s), Shape::Scalar(s)),
                    Shape::Scalar(_) => bail!("dynamic index into scalar"),
                };
                let sk = self.konst(stride, K::U);
                let Lv::S(sr, _) = sk else { unreachable!() };
                let scaled = self.nr();
                self.push(Inst::IMul { d: scaled, a: ir, b: sr });
                let total = match dynr {
                    None => scaled,
                    Some(prev) => {
                        let t = self.nr();
                        self.push(Inst::IAdd { d: t, a: prev, b: scaled });
                        t
                    }
                };
                Ok(Val::P(Ptr::Buf { buf, imm, dynr: Some(total), shape }))
            }
            Val::P(Ptr::Local(Lv::A(items))) => {
                let mut cands = Vec::new();
                for (k, it) in items.iter().enumerate() {
                    let Lv::S(kc, _) = self.konst(k as u32, K::U) else { unreachable!() };
                    let t = self.nr();
                    self.push(Inst::Cmp { d: t, a: ir, b: kc, c: Cmp::IEq });
                    cands.push((t, it.clone()));
                }
                Ok(Val::P(Ptr::LocalDyn(cands)))
            }
            Val::P(Ptr::LocalDyn(prev)) => {
                // a second dynamic index on an already-dynamic pointer:
                // candidates are the product, flags are ANDed
                let mut cands = Vec::new();
                for (flag, lv) in prev {
                    let Lv::A(items) = lv else { bail!("dynamic index into scalar pointer") };
                    for (k, it) in items.iter().enumerate() {
                        let Lv::S(kc, _) = self.konst(k as u32, K::U) else { unreachable!() };
                        let t = self.nr();
                        self.push(Inst::Cmp { d: t, a: ir, b: kc, c: Cmp::IEq });
                        let f = self.nr();
                        self.push(Inst::And { d: f, a: flag, b: t });
                        cands.push((f, it.clone()));
                    }
                }
                Ok(Val::P(Ptr::LocalDyn(cands)))
            }
            _ => bail!("dynamic index on this base"),
        }
    }

    fn load_buf(&mut self, buf: u32, imm: u32, dynr: Option<R>, shape: Shape) -> Res<Lv> {
        let off = dynr.unwrap_or(0);
        let m = self.m;
        match shape {
            Shape::Scalar(s) => {
                let k = scalar_k(s)?;
                let d = self.nr();
                self.push(Inst::LoadBuf { d, buf, off, imm });
                Ok(Lv::S(d, k))
            }
            Shape::Vec(n, s) => {
                let mut v = Vec::new();
                for i in 0..n {
                    v.push(self.load_buf(buf, imm + i * w(s), dynr, Shape::Scalar(s))?);
                }
                Ok(Lv::A(v))
            }
            Shape::Ty(ty) => match &m.types[ty].inner {
                TypeInner::Scalar(s) => self.load_buf(buf, imm, dynr, Shape::Scalar(*s)),
                TypeInner::Vector { size, scalar } => {
                    self.load_buf(buf, imm, dynr, Shape::Vec(vsize(*size), *scalar))
                }
                TypeInner::Matrix { columns, rows, scalar } => {
                    let stride = vec_stride(vsize(*rows), *scalar);
                    let mut cols = Vec::new();
                    for c in 0..vsize(*columns) {
                        cols.push(self.load_buf(
                            buf,
                            imm + c * stride,
                            dynr,
                            Shape::Vec(vsize(*rows), *scalar),
                        )?);
                    }
                    Ok(Lv::A(cols))
                }
                TypeInner::Array { base, size, .. } => {
                    let n = match size {
                        naga::ArraySize::Constant(c) => c.get(),
                        _ => bail!("load of an unsized array"),
                    };
                    let stride = array_stride(m, *base);
                    let mut v = Vec::new();
                    for i in 0..n {
                        v.push(self.load_buf(buf, imm + i * stride, dynr, Shape::Ty(*base))?);
                    }
                    Ok(Lv::A(v))
                }
                TypeInner::Struct { members, .. } => {
                    let offs = member_offsets(m, members);
                    let mut v = Vec::new();
                    for (mm, o) in members.iter().zip(offs) {
                        v.push(self.load_buf(buf, imm + o, dynr, Shape::Ty(mm.ty))?);
                    }
                    Ok(Lv::A(v))
                }
                other => bail!("buffer load of {other:?}"),
            },
        }
    }

    // -- scalar op emission -------------------------------------------------------

    fn map1(&mut self, lv: &Lv, f: &mut dyn FnMut(&mut Self, (R, K)) -> Res<(R, K)>) -> Res<Lv> {
        match lv {
            Lv::S(r, k) => {
                let (d, k2) = f(self, (*r, *k))?;
                Ok(Lv::S(d, k2))
            }
            Lv::A(v) => {
                let mut out = Vec::new();
                for c in v {
                    out.push(self.map1(c, f)?);
                }
                Ok(Lv::A(out))
            }
        }
    }

    fn map2(
        &mut self,
        a: &Lv,
        b: &Lv,
        f: &mut dyn FnMut(&mut Self, (R, K), (R, K)) -> Res<(R, K)>,
    ) -> Res<Lv> {
        match (a, b) {
            (Lv::S(x, kx), Lv::S(y, ky)) => {
                let (d, k) = f(self, (*x, *kx), (*y, *ky))?;
                Ok(Lv::S(d, k))
            }
            (Lv::A(xs), Lv::A(ys)) if xs.len() == ys.len() => {
                let mut out = Vec::new();
                for (x, y) in xs.iter().zip(ys) {
                    out.push(self.map2(x, y, f)?);
                }
                Ok(Lv::A(out))
            }
            // scalar broadcast (vec * scalar, mat * scalar, ...)
            (Lv::A(xs), s @ Lv::S(..)) => {
                let mut out = Vec::new();
                for x in xs {
                    out.push(self.map2(x, s, f)?);
                }
                Ok(Lv::A(out))
            }
            (s @ Lv::S(..), Lv::A(ys)) => {
                let mut out = Vec::new();
                for y in ys {
                    out.push(self.map2(s, y, f)?);
                }
                Ok(Lv::A(out))
            }
            _ => bail!("operand shape mismatch"),
        }
    }

    fn unary(&mut self, op: UnaryOperator, v: &Lv) -> Res<Lv> {
        self.map1(v, &mut |s, (a, k)| {
            let d = s.nr();
            match (op, k) {
                (UnaryOperator::Negate, K::F) => s.push(Inst::FNeg { d, a }),
                (UnaryOperator::Negate, K::I | K::U) => s.push(Inst::ISub { d, a: 0, b: a }),
                (UnaryOperator::LogicalNot, K::B) => {
                    let one = s.konst(1, K::B);
                    let Lv::S(o, _) = one else { unreachable!() };
                    s.push(Inst::Xor { d, a, b: o })
                }
                (UnaryOperator::BitwiseNot, K::I | K::U) => s.push(Inst::Not { d, a }),
                (o, k) => bail!("unary {o:?} on {k:?}"),
            }
            Ok((d, k))
        })
    }

    fn bin_scalar(&mut self, op: B, (a, ka): (R, K), (b, kb): (R, K)) -> Res<(R, K)> {
        use B::*;
        let d = self.nr();
        let cmpk = |f: Cmp, s: &mut Self| {
            s.push(Inst::Cmp { d, a, b, c: f });
            Ok((d, K::B))
        };
        // result kind follows the left operand except for comparisons
        match (op, ka) {
            (Add, K::F) => self.push(Inst::FAdd { d, a, b }),
            (Subtract, K::F) => self.push(Inst::FSub { d, a, b }),
            (Multiply, K::F) => self.push(Inst::FMul { d, a, b }),
            (Divide, K::F) => self.push(Inst::FDiv { d, a, b }),
            (Modulo, K::F) => self.push(Inst::Call { d, a, b, f: Fun::FRem }),
            (Add, K::I | K::U) => self.push(Inst::IAdd { d, a, b }),
            (Subtract, K::I | K::U) => self.push(Inst::ISub { d, a, b }),
            (Multiply, K::I | K::U) => self.push(Inst::IMul { d, a, b }),
            (Divide, K::I) => self.push(Inst::Call { d, a, b, f: Fun::IDivS }),
            (Divide, K::U) => self.push(Inst::Call { d, a, b, f: Fun::IDivU }),
            (Modulo, K::I) => self.push(Inst::Call { d, a, b, f: Fun::IRemS }),
            (Modulo, K::U) => self.push(Inst::Call { d, a, b, f: Fun::IRemU }),
            (And | LogicalAnd, K::I | K::U | K::B) => self.push(Inst::And { d, a, b }),
            (InclusiveOr | LogicalOr, K::I | K::U | K::B) => self.push(Inst::Or { d, a, b }),
            (ExclusiveOr, K::I | K::U | K::B) => self.push(Inst::Xor { d, a, b }),
            (ShiftLeft, K::I | K::U) => self.push(Inst::Shl { d, a, b }),
            (ShiftRight, K::I) => self.push(Inst::ShrS { d, a, b }),
            (ShiftRight, K::U) => self.push(Inst::ShrU { d, a, b }),
            (Equal, K::F) => return cmpk(Cmp::FEq, self),
            (NotEqual, K::F) => return cmpk(Cmp::FNe, self),
            (Less, K::F) => return cmpk(Cmp::FLt, self),
            (LessEqual, K::F) => return cmpk(Cmp::FLe, self),
            (Greater, K::F) => return cmpk(Cmp::FGt, self),
            (GreaterEqual, K::F) => return cmpk(Cmp::FGe, self),
            (Equal, _) => return cmpk(Cmp::IEq, self),
            (NotEqual, _) => return cmpk(Cmp::INe, self),
            (Less, K::I) => return cmpk(Cmp::SLt, self),
            (LessEqual, K::I) => return cmpk(Cmp::SLe, self),
            (Greater, K::I) => return cmpk(Cmp::SGt, self),
            (GreaterEqual, K::I) => return cmpk(Cmp::SGe, self),
            (Less, K::U) => return cmpk(Cmp::ULt, self),
            (LessEqual, K::U) => return cmpk(Cmp::ULe, self),
            (Greater, K::U) => return cmpk(Cmp::UGt, self),
            (GreaterEqual, K::U) => return cmpk(Cmp::UGe, self),
            (o, k) => bail!("binary {o:?} on {k:?}/{kb:?}"),
        }
        Ok((d, ka))
    }

    fn binary(&mut self, op: B, l: &Lv, r: &Lv) -> Res<Lv> {
        if op == B::Multiply && (is_mat(l) || is_mat(r)) {
            return self.mat_mul(l, r);
        }
        self.map2(l, r, &mut |s, a, b| s.bin_scalar(op, a, b))
    }

    fn fmul(&mut self, a: &Lv, b: &Lv) -> Res<Lv> {
        self.map2(a, b, &mut |s, x, y| s.bin_scalar(B::Multiply, x, y))
    }
    fn fadd(&mut self, a: &Lv, b: &Lv) -> Res<Lv> {
        self.map2(a, b, &mut |s, x, y| s.bin_scalar(B::Add, x, y))
    }

    /// matrix products; `m[col][row]`
    fn mat_mul(&mut self, l: &Lv, r: &Lv) -> Res<Lv> {
        match (l, r) {
            // mat * scalar / scalar * mat
            (_, Lv::S(..)) | (Lv::S(..), _) => {
                self.map2(l, r, &mut |s, x, y| s.bin_scalar(B::Multiply, x, y))
            }
            // mat * vec
            (Lv::A(cols), Lv::A(v)) if is_mat(l) && !is_mat(r) => {
                let rows = match &cols[0] {
                    Lv::A(c) => c.len(),
                    _ => bail!("matrix shape"),
                };
                if cols.len() != v.len() {
                    bail!("mat*vec dimension mismatch")
                }
                let mut out = Vec::new();
                for i in 0..rows {
                    let mut acc: Option<Lv> = None;
                    for (j, col) in cols.iter().enumerate() {
                        let Lv::A(c) = col else { bail!("matrix shape") };
                        let p = self.fmul(&c[i], &v[j])?;
                        acc = Some(match acc {
                            None => p,
                            Some(a) => self.fadd(&a, &p)?,
                        });
                    }
                    out.push(acc.unwrap());
                }
                Ok(Lv::A(out))
            }
            // vec * mat
            (Lv::A(v), Lv::A(cols)) if !is_mat(l) && is_mat(r) => {
                let mut out = Vec::new();
                for col in cols {
                    let Lv::A(c) = col else { bail!("matrix shape") };
                    if c.len() != v.len() {
                        bail!("vec*mat dimension mismatch")
                    }
                    let mut acc: Option<Lv> = None;
                    for (i, x) in v.iter().enumerate() {
                        let p = self.fmul(x, &c[i])?;
                        acc = Some(match acc {
                            None => p,
                            Some(a) => self.fadd(&a, &p)?,
                        });
                    }
                    out.push(acc.unwrap());
                }
                Ok(Lv::A(out))
            }
            // mat * mat: each result column is l * (column of r)
            (Lv::A(_), Lv::A(rcols)) => {
                let mut out = Vec::new();
                for c in rcols {
                    out.push(self.mat_mul(l, c)?);
                }
                Ok(Lv::A(out))
            }
        }
    }

    fn select(&mut self, c: &Lv, a: &Lv, b: &Lv) -> Res<Lv> {
        match (c, a, b) {
            (Lv::S(cr, _), _, _) => {
                let cr = *cr;
                self.map2(a, b, &mut |s, (x, kx), (y, _)| {
                    let d = s.nr();
                    s.push(Inst::Select { d, c: cr, a: x, b: y });
                    Ok((d, kx))
                })
            }
            (Lv::A(cs), Lv::A(xs), Lv::A(ys)) if cs.len() == xs.len() && xs.len() == ys.len() => {
                let mut out = Vec::new();
                for ((cc, x), y) in cs.iter().zip(xs).zip(ys) {
                    out.push(self.select(cc, x, y)?);
                }
                Ok(Lv::A(out))
            }
            _ => bail!("select shape"),
        }
    }

    fn relational(&mut self, fun: naga::RelationalFunction, v: &Lv) -> Res<Lv> {
        use naga::RelationalFunction as R_;
        let mut ls = Vec::new();
        leaves(v, &mut ls);
        let mut flags: Vec<R> = Vec::new();
        for (r, _) in &ls {
            let d = self.nr();
            match fun {
                R_::All | R_::Any => flags.push(*r),
                R_::IsNan => {
                    self.push(Inst::Cmp { d, a: *r, b: *r, c: Cmp::FNe });
                    flags.push(d);
                }
                R_::IsInf => {
                    self.push(Inst::Call { d, a: *r, b: 0, f: Fun::IsInf });
                    flags.push(d);
                }
            }
        }
        let all = matches!(fun, R_::All);
        let mut acc = flags[0];
        for &f in &flags[1..] {
            let d = self.nr();
            if all {
                self.push(Inst::And { d, a: acc, b: f });
            } else {
                self.push(Inst::Or { d, a: acc, b: f });
            }
            acc = d;
        }
        Ok(Lv::S(acc, K::B))
    }

    fn cast_scalar(
        &mut self,
        (a, k): (R, K),
        to: ScalarKind,
        convert: Option<naga::Bytes>,
    ) -> Res<(R, K)> {
        let tk = kind_k(to)?;
        if convert.is_none() {
            // bitcast: same bits, new interpretation
            return Ok((a, tk));
        }
        let d = self.nr();
        match (k, tk) {
            (K::F, K::F) | (K::I, K::I) | (K::U, K::U) | (K::B, K::B) => return Ok((a, tk)),
            (K::F, K::U) => self.push(Inst::Call { d, a, b: 0, f: Fun::F2U }),
            (K::F, K::I) => self.push(Inst::Call { d, a, b: 0, f: Fun::F2I }),
            (K::F, K::B) => self.push(Inst::Cmp { d, a, b: 0, c: Cmp::FNe }),
            (K::I, K::F) | (K::B, K::F) => self.push(Inst::I2F { d, a }),
            (K::U, K::F) => self.push(Inst::U2F { d, a }),
            (K::I, K::B) | (K::U, K::B) => self.push(Inst::Cmp { d, a, b: 0, c: Cmp::INe }),
            (K::I, K::U) | (K::U, K::I) | (K::B, K::U) | (K::B, K::I) => return Ok((a, tk)),
        }
        Ok((d, tk))
    }

    fn cast(&mut self, v: &Lv, to: ScalarKind, convert: Option<naga::Bytes>) -> Res<Lv> {
        self.map1(v, &mut |s, x| s.cast_scalar(x, to, convert))
    }

    // -- math -------------------------------------------------------------------

    fn call(&mut self, f: Fun, a: R, b: R, k: K) -> (R, K) {
        let d = self.nr();
        if f.cacheable() {
            // four consecutive persistent registers: valid, key a, key b, result
            let c = self.nregs;
            self.nregs += 4;
            self.push(Inst::CallC { d, a, b, f, c });
        } else {
            self.push(Inst::Call { d, a, b, f });
        }
        (d, k)
    }

    fn math(&mut self, fun: M, args: Vec<Lv>) -> Res<Lv> {
        // aggregate builtins: dot / length / distance / normalize / cross
        match fun {
            M::Dot | M::Length | M::Distance | M::Normalize | M::Cross => {
                let mut xs = Vec::new();
                leaves(&args[0], &mut xs);
                let ys = args.get(1).map(|a| {
                    let mut v = Vec::new();
                    leaves(a, &mut v);
                    v
                });
                return self.aggregate(fun, xs, ys);
            }
            M::Fma | M::Transpose | M::Inverse | M::Determinant | M::Outer => {
                bail!("math {fun:?}")
            }
            _ => {}
        }
        // elementwise, extra scalar args broadcast over a vector first arg
        match &args[0] {
            Lv::A(items) => {
                let n = items.len();
                let mut out = Vec::new();
                for i in 0..n {
                    let mut sa = vec![items[i].clone()];
                    for extra in &args[1..] {
                        sa.push(match extra {
                            Lv::A(e) => e[i].clone(),
                            s => s.clone(),
                        });
                    }
                    out.push(self.math(fun, sa)?);
                }
                Ok(Lv::A(out))
            }
            Lv::S(..) => {
                let mut sc = Vec::new();
                for a in &args {
                    match a {
                        Lv::S(r, k) => sc.push((*r, *k)),
                        _ => bail!("math arg shape"),
                    }
                }
                let (d, k) = self.math_scalar(fun, &sc)?;
                Ok(Lv::S(d, k))
            }
        }
    }

    fn aggregate(&mut self, fun: M, xs: Vec<(R, K)>, ys: Option<Vec<(R, K)>>) -> Res<Lv> {
        let zero = self.fconst(0.0);
        let fm = |s: &mut Self, a: R, b: R| {
            let d = s.nr();
            s.push(Inst::FMul { d, a, b });
            d
        };
        let fa = |s: &mut Self, a: R, b: R| {
            let d = s.nr();
            s.push(Inst::FAdd { d, a, b });
            d
        };
        let fs = |s: &mut Self, a: R, b: R| {
            let d = s.nr();
            s.push(Inst::FSub { d, a, b });
            d
        };
        match fun {
            M::Dot => {
                let ys = ys.ok_or("dot arity")?;
                let mut acc = zero.0;
                for (p, q) in xs.iter().zip(&ys) {
                    let m = fm(self, p.0, q.0);
                    acc = fa(self, acc, m);
                }
                Ok(Lv::S(acc, K::F))
            }
            M::Length | M::Normalize => {
                let mut acc = zero.0;
                for p in &xs {
                    let m = fm(self, p.0, p.0);
                    acc = fa(self, acc, m);
                }
                let len = self.nr();
                self.push(Inst::Sqrt { d: len, a: acc });
                if fun == M::Length {
                    return Ok(Lv::S(len, K::F));
                }
                let mut out = Vec::new();
                for p in &xs {
                    let d = self.nr();
                    self.push(Inst::FDiv { d, a: p.0, b: len });
                    out.push(Lv::S(d, K::F));
                }
                Ok(Lv::A(out))
            }
            M::Distance => {
                let ys = ys.ok_or("distance arity")?;
                let mut acc = zero.0;
                for (p, q) in xs.iter().zip(&ys) {
                    let d = fs(self, p.0, q.0);
                    let m = fm(self, d, d);
                    acc = fa(self, acc, m);
                }
                let d = self.nr();
                self.push(Inst::Sqrt { d, a: acc });
                Ok(Lv::S(d, K::F))
            }
            M::Cross => {
                let ys = ys.ok_or("cross arity")?;
                if xs.len() != 3 || ys.len() != 3 {
                    bail!("cross needs vec3")
                }
                let comp = |s: &mut Self, i: usize, j: usize| {
                    let a = fm(s, xs[i].0, ys[j].0);
                    let b = fm(s, xs[j].0, ys[i].0);
                    fs(s, a, b)
                };
                let (x, y, z) = (comp(self, 1, 2), comp(self, 2, 0), comp(self, 0, 1));
                Ok(Lv::A(vec![Lv::S(x, K::F), Lv::S(y, K::F), Lv::S(z, K::F)]))
            }
            _ => unreachable!(),
        }
    }

    fn math_scalar(&mut self, fun: M, a: &[(R, K)]) -> Res<(R, K)> {
        let x = a[0];
        let arg = |i: usize| a.get(i).copied().ok_or_else(|| format!("math {fun:?}: missing arg {i}"));
        let fbin = |s: &mut Self, op: B, p: (R, K), q: (R, K)| s.bin_scalar(op, p, q);
        Ok(match fun {
            M::Abs => match x.1 {
                K::F => {
                    let d = self.nr();
                    self.push(Inst::FAbs { d, a: x.0 });
                    (d, K::F)
                }
                K::I => self.call(Fun::IAbs, x.0, 0, K::I),
                _ => x,
            },
            M::Min | M::Max => {
                let y = arg(1)?;
                match x.1 {
                    K::F => self.call(if fun == M::Min { Fun::FMin } else { Fun::FMax }, x.0, y.0, K::F),
                    k => {
                        let c = match (fun == M::Min, k) {
                            (true, K::I) => Cmp::SLt,
                            (true, _) => Cmp::ULt,
                            (false, K::I) => Cmp::SGt,
                            (false, _) => Cmp::UGt,
                        };
                        let t = self.nr();
                        self.push(Inst::Cmp { d: t, a: x.0, b: y.0, c });
                        let d = self.nr();
                        self.push(Inst::Select { d, c: t, a: x.0, b: y.0 });
                        (d, k)
                    }
                }
            }
            M::Clamp => {
                let (lo, hi) = (arg(1)?, arg(2)?);
                match x.1 {
                    K::F => {
                        // interp: x.min(hi).max(lo)
                        let m = self.call(Fun::FMin, x.0, hi.0, K::F);
                        self.call(Fun::FMax, m.0, lo.0, K::F)
                    }
                    k => {
                        let (lt, gt) = if k == K::I { (Cmp::SLt, Cmp::SGt) } else { (Cmp::ULt, Cmp::UGt) };
                        // max(x, lo)
                        let c1 = self.nr();
                        self.push(Inst::Cmp { d: c1, a: x.0, b: lo.0, c: gt });
                        let mx = self.nr();
                        self.push(Inst::Select { d: mx, c: c1, a: x.0, b: lo.0 });
                        // min(.., hi)
                        let c2 = self.nr();
                        self.push(Inst::Cmp { d: c2, a: mx, b: hi.0, c: lt });
                        let d = self.nr();
                        self.push(Inst::Select { d, c: c2, a: mx, b: hi.0 });
                        (d, k)
                    }
                }
            }
            M::Saturate => {
                let one = self.fconst(1.0);
                let zero = self.fconst(0.0);
                let m = self.call(Fun::FMin, x.0, one.0, K::F);
                self.call(Fun::FMax, m.0, zero.0, K::F)
            }
            M::Sin => self.call(Fun::Sin, x.0, 0, K::F),
            M::Cos => self.call(Fun::Cos, x.0, 0, K::F),
            M::Tan => self.call(Fun::Tan, x.0, 0, K::F),
            M::Sinh => self.call(Fun::Sinh, x.0, 0, K::F),
            M::Cosh => self.call(Fun::Cosh, x.0, 0, K::F),
            M::Tanh => self.call(Fun::Tanh, x.0, 0, K::F),
            M::Asin => self.call(Fun::Asin, x.0, 0, K::F),
            M::Acos => self.call(Fun::Acos, x.0, 0, K::F),
            M::Atan => self.call(Fun::Atan, x.0, 0, K::F),
            M::Asinh => self.call(Fun::Asinh, x.0, 0, K::F),
            M::Acosh => self.call(Fun::Acosh, x.0, 0, K::F),
            M::Atanh => self.call(Fun::Atanh, x.0, 0, K::F),
            M::Atan2 => self.call(Fun::Atan2, x.0, arg(1)?.0, K::F),
            M::Pow => self.call(Fun::Pow, x.0, arg(1)?.0, K::F),
            M::Exp => self.call(Fun::Exp, x.0, 0, K::F),
            M::Exp2 => self.call(Fun::Exp2, x.0, 0, K::F),
            M::Log => self.call(Fun::Ln, x.0, 0, K::F),
            M::Log2 => self.call(Fun::Log2, x.0, 0, K::F),
            M::Ceil => self.call(Fun::Ceil, x.0, 0, K::F),
            M::Floor => self.call(Fun::Floor, x.0, 0, K::F),
            M::Round => self.call(Fun::Round, x.0, 0, K::F),
            M::Trunc => self.call(Fun::Trunc, x.0, 0, K::F),
            M::Fract => {
                let fl = self.call(Fun::Floor, x.0, 0, K::F);
                fbin(self, B::Subtract, x, fl)?
            }
            M::Sign => match x.1 {
                K::F => self.call(Fun::Sign, x.0, 0, K::F),
                _ => self.call(Fun::ISign, x.0, 0, K::I),
            },
            M::Radians => {
                let c = self.fconst(std::f32::consts::PI / 180.0);
                fbin(self, B::Multiply, x, c)?
            }
            M::Degrees => {
                let c = self.fconst(180.0 / std::f32::consts::PI);
                fbin(self, B::Multiply, x, c)?
            }
            M::Sqrt => {
                let d = self.nr();
                self.push(Inst::Sqrt { d, a: x.0 });
                (d, K::F)
            }
            M::InverseSqrt => {
                let one = self.fconst(1.0);
                let sq = self.nr();
                self.push(Inst::Sqrt { d: sq, a: x.0 });
                fbin(self, B::Divide, one, (sq, K::F))?
            }
            M::Mix => {
                // x * (1 - t) + y * t
                let (y, t) = (arg(1)?, arg(2)?);
                let one = self.fconst(1.0);
                let omt = fbin(self, B::Subtract, one, t)?;
                let l = fbin(self, B::Multiply, x, omt)?;
                let r = fbin(self, B::Multiply, y, t)?;
                fbin(self, B::Add, l, r)?
            }
            M::Step => {
                // edge = x, value = arg1: value < edge ? 0 : 1
                let v = arg(1)?;
                let t = self.nr();
                self.push(Inst::Cmp { d: t, a: v.0, b: x.0, c: Cmp::FLt });
                let zero = self.fconst(0.0);
                let one = self.fconst(1.0);
                let d = self.nr();
                self.push(Inst::Select { d, c: t, a: zero.0, b: one.0 });
                (d, K::F)
            }
            M::SmoothStep => {
                let (hi, v) = (arg(1)?, arg(2)?);
                let num = fbin(self, B::Subtract, v, x)?;
                let den = fbin(self, B::Subtract, hi, x)?;
                let q = fbin(self, B::Divide, num, den)?;
                let t = self.call(Fun::Clamp01, q.0, 0, K::F);
                let two = self.fconst(2.0);
                let three = self.fconst(3.0);
                let t2 = fbin(self, B::Multiply, two, t)?;
                let inner = fbin(self, B::Subtract, three, t2)?;
                let tt = fbin(self, B::Multiply, t, t)?;
                fbin(self, B::Multiply, tt, inner)?
            }
            other => bail!("math {other:?}"),
        })
    }
}
