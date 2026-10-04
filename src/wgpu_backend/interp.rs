//! The naga IR interpreter — the "shader core" of the akuma custom wgpu
//! backend (plan §5b option A).
//!
//! wgpu hands a custom backend the raw WGSL source (`ShaderSource::Wgsl`);
//! this module parses it with naga's `wgsl-in` front-end, validates it, and
//! *walks the IR* per vertex/fragment invocation. Nothing is ever compiled to
//! machine code: the kernel refuses W|X memory (plan §5b), and does not need
//! to — the demo's shaders are a few dozen ops per invocation, and a
//! tree-walking interpreter buys portability and determinism for speed we
//! do not have to pay for on this milestone.
//!
//! Determinism is the whole point of M3: the acceptance test is that the
//! wgpu path produces frames bit-identical to `softrender`. Two rules make
//! that hold:
//!
//! 1. Every arithmetic op maps 1:1 to the Rust scalar op softrender uses
//!    (`a + b`, `f32::sin`, `as`-casts, ...), evaluated in source order,
//!    with no fusing or reassociation anywhere. Rust never contracts float
//!    ops, and naga's front-end does not rewrite them either — the IR tree
//!    is the source tree.
//! 2. Anything naga could legitimately fold at parse time must not carry an
//!    exactness obligation: the shaders in `shaders.rs` keep every value
//!    that must match softrender on a runtime path (uniforms, storage
//!    loads); only literal-only constants are folded, and those are
//!    correctly-rounded in both toolchains.
//!
//! Scope: scalars, vectors, matrices, arrays, structs; uniform + storage
//! buffers; user functions; if/loop/switch; the math builtins the demo and
//! sugarloaf-style shaders use. Sampling, atomics, ray queries and subgroup
//! ops are explicitly unimplemented (they panic with a clear message).

use std::collections::HashMap;

use naga::{
    AddressSpace, BinaryOperator, BuiltIn, Expression, Function, Handle, Literal, MathFunction,
    ScalarKind, SwizzleComponent, TypeInner, UnaryOperator, VectorSize,
};

// ---------------------------------------------------------------------------
// Parsed shader
// ---------------------------------------------------------------------------

/// A WGSL module, parsed and validated once at `create_shader_module` time.
/// The validator run is the correctness gate (a shader that fails
/// `naga::valid` never reaches the interpreter); the interpreter itself is
/// dynamically typed on values, so the per-expression `ModuleInfo` is
/// intentionally not kept.
#[derive(Debug)]
pub struct Shader {
    pub module: naga::Module,
}

impl Shader {
    pub fn parse(wgsl: &str) -> Result<Shader, String> {
        let module =
            naga::front::wgsl::parse_str(wgsl).map_err(|e| format!("wgsl parse: {e}"))?;
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator
            .validate(&module)
            .map_err(|e| format!("wgsl validate: {e}"))?;
        drop(info);
        Ok(Shader { module })
    }

    /// Resolve an entry point by name to its index in `module.entry_points`
    /// (wgpu's `entry_point: None` convention: the module must have exactly
    /// one).
    pub fn entry(&self, name: Option<&str>) -> Result<usize, String> {
        match name {
            Some(want) => self
                .module
                .entry_points
                .iter()
                .position(|ep| ep.name == want)
                .ok_or_else(|| format!("no entry point `{want}`")),
            None => {
                if self.module.entry_points.len() == 1 {
                    Ok(0)
                } else {
                    Err(format!(
                        "no entry point name given and module has {} entry points",
                        self.module.entry_points.len()
                    ))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// One scalar lane, or a composite.
#[derive(Clone, Debug)]
pub enum Value {
    F32(f32),
    I32(i32),
    U32(u32),
    Bool(bool),
    /// vector, length 2..=4, components in declared order
    Vec(Vec<Value>),
    Arr(Vec<Value>),
    Struct(Vec<Value>),
    /// column-major: `cols[c][r]`
    Mat { cols: Vec<[f32; 4]>, rows: usize },
    /// a reference into a local slot or a bound buffer
    Ptr(Ptr),
}

#[derive(Clone, Debug)]
pub struct Ptr {
    pub base: PtrBase,
    /// member/index steps from the base
    pub steps: Vec<PtrStep>,
}

#[derive(Clone, Copy, Debug)]
pub enum PtrBase {
    /// a `LocalVariable` handle (indexes `Frame::slots`)
    Local(Handle<naga::LocalVariable>),
    /// (group, binding) of a buffer global
    Buffer { group: u32, binding: u32 },
    /// a `var<private>` global (indexes the invocation's `Privates`)
    Private(Handle<naga::GlobalVariable>),
}

/// The invocation's `var<private>` globals, by global index; shared by every
/// frame of one entry-point run (`None` for globals in other spaces).
type Privates = std::rc::Rc<std::cell::RefCell<Vec<Option<Value>>>>;

#[derive(Clone, Copy, Debug)]
pub enum PtrStep {
    Idx(u32),
    Member(u32),
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// What a shader invocation can see. Buffers only for now: the M3 shaders
/// use one uniform (frame/time) and one read-only storage buffer (the mesh).
#[derive(Default)]
pub struct Resources<'r> {
    /// bound buffer contents, borrowed for the duration of one draw (the
    /// backend holds the buffers' locks across it, so every stage executor —
    /// interpreter, VM, JIT — reads plain slices)
    pub bufs: HashMap<(u32, u32), &'r [u8]>,
    /// bound textures / samplers (used by compiled stages; the interpreter
    /// does not sample)
    pub texs: HashMap<(u32, u32), super::texture::TexRef>,
    pub smps: HashMap<(u32, u32), super::texture::SmpRef>,
}

impl<'r> Resources<'r> {
    pub fn with_buffer(mut self, group: u32, binding: u32, bytes: &'r [u8]) -> Self {
        self.bufs.insert((group, binding), bytes);
        self
    }
}

// ---------------------------------------------------------------------------
// Stage IO
// ---------------------------------------------------------------------------

/// One vertex-shader invocation's result.
#[derive(Clone, Debug)]
pub struct VertexOut {
    /// @builtin(position): x/y are pixel coords, z the depth our
    /// fixed-function rasterizer compares (the akuma-GPU contract — there
    /// is no NDC and no divide; see shaders.rs).
    pub position: [f32; 4],
    /// (location, value) in declared order; Flat by contract (see backend).
    pub varyings: Vec<(u32, Value)>,
}

// ---------------------------------------------------------------------------
// WGSL layout (storage rules; the demo's uniform is a plain vec4<f32>, so
// the uniform address-space restrictions never bite)
// ---------------------------------------------------------------------------

pub(super) fn vsize(n: VectorSize) -> u32 {
    match n {
        VectorSize::Bi => 2,
        VectorSize::Tri => 3,
        VectorSize::Quad => 4,
    }
}

pub(super) fn w(s: naga::Scalar) -> u32 {
    s.width as u32
}

pub fn round_up(align: u32, val: u32) -> u32 {
    debug_assert!(align.is_power_of_two());
    (val + align - 1) & !(align - 1)
}

/// Array element stride, computed from OUR layout math: element size rounded
/// up to element alignment — the WGSL rule. Idx stepping and member offsets
/// must come from one set of rules (naga's stored stride agrees with these,
/// but deriving the stride here is what keeps the two from drifting apart).
pub(super) fn array_stride(module: &naga::Module, base: Handle<naga::Type>) -> u32 {
    round_up(align_of(module, base), size_of(module, base))
}

pub(super) fn align_of(module: &naga::Module, ty: Handle<naga::Type>) -> u32 {
    match &module.types[ty].inner {
        TypeInner::Scalar(s) | TypeInner::Atomic(s) => w(*s),
        TypeInner::Vector { size, scalar } => match *size {
            VectorSize::Bi => 2 * w(*scalar),
            // WGSL: a vec3 has the alignment of a vec4
            VectorSize::Tri | VectorSize::Quad => 4 * w(*scalar),
        },
        TypeInner::Matrix { rows, scalar, .. } => match *rows {
            VectorSize::Bi => 2 * w(*scalar),
            _ => 4 * w(*scalar),
        },
        TypeInner::Array { base, .. } => align_of(module, *base),
        TypeInner::Struct { members, .. } => members
            .iter()
            .map(|m| align_of(module, m.ty))
            .max()
            .unwrap_or(1),
        other => panic!("interp: align_of unsupported {other:?}"),
    }
}

pub(super) fn size_of(module: &naga::Module, ty: Handle<naga::Type>) -> u32 {
    match &module.types[ty].inner {
        TypeInner::Scalar(s) | TypeInner::Atomic(s) => w(*s),
        TypeInner::Vector { size, scalar } => vsize(*size) * w(*scalar),
        TypeInner::Matrix { columns, rows, scalar } => {
            vec_stride(vsize(*rows), *scalar) * vsize(*columns)
        }
        TypeInner::Array { base, size, .. } => {
            let n = match size {
                naga::ArraySize::Constant(c) => c.get(),
                naga::ArraySize::Dynamic => panic!("interp: unsized array has no size"),
                naga::ArraySize::Pending(_) => panic!("interp: pending array size unsupported"),
            };
            array_stride(module, *base) * n
        }
        TypeInner::Struct { members, .. } => {
            let mut off = 0;
            for m in members {
                off = round_up(align_of(module, m.ty), off) + size_of(module, m.ty);
            }
            round_up(align_of(module, ty), off)
        }
        other => panic!("interp: size_of unsupported {other:?}"),
    }
}

/// stride for a vector standing in for a matrix column
pub(super) fn vec_stride(n: u32, s: naga::Scalar) -> u32 {
    round_up(w(s), n * w(s))
}

/// array stride = element size rounded up to element alignment

/// struct member offsets (storage layout)
pub(super) fn member_offsets(module: &naga::Module, members: &[naga::StructMember]) -> Vec<u32> {
    let mut offs = Vec::with_capacity(members.len());
    let mut off = 0;
    for m in members {
        off = round_up(align_of(module, m.ty), off);
        offs.push(off);
        off += size_of(module, m.ty);
    }
    offs
}

// ---------------------------------------------------------------------------
// Byte-level typed access (bound buffers)
// ---------------------------------------------------------------------------

/// The value shape a buffer pointer resolves to: either a type from the
/// module's arena, or a bare scalar that exists only as part of one.
#[derive(Clone, Copy, Debug)]
enum PtrShape {
    Ty(Handle<naga::Type>),
    Scalar(naga::Scalar),
}

fn read_shape(module: &naga::Module, bytes: &[u8], off: u32, shape: &PtrShape) -> Value {
    match shape {
        PtrShape::Ty(ty) => read_value(module, bytes, off, *ty),
        PtrShape::Scalar(s) => read_scalar(bytes, off, *s),
    }
}

#[allow(dead_code)]
fn write_shape(
    module: &naga::Module,
    bytes: &mut [u8],
    off: u32,
    shape: &PtrShape,
    v: &Value,
) {
    match shape {
        PtrShape::Ty(ty) => write_value(module, bytes, off, *ty, v),
        PtrShape::Scalar(_) => write_scalar(bytes, off, v),
    }
}

fn read_scalar(bytes: &[u8], off: u32, s: naga::Scalar) -> Value {
    let at = off as usize;
    match (s.kind, s.width) {
        (ScalarKind::Float, 4) => {
            Value::F32(f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()))
        }
        (ScalarKind::Sint, 4) => {
            Value::I32(i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()))
        }
        (ScalarKind::Uint, 4) => {
            Value::U32(u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()))
        }
        (ScalarKind::Bool, 1) => Value::Bool(bytes[at] != 0),
        (kind, width) => panic!("interp: read_scalar {kind:?} x{width} unsupported"),
    }
}

#[allow(dead_code)]
fn write_scalar(bytes: &mut [u8], off: u32, v: &Value) {
    let at = off as usize;
    match v {
        Value::F32(x) => bytes[at..at + 4].copy_from_slice(&x.to_le_bytes()),
        Value::I32(x) => bytes[at..at + 4].copy_from_slice(&x.to_le_bytes()),
        Value::U32(x) => bytes[at..at + 4].copy_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => bytes[at] = *x as u8,
        other => panic!("interp: write_scalar {other:?} unsupported"),
    }
}

fn read_value(module: &naga::Module, bytes: &[u8], off: u32, ty: Handle<naga::Type>) -> Value {
    match &module.types[ty].inner {
        TypeInner::Scalar(s) | TypeInner::Atomic(s) => read_scalar(bytes, off, *s),
        TypeInner::Vector { size, scalar } => {
            let n = vsize(*size);
            let sw = w(*scalar);
            Value::Vec(
                (0..n)
                    .map(|i| read_scalar(bytes, off + i * sw, *scalar))
                    .collect(),
            )
        }
        TypeInner::Matrix { columns, rows, scalar } => {
            let stride = vec_stride(vsize(*rows), *scalar);
            let cols = (0..vsize(*columns))
                .map(|c| {
                    let mut col = [0.0f32; 4];
                    for r in 0..vsize(*rows) {
                        if let Value::F32(f) = read_scalar(bytes, off + c * stride + r * 4, *scalar)
                        {
                            col[r as usize] = f;
                        }
                    }
                    col
                })
                .collect();
            Value::Mat {
                cols,
                rows: vsize(*rows) as usize,
            }
        }
        TypeInner::Array { base, size, .. } => {
            let n = match size {
                naga::ArraySize::Constant(c) => c.get(),
                // unsized arrays: the runtime length is not in the IR; the
                // demo's mesh array is fixed-size at declaration
                naga::ArraySize::Dynamic => panic!("interp: dynamic array length unsupported"),
                naga::ArraySize::Pending(_) => panic!("interp: pending array unsupported"),
            };
            let stride = array_stride(module, *base);
            Value::Arr(
                (0..n)
                    .map(|i| read_value(module, bytes, off + i * stride, *base))
                    .collect(),
            )
        }
        TypeInner::Struct { members, .. } => {
            let offs = member_offsets(module, members);
            Value::Struct(
                members
                    .iter()
                    .zip(offs)
                    .map(|(m, o)| read_value(module, bytes, off + o, m.ty))
                    .collect(),
            )
        }
        other => panic!("interp: read_value unsupported {other:?}"),
    }
}

#[allow(dead_code)]
fn write_value(
    module: &naga::Module,
    bytes: &mut [u8],
    off: u32,
    ty: Handle<naga::Type>,
    v: &Value,
) {
    match &module.types[ty].inner {
        TypeInner::Scalar(_) | TypeInner::Atomic(_) => write_scalar(bytes, off, v),
        TypeInner::Vector { size, scalar } => {
            let sw = w(*scalar);
            if let Value::Vec(vs) = v {
                for (i, c) in vs.iter().enumerate().take(vsize(*size) as usize) {
                    write_scalar(bytes, off + i as u32 * sw, c);
                }
            }
        }
        TypeInner::Array { base, .. } => {
            if let Value::Arr(vs) = v {
                let stride = array_stride(module, *base);
                for (i, c) in vs.iter().enumerate() {
                    write_value(module, bytes, off + i as u32 * stride, *base, c);
                }
            }
        }
        TypeInner::Struct { members, .. } => {
            let offs = member_offsets(module, members);
            if let Value::Struct(vs) = v {
                for ((m, o), c) in members.iter().zip(offs).zip(vs) {
                    write_value(module, bytes, off + o, m.ty, c);
                }
            }
        }
        other => panic!("interp: write_value unsupported {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Per-invocation evaluator
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum Flow {
    Next,
    Break,
    Continue,
    Return,
    Kill,
}

const MAX_CALL_DEPTH: u32 = 64;

struct Frame<'a> {
    sh: &'a Shader,
    fun: &'a Function,
    res: &'a Resources<'a>,
    args: Vec<Value>,
    slots: Vec<Option<Value>>,
    memo: Vec<Option<Value>>,
    return_slot: Option<Value>,
    depth: u32,
    privates: Privates,
}

impl<'a> Frame<'a> {
    fn new(
        sh: &'a Shader,
        fun: &'a Function,
        res: &'a Resources<'a>,
        args: Vec<Value>,
        depth: u32,
        privates: Privates,
    ) -> Frame<'a> {
        Frame {
            sh,
            fun,
            res,
            args,
            // WGSL zero-initializes every variable; naga's LocalVariables
            // with no `init` rely on exactly that
            slots: fun
                .local_variables
                .iter()
                .map(|(_, lv)| Some(zero_value_for(sh, lv.ty)))
                .collect(),
            memo: vec![None; fun.expressions.len()],
            return_slot: None,
            depth,
            privates,
        }
    }

    /// `var x = <const expr>` arrives as a `LocalVariable::init`; apply it
    /// before the body runs (slots are otherwise zero-initialized).
    fn init_locals(&mut self) {
        for (h, lv) in self.fun.local_variables.iter() {
            if let Some(init) = lv.init {
                let v = self.eval(init);
                self.slots[h.index()] = Some(v);
            }
        }
    }

    // -- expression evaluation -------------------------------------------

    /// An expression's value: the one captured at its `Emit` (or `Call`)
    /// point if it has one. Anything not emitted — literals, constants,
    /// argument/variable references and the pointer chains built on them — is
    /// pure, so it is simply recomputed on every use, never memoized: a
    /// pointer like `arr[i]` must see the current `i` on every loop iteration.
    fn eval(&mut self, e: Handle<Expression>) -> Value {
        if let Some(v) = &self.memo[e.index()] {
            return v.clone();
        }
        self.eval_inner(e)
    }

    fn eval_inner(&mut self, e: Handle<Expression>) -> Value {
        match &self.fun.expressions[e] {
            Expression::Literal(l) => literal_value(*l),
            Expression::Constant(h) => self.eval_const(self.sh.module.constants[*h].init),
            Expression::Override(_) => panic!("interp: overrides unsupported"),
            Expression::ZeroValue(ty) => zero_value_for(self.sh, *ty),
            Expression::Compose { ty, components } => {
                let comps: Vec<Value> = components.iter().map(|&c| self.eval(c)).collect();
                compose_value(self.sh, *ty, comps)
            }
            Expression::Access { base, index } => {
                let b = self.eval(*base);
                let i = self.eval(*index);
                access_dyn(b, i)
            }
            Expression::AccessIndex { base, index } => {
                let b = self.eval(*base);
                access_static(b, *index)
            }
            Expression::Splat { size, value } => {
                let v = self.eval(*value);
                Value::Vec(vec![v; vsize(*size) as usize])
            }
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let v = self.eval(*vector);
                match v {
                    Value::Vec(vs) => Value::Vec(
                        (0..vsize(*size) as usize)
                            .map(|i| vs[swizzle_index(pattern[i])].clone())
                            .collect(),
                    ),
                    other => panic!("interp: swizzle of {other:?}"),
                }
            }
            Expression::FunctionArgument(i) => self.args[*i as usize].clone(),
            Expression::GlobalVariable(h) => {
                let gv = &self.sh.module.global_variables[*h];
                match gv.space {
                    AddressSpace::Uniform | AddressSpace::Storage { .. } => {
                        let b = gv.binding.as_ref().expect("resource without binding");
                        Value::Ptr(Ptr {
                            base: PtrBase::Buffer {
                                group: b.group,
                                binding: b.binding,
                            },
                            steps: Vec::new(),
                        })
                    }
                    AddressSpace::Private => Value::Ptr(Ptr {
                        base: PtrBase::Private(*h),
                        steps: Vec::new(),
                    }),
                    other => panic!("interp: global address space {other:?} unsupported"),
                }
            }
            Expression::LocalVariable(h) => Value::Ptr(Ptr {
                base: PtrBase::Local(*h),
                steps: Vec::new(),
            }),
            Expression::Load { pointer } => {
                let p = self.eval(*pointer);
                self.load_pointer(p)
            }
            Expression::ImageSample { .. }
            | Expression::ImageLoad { .. }
            | Expression::ImageQuery { .. } => panic!("interp: textures unsupported"),
            Expression::Unary { op, expr } => {
                let v = self.eval(*expr);
                unary(*op, v)
            }
            Expression::Binary { op, left, right } => {
                let l = self.eval(*left);
                let r = self.eval(*right);
                binary(*op, l, r)
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                let c = self.eval(*condition);
                let a = self.eval(*accept);
                let r = self.eval(*reject);
                select(c, a, r)
            }
            Expression::Derivative { .. } => panic!("interp: derivatives unsupported"),
            Expression::Relational { fun, argument } => {
                let v = self.eval(*argument);
                relational(*fun, v)
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                let a = self.eval(*arg);
                let a1 = arg1.map(|h| self.eval(h));
                let a2 = arg2.map(|h| self.eval(h));
                let a3 = arg3.map(|h| self.eval(h));
                math(*fun, a, a1, a2, a3)
            }
            Expression::As {
                expr,
                kind,
                convert,
            } => {
                let v = self.eval(*expr);
                cast(v, *kind, *convert)
            }
            Expression::CallResult(_) => {
                // the value the most recent Call stored into this expression's
                // memo slot (see Statement::Call below)
                self.memo[e.index()]
                    .clone()
                    .expect("interp: CallResult read before its Call")
            }
            Expression::ArrayLength(_) => panic!("interp: arrayLength unsupported"),
            Expression::AtomicResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. }
            | Expression::RayQueryGetIntersection { .. }
            | Expression::RayQueryProceedResult
            | Expression::RayQueryVertexPositions { .. }
            | Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. }
            | Expression::CooperativeLoad { .. }
            | Expression::CooperativeMultiplyAdd { .. } => {
                panic!("interp: unsupported expression (atomics/rayquery/subgroup/cooperative)")
            }
        }
    }

    /// const-expression evaluation in the module's global arena
    fn eval_const(&mut self, e: Handle<Expression>) -> Value {
        match &self.sh.module.global_expressions[e] {
            Expression::Literal(l) => literal_value(*l),
            Expression::Constant(h) => self.eval_const(self.sh.module.constants[*h].init),
            Expression::ZeroValue(ty) => zero_value_for(self.sh, *ty),
            Expression::Compose { ty, components } => {
                let comps: Vec<Value> =
                    components.iter().map(|&c| self.eval_const(c)).collect();
                compose_value(self.sh, *ty, comps)
            }
            Expression::Splat { size, value } => {
                let v = self.eval_const(*value);
                Value::Vec(vec![v; vsize(*size) as usize])
            }
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let v = self.eval_const(*vector);
                match v {
                    Value::Vec(vs) => Value::Vec(
                        (0..vsize(*size) as usize)
                            .map(|i| vs[swizzle_index(pattern[i])].clone())
                            .collect(),
                    ),
                    other => panic!("interp: const swizzle of {other:?}"),
                }
            }
            Expression::Access { base, index } => {
                let b = self.eval_const(*base);
                let i = self.eval_const(*index);
                access_dyn(b, i)
            }
            Expression::AccessIndex { base, index } => {
                let b = self.eval_const(*base);
                access_static(b, *index)
            }
            Expression::Unary { op, expr } => unary(*op, self.eval_const(*expr)),
            Expression::Binary { op, left, right } => {
                let l = self.eval_const(*left);
                let r = self.eval_const(*right);
                binary(*op, l, r)
            }
            Expression::As {
                expr,
                kind,
                convert,
            } => cast(self.eval_const(*expr), *kind, *convert),
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => math(
                *fun,
                self.eval_const(*arg),
                arg1.map(|h| self.eval_const(h)),
                arg2.map(|h| self.eval_const(h)),
                arg3.map(|h| self.eval_const(h)),
            ),
            other => panic!("interp: const-expr {other:?} unsupported"),
        }
    }

    // -- pointer plumbing --------------------------------------------------

    /// walk a pointer's steps down to a byte offset + value shape, for buffer
    /// pointers. Composite types carry their arena handle; component-level
    /// steps through vectors/scalars produce a shape instead.
    fn buffer_ptr_offset_shape(&self, p: &Ptr) -> (u32, PtrShape) {
        let (group, binding) = match p.base {
            PtrBase::Buffer { group, binding } => (group, binding),
            PtrBase::Local(_) | PtrBase::Private(_) => {
                unreachable!("variable pointer in buffer_ptr_offset_shape")
            }
        };
        let gv = self
            .sh
            .module
            .global_variables
            .iter()
            .find(|(_, gv)| {
                gv.binding
                    .as_ref()
                    .is_some_and(|b| b.group == group && b.binding == binding)
            })
            .map(|(_, gv)| gv)
            .expect("interp: no global variable for buffer pointer");
        let mut off = 0;
        let mut shape = PtrShape::Ty(gv.ty);
        for s in &p.steps {
            shape = match (std::mem::replace(&mut shape, PtrShape::Scalar(naga::Scalar::F32)), s) {
                (PtrShape::Ty(ty), PtrStep::Idx(i)) => {
                    let stride = match &self.sh.module.types[ty].inner {
                        TypeInner::Array { base, .. } => array_stride(&self.sh.module, *base),
                        other => panic!("interp: index into {other:?}"),
                    };
                    off += i * stride;
                    let base = elem_ty(self.sh, ty);
                    PtrShape::Ty(base)
                }
                (PtrShape::Ty(ty), PtrStep::Member(m)) => {
                    match self.sh.module.types[ty].inner.clone() {
                        TypeInner::Struct { members, .. } => {
                            let offs = member_offsets(&self.sh.module, &members);
                            off += offs[*m as usize];
                            PtrShape::Ty(members[*m as usize].ty)
                        }
                        TypeInner::Array { base, .. } => {
                            off += m * array_stride(&self.sh.module, base);
                            PtrShape::Ty(base)
                        }
                        TypeInner::Vector { size, scalar } => {
                            debug_assert!(*m < vsize(size));
                            off += *m as u32 * w(scalar);
                            PtrShape::Scalar(scalar)
                        }
                        other => panic!(
                            "interp: member {m} on buffer pointer to {other:?} unsupported"
                        ),
                    }
                }
                (other, step) => {
                    panic!("interp: step {step:?} on buffer pointer shape {other:?}")
                }
            };
        }
        (off, shape)
    }

    fn load_pointer(&mut self, p: Value) -> Value {
        let Value::Ptr(ptr) = p else {
            panic!("interp: load of non-pointer {p:?}")
        };
        match ptr.base {
            PtrBase::Local(slot) => {
                // naga emits `var` declaration-time initialization either as a
                // const `init` on the LocalVariable (applied before first use)
                // or as an explicit `Store` at the declaration point; both are
                // honored here.
                if self.slots[slot.index()].is_none() {
                    if let Some(init) = self.fun.local_variables[slot].init {
                        let v = self.eval(init);
                        self.slots[slot.index()] = Some(v);
                    } else {
                        panic!("interp: read of uninitialized local {}", slot.index());
                    }
                }
                let mut v = self.slots[slot.index()].clone().unwrap();
                for s in &ptr.steps {
                    v = step_value(v, *s);
                }
                v
            }
            PtrBase::Private(g) => {
                let mut v = self.privates.borrow()[g.index()]
                    .clone()
                    .expect("interp: private global not initialized");
                for s in &ptr.steps {
                    v = step_value(v, *s);
                }
                v
            }
            PtrBase::Buffer { group, binding } => {
                let bytes = *self
                    .res
                    .bufs
                    .get(&(group, binding))
                    .expect("interp: shader reads an unbound buffer");
                let (off, shape) = self.buffer_ptr_offset_shape(&ptr);
                read_shape(&self.sh.module, bytes, off, &shape)
            }
        }
    }

    fn store_pointer(&mut self, p: Value, v: Value) {
        let Value::Ptr(ptr) = p else {
            panic!("interp: store to non-pointer {p:?}")
        };
        match ptr.base {
            PtrBase::Local(slot) => {
                // naga lowers `var x = init` to a bare slot plus a Store at
                // the declaration point — so a whole-slot Store to an
                // uninitialized local IS its initialization. (A const init on
                // the LocalVariable, if any, is applied first.) Only a
                // partial store through steps into an unset slot is a bug.
                if self.slots[slot.index()].is_none() {
                    if let Some(init) = self.fun.local_variables[slot].init {
                        let iv = self.eval(init);
                        self.slots[slot.index()] = Some(iv);
                    }
                }
                if ptr.steps.is_empty() {
                    self.slots[slot.index()] = Some(v);
                    return;
                }
                if self.slots[slot.index()].is_none() {
                    panic!(
                        "interp: partial store into uninitialized local {}",
                        slot.index()
                    );
                }
                let cur = self.slots[slot.index()].as_mut().unwrap();
                let mut cur: &mut Value = cur;
                for s in &ptr.steps {
                    cur = descend_mut(cur, *s);
                }
                *cur = v;
            }
            PtrBase::Private(g) => {
                let mut privs = self.privates.borrow_mut();
                let mut cur: &mut Value = privs[g.index()]
                    .as_mut()
                    .expect("interp: private global not initialized");
                for s in &ptr.steps {
                    cur = descend_mut(cur, *s);
                }
                *cur = v;
            }
            PtrBase::Buffer { group, binding } => {
                // vertex/fragment stages here only ever read buffers; the
                // backend holds them as shared slices for the whole draw
                let _ = (group, binding, v);
                panic!("interp: storage-buffer writes unsupported");
            }
        }
    }

    // -- statements ---------------------------------------------------------

    fn exec_block(&mut self, block: &naga::Block) -> Flow {
        for stmt in block.iter() {
            match self.exec_stmt(stmt) {
                Flow::Next => {}
                f => return f,
            }
        }
        Flow::Next
    }

    fn exec_stmt(&mut self, stmt: &naga::Statement) -> Flow {
        match stmt {
            naga::Statement::Emit(range) => {
                // naga semantics: the value is whatever the expression
                // computes *here* (a `let` of a load must not see later
                // stores, and a loop re-emits it every iteration)
                for h in range.clone() {
                    let v = self.eval_inner(h);
                    self.memo[h.index()] = Some(v);
                }
                Flow::Next
            }
            naga::Statement::Block(b) => self.exec_block(b),
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => {
                let c = self.eval(*condition);
                let block = if scalar_bool(&c) { accept } else { reject };
                self.exec_block(block)
            }
            naga::Statement::Switch { selector, cases } => {
                let s = self.eval(*selector);
                let sv = match s {
                    Value::I32(x) => naga::SwitchValue::I32(x),
                    Value::U32(x) => naga::SwitchValue::U32(x),
                    other => panic!("interp: switch on {other:?}"),
                };
                let chosen = cases
                    .iter()
                    .find(|c| c.value == sv || c.value == naga::SwitchValue::Default);
                match chosen {
                    Some(c) => {
                        if c.fall_through {
                            panic!("interp: switch fall-through unsupported");
                        }
                        self.exec_block(&c.body)
                    }
                    None => Flow::Next,
                }
            }
            naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } => loop {
                match self.exec_block(body) {
                    Flow::Next | Flow::Continue => {}
                    Flow::Break => return Flow::Next,
                    Flow::Return => return Flow::Return,
                    Flow::Kill => return Flow::Kill,
                }
                match self.exec_block(continuing) {
                    Flow::Next | Flow::Continue => {}
                    Flow::Break => return Flow::Next,
                    Flow::Return => return Flow::Return,
                    Flow::Kill => return Flow::Kill,
                }
                if let Some(cond) = break_if {
                    if scalar_bool(&self.eval(*cond)) {
                        return Flow::Next;
                    }
                }
            },
            naga::Statement::Break => Flow::Break,
            naga::Statement::Continue => Flow::Continue,
            naga::Statement::Return { value } => {
                if let Some(v) = value {
                    let v = self.eval(*v);
                    self.return_slot = Some(v);
                }
                Flow::Return
            }
            naga::Statement::Kill => Flow::Kill,
            naga::Statement::Store { pointer, value } => {
                let p = self.eval(*pointer);
                let v = self.eval(*value);
                self.store_pointer(p, v);
                Flow::Next
            }
            naga::Statement::Call {
                function,
                arguments,
                result,
            } => {
                let args: Vec<Value> = arguments.iter().map(|&a| self.eval(a)).collect();
                let ret = call_function(
                    self.sh,
                    self.res,
                    *function,
                    args,
                    self.depth + 1,
                    self.privates.clone(),
                );
                if let (Some(slot), Some(v)) = (result, ret) {
                    self.memo[slot.index()] = Some(v);
                }
                Flow::Next
            }
            naga::Statement::ImageStore { .. } => panic!("interp: ImageStore unsupported"),
            naga::Statement::Atomic { .. } | naga::Statement::ImageAtomic { .. } => {
                panic!("interp: atomics unsupported")
            }
            naga::Statement::WorkGroupUniformLoad { .. } => {
                panic!("interp: workgroup unsupported")
            }
            naga::Statement::RayQuery { .. }
            | naga::Statement::RayPipelineFunction(_)
            | naga::Statement::SubgroupBallot { .. }
            | naga::Statement::SubgroupGather { .. }
            | naga::Statement::SubgroupCollectiveOperation { .. }
            | naga::Statement::CooperativeStore { .. }
            | naga::Statement::ControlBarrier(_)
            | naga::Statement::MemoryBarrier(_) => panic!("interp: unsupported statement"),
        }
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Evaluate a stage entry point (by index) with the given arguments.
fn run_entry(sh: &Shader, entry: usize, res: &Resources, args: Vec<Value>) -> Option<Value> {
    let f = &sh.module.entry_points[entry].function;
    let privates: Privates = Default::default();
    let mut fr = Frame::new(sh, f, res, args, 0, privates.clone());
    // `var<private>` globals: fresh per invocation, from their const init
    // (or zero), before the body runs
    let inits: Vec<Option<Value>> = sh
        .module
        .global_variables
        .iter()
        .map(|(_, gv)| {
            (gv.space == AddressSpace::Private).then(|| match gv.init {
                Some(init) => fr.eval_const(init),
                None => zero_value_for(sh, gv.ty),
            })
        })
        .collect();
    *privates.borrow_mut() = inits;
    fr.init_locals();
    let flow = fr.exec_block(&f.body);
    match flow {
        Flow::Return | Flow::Next => fr.return_slot,
        Flow::Kill => None,
        Flow::Break | Flow::Continue => panic!("interp: break/continue escaped entry point"),
    }
}

/// Run a vertex entry point for one vertex/instance.
pub fn run_vertex(
    sh: &Shader,
    entry: usize,
    res: &Resources,
    vertex_index: u32,
    instance_index: u32,
) -> VertexOut {
    let f = &sh.module.entry_points[entry].function;
    let mut args = Vec::with_capacity(f.arguments.len());
    for a in &f.arguments {
        match &a.binding {
            Some(naga::Binding::BuiltIn(BuiltIn::VertexIndex)) => args.push(Value::U32(vertex_index)),
            Some(naga::Binding::BuiltIn(BuiltIn::InstanceIndex)) => {
                args.push(Value::U32(instance_index))
            }
            other => panic!("interp: vertex input {other:?} unsupported"),
        }
    }
    let out = run_entry(sh, entry, res, args).expect("interp: vertex entry must return");
    let result_ty = f.result.as_ref().expect("interp: vertex result").ty;
    // a bare `-> @builtin(position) vec4<f32>` return has no struct around it
    if let Some(res) = &f.result {
        if matches!(res.binding, Some(naga::Binding::BuiltIn(BuiltIn::Position { .. }))) {
            return VertexOut { position: vec4_f32(&out), varyings: Vec::new() };
        }
    }
    let out = match out {
        Value::Struct(m) => m,
        other => panic!("interp: vertex returned {other:?} (expected struct)"),
    };
    let members = match &sh.module.types[result_ty].inner {
        TypeInner::Struct { members, .. } => members,
        other => panic!("interp: vertex result {other:?} not a struct"),
    };
    let mut vo = VertexOut {
        position: [0.0; 4],
        varyings: Vec::new(),
    };
    for (m, v) in members.iter().zip(out) {
        match &m.binding {
            Some(naga::Binding::BuiltIn(BuiltIn::Position { .. })) => vo.position = vec4_f32(&v),
            Some(naga::Binding::Location { location, .. }) => vo.varyings.push((*location, v)),
            other => panic!("interp: vertex output binding {other:?} unsupported"),
        }
    }
    vo
}

/// Run a fragment entry point for one pixel. Returns the 4 target-component
/// values in target order, or None if the invocation killed/discarded.
pub fn run_fragment(
    sh: &Shader,
    entry: usize,
    res: &Resources,
    varyings: &[(u32, Value)],
    fb_position: [f32; 4],
) -> Option<[u32; 4]> {
    let f = &sh.module.entry_points[entry].function;
    let fb_pos_val = || {
        Value::Vec(fb_position.iter().map(|&c| Value::F32(c)).collect())
    };
    let mut args = Vec::with_capacity(f.arguments.len());
    for a in &f.arguments {
        let arg = match &a.binding {
            Some(naga::Binding::BuiltIn(BuiltIn::Position { .. })) => fb_pos_val(),
            Some(naga::Binding::Location { location, .. }) => varying_for(varyings, *location),
            None => {
                // struct input: members carry @location/@builtin(position)
                let mut sval = zero_value_for(sh, a.ty);
                if let TypeInner::Struct { members, .. } = &sh.module.types[a.ty].inner {
                    let members = members.clone();
                    for (i, m) in members.iter().enumerate() {
                        let mv = match &m.binding {
                            Some(naga::Binding::BuiltIn(BuiltIn::Position { .. })) => fb_pos_val(),
                            Some(naga::Binding::Location { location, .. }) => {
                                varying_for(varyings, *location)
                            }
                            other => {
                                panic!("interp: fragment input member binding {other:?}")
                            }
                        };
                        if let Value::Struct(ms) = &mut sval {
                            ms[i] = mv;
                        }
                    }
                }
                sval
            }
            other => panic!("interp: fragment input {other:?} unsupported"),
        };
        args.push(arg);
    }
    let ret = run_entry(sh, entry, res, args)?;
    let comps = match ret {
        Value::Vec(vs) => vs,
        other => panic!("interp: fragment returned {other:?}"),
    };
    let mut out = [0u32; 4];
    for (i, c) in comps.iter().take(4).enumerate() {
        // integer targets return their value, float targets the f32 bits
        out[i] = match c {
            Value::F32(x) => x.to_bits(),
            other => scalar_u32(other),
        };
    }
    Some(out)
}

fn varying_for(varyings: &[(u32, Value)], location: u32) -> Value {
    varyings
        .iter()
        .find(|(l, _)| *l == location)
        .map(|(_, v)| v.clone())
        .unwrap_or_else(|| panic!("interp: fragment input location {location} not provided"))
}

/// Invoke any function (entry points included) with by-value arguments.
fn call_function(
    sh: &Shader,
    res: &Resources,
    h: Handle<Function>,
    args: Vec<Value>,
    depth: u32,
    privates: Privates,
) -> Option<Value> {
    if depth > MAX_CALL_DEPTH {
        panic!("interp: call depth limit exceeded");
    }
    let f = &sh.module.functions[h];
    let mut fr = Frame::new(sh, f, res, args, depth, privates);
    fr.init_locals();
    let flow = fr.exec_block(&f.body);
    match flow {
        Flow::Return | Flow::Next => fr.return_slot,
        Flow::Kill => None,
        Flow::Break | Flow::Continue => panic!("interp: break/continue escaped function"),
    }
}

// ---------------------------------------------------------------------------
// Scalar/vector op implementations — the exactness contract
// ---------------------------------------------------------------------------

fn literal_value(l: Literal) -> Value {
    match l {
        Literal::F32(x) => Value::F32(x),
        Literal::F64(x) => Value::F32(x as f32),
        Literal::F16(_) => panic!("interp: f16 literals unsupported"),
        Literal::U16(x) => Value::U32(x as u32),
        Literal::I16(x) => Value::I32(x as i32),
        Literal::U32(x) => Value::U32(x),
        Literal::I32(x) => Value::I32(x),
        Literal::U64(x) => Value::U32(x as u32),
        Literal::I64(x) => Value::I32(x as i32),
        Literal::Bool(x) => Value::Bool(x),
        Literal::AbstractInt(x) => Value::I32(x as i32),
        Literal::AbstractFloat(x) => Value::F32(x as f32),
    }
}

fn zero_scalar(s: naga::Scalar) -> Value {
    match s.kind {
        ScalarKind::Float => Value::F32(0.0),
        ScalarKind::Sint => Value::I32(0),
        ScalarKind::Uint => Value::U32(0),
        ScalarKind::Bool => Value::Bool(false),
        k => panic!("interp: zero for {k:?}"),
    }
}

fn zero_value_for(sh: &Shader, ty: Handle<naga::Type>) -> Value {
    match &sh.module.types[ty].inner {
        TypeInner::Scalar(s) | TypeInner::Atomic(s) => zero_scalar(*s),
        TypeInner::Vector { size, scalar } => {
            Value::Vec(vec![zero_scalar(*scalar); vsize(*size) as usize])
        }
        TypeInner::Matrix { columns, rows, .. } => Value::Mat {
            cols: vec![[0.0; 4]; vsize(*columns) as usize],
            rows: vsize(*rows) as usize,
        },
        TypeInner::Array { base, size, .. } => {
            let n = match size {
                naga::ArraySize::Constant(c) => c.get(),
                _ => panic!("interp: unsized zero array"),
            };
            Value::Arr(vec![zero_value_for(sh, *base); n as usize])
        }
        TypeInner::Struct { members, .. } => Value::Struct(
            members
                .iter()
                .map(|m| zero_value_for(sh, m.ty))
                .collect(),
        ),
        other => panic!("interp: zero value for {other:?}"),
    }
}

fn compose_value(sh: &Shader, ty: Handle<naga::Type>, comps: Vec<Value>) -> Value {
    match &sh.module.types[ty].inner {
        TypeInner::Vector { .. } => {
            // vec4(vec2, vec2) / vec4(vec3, f32): flatten vector components
            let mut flat = Vec::with_capacity(comps.len());
            for c in comps {
                match c {
                    Value::Vec(vs) => flat.extend(vs),
                    scalar => flat.push(scalar),
                }
            }
            Value::Vec(flat)
        }
        TypeInner::Array { .. } => Value::Arr(comps),
        TypeInner::Struct { .. } => Value::Struct(comps),
        TypeInner::Matrix { columns, rows, .. } => {
            let mut cols = Vec::with_capacity(vsize(*columns) as usize);
            for c in comps {
                match c {
                    Value::Vec(vs) => {
                        let mut col = [0.0f32; 4];
                        for (r, v) in vs.iter().enumerate() {
                            col[r] = scalar_f32(v);
                        }
                        cols.push(col);
                    }
                    other => panic!("interp: matrix compose from {other:?}"),
                }
            }
            Value::Mat {
                cols,
                rows: vsize(*rows) as usize,
            }
        }
        other => panic!("interp: compose into {other:?}"),
    }
}

fn swizzle_index(c: SwizzleComponent) -> usize {
    match c {
        SwizzleComponent::X => 0,
        SwizzleComponent::Y => 1,
        SwizzleComponent::Z => 2,
        SwizzleComponent::W => 3,
    }
}

fn scalar_u32(v: &Value) -> u32 {
    match v {
        Value::U32(x) => *x,
        Value::I32(x) => *x as u32,
        other => panic!("interp: expected int, got {other:?}"),
    }
}

fn scalar_f32(v: &Value) -> f32 {
    match v {
        Value::F32(x) => *x,
        other => panic!("interp: expected f32, got {other:?}"),
    }
}

fn scalar_bool(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        other => panic!("interp: expected bool, got {other:?}"),
    }
}

fn vec4_f32(v: &Value) -> [f32; 4] {
    match v {
        Value::Vec(vs) => {
            let mut out = [0.0; 4];
            for (i, c) in vs.iter().enumerate().take(4) {
                out[i] = scalar_f32(c);
            }
            out
        }
        other => panic!("interp: expected vec4<f32>, got {other:?}"),
    }
}

fn nth(vs: Vec<Value>, i: u32, what: &str) -> Value {
    let n = vs.len();
    vs.into_iter()
        .nth(i as usize)
        .unwrap_or_else(|| panic!("interp: index {i} out of range {n} of {what}"))
}

fn access_dyn(base: Value, index: Value) -> Value {
    match base {
        Value::Ptr(mut p) => {
            p.steps.push(PtrStep::Idx(scalar_u32(&index)));
            Value::Ptr(p)
        }
        Value::Arr(vs) => nth(vs, scalar_u32(&index), "array"),
        Value::Vec(vs) => nth(vs, scalar_u32(&index), "vector"),
        other => panic!("interp: dynamic access into {other:?}"),
    }
}

fn access_static(base: Value, index: u32) -> Value {
    match base {
        Value::Ptr(mut p) => {
            p.steps.push(PtrStep::Member(index));
            Value::Ptr(p)
        }
        Value::Arr(vs) => nth(vs, index, "array"),
        Value::Vec(vs) => nth(vs, index, "vector"),
        Value::Struct(vs) => nth(vs, index, "struct"),
        Value::Mat { cols, rows } => Value::Vec(
            (0..rows)
                .map(|r| Value::F32(cols[index as usize][r]))
                .collect(),
        ),
        other => panic!("interp: access into {other:?}"),
    }
}

fn step_value(v: Value, s: PtrStep) -> Value {
    match (v, s) {
        (Value::Arr(vs), PtrStep::Idx(i)) => nth(vs, i, "array"),
        (Value::Vec(vs), PtrStep::Idx(i)) => nth(vs, i, "vector"),
        (Value::Vec(vs), PtrStep::Member(i)) => nth(vs, i, "vector"),
        (Value::Arr(vs), PtrStep::Member(i)) => nth(vs, i, "array"),
        (Value::Struct(vs), PtrStep::Member(i)) => nth(vs, i, "struct"),
        (Value::Mat { cols, rows }, PtrStep::Member(i)) => Value::Vec(
            (0..rows)
                .map(|r| Value::F32(cols[i as usize][r]))
                .collect(),
        ),
        (other, s) => panic!("interp: step {s:?} on {other:?}"),
    }
}

fn descend_mut<'v>(v: &'v mut Value, s: PtrStep) -> &'v mut Value {
    match (v, s) {
        (Value::Arr(vs), PtrStep::Idx(i)) => &mut vs[i as usize],
        (Value::Vec(vs), PtrStep::Idx(i)) => &mut vs[i as usize],
        // `v.x = ..` is an AccessIndex (Member) step into a vector pointer
        (Value::Vec(vs), PtrStep::Member(i)) => &mut vs[i as usize],
        (Value::Arr(vs), PtrStep::Member(i)) => &mut vs[i as usize],
        (Value::Struct(vs), PtrStep::Member(i)) => &mut vs[i as usize],
        (other, s) => panic!("interp: descend {s:?} on {other:?}"),
    }
}

fn elem_ty(sh: &Shader, ty: Handle<naga::Type>) -> Handle<naga::Type> {
    match &sh.module.types[ty].inner {
        TypeInner::Array { base, .. } => *base,
        other => panic!("interp: element type of {other:?}"),
    }
}

fn unary(op: UnaryOperator, v: Value) -> Value {
    fn s(op: UnaryOperator, v: Value) -> Value {
        match (op, v) {
            (UnaryOperator::Negate, Value::F32(x)) => Value::F32(-x),
            (UnaryOperator::Negate, Value::I32(x)) => Value::I32(x.wrapping_neg()),
            (UnaryOperator::Negate, Value::U32(x)) => Value::U32(x.wrapping_neg()),
            (UnaryOperator::LogicalNot, Value::Bool(b)) => Value::Bool(!b),
            (UnaryOperator::BitwiseNot, Value::I32(x)) => Value::I32(!x),
            (UnaryOperator::BitwiseNot, Value::U32(x)) => Value::U32(!x),
            (o, v) => panic!("interp: unary {o:?} on {v:?}"),
        }
    }
    match v {
        Value::Vec(vs) => Value::Vec(vs.into_iter().map(|c| s(op, c)).collect()),
        v => s(op, v),
    }
}

fn binary(op: BinaryOperator, l: Value, r: Value) -> Value {
    fn s(op: BinaryOperator, l: Value, r: Value) -> Value {
        use BinaryOperator as B;
        match (op, l, r) {
            (B::Add, Value::F32(a), Value::F32(b)) => Value::F32(a + b),
            (B::Add, Value::I32(a), Value::I32(b)) => Value::I32(a.wrapping_add(b)),
            (B::Add, Value::U32(a), Value::U32(b)) => Value::U32(a.wrapping_add(b)),
            (B::Subtract, Value::F32(a), Value::F32(b)) => Value::F32(a - b),
            (B::Subtract, Value::I32(a), Value::I32(b)) => Value::I32(a.wrapping_sub(b)),
            (B::Subtract, Value::U32(a), Value::U32(b)) => Value::U32(a.wrapping_sub(b)),
            (B::Multiply, Value::F32(a), Value::F32(b)) => Value::F32(a * b),
            (B::Multiply, Value::I32(a), Value::I32(b)) => Value::I32(a.wrapping_mul(b)),
            (B::Multiply, Value::U32(a), Value::U32(b)) => Value::U32(a.wrapping_mul(b)),
            (B::Divide, Value::F32(a), Value::F32(b)) => Value::F32(a / b),
            (B::Divide, Value::I32(a), Value::I32(b)) => Value::I32(a.wrapping_div(b)),
            (B::Divide, Value::U32(a), Value::U32(b)) => Value::U32(a.wrapping_div(b)),
            (B::Modulo, Value::F32(a), Value::F32(b)) => Value::F32(a % b),
            (B::Modulo, Value::I32(a), Value::I32(b)) => Value::I32(a.wrapping_rem(b)),
            (B::Modulo, Value::U32(a), Value::U32(b)) => Value::U32(a.wrapping_rem(b)),
            (B::Equal, a, b) => Value::Bool(scalar_eq(&a, &b)),
            (B::NotEqual, a, b) => Value::Bool(!scalar_eq(&a, &b)),
            (B::Less, Value::F32(a), Value::F32(b)) => Value::Bool(a < b),
            (B::Less, Value::I32(a), Value::I32(b)) => Value::Bool(a < b),
            (B::Less, Value::U32(a), Value::U32(b)) => Value::Bool(a < b),
            (B::LessEqual, Value::F32(a), Value::F32(b)) => Value::Bool(a <= b),
            (B::LessEqual, Value::I32(a), Value::I32(b)) => Value::Bool(a <= b),
            (B::LessEqual, Value::U32(a), Value::U32(b)) => Value::Bool(a <= b),
            (B::Greater, Value::F32(a), Value::F32(b)) => Value::Bool(a > b),
            (B::Greater, Value::I32(a), Value::I32(b)) => Value::Bool(a > b),
            (B::Greater, Value::U32(a), Value::U32(b)) => Value::Bool(a > b),
            (B::GreaterEqual, Value::F32(a), Value::F32(b)) => Value::Bool(a >= b),
            (B::GreaterEqual, Value::I32(a), Value::I32(b)) => Value::Bool(a >= b),
            (B::GreaterEqual, Value::U32(a), Value::U32(b)) => Value::Bool(a >= b),
            (B::And, Value::I32(a), Value::I32(b)) => Value::I32(a & b),
            (B::And, Value::U32(a), Value::U32(b)) => Value::U32(a & b),
            (B::ExclusiveOr, Value::I32(a), Value::I32(b)) => Value::I32(a ^ b),
            (B::ExclusiveOr, Value::U32(a), Value::U32(b)) => Value::U32(a ^ b),
            (B::InclusiveOr, Value::I32(a), Value::I32(b)) => Value::I32(a | b),
            (B::InclusiveOr, Value::U32(a), Value::U32(b)) => Value::U32(a | b),
            (B::LogicalAnd, Value::Bool(a), Value::Bool(b)) => Value::Bool(a && b),
            (B::LogicalOr, Value::Bool(a), Value::Bool(b)) => Value::Bool(a || b),
            (B::ShiftLeft, Value::I32(a), Value::U32(b)) => Value::I32(a.wrapping_shl(b & 31)),
            (B::ShiftLeft, Value::U32(a), Value::U32(b)) => Value::U32(a.wrapping_shl(b & 31)),
            (B::ShiftRight, Value::I32(a), Value::U32(b)) => Value::I32(a.wrapping_shr(b & 31)),
            (B::ShiftRight, Value::U32(a), Value::U32(b)) => Value::U32(a.wrapping_shr(b & 31)),
            (o, a, b) => panic!("interp: binary {o:?} on {a:?} / {b:?}"),
        }
    }
    match (&l, &r) {
        (Value::Vec(_), Value::Vec(_)) => {
            let ls = into_vec(&l);
            let rs = into_vec(&r);
            Value::Vec(
                ls.into_iter()
                    .zip(rs)
                    .map(|(a, b)| s(op, a, b))
                    .collect(),
            )
        }
        // scalar broadcast (vec * scalar, scalar + vec, ...)
        (Value::Vec(ls), _) => {
            Value::Vec(ls.iter().cloned().map(|a| s(op, a, r.clone())).collect())
        }
        (_, Value::Vec(rs)) => {
            Value::Vec(rs.iter().cloned().map(|b| s(op, l.clone(), b)).collect())
        }
        _ => s(op, l, r),
    }
}

fn scalar_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::F32(x), Value::F32(y)) => x == y,
        (Value::I32(x), Value::I32(y)) => x == y,
        (Value::U32(x), Value::U32(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (a, b) => panic!("interp: compare {a:?} with {b:?}"),
    }
}

fn into_vec(v: &Value) -> Vec<Value> {
    match v {
        Value::Vec(vs) => vs.clone(),
        other => panic!("interp: expected vector, got {other:?}"),
    }
}

fn select(cond: Value, accept: Value, reject: Value) -> Value {
    match cond {
        Value::Bool(b) => if b { accept } else { reject },
        Value::Vec(cs) => {
            let a = into_vec(&accept);
            let r = into_vec(&reject);
            Value::Vec(
                cs.into_iter()
                    .zip(a.into_iter().zip(r))
                    .map(|(c, (a, r))| if scalar_bool(&c) { a } else { r })
                    .collect(),
            )
        }
        other => panic!("interp: select on {other:?}"),
    }
}

fn relational(fun: naga::RelationalFunction, v: Value) -> Value {
    use naga::RelationalFunction as R;
    let comps = match &v {
        Value::Vec(_) => into_vec(&v),
        _ => vec![v],
    };
    match fun {
        R::All => Value::Bool(comps.iter().all(|c| scalar_bool(c))),
        R::Any => Value::Bool(comps.iter().any(|c| scalar_bool(c))),
        R::IsNan => Value::Bool(
            comps
                .iter()
                .any(|c| matches!(c, Value::F32(x) if x.is_nan())),
        ),
        R::IsInf => Value::Bool(
            comps
                .iter()
                .any(|c| matches!(c, Value::F32(x) if x.is_infinite())),
        ),

    }
}

fn cast(v: Value, kind: ScalarKind, convert: Option<naga::Bytes>) -> Value {
    fn s(v: Value, kind: ScalarKind, convert: Option<naga::Bytes>) -> Value {
        match convert {
            // convert == None is a bitcast (WGSL bitcast<T>)
            None => match (v, kind) {
                (Value::F32(x), ScalarKind::Sint) => Value::I32(x.to_bits() as i32),
                (Value::F32(x), ScalarKind::Uint) => Value::U32(x.to_bits()),
                (Value::I32(x), ScalarKind::Float) => Value::F32(f32::from_bits(x as u32)),
                (Value::U32(x), ScalarKind::Float) => Value::F32(f32::from_bits(x)),
                (Value::I32(x), ScalarKind::Uint) => Value::U32(x as u32),
                (Value::U32(x), ScalarKind::Sint) => Value::I32(x as i32),
                (Value::F32(x), ScalarKind::Float) => Value::F32(x),
                (Value::I32(x), ScalarKind::Sint) => Value::I32(x),
                (Value::U32(x), ScalarKind::Uint) => Value::U32(x),
                (a, k) => panic!("interp: bitcast to {k:?} from {a:?}"),
            },
            Some(_) => match (v, kind) {
                (Value::F32(x), ScalarKind::Uint) => Value::U32(x as u32),
                (Value::F32(x), ScalarKind::Sint) => Value::I32(x as i32),
                (Value::F32(x), ScalarKind::Float) => Value::F32(x),
                (Value::F32(x), ScalarKind::Bool) => Value::Bool(x != 0.0),
                (Value::I32(x), ScalarKind::Float) => Value::F32(x as f32),
                (Value::I32(x), ScalarKind::Uint) => Value::U32(x as u32),
                (Value::I32(x), ScalarKind::Sint) => Value::I32(x),
                (Value::I32(x), ScalarKind::Bool) => Value::Bool(x != 0),
                (Value::U32(x), ScalarKind::Float) => Value::F32(x as f32),
                (Value::U32(x), ScalarKind::Sint) => Value::I32(x as i32),
                (Value::U32(x), ScalarKind::Uint) => Value::U32(x),
                (Value::U32(x), ScalarKind::Bool) => Value::Bool(x != 0),
                (Value::Bool(x), ScalarKind::Float) => Value::F32(x as i32 as f32),
                (Value::Bool(x), ScalarKind::Uint) => Value::U32(x as u32),
                (Value::Bool(x), ScalarKind::Sint) => Value::I32(x as i32),
                (a, k) => panic!("interp: cast {a:?} to {k:?}"),
            },
        }
    }
    match v {
        Value::Vec(vs) => Value::Vec(vs.into_iter().map(|c| s(c, kind, convert)).collect()),
        v => s(v, kind, convert),
    }
}

/// math builtins whose result shape differs from their vector arguments —
/// these run whole, never componentwise
fn is_aggregate(fun: MathFunction) -> bool {
    matches!(
        fun,
        MathFunction::Dot
            | MathFunction::Cross
            | MathFunction::Distance
            | MathFunction::Length
            | MathFunction::Normalize
            | MathFunction::Outer
            | MathFunction::FaceForward
            | MathFunction::Reflect
            | MathFunction::Refract
            | MathFunction::Transpose
            | MathFunction::Inverse
            | MathFunction::Determinant
    )
}

fn math(
    fun: MathFunction,
    a: Value,
    a1: Option<Value>,
    a2: Option<Value>,
    a3: Option<Value>,
) -> Value {
    if !is_aggregate(fun) {
        if let Value::Vec(_) = a {
            // elementwise: broadcast scalar extra args to the vector length
            let n = into_vec(&a).len();
            let bcast = |v: Option<Value>| -> Option<Vec<Value>> {
                v.map(|v| match v {
                    Value::Vec(vs) => vs,
                    scalar => vec![scalar; n],
                })
            };
            let comps = into_vec(&a);
            let c1 = bcast(a1);
            let c2 = bcast(a2);
            let c3 = bcast(a3);
            return Value::Vec(
                (0..n)
                    .map(|i| {
                        math_scalar(
                            fun,
                            comps[i].clone(),
                            c1.as_ref().map(|v| v[i].clone()),
                            c2.as_ref().map(|v| v[i].clone()),
                            c3.as_ref().map(|v| v[i].clone()),
                        )
                    })
                    .collect(),
            );
        }
    }
    match fun {
        MathFunction::Dot => {
            let x = into_vec(&a);
            let y = into_vec(&a1.expect("dot needs two args"));
            let mut acc = 0.0f32;
            for (p, q) in x.into_iter().zip(y) {
                acc = acc + scalar_f32(&p) * scalar_f32(&q);
            }
            Value::F32(acc)
        }
        MathFunction::Cross => {
            let x = into_vec(&a);
            let y = into_vec(&a1.expect("cross needs two args"));
            let g = |v: &[Value], i: usize| scalar_f32(&v[i]);
            Value::Vec(vec![
                Value::F32(g(&x, 1) * g(&y, 2) - g(&x, 2) * g(&y, 1)),
                Value::F32(g(&x, 2) * g(&y, 0) - g(&x, 0) * g(&y, 2)),
                Value::F32(g(&x, 0) * g(&y, 1) - g(&x, 1) * g(&y, 0)),
            ])
        }
        MathFunction::Length => {
            let x = into_vec(&a);
            let mut acc = 0.0f32;
            for p in x {
                let c = scalar_f32(&p);
                acc = acc + c * c;
            }
            Value::F32(acc.sqrt())
        }
        MathFunction::Distance => {
            let x = into_vec(&a);
            let y = into_vec(&a1.expect("distance needs two args"));
            let mut acc = 0.0f32;
            for (p, q) in x.into_iter().zip(y) {
                let d = scalar_f32(&p) - scalar_f32(&q);
                acc = acc + d * d;
            }
            Value::F32(acc.sqrt())
        }
        MathFunction::Normalize => {
            let x = into_vec(&a);
            let mut acc = 0.0f32;
            for p in &x {
                let c = scalar_f32(p);
                acc = acc + c * c;
            }
            let len = acc.sqrt();
            Value::Vec(
                x.into_iter()
                    .map(|p| Value::F32(scalar_f32(&p) / len))
                    .collect(),
            )
        }
        // every other builtin on a scalar (or a scalar-arg elementwise op
        // that reaches this far) is math_scalar's job
        other => math_scalar(other, a, a1, a2, a3),
    }
}

fn math_scalar(
    fun: MathFunction,
    a: Value,
    a1: Option<Value>,
    a2: Option<Value>,
    _a3: Option<Value>,
) -> Value {
    let f = |v: &Value| -> f32 { scalar_f32(v) };
    let bf = |v: &Option<Value>| -> f32 { scalar_f32(v.as_ref().expect("interp: missing math arg")) };
    match fun {
        MathFunction::Abs => match a {
            Value::F32(x) => Value::F32(x.abs()),
            Value::I32(x) => Value::I32(x.wrapping_abs()),
            Value::U32(x) => Value::U32(x),
            o => panic!("interp: abs {o:?}"),
        },
        MathFunction::Min => match (a, a1.unwrap()) {
            (Value::F32(x), Value::F32(y)) => Value::F32(x.min(y)),
            (Value::I32(x), Value::I32(y)) => Value::I32(x.min(y)),
            (Value::U32(x), Value::U32(y)) => Value::U32(x.min(y)),
            (a, b) => panic!("interp: min {a:?} {b:?}"),
        },
        MathFunction::Max => match (a, a1.unwrap()) {
            (Value::F32(x), Value::F32(y)) => Value::F32(x.max(y)),
            (Value::I32(x), Value::I32(y)) => Value::I32(x.max(y)),
            (Value::U32(x), Value::U32(y)) => Value::U32(x.max(y)),
            (a, b) => panic!("interp: max {a:?} {b:?}"),
        },
        MathFunction::Clamp => {
            // WGSL clamp(x, lo, hi) = min(max(x, lo), hi)
            let (lo, hi) = (bf(&a1), bf(&a2));
            Value::F32(f(&a).min(hi).max(lo))
        }
        MathFunction::Saturate => Value::F32(f(&a).min(1.0).max(0.0)),
        MathFunction::Cos => Value::F32(f(&a).cos()),
        MathFunction::Cosh => Value::F32(f(&a).cosh()),
        MathFunction::Sin => Value::F32(f(&a).sin()),
        MathFunction::Sinh => Value::F32(f(&a).sinh()),
        MathFunction::Tan => Value::F32(f(&a).tan()),
        MathFunction::Tanh => Value::F32(f(&a).tanh()),
        MathFunction::Acos => Value::F32(f(&a).acos()),
        MathFunction::Asin => Value::F32(f(&a).asin()),
        MathFunction::Atan => Value::F32(f(&a).atan()),
        MathFunction::Atan2 => Value::F32(f(&a).atan2(bf(&a1))),
        MathFunction::Asinh => Value::F32(f(&a).asinh()),
        MathFunction::Acosh => Value::F32(f(&a).acosh()),
        MathFunction::Atanh => Value::F32(f(&a).atanh()),
        MathFunction::Radians => Value::F32(f(&a) * (std::f32::consts::PI / 180.0)),
        MathFunction::Degrees => Value::F32(f(&a) * (180.0 / std::f32::consts::PI)),
        MathFunction::Ceil => Value::F32(f(&a).ceil()),
        MathFunction::Floor => Value::F32(f(&a).floor()),
        MathFunction::Round => Value::F32(f(&a).round()),
        MathFunction::Fract => {
            let x = f(&a);
            Value::F32(x - x.floor())
        }
        MathFunction::Trunc => Value::F32(f(&a).trunc()),
        MathFunction::Exp => Value::F32(f(&a).exp()),
        MathFunction::Exp2 => Value::F32(f(&a).exp2()),
        MathFunction::Log => Value::F32(f(&a).ln()),
        MathFunction::Log2 => Value::F32(f(&a).log2()),
        MathFunction::Pow => Value::F32(f(&a).powf(bf(&a1))),
        MathFunction::Sign => match a {
            Value::F32(x) => Value::F32(match x.partial_cmp(&0.0) {
                Some(std::cmp::Ordering::Greater) => 1.0,
                Some(std::cmp::Ordering::Less) => -1.0,
                _ => 0.0,
            }),
            Value::I32(x) => Value::I32(x.signum()),
            o => panic!("interp: sign {o:?}"),
        },
        MathFunction::Fma => Value::F32(f(&a).mul_add(bf(&a1), bf(&a2))),
        MathFunction::Mix => {
            // WGSL mix(x, y, a) = x * (1 - a) + y * a
            let x = f(&a);
            let y = bf(&a1);
            let t = bf(&a2);
            Value::F32(x * (1.0 - t) + y * t)
        }
        MathFunction::Step => {
            let edge = f(&a);
            let x = bf(&a1);
            Value::F32(if x < edge { 0.0 } else { 1.0 })
        }
        MathFunction::SmoothStep => {
            let lo = f(&a);
            let hi = bf(&a1);
            let x = bf(&a2);
            let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
            Value::F32(t * t * (3.0 - 2.0 * t))
        }
        MathFunction::Sqrt => Value::F32(f(&a).sqrt()),
        MathFunction::InverseSqrt => Value::F32(1.0 / f(&a).sqrt()),
        other => panic!("interp: math {other:?} unsupported"),
    }
}
