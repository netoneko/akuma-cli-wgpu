//! `akuma-wgpu exec-selftest`: run each snippet through every executor the
//! build offers (tree-walking interpreter, register VM, x86-64 JIT) and
//! demand bit-identical results.
//!
//! Each snippet is the body of a vertex shader that reads a pseudo-random
//! storage buffer (with a few NaN/inf/±0 entries) plus a uniform and writes
//! `pos` / `a` / `b`. A snippet the compiler declines is reported, not failed
//! — the pipeline would simply use the interpreter for it. Snippets marked
//! `interp: false` use ops the interpreter does not implement (matrix
//! products) and are checked VM vs JIT only.

use std::sync::Arc;

use super::exec::{RawVertex, Stage};
use super::interp::{Resources, Shader};

const HEAD: &str = r#"
@group(0) @binding(0) var<storage, read> d: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> u: vec4<f32>;
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) a: vec4<u32>,
    @location(1) @interpolate(flat) b: vec4<f32>,
};
fn helper(x: f32, y: f32) -> f32 { return x * 2.0 - y; }
fn helper_vec(v: vec3<f32>) -> vec3<f32> { return v + vec3<f32>(1.0, 2.0, 3.0); }
var<private> pg: vec4<f32> = vec4<f32>(1.0, 2.0, 3.0, 4.0);
var<private> pc: u32;
var<private> pa: array<f32, 4>;
fn bump(x: f32) { pg.x = pg.x + x; pc = pc + 1u; }
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    let p = d[vi % 64u];
    let q = d[(vi * 7u + 3u) % 64u];
"#;
const TAIL: &str = r#"
    var o: VsOut;
    o.pos = pos;
    o.a = a;
    o.b = b;
    return o;
}
"#;

struct Case {
    name: &'static str,
    body: &'static str,
    interp: bool,
}

const CASES: &[Case] = &[
    Case { name: "var<private> globals (init, helper writes, branches, dyn index)", interp: true, body: r#"
        bump(p.x);
        if (q.x > 0.0) { bump(q.y); pa[vi % 4u] = q.z; } else { pg.w = -1.0; }
        bump(1.0);
        let pos = pg + vec4<f32>(pa[0], pa[1], pa[2], pa[3]);
        let a = vec4<u32>(pc, pc * 3u, vi, 0u);
        let b = vec4<f32>(pa[(vi + 1u) % 4u], pg.y, pg.z, pg.w);"# },
    Case { name: "float arithmetic", interp: true, body: r#"
        let pos = p * q + p / q - q;
        let a = vec4<u32>(vi, vi * 3u, 7u, 9u);
        let b = p - q * 0.5;"# },
    Case { name: "integer arithmetic + shifts + bit ops", interp: true, body: r#"
        let x = vi * 2654435761u;
        let a = vec4<u32>(x ^ (x >> 7u), (x << 3u) | 1u, x & 0xffffu, ~x);
        let i = bitcast<i32>(x);
        let b = vec4<f32>(f32(i >> 4u), f32(i / 7), f32(i % 11), f32(-i));
        let pos = vec4<f32>(f32(x % 1000u), f32(vi / (vi % 5u + 1u)), 0.0, 1.0);"# },
    Case { name: "unary + abs/min/max/clamp/saturate", interp: true, body: r#"
        let pos = vec4<f32>(-p.x, abs(q.y), min(p.z, q.z), max(p.w, q.w));
        let a = vec4<u32>(u32(abs(i32(p.x))), u32(min(vi, 40u)), u32(max(vi, 20u)), 0u);
        let b = vec4<f32>(clamp(p.x, -1.0, 1.0), saturate(q.y), clamp(p.z, q.z, q.w + 1.0), fract(p.w));"# },
    Case { name: "comparisons, NaN, select", interp: true, body: r#"
        let c = p.x < q.x;
        let e = p.y == q.y;
        let n = p.z != q.z;
        let pos = select(p, q, p > q);
        let a = vec4<u32>(u32(c), u32(e), u32(n), u32(p.w >= q.w));
        let b = vec4<f32>(select(1.0, 2.0, c), select(3.0, 4.0, !e), select(5.0, 6.0, c && n), select(7.0, 8.0, c || e));"# },
    Case { name: "float->int->float casts", interp: true, body: r#"
        let pos = vec4<f32>(f32(u32(p.x)), f32(i32(p.y)), f32(vi), f32(i32(vi) - 100));
        let a = vec4<u32>(u32(p.z), bitcast<u32>(i32(q.x)), bitcast<u32>(p.w), u32(q.y > 0.0));
        let b = vec4<f32>(bitcast<f32>(vi), f32(bitcast<i32>(q.z)), 0.0, 1.0);"# },
    Case { name: "transcendentals", interp: true, body: r#"
        let pos = vec4<f32>(sin(p.x), cos(q.x), tan(p.y * 0.01), atan(q.y));
        let a = vec4<u32>(0u, 0u, 0u, 0u);
        let b = vec4<f32>(exp(p.z * 0.01), log(abs(q.z) + 1.0), sqrt(abs(p.w)), inverseSqrt(abs(q.w) + 1.0));"# },
    Case { name: "pow/atan2/floor/ceil/round/trunc/sign", interp: true, body: r#"
        let pos = vec4<f32>(pow(abs(p.x), 1.5), atan2(p.y, q.y), floor(p.z), ceil(q.z));
        let a = vec4<u32>(0u, 0u, 0u, 0u);
        let b = vec4<f32>(round(p.w), trunc(q.w), sign(p.x), sign(q.y));"# },
    Case { name: "dot/length/distance/normalize/cross", interp: true, body: r#"
        let v = p.xyz;
        let w = q.xyz;
        let pos = vec4<f32>(dot(v, w), length(v), distance(v, w), 1.0);
        let n = normalize(w + vec3<f32>(100.0));
        let c = cross(v, w);
        let a = vec4<u32>(0u, 0u, 0u, 0u);
        let b = vec4<f32>(n.x + c.x, n.y + c.y, n.z + c.z, dot(p, q));"# },
    Case { name: "mix/step/smoothstep/radians/degrees", interp: true, body: r#"
        let pos = vec4<f32>(mix(p.x, q.x, 0.25), step(p.y, q.y), smoothstep(-50.0, 50.0, p.z), radians(q.w));
        let a = vec4<u32>(0u, 0u, 0u, 0u);
        let b = vec4<f32>(degrees(p.w), mix(p.x, q.x, q.y * 0.01), step(0.0, p.z), smoothstep(q.x, q.x + 10.0, p.x));"# },
    Case { name: "vector swizzle/splat/compose", interp: true, body: r#"
        let v = vec3<f32>(p.zyx);
        let w = vec4<f32>(q.xy, v.yz);
        let s = vec4<f32>(p.w);
        let pos = w + s;
        let a = vec4<u32>(vec2<u32>(vi, vi + 1u), vec2<u32>(3u));
        let b = vec4<f32>(v, q.w);"# },
    Case { name: "if / else / nested", interp: true, body: r#"
        var pos = vec4<f32>(0.0);
        var a = vec4<u32>(0u);
        if (p.x > 0.0) {
            pos.x = 1.0;
            if (q.x > 0.0) { a.x = 11u; } else { a.x = 22u; }
        } else {
            pos.y = 2.0;
            a.y = 33u;
        }
        if (vi % 2u == 0u) { a.z = 5u; }
        var b = vec4<f32>(pos.x + pos.y, 0.0, 0.0, 1.0);"# },
    Case { name: "loop with break/continue", interp: true, body: r#"
        var acc = 0.0;
        var n = 0u;
        var i = 0u;
        loop {
            if (i >= 12u) { break; }
            i = i + 1u;
            if (i % 3u == 0u) { continue; }
            acc = acc + f32(i) * p.x;
            n = n + i;
        }
        let pos = vec4<f32>(acc, 0.0, 0.0, 1.0);
        let a = vec4<u32>(n, i, 0u, 0u);
        let b = vec4<f32>(acc, f32(n), 0.0, 0.0);"# },
    Case { name: "for loop + local array (static idx)", interp: true, body: r#"
        var arr = array<f32, 4>(1.0, 2.0, 3.0, 4.0);
        arr[1] = p.x;
        arr[3] = q.y;
        var s = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) { s = s + f32(k) + 0.5; }
        let pos = vec4<f32>(arr[0] + s, arr[1], arr[2], arr[3]);
        let a = vec4<u32>(0u);
        let b = vec4<f32>(s);"# },
    Case { name: "switch", interp: true, body: r#"
        var x = 0u;
        switch (vi % 4u) {
            case 0u: { x = 10u; }
            case 1u: { x = 20u; }
            case 2u: { x = 30u; }
            default: { x = 99u; }
        }
        let pos = vec4<f32>(f32(x));
        let a = vec4<u32>(x, vi, 0u, 0u);
        let b = vec4<f32>(0.0);"# },
    Case { name: "user function calls", interp: true, body: r#"
        let h = helper(p.x, q.x);
        let hv = helper_vec(p.xyz);
        let pos = vec4<f32>(h, hv.x, hv.y, hv.z);
        let a = vec4<u32>(0u);
        let b = vec4<f32>(helper(h, 1.0));"# },
    Case { name: "uniform + dynamic buffer index + isNan/all/any", interp: true, body: r#"
        let k = d[(vi + u32(u.x)) % 64u];
        let pos = k * u;
        let a = vec4<u32>(u32(any(vec3<bool>(k.x != k.x, k.y > 0.0, k.z < 0.0))), u32(all(k.xy == k.xy)), vi, 0u);
        let b = vec4<f32>(u.w, k.w, 0.0, 0.0);"# },
    Case { name: "dynamic index into a local array (load + store)", interp: true, body: r#"
        var arr = array<f32, 4>(1.0, 2.0, 3.0, 4.0);
        let i = vi % 4u;
        arr[i] = p.x;
        arr[(i + 1u) % 4u] = arr[(i + 2u) % 4u] + q.y;
        var v = array<vec2<f32>, 3>(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0), vec2<f32>(5.0, 6.0));
        let j = vi % 3u;
        v[j].y = q.z;
        let pos = vec4<f32>(arr[0], arr[1], arr[2], arr[3]);
        let a = vec4<u32>(i, j, 0u, 0u);
        let b = vec4<f32>(v[0].x + v[0].y, v[1].y, v[2].y, v[j].x);"# },
    Case { name: "matrix * vector, vector * matrix, matrix * matrix", interp: false, body: r#"
        let m = mat3x3<f32>(p.xyz, q.xyz, vec3<f32>(1.0, 2.0, 3.0));
        let n = mat3x3<f32>(q.xyz, p.xyz, vec3<f32>(3.0, 2.0, 1.0));
        let v = m * p.xyz;
        let w = q.xyz * n;
        let mm = m * n;
        let pos = vec4<f32>(v, 1.0);
        let a = vec4<u32>(0u);
        let b = vec4<f32>(w.x + mm[0].x, w.y + mm[1].y, w.z + mm[2].z, mm[1].x);"# },
];

fn data_bytes() -> Vec<u8> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut v: Vec<f32> = (0..64 * 4)
        .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 200.0)
        .collect();
    // special values in a few lanes
    v[5 * 4] = f32::NAN;
    v[9 * 4 + 1] = f32::INFINITY;
    v[13 * 4 + 2] = f32::NEG_INFINITY;
    v[17 * 4] = 0.0;
    v[17 * 4 + 1] = -0.0;
    v[21 * 4 + 3] = 1e-30;
    v[25 * 4] = 3.0e38;
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn same_bits(a: u32, b: u32) -> bool {
    a == b || (f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan())
}

fn diff(x: &RawVertex, y: &RawVertex) -> Option<String> {
    for i in 0..4 {
        if !same_bits(x.position[i].to_bits(), y.position[i].to_bits()) {
            return Some(format!(
                "pos[{i}]: {:#010x} vs {:#010x}",
                x.position[i].to_bits(),
                y.position[i].to_bits()
            ));
        }
    }
    for l in 0..2 {
        for c in 0..4 {
            if !same_bits(x.varyings[l][c], y.varyings[l][c]) {
                return Some(format!(
                    "loc{l}[{c}]: {:#010x} vs {:#010x}",
                    x.varyings[l][c], y.varyings[l][c]
                ));
            }
        }
    }
    None
}

const N_VERTS: u32 = 300;

/// through the batch entry point, so a wide stage really runs 4 lanes at a time
fn run_stage(st: &Stage, res: &Resources<'_>) -> Vec<RawVertex> {
    run_stage_with(st, res, false)
}

/// `spec`: run the program specialized to this draw's buffer contents
fn run_stage_with(st: &Stage, res: &Resources<'_>, spec: bool) -> Vec<RawVertex> {
    let plan = if spec { st.plan(res, u64::MAX) } else { None };
    let mut inv = st.begin_with(res, &plan);
    let ids: Vec<(u32, u32)> = (0..N_VERTS).map(|v| (v, 0)).collect();
    let attrs = vec![[[0u32; 4]; super::exec::MAX_LOC]; N_VERTS as usize];
    let mut out = Vec::new();
    inv.run_vertex_batch(&ids, &attrs, &mut out);
    out
}

/// Run one snippet through every executor (and specialized): (ok, report line, declined count).
fn run_case(name: &str, body: &str, interp: bool, res: &Resources<'_>, have_jit: bool) -> (bool, String, u32) {
    let src = format!("{HEAD}{body}{TAIL}");
    let mut declined = 0;
    let sh = match Shader::parse(&src) {
        Ok(s) => Arc::new(s),
        Err(e) => return (false, format!("{name:<52} snippet does not parse: {e}"), 0),
    };
    // the oracle: interpreter if it supports the snippet, else the VM
    let oracle_name = if interp { "interp" } else { "vm" };
    let oracle = match Stage::build(sh.clone(), 0, oracle_name) {
        Ok(s) => s,
        Err(e) => return (true, format!("DECL  {name:<52} {e}"), 1),
    };
    let want = run_stage(&oracle, res);
    let mut line = format!("{name:<52}");
    let mut ok = true;
    for mode in ["vm", "jit", "jitw"] {
        if mode == oracle_name {
            continue;
        }
        match Stage::build(sh.clone(), 0, mode) {
            Err(e) => {
                if mode.starts_with("jit") && !have_jit {
                    line.push_str(&format!("  {mode}: n/a"));
                } else {
                    line.push_str(&format!("  {mode}: declined ({e})"));
                    declined += 1;
                }
            }
            Ok(st) => {
                for spec in [false, true] {
                    let got = run_stage_with(&st, res, spec);
                    let tag = if spec { format!("{mode}+spec") } else { mode.to_string() };
                    match got.iter().zip(&want).enumerate().find_map(|(i, (g, w))| diff(g, w).map(|d| (i, d))) {
                        None => line.push_str(&format!("  {tag}: ok")),
                        Some((i, d)) => {
                            ok = false;
                            line.push_str(&format!("  {tag}: MISMATCH at vertex {i}: {d}"));
                        }
                    }
                }
            }
        }
    }
    if !interp {
        line.push_str("  (vs interp: n/a)");
    }
    (ok, line, declined)
}

/// Random snippet generator (deterministic xorshift): float expressions over
/// the inputs `p`, `q`, `u` and `vi`, `var`s, if/else on comparisons, short
/// `for` loops.
pub struct Gen {
    pub x: u64,
}

impl Gen {
    fn next(&mut self) -> u64 {
        self.x ^= self.x << 13;
        self.x ^= self.x >> 7;
        self.x ^= self.x << 17;
        self.x
    }
    fn pick(&mut self, n: usize) -> usize {
        (self.next() >> 20) as usize % n
    }
    fn leaf(&mut self, nv: usize, in_loop: bool) -> String {
        match self.pick(if in_loop { 8 } else { 7 }) {
            0 => ["p.x", "p.y", "p.z", "p.w"][self.pick(4)].into(),
            1 => ["q.x", "q.y", "q.z", "q.w"][self.pick(4)].into(),
            2 | 3 => ["u.x", "u.y", "u.z", "u.w"][self.pick(4)].into(),
            4 => "f32(vi)".into(),
            5 => format!("{:.2}", (self.pick(400) as f32 - 200.0) / 8.0),
            6 if nv > 0 => format!("v{}", self.pick(nv)),
            6 => "1.5".into(),
            _ => "f32(i)".into(),
        }
    }
    fn fexpr(&mut self, d: u32, nv: usize, in_loop: bool) -> String {
        if d == 0 || self.pick(5) == 0 {
            return self.leaf(nv, in_loop);
        }
        let a = self.fexpr(d - 1, nv, in_loop);
        let b = self.fexpr(d - 1, nv, in_loop);
        match self.pick(14) {
            0 => format!("({a} + {b})"),
            1 => format!("({a} - {b})"),
            2 => format!("({a} * {b})"),
            3 => format!("({a} / (abs({b}) + 1.0))"),
            4 => format!("min({a}, {b})"),
            5 => format!("max({a}, {b})"),
            6 => format!("abs({a})"),
            7 => format!("floor({a})"),
            8 => {
                let c = self.cond(d - 1, nv, in_loop);
                format!("select({a}, {b}, {c})")
            }
            9 => format!("sqrt(abs({a}))"),
            10 => format!("pow(abs({a}) + 0.25, 1.5)"),
            11 => format!("(0.0 - {a})"),
            12 => format!("trunc({a})"),
            _ => format!("clamp({a}, -10.0, 10.0)"),
        }
    }
    fn cond(&mut self, d: u32, nv: usize, in_loop: bool) -> String {
        let a = self.fexpr(d.saturating_sub(1), nv, in_loop);
        let b = self.fexpr(d.saturating_sub(1), nv, in_loop);
        match self.pick(if d == 0 { 4 } else { 7 }) {
            0 => format!("{a} < {b}"),
            1 => format!("{a} > {b}"),
            2 => format!("{a} >= {b}"),
            3 => format!("u.x > {:.2}", self.pick(30) as f32 / 10.0),
            4 => format!("({} && {})", self.cond(d - 1, nv, in_loop), self.cond(d - 1, nv, in_loop)),
            5 => format!("({} || {})", self.cond(d - 1, nv, in_loop), self.cond(d - 1, nv, in_loop)),
            _ => format!("!({})", self.cond(d - 1, nv, in_loop)),
        }
    }
    fn stmts(&mut self, out: &mut String, nv: usize, depth: u32, in_loop: bool) {
        for _ in 0..1 + self.pick(3) {
            let k = self.pick(nv);
            match self.pick(if depth == 0 { 3 } else { 6 }) {
                0 | 1 => {
                    let e = self.fexpr(3, nv, in_loop);
                    out.push_str(&format!("        v{k} = {e};\n"));
                }
                2 => {
                    let e = self.fexpr(2, nv, in_loop);
                    out.push_str(&format!("        v{k} = v{k} + {e};\n"));
                }
                3 | 4 => {
                    let c = self.cond(2, nv, in_loop);
                    out.push_str(&format!("        if ({c}) {{\n"));
                    self.stmts(out, nv, depth - 1, in_loop);
                    if self.pick(2) == 0 {
                        out.push_str("        } else {\n");
                        self.stmts(out, nv, depth - 1, in_loop);
                    }
                    out.push_str("        }\n");
                }
                _ if !in_loop => {
                    let n = 1 + self.pick(4);
                    out.push_str(&format!("        for (var i: u32 = 0u; i < {n}u; i = i + 1u) {{\n"));
                    self.stmts(out, nv, depth - 1, true);
                    out.push_str("        }\n");
                }
                _ => {}
            }
        }
    }
    /// `var v0..v5: f32` with random initializers and statements over them
    /// (reading `p`, `q`, `u`, `vi` as in the vertex snippets)
    pub fn vars(&mut self) -> String {
        let mut s = String::new();
        const NV: usize = 6;
        for k in 0..NV {
            let e = self.fexpr(2, k, false);
            s.push_str(&format!("        var v{k}: f32 = {e};\n"));
        }
        self.stmts(&mut s, NV, 3, false);
        s
    }

    fn body(&mut self) -> String {
        let mut s = self.vars();
        s.push_str(
            "        let pos = vec4<f32>(v0, v1, v2, v3);
        let a = vec4<u32>(u32(abs(v4)), u32(abs(v5)), vi, 0u);
        let b = vec4<f32>(v4, v5, v0 + v1, 1.0);",
        );
        s
    }
}

pub fn run() -> i32 {
    let data = data_bytes();
    let uni: Vec<u8> = [1.0f32, 2.0, 3.0, 0.5].iter().flat_map(|f| f.to_le_bytes()).collect();
    let res = Resources::default().with_buffer(0, 0, &data).with_buffer(0, 1, &uni);

    let have_jit = cfg!(all(target_arch = "x86_64", target_os = "linux"));
    println!("exec-selftest: {} snippets, {N_VERTS} vertices each, jit available: {have_jit}", CASES.len());

    let mut failures = 0;
    let mut declined = 0;
    for c in CASES {
        let (ok, line, d) = run_case(c.name, c.body, c.interp, &res, have_jit);
        declined += d;
        println!("{} {line}", if ok { "ok  " } else { "FAIL" });
        if !ok {
            failures += 1;
        }
    }
    // random shaders: control flow on uniform- and input-dependent conditions,
    // loops, mixed ops — through the optimizer, specialization, both JITs
    let nfuzz: u32 = std::env::var("AKUMA_FUZZ").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    let seed: u64 = std::env::var("AKUMA_FUZZ_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0x5EED);
    let mut fuzz_bad = 0;
    for k in 0..nfuzz {
        let mut g = Gen { x: seed.wrapping_add(k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 };
        let body = g.body();
        let name = format!("fuzz #{k} (seed {:#x})", seed.wrapping_add(k as u64));
        let (ok, line, d) = run_case(&name, &body, true, &res, have_jit);
        declined += d;
        if !ok {
            fuzz_bad += 1;
            failures += 1;
            println!("FAIL {line}\n{body}");
        }
    }
    println!("fuzz: {nfuzz} random shaders, {fuzz_bad} mismatching");
    if failures == 0 {
        println!("EXEC-SELFTEST OK ({declined} declined)");
        0
    } else {
        println!("EXEC-SELFTEST FAILED: {failures} snippet(s)");
        1
    }
}

/// `akuma-wgpu shader-check <file.wgsl>...`: for every entry point of every
/// file, report whether the lowering accepts it, whether it JITs, and why not.
pub fn shader_check(files: &[String]) -> i32 {
    let mut bad = 0;
    for f in files {
        let src = match std::fs::read_to_string(f) {
            Ok(s) => s,
            Err(e) => {
                println!("{f}: {e}");
                bad += 1;
                continue;
            }
        };
        let sh = match Shader::parse(&src) {
            Ok(s) => Arc::new(s),
            Err(e) => {
                println!("{f}: does not parse/validate: {e}");
                bad += 1;
                continue;
            }
        };
        for (i, ep) in sh.module.entry_points.iter().enumerate() {
            let verdict = match super::compile::compile(&sh, i) {
                Err(e) => {
                    bad += 1;
                    format!("DECLINED: {e}")
                }
                Ok(p) => {
                    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
                    let j = match super::jit::compile(&p) {
                        Ok(j) => format!("jit {} B", j.code_len()),
                        Err(e) => format!("jit declined: {e}"),
                    };
                    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
                    let j = "no jit on this target".to_string();
                    // instruction mix: what the per-invocation cost is made of
                    let mut calls = std::collections::BTreeMap::<String, u32>::new();
                    let (mut loads, mut tex, mut jumps) = (0, 0, 0);
                    for i in &p.code {
                        match i {
                            super::program::Inst::Call { f, .. } => *calls.entry(format!("{f:?}")).or_default() += 1,
                            super::program::Inst::LoadBuf { .. } => loads += 1,
                            super::program::Inst::Tex { .. } => tex += 1,
                            super::program::Inst::Jmp { .. } | super::program::Inst::Jz { .. } | super::program::Inst::Jnz { .. } => jumps += 1,
                            _ => {}
                        }
                    }
                    let ncalls: u32 = calls.values().sum();
                    if std::env::var("AKUMA_DUMP").is_ok_and(|n| n == ep.name) {
                        println!("  inputs {:?}\n  outputs {:?}", p.inputs, p.outputs);
                        for (k, i) in p.code.iter().enumerate() {
                            println!("  {k:4}: {i:?}");
                        }
                    }
                    if std::env::var_os("AKUMA_MIX").is_some() {
                        let mut mix = std::collections::BTreeMap::<String, u32>::new();
                        for i in &p.code {
                            let n = format!("{i:?}");
                            *mix.entry(n.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("").to_string()).or_default() += 1;
                        }
                        println!("  mix: {mix:?}");
                    }
                    format!(
                        "ok: {} insts ({ncalls} helper calls {calls:?}, {loads} buf loads, {tex} tex, {jumps} jumps), {} regs, {} buffers, {j}",
                        p.code.len(), p.nregs, p.bufs.len()
                    )
                }
            };
            println!("{f}: {:?} {:<22} {verdict}", ep.stage, ep.name);
        }
    }
    if bad == 0 { 0 } else { 1 }
}
