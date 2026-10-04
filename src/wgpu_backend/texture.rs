//! Texture fetch and sampling for shader stages.
//!
//! Compiled stages call `tex_helper` (from the VM directly, from JIT code
//! through a plain `extern "C"` call) for `textureLoad`, `textureSample*` and
//! `textureDimensions`. The akuma GPU has one mip level per texture, so every
//! sample is a level-0 sample: nearest or bilinear per the sampler, with the
//! sampler's address modes. Float formats are decoded to f32 *before*
//! filtering (sRGB textures filter in linear space, like real hardware).

use wgpu::{AddressMode, FilterMode};

use super::format;
use super::program::R;

/// One bound texture as the helper sees it. Plain data; lives in a per-draw
/// table whose address is handed to the stage.
#[derive(Clone, Copy)]
pub struct TexRef {
    pub data: *const u8,
    pub len: usize,
    pub w: u32,
    pub h: u32,
    pub format: wgpu::TextureFormat,
}

// the pointer targets texture storage that the draw keeps locked and
// read-only for its whole duration; worker threads only read through it
unsafe impl Send for TexRef {}
unsafe impl Sync for TexRef {}

impl TexRef {
    pub const EMPTY: TexRef = TexRef {
        data: std::ptr::null(),
        len: 0,
        w: 0,
        h: 0,
        format: wgpu::TextureFormat::Rgba8Unorm,
    };
}

#[derive(Clone, Copy, Debug)]
pub struct SmpRef {
    pub mag: FilterMode,
    pub min: FilterMode,
    pub addr_u: AddressMode,
    pub addr_v: AddressMode,
}

impl SmpRef {
    pub const DEFAULT: SmpRef = SmpRef {
        mag: FilterMode::Nearest,
        min: FilterMode::Nearest,
        addr_u: AddressMode::ClampToEdge,
        addr_v: AddressMode::ClampToEdge,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexKind {
    /// integer coordinates in x/y, no sampler
    Load,
    /// normalized float coordinates in x/y
    Sample,
    /// writes width, height to d, d+1
    Size,
}

/// What the shader's element type is (sets the result registers' meaning)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    F,
    U,
    I,
}

#[derive(Clone, Copy, Debug)]
pub struct TexOp {
    pub kind: TexKind,
    pub tex: u32,
    /// sampler slot (`u32::MAX` for none)
    pub smp: u32,
    pub x: R,
    pub y: R,
    /// first of 4 consecutive result registers (2 for Size)
    pub d: R,
    pub elem: Elem,
    /// coordinates are signed ints (Load only)
    pub signed: bool,
}

/// # Safety
/// `regs` must point at the invocation's register file, `texs`/`smps` at the
/// draw's tables (indexable by every slot this program uses), `op` at a live
/// `TexOp`.
pub unsafe extern "C" fn tex_helper(
    regs: *mut u32,
    texs: *const TexRef,
    smps: *const SmpRef,
    op: *const TexOp,
    stride: usize,
) {
    let op = unsafe { &*op };
    let t = unsafe { *texs.add(op.tex as usize) };
    // register r of this lane is at regs[r * stride] (scalar code: stride 1;
    // wide code passes the lane's base pointer and stride 4)
    let reg = |r: R| unsafe { *regs.add(r as usize * stride) };
    let mut put = |i: u32, v: u32| unsafe { *regs.add((op.d + i) as usize * stride) = v };
    match op.kind {
        TexKind::Size => {
            put(0, t.w);
            put(1, t.h);
        }
        TexKind::Load => {
            let (x, y) = if op.signed {
                (reg(op.x) as i32 as i64, reg(op.y) as i32 as i64)
            } else {
                (reg(op.x) as i64, reg(op.y) as i64)
            };
            let out = if x < 0 || y < 0 || x >= t.w as i64 || y >= t.h as i64 || t.data.is_null() {
                [0u32; 4]
            } else {
                let px = texel(&t, x as u32, y as u32);
                match op.elem {
                    Elem::F => {
                        let c = format::decode(t.format, px);
                        [c[0].to_bits(), c[1].to_bits(), c[2].to_bits(), c[3].to_bits()]
                    }
                    Elem::U | Elem::I => format::decode_uint(t.format, px),
                }
            };
            for (i, v) in out.iter().enumerate() {
                put(i as u32, *v);
            }
        }
        TexKind::Sample => {
            let s = if op.smp == u32::MAX { SmpRef::DEFAULT } else { unsafe { *smps.add(op.smp as usize) } };
            let (u, v) = (f32::from_bits(reg(op.x)), f32::from_bits(reg(op.y)));
            let c = sample(&t, &s, u, v);
            for (i, f) in c.iter().enumerate() {
                put(i as u32, f.to_bits());
            }
        }
    }
}

#[inline]
fn texel(t: &TexRef, x: u32, y: u32) -> &[u8] {
    let bpt = format::bytes_per_texel(t.format).unwrap() as usize;
    let at = (y as usize * t.w as usize + x as usize) * bpt;
    debug_assert!(at + bpt <= t.len);
    unsafe { std::slice::from_raw_parts(t.data.add(at), bpt) }
}

/// map a texel index through an address mode: Some(index) or None for the
/// transparent-black border
fn address(mode: AddressMode, i: i64, n: i64) -> Option<i64> {
    match mode {
        AddressMode::ClampToEdge => Some(i.clamp(0, n - 1)),
        AddressMode::Repeat => Some(i.rem_euclid(n)),
        AddressMode::MirrorRepeat => {
            let m = i.rem_euclid(2 * n);
            Some(if m < n { m } else { 2 * n - 1 - m })
        }
        AddressMode::ClampToBorder => (0..n).contains(&i).then_some(i),
    }
}

fn fetch(t: &TexRef, s: &SmpRef, xi: i64, yi: i64) -> [f32; 4] {
    match (address(s.addr_u, xi, t.w as i64), address(s.addr_v, yi, t.h as i64)) {
        (Some(x), Some(y)) if !t.data.is_null() => format::decode(t.format, texel(t, x as u32, y as u32)),
        _ => [0.0; 4],
    }
}

fn sample(t: &TexRef, s: &SmpRef, u: f32, v: f32) -> [f32; 4] {
    if t.w == 0 || t.h == 0 {
        return [0.0; 4];
    }
    let (fx, fy) = (u as f64 * t.w as f64, v as f64 * t.h as f64);
    if s.mag == FilterMode::Nearest {
        return fetch(t, s, fx.floor() as i64, fy.floor() as i64);
    }
    // bilinear: texel centres sit at +0.5
    let (px, py) = (fx - 0.5, fy - 0.5);
    let (x0, y0) = (px.floor(), py.floor());
    let (ax, ay) = ((px - x0) as f32, (py - y0) as f32);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let c00 = fetch(t, s, x0, y0);
    let c10 = fetch(t, s, x0 + 1, y0);
    let c01 = fetch(t, s, x0, y0 + 1);
    let c11 = fetch(t, s, x0 + 1, y0 + 1);
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        let top = c00[i] * (1.0 - ax) + c10[i] * ax;
        let bot = c01[i] * (1.0 - ax) + c11[i] * ax;
        out[i] = top * (1.0 - ay) + bot * ay;
    }
    out
}
