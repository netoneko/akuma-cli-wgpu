//! The akuma GPU's standard rasterization path: what a real wgpu client
//! (rio/sugarloaf) expects, as opposed to the demo's legacy raw-bytes contract
//! in `backend.rs::raster_tri`.
//!
//! Pipeline per triangle:
//!
//!   clip-space vertices
//!     → clip against near/far (0 ≤ z ≤ w) and a generous x/y guard band
//!       (only when something is outside; the common case skips it)
//!     → perspective divide, viewport transform (y flipped: framebuffer y is down)
//!     → snap to a 1/256-pixel fixed-point grid
//!     → cull by winding
//!     → edge-function fill at pixel centers with the **top-left rule**, in
//!       exact i64 arithmetic: two triangles sharing an edge cover every pixel
//!       centre on it exactly once (no cracks, no double-blend)
//!     → per-pixel varyings: flat (provoking vertex = first), linear, or
//!       perspective-correct
//!     → depth test, fragment stage, blend, color write mask, encode to format

use wgpu::{BlendFactor, BlendOperation, CompareFunction, Face, FrontFace};

use super::exec::{Invoker, RawVertex, Varyings, MAX_LOC};
use super::format;
use super::program::Interp;

/// sub-pixel precision: coordinates are snapped to 1/256 pixel
const SUB: i64 = 256;
/// clip x/y to this many times w (NDC ±GUARD): keeps snapped coordinates
/// (|px| ≲ GUARD*8192 at 8K) comfortably inside the i64 edge-function range
const GUARD: f64 = 8.0;

#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

pub struct ColorTarget<'a> {
    pub format: wgpu::TextureFormat,
    pub data: &'a mut [u8],
    pub width: u32,
    pub height: u32,
    pub blend: Option<wgpu::BlendState>,
    pub write_mask: wgpu::ColorWrites,
}

pub struct DepthTarget<'a> {
    pub data: &'a mut [f32],
    pub compare: CompareFunction,
    pub write: bool,
}

pub struct Raster<'a> {
    pub color: ColorTarget<'a>,
    pub depth: Option<DepthTarget<'a>>,
    pub viewport: Viewport,
    /// x, y, w, h in pixels
    pub scissor: [u32; 4],
    pub cull: Option<Face>,
    pub front: FrontFace,
    pub blend_constant: [f32; 4],
    /// fragment-stage input locations and how each varies
    pub interp: Vec<(u32, Interp)>,
}

#[derive(Clone, Copy)]
struct CV {
    pos: [f64; 4],
    var: [[u32; 4]; MAX_LOC],
}

/// a vertex after the viewport transform
#[derive(Clone, Copy)]
struct SV {
    x: i64,
    y: i64,
    z: f64,
    invw: f64,
    var: [[u32; 4]; MAX_LOC],
}

impl Raster<'_> {
    /// Rasterize one triangle given its clip-space vertices. `provoking` is
    /// the varyings of the primitive's first vertex (the flat source) — for
    /// odd triangles of a strip, which are emitted reversed to keep their
    /// winding, that is not `v[0]`.
    pub fn triangle(&mut self, fs: &mut Invoker<'_>, v: [&RawVertex; 3], provoking: &Varyings) {
        let cv = |r: &RawVertex| CV {
            pos: [
                r.position[0] as f64,
                r.position[1] as f64,
                r.position[2] as f64,
                r.position[3] as f64,
            ],
            var: r.varyings,
        };
        let verts = [cv(v[0]), cv(v[1]), cv(v[2])];
        if verts.iter().any(|p| p.pos.iter().any(|c| !c.is_finite())) {
            return;
        }
        let inside_all = verts.iter().all(|p| {
            let [x, y, z, w] = p.pos;
            w > 1e-12 && z >= 0.0 && z <= w && x.abs() <= GUARD * w && y.abs() <= GUARD * w
        });
        if inside_all {
            self.fill(fs, verts, provoking);
            return;
        }
        let poly = clip(&verts, &self.interp);
        if poly.len() < 3 {
            return;
        }
        for i in 1..poly.len() - 1 {
            self.fill(fs, [poly[0], poly[i], poly[i + 1]], provoking);
        }
    }

    fn to_screen(&self, p: &CV) -> SV {
        let [x, y, z, w] = p.pos;
        let invw = 1.0 / w;
        let (nx, ny, nz) = (x * invw, y * invw, z * invw);
        let vp = &self.viewport;
        let sx = vp.x as f64 + (nx * 0.5 + 0.5) * vp.w as f64;
        let sy = vp.y as f64 + (0.5 - ny * 0.5) * vp.h as f64;
        let sz = vp.min_depth as f64 + nz * (vp.max_depth - vp.min_depth) as f64;
        SV {
            x: (sx * SUB as f64).round() as i64,
            y: (sy * SUB as f64).round() as i64,
            z: sz,
            invw,
            var: p.var,
        }
    }

    fn fill(&mut self, fs: &mut Invoker<'_>, tri: [CV; 3], provoking: &Varyings) {
        let mut s = [self.to_screen(&tri[0]), self.to_screen(&tri[1]), self.to_screen(&tri[2])];
        // signed area in y-down framebuffer space; visually counter-clockwise
        // triangles come out negative
        let mut area = (s[1].x - s[0].x) * (s[2].y - s[0].y) - (s[1].y - s[0].y) * (s[2].x - s[0].x);
        if area == 0 {
            return;
        }
        let ccw_visual = area < 0;
        let front_facing = match self.front {
            FrontFace::Ccw => ccw_visual,
            FrontFace::Cw => !ccw_visual,
        };
        match self.cull {
            Some(Face::Front) if front_facing => return,
            Some(Face::Back) if !front_facing => return,
            _ => {}
        }
        if area > 0 {
            // normalize to the visually-CCW orientation the edge tests assume
            s.swap(1, 2);
            area = -area;
        }

        // bounding box in whole pixels, clipped to scissor, viewport and
        // target. The guard-band clip lets geometry run past NDC +-1, so the
        // viewport rectangle has to bound the fragments too: a pixel is in
        // iff its centre is inside the rectangle.
        let sc = self.scissor;
        let vp = &self.viewport;
        let vp_x0 = (vp.x as f64 - 0.5).ceil() as i64;
        let vp_x1 = ((vp.x + vp.w) as f64 - 0.5).ceil() as i64;
        let vp_y0 = (vp.y as f64 - 0.5).ceil() as i64;
        let vp_y1 = ((vp.y + vp.h) as f64 - 0.5).ceil() as i64;
        let min_x = (s.iter().map(|p| p.x).min().unwrap() - SUB / 2)
            .div_euclid(SUB)
            .max(sc[0] as i64)
            .max(vp_x0);
        let max_x = ((s.iter().map(|p| p.x).max().unwrap() + SUB / 2).div_euclid(SUB) + 1)
            .min((sc[0] + sc[2]) as i64)
            .min(vp_x1);
        let min_y = (s.iter().map(|p| p.y).min().unwrap() - SUB / 2)
            .div_euclid(SUB)
            .max(sc[1] as i64)
            .max(vp_y0);
        let max_y = ((s.iter().map(|p| p.y).max().unwrap() + SUB / 2).div_euclid(SUB) + 1)
            .min((sc[1] + sc[3]) as i64)
            .min(vp_y1);
        let (min_x, min_y) = (min_x.max(0), min_y.max(0));
        if min_x >= max_x || min_y >= max_y {
            return;
        }

        // edge i is opposite vertex i: e0 = v1->v2, e1 = v2->v0, e2 = v0->v1
        // w_i(p) = (b.y-a.y)*(p.x-a.x) - (b.x-a.x)*(p.y-a.y)  (>= 0 inside)
        let edges = [(1usize, 2usize), (2, 0), (0, 1)];
        let mut dwdx = [0i64; 3];
        let mut dwdy = [0i64; 3];
        let mut w_row = [0i64; 3];
        let mut tl = [false; 3];
        let px0 = min_x * SUB + SUB / 2;
        let py0 = min_y * SUB + SUB / 2;
        for (i, &(a, b)) in edges.iter().enumerate() {
            let dx = s[b].x - s[a].x;
            let dy = s[b].y - s[a].y;
            dwdx[i] = dy * SUB;
            dwdy[i] = -dx * SUB;
            w_row[i] = dy * (px0 - s[a].x) - dx * (py0 - s[a].y);
            tl[i] = dy > 0 || (dy == 0 && dx < 0);
        }
        let inv_sum = 1.0 / (-area) as f64;

        let (cw, ch) = (self.color.width as i64, self.color.height as i64);
        let (max_x, max_y) = (max_x.min(cw), max_y.min(ch));
        let flat_all = self.interp.iter().all(|(_, m)| *m == Interp::Flat);
        let mut var: Varyings = *provoking;

        for py in min_y..max_y {
            let mut w = w_row;
            for px in min_x..max_x {
                let inside = (w[0] > 0 || (w[0] == 0 && tl[0]))
                    && (w[1] > 0 || (w[1] == 0 && tl[1]))
                    && (w[2] > 0 || (w[2] == 0 && tl[2]));
                if inside {
                    // barycentric weights (screen space)
                    let l = [
                        w[0] as f64 * inv_sum,
                        w[1] as f64 * inv_sum,
                        w[2] as f64 * inv_sum,
                    ];
                    let z = l[0] * s[0].z + l[1] * s[1].z + l[2] * s[2].z;
                    let invw = l[0] * s[0].invw + l[1] * s[1].invw + l[2] * s[2].invw;
                    let zf = z as f32;
                    let idx = (py * cw + px) as usize;
                    let pass = match &self.depth {
                        Some(d) => compare(d.compare, zf, d.data[idx]),
                        None => true,
                    };
                    if pass {
                        if !flat_all {
                            self.interpolate(&mut var, provoking, &s, &l, invw);
                        }
                        if let Some(c) = fs.run_fragment(&var, [px as f32 + 0.5, py as f32 + 0.5, zf, invw as f32]) {
                            if let Some(d) = &mut self.depth {
                                if d.write {
                                    d.data[idx] = zf;
                                }
                            }
                            self.write_pixel(idx, c);
                        }
                    }
                }
                for i in 0..3 {
                    w[i] += dwdx[i];
                }
            }
            for i in 0..3 {
                w_row[i] += dwdy[i];
            }
        }
    }

    fn interpolate(
        &self,
        out: &mut Varyings,
        provoking: &Varyings,
        s: &[SV; 3],
        l: &[f64; 3],
        invw: f64,
    ) {
        let persp = [
            l[0] * s[0].invw / invw,
            l[1] * s[1].invw / invw,
            l[2] * s[2].invw / invw,
        ];
        for &(loc, mode) in &self.interp {
            let loc = loc as usize;
            match mode {
                Interp::Flat => out[loc] = provoking[loc],
                Interp::Linear | Interp::Perspective => {
                    let k = if mode == Interp::Linear { l } else { &persp };
                    for c in 0..4 {
                        let v = k[0] * f32::from_bits(s[0].var[loc][c]) as f64
                            + k[1] * f32::from_bits(s[1].var[loc][c]) as f64
                            + k[2] * f32::from_bits(s[2].var[loc][c]) as f64;
                        out[loc][c] = (v as f32).to_bits();
                    }
                }
            }
        }
    }

    fn write_pixel(&mut self, idx: usize, frag: [u32; 4]) {
        let fmt = self.color.format;
        let bpt = format::bytes_per_texel(fmt).unwrap() as usize;
        let texel = &mut self.color.data[idx * bpt..(idx + 1) * bpt];
        let src = [
            f32::from_bits(frag[0]),
            f32::from_bits(frag[1]),
            f32::from_bits(frag[2]),
            f32::from_bits(frag[3]),
        ];
        let mut out = src;
        let wm = self.color.write_mask;
        let needs_dst = self.color.blend.is_some() || wm != wgpu::ColorWrites::ALL;
        let dst = if needs_dst { format::decode(fmt, texel) } else { [0.0; 4] };
        if let Some(b) = &self.color.blend {
            out = blend(b, src, dst, self.blend_constant);
        }
        if wm != wgpu::ColorWrites::ALL {
            let masks = [
                wm.contains(wgpu::ColorWrites::RED),
                wm.contains(wgpu::ColorWrites::GREEN),
                wm.contains(wgpu::ColorWrites::BLUE),
                wm.contains(wgpu::ColorWrites::ALPHA),
            ];
            for i in 0..4 {
                if !masks[i] {
                    out[i] = dst[i];
                }
            }
        }
        format::encode(fmt, out, texel);
    }
}

fn compare(f: CompareFunction, a: f32, b: f32) -> bool {
    match f {
        CompareFunction::Never => false,
        CompareFunction::Less => a < b,
        CompareFunction::Equal => a == b,
        CompareFunction::LessEqual => a <= b,
        CompareFunction::Greater => a > b,
        CompareFunction::NotEqual => a != b,
        CompareFunction::GreaterEqual => a >= b,
        CompareFunction::Always => true,
    }
}

// ---------------------------------------------------------------------------
// blending
// ---------------------------------------------------------------------------

fn blend(b: &wgpu::BlendState, src: [f32; 4], dst: [f32; 4], k: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for c in 0..3 {
        let sf = factor(b.color.src_factor, src, dst, k, c);
        let df = factor(b.color.dst_factor, src, dst, k, c);
        out[c] = op(b.color.operation, src[c], sf, dst[c], df);
    }
    let sf = factor(b.alpha.src_factor, src, dst, k, 3);
    let df = factor(b.alpha.dst_factor, src, dst, k, 3);
    out[3] = op(b.alpha.operation, src[3], sf, dst[3], df);
    out
}

/// blend factor for channel `c` (0..=2 color, 3 alpha); color-based factors
/// use the alpha value when blending the alpha channel
fn factor(f: BlendFactor, src: [f32; 4], dst: [f32; 4], k: [f32; 4], c: usize) -> f32 {
    match f {
        BlendFactor::Zero => 0.0,
        BlendFactor::One => 1.0,
        BlendFactor::Src => src[c],
        BlendFactor::OneMinusSrc => 1.0 - src[c],
        BlendFactor::SrcAlpha => src[3],
        BlendFactor::OneMinusSrcAlpha => 1.0 - src[3],
        BlendFactor::Dst => dst[c],
        BlendFactor::OneMinusDst => 1.0 - dst[c],
        BlendFactor::DstAlpha => dst[3],
        BlendFactor::OneMinusDstAlpha => 1.0 - dst[3],
        BlendFactor::SrcAlphaSaturated => {
            if c == 3 { 1.0 } else { src[3].min(1.0 - dst[3]) }
        }
        BlendFactor::Constant => k[c],
        BlendFactor::OneMinusConstant => 1.0 - k[c],
        other => panic!("akuma backend: blend factor {other:?} unsupported"),
    }
}

fn op(o: BlendOperation, s: f32, sf: f32, d: f32, df: f32) -> f32 {
    match o {
        BlendOperation::Add => s * sf + d * df,
        BlendOperation::Subtract => s * sf - d * df,
        BlendOperation::ReverseSubtract => d * df - s * sf,
        BlendOperation::Min => s.min(d),
        BlendOperation::Max => s.max(d),
    }
}

// ---------------------------------------------------------------------------
// clipping (Sutherland–Hodgman against 0<=z<=w and the x/y guard band)
// ---------------------------------------------------------------------------

/// plane value: >= 0 means inside
fn plane(i: usize, p: &[f64; 4]) -> f64 {
    let [x, y, z, w] = *p;
    match i {
        0 => z,
        1 => w - z,
        2 => GUARD * w - x,
        3 => GUARD * w + x,
        4 => GUARD * w - y,
        _ => GUARD * w + y,
    }
}

fn lerp_cv(a: &CV, b: &CV, t: f64, interp: &[(u32, Interp)]) -> CV {
    let mut pos = [0.0; 4];
    for i in 0..4 {
        pos[i] = a.pos[i] + (b.pos[i] - a.pos[i]) * t;
    }
    // flat varyings keep the first vertex's bits; the others interpolate
    // linearly in clip space (which is what perspective-correctness needs)
    let mut var = a.var;
    for &(loc, mode) in interp {
        if mode == Interp::Flat {
            continue;
        }
        let loc = loc as usize;
        for c in 0..4 {
            let (x, y) = (f32::from_bits(a.var[loc][c]) as f64, f32::from_bits(b.var[loc][c]) as f64);
            var[loc][c] = ((x + (y - x) * t) as f32).to_bits();
        }
    }
    CV { pos, var }
}

fn clip(tri: &[CV; 3], interp: &[(u32, Interp)]) -> Vec<CV> {
    let mut poly: Vec<CV> = tri.to_vec();
    for pl in 0..6 {
        if poly.is_empty() {
            break;
        }
        let mut out = Vec::with_capacity(poly.len() + 1);
        for i in 0..poly.len() {
            let a = &poly[i];
            let b = &poly[(i + 1) % poly.len()];
            let (da, db) = (plane(pl, &a.pos), plane(pl, &b.pos));
            if da >= 0.0 {
                out.push(*a);
            }
            if (da >= 0.0) != (db >= 0.0) {
                let t = da / (da - db);
                out.push(lerp_cv(a, b, t, interp));
            }
        }
        poly = out;
    }
    // a w <= 0 survivor would divide by ~0; the near plane (z >= 0, z <= w)
    // already forces w >= 0, drop degenerate leftovers
    poly.retain(|p| p.pos[3] > 1e-12);
    poly
}
