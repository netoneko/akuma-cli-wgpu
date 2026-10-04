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

/// pixels produced by run replication (tests check the mechanism is exercised)
pub static RUN_PIXELS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    /// rows [band.0, band.1) this rasterizer may touch (for tile-parallel
    /// draws; the whole target otherwise) and the target row that
    /// `color.data` / `depth.data` index 0 corresponds to
    pub band: (i64, i64),
    pub row0: i64,
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
        let (min_x, min_y) = (min_x.max(0), min_y.max(self.band.0));
        let max_y = max_y.min(self.band.1);
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
        // position-independent flat shader: one evaluation per triangle
        let constant_fs = fs.constant_per_primitive();
        let mut cached: Option<Option<[u32; 4]>> = None;
        // barycentrics, depth and 1/w only matter if something consumes them
        let need_geom = self.depth.is_some() || !flat_all || fs.uses_position_zw();
        let plan = PixelPlan::new(&self.color, self.blend_constant);
        // with a constant fragment result the final texel (or the blend
        // factors) can be computed once per triangle
        let mut const_texel: Option<[u8; 4]> = None;
        let mut const_blend: Option<([f32; 4], [f32; 4])> = None;
        // wide stages shade up to `lanes` pixels per call: covered pixels are
        // gathered here and flushed when full and at the end of the triangle
        let lanes = if constant_fs { 1 } else { fs.lanes() };
        let mut pend = [Pending { idx: 0, zf: 0.0, pos: [0.0; 4] }; 4];
        // per-pixel varyings, only for stages with non-flat inputs
        let mut pend_var = [[[0u32; 4]; MAX_LOC]; 4];
        let mut np = 0usize;

        // edge i admits a pixel iff w_i >= thr_i (strict inside, or on a
        // top/left edge)
        let thr = [!tl[0] as i64, !tl[1] as i64, !tl[2] as i64];
        let width = max_x - min_x;
        // the straight-line case: flat varyings, nothing reading depth or
        // barycentrics, a 4-lane stage, a plain unorm8 target
        let span_fast = !constant_fs
            && self.depth.is_none()
            && lanes == 4
            && plan.unorm8.is_some()
            && !fs.uses_position_zw();
        let geo = SpanGeo { s: &s, dwdx, inv_sum };
        let tri_planes = if span_fast && !flat_all {
            let mut locs = [(0u32, 0u32); 32];
            let nl = fs.span_locs(&mut locs);
            let mut tp = TriPlanes { n: nl, p: [TriPlane::Const(0); 32] };
            let ortho = s[0].invw == s[1].invw && s[1].invw == s[2].invw;
            // edge functions at the bounding-box corner (min_x, min_y)
            let w00 = w_row;
            for i in 0..nl {
                let (loc, comp) = (locs[i].0 as usize, locs[i].1 as usize);
                let mode = self.interp.iter().find(|(x, _)| *x as usize == loc).map_or(Interp::Flat, |(_, m)| *m);
                tp.p[i] = match mode {
                    Interp::Flat => TriPlane::Const(provoking[loc][comp]),
                    _ => {
                        let v = [
                            f32::from_bits(s[0].var[loc][comp]) as f64,
                            f32::from_bits(s[1].var[loc][comp]) as f64,
                            f32::from_bits(s[2].var[loc][comp]) as f64,
                        ];
                        // inv_sum * sum(w_i * g_i) for the origin value and the x / y slopes
                        let plane = |g: [f64; 3]| {
                            let f = |e: [i64; 3]| inv_sum * (e[0] as f64 * g[0] + e[1] as f64 * g[1] + e[2] as f64 * g[2]);
                            (f(w00), f(dwdx), f(dwdy))
                        };
                        if mode == Interp::Linear || ortho {
                            let (a, bx, by) = plane(v);
                            TriPlane::Lin { a, bx, by }
                        } else {
                            let iw = [s[0].invw, s[1].invw, s[2].invw];
                            let (a, bx, by) = plane([v[0] * iw[0], v[1] * iw[1], v[2] * iw[2]]);
                            let (da, dbx, dby) = plane(iw);
                            TriPlane::Persp { a, bx, by, da, dbx, dby }
                        }
                    }
                };
            }
            Some(tp)
        } else {
            None
        };

        let (runs_x, runs_y) = if span_fast { fs.has_runs() } else { (false, false) };
        // y runs: the previous row's covered interval when it was produced purely
        // by stores, and how many further rows share its quantized y values
        let mut prev_row: Option<(i64, i64)> = None;
        let mut y_budget = 0usize;
        for py in min_y..max_y {
            // the covered pixels of this row are one interval [lo, hi) of
            // offsets from min_x: each edge function is linear in x, so each
            // edge bounds the interval from one side
            let (mut lo, mut hi) = (0i64, width);
            for i in 0..3 {
                let (d, w0, t) = (dwdx[i], w_row[i], thr[i]);
                if d == 0 {
                    if w0 < t {
                        hi = 0;
                    }
                } else if d > 0 {
                    // w0 + d*k >= t  <=>  k >= ceil((t - w0) / d)
                    lo = lo.max(-((w0 - t).div_euclid(d)));
                } else {
                    // k <= floor((w0 - t) / -d)
                    hi = hi.min((w0 - t).div_euclid(-d) + 1);
                }
            }
            // a position-independent shader whose result simply replaces the
            // destination: fill the run
            let mut filled = false;
            if lo < hi && constant_fs && self.depth.is_none() && plan.bpt == 4 {
                let frag = match cached {
                    Some(c) => c,
                    None => {
                        let c = fs.run_fragment(&var, [(min_x + lo) as f32 + 0.5, py as f32 + 0.5, 0.0, 0.0]);
                        cached = Some(c);
                        c
                    }
                };
                match frag {
                    None => filled = true, // discarded
                    Some(col) => {
                        let src = [f32::from_bits(col[0]), f32::from_bits(col[1]), f32::from_bits(col[2]), f32::from_bits(col[3])];
                        let stores = (plan.blend.is_none() && plan.mask_all) || (plan.opaque_is_store && src[3] == 1.0);
                        if stores {
                            let t = *const_texel.get_or_insert_with(|| {
                                let mut t = [0u8; 4];
                                format::encode(plan.fmt, src, &mut t);
                                t
                            });
                            let base = ((py - self.row0) * cw + min_x + lo) as usize * 4;
                            let dst = &mut self.color.data[base..base + (hi - lo) as usize * 4];
                            for px in dst.chunks_exact_mut(4) {
                                px.copy_from_slice(&t);
                            }
                            filled = true;
                        }
                    }
                }
            }
            if filled {
                // nothing more for this row
                prev_row = None;
                y_budget = 0;
            } else if lo < hi && span_fast {
                let mut row = |this: &mut Self, fs: &mut Invoker<'_>, a: i64, b: i64| -> bool {
                    let w0 = [w_row[0] + dwdx[0] * a, w_row[1] + dwdx[1] * a, w_row[2] + dwdx[2] * a];
                    this.fast_span(
                        fs, &plan, &geo, tri_planes.as_ref(), provoking, flat_all, py, min_x + a, w0, (b - a) as usize,
                        (a, py - min_y), runs_x,
                    )
                };
                let mut copied = false;
                if runs_y && y_budget > 0 {
                    if let Some((plo, phi)) = prev_row {
                        let (olo, ohi) = (lo.max(plo), hi.min(phi));
                        if ohi - olo >= 16 {
                            // same quantized y as the row above: copy what overlaps
                            // it, shade the few pixels that do not
                            let mut pure = true;
                            if lo < olo {
                                pure &= row(self, fs, lo, olo);
                            }
                            let src = ((py - 1 - self.row0) * cw + min_x + olo) as usize * 4;
                            let dst = ((py - self.row0) * cw + min_x + olo) as usize * 4;
                            self.color.data.copy_within(src..src + (ohi - olo) as usize * 4, dst);
                            RUN_PIXELS.fetch_add((ohi - olo) as u64, std::sync::atomic::Ordering::Relaxed);
                            if ohi < hi {
                                pure &= row(self, fs, ohi, hi);
                            }
                            y_budget -= 1;
                            prev_row = pure.then_some((lo, hi));
                            if !pure {
                                y_budget = 0;
                            }
                            copied = true;
                        }
                    }
                }
                if !copied {
                    let pure = row(self, fs, lo, hi);
                    if runs_y && pure {
                        prev_row = Some((lo, hi));
                        y_budget = fs.y_extent(py, (max_y - py - 1) as usize);
                    } else {
                        prev_row = None;
                        y_budget = 0;
                    }
                }
            } else {
                prev_row = None;
                y_budget = 0;
                for k in lo..hi {
                    let px = min_x + k;
                    let w = [w_row[0] + dwdx[0] * k, w_row[1] + dwdx[1] * k, w_row[2] + dwdx[2] * k];
                    let (mut zf, mut invw, mut l) = (0.0f32, 0.0f64, [0.0f64; 3]);
                    if need_geom {
                        // barycentric weights (screen space)
                        l = [
                            w[0] as f64 * inv_sum,
                            w[1] as f64 * inv_sum,
                            w[2] as f64 * inv_sum,
                        ];
                        zf = (l[0] * s[0].z + l[1] * s[1].z + l[2] * s[2].z) as f32;
                        invw = l[0] * s[0].invw + l[1] * s[1].invw + l[2] * s[2].invw;
                    }
                    let idx = ((py - self.row0) * cw + px) as usize;
                    let pass = match &self.depth {
                        Some(d) => compare(d.compare, zf, d.data[idx]),
                        None => true,
                    };
                    if pass && lanes > 1 {
                        if !flat_all {
                            self.interpolate(&mut var, provoking, &s, &l, invw);
                        }
                        pend[np] = Pending {
                            idx,
                            zf,
                            pos: [px as f32 + 0.5, py as f32 + 0.5, zf, invw as f32],
                        };
                        // flat-only: every pixel shares the triangle's varyings
                        if !flat_all {
                            pend_var[np] = var;
                        }
                        np += 1;
                        if np == lanes {
                            self.flush(fs, &plan, &pend[..np], &pend_var, &var, flat_all);
                            np = 0;
                        }
                    } else if pass {
                        if !flat_all {
                            self.interpolate(&mut var, provoking, &s, &l, invw);
                        }
                        let frag = match (constant_fs, cached) {
                            (true, Some(c)) => c,
                            _ => {
                                let c = fs.run_fragment(&var, [px as f32 + 0.5, py as f32 + 0.5, zf, invw as f32]);
                                cached = Some(c);
                                c
                            }
                        };
                        if let Some(c) = frag {
                            if let Some(d) = &mut self.depth {
                                if d.write {
                                    d.data[idx] = zf;
                                }
                            }
                            let bpt = plan.bpt;
                            let texel = &mut self.color.data[idx * bpt..(idx + 1) * bpt];
                            plan.write(texel, c, constant_fs, &mut const_texel, &mut const_blend);
                        }
                    }
                }
            }
            for i in 0..3 {
                w_row[i] += dwdy[i];
            }
        }
        if np > 0 {
            self.flush(fs, &plan, &pend[..np], &pend_var, &var, flat_all);
        }
    }

    /// One run of `n` covered pixels starting at (`px0`, `py`), shaded four at
    /// a time with the wide stage and written straight to the target: no
    /// per-pixel gather, marshalling or format dispatch. `w0` = the edge
    /// functions at the first pixel; varyings are interpolated only for the
    /// inputs the shader reads.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    #[allow(clippy::too_many_arguments)]
    fn fast_span(
        &mut self,
        fs: &mut Invoker<'_>,
        plan: &PixelPlan,
        geo: &SpanGeo<'_>,
        tp: Option<&TriPlanes>,
        provoking: &Varyings,
        flat_all: bool,
        py: i64,
        px0: i64,
        w0: [i64; 3],
        n: usize,
        (kx0, ky): (i64, i64),
        runs_x: bool,
    ) -> bool {
        use super::exec::Shade4;
        let bgra = plan.unorm8.unwrap();
        let ypos = py as f32 + 0.5;
        if !fs.span_begin(provoking, ypos, !flat_all) {
            return false;
        }
        // every pixel of the span came straight from the shader (no blending
        // with the destination, no discard): the row can be copied downwards
        let mut pure = true;
        let cw = self.color.width as i64;
        let base = ((py - self.row0) * cw + px0) as usize * 4;
        // the triangle's planes, evaluated at this span's first pixel:
        // value(k) = a + b*k for the k-th pixel of the span
        let mut planes = [Plane::Const(0); 32];
        if let Some(tp) = tp {
            for i in 0..tp.n {
                planes[i] = match tp.p[i] {
                    TriPlane::Const(b) => Plane::Const(b),
                    TriPlane::Lin { a, bx, by } => Plane::Lin { a: a + bx * kx0 as f64 + by * ky as f64, b: bx },
                    TriPlane::Persp { a, bx, by, da, dbx, dby } => Plane::Persp {
                        a: a + bx * kx0 as f64 + by * ky as f64,
                        b: bx,
                        da: da + dbx * kx0 as f64 + dby * ky as f64,
                        db: dbx,
                    },
                };
            }
        }
        let mut done = 0usize;
        while done < n {
            let m = (n - done).min(4);
            let x0 = (px0 + done as i64) as f32;
            let shade = fs.shade4_with(x0, ypos, m, |lane, i| {
                let k = (done + lane) as f64;
                match planes[i] {
                    Plane::Const(bits) => bits,
                    Plane::Lin { a, b } => ((a + b * k) as f32).to_bits(),
                    Plane::Persp { a, b, da, db } => (((a + b * k) / (da + db * k)) as f32).to_bits(),
                }
            });
            let out = &mut self.color.data[base + done * 4..base + (done + m) * 4];
            match shade {
                Shade4::Colors(c) => {
                    let (px, all_opaque) = format::encode4_unorm8([&c[0].0, &c[1].0, &c[2].0, &c[3].0], bgra);
                    let stored = plan.blend.is_none() || (plan.opaque_is_store && all_opaque);
                    if stored {
                        for (i, p) in px.iter().take(m).enumerate() {
                            out[i * 4..i * 4 + 4].copy_from_slice(&p.to_le_bytes());
                        }
                    } else if let Some((sfc, sfa)) = plan.over {
                        pure = false;
                        // blend all four against the destination at once
                        let mut d = [0u8; 16];
                        d[..m * 4].copy_from_slice(out);
                        let px = format::blend_over4_unorm8([&c[0].0, &c[1].0, &c[2].0, &c[3].0], &d, bgra, sfc, sfa);
                        for (i, p) in px.iter().take(m).enumerate() {
                            out[i * 4..i * 4 + 4].copy_from_slice(&p.to_le_bytes());
                        }
                    } else {
                        pure = false;
                        for i in 0..m {
                            let col = [c[0].0[i], c[1].0[i], c[2].0[i], c[3].0[i]];
                            plan.write(&mut out[i * 4..i * 4 + 4], col, false, &mut None, &mut None);
                        }
                    }
                    // a run: the last pixel's result also holds for the pixels after
                    // it while the shader's quantized position values stay the same
                    if runs_x
                        && m == 4
                        && (stored || (plan.opaque_is_store && c[3].0[3] == 1.0f32.to_bits()))
                        && n - done > 4
                    {
                        let extra = fs.x_extent(px0 + done as i64 + 3, n - done - 4);
                        if extra > 0 {
                            let t = px[3].to_le_bytes();
                            let at = base + (done + 4) * 4;
                            for q in self.color.data[at..at + extra * 4].chunks_exact_mut(4) {
                                q.copy_from_slice(&t);
                            }
                            RUN_PIXELS.fetch_add(extra as u64, std::sync::atomic::Ordering::Relaxed);
                            done += extra;
                        }
                    }
                }
                Shade4::Diverged => {
                    pure = false;
                    // the lanes branched differently: one scalar run each
                    for i in 0..m {
                        let mut var = *provoking;
                        if !flat_all {
                            let kk = (done + i) as i64;
                            let w = [w0[0] + geo.dwdx[0] * kk, w0[1] + geo.dwdx[1] * kk, w0[2] + geo.dwdx[2] * kk];
                            let l = [
                                w[0] as f64 * geo.inv_sum,
                                w[1] as f64 * geo.inv_sum,
                                w[2] as f64 * geo.inv_sum,
                            ];
                            let invw = l[0] * geo.s[0].invw + l[1] * geo.s[1].invw + l[2] * geo.s[2].invw;
                            self.interpolate(&mut var, provoking, geo.s, &l, invw);
                        }
                        if let Some(col) = fs.run_fragment(&var, [x0 + i as f32 + 0.5, ypos, 0.0, 0.0]) {
                            let out = &mut self.color.data[base + (done + i) * 4..base + (done + i) * 4 + 4];
                            plan.write(out, col, false, &mut None, &mut None);
                        }
                    }
                }
                Shade4::Killed => pure = false,
            }
            done += m;
        }
        pure
    }

    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    #[allow(clippy::too_many_arguments)]
    fn fast_span(
        &mut self,
        _fs: &mut Invoker<'_>,
        _plan: &PixelPlan,
        _geo: &SpanGeo<'_>,
        _tp: Option<&TriPlanes>,
        _provoking: &Varyings,
        _flat_all: bool,
        _py: i64,
        _px0: i64,
        _w0: [i64; 3],
        _n: usize,
        _k: (i64, i64),
        _runs_x: bool,
    ) -> bool {
        unreachable!("span_fast needs a wide stage")
    }

    /// shade the gathered pixels in one batch, then depth-write and blend each
    fn flush(
        &mut self,
        fs: &mut Invoker<'_>,
        plan: &PixelPlan,
        pend: &[Pending],
        pend_var: &[Varyings; 4],
        tri_var: &Varyings,
        flat: bool,
    ) {
        let mut ins: [([f32; 4], &Varyings); 4] = [([0.0; 4], tri_var); 4];
        for (i, p) in pend.iter().enumerate() {
            ins[i] = (p.pos, if flat { tri_var } else { &pend_var[i] });
        }
        let mut outs = [None; 4];
        fs.run_fragment_batch(&ins[..pend.len()], &mut outs[..pend.len()]);
        let (mut ct, mut cb) = (None, None);
        for (p, frag) in pend.iter().zip(outs) {
            if let Some(c) = frag {
                if let Some(d) = &mut self.depth {
                    if d.write {
                        d.data[p.idx] = p.zf;
                    }
                }
                let bpt = plan.bpt;
                let texel = &mut self.color.data[p.idx * bpt..(p.idx + 1) * bpt];
                plan.write(texel, c, false, &mut ct, &mut cb);
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

// ---------------------------------------------------------------------------
// per-triangle pixel write plan
// ---------------------------------------------------------------------------

/// A varying component over a whole triangle: value(kx, ky) = a + bx*kx + by*ky
/// at pixel offsets (kx, ky) from the triangle's bounding-box corner
/// (perspective: a ratio of two such planes).
#[derive(Clone, Copy)]
enum TriPlane {
    Const(u32),
    Lin { a: f64, bx: f64, by: f64 },
    Persp { a: f64, bx: f64, by: f64, da: f64, dbx: f64, dby: f64 },
}

/// the planes of every varying input the shader reads, built once per triangle
struct TriPlanes {
    n: usize,
    p: [TriPlane; 32],
}

#[derive(Clone, Copy)]
enum Plane {
    Const(u32),
    Lin { a: f64, b: f64 },
    Persp { a: f64, b: f64, da: f64, db: f64 },
}

/// per-triangle constants of the fast span path
struct SpanGeo<'a> {
    s: &'a [SV; 3],
    dwdx: [i64; 3],
    inv_sum: f64,
}

#[derive(Clone, Copy)]
struct Pending {
    idx: usize,
    zf: f32,
    pos: [f32; 4],
}

/// Everything about writing a pixel that does not change within a triangle,
/// resolved once instead of per pixel.
struct PixelPlan {
    fmt: wgpu::TextureFormat,
    bpt: usize,
    blend: Option<wgpu::BlendState>,
    mask: [bool; 4],
    mask_all: bool,
    k: [f32; 4],
    /// the blend leaves a fully opaque source untouched (premultiplied or
    /// straight "over": dst factor is 1 - src alpha, src factor is 1 or src
    /// alpha), so alpha == 1 pixels can skip the destination read
    opaque_is_store: bool,
    /// 8-bit RGBA/BGRA unorm target (Some(true) = BGRA) whose texels the
    /// vector encoder can write: 4 bytes, no sRGB conversion, all channels
    unorm8: Option<bool>,
    /// premultiplied / straight "over" on an `unorm8` target:
    /// (colour src factor is src alpha, alpha src factor is src alpha)
    over: Option<(bool, bool)>,
}

impl PixelPlan {
    fn new(c: &ColorTarget<'_>, k: [f32; 4]) -> PixelPlan {
        let wm = c.write_mask;
        PixelPlan {
            fmt: c.format,
            bpt: format::bytes_per_texel(c.format).unwrap() as usize,
            blend: c.blend,
            mask: [
                wm.contains(wgpu::ColorWrites::RED),
                wm.contains(wgpu::ColorWrites::GREEN),
                wm.contains(wgpu::ColorWrites::BLUE),
                wm.contains(wgpu::ColorWrites::ALPHA),
            ],
            mask_all: wm == wgpu::ColorWrites::ALL,
            k,
            unorm8: match c.format {
                wgpu::TextureFormat::Bgra8Unorm if wm == wgpu::ColorWrites::ALL => Some(true),
                wgpu::TextureFormat::Rgba8Unorm if wm == wgpu::ColorWrites::ALL => Some(false),
                _ => None,
            },
            over: c.blend.and_then(|b| {
                let ok = |c: &wgpu::BlendComponent| {
                    c.operation == BlendOperation::Add
                        && matches!(c.src_factor, BlendFactor::One | BlendFactor::SrcAlpha)
                        && c.dst_factor == BlendFactor::OneMinusSrcAlpha
                };
                (ok(&b.color) && ok(&b.alpha) && matches!(c.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm) && wm == wgpu::ColorWrites::ALL)
                    .then_some((b.color.src_factor == BlendFactor::SrcAlpha, b.alpha.src_factor == BlendFactor::SrcAlpha))
            }),
            opaque_is_store: c.blend.is_some_and(|b| {
                let over = |c: &wgpu::BlendComponent| {
                    c.operation == BlendOperation::Add
                        && matches!(c.src_factor, BlendFactor::One | BlendFactor::SrcAlpha)
                        && c.dst_factor == BlendFactor::OneMinusSrcAlpha
                };
                over(&b.color) && over(&b.alpha) && wm == wgpu::ColorWrites::ALL
            }),
        }
    }

    /// blend factors that depend only on the source and the constant: the
    /// destination-dependent ones cannot be hoisted
    fn src_only(f: BlendFactor) -> bool {
        !matches!(
            f,
            BlendFactor::Dst
                | BlendFactor::OneMinusDst
                | BlendFactor::DstAlpha
                | BlendFactor::OneMinusDstAlpha
                | BlendFactor::SrcAlphaSaturated
        )
    }

    #[inline]
    fn write(
        &self,
        texel: &mut [u8],
        frag: [u32; 4],
        constant_src: bool,
        const_texel: &mut Option<[u8; 4]>,
        const_blend: &mut Option<([f32; 4], [f32; 4])>,
    ) {
        let src = [
            f32::from_bits(frag[0]),
            f32::from_bits(frag[1]),
            f32::from_bits(frag[2]),
            f32::from_bits(frag[3]),
        ];
        if self.opaque_is_store && src[3] == 1.0 {
            // src*1 + dst*(1 - 1) == src
            format::encode(self.fmt, src, texel);
            return;
        }
        let Some(b) = &self.blend else {
            if self.mask_all {
                // plain store: for a constant source the texel is the same
                // for the whole triangle
                if constant_src {
                    let t = const_texel.get_or_insert_with(|| {
                        let mut t = [0u8; 4];
                        format::encode(self.fmt, src, &mut t);
                        t
                    });
                    // fixed-size stores: a variable-length copy_from_slice is a
                    // real memcpy call per pixel on musl
                    match self.bpt {
                        4 => texel.copy_from_slice(&t[..4]),
                        2 => texel.copy_from_slice(&t[..2]),
                        _ => texel[0] = t[0],
                    }
                } else {
                    format::encode(self.fmt, src, texel);
                }
                return;
            }
            return self.write_generic(texel, src, None);
        };
        // blend with source-and-constant-only factors, Add, all channels
        // written, constant source: out = src*sf + dst*df with sf/df fixed
        let hoistable = constant_src
            && self.mask_all
            && b.color.operation == BlendOperation::Add
            && b.alpha.operation == BlendOperation::Add
            && Self::src_only(b.color.src_factor)
            && Self::src_only(b.color.dst_factor)
            && Self::src_only(b.alpha.src_factor)
            && Self::src_only(b.alpha.dst_factor);
        if hoistable {
            let (sf, df) = const_blend.get_or_insert_with(|| {
                let z = [0.0; 4];
                let (mut sf, mut df) = ([0.0f32; 4], [0.0f32; 4]);
                for c in 0..4 {
                    let comp = if c == 3 { &b.alpha } else { &b.color };
                    sf[c] = factor(comp.src_factor, src, z, self.k, c);
                    df[c] = factor(comp.dst_factor, src, z, self.k, c);
                }
                (sf, df)
            });
            let d = format::decode(self.fmt, texel);
            let out = [
                src[0] * sf[0] + d[0] * df[0],
                src[1] * sf[1] + d[1] * df[1],
                src[2] * sf[2] + d[2] * df[2],
                src[3] * sf[3] + d[3] * df[3],
            ];
            format::encode(self.fmt, out, texel);
            return;
        }
        self.write_generic(texel, src, Some(b))
    }

    fn write_generic(&self, texel: &mut [u8], src: [f32; 4], blend_state: Option<&wgpu::BlendState>) {
        let mut out = src;
        let dst = format::decode(self.fmt, texel);
        if let Some(b) = blend_state {
            out = blend(b, src, dst, self.k);
        }
        if !self.mask_all {
            for i in 0..4 {
                if !self.mask[i] {
                    out[i] = dst[i];
                }
            }
        }
        format::encode(self.fmt, out, texel);
    }
}
