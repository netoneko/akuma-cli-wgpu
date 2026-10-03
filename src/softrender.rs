//! The software rasterizer: the direct path of the demo (milestone M2).
//!
//! One triangle pipeline, no dependencies:
//!
//!   mesh (catlogo::extrude) -> rotate -> perspective project -> cull ->
//!   shade (lambert + purple-blue hue wave) -> scanline spans into a
//!   z-buffered RAM `Frame` -> one contiguous present into /dev/fb0.
//!
//! Design notes tied to the platform:
//!
//! * Composing in cached WB memory and presenting as full-span copies is the
//!   WC story of the whole runbook: random writes into a WC mapping are the
//!   71 MB/s failure mode; full spans are the 3026 MB/s one.
//! * The rain backdrop is the template's Matrix columns on a *virtual
//!   character grid*: cells sized like terminal cells (height = frame/45,
//!   the kernel console's HD-font row on the 4K panel; width 1:2), a rain
//!   column per cell column (packed twice denser than the template's
//!   wide-char `step_by(2)`),
//!   trails of discrete glyphs — near-white head, purple->cyan fading tail,
//!   per-cell flicker with the occasional dark cell. (The first port drew
//!   1-px streaks, which read as shooting stars on a 4K panel.) Still no
//!   font dependency — a glyph is a filled rect; rio's sugarloaf supplies
//!   real glyphs later.
//! * The z-buffer is refilled with +inf each frame. At 4K that is a 33 MB
//!   WB fill — the same class of write as the kernel's timed clear, so the
//!   exit metrics put an upper bound on it for free.

use crate::fb::Frame;
use crate::rng::Rng;

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Tri {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub c: [f32; 3],
}

impl Tri {
    pub fn new(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Tri {
        Tri { a, b, c }
    }
}

/// The scene: an extruded asset plus the rain state.
pub struct Scene {
    pub tris: Vec<Tri>,
    /// world extent of the larger grid axis (cells are 1.0 units)
    pub extent: f32,
    /// rain state; empty when the backdrop is off
    rain: Vec<Column>,
    /// screen x the mesh orbits around. Defaults to mid-frame; the live
    /// screensaver parks it at quarter width (the trashcan's panel is dead
    /// on the right half — the rain still spans the whole frame). selftest
    /// keeps the default so its checksums stay about the renderers.
    pub center_x: f32,
}

/// Seed for the rain's initial column stats (see `rng.rs` on why the render
/// path must not draw from the wall clock): fixed, so a scene's first frame
/// depends only on the scene.
const RAIN_SEED: u64 = 0x9E37_79B9_7F4A_7C15; // the golden-ratio constant

// The template's PURPLE_BLUE_HUES, decoded from xterm-256 to RGB.
const HUES: [[u8; 3]; 10] = [
    [0x5f, 0x00, 0xff], // 57  dark purple
    [0x5f, 0x5f, 0xff], // 61  purple
    [0x87, 0x00, 0xff], // 93  light purple
    [0xaf, 0x00, 0xff], // 129 pink-purple
    [0x00, 0x00, 0xd7], // 20  dark blue
    [0x00, 0x5f, 0xff], // 27  blue
    [0x00, 0x87, 0xff], // 33  light blue
    [0x00, 0xaf, 0xff], // 39  cyan-blue
    [0x00, 0xd7, 0xff], // 45  light cyan
    [0x00, 0xff, 0xff], // 51  bright cyan
];

const HEAD: u32 = 0xccffff; // rain head: near-white cyan
pub const BG: u32 = 0x05030a; // near-black, a hint of purple

#[inline]
fn rgb([r, g, b]: [u8; 3]) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

#[inline]
fn shade(c: u32, k: f32) -> u32 {
    let r = (((c >> 16) & 0xff) as f32 * k).min(255.0) as u32;
    let g = (((c >> 8) & 0xff) as f32 * k).min(255.0) as u32;
    let b = ((c & 0xff) as f32 * k).min(255.0) as u32;
    (r << 16) | (g << 8) | b
}

#[inline]
fn frac01(x: f32) -> f32 {
    x - x.floor()
}

/// Rain cell geometry: the template rains on a terminal character grid, so
/// the pixel rain does too — one "glyph" per cell, cells stacked with row
/// gaps, trails measured in cells. Height is a 45th of the frame (the
/// kernel console's HD-font grid is ~44 rows on the 4K panel), width half
/// of that (terminal cells are 1:2), clamped so odd frame sizes stay sane.
fn cell_metrics(frame_h: usize) -> (i64, i64) {
    let ch = (frame_h / 45).clamp(8, 48) as i64;
    let cw = (ch / 2).max(4);
    (cw, ch)
}

/// One Matrix column, transplanted from the template (`MatrixColumn`) —
/// same speed classes and hue re-rolls, but in *cell* units so the rain
/// reads as columns of glyphs instead of 1-px streaks.
struct Column {
    /// cell column index; one rain column every two cell columns, the
    /// template's wide-char `step_by(2)`
    col: i64,
    /// head row in cells (y grows downward, the tail trails above)
    y: f32,
    /// cells per frame
    speed: f32,
    /// trail length in cells
    len: i64,
    hue: usize,
}

impl Column {
    fn new(col: i64, rows: i64, rng: &mut Rng) -> Column {
        Column {
            col,
            y: -(rng.below(rows.max(1) as u64) as f32),
            speed: (1.0 + rng.below(4) as f32) * 0.055,
            len: 6 + rng.below(25) as i64,
            hue: rng.below(HUES.len() as u64) as usize,
        }
    }

    fn step(&mut self, rows: i64, rng: &mut Rng) {
        self.y += self.speed;
        if self.y - self.len as f32 > rows as f32 {
            // respawn above the top with fresh stats
            self.y = -(rng.below(rows.max(1) as u64 / 2 + 1) as f32);
            self.speed = (1.0 + rng.below(4) as f32) * 0.055;
            self.len = 6 + rng.below(25) as i64;
            self.hue = rng.below(HUES.len() as u64) as usize;
        } else if rng.pct(10) {
            self.hue = rng.below(HUES.len() as u64) as usize;
        }
    }

    fn draw(&self, frame: &mut Frame, time: f32) {
        let (cw, ch) = cell_metrics(frame.height);
        let rows = frame.height as i64 / ch;
        // the glyph box: ~3/4 of the cell wide, cell height minus a row
        // gap, so stacked cells read as separate glyphs on a character grid
        let gw = (cw * 3 / 4).max(2);
        let gh = (ch * 5 / 6).max(2);
        let gx = self.col * cw + (cw - gw) / 2;
        // flicker re-rolls ~12x/s; the same tick for the whole frame keeps
        // every cell stable within the frame (a pure function of scene+time)
        let tick = (time * 12.0) as u64;
        for i in 0..self.len {
            let row = self.y as i64 - i;
            if row < 0 || row >= rows {
                continue;
            }
            let color = if i == 0 {
                HEAD
            } else {
                let t = i as f32 / self.len as f32; // 0 at head
                // per-cell flicker, deterministic per (column, row, tick):
                // ~1 in 5 tail cells stays dark (the missing-glyph texture
                // of the template's columns), the rest ride the fade
                let n = Rng::with_seed(
                    (self.col as u64).rotate_left(32) ^ (row as u64).rotate_left(16) ^ tick,
                )
                .below(100);
                if n < 18 {
                    continue;
                }
                let flick = 0.6 + (n - 18) as f32 * (0.4 / 82.0);
                // hue drifts along the tail so long streaks sweep
                // purple -> cyan like the template's columns
                let idx =
                    (self.hue as f32 + t * 6.0 + time * 0.35) as usize % HUES.len();
                shade(rgb(HUES[idx]), (1.0 - t * 0.85) * flick)
            };
            for dy in 0..gh {
                frame.span(row * ch + dy, gx, gx + gw, color);
            }
        }
    }
}

impl Scene {
    pub fn new(tris: Vec<Tri>, extent: f32, with_rain: bool, frame_w: usize, frame_h: usize) -> Scene {
        let mut rng = Rng::with_seed(RAIN_SEED);
        // a column per cell column — packed twice denser than the
        // template's wide-char `step_by(2)`; density tracks the cell, so
        // the rain looks the same at any resolution (~160 columns on 16:9)
        let rain = if with_rain {
            rain_columns(frame_w, frame_h, &mut rng)
        } else {
            Vec::new()
        };
        Scene {
            tris,
            extent,
            rain,
            center_x: frame_w as f32 / 2.0,
        }
    }

    /// Rebuild the rain when the frame size changes (template: handle_resize).
    pub fn resize_rain(&mut self, frame_w: usize, frame_h: usize, with_rain: bool) {
        let want = if with_rain {
            let (cw, _) = cell_metrics(frame_h);
            (frame_w as i64 / cw).max(0) as usize
        } else {
            0
        };
        if want != self.rain.len() {
            let mut rng = Rng::with_seed(RAIN_SEED);
            self.rain = rain_columns(frame_w, frame_h, &mut rng);
        }
    }
}

fn rain_columns(frame_w: usize, frame_h: usize, rng: &mut Rng) -> Vec<Column> {
    let (cw, ch) = cell_metrics(frame_h);
    let rows = (frame_h as i64 / ch).max(1);
    let cols = frame_w as i64 / (cw * 2);
    (0..cols).map(|col| Column::new(col, rows, rng)).collect()
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

/// The shared backdrop of both render paths: clear the frame, then step and
/// draw the matrix rain. `Renderer::render` (software) and
/// `wgpu_backend::WgpuRenderer::render` both call this so the backdrop is
/// bit-identical by construction — the M3 acceptance is about the *mesh*
/// pixels, and keeping the rain in one place on the CPU is what makes the
/// frame checksums comparable at all. (The rain is a u64 xorshift; it has no
/// business in a WGSL shader, and rio's sugarloaf will not ship one either.)
///
/// Seeded from the frame time: identical across a selftest's runs (its times
/// are exact), never repeated in a live show — the rain steps stay a pure
/// function of (scene, time).
pub fn backdrop(frame: &mut Frame, scene: &mut Scene, time: f32, with_rain: bool) {
    frame.clear(BG);
    if with_rain {
        let mut rng = Rng::with_seed(time.to_bits() as u64 ^ RAIN_SEED);
        for col in &mut scene.rain {
            col.step(frame.height as i64, &mut rng);
            col.draw(frame, time);
        }
    }
}

/// Fixed directional light, roughly "from the upper left, out of the screen".
const LIGHT: [f32; 3] = [-0.45, 0.65, 0.62];

#[derive(Clone, Copy)]
struct Vtx {
    /// screen coords (pixels, y down)
    x: f32,
    y: f32,
    /// view-space depth (positive into the screen)
    z: f32,
}

pub struct Renderer {
    z: Vec<f32>,
    /// 3 projected vertices per triangle, in triangle order
    scratch: Vec<Vtx>,
    /// seconds since start, set per frame; drives the hue wave
    time: f32,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Renderer {
        Renderer {
            z: vec![f32::INFINITY; width * height],
            scratch: Vec::new(),
            time: 0.0,
        }
    }

    /// Render one frame. `time` is seconds since start (drives the orbit and
    /// the hue wave); the rain steps and draws as the backdrop first.
    pub fn render(&mut self, frame: &mut Frame, scene: &mut Scene, time: f32, with_rain: bool) {
        self.time = time;
        backdrop(frame, scene, time, with_rain);
        // Refill the z-buffer — the module doc promises this per frame and
        // the frame-cost budget counts on it (33 MB WB fill at 4K). Without
        // it, stale depths from the previous frame reject the rotating mesh
        // pixel-by-pixel: coverage erodes every frame, and the big-grid
        // assets (79/120, whose surfaces travel farthest in z per frame)
        // vanish entirely within a few dozen frames.
        self.z.fill(f32::INFINITY);

        // the screensaver orbit: slow yaw, capped pitch wobble. The cap
        // matters: catlogo::extrude skips bottom faces assuming the camera
        // stays above the z=0 plane.
        let yaw = time * 0.55;
        let pitch = 0.16 + 0.10 * (time * 0.7).sin();
        let dist = scene.extent * 2.1;

        self.transform(frame, scene, yaw, pitch, dist);
        for i in 0..scene.tris.len() {
            let t = scene.tris[i];
            self.raster_tri(frame, &t, i * 3);
        }
    }

    fn transform(&mut self, frame: &Frame, scene: &Scene, yaw: f32, pitch: f32, dist: f32) {
        // fit: world extent -> ~72% of the shorter screen axis at center depth
        let focal = 0.72 * frame.height.min(frame.width) as f32 * dist / scene.extent;
        let (cy, sy) = (yaw.cos(), yaw.sin());
        let (cp, sp) = (pitch.cos(), pitch.sin());
        // horizontal center comes from the scene: mid-frame by default,
        // quarter width in the live screensaver (dead right half of the
        // panel); see `Scene::center_x`
        let cx = scene.center_x;
        let cypix = frame.height as f32 / 2.0;

        self.scratch.clear();
        self.scratch.reserve(scene.tris.len() * 3);
        for t in &scene.tris {
            for p in [t.a, t.b, t.c] {
                // yaw around Y
                let x = p[0] * cy + p[2] * sy;
                let z1 = -p[0] * sy + p[2] * cy;
                // pitch around X
                let y = p[1] * cp - z1 * sp;
                let z2 = p[1] * sp + z1 * cp;
                // camera pulled back on -z
                let zc = z2 + dist;
                let inv = focal / zc;
                self.scratch.push(Vtx {
                    x: cx + x * inv,
                    y: cypix - y * inv,
                    z: zc,
                });
            }
        }
    }

    /// Rasterize one triangle; its projected vertices live at
    /// `scratch[base..base + 3]` (pushed in triangle order by `transform`).
    fn raster_tri(&mut self, frame: &mut Frame, t: &Tri, base: usize) {
        let v = [self.scratch[base], self.scratch[base + 1], self.scratch[base + 2]];

        // Signed area in y-down screen space. CCW-from-outside world
        // triangles project to negative area (see catlogo::extrude); cull
        // everything else, including degenerates.
        let area = (v[1].x - v[0].x).mul_add(v[2].y - v[0].y, -((v[1].y - v[0].y) * (v[2].x - v[0].x)));
        if area >= -0.01 {
            return;
        }

        // flat shading: world-space face normal (light is fixed in world
        // space, so no rotation needed here)
        let e1 = [t.b[0] - t.a[0], t.b[1] - t.a[1], t.b[2] - t.a[2]];
        let e2 = [t.c[0] - t.a[0], t.c[1] - t.a[1], t.c[2] - t.a[2]];
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];
        let nl = (nx * nx + ny * ny + nz * nz).sqrt();
        if nl <= 1e-9 {
            return;
        }
        let lam = ((nx / nl) * LIGHT[0] + (ny / nl) * LIGHT[1] + (nz / nl) * LIGHT[2]).max(0.0);
        let bright = 0.35 + 0.75 * lam;

        // hue wave: purple->cyan sweeping across world height and extrusion
        // depth, rolling in time — the splash's palette idea, transplanted
        let wave = frac01(
            (t.a[1] + t.b[1] + t.c[1]) * 0.055
                + (t.a[2] + t.b[2] + t.c[2]) * 0.14
                + self.time * 0.12,
        );
        let hue_idx = (wave * (HUES.len() as f32 - 1.0)).round() as usize % HUES.len();
        let color = shade(rgb(HUES[hue_idx]), bright);

        // scanline raster with per-pixel z interpolated along the edges
        let (w, h) = (frame.width as i64, frame.height as i64);
        let mut p = [v[0], v[1], v[2]];
        p.sort_by(|a, b| a.y.total_cmp(&b.y));

        let y0 = (p[0].y.ceil() as i64).max(0);
        let y1 = (p[2].y.ceil() as i64).min(h);
        for y in y0..y1 {
            let yy = y as f32 + 0.5;
            if yy < p[0].y || yy >= p[2].y {
                continue;
            }
            // long edge p0->p2; short edge is p0->p1 above p1.y, else p1->p2
            let (xl, zl) = edge_xz(p[0], p[2], yy);
            let (xs, zs) = if yy < p[1].y {
                edge_xz(p[0], p[1], yy)
            } else {
                edge_xz(p[1], p[2], yy)
            };
            let (xa, za, xb, zb) = if xl <= xs {
                (xl, zl, xs, zs)
            } else {
                (xs, zs, xl, zl)
            };
            let x0 = (xa.ceil() as i64).max(0);
            let x1 = (xb.ceil() as i64).min(w);
            if x0 >= x1 {
                continue;
            }
            let row = y as usize * frame.width as usize;
            let dx = xb - xa;
            let span_inv = if dx.abs() > 1e-6 { 1.0 / dx } else { 0.0 };
            for x in x0..x1 {
                let ts = ((x as f32 + 0.5) - xa) * span_inv;
                let z = za + (zb - za) * ts;
                let zi = &mut self.z[row + x as usize];
                if z < *zi {
                    *zi = z;
                    frame.buf[row + x as usize] = color;
                }
            }
        }
    }
}

/// x and z where the segment p->q crosses scanline `yy`. Callers split the
/// triangle at p1.y, so each call stays within the segment's y range.
#[inline]
fn edge_xz(p: Vtx, q: Vtx, yy: f32) -> (f32, f32) {
    let dy = q.y - p.y;
    if dy.abs() < 1e-6 {
        return (p.x, p.z);
    }
    let s = ((yy - p.y) / dy).clamp(0.0, 1.0);
    (p.x + (q.x - p.x) * s, p.z + (q.z - p.z) * s)
}
