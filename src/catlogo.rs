//! The cat mark as a 3D height field.
//!
//! The template (`akuma-cli`) centers the ASCII art in a terminal cell grid;
//! we do the same to the pixels of the framebuffer, but first we extrude it:
//! each glyph's ink density (the ` .:-=+*#%@` ramp — exactly the charset the
//! four assets use) becomes a height, and the flat art becomes a blocky 3D
//! mesh. No modelling, and the logo is genuinely three-dimensional — the same
//! trick the kernel's splash uses to make a static asset feel alive, one
//! dimension up.
//!
//! The assets are the same files the template embeds (`akuma_20/40/79/120.txt`);
//! `akuma_40.txt` is byte-identical to `amd64/src/akuma_40.txt`, the asset the
//! kernel banner and splash paint.

use akuma_cli_wgpu::softrender::{Tri, KIND_FRONT, KIND_WALL};

/// Ink density ramp, darkest last. Every non-space character in the assets is
/// in here; anything unknown maps to mid density so future assets degrade
/// gracefully instead of vanishing.
const RAMP: &[u8] = b" .:-=+*#%@";

pub struct HeightField {
    pub width: usize,
    pub height: usize,
    /// Row-major, `height * width`, in `0.0..=1.0` (0 = space).
    pub cells: Vec<f32>,
}

impl HeightField {
    pub fn parse(text: &str) -> HeightField {
        let lines: Vec<&str> = text.lines().collect();
        let width = lines.iter().map(|l| l.len()).max().unwrap_or(0);
        let height = lines.len();
        let mut cells = vec![0.0f32; width * height];
        for (y, line) in lines.iter().enumerate() {
            for (x, &b) in line.as_bytes().iter().enumerate() {
                let ramp_pos = RAMP.iter().position(|&r| r == b);
                let d = match ramp_pos {
                    Some(i) => i as f32 / (RAMP.len() - 1) as f32,
                    // unknown printable -> mid; the assets never hit this
                    None if b.is_ascii_graphic() => 0.5,
                    None => 0.0,
                };
                cells[y * width + x] = d;
            }
        }
        HeightField {
            width,
            height,
            cells,
        }
    }
}

/// Extrude the height field into triangles.
///
/// Cell size is 1.0 world unit; cell `(x, y)` spans `x..x+1` across and (with
/// y flipped so the art reads upright in a y-up world) `..` in y, with its top
/// face lifted to `density * depth`. For a closed, watertight-looking solid
/// we emit:
///
/// * the top face of every ink cell, and
/// * a side wall on every edge where the neighbour's height is lower or the
///   neighbour is outside the shape (a "stair riser").
///
/// Bottom faces are skipped: the camera never goes below the z=0 plane (the
/// pitch wobble in `softrender` is capped well under the horizon), so they
/// are always backfacing.
///
/// Winding: all triangles are CCW when viewed from outside, so after the
/// y-down screen projection the front faces carry a *negative* signed area
/// (see `softrender::raster`); that is what the backface cull keys on.
pub fn extrude(hf: &HeightField, depth: f32) -> Vec<Tri> {
    let mut tris = Vec::new();
    let w = hf.width as f32;
    let h = hf.height as f32;

    let at = |x: i64, y: i64| -> f32 {
        if x < 0 || y < 0 || x >= hf.width as i64 || y >= hf.height as i64 {
            0.0
        } else {
            hf.cells[y as usize * hf.width + x as usize]
        }
    };

    // World-space corners of cell (x, y), y flipped, centered on the origin.
    for cy in 0..hf.height {
        for cx in 0..hf.width {
            let d = at(cx as i64, cy as i64);
            if d <= 0.0 {
                continue;
            }
            let z = d * depth;
            // y-up: art row 0 is the top -> world +y
            let ax = cx as f32 - w / 2.0;
            let bx = ax + 1.0;
            let ay = (h / 2.0) - cy as f32 - 1.0;
            let by = ay + 1.0;

            // top face at z (CCW seen from +z) — the plate's FRONT
            quad(
                &mut tris,
                KIND_FRONT,
                [ax, ay, z],
                [bx, ay, z],
                [bx, by, z],
                [ax, by, z],
            );

            // walls (risers) where the neighbour is lower or absent.
            // right neighbour (+x): wall plane x = bx, CCW seen from +x
            if at(cx as i64 + 1, cy as i64) < d {
                quad(
                    &mut tris,
                    KIND_WALL,
                    [bx, ay, 0.0],
                    [bx, by, 0.0],
                    [bx, by, z],
                    [bx, ay, z],
                );
            }
            // left neighbour (-x): plane x = ax, CCW seen from -x
            if at(cx as i64 - 1, cy as i64) < d {
                quad(
                    &mut tris,
                    KIND_WALL,
                    [ax, by, 0.0],
                    [ax, ay, 0.0],
                    [ax, ay, z],
                    [ax, by, z],
                );
            }
            // neighbour above (+y, art row cy-1): plane y = by, CCW from +y
            if at(cx as i64, cy as i64 - 1) < d {
                quad(
                    &mut tris,
                    KIND_WALL,
                    [bx, by, 0.0],
                    [ax, by, 0.0],
                    [ax, by, z],
                    [bx, by, z],
                );
            }
            // neighbour below (-y): plane y = ay, CCW from -y
            if at(cx as i64, cy as i64 + 1) < d {
                quad(
                    &mut tris,
                    KIND_WALL,
                    [ax, ay, 0.0],
                    [bx, ay, 0.0],
                    [bx, ay, z],
                    [ax, ay, z],
                );
            }
        }
    }
    tris
}

/// Extrusion depth for an asset: a *slim plate*. A logo reads thin — the
/// old 35%-of-short-side slabs read as candy blocks — but the floor keeps
/// even the 20-column cat's edge visible instead of collapsing to a wafer.
pub fn depth_for(hf: &HeightField) -> f32 {
    ((hf.height.min(hf.width) as f32).max(6.0) * 0.14).max(2.0)
}

/// Two CCW triangles (a, b, c) + (a, c, d), tagged with the face `kind`.
fn quad(tris: &mut Vec<Tri>, kind: u32, a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) {
    tris.push(Tri::new(a, b, c, kind));
    tris.push(Tri::new(a, c, d, kind));
}
