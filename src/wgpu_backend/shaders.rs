//! The WGSL programs of the wgpu path, and the fixed-function contract they
//! run against.
//!
//! These are line-for-line ports of `softrender.rs`'s per-frame math. The
//! M3 acceptance test is that this path produces frames bit-identical to the
//! software rasterizer, so every expression below mirrors the Rust source in
//! the exact operation order, at `f32` throughout:
//!
//!   `vs_main`     = `Renderer::transform` (per corner) + the degenerate-
//!                   normal check of `raster_tri`'s flat-shading head; the
//!                   color is the face-kind palette pick (softrender's
//!                   solid two-tone logo colors, FRONT_COLOR / WALL_COLOR)
//!   the rasterizer = `raster_tri`'s scanline walk, as backend fixed-function
//!                   (see backend.rs — it is the same edge/span/z math,
//!                   including the `mul_add` area test and the `-0.01` cull)
//!   `fs_main`     = pass-through of the flat per-triangle color into the
//!                   exact target byte order
//!
//! The rain backdrop stays on the CPU in both paths (shared code, so the
//! frames match by construction); see mod.rs.
//!
//! The akuma-GPU position contract: `@builtin(position)` from the vertex
//! stage is ALREADY in framebuffer pixels, and `.z` is the view-space depth
//! the rasterizer compares directly. There is no clip space, no perspective
//! divide, no viewport transform — those fixed-function stages do not exist
//! on this device, and the demo's math never wanted them.

/// Per-frame scalars, all f32 (no padding rules involved anywhere):
///
/// * time     — seconds since start, the exact f32 softrender received
/// * width    — frame width as f32 (exact: <= 2^24)
/// * height   — frame height as f32
/// * extent   — scene.extent: world extent of the larger grid axis
/// * center_x — screen x the mesh is centered on (`Scene::center_x`; the
///              live screensaver parks the logo on the left half of the
///              panel, selftest keeps the mid-frame default)
///
/// (The uniform is a plain 8xf32 struct — two vec4s' worth, no struct-
/// padding rules involved — bound at group 0 / binding 0; the mesh is
/// `array<Tri>` at group 0 / binding 1 with a 48-byte stride — vec3<f32>
/// has 16-byte alignment.)

/// The cat-logo mesh: one `Tri` per softrender `Tri`, padded to the WGSL
/// storage stride: vec3<f32> members are 12 bytes at 16-byte alignment
/// (a/b/c at 0/16/32), the u32 `kind` at 44, stride 48.
pub const SHADERS_WGSL: &str = r#"
struct Tri {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    // face family tagged by catlogo::extrude: 0 = front, 1 = wall —
    // softrender's KIND_FRONT / KIND_WALL
    kind: u32,
};

// (time, width, height, extent, center_x) — see mod.rs for who fills it
struct Uniforms {
    time: f32,
    width: f32,
    height: f32,
    extent: f32,
    // screen x the mesh is centered on: scene.center_x — mid-frame in the
    // selftest, quarter width in the live screensaver (dead right half)
    center_x: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<uniform> frame: Uniforms;
@group(0) @binding(1) var<storage, read> tris: array<Tri>;

// the logo palette — softrender's FRONT_COLOR / WALL_COLOR: one solid
// color per face family, no lambert, no hue wave. Solid colors are what
// make the mark read as a logo.
const FRONT_COLOR = vec3<u32>(0x00u, 0xffu, 0xffu); // bright cyan, the front
const WALL_COLOR = vec3<u32>(0x2fu, 0x00u, 0x7fu); // deep purple, the walls

struct VsOut {
    // akuma-GPU contract: pixel coords in .xy, view depth in .z, no divide
    @builtin(position) pos: vec4<f32>,
    // per-triangle flat color, computed entirely in the vertex stage below
    @location(0) @interpolate(flat) color: vec3<u32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // one invocation per corner; the whole triangle is in the storage buffer
    let ti = vi / 3u;
    let corner = vi % 3u;
    // NOTE: the corners are read through direct indexed accesses rather
    // than a `let t = tris[ti]` local — the interpreter scrambles
    // struct-valued buffer reads materialized into a local (pre-existing
    // M3 parity bug), and direct loads take the plain pointer path.
    var p = tris[ti].a;
    if (corner == 1u) {
        p = tris[ti].b;
    }
    if (corner == 2u) {
        p = tris[ti].c;
    }

    // ---- Renderer::transform, same op order ----
    let time = frame.time;
    let width = frame.width;
    let height = frame.height;
    let extent = frame.extent;

    let yaw = time * 0.55;
    let pitch = 0.16 + 0.10 * sin(time * 0.7);
    let dist = extent * 2.1;

    // fit: world extent -> ~72% of the shorter screen axis at center depth
    let focal = 0.72 * min(height, width) * dist / extent;
    let cy = cos(yaw);
    let sy = sin(yaw);
    let cp = cos(pitch);
    let sp = sin(pitch);
    let cx = frame.center_x;
    let cypix = height / 2.0;

    // yaw around Y
    let rx = p.x * cy + p.z * sy;
    let z1 = -p.x * sy + p.z * cy;
    // pitch around X
    let ry = p.y * cp - z1 * sp;
    let z2 = p.y * sp + z1 * cp;
    // camera pulled back on -z
    let zc = z2 + dist;
    let inv = focal / zc;

    var out: VsOut;
    out.pos = vec4<f32>(cx + rx * inv, cypix - ry * inv, zc, 1.0);

    // ---- the flat-shading head of raster_tri, same op order ----
    // (direct indexed reads — see the note above about `let t = tris[ti]`)
    let ax = tris[ti].a.x;
    let ay = tris[ti].a.y;
    let az = tris[ti].a.z;
    let bx = tris[ti].b.x;
    let by = tris[ti].b.y;
    let bz = tris[ti].b.z;
    let ccx = tris[ti].c.x;
    let ccy = tris[ti].c.y;
    let ccz = tris[ti].c.z;
    let e1y = by - ay;
    let e1z = bz - az;
    let e2y = ccy - ay;
    let e2z = ccz - az;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * (ccx - ax) - (bx - ax) * e2z;
    let nz = (bx - ax) * e2y - e1y * (ccx - ax);
    let nl = sqrt(nx * nx + ny * ny + nz * nz);
    if (nl <= 1e-9) {
        // degenerate face: softrender returns here; park the triangle on a
        // zero-area position so the fixed-function area cull drops it
        out.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.color = vec3<u32>(0u, 0u, 0u);
        return out;
    }

    // logo shading: one solid color per face family, keyed on `kind` —
    // the exact colors raster_tri picks (same const values)
    if (tris[ti].kind == 1u) {
        out.color = WALL_COLOR;
    } else {
        out.color = FRONT_COLOR;
    }
    return out;
}

struct FsIn {
    // pixel-center coords from the rasterizer (the akuma-GPU contract)
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) color: vec3<u32>,
};

// Target is bgra8uint: raw bytes, no conversion of any kind between the
// shader's integers and the framebuffer word softrender would have written
// (0x00RRGGBB little-endian = bytes b, g, r, 0).
@fragment
fn fs_main(in: FsIn) -> @location(0) vec4<u32> {
    return vec4<u32>(in.color.b, in.color.g, in.color.r, 0u);
}
"#;
