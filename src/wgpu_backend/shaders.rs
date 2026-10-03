//! The WGSL programs of the wgpu path, and the fixed-function contract they
//! run against.
//!
//! These are line-for-line ports of `softrender.rs`'s per-frame math. The
//! M3 acceptance test is that this path produces frames bit-identical to the
//! software rasterizer, so every expression below mirrors the Rust source in
//! the exact operation order, at `f32` throughout:
//!
//!   `vs_main`     = `Renderer::transform` (per corner) + the flat-shading
//!                   head of `raster_tri` (normal, lambert, hue wave, shade)
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
/// storage stride (vec3<f32> has 16-byte alignment, so the struct is 48).
pub const SHADERS_WGSL: &str = r#"
struct Tri {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
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

// the ten xterm-256 PURPLE_BLUE_HUES of akuma-cli, decoded to RGB, in
// softrender.rs order; index math below mirrors `hue_idx` exactly.
const HUES = array<vec3<u32>, 10>(
    vec3<u32>(0x5fu, 0x00u, 0xffu), // 57  dark purple
    vec3<u32>(0x5fu, 0x5fu, 0xffu), // 61  purple
    vec3<u32>(0x87u, 0x00u, 0xffu), // 93  light purple
    vec3<u32>(0xafu, 0x00u, 0xffu), // 129 pink-purple
    vec3<u32>(0x00u, 0x00u, 0xd7u), // 20  dark blue
    vec3<u32>(0x00u, 0x5fu, 0xffu), // 27  blue
    vec3<u32>(0x00u, 0x87u, 0xffu), // 33  light blue
    vec3<u32>(0x00u, 0xafu, 0xffu), // 39  cyan-blue
    vec3<u32>(0x00u, 0xd7u, 0xffu), // 45  light cyan
    vec3<u32>(0x00u, 0xffu, 0xffu), // 51  bright cyan
);

// "from the upper left, out of the screen" — softrender's LIGHT
const LIGHT = vec3<f32>(-0.45, 0.65, 0.62);

// softrender's frac01
fn frac01(x: f32) -> f32 {
    return x - floor(x);
}

// softrender's shade(rgb, k): per channel, (c as f32 * k).min(255.0) as u32
fn shade(c: vec3<u32>, k: f32) -> vec3<u32> {
    let r = u32(min(f32(c.r) * k, 255.0));
    let g = u32(min(f32(c.g) * k, 255.0));
    let b = u32(min(f32(c.b) * k, 255.0));
    return vec3<u32>(r, g, b);
}

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
    let t = tris[ti];
    var p = t.a;
    if (corner == 1u) {
        p = t.b;
    }
    if (corner == 2u) {
        p = t.c;
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
    let ax = t.a.x;
    let ay = t.a.y;
    let az = t.a.z;
    let bx = t.b.x;
    let by = t.b.y;
    let bz = t.b.z;
    let ccx = t.c.x;
    let ccy = t.c.y;
    let ccz = t.c.z;
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
    let lam = max((nx / nl) * LIGHT.x + (ny / nl) * LIGHT.y + (nz / nl) * LIGHT.z, 0.0);
    let bright = 0.35 + 0.75 * lam;

    // hue wave: purple->cyan sweeping across world height and extrusion
    // depth, rolling in time
    let wave = frac01(
        (ay + by + ccy) * 0.055
            + (az + bz + ccz) * 0.14
            + time * 0.12
    );
    let hue_idx = u32(round(wave * 9.0)) % 10u;
    out.color = shade(HUES[hue_idx], bright);
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
