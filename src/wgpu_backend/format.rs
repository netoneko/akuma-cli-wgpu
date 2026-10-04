//! Texel formats of the akuma GPU, and conversion between stored bytes and
//! the values shaders and blending see.
//!
//! Color storage is always raw bytes, row-major, `bytes_per_texel` each. The
//! set is what real wgpu clients (rio/sugarloaf) use: 8-bit unorm color
//! (`R8`, `Rg8`, `Rgba8`, `Bgra8`, with `Srgb` variants where it matters) and
//! `Rgba8Uint`, the demo's raw-bytes target.

use wgpu::TextureFormat as F;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleKind {
    /// unorm / srgb: shaders see f32
    Float,
    /// integer formats: shaders see u32
    Uint,
}

pub fn bytes_per_texel(f: F) -> Option<u32> {
    Some(match f {
        F::R8Unorm | F::R8Uint => 1,
        F::Rg8Unorm => 2,
        F::Rgba8Unorm | F::Rgba8UnormSrgb | F::Bgra8Unorm | F::Bgra8UnormSrgb | F::Rgba8Uint => 4,
        F::Depth32Float => 4,
        _ => return None,
    })
}

pub fn is_depth(f: F) -> bool {
    matches!(f, F::Depth32Float)
}

/// the demo's raw-bytes target: positions already in pixels, no blending,
/// shader integers land in the texel bytes untouched
pub fn is_legacy_raw(f: F) -> bool {
    matches!(f, F::Rgba8Uint)
}

pub fn sample_kind(f: F) -> SampleKind {
    match f {
        F::R8Uint | F::Rgba8Uint => SampleKind::Uint,
        _ => SampleKind::Float,
    }
}

pub fn is_srgb(f: F) -> bool {
    matches!(f, F::Rgba8UnormSrgb | F::Bgra8UnormSrgb)
}

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn unorm(b: u8) -> f32 {
    b as f32 / 255.0
}

fn to_unorm(c: f32) -> u8 {
    // WebGPU: clamp, scale, round to nearest (ties away from zero is fine
    // here: x*255 is never exactly .5 for the representable inputs we care about)
    (c.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
}

/// stored texel -> (r, g, b, a) as the shader/blender sees it; missing
/// channels follow WebGPU (g, b = 0, a = 1). sRGB formats decode to linear.
pub fn decode(f: F, px: &[u8]) -> [f32; 4] {
    match f {
        F::R8Unorm => [unorm(px[0]), 0.0, 0.0, 1.0],
        F::Rg8Unorm => [unorm(px[0]), unorm(px[1]), 0.0, 1.0],
        F::Rgba8Unorm => [unorm(px[0]), unorm(px[1]), unorm(px[2]), unorm(px[3])],
        F::Rgba8UnormSrgb => [
            srgb_to_linear(unorm(px[0])),
            srgb_to_linear(unorm(px[1])),
            srgb_to_linear(unorm(px[2])),
            unorm(px[3]),
        ],
        F::Bgra8Unorm => [unorm(px[2]), unorm(px[1]), unorm(px[0]), unorm(px[3])],
        F::Bgra8UnormSrgb => [
            srgb_to_linear(unorm(px[2])),
            srgb_to_linear(unorm(px[1])),
            srgb_to_linear(unorm(px[0])),
            unorm(px[3]),
        ],
        other => panic!("akuma backend: decode of {other:?}"),
    }
}

/// integer formats: raw channels as u32 (g, b = 0, a = 1 when absent)
pub fn decode_uint(f: F, px: &[u8]) -> [u32; 4] {
    match f {
        F::R8Uint => [px[0] as u32, 0, 0, 1],
        F::Rgba8Uint => [px[0] as u32, px[1] as u32, px[2] as u32, px[3] as u32],
        other => panic!("akuma backend: decode_uint of {other:?}"),
    }
}

/// (r, g, b, a) in linear/shader space -> stored texel bytes
pub fn encode(f: F, c: [f32; 4], out: &mut [u8]) {
    match f {
        F::R8Unorm => out[0] = to_unorm(c[0]),
        F::Rg8Unorm => {
            out[0] = to_unorm(c[0]);
            out[1] = to_unorm(c[1]);
        }
        F::Rgba8Unorm => {
            for i in 0..4 {
                out[i] = to_unorm(c[i]);
            }
        }
        F::Rgba8UnormSrgb => {
            for i in 0..3 {
                out[i] = to_unorm(linear_to_srgb(c[i].clamp(0.0, 1.0)));
            }
            out[3] = to_unorm(c[3]);
        }
        F::Bgra8Unorm => {
            out[0] = to_unorm(c[2]);
            out[1] = to_unorm(c[1]);
            out[2] = to_unorm(c[0]);
            out[3] = to_unorm(c[3]);
        }
        F::Bgra8UnormSrgb => {
            out[0] = to_unorm(linear_to_srgb(c[2].clamp(0.0, 1.0)));
            out[1] = to_unorm(linear_to_srgb(c[1].clamp(0.0, 1.0)));
            out[2] = to_unorm(linear_to_srgb(c[0].clamp(0.0, 1.0)));
            out[3] = to_unorm(c[3]);
        }
        other => panic!("akuma backend: encode to {other:?}"),
    }
}
