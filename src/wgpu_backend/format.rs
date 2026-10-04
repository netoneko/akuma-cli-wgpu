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

const UNORM_LUT: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = i as f32 / 255.0;
        i += 1;
    }
    t
};

#[inline]
fn unorm(b: u8) -> f32 {
    UNORM_LUT[b as usize]
}

/// `unorm(b)` as bits
#[inline]
pub fn unorm_bits(b: u8) -> u32 {
    UNORM_LUT[b as usize].to_bits()
}

fn to_unorm(c: f32) -> u8 {
    // WebGPU: clamp, scale, round to nearest (ties away from zero is fine
    // here: x*255 is never exactly .5 for the representable inputs we care about)
    // the value is non-negative after the clamp, so truncation is the floor;
    // `f32::floor` would be a libm call per channel on a baseline x86-64 build
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
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

/// 4 pixels of an 8-bit unorm RGBA/BGRA target at once (SSE2, in the baseline
/// x86-64 feature set): `lanes` = the r, g, b, a channels, each holding the 4
/// pixels' values as f32 bits.
/// Returns the 4 packed texels (little-endian u32: bytes in memory order) and
/// whether every alpha is exactly 1.0. Bit-identical to `encode` per pixel:
/// clamp (NaN -> 0), x255, +0.5, truncate.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn encode4_unorm8(lanes: [&[u32; 4]; 4], bgra: bool) -> ([u32; 4], bool) {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 is part of the x86_64 baseline; loads are unaligned-safe
    unsafe {
        let zero = _mm_setzero_ps();
        let one = _mm_set1_ps(1.0);
        let k255 = _mm_set1_ps(255.0);
        let half = _mm_set1_ps(0.5);
        let ld = |p: &[u32; 4]| _mm_loadu_ps(p.as_ptr() as *const f32);
        let q = |v: __m128| {
            // max(v, 0) returns 0 for NaN (second operand), then min(.., 1)
            let c = _mm_min_ps(_mm_max_ps(v, zero), one);
            _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(c, k255), half))
        };
        let a_raw = ld(lanes[3]);
        let all_one = _mm_movemask_ps(_mm_cmpeq_ps(a_raw, one)) == 15;
        let (c0, c2) = if bgra { (lanes[2], lanes[0]) } else { (lanes[0], lanes[2]) };
        let b0 = q(ld(c0));
        let b1 = _mm_slli_epi32(q(ld(lanes[1])), 8);
        let b2 = _mm_slli_epi32(q(ld(c2)), 16);
        let b3 = _mm_slli_epi32(q(a_raw), 24);
        let px = _mm_or_si128(_mm_or_si128(b0, b1), _mm_or_si128(b2, b3));
        let mut out = [0u32; 4];
        _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, px);
        (out, all_one)
    }
}

/// 4 pixels of "source over destination" into an 8-bit unorm RGBA/BGRA target
/// (SSE2): `out = src * sf + dst * (1 - src.a)` per channel, where `sf` is 1
/// or the source alpha (`sf_color_alpha` for the colour channels,
/// `sf_alpha_alpha` for the alpha channel). `dst` holds the 4 destination
/// texels in memory order. Bit-identical to `blend` + `decode` + `encode` for
/// that blend state: the same f32 operations in the same order.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn blend_over4_unorm8(
    lanes: [&[u32; 4]; 4],
    dst: &[u8; 16],
    bgra: bool,
    sf_color_alpha: bool,
    sf_alpha_alpha: bool,
) -> [u32; 4] {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 is part of the x86_64 baseline
    unsafe {
        let zero = _mm_setzero_ps();
        let one = _mm_set1_ps(1.0);
        let k255 = _mm_set1_ps(255.0);
        let half = _mm_set1_ps(0.5);
        let ld = |p: &[u32; 4]| _mm_loadu_ps(p.as_ptr() as *const f32);
        let a = ld(lanes[3]);
        let df = _mm_sub_ps(one, a);
        let d = _mm_loadu_si128(dst.as_ptr() as *const __m128i);
        let mask = _mm_set1_epi32(0xff);
        // destination channel j (memory order) as unorm f32
        let dec = |j: u32| -> __m128 {
            let b = match j {
                0 => _mm_and_si128(d, mask),
                1 => _mm_and_si128(_mm_srli_epi32(d, 8), mask),
                2 => _mm_and_si128(_mm_srli_epi32(d, 16), mask),
                _ => _mm_srli_epi32(d, 24),
            };
            _mm_div_ps(_mm_cvtepi32_ps(b), k255)
        };
        let q = |v: __m128| {
            let c = _mm_min_ps(_mm_max_ps(v, zero), one);
            _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(c, k255), half))
        };
        // shader channels in memory order
        let (c0, c2) = if bgra { (lanes[2], lanes[0]) } else { (lanes[0], lanes[2]) };
        let mem = [ld(c0), ld(lanes[1]), ld(c2), a];
        let mut px = _mm_setzero_si128();
        for j in 0..4u32 {
            let s = mem[j as usize];
            let is_alpha = j == 3;
            let sf_alpha = if is_alpha { sf_alpha_alpha } else { sf_color_alpha };
            let sterm = if sf_alpha { _mm_mul_ps(s, a) } else { s };
            let out = _mm_add_ps(sterm, _mm_mul_ps(dec(j), df));
            px = _mm_or_si128(px, _mm_sll_epi32(q(out), _mm_cvtsi32_si128(8 * j as i32)));
        }
        let mut out = [0u32; 4];
        _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, px);
        out
    }
}
