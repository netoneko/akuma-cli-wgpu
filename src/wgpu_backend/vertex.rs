//! Vertex attribute fetch: bytes in a vertex buffer -> the 4 raw words a
//! shader's `@location` input sees. Missing components default to
//! (0, 0, 0, 1) like WebGPU, with the 1 typed to match the format.

use wgpu::VertexFormat as V;

fn rd(b: &[u8], at: usize, n: usize) -> Option<&[u8]> {
    b.get(at..at + n)
}

/// n little-endian components of `size` bytes each, each converted by `f`
fn comps(b: &[u8], at: usize, n: usize, size: usize, f: impl Fn(&[u8]) -> u32) -> [Option<u32>; 4] {
    let mut out = [None; 4];
    for (i, o) in out.iter_mut().enumerate().take(n) {
        *o = rd(b, at + i * size, size).map(&f);
    }
    out
}

pub fn fetch(fmt: V, bytes: &[u8], at: usize) -> [u32; 4] {
    let f32_1 = 1.0f32.to_bits();
    let u8_f = |s: &[u8]| (s[0] as f32 / 255.0).to_bits();
    let i8_f = |s: &[u8]| ((s[0] as i8 as f32) / 127.0).max(-1.0).to_bits();
    let u16_f = |s: &[u8]| (u16::from_le_bytes([s[0], s[1]]) as f32 / 65535.0).to_bits();
    let (c, one): ([Option<u32>; 4], u32) = match fmt {
        V::Uint8 => (comps(bytes, at, 1, 1, |s| s[0] as u32), 1),
        V::Uint8x2 => (comps(bytes, at, 2, 1, |s| s[0] as u32), 1),
        V::Uint8x4 => (comps(bytes, at, 4, 1, |s| s[0] as u32), 1),
        V::Sint8 => (comps(bytes, at, 1, 1, |s| s[0] as i8 as i32 as u32), 1),
        V::Sint8x2 => (comps(bytes, at, 2, 1, |s| s[0] as i8 as i32 as u32), 1),
        V::Sint8x4 => (comps(bytes, at, 4, 1, |s| s[0] as i8 as i32 as u32), 1),
        V::Unorm8 => (comps(bytes, at, 1, 1, u8_f), f32_1),
        V::Unorm8x2 => (comps(bytes, at, 2, 1, u8_f), f32_1),
        V::Unorm8x4 => (comps(bytes, at, 4, 1, u8_f), f32_1),
        V::Snorm8 => (comps(bytes, at, 1, 1, i8_f), f32_1),
        V::Snorm8x2 => (comps(bytes, at, 2, 1, i8_f), f32_1),
        V::Snorm8x4 => (comps(bytes, at, 4, 1, i8_f), f32_1),
        V::Uint16 => (comps(bytes, at, 1, 2, |s| u16::from_le_bytes([s[0], s[1]]) as u32), 1),
        V::Uint16x2 => (comps(bytes, at, 2, 2, |s| u16::from_le_bytes([s[0], s[1]]) as u32), 1),
        V::Uint16x4 => (comps(bytes, at, 4, 2, |s| u16::from_le_bytes([s[0], s[1]]) as u32), 1),
        V::Sint16 => (comps(bytes, at, 1, 2, |s| i16::from_le_bytes([s[0], s[1]]) as i32 as u32), 1),
        V::Sint16x2 => (comps(bytes, at, 2, 2, |s| i16::from_le_bytes([s[0], s[1]]) as i32 as u32), 1),
        V::Sint16x4 => (comps(bytes, at, 4, 2, |s| i16::from_le_bytes([s[0], s[1]]) as i32 as u32), 1),
        V::Unorm16 => (comps(bytes, at, 1, 2, u16_f), f32_1),
        V::Unorm16x2 => (comps(bytes, at, 2, 2, u16_f), f32_1),
        V::Unorm16x4 => (comps(bytes, at, 4, 2, u16_f), f32_1),
        V::Float32 => (comps(bytes, at, 1, 4, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])), f32_1),
        V::Float32x2 => (comps(bytes, at, 2, 4, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])), f32_1),
        V::Float32x3 => (comps(bytes, at, 3, 4, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])), f32_1),
        V::Float32x4 => (comps(bytes, at, 4, 4, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])), f32_1),
        V::Uint32 | V::Sint32 => (comps(bytes, at, 1, 4, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])), 1),
        V::Uint32x2 | V::Sint32x2 => (comps(bytes, at, 2, 4, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])), 1),
        V::Uint32x3 | V::Sint32x3 => (comps(bytes, at, 3, 4, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])), 1),
        V::Uint32x4 | V::Sint32x4 => (comps(bytes, at, 4, 4, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])), 1),
        other => panic!("akuma backend: vertex format {other:?} unsupported"),
    };
    [c[0].unwrap_or(0), c[1].unwrap_or(0), c[2].unwrap_or(0), c[3].unwrap_or(one)]
}
