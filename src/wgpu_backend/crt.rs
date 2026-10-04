//! A cheap CRT look, applied while presenting to the panel.
//!
//! rio's real CRT shaders (librashader's `newpixiecrt`) run through this
//! backend but cost 10-20 s per 4K frame on the CPU. This does the part that
//! reads as "old tube" for a few ms instead: barrel curvature, scanlines, a
//! vignette, soft rounded edges and a little horizontal beam spread. The
//! geometry is precomputed once per size into a source-pixel map and a gain
//! per pixel, so a frame is a lookup, a 3-tap blur and a multiply per pixel.
//!
//! The screen can be split into side-by-side tubes (one per pane of a
//! two-way split). Off by default; rio turns it on through its filter list
//! (`filters = ["akuma-crt"]` or `["akuma-crt-2"]`, see the rio fork's
//! sugarloaf; a `-flat` suffix turns the curvature off and keeps the rest),
//! anything else sets it with [`set_tubes`] / [`set_curved`].

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

/// 0 = off, n = n side-by-side tubes
static TUBES: AtomicU32 = AtomicU32::new(0);

pub fn set_tubes(n: u32) {
    TUBES.store(n, Ordering::Relaxed);
}

pub fn tubes() -> u32 {
    TUBES.load(Ordering::Relaxed)
}

/// Barrel curvature on (the default) or off: off keeps the scanlines,
/// vignette, soft edges and beam spread on a flat picture.
static CURVED: AtomicBool = AtomicBool::new(true);

pub fn set_curved(on: bool) {
    CURVED.store(on, Ordering::Relaxed);
}

pub fn curved() -> bool {
    CURVED.load(Ordering::Relaxed)
}

/// Curvature: how far the sampled point is pushed out at the far edge.
const BEND_X: f32 = 0.045;
const BEND_Y: f32 = 0.065;
/// Every third row is the gap between scanlines.
const SCAN_PERIOD: usize = 3;
const SCAN_GAP_GAIN: f32 = 0.62;
/// Vignette strength (at the very edge, per axis).
const VIGNETTE: f32 = 0.30;
/// Width of the soft fade at the tube's edge, in normalized units.
const EDGE: f32 = 0.025;
/// Black outside the glass.
const NONE: u32 = u32::MAX;

pub struct Crt {
    w: usize,
    h: usize,
    tubes: usize,
    curved: bool,
    /// per output pixel: source pixel index, or NONE
    map: Vec<u32>,
    /// per output pixel: brightness, 0..=256
    gain: Vec<u16>,
    out: Vec<u8>,
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Crt {
    pub fn new(w: usize, h: usize, tubes: usize, curved: bool) -> Crt {
        let (bend_x, bend_y) = if curved { (BEND_X, BEND_Y) } else { (0.0, 0.0) };
        let tubes = tubes.clamp(1, 8);
        let mut map = vec![NONE; w * h];
        let mut gain = vec![0u16; w * h];
        for t in 0..tubes {
            let x0 = w * t / tubes;
            let x1 = w * (t + 1) / tubes;
            let tw = (x1 - x0) as f32;
            for y in 0..h {
                let v = (y as f32 + 0.5) / h as f32 * 2.0 - 1.0;
                let scan = if y % SCAN_PERIOD == SCAN_PERIOD - 1 { SCAN_GAP_GAIN } else { 1.0 };
                for x in x0..x1 {
                    let u = ((x - x0) as f32 + 0.5) / tw * 2.0 - 1.0;
                    // barrel: sample further out towards the edges, so the
                    // picture bulges and its border curves inward
                    let su = u * (1.0 + bend_x * v * v);
                    let sv = v * (1.0 + bend_y * u * u);
                    if su.abs() >= 1.0 || sv.abs() >= 1.0 {
                        continue;
                    }
                    let sx = (((su + 1.0) * 0.5 * tw) as usize).min(x1 - x0 - 1) + x0;
                    let sy = (((sv + 1.0) * 0.5 * h as f32) as usize).min(h - 1);
                    let vig = (1.0 - VIGNETTE * su * su) * (1.0 - VIGNETTE * sv * sv);
                    let edge = smooth(1.0, 1.0 - EDGE, su.abs()) * smooth(1.0, 1.0 - EDGE, sv.abs());
                    let i = y * w + x;
                    map[i] = (sy * w + sx) as u32;
                    gain[i] = (scan * vig * edge * 256.0).round().clamp(0.0, 256.0) as u16;
                }
            }
        }
        Crt { w, h, tubes, curved, map, gain, out: vec![0u8; w * h * 4] }
    }

    pub fn fits(&self, w: usize, h: usize, tubes: usize, curved: bool) -> bool {
        self.w == w && self.h == h && self.tubes == tubes.clamp(1, 8) && self.curved == curved
    }

    /// The CRT image of `src` (4 bytes per pixel, any channel order; the
    /// 4th byte is passed through).
    pub fn apply(&mut self, src: &[u8]) -> &[u8] {
        assert_eq!(src.len(), self.w * self.h * 4, "crt: source size");
        struct Out(*mut u8);
        unsafe impl Sync for Out {}
        let out = Out(self.out.as_mut_ptr());
        let (w, h) = (self.w, self.h);
        let (map, gain) = (&self.map, &self.gain);
        const CHUNKS: usize = 8;
        struct Src(*const u32);
        unsafe impl Sync for Src {}
        let sp = Src(src.as_ptr() as *const u32);
        super::pool::run(CHUNKS, &|c| {
            let out = &out;
            let sp = &sp;
            // one pixel = one u32, channels in its low three bytes; the
            // reads are unaligned-safe and unchecked (indices come from
            // `map`, built in-bounds by `new`; `apply` checked src's size)
            let px = |i: usize| unsafe { sp.0.add(i).read_unaligned() };
            let ch = |p: u32, c: u32| (p >> (8 * c)) & 255;
            for y in (h * c / CHUNKS)..(h * (c + 1) / CHUNKS) {
                // SAFETY: chunks own disjoint row ranges of `out`
                let row = unsafe { out.0.add(y * w * 4) } as *mut u32;
                for x in 0..w {
                    let i = y * w + x;
                    let s = unsafe { *map.get_unchecked(i) };
                    let v = if s == NONE {
                        0xff00_0000
                    } else {
                        let s = s as usize;
                        // beam spread: the neighbours' samples, where they exist
                        let ml = if x > 0 { unsafe { *map.get_unchecked(i - 1) } } else { NONE };
                        let mr = if x + 1 < w { unsafe { *map.get_unchecked(i + 1) } } else { NONE };
                        let (p, l, r) = (
                            px(s),
                            px(if ml != NONE { ml as usize } else { s }),
                            px(if mr != NONE { mr as usize } else { s }),
                        );
                        let g = unsafe { *gain.get_unchecked(i) } as u32;
                        let mut v = p & 0xff00_0000;
                        for k in 0..3 {
                            let sum = 2 * ch(p, k) + ch(l, k) + ch(r, k);
                            v |= ((sum * g) >> 10).min(255) << (8 * k);
                        }
                        v
                    };
                    unsafe { row.add(x).write_unaligned(v) };
                }
            }
        });
        &self.out
    }
}

/// The CRT state the surface reuses between frames.
pub static CACHE: Mutex<Option<Crt>> = Mutex::new(None);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centre_maps_to_centre_and_corners_are_black() {
        let (w, h) = (64, 48);
        let crt = Crt::new(w, h, 1, true);
        let c = (h / 2) * w + w / 2;
        let s = crt.map[c] as usize;
        assert!((s % w).abs_diff(w / 2) <= 1 && (s / w).abs_diff(h / 2) <= 1);
        assert_eq!(crt.map[0], NONE);
        assert_eq!(crt.map[w * h - 1], NONE);
    }

    #[test]
    fn flat_maps_pixels_to_themselves() {
        let (w, h) = (64, 48);
        let crt = Crt::new(w, h, 2, false);
        for i in 0..w * h {
            assert_eq!(crt.map[i], i as u32);
        }
    }

    #[test]
    fn two_tubes_sample_their_own_half() {
        let (w, h) = (64, 32);
        let crt = Crt::new(w, h, 2, true);
        for y in 0..h {
            for x in 0..w {
                let s = crt.map[y * w + x];
                if s != NONE {
                    assert_eq!((s as usize % w) < w / 2, x < w / 2);
                }
            }
        }
    }

    #[test]
    fn deterministic_and_flat_white_stays_bright_in_the_middle() {
        let (w, h) = (40, 30);
        let mut a = Crt::new(w, h, 1, true);
        let src = vec![255u8; w * h * 4];
        let out_a = a.apply(&src).to_vec();
        let mut b = Crt::new(w, h, 1, true);
        assert_eq!(out_a, b.apply(&src));
        let mid = ((h / 2 - 2) * w + w / 2) * 4; // row 13: a scanline, not a gap
        assert!(out_a[mid] > 200, "centre too dark: {}", out_a[mid]);
    }
}
