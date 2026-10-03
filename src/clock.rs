//! Frame pacing and rate measurement.
//!
//! The template (`akuma-cli`) paces at a fixed 50 ms tick. A 3D renderer wants
//! a real target rate instead, but the failure mode is the same one the
//! kernel's splash solves with adaptive `tick_ms`: never spin, never fall
//! behind wall-clock. `Pacer` sleeps out the remainder of the frame budget;
//! `FpsMeter` accumulates what the exit report prints (the template's
//! metrics-on-exit culture, in pixels instead of characters).

use std::thread;
use std::time::{Duration, Instant};

pub struct Pacer {
    budget: Duration,
    frame_start: Instant,
}

impl Pacer {
    pub fn new(fps: u32) -> Self {
        let fps = fps.max(1);
        Pacer {
            budget: Duration::from_nanos(1_000_000_000 / u64::from(fps)),
            frame_start: Instant::now(),
        }
    }

    /// Sleep out the rest of this frame's budget. Overshoot is not an error —
    /// the next frame simply starts late and `FpsMeter` reports it.
    pub fn finish_frame(&mut self) {
        let elapsed = self.frame_start.elapsed();
        if elapsed < self.budget {
            thread::sleep(self.budget - elapsed);
        }
        self.frame_start = Instant::now();
    }
}

pub struct FpsMeter {
    start: Instant,
    frames: u64,
    over_budget: u64,
    budget: Duration,
    slowest: Duration,
    last_mark: Instant,
    last_frames: u64,
    /// Frames-per-second of the most recently completed 1 s window.
    pub live_fps: f64,
}

impl FpsMeter {
    pub fn new(budget: Duration) -> Self {
        let now = Instant::now();
        FpsMeter {
            start: now,
            frames: 0,
            over_budget: 0,
            budget,
            slowest: Duration::ZERO,
            last_mark: now,
            last_frames: 0,
            live_fps: 0.0,
        }
    }

    /// Count a rendered frame; `render_time` is just the draw (no sleep).
    pub fn frame(&mut self, render_time: Duration) {
        self.frames += 1;
        if render_time > self.budget {
            self.over_budget += 1;
        }
        if render_time > self.slowest {
            self.slowest = render_time;
        }
        if self.last_mark.elapsed() >= Duration::from_secs(1) {
            let dt = self.last_mark.elapsed().as_secs_f64();
            self.live_fps = (self.frames - self.last_frames) as f64 / dt;
            self.last_mark = Instant::now();
            self.last_frames = self.frames;
        }
    }

    /// The exit report: (duration, frames, avg fps, over-budget frames,
    /// slowest frame). The caller adds bytes/pixels, which it knows better.
    pub fn summary(&self) -> (Duration, u64, f64, u64, Duration) {
        (
            self.start.elapsed(),
            self.frames,
            self.frames as f64 / self.start.elapsed().as_secs_f64().max(f64::EPSILON),
            self.over_budget,
            self.slowest,
        )
    }
}
