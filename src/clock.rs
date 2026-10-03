//! Frame pacing and rate measurement.
//!
//! The template (`akuma-cli`) paces at a fixed 50 ms tick. A 3D renderer wants
//! a real target rate instead, but the failure mode is the same one the
//! kernel's splash solves with adaptive `tick_ms`: never spin, never fall
//! behind wall-clock. `Pacer` sleeps out the remainder of the frame budget;
//! `FpsMeter` accumulates what the exit report prints (the template's
//! metrics-on-exit culture, in pixels instead of characters).
//!
//! Everything here runs on `monotonic()`/`sleep_until()` — raw
//! CLOCK_MONOTONIC/nanosleep with EINTR handled by us — and deliberately NOT
//! on `std::time::Instant`/`Instant::elapsed`. Kernel note (Akuma, probed on
//! the box 2026-10-03): the arrival of a signal that has a user handler
//! EINTRs the in-flight syscall, and std's `Instant::now` unwraps that EINTR,
//! so a SIGTERM mid-frame used to kill the demo with
//! "called `Result::unwrap()` on an `Err` value: Os { code: 4, kind:
//! Interrupted }". std's `thread::sleep` already retries EINTR; our clock
//! and sleep do the same explicitly (see input.rs for the full story).

/// CLOCK_MONOTONIC in seconds (f64). Retries EINTR (that is signal arrival
/// on this kernel). Any other error returns 0.0 — pacing degrades to
/// free-run, which beats aborting a screensaver over a broken clock.
pub fn monotonic() -> f64 {
    loop {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) } == 0 {
            return ts.tv_sec as f64 + ts.tv_nsec as f64 / 1e9;
        }
        match std::io::Error::last_os_error().raw_os_error() {
            Some(libc::EINTR) => continue,
            _ => return 0.0,
        }
    }
}

/// Sleep until `deadline` (as returned by `monotonic`). EINTR recomputes the
/// remainder from the clock; a non-EINTR error gives up pacing for this
/// frame (free-run) rather than spin or abort.
pub fn sleep_until(deadline: f64) {
    loop {
        let remain = deadline - monotonic();
        if remain <= 0.0 {
            return;
        }
        let ts = libc::timespec {
            tv_sec: remain as i64,
            tv_nsec: (remain.fract() * 1e9) as i64,
        };
        if unsafe { libc::nanosleep(&ts, std::ptr::null_mut()) } == 0 {
            return;
        }
        match std::io::Error::last_os_error().raw_os_error() {
            Some(libc::EINTR) => continue,
            _ => return,
        }
    }
}

pub struct Pacer {
    budget: f64,
    frame_start: f64,
}

impl Pacer {
    pub fn new(fps: u32) -> Self {
        let fps = fps.max(1) as f64;
        Pacer {
            budget: 1.0 / fps,
            frame_start: monotonic(),
        }
    }

    /// Sleep out the rest of this frame's budget. Overshoot is not an error —
    /// the next frame simply starts late and `FpsMeter` reports it.
    pub fn finish_frame(&mut self) {
        sleep_until(self.frame_start + self.budget);
        self.frame_start = monotonic();
    }
}

pub struct FpsMeter {
    start: f64,
    frames: u64,
    over_budget: u64,
    budget: f64,
    slowest: f64,
    last_mark: f64,
    last_frames: u64,
    /// Frames-per-second of the most recently completed 1 s window.
    pub live_fps: f64,
}

impl FpsMeter {
    pub fn new(budget_secs: f64) -> Self {
        let now = monotonic();
        FpsMeter {
            start: now,
            frames: 0,
            over_budget: 0,
            budget: budget_secs,
            slowest: 0.0,
            last_mark: now,
            last_frames: 0,
            live_fps: 0.0,
        }
    }

    /// Count a rendered frame; `render_time` is just the draw (no sleep).
    pub fn frame(&mut self, render_time: f64) {
        self.frames += 1;
        if render_time > self.budget {
            self.over_budget += 1;
        }
        if render_time > self.slowest {
            self.slowest = render_time;
        }
        if monotonic() - self.last_mark >= 1.0 {
            let dt = (monotonic() - self.last_mark).max(f64::EPSILON);
            self.live_fps = (self.frames - self.last_frames) as f64 / dt;
            self.last_mark = monotonic();
            self.last_frames = self.frames;
        }
    }

    /// The exit report: (duration, frames, avg fps, over-budget frames,
    /// slowest frame) — durations in seconds. The caller adds bytes/pixels,
    /// which it knows better.
    pub fn summary(&self) -> (f64, u64, f64, u64, f64) {
        let dur = (monotonic() - self.start).max(0.0);
        (
            dur,
            self.frames,
            self.frames as f64 / dur.max(f64::EPSILON),
            self.over_budget,
            self.slowest,
        )
    }
}
