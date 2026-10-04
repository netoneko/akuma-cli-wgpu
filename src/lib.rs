//! Library half of akuma-cli-wgpu: the wgpu custom backend and the modules it
//! needs (`fb` for the Frame it renders into, `softrender` for the Scene,
//! `clock`/`rng` for timing and determinism). The rio patch depends on this
//! target by path; the `akuma-wgpu` binary is the demo/selftest driver on top.

pub mod clock;
pub mod fb;
pub mod rng;
pub mod softrender;
pub mod wgpu_backend;
