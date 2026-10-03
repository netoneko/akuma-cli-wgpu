//! Milestone M3 placeholder for the custom wgpu backend.
//!
//! Nothing is wired here yet — `--wgpu` reports status and exits. The plan
//! (docs/runbooks/amd64-fbdev-wgpu-demo.md §5) is:
//!
//! * `wgpu = { features = ["custom", "wgsl"], default-features = false }` —
//!   custom backends are officially supported, no fork.
//! * Implement `wgpu::custom::{InstanceInterface, AdapterInterface,
//!   DeviceInterface, QueueInterface}` plus the Dispatch* types; enter via
//!   `wgpu::Instance::from_custom(..)`, recover handles via
//!   `Resource::as_custom::<T>()`. Proven by gfx-rs/wgpu's
//!   `examples/standalone/custom_backend` (wgpu 30).
//! * Shader execution is WGSL -> naga IR -> interpretation in Rust (option
//!   A): rio's sugarloaf shaders are simple glyph-atlas/SDF programs and the
//!   kernel's W^X policy (PROT_WRITE|PROT_EXEC -> EINVAL) makes the
//!   cranelift-JIT route (option B, wgpu-cpu's design) a later conversation
//!   about memfd dual-mapping.
//! * This module will render the same scene as `softrender` and the demo's
//!   acceptance test is that both paths produce the same frames (M3).

pub const STATUS: &str = "wgpu backend not wired yet (milestone M3) — \
see docs/runbooks/amd64-fbdev-wgpu-demo.md §5";
