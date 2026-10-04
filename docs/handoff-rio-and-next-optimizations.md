# Handoff prompt: build rio against the akuma wgpu backend, then a new optimization round

Paste everything below the line into a fresh session.

---

You are continuing work in /Users/netoneko/github.com/netoneko/akuma-cli-wgpu on branch
`jit-shader-executor` (clean, everything committed, nothing pushed). Read README.md fully first
(especially "Rules that are not negotiable", "The wgpu backend (M3)", "M4: the standard-mode GPU",
"rio-scale passes", "Things this kernel taught us", done-log Tasks 8-11, "Next steps"), then
docs/fbdev-wgpu-plan.md §5.

Rules: plain Linux only (no libakuma), tiny deps, small edits and rebuild after each, deterministic
rendering, never claim something works unless you ran it, no background agents without asking the
user, commit checkpoints with the Co-Authored-By trailer, do not push.

## Environment
- Dev host is aarch64 macOS. The JIT is x86-only, so anything JIT/wide/memo/runs related is tested
  ON THE BOX (`ssh akuma`, x86-64 Akuma "trashcan": 4 cores, SSE4.2, no AVX, no /proc/cpuinfo,
  `rg` and python3 absent there).
- `./deploy.sh` cross-builds musl and pushes /tmp/akuma-wgpu to the box over HTTP.
- sugarloaf shaders: clone raphamorim/rio somewhere temporary (already at
  <scratchpad>/rio if it still exists). On the box they live under /tmp/sl (grid/shaders/grid.wgsl,
  renderer/renderer.wgsl); point `AKUMA_SUGARLOAF` at <rio>/sugarloaf/src (host) or /tmp/sl (box).
- Tests that must stay green on host AND box after every change:
  `akuma-wgpu exec-selftest` (18 snippets x executors x +spec, plus random shaders:
  `AKUMA_FUZZ=n AKUMA_FUZZ_SEED=s`), `AKUMA_SUGARLOAF=... akuma-wgpu gpu-selftest` (21 tests; also
  `AKUMA_FUZZ=150` on the box), `akuma-wgpu selftest --w 1280 --h 720 --frames 120 --wgpu` and
  `--w 3840 --h 2160` (checksums in the README baseline table must be IDENTICAL), and
  `screensaver --wgpu --timeout 6` (~35 fps, present-bound; must not regress).
- Measure with `AKUMA_SUGARLOAF=... akuma-wgpu gpu-bench` (reports mean and best, verifies pixels),
  `AKUMA_PROF=1` for per-draw phase times; A/B switches: AKUMA_THREADS, AKUMA_WIDE, AKUMA_OPT,
  AKUMA_SPEC, AKUMA_MEMO, AKUMA_RUNS, AKUMA_SPIN_MS. Profile before optimizing; many guesses in this
  project were wrong (rdtsc counters around phases were what found the glyph vertex divergence).

## State (what exists, measured on the box, 3840x2144)
Full terminal-like redraw (grid bg + 60% glyphs) ~19 ms mean / 14.5 best (~50 fps); worst case
(every cell coloured + glyph) ~25 ms; bg pass ~5 ms. Machinery: opt.rs (optimizer, per-draw
specialization on constant buffer words, if-conversion), memo.rs (region memoization), runs.rs
(position-quantization run detection: x replication, y row copies), pool.rs (spinning worker pool),
raster.rs span path + narrow-triangle gather, SSE2 unorm8 encode/over-blend, wide JIT (4 lanes,
SSE4.1, uniform-branch-or-bail). Glyph pass is now most of a frame.

## Part 1 — build rio against this backend (the user's main goal)
Outside this repo's current scope until now; do it in a separate clone (e.g. the user's fork
netoneko/rio; do not vendor rio code into this repo). Steps, in order, each verified by running:
1. Decide the integration shape with the user if unclear: rio uses wgpu through sugarloaf. The
   cleanest seam is the same one this repo uses: `wgpu::Instance::from_custom(backend::Instance)`.
   Likely extract `src/wgpu_backend/` into a library crate (e.g. `akuma-wgpu-backend`) that this
   binary and a rio patch both depend on; keep the demo binary building and its checksums identical.
2. Make rio build for x86_64-unknown-linux-musl with only what it needs; find what pulls in winit /
   real GPU backends and what sugarloaf needs from `Surface`/swapchain.
3. A framebuffer platform: `rio-window` (screen = /dev/fb0 via src/fb.rs semantics, input = console
   tty/evdev), a `Surface` whose `get_current_texture` hands out a Bgra8 texture rendered in place and
   whose `present` copies whole rows into the WC mapping (see fb.rs: never scattered writes).
4. Known gaps to check against rio's real use (from README Next steps #3): swapchain semantics,
   Rgba16Float/HDR filter targets, Rgba8Snorm, mipmapped textures (filter chain), depth/stencil,
   multisampling (rio uses sample_count 1), the copy_texture_to_texture / set_viewport call sites,
   and the sugarloaf shaders not yet rendered under test (image.wgsl, text_shader.wgsl, filters).
   Add gpu-selftest cases for each gap you close, checked numerically.
5. Run it on the box, look at the panel (the user can watch /dev/fb0 and give feedback), record real
   frame times with AKUMA_PROF=1 for a typical screen (htop, vim, scrolling `cat`) at 4K.
6. A small viewer is also handy and was about to be written: `gpu-show` (animated grid scene drawn
   by sugarloaf's grid.wgsl straight to /dev/fb0 with a procedural seven-segment glyph atlas) so the
   rio-scale output can be eyeballed without rio. Optional.

## Part 2 — new round of optimizations (after rio runs; profile with rio's real frames first)
Candidates from the README, verify each with measurements:
1. Glyph pass: merge the two triangles of an axis-aligned quad into one rectangle fill (needs a
   gpu-selftest comparing rect path vs triangle path pixel for pixel); cheaper vertex input/output
   marshalling in `Invoker::run_vertex_batch` (input marshal ~250 cycles/batch, output extraction
   builds a 144-byte RawVertex per lane); hoist `PixelPlan::new`; avoid per-triangle copies of the
   128-byte varyings arrays.
2. Wide JIT quality: a register allocator (or at least block-local xmm caching with REX support)
   and/or 8-lane code (two register files, interleaved) for ILP; the template JIT is load-op-store.
3. Pool/scheduler: mean vs best differs by ~5 ms because of Akuma's tick-granular wake-ups and
   thread placement (sched_setaffinity returns -1, sched_yield in the spin loop made it WORSE).
   Ideas: fewer parallel phases per draw (fuse vertex + raster into one job), adaptive thread count.
4. Whatever rio's real frames show: other sugarloaf shaders (rounded-rect fragment shader with
   discard, image/filters), damage-based redraw, texture upload paths, `copy_texture_to_texture`.
5. The demo's legacy path (`backend.rs::draw_legacy`) is single-threaded scalar; the live demo is
   present-bound (~11 ms for the 33 MB WC copy), so low priority.
Kernel-side findings to pass to the kernel repo are in README "Things this kernel taught us".

## Report at the end
Commands run with real output, files changed, per-step before/after numbers, and what you could not
verify. Update the README (measured numbers, what changed, what you could not verify) as you go.
