# akuma-cli-wgpu

A 3D screensaver of the Akuma cat logo that draws straight into the framebuffer. It is the first
user of a Linux-standard framebuffer device (`/dev/fb0`) on the Akuma amd64 kernel. The real goal
behind it is to run the **rio terminal** on Akuma's screen later, through a wgpu backend that
renders into that same framebuffer. The wgpu-backend half of that exists now
(`src/wgpu_backend/`, milestone M3 — see "The wgpu backend (M3)" below); rio is still ahead.

This README is the brief for whoever works on this repo next. If you are an agent, read all of it
before changing anything.

## Rules that are not negotiable

1. **Plain Linux only.** This is an ordinary `x86_64-unknown-linux-musl` program using `std` and
   the `libc` crate. **Never use `libakuma`** or any Akuma-private syscall. rio and wgpu will run
   on this substrate without knowing Akuma exists, so this demo must too.
2. **Keep dependencies tiny.** Three, no more: `libc`; `wgpu` with
   `default-features = false, features = ["std", "custom", "wgsl"]` (no real GPU backend
   gets built); and `naga` with `wgsl-in` only (our half of shader parsing on the custom
   path — see `Cargo.toml`). It all has to build with the nightly musl toolchain that runs
   *on the Akuma box itself*. Do not add `clap`, `rand`, `crossterm` or similar; the code
   has its own replacements on purpose.
3. **Standalone crate.** `Cargo.toml` has an empty `[workspace]` table on purpose. Do not join
   any other workspace.
4. **Small edits, rebuild after each one.** Two sessions in a row have lost code to large edits
   (see the done-log below: one ate a function signature, another silently dropped the z-buffer
   refill the module doc promised). Change one function at a time, run `cargo build`, and read
   the error before the next edit.
5. **Rendering must be deterministic.** The selftest checksums are the M3 acceptance test, so
   nothing on the render path may seed from the wall clock, addresses, or any other entropy.
   All rng goes through `Rng::with_seed` (`src/rng.rs`); `Scene` uses the fixed `RAIN_SEED`, and
   per-frame rain stepping seeds from the frame time. Do not reintroduce `Rng` seeds from
   `SystemTime` or pointer values.

## Repo state note (2026-10-03)

The `.git` directory in this checkout was found **empty** (no `HEAD`) on 2026-10-03, mid-session;
`git log` had worked at the start of that same session. The worktree is intact. Until this is
resolved, treat GitHub (`netoneko/akuma-cli-wgpu`) as the only source of history, and do not
assume local git operations work.

## Where things are

| what | where |
|---|---|
| this repo, on the box | `/src/github.com/netoneko/akuma-cli-wgpu` (GitHub: `netoneko/akuma-cli-wgpu`) |
| the kernel repo, on the box | `/src/github.com/netoneko/akuma` |
| **the plan** (read it) | [`docs/fbdev-wgpu-plan.md`](docs/fbdev-wgpu-plan.md) in this repo — a copy of `docs/runbooks/amd64-fbdev-wgpu-demo.md` from branch `cats/meow/fbdev-wgpu-plan` of `netoneko/akuma-litter` |
| the fbdev C probe | [`probes/fbprobe.c`](probes/fbprobe.c) — build command in its header comment |
| the kernel's framebuffer code | `amd64/src/multiboot2.rs` (`kmain_mb2`, `map_wc`), `crates/akuma-fbcon/`, `docs/runbooks/amd64-console-shell.md` |
| the `/dev` device model to copy | `FileDescriptor::DevDsp` (`/dev/dsp`) in `crates/akuma-syscalls-glue` |

When a comment in `src/` says `docs/runbooks/amd64-fbdev-wgpu-demo.md`, read
`docs/fbdev-wgpu-plan.md` here; it is the same document. Paths *inside* the plan (`amd64/…`,
`crates/…`) are in the kernel repo.

## Working on the box

The box runs Akuma, a hobby kernel with a Linux x86-64 ABI. A few things differ from a normal
Linux machine:

```sh
. /etc/akuma-dev.env                 # PATH, HOME and the Rust toolchain; do this first in every shell
cd /src/github.com/netoneko/akuma-cli-wgpu
cargo build --release --target x86_64-unknown-linux-musl
./target/x86_64-unknown-linux-musl/release/akuma-wgpu selftest
```

- Read files with `cat`, `sed -n 'A,Bp'` or `grep -n`. **`rg` is not installed.** There is no
  `python3` on `PATH`. The shell has no process substitution (`<(...)` fails with
  `/dev/fd/…: No such file`); use temp files.
- The binary is called `akuma-wgpu` (the package is `akuma-cli-wgpu`).
- `/dev/fb0` **exists now** (it appeared mid-session 2026-10-03; char dev 29:0). `screensaver`
  and `matrix` run on metal: ~46-50 fps at 4K including the present pass (the selftest math
  predicted ~45), write rate 1.5-1.8 GB/s effective. `selftest` needs no device and remains
  the test you can always run.
- `selftest --wgpu` runs the same headless test through the wgpu backend (M3). Its rain
  backdrop is bit-identical to the software path (shared CPU code), but the **mesh is not**
  yet — see "The wgpu backend (M3)" — so its checksums do not match the baseline table.
- SIGINT/SIGTERM kill the process immediately with exit status 0 — see the kernel
  signal-delivery bug in the done-log (a handler returning to userspace is not survivable on
  this kernel yet). There is **no metrics report on a signal death**; use `--timeout N` for a
  graceful self-terminated run with the full report. `q`/Esc also exit gracefully.
- It also builds on a Mac or Linux laptop with `cargo build --release` for the host, and
  `selftest` runs there too.
- `selftest` measures **render only**. On metal, each frame additionally pays `present` — a
  33 MB copy into the WC mapping, ≈11 ms at the kernel's 3026 MB/s — so real 4K frames land
  around 22 ms ≈ 45 fps. Fine for a screensaver; do not expect selftest fps on the trashcan.

## What exists

`akuma-wgpu <screensaver | matrix | selftest> [options]`, modelled on `akuma-cli` (an
ANSI-terminal screensaver with the same subcommands, `--latin`, arrow keys to switch the logo,
`q`/Esc to quit, and a metrics report on exit).

| file | what it does |
|---|---|
| `src/main.rs` | argument parsing, the render loop (`run_show`), `selftest` (+ `--dump`: `dump_ascii`, an ~80×40 luminance-ramp ASCII picture of the last frame), the exit report |
| `src/fb.rs` | the fbdev client: `open("/dev/fb0")`, `FBIOGET_VSCREENINFO`/`FBIOGET_FSCREENINFO`, `mmap`, pixel-format packing, `present` (row-wise copy of a RAM `Frame` into the mapping). The fbdev structs are declared by hand to match `<linux/fb.h>` byte for byte |
| `src/catlogo.rs` | turns the ASCII cat (`akuma_*.txt`, density ramp ` .:-=+*#%@`) into a height field and extrudes it into a 3D triangle mesh |
| `src/softrender.rs` | the software rasterizer: rotate → perspective project → cull → shade (solid two-tone logo colors — flat cyan front, flat purple walls) → z-buffered spans; plus the Matrix-rain backdrop. The z-buffer is refilled with +inf at the top of `render()` **every frame** — this refill is load-bearing (see done-log) |
| `src/rng.rs` | xorshift64* with a splitmix64-seeded `with_seed` constructor; the render path never touches wall-clock or ASLR entropy (rule 5) |
| `src/input.rs` | termios raw mode, `poll(2)` and escape-sequence decoding; SIGINT/SIGTERM exit the process immediately (kernel signal bug — see done-log) |
| `src/clock.rs` | `Pacer` (sleeps out the frame budget) and `FpsMeter` (live fps, over-budget count, slowest frame) — EINTR-tolerant `CLOCK_MONOTONIC`/`nanosleep`, never std `Instant` (kernel note in `input.rs`) |
| `src/wgpu_backend/` | **milestone M3, the wgpu custom backend — exists now.** `mod.rs`: `WgpuRenderer`, the exact `new(w, h)`/`render(&mut frame, &mut scene, time, with_rain)` surface of the software `Renderer`, so `--wgpu` swaps paths in one place. `backend.rs`: the `wgpu::custom::*` device/adapter/queue plus the fixed-function rasterizer under contract to mirror softrender's scanline walk. `interp.rs`: the naga-IR interpreter — no JIT *yet* (plan §5b A; a JIT is feasible, see "Next steps"). `shaders.rs`: the WGSL, line-for-line ports of softrender's per-frame math. Status: runs, rain identical, **mesh not yet frame-identical** — see "The wgpu backend (M3)" |
| `src/wgpu_backend/{opt,memo,runs,pool}.rs` | the performance machinery added in Task 11 (see "rio-scale passes"): program optimizer + per-draw specialization + if-conversion; region memoization; run detection; the spinning worker pool |
| `src/akuma_{20,40,79,120}.txt` | the logo at four sizes; `akuma_40.txt` is byte-identical to the kernel's boot-banner asset |

Composing each frame in ordinary RAM and copying whole rows into the mapping is deliberate. The
kernel maps the framebuffer write-combining, which is fast for full-row copies (~3000 MB/s on the
box) and slow for scattered writes (~71 MB/s). Keep that property in anything you change.

## The wgpu backend (M3) — acceptance MET (2026-10-04)

`--wgpu` renders any mode through the custom backend instead of the software rasterizer:
`wgpu::Instance::from_custom(backend::Instance)` (wgpu 30, `features = ["custom", "wgsl"]`, no
real GPU backends in the build). WGSL is parsed by naga into IR; rasterization is fixed-function
in `backend.rs`, under contract to mirror softrender's scanline walk exactly (same `mul_add`
area test and `-0.01` cull, same edge/z interpolation, strict `z <` depth test).

**Frames are bit-identical to the software path** on the trashcan: same `fnv1a` for all four
assets at 256x144, 1280x720 and 3840x2160 (120 frames), under every executor below. The
earlier mesh defect was the triangle storage buffer being packed without WGSL's 16-byte
`vec3` alignment (fixed in `e20359c`).

### Shader executors (`src/wgpu_backend/`)

A pipeline stage is run by one of three executors behind `exec.rs::Stage`; the rasterizer never
knows which:

| executor | what | where it lives |
|---|---|---|
| **jit** (default on x86-64 Linux/Akuma) | machine code emitted from the register program; `mmap` RW → write → `mprotect(R+X)`, never a W+X page (the kernel refuses those, but a JIT does not need one — `userspace/jitprobe` in the kernel repo proved it on the trashcan) | `jit.rs` |
| **vm** | bytecode interpreter over the same register program; portable, no executable memory | `vm.rs` |
| **interp** | the original naga-IR tree walker; handles everything the backend supports, slow; the fallback when the lowering declines a shader | `interp.rs` |

`compile.rs` lowers naga IR **once, at pipeline creation**, into `program.rs`'s `Program`: a flat
register machine over 32-bit words (vectors/matrices/structs/arrays scalarized away, user
functions inlined, locals as mutable registers, everything else single-assignment; transcendentals
and Rust-vs-CPU-divergent ops go through shared `extern "C"` helpers, so VM and JIT agree by
construction). Anything it does not understand returns `Err` and the stage silently uses the
interpreter — the compiler can grow feature by feature without ever being a correctness gate.

Around the lowering sit four more passes/helpers, all `Program → Program` or pure helpers so that
the VM stays the oracle for them: `opt.rs` (optimizer + per-draw specialization), `memo.rs`
(region memoization), `runs.rs` (run detection, see "rio-scale passes"), `pool.rs` (the worker
pool). Switches: `AKUMA_OPT=0`, `AKUMA_SPEC=0`, `AKUMA_MEMO=0`, `AKUMA_RUNS=0` turn each off
(for A/B timing and for differential tests), `AKUMA_MEMO_BITS=n` sizes the memo table,
`AKUMA_DUMP=<entry>` (with `shader-check`) and `AKUMA_DUMP_SPEC=1` print programs.

Tools and switches:

* `akuma-wgpu exec-selftest` — **the differential test.** 18 WGSL snippets, each also run
  specialized to its buffers (`+spec`), (float/int/bit ops,
  casts, comparisons with NaN, transcendentals, aggregates, swizzles, if/loop/switch, locals,
  function calls, uniform + dynamic buffer indexing, matrix products) run through interp, VM and
  JIT over 300 vertices fed pseudo-random data including NaN/±inf/±0; results must be
  bit-identical. 17/17 on the trashcan. Writing it found eight interpreter bugs (see done-log
  Task 8). Run it after touching `compile.rs`, `vm.rs`, `jit.rs` or `interp.rs`.
* `akuma-wgpu shader-check <file.wgsl>...` — per entry point: does it lower, does it JIT, and if
  not, why. Point it at real shaders.
* `AKUMA_EXEC=interp|vm|jit` forces an executor; `AKUMA_EXEC_VERBOSE=1` reports each stage's
  choice and why a fallback happened.
* `AKUMA_PROF=1` prints a per-phase breakdown (backdrop, upload, vertex, raster, readback);
  `AKUMA_PROF=2` adds per-fragment timers — each is a syscall on Akuma, so that level distorts
  the numbers around it; use it only for fragment counts.
* `./deploy.sh` cross-builds `x86_64-unknown-linux-musl` on the dev machine and pushes the
  binary to the trashcan over HTTP (`/tmp/akuma-wgpu`; the box has `wget`, no `scp`). Needs
  `x86_64-linux-musl-gcc` (musl-cross) and `ssh akuma` working.

### Measured on the trashcan (2026-10-04, 120 frames, render only)

| config | software | wgpu: interp | wgpu: vm | wgpu: jit |
|---|---|---|---|---|
| 1280×720 | 1.25 ms | 359 ms | 10.3 ms | **7.2 ms** |
| 3840×2160 | 9.9 ms | (seconds) | — | **43.8 ms** |

Where the wgpu 4K frame goes now (jit): ~25 ms is demo-integration plumbing around full-frame
33 MB buffers (backdrop, texture upload memcpy, depth fill, `copy_texture_to_buffer`, readback
blit), ~17 ms is raster + fragment (533k flat-colored fragments, ~33 ns each), 2 ms is the vertex
stage (10.8k invocations at ~200 ns). The vertex stage was 227 ms/frame at 256x144 before the
compiler. Of the vertex time that remains, much is `sin`/`cos` of per-frame uniforms recomputed
for every vertex — hoisting uniform-only computation out of the per-invocation path is the
obvious next win for the executor itself.


## M4: the standard-mode GPU — what rio/sugarloaf needs (2026-10-04)

The M3 backend had a private contract (pixel-space positions, flat varyings, one raw
`Rgba8Uint` target) that is exactly right for the demo's bit-exact acceptance and wrong for any
real wgpu client. M4 adds the real thing **beside** it: a pipeline whose target is anything but
`Rgba8Uint` takes the *standard* path (`PipelineData::legacy()` decides); the demo path is
untouched and its checksums still match softrender.

| file | what |
|---|---|
| `raster.rs` | clip (0≤z≤w + x/y guard band) → perspective divide → viewport (y flipped) → 1/256-pixel fixed point → winding cull → i64 edge-function fill with the **top-left rule** → flat / linear / perspective-correct varyings → depth test → fragment stage → blend → write mask → format encode. Fragments are bounded by viewport ∩ scissor ∩ target |
| `format.rs` | `R8`, `Rg8`, `Rgba8`, `Bgra8` unorm (+ sRGB variants), `Rgba8Uint`; decode/encode |
| `vertex.rs` | vertex-buffer attribute fetch (Float32xN, Uint/Sint 8/16/32, Unorm/Snorm 8/16, defaults (0,0,0,1)) |
| `texture.rs` | `textureLoad`, `textureSample*`, `textureDimensions`: nearest/bilinear, clamp/repeat/mirror/border, sRGB decoded before filtering; level 0 only (one mip) |
| `backend.rs` | pipeline state capture (targets, blend, topology, cull, vertex layouts), vertex/index buffers, viewport, scissor, blend constant, `draw`/`draw_indexed`, instancing, triangle lists and strips, `copy_buffer_to_texture` / `copy_texture_to_texture` / origins in `write_texture` |
| `compile.rs` | now also: texture/sampler globals, dynamic indexing of local arrays, fragment interpolation modes |

`akuma-wgpu gpu-selftest` drives all of it through the real wgpu API and checks pixels against
hand-computed values — 16 tests, all passing on host (VM) and on the trashcan (JIT): fullscreen
triangle, **watertight shared edge** (a quad's two triangles cover every pixel centre on the
diagonal exactly once, top-left rule at the boundary), linear and **perspective-correct**
varyings, **instancing** with `Float32x2` + `Uint32x2` + `Unorm8x4` + `Sint16x2` attributes on
a triangle strip, back-face culling, premultiplied-alpha blending, scissor and viewport, target
formats, `textureLoad`, nearest/bilinear/repeat sampling, sRGB decode, texture copies — and
**sugarloaf's real `grid.wgsl`**: its cell-background pass (160-byte uniforms, storage-buffer
cells, colour-space maths) and instanced glyph pass (7 attributes in 4 formats, mat4
projection, atlas `textureLoad`), through the same pipeline layout rio uses, pixel-correct
(±1) in Bgra8Unorm. Run it with `AKUMA_SUGARLOAF=<rio>/sugarloaf/src` (the shader is read from
rio's checkout, not vendored here). `shader-check` on all of sugarloaf's WGSL:
**15 of 15 entry points lower and JIT.**

### Performance work, 2026-10-04 (fps), and what is left

Demo, the trashcan, 3840×2160, JIT, bit-identical to softrender throughout:

| | start of the day | now |
|---|---|---|
| wgpu path render (selftest) | seconds/frame (254 ms already at 256×144) | **14 ms** (software 10 ms) |
| live `screensaver --wgpu` on `/dev/fb0` | < 1 fps | **35.6 fps** (software 46.7) |

What got it there (each step measured, checksums unchanged): compile + VM + JIT (the executors
section); per-frame full-buffer copies cut (render **in place** into the frame's memory —
the colour texture's storage is pointed at `frame.buf` for the pass, so there is no upload,
readback copy or blit); unread stage inputs pruned so a flat position-independent fragment
shader runs once per triangle (raster 18.6 → 4.6 ms); a `memcpy` call per 4-byte texel store
replaced by fixed-size stores (musl's `memcpy` is not free for tiny sizes: 25 → 8 ns/fragment);
render-pass clears as a doubling `copy_within` instead of a per-pixel loop (4K empty pass 128 → 7 ms —
yes, "just replace it with a new canvas" was the right instinct); transcendental result caches.

#### rio-scale passes: `gpu-bench` (3840×2144, sugarloaf's real `grid.wgsl`)

`akuma-wgpu gpu-bench` with `AKUMA_SUGARLOAF=<rio>/sugarloaf/src`. "Colourful" gives every cell
its own colour and a glyph (the worst case, 16k cells, 16k glyph instances); "terminal-like" is
a dark background with runs of three highlight colours and glyphs in 60% of the cells.

| pass | start of session 2 | now (mean, best of 24) |
|---|---|---|
| cell backgrounds, 4 threads | 211–223 ms | **5 ms (3.5)** |
| cell backgrounds, 1 thread | 784 ms | 10.6 ms |
| bg + a glyph in every cell | 303–320 ms | **25 ms (21)** |
| terminal-like, backgrounds | — | **4–5 ms (2.8)** |
| terminal-like, backgrounds + 60% glyphs (a full redraw) | — | **19–21 ms (14.5) ≈ 50 fps** |
| terminal-like full redraw, 1 thread | — | 33 ms |
| trivial position-dependent shader, whole 4K | 87.6 ms | 19.8 ms |
| constant-colour fill, whole 4K | 29–34 ms | 3.4–8 ms |

`gpu-bench` verifies the rendered pixels of every scene (cell backgrounds at four corners of each
cell) before it prints a time, so the numbers cannot come from a broken fast path; it also works with
`AKUMA_THREADS=1`, `AKUMA_RUNS=0` etc. for A/B runs (runs off: bg pass 36 ms, terminal redraw 50 ms).

The mean–best gap (~5 ms) is the kernel scheduler: whether the four pool threads actually get four
CPUs for a given phase varies from frame to frame (see "Things this kernel taught us").

The target was a full redraw under ~50 ms; it is at ~15–20 ms. Where it came from, in the order
it was built (each step measured on the box; `exec-selftest`, `gpu-selftest` and the demo's
`selftest --wgpu` checksums stayed green at every step):

1. **Exact per-row coverage spans + a straight-line "span" path** (`raster.rs::fill`,
   `fast_span`). The per-pixel inside test became an interval per row (each edge function is
   linear in x, so each edge bounds the interval from one side — same top-left rule, same
   pixels). For wide stages the covered pixels are shaded four at a time with *no* gather,
   marshalling or format dispatch: the input registers are written directly, results are
   encoded with SSE2 (`format::encode4_unorm8`, bit-identical to the scalar `encode`) and stored
   as one 16-byte unit. Varyings are evaluated as per-triangle planes. Premultiplied / straight
   "over" blending is vectorized too (`blend_over4_unorm8`, bit-identical, tested against the
   scalar definition on random destinations). 87.6 → 19.9 ms for a trivial shader, 211 → 148 ms
   for the grid pass.
2. **A program optimizer** (`opt.rs`): constant and copy propagation, constant folding (done by
   running the VM on the instruction, so folded values are bit-identical to what any executor
   computes), constant-branch folding, unreachable-code removal, jump threading, dead code by
   liveness. Run at pipeline creation (384 → 304 instructions for the grid fragment shader) and
   again **per draw, specialized to the draw's buffers**: every load at a constant offset from a
   bound buffer (uniforms: padding, cell size, colour-space flags, cursor) becomes a constant,
   which folds the colour-space branches, the cursor overlay and the padding-extension logic
   (302 → 100 instructions, 1.4 ms to build on the box). Specializations are cached by the
   *values* of the words they folded, so a new frame with unchanged uniforms reuses the machine
   code; a changed uniform recompiles. Only draws big enough to repay it are specialized
   (estimated fragments × program length ≥ 2·10⁷). Vertex stages are specialized the same way.
3. **Region memoization** (`memo.rs`, `MemoGet`/`MemoPut`): the grid shader's colour maths is a
   pure function of the 32-bit cell word it loaded. The pass finds a single-entry/single-exit
   region whose live-in registers are few, never rewritten and not stage inputs, and brackets it
   with a lookup in a 1024-entry direct-mapped cache kept in the persistent register file
   (private to the invoker/thread). Keys loaded from memory are preferred over keys computed
   from the position. The lookup is inlined in the wide JIT (single key); other shapes call
   `program::memo_get`, so VM and JITs agree by construction. 148 → 113 → 69 ms (with the
   optimizer) → 40 ms.
4. **Run detection** (`runs.rs`) — the big one. If a fragment program's dependence on
   `@builtin(position).x` flows only through quantization (`floor((x - pad) / cell_w)` and the
   like), the result is constant over runs of pixels. A conservative per-axis classification
   (clean / affine chain / step / bad) proves it; at run time the exact end of the run is found by
   re-evaluating the affine chain with the program's own f32 operations (exponential + binary
   search; all f32 ops are monotone). Horizontally the result is replicated over the run;
   vertically a row identical in its quantized y is *copied* from the row above (only the few
   pixels at a slanted edge are shaded). Applies when the pixels were produced by plain stores
   (no blending with the destination, no discard). 40 → 5 ms for the background pass.
   `gpu-selftest` renders cell grids with awkward sizes/offsets (7.3×13.9 cells at half-pixel
   offsets, a slanted two-triangle quad, ...) with runs on and off and demands identical bytes —
   and asserts the mechanism really was exercised for the grid shader and not for one that
   leaks the position into the colour.
5. **Everything around the shader**: a lazy render-pass clear folded into the first draw's per-band
   work (the band is still in cache; one trip to memory instead of two); primitives binned per
   band (every band used to set up every triangle); the vertex stage parallelized in recycled
   chunks; constant-colour spans filled directly; a wide `textureLoad`; xmm0 forwarding and an
   inlined, broadcasting `LoadBuf` in the wide JIT.
6. **A persistent worker pool** (`pool.rs`). Measured on this kernel (see below): creating a
   thread costs ~1.5 ms, and a *woken* thread only gets a CPU at the next scheduler tick
   (~4 ms) because idle CPUs are not kicked. `std::thread::scope` per phase therefore cost ~8 ms
   per draw. Workers are now started once, spin for 50 ms after the last job (picking one up in
   microseconds) and only then sleep; `begin_render_pass` calls `pool::warm()` so the first
   parallel phase of a pass finds them spinning.

7. **If-conversion** (`opt.rs::ifconv`). The wide JIT runs four pixels/vertices per instruction
   with *uniform* control flow: a branch whose lanes disagree makes the batch bail out to
   one-lane-at-a-time. WGSL's short-circuit `a || b` is lowered by naga into a branch and a
   temporary, and sugarloaf's glyph vertex shader has `vid == 1u || vid == 3u`, which the four
   vertices of a quad *always* disagree on: **every vertex batch bailed** (found with an rdtsc
   probe: 9648 of 9648 batches diverged). Small `if`/`else` diamonds whose arms are straight-line
   pure code now execute both arms and `Select` the results (arms ≤ 16 instructions; no texture
   fetches or result-cached libm calls, so speculation is safe). Vertex stage 8.2 → 4.3 ms
   (single thread), no divergence left in the grid shaders.
8. **Gathering narrow triangles** (`raster.rs::flush_gather`): a 6-pixel-wide glyph quad leaves
   half of every four-lane batch empty if each row is its own batch; the covered pixels of
   successive rows are now gathered into full batches and scattered to their texels
   (per-lane planes, per-lane `@builtin(position).y`). Glyph raster ~13 → ~9 ms.

What is left, honestly: the **glyph pass** is still most of a frame. Per glyph quad ~3 µs of CPU for
~50 pixels — two triangles' setup, ~12 four-pixel batches with a texture fetch and an alpha
blend each, and a vertex stage that spends ~100 cycles/vertex mostly outside the JIT'd code (input
marshalling 250 cycles/batch, output extraction). Ideas, in order: merge the two triangles of an
axis-aligned quad into one rectangle fill; cheaper vertex input/output marshalling (SSE transposes
instead of per-lane `match`); 8-lane (two-register-file) JIT code for ILP and a real register
allocator for the wide JIT (it is still load-op-store, ~5 cycles per instruction on the critical
chain — the hit path of the memoized grid shader costs ~200 cycles/batch before run detection
removed most of it). Run detection does not help the glyph shader (an atlas lookup at an affine
coordinate is not a quantization).

#### Things this kernel taught us (for the kernel repo)

* **Thread wake-up latency is a scheduler tick.** `thread-probe` (pool rounds): a freshly woken or
  created thread starts ~4.3 ms late if its CPU is busy and stays that way until balanced;
  four spinning workers converge to four CPUs after ~3 rounds. `sched_setaffinity` returns -1.
  Idle CPUs are evidently not sent an IPI when a thread becomes runnable.
* **Thread creation is ~1.5 ms.**
* **First-touch page faults cost several µs each** (a 2.3 MB `Vec` filled by a worker was 3–5 ms),
  so per-draw temporaries (vertex output, attribute arrays) are recycled or kept small.
* Memory write bandwidth tops out around 5–10 GB/s: a 4K clear is 3–6 ms no matter how many threads.

## Baseline: selftest checksums and timings (re-measured 2026-10-04)

These are the acceptance numbers for M3: the wgpu path must reproduce these frames (same
`fnv1a` per asset). They are stable across runs on the trashcan (nightly
`1.100.0-nightly (420ed2a0c)`, release profile). If you change rendering intentionally, expect
these to change — update this table and say why in your report.

| config | akuma_40 | akuma_79 | akuma_120 | akuma_20 | avg render |
|---|---|---|---|---|---|
| 1280×720, 120 frames | `9c5f8a2d` | `3e6005a8` | `1a65844b` | `3ec5edbf` | 1.25 ms/frame (~799 fps) |
| 3840×2160, 120 frames | `3faaed4e` | `ef795024` | `eefd7b18` | `8ef751d6` | 9.89 ms/frame (~101 fps) |

**The table above was stale** — the 2026-10-03 solid-two-tone-logo-colors change (`e20359c`) altered every
frame's bits and the previous values were never re-measured. These are measured on the trashcan
2026-10-04 and **the wgpu path (`--wgpu`, every executor) produces the same checksums**: that is the
M3 acceptance, met.

**Changed 2026-10-03 twice — intentional.** (1) Matrix rain rework (done-log Task 4): the
backdrop went from 1-px streaks to cell-based Matrix columns. (2) Rain packed to a column per
cell column (Task 5) — the user asked for a denser, always-present matrix. Every frame's bits
changed; the table was re-measured on the box after each change. (3) Task 7 fixed the frozen
live rain (see done-log): column count now matches between builder and resizer, so the show
loop no longer rebuilds the rain every frame — positions and per-column speeds are back. The
mesh math is untouched
(`Scene::center_x` defaults to mid-frame in selftest), so these remain the
renderer-equivalence target.

Re-verified 2026-10-03 after the fb.rs u16 fix, the clock hardening and the signal-handler
change: **checksums identical**, `SELFTEST OK` on every build.

Where the 4K render milliseconds go (probe measurements): the two 33.2 MB fills — `frame.clear`
(~3.5 ms) and the z-buffer refill (~3.4 ms) — are ~64% of the frame; the raster is the rest
(~1 ms for the 442-tri `akuma_20` up to ~2.8 ms for the 8538-tri `akuma_120`); rain is no
longer free since the cell rework but stays small (the 4K average moved 10.7 → 10.8 ms/frame,
so the new rain costs ≲0.15 ms). Do not "optimise" the z refill away — see the done-log for
what its absence does.

## Done-log

### Task 1 — make it build again — DONE 2026-10-03

- Restored the `print_metrics` signature exactly as this README specified (body had survived).
- Wrote `dump_ascii(&frame)` (`src/main.rs`): ~80×40 cells, mean BT.601 luminance per cell on
  the ` .:-=+*#%@` ramp; `BG` maps to `' '` with no special case. `selftest --dump` shows the
  cat for all four assets.
- **Found and fixed a third breakage this README did not list:** `Renderer::render` never
  refilled the z-buffer, though the module doc promised a per-frame refill. Stale depths from
  frame N−1 reject the rotating mesh pixel-by-pixel; coverage erodes every frame and the
  big-grid assets die first (`akuma_79`/`akuma_120` rendered 0.0% coverage — blank dumps, and
  checksums of blank frames). Fix: `self.z.fill(f32::INFINITY)` at the top of `render()`.
  Proven with a throwaway probe: same mesh + rasterizer with rain off renders healthy 5%
  coverage; over 30 frames with stale z it erodes steadily; with the refill it is stable.
- Build: 0 errors, 0 warnings; `selftest` ends with `SELFTEST OK`; coverage 5.1–6.0% on all
  four assets.

### Task 2 — check the selftest numbers mean something — DONE 2026-10-03

- **Checksums were not deterministic as shipped**: the render path seeded rain from
  `CLOCK_REALTIME` xor a stack address (ASLR), re-seeded per frame. Fixed via
  `Rng::with_seed` (rule 5 above). Two consecutive runs now print identical `fnv1a` values at
  720p and at 4K (see baseline table). The live show still varies frame to frame, because the
  frame time never repeats; rendering is a pure function of (scene, time).
- Timings measured (see baseline table). 60 fps is met at 4K with ~6 ms of render headroom;
  the present pass changes that on metal (see "Working on the box"). No optimisation was done,
  per the task's instructions.

### Task 3 — a raw C probe for `/dev/fb0` — DONE 2026-10-03

- `probes/fbprobe.c`, exactly as specified: open `/dev/fb0`, print every field of both
  `FBIOGET_*` structs (hand-declared ABI, `_Static_assert` size checks 160/80, so no kernel
  headers needed anywhere), `mmap` `MAP_SHARED` RW, fill a self-identifying gradient
  (word i = `0xFF000000|i`) timed with `CLOCK_MONOTONIC`, read every word back and compare
  (prints sampled pixels too), exit 0 only on an all-pass verdict.
- `/dev/fb0` appeared on the box mid-task, so the probe ran against the real thing — PASS
  twice: fill 35,389,440 B in 12.35/12.36 ms = **2866/2863 MB/s** → the user mapping is
  write-combining (kernel S2 works; ~71 MB/s would have meant UC). Read-back
  8,847,360/8,847,360 words match (S5 device mmap works); reads run ~4 MB/s, expected from WC
  memory. Struct dump matches plan §2: `id="akuma-fb"`, 3840x2160, 32 bpp 8/8/8 (r16 g8 b0),
  TRUECOLOR, PACKED_PIXEL, `line_length` 16384, `smem_len` 35389440.
- Build: host `gcc -O1 -static -Wall -Wextra`, 0 warnings. `x86_64-linux-musl-gcc` is **not
  installed** on the box, so the exact build line from the task is unverified; the musl gcc
  also is not needed for correctness (no headers, no libakuma), but run it once when it
  exists. Calibration against a real Linux control box with a framebuffer: **not done**
  (none available from here).

### Found while running Task 3 — fb.rs ABI fix + kernel signal bug + clock hardening

- **fb.rs ABI bug, FIXED.** `FbFixScreeninfo` declared `xpanstep`/`ypanstep`/`ywrapstep` as
  `u32`; the Linux ABI (`<linux/fb.h>`, and the C probe's declaration) uses `__u16`. The u32s
  pushed `line_length` to offset 52 — where the real ABI keeps alignment padding — so the
  demo would have read pitch 0 the moment S4 landed. Now `u16`, byte-identical to the probe.
  Rendering untouched: selftest checksums identical to the baseline table, re-verified after
  every change below.
- **First on-metal runs of the demo** (possible because `/dev/fb0` exists): `[fb]` line
  correct (`akuma-fb 3840x2160 pitch 16384 bpp 32`), **46-50 fps at 4K including present**
  (selftest's math predicted ~45), write rate 1.5-1.8 GB/s effective. A 90-second run and a
  60-second run with no signals: no failures.
- **Kernel bug — signal delivery (kernel-repo work; documented here because the demo had to
  work around it).** Symptoms, before the workaround: every SIGTERM'd run died as a
  `clock_gettime` EINTR abort (`Instant::now` unwrap), an impossible out-of-bounds index
  (softrender.rs:359: len 8294400, index ~2.73e8 — unreachable through the clamped code for
  any f32 input), or a silent segfault; no-signal runs never crashed. Probes run on the box:
  (a) `sigprocmask(SIG_BLOCK)` with no handler → TERM truly silent, process survives; (b) the
  same block with `signal(SIGTERM, h)` installed → the in-flight syscall EINTRs anyway (mask
  verified set) — the mask gates handler invocation, not the EINTR; (c) 172 mid-SSE-loop
  deliveries restore xmm/GPR state perfectly; (d) `sigpending`/`sigtimedwait`/`signalfd` are
  ENOSYS; `ppoll` exists, honors its mask, but does not deliver pending signals; (e) with the
  clock EINTR-hardened, remaining TERMs died as SEGV_MAPERR at definitely-mapped VAs (fb map
  `0x40037xxx` x3, heap `0x10xxxxxx`), instantly, with the handler having run. Reading: the
  EINTR is one symptom; the address space itself is not intact across a delivery that returns
  to userspace. **Workaround in this repo:** the INT/TERM handler `_exit(0)`s immediately —
  delivery → handler → exit never returns to the broken continuation. 10/10 direct TERMs,
  matrix TERM, and `--timeout` self-exit all clean afterwards; checksums unchanged. Kernel
  repo should look at: EINTR-on-masked-signals, the delivery/return address-space corruption,
  missing sigpending/sigtimedwait/signalfd, the interval timer not reloading (one delivery
  per setitimer), and a ucontext layout that differs from musl's (RIP read as 0 from the
  documented musl offsets).
- **clock.rs hardened:** `monotonic()`/`sleep_until()` replace std `Instant`/`thread::sleep`
  on the render path — raw `CLOCK_MONOTONIC`/`nanosleep` that retry EINTR instead of
  unwrapping it (std's `Instant::now` aborts on EINTR; that was the original crash signature).
  selftest numbers unchanged.

### Task 4 — docs catch-up on the wgpu backend, a Matrix that reads as Matrix, logo on the living half of the panel — DONE 2026-10-03

- **Docs caught up with the wgpu backend (M3), which a previous session built without
  updating this README** (`src/wgpu_backend.rs` "placeholder" → the real
  `src/wgpu_backend/{mod,backend,interp,shaders}.rs`; rule 2's dependency list; a status
  section "The wgpu backend (M3)"; `main.rs` USAGE and `Options::wgpu` no longer claim "not
  wired yet"; `docs/fbdev-wgpu-plan.md` got a status banner). Documenting included *running*
  it: the rain is bit-identical across paths, the mesh is **not** — pre-existing defect,
  details in the M3 section, not debugged in this task.
- **Matrix rain rework** (asked for: it "looked like shooting stars, not a matrix"). The old
  rain drew 1-px-wide, 4–17-px streaks — shooting stars at 4K. Now the template's
  character-grid idea, made explicit in `softrender.rs`: `cell_metrics()` sizes a virtual
  terminal cell (height = frame_h/45 — the kernel console's HD-font row at 4K — width 1:2,
  clamped), one rain column every two cell columns (the template's wide-char `step_by(2)`,
  ~80 columns at 16:9), trails of 6–30 discrete glyph boxes (¾ cell wide, ⅚ cell tall, drawn
  with `Frame::span` — full horizontal spans, the WC-friendly shape), near-white head,
  purple→cyan hue drift down the tail, per-cell flicker with ~1 in 5 cells dark, keyed
  deterministically by (column, row, 12 Hz tick) through `Rng::with_seed` — no new entropy,
  rule 5 intact. Speeds 0.055–0.22 cells/frame ≈ 3–13 rows/s at 60 fps. `Frame::pixel` is
  unused since (kept with `#[allow(dead_code)]`, like the FBIOPUT constants).
- **Logo parked on the left half** (asked for: the panel is dead on its right half).
  `Scene::center_x` (default: mid-frame) is read by the software `transform` and shipped to
  the WGSL via the uniform, now 32 bytes / 8 f32s. The live `screensaver` sets quarter width
  in `build_scene`; the rain still spans the whole frame; `selftest` keeps the default so
  its checksums stay about the renderers. A/B: old vec4 uniform vs new struct → byte-identical
  wgpu output, so the uniform change is exonerated (see the M3 section).
- **Checksums re-measured** — the rain rework changes every frame's bits, so the baseline
  table was updated per this README's own rule; two consecutive 720p runs identical; 4K avg
  10.81 ms/frame. Live metal after the rework: `matrix --timeout 3` = 167 frames @ 53.9 fps,
  `screensaver --timeout 2` = 96 frames @ 45.9 fps, both 4K, clean timeout exits.
- **Could not verify:** how the new rain actually looks to a human is on the panel right now —
  the ASCII dumps only approximate it. The wgpu mesh defect is documented, not fixed.

### Task 5 — a packed Matrix, the biggest cat by default, and the start of the wgpu mesh hunt — DONE 2026-10-03

- **Rain packed** (asked for: "more packed … there the whole time"): one rain column per cell
  column instead of every second one — ~160 columns at 4K/16:9, double Task 4's density. The
  rain was already drawn every frame of every mode (shared `backdrop`); the density is what
  makes it read as always-there. Checksums re-measured (table above); 4K avg 10.74 ms/frame;
  live `screensaver --timeout 2` at 4K: 39.9 fps (bigger cat + denser rain, still comfortably
  screensaver territory).
- **The show starts on `akuma_120`** (asked for: "the biggest one"): new `DEFAULT_ASSET = 2`
  in `main.rs`; Left/Right still cycles all four. selftest still walks the four assets in
  ASSETS order, so the table's columns keep their meaning.
- **wgpu mesh hunt, started** (the next list item: the M3 acceptance). Timeboxed to a survey
  this session: `interp.rs`'s `MathFunction::Sin/Cos/Min` and `backend.rs`'s rasterizer
  (`edge_xz` verbatim, fma area test, strict `z <`) look correct — suspicion narrows to
  `vs_main` expression evaluation or the tri storage-buffer read. Repro in "The wgpu backend
  (M3)" stands; not cracked here.

### Task 6 — rain motion is time-based — DONE 2026-10-03

- Reported on the box: on the wgpu path the rain sat frozen at its initial layout. Root cause:
  rain stepped per *frame*, and the interpreter runs ~seconds per frame at 4K, so between two
  presented frames the rain moved <1 cell — visually frozen (its slowness and the tiny mesh
  are the known M3 items). Fix: `backdrop` now derives a clamped `dt` from `time`
  (`Scene::last_t`), columns step `speed * dt * 60` cells — identical visuals at 60 fps,
  correct motion at any fps, still a pure function of `time` (rule 5). Checksums re-measured
  (table above; asset-switch jumps clamp to 0.1 s, hence slightly higher last-frame coverage);
  4K avg 10.98 ms/frame.

### Task 7 — the live rain was frozen at spawn — DONE 2026-10-03

- Reported on the box: on the live show the matrix columns never moved down, but their colors
  kept animating. Root cause: Task 5 changed `resize_rain` and `Column::draw` to one column
  per *cell* column (`frame_w / cw`), but the builder `rain_columns` still produced half that
  (`frame_w / (cw * 2)`). The show loop calls `scene.resize_rain` every frame, saw
  `want != rain.len()`, and rebuilt the rain from `RAIN_SEED` every frame — positions reset
  constantly (frozen), while hue drift and flicker (pure functions of `time` inside `draw`)
  kept changing. The selftest never resizes, which is why its frames looked right.
- Fix, three lines of substance: `rain_columns` now builds `frame_w / cw` columns (matching
  the resizer, so the per-frame rebuild is gone); `Column::step` applies the per-column speed
  again (`y += dt_cells * speed` — Task 6's dt conversion had dropped it, making every column
  uniform-speed); `backdrop` passes rows in *cell* units (`frame_h / ch`) to `step`, the same
  unit `Column::new` spawns in, so respawn off the bottom actually fires.
- Verified: 720p selftest twice — bit-identical across runs; new checksums in the table
  (coverage back up to ~6%, the doubled column count). Movement proved by diffing
  `selftest --dump` last frames at `--frames 40` vs `--frames 100`: trails advance several
  rows deeper over the extra second (frozen rain dumps were identical). Live
  `matrix --timeout 2`: 113 frames @ 54 fps on the panel.

### Task 8 — profile, then a shader compiler, a VM and an x86-64 JIT — DONE 2026-10-04

- **Profile first** (`AKUMA_PROF`): vertex-stage interpretation was 96-98% of the wgpu frame
  (22.6 µs per vertex invocation on the box, ~4x the host — musl's allocator under an
  interpreter that allocates for every value). Raster and fragment together were ~1 ms.
- **`jit_probe`** (kernel repo, `userspace/jitprobe`) showed RW → `mprotect(R+X)` works on the
  trashcan: no kernel change needed (the plan's `memfd_create` dual-mapping idea was wrong).
- Built `exec.rs` (Stage/Invoker), `compile.rs` + `program.rs` (lowering), `vm.rs`, `jit.rs`;
  per-draw buffer slices instead of per-read mutex locks; `exec-selftest`, `shader-check`,
  `deploy.sh`. Vertex invocation on the box: interp 21.1 µs → vm 444 ns → jit 201 ns.
- **`exec-selftest` found eight interpreter bugs** that the demo's two tiny shaders never hit
  and that would have bitten sugarloaf's the moment the fallback ran them: no
  `vec * scalar`; no int↔int bitcasts; `vec4(vec2, vec2)` did not flatten; no store to a vector
  component (`v.x = ..`); no static index step into arrays; **local `var x = <const>` ignored
  its initializer**; **loops never re-evaluated their conditions** (expressions were memoized
  on first use for the whole invocation, so `loop { if i >= 12 { break } .. }` spun forever);
  and `let` values were read at first *use* instead of at their `Emit` point. The interpreter
  now follows naga's Emit semantics.
- **Cut the 4K plumbing**: the frame is handed to wgpu as bytes instead of converted per
  pixel; `get_mapped_range` no longer clones a 33 MB buffer per map; vectorizable masked
  blit. 4K wgpu frame 90 → 38 ms.
- **Could not verify:** the live `screensaver --wgpu` on `/dev/fb0` after these changes (only
  headless `selftest` was run end to end); how it looks to a human.
- **Not done:** textures/samplers, vertex buffers, blending, standard viewport semantics —
  see "Next steps".

### Task 9 — the standard-mode GPU (M4) — DONE 2026-10-04

- Surveyed rio's real wgpu use first (formats, vertex formats, blend factors, samplers, draw
  calls) so the subset is the one that matters; see the M4 section for what was built.
- `gpu-selftest` (16 tests) is the acceptance; writing it exposed real bugs before any client
  could: the compiler could not index a local array dynamically (sugarloaf does), the
  interpreter could not return a bare `@builtin(position)`, and fragments were not bounded by
  the viewport rectangle (the guard-band clip lets geometry run past NDC ±1).
- sugarloaf's own `grid.wgsl` renders pixel-correct through its own pipeline layout, on host
  and on the trashcan. 15/15 sugarloaf entry points lower and JIT.
- `gpu-bench` found the next wall: ~65 ns of fixed per-fragment cost plus the shader, i.e. a
  full 4K redraw takes ~1.6 s. The transcendental result cache (`CallC`) already took the
  grid shader from 2.8 s to 1.6 s.
- **Could not verify:** rio itself (not built against this backend); any `Surface`/swapchain
  path; mipmaps; HDR formats; that bilinear sampling matches a real GPU's rounding bit for
  bit (it does not claim to).

### Task 10 — fps: in-place rendering, threads, a wide JIT — DONE 2026-10-04

- Live demo through wgpu: < 1 fps → 35.6 fps at 4K (software 46.7); rio-scale grid pass 2.8 s →
  0.22 s. Details and per-step numbers in "Performance work" under M4.
- Found by measuring, not guessing: the demo's fragment shader declares an unused
  `@builtin(position)` so the "position-independent" test never fired; `floor()` and 4-byte
  `memcpy` are real function calls on musl; a render-pass clear was 128 ms because it ran
  per pixel.
- **Could not verify:** the picture on the panel after the threading/wide-JIT work (the demo
  uses the legacy single-threaded raster path, so it is unaffected, but nobody looked); wide
  JIT on anything but this CPU (SSE4.1 only — falls back to the scalar JIT without it).
- Environment switches added: `AKUMA_THREADS`, `AKUMA_WIDE`, `AKUMA_EXEC=jitw`.

### Task 11 — rio-scale redraw: from 320 ms to ~20 ms — DONE 2026-10-04 (second session)

Goal given: make sugarloaf's real shaders redraw a 4K terminal interactively on the trashcan
(4 cores, SSE4.2, no AVX), target a full redraw under ~50 ms, without regressing the demo.
Result (`gpu-bench`, 3840×2144, details and per-step numbers in "rio-scale passes"):
cell-background pass 223 → 5 ms, a full terminal-like redraw (backgrounds + glyphs in 60% of
the cells) ~20 ms mean / 14.5 ms best (≈ 50 fps), worst case (every cell its own colour and
a glyph) 320 → 25 ms. The demo's four 4K checksums and 720p checksums are unchanged and
`screensaver --wgpu` still runs at ~35 fps (present-bound).

- New: `opt.rs` (optimizer, per-draw specialization, if-conversion), `memo.rs` (region
  memoization), `runs.rs` (position-quantization run detection), `pool.rs` (spinning worker
  pool); raster span/gather paths, SSE2 encode/over-blend, wide `textureLoad`, lazy clear.
- Found by measuring: `std::thread::scope` costs ~8 ms per phase here (thread creation 1.5 ms,
  wake-up at a scheduler tick); first-touch page faults are µs each; every glyph-vertex batch
  diverged because of naga's lowering of `||`; a 64-bit `idiv` per row edge; the interpreter
  could not return float fragment colours at all (so the "fallback" for standard-mode
  fragment shaders never worked — fixed).
- Tests added (all on host *and* box): `exec-selftest` runs every snippet specialized too
  and 60+ random vertex shaders (`AKUMA_FUZZ=n`, `AKUMA_FUZZ_SEED=s`) against the
  interpreter; `gpu-selftest` (now 21 tests) adds: memoized+specialized fragment shader vs a CPU
  evaluation, vector over-blend vs the scalar definition, run replication on/off over awkward
  cell geometries (asserting it really fires for the grid shader and not for one that leaks the
  position), random fragment shaders vs the interpreter (default, runs off, forced
  specialization, two uniform values), sugarloaf's `renderer.wgsl` rects. Fuzzing: 800 random
  vertex shaders and 150 random fragment shaders × variants, all executors bit-identical.
- **Could not verify:** rio itself; anything about how the output looks to a human (all
  pixel checks are numeric); the wide JIT's numbers on a CPU other than this one; behaviour
  of the worker pool's 50 ms spin on a busy desktop (it burns up to three cores for 50 ms
  after each frame — `AKUMA_SPIN_MS` tunes it); `sched_setaffinity` (returns -1 here, so workers cannot be pinned).

## Next steps (updated 2026-10-04, branch `jit-shader-executor`)

Done: the compiler/VM/JIT executors (M3 speed), the standard-mode GPU and its test suite
(M4), real sugarloaf shaders running on the trashcan, and (Task 11) a full 4K terminal redraw
in ~20 ms. In order:

1. **Glyph pass** (now most of a frame): merge the two triangles of an axis-aligned quad into
   one rectangle fill; cheaper vertex marshalling; a register allocator / 8-lane code for the
   wide JIT. Re-run `gpu-bench` after every change (it reports mean and best).
2. **Verify on `/dev/fb0`** that the demo still looks right with everything above (only
   headless `selftest` and `screensaver --timeout` runs were done, no human looked at the panel).
3. **rio itself**, which is outside this repo: a framebuffer platform in `rio-window`
   (screen = `/dev/fb0`, input = the console tty), a `Surface` whose texture is presented
   into the mapping with whole-row copies, and building rio against this wgpu backend. Known
   gaps to check against rio's real use once it builds: surface/swapchain semantics
   (`get_current_texture`, `present`), `Rgba16Float` / HDR filter targets, `Rgba8Snorm`,
   mipmapped textures (rio's filter chain creates them), depth/stencil, multisampling
   (rio uses `sample_count: 1`), and the 2 `copy_texture_to_texture` / 2 `set_viewport`
   call sites. Not yet exercised by a test: sugarloaf's `image.wgsl`, `text_shader.wgsl` and the
   filter shaders (they compile and JIT; only `grid.wgsl` and `renderer.wgsl` render under test).
4. The demo's own legacy raster path (`backend.rs::draw_legacy`) still shades fragment by
   fragment on one thread; the live demo is present-bound (33 MB write-combined copy ≈ 11 ms)
   so there is little to gain, but its vertex stage could use the wide path and the worker pool.
5. Kernel repo: the scheduler/wake-up behaviour in "Things this kernel taught us".

## Tasks, in order

### Task 3 — a raw C probe for `/dev/fb0` (kernel-side preparation) — DONE 2026-10-03 (see done-log; the probe found a live `/dev/fb0` and an fb.rs ABI bug)

The task, as written when it was NEXT:

Write `probes/fbprobe.c`: a plain C program (no libakuma) built with
`x86_64-linux-musl-gcc -O1 -static`. It should:

1. open `/dev/fb0`;
2. print every field of both `FBIOGET_*` structs;
3. `mmap` the framebuffer `MAP_SHARED`, `PROT_READ|PROT_WRITE`;
4. fill it with a gradient and time the fill with `clock_gettime(CLOCK_MONOTONIC)`;
5. read a few pixels back;
6. exit 0 only if everything worked.

Print the MB/s figure. If the kernel side is right it lands near 3000 MB/s; ~71 MB/s means the
mapping is not write-combining. It must also run unchanged on a real Linux box with a
framebuffer, which is how its expected output gets calibrated.

### Later — not now, unless asked

- **M3, the wgpu backend:** implemented since (see "The wgpu backend (M3)" above). What
  remains of it is the acceptance: frames bit-identical to `softrender` — the checksums in
  "Baseline" are the concrete target, and the mesh defect described in the M3 section is the
  blocker.
- **rio:** a framebuffer platform in rio's `rio-window`, plus the wgpu backend above.

## The kernel side (not in this repo)

**Status update 2026-10-03:** `/dev/fb0` exists on the box now (char dev 29:0) and the demo
plus `probes/fbprobe.c` exercise S2 (WC at ~2866 MB/s user-side), S3 (single-open device) and
S4/S5 (ioctls byte-identical to the plan §2 dump, device-backed mmap). The kernel-side signal
bugs found while getting the demo to survive SIGTERM are listed in the done-log — they are
kernel-repo work, this repo carries only the `_exit(0)` handler workaround in `input.rs`.
The slice list below is kept as reference:

The plan's slices, all in the kernel
repo:

1. **S1:** keep the framebuffer's geometry in a static at boot.
2. **S2:** add a write-combining memory type for user page tables.
3. **S3:** a `/dev/fb0` file descriptor; a second open gets `EBUSY`.
4. **S4:** the fbdev ioctls (`FBIOGET_VSCREENINFO` 0x4600, `FBIOPUT_VSCREENINFO` 0x4601 which
   refuses mode changes, `FBIOGET_FSCREENINFO` 0x4602 with `id = "akuma-fb"`, `FBIOPAN_DISPLAY`
   0x4606 as a no-op).
5. **S5:** device-backed `mmap`.
6. **S6:** hand the screen back to the console on close, exit or kernel panic.

Do **not** start on the kernel from this brief. It is a separate piece of work with its own rules
(see the kernel repo's `CLAUDE.md`). Task 3 above is designed to be useful before it exists.

## Reporting

When you finish a task, report:

- the commands you ran and their real output (the selftest lines and any errors);
- the files you changed;
- what you could **not** verify.

Never describe something as working unless you ran it. Do not commit or push unless you are told
to.
