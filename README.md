# akuma-cli-wgpu

A 3D screensaver of the Akuma cat logo that draws straight into the framebuffer. It is the first
user of a Linux-standard framebuffer device (`/dev/fb0`) on the Akuma amd64 kernel. The real goal
behind it is to run the **rio terminal** on Akuma's screen later, through a wgpu backend that
renders into that same framebuffer.

This README is the brief for whoever works on this repo next. If you are an agent, read all of it
before changing anything.

## Rules that are not negotiable

1. **Plain Linux only.** This is an ordinary `x86_64-unknown-linux-musl` program using `std` and
   the `libc` crate. **Never use `libakuma`** or any Akuma-private syscall. rio and wgpu will run
   on this substrate without knowing Akuma exists, so this demo must too.
2. **Keep dependencies tiny.** The only dependency today is `libc`. It has to build with the
   nightly musl toolchain that runs *on the Akuma box itself*. Do not add `clap`, `rand`,
   `crossterm` or similar; the code has its own replacements on purpose (see `Cargo.toml`).
   wgpu and naga arrive later, in milestone M3, and only there.
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
| `src/softrender.rs` | the software rasterizer: rotate → perspective project → cull → shade (lambert + purple-blue hue wave) → z-buffered spans; plus the Matrix-rain backdrop. The z-buffer is refilled with +inf at the top of `render()` **every frame** — this refill is load-bearing (see done-log) |
| `src/rng.rs` | xorshift64* with a splitmix64-seeded `with_seed` constructor; the render path never touches wall-clock or ASLR entropy (rule 5) |
| `src/input.rs` | termios raw mode, `poll(2)` and escape-sequence decoding; SIGINT/SIGTERM exit the process immediately (kernel signal bug — see done-log) |
| `src/clock.rs` | `Pacer` (sleeps out the frame budget) and `FpsMeter` (live fps, over-budget count, slowest frame) — EINTR-tolerant `CLOCK_MONOTONIC`/`nanosleep`, never std `Instant` (kernel note in `input.rs`) |
| `src/wgpu_backend.rs` | placeholder for milestone M3; `--wgpu` prints its status and exits |
| `src/akuma_{20,40,79,120}.txt` | the logo at four sizes; `akuma_40.txt` is byte-identical to the kernel's boot-banner asset |

Composing each frame in ordinary RAM and copying whole rows into the mapping is deliberate. The
kernel maps the framebuffer write-combining, which is fast for full-row copies (~3000 MB/s on the
box) and slow for scattered writes (~71 MB/s). Keep that property in anything you change.

## Baseline: selftest checksums and timings (2026-10-03)

These are the acceptance numbers for M3: the wgpu path must reproduce these frames (same
`fnv1a` per asset). They are stable across runs on the trashcan (nightly
`1.100.0-nightly (420ed2a0c)`, release profile). If you change rendering intentionally, expect
these to change — update this table and say why in your report.

| config | akuma_40 | akuma_79 | akuma_120 | akuma_20 | avg render |
|---|---|---|---|---|---|
| 1280×720, 120 frames | `1fb62b0e` | `106fe365` | `2000460c` | `d9cef3ca` | 1.46 ms/frame (686 fps) |
| 3840×2160, 120 frames | `6ad3744e` | `caf9295f` | `38f2e499` | `45c6878d` | 10.65–10.74 ms/frame (~94 fps) |

Re-verified 2026-10-03 after the fb.rs u16 fix, the clock hardening and the signal-handler
change: **checksums identical**, `SELFTEST OK` on every build.

Where the 4K render milliseconds go (probe measurements): the two 33.2 MB fills — `frame.clear`
(~3.5 ms) and the z-buffer refill (~3.4 ms) — are ~64% of the frame; the raster is the rest
(~1 ms for the 442-tri `akuma_20` up to ~2.8 ms for the 8538-tri `akuma_120`); rain is
negligible (~0.03 ms). Do not "optimise" the z refill away — see the done-log for what its
absence does.

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

- **M3, the wgpu backend:** wgpu 30 with `features = ["custom", "wgsl"]`, a custom backend
  through `wgpu::Instance::from_custom`, and WGSL shaders *interpreted* from naga IR. The
  kernel refuses writable-and-executable memory, so a cranelift JIT is not an option.
  `jgraef/wgpu-cpu` is a useful reference, but it has **no license**, so read it and do not copy
  it. Acceptance: the wgpu path produces the same frames as `softrender` — the checksums in
  "Baseline" above are the concrete target.
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
