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
4. **Small edits, rebuild after each one.** The last session broke the build with a large
   search-and-replace that deleted lines it did not mean to (see "Task 1"). Change one function at
   a time, run `cargo build`, and read the error before the next edit.

## Where things are

| what | where |
|---|---|
| this repo, on the box | `/src/github.com/netoneko/akuma-cli-wgpu` (GitHub: `netoneko/akuma-cli-wgpu`) |
| the kernel repo, on the box | `/src/github.com/netoneko/akuma` |
| **the plan** (read it) | [`docs/fbdev-wgpu-plan.md`](docs/fbdev-wgpu-plan.md) in this repo — a copy of `docs/runbooks/amd64-fbdev-wgpu-demo.md` from branch `cats/meow/fbdev-wgpu-plan` of `netoneko/akuma-litter` |
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
  `python3` on `PATH`.
- The binary is called `akuma-wgpu` (the package is `akuma-cli-wgpu`).
- `/dev/fb0` **does not exist yet** (see "The kernel side"). So `akuma-wgpu screensaver` and
  `akuma-wgpu matrix` will print `cannot open /dev/fb0`. That is expected today, not a bug in
  this repo. `selftest` needs no device and is the test you can always run.
- It also builds on a Mac or Linux laptop with `cargo build --release` for the host, and
  `selftest` runs there too.

## What exists

`akuma-wgpu <screensaver | matrix | selftest> [options]`, modelled on `akuma-cli` (an
ANSI-terminal screensaver with the same subcommands, `--latin`, arrow keys to switch the logo,
`q`/Esc to quit, and a metrics report on exit).

| file | what it does |
|---|---|
| `src/main.rs` | argument parsing, the render loop (`run_show`), `selftest`, the exit report |
| `src/fb.rs` | the fbdev client: `open("/dev/fb0")`, `FBIOGET_VSCREENINFO`/`FBIOGET_FSCREENINFO`, `mmap`, pixel-format packing, `present` (row-wise copy of a RAM `Frame` into the mapping). The fbdev structs are declared by hand to match `<linux/fb.h>` byte for byte |
| `src/catlogo.rs` | turns the ASCII cat (`akuma_*.txt`, density ramp ` .:-=+*#%@`) into a height field and extrudes it into a 3D triangle mesh |
| `src/softrender.rs` | the software rasterizer: rotate → perspective project → cull → shade (lambert + purple-blue hue wave) → z-buffered spans; plus the Matrix-rain backdrop |
| `src/input.rs` | termios raw mode, `poll(2)` and escape-sequence decoding; SIGINT/SIGTERM set a quit flag |
| `src/clock.rs` | `Pacer` (sleeps out the frame budget) and `FpsMeter` (live fps, over-budget count, slowest frame) |
| `src/rng.rs` | xorshift64* for the visuals |
| `src/wgpu_backend.rs` | placeholder for milestone M3; `--wgpu` prints its status and exits |
| `src/akuma_{20,40,79,120}.txt` | the logo at four sizes; `akuma_40.txt` is byte-identical to the kernel's boot-banner asset |

Composing each frame in ordinary RAM and copying whole rows into the mapping is deliberate. The
kernel maps the framebuffer write-combining, which is fast for full-row copies (~3000 MB/s on the
box) and slow for scattered writes (~71 MB/s). Keep that property in anything you change.

## Tasks, in order

### Task 1 — make it build again (do this first)

`cargo build` fails with `unexpected closing delimiter` at the last line of `src/main.rs`. An
earlier edit deleted two things:

- **The signature of `print_metrics`.** Its body survives after the `// Exit report` comment
  near the end of `main.rs`, but its `fn` line is gone. From the call in `run_show` and the body,
  it was:
  ```rust
  fn print_metrics(opts: &Options, meter: &FpsMeter, frame: &Frame, switches: u64, why: &str) {
  ```
- **`dump_ascii(&frame)`**, called from `run_selftest` when `--dump` is given. Nothing defines it
  any more. Write a small one: downsample the `Frame` to about 80×40 cells and print one character
  per cell, using the same ` .:-=+*#%@` ramp by brightness. Read `Frame` in `src/fb.rs` first to
  see its fields; `softrender::BG` is the background colour.

**Done when** `cargo build --release --target x86_64-unknown-linux-musl` finishes with no errors
and **no warnings**, and `akuma-wgpu selftest` ends with a `SELFTEST OK` line. Also run
`akuma-wgpu selftest --dump` and check the ASCII picture really is a cat. Put the selftest output
in your report.

### Task 2 — check the selftest numbers mean something

- **The checksums must be deterministic.** Run `selftest` twice; both runs must print the same
  `fnv1a` values. If they differ, find out why (time or randomness leaking into rendering) and
  fix it, because these checksums become the acceptance test for M3.
- **Report the speed.** Report `avg render: … ms/frame` at the default 1280×720 and at
  `--w 3840 --h 2160` (the box's real screen). A 60 fps target needs under 16.6 ms per frame at
  4K. If it is far off, say where the time goes (for example, the z-buffer clear is 33 MB per
  frame at 4K). Do not optimise yet; just measure and report.

### Task 3 — a raw C probe for `/dev/fb0` (kernel-side preparation)

Write `probes/fbprobe.c`: a plain C program (no libakuma) built with
`x86_64-linux-musl-gcc -O1 -static`. It should:

1. open `/dev/fb0`;
2. print every field of both `FBIOGET_*` structs;
3. `mmap` the framebuffer `MAP_SHARED`, `PROT_READ|PROT_WRITE`;
4. fill it with a gradient and time the fill with `clock_gettime(CLOCK_MONOTONIC)`;
5. read a few pixels back;
6. exit 0 only if everything worked.

Print the MB/s figure. If the kernel side is right it lands near 3000 MB/s; ~71 MB/s means the
mapping is not write-combining. It must also run unchanged on a real Linux box with a framebuffer,
which is how its expected output gets calibrated.

### Later — not now, unless asked

- **M3, the wgpu backend:** wgpu 30 with `features = ["custom", "wgsl"]`, a custom backend
  through `wgpu::Instance::from_custom`, and WGSL shaders *interpreted* from naga IR. The
  kernel refuses writable-and-executable memory, so a cranelift JIT is not an option.
  `jgraef/wgpu-cpu` is a useful reference, but it has **no license**, so read it and do not copy
  it. Acceptance: the wgpu path produces the same frames (same checksums) as `softrender`.
- **rio:** a framebuffer platform in rio's `rio-window`, plus the wgpu backend above.

## The kernel side (not in this repo)

The kernel does not expose a framebuffer to userspace yet. The plan's slices, all in the kernel
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
(see the kernel repo's `CLAUDE.md`). Tasks 1–3 above are designed to be useful before it exists.

## Reporting

When you finish a task, report:

- the commands you ran and their real output (the selftest lines and any errors);
- the files you changed;
- what you could **not** verify.

Never describe something as working unless you ran it. Do not commit or push unless you are told
to.
