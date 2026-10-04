# Handoff prompt: make rio's shell work on the Akuma framebuffer, then the next optimization round

Paste everything below the line into a fresh session.

---

You are continuing work started in /Users/netoneko/github.com/netoneko/akuma-cli-wgpu on branch
`jit-shader-executor` (committed, pushed as far as allowed) and in the forks `netoneko/rio`
(branch with the musl/fb patches) and `netoneko/akuma-wgpu-backend` (empty, future home of the
backend). Read README.md here fully first ("Rules that are not negotiable", the M4 section,
"The wgpu Surface over the framebuffer"), then
`/Users/netoneko/github.com/netoneko/akuma/docs/archive/AKUMA_AMD64_RIO_FBDEV_BUILD.md` — it has
the full build story, the debug tooling, every bug fixed so far, and the kernel-side findings.

Rules: plain Linux only (no libakuma), tiny deps, small edits and rebuild after each, deterministic
rendering, never claim something works unless you ran it, no background agents without asking the
user, commit checkpoints with the Co-Authored-By trailer, do not push the kernel repo (the user
drives those commits); the two fork repos (rio, akuma-cli-wgpu) may be pushed.

## Environment

- Dev host is aarch64 macOS; the box (`ssh akuma`, port 2222 in some configs) is the x86-64
  Akuma trashcan with an Alpine userland (`apk` works; no /etc/passwd, rg/python3 absent,
  busybox ps/pidof). `/tmp/rio-bin` is the deployed rio; `userspace/rio/build.sh` in the kernel
  repo rebuilds + redeploys it (needs `RIO_DIR=~/github.com/netoneko/rio`, musl-cross gcc,
  rustup toolchain 1.96.1 with the musl target).
- The build: `cargo +1.96.1 build --release -p rioterm --no-default-features --features wgpu,fb
  --target x86_64-unknown-linux-musl` with `CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER`
  / `CC_...` / `AR_...` set to the musl cross tools.
- rio renders on the panel through our wgpu backend (sugarloaf `--features wgpu`, custom backend
  instance in `sugarloaf/src/context/webgpu.rs` under `cfg(target_env = "musl")`; surface =
  /dev/fb0, shared via AKUMA_FB_FD because the device is single-open).
- rio config on the box: `/root/.config/rio/config.toml` — `shell = { program = "/bin/sh" }`,
  `[fonts] size = 18.0, family = "Source Code Pro"` (font installed via apk).
- Debug tooling: the fb platform appends its full lifecycle trace to `/tmp/akuma-fb.log`
  (per-event handler enter/exit lines, raw tty reads, decoded keys). `AKUMA_FB_DEBUG_INPUT=1`
  mirrors to stderr. The console shell cannot parse `VAR=x cmd 2>file` — never rely on it.
  `/tmp/rio-expect.sh` on the host drives rio over `ssh -tt` with scripted keystrokes; the
  whole input path is testable without anyone at the panel.
- Tests that must stay green: on the host AND box `akuma-wgpu exec-selftest` (fuzz:
  AKUMA_FUZZ=n AKUMA_FUZZ_SEED=s), `AKUMA_SUGARLOAF=<rio>/sugarloaf/src akuma-wgpu gpu-selftest`
  (22 tests), `selftest --w 1280 --h 720 --frames 120 --wgpu` and `--w 3840 --h 2160`
  (checksums in the README table IDENTICAL), `screensaver --wgpu --timeout 6` (~35 fps,
  present-bound). rio-window decoder tests: `cargo +1.96.1 test -p rio-window --lib --no-run
  --features fb --target x86_64-unknown-linux-musl` then run the test binary ON THE BOX
  (macOS cannot execute musl binaries).
- Kernel-side findings so far are in the archive doc; do not fix the kernel from here, just
  document anything new.

## State (2026-10-04, end of session)

rio renders on the panel (grid, cursor, config-error screens all drew) and takes input: the fb
platform decodes console bytes, delivers KeyboardInput to rioterm (confirmed in
/tmp/akuma-fb.log), ctrl+enter dismisses rio's own screens. Fixed along the way: a redraw-queue
livelock in the fb event loop (drain a snapshot, not the live queue), console reads that ignore
VMIN=0/O_NONBLOCK (tty is O_NONBLOCK now, poll sliced at 33 ms), incomplete escape-sequence
decoding, clipboard panic without X11, no-shell (config now pins /bin/sh), write_buffer_with in
the wgpu backend.

## Part 1 — make typing work — DONE 2026-10-05

Resolved: the kernel has no ptys, so rio ran a dead context. The rio fork
now falls back to a pipe pty with a userspace line discipline, and the fb
platform sends modifiers before the key. Verified at the panel. See README
"rio on the panel" and the archive doc's "Resolved 2026-10-05" section and
kernel pty spec. Still open from this part: the Ctrl+Q quit binding. The
original notes follow for the record.


Symptom: keys reach rio, but typed text never appears; cursor frozen. Suspects in order:

1. **Start from rio's own log.** The user launched with `--enable-log-file`; a fresh 34 KB
   /tmp/rio.log (2026-10-04 20:55) was never examined. Read it first.
2. **The pty master read path.** rio's reactor is corcovado (epoll wrapper). If epoll on a pty
   master never reports readable on this kernel, the shell's echo never renders. Trace points:
   add flog-style logging (rio_window::platform::fb has flog; make it public if needed) around
   rioterm's `messenger.send_write` (does the write to the master succeed? how many bytes?) and
   around corcovado's poll/epoll + read on the master fd.
3. **The pty master write** blocking (the kernel ignores non-blocking semantics on the console
   channel — ptys may share the disease; the console read ignores VMIN=0 entirely, documented).
4. **ash spawn**: verify a live ash child of rio-bin exists (`pidof ash` before/after; note
   stacked rio instances confuse this — kill all first: `pidof rio-bin | xargs -r kill -9`).
5. Watch for the console channel delivering input to the console shell instead of rio (earlier
   evidence: typed keys appeared in /bin/sh after rio died). If pts vs console routing is the
   issue, the trace will show zero master-write bytes.

Verification: over the ssh expect harness first (full control), then the real console with the
user watching the panel. Success = typed keys echo and commands run, on the panel.

Also small: rio's quit binding is Super+Q which the console cannot express (Ctrl+D/EOF works);
add a rio binding that works on the fb platform (e.g. Ctrl+Q via a config binding in the shipped
config or a fork patch).

## Part 2 — polish once typing works

1. Config-error screen: rio showed a "press enter" config screen on early runs; confirm it is
   gone with the current config, and if not, capture its text and fix the config keys.
2. Console shell quirks: verify whether `VAR=x`/`2>` really fail in the kernel console shell or
   that was a red herring (it explained empty logs once; the fb-log-file approach removed the
   dependency).
3. Cursor blink / time-based updates (currently frozen cursor even when idle — rio may need
   ControlFlow::WaitUntil wakeups; check the fb platform's WaitUntil path).
4. Update README (this repo) and the archive doc with anything new; keep the
   "Things this kernel taught us" list current for the kernel repo.

## Part 3 — the previous optimization round (unchanged priorities, do after rio works)

Glyph pass first (quad rectangle merge with a gpu-selftest comparing rect vs triangle path pixel
for pixel; cheaper vertex marshalling in `Invoker::run_vertex_batch`; hoist PixelPlan::new; avoid
per-triangle 128-byte varyings copies), then the wide JIT (register allocator / block-local xmm
caching, 8-lane code), then the scheduler problems (mean vs best ~5 ms gap; sched_yield made it
WORSE). Profile with AKUMA_PROF=1 and gpu-bench before each change; checksums stay identical.

## Report at the end

Commands run with real output, files changed, per-step before/after numbers, and what you could
not verify. Update the README and the archive doc as you go.
