# A wgpu demo on the framebuffer — `akuma-cli-wgpu` and the road to rio

**Status: plan (researched 2026-10-02). Update 2026-10-03: the userspace half of this is
real — `netoneko/akuma-cli-wgpu` runs on `/dev/fb0` with the software rasterizer (M2) and the
§5 wgpu custom backend (M3: `Instance::from_custom`, WGSL via the naga-IR interpreter, no
JIT). Its README tracks what works; one open defect: the wgpu mesh path does not yet
reproduce the software frames bit-for-bit. The kernel-side slices below are still the
reference for that repo's work.**

Not implemented yet (kernel side). This document is the
design + slice plan for two things that share one substrate:

1. `userspace/akuma-cli-wgpu` — a userspace demo that draws a 3D screensaver of the
   cat logo straight into the framebuffer.
2. The Linux-standard device interface under it (`/dev/fb0`), which is also what a
   later **rio terminal** launch would sit on, via a wgpu *framebuffer backend*.

Constraints, non-negotiable:

- **Raw Linux syscalls only** in userspace: `std`/`libc`/`rustix` style, targeting the
  kernel's Linux x86-64 ABI. **No `libakuma`**, no private syscalls. rio and wgpu must
  be able to run without knowing Akuma exists.
- **Build on the 2026-10 console/framebuffer work** (branch `local-console`): the
  multiboot2 framebuffer + write-combining/PAT mapping, `akuma-fbcon`, the HD font and
  the splash. `docs/archive/FRAMEBUFFER_REMOVED.md` and `docs/archive/DOOM.md` are old
  AArch64 history; their only lesson here is the one they teach by absence — expose a
  *standard* device (`/dev/fb0`), not `sys_fb_*`-style private syscalls.

---

## 1. What the kernel has today (verified 2026-10-02)

Everything a framebuffer device needs already exists **inside the kernel**; none of it
reaches userspace.

| piece | where | what it gives the demo |
|---|---|---|
| the scanout | `amd64/src/multiboot2.rs` `kmain_mb2` | GRUB hands a direct-colour linear framebuffer via the multiboot2 tag; width/height/pitch/bpp/format parsed; refused above `MAPPED_LIMIT` (4 GiB). Trashcan: 3840x2160x32, pitch 16384 |
| WC mapping | `multiboot2::map_wc`, `pat_init_ap` | PAT entry 4 re-typed WC (MSR `IA32_PAT`), PAT bit set on the fb's 2 MiB PDEs in `__pd0`, ragged ends split to 4 KiB. Measured 3026 MB/s full-screen fill (was 71 MB/s UC). The `[fb]` line reports `wc on` + clear time |
| text/graphics stack | `crates/akuma-fbcon` | `Surface` (`put`/`fill`/`flush`), the terminal emulator (vte 0.15 + unicode-width), the HD font (IBM Plex Mono 24x48, scale 1 on 4K, grid ~157x44), `splash::paint` (the hue-wave cat), the `[fb]`/`stty size` glue (`fb_grid`, `cursor_idle`) |
| input + tty | `amd64/src/console.rs`, `xhci.rs` | keyboard → `ProcessChannel` → line discipline (`^C`→`SIGINT`), one shared `TerminalState`; the pump is the model for "who owns the screen" arbitration |
| user page tables | `crates/akuma-mmu`, `amd64/src/paging.rs` | `map_page_in(root, va, pa, prot, attr)` takes a **`MemAttr`**; `PteProt::USER_RW` exists; `MemAttr::Device` (PCD\|PWT) is already anticipated for "a future device-mapping method". Missing only a `WriteCombine` attr (PWT=0, PCD=0, PAT=1) — PAT entry 4 is *already* WC on every CPU |
| device pattern | `FileDescriptor::DevDsp` | the /dev/dsp recipe: an fd variant, an `openat` path arm (glue `fs.rs`), ioctl arms (glue `term.rs`), a `dev_node` registry entry (`akuma_vfs::dev`) so `ls /dev` shows it |
| Linux ABI userspace | the box itself | nightly musl `rustc`/`cargo` run on Akuma (`stage-rust-toolchain-amd64.md`); `userspace/amd64/ruststd` proves real `std` (static-PIE musl) works: `println!`, `Vec`, `std::thread` |

**What userspace sees: nothing.** On the live box `/dev` holds `null random tty
urandom zero`. There is no `FBIO*`, no `fb0`, no `fbdev` anywhere under `amd64/` or
`crates/`. `sys_fb_*` (mentioned in a glue comment) was the old AArch64 path, removed
2026-08-31. The console's output path (`serial::mirror_byte` → fbcon) is kernel-
internal; userspace "draws" only by *writing text* into the channel.

Two more facts that shape the plan:

- **The kernel refuses a page that is writable and executable at once** (`mm.rs`:
  `PROT_WRITE|PROT_EXEC` in one `mmap`/`mprotect` call → `EINVAL`). **But a JIT does not
  need one — measured 2026-10-04** with `userspace/jitprobe/c/jit_probe.c` in the
  kernel repo, on the trashcan: mmap RW → emit code → `mprotect(R+X)` → call works,
  RX→RW→rewrite→RX re-JIT cycles work (no stale code), an RX page survives `fork`, and
  a JITed loop runs at native speed (0.39 ns/iter). See §5b.
- **No `memfd_create`.** Only needed by any Mesa/lavapipe escape hatch now (the
  dual-mapping JIT it was once listed for is unnecessary).
- **QEMU `microvm` has no framebuffer at all** (`run.sh` proves this: `FBTRACE`
  writes what the screen *would* show to port 0xE9). The fb rig is the trashcan, or a
  new QEMU rig booted through GRUB (§7).

---

## 2. The interface: fbdev (`/dev/fb0`), not DRM

**Decision: a Linux fbdev-style device — `/dev/fb0`, the three standard ioctls, and
`mmap`.**

| | fbdev `/dev/fb0` | DRM dumb buffers (`/dev/dri/card0`) |
|---|---|---|
| fits the hardware | exactly: one static scanout, fixed mode set by GRUB, no modesetting, no flips | assumes a KMS driver: planes, encoders, commit ioctls |
| kernel work | small: one fd type, three ioctl arms, one mmap arm | large: an entire drm-mini subsystem for zero benefit without a real GPU driver |
| what programs get | fixed geometry + a WC mapping of the live scanout | dumb-buffer alloc/attach, which still cannot flip without KMS |
| compat | fbdev is the classic "small kernel" display API (uapi `linux/fb.h`) | what mesa/X/wayland stack wants — irrelevant until there is a GPU driver |

The demo (and rio's backend) is the only consumer, and it wants exactly what fbdev
gives: "here are the pixels, they are the screen". If a DRM story is ever needed
(a real i915-class driver is a multi-year project by itself), dumb buffers can be
layered on the same internals later; nothing in the fbdev design blocks it.

### The device surface (what `/dev/fb0` must answer)

Values from uapi `linux/fb.h` (the `libc` crate ships the constants and
`fb_var_screeninfo`/`fb_fix_screeninfo` structs on Linux targets, so userspace needs
no hand-rolled ABI):

- `FBIOGET_VSCREENINFO` (0x4600) — xres, yres, `xres_virtual = xres`,
  `yres_virtual = yres` (single buffer), `bits_per_pixel`, the red/green/blue
  `fb_bitfield`s **copied from the multiboot2 tag's pixel format**, `vmode =
  FB_VMODE_NONINTERLACED`.
- `FBIOPUT_VSCREENINFO` (0x4601) — refuse mode *changes* (`EINVAL`) but accept
  identical settings (some programs round-trip it).
- `FBIOGET_FSCREENINFO` (0x4602) — `id = "akuma-fb"`, `smem_len = fb size`,
  `line_length = pitch`, `visual = FB_VISUAL_TRUECOLOR`, `type = FB_TYPE_PACKED_PIXEL`.
- `FBIOPAN_DISPLAY` (0x4606) — returns 0, no-op: single buffer, `yoffset` stays 0.
- `mmap` — the whole point; see §4 S5.

No `FBIOGET_VBLANK`/vsync: the scanout is firmware-driven; there is no flip and no
interrupt. Clients that care about tearing double-buffer in their own memory and blit
(the demo does exactly this; see §6).

---

## 3. Ownership: the console vs. the app

Today exactly one writer exists (fbcon, via the `CONSOLE` static, the serial
mirror and the splash daemon). A userspace fb client and the console would fight for
pixels. Linux answers this with VTs (`KD_GRAPHICS`/`KD_TEXT`); the minimal Akuma
version is an ownership bit, in the spirit of `SPAWN_FLAG_CONSOLE`'s
refuse-if-not-yours rule:

- **`open("/dev/fb0")` succeeds only for the first client** (single owner, like the
  console-attachment rule); a second open gets `EBUSY`.
- On open: kernel stops drawing — one atomic (`FB_USER_OWNS`) gates every fbcon write
  entry point (`serial::mirror_byte`/`putb_tty` fb arm, `cursor_idle`, splash, panic
  banner). Same shape as the existing `set_fb_quiet` policy.
- On last `close`/process exit: ownership released, screen cleared, banner redrawn
  (mirror what `fb_splash_end` does). A **panic while a user owns the fb** must
  reopen the console (extend the `begin_fatal` path: drop ownership, redraw, then
  print) — a failed boot is never hidden, same rule as the splash's 90 s timeout.

---

## 4. Kernel slices (in order, each independently testable)

**S1 — stash the geometry.** `kmain_mb2` currently consumes the fb's physical address
into the physmap pointer. Keep a `static FB_INFO` (phys addr, size, width, height,
pitch, bpp, rgb bitfields) written once at boot; `[fb]` already prints most of it.

**S2 — `MemAttr::WriteCombine`.** New `akuma-mmu` variant: x86_64 encode = PWT 0,
PCD 0, **PAT bit (12) 1** → PAT entry 4 = WC (already programmed on every CPU by
`map_wc`/`pat_init_ap`). Pin it with the existing `encode()` bit-pattern test style
(`paging.rs` has the precedent, including the "bit 12 of a large PDE is PAT, not an
address bit" note).

**S3 — the fd.** `FileDescriptor::DevFb` + `openat` arm (`/dev/fb0`; `EBUSY` per §3)
+ `dev_node` registry entry (char device, so `ls /dev` is honest) + `fstat` size =
fb size. Model: the `DevDsp` arm at glue `fs.rs:1835`.

**S4 — the ioctls.** Glue ioctl arm keyed on `DevFb` (model: the OSS ioctls in glue
`term.rs`). §2 defines the answers; all fields come from `FB_INFO`. A C probe can
diff every byte of both structs against a real Linux `/dev/fb0` (same-binary-on-both
calibration, the `fbstress`/`md5probe` convention).

**S5 — device mmap.** A third mapping kind in the mmap plan (`akuma-syscalls-mem`
distinguishes anonymous/file-backed today): fd is `DevFb` → region records
`(phys = FB_INFO.addr, len = fb size, attr = WriteCombine)`; the amd64 `mm` fault/
establish path maps each 4 KiB with `map_page_in(root, va, pa + off, USER_RW,
WriteCombine)`. Eager is fine (~4.3k PTEs for 16.9 MiB). Require
`MAP_SHARED | PROT_READ|PROT_WRITE`, reject `MAP_FIXED` onto kernel VA (existing
check), no `PROT_EXEC` (W^X untouched). Teardown: existing munmap/exit paths.

**S6 — ownership + handoff.** §3. Includes the panic path.

**S7 — optional, later: `memfd_create`** (x86-64 #319). *No longer needed for the JIT*
(§5b B works with plain `mprotect`, measured 2026-10-04); only a Mesa escape hatch
(§5 C) would want it. Small, known-shaped syscall; not on the critical path.

**Verification harness.**

- C probe `userspace/amd64/fbprobe/fbprobe.c` (the `fdprobe` convention, raw
  syscalls, built by `mkdisk.sh` like `ruststd`): open → ioctl both structs → mmap →
  write a self-identifying gradient → read back → exit-status verdict. Runs
  identically on Linux with a real fbdev as the control.
- Std-Rust probe (first cut of the demo's `fb.rs`) does the same through
  `libc`/`std::fs`.
- Timing printed with `clock_gettime(CLOCK_MONOTONIC)`: full-screen fill MB/s must
  land near the kernel's WC number (3 GB/s). If it reads ~71 MB/s, the user PTEs got
  no PAT bit — that failure mode is what the `[fb] wc` line taught us to look for.
- Boot-suite arm: spawn a tiny no-std guest program that maps 4 KiB of the fb and
  round-trips a pattern (the `fdprobe` pattern) so the mmap path fails a boot rather
  than a demo.

---

## 5. The wgpu side: what the research found

### 5a. Custom backends are a supported wgpu mechanism — no fork needed

Verified against wgpu trunk (`examples/standalone/custom_backend`, wgpu 30):

```toml
wgpu = { version = "30", default-features = false, features = ["custom", "wgsl"] }
```

A backend implements `wgpu::custom::{InstanceInterface, AdapterInterface,
DeviceInterface, QueueInterface}` (+ the per-resource dispatch types: buffer,
texture, shader module, ...) and hands it to `wgpu::Instance::from_custom(...)`;
callers recover it with `Resource::as_custom::<T>()`. Upstream is actively improving
this path (issues #8826/#9761/#9232, PR #9605). **This is exactly the seam for a
"wgpu-over-framebuffer backend".**

Prior art: **`jgraef/wgpu-cpu`** — a software wgpu backend: CPU rasterizer +
**`naga-cranelift`** (a naga→cranelift compiler backend, so WGSL executes as JITed
x86) + optional `softbuffer` presentation; runs teapot/bunny examples. Caveat: the
repo **has no license file** — use it as an architecture reference only, do not
vendor code from it.

### 5b. Executing shaders on Akuma: three options

| option | mechanism | verdict |
|---|---|---|
| **A. interpret, don't JIT** | backend parses WGSL with `naga` (`wgsl-in`, pure Rust) and walks the IR per-vertex/fragment in Rust; no executable memory at all | **done first (M3); now the slow path to replace.** Slow (maybe 10-50x) but the demo's shaders are trivial and rio's sugarloaf shaders are simple (glyph-atlas quads + SDF). Zero kernel changes beyond §4 |
| **B. JIT via RW→RX `mprotect`** | emit code into an RW anonymous page, `mprotect` it to R+X, run it; to patch, flip back to RW. **No W+X page ever exists, so the kernel's W^X policy is not violated and no kernel change or `memfd_create` is needed.** Originally written up here as "needs dual mapping"; that was wrong — `jit_probe` (kernel repo, `userspace/jitprobe/c/`) passed every arm on the trashcan on 2026-10-04, native-speed loop included | **viable today.** Cost is userspace only: someone must lower naga IR to machine code (hand-rolled x86-64 emitter for the few ops sugarloaf needs, or cranelift — which strains the tiny-deps rule and its on-box musl build is unverified) |
| **C. Mesa lavapipe + stock wgpu** | CPU Vulkan ICD; wgpu's normal Vulkan backend talks to it | no custom-backend work at all, but a large foreign dependency, needs `memfd_create` and more ABI surface; keep as an escape hatch, not the plan |

The demo therefore has **two render paths behind one screen**:

1. `softrender` — a ~300-line software rasterizer (perspective-correct textured
   triangles + z-buffer), used until (and as a fallback beside) the wgpu path. This
   is also the guaranteed-looking path: it can reuse `akuma-fbcon`'s splash math
   (hue wave, brightness swell) for the logo's shading.
2. `wgpu-fb` — the custom backend (option A) rendering the same scene from WGSL.
   Milestone: the same pixels out of both paths is the acceptance test.

### 5c. What this buys for rio (the actual goal)

rio's stack is `rio-window` (their in-tree winit fork — it already carries a
non-X11/Wayland platform precedent: Redox's `orbital`) → **sugarloaf** (their wgpu
rendering library) → wgpu. On Akuma that becomes:

- `rio-window`: a new *framebuffer platform* — screen = `/dev/fb0`, input = the
  console tty (the pump/line discipline already deliver keys and `^C`).
- sugarloaf/wgpu: our custom backend (§5b), one `Surface` = the fb mapping.

Nothing in that stack needs to know about Akuma-private syscalls — which is the whole
reason the substrate must be `/dev/fb0` + standard ABI, and why `libakuma` is banned
from the demo.

---

## 6. The demo: `userspace/akuma-cli-wgpu`

**Naming note:** `userspace/akuma-cli` does not exist in any branch (checked all
refs). The model to copy is `userspace/amd64/ruststd`: an **ordinary Linux binary**,
`x86_64-unknown-linux-musl`, real `std`, built by `mkdisk.sh` with the musl
toolchain, staged onto the disk — *not* a `userspace/` workspace member (that
workspace targets `aarch64-unknown-none` + libakuma, exactly what we must not use).

Layout:

```
userspace/akuma-cli-wgpu/
  Cargo.toml            # std, libc (or rustix); no libakuma
  src/main.rs           # arg parsing (--direct | --wgpu), fps, exit on q/^C
  src/fb.rs             # open /dev/fb0, FBIOGET_*, mmap, pixel-format adapters
  src/catlogo.rs        # include_str!("../../../amd64/src/akuma_40.txt") -> alpha/height field
  src/softrender.rs     # software rasterizer: perspective triangles, z-buffer, hue-wave shading
  src/clock.rs          # CLOCK_MONOTONIC frame pacing (adaptive, the splash tick_ms idea)
  src/wgpu_backend/     # the custom wgpu backend (§5b option A) -- milestone 2
```

The screensaver: the cat mark extruded into a 3D mesh (the ASCII art's ink density
drives displacement — the "height field" trick makes the 2D logo genuinely 3D without
modelling), spinning with a slow hue wave over it (the splash's own palette math),
density-adaptive resolution (render at half res and blit when frames get expensive —
`splash::tick_ms` precedent). Single-buffered present is acceptable for a screensaver;
the `softrender` shadow-buffer blit is there if tearing offends.

Milestones:

1. **M1 (kernel S1-S5 + fbprobe):** `/dev/fb0` opens, ioctls match Linux byte for
   byte, mmap fill runs at WC speed on the trashcan.
2. **M2 (demo direct path):** cat logo spins via `softrender` on the metal. Ships the
   moment M1 does.
3. **M3 (wgpu backend, option A):** same scene through `wgpu` with the custom
   backend; WGSL interpreted by naga. Acceptance: both paths produce the same frames.
4. **M4 (host CI):** the backend + softrender run on Linux against a
   `/dev/fb0`-shaped mock (the `fbstress` same-binary convention), so wgpu-path
   regressions are caught without the metal.

---

## 7. Testing rigs — a known gap

`microvm` (the standard local rig) has **no framebuffer**, and the console probe's
PVH boot likewise has none (`map_wc` is not even exercised there). Options:

- **The trashcan** is the only current real fb (3840x2160, `wc on`). Every M1/M2
  acceptance lands there first. GRUB menu has the `.prev` rollback kernels; add
  `/boot/akuma-amd64.prev2` discipline before installing the fbdev kernel.
- **A GRUB rig for QEMU:** boot the kernel via multiboot2 under `q35`/`pc` with
  `-vga std` and `-display sdl|vnc` — this is the same shape the console probe's USB
  rig already uses (`--usb` boots GRUB under q35). This gives a *local* fb for
  development and screenshots, at 1024x768-ish instead of 4K. Small `run.sh`
  variant (`FB=grub`), worth building before M1.

## 8. Risks / open questions

- **Tearing:** no flip mechanism (firmware scanout). Screensaver: fine. rio: acceptable
  initially; the real fix is a real KMS driver someday, not an fbdev concern.
- **Panic while the fb is owned** — S6's must-not-hide-failure rule; test it
  deliberately (a probe that owns the fb then triggers `ud`).
- **wgpu-cpu is unlicensed** — architecture reference only; our backend is
  first-party code (BSD-2-Clause like the rest).
- **`naga` + `wgpu` custom + `cranelift`-free build size** on the box: heavy crates;
  the on-box nightly musl toolchain has built heavier (llama.cpp), but M3 compile
  times on the trashcan should be measured before promising in-box builds; the dev
  loop can cross-build from the Ubuntu personality.
- **WGSL interpreter completeness** (option A) is scoped to what sugarloaf needs
  (fragment shaders over an atlas, simple uniforms) — enumerate sugarloaf's shaders
  before promising M3 dates.
- **`memfd_create`** only if the Mesa escape hatch is ever taken (S7); option B does not need it.
