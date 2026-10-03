//! The fbdev interface: `/dev/fb0`, three ioctls, one mmap.
//!
//! This is the userspace half of kernel slices S1–S5 in
//! `docs/runbooks/amd64-fbdev-wgpu-demo.md`. The whole surface is the
//! Linux-standard framebuffer API — nothing akuma-private, so rio and wgpu
//! can sit on it unmodified:
//!
//! * `FBIOGET_VSCREENINFO` (0x4600) — visible geometry + RGB bitfields
//! * `FBIOPUT_VSCREENINFO` (0x4601) — the kernel refuses mode changes with
//!   EINVAL and accepts an identical write-back; we never need it, it exists
//!   so generic fbdev clients (and the fbprobe calibration) behave.
//! * `FBIOGET_FSCREENINFO` (0x4602) — `id: "akuma-fb"`, `smem_len`, the pitch
//!   (`line_length`), `FB_VISUAL_TRUECOLOR`, `FB_TYPE_PACKED_PIXEL`
//! * `mmap(MAP_SHARED, PROT_READ|PROT_WRITE)` — the pixels themselves,
//!   mapped write-combining (kernel slice S5: user PTEs carry the PAT bit;
//!   ~3 GB/s on the trashcan, vs 71 MB/s if the bit were missing).
//!
//! `FBIOPAN_DISPLAY` (0x4606) is a no-op on the kernel side (single static
//! scanout); present = one contiguous row-wise copy into the mapping, the
//! same shape as the kernel's own `Framebuffer::fill` (row-wise `rep stosd`
//! + one sfence), which is what keeps WC happy: full spans, no straddled
//! lines.
//!
//! The fbdev structs are declared here rather than taken from libc so the
//! exact ABI bytes are visible in our tree — they must match Linux's
//! `<linux/fb.h>` field for field on x86_64, and the kernel implements the
//! other side of exactly these bytes.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;

// ---------------------------------------------------------------------------
// <linux/fb.h> ioctl numbers (u32 ioctls on x86_64).
// musl's `ioctl(2)` declares the request as `int`; glibc as `unsigned long`.
// ---------------------------------------------------------------------------
#[cfg(target_env = "musl")]
type IoctlNum = libc::c_int;
#[cfg(not(target_env = "musl"))]
type IoctlNum = libc::c_ulong;

pub const FBIOGET_VSCREENINFO: IoctlNum = 0x4600;
// Implemented by the kernel (S4) and exercised by fbprobe; the demo itself
// never puts a mode.
#[allow(dead_code)]
pub const FBIOPUT_VSCREENINFO: IoctlNum = 0x4601;
pub const FBIOGET_FSCREENINFO: IoctlNum = 0x4602;
// A no-op on the kernel side (single static scanout, nothing to pan).
#[allow(dead_code)]
pub const FBIOPAN_DISPLAY: IoctlNum = 0x4606;

pub const FB_TYPE_PACKED_PIXEL: u32 = 0;
pub const FB_VISUAL_TRUECOLOR: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FbBitfield {
    offset: u32,
    length: u32,
    msb_right: u32,
}

/// Mirrors `struct fb_var_screeninfo`. Every field u32; the ioctl only reads
/// this, but the size must be exact (the kernel `copy_to_user`s the whole
/// struct, slice S4).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FbVarScreeninfo {
    pub xres: u32,
    pub yres: u32,
    pub xres_virtual: u32,
    pub yres_virtual: u32,
    pub xoffset: u32,
    pub yoffset: u32,
    pub bits_per_pixel: u32,
    pub grayscale: u32,
    red: FbBitfield,
    green: FbBitfield,
    blue: FbBitfield,
    transp: FbBitfield,
    pub nonstd: u32,
    pub activate: u32,
    pub height: u32,
    pub width: u32,
    pub accel_flags: u32,
    // timings — we carry them verbatim (FBIOPUT round-trip checks identical)
    pub pixclock: u32,
    pub left_margin: u32,
    pub right_margin: u32,
    pub upper_margin: u32,
    pub lower_margin: u32,
    pub hsync_len: u32,
    pub vsync_len: u32,
    pub sync: u32,
    pub vmode: u32,
    pub rotate: u32,
    pub colorspace: u32,
    spare: [u32; 4],
}

/// Mirrors `struct fb_fix_screeninfo` — note the two `unsigned long`s that
/// make this x86_64-shaped (this struct is 80 bytes here; it is not portable
/// to ILP32, which we do not care about: the kernel is x86_64-only).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FbFixScreeninfo {
    pub id: [libc::c_char; 16],
    pub smem_start: libc::c_ulong,
    pub smem_len: u32,
    pub type_: u32,
    pub type_aux: u32,
    pub visual: u32,
    pub xpanstep: u32,
    pub ypanstep: u32,
    pub ywrapstep: u32,
    pub line_length: u32,
    pub mmio_start: libc::c_ulong,
    pub mmio_len: u32,
    pub accel: u32,
    pub capabilities: u16,
    reserved: [u16; 2],
}

impl Default for FbFixScreeninfo {
    fn default() -> Self {
        // id[16] zeroed, everything else zero — `..Default::default()` is not
        // derivable because of the array, and a zeroed fix-screeninfo is the
        // only sane default.
        unsafe { std::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// Pixel formats
// ---------------------------------------------------------------------------

/// Where the channels live inside one pixel, straight from the var-info
/// bitfields. The kernel reports whatever the firmware scanout gave GRUB
/// (the trashcan: 32 bpp, 8/8/8); we render into a canonical 0x00RRGGBB
/// `u32` and translate once, in `present`, on the way out.
#[derive(Clone, Copy, Debug)]
pub struct PixelFormat {
    pub bits_per_pixel: u32,
    r: (u32, u32),
    g: (u32, u32),
    b: (u32, u32),
}

impl PixelFormat {
    fn from_var(v: &FbVarScreeninfo) -> Option<PixelFormat> {
        // The kernel fills the red/green/blue bitfields from the multiboot2
        // RGB framebuffer tag (S1); transp stays zero on this scanout.
        if v.red.length == 0 || v.green.length == 0 || v.blue.length == 0 {
            return None;
        }
        Some(PixelFormat {
            bits_per_pixel: v.bits_per_pixel,
            r: (v.red.offset, v.red.length),
            g: (v.green.offset, v.green.length),
            b: (v.blue.offset, v.blue.length),
        })
    }

    /// Canonical `0x00RRGGBB` -> device pixel bits. Channel values are taken
    /// from the top of each byte and shifted into their slot; narrow slots
    /// (<8 bits) get their top bits replicated down (5-bit blue ends up with
    /// its top 3 bits duplicated), which is what fbcon's `PixelFormat` does
    /// kernel-side. Depths here are <= 32, so u32 carries any pixel.
    #[inline]
    pub fn pack(&self, rgb: u32) -> u32 {
        fn chan(rgb: u32, shift: u32) -> u32 {
            (rgb >> shift) & 0xff
        }
        fn stretch(v: u32, len: u32) -> u32 {
            match len {
                0 => 0,
                8 => v,
                n if n < 8 => {
                    let m = v >> (8 - n);
                    let mut out = m;
                    let mut filled = n;
                    while filled < 8 {
                        out = (out << n) | m;
                        filled += n;
                    }
                    out >> (filled - 8)
                }
                _ => v & 0xff, // >8-bit channels: keep low byte
            }
        }
        (stretch(chan(rgb, 16), self.r.1) << self.r.0)
            | (stretch(chan(rgb, 8), self.g.1) << self.g.0)
            | (stretch(chan(rgb, 0), self.b.1) << self.b.0)
    }
}

// ---------------------------------------------------------------------------
// Frame — the renderer's RAM surface (canonical 0x00RRGGBB)
// ---------------------------------------------------------------------------

/// An in-memory RGBA surface the rasterizer draws into. Composing in cached
/// WB memory and presenting as one contiguous copy is deliberate: random
/// access into a WC mapping is the 71 MB/s failure mode; full-span writes
/// are the 3 GB/s one.
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub buf: Vec<u32>,
}

impl Frame {
    pub fn new(width: usize, height: usize) -> Frame {
        Frame {
            width,
            height,
            buf: vec![0; width * height],
        }
    }

    #[inline]
    pub fn clear(&mut self, rgb: u32) {
        self.buf.fill(rgb);
    }

    #[inline]
    pub fn pixel(&mut self, x: i64, y: i64, rgb: u32) {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return;
        }
        let (x, y) = (x as usize, y as usize);
        self.buf[y * self.width + x] = rgb;
    }

    /// Fill `[xa, xb)` on scanline `y` — the rasterizer's emit primitive.
    /// Kept for the M3 wgpu backend, which fills spans the same way.
    #[allow(dead_code)]
    #[inline]
    pub fn span(&mut self, y: i64, xa: i64, xb: i64, rgb: u32) {
        if y < 0 || y >= self.height as i64 {
            return;
        }
        let xa = xa.clamp(0, self.width as i64) as usize;
        let xb = xb.clamp(0, self.width as i64) as usize;
        if xa >= xb {
            return;
        }
        let y = y as usize;
        self.buf[y * self.width + xa..y * self.width + xb].fill(rgb);
    }
}

// ---------------------------------------------------------------------------
// FbDevice — /dev/fb0 open + mmap + present
// ---------------------------------------------------------------------------

/// An open `/dev/fb0`. Per the ownership model (slice S6): first open wins,
/// a second concurrent open gets EBUSY from the kernel; our `Drop` closes
/// the fd, which releases the screen back to fbcon.
pub struct FbDevice {
    // Not read directly — this is the fd's lifetime. Closing it in `Drop`
    // is what releases the screen back to fbcon (kernel slice S6).
    #[allow(dead_code)]
    dev: File,
    map: *mut u8,
    map_len: usize,
    pub width: usize,
    pub height: usize,
    /// stride in bytes, from `fix.line_length`
    pub pitch: usize,
    pub format: PixelFormat,
    pub id: String,
    pub smem_len: u32,
}

// The mapping is only ever touched through `present` on `&self`, and the
// pointer is page-aligned kernel memory for this fd alone. Sending it
// across threads is fine; `Sync` keeps the door open for a render thread.
unsafe impl Send for FbDevice {}
unsafe impl Sync for FbDevice {}

impl FbDevice {
    /// Open and map the framebuffer. `path` defaults to `/dev/fb0`.
    pub fn open(path: &str) -> io::Result<FbDevice> {
        let dev = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;

        let mut var = FbVarScreeninfo::default();
        // SAFETY: ioctl with a pointer to exactly the struct the fbdev ABI
        // expects; the kernel copies out `std::mem::size_of` bytes.
        if unsafe { libc::ioctl(dev.as_raw_fd(), FBIOGET_VSCREENINFO, &mut var) } == -1 {
            return Err(io::Error::last_os_error());
        }
        let mut fix = FbFixScreeninfo::default();
        if unsafe { libc::ioctl(dev.as_raw_fd(), FBIOGET_FSCREENINFO, &mut fix) } == -1 {
            return Err(io::Error::last_os_error());
        }

        if fix.type_ != FB_TYPE_PACKED_PIXEL || fix.visual != FB_VISUAL_TRUECOLOR {
            return Err(io::Error::other(format!(
                "unsupported framebuffer type {} visual {} (want packed truecolor)",
                fix.type_, fix.visual
            )));
        }
        let format = PixelFormat::from_var(&var).ok_or_else(|| {
            io::Error::other("var-screeninfo carries empty RGB bitfields")
        })?;
        if !matches!(format.bits_per_pixel, 16 | 24 | 32) {
            return Err(io::Error::other(format!(
                "unsupported depth {} (want 16/24/32)",
                format.bits_per_pixel
            )));
        }

        let width = var.xres as usize;
        let height = var.yres as usize;
        let pitch = fix.line_length as usize;
        let map_len = fix.smem_len as usize;
        if width == 0 || height == 0 || pitch == 0 || map_len < height * pitch {
            return Err(io::Error::other(format!(
                "nonsensical geometry {width}x{height} pitch {pitch} smem {map_len}"
            )));
        }

        // Slice S5: the kernel maps these pages into our address space with
        // MemAttr::WriteCombine (PWT=0, PCD=0, PAT bit set -> PAT entry 4,
        // programmed WC by map_wc on every CPU). MAP_SHARED is required —
        // the kernel refuses MAP_PRIVATE with EINVAL because write-combined
        // dirty-page writeback of a device mapping is nonsense.
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                dev.as_raw_fd(),
                0,
            )
        };
        if map == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(FbDevice {
            id: String::from_utf8_lossy(
                &fix.id.iter().map(|&c| c as u8).collect::<Vec<_>>(),
            )
            .trim_end_matches('\0')
            .to_string(),
            dev,
            map: map as *mut u8,
            map_len,
            width,
            height,
            pitch,
            format,
            smem_len: fix.smem_len,
        })
    }

    /// Present a frame: canonical RGBA -> device pixels, one full-span
    /// row at a time. This is the only writer of the mapping, and it
    /// writes every byte of every visible row (WC-friendly, no
    /// read-modify-write of the device memory).
    pub fn present(&self, frame: &Frame) {
        debug_assert_eq!(frame.width, self.width);
        debug_assert_eq!(frame.height, self.height);
        let bpp = (self.format.bits_per_pixel / 8) as usize;
        for (row, src) in frame.buf.chunks_exact(frame.width).enumerate() {
            let dst = unsafe { self.map.add(row * self.pitch) };
            match bpp {
                4 => {
                    let d = dst as *mut u32;
                    for (i, &px) in src.iter().enumerate() {
                        // pack() returns a u64; 32-bit formats only need the
                        // low word.
                        unsafe { d.add(i).write(self.pack32(px)) };
                    }
                }
                3 => {
                    for (i, &px) in src.iter().enumerate() {
                        let p = self.pack32(px);
                        let b = dst as *mut u8;
                        unsafe {
                            b.add(i * 3).write((p & 0xff) as u8);
                            b.add(i * 3 + 1).write(((p >> 8) & 0xff) as u8);
                            b.add(i * 3 + 2).write(((p >> 16) & 0xff) as u8);
                        }
                    }
                }
                2 => {
                    let d = dst as *mut u16;
                    for (i, &px) in src.iter().enumerate() {
                        unsafe { d.add(i).write(self.pack32(px) as u16) };
                    }
                }
                _ => unreachable!("checked at open"),
            }
        }
        // Single-buffer static scanout: no panning, so no FBIOPAN_DISPLAY.
        // The kernel's fbcon does an sfence on its own writes; user stores
        // to WC memory are made visible by the chipset on their own — there
        // is nothing to flush from userspace.
    }

    #[inline]
    fn pack32(&self, rgb: u32) -> u32 {
        self.format.pack(rgb)
    }
}

impl Drop for FbDevice {
    fn drop(&mut self) {
        // Order matters: unmap before close (close releases ownership back
        // to fbcon, slice S6 — the kernel clears and redraws the banner).
        unsafe { libc::munmap(self.map as *mut libc::c_void, self.map_len) };
        // `dev: File` closes itself after this.
    }
}
