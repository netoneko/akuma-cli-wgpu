//! `akuma-wgpu` — the akuma-cli screensaver, extruded into 3D and rendered
//! straight into the framebuffer.
//!
//! Template: github.com/netoneko/akuma-cli (same CLI surface — `screensaver`,
//! `matrix`, `--latin`, arrow keys switch assets, `q`/Esc quits, metrics on
//! exit). Difference: instead of ANSI escape codes into a terminal grid, a
//! software rasterizer draws into `/dev/fb0` via the Linux-standard fbdev
//! interface — three ioctls and an mmap, zero akuma-private syscalls, the
//! surface rio and wgpu will eventually sit on.
//!
//! Subcommands:
//!   screensaver   3D cat logo over the Matrix backdrop (the default show)
//!   matrix        backdrop only
//!   selftest      render headlessly into RAM and print stats — runs
//!                 anywhere, no /dev/fb0 needed (the fbstress convention:
//!                 same binary, Linux and Akuma)
//!
//! Exit metrics mirror the template's, in the frame buffer's units: frames,
//! fps, MB written, MB/s — directly comparable to the kernel banner's
//! `[fb] ... clear 10.9ms = 3026MB/s` line.

// deep trait-solver recursion: proving our custom-backend futures Send walks
// the whole wgpu dispatch enum (including wgpu-core's hub types), which
// overflows the default depth of 128 with a spurious "overflow evaluating
// the requirement" warning
#![recursion_limit = "512"]

mod catlogo;
mod clock;
mod fb;
mod input;
mod rng;
mod softrender;
mod wgpu_backend;

use std::io::Write as _;
use clock::{monotonic, FpsMeter, Pacer};

use catlogo::HeightField;
use fb::{FbDevice, Frame};
use input::{Key, RawTty};
use softrender::{Renderer, Scene};

/// Which renderer draws the frames: the software rasterizer (M2, the
/// reference) or the wgpu custom backend (M3). Same frame signature on both
/// — that is the milestone's whole point.
enum RenderPath {
    Soft(Renderer),
    Wgpu(wgpu_backend::WgpuRenderer),
}

impl RenderPath {
    fn new(wgpu_path: bool, w: usize, h: usize) -> RenderPath {
        if wgpu_path {
            eprintln!("[wgpu] {}", wgpu_backend::STATUS);
            RenderPath::Wgpu(wgpu_backend::WgpuRenderer::new(w, h))
        } else {
            RenderPath::Soft(Renderer::new(w, h))
        }
    }

    fn render(&mut self, frame: &mut Frame, scene: &mut Scene, time: f32, with_rain: bool) {
        match self {
            RenderPath::Soft(r) => r.render(frame, scene, time, with_rain),
            RenderPath::Wgpu(r) => r.render(frame, scene, time, with_rain),
        }
    }
}

/// The assets, embedded like the template embeds them.
const AKUMA_20: &str = include_str!("akuma_20.txt");
const AKUMA_40: &str = include_str!("akuma_40.txt");
const AKUMA_79: &str = include_str!("akuma_79.txt");
const AKUMA_120: &str = include_str!("akuma_120.txt");
const ASSETS: [&str; 4] = [AKUMA_40, AKUMA_79, AKUMA_120, AKUMA_20];
const ASSET_NAMES: [&str; 4] = ["akuma_40", "akuma_79", "akuma_120", "akuma_20"];
/// where the live show starts: the biggest logo (index into ASSETS). The
/// selftest still walks all four in ASSETS order; Left/Right cycles from here.
const DEFAULT_ASSET: usize = 2;

struct Options {
    mode: Mode,
    latin: bool,
    fps: u32,
    timeout_secs: u64,
    fb_path: String,
    width: usize,
    height: usize,
    frames: u64,
    /// render through the wgpu custom backend (milestone M3) instead of
    /// the software rasterizer
    wgpu: bool,
    /// selftest: print ASCII maps of final frames
    dump: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Screensaver,
    Matrix,
    Selftest,
    ExecSelftest,
}

const USAGE: &str = "\
akuma-wgpu — the akuma-cli screensaver, 3D, straight into the framebuffer

USAGE:
    akuma-wgpu <COMMAND> [OPTIONS]

COMMANDS:
    screensaver    3D cat logo over the Matrix backdrop
    matrix         Matrix backdrop only
    selftest       render headlessly into RAM, print stats, exit
    exec-selftest  diff the shader executors (interp/vm/jit) bit for bit

OPTIONS:
    --latin        Latin characters in the backdrop (default: katakana)
    --fps <N>      target frame rate (default 60)
    --timeout <S>  exit after S seconds (0 = run until q/Esc/^C; default 0)
    --fb <PATH>    framebuffer device (default /dev/fb0)
    --wgpu         render through the wgpu custom backend (M3) instead of
                   the software rasterizer (both draw the same frames)
    --w <W>        selftest frame width (default 1280)
    --h <H>        selftest frame height (default 720)
    --frames <N>   selftest frame count (default 120)
    --dump         selftest also prints an ASCII map of each last frame
    -h, --help     this text

CONTROLS:
    Left/Right     switch asset     q / Esc / ^C    quit (metrics on exit)
";

fn parse_args(argv: &[String]) -> Result<Options, String> {
    let mut mode = Mode::Screensaver;
    let mut opt = Options {
        mode,
        latin: false,
        fps: 60,
        timeout_secs: 0,
        fb_path: "/dev/fb0".to_string(),
        width: 1280,
        height: 720,
        frames: 120,
        wgpu: false,
        dump: false,
    };
    let mut it = argv.iter();
    if let Some(first) = it.next() {
        match first.as_str() {
            "screensaver" => mode = Mode::Screensaver,
            "matrix" => mode = Mode::Matrix,
            "selftest" => mode = Mode::Selftest,
            "exec-selftest" => mode = Mode::ExecSelftest,
            "-h" | "--help" | "help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown command `{other}` (try --help)")),
        }
    }
    let mut flags = it.peekable();
    while let Some(flag) = flags.next() {
        let mut value = |name: &str| -> Result<String, String> {
            flags
                .next()
                .ok_or_else(|| format!("{name} needs a value"))
                .cloned()
        };
        match flag.as_str() {
            "--latin" => opt.latin = true,
            "--wgpu" => opt.wgpu = true,
            "--dump" => opt.dump = true,
            "--fps" => opt.fps = value("--fps")?.parse().map_err(|_| "--fps wants a number")?,
            "--timeout" => {
                opt.timeout_secs = value("--timeout")?
                    .parse()
                    .map_err(|_| "--timeout wants seconds")?
            }
            "--fb" => opt.fb_path = value("--fb")?,
            "--w" => opt.width = value("--w")?.parse().map_err(|_| "--w wants pixels")?,
            "--h" => opt.height = value("--h")?.parse().map_err(|_| "--h wants pixels")?,
            "--frames" => {
                opt.frames = value("--frames")?
                    .parse()
                    .map_err(|_| "--frames wants a count")?
            }
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown option `{other}` (try --help)")),
        }
    }
    opt.mode = mode;
    Ok(opt)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let opts = match parse_args(&argv) {
        Ok(o) => o,
        Err(msg) => {
            // usage text goes to stdout, errors to stderr — like clap
            if msg.trim_start().starts_with("akuma-wgpu") {
                println!("{msg}");
            } else {
                eprintln!("error: {msg}");
                eprintln!("{USAGE}");
                std::process::exit(2);
            }
            return;
        }
    };

    input::install_signal_handlers();

    let code = match opts.mode {
        Mode::Selftest => run_selftest(&opts),
        Mode::ExecSelftest => wgpu_backend::exec_selftest::run(),
        Mode::Matrix => run_show(&opts, /*asset overlay*/ false),
        Mode::Screensaver => run_show(&opts, true),
    };
    std::process::exit(code);
}

// ---------------------------------------------------------------------------
// The show (screensaver | matrix) — the /dev/fb0 path
// ---------------------------------------------------------------------------

fn run_show(opts: &Options, with_asset: bool) -> i32 {
    let dev = match FbDevice::open(&opts.fb_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "error: cannot open {}: {} — is this the akuma kernel with \
                 slices S1-S5 (see docs/runbooks/amd64-fbdev-wgpu-demo.md)?",
                opts.fb_path, e
            );
            return 1;
        }
    };
    eprintln!(
        "[fb] {} {}x{} pitch {} bpp {} smem {} KiB",
        dev.id,
        dev.width,
        dev.height,
        dev.pitch,
        dev.format.bits_per_pixel,
        dev.smem_len / 1024
    );

    let mut frame = Frame::new(dev.width, dev.height);
    let mut renderer = RenderPath::new(opts.wgpu, dev.width, dev.height);

    // the show starts on the biggest logo (akuma_120); Left/Right cycles
    // all four. The matrix subcommand runs with no mesh at all — rain only,
    // like the template's `matrix` mode.
    let mut asset_idx: usize = DEFAULT_ASSET;
    let mut scene = if with_asset {
        build_scene(opts, ASSETS[asset_idx], dev.width, dev.height)
    } else {
        Scene::new(Vec::new(), 1.0, true, dev.width, dev.height)
    };

    let mut tty = RawTty::new().ok();
    let started = monotonic();
    let mut pacer = Pacer::new(opts.fps);
    let mut meter = FpsMeter::new(1.0 / f64::from(opts.fps.max(1)));
    let mut switches: u64 = 0;
    let mut quit_reason = "q/Esc";

    loop {
        let t0 = monotonic();
        let time = (t0 - started) as f32;

        // --- input ---
        if let Some(tty) = &mut tty {
            for key in tty.poll_keys() {
                match key {
                    Key::Quit => {
                        quit_reason = "key";
                    }
                    Key::Left => {
                        if with_asset {
                            asset_idx = (asset_idx + ASSETS.len() - 1) % ASSETS.len();
                            scene = build_scene(opts, ASSETS[asset_idx], dev.width, dev.height);
                            switches += 1;
                        }
                    }
                    Key::Right => {
                        if with_asset {
                            asset_idx = (asset_idx + 1) % ASSETS.len();
                            scene = build_scene(opts, ASSETS[asset_idx], dev.width, dev.height);
                            switches += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
        if input::quit_requested() {
            quit_reason = "^C";
        }

        // --- timeout ---
        if opts.timeout_secs > 0
            && monotonic() - started >= opts.timeout_secs as f64
        {
            quit_reason = "timeout";
        }

        // --- render + present ---
        scene.resize_rain(dev.width, dev.height, true);
        renderer.render(&mut frame, &mut scene, time, true);
        dev.present(&frame);

        meter.frame(monotonic() - t0);
        pacer.finish_frame();

        if quit_reason != "q/Esc" {
            break;
        }
    }

    // Leave the console how we found it: drop closes /dev/fb0 (releasing
    // ownership back to fbcon, kernel slice S6) and RawTty restores termios.
    drop(tty);
    drop(dev);
    print_metrics(opts, &meter, &frame, switches, quit_reason);
    0
}

fn build_scene(_opts: &Options, asset: &str, w: usize, h: usize) -> Scene {
    let hf = HeightField::parse(asset);
    let depth = catlogo::depth_for(&hf);
    let tris = catlogo::extrude(&hf, depth);
    let extent = hf.width.max(hf.height) as f32;
    let mut scene = Scene::new(tris, extent, true, w, h);
    // this panel is dead on the right half: park the logo in the middle of
    // the left half (the rain still spans the whole frame). Only the live
    // show moves; selftest keeps the mid-frame default so its checksums stay
    // about the two renderers.
    scene.center_x = w as f32 * 0.25;
    scene
}

// ---------------------------------------------------------------------------
// selftest — headless, no /dev/fb0, the same binary on Linux and Akuma
// ---------------------------------------------------------------------------

fn run_selftest(opts: &Options) -> i32 {
    println!(
        "selftest: {}x{}, {} frames, fps target {}, latin={}",
        opts.width, opts.height, opts.frames, opts.fps, opts.latin
    );
    let mut frame = Frame::new(opts.width, opts.height);
    let mut renderer = RenderPath::new(opts.wgpu, opts.width, opts.height);
    let mut total_ns: u128 = 0;
    let mut checksum: u32 = 0;

    // one scene per asset: exercises every mesh shape
    for (i, (name, asset)) in ASSET_NAMES.iter().zip(ASSETS.iter()).enumerate() {
        let hf = HeightField::parse(asset);
        let depth = catlogo::depth_for(&hf);
        let tris = catlogo::extrude(&hf, depth);
        let mut scene =
            Scene::new(tris, hf.width.max(hf.height) as f32, true, opts.width, opts.height);
        println!(
            "  {name}: {}x{} cells -> {} triangles",
            hf.width,
            hf.height,
            scene.tris.len()
        );

        let per = (opts.frames / ASSETS.len() as u64).max(1);
        let mut asset_checksum: u32 = 0;
        for k in 0..per {
            let time = (i as f64 * 10.0 + k as f64) as f32 / opts.fps.max(1) as f32;
            let t0 = monotonic();
            renderer.render(&mut frame, &mut scene, time, true);
            total_ns += ((monotonic() - t0) * 1e9) as u128;
            if k == per - 1 {
                let coverage = frame
                    .buf
                    .iter()
                    .filter(|&&p| p != softrender::BG)
                    .count();
                for &p in &frame.buf {
                    asset_checksum =
                        asset_checksum.wrapping_mul(0x0100_0193).wrapping_add(p & 0xff_ffff);
                }
                println!(
                    "    last frame: {:.1}% coverage, fnv1a {:08x}",
                    100.0 * coverage as f64 / frame.buf.len() as f64,
                    asset_checksum
                );
                checksum = checksum.wrapping_add(asset_checksum);
                if opts.dump {
                    dump_ascii(&frame);
                }
            }
        }
    }

    let frames_done = opts.frames.max(1);
    let avg_ms = total_ns as f64 / frames_done as f64 / 1e6;
    let total_px = opts.width as u64 * opts.height as u64 * frames_done;
    println!(
        "  avg render: {avg_ms:.2} ms/frame ({} fps theoretical)",
        (1000.0 / avg_ms).round()
    );
    println!(
        "SELFTEST OK — {} frames, {:.1} Mpx/s, {:.0} MB/s equivalent write rate",
        frames_done,
        total_px as f64 / 1e6 / (total_ns as f64 / 1e9),
        total_px as f64 * 4.0 / 1e6 / (total_ns as f64 / 1e9)
    );
    0
}

// ---------------------------------------------------------------------------
// Selftest helpers
// ---------------------------------------------------------------------------

/// `selftest --dump`: downsample the last frame to ~80x40 cells and print
/// one character per cell, brightness on the same ` .:-=+*#%@` ramp the
/// logo's density ramp uses — the text-mode picture of what would have
/// reached `/dev/fb0`, so the headless selftest shows its work on any
/// machine (the `fbstress`/`fbprobe` show-me convention).
fn dump_ascii(frame: &Frame) {
    const RAMP: &[u8; 10] = b" .:-=+*#%@";
    let cols = frame.width.min(80);
    let rows = frame.height.min(40);
    let mut out = String::with_capacity((cols + 1) * rows);
    for cy in 0..rows {
        let y0 = cy * frame.height / rows;
        let y1 = ((cy + 1) * frame.height / rows).max(y0 + 1);
        for cx in 0..cols {
            let x0 = cx * frame.width / cols;
            let x1 = ((cx + 1) * frame.width / cols).max(x0 + 1);
            // mean luminance of the cell (ITU-R BT.601 weights, scaled x1000
            // to stay in integers). `BG` is near-black, so background cells
            // land on the ramp's first char (` `) with no special case.
            let mut sum: u64 = 0;
            let mut n: u64 = 0;
            for y in y0..y1 {
                let row = &frame.buf[y * frame.width + x0..y * frame.width + x1];
                for &px in row {
                    sum += u64::from((px >> 16) & 0xff) * 299
                        + u64::from((px >> 8) & 0xff) * 587
                        + u64::from(px & 0xff) * 114;
                    n += 1;
                }
            }
            let lum = if n == 0 { 0 } else { (sum / n / 1000) as usize }; // 0..=255
            out.push(RAMP[lum * (RAMP.len() - 1) / 255] as char);
        }
        out.push('\n');
    }
    print!("{out}");
}

// ---------------------------------------------------------------------------
// Exit report — the template's metrics culture, in framebuffer units
// ---------------------------------------------------------------------------
fn print_metrics(opts: &Options, meter: &FpsMeter, frame: &Frame, switches: u64, why: &str) {
    let (dur, frames, fps, over, slowest) = meter.summary();
    let px = frame.width as u64 * frame.height as u64;
    let bytes = px * 4 * frames; // canonical 0x00RRGGBB frames written
    let mbps = if dur > 0.0 {
        bytes as f64 / 1e6 / dur
    } else {
        0.0
    };
    let mut out = String::new();
    out.push_str("\n=== Screensaver Metrics ===\n");
    out.push_str(&format!("Exit: {why}\n"));
    out.push_str(&format!("Duration: {:.2} seconds\n", dur));
    out.push_str(&format!("Frames: {frames}\n"));
    out.push_str(&format!("Average FPS: {fps:.2}\n"));
    out.push_str(&format!("Frame budget: {:.1} ms at {} fps\n", 1000.0 / opts.fps.max(1) as f64, opts.fps));
    out.push_str(&format!("Over-budget frames: {over}\n"));
    out.push_str(&format!("Slowest frame render: {:.2} ms\n", slowest * 1e3));
    out.push_str(&format!(
        "Total pixels: {:.2} Gpx ({}x{} per frame)\n",
        (px * frames) as f64 / 1e9,
        frame.width,
        frame.height
    ));
    out.push_str(&format!("Total bytes written: {:.2} MB\n", bytes as f64 / 1e6));
    out.push_str(&format!("Write rate: {mbps:.0} MB/s  (kernel [fb] clear reference: ~3026 MB/s)\n"));
    out.push_str(&format!("Asset switches: {switches}\n"));
    out.push_str("============================\n");
    let _ = std::io::stderr().write_all(out.as_bytes());
}
