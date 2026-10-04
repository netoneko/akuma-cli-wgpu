//! Milestone M3: the same scene through wgpu.
//!
//! Layout of this module:
//!
//! * `interp`   — the naga IR interpreter (no JIT yet: plan §5b A; RW->RX JIT is feasible)
//! * `backend`  — the `wgpu::custom::*` implementations, including the
//!   fixed-function rasterizer whose contract is softrender's scanline walk
//! * `shaders`  — the WGSL programs, line-for-line ports of softrender's math
//!
//! `WgpuRenderer` exposes exactly softrender's `Renderer` surface
//! (`new(w, h)` / `render(&mut frame, &mut scene, time, with_rain)`) so the
//! two paths are interchangeable in main.rs.
//!
//! Frame flow per render:
//!
//!   1. shared backdrop — `softrender::backdrop` clears the frame and steps
//!      + draws the rain on the CPU. This is *the same code* the software
//!      path runs, so the backdrop is identical by construction (the rain is
//!      a u64-xorshift CPU effect; it has no business in a shader).
//!   2. `queue.write_buffer` — per-frame uniform (time, w, h, extent,
//!      center_x) and the triangle mesh into the storage buffer.
//!   3. `queue.write_texture` — the frame (including rain) into the color
//!      target; the render pass loads it so the logo draws on top, exactly
//!      like the software path draws on top of the rain.
//!   4. render pass — `draw(0..3*n)`: vertex stage per corner through the
//!      interpreter, fixed-function raster + z-test, fragment stage per
//!      covered pixel.
//!   5. `copy_texture_to_buffer` + submit + map — blit rows back into the
//!      frame. bgra8uint means the bytes land in the framebuffer word
//!      untouched.
//!
//! The acceptance test is that the fnv1a checksums of `selftest --wgpu`
//! equal the software-path baselines in the README.

pub mod backend;
pub mod compile;
pub mod exec;
pub mod exec_selftest;
pub mod format;
pub mod gpu_selftest;
pub mod interp;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub mod jit;
pub mod memo;
pub mod opt;
pub mod pool;
pub mod runs;
pub mod program;
pub mod raster;
pub mod vertex;
pub mod vm;
pub mod shaders;
pub mod texture;

/// One-line status of the wgpu path (main.rs prints it when the wgpu path
/// starts).
pub const STATUS: &str = "wgpu path: akuma custom backend — WGSL via naga 30 \
interpreter (JIT on x86-64 / register VM / interpreter fallback), fixed-function raster per softrender contract";

use crate::fb::Frame;
use crate::softrender::{self, Scene};

// ---------------------------------------------------------------------------
// tiny block_on: the custom backend's futures resolve immediately, so a
// noop-waker poll loop is all a single-threaded program needs (no pollster)
// ---------------------------------------------------------------------------

pub(crate) fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    unsafe fn noop(_: *const ()) {}
    unsafe fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        if let Poll::Ready(v) = std::future::Future::poll(fut.as_mut(), &mut cx) {
            return v;
        }
    }
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

pub struct WgpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    color: wgpu::Texture,
    depth: wgpu::Texture,
    /// readback target for the color texture (256-aligned rows)
    readback: wgpu::Buffer,
    readback_pitch: usize,
    /// the mesh storage buffer; grown when a bigger scene arrives
    tris: Option<(wgpu::Buffer, usize)>,
    uniform: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    w: usize,
    h: usize,
}

// WGSL `Tri` layout, per the storage rules the interpreter applies: vec3f
// members are 12 bytes but align 16, so a/b/c sit at 0/16/32; `kind` (u32,
// align 4) follows c at 44; struct size rounds up to align: 48.
const TRI_WGSL_STRIDE: usize = 48;

impl WgpuRenderer {
    pub fn new(width: usize, height: usize) -> WgpuRenderer {
        let instance = wgpu::Instance::from_custom(backend::Instance);
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("wgpu: request_adapter failed on the akuma backend");
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("akuma"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::wgt::ExperimentalFeatures::disabled(),
        }))
        .expect("wgpu: request_device failed on the akuma backend");

        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("akuma color"),
            size: extent(width, height),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("akuma depth"),
            size: extent(width, height),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            // softrender refills its z-buffer with +inf every frame; the
            // render pass clear value below does the same job
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // readback rows are 256-aligned like every wgpu buffer copy
        let readback_pitch = (width * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("akuma readback"),
            size: (readback_pitch * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("akuma uniform"),
            // two vec4<f32>s: time, width, height, extent | center_x, pad
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("akuma scene"),
            source: wgpu::ShaderSource::Wgsl(shaders::SHADERS_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("akuma group 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(32),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(TRI_WGSL_STRIDE as u64),
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("akuma layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("akuma logo"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Uint,
                    // softrender writes opaque color words; no blending of
                    // any kind on this path
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            // culling and the -0.01 epsilon are fixed-function on this GPU
            // (backend.rs raster_tri); the pipeline asks for none
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                // softrender: draw iff z < zbuf (strictly)
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("akuma bindings"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform,
                        offset: 0,
                        size: None,
                    }),
                },
                // the storage buffer is bound lazily (see ensure_tris); a
                // 16-byte placeholder keeps creation honest until then
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform,
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
            ],
        });

        WgpuRenderer {
            device,
            queue,
            color,
            depth,
            readback,
            readback_pitch,
            tris: None,
            uniform,
            pipeline,
            bind_group,
            w: width,
            h: height,
        }
    }

    /// (re)create the mesh storage buffer for a scene of `n` triangles and
    /// rebuild the bind group against it.
    fn ensure_tris(&mut self, n: usize) {
        if let Some((_, cap)) = &self.tris {
            if n <= *cap {
                return;
            }
        }
        let cap = n.max(1024);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("akuma tris"),
            size: (cap * TRI_WGSL_STRIDE) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("akuma bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buf,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        self.tris = Some((buf, cap));
    }

    /// Render one frame through wgpu. Signature-compatible with
    /// `softrender::Renderer::render`.
    pub fn render(&mut self, frame: &mut Frame, scene: &mut Scene, time: f32, with_rain: bool) {
        // 1. the shared backdrop: clear + rain, identical code to the other
        // path — identical bits by construction
        let t0 = crate::clock::monotonic();
        softrender::backdrop(frame, scene, time, with_rain);
        prof::add_ns(0, crate::clock::monotonic() - t0);
        prof::inc(10, 1);

        let n = scene.tris.len();
        if n > 0 {
            self.ensure_tris(n);

            // 2. uniform: time, width, height, extent, center_x (+3 pad)
            let mut uni = [0u8; 32];
            uni[0..4].copy_from_slice(&time.to_le_bytes());
            uni[4..8].copy_from_slice(&(self.w as f32).to_le_bytes());
            uni[8..12].copy_from_slice(&(self.h as f32).to_le_bytes());
            uni[12..16].copy_from_slice(&scene.extent.to_le_bytes());
            uni[16..20].copy_from_slice(&scene.center_x.to_le_bytes());
            self.queue.write_buffer(&self.uniform, 0, &uni);

            // mesh: pack each tri to the WGSL storage stride. Each vec3<f32>
            // member occupies 12 bytes but ALIGNS to 16, so every vertex
            // gets its own 16-byte lane (a@0, b@16, c@32) and `kind` lands
            // right after c, at +44. (The original code packed the 12 floats
            // contiguously — b/c then read shifted by one float, which is
            // the squashed-cat M3 parity bug.)
            let t0 = crate::clock::monotonic();
            let mut mesh = vec![0u8; n * TRI_WGSL_STRIDE];
            for (i, t) in scene.tris.iter().enumerate() {
                let o = i * TRI_WGSL_STRIDE;
                for (v, p3) in [t.a, t.b, t.c].iter().enumerate() {
                    for (c, p) in p3.iter().enumerate() {
                        let at = o + v * 16 + c * 4;
                        mesh[at..at + 4].copy_from_slice(&p.to_le_bytes());
                    }
                }
                mesh[o + 44..o + 48].copy_from_slice(&t.kind.to_le_bytes());
            }
            let (tris_buf, _) = self.tris.as_ref().unwrap();
            self.queue.write_buffer(tris_buf, 0, &mesh);
            prof::add_ns(1, crate::clock::monotonic() - t0);

            // 3. Render in place. The frame (BG + rain so far) already holds
            // exactly the texture's bytes (a little-endian 0x00RRGGBB word is
            // the bytes b, g, r, 0 that the fragment stage writes), so rather
            // than uploading 33 MB at 4K, drawing, copying it to a readback
            // buffer and blitting it back, the colour texture's storage is
            // pointed at the frame's own memory for the duration of the pass.
            const _: () = assert!(cfg!(target_endian = "little"));
            let tex = self
                .color
                .as_custom::<backend::TextureData>()
                .expect("wgpu: colour target is not an akuma texture");
            let backend::TexStore::Color(store) = &*tex.store else { unreachable!() };
            let nbytes = frame.buf.len() * 4;
            assert_eq!(nbytes, self.w * self.h * 4);
            // SAFETY: a Vec<u8> view of the frame's u32 allocation. It is only
            // ever indexed (the pass reads and writes pixels in place), never
            // grown or dropped: it is swapped out again and `forget`-ten below,
            // so the allocation is freed once, by `frame`, with its own layout.
            let alias = unsafe { Vec::from_raw_parts(frame.buf.as_mut_ptr() as *mut u8, nbytes, nbytes) };
            let original = std::mem::replace(&mut *store.lock().unwrap(), alias);

            // 4. render pass: draw the logo over the backdrop already in place
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("akuma") });
            {
                let color_view = self.color.create_view(&wgpu::TextureViewDescriptor::default());
                let depth_view = self.depth.create_view(&wgpu::TextureViewDescriptor::default());
                let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("akuma frame"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            // the z-buffer refill of softrender::render
                            load: wgpu::LoadOp::Clear(f32::INFINITY),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rpass.set_pipeline(&self.pipeline);
                rpass.set_bind_group(0, &self.bind_group, &[]);
                rpass.draw(0..(n as u32 * 3), 0..1);
            }
            let t0 = crate::clock::monotonic();
            let done = self.queue.submit([enc.finish()]);
            self.device
                .poll(wgpu::PollType::Wait { submission_index: Some(done), timeout: None })
                .expect("wgpu: poll failed");
            prof::add_ns(3, crate::clock::monotonic() - t0);

            // put the texture's own storage back; the alias is never dropped
            let alias = std::mem::replace(&mut *store.lock().unwrap(), original);
            std::mem::forget(alias);
        }
    }
}

fn extent(w: usize, h: usize) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: w as u32,
        height: h as u32,
        depth_or_array_layers: 1,
    }
}

/// Phase counters for `AKUMA_PROF=1` (stderr report when a `WgpuRenderer`
/// drops). Plain atomics so the backend's executor can bump them from
/// wherever it runs; nanoseconds via `clock::monotonic`, never std `Instant`.
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub const NAMES: [&str; 13] = [
        "backdrop (cpu rain)",
        "pack mesh",
        "upload color tex",
        "submit+poll total",
        "  vertex stage",
        "  raster excl. fragment",
        "  fragment stage",
        "readback copy+blit",
        "vertex invocations",
        "fragment invocations",
        "frames",
        "wide batches",
        "wide batches diverged",
    ];
    pub static C: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];

    pub fn add_ns(i: usize, secs: f64) {
        C[i].fetch_add((secs * 1e9) as u64, Relaxed);
    }
    pub fn inc(i: usize, n: u64) {
        C[i].fetch_add(n, Relaxed);
    }
    /// 0 = off, 1 = per-phase (cheap), 2 = also per-fragment timers (each is
    /// a syscall on Akuma, so it distorts everything it sits inside)
    pub fn level() -> u8 {
        use std::sync::OnceLock;
        static E: OnceLock<u8> = OnceLock::new();
        *E.get_or_init(|| match std::env::var("AKUMA_PROF").as_deref() {
            Ok("2") => 2,
            Ok(_) => 1,
            Err(_) => 0,
        })
    }
    pub fn enabled() -> bool {
        level() > 0
    }
    pub fn report() {
        let frames = C[10].load(Relaxed).max(1);
        eprintln!("[prof] {frames} frames");
        for i in 0..8 {
            let mut ns = C[i].load(Relaxed);
            if i == 5 {
                // slot 5 holds the whole raster loop; fragment time is nested in it
                ns = ns.saturating_sub(C[6].load(Relaxed));
            }
            eprintln!("[prof] {:<26} {:>9.3} ms/frame", NAMES[i], ns as f64 / 1e6 / frames as f64);
        }
        let v = C[8].load(Relaxed);
        let f = C[9].load(Relaxed);
        eprintln!("[prof] vertex invocations   {:>9}/frame, {:.0} ns each", v / frames, C[4].load(Relaxed) as f64 / v.max(1) as f64);
        eprintln!("[prof] fragment invocations {:>9}/frame, {:.0} ns each", f / frames, C[6].load(Relaxed) as f64 / f.max(1) as f64);
        let (wb, wd) = (C[11].load(Relaxed), C[12].load(Relaxed));
        if wb > 0 {
            eprintln!("[prof] wide batches {wb}, diverged {wd} ({:.1}%)", wd as f64 * 100.0 / wb as f64);
        }
        for c in C.iter() {
            c.store(0, Relaxed);
        }
    }
}

impl Drop for WgpuRenderer {
    fn drop(&mut self) {
        if prof::enabled() {
            prof::report();
        }
    }
}
