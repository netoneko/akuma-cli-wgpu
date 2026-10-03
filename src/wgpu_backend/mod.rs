//! Milestone M3: the same scene through wgpu.
//!
//! Layout of this module:
//!
//! * `interp`   — the naga IR interpreter (no JIT: kernel W^X, plan §5b A)
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
pub mod interp;
pub mod shaders;

/// One-line status of the wgpu path (main.rs prints it when the wgpu path
/// starts).
pub const STATUS: &str = "wgpu path: akuma custom backend — WGSL via naga 30 \
interpreter (no JIT), fixed-function raster per softrender contract";

use crate::fb::Frame;
use crate::softrender::{self, Scene};

// ---------------------------------------------------------------------------
// tiny block_on: the custom backend's futures resolve immediately, so a
// noop-waker poll loop is all a single-threaded program needs (no pollster)
// ---------------------------------------------------------------------------

fn block_on<F: std::future::Future>(fut: F) -> F::Output {
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

const TRI_WGSL_STRIDE: usize = 48; // vec3f has 16-byte alignment: 3 * 16

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
        softrender::backdrop(frame, scene, time, with_rain);

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

            // mesh: pad each tri to the WGSL storage stride
            let mut mesh = vec![0u8; n * TRI_WGSL_STRIDE];
            for (i, t) in scene.tris.iter().enumerate() {
                let o = i * TRI_WGSL_STRIDE;
                for (c, p) in t.a.iter().chain(t.b.iter()).chain(t.c.iter()).enumerate() {
                    mesh[o + c * 4..o + c * 4 + 4].copy_from_slice(&p.to_le_bytes());
                }
            }
            let (tris_buf, _) = self.tris.as_ref().unwrap();
            self.queue.write_buffer(tris_buf, 0, &mesh);

            // 3. the current frame (BG + rain so far) becomes the load
            // contents of the color target
            let mut pixels = Vec::with_capacity(frame.buf.len() * 4);
            for &px in &frame.buf {
                pixels.extend_from_slice(&px.to_le_bytes());
            }
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.color,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some((self.w * 4) as u32),
                    rows_per_image: None,
                },
                extent(self.w, self.h),
            );

            // 4. render pass: draw the logo over the loaded backdrop
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

            // 5. read back and blit into the frame
            enc.copy_texture_to_buffer(
                self.color.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(self.readback_pitch as u32),
                        rows_per_image: None,
                    },
                },
                extent(self.w, self.h),
            );
            let done = self.queue.submit([enc.finish()]);

            self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(done),
                timeout: None,
            })
            .expect("wgpu: poll failed");

            let (tx, rx) = std::sync::mpsc::channel();
            self.readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
            if let Ok(Err(e)) = rx.recv() {
                panic!("wgpu: map_async failed: {e:?}");
            }
            let mapped = self
                .readback
                .get_mapped_range(..)
                .expect("wgpu: get_mapped_range failed");
            let bytes: Vec<u8> = mapped.to_vec();
            drop(mapped);
            self.readback.unmap();

            for y in 0..self.h {
                let row = &bytes[y * self.readback_pitch..y * self.readback_pitch + self.w * 4];
                for x in 0..self.w {
                    let o = x * 4;
                    frame.buf[y * self.w + x] =
                        u32::from_le_bytes([row[o], row[o + 1], row[o + 2], 0]);
                }
            }
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
