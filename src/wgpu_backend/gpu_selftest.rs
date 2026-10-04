//! `akuma-wgpu gpu-selftest`: drive the akuma GPU through the real wgpu API
//! and check pixels against values worked out by hand. This is the standard
//! mode's acceptance test — the part of the backend rio/sugarloaf will use —
//! as `selftest --wgpu` is for the demo's legacy contract.

use super::block_on;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    fn new() -> Gpu {
        let instance = wgpu::Instance::from_custom(super::backend::Instance);
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("request_adapter");
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("akuma gpu-selftest"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::wgt::ExperimentalFeatures::disabled(),
        }))
        .expect("request_device");
        Gpu { device, queue }
    }

    fn texture(&self, w: u32, h: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn buffer(&self, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: data.len().max(4) as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&b, 0, data);
        b
    }

    fn module(&self, src: &str) -> wgpu::ShaderModule {
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(src.into()),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn pipeline(
        &self,
        m: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
        blend: Option<wgpu::BlendState>,
        topology: wgpu::PrimitiveTopology,
        cull: Option<wgpu::Face>,
        bufs: &[Option<wgpu::VertexBufferLayout<'_>>],
    ) -> wgpu::RenderPipeline {
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: m,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: bufs,
            },
            fragment: Some(wgpu::FragmentState {
                module: m,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology, cull_mode: cull, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn empty_layout(&self) -> wgpu::PipelineLayout {
        self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        })
    }

    /// run `f` inside a render pass on `tex` (clearing it first if asked)
    fn pass(
        &self,
        tex: &wgpu::Texture,
        clear: Option<wgpu::Color>,
        f: impl FnOnce(&mut wgpu::RenderPass<'_>),
    ) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: match clear {
                            Some(c) => wgpu::LoadOp::Clear(c),
                            None => wgpu::LoadOp::Load,
                        },
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            f(&mut rp);
        }
        let done = self.queue.submit([enc.finish()]);
        self.device
            .poll(wgpu::PollType::Wait { submission_index: Some(done), timeout: None })
            .expect("poll");
    }

    /// tightly packed bytes of the whole texture
    fn read(&self, tex: &wgpu::Texture, w: u32, h: u32, bpt: u32) -> Vec<u8> {
        let pitch = (w * bpt).div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (pitch * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let done = self.queue.submit([enc.finish()]);
        self.device
            .poll(wgpu::PollType::Wait { submission_index: Some(done), timeout: None })
            .expect("poll");
        let (tx, rx) = std::sync::mpsc::channel();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        rx.recv().unwrap().expect("map");
        let mapped = buf.get_mapped_range(..).expect("range");
        let mut out = Vec::with_capacity((w * h * bpt) as usize);
        for y in 0..h {
            let o = (y * pitch) as usize;
            out.extend_from_slice(&mapped[o..o + (w * bpt) as usize]);
        }
        out
    }
}

impl Gpu {
    fn upload(&self, tex: &wgpu::Texture, w: u32, h: u32, bpt: u32, data: &[u8]) {
        self.queue.write_texture(
            tex.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * bpt), rows_per_image: None },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }

    fn sampler(&self, filter: wgpu::FilterMode, addr: wgpu::AddressMode) -> wgpu::Sampler {
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: addr,
            address_mode_v: addr,
            address_mode_w: addr,
            mag_filter: filter,
            min_filter: filter,
            ..Default::default()
        })
    }

    /// group 0: texture at binding 0, sampler at binding 1, both fragment-visible
    fn tex_layout(&self) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        (bgl, pl)
    }

    /// draw a fullscreen triangle running `fs_body` (a fragment function body
    /// with `tex`, `smp` and `uv` in scope) onto a fresh `fmt` target
    fn draw_textured(
        &self,
        src_tex: &wgpu::Texture,
        smp: &wgpu::Sampler,
        fs_body: &str,
        (w, h): (u32, u32),
        fmt: wgpu::TextureFormat,
        bpt: u32,
    ) -> Vec<u8> {
        let src = format!(
            "{FULL_TRI}
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct V {{ @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> }};
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {{
    var o: V;
    o.p = full(vi);
    // (-1,-3) (-1,1) (3,1) -> uv spanning 0..1 over the visible square
    o.uv = vec2<f32>(o.p.x * 0.5 + 0.5, 0.5 - o.p.y * 0.5);
    return o;
}}
@fragment fn fs(v: V) -> @location(0) vec4<f32> {{
    let uv = v.uv;
    {fs_body}
}}"
        );
        let (bgl, pl) = self.tex_layout();
        let m = self.module(&src);
        let p = self.pipeline(&m, &pl, fmt, None, wgpu::PrimitiveTopology::TriangleList, None, &[]);
        let view = src_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(smp) },
            ],
        });
        let target = self.texture(w, h, fmt);
        self.pass(&target, Some(BLACK), |rp| {
            rp.set_pipeline(&p);
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        });
        self.read(&target, w, h, bpt)
    }
}

type TestResult = Result<(), String>;

fn expect(cond: bool, what: impl FnOnce() -> String) -> TestResult {
    if cond { Ok(()) } else { Err(what()) }
}

const BLACK: wgpu::Color = wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };
const FULL_TRI: &str = r#"
fn full(vi: u32) -> vec4<f32> {
    // (-1,-3) (-1,1) (3,1)
    var x = -1.0; var y = 1.0;
    if (vi == 2u) { x = 3.0; }
    if (vi == 0u) { y = -3.0; }
    return vec4<f32>(x, y, 0.5, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

fn t_fullscreen_solid(g: &Gpu) -> TestResult {
    let src = format!(
        "{FULL_TRI}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{ return full(vi); }}
@fragment fn fs() -> @location(0) vec4<f32> {{ return vec4<f32>(1.0, 0.0, 0.0, 1.0); }}"
    );
    let (w, h) = (37, 23);
    let tex = g.texture(w, h, wgpu::TextureFormat::Rgba8Unorm);
    let m = g.module(&src);
    let p = g.pipeline(
        &m,
        &g.empty_layout(),
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        wgpu::PrimitiveTopology::TriangleList,
        None,
        &[],
    );
    g.pass(&tex, Some(wgpu::Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 }), |rp| {
        rp.set_pipeline(&p);
        rp.draw(0..3, 0..1);
    });
    let px = g.read(&tex, w, h, 4);
    let bad = px.chunks_exact(4).filter(|p| p != &[255, 0, 0, 255]).count();
    expect(bad == 0, || format!("{bad} of {} pixels are not (255,0,0,255)", w * h))
}

/// a quad = two triangles sharing a diagonal, with its edges ON pixel centres:
/// every pixel centre on the shared diagonal must be covered exactly once
/// (additive blend turns a double hit into 2), and the top-left rule decides
/// the boundary pixels
fn t_watertight_quad(g: &Gpu) -> TestResult {
    // target 40x30; quad x: 10.5..30.5, y: 7.5..22.5 (all pixel centres)
    let (w, h) = (40u32, 30u32);
    let nx = |px: f32| px / w as f32 * 2.0 - 1.0;
    let ny = |py: f32| 1.0 - py / h as f32 * 2.0;
    let (x0, x1, y0, y1) = (nx(10.5), nx(30.5), ny(7.5), ny(22.5));
    let src = format!(
        "
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{
    // two triangles: (tl, bl, tr) (tr, bl, br)
    var p = array<vec2<f32>, 6>(
        vec2<f32>({x0}, {y0}), vec2<f32>({x0}, {y1}), vec2<f32>({x1}, {y0}),
        vec2<f32>({x1}, {y0}), vec2<f32>({x0}, {y1}), vec2<f32>({x1}, {y1}));
    return vec4<f32>(p[vi], 0.5, 1.0);
}}
@fragment fn fs() -> @location(0) vec4<f32> {{ return vec4<f32>(1.0 / 255.0, 0.0, 0.0, 1.0); }}"
    );
    let tex = g.texture(w, h, wgpu::TextureFormat::R8Unorm);
    let m = g.module(&src);
    let add = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    let p = g.pipeline(
        &m,
        &g.empty_layout(),
        wgpu::TextureFormat::R8Unorm,
        Some(wgpu::BlendState { color: add, alpha: add }),
        wgpu::PrimitiveTopology::TriangleList,
        None,
        &[],
    );
    g.pass(&tex, Some(BLACK), |rp| {
        rp.set_pipeline(&p);
        rp.draw(0..6, 0..1);
    });
    let px = g.read(&tex, w, h, 1);
    let at = |x: u32, y: u32| px[(y * w + x) as usize];
    let doubles = px.iter().filter(|&&v| v > 1).count();
    let covered = px.iter().filter(|&&v| v == 1).count();
    expect(doubles == 0, || format!("{doubles} pixels were covered twice (cracked/overlapping shared edge)"))?;
    // columns 10..=29 (centre 30.5 is the right edge: excluded), rows 7..=21
    // (centre 7.5 is the top edge: included; 22.5 bottom edge: excluded)
    expect(covered == 20 * 15, || format!("covered {covered}, expected 300"))?;
    expect(at(10, 7) == 1, || "top-left corner pixel (on the top and left edges) must be covered".into())?;
    expect(at(30, 7) == 0 && at(10, 22) == 0, || "right/bottom edge pixels must not be covered".into())?;
    // every pixel centre on the diagonal tr->bl: centres (x+.5, y+.5) with
    // x+.5 = 30.5 - t*20, y+.5 = 7.5 + t*15 -> t multiples of 1/5: x+.5=30.5-4k, y+.5=7.5+3k
    for k in 0..=5u32 {
        let (x, y) = (30 - 4 * k, 7 + 3 * k);
        if x < 30 && y < 22 {
            expect(at(x, y) == 1, || format!("diagonal pixel ({x},{y}) coverage {}", at(x, y)))?;
        }
    }
    Ok(())
}

fn t_linear_varying(g: &Gpu) -> TestResult {
    let (w, h) = (64u32, 4u32);
    let src = "
struct V { @builtin(position) p: vec4<f32>, @location(0) u: f32 };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
    var o: V;
    var c = array<vec2<f32>, 6>(vec2<f32>(-1.0,-1.0), vec2<f32>(-1.0,1.0), vec2<f32>(1.0,-1.0),
                                vec2<f32>(1.0,-1.0), vec2<f32>(-1.0,1.0), vec2<f32>(1.0,1.0));
    let q = c[vi];
    o.p = vec4<f32>(q, 0.5, 1.0);
    o.u = q.x * 0.5 + 0.5;
    return o;
}
@fragment fn fs(v: V) -> @location(0) vec4<f32> { return vec4<f32>(v.u, 0.0, 0.0, 1.0); }";
    let tex = g.texture(w, h, wgpu::TextureFormat::Rgba8Unorm);
    let m = g.module(src);
    let p = g.pipeline(&m, &g.empty_layout(), wgpu::TextureFormat::Rgba8Unorm, None,
        wgpu::PrimitiveTopology::TriangleList, None, &[]);
    g.pass(&tex, Some(BLACK), |rp| {
        rp.set_pipeline(&p);
        rp.draw(0..6, 0..1);
    });
    let px = g.read(&tex, w, h, 4);
    for x in 0..w {
        let want = ((x as f32 + 0.5) / w as f32 * 255.0 + 0.5).floor() as i32;
        let got = px[((2 * w + x) * 4) as usize] as i32;
        expect((got - want).abs() <= 1, || format!("x={x}: red {got}, expected {want}"))?;
    }
    Ok(())
}

/// left vertices at w=1, right vertices at w=2; the colour is 0 at the left and
/// 1 at the right. Perspective-correct interpolation is not linear in screen x.
fn t_perspective_varying(g: &Gpu) -> TestResult {
    let (w, h) = (64u32, 4u32);
    let src = "
struct V { @builtin(position) p: vec4<f32>, @location(0) u: f32 };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
    var o: V;
    var c = array<vec2<f32>, 6>(vec2<f32>(-1.0,-1.0), vec2<f32>(-1.0,1.0), vec2<f32>(1.0,-1.0),
                                vec2<f32>(1.0,-1.0), vec2<f32>(-1.0,1.0), vec2<f32>(1.0,1.0));
    let q = c[vi];
    let right = q.x > 0.0;
    var wv = 1.0;
    if (right) { wv = 2.0; }
    o.p = vec4<f32>(q.x * wv, q.y * wv, 0.5 * wv, wv);
    o.u = select(0.0, 1.0, right);
    return o;
}
@fragment fn fs(v: V) -> @location(0) vec4<f32> { return vec4<f32>(v.u, 0.0, 0.0, 1.0); }";
    let tex = g.texture(w, h, wgpu::TextureFormat::Rgba8Unorm);
    let m = g.module(src);
    let p = g.pipeline(&m, &g.empty_layout(), wgpu::TextureFormat::Rgba8Unorm, None,
        wgpu::PrimitiveTopology::TriangleList, None, &[]);
    g.pass(&tex, Some(BLACK), |rp| {
        rp.set_pipeline(&p);
        rp.draw(0..6, 0..1);
    });
    let px = g.read(&tex, w, h, 4);
    for x in 0..w {
        let s = (x as f64 + 0.5) / w as f64; // screen-space parameter
        let a = (s / 2.0) / ((1.0 - s) + s / 2.0);
        let want = (a * 255.0 + 0.5).floor() as i32;
        let got = px[((2 * w + x) * 4) as usize] as i32;
        expect((got - want).abs() <= 1, || format!("x={x}: red {got}, expected {want} (perspective)"))?;
    }
    // and it must differ visibly from the linear answer in the middle
    let mid = px[((2 * w + w / 2) * 4) as usize] as i32;
    expect(mid < 120, || format!("middle pixel {mid}: looks affine, not perspective-correct (want ~85)"))
}

/// instanced attribute quads: per-vertex Float32x2 corners as a triangle strip,
/// per-instance Uint32x2 cell + Unorm8x4 colour + Sint16x2 offset
fn t_instancing(g: &Gpu) -> TestResult {
    let (w, h) = (48u32, 32u32);
    let src = "
struct In {
    @location(0) corner: vec2<f32>,
    @location(1) cell: vec2<u32>,
    @location(2) color: vec4<f32>,
    @location(3) shift: vec2<i32>,
};
struct V { @builtin(position) p: vec4<f32>, @location(0) @interpolate(flat) c: vec4<f32> };
@vertex fn vs(i: In) -> V {
    var o: V;
    // 16x16-pixel cells on a 48x32 target, shifted by `shift` pixels
    let px = (vec2<f32>(i.cell) + i.corner) * 16.0 + vec2<f32>(i.shift);
    o.p = vec4<f32>(px.x / 24.0 - 1.0, 1.0 - px.y / 16.0, 0.5, 1.0);
    o.c = i.color;
    return o;
}
@fragment fn fs(v: V) -> @location(0) vec4<f32> { return v.c; }";
    let corners: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let vb0 = g.buffer(&corners.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>(), wgpu::BufferUsages::VERTEX);
    // per instance: cell u32x2 (8) + color unorm8x4 (4) + shift sint16x2 (4) = 16 bytes
    let inst: [(u32, u32, [u8; 4], [i16; 2]); 3] = [
        (0, 0, [255, 0, 0, 255], [0, 0]),
        (1, 0, [0, 255, 0, 255], [0, 0]),
        (2, 1, [0, 0, 255, 255], [-4, 3]),
    ];
    let mut ib = Vec::new();
    for (cx, cy, col, sh) in inst {
        ib.extend_from_slice(&cx.to_le_bytes());
        ib.extend_from_slice(&cy.to_le_bytes());
        ib.extend_from_slice(&col);
        ib.extend_from_slice(&sh[0].to_le_bytes());
        ib.extend_from_slice(&sh[1].to_le_bytes());
    }
    let vb1 = g.buffer(&ib, wgpu::BufferUsages::VERTEX);
    let l0 = wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 }],
    };
    let l1 = wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32x2, offset: 0, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Unorm8x4, offset: 8, shader_location: 2 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Sint16x2, offset: 12, shader_location: 3 },
        ],
    };
    let tex = g.texture(w, h, wgpu::TextureFormat::Rgba8Unorm);
    let m = g.module(src);
    let p = g.pipeline(&m, &g.empty_layout(), wgpu::TextureFormat::Rgba8Unorm, None,
        wgpu::PrimitiveTopology::TriangleStrip, None, &[Some(l0), Some(l1)]);
    g.pass(&tex, Some(BLACK), |rp| {
        rp.set_pipeline(&p);
        rp.set_vertex_buffer(0, vb0.slice(..));
        rp.set_vertex_buffer(1, vb1.slice(..));
        rp.draw(0..4, 0..3);
    });
    let px = g.read(&tex, w, h, 4);
    let at = |x: u32, y: u32| {
        let o = ((y * w + x) * 4) as usize;
        [px[o], px[o + 1], px[o + 2], px[o + 3]]
    };
    expect(at(8, 8) == [255, 0, 0, 255], || format!("cell (0,0) = {:?}", at(8, 8)))?;
    expect(at(24, 8) == [0, 255, 0, 255], || format!("cell (1,0) = {:?}", at(24, 8)))?;
    // third quad: cell (2,1) = x 32..48, y 16..32, shifted (-4,+3) -> x 28..44, y 19..35 (clipped to 32)
    expect(at(36, 25) == [0, 0, 255, 255], || format!("shifted cell (2,1) = {:?}", at(36, 25)))?;
    expect(at(29, 20) == [0, 0, 255, 255], || format!("shifted cell inner corner = {:?}", at(29, 20)))?;
    expect(at(27, 20) == [0, 0, 0, 0], || format!("left of shifted cell = {:?}", at(27, 20)))?;
    expect(at(40, 8) == [0, 0, 0, 0], || format!("empty cell = {:?}", at(40, 8)))
}

fn t_cull_back(g: &Gpu) -> TestResult {
    // CCW triangle on the left, CW on the right; cull_mode Back drops the CW one
    let (w, h) = (40u32, 20u32);
    let src = "
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 6>(
        vec2<f32>(-0.9, -0.9), vec2<f32>(-0.1, -0.9), vec2<f32>(-0.5, 0.9),   // CCW (y up)
        vec2<f32>(0.1, -0.9), vec2<f32>(0.5, 0.9), vec2<f32>(0.9, -0.9));      // CW
    return vec4<f32>(p[vi], 0.5, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }";
    let tex = g.texture(w, h, wgpu::TextureFormat::Rgba8Unorm);
    let m = g.module(src);
    let p = g.pipeline(&m, &g.empty_layout(), wgpu::TextureFormat::Rgba8Unorm, None,
        wgpu::PrimitiveTopology::TriangleList, Some(wgpu::Face::Back), &[]);
    g.pass(&tex, Some(BLACK), |rp| {
        rp.set_pipeline(&p);
        rp.draw(0..6, 0..1);
    });
    let px = g.read(&tex, w, h, 4);
    let lit = |x0: u32, x1: u32| {
        (0..h).flat_map(|y| (x0..x1).map(move |x| (x, y)))
            .filter(|&(x, y)| px[((y * w + x) * 4) as usize] != 0).count()
    };
    let (left, right) = (lit(0, 20), lit(20, 40));
    expect(left > 50, || format!("front-facing (CCW) triangle not drawn: {left} px"))?;
    expect(right == 0, || format!("back-facing (CW) triangle drawn despite cull: {right} px"))
}

fn t_premultiplied_blend(g: &Gpu) -> TestResult {
    let (w, h) = (8u32, 8u32);
    let src = format!(
        "{FULL_TRI}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{ return full(vi); }}
@fragment fn fs() -> @location(0) vec4<f32> {{ return vec4<f32>(0.5, 0.0, 0.0, 0.5); }}"
    );
    let tex = g.texture(w, h, wgpu::TextureFormat::Rgba8Unorm);
    let m = g.module(&src);
    let c = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    let p = g.pipeline(&m, &g.empty_layout(), wgpu::TextureFormat::Rgba8Unorm,
        Some(wgpu::BlendState { color: c, alpha: c }), wgpu::PrimitiveTopology::TriangleList, None, &[]);
    g.pass(&tex, Some(wgpu::Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 }), |rp| {
        rp.set_pipeline(&p);
        rp.draw(0..3, 0..1);
    });
    let px = g.read(&tex, w, h, 4);
    // clear 0.5 -> 128 stored (127.5+.5 floor = 128 -> 0.50196); blend: r = .5 + .50196*.5 = .75098 -> 191.5 -> 192? compute exactly
    let dst = 128.0f32 / 255.0;
    let want_r = ((0.5 + dst * 0.5) * 255.0 + 0.5).floor() as i32;
    let want_g = ((dst * 0.5) * 255.0 + 0.5).floor() as i32;
    let want_a = ((0.5f32 + 1.0 * 0.5) * 255.0 + 0.5).floor() as i32;
    let got = [px[0] as i32, px[1] as i32, px[2] as i32, px[3] as i32];
    expect(got == [want_r, want_g, want_g, want_a], || {
        format!("blend result {got:?}, expected {:?}", [want_r, want_g, want_g, want_a])
    })
}

fn t_scissor_viewport(g: &Gpu) -> TestResult {
    let (w, h) = (64u32, 48u32);
    let src = format!(
        "{FULL_TRI}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{ return full(vi); }}
@fragment fn fs() -> @location(0) vec4<f32> {{ return vec4<f32>(1.0, 1.0, 1.0, 1.0); }}"
    );
    let tex = g.texture(w, h, wgpu::TextureFormat::Rgba8Unorm);
    let m = g.module(&src);
    let p = g.pipeline(&m, &g.empty_layout(), wgpu::TextureFormat::Rgba8Unorm, None,
        wgpu::PrimitiveTopology::TriangleList, None, &[]);
    g.pass(&tex, Some(BLACK), |rp| {
        rp.set_pipeline(&p);
        rp.set_scissor_rect(10, 5, 20, 10);
        rp.draw(0..3, 0..1);
    });
    let px = g.read(&tex, w, h, 4);
    let lit: Vec<(u32, u32)> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| px[((y * w + x) * 4) as usize] != 0).collect();
    expect(lit.len() == 200, || format!("scissor: {} pixels lit, expected 200", lit.len()))?;
    expect(lit.iter().all(|&(x, y)| (10..30).contains(&x) && (5..15).contains(&y)), || "scissor leaked".into())?;
    // viewport: only the 32x24 top-left quarter may be written
    g.pass(&tex, Some(BLACK), |rp| {
        rp.set_pipeline(&p);
        rp.set_viewport(0.0, 0.0, 32.0, 24.0, 0.0, 1.0);
        rp.draw(0..3, 0..1);
    });
    let px = g.read(&tex, w, h, 4);
    let n = (0..h).flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| px[((y * w + x) * 4) as usize] != 0).count();
    expect(n == 32 * 24, || format!("viewport: {n} pixels lit, expected {}", 32 * 24))
}

fn t_formats(g: &Gpu) -> TestResult {
    let (w, h) = (4u32, 4u32);
    let src = format!(
        "{FULL_TRI}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{ return full(vi); }}
@fragment fn fs() -> @location(0) vec4<f32> {{ return vec4<f32>(1.0, 0.5, 0.25, 1.0); }}"
    );
    let m = g.module(&src);
    let check = |fmt: wgpu::TextureFormat, bpt: u32, want: &[u8]| -> TestResult {
        let tex = g.texture(w, h, fmt);
        let p = g.pipeline(&m, &g.empty_layout(), fmt, None, wgpu::PrimitiveTopology::TriangleList, None, &[]);
        g.pass(&tex, Some(BLACK), |rp| {
            rp.set_pipeline(&p);
            rp.draw(0..3, 0..1);
        });
        let px = g.read(&tex, w, h, bpt);
        expect(&px[..bpt as usize] == want, || format!("{fmt:?}: {:?}, expected {want:?}", &px[..bpt as usize]))
    };
    check(wgpu::TextureFormat::Rgba8Unorm, 4, &[255, 128, 64, 255])?;
    check(wgpu::TextureFormat::Bgra8Unorm, 4, &[64, 128, 255, 255])?;
    check(wgpu::TextureFormat::R8Unorm, 1, &[255])?;
    check(wgpu::TextureFormat::Rg8Unorm, 2, &[255, 128])?;
    // sRGB: linear 0.5 encodes to ~188, linear 0.25 to ~137
    check(wgpu::TextureFormat::Rgba8UnormSrgb, 4, &[255, 188, 137, 255])
}

fn rgba(px: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
    let o = ((y * w + x) * 4) as usize;
    [px[o], px[o + 1], px[o + 2], px[o + 3]]
}

const QUAD4: [u8; 16] = [
    255, 0, 0, 255, 0, 255, 0, 255, // red, green
    0, 0, 255, 255, 255, 255, 255, 255, // blue, white
];

fn t_texture_load(g: &Gpu) -> TestResult {
    let tex = g.texture(2, 2, wgpu::TextureFormat::Rgba8Unorm);
    g.upload(&tex, 2, 2, 4, &QUAD4);
    let smp = g.sampler(wgpu::FilterMode::Nearest, wgpu::AddressMode::ClampToEdge);
    let px = g.draw_textured(
        &tex, &smp,
        "return textureLoad(tex, vec2<i32>(floor(uv * 2.0)), 0);",
        (8, 8), wgpu::TextureFormat::Rgba8Unorm, 4,
    );
    expect(rgba(&px, 8, 1, 1) == [255, 0, 0, 255], || format!("tl {:?}", rgba(&px, 8, 1, 1)))?;
    expect(rgba(&px, 8, 6, 1) == [0, 255, 0, 255], || format!("tr {:?}", rgba(&px, 8, 6, 1)))?;
    expect(rgba(&px, 8, 1, 6) == [0, 0, 255, 255], || format!("bl {:?}", rgba(&px, 8, 1, 6)))?;
    expect(rgba(&px, 8, 6, 6) == [255, 255, 255, 255], || format!("br {:?}", rgba(&px, 8, 6, 6)))
}

fn t_texture_sample_nearest(g: &Gpu) -> TestResult {
    let tex = g.texture(2, 2, wgpu::TextureFormat::Rgba8Unorm);
    g.upload(&tex, 2, 2, 4, &QUAD4);
    let smp = g.sampler(wgpu::FilterMode::Nearest, wgpu::AddressMode::ClampToEdge);
    let px = g.draw_textured(
        &tex, &smp,
        "return textureSample(tex, smp, uv);",
        (8, 8), wgpu::TextureFormat::Rgba8Unorm, 4,
    );
    expect(rgba(&px, 8, 1, 1) == [255, 0, 0, 255], || format!("tl {:?}", rgba(&px, 8, 1, 1)))?;
    expect(rgba(&px, 8, 6, 6) == [255, 255, 255, 255], || format!("br {:?}", rgba(&px, 8, 6, 6)))?;
    // clamp-to-edge: sampling outside 0..1 repeats the edge texel
    let px = g.draw_textured(
        &tex, &smp,
        "return textureSample(tex, smp, uv * 3.0 - vec2<f32>(1.0, 1.0));",
        (9, 9), wgpu::TextureFormat::Rgba8Unorm, 4,
    );
    expect(rgba(&px, 9, 0, 0) == [255, 0, 0, 255], || format!("clamped corner {:?}", rgba(&px, 9, 0, 0)))?;
    expect(rgba(&px, 9, 8, 8) == [255, 255, 255, 255], || format!("clamped corner {:?}", rgba(&px, 9, 8, 8)))
}

/// a 2x1 R8 texture (0, 255): bilinear between the texel centres at u=0.25
/// and u=0.75 ramps linearly and clamps outside them
fn t_texture_bilinear(g: &Gpu) -> TestResult {
    let tex = g.texture(2, 1, wgpu::TextureFormat::R8Unorm);
    g.upload(&tex, 2, 1, 1, &[0, 255]);
    let smp = g.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge);
    let (w, h) = (16u32, 2u32);
    let px = g.draw_textured(&tex, &smp, "return vec4<f32>(textureSample(tex, smp, uv).r, 0.0, 0.0, 1.0);",
        (w, h), wgpu::TextureFormat::Rgba8Unorm, 4);
    for x in 0..w {
        let u = (x as f32 + 0.5) / w as f32;
        let want = (((u - 0.25) / 0.5).clamp(0.0, 1.0) * 255.0 + 0.5).floor() as i32;
        let got = rgba(&px, w, x, 0)[0] as i32;
        expect((got - want).abs() <= 1, || format!("x={x} u={u}: {got}, expected {want}"))?;
    }
    Ok(())
}

fn t_texture_repeat_and_dims(g: &Gpu) -> TestResult {
    let tex = g.texture(2, 1, wgpu::TextureFormat::R8Unorm);
    g.upload(&tex, 2, 1, 1, &[10, 200]);
    let smp = g.sampler(wgpu::FilterMode::Nearest, wgpu::AddressMode::Repeat);
    // u*2 over 0..1 -> two repeats of (10, 200): texel columns 10 200 10 200
    let (w, h) = (8u32, 2u32);
    let px = g.draw_textured(&tex, &smp, "return vec4<f32>(textureSample(tex, smp, vec2<f32>(uv.x * 2.0, 0.5)).r, 0.0, 0.0, 1.0);",
        (w, h), wgpu::TextureFormat::Rgba8Unorm, 4);
    let want = [10, 10, 200, 200, 10, 10, 200, 200];
    for x in 0..w {
        let got = rgba(&px, w, x, 0)[0];
        expect(got == want[x as usize], || format!("repeat x={x}: {got}, expected {}", want[x as usize]))?;
    }
    // textureDimensions
    let px = g.draw_textured(&tex, &smp, "let d = textureDimensions(tex); return vec4<f32>(f32(d.x) / 255.0, f32(d.y) / 255.0, 0.0, 1.0);",
        (4, 4), wgpu::TextureFormat::Rgba8Unorm, 4);
    expect(rgba(&px, 4, 1, 1)[..2] == [2, 1], || format!("dims {:?}", &rgba(&px, 4, 1, 1)[..2]))
}

fn t_texture_srgb(g: &Gpu) -> TestResult {
    let tex = g.texture(1, 1, wgpu::TextureFormat::Rgba8UnormSrgb);
    g.upload(&tex, 1, 1, 4, &[128, 128, 128, 255]);
    let smp = g.sampler(wgpu::FilterMode::Nearest, wgpu::AddressMode::ClampToEdge);
    let px = g.draw_textured(&tex, &smp, "return textureSample(tex, smp, uv);", (2, 2), wgpu::TextureFormat::Rgba8Unorm, 4);
    // sRGB 128/255 = 0.50196 -> linear 0.2158 -> 55
    expect(rgba(&px, 2, 0, 0) == [55, 55, 55, 255], || format!("srgb decode {:?}", rgba(&px, 2, 0, 0)))
}

fn t_texture_copy(g: &Gpu) -> TestResult {
    // a 4x4 R8 texture filled with its index; copy the 2x2 block at (1,1) to a
    // second texture at (2,0), and a row buffer->texture; read both back
    let a = g.texture(4, 4, wgpu::TextureFormat::R8Unorm);
    let data: Vec<u8> = (0..16).collect();
    g.upload(&a, 4, 4, 1, &data);
    let b = g.texture(4, 4, wgpu::TextureFormat::R8Unorm);
    let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo { texture: &a, mip_level: 0, origin: wgpu::Origin3d { x: 1, y: 1, z: 0 }, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyTextureInfo { texture: &b, mip_level: 0, origin: wgpu::Origin3d { x: 2, y: 0, z: 0 }, aspect: wgpu::TextureAspect::All },
        wgpu::Extent3d { width: 2, height: 2, depth_or_array_layers: 1 },
    );
    let done = g.queue.submit([enc.finish()]);
    g.device.poll(wgpu::PollType::Wait { submission_index: Some(done), timeout: None }).expect("poll");
    let got = g.read(&b, 4, 4, 1);
    // block from a: (1,1)=5 (2,1)=6 / (1,2)=9 (2,2)=10 -> b at (2,0),(3,0) / (2,1),(3,1)
    expect(got[2] == 5 && got[3] == 6 && got[4 + 2] == 9 && got[4 + 3] == 10, || format!("copy: {got:?}"))?;
    expect(got[0] == 0 && got[1] == 0 && got[8] == 0, || "copy touched outside the block".into())
}

pub fn run() -> i32 {
    let g = Gpu::new();
    let tests: &[(&str, fn(&Gpu) -> TestResult)] = &[
        ("fullscreen triangle, clear + draw", t_fullscreen_solid),
        ("watertight shared edge + top-left rule", t_watertight_quad),
        ("linear varying across a quad", t_linear_varying),
        ("perspective-correct varying", t_perspective_varying),
        ("vertex buffers, instancing, strips, attribute formats", t_instancing),
        ("back-face culling by winding", t_cull_back),
        ("premultiplied-alpha blending", t_premultiplied_blend),
        ("scissor rect + viewport", t_scissor_viewport),
        ("target formats (rgba/bgra/r8/rg8/srgb)", t_formats),
        ("textureLoad", t_texture_load),
        ("textureSample, nearest, clamp-to-edge", t_texture_sample_nearest),
        ("textureSample, bilinear", t_texture_bilinear),
        ("textureSample repeat + textureDimensions", t_texture_repeat_and_dims),
        ("sRGB texture decode on sample", t_texture_srgb),
        ("texture-to-texture copy with origins", t_texture_copy),
    ];
    let mut failed = 0;
    for (name, f) in tests {
        match f(&g) {
            Ok(()) => println!("ok    {name}"),
            Err(e) => {
                failed += 1;
                println!("FAIL  {name}: {e}");
            }
        }
    }
    if failed == 0 {
        println!("GPU-SELFTEST OK ({} tests)", tests.len());
        0
    } else {
        println!("GPU-SELFTEST FAILED: {failed} of {}", tests.len());
        1
    }
}
