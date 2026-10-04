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


/// A fragment shader built to exercise memoization and specialization: its
/// colour is a pure function of a loaded cell word (pow-heavy, with a branch
/// on the word and one on a uniform), evaluated over cells of widths that
/// make 4-pixel batches uniform (16), straddle cell edges (3, 5) and never
/// repeat (1). Results must equal a CPU evaluation of the same f32 maths,
/// with and without forced specialization, and again after the uniform that
/// the specialization folded changes.
fn t_memo_spec(g: &Gpu) -> TestResult {
    use std::sync::atomic::Ordering::Relaxed;
    let (w, h) = (64u32, 16u32);
    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    let vis = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
    let bgl = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: vis,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: vis,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });
    let pl = g.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let to_u8 = |c: f32| (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    for cw in [1u32, 3, 5, 16] {
        let src = format!(
            "{FULL_TRI}
struct U {{ mode: u32, cols: u32, a: u32, b: u32 }};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> cells: array<u32>;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{ return full(vi); }}
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{
    let cx = min(u32(p.x) / {cw}u, u.cols - 1u);
    let cy = u32(p.y) / 4u;
    let w = cells[cy * u.cols + cx];
    var r = f32(w & 255u) / 255.0;
    var g = f32((w >> 8u) & 255u) / 255.0;
    var b = f32((w >> 16u) & 255u) / 255.0;
    if (u.mode == 0u) {{
        r = pow(r, 2.2); g = pow(g, 0.45); b = pow(b + 0.1, 1.7);
    }} else {{
        r = pow(r, 0.5) * 0.9; g = pow(g + 0.2, 3.0); b = pow(b, 1.25);
    }}
    if ((w & 1u) != 0u) {{ r = r * 0.5; }}
    return vec4<f32>(r, g, b, 1.0);
}}"
        );
        let m = g.module(&src);
        let p = g.pipeline(&m, &pl, fmt, None, wgpu::PrimitiveTopology::TriangleList, None, &[]);
        let cols = w.div_ceil(cw);
        // runs of equal words, so batches hit, plus enough variety to miss
        let mut cells = Vec::new();
        let mut x = 0x2545_F491u32;
        for r in 0..4u32 {
            for c in 0..cols {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let v = if (c + r) % 4 == 0 { x } else { 0x00_80_40_21 };
                cells.extend_from_slice(&(v & 0x00ff_ffff).to_le_bytes());
            }
        }
        let cbuf = g.buffer(&cells, wgpu::BufferUsages::STORAGE);
        for spec in [false, true] {
            crate::wgpu_backend::exec::FORCE_SPEC.store(spec, Relaxed);
            for mode in [0u32, 1, 0] {
                let mut u = Vec::new();
                for v in [mode, cols, 0, 0] {
                    u.extend_from_slice(&v.to_le_bytes());
                }
                let ubuf = g.buffer(&u, wgpu::BufferUsages::UNIFORM);
                let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: cbuf.as_entire_binding() },
                    ],
                });
                let target = g.texture(w, h, fmt);
                g.pass(&target, Some(BLACK), |rp| {
                    rp.set_pipeline(&p);
                    rp.set_bind_group(0, &bg, &[]);
                    rp.draw(0..3, 0..1);
                });
                let px = g.read(&target, w, h, 4);
                for y in 0..h {
                    for xx in 0..w {
                        let cx = (xx / cw).min(cols - 1);
                        let o = ((y / 4) * cols + cx) as usize * 4;
                        let word = u32::from_le_bytes([cells[o], cells[o + 1], cells[o + 2], cells[o + 3]]);
                        let mut r = (word & 255) as f32 / 255.0;
                        let mut gg = ((word >> 8) & 255) as f32 / 255.0;
                        let mut b = ((word >> 16) & 255) as f32 / 255.0;
                        if mode == 0 {
                            r = r.powf(2.2);
                            gg = gg.powf(0.45);
                            b = (b + 0.1).powf(1.7);
                        } else {
                            r = r.powf(0.5) * 0.9;
                            gg = (gg + 0.2).powf(3.0);
                            b = b.powf(1.25);
                        }
                        if word & 1 != 0 {
                            r *= 0.5;
                        }
                        let want = [to_u8(r), to_u8(gg), to_u8(b), 255];
                        let at = ((y * w + xx) * 4) as usize;
                        let got = [px[at], px[at + 1], px[at + 2], px[at + 3]];
                        if got != want {
                            crate::wgpu_backend::exec::FORCE_SPEC.store(false, Relaxed);
                            return Err(format!(
                                "cell width {cw}, spec {spec}, mode {mode}: pixel ({xx},{y}) = {got:?}, expected {want:?}"
                            ));
                        }
                    }
                }
            }
        }
    }
    crate::wgpu_backend::exec::FORCE_SPEC.store(false, Relaxed);
    Ok(())
}


/// "over" blending of a position-dependent source (alpha 0..1 in steps) onto
/// random destination texels, for both channel orders and both source factors,
/// against the scalar definition (decode, f32 blend, encode).
fn t_over_blend_vector(g: &Gpu) -> TestResult {
    use crate::wgpu_backend::format;
    let (w, h) = (67u32, 13u32);
    let mut x = 0x1234_5678u32;
    let mut dst = vec![0u8; (w * h * 4) as usize];
    for b in dst.iter_mut() {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *b = (x >> 24) as u8;
    }
    for fmt in [wgpu::TextureFormat::Bgra8Unorm, wgpu::TextureFormat::Rgba8Unorm] {
        for straight in [false, true] {
            let mul = if straight { "1.0" } else { "a" };
            let src = format!(
                "{FULL_TRI}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{ return full(vi); }}
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{
    let ix = u32(p.x); let iy = u32(p.y);
    let a = f32((ix * 7u + iy * 13u) % 31u) / 30.0;
    let r = f32((ix * 5u) % 17u) / 16.0;
    let g = f32((iy * 3u + ix) % 19u) / 18.0;
    let b = f32((ix + iy * 11u) % 23u) / 22.0;
    return vec4<f32>(r * {mul}, g * {mul}, b * {mul}, a);
}}"
            );
            let c = wgpu::BlendComponent {
                src_factor: if straight { wgpu::BlendFactor::SrcAlpha } else { wgpu::BlendFactor::One },
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            };
            let m = g.module(&src);
            let p = g.pipeline(&m, &g.empty_layout(), fmt, Some(wgpu::BlendState { color: c, alpha: c }), wgpu::PrimitiveTopology::TriangleList, None, &[]);
            let tex = g.texture(w, h, fmt);
            g.upload(&tex, w, h, 4, &dst);
            g.pass(&tex, None, |rp| {
                rp.set_pipeline(&p);
                rp.draw(0..3, 0..1);
            });
            let got = g.read(&tex, w, h, 4);
            for iy in 0..h {
                for ix in 0..w {
                    let a = ((ix * 7 + iy * 13) % 31) as f32 / 30.0;
                    let r = ((ix * 5) % 17) as f32 / 16.0;
                    let gg = ((iy * 3 + ix) % 19) as f32 / 18.0;
                    let b = ((ix + iy * 11) % 23) as f32 / 22.0;
                    let m = if straight { 1.0 } else { a };
                    let s = [r * m, gg * m, b * m, a];
                    let at = ((iy * w + ix) * 4) as usize;
                    let d = format::decode(fmt, &dst[at..at + 4]);
                    let df = 1.0 - a;
                    let sf = if straight { a } else { 1.0 };
                    let o = [s[0] * sf + d[0] * df, s[1] * sf + d[1] * df, s[2] * sf + d[2] * df, s[3] * sf + d[3] * df];
                    let mut want = [0u8; 4];
                    format::encode(fmt, o, &mut want);
                    if got[at..at + 4] != want {
                        return Err(format!(
                            "{fmt:?} straight={straight}: pixel ({ix},{iy}) = {:?}, expected {want:?}",
                            &got[at..at + 4]
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}


/// Run detection (`runs.rs`): a cell-grid shader whose position dependence is
/// pure quantization must render identically with run replication on and off,
/// across cell sizes and offsets that are not multiples of anything, for a
/// fullscreen triangle and for a slanted two-triangle quad; a shader that
/// uses the position directly must be unaffected.
fn t_runs(g: &Gpu) -> TestResult {
    use std::sync::atomic::Ordering::Relaxed;
    let (w, h) = (211u32, 131u32);
    let fmt = wgpu::TextureFormat::Bgra8Unorm;
    let vis = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
    let bgl = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: vis,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: vis,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });
    let pl = g.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let grid_fs = "
struct U { pad: vec4<f32>, dims: vec4<u32> };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> cells: array<u32>;
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let f = (p.xy - u.pad.xy) / u.pad.zw;
    var gp = vec2<i32>(floor(f));
    let cols = i32(u.dims.x);
    let rows = i32(u.dims.y);
    if (gp.x < 0) { gp.x = 0; } else if (gp.x > cols - 1) { gp.x = cols - 1; }
    if (gp.y < 0) { gp.y = 0; } else if (gp.y > rows - 1) { gp.y = rows - 1; }
    let wd = cells[u32(gp.y) * u.dims.x + u32(gp.x)];
    let r = pow(f32(wd & 255u) / 255.0, 2.2);
    let gg = pow(f32((wd >> 8u) & 255u) / 255.0, 0.7);
    let b = pow(f32((wd >> 16u) & 255u) / 255.0, 1.4);
    return vec4<f32>(r, gg, b, 1.0);
}";
    // the position leaks into the colour: no run may be assumed
    let leaky_fs = "
struct U { pad: vec4<f32>, dims: vec4<u32> };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> cells: array<u32>;
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let f = (p.xy - u.pad.xy) / u.pad.zw;
    let gp = vec2<i32>(floor(f));
    let wd = cells[u32(clamp(gp.y, 0, i32(u.dims.y) - 1)) * u.dims.x + u32(clamp(gp.x, 0, i32(u.dims.x) - 1))];
    return vec4<f32>(f32(wd & 255u) / 255.0, fract(p.x * 0.05), fract(p.y * 0.03), 1.0);
}";
    let vs_tri = "@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> { return full(vi); }";
    // a quad inset from the edges; its two triangles meet on a diagonal
    let vs_quad = "@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = select(-0.93, 0.88, (vi & 1u) == 1u);
    let y = select(-0.81, 0.91, (vi & 2u) == 2u);
    return vec4<f32>(x, y, 0.5, 1.0);
}";
    let geoms: [[f32; 4]; 5] = [
        [0.0, 0.0, 16.0, 32.0],
        [3.5, 7.25, 9.3, 13.7],
        [-5.5, 2.0, 5.0, 5.0],
        [0.0, 0.0, 3.0, 2.5],
        [10.0, 4.0, 40.0, 1.0],
    ];
    let run = |vs: &str, fs: &str, topo: wgpu::PrimitiveTopology, verts: u32, geo: [f32; 4], force_spec: bool| -> Vec<u8> {
        let (cols, rows) = (((w as f32 - geo[0]) / geo[2]).ceil().max(1.0) as u32 + 1, ((h as f32 - geo[1]) / geo[3]).ceil().max(1.0) as u32 + 1);
        let mut cells = Vec::new();
        let mut x = 0xCAFE_F00Du32 ^ cols;
        for i in 0..cols * rows {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = if i % 5 == 0 { 0x00_40_80_20 } else { x & 0x00ff_ffff };
            cells.extend_from_slice(&v.to_le_bytes());
        }
        let mut u = Vec::new();
        for v in geo {
            u.extend_from_slice(&v.to_le_bytes());
        }
        for v in [cols, rows, 0, 0] {
            u.extend_from_slice(&v.to_le_bytes());
        }
        let src = format!("{FULL_TRI}\n{vs}\n{fs}");
        let m = g.module(&src);
        let p = g.pipeline(&m, &pl, fmt, None, topo, None, &[]);
        let ubuf = g.buffer(&u, wgpu::BufferUsages::UNIFORM);
        let cbuf = g.buffer(&cells, wgpu::BufferUsages::STORAGE);
        let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cbuf.as_entire_binding() },
            ],
        });
        crate::wgpu_backend::exec::FORCE_SPEC.store(force_spec, Relaxed);
        let target = g.texture(w, h, fmt);
        g.pass(&target, Some(wgpu::Color { r: 0.25, g: 0.5, b: 0.75, a: 1.0 }), |rp| {
            rp.set_pipeline(&p);
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..verts, 0..1);
        });
        crate::wgpu_backend::exec::FORCE_SPEC.store(false, Relaxed);
        g.read(&target, w, h, 4)
    };
    for (name, vs, verts, topo) in [
        ("fullscreen triangle", vs_tri, 3u32, wgpu::PrimitiveTopology::TriangleList),
        ("slanted quad", vs_quad, 4u32, wgpu::PrimitiveTopology::TriangleStrip),
    ] {
        for fs in [grid_fs, leaky_fs] {
            for &geo in &geoms {
                for spec in [false, true] {
                    crate::wgpu_backend::runs::ENABLED.store(false, Relaxed);
                    let want = run(vs, fs, topo, verts, geo, spec);
                    crate::wgpu_backend::runs::ENABLED.store(true, Relaxed);
                    let before = crate::wgpu_backend::raster::RUN_PIXELS.load(Relaxed);
                    let got = run(vs, fs, topo, verts, geo, spec);
                    let replicated = crate::wgpu_backend::raster::RUN_PIXELS.load(Relaxed) - before;
                    // x86-64 only (the wide JIT): the grid shader must really
                    // take the run path, the leaky one must not
                    if cfg!(all(target_arch = "x86_64", target_os = "linux")) && geo[2] >= 5.0 {
                        let leaky = fs == leaky_fs;
                        if (replicated == 0) != leaky {
                            return Err(format!(
                                "{name}, {} shader, cells {geo:?}: {replicated} pixels replicated",
                                if leaky { "leaky" } else { "grid" }
                            ));
                        }
                    }
                    if got != want {
                        let i = got.iter().zip(&want).position(|(a, b)| a != b).unwrap() / 4;
                        return Err(format!(
                            "{name}, {} shader, cells {geo:?}, spec {spec}: first difference at pixel ({}, {}): runs {:?}, no runs {:?}",
                            if fs == grid_fs { "grid" } else { "leaky" },
                            i as u32 % w,
                            i as u32 / w,
                            &got[i * 4..i * 4 + 4],
                            &want[i * 4..i * 4 + 4]
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}


/// Random fragment shaders — a cell lookup, position-quantized coordinates,
/// random float maths with branches and short loops on top — rendered by the
/// default executors (JIT, memoization, run replication) and with forced
/// specialization, against the interpreter. Some shaders leak the raw pixel
/// position into the colour (run replication must back off), some depend only
/// on the quantized position.
fn t_fuzz_fragment(g: &Gpu) -> TestResult {
    use crate::wgpu_backend::exec_selftest::Gen;
    use std::sync::atomic::Ordering::Relaxed;
    let n: u32 = std::env::var("AKUMA_FUZZ").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let (w, h) = (101u32, 67u32);
    let fmt = wgpu::TextureFormat::Bgra8Unorm;
    let vis = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
    let bgl = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: vis,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: vis,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });
    let pl = g.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let mut cells = Vec::new();
    let mut x = 0xBADC_0FFEu32;
    for _ in 0..40 * 40 {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        cells.extend_from_slice(&x.to_le_bytes());
    }
    let cbuf = g.buffer(&cells, wgpu::BufferUsages::STORAGE);
    for k in 0..n {
        let mut rng = Gen { x: (0xF00D_u64 + k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 };
        let vars = rng.vars();
        for leaky in [false, true] {
            let cw = [3.0f32, 5.5, 8.0, 13.0][(k % 4) as usize];
            let ch = [4.0f32, 2.5, 9.0, 16.0][((k / 4) % 4) as usize];
            let src = format!(
                "{FULL_TRI}
@group(0) @binding(0) var<uniform> u: vec4<f32>;
@group(0) @binding(1) var<storage, read> cells: array<u32>;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{ return full(vi); }}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {{
    let gp = vec2<i32>(floor((pos.xy - vec2<f32>(1.5, 0.25)) / vec2<f32>({cw:?}, {ch:?})));
    let cx = u32(max(0, min(gp.x, 39)));
    let cy = u32(max(0, min(gp.y, 39)));
    let wd = cells[cy * 40u + cx];
    let p = vec4<f32>(f32(wd & 255u) / 255.0, f32((wd >> 8u) & 255u) / 255.0, f32((wd >> 16u) & 255u) / 255.0, f32(gp.x));
    let q = vec4<f32>(f32(gp.y), f32(wd >> 24u), {leak}, 1.0);
    let vi = wd & 1023u;
{vars}
    return vec4<f32>(clamp(v0 * 0.1, 0.0, 1.0), clamp(v1 * 0.1, 0.0, 1.0), clamp(v2 * 0.1, 0.0, 1.0), 1.0);
}}",
                leak = if leaky { "pos.x * 0.07" } else { "q_const()" },
            )
            .replace("q_const()", "0.5");
            if std::env::var_os("AKUMA_FUZZ_VERBOSE").is_some() {
                eprintln!("--- fuzz #{k} leaky={leaky}\n{src}");
            }
            let make = || {
                let m = g.module(&src);
                g.pipeline(&m, &pl, fmt, None, wgpu::PrimitiveTopology::TriangleList, None, &[])
            };
            let draw = |p: &wgpu::RenderPipeline, uval: [f32; 4]| -> Vec<u8> {
                let mut u = Vec::new();
                for v in uval {
                    u.extend_from_slice(&v.to_le_bytes());
                }
                let ubuf = g.buffer(&u, wgpu::BufferUsages::UNIFORM);
                let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: cbuf.as_entire_binding() },
                    ],
                });
                let target = g.texture(w, h, fmt);
                g.pass(&target, Some(BLACK), |rp| {
                    rp.set_pipeline(p);
                    rp.set_bind_group(0, &bg, &[]);
                    rp.draw(0..3, 0..1);
                });
                g.read(&target, w, h, 4)
            };
            // the reference: the interpreter
            // SAFETY: tests run on one thread; nothing else reads the environment meanwhile
            unsafe { std::env::set_var("AKUMA_EXEC", "interp") };
            let pi = make();
            unsafe { std::env::remove_var("AKUMA_EXEC") };
            let pd = make();
            for uval in [[1.0f32, 2.0, 3.0, 0.5], [0.25, -1.0, 7.5, 2.0]] {
                let want = draw(&pi, uval);
                for (what, spec, runs) in [("default", false, true), ("no runs", false, false), ("forced spec", true, true)] {
                    crate::wgpu_backend::exec::FORCE_SPEC.store(spec, Relaxed);
                    crate::wgpu_backend::runs::ENABLED.store(runs, Relaxed);
                    let got = draw(&pd, uval);
                    crate::wgpu_backend::exec::FORCE_SPEC.store(false, Relaxed);
                    crate::wgpu_backend::runs::ENABLED.store(true, Relaxed);
                    if got != want {
                        let i = got.iter().zip(&want).position(|(a, b)| a != b).unwrap() / 4;
                        return Err(format!(
                            "fuzz #{k} leaky={leaky} u={uval:?} ({what}): pixel ({}, {}) = {:?}, interpreter {:?}\n{vars}",
                            i as u32 % w,
                            i as u32 / w,
                            &got[i * 4..i * 4 + 4],
                            &want[i * 4..i * 4 + 4]
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}


/// sugarloaf's rectangle shader (`renderer/renderer.wgsl`: instanced quads
/// with corner radii, clip rectangles, discard) through its own pipeline
/// layout: a plain rect, a rounded rect and a clipped rect, checked at the
/// pixels that tell them apart. Needs `AKUMA_SUGARLOAF`.
fn t_sugarloaf_quads(g: &Gpu) -> TestResult {
    let Some(dir) = std::env::var_os("AKUMA_SUGARLOAF") else {
        return Ok(());
    };
    let path = std::path::Path::new(&dir).join("renderer/renderer.wgsl");
    let Ok(src) = std::fs::read_to_string(&path) else {
        println!("      ({} not found: skipped)", path.display());
        return Ok(());
    };
    let m = g.module(&src);
    let (w, h) = (80u32, 50u32);
    let fmt = wgpu::TextureFormat::Bgra8Unorm;
    let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None };
    let bgl0 = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            entry(
                0,
                wgpu::ShaderStages::VERTEX,
                wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            ),
            entry(1, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
        ],
    });
    let tex_ty = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    };
    let bgl1 = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[entry(0, wgpu::ShaderStages::FRAGMENT, tex_ty), entry(1, wgpu::ShaderStages::FRAGMENT, tex_ty)],
    });
    let pl = g.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
        immediate_size: 0,
    });
    use wgpu::VertexFormat as F;
    let attrs = [
        (F::Float32x3, 0u64),
        (F::Float32x4, 12),
        (F::Float32x4, 28),
        (F::Sint32x2, 44),
        (F::Float32x2, 52),
        (F::Float32x4, 60),
        (F::Sint32, 76),
        (F::Float32x4, 80),
    ]
    .iter()
    .enumerate()
    .map(|(i, &(format, offset))| wgpu::VertexAttribute { format, offset, shader_location: i as u32 })
    .collect::<Vec<_>>();
    let vbl = wgpu::VertexBufferLayout { array_stride: 96, step_mode: wgpu::VertexStepMode::Instance, attributes: &attrs };
    let c = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    let pipe = g.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pl),
        vertex: wgpu::VertexState { module: &m, entry_point: Some("vs_instanced"), buffers: &[Some(vbl)], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &m,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState { format: fmt, blend: Some(wgpu::BlendState { color: c, alpha: c }), write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    // pixels -> NDC, column-major
    let mut u = Vec::new();
    for v in [2.0 / w as f32, 0.0, 0.0, 0.0, 0.0, -2.0 / h as f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0f32] {
        u.extend_from_slice(&v.to_le_bytes());
    }
    let ubuf = g.buffer(&u, wgpu::BufferUsages::UNIFORM);
    let smp = g.sampler(wgpu::FilterMode::Nearest, wgpu::AddressMode::ClampToEdge);
    let bg0 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl0,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&smp) },
        ],
    });
    let color_tex = g.texture(1, 1, wgpu::TextureFormat::Rgba8Unorm);
    g.upload(&color_tex, 1, 1, 4, &[255, 255, 255, 255]);
    let mask_tex = g.texture(1, 1, wgpu::TextureFormat::R8Unorm);
    g.upload(&mask_tex, 1, 1, 1, &[255]);
    let (cv, mv) = (
        color_tex.create_view(&wgpu::TextureViewDescriptor::default()),
        mask_tex.create_view(&wgpu::TextureViewDescriptor::default()),
    );
    let bg1 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl1,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&cv) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&mv) },
        ],
    });
    // (x, y, size w, h, radius, clip rect)
    let rects: [(f32, f32, f32, f32, f32, [f32; 4]); 3] = [
        (2.0, 2.0, 20.0, 15.0, 0.0, [0.0; 4]),
        (26.0, 2.0, 40.0, 30.0, 8.0, [0.0; 4]),
        (2.0, 24.0, 20.0, 20.0, 0.0, [0.0, 0.0, 12.0, 100.0]),
    ];
    let mut inst = Vec::new();
    for (x, y, rw, rh, r, clip) in rects {
        let f = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let i = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        inst.extend(f(&[x, y, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]));
        inst.extend(i(&[0, 0]));
        inst.extend(f(&[rw, rh, r, r, r, r]));
        inst.extend(i(&[0]));
        inst.extend(f(&clip));
    }
    let vb = g.buffer(&inst, wgpu::BufferUsages::VERTEX);
    let target = g.texture(w, h, fmt);
    g.pass(&target, Some(wgpu::Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 }), |rp| {
        rp.set_pipeline(&pipe);
        rp.set_bind_group(0, &bg0, &[]);
        rp.set_bind_group(1, &bg1, &[]);
        rp.set_vertex_buffer(0, vb.slice(..));
        rp.draw(0..4, 0..3);
    });
    let px = g.read(&target, w, h, 4);
    let at = |x: u32, y: u32| {
        let o = ((y * w + x) * 4) as usize;
        [px[o + 2], px[o + 1], px[o], px[o + 3]] // r g b a from BGRA
    };
    let (red, blue) = ([255u8, 0, 0, 255], [0u8, 0, 255, 255]);
    let checks: [(&str, u32, u32, [u8; 4]); 8] = [
        ("plain rect, inside", 10, 8, red),
        ("plain rect, just outside", 22, 8, blue),
        ("rounded rect, centre", 46, 17, red),
        ("rounded rect, cut-off corner", 26, 2, blue),
        ("rounded rect, straight left edge", 26, 17, red),
        ("clipped rect, inside the clip", 5, 30, red),
        ("clipped rect, outside the clip", 15, 30, blue),
        ("clipped rect, outside the rect", 25, 30, blue),
    ];
    for (what, x, y, want) in checks {
        let got = at(x, y);
        expect(got.iter().zip(&want).all(|(a, b)| (*a as i32 - *b as i32).abs() <= 1), || {
            format!("{what}: pixel ({x},{y}) = {got:?}, expected {want:?}")
        })?;
    }
    Ok(())
}

/// rio/sugarloaf's real `grid.wgsl` through the real wgpu API with its own
/// pipeline layout: the cell-background pass (storage-buffer cells, 160-byte
/// uniforms, colour-space maths, premultiplied blend into Bgra8Unorm) and the
/// instanced glyph pass (7 vertex attributes in 4 formats, a mat4 projection,
/// triangle strips, `textureLoad` from an R8 atlas). Needs
/// `AKUMA_SUGARLOAF=<path to rio>/sugarloaf/src`; skipped without it.
fn t_sugarloaf_grid(g: &Gpu) -> TestResult {
    let Some(dir) = std::env::var_os("AKUMA_SUGARLOAF") else {
        println!("      (sugarloaf grid.wgsl: set AKUMA_SUGARLOAF=<rio>/sugarloaf/src to run)");
        return Ok(());
    };
    let (cols, rows, cw, ch) = (6u32, 3u32, 12u32, 16u32);
    let (w, h) = (cols * cw, rows * ch);
    let (px, _ms) = sugarloaf_scene(g, &dir, cols, rows, cw, ch, 1, 1)?;
    sugarloaf_check(&px, cols, rows, cw, ch, w)
}

/// `AKUMA_SUGARLOAF=<rio>/sugarloaf/src akuma-wgpu gpu-bench`: the same two
/// passes at 4K scale — a full-screen cell-background pass over a 240x67 grid
/// and a glyph quad in every cell — timed per frame.
pub fn bench() -> i32 {
    let Some(dir) = std::env::var_os("AKUMA_SUGARLOAF") else {
        eprintln!("gpu-bench needs AKUMA_SUGARLOAF=<rio>/sugarloaf/src");
        return 2;
    };
    let g = Gpu::new();
    // fixed per-fragment overhead of the pipeline: a constant-colour shader
    // over the same 3840x2144 target, without and with premultiplied blending
    {
        let (w, h) = (3840u32, 2144u32);
        let src = format!(
            "{FULL_TRI}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{ return full(vi); }}
@fragment fn fs() -> @location(0) vec4<f32> {{ return vec4<f32>(0.25, 0.5, 0.75, 1.0); }}"
        );
        let m = g.module(&src);
        let c = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        {
            let fmt = wgpu::TextureFormat::Bgra8Unorm;
            let tex = g.texture(w, h, fmt);
            let p = g.pipeline(&m, &g.empty_layout(), fmt, None, wgpu::PrimitiveTopology::TriangleList, None, &[]);
            let _ = &p;
            let t0 = crate::clock::monotonic();
            for _ in 0..3 {
                g.pass(&tex, Some(BLACK), |_rp| {});
            }
            let ms = (crate::clock::monotonic() - t0) * 1000.0 / 3.0;
            println!("{w}x{h}, clear only (empty pass): {ms:.1} ms/frame");
            let t0 = crate::clock::monotonic();
            for _ in 0..3 {
                g.pass(&tex, None, |rp| {
                    rp.set_pipeline(&p);
                    rp.draw(0..3, 0..1);
                });
            }
            let ms = (crate::clock::monotonic() - t0) * 1000.0 / 3.0;
            println!("{w}x{h}, constant shader, no clear, no blend: {ms:.1} ms/frame ({:.0} ns/fragment)", ms * 1e6 / (w * h) as f64);
        }
        // a trivial shader that is NOT constant per primitive (reads position):
        // isolates the per-fragment pipeline cost of the batched/wide path
        {
            let src2 = format!(
                "{FULL_TRI}
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{ return full(vi); }}
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{ return vec4<f32>(p.x / 3840.0, p.y / 2144.0, 0.5, 1.0); }}"
            );
            let m2 = g.module(&src2);
            let fmt = wgpu::TextureFormat::Bgra8Unorm;
            let tex = g.texture(w, h, fmt);
            let p = g.pipeline(&m2, &g.empty_layout(), fmt, None, wgpu::PrimitiveTopology::TriangleList, None, &[]);
            let t0 = crate::clock::monotonic();
            for _ in 0..3 {
                g.pass(&tex, None, |rp| {
                    rp.set_pipeline(&p);
                    rp.draw(0..3, 0..1);
                });
            }
            let ms = (crate::clock::monotonic() - t0) * 1000.0 / 3.0;
            println!("{w}x{h}, position-dependent shader (4 insts), no blend: {ms:.1} ms/frame ({:.0} ns/fragment)", ms * 1e6 / (w * h) as f64);
        }
        for (what, blend) in [
            ("constant shader, no blend", None),
            ("constant shader, premultiplied blend", Some(wgpu::BlendState { color: c, alpha: c })),
        ] {
            let fmt = wgpu::TextureFormat::Bgra8Unorm;
            let tex = g.texture(w, h, fmt);
            let p = g.pipeline(&m, &g.empty_layout(), fmt, blend, wgpu::PrimitiveTopology::TriangleList, None, &[]);
            let t0 = crate::clock::monotonic();
            for _ in 0..3 {
                g.pass(&tex, Some(BLACK), |rp| {
                    rp.set_pipeline(&p);
                    rp.draw(0..3, 0..1);
                });
            }
            let ms = (crate::clock::monotonic() - t0) * 1000.0 / 3.0;
            println!("{w}x{h}, {what}: {ms:.1} ms/frame ({:.0} ns/fragment)", ms * 1e6 / (w * h) as f64);
        }
    }
    bench_quads(&g);
    let (cols, rows, cw, ch) = (240u32, 67u32, 16u32, 32u32);
    for (what, glyphs) in [("bg pass only", 0u32), ("bg + a glyph in every cell", cols * rows)] {
        match sugarloaf_scene(&g, &dir, cols, rows, cw, ch, glyphs, 24) {
            Ok((_px, ms)) => println!(
                "{}x{} grid, {what}: {ms:.1} ms/frame, best {:.1} ({:.1} fps)",
                cols * cw, rows * ch, last_best(), 1000.0 / ms
            ),
            Err(e) => {
                println!("FAIL {what}: {e}");
                return 1;
            }
        }
    }
    // what a terminal actually looks like: a few distinct cell colours in runs
    // and glyphs in ~60% of the cells
    TERMINAL_SCENE.store(true, std::sync::atomic::Ordering::Relaxed);
    for (what, glyphs) in [("terminal-like, bg pass only", 0u32), ("terminal-like, bg + glyphs in 60% of cells", cols * rows * 6 / 10)] {
        match sugarloaf_scene(&g, &dir, cols, rows, cw, ch, glyphs, 24) {
            Ok((_px, ms)) => println!("{}x{} grid, {what}: {ms:.1} ms/frame, best {:.1} ({:.1} fps)", cols * cw, rows * ch, last_best(), 1000.0 / ms),
            Err(e) => {
                println!("FAIL {what}: {e}");
                return 1;
            }
        }
    }
    TERMINAL_SCENE.store(false, std::sync::atomic::Ordering::Relaxed);
    if super::prof::enabled() {
        super::prof::report();
    }
    0
}

/// 16k small instanced quads (6x8 px, a terminal's glyphs) with fragment
/// shaders of increasing weight: separates per-triangle, per-span and
/// per-fragment costs of the glyph pass.
fn bench_quads(g: &Gpu) {
    let (w, h) = (3840u32, 2144u32);
    const HEAD: &str = "
struct VO { @builtin(position) p: vec4<f32>, @location(0) @interpolate(flat) c: vec4<f32>, @location(1) t: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
    let col = ii % 240u; let row = (ii / 240u) % 67u;
    let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
    let px = vec2<f32>(f32(col) * 16.0 + 3.0, f32(row) * 32.0 + 12.0) + corner * vec2<f32>(SX, SY);
    var o: VO;
    o.p = vec4<f32>(px.x / 1920.0 - 1.0, 1.0 - px.y / 1072.0, 0.0, 1.0);
    o.c = vec4<f32>(1.0, 0.5, 0.25, 1.0);
    o.t = corner * 6.0;
    return o;
}
";
    let fmt = wgpu::TextureFormat::Bgra8Unorm;
    let c = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    let premul = Some(wgpu::BlendState { color: c, alpha: c });
    let n = 240 * 67;
    let flat = "@fragment fn fs(i: VO) -> @location(0) vec4<f32> { return i.c; }";
    let lin = "@fragment fn fs(i: VO) -> @location(0) vec4<f32> { return vec4<f32>(i.t.x * 0.1, i.t.y * 0.1, 0.5, 1.0); }";
    let alpha = "@fragment fn fs(i: VO) -> @location(0) vec4<f32> { let a = i.t.x * 0.1 + 0.2; return vec4<f32>(a, a, a, a); }";
    for (what, size, fs) in [
        ("6x8 flat colour, opaque", ("6.0", "8.0"), flat),
        ("6x8 linear varying", ("6.0", "8.0"), lin),
        ("6x8 linear varying, alpha < 1 (blend reads dst)", ("6.0", "8.0"), alpha),
        ("1x1 flat colour (per-triangle overhead)", ("1.0", "1.0"), flat),
    ] {
        let m = g.module(&format!("{}{fs}", HEAD.replace("SX", size.0).replace("SY", size.1)));
        let tex = g.texture(w, h, fmt);
        let p = g.pipeline(&m, &g.empty_layout(), fmt, premul, wgpu::PrimitiveTopology::TriangleStrip, None, &[]);
        let run = || {
            g.pass(&tex, None, |rp| {
                rp.set_pipeline(&p);
                rp.draw(0..4, 0..n);
            })
        };
        run();
        let t0 = crate::clock::monotonic();
        for _ in 0..5 {
            run();
        }
        let ms = (crate::clock::monotonic() - t0) * 1000.0 / 5.0;
        println!("{n} quads, {what}: {ms:.1} ms ({:.0} ns/quad)", ms * 1e6 / n as f64);
    }
}

/// fastest timed frame of the last `sugarloaf_scene` (f64 bits, ms)
static LAST_BEST_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn last_best() -> f64 {
    f64::from_bits(LAST_BEST_MS.load(std::sync::atomic::Ordering::Relaxed))
}

static TERMINAL_SCENE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Run sugarloaf's `grid.wgsl` bg pass and, if `glyphs > 0`, a glyph pass with
/// that many instances (the first at grid (2,1) as the verified glyph, the
/// rest spread over the grid). Returns the pixels of the last frame and the
/// mean milliseconds per frame over `frames` timed repetitions.
#[allow(clippy::too_many_arguments)]
fn sugarloaf_scene(
    g: &Gpu,
    dir: &std::ffi::OsStr,
    cols: u32,
    rows: u32,
    cw: u32,
    ch: u32,
    glyphs: u32,
    frames: u32,
) -> Result<(Vec<u8>, f64), String> {
    let path = std::path::Path::new(dir).join("grid/shaders/grid.wgsl");
    let src = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let m = g.module(&src);

    let (w, h) = (cols * cw, rows * ch);
    let fmt = wgpu::TextureFormat::Bgra8Unorm;

    // ---- uniforms (160 bytes; see the struct in grid.wgsl) ----
    let mut u = vec![0u8; 160];
    let f32s = |u: &mut [u8], at: usize, v: &[f32]| {
        for (i, x) in v.iter().enumerate() {
            u[at + i * 4..at + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
    };
    let u32s = |u: &mut [u8], at: usize, v: &[u32]| {
        for (i, x) in v.iter().enumerate() {
            u[at + i * 4..at + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
    };
    // column-major ortho: pixels (y down) -> NDC
    f32s(&mut u, 0, &[2.0 / w as f32, 0.0, 0.0, 0.0, 0.0, -2.0 / h as f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0]);
    f32s(&mut u, 112, &[cw as f32, ch as f32]); // cell_size
    u32s(&mut u, 120, &[cols, rows]); // grid_size
    u32s(&mut u, 156, &[1]); // input_colorspace: plain sRGB (no gamut remap)
    let ubuf = g.buffer(&u, wgpu::BufferUsages::UNIFORM);

    // ---- cells: one u32 each, rgba little-endian ----
    // the default scene gives every cell its own colour; the "terminal" scene
    // is a dark background with runs of a few highlight colours
    let terminal = TERMINAL_SCENE.load(std::sync::atomic::Ordering::Relaxed);
    let cell_rgb = |c: u32, r: u32| {
        if terminal {
            const PAL: [[u32; 3]; 3] = [[70, 40, 90], [30, 80, 60], [110, 70, 30]];
            let k = ((c / 6) * 7 + r * 13) % 10;
            if k < 7 { [30, 30, 46] } else { PAL[(k - 7) as usize] }
        } else {
            [40 * c + 20, 80 * r + 30, 200 - 20 * c]
        }
    };
    let mut cells = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let [cr, cg, cb] = cell_rgb(c, r);
            cells.extend_from_slice(&(cr | cg << 8 | cb << 16 | 255 << 24).to_le_bytes());
        }
    }
    let cbuf = g.buffer(&cells, wgpu::BufferUsages::STORAGE);

    let vis = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
    let bg_bgl = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: vis,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: vis,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });
    let bg0 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bg_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cbuf.as_entire_binding() },
        ],
    });
    let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let atlas_bgl = g.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[tex_entry(0), tex_entry(1)],
    });

    // grayscale atlas 16x16: texel (x,y) = 255 when (x+y) is even
    let atlas = g.texture(16, 16, wgpu::TextureFormat::R8Unorm);
    let atlas_px: Vec<u8> = (0..16 * 16).map(|i| if (i % 16 + i / 16) % 2 == 0 { 255 } else { 0 }).collect();
    g.upload(&atlas, 16, 16, 1, &atlas_px);
    let color_atlas = g.texture(1, 1, wgpu::TextureFormat::Rgba8Unorm);
    g.upload(&color_atlas, 1, 1, 4, &[255, 0, 255, 255]);
    let (av, cv) = (
        atlas.create_view(&wgpu::TextureViewDescriptor::default()),
        color_atlas.create_view(&wgpu::TextureViewDescriptor::default()),
    );
    let bg1 = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &atlas_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&av) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&cv) },
        ],
    });

    let premul = {
        let c = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        wgpu::BlendState { color: c, alpha: c }
    };
    let target_state = [Some(wgpu::ColorTargetState { format: fmt, blend: Some(premul), write_mask: wgpu::ColorWrites::ALL })];

    // ---- background pipeline ----
    let bg_pl = g.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bg_bgl)],
        immediate_size: 0,
    });
    let bg_pipe = g.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&bg_pl),
        vertex: wgpu::VertexState { module: &m, entry_point: Some("grid_bg_vertex"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &m, entry_point: Some("grid_bg_fragment"), targets: &target_state, compilation_options: Default::default() }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // ---- text pipeline (attribute layout copied from grid/webgpu.rs) ----
    let text_pl = g.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bg_bgl), Some(&atlas_bgl)],
        immediate_size: 0,
    });
    let attrs = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32x2, offset: 0, shader_location: 0 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32x2, offset: 8, shader_location: 1 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Sint16x2, offset: 16, shader_location: 2 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint16x2, offset: 20, shader_location: 3 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Unorm8x4, offset: 24, shader_location: 4 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint8, offset: 28, shader_location: 5 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint8, offset: 29, shader_location: 6 },
    ];
    let vbl = wgpu::VertexBufferLayout { array_stride: 32, step_mode: wgpu::VertexStepMode::Instance, attributes: &attrs };
    let text_pipe = g.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&text_pl),
        vertex: wgpu::VertexState { module: &m, entry_point: Some("grid_text_vertex"), buffers: &[Some(vbl)], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &m, entry_point: Some("grid_text_fragment"), targets: &target_state, compilation_options: Default::default() }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    // glyph instances; the first is the verified one: grid (2,1), atlas pos
    // (1,1), size 6x8, bearings (3,12), white. The rest are the same glyph
    // spread over the other cells (cell 16x32 in the bench has room for it).
    let mut inst = Vec::new();
    for i in 0..glyphs {
        let (gx, gy) = if i == 0 { (2, 1) } else { (i % cols, (i / cols) % rows) };
        inst.extend_from_slice(&1u32.to_le_bytes());
        inst.extend_from_slice(&1u32.to_le_bytes());
        inst.extend_from_slice(&6u32.to_le_bytes());
        inst.extend_from_slice(&8u32.to_le_bytes());
        inst.extend_from_slice(&3i16.to_le_bytes());
        inst.extend_from_slice(&12i16.to_le_bytes());
        inst.extend_from_slice(&(gx as u16).to_le_bytes());
        inst.extend_from_slice(&(gy as u16).to_le_bytes());
        inst.extend_from_slice(&[255, 255, 255, 255]);
        inst.extend_from_slice(&[0, 0, 0, 0]); // atlas = grayscale, bools = 0, pad
    }
    let vb = g.buffer(&inst, wgpu::BufferUsages::VERTEX);

    let target = g.texture(w, h, fmt);
    let mut total = 0.0;
    let mut best = f64::INFINITY;
    // with several frames, the first is a warm-up (specialization, page faults)
    // and not timed
    for i in 0..frames.max(1) + (frames > 1) as u32 {
        let t0 = crate::clock::monotonic();
        g.pass(&target, Some(BLACK), |rp| {
            rp.set_pipeline(&bg_pipe);
            rp.set_bind_group(0, &bg0, &[]);
            rp.draw(0..3, 0..1);
            if glyphs > 0 {
                rp.set_pipeline(&text_pipe);
                rp.set_bind_group(0, &bg0, &[]);
                rp.set_bind_group(1, &bg1, &[]);
                rp.set_vertex_buffer(0, vb.slice(..));
                rp.draw(0..4, 0..glyphs);
            }
        });
        if frames <= 1 || i > 0 {
            let dt = crate::clock::monotonic() - t0;
            total += dt;
            best = best.min(dt);
        }
    }
    let px = g.read(&target, w, h, 4);
    LAST_BEST_MS.store((best * 1000.0).to_bits(), std::sync::atomic::Ordering::Relaxed);
    Ok((px, total / frames.max(1) as f64 * 1000.0))
}

fn sugarloaf_check(px: &[u8], cols: u32, rows: u32, cw: u32, ch: u32, w: u32) -> TestResult {
    let cell_rgb = |c: u32, r: u32| [40 * c + 20, 80 * r + 30, 200 - 20 * c];
    // Bgra8Unorm bytes: b, g, r, a
    let at = |x: u32, y: u32| {
        let o = ((y * w + x) * 4) as usize;
        [px[o + 2] as i32, px[o + 1] as i32, px[o] as i32, px[o + 3] as i32]
    };
    let near = |got: [i32; 4], want: [i32; 4]| got.iter().zip(&want).all(|(a, b)| (a - b).abs() <= 1);
    // every cell centre carries its colour (corners avoid the glyph's cell (2,1))
    for r in 0..rows {
        for c in 0..cols {
            if (c, r) == (2, 1) {
                continue;
            }
            let [cr, cg, cb] = cell_rgb(c, r);
            let got = at(c * cw + 1, r * ch + 1);
            expect(near(got, [cr as i32, cg as i32, cb as i32, 255]), || {
                format!("cell ({c},{r}) = {got:?}, expected {:?}", [cr, cg, cb, 255])
            })?;
        }
    }
    // the glyph: pixel (px,py) in 27..33 x 20..28 shows atlas[(1+px-27, 1+py-20)]
    let bg = {
        let [cr, cg, cb] = cell_rgb(2, 1);
        [cr as i32, cg as i32, cb as i32, 255]
    };
    for gy in 0..8u32 {
        for gx in 0..6u32 {
            let (ax, ay) = (1 + gx, 1 + gy);
            let on = (ax + ay) % 2 == 0;
            let got = at(27 + gx, 20 + gy);
            let want = if on { [255, 255, 255, 255] } else { bg };
            expect(near(got, want), || {
                format!("glyph texel ({gx},{gy}) atlas({ax},{ay}) on={on}: {got:?}, expected {want:?}")
            })?;
        }
    }
    // outside the glyph box, inside the same cell: untouched background
    expect(near(at(24 + 1, 16 + 1), bg), || format!("cell bg near the glyph = {:?}", at(25, 17)))
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
        ("memoized + specialized fragment shader vs CPU", t_memo_spec),
        ("over-blending onto random texels vs scalar definition", t_over_blend_vector),
        ("position-quantization runs match per-pixel shading", t_runs),
        ("random fragment shaders vs the interpreter", t_fuzz_fragment),
        ("sugarloaf grid.wgsl: cell backgrounds + instanced glyphs", t_sugarloaf_grid),
        ("sugarloaf renderer.wgsl: plain, rounded and clipped rects", t_sugarloaf_quads),
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
