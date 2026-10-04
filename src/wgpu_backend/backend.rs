//! The akuma custom wgpu backend: the `wgpu::custom::*` implementations that
//! turn wgpu into a framebuffer GPU (plan §5b option A).
//!
//! What this "hardware" is:
//!
//! * a CPU that executes WGSL by interpreting naga IR (`interp.rs`), so
//!   shader execution needs no executable memory — the kernel's W^X policy
//!   is respected by construction;
//! * a fixed-function rasterizer whose contract is *the demo's original
//!   scanline walk*: `raster_tri`'s fma area test with the -0.01 epsilon
//!   cull, `total_cmp` y-sort, `edge_xz` spans, per-pixel strict `z <`
//!   depth-test, provoking-vertex Flat varyings, draws in submission order.
//!   That contract is what makes M3's acceptance test (bit-identical frames
//!   vs softrender) well-posed: fixed-function IS the specification here,
//!   and ours is written down in `raster_tri` below, verbatim from
//!   softrender.rs.
//!
//! Presentation is the plain wgpu texture path: the scene renders into a
//! `bgra8uint` texture (raw bytes — no conversion anywhere between shader
//! integers and the framebuffer word), `copy_texture_to_buffer` reads it
//! back, and the caller blits rows into the fbdev `Frame`.
//!
//! Threading: everything runs on the calling thread — `submit` executes
//! synchronously and the GPU has no other threads. The interface traits are
//! Send+Sync, so shared buffer/texture state hides behind `Arc<Mutex<..>>`
//! *inside* the resource data; handles recovered through
//! `Resource::as_custom` clone those interior Arcs, which is what keeps
//! buffer identity stable across handles.

use std::sync::{Arc, Mutex};

use super::exec::{Invoker, RawVertex, Stage, Varyings};
use super::interp::{Resources, Shader};
use wgpu::{
    BufferAddress, BufferSize, MapMode,
};

// ---------------------------------------------------------------------------
// Resource data
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct BufferData {
    pub bytes: Arc<Mutex<Vec<u8>>>,
    pub size: BufferAddress,
}

#[derive(Debug)]
pub enum TexStore {
    /// 4 raw bytes per texel
    Color(Mutex<Vec<u8>>),
    /// one f32 depth per texel
    Depth(Mutex<Vec<f32>>),
}

#[derive(Debug)]
pub struct TextureData {
    pub size: wgpu::Extent3d,
    pub format: wgpu::TextureFormat,
    pub store: Arc<TexStore>,
}

impl TextureData {
    pub fn bytes_per_texel(&self) -> u32 {
        super::format::bytes_per_texel(self.format)
            .unwrap_or_else(|| panic!("akuma backend: unsupported texture format {:?}", self.format))
    }
}

#[derive(Debug, Clone)]
pub struct ViewData {
    pub tex: Arc<TextureData>,
}

#[derive(Debug, Clone)]
pub enum BindRes {
    Buffer(Arc<Mutex<Vec<u8>>>),
    /// recorded into bind groups for completeness; the draw executor rejects
    /// texture bindings (the demo's shaders are buffer-only)
    Texture(ViewData),
    Sampler(super::texture::SmpRef),
}

#[derive(Debug, Clone)]
pub struct BindGroupData {
    /// (binding number -> resource), in entry order
    pub entries: Vec<(u32, BindRes)>,
}

#[derive(Clone, Debug)]
pub struct StageData {
    pub stage: Arc<Stage>,
}

impl StageData {
    fn resolve(shader: Arc<Shader>, entry_point: Option<&str>, what: &str) -> StageData {
        let entry = shader
            .entry(entry_point)
            .unwrap_or_else(|e| panic!("akuma backend: {what} entry: {e}"));
        StageData { stage: Arc::new(Stage::new(shader, entry)) }
    }
}

#[derive(Clone, Debug)]
pub struct RenderPipelineData {
    pub vs: StageData,
    pub fs: Option<StageData>,
    pub depth: Option<wgpu::DepthStencilState>,
    pub groups: Vec<Arc<BindGroupLayoutData>>,
    pub topology: wgpu::PrimitiveTopology,
    pub cull: Option<wgpu::Face>,
    pub front: wgpu::FrontFace,
    /// color target 0 (the only one the backend draws to)
    pub target: Option<TargetInfo>,
    pub vbufs: Vec<VtxLayout>,
}

#[derive(Clone, Debug)]
pub struct TargetInfo {
    pub format: wgpu::TextureFormat,
    pub blend: Option<wgpu::BlendState>,
    pub write_mask: wgpu::ColorWrites,
}

#[derive(Clone, Debug)]
pub struct VtxAttr {
    pub format: wgpu::VertexFormat,
    pub offset: u64,
    pub location: u32,
}

#[derive(Clone, Debug)]
pub struct VtxLayout {
    pub stride: u64,
    pub step: wgpu::VertexStepMode,
    pub attrs: Vec<VtxAttr>,
}

impl RenderPipelineData {
    /// the demo's raw-bytes contract (pixel-space positions, flat varyings,
    /// no blending) vs. standard wgpu semantics
    pub fn legacy(&self) -> bool {
        self.target.as_ref().is_none_or(|t| super::format::is_legacy_raw(t.format))
    }
}

pub struct CommandBufferShared {
    pub cmds: Mutex<Vec<Cmd>>,
}

// Vec<Cmd> is not Debug; the shared list only needs a length in Debug output
impl std::fmt::Debug for CommandBufferShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandBufferShared")
            .field("cmds.len", &self.cmds.lock().unwrap().len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Recorded commands (the "ring buffer" of our GPU)
// ---------------------------------------------------------------------------

pub enum Cmd {
    CopyBufferToBuffer {
        src_bytes: Arc<Mutex<Vec<u8>>>,
        src_offset: BufferAddress,
        src_size: BufferAddress,
        dst_bytes: Arc<Mutex<Vec<u8>>>,
        dst_offset: BufferAddress,
        size: Option<BufferAddress>,
    },
    CopyTextureToBuffer {
        src_store: Arc<TexStore>,
        src_width: u32,
        src_bpt: u32,
        src_origin: wgpu::Origin3d,
        dst_bytes: Arc<Mutex<Vec<u8>>>,
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    },
    CopyBufferToTexture {
        src_bytes: Arc<Mutex<Vec<u8>>>,
        layout: wgpu::TexelCopyBufferLayout,
        dst_store: Arc<TexStore>,
        dst_width: u32,
        dst_bpt: u32,
        dst_origin: wgpu::Origin3d,
        size: wgpu::Extent3d,
    },
    CopyTextureToTexture {
        src_store: Arc<TexStore>,
        src_width: u32,
        src_origin: wgpu::Origin3d,
        dst_store: Arc<TexStore>,
        dst_width: u32,
        dst_origin: wgpu::Origin3d,
        bpt: u32,
        size: wgpu::Extent3d,
    },
    ClearBuffer {
        bytes: Arc<Mutex<Vec<u8>>>,
        buf_size: BufferAddress,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    },
    Render {
        data: RenderPassData,
    },
}

#[derive(Debug)]
pub struct RenderPassData {
    /// (texture, load: true = clear to the color, false = keep contents)
    pub color: Option<(Arc<TextureData>, bool, wgpu::Color)>,
    /// (depth texture, clear value; None = keep contents)
    pub depth: Option<(Arc<TextureData>, Option<f32>)>,
    pub cmds: Vec<PassCmd>,
}

#[derive(Debug)]
pub enum PassCmd {
    SetPipeline(RenderPipelineData),
    SetBindGroup(u32, BindGroupData),
    Draw {
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    },
    DrawIndexed {
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    },
    SetVertexBuffer {
        slot: u32,
        buf: Option<BufBind>,
    },
    SetIndexBuffer {
        buf: BufBind,
        format: wgpu::IndexFormat,
    },
    SetViewport(super::raster::Viewport),
    SetScissor([u32; 4]),
    SetBlendConstant(wgpu::Color),
}

/// a buffer bound at an offset (vertex/index buffers)
#[derive(Debug, Clone)]
pub struct BufBind {
    pub bytes: Arc<Mutex<Vec<u8>>>,
    pub offset: u64,
}

// ---------------------------------------------------------------------------
// Instance / Adapter / Device / Queue
// ---------------------------------------------------------------------------

/// Marker instance type; enters via `wgpu::Instance::from_custom(Instance)`.
#[derive(Debug)]
pub struct Instance;

/// A local immediately-ready future — same job as `std::future::Ready`, but
/// a private type keeps rustc's CoerceUnsized solver from overflowing on the
/// blanket-impl chain behind wgpu's future trait aliases.
struct Ready<T>(Option<T>);

impl<T: Unpin> std::future::Future for Ready<T> {
    type Output = T;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<T> {
        std::task::Poll::Ready(self.get_mut().0.take().expect("polled after ready"))
    }
}

impl wgpu::custom::InstanceInterface for Instance {
    fn new(_desc: wgpu::InstanceDescriptor) -> Self {
        // only `Instance::from_custom` is used in this crate
        Instance
    }

    unsafe fn create_surface(
        &self,
        _target: wgpu::SurfaceTargetUnsafe,
    ) -> Result<wgpu::custom::DispatchSurface, wgpu::CreateSurfaceError> {
        panic!("akuma backend: no surfaces (the framebuffer is not a wgpu surface)");
    }

    fn request_adapter(
        &self,
        _options: &wgpu::RequestAdapterOptions<'_, '_>,
    ) -> std::pin::Pin<Box<dyn wgpu::custom::RequestAdapterFuture>> {
        Box::pin(Ready(Some(Ok(wgpu::custom::DispatchAdapter::custom(Adapter)))))
    }

    fn poll_all_devices(&self, _force_wait: bool) -> bool {
        true
    }

    fn wgsl_language_features(&self) -> wgpu::WgslLanguageFeatures {
        wgpu::WgslLanguageFeatures::empty()
    }

    fn enumerate_adapters(
        &self,
        _backends: wgpu::Backends,
    ) -> std::pin::Pin<Box<dyn wgpu::custom::EnumerateAdapterFuture>> {
        Box::pin(Ready(Some(vec![
            wgpu::custom::DispatchAdapter::custom(Adapter),
        ])))
    }
}

#[derive(Debug)]
pub struct Adapter;

fn adapter_info() -> wgpu::AdapterInfo {
    wgpu::AdapterInfo::new(wgpu::DeviceType::Cpu, wgt_backend())
}

fn wgt_backend() -> wgpu::Backend {
    wgpu::Backend::Noop
}

fn device_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_texture_dimension_2d: 16384,
        ..wgpu::Limits::default()
    }
}

impl wgpu::custom::AdapterInterface for Adapter {
    fn request_device(
        &self,
        _desc: &wgpu::DeviceDescriptor<'_>,
    ) -> std::pin::Pin<Box<dyn wgpu::custom::RequestDeviceFuture>> {
        let device = wgpu::custom::DispatchDevice::custom(Device);
        let queue = wgpu::custom::DispatchQueue::custom(Queue);
        Box::pin(Ready(Some(Ok((device, queue)))))
    }

    fn is_surface_supported(&self, _surface: &wgpu::custom::DispatchSurface) -> bool {
        false
    }

    fn features(&self) -> wgpu::Features {
        wgpu::Features::empty()
    }

    fn limits(&self) -> wgpu::Limits {
        device_limits()
    }

    fn downlevel_capabilities(&self) -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities::default()
    }

    fn get_info(&self) -> wgpu::AdapterInfo {
        adapter_info()
    }

    fn get_texture_format_features(
        &self,
        _format: wgpu::TextureFormat,
    ) -> wgpu::TextureFormatFeatures {
        wgpu::TextureFormatFeatures { allowed_usages: wgpu::TextureUsages::all(), flags: wgpu::TextureFormatFeatureFlags::empty() }
    }

    fn get_presentation_timestamp(&self) -> wgpu::PresentationTimestamp {
        wgpu::PresentationTimestamp::INVALID_TIMESTAMP
    }

    fn cooperative_matrix_properties(&self) -> Vec<wgpu::wgt::CooperativeMatrixProperties> {
        Vec::new()
    }
}

#[derive(Debug)]
pub struct Device;

impl Device {
    fn shader_data(module: &wgpu::ShaderModule) -> Arc<Shader> {
        module
            .as_custom::<ShaderData>()
            .expect("akuma backend: foreign shader module")
            .shader
            .clone()
    }
}

impl wgpu::custom::DeviceInterface for Device {
    fn features(&self) -> wgpu::Features {
        wgpu::Features::empty()
    }

    fn limits(&self) -> wgpu::Limits {
        device_limits()
    }

    fn adapter_info(&self) -> wgpu::AdapterInfo {
        adapter_info()
    }

    fn create_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor<'_>,
        _shader_bound_checks: wgpu::ShaderRuntimeChecks,
    ) -> wgpu::custom::DispatchShaderModule {
        let wgsl = match &desc.source {
            wgpu::ShaderSource::Wgsl(s) => s,
            other => panic!("akuma backend: only WGSL sources are supported, got {other:?}"),
        };
        let shader = Shader::parse(wgsl)
            .unwrap_or_else(|e| panic!("akuma backend: {e}\nshader source:\n{wgsl}"));
        wgpu::custom::DispatchShaderModule::custom(ShaderData {
            shader: Arc::new(shader),
        })
    }

    unsafe fn create_shader_module_passthrough(
        &self,
        _desc: &wgpu::ShaderModuleDescriptorPassthrough<'_>,
    ) -> wgpu::custom::DispatchShaderModule {
        panic!("akuma backend: passthrough shader modules unsupported");
    }

    fn create_bind_group_layout(
        &self,
        desc: &wgpu::BindGroupLayoutDescriptor<'_>,
    ) -> wgpu::custom::DispatchBindGroupLayout {
        wgpu::custom::DispatchBindGroupLayout::custom(BindGroupLayoutData {
            entries: desc.entries.to_vec(),
        })
    }

    fn create_bind_group(
        &self,
        desc: &wgpu::BindGroupDescriptor<'_>,
    ) -> wgpu::custom::DispatchBindGroup {
        let mut entries = Vec::new();
        for e in desc.entries {
            let res = match &e.resource {
                wgpu::BindingResource::Buffer(b) => {
                    let bd = b
                        .buffer
                        .as_custom::<BufferData>()
                        .expect("akuma backend: foreign buffer");
                    BindRes::Buffer(Arc::clone(&bd.bytes))
                }
                wgpu::BindingResource::TextureView(v) => {
                    let vd = v
                        .as_custom::<ViewData>()
                        .expect("akuma backend: foreign texture view");
                    BindRes::Texture(vd.clone())
                }
                wgpu::BindingResource::Sampler(smp) => {
                    let sd = smp
                        .as_custom::<SamplerData>()
                        .expect("akuma backend: foreign sampler");
                    BindRes::Sampler(sd.desc)
                }
                other => panic!("akuma backend: bind group resource {other:?} unsupported"),
            };
            entries.push((e.binding, res));
        }
        wgpu::custom::DispatchBindGroup::custom(BindGroupData { entries })
    }

    fn create_pipeline_layout(
        &self,
        desc: &wgpu::PipelineLayoutDescriptor<'_>,
    ) -> wgpu::custom::DispatchPipelineLayout {
        let mut groups = Vec::new();
        for l in desc.bind_group_layouts.iter().flatten().copied() {
            let data = l
                .as_custom::<BindGroupLayoutData>()
                .expect("akuma backend: foreign bind group layout");
            groups.push(Arc::new(BindGroupLayoutData {
                entries: data.entries.clone(),
            }));
        }
        wgpu::custom::DispatchPipelineLayout::custom(PipelineLayoutData { groups })
    }

    fn create_render_pipeline(
        &self,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> wgpu::custom::DispatchRenderPipeline {
        if let Some(f) = &desc.fragment {
            for t in f.targets.iter().flatten() {
                if super::format::bytes_per_texel(t.format).is_none()
                    || super::format::is_depth(t.format)
                {
                    panic!("akuma backend: unsupported render target format {:?}", t.format);
                }
            }
        }
        let vs = StageData::resolve(
            Device::shader_data(desc.vertex.module),
            desc.vertex.entry_point.as_deref(),
            "vertex",
        );
        let fs = desc.fragment.as_ref().map(|f| {
            StageData::resolve(Device::shader_data(f.module), f.entry_point.as_deref(), "fragment")
        });
        let groups = desc
            .layout
            .and_then(|l| l.as_custom::<PipelineLayoutData>())
            .map(|pl| pl.groups.clone())
            .unwrap_or_default();
        let target = desc
            .fragment
            .as_ref()
            .and_then(|f| f.targets.first().cloned().flatten())
            .map(|t| TargetInfo { format: t.format, blend: t.blend, write_mask: t.write_mask });
        let vbufs = desc
            .vertex
            .buffers
            .iter()
            .map(|b| match b {
                None => VtxLayout { stride: 0, step: wgpu::VertexStepMode::Vertex, attrs: Vec::new() },
                Some(b) => VtxLayout {
                stride: b.array_stride,
                step: b.step_mode,
                attrs: b
                    .attributes
                    .iter()
                    .map(|a| {
                        assert!(
                            (a.shader_location as usize) < super::exec::MAX_LOC,
                            "akuma backend: vertex attribute location {} >= {}",
                            a.shader_location,
                            super::exec::MAX_LOC
                        );
                        VtxAttr { format: a.format, offset: a.offset, location: a.shader_location }
                    })
                    .collect(),
                },
            })
            .collect();
        wgpu::custom::DispatchRenderPipeline::custom(RenderPipelineData {
            vs,
            fs,
            depth: desc.depth_stencil.clone(),
            groups,
            topology: desc.primitive.topology,
            cull: desc.primitive.cull_mode,
            front: desc.primitive.front_face,
            target,
            vbufs,
        })
    }

    fn create_mesh_pipeline(
        &self,
        _desc: &wgpu::MeshPipelineDescriptor<'_>,
    ) -> wgpu::custom::DispatchRenderPipeline {
        panic!("akuma backend: mesh pipelines unsupported");
    }

    fn create_compute_pipeline(
        &self,
        desc: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> wgpu::custom::DispatchComputePipeline {
        let cs = StageData::resolve(
            Device::shader_data(desc.module),
            desc.entry_point.as_deref(),
            "compute",
        );
        wgpu::custom::DispatchComputePipeline::custom(ComputePipelineData { cs })
    }

    unsafe fn create_pipeline_cache(
        &self,
        _desc: &wgpu::PipelineCacheDescriptor<'_>,
    ) -> wgpu::custom::DispatchPipelineCache {
        panic!("akuma backend: pipeline caches unsupported");
    }

    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> wgpu::custom::DispatchBuffer {
        wgpu::custom::DispatchBuffer::custom(BufferData {
            bytes: Arc::new(Mutex::new(vec![0u8; desc.size as usize])),
            size: desc.size,
        })
    }

    fn create_texture(&self, desc: &wgpu::TextureDescriptor<'_>) -> wgpu::custom::DispatchTexture {
        let pixels = desc.size.width as usize
            * desc.size.height as usize
            * desc.size.depth_or_array_layers as usize;
        assert_eq!(desc.mip_level_count, 1, "akuma backend: mip chains unsupported");
        assert_eq!(desc.sample_count, 1, "akuma backend: multisampling unsupported");
        assert_eq!(desc.dimension, wgpu::TextureDimension::D2, "akuma backend: only 2D textures");
        let store = match desc.format {
            wgpu::TextureFormat::Depth32Float => TexStore::Depth(Mutex::new(vec![0.0f32; pixels])),
            f => match super::format::bytes_per_texel(f) {
                Some(bpt) => TexStore::Color(Mutex::new(vec![0u8; pixels * bpt as usize])),
                None => panic!("akuma backend: unsupported texture format {f:?}"),
            },
        };
        wgpu::custom::DispatchTexture::custom(TextureData {
            size: desc.size,
            format: desc.format,
            store: Arc::new(store),
        })
    }

    fn create_external_texture(
        &self,
        _desc: &wgpu::ExternalTextureDescriptor<'_>,
        _planes: &[&wgpu::TextureView],
    ) -> wgpu::custom::DispatchExternalTexture {
        panic!("akuma backend: external textures unsupported");
    }

    fn create_blas(
        &self,
        _desc: &wgpu::CreateBlasDescriptor<'_>,
        _sizes: wgpu::BlasGeometrySizeDescriptors,
    ) -> (Option<u64>, wgpu::custom::DispatchBlas) {
        panic!("akuma backend: acceleration structures unsupported");
    }

    fn create_tlas(
        &self,
        _desc: &wgpu::CreateTlasDescriptor<'_>,
    ) -> wgpu::custom::DispatchTlas {
        panic!("akuma backend: acceleration structures unsupported");
    }

    fn create_sampler(
        &self,
        desc: &wgpu::SamplerDescriptor<'_>,
    ) -> wgpu::custom::DispatchSampler {
        assert!(desc.compare.is_none(), "akuma backend: comparison samplers unsupported");
        wgpu::custom::DispatchSampler::custom(SamplerData {
            desc: super::texture::SmpRef {
                mag: desc.mag_filter,
                min: desc.min_filter,
                addr_u: desc.address_mode_u,
                addr_v: desc.address_mode_v,
            },
        })
    }

    fn create_query_set(
        &self,
        _desc: &wgpu::QuerySetDescriptor<'_>,
    ) -> wgpu::custom::DispatchQuerySet {
        wgpu::custom::DispatchQuerySet::custom(QuerySetData)
    }

    fn create_command_encoder(
        &self,
        _desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> wgpu::custom::DispatchCommandEncoder {
        wgpu::custom::DispatchCommandEncoder::custom(Encoder {
            shared: Arc::new(CommandBufferShared {
                cmds: Mutex::new(Vec::new()),
            }),
        })
    }

    fn create_render_bundle_encoder(
        &self,
        _desc: &wgpu::RenderBundleEncoderDescriptor<'_>,
    ) -> wgpu::custom::DispatchRenderBundleEncoder {
        panic!("akuma backend: render bundles unsupported");
    }

    fn set_device_lost_callback(&self, _cb: wgpu::custom::BoxDeviceLostCallback) {}

    fn on_uncaptured_error(&self, _handler: Arc<dyn wgpu::UncapturedErrorHandler>) {}

    fn push_error_scope(&self, _filter: wgpu::ErrorFilter) -> u32 {
        0
    }

    fn pop_error_scope(
        &self,
        _index: u32,
    ) -> std::pin::Pin<Box<dyn wgpu::custom::PopErrorScopeFuture>> {
        Box::pin(Ready(Some(None)))
    }

    unsafe fn start_graphics_debugger_capture(&self) {}

    unsafe fn stop_graphics_debugger_capture(&self) {}

    fn poll(
        &self,
        _poll_type: wgpu::wgt::PollType<u64>,
    ) -> Result<wgpu::PollStatus, wgpu::PollError> {
        Ok(wgpu::PollStatus::QueueEmpty)
    }

    fn get_internal_counters(&self) -> wgpu::InternalCounters {
        wgpu::InternalCounters::default()
    }

    fn generate_allocator_report(&self) -> Option<wgpu::AllocatorReport> {
        None
    }

    fn destroy(&self) {}
}

#[derive(Debug)]
pub struct ShaderData {
    pub shader: Arc<Shader>,
}

impl wgpu::custom::ShaderModuleInterface for ShaderData {
    fn get_compilation_info(
        &self,
    ) -> std::pin::Pin<Box<dyn wgpu::custom::ShaderCompilationInfoFuture>> {
        Box::pin(Ready(Some(wgpu::CompilationInfo { messages: vec![] })))
    }
}

#[derive(Debug)]
pub struct BindGroupLayoutData {
    pub entries: Vec<wgpu::BindGroupLayoutEntry>,
}

impl wgpu::custom::BindGroupLayoutInterface for BindGroupLayoutData {}

/// The bind group contents. Cloned into recorded draw state; identity lives
/// in the interior Arcs.
impl wgpu::custom::BindGroupInterface for BindGroupData {}

#[derive(Debug)]
pub struct PipelineLayoutData {
    pub groups: Vec<Arc<BindGroupLayoutData>>,
}

impl wgpu::custom::PipelineLayoutInterface for PipelineLayoutData {}

#[derive(Debug)]
pub struct ComputePipelineData {
    /// kept for when a compute path (rain-as-compute?) arrives; dispatching
    /// is not wired on this backend yet
    #[allow(dead_code)]
    pub cs: StageData,
}

impl wgpu::custom::ComputePipelineInterface for ComputePipelineData {
    fn get_bind_group_layout(&self, _index: u32) -> wgpu::custom::DispatchBindGroupLayout {
        panic!("akuma backend: get_bind_group_layout on compute pipelines unused");
    }
}

#[derive(Debug)]
pub struct SamplerData {
    pub desc: super::texture::SmpRef,
}
impl wgpu::custom::SamplerInterface for SamplerData {}

#[derive(Debug)]
pub struct QuerySetData;
impl wgpu::custom::QuerySetInterface for QuerySetData {
    fn destroy(&self) {}
}

#[derive(Debug)]
pub struct Queue;

impl wgpu::custom::QueueInterface for Queue {
    fn write_buffer(
        &self,
        buffer: &wgpu::custom::DispatchBuffer,
        offset: BufferAddress,
        data: &[u8],
    ) {
        let b = buffer
            .as_custom::<BufferData>()
            .expect("akuma backend: foreign buffer");
        let mut bytes = b.bytes.lock().unwrap();
        let end = offset as usize + data.len();
        assert!(end <= bytes.len(), "write_buffer out of range");
        bytes[offset as usize..end].copy_from_slice(data);
    }

    fn create_staging_buffer(
        &self,
        size: BufferSize,
    ) -> Option<wgpu::custom::DispatchQueueWriteBuffer> {
        Some(wgpu::custom::DispatchQueueWriteBuffer::custom(
            QueueWriteBuffer {
                bytes: vec![0u8; size.get() as usize],
            },
        ))
    }

    fn validate_write_buffer(
        &self,
        buffer: &wgpu::custom::DispatchBuffer,
        offset: BufferAddress,
        size: BufferSize,
    ) -> Option<()> {
        let b = buffer
            .as_custom::<BufferData>()
            .expect("akuma backend: foreign buffer");
        (offset + size.get() <= b.size).then_some(())
    }

    fn write_staging_buffer(
        &self,
        buffer: &wgpu::custom::DispatchBuffer,
        offset: BufferAddress,
        staging_buffer: &wgpu::custom::DispatchQueueWriteBuffer,
    ) {
        let staging: &[u8] = {
            let s = staging_buffer
                .as_custom::<QueueWriteBuffer>()
                .expect("akuma backend: foreign staging buffer");
            &s.bytes
        };
        let b = buffer
            .as_custom::<BufferData>()
            .expect("akuma backend: foreign buffer");
        let mut bytes = b.bytes.lock().unwrap();
        let end = offset as usize + staging.len();
        assert!(end <= bytes.len(), "write_staging_buffer out of range");
        bytes[offset as usize..end].copy_from_slice(staging);
    }

    fn write_texture(
        &self,
        texture: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        data_layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        assert_eq!(
            size.depth_or_array_layers, 1,
            "akuma backend: layered write_texture unused"
        );
        let tex = texture
            .texture
            .as_custom::<TextureData>()
            .expect("akuma backend: foreign texture");
        let bpt = tex.bytes_per_texel();
        let dst_row_pitch = tex.size.width * bpt;
        let src_pitch = data_layout
            .bytes_per_row
            .unwrap_or(size.width * bpt) as usize;
        let mut store = match &*tex.store {
            TexStore::Color(c) => c.lock().unwrap(),
            TexStore::Depth(_) => panic!("write_texture into a depth texture"),
        };
        for y in 0..size.height {
            let src = data_layout.offset as usize + y as usize * src_pitch;
            let dst = (texture.origin.y + y) as usize * dst_row_pitch as usize
                + (texture.origin.x * bpt) as usize;
            let count = (size.width * bpt) as usize;
            store[dst..dst + count].copy_from_slice(&data[src..src + count]);
        }
    }

    fn submit(&self, command_buffers: &mut dyn Iterator<Item = wgpu::custom::DispatchCommandBuffer>) -> u64 {
        let mut n: u64 = 0;
        for cb in command_buffers {
            let cb = cb
                .as_custom::<CommandBuffer>()
                .expect("akuma backend: foreign command buffer");
            let cmds: Vec<Cmd> = std::mem::take(&mut *cb.shared.cmds.lock().unwrap());
            for cmd in cmds {
                execute(cmd);
            }
            n += 1;
        }
        n
    }

    fn get_timestamp_period(&self) -> f32 {
        1.0
    }

    fn on_submitted_work_done(&self, cb: wgpu::custom::BoxSubmittedWorkDoneCallback) {
        // work completes synchronously in submit; honor the callback at once
        cb();
    }

    fn compact_blas(
        &self,
        _blas: &wgpu::custom::DispatchBlas,
    ) -> (Option<u64>, wgpu::custom::DispatchBlas) {
        panic!("akuma backend: acceleration structures unsupported");
    }

    fn present(&self, _detail: &wgpu::custom::DispatchSurfaceOutputDetail) {
        panic!("akuma backend: no surfaces");
    }
}

/// Execute one recorded command. This is the "GPU".
fn execute(cmd: Cmd) {
    match cmd {
        Cmd::CopyBufferToBuffer {
            src_bytes,
            src_offset,
            src_size,
            dst_bytes,
            dst_offset,
            size,
        } => {
            let n = size.unwrap_or(src_size - src_offset) as usize;
            let s = src_bytes.lock().unwrap();
            let mut d = dst_bytes.lock().unwrap();
            d[dst_offset as usize..dst_offset as usize + n]
                .copy_from_slice(&s[src_offset as usize..src_offset as usize + n]);
        }
        Cmd::CopyTextureToBuffer {
            src_store,
            src_width,
            src_bpt,
            src_origin,
            dst_bytes,
            layout,
            size,
        } => {
            assert_eq!(size.depth_or_array_layers, 1);
            let bpt = src_bpt as usize;
            let count = size.width as usize * bpt;
            let dst_pitch = layout.bytes_per_row.map(|p| p as usize).unwrap_or(count);
            let mut d = dst_bytes.lock().unwrap();
            match &*src_store {
                TexStore::Color(c) => {
                    let s = c.lock().unwrap();
                    for y in 0..size.height as usize {
                        let so = ((src_origin.y as usize + y) * src_width as usize
                            + src_origin.x as usize)
                            * bpt;
                        let doff = layout.offset as usize + y * dst_pitch;
                        d[doff..doff + count].copy_from_slice(&s[so..so + count]);
                    }
                }
                TexStore::Depth(_) => panic!("copy depth texture to buffer unused"),
            }
        }
        Cmd::CopyBufferToTexture {
            src_bytes,
            layout,
            dst_store,
            dst_width,
            dst_bpt,
            dst_origin,
            size,
        } => {
            assert_eq!(size.depth_or_array_layers, 1);
            let bpt = dst_bpt as usize;
            let count = size.width as usize * bpt;
            let src_pitch = layout.bytes_per_row.map(|p| p as usize).unwrap_or(count);
            let s = src_bytes.lock().unwrap();
            match &*dst_store {
                TexStore::Color(c) => {
                    let mut d = c.lock().unwrap();
                    for y in 0..size.height as usize {
                        let so = layout.offset as usize + y * src_pitch;
                        let doff = ((dst_origin.y as usize + y) * dst_width as usize
                            + dst_origin.x as usize)
                            * bpt;
                        d[doff..doff + count].copy_from_slice(&s[so..so + count]);
                    }
                }
                TexStore::Depth(_) => panic!("copy buffer to depth texture unused"),
            }
        }
        Cmd::CopyTextureToTexture {
            src_store,
            src_width,
            src_origin,
            dst_store,
            dst_width,
            dst_origin,
            bpt,
            size,
        } => {
            assert_eq!(size.depth_or_array_layers, 1);
            let bpt = bpt as usize;
            let count = size.width as usize * bpt;
            // stage through a temporary so a copy within one texture (or
            // overlapping regions) behaves like a memmove
            let mut tmp = vec![0u8; count * size.height as usize];
            match &*src_store {
                TexStore::Color(c) => {
                    let s = c.lock().unwrap();
                    for y in 0..size.height as usize {
                        let so = ((src_origin.y as usize + y) * src_width as usize
                            + src_origin.x as usize)
                            * bpt;
                        tmp[y * count..(y + 1) * count].copy_from_slice(&s[so..so + count]);
                    }
                }
                TexStore::Depth(_) => panic!("depth texture copies unused"),
            }
            match &*dst_store {
                TexStore::Color(c) => {
                    let mut d = c.lock().unwrap();
                    for y in 0..size.height as usize {
                        let doff = ((dst_origin.y as usize + y) * dst_width as usize
                            + dst_origin.x as usize)
                            * bpt;
                        d[doff..doff + count].copy_from_slice(&tmp[y * count..(y + 1) * count]);
                    }
                }
                TexStore::Depth(_) => panic!("depth texture copies unused"),
            }
        }
        Cmd::ClearBuffer {
            bytes,
            buf_size,
            offset,
            size,
        } => {
            let n = size.unwrap_or(buf_size - offset) as usize;
            let mut b = bytes.lock().unwrap();
            b[offset as usize..offset as usize + n].fill(0);
        }
        Cmd::Render { data } => execute_render(data),
    }
}

// ---------------------------------------------------------------------------
// The fixed-function rasterizer — the specification is softrender's
// raster_tri; every line below has a twin there.
// ---------------------------------------------------------------------------

/// x, y, z of one projected vertex (softrender's `Vtx`)
#[derive(Clone, Copy)]
struct Vtx {
    x: f32,
    y: f32,
    z: f32,
}

fn execute_render(data: RenderPassData) {
    let (color_tex, do_clear, clear) = match &data.color {
        Some((tex, do_clear, clear)) => (tex.clone(), *do_clear, *clear),
        None => panic!("akuma backend: render pass without color attachment"),
    };

    // color clear: the demo's raw Rgba8Uint target takes the components as
    // bytes; every other format goes through the format encoder (so sRGB
    // targets get the linear clear color encoded properly)
    if do_clear {
        let mut c = match &*color_tex.store {
            TexStore::Color(c) => c.lock().unwrap(),
            TexStore::Depth(_) => unreachable!(),
        };
        if super::format::is_legacy_raw(color_tex.format) {
            for px in c.chunks_exact_mut(4) {
                px[0] = clear.r as u8;
                px[1] = clear.g as u8;
                px[2] = clear.b as u8;
                px[3] = clear.a as u8;
            }
        } else {
            let bpt = color_tex.bytes_per_texel() as usize;
            let mut one = [0u8; 4];
            super::format::encode(
                color_tex.format,
                [clear.r as f32, clear.g as f32, clear.b as f32, clear.a as f32],
                &mut one,
            );
            fill_pattern(&mut c, &one[..bpt]);
        }
    }
    let mut depth: Option<Arc<TextureData>> = None;
    if let Some((dtex, clear)) = &data.depth {
        if let Some(v) = clear {
            let mut z = match &*dtex.store {
                TexStore::Depth(d) => d.lock().unwrap(),
                TexStore::Color(_) => unreachable!(),
            };
            z.fill(*v);
        }
        depth = Some(dtex.clone());
    }

    let (w, h) = (color_tex.size.width as usize, color_tex.size.height as usize);

    let mut pipe: Option<RenderPipelineData> = None;
    let mut groups: Vec<Option<BindGroupData>> = Vec::new();
    let mut vbufs: Vec<Option<BufBind>> = Vec::new();
    let mut ibuf: Option<(BufBind, wgpu::IndexFormat)> = None;
    let mut viewport: Option<super::raster::Viewport> = None;
    let mut scissor: Option<[u32; 4]> = None;
    let mut blend_const = [0.0f32; 4];

    for cmd in data.cmds {
        match cmd {
            PassCmd::SetPipeline(p) => pipe = Some(p),
            PassCmd::SetBindGroup(i, bg) => {
                if groups.len() <= i as usize {
                    groups.resize(i as usize + 1, None);
                }
                groups[i as usize] = Some(bg);
            }
            PassCmd::SetVertexBuffer { slot, buf } => {
                if vbufs.len() <= slot as usize {
                    vbufs.resize(slot as usize + 1, None);
                }
                vbufs[slot as usize] = buf;
            }
            PassCmd::SetIndexBuffer { buf, format } => ibuf = Some((buf, format)),
            PassCmd::SetViewport(v) => viewport = Some(v),
            PassCmd::SetScissor(s) => scissor = Some(s),
            PassCmd::SetBlendConstant(c) => {
                blend_const = [c.r as f32, c.g as f32, c.b as f32, c.a as f32]
            }
            PassCmd::Draw { vertices, instances } => {
                let pipe = pipe.as_ref().expect("draw without a pipeline");
                if pipe.legacy() {
                    draw_legacy(pipe, &groups, &color_tex, depth.as_ref(), w, h, vertices, instances);
                } else {
                    let st = StdDrawState {
                        pipe,
                        groups: &groups,
                        vbufs: &vbufs,
                        ibuf: None,
                        color: &color_tex,
                        depth: depth.as_ref(),
                        viewport,
                        scissor,
                        blend_const,
                    };
                    draw_standard(&st, StdDraw::Direct { vertices, instances });
                }
            }
            PassCmd::DrawIndexed { indices, base_vertex, instances } => {
                let pipe = pipe.as_ref().expect("draw without a pipeline");
                assert!(!pipe.legacy(), "akuma backend: indexed draws need a standard-mode target");
                let st = StdDrawState {
                    pipe,
                    groups: &groups,
                    vbufs: &vbufs,
                    ibuf: ibuf.as_ref().map(|(b, f)| (b, *f)),
                    color: &color_tex,
                    depth: depth.as_ref(),
                    viewport,
                    scissor,
                    blend_const,
                };
                draw_standard(&st, StdDraw::Indexed { indices, base_vertex, instances });
            }
        }
    }
}

/// Fill `buf` with the repeating `pat` at memset/memcpy speed: write the
/// pattern once, then double the filled prefix with `copy_within`.
fn fill_pattern(buf: &mut [u8], pat: &[u8]) {
    if buf.is_empty() {
        return;
    }
    if pat.len() == 1 {
        buf.fill(pat[0]);
        return;
    }
    let n0 = pat.len().min(buf.len());
    buf[..n0].copy_from_slice(&pat[..n0]);
    let mut n = n0;
    while n < buf.len() {
        let m = n.min(buf.len() - n);
        buf.copy_within(0..m, n);
        n += m;
    }
}

/// The demo's draw: raw Rgba8Uint target, pixel-space positions, flat
/// varyings, the softrender-contract scanline walk (`raster_tri`).
#[allow(clippy::too_many_arguments)]
fn draw_legacy(
    pipe: &RenderPipelineData,
    groups: &[Option<BindGroupData>],
    color_tex: &Arc<TextureData>,
    depth: Option<&Arc<TextureData>>,
    w: usize,
    h: usize,
    vertices: std::ops::Range<u32>,
    instances: std::ops::Range<u32>,
) {
    let depth_st = pipe
        .depth
        .as_ref()
        .expect("akuma backend: draws require a depth-stencil state");
    let ztex = depth.expect("akuma backend: draws require a depth attachment");
    let guards = lock_buffers(groups);
    let res = build_resources(&guards);
    let mut zbuf = match &*ztex.store {
        TexStore::Depth(d) => d.lock().unwrap(),
        TexStore::Color(_) => unreachable!(),
    };
    let mut color = match &*color_tex.store {
        TexStore::Color(c) => c.lock().unwrap(),
        TexStore::Depth(_) => unreachable!(),
    };
    let mut vs_inv = pipe.vs.stage.begin(&res);
    let mut fs_inv = pipe
        .fs
        .as_ref()
        .expect("akuma backend: draw needs a fragment stage")
        .stage
        .begin(&res);
    for inst in instances {
        // vertex stage: one invocation per corner
        let tv = crate::clock::monotonic();
        let mut verts = Vec::with_capacity((vertices.end - vertices.start) as usize);
        for vi in vertices.clone() {
            verts.push(vs_inv.run_vertex(vi, inst, &[[0u32; 4]; super::exec::MAX_LOC]));
        }
        super::prof::add_ns(4, crate::clock::monotonic() - tv);
        super::prof::inc(8, verts.len() as u64);
        let tr = crate::clock::monotonic();
        for tri in verts.chunks_exact(3) {
            raster_tri(tri, &mut fs_inv, depth_st, &mut zbuf, &mut color, w, h);
        }
        // raster total minus the fragment time raster_tri booked
        super::prof::add_ns(5, crate::clock::monotonic() - tr);
    }
}

// ---------------------------------------------------------------------------
// Standard-mode draws (real wgpu semantics): vertex fetch, primitive
// assembly, `raster.rs`
// ---------------------------------------------------------------------------

struct StdDrawState<'a> {
    pipe: &'a RenderPipelineData,
    groups: &'a [Option<BindGroupData>],
    vbufs: &'a [Option<BufBind>],
    ibuf: Option<(&'a BufBind, wgpu::IndexFormat)>,
    color: &'a Arc<TextureData>,
    depth: Option<&'a Arc<TextureData>>,
    viewport: Option<super::raster::Viewport>,
    scissor: Option<[u32; 4]>,
    blend_const: [f32; 4],
}

enum StdDraw {
    Direct { vertices: std::ops::Range<u32>, instances: std::ops::Range<u32> },
    Indexed { indices: std::ops::Range<u32>, base_vertex: i32, instances: std::ops::Range<u32> },
}

/// Collects buffer locks without ever locking the same mutex twice.
struct Locks<'a> {
    guards: Vec<std::sync::MutexGuard<'a, Vec<u8>>>,
    seen: Vec<*const Mutex<Vec<u8>>>,
}

impl<'a> Locks<'a> {
    fn new() -> Self {
        Locks { guards: Vec::new(), seen: Vec::new() }
    }
    fn lock(&mut self, a: &'a Arc<Mutex<Vec<u8>>>) -> usize {
        let p = Arc::as_ptr(a);
        if let Some(i) = self.seen.iter().position(|&q| q == p) {
            return i;
        }
        self.seen.push(p);
        self.guards.push(a.lock().unwrap());
        self.guards.len() - 1
    }
}

fn draw_standard(st: &StdDrawState<'_>, what: StdDraw) {
    use wgpu::PrimitiveTopology as T;
    let pipe = st.pipe;
    let target = pipe.target.as_ref().expect("standard draw needs a color target");
    let fs_stage = &pipe
        .fs
        .as_ref()
        .expect("akuma backend: draw needs a fragment stage")
        .stage;

    // lock every buffer once
    let mut locks = Locks::new();
    let mut bind_idx: Vec<(u32, u32, usize)> = Vec::new();
    // bound textures: lock each distinct store once, keep the guards for the draw
    let mut tex_guards: Vec<std::sync::MutexGuard<'_, Vec<u8>>> = Vec::new();
    let mut tex_seen: Vec<*const TexStore> = Vec::new();
    let mut tex_binds: Vec<(u32, u32, usize, &TextureData)> = Vec::new();
    let mut smp_binds: Vec<(u32, u32, super::texture::SmpRef)> = Vec::new();
    for (gi, g) in st.groups.iter().enumerate() {
        let g = g.as_ref().expect("bind group not set");
        for (binding, r) in &g.entries {
            match r {
                BindRes::Buffer(bytes) => bind_idx.push((gi as u32, *binding, locks.lock(bytes))),
                BindRes::Texture(view) => {
                    assert!(
                        !Arc::ptr_eq(&view.tex.store, &st.color.store),
                        "akuma backend: a texture bound for sampling is also the render target"
                    );
                    let p = Arc::as_ptr(&view.tex.store);
                    let at = match tex_seen.iter().position(|&q| q == p) {
                        Some(i) => i,
                        None => {
                            let TexStore::Color(c) = &*view.tex.store else {
                                panic!("akuma backend: depth textures cannot be sampled yet")
                            };
                            tex_seen.push(p);
                            tex_guards.push(c.lock().unwrap());
                            tex_guards.len() - 1
                        }
                    };
                    tex_binds.push((gi as u32, *binding, at, &view.tex));
                }
                BindRes::Sampler(d) => smp_binds.push((gi as u32, *binding, *d)),
            }
        }
    }
    let vb_idx: Vec<Option<usize>> = st
        .vbufs
        .iter()
        .map(|b| b.as_ref().map(|b| locks.lock(&b.bytes)))
        .collect();
    let ib_idx = st.ibuf.map(|(b, _)| locks.lock(&b.bytes));

    let mut res = Resources::default();
    for &(g, b, at) in &bind_idx {
        res = res.with_buffer(g, b, &locks.guards[at][..]);
    }
    for &(g, b, at, t) in &tex_binds {
        let bytes = &tex_guards[at];
        res.texs.insert(
            (g, b),
            super::texture::TexRef {
                data: bytes.as_ptr(),
                len: bytes.len(),
                w: t.size.width,
                h: t.size.height,
                format: t.format,
            },
        );
    }
    for &(g, b, d) in &smp_binds {
        res.smps.insert((g, b), d);
    }
    let (cw, ch) = (st.color.size.width, st.color.size.height);
    let mut color_guard = match &*st.color.store {
        TexStore::Color(c) => c.lock().unwrap(),
        TexStore::Depth(_) => unreachable!(),
    };
    let mut depth_guard = st.depth.map(|d| match &*d.store {
        TexStore::Depth(z) => z.lock().unwrap(),
        TexStore::Color(_) => unreachable!(),
    });
    let depth_state = pipe.depth.as_ref();

    let (instances, vertex_ids): (std::ops::Range<u32>, Vec<i64>) = match &what {
        StdDraw::Direct { vertices, instances } => {
            (instances.clone(), vertices.clone().map(|v| v as i64).collect())
        }
        StdDraw::Indexed { indices, base_vertex, instances } => {
            let (b, fmt) = st.ibuf.expect("indexed draw without an index buffer");
            let bytes = &locks.guards[ib_idx.unwrap()][..];
            let ids = indices
                .clone()
                .map(|i| {
                    let off = b.offset as usize
                        + i as usize * if fmt == wgpu::IndexFormat::Uint16 { 2 } else { 4 };
                    let v = match fmt {
                        wgpu::IndexFormat::Uint16 => bytes
                            .get(off..off + 2)
                            .map_or(0, |s| u16::from_le_bytes([s[0], s[1]]) as i64),
                        wgpu::IndexFormat::Uint32 => bytes
                            .get(off..off + 4)
                            .map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as i64),
                    };
                    v + *base_vertex as i64
                })
                .collect();
            (instances.clone(), ids)
        }
    };

    // ---- vertex stage (batched) + primitive assembly ----
    let mut ids: Vec<(u32, u32)> = Vec::with_capacity(instances.len() * vertex_ids.len());
    let mut attrs: Vec<super::exec::Varyings> = Vec::with_capacity(ids.capacity());
    for inst in instances.clone() {
        for &vid in &vertex_ids {
            ids.push((vid as u32, inst));
            attrs.push(fetch_attrs(pipe, st.vbufs, &vb_idx, &locks, vid, inst));
        }
    }
    let mut verts: Vec<super::exec::RawVertex> = Vec::with_capacity(ids.len());
    let vplan = pipe.vs.stage.plan(&res, ids.len() as u64);
    let mut vs_inv = pipe.vs.stage.begin_with(&res, &vplan);
    vs_inv.run_vertex_batch(&ids, &attrs, &mut verts);
    // (a, b, c, provoking) as indices into `verts`
    let mut prims: Vec<[u32; 4]> = Vec::new();
    for inst_i in 0..instances.len() as u32 {
        let base = inst_i * vertex_ids.len() as u32;
        let n = vertex_ids.len() as u32;
        match pipe.topology {
            T::TriangleList => {
                for t in 0..n / 3 {
                    let a = base + t * 3;
                    prims.push([a, a + 1, a + 2, a]);
                }
            }
            T::TriangleStrip => {
                for i in 0..n.saturating_sub(2) {
                    let a = base + i;
                    // odd triangles are reversed to keep the winding; the
                    // provoking vertex stays the primitive's first
                    if i % 2 == 0 {
                        prims.push([a, a + 1, a + 2, a]);
                    } else {
                        prims.push([a + 1, a, a + 2, a]);
                    }
                }
            }
            other => panic!("akuma backend: topology {other:?} unsupported"),
        }
    }

    // ---- raster: row bands, in parallel when the draw is big enough ----
    let viewport = st.viewport.unwrap_or(super::raster::Viewport {
        x: 0.0,
        y: 0.0,
        w: cw as f32,
        h: ch as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    });
    let scissor = st.scissor.unwrap_or([0, 0, cw, ch]);
    let est_frags = estimate_fragments(&prims, &verts, &viewport, cw, ch);
    let fplan = fs_stage.plan(&res, est_frags);
    let mut fs_inv = fs_stage.begin_with(&res, &fplan);
    let interp = fs_stage.frag_interp();
    let bpt = super::format::bytes_per_texel(target.format).unwrap() as usize;
    let depth_cfg = depth_state.map(|ds| {
        (
            ds.depth_compare.unwrap_or(wgpu::CompareFunction::Always),
            ds.depth_write_enabled.unwrap_or(false),
        )
    });
    let threads = raster_threads(&prims, cw, ch);
    // many more bands than threads: bands cost very different amounts
    let n_bands = if threads > 1 { (threads * 8).min(ch as usize).max(1) } else { 1 };
    let rows_per_band = (ch as usize).div_ceil(n_bands);

    struct Band<'b> {
        y0: i64,
        y1: i64,
        color: &'b mut [u8],
        depth: Option<&'b mut [f32]>,
    }
    let color_all: &mut [u8] = &mut color_guard[..];
    let mut depth_bands: Vec<Option<&mut [f32]>> = match (&mut depth_guard, depth_cfg) {
        (Some(z), Some(_)) => z[..].chunks_mut(rows_per_band * cw as usize).map(Some).collect(),
        _ => (0..n_bands).map(|_| None).collect(),
    };
    depth_bands.reverse();
    let mut bands: Vec<Band<'_>> = Vec::with_capacity(n_bands);
    for (bi, c) in color_all.chunks_mut(rows_per_band * cw as usize * bpt).enumerate() {
        let y0 = (bi * rows_per_band) as i64;
        bands.push(Band {
            y0,
            y1: (y0 + (c.len() / (cw as usize * bpt)) as i64),
            color: c,
            depth: depth_bands.pop().flatten(),
        });
    }

    let run_band = |band: Band<'_>, fs: &mut Invoker<'_>| {
        let mut raster = super::raster::Raster {
            color: super::raster::ColorTarget {
                format: target.format,
                data: band.color,
                width: cw,
                height: ch,
                blend: target.blend,
                write_mask: target.write_mask,
            },
            depth: match (band.depth, depth_cfg) {
                (Some(z), Some((compare, write))) => {
                    Some(super::raster::DepthTarget { data: z, compare, write })
                }
                _ => None,
            },
            viewport,
            scissor,
            cull: pipe.cull,
            front: pipe.front,
            blend_constant: st.blend_const,
            interp: interp.clone(),
            band: (band.y0, band.y1),
            row0: band.y0,
        };
        for p in &prims {
            raster.triangle(
                fs,
                [&verts[p[0] as usize], &verts[p[1] as usize], &verts[p[2] as usize]],
                &verts[p[3] as usize].varyings,
            );
        }
    };

    if threads <= 1 {
        for band in bands {
            run_band(band, &mut fs_inv);
        }
    } else {
        let queue = Mutex::new(bands);
        let res_ref = &res;
        let fplan_ref = &fplan;
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| {
                    let mut fs = fs_stage.begin_with(res_ref, fplan_ref);
                    loop {
                        let next = queue.lock().unwrap().pop();
                        match next {
                            Some(b) => run_band(b, &mut fs),
                            None => break,
                        }
                    }
                });
            }
        });
    }
}

/// Rough upper bound on the fragments of a draw: the summed screen bounding
/// boxes of its primitives, clipped to the target. Decides whether a stage is
/// worth specializing; precision does not matter.
fn estimate_fragments(
    prims: &[[u32; 4]],
    verts: &[super::exec::RawVertex],
    vp: &super::raster::Viewport,
    cw: u32,
    ch: u32,
) -> u64 {
    let full = cw as u64 * ch as u64;
    let mut total = 0u64;
    for p in prims {
        let mut x0 = f32::INFINITY;
        let mut x1 = f32::NEG_INFINITY;
        let mut y0 = f32::INFINITY;
        let mut y1 = f32::NEG_INFINITY;
        let mut behind = false;
        for &i in &p[..3] {
            let [x, y, _, w] = verts[i as usize].position;
            if !(w > 1e-9) {
                behind = true;
                break;
            }
            let sx = vp.x + (x / w * 0.5 + 0.5) * vp.w;
            let sy = vp.y + (0.5 - y / w * 0.5) * vp.h;
            x0 = x0.min(sx);
            x1 = x1.max(sx);
            y0 = y0.min(sy);
            y1 = y1.max(sy);
        }
        if behind {
            total += full;
        } else {
            let w = (x1.min(cw as f32) - x0.max(0.0)).max(0.0);
            let h = (y1.min(ch as f32) - y0.max(0.0)).max(0.0);
            total += (w * h) as u64;
        }
        if total >= 1 << 40 {
            break;
        }
    }
    total
}

/// Worker count for a draw: 1 unless the draw is big enough to repay thread
/// start-up (`AKUMA_THREADS=n` overrides; default = available parallelism).
fn raster_threads(prims: &[[u32; 4]], w: u32, h: u32) -> usize {
    let want = std::env::var("AKUMA_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .clamp(1, 8);
    // threads cost ~100 us each to start; a draw of a handful of small
    // triangles is not worth it. Coarse test: enough primitives, or a big target.
    if want > 1 && (prims.len() >= 64 || (prims.len() >= 1 && (w as u64 * h as u64) >= 1_000_000 && prims.len() <= 4)) {
        want
    } else {
        1
    }
}

/// One vertex's attributes from the bound vertex buffers, by location.
fn fetch_attrs(
    pipe: &RenderPipelineData,
    vbufs: &[Option<BufBind>],
    vb_idx: &[Option<usize>],
    locks: &Locks<'_>,
    vertex_id: i64,
    instance: u32,
) -> super::exec::Varyings {
    let mut out = [[0u32; 4]; super::exec::MAX_LOC];
    for (slot, layout) in pipe.vbufs.iter().enumerate() {
        let (Some(Some(bind)), Some(Some(gi))) = (vbufs.get(slot), vb_idx.get(slot)) else {
            continue;
        };
        let bytes = &locks.guards[*gi][..];
        let index = match layout.step {
            wgpu::VertexStepMode::Vertex => vertex_id.max(0) as u64,
            wgpu::VertexStepMode::Instance => instance as u64,
        };
        let base = bind.offset + index * layout.stride;
        for a in &layout.attrs {
            out[a.location as usize] = super::vertex::fetch(a.format, bytes, (base + a.offset) as usize);
        }
    }
    out
}

/// Lock every buffer the bound groups reference, once per draw. The same
/// buffer bound at two slots is locked once (a second `lock()` on the same
/// mutex would deadlock). Returns (group, binding, guard index) + the guards.
type Locked<'a> = (Vec<(u32, u32, usize)>, Vec<std::sync::MutexGuard<'a, Vec<u8>>>);

fn lock_buffers(groups: &[Option<BindGroupData>]) -> Locked<'_> {
    let mut guards = Vec::new();
    let mut seen: Vec<*const Mutex<Vec<u8>>> = Vec::new();
    let mut idx = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        let g = g.as_ref().expect("bind group not set");
        for (binding, r) in &g.entries {
            match r {
                BindRes::Buffer(bytes) => {
                    let p = Arc::as_ptr(bytes);
                    let at = match seen.iter().position(|&q| q == p) {
                        Some(i) => i,
                        None => {
                            seen.push(p);
                            guards.push(bytes.lock().unwrap());
                            guards.len() - 1
                        }
                    };
                    idx.push((gi as u32, *binding, at));
                }
                BindRes::Texture(_) | BindRes::Sampler(_) => {
                    panic!("akuma backend: texture/sampler bindings unsupported in the legacy path")
                }
            }
        }
    }
    (idx, guards)
}

fn build_resources<'a>(locked: &'a Locked<'_>) -> Resources<'a> {
    let (idx, guards) = locked;
    let mut res = Resources::default();
    for &(g, b, at) in idx {
        res = res.with_buffer(g, b, &guards[at][..]);
    }
    res
}

/// softrender's edge_xz, verbatim.
#[inline]
fn edge_xz(p: Vtx, q: Vtx, yy: f32) -> (f32, f32) {
    let dy = q.y - p.y;
    if dy.abs() < 1e-6 {
        return (p.x, p.z);
    }
    let s = ((yy - p.y) / dy).clamp(0.0, 1.0);
    (p.x + (q.x - p.x) * s, p.z + (q.z - p.z) * s)
}

/// softrender's raster_tri with shader-produced vertices. The flat color
/// comes from the provoking (first) vertex; the fragment stage runs per
/// covered pixel and hands back the exact target bytes.
fn raster_tri(
    v: &[RawVertex],
    fs: &mut Invoker<'_>,
    depth_st: &wgpu::DepthStencilState,
    zbuf: &mut [f32],
    color: &mut [u8],
    w: usize,
    h: usize,
) {
    debug_assert!(matches!(
        depth_st.depth_compare,
        Some(wgpu::CompareFunction::Less)
    ));
    let pos = [v[0].position, v[1].position, v[2].position];
    let p0 = Vtx { x: pos[0][0], y: pos[0][1], z: pos[0][2] };
    let p1 = Vtx { x: pos[1][0], y: pos[1][1], z: pos[1][2] };
    let p2 = Vtx { x: pos[2][0], y: pos[2][1], z: pos[2][2] };

    // Signed area in y-down screen space; CCW-from-outside triangles project
    // to negative area. Cull everything else, including degenerates.
    let area = (p1.x - p0.x).mul_add(p2.y - p0.y, -((p1.y - p0.y) * (p2.x - p0.x)));
    if area >= -0.01 {
        return;
    }

    // provoking vertex = first corner: the flat varying source
    let provoking: &Varyings = &v[0].varyings;
    // per-fragment timers cost a syscall each on this kernel: only when asked
    let profiling = super::prof::level() >= 2;
    // a stage that depends on neither position nor non-flat varyings gives
    // the same answer for every pixel of this triangle: run it once
    let constant_fs = fs.constant_per_primitive();
    let mut cached: Option<Option<[u32; 4]>> = None;

    // scanline raster with per-pixel z interpolated along the edges
    let (wi, hi) = (w as i64, h as i64);
    let mut p = [p0, p1, p2];
    p.sort_by(|a, b| a.y.total_cmp(&b.y));

    let y0 = (p[0].y.ceil() as i64).max(0);
    let y1 = (p[2].y.ceil() as i64).min(hi);
    for y in y0..y1 {
        let yy = y as f32 + 0.5;
        if yy < p[0].y || yy >= p[2].y {
            continue;
        }
        // long edge p0->p2; short edge is p0->p1 above p1.y, else p1->p2
        let (xl, zl) = edge_xz(p[0], p[2], yy);
        let (xs, zs) = if yy < p[1].y {
            edge_xz(p[0], p[1], yy)
        } else {
            edge_xz(p[1], p[2], yy)
        };
        let (xa, za, xb, zb) = if xl <= xs {
            (xl, zl, xs, zs)
        } else {
            (xs, zs, xl, zl)
        };
        let x0 = (xa.ceil() as i64).max(0);
        let x1 = (xb.ceil() as i64).min(wi);
        if x0 >= x1 {
            continue;
        }
        let row = y as usize * w;
        let dx = xb - xa;
        let span_inv = if dx.abs() > 1e-6 { 1.0 / dx } else { 0.0 };
        for x in x0..x1 {
            let ts = ((x as f32 + 0.5) - xa) * span_inv;
            let z = za + (zb - za) * ts;
            let zi = &mut zbuf[row + x as usize];
            if z < *zi {
                *zi = z;
                // fragment stage for this pixel; it returns the four raw
                // target-component values
                let tf = if profiling { crate::clock::monotonic() } else { 0.0 };
                let frag = match (constant_fs, cached) {
                    (true, Some(c)) => c,
                    _ => {
                        let c = fs.run_fragment(provoking, [x as f32 + 0.5, yy, z, 1.0]);
                        cached = Some(c);
                        c
                    }
                };
                if profiling {
                    super::prof::add_ns(6, crate::clock::monotonic() - tf);
                    super::prof::inc(9, 1);
                }
                if let Some(px) = frag {
                    let o = (row + x as usize) * 4;
                    color[o..o + 4].copy_from_slice(&[
                        px[0] as u8,
                        px[1] as u8,
                        px[2] as u8,
                        px[3] as u8,
                    ]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Command encoder + render pass (recording into a shared Vec; submit runs it)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Encoder {
    pub shared: Arc<CommandBufferShared>,
}

impl wgpu::custom::CommandEncoderInterface for Encoder {
    fn copy_buffer_to_buffer(
        &self,
        source: &wgpu::custom::DispatchBuffer,
        source_offset: BufferAddress,
        destination: &wgpu::custom::DispatchBuffer,
        destination_offset: BufferAddress,
        copy_size: Option<BufferAddress>,
    ) {
        let src = source
            .as_custom::<BufferData>()
            .expect("akuma backend: foreign buffer");
        let dst = destination
            .as_custom::<BufferData>()
            .expect("akuma backend: foreign buffer");
        self.shared.cmds.lock().unwrap().push(Cmd::CopyBufferToBuffer {
            src_bytes: Arc::clone(&src.bytes),
            src_offset: source_offset,
            src_size: src.size,
            dst_bytes: Arc::clone(&dst.bytes),
            dst_offset: destination_offset,
            size: copy_size,
        });
    }

    fn copy_buffer_to_texture(
        &self,
        source: wgpu::TexelCopyBufferInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        let src = source
            .buffer
            .as_custom::<BufferData>()
            .expect("akuma backend: foreign buffer");
        let dst = destination
            .texture
            .as_custom::<TextureData>()
            .expect("akuma backend: foreign texture");
        self.shared.cmds.lock().unwrap().push(Cmd::CopyBufferToTexture {
            src_bytes: Arc::clone(&src.bytes),
            layout: source.layout,
            dst_store: Arc::clone(&dst.store),
            dst_width: dst.size.width,
            dst_bpt: dst.bytes_per_texel(),
            dst_origin: destination.origin,
            size: copy_size,
        });
    }

    fn copy_texture_to_buffer(
        &self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyBufferInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        let src = source
            .texture
            .as_custom::<TextureData>()
            .expect("akuma backend: foreign texture");
        let dst = destination
            .buffer
            .as_custom::<BufferData>()
            .expect("akuma backend: foreign buffer");
        self.shared
            .cmds
            .lock()
            .unwrap()
            .push(Cmd::CopyTextureToBuffer {
                src_store: Arc::clone(&src.store),
                src_width: src.size.width,
                src_bpt: src.bytes_per_texel(),
                src_origin: source.origin,
                dst_bytes: Arc::clone(&dst.bytes),
                layout: destination.layout,
                size: copy_size,
            });
    }

    fn copy_texture_to_texture(
        &self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        let src = source
            .texture
            .as_custom::<TextureData>()
            .expect("akuma backend: foreign texture");
        let dst = destination
            .texture
            .as_custom::<TextureData>()
            .expect("akuma backend: foreign texture");
        assert_eq!(src.format, dst.format, "akuma backend: copy between different formats");
        self.shared.cmds.lock().unwrap().push(Cmd::CopyTextureToTexture {
            src_store: Arc::clone(&src.store),
            src_width: src.size.width,
            src_origin: source.origin,
            dst_store: Arc::clone(&dst.store),
            dst_width: dst.size.width,
            dst_origin: destination.origin,
            bpt: src.bytes_per_texel(),
            size: copy_size,
        });
    }

    fn begin_compute_pass(
        &self,
        _desc: &wgpu::ComputePassDescriptor<'_>,
    ) -> wgpu::custom::DispatchComputePass {
        panic!("akuma backend: compute passes unused");
    }

    fn begin_render_pass(
        &self,
        desc: &wgpu::RenderPassDescriptor<'_>,
    ) -> wgpu::custom::DispatchRenderPass {
        let mut color = None;
        let mut depth = None;
        for a in desc.color_attachments.iter().flatten() {
            let view = a
                .view
                .as_custom::<ViewData>()
                .expect("akuma backend: foreign texture view");
            let (do_clear, clear) = match a.ops.load {
                wgpu::LoadOp::Clear(c) => (true, c),
                wgpu::LoadOp::Load => (
                    false,
                    wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    },
                ),
                other => panic!("akuma backend: unsupported color load op {other:?}"),
            };
            color = Some((view.tex.clone(), do_clear, clear));
        }
        if let Some(d) = &desc.depth_stencil_attachment {
            let view = d
                .view
                .as_custom::<ViewData>()
                .expect("akuma backend: foreign texture view");
            let clear = match &d.depth_ops {
                Some(ops) => match ops.load {
                    wgpu::LoadOp::Clear(v) => Some(v),
                    wgpu::LoadOp::Load => None,
                    other => panic!("akuma backend: unsupported depth load op {other:?}"),
                },
                None => None,
            };
            depth = Some((view.tex.clone(), clear));
        }
        let pass = RenderPassRec {
            data: RenderPassData {
                color,
                depth,
                cmds: Vec::new(),
            },
            shared: Arc::clone(&self.shared),
        };
        wgpu::custom::DispatchRenderPass::custom(pass)
    }

    fn finish(&mut self) -> wgpu::custom::DispatchCommandBuffer {
        wgpu::custom::DispatchCommandBuffer::custom(CommandBuffer {
            shared: Arc::clone(&self.shared),
        })
    }

    fn clear_texture(
        &self,
        _texture: &wgpu::custom::DispatchTexture,
        _subresource_range: &wgpu::ImageSubresourceRange,
    ) {
        panic!("akuma backend: clear_texture unused");
    }

    fn clear_buffer(
        &self,
        buffer: &wgpu::custom::DispatchBuffer,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) {
        let buf = buffer
            .as_custom::<BufferData>()
            .expect("akuma backend: foreign buffer");
        self.shared.cmds.lock().unwrap().push(Cmd::ClearBuffer {
            bytes: Arc::clone(&buf.bytes),
            buf_size: buf.size,
            offset,
            size,
        });
    }

    fn insert_debug_marker(&self, _label: &str) {}
    fn push_debug_group(&self, _label: &str) {}
    fn pop_debug_group(&self) {}

    fn write_timestamp(
        &self,
        _query_set: &wgpu::custom::DispatchQuerySet,
        _query_index: u32,
    ) {
    }

    fn resolve_query_set(
        &self,
        _query_set: &wgpu::custom::DispatchQuerySet,
        _first_query: u32,
        _query_count: u32,
        _destination: &wgpu::custom::DispatchBuffer,
        _destination_offset: BufferAddress,
    ) {
        panic!("akuma backend: query sets unused");
    }

    fn mark_acceleration_structures_built<'a>(
        &self,
        _blas: &mut dyn Iterator<Item = &'a wgpu::Blas>,
        _tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>,
    ) {
        panic!("akuma backend: acceleration structures unsupported");
    }

    fn build_acceleration_structures<'a>(
        &self,
        _blas: &mut dyn Iterator<Item = &'a wgpu::BlasBuildEntry<'a>>,
        _tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>,
    ) {
        panic!("akuma backend: acceleration structures unsupported");
    }

    fn transition_resources<'a>(
        &mut self,
        buffer_transitions: &mut dyn Iterator<
            Item = wgpu::wgt::BufferTransition<&'a wgpu::custom::DispatchBuffer>,
        >,
        texture_transitions: &mut dyn Iterator<
            Item = wgpu::wgt::TextureTransition<&'a wgpu::custom::DispatchTexture>,
        >,
    ) {
        // no barriers on a single-threaded immediate GPU; drain as promised
        buffer_transitions.for_each(|_| {});
        texture_transitions.for_each(|_| {});
    }
}

/// The render pass recorder. On Drop it lands in the encoder's command list —
/// exactly when the user's `rpass` binding goes out of scope.
#[derive(Debug)]
pub struct RenderPassRec {
    pub data: RenderPassData,
    pub shared: Arc<CommandBufferShared>,
}

impl Drop for RenderPassRec {
    fn drop(&mut self) {
        let data = std::mem::replace(
            &mut self.data,
            RenderPassData {
                color: None,
                depth: None,
                cmds: Vec::new(),
            },
        );
        self.shared.cmds.lock().unwrap().push(Cmd::Render { data });
    }
}

impl wgpu::custom::RenderPassInterface for RenderPassRec {
    fn set_pipeline(&mut self, pipeline: &wgpu::custom::DispatchRenderPipeline) {
        let p = pipeline
            .as_custom::<RenderPipelineData>()
            .expect("akuma backend: foreign render pipeline")
            .clone();
        self.data.cmds.push(PassCmd::SetPipeline(p));
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&wgpu::custom::DispatchBindGroup>,
        _offsets: &[wgpu::DynamicOffset],
    ) {
        let bg = bind_group
            .expect("akuma backend: unset bind group")
            .as_custom::<BindGroupData>()
            .expect("akuma backend: foreign bind group")
            .clone();
        self.data.cmds.push(PassCmd::SetBindGroup(index, bg));
    }

    fn set_index_buffer(
        &mut self,
        buffer: &wgpu::custom::DispatchBuffer,
        index_format: wgpu::IndexFormat,
        offset: BufferAddress,
        _size: Option<BufferSize>,
    ) {
        let b = buffer
            .as_custom::<BufferData>()
            .expect("akuma backend: foreign buffer");
        self.data.cmds.push(PassCmd::SetIndexBuffer {
            buf: BufBind { bytes: Arc::clone(&b.bytes), offset },
            format: index_format,
        });
    }

    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: Option<&wgpu::custom::DispatchBuffer>,
        offset: BufferAddress,
        _size: Option<BufferSize>,
    ) {
        let buf = buffer.map(|b| {
            let b = b
                .as_custom::<BufferData>()
                .expect("akuma backend: foreign buffer");
            BufBind { bytes: Arc::clone(&b.bytes), offset }
        });
        self.data.cmds.push(PassCmd::SetVertexBuffer { slot, buf });
    }

    fn set_immediates(&mut self, _offset: u32, _data: &[u8]) {
        panic!("akuma backend: immediates unused");
    }

    fn set_blend_constant(&mut self, color: wgpu::Color) {
        self.data.cmds.push(PassCmd::SetBlendConstant(color));
    }

    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.data.cmds.push(PassCmd::SetScissor([x, y, width, height]));
    }

    fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        self.data.cmds.push(PassCmd::SetViewport(super::raster::Viewport {
            x,
            y,
            w: width,
            h: height,
            min_depth,
            max_depth,
        }));
    }

    fn set_stencil_reference(&mut self, _reference: u32) {
        panic!("akuma backend: stencil unused");
    }

    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        self.data
            .cmds
            .push(PassCmd::Draw { vertices, instances });
    }

    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        self.data
            .cmds
            .push(PassCmd::DrawIndexed { indices, base_vertex, instances });
    }

    fn draw_mesh_tasks(&mut self, _x: u32, _y: u32, _z: u32) {
        panic!("akuma backend: mesh pipelines unsupported");
    }

    fn draw_indirect(
        &mut self,
        _indirect_buffer: &wgpu::custom::DispatchBuffer,
        _indirect_offset: BufferAddress,
    ) {
        panic!("akuma backend: indirect draws unused");
    }

    fn draw_indexed_indirect(
        &mut self,
        _indirect_buffer: &wgpu::custom::DispatchBuffer,
        _indirect_offset: BufferAddress,
    ) {
        panic!("akuma backend: indirect draws unused");
    }

    fn draw_mesh_tasks_indirect(
        &mut self,
        _indirect_buffer: &wgpu::custom::DispatchBuffer,
        _indirect_offset: BufferAddress,
    ) {
        panic!("akuma backend: mesh pipelines unsupported");
    }

    fn multi_draw_indirect(
        &mut self,
        _indirect_buffer: &wgpu::custom::DispatchBuffer,
        _indirect_offset: BufferAddress,
        _count: u32,
    ) {
        panic!("akuma backend: indirect draws unused");
    }

    fn multi_draw_indexed_indirect(
        &mut self,
        _indirect_buffer: &wgpu::custom::DispatchBuffer,
        _indirect_offset: BufferAddress,
        _count: u32,
    ) {
        panic!("akuma backend: indirect draws unused");
    }

    fn multi_draw_indirect_count(
        &mut self,
        _indirect_buffer: &wgpu::custom::DispatchBuffer,
        _indirect_offset: BufferAddress,
        _count_buffer: &wgpu::custom::DispatchBuffer,
        _count_buffer_offset: BufferAddress,
        _max_count: u32,
    ) {
        panic!("akuma backend: indirect draws unused");
    }

    fn multi_draw_mesh_tasks_indirect_count(
        &mut self,
        _indirect_buffer: &wgpu::custom::DispatchBuffer,
        _indirect_offset: BufferAddress,
        _count_buffer: &wgpu::custom::DispatchBuffer,
        _count_buffer_offset: BufferAddress,
        _max_count: u32,
    ) {
        panic!("akuma backend: mesh pipelines unsupported");
    }

    fn multi_draw_indexed_indirect_count(
        &mut self,
        _indirect_buffer: &wgpu::custom::DispatchBuffer,
        _indirect_offset: BufferAddress,
        _count_buffer: &wgpu::custom::DispatchBuffer,
        _count_buffer_offset: BufferAddress,
        _max_count: u32,
    ) {
        panic!("akuma backend: indirect draws unused");
    }

    fn multi_draw_mesh_tasks_indirect(
        &mut self,
        _indirect_buffer: &wgpu::custom::DispatchBuffer,
        _indirect_offset: BufferAddress,
        _count: u32,
    ) {
        panic!("akuma backend: mesh pipelines unsupported");
    }

    fn insert_debug_marker(&mut self, _label: &str) {}
    fn push_debug_group(&mut self, _label: &str) {}
    fn pop_debug_group(&mut self) {}

    fn write_timestamp(
        &mut self,
        _query_set: &wgpu::custom::DispatchQuerySet,
        _query_index: u32,
    ) {
    }

    fn begin_occlusion_query(&mut self, _query_index: u32) {
        panic!("akuma backend: occlusion queries unused");
    }

    fn end_occlusion_query(&mut self) {
        panic!("akuma backend: occlusion queries unused");
    }

    fn begin_pipeline_statistics_query(
        &mut self,
        _query_set: &wgpu::custom::DispatchQuerySet,
        _query_index: u32,
    ) {
        panic!("akuma backend: pipeline statistics unused");
    }

    fn end_pipeline_statistics_query(&mut self) {
        panic!("akuma backend: pipeline statistics unused");
    }

    fn execute_bundles(
        &mut self,
        _render_bundles: &mut dyn Iterator<Item = &wgpu::custom::DispatchRenderBundle>,
    ) {
        panic!("akuma backend: render bundles unsupported");
    }
}

#[derive(Debug)]
pub struct CommandBuffer {
    pub shared: Arc<CommandBufferShared>,
}

impl wgpu::custom::CommandBufferInterface for CommandBuffer {}

#[derive(Debug)]
pub struct QueueWriteBuffer {
    pub bytes: Vec<u8>,
}

impl wgpu::custom::QueueWriteBufferInterface for QueueWriteBuffer {
    fn len(&self) -> usize {
        self.bytes.len()
    }

    unsafe fn write_slice(&mut self) -> wgpu::WriteOnly<'_, [u8]> {
        // WriteOnly::new is unsafe: the caller (wgpu) promises not to read through
        // it; we only hand it our own staging bytes
        unsafe { wgpu::WriteOnly::new(std::ptr::NonNull::from(&mut self.bytes[..])) }
    }
}

#[derive(Debug)]
pub struct BufferMapped {
    /// keeps the buffer storage alive while the range is exposed
    _keep: Arc<Mutex<Vec<u8>>>,
    /// start of the buffer's bytes. Stable: buffers are allocated once at
    /// their full size and never resized, and this backend is
    /// single-threaded with a synchronous "GPU" — nothing writes a buffer
    /// while it is mapped (wgpu's own contract forbids it). Avoids copying
    /// the whole buffer on every map (33 MB per frame at 4K).
    ptr: *const u8,
    len: usize,
    pub offset: BufferAddress,
    pub size: usize,
}

// the pointee is owned by `_keep` and is only ever read through this handle
unsafe impl Send for BufferMapped {}
unsafe impl Sync for BufferMapped {}

impl wgpu::custom::BufferMappedRangeInterface for BufferMapped {
    fn len(&self) -> usize {
        self.size
    }

    unsafe fn read_slice(&self) -> &[u8] {
        let all = unsafe { std::slice::from_raw_parts(self.ptr, self.len) };
        &all[self.offset as usize..self.offset as usize + self.size]
    }

    unsafe fn write_slice(&mut self) -> wgpu::WriteOnly<'_, [u8]> {
        panic!("akuma backend: write mappings unused");
    }
}

impl wgpu::custom::BufferInterface for BufferData {
    fn map_async(
        &self,
        _mode: MapMode,
        _range: std::ops::Range<BufferAddress>,
        callback: wgpu::custom::BufferMapCallback,
    ) {
        // the GPU is synchronous: by map time submit() has finished, so the
        // map always succeeds immediately
        callback(Ok(()));
    }

    fn get_mapped_range(
        &self,
        sub_range: std::ops::Range<BufferAddress>,
    ) -> Result<wgpu::custom::DispatchBufferMappedRange, wgpu::MapRangeError> {
        let (ptr, len) = {
            let g = self.bytes.lock().unwrap();
            (g.as_ptr(), g.len())
        };
        let size = (sub_range.end.min(self.size) - sub_range.start) as usize;
        Ok(wgpu::custom::DispatchBufferMappedRange::custom(
            BufferMapped {
                _keep: Arc::clone(&self.bytes),
                ptr,
                len,
                offset: sub_range.start,
                size,
            },
        ))
    }

    fn unmap(&self) {}

    fn destroy(&self) {}
}

impl wgpu::custom::TextureInterface for TextureData {
    fn create_view(&self, _desc: &wgpu::TextureViewDescriptor<'_>) -> wgpu::custom::DispatchTextureView {
        wgpu::custom::DispatchTextureView::custom(ViewData {
            tex: Arc::new(TextureData {
                size: self.size,
                format: self.format,
                store: Arc::clone(&self.store),
            }),
        })
    }

    fn destroy(&self) {}
}

impl wgpu::custom::TextureViewInterface for ViewData {}

impl wgpu::custom::RenderPipelineInterface for RenderPipelineData {
    fn get_bind_group_layout(&self, index: u32) -> wgpu::custom::DispatchBindGroupLayout {
        let l = self
            .groups
            .get(index as usize)
            .expect("akuma backend: no bind group layout at that index");
        wgpu::custom::DispatchBindGroupLayout::custom(BindGroupLayoutData {
            entries: l.entries.clone(),
        })
    }
}
