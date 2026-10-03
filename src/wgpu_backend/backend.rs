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

use super::interp::{self, Resources, Shader};
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
    fn bytes_per_texel(&self) -> u32 {
        match self.format {
            wgpu::TextureFormat::Rgba8Uint | wgpu::TextureFormat::Depth32Float => 4,
            other => panic!("akuma backend: unsupported texture format {other:?}"),
        }
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
    #[allow(dead_code)]
    Texture(ViewData),
    Sampler,
}

#[derive(Debug, Clone)]
pub struct BindGroupData {
    /// (binding number -> resource), in entry order
    pub entries: Vec<(u32, BindRes)>,
}

#[derive(Clone, Debug)]
pub struct StageData {
    pub shader: Arc<Shader>,
    pub entry: usize,
}

impl StageData {
    fn resolve(shader: Arc<Shader>, entry_point: Option<&str>, what: &str) -> StageData {
        let entry = shader
            .entry(entry_point)
            .unwrap_or_else(|e| panic!("akuma backend: {what} entry: {e}"));
        StageData { shader, entry }
    }
}

#[derive(Clone, Debug)]
pub struct RenderPipelineData {
    pub vs: StageData,
    pub fs: Option<StageData>,
    pub depth: Option<wgpu::DepthStencilState>,
    pub groups: Vec<Arc<BindGroupLayoutData>>,
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
        src_origin: wgpu::Origin3d,
        dst_bytes: Arc<Mutex<Vec<u8>>>,
        layout: wgpu::TexelCopyBufferLayout,
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
    /// (depth texture, clear value)
    pub depth: Option<(Arc<TextureData>, f32)>,
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
                wgpu::BindingResource::Sampler(_) => BindRes::Sampler,
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
                match t.format {
                    wgpu::TextureFormat::Rgba8Uint => {}
                    other => panic!(
                        "akuma backend: only bgra8uint render targets are supported, got {other:?}"
                    ),
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
        wgpu::custom::DispatchRenderPipeline::custom(RenderPipelineData {
            vs,
            fs,
            depth: desc.depth_stencil.clone(),
            groups,
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
        let store = match desc.format {
            wgpu::TextureFormat::Rgba8Uint => {
                TexStore::Color(Mutex::new(vec![0u8; pixels * 4]))
            }
            wgpu::TextureFormat::Depth32Float => TexStore::Depth(Mutex::new(vec![0.0f32; pixels])),
            other => panic!("akuma backend: unsupported texture format {other:?}"),
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
        _desc: &wgpu::SamplerDescriptor<'_>,
    ) -> wgpu::custom::DispatchSampler {
        wgpu::custom::DispatchSampler::custom(SamplerData)
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
pub struct SamplerData;
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
            texture.origin,
            wgpu::Origin3d::ZERO,
            "akuma backend: nonzero write_texture origin unused"
        );
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
            let src = y as usize * src_pitch;
            let dst = y as usize * dst_row_pitch as usize;
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
            src_origin,
            dst_bytes,
            layout,
            size,
        } => {
            assert_eq!(src_origin, wgpu::Origin3d::ZERO);
            assert_eq!(size.depth_or_array_layers, 1);
            let bpt = 4u32;
            let row_pitch = src_width * bpt;
            let dst_pitch = layout.bytes_per_row.unwrap_or(size.width * bpt) as usize;
            let mut d = dst_bytes.lock().unwrap();
            match &*src_store {
                TexStore::Color(c) => {
                    let s = c.lock().unwrap();
                    for y in 0..size.height {
                        let so = (y * row_pitch) as usize;
                        let doff = layout.offset as usize + y as usize * dst_pitch;
                        let count = (size.width * bpt) as usize;
                        d[doff..doff + count].copy_from_slice(&s[so..so + count]);
                    }
                }
                TexStore::Depth(_) => panic!("copy depth texture to buffer unused"),
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
    // clear / load
    if let Some((tex, do_clear, clear)) = &data.color {
        if *do_clear {
            // rgba8uint: the clear color components land on the raw texel
            // components in r,g,b,a order (no conversion — this target is
            // bytes; the demo only ever uses Load anyway)
            let mut c = match &*tex.store {
                TexStore::Color(c) => c.lock().unwrap(),
                TexStore::Depth(_) => unreachable!(),
            };
            for px in c.chunks_exact_mut(4) {
                px[0] = clear.r as u8;
                px[1] = clear.g as u8;
                px[2] = clear.b as u8;
                px[3] = clear.a as u8;
            }
        }
    }
    let mut depth: Option<Arc<TextureData>> = None;
    if let Some((dtex, clear)) = &data.depth {
        let mut z = match &*dtex.store {
            TexStore::Depth(d) => d.lock().unwrap(),
            TexStore::Color(_) => unreachable!(),
        };
        z.fill(*clear);
        depth = Some(dtex.clone());
    }

    let (color_store, w, h) = match &data.color {
        Some((tex, ..)) => (
            tex.store.clone(),
            tex.size.width as usize,
            tex.size.height as usize,
        ),
        None => panic!("akuma backend: render pass without color attachment"),
    };

    let mut pipe: Option<RenderPipelineData> = None;
    let mut groups: Vec<Option<BindGroupData>> = Vec::new();

    for cmd in data.cmds {
        match cmd {
            PassCmd::SetPipeline(p) => pipe = Some(p),
            PassCmd::SetBindGroup(i, bg) => {
                if groups.len() <= i as usize {
                    groups.resize(i as usize + 1, None);
                }
                groups[i as usize] = Some(bg);
            }
            PassCmd::Draw {
                vertices,
                instances,
            } => {
                let pipe = pipe.as_ref().expect("draw without a pipeline");
                let depth_st = pipe
                    .depth
                    .as_ref()
                    .expect("akuma backend: draws require a depth-stencil state");
                let ztex = depth
                    .as_ref()
                    .expect("akuma backend: draws require a depth attachment");
                let res = build_resources(&groups);
                let mut zbuf = match &*ztex.store {
                    TexStore::Depth(d) => d.lock().unwrap(),
                    TexStore::Color(_) => unreachable!(),
                };
                let mut color = match &*color_store {
                    TexStore::Color(c) => c.lock().unwrap(),
                    TexStore::Depth(_) => unreachable!(),
                };
                for inst in instances.clone() {
                    // vertex stage: one interpreter invocation per corner
                    let mut verts = Vec::with_capacity((vertices.end - vertices.start) as usize);
                    for vi in vertices.clone() {
                        verts.push(interp::run_vertex(
                            &pipe.vs.shader,
                            pipe.vs.entry,
                            &res,
                            vi,
                            inst,
                        ));
                    }
                    for tri in verts.chunks_exact(3) {
                        raster_tri(tri, pipe, &res, depth_st, &mut zbuf, &mut color, w, h);
                    }
                }
            }
        }
    }
}

fn build_resources(groups: &[Option<BindGroupData>]) -> Resources {
    let mut res = Resources::default();
    for (gi, g) in groups.iter().enumerate() {
        let g = g.as_ref().expect("bind group not set");
        for (binding, r) in &g.entries {
            match r {
                BindRes::Buffer(bytes) => {
                    res = res.with_buffer(gi as u32, *binding, Arc::clone(bytes));
                }
                BindRes::Texture(_) | BindRes::Sampler => {
                    panic!("akuma backend: texture/sampler bindings unsupported")
                }
            }
        }
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
    v: &[interp::VertexOut],
    pipe: &RenderPipelineData,
    res: &Resources,
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
    let provoking = &v[0].varyings;
    let fs = pipe.fs.as_ref().expect("akuma backend: draw needs a fragment stage");

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
                if let Some(px) = interp::run_fragment(
                    &fs.shader,
                    fs.entry,
                    res,
                    provoking,
                    [x as f32 + 0.5, yy, z, 1.0],
                ) {
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
        _source: wgpu::TexelCopyBufferInfo<'_>,
        _destination: wgpu::TexelCopyTextureInfo<'_>,
        _copy_size: wgpu::Extent3d,
    ) {
        panic!("akuma backend: buffer->texture copy unused");
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
                src_origin: source.origin,
                dst_bytes: Arc::clone(&dst.bytes),
                layout: destination.layout,
                size: copy_size,
            });
    }

    fn copy_texture_to_texture(
        &self,
        _source: wgpu::TexelCopyTextureInfo<'_>,
        _destination: wgpu::TexelCopyTextureInfo<'_>,
        _copy_size: wgpu::Extent3d,
    ) {
        panic!("akuma backend: texture->texture copy unused");
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
                    wgpu::LoadOp::Clear(v) => v,
                    wgpu::LoadOp::Load => f32::INFINITY,
                    other => panic!("akuma backend: unsupported depth load op {other:?}"),
                },
                None => f32::INFINITY,
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
        _buffer: &wgpu::custom::DispatchBuffer,
        _index_format: wgpu::IndexFormat,
        _offset: BufferAddress,
        _size: Option<BufferSize>,
    ) {
        panic!("akuma backend: index buffers unused");
    }

    fn set_vertex_buffer(
        &mut self,
        _slot: u32,
        _buffer: Option<&wgpu::custom::DispatchBuffer>,
        _offset: BufferAddress,
        _size: Option<BufferSize>,
    ) {
        panic!("akuma backend: vertex buffers unused (vertex_index-driven reads only)");
    }

    fn set_immediates(&mut self, _offset: u32, _data: &[u8]) {
        panic!("akuma backend: immediates unused");
    }

    fn set_blend_constant(&mut self, _color: wgpu::Color) {
        panic!("akuma backend: blending unused");
    }

    fn set_scissor_rect(&mut self, _x: u32, _y: u32, _width: u32, _height: u32) {
        panic!("akuma backend: scissor unused");
    }

    fn set_viewport(
        &mut self,
        _x: f32,
        _y: f32,
        _width: f32,
        _height: f32,
        _min_depth: f32,
        _max_depth: f32,
    ) {
        panic!("akuma backend: viewport unused (positions are already pixels)");
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
        _indices: std::ops::Range<u32>,
        _base_vertex: i32,
        _instances: std::ops::Range<u32>,
    ) {
        panic!("akuma backend: indexed draws unused");
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
    /// snapshot of the mapped bytes (map only happens on quiesced state)
    pub bytes: Arc<Vec<u8>>,
    pub offset: BufferAddress,
    pub size: usize,
}

impl wgpu::custom::BufferMappedRangeInterface for BufferMapped {
    fn len(&self) -> usize {
        self.size
    }

    unsafe fn read_slice(&self) -> &[u8] {
        &self.bytes[self.offset as usize..self.offset as usize + self.size]
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
        // snapshot the quiesced bytes; read_slice hands back the snapshot
        let bytes = self.bytes.lock().unwrap().clone();
        let size = (sub_range.end.min(self.size) - sub_range.start) as usize;
        Ok(wgpu::custom::DispatchBufferMappedRange::custom(
            BufferMapped {
                bytes: Arc::new(bytes),
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
