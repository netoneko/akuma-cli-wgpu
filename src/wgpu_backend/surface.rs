//! The wgpu `Surface` over the framebuffer: the swapchain half of the rio
//! port. `get_current_texture` hands out a `Bgra8`-family texture whose
//! storage lives in ordinary RAM and is rendered in place by the normal
//! pipeline paths; `Queue::present` copies whole rows into the `/dev/fb0`
//! WC mapping (see `fb.rs`: never scattered writes).
//!
//! Surfaces are created by `backend::Instance::create_surface`, which opens
//! `/dev/fb0` and falls back to an in-RAM sink when there is no framebuffer
//! (host machines, `gpu-selftest`) — the same code runs in both, and present
//! into the RAM sink is checked numerically by the selftest.
//!
//! One surface texture, reused frame to frame (the fb is single-buffer:
//! there is no swap chain to rotate). `configure` recreates it on size or
//! format change.

use std::sync::{Arc, Mutex};

use super::backend::TextureData;
use crate::fb::FbDevice;

/// Where `present` sends the finished texture.
enum Sink {
    /// the real panel
    Fb(Arc<FbDevice>),
    /// no framebuffer (host tests): present is a no-op the selftest can
    /// observe through the texture bytes instead
    Ram,
}

struct Config {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

impl std::fmt::Debug for Sink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sink::Fb(fb) => f
                .debug_struct("Fb")
                .field("size", &(fb.width, fb.height))
                .finish(),
            Sink::Ram => f.write_str("Ram"),
        }
    }
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("size", &(self.width, self.height))
            .field("format", &self.format)
            .finish()
    }
}

impl std::fmt::Debug for SinkRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SinkRef::Fb(fb) => f
                .debug_struct("Fb")
                .field("size", &(fb.width, fb.height))
                .finish(),
            SinkRef::Ram => f.write_str("Ram"),
        }
    }
}

#[derive(Debug)]
pub struct SurfaceData {
    sink: Mutex<Sink>,
    config: Mutex<Option<Config>>,
    /// the single surface texture, recreated by `configure`
    tex: Mutex<Option<Arc<TextureData>>>,
}

impl SurfaceData {
    /// Open the panel; `None` if there is no usable framebuffer (then
    /// present goes to the RAM sink).
    pub fn open(path: &str) -> Result<SurfaceData, String> {
        match FbDevice::open(path) {
            Ok(fb) => {
                if fb.format.bits_per_pixel != 32 {
                    return Err(format!(
                        "akuma surface: {path} is {} bpp; only 32-bit formats are supported",
                        fb.format.bits_per_pixel
                    ));
                }
                Ok(SurfaceData {
                    sink: Mutex::new(Sink::Fb(Arc::new(fb))),
                    config: Mutex::new(None),
                    tex: Mutex::new(None),
                })
            }
            Err(e) => Err(format!("akuma surface: cannot open {path}: {e}")),
        }
    }

    /// The RAM sink the selftest renders into when there is no panel.
    pub fn ram() -> SurfaceData {
        SurfaceData {
            sink: Mutex::new(Sink::Ram),
            config: Mutex::new(None),
            tex: Mutex::new(None),
        }
    }

    /// The wgpu texture formats this surface can be configured with, most
    /// preferred first (the fb's channel order first — the panel shows
    /// whatever lands in the mapping verbatim).
    fn formats(&self) -> Vec<wgpu::TextureFormat> {
        let first = match &*self.sink.lock().unwrap() {
            Sink::Fb(fb) => fb_format(fb),
            Sink::Ram => wgpu::TextureFormat::Bgra8Unorm,
        };
        let mut formats = vec![first];
        let other = if first == wgpu::TextureFormat::Bgra8Unorm {
            wgpu::TextureFormat::Rgba8Unorm
        } else {
            wgpu::TextureFormat::Bgra8Unorm
        };
        formats.push(other);
        formats
    }
}

/// Map the fbdev channel layout to the wgpu format. The box's panel is
/// 32 bpp with blue in byte 0 (`r16 g8 b0` in the fbdev notation) = Bgra8.
fn fb_format(fb: &FbDevice) -> wgpu::TextureFormat {
    match fb.format.layout() {
        (32, 16, 8, 0) => wgpu::TextureFormat::Bgra8Unorm,
        (32, 0, 8, 16) => wgpu::TextureFormat::Rgba8Unorm,
        other => panic!(
            "akuma surface: unsupported fb layout {other:?} (need 32bpp x8:8:8)"
        ),
    }
}

impl wgpu::custom::SurfaceInterface for SurfaceData {
    fn get_capabilities(
        &self,
        _adapter: &wgpu::custom::DispatchAdapter,
    ) -> wgpu::SurfaceCapabilities {
        let formats = self.formats();
        wgpu::SurfaceCapabilities {
            // every format we hand out is an srgb-unorm byte format
            format_capabilities: formats
                .iter()
                .map(|f| wgpu::SurfaceFormatCapabilities {
                    format: *f,
                    color_spaces: wgpu::SurfaceColorSpaces::SRGB,
                })
                .collect(),
            formats,
            present_modes: vec![wgpu::PresentMode::Fifo],
            // no compositing: the fb is the whole screen
            alpha_modes: vec![wgpu::CompositeAlphaMode::Opaque],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
        }
    }

    fn configure(
        &self,
        _device: &wgpu::custom::DispatchDevice,
        config: &wgpu::SurfaceConfiguration,
    ) {
        let formats = self.formats();
        assert!(
            formats.contains(&config.format),
            "akuma surface: format {:?} not supported (have {formats:?})",
            config.format
        );
        assert!(
            matches!(
                config.alpha_mode,
                wgpu::CompositeAlphaMode::Opaque | wgpu::CompositeAlphaMode::Auto
            ),
            "akuma surface: alpha mode {:?} unsupported (the fb does not composite)",
            config.alpha_mode
        );
        let bpt = super::format::bytes_per_texel(config.format)
            .expect("akuma surface: unsupported format") as usize;
        let bytes = config.width as usize * config.height as usize * bpt;
        let tex = Arc::new(TextureData {
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            format: config.format,
            store: Arc::new(super::backend::TexStore::Color(Mutex::new(vec![
                0u8;
                bytes
            ]))),
        });
        *self.tex.lock().unwrap() = Some(tex);
        *self.config.lock().unwrap() = Some(Config {
            width: config.width,
            height: config.height,
            format: config.format,
        });
    }

    fn get_current_texture(
        &self,
    ) -> (
        Option<wgpu::custom::DispatchTexture>,
        wgpu::SurfaceStatus,
        wgpu::custom::DispatchSurfaceOutputDetail,
    ) {
        let tex = self.tex.lock().unwrap().clone();
        match tex {
            Some(tex) => (
                Some(wgpu::custom::DispatchTexture::custom(TextureData {
                    size: tex.size,
                    format: tex.format,
                    store: Arc::clone(&tex.store),
                })),
                wgpu::SurfaceStatus::Good,
                wgpu::custom::DispatchSurfaceOutputDetail::custom(SurfaceOutputDetail {
                    tex,
                    sink: self.sink.lock().unwrap().share(),
                }),
            ),
            // surface not configured yet
            None => (
                None,
                wgpu::SurfaceStatus::Lost,
                wgpu::custom::DispatchSurfaceOutputDetail::custom(SurfaceOutputDetail {
                    tex: Arc::new(TextureData {
                        size: wgpu::Extent3d::default(),
                        format: wgpu::TextureFormat::Bgra8Unorm,
                        store: Arc::new(super::backend::TexStore::Color(Mutex::new(
                            Vec::new(),
                        ))),
                    }),
                    sink: SinkRef::Ram,
                }),
            ),
        }
    }
}

/// A cheap clone of the sink for the per-frame output detail.
enum SinkRef {
    Fb(Arc<FbDevice>),
    Ram,
}

impl Sink {
    fn share(&self) -> SinkRef {
        match self {
            Sink::Fb(fb) => SinkRef::Fb(Arc::clone(fb)),
            Sink::Ram => SinkRef::Ram,
        }
    }
}

#[derive(Debug)]
pub struct SurfaceOutputDetail {
    tex: Arc<TextureData>,
    sink: SinkRef,
}

impl wgpu::custom::SurfaceOutputDetailInterface for SurfaceOutputDetail {
    fn texture_discard(&self) {
        // single-buffer surface: the texture is reused as-is
    }

    fn texture_release(&self) {
        // nothing to retire
    }
}

impl SurfaceOutputDetail {
    /// Copy the texture's rows into the presentation sink, one contiguous
    /// copy per row (the WC contract); a surface smaller than the panel
    /// lands in its top-left corner.
    pub fn present(&self) {
        let (w, h) = (self.tex.size.width as usize, self.tex.size.height as usize);
        let bpt = super::format::bytes_per_texel(self.tex.format)
            .expect("akuma surface: unsupported texture format") as usize;
        let row_bytes = w * bpt;
        let store = match &*self.tex.store {
            super::backend::TexStore::Color(bytes) => bytes,
            super::backend::TexStore::Depth(_) => {
                panic!("akuma surface: depth texture presented")
            }
        };
        let bytes = store.lock().unwrap();
        debug_assert_eq!(bytes.len(), row_bytes * h);
        match &self.sink {
            SinkRef::Fb(fb) => {
                // smaller than the panel = top-left corner (see present_raw)
                assert!(w <= fb.width && h <= fb.height, "surface larger than fb");
                let tubes = super::crt::tubes() as usize;
                if tubes > 0 && bpt == 4 {
                    // the CRT look (crt.rs): curvature, scanlines, vignette
                    let mut cache = super::crt::CACHE.lock().unwrap();
                    let curved = super::crt::curved();
                    if !cache.as_ref().is_some_and(|c| c.fits(w, h, tubes, curved)) {
                        *cache = Some(super::crt::Crt::new(w, h, tubes, curved));
                    }
                    let t0 = crate::clock::monotonic();
                    let out = cache.as_mut().unwrap().apply(&bytes);
                    let t1 = crate::clock::monotonic();
                    fb.present_raw(out, row_bytes);
                    if std::env::var_os("AKUMA_EXEC_VERBOSE").is_some() {
                        eprintln!(
                            "[crt] {w}x{h}, {tubes} tube(s): effect {:.1} ms, copy {:.1} ms",
                            (t1 - t0) * 1e3,
                            (crate::clock::monotonic() - t1) * 1e3
                        );
                    }
                } else {
                    fb.present_raw(&bytes, row_bytes);
                }
            }
            SinkRef::Ram => {
                // the selftest reads the texture bytes directly; presenting
                // is a no-op (kept so the code path is identical)
            }
        }
    }
}
