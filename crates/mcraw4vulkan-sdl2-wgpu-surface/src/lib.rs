use std::error::Error;
use std::fmt;

use sdl2::video::{FullscreenType, Window};

pub const DEFAULT_WINDOW_WIDTH: u32 = 1600;
pub const DEFAULT_WINDOW_HEIGHT: u32 = 900;
pub const DEFAULT_WINDOW_TITLE: &str = "mcraw4vulkan SDL2 wgpu surface";

#[derive(Debug, Clone)]
pub struct Sdl2WgpuSurfaceConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub backends: wgpu::Backends,
    pub power_preference: wgpu::PowerPreference,
    pub preferred_present_mode: Option<wgpu::PresentMode>,
}

impl Default for Sdl2WgpuSurfaceConfig {
    fn default() -> Self {
        Self {
            title: DEFAULT_WINDOW_TITLE.to_string(),
            width: DEFAULT_WINDOW_WIDTH,
            height: DEFAULT_WINDOW_HEIGHT,
            backends: wgpu::Backends::VULKAN,
            power_preference: wgpu::PowerPreference::HighPerformance,
            preferred_present_mode: None,
        }
    }
}

impl Sdl2WgpuSurfaceConfig {
    pub fn new(title: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            title: title.into(),
            width,
            height,
            ..Self::default()
        }
    }

    pub fn with_preferred_present_mode(mut self, present_mode: wgpu::PresentMode) -> Self {
        self.preferred_present_mode = Some(present_mode);
        self
    }

    pub fn validate(&self) -> Result<(), Sdl2WgpuSurfaceError> {
        if self.width == 0 || self.height == 0 {
            return Err(Sdl2WgpuSurfaceError::InvalidWindowSize {
                width: self.width,
                height: self.height,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowSize {
    pub window_width: u32,
    pub window_height: u32,
    pub drawable_width: u32,
    pub drawable_height: u32,
}

impl WindowSize {
    pub fn new(
        window_width: u32,
        window_height: u32,
        drawable_width: u32,
        drawable_height: u32,
    ) -> Self {
        Self {
            window_width,
            window_height,
            drawable_width,
            drawable_height,
        }
    }

    pub fn is_zero(self) -> bool {
        self.drawable_width == 0 || self.drawable_height == 0
    }

    pub fn scale_factor(self) -> f64 {
        if self.window_width == 0 || self.window_height == 0 || self.is_zero() {
            return 1.0;
        }

        let x_scale = f64::from(self.drawable_width) / f64::from(self.window_width);
        let y_scale = f64::from(self.drawable_height) / f64::from(self.window_height);
        let scale = x_scale.min(y_scale);

        if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        }
    }

    fn from_window(window: &Window) -> Self {
        let (window_width, window_height) = window.size();
        let (drawable_width, drawable_height) = window.drawable_size();
        Self::new(window_width, window_height, drawable_width, drawable_height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconfigureStatus {
    Configured,
    SkippedZeroSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderFrameStatus {
    Submitted { suboptimal: bool },
    SkippedZeroSize,
    SurfaceChanged,
    Timeout,
}

pub struct RenderFrameContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub view: &'a wgpu::TextureView,
    pub surface_format: wgpu::TextureFormat,
    pub size: WindowSize,
}

#[derive(Debug)]
pub enum Sdl2WgpuSurfaceError {
    InvalidWindowSize { width: u32, height: u32 },
    SdlInit(String),
    VideoInit(String),
    WindowBuild(String),
    EventPump(String),
    RawWindowHandle(String),
    CreateSurface(wgpu::CreateSurfaceError),
    NoAdapter { backends: wgpu::Backends },
    RequestDevice(wgpu::RequestDeviceError),
    NoSurfaceFormat,
    NoPresentMode,
    NoAlphaMode,
    Surface(wgpu::SurfaceError),
}

impl fmt::Display for Sdl2WgpuSurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWindowSize { width, height } => {
                write!(f, "invalid SDL2 window size {width}x{height}")
            }
            Self::SdlInit(error) => write!(f, "failed to initialize SDL2: {error}"),
            Self::VideoInit(error) => {
                write!(f, "failed to initialize SDL2 video subsystem: {error}")
            }
            Self::WindowBuild(error) => write!(f, "failed to create SDL2 window: {error}"),
            Self::EventPump(error) => write!(f, "failed to create SDL2 event pump: {error}"),
            Self::RawWindowHandle(error) => {
                write!(f, "failed to read SDL2 raw window/display handle: {error}")
            }
            Self::CreateSurface(error) => write!(f, "failed to create wgpu surface: {error}"),
            Self::NoAdapter { backends } => {
                write!(
                    f,
                    "no surface-compatible wgpu adapter found for backends {backends:?}"
                )
            }
            Self::RequestDevice(error) => write!(f, "failed to request wgpu device: {error}"),
            Self::NoSurfaceFormat => write!(f, "surface reported no supported texture formats"),
            Self::NoPresentMode => write!(f, "surface reported no supported present modes"),
            Self::NoAlphaMode => write!(f, "surface reported no supported alpha modes"),
            Self::Surface(error) => write!(f, "wgpu surface error: {error}"),
        }
    }
}

impl Error for Sdl2WgpuSurfaceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CreateSurface(error) => Some(error),
            Self::RequestDevice(error) => Some(error),
            Self::Surface(error) => Some(error),
            _ => None,
        }
    }
}

pub struct Sdl2WgpuSurface {
    // The surface is declared before the SDL window and is also taken in Drop.
    // This keeps the wgpu surface teardown before the SDL-owned native handles.
    surface: Option<wgpu::Surface<'static>>,
    queue: wgpu::Queue,
    device: wgpu::Device,
    adapter: wgpu::Adapter,
    instance: wgpu::Instance,
    window: Window,
    video: sdl2::VideoSubsystem,
    sdl: sdl2::Sdl,
    adapter_info: wgpu::AdapterInfo,
    config: Option<wgpu::SurfaceConfiguration>,
    size: WindowSize,
    surface_format: wgpu::TextureFormat,
    present_mode: wgpu::PresentMode,
    alpha_mode: wgpu::CompositeAlphaMode,
}

impl Sdl2WgpuSurface {
    pub fn new(config: Sdl2WgpuSurfaceConfig) -> Result<Self, Sdl2WgpuSurfaceError> {
        config.validate()?;

        let sdl = sdl2::init().map_err(Sdl2WgpuSurfaceError::SdlInit)?;
        let video = sdl.video().map_err(Sdl2WgpuSurfaceError::VideoInit)?;
        let mut window_builder = video.window(&config.title, config.width, config.height);
        window_builder
            .position_centered()
            .resizable()
            .allow_highdpi()
            .vulkan();
        #[cfg(target_os = "macos")]
        {
            window_builder.metal_view();
        }
        let window = window_builder
            .build()
            .map_err(|error| Sdl2WgpuSurfaceError::WindowBuild(error.to_string()))?;

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: config.backends,
            ..wgpu::InstanceDescriptor::default()
        });

        let surface = {
            // SAFETY: The SDL window is owned by this Sdl2WgpuSurface value and outlives
            // the wgpu Surface. The surface field is declared before the window field and
            // Drop explicitly takes the surface before SDL window teardown, so the surface is
            // dropped before the native SDL handles. Surface use stays on the owning thread
            // through the caller's foreground SDL event loop; resize and reconfigure are safe
            // methods intended to be called from that same SDL/caller-owned event loop. The
            // raw window and display handles are copied only into wgpu's SurfaceTargetUnsafe
            // for surface creation, are not cached by this crate, and are not exposed to callers.
            unsafe {
                let target = wgpu::SurfaceTargetUnsafe::from_window(&window)
                    .map_err(|error| Sdl2WgpuSurfaceError::RawWindowHandle(error.to_string()))?;
                instance
                    .create_surface_unsafe(target)
                    .map_err(Sdl2WgpuSurfaceError::CreateSurface)?
            }
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: config.power_preference,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .ok_or(Sdl2WgpuSurfaceError::NoAdapter {
            backends: config.backends,
        })?;
        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("mcraw4vulkan SDL2 wgpu boundary device"),
                required_features: wgpu::Features::empty(),
                required_limits:
                    wgpu::Limits::downlevel_defaults().using_resolution(adapter_limits),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(Sdl2WgpuSurfaceError::RequestDevice)?;

        let capabilities = surface.get_capabilities(&adapter);
        let surface_format = choose_surface_format(&capabilities.formats)
            .ok_or(Sdl2WgpuSurfaceError::NoSurfaceFormat)?;
        let present_mode = choose_present_mode_with_preference(
            &capabilities.present_modes,
            config.preferred_present_mode,
        )
        .ok_or(Sdl2WgpuSurfaceError::NoPresentMode)?;
        let alpha_mode = choose_alpha_mode(&capabilities.alpha_modes)
            .ok_or(Sdl2WgpuSurfaceError::NoAlphaMode)?;
        let size = WindowSize::from_window(&window);

        let mut boundary = Self {
            surface: Some(surface),
            queue,
            device,
            adapter,
            instance,
            window,
            video,
            sdl,
            adapter_info,
            config: None,
            size,
            surface_format,
            present_mode,
            alpha_mode,
        };

        let _ = boundary.reconfigure()?;
        Ok(boundary)
    }

    pub fn current_video_driver(&self) -> &'static str {
        self.video.current_video_driver()
    }

    pub fn set_clipboard_text(&self, text: &str) -> Result<(), String> {
        self.video.clipboard().set_clipboard_text(text)
    }

    pub fn event_pump(&self) -> Result<sdl2::EventPump, Sdl2WgpuSurfaceError> {
        self.sdl
            .event_pump()
            .map_err(Sdl2WgpuSurfaceError::EventPump)
    }

    pub fn window_id(&self) -> u32 {
        self.window.id()
    }

    pub fn set_fullscreen_desktop(&mut self, enabled: bool) -> Result<(), Sdl2WgpuSurfaceError> {
        let mode = if enabled {
            FullscreenType::Desktop
        } else {
            FullscreenType::Off
        };
        self.window
            .set_fullscreen(mode)
            .map_err(Sdl2WgpuSurfaceError::WindowBuild)
    }

    pub fn maximize_window(&mut self) -> Result<(), Sdl2WgpuSurfaceError> {
        self.window.maximize();
        Ok(())
    }

    pub fn set_window_size(&mut self, width: u32, height: u32) -> Result<(), Sdl2WgpuSurfaceError> {
        if width == 0 || height == 0 {
            return Err(Sdl2WgpuSurfaceError::InvalidWindowSize { width, height });
        }
        self.window.restore();
        self.window
            .set_size(width, height)
            .map_err(|error| Sdl2WgpuSurfaceError::WindowBuild(error.to_string()))?;
        self.size = WindowSize::from_window(&self.window);
        Ok(())
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    pub fn backend(&self) -> wgpu::Backend {
        self.adapter_info.backend
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }

    pub fn current_present_mode(&self) -> wgpu::PresentMode {
        self.present_mode
    }

    pub fn present_mode(&self) -> wgpu::PresentMode {
        self.current_present_mode()
    }

    pub fn alpha_mode(&self) -> wgpu::CompositeAlphaMode {
        self.alpha_mode
    }

    pub fn size(&self) -> WindowSize {
        self.size
    }

    pub fn scale_factor(&self) -> f64 {
        self.size.scale_factor()
    }

    pub fn surface_configuration(&self) -> Option<&wgpu::SurfaceConfiguration> {
        self.config.as_ref()
    }

    pub fn set_preferred_present_mode(
        &mut self,
        preferred_present_mode: wgpu::PresentMode,
    ) -> Result<ReconfigureStatus, Sdl2WgpuSurfaceError> {
        let capabilities = self.surface().get_capabilities(&self.adapter);
        let present_mode = choose_present_mode_with_preference(
            &capabilities.present_modes,
            Some(preferred_present_mode),
        )
        .ok_or(Sdl2WgpuSurfaceError::NoPresentMode)?;
        self.reconfigure_present_mode(present_mode)
    }

    pub fn reconfigure_present_mode(
        &mut self,
        present_mode: wgpu::PresentMode,
    ) -> Result<ReconfigureStatus, Sdl2WgpuSurfaceError> {
        self.present_mode = present_mode;
        self.reconfigure()
    }

    pub fn reconfigure(&mut self) -> Result<ReconfigureStatus, Sdl2WgpuSurfaceError> {
        self.size = WindowSize::from_window(&self.window);

        if self.size.is_zero() {
            self.config = None;
            return Ok(ReconfigureStatus::SkippedZeroSize);
        }

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            width: self.size.drawable_width,
            height: self.size.drawable_height,
            present_mode: self.present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: self.alpha_mode,
            view_formats: vec![],
        };

        self.surface().configure(&self.device, &surface_config);
        self.config = Some(surface_config);
        Ok(ReconfigureStatus::Configured)
    }

    pub fn render_frame(
        &mut self,
        render: impl FnOnce(RenderFrameContext<'_>),
    ) -> Result<RenderFrameStatus, Sdl2WgpuSurfaceError> {
        self.size = WindowSize::from_window(&self.window);

        if self.size.is_zero() {
            self.config = None;
            return Ok(RenderFrameStatus::SkippedZeroSize);
        }

        if self.config.as_ref().is_none_or(|config| {
            config.width != self.size.drawable_width || config.height != self.size.drawable_height
        }) {
            let status = self.reconfigure()?;
            if status == ReconfigureStatus::SkippedZeroSize {
                return Ok(RenderFrameStatus::SkippedZeroSize);
            }
        }

        let surface_texture = match self.surface().get_current_texture() {
            Ok(surface_texture) => surface_texture,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let _ = self.reconfigure()?;
                return Ok(RenderFrameStatus::SurfaceChanged);
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(RenderFrameStatus::Timeout),
            Err(error @ (wgpu::SurfaceError::OutOfMemory | wgpu::SurfaceError::Other)) => {
                return Err(Sdl2WgpuSurfaceError::Surface(error));
            }
        };

        let suboptimal = surface_texture.suboptimal;
        let texture_size = surface_texture.texture.size();
        let frame_size = WindowSize::new(
            self.size.window_width,
            self.size.window_height,
            texture_size.width,
            texture_size.height,
        );
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mcraw4vulkan SDL2 wgpu boundary frame encoder"),
            });

        render(RenderFrameContext {
            device: &self.device,
            queue: &self.queue,
            encoder: &mut encoder,
            view: &view,
            surface_format: self.surface_format,
            size: frame_size,
        });

        self.queue.submit(Some(encoder.finish()));
        surface_texture.present();

        Ok(RenderFrameStatus::Submitted { suboptimal })
    }

    fn surface(&self) -> &wgpu::Surface<'static> {
        self.surface
            .as_ref()
            .expect("surface is present until Sdl2WgpuSurface::drop")
    }
}

impl Drop for Sdl2WgpuSurface {
    fn drop(&mut self) {
        self.config = None;
        let _ = self.surface.take();
    }
}

pub fn choose_surface_format(formats: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
    const PREFERRED: &[wgpu::TextureFormat] = &[
        wgpu::TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Rgba8Unorm,
    ];

    PREFERRED
        .iter()
        .copied()
        .find(|format| formats.contains(format))
        .or_else(|| formats.first().copied())
}

pub fn choose_present_mode(modes: &[wgpu::PresentMode]) -> Option<wgpu::PresentMode> {
    choose_present_mode_with_preference(modes, None)
}

pub fn choose_present_mode_with_preference(
    modes: &[wgpu::PresentMode],
    preferred: Option<wgpu::PresentMode>,
) -> Option<wgpu::PresentMode> {
    if let Some(preferred) = preferred {
        if is_automatic_present_mode(preferred) {
            return Some(preferred);
        }
        if modes.contains(&preferred) {
            return Some(preferred);
        }
    }

    const PREFERRED: &[wgpu::PresentMode] = &[
        wgpu::PresentMode::AutoVsync,
        wgpu::PresentMode::Fifo,
        wgpu::PresentMode::FifoRelaxed,
        wgpu::PresentMode::Mailbox,
        wgpu::PresentMode::Immediate,
    ];

    PREFERRED
        .iter()
        .copied()
        .find(|mode| modes.contains(mode))
        .or_else(|| modes.first().copied())
}

fn is_automatic_present_mode(mode: wgpu::PresentMode) -> bool {
    matches!(
        mode,
        wgpu::PresentMode::AutoVsync | wgpu::PresentMode::AutoNoVsync
    )
}

pub fn choose_alpha_mode(modes: &[wgpu::CompositeAlphaMode]) -> Option<wgpu::CompositeAlphaMode> {
    const PREFERRED: &[wgpu::CompositeAlphaMode] = &[
        wgpu::CompositeAlphaMode::Opaque,
        wgpu::CompositeAlphaMode::Auto,
        wgpu::CompositeAlphaMode::Inherit,
        wgpu::CompositeAlphaMode::PreMultiplied,
        wgpu::CompositeAlphaMode::PostMultiplied,
    ];

    PREFERRED
        .iter()
        .copied()
        .find(|mode| modes.contains(mode))
        .or_else(|| modes.first().copied())
}
