use std::time::{Duration, Instant};

pub const DEFAULT_DISPLAY_WINDOW_WIDTH: u32 = 1920;
pub const DEFAULT_DISPLAY_WINDOW_HEIGHT: u32 = 1080;

#[derive(Debug, thiserror::Error)]
pub enum DisplayError {
    #[error("surface has no supported texture formats")]
    NoSurfaceFormats,
    #[error("surface has no supported present modes")]
    NoPresentModes,
    #[error("surface size must be non-zero, got {width}x{height}")]
    InvalidSurfaceSize { width: u32, height: u32 },
    #[error("source texture size must be non-zero, got {width}x{height}")]
    InvalidSourceSize { width: u32, height: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplaySize {
    pub width: u32,
    pub height: u32,
}

impl DisplaySize {
    pub fn new(width: u32, height: u32) -> Result<Self, DisplayError> {
        if width == 0 || height == 0 {
            return Err(DisplayError::InvalidSurfaceSize { width, height });
        }
        Ok(Self { width, height })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Default for DisplayRect {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DisplayScaleMode {
    #[default]
    FitContain,
    Stretch,
}

impl DisplayScaleMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::FitContain => "fit-contain",
            Self::Stretch => "stretch",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DisplayPresentMode {
    AutoVsync,
    AutoNoVsync,
    #[default]
    Fifo,
    Immediate,
    Mailbox,
}

impl DisplayPresentMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::AutoVsync => "auto-vsync",
            Self::AutoNoVsync => "auto-no-vsync",
            Self::Fifo => "fifo",
            Self::Immediate => "immediate",
            Self::Mailbox => "mailbox",
        }
    }

    pub fn parse_label(value: &str) -> Option<Self> {
        match value {
            "auto-vsync" | "auto_vsync" => Some(Self::AutoVsync),
            "auto-no-vsync" | "auto_no_vsync" => Some(Self::AutoNoVsync),
            "fifo" => Some(Self::Fifo),
            "immediate" => Some(Self::Immediate),
            "mailbox" => Some(Self::Mailbox),
            _ => None,
        }
    }

    pub fn to_wgpu(self) -> wgpu::PresentMode {
        match self {
            Self::AutoVsync => wgpu::PresentMode::AutoVsync,
            Self::AutoNoVsync => wgpu::PresentMode::AutoNoVsync,
            Self::Fifo => wgpu::PresentMode::Fifo,
            Self::Immediate => wgpu::PresentMode::Immediate,
            Self::Mailbox => wgpu::PresentMode::Mailbox,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayPresentPolicy {
    Vsync,
}

impl DisplayPresentPolicy {
    pub fn present_mode(self) -> DisplayPresentMode {
        match self {
            Self::Vsync => DisplayPresentMode::Fifo,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayWindowMode {
    Windowed,
    FullscreenBorderless,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayWindowPolicy {
    pub default_window_width: u32,
    pub default_window_height: u32,
    pub preserve_aspect: bool,
    pub fullscreen_uses_current_monitor: bool,
    pub present_policy: DisplayPresentPolicy,
}

impl Default for DisplayWindowPolicy {
    fn default() -> Self {
        Self {
            default_window_width: DEFAULT_DISPLAY_WINDOW_WIDTH,
            default_window_height: DEFAULT_DISPLAY_WINDOW_HEIGHT,
            preserve_aspect: true,
            fullscreen_uses_current_monitor: true,
            present_policy: DisplayPresentPolicy::Vsync,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DisplayViewport {
    pub surface_width: u32,
    pub surface_height: u32,
    pub content_x: u32,
    pub content_y: u32,
    pub content_width: u32,
    pub content_height: u32,
    pub source_width: u32,
    pub source_height: u32,
    pub scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplaySourceTransfer {
    ShaderSrgb,
    HardwareSrgb,
}

impl DisplaySourceTransfer {
    pub fn label(self) -> &'static str {
        match self {
            Self::ShaderSrgb => "shader-srgb",
            Self::HardwareSrgb => "hardware-srgb",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayTransferStrategy {
    pub source_is_srgb_encoded: bool,
    pub surface_is_srgb: bool,
    pub present_shader_srgb_decode: bool,
    pub present_shader_oetf: bool,
    pub double_srgb_avoided: bool,
}

impl DisplayTransferStrategy {
    // An sRGB surface encodes linear fragment output as it is stored. Decode an
    // encoded source before that step, or encode in shader for a linear surface, so
    // the transfer function is applied exactly once.
    pub fn for_source_and_surface(
        source_transfer: DisplaySourceTransfer,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let source_is_srgb_encoded = matches!(source_transfer, DisplaySourceTransfer::ShaderSrgb);
        let surface_is_srgb = is_srgb_format(surface_format);
        let present_shader_srgb_decode = source_is_srgb_encoded && surface_is_srgb;
        let present_shader_oetf = !source_is_srgb_encoded && !surface_is_srgb;
        Self {
            source_is_srgb_encoded,
            surface_is_srgb,
            present_shader_srgb_decode,
            present_shader_oetf,
            double_srgb_avoided: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplaySurfaceSelection {
    pub format: wgpu::TextureFormat,
    pub requested_present_mode: DisplayPresentMode,
    pub present_mode: wgpu::PresentMode,
    pub alpha_mode: wgpu::CompositeAlphaMode,
    pub configuration: wgpu::SurfaceConfiguration,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DisplayPresentEncodeStats {
    pub params_upload: Duration,
    pub bind_group: Duration,
    pub render_pass_encode: Duration,
    pub viewport: DisplayRect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayTargetLoadOp {
    Clear,
    Load,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayPresentTarget {
    pub origin_x: u32,
    pub origin_y: u32,
    pub size: DisplaySize,
    pub load_op: DisplayTargetLoadOp,
}

impl DisplayPresentTarget {
    pub fn full(size: DisplaySize) -> Self {
        Self {
            origin_x: 0,
            origin_y: 0,
            size,
            load_op: DisplayTargetLoadOp::Clear,
        }
    }

    pub fn embedded(
        origin_x: u32,
        origin_y: u32,
        width: u32,
        height: u32,
    ) -> Result<Self, DisplayError> {
        Ok(Self {
            origin_x,
            origin_y,
            size: DisplaySize::new(width, height)?,
            load_op: DisplayTargetLoadOp::Load,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayFpsOverlay {
    pub fps_x100: u32,
    pub source_width: u32,
    pub source_height: u32,
}

pub const DISPLAY_FPS_OVERLAY_MAX_X100: u32 = 999_999;
pub const DISPLAY_FPS_OVERLAY_REQUIRED_GLYPHS: &str = "0123456789. fpsx";

impl DisplayFpsOverlay {
    pub fn from_fps(fps: f64) -> Option<Self> {
        if !fps.is_finite() || fps < 0.0 {
            return None;
        }
        let fps_x100 = (fps * 100.0)
            .round()
            .clamp(0.0, f64::from(DISPLAY_FPS_OVERLAY_MAX_X100)) as u32;
        Some(Self {
            fps_x100,
            source_width: 0,
            source_height: 0,
        })
    }

    pub fn fps(self) -> f64 {
        f64::from(self.fps_x100) / 100.0
    }

    pub fn with_source_size(mut self, source_size: DisplaySize) -> Self {
        self.source_width = source_size.width;
        self.source_height = source_size.height;
        self
    }

    pub fn source_size(self) -> Option<DisplaySize> {
        (self.source_width != 0 && self.source_height != 0).then_some(DisplaySize {
            width: self.source_width,
            height: self.source_height,
        })
    }

    pub fn text(self) -> String {
        let fps = format!("{:.2} fps", self.fps());
        match self.source_size() {
            Some(source_size) => format!("{} x {}\n\n{fps}", source_size.width, source_size.height),
            None => fps,
        }
    }
}

pub fn fps_overlay_text_from_fps(fps: f64) -> Option<String> {
    DisplayFpsOverlay::from_fps(fps).map(DisplayFpsOverlay::text)
}

pub fn fps_overlay_supports_glyph(glyph: char) -> bool {
    matches!(glyph, '0'..='9' | '.' | ' ' | 'f' | 'p' | 's' | 'x')
}

#[derive(Debug, Clone, Copy)]
pub struct DisplayFpsMeter {
    sample_start: Instant,
    frames_since_sample: u32,
    measured_fps: Option<f64>,
    min_sample_duration: Duration,
}

impl DisplayFpsMeter {
    pub const DEFAULT_MIN_SAMPLE_DURATION: Duration = Duration::from_millis(500);

    pub fn new(start: Instant) -> Self {
        Self::with_min_sample_duration(start, Self::DEFAULT_MIN_SAMPLE_DURATION)
    }

    pub fn with_min_sample_duration(start: Instant, min_sample_duration: Duration) -> Self {
        Self {
            sample_start: start,
            frames_since_sample: 0,
            measured_fps: None,
            min_sample_duration: min_sample_duration.max(Duration::from_nanos(1)),
        }
    }

    pub fn current_fps(self) -> Option<f64> {
        self.measured_fps
    }

    pub fn fps_for_overlay(self, initial_estimate_fps: f64) -> Option<f64> {
        self.measured_fps.or_else(|| {
            (initial_estimate_fps.is_finite() && initial_estimate_fps >= 0.0)
                .then_some(initial_estimate_fps)
        })
    }

    pub fn record_presented(&mut self, now: Instant) -> Option<f64> {
        self.record_presented_frames(1, now)
    }

    pub fn record_presented_frames(&mut self, frames: u32, now: Instant) -> Option<f64> {
        if frames == 0 {
            return self.measured_fps;
        }
        self.frames_since_sample = self.frames_since_sample.saturating_add(frames);
        let elapsed = now.saturating_duration_since(self.sample_start);
        if elapsed >= self.min_sample_duration && !elapsed.is_zero() {
            let fps = f64::from(self.frames_since_sample) / elapsed.as_secs_f64();
            if fps.is_finite() && fps >= 0.0 {
                self.measured_fps = Some(fps);
            }
            self.sample_start = now;
            self.frames_since_sample = 0;
        }
        self.measured_fps
    }
}

pub struct DisplayTexturePresenter {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params_buffer: wgpu::Buffer,
    surface_format: wgpu::TextureFormat,
    scale_mode: DisplayScaleMode,
}

pub struct DisplayPresentInput<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub source_view: &'a wgpu::TextureView,
    pub source_size: DisplaySize,
    pub target_view: &'a wgpu::TextureView,
    pub target: DisplayPresentTarget,
    pub transfer_strategy: DisplayTransferStrategy,
    pub fps_overlay: Option<DisplayFpsOverlay>,
}

impl DisplayTexturePresenter {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        scale_mode: DisplayScaleMode,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mcraw4vulkan display texture presenter shader"),
            source: wgpu::ShaderSource::Wgsl(DISPLAY_PRESENT_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mcraw4vulkan display texture presenter bind group layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(32),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mcraw4vulkan display texture presenter pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mcraw4vulkan display texture presenter pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mcraw4vulkan display texture presenter sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcraw4vulkan display texture presenter params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bind_group_layout,
            sampler,
            params_buffer,
            surface_format,
            scale_mode,
        }
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }

    pub fn scale_mode(&self) -> DisplayScaleMode {
        self.scale_mode
    }

    pub fn encode_present(
        &mut self,
        input: DisplayPresentInput<'_>,
    ) -> Result<DisplayPresentEncodeStats, DisplayError> {
        let DisplayPresentInput {
            device,
            queue,
            encoder,
            source_view,
            source_size,
            target_view,
            target,
            transfer_strategy,
            fps_overlay,
        } = input;
        let target_size = target.size;

        if source_size.width == 0 || source_size.height == 0 {
            return Err(DisplayError::InvalidSourceSize {
                width: source_size.width,
                height: source_size.height,
            });
        }
        if target_size.width == 0 || target_size.height == 0 {
            return Err(DisplayError::InvalidSurfaceSize {
                width: target_size.width,
                height: target_size.height,
            });
        }

        let viewport = match self.scale_mode {
            DisplayScaleMode::FitContain => aspect_fit_rect(source_size, target_size),
            DisplayScaleMode::Stretch => DisplayRect {
                x: 0.0,
                y: 0.0,
                width: target_size.width as f32,
                height: target_size.height as f32,
            },
        };

        let overlay_source_size = fps_overlay
            .and_then(DisplayFpsOverlay::source_size)
            .unwrap_or(source_size);
        // Keep these eight words in PresentParams field order; this 32-byte
        // little-endian payload is the host half of the WGSL uniform contract.
        let params = [
            u32::from(transfer_strategy.present_shader_srgb_decode),
            u32::from(transfer_strategy.present_shader_oetf),
            u32::from(fps_overlay.is_some()),
            fps_overlay.map(|overlay| overlay.fps_x100).unwrap_or(0),
            viewport.width.to_bits(),
            viewport.height.to_bits(),
            if fps_overlay.is_some() {
                overlay_source_size.width
            } else {
                0
            },
            if fps_overlay.is_some() {
                overlay_source_size.height
            } else {
                0
            },
        ];
        let mut bytes = [0_u8; 32];
        for (index, word) in params.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }

        let params_upload_start = Instant::now();
        queue.write_buffer(&self.params_buffer, 0, &bytes);
        let params_upload = params_upload_start.elapsed();

        let bind_group_start = Instant::now();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mcraw4vulkan display texture presenter bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });
        let bind_group_elapsed = bind_group_start.elapsed();

        let render_pass_encode_start = Instant::now();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mcraw4vulkan display texture presenter pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: match target.load_op {
                            DisplayTargetLoadOp::Clear => wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            DisplayTargetLoadOp::Load => wgpu::LoadOp::Load,
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_scissor_rect(
                target.origin_x,
                target.origin_y,
                target_size.width,
                target_size.height,
            );
            pass.set_viewport(
                target.origin_x as f32 + viewport.x,
                target.origin_y as f32 + viewport.y,
                viewport.width,
                viewport.height,
                0.0,
                1.0,
            );
            pass.draw(0..3, 0..1);
        }
        let render_pass_encode = render_pass_encode_start.elapsed();

        Ok(DisplayPresentEncodeStats {
            params_upload,
            bind_group: bind_group_elapsed,
            render_pass_encode,
            viewport,
        })
    }
}

pub fn choose_surface_format(formats: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
    const PREFERENCE: &[wgpu::TextureFormat] = &[
        wgpu::TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Rgba8Unorm,
    ];
    PREFERENCE
        .iter()
        .copied()
        .find(|format| formats.contains(format))
        .or_else(|| formats.first().copied())
}

pub fn choose_present_mode(
    supported: &[wgpu::PresentMode],
    requested: DisplayPresentMode,
) -> wgpu::PresentMode {
    match requested {
        DisplayPresentMode::AutoVsync => wgpu::PresentMode::AutoVsync,
        DisplayPresentMode::AutoNoVsync => wgpu::PresentMode::AutoNoVsync,
        DisplayPresentMode::Fifo => wgpu::PresentMode::Fifo,
        DisplayPresentMode::Immediate => {
            if supported.contains(&wgpu::PresentMode::Immediate) {
                wgpu::PresentMode::Immediate
            } else {
                wgpu::PresentMode::Fifo
            }
        }
        DisplayPresentMode::Mailbox => {
            if supported.contains(&wgpu::PresentMode::Mailbox) {
                wgpu::PresentMode::Mailbox
            } else {
                wgpu::PresentMode::Fifo
            }
        }
    }
}

pub fn build_surface_selection(
    capabilities: &wgpu::SurfaceCapabilities,
    width: u32,
    height: u32,
    requested_present_mode: DisplayPresentMode,
) -> Result<DisplaySurfaceSelection, DisplayError> {
    if width == 0 || height == 0 {
        return Err(DisplayError::InvalidSurfaceSize { width, height });
    }
    let format =
        choose_surface_format(&capabilities.formats).ok_or(DisplayError::NoSurfaceFormats)?;
    if capabilities.present_modes.is_empty() {
        return Err(DisplayError::NoPresentModes);
    }
    let present_mode = choose_present_mode(&capabilities.present_modes, requested_present_mode);
    let alpha_mode = capabilities
        .alpha_modes
        .first()
        .copied()
        .unwrap_or(wgpu::CompositeAlphaMode::Auto);
    let configuration = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode,
        desired_maximum_frame_latency: 2,
        alpha_mode,
        view_formats: vec![],
    };
    Ok(DisplaySurfaceSelection {
        format,
        requested_present_mode,
        present_mode,
        alpha_mode,
        configuration,
    })
}

pub fn is_srgb_format(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Rgba8UnormSrgb
    )
}

pub fn aspect_fit_rect(source: DisplaySize, target: DisplaySize) -> DisplayRect {
    let viewport = fit_aspect_preserving(target.width, target.height, source.width, source.height);
    DisplayRect {
        x: viewport.content_x as f32,
        y: viewport.content_y as f32,
        width: viewport.content_width as f32,
        height: viewport.content_height as f32,
    }
}

pub fn fit_aspect_preserving(
    surface_width: u32,
    surface_height: u32,
    source_width: u32,
    source_height: u32,
) -> DisplayViewport {
    if surface_width == 0 || surface_height == 0 || source_width == 0 || source_height == 0 {
        return DisplayViewport {
            surface_width,
            surface_height,
            source_width,
            source_height,
            ..DisplayViewport::default()
        };
    }

    let source_w = u128::from(source_width);
    let source_h = u128::from(source_height);
    let surface_w = u128::from(surface_width);
    let surface_h = u128::from(surface_height);
    let source_is_wider = source_w.saturating_mul(surface_h) > source_h.saturating_mul(surface_w);
    let (content_width, content_height) = if source_is_wider {
        let height = surface_w.saturating_mul(source_h) / source_w;
        (
            surface_width,
            u32::try_from(height)
                .unwrap_or(u32::MAX)
                .clamp(1, surface_height),
        )
    } else {
        let width = surface_h.saturating_mul(source_w) / source_h;
        (
            u32::try_from(width)
                .unwrap_or(u32::MAX)
                .clamp(1, surface_width),
            surface_height,
        )
    };
    let content_x = (surface_width - content_width) / 2;
    let content_y = (surface_height - content_height) / 2;
    let scale = content_width as f32 / source_width as f32;

    DisplayViewport {
        surface_width,
        surface_height,
        content_x,
        content_y,
        content_width,
        content_height,
        source_width,
        source_height,
        scale,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayFramePacingDecision {
    PresentFrame(u64),
    RepeatCurrent,
    DropToFrame(u64),
    WaitUntil(Instant),
}

#[derive(Debug, Clone, Copy)]
pub struct DisplayFramePacer {
    frame_duration: Duration,
    start_instant: Instant,
    start_frame_index: u64,
    last_presented_frame: Option<u64>,
}

impl DisplayFramePacer {
    pub fn new(
        source_frame_rate_fps: f64,
        start_instant: Instant,
        start_frame_index: u64,
    ) -> Option<Self> {
        if !source_frame_rate_fps.is_finite() || source_frame_rate_fps <= 0.0 {
            return None;
        }
        let frame_duration = Duration::from_secs_f64(1.0 / source_frame_rate_fps);
        Self::from_frame_duration(frame_duration, start_instant, start_frame_index)
    }

    pub fn from_frame_duration(
        frame_duration: Duration,
        start_instant: Instant,
        start_frame_index: u64,
    ) -> Option<Self> {
        if frame_duration.is_zero() {
            return None;
        }
        Some(Self {
            frame_duration,
            start_instant,
            start_frame_index,
            last_presented_frame: None,
        })
    }

    pub fn frame_duration(self) -> Duration {
        self.frame_duration
    }

    pub fn last_presented_frame(self) -> Option<u64> {
        self.last_presented_frame
    }

    pub fn next_frame_instant(self, frame_index: u64) -> Instant {
        let offset = frame_index.saturating_sub(self.start_frame_index);
        self.start_instant + duration_mul_u64(self.frame_duration, offset)
    }

    pub fn sleep_duration_until(self, frame_index: u64, now: Instant) -> Option<Duration> {
        self.next_frame_instant(frame_index)
            .checked_duration_since(now)
    }

    pub fn due_frame_at(self, now: Instant) -> u64 {
        if now <= self.start_instant {
            return self.start_frame_index;
        }
        let elapsed = now.duration_since(self.start_instant);
        let frame_ns = self.frame_duration.as_nanos().max(1);
        let elapsed_frames = elapsed.as_nanos() / frame_ns;
        self.start_frame_index
            .saturating_add(u64::try_from(elapsed_frames).unwrap_or(u64::MAX))
    }

    pub fn decision(self, now: Instant, decoded_max_frame: u64) -> DisplayFramePacingDecision {
        let due_frame = self.due_frame_at(now);
        match self.last_presented_frame {
            None => DisplayFramePacingDecision::PresentFrame(due_frame.min(decoded_max_frame)),
            Some(last_presented) => {
                let next_frame = last_presented.saturating_add(1);
                let next_instant = self.next_frame_instant(next_frame);
                if now < next_instant {
                    return DisplayFramePacingDecision::WaitUntil(next_instant);
                }
                if decoded_max_frame < next_frame {
                    return DisplayFramePacingDecision::RepeatCurrent;
                }
                let target = due_frame.min(decoded_max_frame).max(next_frame);
                if target > next_frame {
                    DisplayFramePacingDecision::DropToFrame(target)
                } else {
                    DisplayFramePacingDecision::PresentFrame(next_frame)
                }
            }
        }
    }

    pub fn record_presented(&mut self, frame_index: u64) {
        self.last_presented_frame = Some(frame_index);
    }

    pub fn reset_next_frame_due_now(&mut self, frame_index: u64, now: Instant) {
        self.start_instant = now.checked_sub(Duration::from_nanos(1)).unwrap_or(now);
        self.start_frame_index = frame_index;
        self.last_presented_frame = None;
    }
}

fn duration_mul_u64(duration: Duration, factor: u64) -> Duration {
    let nanos = duration
        .as_nanos()
        .saturating_mul(u128::from(factor))
        .min(u128::from(u64::MAX));
    Duration::from_nanos(nanos as u64)
}

const DISPLAY_PRESENT_WGSL: &str = r#"
struct PresentParams {
    srgb_decode: u32,
    srgb_encode: u32,
    fps_overlay_enabled: u32,
    fps_x100: u32,
    content_width: f32,
    content_height: f32,
    source_width: u32,
    source_height: u32,
};

@group(0) @binding(0)
var preview_texture: texture_2d<f32>;

@group(0) @binding(1)
var preview_sampler: sampler;

@group(0) @binding(2)
var<uniform> params: PresentParams;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VertexOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

fn srgb_eotf_one(encoded: f32) -> f32 {
    let clamped = clamp(encoded, 0.0, 1.0);
    if (clamped <= 0.04045) {
        return clamped / 12.92;
    }
    return pow((clamped + 0.055) / 1.055, 2.4);
}

fn srgb_oetf_one(linear: f32) -> f32 {
    let clamped = clamp(linear, 0.0, 1.0);
    if (clamped <= 0.0031308) {
        return 12.92 * clamped;
    }
    return 1.055 * pow(clamped, 1.0 / 2.4) - 0.055;
}

fn srgb_eotf(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_eotf_one(rgb.x),
        srgb_eotf_one(rgb.y),
        srgb_eotf_one(rgb.z),
    );
}

fn srgb_oetf(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_oetf_one(rgb.x),
        srgb_oetf_one(rgb.y),
        srgb_oetf_one(rgb.z),
    );
}

fn digit_mask(digit: u32) -> u32 {
    switch digit {
        case 0u: { return 0x3fu; }
        case 1u: { return 0x06u; }
        case 2u: { return 0x5bu; }
        case 3u: { return 0x4fu; }
        case 4u: { return 0x66u; }
        case 5u: { return 0x6du; }
        case 6u: { return 0x7du; }
        case 7u: { return 0x07u; }
        case 8u: { return 0x7fu; }
        case 9u: { return 0x6fu; }
        default: { return 0u; }
    }
}

fn in_rect(p: vec2<f32>, x0: f32, y0: f32, x1: f32, y1: f32) -> bool {
    return p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1;
}

fn seven_segment_digit(local: vec2<f32>, digit: u32, scale: f32) -> bool {
    let mask = digit_mask(digit);
    let w = 8.0 * scale;
    let h = 14.0 * scale;
    let t = max(1.0, 2.0 * scale);
    let mid = 6.0 * scale;
    if (local.x < 0.0 || local.y < 0.0 || local.x > w || local.y > h) {
        return false;
    }
    if ((mask & 0x01u) != 0u && in_rect(local, t, 0.0, w - t, t)) {
        return true;
    }
    if ((mask & 0x02u) != 0u && in_rect(local, w - t, t, w, mid)) {
        return true;
    }
    if ((mask & 0x04u) != 0u && in_rect(local, w - t, mid + t, w, h - t)) {
        return true;
    }
    if ((mask & 0x08u) != 0u && in_rect(local, t, h - t, w - t, h)) {
        return true;
    }
    if ((mask & 0x10u) != 0u && in_rect(local, 0.0, mid + t, t, h - t)) {
        return true;
    }
    if ((mask & 0x20u) != 0u && in_rect(local, 0.0, t, t, mid)) {
        return true;
    }
    if ((mask & 0x40u) != 0u && in_rect(local, t, mid, w - t, mid + t)) {
        return true;
    }
    return false;
}

fn decimal_point_glyph(local: vec2<f32>, scale: f32) -> bool {
    let dot = max(1.0, 2.0 * scale);
    let x0 = 3.0 * scale;
    let y0 = 12.0 * scale;
    return in_rect(local, x0, y0, x0 + dot, y0 + dot);
}

fn glyph_f(local: vec2<f32>, scale: f32) -> bool {
    let w = 8.0 * scale;
    let h = 14.0 * scale;
    let t = max(1.0, 2.0 * scale);
    let mid = 6.0 * scale;
    if (local.x < 0.0 || local.y < 0.0 || local.x > w || local.y > h) {
        return false;
    }
    return in_rect(local, t, 0.0, w, t) ||
        in_rect(local, 0.0, t, t, h) ||
        in_rect(local, t, mid, w - t, mid + t);
}

fn glyph_p(local: vec2<f32>, scale: f32) -> bool {
    let w = 8.0 * scale;
    let h = 14.0 * scale;
    let t = max(1.0, 2.0 * scale);
    let mid = 6.0 * scale;
    if (local.x < 0.0 || local.y < 0.0 || local.x > w || local.y > h) {
        return false;
    }
    return in_rect(local, 0.0, 0.0, t, h) ||
        in_rect(local, t, 0.0, w - t, t) ||
        in_rect(local, w - t, t, w, mid) ||
        in_rect(local, t, mid, w - t, mid + t);
}

fn glyph_x(local: vec2<f32>, scale: f32) -> bool {
    let w = 8.0 * scale;
    let h = 14.0 * scale;
    let t = max(1.0, 1.5 * scale);
    if (local.x < 0.0 || local.y < 0.0 || local.x > w || local.y > h) {
        return false;
    }
    let y_down = local.x * h / w;
    let y_up = h - y_down;
    return abs(local.y - y_down) <= t || abs(local.y - y_up) <= t;
}

fn decimal_digit_count(value: u32) -> u32 {
    var digits = 1u;
    if (value >= 10u) { digits = 2u; }
    if (value >= 100u) { digits = 3u; }
    if (value >= 1000u) { digits = 4u; }
    if (value >= 10000u) { digits = 5u; }
    if (value >= 100000u) { digits = 6u; }
    if (value >= 1000000u) { digits = 7u; }
    if (value >= 10000000u) { digits = 8u; }
    if (value >= 100000000u) { digits = 9u; }
    if (value >= 1000000000u) { digits = 10u; }
    return digits;
}

fn pow10_u32(exp: u32) -> u32 {
    switch exp {
        case 0u: { return 1u; }
        case 1u: { return 10u; }
        case 2u: { return 100u; }
        case 3u: { return 1000u; }
        case 4u: { return 10000u; }
        case 5u: { return 100000u; }
        case 6u: { return 1000000u; }
        case 7u: { return 10000000u; }
        case 8u: { return 100000000u; }
        default: { return 1000000000u; }
    }
}

fn decimal_digit(value: u32, digit_count: u32, index: u32) -> u32 {
    let divisor = pow10_u32(digit_count - index - 1u);
    return (value / divisor) % 10u;
}

fn fps_integer_digit(whole: u32, digit_count: u32, index: u32) -> u32 {
    if (digit_count == 4u) {
        if (index == 0u) { return whole / 1000u; }
        if (index == 1u) { return (whole / 100u) % 10u; }
        if (index == 2u) { return (whole / 10u) % 10u; }
        return whole % 10u;
    }
    if (digit_count == 3u) {
        if (index == 0u) { return whole / 100u; }
        if (index == 1u) { return (whole / 10u) % 10u; }
        return whole % 10u;
    }
    if (digit_count == 2u) {
        if (index == 0u) { return whole / 10u; }
        return whole % 10u;
    }
    return whole % 10u;
}

fn fps_overlay_char(local: vec2<f32>, char_index: u32, digit_count: u32, whole: u32, fraction: u32, scale: f32) -> bool {
    if (char_index < digit_count) {
        return seven_segment_digit(local, fps_integer_digit(whole, digit_count, char_index), scale);
    }

    let suffix_index = char_index - digit_count;
    if (suffix_index == 0u) {
        return decimal_point_glyph(local, scale);
    }
    if (suffix_index == 1u) {
        return seven_segment_digit(local, fraction / 10u, scale);
    }
    if (suffix_index == 2u) {
        return seven_segment_digit(local, fraction % 10u, scale);
    }
    if (suffix_index == 3u) {
        return false;
    }
    if (suffix_index == 4u) {
        return glyph_f(local, scale);
    }
    if (suffix_index == 5u) {
        return glyph_p(local, scale);
    }
    if (suffix_index == 6u) {
        return seven_segment_digit(local, 5u, scale);
    }
    return false;
}

fn resolution_overlay_char(
    local: vec2<f32>,
    char_index: u32,
    width_digit_count: u32,
    height_digit_count: u32,
    source_width: u32,
    source_height: u32,
    scale: f32
) -> bool {
    if (char_index < width_digit_count) {
        return seven_segment_digit(
            local,
            decimal_digit(source_width, width_digit_count, char_index),
            scale
        );
    }

    let separator_index = char_index - width_digit_count;
    if (separator_index == 0u) {
        return false;
    }
    if (separator_index == 1u) {
        return glyph_x(local, scale);
    }
    if (separator_index == 2u) {
        return false;
    }

    let height_index = separator_index - 3u;
    if (height_index < height_digit_count) {
        return seven_segment_digit(
            local,
            decimal_digit(source_height, height_digit_count, height_index),
            scale
        );
    }
    return false;
}

fn fps_overlay_line_alpha(local: vec2<f32>, char_count: u32, digit_count: u32, whole: u32, fraction: u32, scale: f32) -> f32 {
    let glyph_w = 8.0 * scale;
    let gap = 2.0 * scale;
    let block_w = f32(char_count) * glyph_w + f32(char_count - 1u) * gap;
    if (local.x < 0.0 || local.y < 0.0 || local.x >= block_w || local.y > 14.0 * scale) {
        return 0.0;
    }
    let cell_w = glyph_w + gap;
    let char_index = u32(floor(local.x / cell_w));
    if (char_index >= char_count) {
        return 0.0;
    }
    let char_local = vec2<f32>(local.x - f32(char_index) * cell_w, local.y);
    if (char_local.x > glyph_w) {
        return 0.0;
    }
    if (fps_overlay_char(char_local, char_index, digit_count, whole, fraction, scale)) {
        return 1.0;
    }
    return 0.0;
}

fn resolution_overlay_line_alpha(
    local: vec2<f32>,
    char_count: u32,
    width_digit_count: u32,
    height_digit_count: u32,
    source_width: u32,
    source_height: u32,
    scale: f32
) -> f32 {
    let glyph_w = 8.0 * scale;
    let gap = 2.0 * scale;
    let block_w = f32(char_count) * glyph_w + f32(char_count - 1u) * gap;
    if (local.x < 0.0 || local.y < 0.0 || local.x >= block_w || local.y > 14.0 * scale) {
        return 0.0;
    }
    let cell_w = glyph_w + gap;
    let char_index = u32(floor(local.x / cell_w));
    if (char_index >= char_count) {
        return 0.0;
    }
    let char_local = vec2<f32>(local.x - f32(char_index) * cell_w, local.y);
    if (char_local.x > glyph_w) {
        return 0.0;
    }
    if (resolution_overlay_char(
        char_local,
        char_index,
        width_digit_count,
        height_digit_count,
        source_width,
        source_height,
        scale
    )) {
        return 1.0;
    }
    return 0.0;
}

fn fps_overlay_alpha(uv: vec2<f32>) -> f32 {
    if (
        params.fps_overlay_enabled == 0u ||
        params.content_width <= 0.0 ||
        params.content_height <= 0.0 ||
        params.source_width == 0u ||
        params.source_height == 0u
    ) {
        return 0.0;
    }
    let scale = max(1.0, params.content_height / 1080.0);
    let glyph_w = 8.0 * scale;
    let gap = 2.0 * scale;
    let margin = 8.0 * scale;
    let line_h = 14.0 * scale;
    let fps_x100 = min(999999u, params.fps_x100);
    let whole = min(9999u, fps_x100 / 100u);
    let fraction = fps_x100 % 100u;
    let digit_count = select(select(select(1u, 2u, whole >= 10u), 3u, whole >= 100u), 4u, whole >= 1000u);
    let fps_char_count = digit_count + 7u;
    let fps_block_w = f32(fps_char_count) * glyph_w + f32(fps_char_count - 1u) * gap;
    let width_digit_count = decimal_digit_count(params.source_width);
    let height_digit_count = decimal_digit_count(params.source_height);
    let resolution_char_count = width_digit_count + 3u + height_digit_count;
    let resolution_block_w = f32(resolution_char_count) * glyph_w + f32(resolution_char_count - 1u) * gap;
    let p = uv * vec2<f32>(params.content_width, params.content_height);
    let resolution_origin = vec2<f32>(params.content_width - margin - resolution_block_w, margin);
    let resolution_alpha = resolution_overlay_line_alpha(
        p - resolution_origin,
        resolution_char_count,
        width_digit_count,
        height_digit_count,
        params.source_width,
        params.source_height,
        scale
    );
    if (resolution_alpha > 0.0) {
        return resolution_alpha;
    }

    let fps_origin = vec2<f32>(
        params.content_width - margin - fps_block_w,
        margin + 2.0 * (line_h + gap),
    );
    return fps_overlay_line_alpha(
        p - fps_origin,
        fps_char_count,
        digit_count,
        whole,
        fraction,
        scale
    );
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var rgb = textureSample(preview_texture, preview_sampler, in.uv).rgb;
    if (params.srgb_decode != 0u) {
        rgb = srgb_eotf(rgb);
    }
    if (params.srgb_encode != 0u) {
        rgb = srgb_oetf(rgb);
    }
    let overlay_alpha = fps_overlay_alpha(in.uv);
    if (overlay_alpha > 0.0) {
        rgb = mix(rgb, vec3<f32>(1.0), 0.9);
    }
    return vec4<f32>(rgb, 1.0);
}
"#;
