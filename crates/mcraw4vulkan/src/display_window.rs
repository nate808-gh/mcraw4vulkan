use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use mcraw4vulkan_core::{BayerPattern, FrameDimensions, FrameNumber, FramePayloadLayout};
use mcraw4vulkan_cpu::CpuFrameDecoder;
use mcraw4vulkan_display::{
    DisplayFpsMeter, DisplayFpsOverlay, DisplayFramePacer, DisplayPresentInput, DisplayPresentMode,
    DisplayPresentTarget, DisplayScaleMode, DisplaySize, DisplaySourceTransfer,
    DisplayTexturePresenter, DisplayTransferStrategy, fit_aspect_preserving,
};
use mcraw4vulkan_display_sound::{DisplaySoundError, DisplaySoundPlan, DisplaySoundSession};
use mcraw4vulkan_gpu::{GpuDecodeBackend, OptionalGpuVignetteCorrection};
use mcraw4vulkan_mcrawcontainer::{
    ColorIlluminant, ContainerMetadata, FrameMetadata, LensShadingMap, McrawContainer,
    SensorArrangement,
    payload_reader::{
        PayloadFeeder, PayloadFeederOptions, PayloadFeederPoll, PayloadFrame, PayloadReadPlan,
    },
};
use mcraw4vulkan_render::{
    DisplayRgbTonePolicy, GpuDisplayP999Histogram, GpuP999HistogramConfig, GpuP999HistogramInput,
    GpuP999HistogramParams, GpuP999HistogramResult, GpuPreviewRenderEncodeOutput,
    GpuPreviewTextureRenderer, GpuRenderCalibrationIlluminant, GpuRenderColorMetadata,
    GpuRenderColorMode, GpuRenderColorParams, GpuRgbSinkGuard, GpuRgbSinkHighlightDesat,
    PreviewRenderConfig, PreviewRgbSinkPolicy, PreviewScaleMode, PreviewTextureFormat,
    PreviewTransferMode, RenderSampleDomain,
};
use mcraw4vulkan_sdl2_wgpu_surface::{
    RenderFrameContext, RenderFrameStatus, Sdl2WgpuSurface, Sdl2WgpuSurfaceConfig, WindowSize,
};
use mcraw4vulkan_vignette::{
    FixedPointVignetteInputFacts, GpuUploadedFullResolutionGainMap, GpuVignetteCorrectionParams,
    GpuVignetteCorrector, GpuVignettePackedU16DispatchInput, PreparedFixedLensShadingMap,
    VignetteCoordinateMapping, VignetteCorrectionInputFacts, VignetteCorrectionMode,
    VignetteGainMapFingerprint,
};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;

const DEFAULT_DISPLAY_FRAME_RATE_FPS: f64 = 24.0;
pub const DEFAULT_STANDALONE_DISPLAY_WINDOW_WIDTH: u32 = 1280;
pub const DEFAULT_STANDALONE_DISPLAY_WINDOW_HEIGHT: u32 = 720;
const EMBEDDED_PREVIEW_PAYLOAD_RETRY_INTERVAL: Duration = Duration::from_millis(8);
const EMBEDDED_PREVIEW_MAX_SPEED_REPAINT_INTERVAL: Duration = Duration::from_millis(4);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayCliBackend {
    Gpu,
    Cpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayCliVignette {
    NoCorrection,
    WithCorrection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayCliVsync {
    Vsync,
    NoVsync,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayCliSettings {
    Default,
    Optimized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayCliOverlay {
    WithOverlay,
    NoOverlay,
}

impl DisplayCliOverlay {
    fn enabled(self) -> bool {
        self == Self::WithOverlay
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayCliSound {
    Silent,
    WithSound,
}

impl DisplayCliSound {
    fn enabled(self) -> bool {
        self == Self::WithSound
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplayCliRunConfig {
    pub input_path: PathBuf,
    pub backend: DisplayCliBackend,
    pub vignette: DisplayCliVignette,
    pub vsync: DisplayCliVsync,
    pub settings: DisplayCliSettings,
    pub overlay: DisplayCliOverlay,
    pub sound: DisplayCliSound,
    pub payload_feeder_options: PayloadFeederOptions,
}

impl DisplayCliRunConfig {
    pub fn production_default(input_path: impl Into<PathBuf>) -> Self {
        Self {
            input_path: input_path.into(),
            backend: DisplayCliBackend::Gpu,
            vignette: DisplayCliVignette::NoCorrection,
            vsync: DisplayCliVsync::Vsync,
            settings: DisplayCliSettings::Default,
            overlay: DisplayCliOverlay::WithOverlay,
            sound: DisplayCliSound::Silent,
            payload_feeder_options: PayloadFeederOptions::production_default(),
        }
    }

    pub fn validate_before_open(&self) -> Result<()> {
        if self.sound.enabled() && self.vsync == DisplayCliVsync::NoVsync {
            bail!("--with-sound requires Vsync display and cannot be used with --no-vsync");
        }

        Ok(())
    }

    fn present_mode(&self) -> DisplayPresentMode {
        match self.vsync {
            DisplayCliVsync::Vsync => DisplayPresentMode::Fifo,
            DisplayCliVsync::NoVsync => DisplayPresentMode::AutoNoVsync,
        }
    }

    fn playback_mode(&self) -> DisplayPlaybackMode {
        match self.vsync {
            DisplayCliVsync::Vsync => DisplayPlaybackMode::SourcePaced,
            DisplayCliVsync::NoVsync => DisplayPlaybackMode::MaxSpeed,
        }
    }

    pub fn window_title(&self) -> String {
        let basename = self
            .input_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("mcraw clip");
        format!("mcraw4vulkan display - {basename}")
    }
}

pub fn apply_effective_display_settings(config: &mut DisplayCliRunConfig) -> Option<String> {
    let effective = match config.settings {
        DisplayCliSettings::Default => mcraw4vulkan_optimizer::built_in_default_settings(),
        DisplayCliSettings::Optimized => mcraw4vulkan_optimizer::resolve_effective_settings(
            mcraw4vulkan_optimizer::SettingsSourceSelection::Optimized,
        ),
    };
    apply_effective_optimizer_settings(config, &effective);
    effective.warning().map(ToString::to_string)
}

fn apply_effective_optimizer_settings(
    config: &mut DisplayCliRunConfig,
    effective: &mcraw4vulkan_optimizer::EffectiveOptimizerSettings,
) {
    config.payload_feeder_options = effective.payload_profile.payload_feeder_options();
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DisplayPlaybackMode {
    SourcePaced,
    MaxSpeed,
}

impl DisplayPlaybackMode {
    fn source_paced_playback(self) -> bool {
        self == Self::SourcePaced
    }

    fn create_frame_pacer(
        self,
        source_frame_rate_fps: f64,
        start_instant: Instant,
        start_frame_index: u64,
    ) -> Option<DisplayFramePacer> {
        if self.source_paced_playback() {
            DisplayFramePacer::new(source_frame_rate_fps, start_instant, start_frame_index)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DisplayPreviewResolutionMode {
    WindowedPhysical { canvas_size: DisplaySize },
    FullscreenPhysical { canvas_size: DisplaySize },
}

impl DisplayPreviewResolutionMode {
    fn windowed(canvas_size: DisplaySize) -> Self {
        Self::WindowedPhysical { canvas_size }
    }

    fn fullscreen(canvas_size: DisplaySize) -> Self {
        Self::FullscreenPhysical { canvas_size }
    }

    fn resized(self, size: PhysicalSize<u32>) -> Self {
        match (self, display_size_from_physical(size)) {
            (Self::WindowedPhysical { .. }, Some(canvas_size)) => Self::windowed(canvas_size),
            (Self::FullscreenPhysical { .. }, Some(canvas_size)) => Self::fullscreen(canvas_size),
            _ => self,
        }
    }

    fn preview_scale_mode(self, source_dimensions: FrameDimensions) -> PreviewScaleMode {
        let target_dimensions = match self {
            Self::WindowedPhysical { canvas_size } => {
                preview_target_dimensions_for_canvas(source_dimensions, canvas_size)
            }
            Self::FullscreenPhysical { canvas_size } => {
                fullscreen_preview_target_dimensions(source_dimensions, canvas_size)
            }
        };
        target_dimensions
            .map(|dimensions| PreviewScaleMode::Explicit {
                width: dimensions.width,
                height: dimensions.height,
            })
            .unwrap_or(PreviewScaleMode::FullResolution)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PhysicalSize<T> {
    width: T,
    height: T,
}

impl<T> PhysicalSize<T> {
    fn new(width: T, height: T) -> Self {
        Self { width, height }
    }
}

fn display_size_from_physical(size: PhysicalSize<u32>) -> Option<DisplaySize> {
    if size.width == 0 || size.height == 0 {
        return None;
    }
    Some(DisplaySize {
        width: size.width,
        height: size.height,
    })
}

fn preview_target_dimensions_for_canvas(
    source_dimensions: FrameDimensions,
    canvas_size: DisplaySize,
) -> Option<FrameDimensions> {
    if source_dimensions.width == 0
        || source_dimensions.height == 0
        || canvas_size.width == 0
        || canvas_size.height == 0
    {
        return None;
    }

    let viewport = fit_aspect_preserving(
        canvas_size.width,
        canvas_size.height,
        source_dimensions.width,
        source_dimensions.height,
    );
    if viewport.content_width == 0 || viewport.content_height == 0 {
        return None;
    }

    Some(FrameDimensions {
        width: viewport.content_width.min(source_dimensions.width).max(1),
        height: viewport.content_height.min(source_dimensions.height).max(1),
    })
}

fn fullscreen_preview_target_dimensions(
    source_dimensions: FrameDimensions,
    canvas_size: DisplaySize,
) -> Option<FrameDimensions> {
    preview_target_dimensions_for_canvas(source_dimensions, canvas_size)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayKeyboardAction {
    None,
    ToggleFullscreen,
    Exit,
}

pub(crate) fn display_keyboard_action(
    keycode: Option<Keycode>,
    pressed: bool,
    repeat: bool,
) -> DisplayKeyboardAction {
    if !pressed || repeat {
        return DisplayKeyboardAction::None;
    }

    match keycode {
        Some(Keycode::Escape) => DisplayKeyboardAction::Exit,
        Some(Keycode::F) => DisplayKeyboardAction::ToggleFullscreen,
        _ => DisplayKeyboardAction::None,
    }
}

// The SDL event pump, surface operations, and frame submission remain on this
// thread; fullscreen and resize changes are applied by the same event loop.
pub fn run_display_window(config: DisplayCliRunConfig) -> Result<()> {
    config.validate_before_open()?;

    let mut surface = Sdl2WgpuSurface::new(display_surface_config(&config))?;
    let mut event_pump = surface.event_pump()?;
    let initial_size = display_size_from_window_size(surface.size()).unwrap_or(DisplaySize {
        width: DEFAULT_STANDALONE_DISPLAY_WINDOW_WIDTH,
        height: DEFAULT_STANDALONE_DISPLAY_WINDOW_HEIGHT,
    });
    let mut state = DisplayWindowState::create(
        config,
        DisplayWindowCreateContext {
            adapter_info: surface.adapter_info().clone(),
            device: surface.device().clone(),
            queue: surface.queue().clone(),
            surface_format: surface.surface_format(),
            display_size: initial_size,
        },
    )?;
    let mut fullscreen = false;
    let mut running = true;

    while running {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => running = false,
                Event::Window {
                    win_event: WindowEvent::Close,
                    ..
                } => running = false,
                Event::KeyDown {
                    keycode, repeat, ..
                } => match display_keyboard_action(keycode, true, repeat) {
                    DisplayKeyboardAction::Exit => running = false,
                    DisplayKeyboardAction::ToggleFullscreen => {
                        fullscreen = !fullscreen;
                        surface.set_fullscreen_desktop(fullscreen)?;
                        state.set_fullscreen(fullscreen);
                    }
                    DisplayKeyboardAction::None => {}
                },
                _ => {}
            }
        }

        if !running {
            break;
        }

        let mut progress = None;
        let status = surface.render_frame(|frame| {
            progress = Some(state.render_next_frame(frame));
        })?;
        match status {
            RenderFrameStatus::Submitted { .. } => {
                if let Some(progress) = progress {
                    let progress = progress?;
                    state.after_surface_submit()?;
                    match progress {
                        DisplayWindowProgress::Continue { .. } => {}
                        DisplayWindowProgress::Waiting { .. } => {}
                        DisplayWindowProgress::Finished => running = false,
                    }
                } else {
                    state.after_surface_submit()?;
                }
            }
            RenderFrameStatus::SurfaceChanged => {}
            RenderFrameStatus::SkippedZeroSize | RenderFrameStatus::Timeout => {
                std::thread::sleep(Duration::from_millis(8));
            }
        }
    }

    state.finish()
}

fn display_surface_config(config: &DisplayCliRunConfig) -> Sdl2WgpuSurfaceConfig {
    Sdl2WgpuSurfaceConfig::new(
        config.window_title(),
        DEFAULT_STANDALONE_DISPLAY_WINDOW_WIDTH,
        DEFAULT_STANDALONE_DISPLAY_WINDOW_HEIGHT,
    )
    .with_preferred_present_mode(config.present_mode().to_wgpu())
}

fn display_size_from_window_size(size: WindowSize) -> Option<DisplaySize> {
    if size.drawable_width == 0 || size.drawable_height == 0 {
        return None;
    }
    Some(DisplaySize {
        width: size.drawable_width,
        height: size.drawable_height,
    })
}

enum DisplayWindowProgress {
    Continue { repaint_after: Duration },
    Waiting { repaint_after: Duration },
    Finished,
}

struct DisplayWindowFrameResult {
    progress: DisplayWindowProgress,
    timings: EmbeddedDisplayPreviewFrameTimings,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedDisplayPreviewProgress {
    Continue { repaint_after: Duration },
    Finished,
}

impl EmbeddedDisplayPreviewProgress {
    pub fn repaint_after(self) -> Option<Duration> {
        match self {
            Self::Continue { repaint_after } => Some(repaint_after),
            Self::Finished => None,
        }
    }
}

fn embedded_progress_from_display(
    progress: DisplayWindowProgress,
) -> EmbeddedDisplayPreviewProgress {
    match progress {
        DisplayWindowProgress::Continue { repaint_after }
        | DisplayWindowProgress::Waiting { repaint_after } => {
            EmbeddedDisplayPreviewProgress::Continue { repaint_after }
        }
        DisplayWindowProgress::Finished => EmbeddedDisplayPreviewProgress::Finished,
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EmbeddedDisplayPreviewFrameTimings {
    pub new_video_frame: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct EmbeddedDisplayPreviewFrameResult {
    pub progress: EmbeddedDisplayPreviewProgress,
    pub timings: EmbeddedDisplayPreviewFrameTimings,
}

enum DisplayPayloadFramePoll {
    Ready(PayloadFrame),
    Pending,
    Finished,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddedDisplayPreviewTarget {
    pub origin_x: u32,
    pub origin_y: u32,
    pub width: u32,
    pub height: u32,
}

impl EmbeddedDisplayPreviewTarget {
    pub fn new(origin_x: u32, origin_y: u32, width: u32, height: u32) -> Option<Self> {
        (width != 0 && height != 0).then_some(Self {
            origin_x,
            origin_y,
            width,
            height,
        })
    }

    fn display_size(self) -> DisplaySize {
        DisplaySize {
            width: self.width,
            height: self.height,
        }
    }

    fn present_target(self) -> Result<DisplayPresentTarget> {
        DisplayPresentTarget::embedded(self.origin_x, self.origin_y, self.width, self.height)
            .map_err(|error| anyhow!("invalid embedded preview target: {error}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddedDisplayPreviewPosition {
    pub current_frame_index: Option<usize>,
    pub frame_count: usize,
}

// This value owns the payload feeder and render resources across frames.
// Cooperative shutdown completes only after poll_shutdown joins the feeder.
pub struct EmbeddedDisplayPreview {
    state: DisplayWindowState,
}

pub struct EmbeddedDisplayPreviewPrepared {
    prepared: DisplayWindowPreparedState,
}

impl EmbeddedDisplayPreviewPrepared {
    #[rustfmt::skip]
    pub fn prepare(config: DisplayCliRunConfig) -> Result<Self> {
        Self::prepare_cancellable(config, || false)
    }

    #[rustfmt::skip]
    pub fn prepare_cancellable(
        config: DisplayCliRunConfig,
        cancelled: impl Fn() -> bool,
    ) -> Result<Self> {
        config.validate_before_open()?;
        if config.sound.enabled() {
            bail!("display sound is only supported by standalone CLI display");
        }
        Ok(Self {
            prepared: DisplayWindowPreparedState::prepare_with_cancel(
                config,
                cancelled,
            )?,
        })
    }

    pub fn prepare_with_sound_plan(
        config: DisplayCliRunConfig,
    ) -> Result<(Self, Result<DisplaySoundPlan, DisplaySoundError>)> {
        Self::prepare_with_sound_plan_cancellable(config, || false)
    }

    #[rustfmt::skip]
    pub fn prepare_with_sound_plan_cancellable(
        config: DisplayCliRunConfig,
        cancelled: impl Fn() -> bool,
    ) -> Result<(Self, Result<DisplaySoundPlan, DisplaySoundError>)> {
        config.validate_before_open()?;
        if !config.sound.enabled() {
            bail!("embedded display sound preparation requires sound-enabled config");
        }
        let prepared = DisplayWindowPreparedState::prepare_with_cancel(
                config,
                cancelled,
            )?;
        let sound_plan = DisplaySoundPlan::from_container(&prepared.source.container);
        Ok((Self { prepared }, sound_plan))
    }
}

impl EmbeddedDisplayPreview {
    #[rustfmt::skip]
    pub fn create(
        config: DisplayCliRunConfig,
        adapter_info: wgpu::AdapterInfo,
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        target: EmbeddedDisplayPreviewTarget,
    ) -> Result<Self> {
        let prepared = EmbeddedDisplayPreviewPrepared::prepare(config)?;
        Self::from_prepared(
            prepared,
            adapter_info,
            device,
            queue,
            surface_format,
            target,
        )
    }

    #[rustfmt::skip]
    pub fn from_prepared(
        prepared: EmbeddedDisplayPreviewPrepared,
        adapter_info: wgpu::AdapterInfo,
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        target: EmbeddedDisplayPreviewTarget,
    ) -> Result<Self> {
        let state = DisplayWindowState::create_from_prepared(
            prepared.prepared,
            DisplayWindowCreateContext {
                adapter_info,
                device,
                queue,
                surface_format,
                display_size: target.display_size(),
            },
            None,
        )?;
        Ok(Self { state })
    }

    pub fn resize(&mut self, target: EmbeddedDisplayPreviewTarget) {
        self.state.resize(target.display_size());
    }

    pub fn render_next_frame(
        &mut self,
        target: RenderFrameContext<'_>,
        preview_target: EmbeddedDisplayPreviewTarget,
    ) -> Result<EmbeddedDisplayPreviewProgress> {
        Ok(self
            .render_next_frame_with_timings(target, preview_target)?
            .progress)
    }

    pub fn render_next_frame_with_timings(
        &mut self,
        target: RenderFrameContext<'_>,
        preview_target: EmbeddedDisplayPreviewTarget,
    ) -> Result<EmbeddedDisplayPreviewFrameResult> {
        self.state.resize(preview_target.display_size());
        let result = self
            .state
            .render_next_frame_to_cooperative(target, preview_target.present_target()?)?;
        Ok(EmbeddedDisplayPreviewFrameResult {
            progress: embedded_progress_from_display(result.progress),
            timings: result.timings,
        })
    }

    pub fn present_current_frame(
        &mut self,
        target: RenderFrameContext<'_>,
        preview_target: EmbeddedDisplayPreviewTarget,
    ) -> Result<EmbeddedDisplayPreviewProgress> {
        Ok(self
            .present_current_frame_with_timings(target, preview_target)?
            .progress)
    }

    pub fn present_current_frame_with_timings(
        &mut self,
        target: RenderFrameContext<'_>,
        preview_target: EmbeddedDisplayPreviewTarget,
    ) -> Result<EmbeddedDisplayPreviewFrameResult> {
        self.state.resize(preview_target.display_size());
        self.state
            .present_existing_preview_if_available(target, preview_target.present_target()?)?;
        let repaint_after = self.state.cooperative_repaint_after(Instant::now());
        Ok(EmbeddedDisplayPreviewFrameResult {
            progress: EmbeddedDisplayPreviewProgress::Continue { repaint_after },
            timings: EmbeddedDisplayPreviewFrameTimings {
                ..EmbeddedDisplayPreviewFrameTimings::default()
            },
        })
    }

    pub fn request_shutdown(&mut self) {
        self.state.request_shutdown();
    }

    pub fn poll_shutdown(&mut self) -> Result<bool> {
        self.state.poll_shutdown()
    }

    pub fn resume_after_pause(&mut self) {
        self.state.resume_after_pause();
    }

    pub fn playback_position(&self) -> EmbeddedDisplayPreviewPosition {
        self.state.playback_position()
    }

    pub fn seek_to_frame_index(&mut self, frame_index: usize) -> Result<()> {
        self.state.seek_to_frame_index(frame_index)
    }

    pub fn source_frame_duration(&self) -> Option<Duration> {
        self.state.source.source_frame_duration()
    }

    pub fn finish(&mut self) -> Result<()> {
        self.state.finish()
    }
}

struct DisplayWindowState {
    config: DisplayCliRunConfig,
    display_size: DisplaySize,
    backend: GpuDecodeBackend,
    preview_renderer: GpuPreviewTextureRenderer,
    display_histogram: Option<GpuDisplayP999Histogram>,
    preview_uploader: Option<CpuPreviewUploadBuffer>,
    presenter: DisplayTexturePresenter,
    source: DisplayWindowSource,
    fps_meter: DisplayFpsMeter,
    vignette_corrector: Option<GpuVignetteCorrector>,
    gain_map: Option<UploadedGainMapCache>,
    preview_resolution_mode: DisplayPreviewResolutionMode,
    current_frame_index: Option<usize>,
    presented_frames: u64,
    sound: Option<DisplaySoundSession>,
}

struct DisplayWindowPreparedState {
    config: DisplayCliRunConfig,
    source: DisplayWindowPreparedSource,
}

struct DisplayWindowCreateContext {
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_format: wgpu::TextureFormat,
    display_size: DisplaySize,
}

#[derive(Default)]
struct CpuPreviewUploadBuffer {
    buffer: Option<wgpu::Buffer>,
    byte_len: u64,
}

impl CpuPreviewUploadBuffer {
    fn ensure(&mut self, device: &wgpu::Device, byte_len: u64) -> &wgpu::Buffer {
        let needs_allocation = self.buffer.is_none() || self.byte_len < byte_len;
        if needs_allocation {
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mcraw4vulkan display CPU preview upload buffer"),
                size: byte_len.max(1),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.byte_len = byte_len;
        }
        self.buffer
            .as_ref()
            .expect("CPU preview upload buffer was just ensured")
    }
}

#[derive(Clone)]
struct UploadedGainMapCache {
    fingerprint: VignetteGainMapFingerprint,
    uploaded: GpuUploadedFullResolutionGainMap,
}

#[derive(Debug)]
struct PreviewStageOutput {
    encoded: GpuPreviewRenderEncodeOutput,
}

#[derive(Debug)]
struct DecodedDisplayStage {
    buffer: wgpu::Buffer,
    byte_len: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DisplayPreviewTonePolicy {
    enabled: bool,
    histogram_config: GpuP999HistogramConfig,
    rgb_policy: DisplayRgbTonePolicy,
    histogram_extra_scale: f32,
}

impl DisplayPreviewTonePolicy {
    fn for_vignette(vignette: DisplayCliVignette) -> Self {
        let enabled = vignette == DisplayCliVignette::WithCorrection;
        let rgb_policy = DisplayRgbTonePolicy::canonical();
        Self {
            enabled,
            histogram_config: GpuP999HistogramConfig::canonical(),
            rgb_policy,
            histogram_extra_scale: rgb_policy.extra_scale(enabled),
        }
    }

    fn preview_adjustments(
        self,
        histogram: Option<GpuP999HistogramResult>,
    ) -> DisplayPreviewToneAdjustments {
        let Some(histogram) = histogram else {
            return DisplayPreviewToneAdjustments::none();
        };
        if !self.enabled {
            return DisplayPreviewToneAdjustments::none();
        }

        DisplayPreviewToneAdjustments {
            tone_scale: histogram.final_scale,
            guard: self
                .rgb_policy
                .render_guard((histogram.luma_p999 * histogram.final_scale).max(0.0)),
            desat: self.rgb_policy.highlight_desat(
                true,
                histogram.luma_p999,
                histogram.final_scale,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DisplayPreviewToneAdjustments {
    tone_scale: f32,
    guard: GpuRgbSinkGuard,
    desat: GpuRgbSinkHighlightDesat,
}

impl DisplayPreviewToneAdjustments {
    fn none() -> Self {
        Self {
            tone_scale: 1.0,
            guard: GpuRgbSinkGuard::none(),
            desat: GpuRgbSinkHighlightDesat::none(),
        }
    }

    fn preview_rgb_sink_policy(self) -> PreviewRgbSinkPolicy {
        PreviewRgbSinkPolicy {
            rgb_sink_tone_scale: self.tone_scale,
            rgb_sink_guard: self.guard,
            rgb_sink_desat: self.desat,
        }
    }
}

fn encode_preview_stage(
    preview: &mut GpuPreviewTextureRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    input_buffer: &wgpu::Buffer,
    input_buffer_bytes: u64,
    preview_config: PreviewRenderConfig,
) -> Result<PreviewStageOutput> {
    let encoded = preview
        .encode_render_to_texture(
            device,
            queue,
            encoder,
            input_buffer,
            input_buffer_bytes,
            preview_config,
        )
        .map_err(|error| anyhow!("failed to encode display preview texture: {error}"))?;
    Ok(PreviewStageOutput { encoded })
}

fn display_source_frame_rate_fps(metadata_frame_rate_fps: Option<f64>) -> f64 {
    metadata_frame_rate_fps
        .filter(|fps| fps.is_finite() && *fps > 0.0)
        .unwrap_or(DEFAULT_DISPLAY_FRAME_RATE_FPS)
}

fn display_fps_overlay_for_present(
    overlay: DisplayCliOverlay,
    fps_meter: DisplayFpsMeter,
    source_frame_rate_fps: f64,
    source_size: DisplaySize,
) -> Option<DisplayFpsOverlay> {
    if !overlay.enabled() {
        return None;
    }
    fps_meter
        .fps_for_overlay(source_frame_rate_fps)
        .and_then(DisplayFpsOverlay::from_fps)
        .map(|overlay| overlay.with_source_size(source_size))
}

impl DisplayWindowPreparedState {
    #[rustfmt::skip]
    fn prepare(
        config: DisplayCliRunConfig,
    ) -> Result<Self> {
        Self::prepare_with_cancel(
            config,
            || false,
        )
    }

    #[rustfmt::skip]
    fn prepare_with_cancel(
        config: DisplayCliRunConfig,
        cancelled: impl Fn() -> bool,
    ) -> Result<Self> {
        let container = match (
            config.sound.enabled(),
        ) {
            (
                true,
            ) => McrawContainer::open_for_display_with_audio(&config.input_path),
            (
                false,
            ) => McrawContainer::open_for_display(&config.input_path),
        }
        .with_context(|| format!("opening {}", config.input_path.display()))?;
        container.prepare_geometry_with_cancel(cancelled)?;

        let source = DisplayWindowPreparedSource::prepare(
            &config.input_path,
            container,
            config.playback_mode(),
            config.payload_feeder_options,
        )?;

        Ok(Self {
            config,
            source,
        })
    }
}

fn prepare_display_sound(
    config: &DisplayCliRunConfig,
    container: &McrawContainer,
) -> Result<Option<DisplaySoundSession>> {
    if !config.sound.enabled() {
        return Ok(None);
    }

    let sound = DisplaySoundSession::from_container(container)
        .context("preparing display audio for --with-sound")?;
    Ok(Some(sound))
}

impl DisplayWindowState {
    #[rustfmt::skip]
    fn create(
        config: DisplayCliRunConfig,
        context: DisplayWindowCreateContext,
    ) -> Result<Self> {
        let prepared = DisplayWindowPreparedState::prepare(
            config,
        )?;
        let sound = prepare_display_sound(
            &prepared.config,
            &prepared.source.container,
        )?;
        Self::create_from_prepared(
            prepared, context,
            sound,
        )
    }

    #[rustfmt::skip]
    fn create_from_prepared(
        prepared: DisplayWindowPreparedState,
        context: DisplayWindowCreateContext,
        sound: Option<DisplaySoundSession>,
    ) -> Result<Self> {
        let DisplayWindowPreparedState {
            config,
            source,
        } = prepared;
        let DisplayWindowCreateContext {
            adapter_info,
            device,
            queue,
            surface_format,
            display_size,
        } = context;

        let source = DisplayWindowSource::from_prepared(
            source,
        )?;
        let backend = GpuDecodeBackend::from_wgpu_device(
            adapter_info,
            device,
            queue,
        )?;
        let preview_renderer = GpuPreviewTextureRenderer::new(backend.device())
            .map_err(|error| anyhow!("creating display preview renderer: {error}"))?;
        let display_histogram = if config.vignette == DisplayCliVignette::WithCorrection {
            Some(
                GpuDisplayP999Histogram::new(backend.device())
                    .map_err(|error| anyhow!("creating display p999 histogram: {error}"))?,
            )
        } else {
            None
        };
        let preview_uploader = match config.backend {
            DisplayCliBackend::Cpu => Some(CpuPreviewUploadBuffer::default()),
            DisplayCliBackend::Gpu => None,
        };

        let presenter = DisplayTexturePresenter::new(
            backend.device(),
            surface_format,
            DisplayScaleMode::FitContain,
        );
        let vignette_corrector = if config.vignette == DisplayCliVignette::WithCorrection {
            Some(backend.create_vignette_corrector()?)
        } else {
            None
        };

        eprintln!(
            "displaying {} with {} frames",
            config.input_path.display(),
            source.frame_count()
        );

        Ok(Self {
            config,
            display_size,
            backend,
            preview_renderer,
            display_histogram,
            preview_uploader,
            presenter,
            source,
            fps_meter: DisplayFpsMeter::new(Instant::now()),
            vignette_corrector,
            gain_map: None,
            preview_resolution_mode: DisplayPreviewResolutionMode::windowed(display_size),
            current_frame_index: None,
            presented_frames: 0,
            sound,
        })
    }

    fn resize(&mut self, size: DisplaySize) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.display_size = size;
        self.preview_resolution_mode = self
            .preview_resolution_mode
            .resized(PhysicalSize::new(size.width, size.height));
    }

    fn set_fullscreen(&mut self, fullscreen: bool) {
        if fullscreen {
            self.preview_resolution_mode =
                DisplayPreviewResolutionMode::fullscreen(self.display_size);
        } else {
            self.preview_resolution_mode =
                DisplayPreviewResolutionMode::windowed(self.display_size);
        }
    }

    fn render_next_frame(
        &mut self,
        target: RenderFrameContext<'_>,
    ) -> Result<DisplayWindowProgress> {
        if let Some(size) = display_size_from_window_size(target.size) {
            self.resize(size);
        }

        self.render_next_frame_to(target, DisplayPresentTarget::full(self.display_size))
    }

    fn render_next_frame_to(
        &mut self,
        target: RenderFrameContext<'_>,
        present_target: DisplayPresentTarget,
    ) -> Result<DisplayWindowProgress> {
        self.resize(present_target.size);

        let frame = self.source.next_payload_frame()?;
        let Some(frame) = frame else {
            return Ok(DisplayWindowProgress::Finished);
        };

        let displayed_frame_index = self.source.next_frame_index;
        let facts = self.source.frame_facts(frame.frame_number_core()?)?;
        match self.config.backend {
            DisplayCliBackend::Gpu => self.render_gpu_frame(&facts, &frame)?,
            DisplayCliBackend::Cpu => self.render_cpu_frame(&facts, &frame)?,
        }
        self.source.advance_after_frame_blocking();
        self.present_preview_to(target, present_target, true)?;
        self.current_frame_index = Some(displayed_frame_index);
        self.presented_frames += 1;

        Ok(DisplayWindowProgress::Continue {
            repaint_after: Duration::ZERO,
        })
    }

    #[rustfmt::skip]
    fn render_next_frame_to_cooperative(
        &mut self,
        target: RenderFrameContext<'_>,
        present_target: DisplayPresentTarget,
    ) -> Result<DisplayWindowFrameResult> {
        self.resize(present_target.size);

        let now = Instant::now();
        let displayed_frame_index = self.source.next_frame_index;
        if let Some(repaint_after) = self.source.repaint_after_until_next_frame(now) {
            self.present_existing_preview_if_available(target, present_target)?;
            return Ok(DisplayWindowFrameResult {
                progress: DisplayWindowProgress::Waiting { repaint_after },
                timings: EmbeddedDisplayPreviewFrameTimings {
                    ..EmbeddedDisplayPreviewFrameTimings::default()
                },
            });
        }

        let frame = match self.source.poll_next_payload_frame()? {
            DisplayPayloadFramePoll::Ready(frame) => frame,
            DisplayPayloadFramePoll::Pending => {
                self.present_existing_preview_if_available(target, present_target)?;
                return Ok(DisplayWindowFrameResult {
                    progress: DisplayWindowProgress::Waiting {
                        repaint_after: EMBEDDED_PREVIEW_PAYLOAD_RETRY_INTERVAL,
                    },
                    timings: EmbeddedDisplayPreviewFrameTimings {
                        ..EmbeddedDisplayPreviewFrameTimings::default()
                    },
                });
            }
            DisplayPayloadFramePoll::Finished => {
                return Ok(DisplayWindowFrameResult {
                    progress: DisplayWindowProgress::Finished,
                    timings: EmbeddedDisplayPreviewFrameTimings {
                        ..EmbeddedDisplayPreviewFrameTimings::default()
                    },
                });
            }
        };

        let facts = self.source.frame_facts(frame.frame_number_core()?)?;
        match self.config.backend {
            DisplayCliBackend::Gpu => self.render_gpu_frame(&facts, &frame)?,
            DisplayCliBackend::Cpu => self.render_cpu_frame(&facts, &frame)?,
        }
        self.source.advance_after_frame_cooperative();
        self.present_preview_to(target, present_target, true)?;
        self.current_frame_index = Some(displayed_frame_index);
        self.presented_frames += 1;
        let repaint_after = self.cooperative_repaint_after(Instant::now());

        Ok(DisplayWindowFrameResult {
            progress: DisplayWindowProgress::Continue { repaint_after },
            timings: EmbeddedDisplayPreviewFrameTimings {
                new_video_frame: true,
            },
        })
    }

    fn cooperative_repaint_after(&self, now: Instant) -> Duration {
        self.source.cooperative_repaint_after(now)
    }

    fn after_surface_submit(&mut self) -> Result<()> {
        if let Some(sound) = self.sound.as_mut() {
            sound.start_after_first_video_submit()?;
            sound.pump()?;
        }
        Ok(())
    }

    fn render_gpu_frame(&mut self, facts: &DisplayFrameFacts, frame: &PayloadFrame) -> Result<()> {
        if self.config.vignette == DisplayCliVignette::WithCorrection {
            return self.render_gpu_frame_with_vignette(facts, frame);
        }

        let preview_config = facts.preview_config(
            DisplayCliVignette::NoCorrection,
            self.preview_resolution_mode
                .preview_scale_mode(facts.dimensions),
        )?;
        let backend = &mut self.backend;
        let preview = &mut self.preview_renderer;
        let output = match facts.payload_layout {
            FramePayloadLayout::CompressedRawcodecType7 => backend
                .decode_raw_payload_packed_u16_gpu_stage_no_readback_with_vignette(
                    frame.data.as_slice(),
                    facts.geometry.extraction(),
                    None,
                    |device, queue, encoder, decoded| {
                        encode_preview_stage(
                            preview,
                            device,
                            queue,
                            encoder,
                            decoded.buffer,
                            decoded.byte_len,
                            preview_config,
                        )
                    },
                )?,
            FramePayloadLayout::BinnedRaw16Type6 { row_stride } => backend
                .decode_legacy_raw16_payload_packed_u16_gpu_stage_no_readback_with_vignette(
                    frame.data.as_slice(),
                    facts.geometry.extraction(),
                    row_stride,
                    None,
                    |device, queue, encoder, decoded| {
                        encode_preview_stage(
                            preview,
                            device,
                            queue,
                            encoder,
                            decoded.buffer,
                            decoded.byte_len,
                            preview_config,
                        )
                    },
                )?,
        };
        let _ = output.stage.encoded.info;
        Ok(())
    }

    fn render_gpu_frame_with_vignette(
        &mut self,
        facts: &DisplayFrameFacts,
        frame: &PayloadFrame,
    ) -> Result<()> {
        let (uploaded_gain_map, params) = self.ensure_uploaded_gain_map(facts)?;
        let corrector = self
            .vignette_corrector
            .as_mut()
            .context("display vignette corrector missing")?;
        let correction = Some(OptionalGpuVignetteCorrection {
            corrector,
            uploaded_gain_map: &uploaded_gain_map,
            params,
        });
        let decode_output = match facts.payload_layout {
            FramePayloadLayout::CompressedRawcodecType7 => self
                .backend
                .decode_raw_payload_packed_u16_gpu_stage_no_readback_with_vignette(
                    frame.data.as_slice(),
                    facts.geometry.extraction(),
                    correction,
                    |_device, _queue, _encoder, decoded| {
                        Ok(DecodedDisplayStage {
                            buffer: decoded.buffer.clone(),
                            byte_len: decoded.byte_len,
                        })
                    },
                )?,
            FramePayloadLayout::BinnedRaw16Type6 { row_stride } => self
                .backend
                .decode_legacy_raw16_payload_packed_u16_gpu_stage_no_readback_with_vignette(
                    frame.data.as_slice(),
                    facts.geometry.extraction(),
                    row_stride,
                    correction,
                    |_device, _queue, _encoder, decoded| {
                        Ok(DecodedDisplayStage {
                            buffer: decoded.buffer.clone(),
                            byte_len: decoded.byte_len,
                        })
                    },
                )?,
        };
        self.render_corrected_frame(facts, &decode_output.stage)
    }

    // Both RAW backends feed the same corrected camera-domain buffer and the
    // same histogram/tone/preview path. CPU RAW decoding still uses GPU spatial
    // correction after its existing upload; it does not run a scalar corrector.
    fn render_corrected_frame(
        &mut self,
        facts: &DisplayFrameFacts,
        decoded: &DecodedDisplayStage,
    ) -> Result<()> {
        let tone_policy =
            DisplayPreviewTonePolicy::for_vignette(DisplayCliVignette::WithCorrection);
        let render_params = facts.p999_histogram_params(DisplayCliVignette::WithCorrection)?;
        let display_histogram = self
            .display_histogram
            .as_mut()
            .context("display p999 histogram missing")?;
        let histogram = display_histogram
            .compute_p999_luminance_histogram(GpuP999HistogramInput {
                device: self.backend.device(),
                queue: self.backend.queue(),
                input_buffer: &decoded.buffer,
                input_buffer_bytes: decoded.byte_len,
                params: render_params,
                config: tone_policy.histogram_config,
                extra_scale: tone_policy.histogram_extra_scale,
            })
            .map_err(|error| anyhow!("failed to compute display p999 histogram: {error}"))?;
        let preview_config = facts.preview_config(
            DisplayCliVignette::WithCorrection,
            self.preview_resolution_mode
                .preview_scale_mode(facts.dimensions),
        )?;
        let preview_rgb_sink_policy = tone_policy
            .preview_adjustments(Some(histogram))
            .preview_rgb_sink_policy();
        let mut encoder =
            self.backend
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mcraw4vulkan display with-vignette preview encoder"),
                });
        let encoded = self
            .preview_renderer
            .encode_render_to_texture_with_rgb_sink_policy(
                self.backend.device(),
                self.backend.queue(),
                &mut encoder,
                &decoded.buffer,
                decoded.byte_len,
                preview_config,
                preview_rgb_sink_policy,
            )
            .map_err(|error| anyhow!("failed to encode display preview texture: {error}"))?;
        self.backend.queue().submit([encoder.finish()]);
        self.backend.device().poll(wgpu::Maintain::Wait);
        let _ = encoded.info;
        Ok(())
    }

    fn ensure_uploaded_gain_map(
        &mut self,
        facts: &DisplayFrameFacts,
    ) -> Result<(
        GpuUploadedFullResolutionGainMap,
        GpuVignetteCorrectionParams,
    )> {
        let corrector = self
            .vignette_corrector
            .as_mut()
            .context("display vignette corrector missing")?;
        let fixed_facts = facts.fixed_facts_enabled()?;
        let fingerprint = VignetteGainMapFingerprint::from_fixed_facts(&fixed_facts)?;
        if let Some(cache) = self
            .gain_map
            .as_ref()
            .filter(|cache| cache.fingerprint == fingerprint)
        {
            return Ok((
                cache.uploaded.clone(),
                GpuVignetteCorrectionParams::from_fixed_facts(&fixed_facts)?,
            ));
        }

        let uploaded = self
            .backend
            .upload_compact_vignette_gain_map(corrector, &fixed_facts)?;
        self.gain_map = Some(UploadedGainMapCache {
            fingerprint,
            uploaded: uploaded.clone(),
        });
        Ok((
            uploaded,
            GpuVignetteCorrectionParams::from_fixed_facts(&fixed_facts)?,
        ))
    }

    fn render_cpu_frame(&mut self, facts: &DisplayFrameFacts, frame: &PayloadFrame) -> Result<()> {
        if self.preview_uploader.is_none() {
            bail!("CPU display upload buffer was not created");
        }
        self.source.cpu_decoder.prepare_compressed(frame.data.len());
        frame
            .data
            .copy_into(self.source.cpu_decoder.compressed_mut());
        let (decoded, _) = self
            .source
            .cpu_decoder
            .decode_loaded_payload_to_decoded_bayer_u16_frame_with_layout(
                facts.geometry.extraction(),
                facts.payload_layout,
            )
            .context("CPU decoding display frame")?;
        let bytes = decoded
            .with_resolved_geometry(facts.geometry)?
            .into_owned_le_bytes();

        let byte_len =
            u64::try_from(bytes.len()).context("CPU preview byte length overflows u64")?;
        let uploader = self
            .preview_uploader
            .as_mut()
            .context("CPU display upload buffer missing")?;
        let buffer = uploader.ensure(self.backend.device(), byte_len).clone();
        self.backend.queue().write_buffer(&buffer, 0, &bytes);
        if self.config.vignette == DisplayCliVignette::WithCorrection {
            let (uploaded_gain_map, params) = self.ensure_uploaded_gain_map(facts)?;
            let mut encoder =
                self.backend
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("mcraw4vulkan CPU RAW GPU vignette encoder"),
                    });
            let corrected = self
                .vignette_corrector
                .as_mut()
                .context("display vignette corrector missing")?
                .dispatch_packed_u16(GpuVignettePackedU16DispatchInput {
                    device: self.backend.device(),
                    queue: self.backend.queue(),
                    encoder: &mut encoder,
                    input_buffer: &buffer,
                    input_buffer_bytes: byte_len,
                    uploaded_gain_map: &uploaded_gain_map,
                    params,
                })?;
            let decoded = DecodedDisplayStage {
                buffer: corrected.output.buffer().clone(),
                byte_len,
            };
            self.backend.queue().submit([encoder.finish()]);
            return self.render_corrected_frame(facts, &decoded);
        }
        let mut encoder =
            self.backend
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mcraw4vulkan display CPU preview encoder"),
                });
        self.preview_renderer.encode_render_to_texture(
            self.backend.device(),
            self.backend.queue(),
            &mut encoder,
            &buffer,
            byte_len,
            facts.preview_config(
                DisplayCliVignette::NoCorrection,
                self.preview_resolution_mode
                    .preview_scale_mode(facts.dimensions),
            )?,
        )?;
        self.backend.queue().submit([encoder.finish()]);
        self.backend.device().poll(wgpu::Maintain::Wait);
        Ok(())
    }

    fn present_preview_to(
        &mut self,
        target: RenderFrameContext<'_>,
        present_target: DisplayPresentTarget,
        record_presented: bool,
    ) -> Result<()> {
        let texture_view = self.preview_renderer.texture_view()?;
        let fps_overlay = display_fps_overlay_for_present(
            self.config.overlay,
            self.fps_meter,
            self.source.frame_rate_fps,
            self.source.source_size(),
        );
        self.presenter.encode_present(DisplayPresentInput {
            device: self.backend.device(),
            queue: self.backend.queue(),
            encoder: target.encoder,
            source_view: texture_view,
            source_size: DisplaySize {
                width: self.preview_renderer.texture()?.width(),
                height: self.preview_renderer.texture()?.height(),
            },
            target_view: target.view,
            target: present_target,
            transfer_strategy: DisplayTransferStrategy::for_source_and_surface(
                DisplaySourceTransfer::ShaderSrgb,
                self.presenter.surface_format(),
            ),
            fps_overlay,
        })?;
        if record_presented {
            self.fps_meter.record_presented(Instant::now());
        }
        Ok(())
    }

    fn present_existing_preview_if_available(
        &mut self,
        target: RenderFrameContext<'_>,
        present_target: DisplayPresentTarget,
    ) -> Result<()> {
        if self.presented_frames == 0 {
            return Ok(());
        }
        self.present_preview_to(target, present_target, false)
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(sound) = self.sound.as_mut() {
            sound.stop();
        }
        self.source.finish()?;
        if self.presented_frames > 0 {
            eprintln!("displayed {} frames", self.presented_frames);
        }
        Ok(())
    }

    fn request_shutdown(&mut self) {
        self.source.request_shutdown();
    }

    fn poll_shutdown(&mut self) -> Result<bool> {
        self.source.poll_shutdown()
    }

    fn resume_after_pause(&mut self) {
        self.source.resume_after_pause();
    }

    fn playback_position(&self) -> EmbeddedDisplayPreviewPosition {
        EmbeddedDisplayPreviewPosition {
            current_frame_index: self.current_frame_index,
            frame_count: self.source.frame_count(),
        }
    }

    fn seek_to_frame_index(&mut self, frame_index: usize) -> Result<()> {
        self.source.seek_to_frame_index(frame_index)
    }
}

// Frame order and the payload feeder advance together. Seeking replaces the
// feeder with a plan beginning at the requested frame before updating the index.
struct DisplayWindowSource {
    input_path: PathBuf,
    container: McrawContainer,
    frames: Vec<FrameNumber>,
    next_frame_index: usize,
    payload_feeder: Option<PayloadFeeder>,
    payload_feeder_options: PayloadFeederOptions,
    cpu_decoder: CpuFrameDecoder,
    frame_rate_fps: f64,
    frame_pacer: Option<DisplayFramePacer>,
}

struct DisplayWindowPreparedSource {
    input_path: PathBuf,
    container: McrawContainer,
    frames: Vec<FrameNumber>,
    payload_plan: PayloadReadPlan,
    playback_mode: DisplayPlaybackMode,
    payload_feeder_options: PayloadFeederOptions,
    frame_rate_fps: f64,
}

impl DisplayWindowPreparedSource {
    #[rustfmt::skip]
    fn prepare(
        input_path: &Path,
        container: McrawContainer,
        playback_mode: DisplayPlaybackMode,
        payload_feeder_options: PayloadFeederOptions,
    ) -> Result<Self> {
        let geometry = container.resolved_frame_geometry(FrameNumber(0))?;
        if let Some(message) = geometry.recovery_message() {
            tracing::warn!("{message}");
        }
        let frame_count = container.frame_count();
        if frame_count == 0 {
            bail!("{} contains no frames", input_path.display());
        }

        let frames = (0..frame_count)
            .map(|frame| {
                u32::try_from(frame)
                    .map(FrameNumber)
                    .context("frame index does not fit in u32")
            })
            .collect::<Result<Vec<_>>>()?;
        let payload_plan = PayloadReadPlan::from_core_frame_numbers(&container, &frames)
            .context("planning display payload reads")?;
        let frame_rate_fps =
            display_source_frame_rate_fps(container.clip_info().frame_rate.as_f64());

        Ok(Self {
            input_path: input_path.to_path_buf(),
            container,
            frames,
            payload_plan,
            playback_mode,
            payload_feeder_options,
            frame_rate_fps,
        })
    }
}

impl DisplayWindowSource {
    #[rustfmt::skip]
    fn from_prepared(
        prepared: DisplayWindowPreparedSource,
    ) -> Result<Self> {
        let DisplayWindowPreparedSource {
            input_path,
            container,
            frames,
            payload_plan,
            playback_mode,
            payload_feeder_options,
            frame_rate_fps,
        } = prepared;
        let payload_feeder = {
            PayloadFeeder::spawn(&input_path, payload_plan, payload_feeder_options)
        }
        .context("starting display payload reader")?;
        let frame_pacer = playback_mode.create_frame_pacer(frame_rate_fps, Instant::now(), 0);

        Ok(Self {
            input_path,
            container,
            frames,
            next_frame_index: 0,
            payload_feeder: Some(payload_feeder),
            payload_feeder_options,
            cpu_decoder: CpuFrameDecoder::new(),
            frame_rate_fps,
            frame_pacer,
        })
    }

    fn frame_count(&self) -> usize {
        self.frames.len()
    }

    fn source_size(&self) -> DisplaySize {
        let dimensions = self
            .container
            .resolved_frame_geometry(FrameNumber(0))
            .expect("prepared display geometry")
            .effective;
        DisplaySize {
            width: dimensions.width,
            height: dimensions.height,
        }
    }

    fn next_payload_frame(&mut self) -> Result<Option<PayloadFrame>> {
        if self.next_frame_index >= self.frames.len() {
            return Ok(None);
        }
        let expected = self.frames[self.next_frame_index];
        let payload = self
            .payload_feeder
            .as_mut()
            .context("display payload reader already finished")?
            .next_frame()
            .with_context(|| format!("reading display frame {}", self.next_frame_index))?
            .context("display payload reader ended before selected frames")?;
        if payload.frame_number_core()? != expected {
            bail!(
                "display payload reader returned frame {:?}, expected {:?}",
                payload.frame_number_core()?,
                expected
            );
        }
        self.container
            .validate_payload_geometry(expected, payload.data.as_slice())?;
        Ok(Some(payload))
    }

    fn poll_next_payload_frame(&mut self) -> Result<DisplayPayloadFramePoll> {
        if self.next_frame_index >= self.frames.len() {
            return Ok(DisplayPayloadFramePoll::Finished);
        }
        let expected = self.frames[self.next_frame_index];
        let payload = match self
            .payload_feeder
            .as_mut()
            .context("display payload reader already finished")?
            .try_next_frame()
            .with_context(|| format!("polling display frame {}", self.next_frame_index))?
        {
            PayloadFeederPoll::Pending => return Ok(DisplayPayloadFramePoll::Pending),
            PayloadFeederPoll::Ready(None) => {
                return Err(anyhow!(
                    "display payload reader ended before selected frame {}",
                    self.next_frame_index
                ));
            }
            PayloadFeederPoll::Ready(Some(payload)) => payload,
        };
        if payload.frame_number_core()? != expected {
            bail!(
                "display payload reader returned frame {:?}, expected {:?}",
                payload.frame_number_core()?,
                expected
            );
        }
        self.container
            .validate_payload_geometry(expected, payload.data.as_slice())?;
        Ok(DisplayPayloadFramePoll::Ready(payload))
    }

    fn repaint_after_until_next_frame(&self, now: Instant) -> Option<Duration> {
        let frame_index = self.next_frame_index as u64;
        self.frame_pacer
            .and_then(|pacer| pacer.sleep_duration_until(frame_index, now))
    }

    fn cooperative_repaint_after(&self, now: Instant) -> Duration {
        match self.repaint_after_until_next_frame(now) {
            Some(repaint_after) => repaint_after,
            None if self.frame_pacer.is_some() => Duration::ZERO,
            None => EMBEDDED_PREVIEW_MAX_SPEED_REPAINT_INTERVAL,
        }
    }

    fn source_frame_duration(&self) -> Option<Duration> {
        (self.frame_rate_fps.is_finite() && self.frame_rate_fps > 0.0)
            .then(|| Duration::from_secs_f64(1.0 / self.frame_rate_fps))
    }

    fn advance_after_frame_blocking(&mut self) {
        if let Some(pacer) = self.frame_pacer.as_mut() {
            let frame_index = self.next_frame_index as u64;
            let now = Instant::now();
            if let Some(sleep_duration) = pacer.sleep_duration_until(frame_index, now) {
                std::thread::sleep(sleep_duration);
            }
            pacer.record_presented(frame_index);
        }
        self.next_frame_index += 1;
    }

    fn advance_after_frame_cooperative(&mut self) {
        if let Some(pacer) = self.frame_pacer.as_mut() {
            let frame_index = self.next_frame_index as u64;
            pacer.record_presented(frame_index);
        }
        self.next_frame_index += 1;
    }

    fn resume_after_pause(&mut self) {
        if let Some(pacer) = self.frame_pacer.as_mut() {
            pacer.reset_next_frame_due_now(self.next_frame_index as u64, Instant::now());
        }
    }

    fn seek_to_frame_index(&mut self, frame_index: usize) -> Result<()> {
        if self.frames.is_empty() {
            bail!("cannot seek an empty display source");
        }
        let frame_index = frame_index.min(self.frames.len() - 1);
        let payload_plan =
            PayloadReadPlan::from_core_frame_numbers(&self.container, &self.frames[frame_index..])
                .with_context(|| format!("planning display seek to frame {frame_index}"))?;
        let payload_feeder =
            PayloadFeeder::spawn(&self.input_path, payload_plan, self.payload_feeder_options)
                .with_context(|| {
                    format!("starting display payload reader at frame {frame_index}")
                })?;
        self.payload_feeder = Some(payload_feeder);
        self.next_frame_index = frame_index;
        self.resume_after_pause();
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(feeder) = self.payload_feeder.take() {
            if self.next_frame_index >= self.frames.len() {
                feeder.finish().with_context(|| {
                    format!(
                        "finishing display payload reader for {}",
                        self.input_path.display()
                    )
                })?;
            } else {
                drop(feeder);
            }
        }
        Ok(())
    }

    fn request_shutdown(&mut self) {
        if let Some(feeder) = self.payload_feeder.as_mut() {
            feeder.request_stop();
        }
    }

    fn poll_shutdown(&mut self) -> Result<bool> {
        let Some(feeder) = self.payload_feeder.as_mut() else {
            return Ok(true);
        };

        if feeder.try_join_after_stop().with_context(|| {
            format!(
                "stopping display payload reader for {}",
                self.input_path.display()
            )
        })? {
            self.payload_feeder = None;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn frame_facts(&self, frame_number: FrameNumber) -> Result<DisplayFrameFacts> {
        let frame_metadata = self
            .container
            .frame_metadata(frame_number)
            .with_context(|| format!("reading metadata for frame {:?}", frame_number))?;
        let container_metadata = self.container.container_metadata();
        let geometry = self.container.resolved_frame_geometry(frame_number)?;
        let dimensions = geometry.effective;
        let bayer_pattern = bayer_pattern_from_sensor(&container_metadata.sensor_arrangement)
            .context("unsupported Bayer arrangement for display")?;
        let black_level =
            frame_black_level(frame_metadata, container_metadata).context("missing black level")?;
        let white_level =
            frame_white_level(frame_metadata, container_metadata).context("missing white level")?;
        let source_bits = source_bits_from_white_level(white_level);
        let effective = container_metadata.with_frame_color(frame_metadata);
        let color_metadata = color_metadata_from_container(&effective);
        let adapted_color = if effective.forward_matrix1.is_none()
            && effective.forward_matrix2.is_none()
            && (effective.color_matrix1.is_some() || effective.color_matrix2.is_some())
        {
            use crate::strict_motioncam_color::{
                ClipSourceSha256, StrictMotionCamForwardMatrixColorV2,
                StrictMotionCamFrameColorInput,
            };
            use mcraw4vulkan_mcrawcontainer::{RawCamera2FrameColor, StrictColorProfileProvenance};
            let provenance = StrictColorProfileProvenance::from_source_sha256(
                ClipSourceSha256::deferred_stream_identity().bytes(),
            );
            let resolver = StrictMotionCamForwardMatrixColorV2::new()?;
            let profile = resolver
                .parse_supported_profile(self.container.container_metadata_json(), provenance)?;
            let profile = resolver.effective_profile(&profile, &frame_metadata.color_overrides)?;
            let input = StrictMotionCamFrameColorInput::from_raw(RawCamera2FrameColor {
                source_frame_index: u64::from(frame_number.0),
                as_shot_neutral: frame_metadata.as_shot_neutral,
                provenance,
            });
            Some(GpuRenderColorParams::from_camera_to_xyz_d50(
                resolver.resolve(&profile, &input)?.t50(),
            )?)
        } else {
            if effective.forward_matrix1.is_some() != effective.forward_matrix2.is_some()
                && effective.color_matrix1.is_some()
                && effective.color_matrix2.is_some()
            {
                bail!("incomplete ForwardMatrix calibration set");
            }
            None
        };
        let payload_layout = frame_metadata.payload_layout()?;
        Ok(DisplayFrameFacts {
            geometry,
            dimensions,
            payload_layout,
            bayer_pattern,
            black_level,
            white_level,
            source_bits,
            lens_shading_map: frame_metadata.lens_shading_map.clone(),
            as_shot_neutral: frame_metadata.as_shot_neutral,
            color_metadata,
            adapted_color,
        })
    }
}

struct DisplayFrameFacts {
    geometry: mcraw4vulkan_core::ResolvedFrameGeometry,
    dimensions: FrameDimensions,
    payload_layout: FramePayloadLayout,
    bayer_pattern: BayerPattern,
    black_level: [f32; 4],
    white_level: f32,
    source_bits: u16,
    lens_shading_map: Option<LensShadingMap>,
    as_shot_neutral: Option<[f64; 3]>,
    color_metadata: GpuRenderColorMetadata,
    adapted_color: Option<GpuRenderColorParams>,
}

impl DisplayFrameFacts {
    fn fixed_facts_enabled(&self) -> Result<FixedPointVignetteInputFacts<'_>> {
        let lens_shading_map = self
            .lens_shading_map
            .as_ref()
            .context("frame metadata is missing lensShadingMap")?;
        let fixed_map = PreparedFixedLensShadingMap::from_typed_map(lens_shading_map)?;
        let input_facts = VignetteCorrectionInputFacts::new(
            VignetteCorrectionMode::Enabled,
            VignetteCoordinateMapping::VisibleFrame,
            self.geometry.declared,
            self.bayer_pattern,
            None,
            self.black_level,
            self.white_level as u16,
        )?
        .with_resolved_geometry(self.geometry)?;
        FixedPointVignetteInputFacts::from_input_facts_with_fixed_map(&input_facts, Some(fixed_map))
            .context("failed to build fixed-point vignette input facts")
    }

    fn preview_config(
        &self,
        vignette: DisplayCliVignette,
        scale_mode: PreviewScaleMode,
    ) -> Result<PreviewRenderConfig> {
        let vignette_applied = vignette == DisplayCliVignette::WithCorrection;
        let color = self
            .adapted_color
            .map_or_else(
                || {
                    GpuRenderColorParams::from_metadata(
                        GpuRenderColorMode::MetadataSrgb,
                        self.as_shot_neutral,
                        self.color_metadata,
                    )
                },
                Ok,
            )
            .map_err(|error| anyhow!("failed to build display preview color params: {error}"))?;
        Ok(PreviewRenderConfig {
            dimensions: self.dimensions,
            bayer_pattern: self.bayer_pattern,
            source_bits: self.source_bits,
            black_level: if vignette_applied {
                [0.0; 4]
            } else {
                self.black_level
            },
            white_level: self.white_level,
            sample_domain: if vignette_applied {
                RenderSampleDomain::motioncam_compatible_pixel_v1(self.white_level)
            } else {
                RenderSampleDomain::raw(self.white_level)
            },
            color,
            texture_format: PreviewTextureFormat::default(),
            transfer_mode: PreviewTransferMode::default(),
            scale_mode,
        })
    }

    fn p999_histogram_params(
        &self,
        vignette: DisplayCliVignette,
    ) -> Result<GpuP999HistogramParams> {
        let vignette_applied = vignette == DisplayCliVignette::WithCorrection;
        let color = self
            .adapted_color
            .map_or_else(
                || {
                    GpuRenderColorParams::from_metadata(
                        GpuRenderColorMode::MetadataSrgb,
                        self.as_shot_neutral,
                        self.color_metadata,
                    )
                },
                Ok,
            )
            .map_err(|error| anyhow!("failed to build display p999 color params: {error}"))?;
        Ok(GpuP999HistogramParams {
            dimensions: self.dimensions,
            bayer_pattern: self.bayer_pattern,
            source_bits: self.source_bits,
            black_level: if vignette_applied {
                [0.0; 4]
            } else {
                self.black_level
            },
            white_level: self.white_level,
            sample_limit: if vignette_applied {
                mcraw4vulkan_render::motioncam_pixel_v1_display_sample_limit(self.white_level)
            } else {
                self.white_level
            },
            color,
        })
    }
}

fn bayer_pattern_from_sensor(sensor: &SensorArrangement) -> Option<BayerPattern> {
    match sensor {
        SensorArrangement::Rggb => Some(BayerPattern::Rggb),
        SensorArrangement::Grbg => Some(BayerPattern::Grbg),
        SensorArrangement::Gbrg => Some(BayerPattern::Gbrg),
        SensorArrangement::Bggr => Some(BayerPattern::Bggr),
        _ => None,
    }
}

fn frame_black_level(
    frame_metadata: &FrameMetadata,
    container_metadata: &ContainerMetadata,
) -> Option<[f32; 4]> {
    frame_metadata
        .dynamic_black_level
        .map(|level| level.map(|value| value as f32))
        .or_else(|| {
            container_metadata
                .black_level
                .as_ref()
                .map(|levels| levels.values.map(|value| value as f32))
        })
}

fn frame_white_level(
    frame_metadata: &FrameMetadata,
    container_metadata: &ContainerMetadata,
) -> Option<f32> {
    frame_metadata
        .dynamic_white_level
        .map(|level| level as f32)
        .or_else(|| {
            container_metadata
                .white_level
                .as_ref()
                .map(|level| level.values[0] as f32)
        })
}

fn source_bits_from_white_level(white_level: f32) -> u16 {
    let white = white_level.round().clamp(1.0, f32::from(u16::MAX)) as u16;
    let levels = u32::from(white).saturating_add(1).max(2);
    (u32::BITS - (levels - 1).leading_zeros()) as u16
}

fn color_metadata_from_container(metadata: &ContainerMetadata) -> GpuRenderColorMetadata {
    GpuRenderColorMetadata {
        color_matrix1: metadata.color_matrix1.map(|matrix| matrix.values),
        color_matrix2: metadata.color_matrix2.map(|matrix| matrix.values),
        forward_matrix1: metadata.forward_matrix1.map(|matrix| matrix.values),
        forward_matrix2: metadata.forward_matrix2.map(|matrix| matrix.values),
        illuminant1: metadata
            .color_illuminant1
            .as_ref()
            .map(render_illuminant_from_container),
        illuminant2: metadata
            .color_illuminant2
            .as_ref()
            .map(render_illuminant_from_container),
    }
}

fn render_illuminant_from_container(
    illuminant: &ColorIlluminant,
) -> GpuRenderCalibrationIlluminant {
    match illuminant {
        ColorIlluminant::StandardA => GpuRenderCalibrationIlluminant::StandardA,
        ColorIlluminant::D65 => GpuRenderCalibrationIlluminant::D65,
        ColorIlluminant::D50 | ColorIlluminant::Other(_) => GpuRenderCalibrationIlluminant::Other,
    }
}
