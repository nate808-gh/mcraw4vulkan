use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use mcraw4vulkan_core::{BayerPattern, FrameDimensions, FrameNumber, FramePayloadLayout};
use mcraw4vulkan_cpu::{CpuFrameDecoder, DecodeFrameTimings};
use mcraw4vulkan_display::{
    DisplayFpsMeter, DisplayFpsOverlay, DisplayFramePacer, DisplayPresentInput, DisplayPresentMode,
    DisplayPresentTarget, DisplayScaleMode, DisplaySize, DisplaySourceTransfer,
    DisplayTexturePresenter, DisplayTransferStrategy, fit_aspect_preserving,
};
use mcraw4vulkan_display_sound::{DisplaySoundError, DisplaySoundPlan, DisplaySoundSession};
use mcraw4vulkan_gpu::{
    GpuBackendPreference, GpuDecodeBackend, GpuDecodeConfig, OptionalGpuVignetteCorrection,
};
use mcraw4vulkan_mcrawcontainer::{
    ColorIlluminant, ContainerMetadata, FrameMetadata, LensShadingMap, McrawContainer,
    McrawContainerOpenTimings, SensorArrangement,
    payload_reader::{
        PayloadFeeder, PayloadFeederOptions, PayloadFeederPoll, PayloadFeederSpawnTimings,
        PayloadFrame, PayloadReadPlan,
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
    GpuVignetteCorrector, PreparedFixedLensShadingMap, VignetteCoordinateMapping,
    VignetteCorrectionInputFacts, VignetteCorrectionMode, VignetteGainMapFingerprint,
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
    pub startup_timing: bool,
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
            startup_timing: false,
        }
    }

    pub fn validate_before_open(&self) -> Result<()> {
        if self.backend == DisplayCliBackend::Cpu
            && self.vignette == DisplayCliVignette::WithCorrection
        {
            bail!(
                "Vignette correction using CPU is too slow; use `--gpu --with-vig-correction` or `--cpu --no-vig-correction`"
            );
        }

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

    fn backend_preference(&self) -> GpuBackendPreference {
        match self.backend {
            DisplayCliBackend::Gpu => GpuBackendPreference::VulkanOnly,
            DisplayCliBackend::Cpu => GpuBackendPreference::Auto,
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
    #[cfg(test)]
    fn is_fullscreen(self) -> bool {
        matches!(self, Self::FullscreenPhysical { .. })
    }

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

    let mut startup_timing = config.startup_timing.then(DisplayStartupTimings::new);
    let surface_start = Instant::now();
    let mut surface = Sdl2WgpuSurface::new(display_surface_config(&config))?;
    if let Some(timing) = startup_timing.as_mut() {
        timing.record_phase("display.sdl_wgpu_surface_setup", surface_start);
    }
    let mut event_pump = surface.event_pump()?;
    let initial_size = display_size_from_window_size(surface.size()).unwrap_or(DisplaySize {
        width: DEFAULT_STANDALONE_DISPLAY_WINDOW_WIDTH,
        height: DEFAULT_STANDALONE_DISPLAY_WINDOW_HEIGHT,
    });
    let adapter_features = surface.adapter().features();
    let mut state = DisplayWindowState::create(
        config,
        DisplayWindowCreateContext {
            adapter_info: surface.adapter_info().clone(),
            adapter_features,
            device: surface.device().clone(),
            queue: surface.queue().clone(),
            surface_format: surface.surface_format(),
            display_size: initial_size,
            startup_timing,
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
pub struct EmbeddedDisplayPreviewCreateTimings {
    pub display_source_from_prepared: Option<Duration>,
    pub payload_feeder_spawn: Option<Duration>,
    pub payload_chunk_plan_from_plan: Option<Duration>,
    pub gpu_decode_backend_from_wgpu_device: Option<Duration>,
    pub gpu_preview_texture_renderer_new: Option<Duration>,
    pub display_texture_presenter_new: Option<Duration>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EmbeddedDisplayPreviewFrameTimings {
    pub frame_index_before: u64,
    pub frame_index_after: u64,
    pub frame_number: Option<u32>,
    pub source_frame_duration: Option<Duration>,
    pub repaint_after: Option<Duration>,
    pub payload_receive_duration: Option<Duration>,
    pub decode_upload_duration: Option<Duration>,
    pub present_encode_duration: Option<Duration>,
    pub new_video_frame: bool,
    pub reused_existing_frame: bool,
    pub payload_pending: bool,
    pub waiting_for_pacer: bool,
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
    pub fn prepare(config: DisplayCliRunConfig) -> Result<Self> {
        config.validate_before_open()?;
        if config.sound.enabled() {
            bail!("display sound is only supported by standalone CLI display");
        }
        Ok(Self {
            prepared: DisplayWindowPreparedState::prepare(config, None)?,
        })
    }

    pub fn prepare_with_sound_plan(
        config: DisplayCliRunConfig,
    ) -> Result<(Self, Result<DisplaySoundPlan, DisplaySoundError>)> {
        config.validate_before_open()?;
        if !config.sound.enabled() {
            bail!("embedded display sound preparation requires sound-enabled config");
        }
        let prepared = DisplayWindowPreparedState::prepare(config, None)?;
        let sound_plan = DisplaySoundPlan::from_container(&prepared.source.container);
        Ok((Self { prepared }, sound_plan))
    }
}

impl EmbeddedDisplayPreview {
    pub fn create(
        config: DisplayCliRunConfig,
        adapter_info: wgpu::AdapterInfo,
        adapter_features: wgpu::Features,
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        target: EmbeddedDisplayPreviewTarget,
    ) -> Result<Self> {
        let prepared = EmbeddedDisplayPreviewPrepared::prepare(config)?;
        Self::from_prepared(
            prepared,
            adapter_info,
            adapter_features,
            device,
            queue,
            surface_format,
            target,
        )
    }

    pub fn from_prepared(
        prepared: EmbeddedDisplayPreviewPrepared,
        adapter_info: wgpu::AdapterInfo,
        adapter_features: wgpu::Features,
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        target: EmbeddedDisplayPreviewTarget,
    ) -> Result<Self> {
        let state = DisplayWindowState::create_from_prepared(
            prepared.prepared,
            DisplayWindowCreateContext {
                adapter_info,
                adapter_features,
                device,
                queue,
                surface_format,
                display_size: target.display_size(),
                startup_timing: None,
            },
            None,
            None,
        )?;
        Ok(Self { state })
    }

    pub fn from_prepared_with_timings(
        prepared: EmbeddedDisplayPreviewPrepared,
        adapter_info: wgpu::AdapterInfo,
        adapter_features: wgpu::Features,
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        target: EmbeddedDisplayPreviewTarget,
    ) -> Result<(Self, EmbeddedDisplayPreviewCreateTimings)> {
        let mut timings = EmbeddedDisplayPreviewCreateTimings::default();
        let state = DisplayWindowState::create_from_prepared(
            prepared.prepared,
            DisplayWindowCreateContext {
                adapter_info,
                adapter_features,
                device,
                queue,
                surface_format,
                display_size: target.display_size(),
                startup_timing: None,
            },
            Some(&mut timings),
            None,
        )?;
        Ok((Self { state }, timings))
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
        let frame_index_before = self.state.source.next_frame_index as u64;
        let source_frame_duration = self.state.source.source_frame_duration();
        self.state
            .present_existing_preview_if_available(target, preview_target.present_target()?)?;
        let repaint_after = self.state.cooperative_repaint_after(Instant::now());
        Ok(EmbeddedDisplayPreviewFrameResult {
            progress: EmbeddedDisplayPreviewProgress::Continue { repaint_after },
            timings: EmbeddedDisplayPreviewFrameTimings {
                frame_index_before,
                frame_index_after: self.state.source.next_frame_index as u64,
                source_frame_duration,
                repaint_after: Some(repaint_after),
                reused_existing_frame: self.state.presented_frames > 0,
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
    startup_timing: Option<DisplayStartupTimings>,
}

struct DisplayWindowPreparedState {
    config: DisplayCliRunConfig,
    source: DisplayWindowPreparedSource,
    startup_timing: Option<DisplayStartupTimings>,
}

struct DisplayWindowCreateContext {
    adapter_info: wgpu::AdapterInfo,
    adapter_features: wgpu::Features,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_format: wgpu::TextureFormat,
    display_size: DisplaySize,
    startup_timing: Option<DisplayStartupTimings>,
}

#[derive(Debug)]
struct DisplayStartupTimingEvent {
    phase: &'static str,
    start_offset: Duration,
    duration: Duration,
}

#[derive(Debug)]
struct DisplayStartupTimings {
    start: Instant,
    events: Vec<DisplayStartupTimingEvent>,
    first_payload_received: bool,
    first_decode_upload: bool,
    first_present_encode: bool,
    first_surface_submit: bool,
    emitted: bool,
}

impl DisplayStartupTimings {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            events: Vec::new(),
            first_payload_received: false,
            first_decode_upload: false,
            first_present_encode: false,
            first_surface_submit: false,
            emitted: false,
        }
    }

    fn record_phase(&mut self, phase: &'static str, phase_start: Instant) {
        self.events.push(DisplayStartupTimingEvent {
            phase,
            start_offset: phase_start.duration_since(self.start),
            duration: phase_start.elapsed(),
        });
    }

    fn record_nested_phase(
        &mut self,
        phase: &'static str,
        parent_start: Instant,
        nested_start_offset: Duration,
        duration: Duration,
    ) {
        self.events.push(DisplayStartupTimingEvent {
            phase,
            start_offset: parent_start.duration_since(self.start) + nested_start_offset,
            duration,
        });
    }

    fn emit(&mut self, input_path: &Path) {
        if self.emitted {
            return;
        }
        self.emitted = true;
        eprintln!(
            "[display-startup-timing] file={} total_elapsed_ms={:.3}",
            input_path.display(),
            duration_ms(self.start.elapsed())
        );
        for event in &self.events {
            eprintln!(
                "[display-startup-timing] phase={} start_ms={:.3} duration_ms={:.3}",
                event.phase,
                duration_ms(event.start_offset),
                duration_ms(event.duration)
            );
        }
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
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
    fn prepare(
        config: DisplayCliRunConfig,
        mut startup_timing: Option<DisplayStartupTimings>,
    ) -> Result<Self> {
        let container_start = Instant::now();
        let mut container_timings = McrawContainerOpenTimings::default();
        let container = match (startup_timing.is_some(), config.sound.enabled()) {
            (true, true) => McrawContainer::open_for_display_with_audio_with_timing(
                &config.input_path,
                &mut container_timings,
            ),
            (true, false) => McrawContainer::open_for_display_with_timing(
                &config.input_path,
                &mut container_timings,
            ),
            (false, true) => McrawContainer::open_for_display_with_audio(&config.input_path),
            (false, false) => McrawContainer::open_for_display(&config.input_path),
        }
        .with_context(|| format!("opening {}", config.input_path.display()))?;
        if let Some(timing) = startup_timing.as_mut() {
            timing.record_phase("mcraw_container.open.total", container_start);
            timing.record_nested_phase(
                "mcraw_container.open.parse_clip",
                container_start,
                container_timings.parse_clip.start_offset,
                container_timings.parse_clip.duration,
            );
            timing.record_nested_phase(
                "mcraw_container.open.container_metadata_parse",
                container_start,
                container_timings.container_metadata_parse.start_offset,
                container_timings.container_metadata_parse.duration,
            );
            let frame_metadata_phase = if container_timings.parsed_all_frame_metadata {
                "mcraw_container.open.frame_metadata_json_parse_loop"
            } else {
                "mcraw_container.open.display_initial_frame_metadata_parse"
            };
            timing.record_nested_phase(
                frame_metadata_phase,
                container_start,
                container_timings
                    .frame_metadata_json_parse_loop
                    .start_offset,
                container_timings.frame_metadata_json_parse_loop.duration,
            );
            if !container_timings.parsed_all_frame_metadata {
                timing.record_nested_phase(
                    "mcraw_container.open.full_frame_metadata_deferred",
                    container_start,
                    container_timings
                        .frame_metadata_json_parse_loop
                        .start_offset
                        + container_timings.frame_metadata_json_parse_loop.duration,
                    Duration::ZERO,
                );
            }
            if container_timings.parsed_audio_metadata {
                timing.record_nested_phase(
                    "mcraw_container.open.audio_metadata_index_sync_setup",
                    container_start,
                    container_timings
                        .audio_metadata_index_sync_setup
                        .start_offset,
                    container_timings.audio_metadata_index_sync_setup.duration,
                );
            } else {
                timing.record_nested_phase(
                    "mcraw_container.open.audio_metadata_index_sync_skipped_for_display",
                    container_start,
                    container_timings
                        .frame_metadata_json_parse_loop
                        .start_offset
                        + container_timings.frame_metadata_json_parse_loop.duration,
                    Duration::ZERO,
                );
            }
        }

        let source = DisplayWindowPreparedSource::prepare(
            &config.input_path,
            container,
            config.playback_mode(),
            config.payload_feeder_options,
            startup_timing.as_mut(),
        )?;

        Ok(Self {
            config,
            source,
            startup_timing,
        })
    }
}

fn prepare_display_sound(
    config: &DisplayCliRunConfig,
    container: &McrawContainer,
    startup_timing: Option<&mut DisplayStartupTimings>,
) -> Result<Option<DisplaySoundSession>> {
    if !config.sound.enabled() {
        return Ok(None);
    }

    let audio_start = Instant::now();
    let sound = DisplaySoundSession::from_container(container)
        .context("preparing display audio for --with-sound")?;
    if let Some(timing) = startup_timing {
        timing.record_phase("display.audio.prepare", audio_start);
    }
    Ok(Some(sound))
}

impl DisplayWindowState {
    fn create(
        config: DisplayCliRunConfig,
        mut context: DisplayWindowCreateContext,
    ) -> Result<Self> {
        let mut prepared =
            DisplayWindowPreparedState::prepare(config, context.startup_timing.take())?;
        let sound = prepare_display_sound(
            &prepared.config,
            &prepared.source.container,
            prepared.startup_timing.as_mut(),
        )?;
        Self::create_from_prepared(prepared, context, None, sound)
    }

    fn create_from_prepared(
        prepared: DisplayWindowPreparedState,
        context: DisplayWindowCreateContext,
        mut embedded_timings: Option<&mut EmbeddedDisplayPreviewCreateTimings>,
        sound: Option<DisplaySoundSession>,
    ) -> Result<Self> {
        let DisplayWindowPreparedState {
            config,
            source,
            mut startup_timing,
        } = prepared;
        let DisplayWindowCreateContext {
            adapter_info,
            adapter_features,
            device,
            queue,
            surface_format,
            display_size,
            startup_timing: _,
        } = context;

        let source_start = Instant::now();
        let source = DisplayWindowSource::from_prepared(
            source,
            startup_timing.as_mut(),
            embedded_timings.as_deref_mut(),
        )?;
        if let Some(timings) = embedded_timings.as_deref_mut() {
            timings.display_source_from_prepared = Some(source_start.elapsed());
        }

        let backend_start = Instant::now();
        let backend = GpuDecodeBackend::from_wgpu_device(
            adapter_info,
            adapter_features,
            device,
            queue,
            GpuDecodeConfig {
                backend_preference: config.backend_preference(),
                enable_gpu_timestamps: false,
            },
        )?;
        if let Some(timing) = startup_timing.as_mut() {
            timing.record_phase("gpu_decode_backend.from_wgpu_device", backend_start);
        }
        if let Some(timings) = embedded_timings.as_mut() {
            timings.gpu_decode_backend_from_wgpu_device = Some(backend_start.elapsed());
        }

        let preview_renderer_start = Instant::now();
        let preview_renderer = GpuPreviewTextureRenderer::new(backend.device())
            .map_err(|error| anyhow!("creating display preview renderer: {error}"))?;
        if let Some(timing) = startup_timing.as_mut() {
            timing.record_phase("gpu_preview_texture_renderer.new", preview_renderer_start);
        }
        if let Some(timings) = embedded_timings.as_mut() {
            timings.gpu_preview_texture_renderer_new = Some(preview_renderer_start.elapsed());
        }
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

        let presenter_start = Instant::now();
        let presenter = DisplayTexturePresenter::new(
            backend.device(),
            surface_format,
            DisplayScaleMode::FitContain,
        );
        if let Some(timing) = startup_timing.as_mut() {
            timing.record_phase("display_texture_presenter.new", presenter_start);
        }
        if let Some(timings) = embedded_timings.as_mut() {
            timings.display_texture_presenter_new = Some(presenter_start.elapsed());
        }
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
            startup_timing,
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

        let first_payload_start = self
            .startup_timing
            .as_ref()
            .filter(|timing| !timing.first_payload_received)
            .map(|_| Instant::now());
        let frame = self.source.next_payload_frame()?;
        if frame.is_some() {
            self.record_first_payload_received(first_payload_start);
        }
        let Some(frame) = frame else {
            return Ok(DisplayWindowProgress::Finished);
        };

        let displayed_frame_index = self.source.next_frame_index;
        let facts = self.source.frame_facts(frame.frame_number_core()?)?;
        let first_decode_upload_start = self
            .startup_timing
            .as_ref()
            .filter(|timing| !timing.first_decode_upload)
            .map(|_| Instant::now());
        match self.config.backend {
            DisplayCliBackend::Gpu => self.render_gpu_frame(&facts, &frame)?,
            DisplayCliBackend::Cpu => self.render_cpu_frame(&facts, &frame)?,
        }
        self.record_first_decode_upload(first_decode_upload_start);
        self.source.advance_after_frame_blocking();
        self.present_preview_to(target, present_target, true)?;
        self.current_frame_index = Some(displayed_frame_index);
        self.presented_frames += 1;

        Ok(DisplayWindowProgress::Continue {
            repaint_after: Duration::ZERO,
        })
    }

    fn render_next_frame_to_cooperative(
        &mut self,
        target: RenderFrameContext<'_>,
        present_target: DisplayPresentTarget,
    ) -> Result<DisplayWindowFrameResult> {
        self.resize(present_target.size);

        let now = Instant::now();
        let displayed_frame_index = self.source.next_frame_index;
        let frame_index_before = displayed_frame_index as u64;
        let source_frame_duration = self.source.source_frame_duration();
        if let Some(repaint_after) = self.source.repaint_after_until_next_frame(now) {
            self.present_existing_preview_if_available(target, present_target)?;
            return Ok(DisplayWindowFrameResult {
                progress: DisplayWindowProgress::Waiting { repaint_after },
                timings: EmbeddedDisplayPreviewFrameTimings {
                    frame_index_before,
                    frame_index_after: self.source.next_frame_index as u64,
                    source_frame_duration,
                    repaint_after: Some(repaint_after),
                    reused_existing_frame: self.presented_frames > 0,
                    waiting_for_pacer: true,
                    ..EmbeddedDisplayPreviewFrameTimings::default()
                },
            });
        }

        let payload_start = Instant::now();
        let frame = match self.source.poll_next_payload_frame()? {
            DisplayPayloadFramePoll::Ready(frame) => frame,
            DisplayPayloadFramePoll::Pending => {
                self.present_existing_preview_if_available(target, present_target)?;
                return Ok(DisplayWindowFrameResult {
                    progress: DisplayWindowProgress::Waiting {
                        repaint_after: EMBEDDED_PREVIEW_PAYLOAD_RETRY_INTERVAL,
                    },
                    timings: EmbeddedDisplayPreviewFrameTimings {
                        frame_index_before,
                        frame_index_after: self.source.next_frame_index as u64,
                        source_frame_duration,
                        repaint_after: Some(EMBEDDED_PREVIEW_PAYLOAD_RETRY_INTERVAL),
                        reused_existing_frame: self.presented_frames > 0,
                        payload_pending: true,
                        ..EmbeddedDisplayPreviewFrameTimings::default()
                    },
                });
            }
            DisplayPayloadFramePoll::Finished => {
                return Ok(DisplayWindowFrameResult {
                    progress: DisplayWindowProgress::Finished,
                    timings: EmbeddedDisplayPreviewFrameTimings {
                        frame_index_before,
                        frame_index_after: self.source.next_frame_index as u64,
                        source_frame_duration,
                        ..EmbeddedDisplayPreviewFrameTimings::default()
                    },
                });
            }
        };
        let payload_receive_duration = payload_start.elapsed();
        self.record_first_payload_received(Some(payload_start));

        let facts = self.source.frame_facts(frame.frame_number_core()?)?;
        let first_decode_upload_start = self
            .startup_timing
            .as_ref()
            .filter(|timing| !timing.first_decode_upload)
            .map(|_| Instant::now());
        let decode_upload_start = Instant::now();
        match self.config.backend {
            DisplayCliBackend::Gpu => self.render_gpu_frame(&facts, &frame)?,
            DisplayCliBackend::Cpu => self.render_cpu_frame(&facts, &frame)?,
        }
        let decode_upload_duration = decode_upload_start.elapsed();
        self.record_first_decode_upload(first_decode_upload_start);
        self.source.advance_after_frame_cooperative();
        let present_encode_start = Instant::now();
        self.present_preview_to(target, present_target, true)?;
        let present_encode_duration = present_encode_start.elapsed();
        self.current_frame_index = Some(displayed_frame_index);
        self.presented_frames += 1;
        let repaint_after = self.cooperative_repaint_after(Instant::now());

        Ok(DisplayWindowFrameResult {
            progress: DisplayWindowProgress::Continue { repaint_after },
            timings: EmbeddedDisplayPreviewFrameTimings {
                frame_index_before,
                frame_index_after: self.source.next_frame_index as u64,
                frame_number: Some(frame.frame_number_core()?.0),
                source_frame_duration,
                repaint_after: Some(repaint_after),
                payload_receive_duration: Some(payload_receive_duration),
                decode_upload_duration: Some(decode_upload_duration),
                present_encode_duration: Some(present_encode_duration),
                new_video_frame: true,
                ..EmbeddedDisplayPreviewFrameTimings::default()
            },
        })
    }

    fn cooperative_repaint_after(&self, now: Instant) -> Duration {
        self.source.cooperative_repaint_after(now)
    }

    fn record_first_payload_received(&mut self, phase_start: Option<Instant>) {
        let Some(phase_start) = phase_start else {
            return;
        };
        let Some(timing) = self.startup_timing.as_mut() else {
            return;
        };
        if timing.first_payload_received {
            return;
        }
        timing.first_payload_received = true;
        timing.record_phase("display.first_payload_received", phase_start);
    }

    fn record_first_decode_upload(&mut self, phase_start: Option<Instant>) {
        let Some(phase_start) = phase_start else {
            return;
        };
        let Some(timing) = self.startup_timing.as_mut() else {
            return;
        };
        if timing.first_decode_upload {
            return;
        }
        timing.first_decode_upload = true;
        timing.record_phase("display.first_decode_upload", phase_start);
    }

    fn record_first_present_encode(&mut self, phase_start: Option<Instant>) {
        let Some(phase_start) = phase_start else {
            return;
        };
        let Some(timing) = self.startup_timing.as_mut() else {
            return;
        };
        if timing.first_present_encode {
            return;
        }
        timing.first_present_encode = true;
        timing.record_phase("display.first_present_encode", phase_start);
    }

    fn after_surface_submit(&mut self) -> Result<()> {
        self.record_first_surface_submit();
        if let Some(sound) = self.sound.as_mut() {
            sound.start_after_first_video_submit()?;
            sound.pump()?;
        }
        Ok(())
    }

    fn record_first_surface_submit(&mut self) {
        let Some(timing) = self.startup_timing.as_mut() else {
            return;
        };
        if timing.first_surface_submit {
            return;
        }
        timing.first_surface_submit = true;
        timing.events.push(DisplayStartupTimingEvent {
            phase: "display.first_surface_submit",
            start_offset: timing.start.elapsed(),
            duration: Duration::ZERO,
        });
        timing.emit(&self.config.input_path);
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
                    facts.dimensions,
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
                    facts.dimensions,
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
        let tone_policy =
            DisplayPreviewTonePolicy::for_vignette(DisplayCliVignette::WithCorrection);
        let render_params = facts.p999_histogram_params(DisplayCliVignette::WithCorrection)?;
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
                    facts.dimensions,
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
                    facts.dimensions,
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
        let display_histogram = self
            .display_histogram
            .as_mut()
            .context("display p999 histogram missing")?;
        let histogram = display_histogram
            .compute_p999_luminance_histogram(GpuP999HistogramInput {
                device: self.backend.device(),
                queue: self.backend.queue(),
                input_buffer: &decode_output.stage.buffer,
                input_buffer_bytes: decode_output.stage.byte_len,
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
                &decode_output.stage.buffer,
                decode_output.stage.byte_len,
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
        let mut timings = DecodeFrameTimings::default();
        self.source.cpu_decoder.prepare_compressed(frame.data.len());
        frame
            .data
            .copy_into(self.source.cpu_decoder.compressed_mut());
        let (decoded, _) = self
            .source
            .cpu_decoder
            .decode_loaded_payload_to_decoded_bayer_u16_frame_with_layout(
                facts.dimensions,
                facts.payload_layout,
                &mut timings,
            )
            .context("CPU decoding display frame")?;
        let bytes = decoded.into_owned_le_bytes();

        let byte_len =
            u64::try_from(bytes.len()).context("CPU preview byte length overflows u64")?;
        let uploader = self
            .preview_uploader
            .as_mut()
            .context("CPU display upload buffer missing")?;
        let buffer = uploader.ensure(self.backend.device(), byte_len);
        self.backend.queue().write_buffer(buffer, 0, &bytes);
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
            buffer,
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
        let first_present_encode_start = self
            .startup_timing
            .as_ref()
            .filter(|timing| record_presented && !timing.first_present_encode)
            .map(|_| Instant::now());
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
        self.record_first_present_encode(first_present_encode_start);
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
        if let Some(timing) = self.startup_timing.as_mut() {
            timing.emit(&self.config.input_path);
        }
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
    source_create_start: Option<Instant>,
}

impl DisplayWindowPreparedSource {
    fn prepare(
        input_path: &Path,
        container: McrawContainer,
        playback_mode: DisplayPlaybackMode,
        payload_feeder_options: PayloadFeederOptions,
        startup_timing: Option<&mut DisplayStartupTimings>,
    ) -> Result<Self> {
        let source_create_start = startup_timing.as_ref().map(|_| Instant::now());
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
        let plan_start = Instant::now();
        let payload_plan = PayloadReadPlan::from_core_frame_numbers(&container, &frames)
            .context("planning display payload reads")?;
        if let Some(timing) = startup_timing {
            timing.record_phase("payload_read_plan.from_core_frame_numbers", plan_start);
        }
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
            source_create_start,
        })
    }
}

impl DisplayWindowSource {
    fn from_prepared(
        prepared: DisplayWindowPreparedSource,
        startup_timing: Option<&mut DisplayStartupTimings>,
        embedded_timings: Option<&mut EmbeddedDisplayPreviewCreateTimings>,
    ) -> Result<Self> {
        let DisplayWindowPreparedSource {
            input_path,
            container,
            frames,
            payload_plan,
            playback_mode,
            payload_feeder_options,
            frame_rate_fps,
            source_create_start,
        } = prepared;
        let feeder_start = Instant::now();
        let mut feeder_timings = PayloadFeederSpawnTimings::default();
        let timing_payload_feeder = startup_timing.is_some() || embedded_timings.is_some();
        let payload_feeder = if timing_payload_feeder {
            PayloadFeeder::spawn_with_timing(
                &input_path,
                payload_plan,
                payload_feeder_options,
                &mut feeder_timings,
            )
        } else {
            PayloadFeeder::spawn(&input_path, payload_plan, payload_feeder_options)
        }
        .context("starting display payload reader")?;
        if let Some(timing) = startup_timing {
            timing.record_phase("payload_feeder.spawn.total", feeder_start);
            if let Some(chunk_plan) = feeder_timings.chunk_plan_from_plan {
                timing.record_nested_phase(
                    "payload_feeder.spawn.payload_chunk_plan.from_plan",
                    feeder_start,
                    chunk_plan.start_offset,
                    chunk_plan.duration,
                );
            }
            if let Some(source_create_start) = source_create_start {
                timing.record_phase("display_source.create.total", source_create_start);
            }
        }
        if let Some(timings) = embedded_timings {
            timings.payload_feeder_spawn = Some(feeder_start.elapsed());
            if let Some(chunk_plan) = feeder_timings.chunk_plan_from_plan {
                timings.payload_chunk_plan_from_plan = Some(chunk_plan.duration);
            }
        }
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
        let clip_info = self.container.clip_info();
        DisplaySize {
            width: clip_info.width,
            height: clip_info.height,
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
        let dimensions = frame_metadata.dimensions;
        let bayer_pattern = bayer_pattern_from_sensor(&container_metadata.sensor_arrangement)
            .context("unsupported Bayer arrangement for display")?;
        let black_level =
            frame_black_level(frame_metadata, container_metadata).context("missing black level")?;
        let white_level =
            frame_white_level(frame_metadata, container_metadata).context("missing white level")?;
        let source_bits = source_bits_from_white_level(white_level);
        let color_metadata = color_metadata_from_container(container_metadata);
        let payload_layout = frame_metadata.payload_layout()?;
        Ok(DisplayFrameFacts {
            dimensions,
            payload_layout,
            bayer_pattern,
            black_level,
            white_level,
            source_bits,
            lens_shading_map: frame_metadata.lens_shading_map.clone(),
            as_shot_neutral: frame_metadata.as_shot_neutral,
            color_metadata,
        })
    }
}

struct DisplayFrameFacts {
    dimensions: FrameDimensions,
    payload_layout: FramePayloadLayout,
    bayer_pattern: BayerPattern,
    black_level: [f32; 4],
    white_level: f32,
    source_bits: u16,
    lens_shading_map: Option<LensShadingMap>,
    as_shot_neutral: Option<[f64; 3]>,
    color_metadata: GpuRenderColorMetadata,
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
            self.dimensions,
            self.bayer_pattern,
            None,
            self.black_level,
            self.white_level as u16,
        )?;
        FixedPointVignetteInputFacts::from_input_facts_with_fixed_map(&input_facts, Some(fixed_map))
            .context("failed to build fixed-point vignette input facts")
    }

    fn preview_config(
        &self,
        vignette: DisplayCliVignette,
        scale_mode: PreviewScaleMode,
    ) -> Result<PreviewRenderConfig> {
        let vignette_applied = vignette == DisplayCliVignette::WithCorrection;
        let color = GpuRenderColorParams::from_metadata(
            GpuRenderColorMode::MetadataSrgb,
            self.as_shot_neutral,
            self.color_metadata,
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
        let color = GpuRenderColorParams::from_metadata(
            GpuRenderColorMode::MetadataSrgb,
            self.as_shot_neutral,
            self.color_metadata,
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
        ColorIlluminant::Other(_) => GpuRenderCalibrationIlluminant::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcraw4vulkan_render::{GpuRgbSinkGuardMode, p999_scale_for_luma};

    #[test]
    fn display_cli_default_config_matches_production_contract() {
        let config = DisplayCliRunConfig::production_default("clip.mcraw");
        assert_eq!(config.backend, DisplayCliBackend::Gpu);
        assert_eq!(config.vignette, DisplayCliVignette::NoCorrection);
        assert_eq!(config.vsync, DisplayCliVsync::Vsync);
        assert_eq!(config.settings, DisplayCliSettings::Default);
        assert_eq!(config.overlay, DisplayCliOverlay::WithOverlay);
        assert_eq!(config.sound, DisplayCliSound::Silent);
    }

    #[test]
    fn standalone_display_surface_config_keeps_accepted_default_size() {
        let config = DisplayCliRunConfig::production_default("clip.mcraw");
        let surface_config = display_surface_config(&config);

        assert_eq!(
            surface_config.width,
            DEFAULT_STANDALONE_DISPLAY_WINDOW_WIDTH
        );
        assert_eq!(
            surface_config.height,
            DEFAULT_STANDALONE_DISPLAY_WINDOW_HEIGHT
        );
        assert_eq!(
            surface_config.preferred_present_mode,
            Some(wgpu::PresentMode::Fifo)
        );
    }

    #[test]
    fn embedded_preview_target_rejects_zero_dimensions() {
        assert_eq!(
            EmbeddedDisplayPreviewTarget::new(0, 0, 1280, 720),
            Some(EmbeddedDisplayPreviewTarget {
                origin_x: 0,
                origin_y: 0,
                width: 1280,
                height: 720,
            })
        );
        assert_eq!(EmbeddedDisplayPreviewTarget::new(0, 0, 0, 720), None);
        assert_eq!(EmbeddedDisplayPreviewTarget::new(0, 0, 1280, 0), None);
    }

    #[test]
    fn display_optimized_settings_validate_without_state_failure() {
        let mut config = DisplayCliRunConfig::production_default("clip.mcraw");
        config.settings = DisplayCliSettings::Optimized;
        config.validate_before_open().unwrap();
    }

    #[test]
    fn effective_display_settings_apply_payload_profile() {
        let mut config = DisplayCliRunConfig::production_default("clip.mcraw");
        config.settings = DisplayCliSettings::Optimized;
        let effective = mcraw4vulkan_optimizer::resolve_effective_settings_from_load(
            mcraw4vulkan_optimizer::OptimizedStateLoadOutcome::Loaded {
                path: PathBuf::from("optimized-state.json"),
                state: mcraw4vulkan_optimizer::OptimizedState::new(
                    mcraw4vulkan_optimizer::PayloadProfile::OffsetPrefetch,
                ),
            },
        );

        apply_effective_optimizer_settings(&mut config, &effective);

        assert_eq!(
            config.payload_feeder_options,
            mcraw4vulkan_optimizer::PayloadProfile::OffsetPrefetch.payload_feeder_options()
        );
    }

    #[test]
    fn display_cpu_vignette_reports_unsupported_combination() {
        for settings in [DisplayCliSettings::Default, DisplayCliSettings::Optimized] {
            let config = DisplayCliRunConfig {
                input_path: PathBuf::from("clip.mcraw"),
                backend: DisplayCliBackend::Cpu,
                vignette: DisplayCliVignette::WithCorrection,
                vsync: DisplayCliVsync::Vsync,
                settings,
                overlay: DisplayCliOverlay::WithOverlay,
                sound: DisplayCliSound::Silent,
                payload_feeder_options: PayloadFeederOptions::production_default(),
                startup_timing: false,
            };
            let error = config.validate_before_open().unwrap_err().to_string();
            assert!(error.contains("Vignette correction"), "{error}");
            assert!(error.contains("CPU"), "{error}");
            assert!(error.contains("--gpu --with-vig-correction"), "{error}");
            assert!(error.contains("--cpu --no-vig-correction"), "{error}");
        }
    }

    #[test]
    fn type6_gpu_display_route_is_native_supported() {
        assert!(
            FramePayloadLayout::BinnedRaw16Type6 { row_stride: 3840 }.supports_native_gpu_decode()
        );
        assert_eq!(
            FramePayloadLayout::BinnedRaw16Type6 { row_stride: 3840 }
                .unsupported_gpu_decode_message(),
            None
        );
    }

    fn synthetic_display_frame_facts() -> DisplayFrameFacts {
        DisplayFrameFacts {
            dimensions: FrameDimensions {
                width: 640,
                height: 360,
            },
            payload_layout: FramePayloadLayout::CompressedRawcodecType7,
            bayer_pattern: BayerPattern::Gbrg,
            black_level: [64.0, 65.0, 66.0, 67.0],
            white_level: 1023.0,
            source_bits: 10,
            lens_shading_map: None,
            as_shot_neutral: Some([0.5, 1.0, 0.25]),
            color_metadata: GpuRenderColorMetadata::default(),
        }
    }

    fn frame_dimensions(width: u32, height: u32) -> FrameDimensions {
        FrameDimensions { width, height }
    }

    fn display_size(width: u32, height: u32) -> DisplaySize {
        DisplaySize { width, height }
    }

    fn assert_fullscreen_target(
        source: FrameDimensions,
        canvas: DisplaySize,
        expected: FrameDimensions,
    ) {
        assert_eq!(
            fullscreen_preview_target_dimensions(source, canvas),
            Some(expected)
        );
        assert_eq!(
            DisplayPreviewResolutionMode::fullscreen(canvas).preview_scale_mode(source),
            PreviewScaleMode::Explicit {
                width: expected.width,
                height: expected.height,
            }
        );
    }

    fn assert_windowed_target(
        source: FrameDimensions,
        canvas: DisplaySize,
        expected: FrameDimensions,
    ) {
        assert_eq!(
            preview_target_dimensions_for_canvas(source, canvas),
            Some(expected)
        );
        assert_eq!(
            DisplayPreviewResolutionMode::windowed(canvas).preview_scale_mode(source),
            PreviewScaleMode::Explicit {
                width: expected.width,
                height: expected.height,
            }
        );
    }

    fn synthetic_p999_result() -> GpuP999HistogramResult {
        let histogram_config = GpuP999HistogramConfig::canonical();
        let rgb_policy = DisplayRgbTonePolicy::canonical();
        let luma_p999 = 2.0;
        let scale_p999 = p999_scale_for_luma(luma_p999, histogram_config);
        let final_scale = scale_p999 * rgb_policy.extra_scale(true);
        GpuP999HistogramResult {
            luma_p999,
            scale_p999,
            final_scale,
            total_samples: 1000,
            overflow_count: 0,
            max_luma_observed: luma_p999,
            histogram_bins: histogram_config.bin_count,
            readback_bytes: histogram_config.histogram_readback_bytes().unwrap(),
        }
    }

    #[test]
    fn display_with_vig_policy_selects_canonical_rgb_sink_policy() {
        let policy = DisplayPreviewTonePolicy::for_vignette(DisplayCliVignette::WithCorrection);

        assert!(policy.enabled);
        assert_eq!(policy.rgb_policy, DisplayRgbTonePolicy::canonical());
        assert_eq!(policy.histogram_config, GpuP999HistogramConfig::canonical());
    }

    #[test]
    fn display_with_vig_policy_uses_canonical_p999_target_and_clamps() {
        let policy = DisplayPreviewTonePolicy::for_vignette(DisplayCliVignette::WithCorrection);
        let histogram_config = policy.histogram_config;

        assert_eq!(histogram_config.target_percentile, 0.999);
        assert_eq!(histogram_config.target_output_luma, 0.96);
        assert_eq!(histogram_config.min_scale, 1.0 / 64.0);
        assert_eq!(histogram_config.max_scale, 8.0);
        assert!((p999_scale_for_luma(2.0, histogram_config) - 0.48).abs() < 1.0e-6);
    }

    #[test]
    fn display_with_vig_policy_applies_canonical_extra_vig_scale() {
        let policy = DisplayPreviewTonePolicy::for_vignette(DisplayCliVignette::WithCorrection);

        assert!(
            (policy.histogram_extra_scale - policy.rgb_policy.extra_scale(true)).abs() < 1.0e-7
        );
        assert!((policy.histogram_extra_scale - std::f32::consts::FRAC_1_SQRT_2).abs() < 1.0e-7);
    }

    #[test]
    fn display_with_vig_policy_enables_rb_guard() {
        let policy = DisplayPreviewTonePolicy::for_vignette(DisplayCliVignette::WithCorrection);
        let histogram = synthetic_p999_result();
        let adjustments = policy.preview_adjustments(Some(histogram));

        assert_eq!(adjustments.guard.mode, GpuRgbSinkGuardMode::RbSumLimit);
        assert!(
            (adjustments.guard.highlight_luma - (histogram.luma_p999 * histogram.final_scale))
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn display_with_vig_policy_enables_canonical_highlight_desat() {
        let policy = DisplayPreviewTonePolicy::for_vignette(DisplayCliVignette::WithCorrection);
        let histogram = synthetic_p999_result();
        let adjustments = policy.preview_adjustments(Some(histogram));
        let threshold = histogram.luma_p999 * histogram.final_scale;

        assert!(adjustments.desat.enabled);
        assert!(
            (adjustments.desat.start_luma - threshold * policy.rgb_policy.vig_desat_start_scale())
                .abs()
                < 1.0e-6
        );
        assert!(
            (adjustments.desat.end_luma - threshold * policy.rgb_policy.vig_desat_end_scale())
                .abs()
                < 1.0e-6
        );
        assert_eq!(
            adjustments.desat.strength,
            policy.rgb_policy.vig_desat_strength()
        );
    }

    #[test]
    fn display_no_vig_preview_policy_remains_unchanged() {
        let facts = synthetic_display_frame_facts();
        let config = facts
            .preview_config(
                DisplayCliVignette::NoCorrection,
                PreviewScaleMode::FullResolution,
            )
            .expect("preview config");
        let policy = DisplayPreviewTonePolicy::for_vignette(DisplayCliVignette::NoCorrection);
        let adjustments = policy.preview_adjustments(None);

        assert_eq!(config.black_level, facts.black_level);
        assert_eq!(
            config.sample_domain,
            RenderSampleDomain::raw(facts.white_level)
        );
        assert_eq!(adjustments, DisplayPreviewToneAdjustments::none());
        assert_eq!(
            adjustments.preview_rgb_sink_policy(),
            PreviewRgbSinkPolicy::none()
        );
    }

    #[test]
    fn display_with_vig_histogram_params_match_preview_input_domain() {
        let facts = synthetic_display_frame_facts();
        let params = facts
            .p999_histogram_params(DisplayCliVignette::WithCorrection)
            .expect("histogram params");

        assert_eq!(params.black_level, [0.0; 4]);
        assert_eq!(
            params.sample_limit,
            mcraw4vulkan_render::motioncam_pixel_v1_display_sample_limit(facts.white_level)
        );
        assert_eq!(params.color.mode, GpuRenderColorMode::MetadataSrgb);
    }

    #[test]
    fn display_with_vig_preview_params_include_tone_guard_desat_policy() {
        let policy = DisplayPreviewTonePolicy::for_vignette(DisplayCliVignette::WithCorrection);
        let adjustments = policy.preview_adjustments(Some(synthetic_p999_result()));
        let preview_policy = adjustments.preview_rgb_sink_policy();

        assert_eq!(preview_policy.rgb_sink_tone_scale, adjustments.tone_scale);
        assert_eq!(preview_policy.rgb_sink_guard, adjustments.guard);
        assert_eq!(preview_policy.rgb_sink_desat, adjustments.desat);
    }

    #[test]
    fn display_window_title_includes_input_basename() {
        let config = DisplayCliRunConfig::production_default("tmp/example.mcraw");
        assert!(config.window_title().contains("example.mcraw"));
    }

    #[test]
    fn display_present_mode_follows_vsync_choice() {
        let mut config = DisplayCliRunConfig::production_default("clip.mcraw");
        assert_eq!(config.present_mode(), DisplayPresentMode::Fifo);
        config.vsync = DisplayCliVsync::NoVsync;
        assert_eq!(config.present_mode(), DisplayPresentMode::AutoNoVsync);
    }

    #[test]
    fn display_vsync_maps_to_source_paced_playback() {
        let config = DisplayCliRunConfig::production_default("clip.mcraw");

        assert_eq!(config.playback_mode(), DisplayPlaybackMode::SourcePaced);
        assert!(config.playback_mode().source_paced_playback());
    }

    #[test]
    fn display_no_vsync_maps_to_max_speed_playback() {
        let mut config = DisplayCliRunConfig::production_default("clip.mcraw");
        config.vsync = DisplayCliVsync::NoVsync;

        assert_eq!(config.playback_mode(), DisplayPlaybackMode::MaxSpeed);
        assert!(!config.playback_mode().source_paced_playback());
    }

    #[test]
    fn display_with_sound_requires_vsync() {
        let mut config = DisplayCliRunConfig::production_default("clip.mcraw");
        config.sound = DisplayCliSound::WithSound;
        config.validate_before_open().unwrap();

        config.vsync = DisplayCliVsync::NoVsync;
        let error = config.validate_before_open().unwrap_err().to_string();
        assert!(
            error.contains("--with-sound requires Vsync display"),
            "{error}"
        );
        assert!(error.contains("--no-vsync"), "{error}");
    }

    #[test]
    fn display_no_vsync_does_not_construct_source_frame_pacer() {
        let start = Instant::now();
        let pacer = DisplayPlaybackMode::MaxSpeed.create_frame_pacer(24.0, start, 0);

        assert!(pacer.is_none());
    }

    #[test]
    fn display_vsync_constructs_source_frame_pacer() {
        let start = Instant::now();
        let pacer = DisplayPlaybackMode::SourcePaced
            .create_frame_pacer(24.0, start, 0)
            .expect("source pacer");

        assert!((pacer.frame_duration().as_secs_f64() - (1.0 / 24.0)).abs() < 0.000001);
        assert_eq!(
            pacer.sleep_duration_until(1, start),
            Some(pacer.next_frame_instant(1).duration_since(start))
        );
    }

    #[test]
    fn display_metadata_frame_rate_falls_back_to_documented_default() {
        assert_eq!(
            display_source_frame_rate_fps(None),
            DEFAULT_DISPLAY_FRAME_RATE_FPS
        );
        assert_eq!(
            display_source_frame_rate_fps(Some(f64::NAN)),
            DEFAULT_DISPLAY_FRAME_RATE_FPS
        );
        assert_eq!(
            display_source_frame_rate_fps(Some(0.0)),
            DEFAULT_DISPLAY_FRAME_RATE_FPS
        );
        assert_eq!(display_source_frame_rate_fps(Some(29.97)), 29.97);
    }

    #[test]
    fn fullscreen_target_4_3_source_into_uhd_is_2880_by_2160() {
        assert_fullscreen_target(
            frame_dimensions(4032, 3024),
            display_size(3840, 2160),
            frame_dimensions(2880, 2160),
        );
    }

    #[test]
    fn fullscreen_target_4_3_source_into_1440p_is_1920_by_1440() {
        assert_fullscreen_target(
            frame_dimensions(4032, 3024),
            display_size(2560, 1440),
            frame_dimensions(1920, 1440),
        );
    }

    #[test]
    fn fullscreen_target_4_3_source_into_1080p_is_1440_by_1080() {
        assert_fullscreen_target(
            frame_dimensions(4032, 3024),
            display_size(1920, 1080),
            frame_dimensions(1440, 1080),
        );
    }

    #[test]
    fn fullscreen_target_16_9_source_into_uhd_is_3840_by_2160() {
        assert_fullscreen_target(
            frame_dimensions(8192, 4608),
            display_size(3840, 2160),
            frame_dimensions(3840, 2160),
        );
    }

    #[test]
    fn fullscreen_target_16_9_source_into_1440p_is_2560_by_1440() {
        assert_fullscreen_target(
            frame_dimensions(8192, 4608),
            display_size(2560, 1440),
            frame_dimensions(2560, 1440),
        );
    }

    #[test]
    fn fullscreen_target_16_9_source_into_1080p_is_1920_by_1080() {
        assert_fullscreen_target(
            frame_dimensions(8192, 4608),
            display_size(1920, 1080),
            frame_dimensions(1920, 1080),
        );
    }

    #[test]
    fn fullscreen_target_dciish_source_preserves_aspect() {
        let target = fullscreen_preview_target_dimensions(
            frame_dimensions(8192, 4320),
            display_size(3840, 2160),
        )
        .expect("target");

        assert_eq!(target, frame_dimensions(3840, 2025));
        let source_aspect = 8192.0_f32 / 4320.0_f32;
        let target_aspect = target.width as f32 / target.height as f32;
        assert!((source_aspect - target_aspect).abs() < 0.001);
    }

    #[test]
    fn windowed_target_4_3_source_into_720p_window_is_960_by_720() {
        assert_windowed_target(
            frame_dimensions(4032, 3024),
            display_size(1280, 720),
            frame_dimensions(960, 720),
        );
    }

    #[test]
    fn windowed_target_16_9_source_into_720p_window_is_1280_by_720() {
        assert_windowed_target(
            frame_dimensions(8192, 4608),
            display_size(1280, 720),
            frame_dimensions(1280, 720),
        );
    }

    #[test]
    fn windowed_target_4_3_source_into_1440p_window_is_1920_by_1440() {
        assert_windowed_target(
            frame_dimensions(4032, 3024),
            display_size(2560, 1440),
            frame_dimensions(1920, 1440),
        );
    }

    #[test]
    fn windowed_target_16_9_source_into_1440p_window_is_2560_by_1440() {
        assert_windowed_target(
            frame_dimensions(8192, 4608),
            display_size(2560, 1440),
            frame_dimensions(2560, 1440),
        );
    }

    #[test]
    fn windowed_target_clamps_small_16_9_source_to_source_dimensions() {
        assert_windowed_target(
            frame_dimensions(640, 360),
            display_size(1280, 720),
            frame_dimensions(640, 360),
        );
    }

    #[test]
    fn windowed_target_clamps_small_4_3_source_to_source_dimensions() {
        assert_windowed_target(
            frame_dimensions(640, 480),
            display_size(1280, 720),
            frame_dimensions(640, 480),
        );
    }

    #[test]
    fn preview_target_zero_canvas_is_safe() {
        assert_eq!(
            preview_target_dimensions_for_canvas(
                frame_dimensions(4032, 3024),
                display_size(0, 720)
            ),
            None
        );
    }

    #[test]
    fn preview_target_zero_source_is_safe() {
        assert_eq!(
            preview_target_dimensions_for_canvas(
                frame_dimensions(0, 3024),
                display_size(1280, 720)
            ),
            None
        );
    }

    #[test]
    fn windowed_target_dciish_source_preserves_aspect() {
        let target = preview_target_dimensions_for_canvas(
            frame_dimensions(8192, 4320),
            display_size(3840, 2160),
        )
        .expect("target");

        assert_eq!(target, frame_dimensions(3840, 2025));
        let source_aspect = 8192.0_f32 / 4320.0_f32;
        let target_aspect = target.width as f32 / target.height as f32;
        assert!((source_aspect - target_aspect).abs() < 0.001);
    }

    #[test]
    fn preview_target_never_exceeds_source_or_canvas() {
        for (source, canvas) in [
            (frame_dimensions(4032, 3024), display_size(1280, 720)),
            (frame_dimensions(8192, 4608), display_size(2560, 1440)),
            (frame_dimensions(640, 480), display_size(1280, 720)),
            (frame_dimensions(8192, 4320), display_size(3840, 2160)),
        ] {
            let target = preview_target_dimensions_for_canvas(source, canvas).expect("target");
            assert!(target.width <= source.width);
            assert!(target.height <= source.height);
            assert!(target.width <= canvas.width);
            assert!(target.height <= canvas.height);
        }
    }

    #[test]
    fn fullscreen_target_zero_canvas_is_safe() {
        assert_eq!(
            fullscreen_preview_target_dimensions(
                frame_dimensions(4032, 3024),
                display_size(0, 2160)
            ),
            None
        );
        assert_eq!(
            DisplayPreviewResolutionMode::fullscreen(display_size(0, 2160))
                .preview_scale_mode(frame_dimensions(4032, 3024)),
            PreviewScaleMode::FullResolution
        );
    }

    #[test]
    fn fullscreen_target_clamps_to_source_when_source_is_smaller() {
        assert_fullscreen_target(
            frame_dimensions(1280, 720),
            display_size(3840, 2160),
            frame_dimensions(1280, 720),
        );
    }

    #[test]
    fn fullscreen_target_is_at_least_one_by_one_when_displayable() {
        assert_fullscreen_target(
            frame_dimensions(1, 1000),
            display_size(1000, 1),
            frame_dimensions(1, 1),
        );
    }

    #[test]
    fn windowed_preview_mode_uses_explicit_physical_canvas_target() {
        let source = frame_dimensions(4032, 3024);
        let mode = DisplayPreviewResolutionMode::windowed(display_size(1280, 720));

        assert_eq!(
            mode.preview_scale_mode(source),
            PreviewScaleMode::Explicit {
                width: 960,
                height: 720,
            }
        );
        assert_eq!(
            mode.preview_scale_mode(source).resolve(source).unwrap(),
            frame_dimensions(960, 720)
        );
    }

    #[test]
    fn entering_fullscreen_switches_to_fullscreen_target_mode() {
        let source = frame_dimensions(4032, 3024);
        let mode = DisplayPreviewResolutionMode::fullscreen(display_size(3840, 2160));

        assert!(mode.is_fullscreen());
        assert_eq!(
            mode.preview_scale_mode(source),
            PreviewScaleMode::Explicit {
                width: 2880,
                height: 2160,
            }
        );
    }

    #[test]
    fn exiting_fullscreen_switches_back_to_windowed_mode() {
        let source = frame_dimensions(4032, 3024);
        let mode = DisplayPreviewResolutionMode::windowed(display_size(1280, 720));

        assert!(!mode.is_fullscreen());
        assert_eq!(
            mode.preview_scale_mode(source),
            PreviewScaleMode::Explicit {
                width: 960,
                height: 720,
            }
        );
    }

    #[test]
    fn fullscreen_resize_recomputes_target_from_physical_canvas() {
        let mode = DisplayPreviewResolutionMode::fullscreen(display_size(3840, 2160))
            .resized(PhysicalSize::new(2560, 1440));

        assert_eq!(
            mode.preview_scale_mode(frame_dimensions(4032, 3024)),
            PreviewScaleMode::Explicit {
                width: 1920,
                height: 1440,
            }
        );
    }

    #[test]
    fn windowed_resize_updates_physical_preview_canvas() {
        let mode = DisplayPreviewResolutionMode::windowed(display_size(1280, 720))
            .resized(PhysicalSize::new(2560, 1440));

        assert_eq!(
            mode.preview_scale_mode(frame_dimensions(4032, 3024)),
            PreviewScaleMode::Explicit {
                width: 1920,
                height: 1440,
            }
        );
    }

    #[test]
    fn zero_size_resize_preserves_current_preview_mode() {
        let windowed = DisplayPreviewResolutionMode::windowed(display_size(1280, 720));
        let fullscreen = DisplayPreviewResolutionMode::fullscreen(display_size(3840, 2160));

        assert_eq!(windowed.resized(PhysicalSize::new(0, 720)), windowed);
        assert_eq!(fullscreen.resized(PhysicalSize::new(3840, 0)), fullscreen);
    }

    #[test]
    fn unchanged_fullscreen_resize_keeps_same_mode() {
        let mode = DisplayPreviewResolutionMode::fullscreen(display_size(2560, 1440));

        assert_eq!(mode.resized(PhysicalSize::new(2560, 1440)), mode);
    }

    #[test]
    fn display_overlay_enabled_builds_present_overlay() {
        let start = Instant::now();
        let meter = DisplayFpsMeter::new(start);
        let overlay = display_fps_overlay_for_present(
            DisplayCliOverlay::WithOverlay,
            meter,
            24.0,
            display_size(3840, 2160),
        )
        .expect("overlay");

        assert_eq!(overlay.text(), "3840 x 2160\n\n24.00 fps");
    }

    #[test]
    fn display_overlay_disabled_skips_present_overlay() {
        let start = Instant::now();
        let meter = DisplayFpsMeter::new(start);

        assert_eq!(
            display_fps_overlay_for_present(
                DisplayCliOverlay::NoOverlay,
                meter,
                24.0,
                display_size(3840, 2160)
            ),
            None
        );
    }

    #[test]
    fn display_usage_does_not_expose_preview_target_flags() {
        let usage = crate::cli::Cli::usage();

        for hidden_or_unwanted in [
            "--display-resolution",
            "--preview-resolution",
            "--preview-target",
            "--window-size",
            "--fullscreen",
            "--internal-measure",
        ] {
            assert!(
                !usage.contains(hidden_or_unwanted),
                "usage unexpectedly exposes {hidden_or_unwanted}"
            );
        }
    }

    #[test]
    fn display_keyboard_f_toggles_fullscreen() {
        assert_eq!(
            display_keyboard_action(Some(Keycode::F), true, false),
            DisplayKeyboardAction::ToggleFullscreen
        );
    }

    #[test]
    fn display_keyboard_escape_exits() {
        assert_eq!(
            display_keyboard_action(Some(Keycode::Escape), true, false),
            DisplayKeyboardAction::Exit
        );
    }

    #[test]
    fn display_keyboard_ignores_repeated_or_released_keys() {
        assert_eq!(
            display_keyboard_action(Some(Keycode::F), true, true),
            DisplayKeyboardAction::None
        );
        assert_eq!(
            display_keyboard_action(Some(Keycode::F), false, false),
            DisplayKeyboardAction::None
        );
    }
}
