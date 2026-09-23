use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::Duration;

#[rustfmt::skip]
use mcraw4vulkan::display_window::{
    DisplayCliBackend,
    DisplayCliOverlay,
    DisplayCliRunConfig,
    DisplayCliSettings,
    DisplayCliSound,
    DisplayCliVignette,
    DisplayCliVsync,
    EmbeddedDisplayPreview,
    EmbeddedDisplayPreviewPrepared,
    EmbeddedDisplayPreviewPosition,
    EmbeddedDisplayPreviewProgress,
    EmbeddedDisplayPreviewTarget,
};
use mcraw4vulkan_display_sound::{DisplaySoundPlan, DisplaySoundSession};
use mcraw4vulkan_sdl2_wgpu_surface::{RenderFrameContext, WindowSize};

use crate::gui_settings::{DecodeMode, GuiSettings, PreviewTiming};

pub const NO_SELECTION_MESSAGE: &str = "Select a playlist file to preview.";
pub const ALREADY_ACTIVE_MESSAGE: &str = "Preview is already active.";
pub const PREVIEW_PAUSED_MESSAGE: &str = "Preview paused.";
pub const PREVIEW_RESUMED_MESSAGE: &str = "Preview resumed.";
pub const PREVIEW_LOADING_MESSAGE: &str = "Preview is still loading.";
pub const PREVIEW_REPAINT_INTERVAL: Duration = Duration::from_millis(16);
pub const PREVIEW_EVENT_SKIP_RETRY_INTERVAL: Duration = Duration::ZERO;
pub const PREVIEW_NORMAL_PRESENT_MODE: wgpu::PresentMode = wgpu::PresentMode::AutoVsync;
pub const PREVIEW_VSYNC_PRESENT_MODE: wgpu::PresentMode = wgpu::PresentMode::Fifo;
pub const PREVIEW_MAX_SPEED_PRESENT_MODE: wgpu::PresentMode = wgpu::PresentMode::AutoNoVsync;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreviewLogicalRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl PreviewLogicalRect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Option<Self> {
        (width > 0.0 && height > 0.0).then_some(Self {
            x,
            y,
            width,
            height,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreviewSettings {
    pub backend: DisplayCliBackend,
    pub vignette: DisplayCliVignette,
    pub vsync: DisplayCliVsync,
    pub settings: DisplayCliSettings,
    pub overlay: DisplayCliOverlay,
}

impl PreviewSettings {
    pub fn from_gui(settings: &GuiSettings) -> Self {
        Self {
            backend: match settings.decode_mode {
                DecodeMode::Gpu => DisplayCliBackend::Gpu,
                DecodeMode::Cpu => DisplayCliBackend::Cpu,
            },
            vignette: if settings.quick_preview_vignette {
                DisplayCliVignette::WithCorrection
            } else {
                DisplayCliVignette::NoCorrection
            },
            vsync: match settings.quick_preview_timing {
                PreviewTiming::VsyncOn => DisplayCliVsync::Vsync,
                PreviewTiming::MaxSpeed => DisplayCliVsync::NoVsync,
            },
            settings: settings.optimizer_profile.display_settings(),
            overlay: if settings.quick_preview_fps_overlay {
                DisplayCliOverlay::WithOverlay
            } else {
                DisplayCliOverlay::NoOverlay
            },
        }
    }

    pub fn display_config(self, source_path: impl Into<PathBuf>) -> DisplayCliRunConfig {
        let mut config = DisplayCliRunConfig::production_default(source_path);
        config.backend = self.backend;
        config.vignette = self.vignette;
        config.vsync = self.vsync;
        config.settings = self.settings;
        config.overlay = self.overlay;
        let _ = mcraw4vulkan::display_window::apply_effective_display_settings(&mut config);
        config
    }

    pub fn display_config_with_sound(
        self,
        source_path: impl Into<PathBuf>,
        sound_enabled: bool,
    ) -> DisplayCliRunConfig {
        let mut config = self.display_config(source_path);
        if sound_enabled {
            config.sound = DisplayCliSound::WithSound;
        }
        config
    }

    pub fn gui_sound_enabled(self) -> bool {
        self.vsync == DisplayCliVsync::Vsync
    }

    pub fn surface_present_mode(self) -> wgpu::PresentMode {
        match self.vsync {
            DisplayCliVsync::Vsync => PREVIEW_VSYNC_PRESENT_MODE,
            DisplayCliVsync::NoVsync => PREVIEW_MAX_SPEED_PRESENT_MODE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreviewStartSpec {
    pub source_path: PathBuf,
    pub display_name: String,
    pub settings: PreviewSettings,
    pub sound_enabled: bool,
}

struct ActivePreview {
    core: EmbeddedDisplayPreview,
    spec: PreviewStartSpec,
    target: EmbeddedDisplayPreviewTarget,
    request_id: u64,
    paused: bool,
    seek_pending: bool,
    sound: Option<DisplaySoundSession>,
    sound_start_pending: bool,
    sound_start_requires_video_advance: bool,
}

impl ActivePreview {
    fn stop_sound(&mut self) {
        if let Some(sound) = self.sound.as_mut() {
            sound.stop();
        }
        self.sound = None;
        self.sound_start_pending = false;
        self.sound_start_requires_video_advance = false;
    }

    fn pause_sound(&mut self) {
        if let Some(sound) = self.sound.as_mut() {
            sound.pause();
        }
        self.sound_start_pending = false;
        self.sound_start_requires_video_advance = false;
    }

    fn rebase_sound_to_current_frame(
        &mut self,
        start_pending: bool,
        requires_video_advance: bool,
    ) -> Option<String> {
        let position = self.core.playback_position();
        let frame_index = position.current_frame_index.unwrap_or(0);
        self.rebase_sound_to_frame(
            frame_index,
            position.frame_count,
            start_pending,
            requires_video_advance,
        )
    }

    fn rebase_sound_to_frame(
        &mut self,
        frame_index: usize,
        frame_count: usize,
        start_pending: bool,
        requires_video_advance: bool,
    ) -> Option<String> {
        let sound = self.sound.as_mut()?;
        let sample_frame = sound.sample_frame_offset_for_frame(frame_index, frame_count);
        if let Err(error) = sound.seek_to_sample_frame_paused(sample_frame) {
            self.sound = None;
            self.sound_start_pending = false;
            self.sound_start_requires_video_advance = false;
            return Some(preview_audio_failed_message(&error));
        }
        self.sound_start_pending = start_pending;
        self.sound_start_requires_video_advance = requires_video_advance;
        None
    }

    fn after_surface_submit(&mut self, video_advanced: bool) -> Option<String> {
        if self.paused {
            return None;
        }
        let can_start = self.sound_start_pending
            && (!self.sound_start_requires_video_advance || video_advanced);
        if can_start {
            let position = self.core.playback_position();
            let frame_index = position.current_frame_index.unwrap_or(0);
            let result = if let Some(sound) = self.sound.as_mut() {
                let sample_frame =
                    sound.sample_frame_offset_for_frame(frame_index, position.frame_count);
                sound.start_from_sample_frame_after_video_submit(sample_frame)
            } else {
                Ok(())
            };
            match result {
                Ok(()) => {
                    self.sound_start_pending = false;
                    self.sound_start_requires_video_advance = false;
                }
                Err(error) => {
                    self.stop_sound();
                    return Some(preview_audio_failed_message(&error));
                }
            }
        }
        let pump_result = self
            .sound
            .as_mut()
            .filter(|sound| sound.is_started())
            .map(DisplaySoundSession::pump)
            .transpose();
        if let Err(error) = pump_result {
            self.stop_sound();
            return Some(preview_audio_failed_message(&error));
        }
        None
    }
}

struct PreviewPreparedBundle {
    prepared: EmbeddedDisplayPreviewPrepared,
    sound_plan: Option<DisplaySoundPlan>,
    sound_warning: Option<String>,
}

struct PreviewStartupWorkerMessage {
    result: Result<PreviewPreparedBundle, String>,
}

type PreviewStartupResult = PreviewStartupWorkerMessage;

struct PreviewStartup {
    spec: PreviewStartSpec,
    request_id: u64,
    receiver: Receiver<PreviewStartupResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveVsyncPreviewCadence {
    pub frame_duration: Duration,
    pub present_mode: wgpu::PresentMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreviewPlaybackPosition {
    pub current_frame_index: Option<usize>,
    pub frame_count: usize,
}

impl From<EmbeddedDisplayPreviewPosition> for PreviewPlaybackPosition {
    fn from(position: EmbeddedDisplayPreviewPosition) -> Self {
        Self {
            current_frame_index: position.current_frame_index,
            frame_count: position.frame_count,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewPlaybackControl {
    Disabled,
    Playing,
    Paused,
}

impl PreviewPlaybackControl {
    pub fn label(self) -> &'static str {
        match self {
            Self::Disabled | Self::Paused => "Play",
            Self::Playing => "Pause",
        }
    }

    pub fn enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

struct PreviewAdapterContext {
    info: wgpu::AdapterInfo,
    features: wgpu::Features,
}

enum PreviewStopReason {
    Stopped,
    Finished,
}

// Stopping retains the renderer while the payload reader shuts down
// asynchronously; the sound session is stopped before entering that state.
enum PreviewState {
    Inactive,
    Starting(PreviewStartup),
    Active(Box<ActivePreview>),
    Stopping {
        active: Box<ActivePreview>,
        reason: PreviewStopReason,
    },
    Finished,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PostStopSurfaceRefresh {
    None,
    AwaitingGuiOnlyPresent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreviewStartSurfaceRefresh {
    None,
    AwaitingFirstSubmit { request_id: u64 },
    Pending { request_id: u64 },
    Consumed { request_id: u64 },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerminalSurfaceRefreshRequest {
    follow_up_after_suboptimal: bool,
    stop_owned: bool,
}

impl TerminalSurfaceRefreshRequest {
    pub fn follow_up_after_suboptimal(self) -> bool {
        self.follow_up_after_suboptimal
    }
}

pub enum PreviewFrameStatus {
    Idle,
    Continue { video_advanced: bool },
    Finished(String),
    Error(String),
}

pub struct GuiPreview {
    state: PreviewState,
    repaint_after: Option<Duration>,
    next_request_id: u64,
    pending_status: Option<String>,
    terminal_surface_refresh_pending: Option<TerminalSurfaceRefreshRequest>,
    post_stop_surface_refresh: PostStopSurfaceRefresh,
    preview_start_surface_refresh: PreviewStartSurfaceRefresh,
}

impl Default for GuiPreview {
    fn default() -> Self {
        Self {
            state: PreviewState::Inactive,
            repaint_after: None,
            next_request_id: 1,
            pending_status: None,
            terminal_surface_refresh_pending: None,
            post_stop_surface_refresh: PostStopSurfaceRefresh::None,
            preview_start_surface_refresh: PreviewStartSurfaceRefresh::None,
        }
    }
}

#[rustfmt::skip]
fn spawn_preview_startup_worker(spec: &PreviewStartSpec) -> Receiver<PreviewStartupResult> {
    let config = spec
        .settings
        .display_config_with_sound(spec.source_path.clone(), spec.sound_enabled);
    let sound_enabled = spec.sound_enabled;
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let result = if sound_enabled {
            EmbeddedDisplayPreviewPrepared::prepare_with_sound_plan(config)
                .map(|(prepared, sound_plan)| match sound_plan {
                    Ok(sound_plan) => PreviewPreparedBundle {
                        prepared,
                        sound_plan: Some(sound_plan),
                        sound_warning: None,
                    },
                    Err(error) => PreviewPreparedBundle {
                        prepared,
                        sound_plan: None,
                        sound_warning: Some(preview_audio_unavailable_message(&error)),
                    },
                })
                .map_err(|error| error.to_string())
        } else {
            EmbeddedDisplayPreviewPrepared::prepare(config)
                .map(|prepared| PreviewPreparedBundle {
                    prepared,
                    sound_plan: None,
                    sound_warning: None,
                })
                .map_err(|error| error.to_string())
        };
        let _ = sender.send(PreviewStartupWorkerMessage {
            result,
        });
    });
    receiver
}

fn preview_audio_unavailable_message(error: &impl std::fmt::Display) -> String {
    format!("Preview audio unavailable: {error}. Continuing without sound.")
}

fn preview_audio_failed_message(error: &impl std::fmt::Display) -> String {
    format!("Preview audio failed: {error}. Continuing without sound.")
}

impl GuiPreview {
    pub fn is_running(&self) -> bool {
        matches!(
            self.state,
            PreviewState::Starting(_) | PreviewState::Active(_)
        )
    }

    pub fn needs_frame_work(&self) -> bool {
        matches!(
            self.state,
            PreviewState::Starting(_) | PreviewState::Active(_) | PreviewState::Stopping { .. }
        )
    }

    // The pane is not transferable until frame work and the terminal surface
    // refresh are both drained, even after playback itself has stopped.
    pub fn right_pane_handoff_ready(&self) -> bool {
        !self.needs_frame_work()
            && self.terminal_surface_refresh_pending.is_none()
            && self.post_stop_surface_refresh == PostStopSurfaceRefresh::None
    }

    pub fn repaint_after(&self) -> Option<Duration> {
        if matches!(&self.state, PreviewState::Active(active) if active.paused && !active.seek_pending)
        {
            return None;
        }
        self.needs_frame_work()
            .then_some(self.repaint_after.unwrap_or(PREVIEW_REPAINT_INTERVAL))
    }

    pub fn surface_present_mode(&self) -> wgpu::PresentMode {
        match &self.state {
            PreviewState::Starting(startup) => startup.spec.settings.surface_present_mode(),
            PreviewState::Active(active) => active.spec.settings.surface_present_mode(),
            PreviewState::Inactive
            | PreviewState::Stopping { .. }
            | PreviewState::Finished
            | PreviewState::Error => PREVIEW_NORMAL_PRESENT_MODE,
        }
    }

    pub fn active_vsync_cadence(&self) -> Option<ActiveVsyncPreviewCadence> {
        let PreviewState::Active(active) = &self.state else {
            return None;
        };
        if active.paused {
            return None;
        }
        active_vsync_cadence_from_parts(
            active.spec.settings.vsync,
            active.spec.settings.surface_present_mode(),
            active.core.source_frame_duration(),
        )
    }

    pub fn take_terminal_surface_refresh_request(
        &mut self,
    ) -> Option<TerminalSurfaceRefreshRequest> {
        self.terminal_surface_refresh_pending.take()
    }

    pub fn observe_preview_start_surface_submit(&mut self, suboptimal: bool) -> bool {
        let PreviewStartSurfaceRefresh::AwaitingFirstSubmit { request_id } =
            self.preview_start_surface_refresh
        else {
            return false;
        };
        if self.current_running_request_id() != Some(request_id) {
            self.preview_start_surface_refresh = PreviewStartSurfaceRefresh::None;
            return false;
        }

        self.preview_start_surface_refresh = if suboptimal {
            PreviewStartSurfaceRefresh::Pending { request_id }
        } else {
            PreviewStartSurfaceRefresh::Consumed { request_id }
        };
        suboptimal
    }

    pub fn take_preview_start_surface_refresh_request(&mut self) -> Option<u64> {
        let PreviewStartSurfaceRefresh::Pending { request_id } = self.preview_start_surface_refresh
        else {
            return None;
        };
        if self.current_running_request_id() != Some(request_id) {
            self.preview_start_surface_refresh = PreviewStartSurfaceRefresh::None;
            return None;
        }

        self.preview_start_surface_refresh = PreviewStartSurfaceRefresh::Consumed { request_id };
        Some(request_id)
    }

    fn current_running_request_id(&self) -> Option<u64> {
        match &self.state {
            PreviewState::Starting(startup) => Some(startup.request_id),
            PreviewState::Active(active) => Some(active.request_id),
            PreviewState::Inactive
            | PreviewState::Stopping { .. }
            | PreviewState::Finished
            | PreviewState::Error => None,
        }
    }

    fn clear_preview_start_surface_refresh(&mut self) {
        self.preview_start_surface_refresh = PreviewStartSurfaceRefresh::None;
    }

    fn request_terminal_surface_refresh(&mut self) {
        self.clear_preview_start_surface_refresh();
        self.post_stop_surface_refresh = PostStopSurfaceRefresh::None;
        self.terminal_surface_refresh_pending = Some(TerminalSurfaceRefreshRequest::default());
    }

    fn request_post_stop_surface_refresh_after_gui_present(&mut self) {
        self.clear_preview_start_surface_refresh();
        self.post_stop_surface_refresh = PostStopSurfaceRefresh::AwaitingGuiOnlyPresent;
    }

    pub fn advance_post_stop_surface_refresh(&mut self, preview_frame_work_ran: bool) -> bool {
        if preview_frame_work_ran {
            return false;
        }

        match self.post_stop_surface_refresh {
            PostStopSurfaceRefresh::AwaitingGuiOnlyPresent => {
                self.post_stop_surface_refresh = PostStopSurfaceRefresh::None;
                self.terminal_surface_refresh_pending = Some(TerminalSurfaceRefreshRequest {
                    follow_up_after_suboptimal: true,
                    stop_owned: true,
                });
                true
            }
            PostStopSurfaceRefresh::None => false,
        }
    }

    pub fn request_post_stop_follow_up_surface_refresh(&mut self) {
        self.post_stop_surface_refresh = PostStopSurfaceRefresh::None;
        self.terminal_surface_refresh_pending = Some(TerminalSurfaceRefreshRequest {
            follow_up_after_suboptimal: false,
            stop_owned: true,
        });
    }

    fn clear_pending_terminal_surface_refresh_for_preview_start(&mut self) {
        self.clear_preview_start_surface_refresh();
        self.post_stop_surface_refresh = PostStopSurfaceRefresh::None;
        self.terminal_surface_refresh_pending = None;
    }

    fn enter_error(&mut self) {
        self.request_terminal_surface_refresh();
        self.state = PreviewState::Error;
        self.repaint_after = None;
    }

    fn enter_finished(&mut self) {
        self.request_terminal_surface_refresh();
        self.state = PreviewState::Finished;
        self.repaint_after = None;
    }

    fn complete_stopping_preview(&mut self, reason: PreviewStopReason) -> PreviewFrameStatus {
        match reason {
            PreviewStopReason::Stopped => {
                self.request_post_stop_surface_refresh_after_gui_present();
                self.state = PreviewState::Inactive;
                self.repaint_after = None;
                PreviewFrameStatus::Finished("Preview stopped.".to_string())
            }
            PreviewStopReason::Finished => {
                self.enter_finished();
                PreviewFrameStatus::Finished("Preview finished.".to_string())
            }
        }
    }

    pub fn playback_control(&self) -> PreviewPlaybackControl {
        match &self.state {
            PreviewState::Active(active) if active.paused => PreviewPlaybackControl::Paused,
            PreviewState::Active(_) => PreviewPlaybackControl::Playing,
            PreviewState::Inactive
            | PreviewState::Starting(_)
            | PreviewState::Stopping { .. }
            | PreviewState::Finished
            | PreviewState::Error => PreviewPlaybackControl::Disabled,
        }
    }

    pub fn playback_position(&self) -> Option<PreviewPlaybackPosition> {
        match &self.state {
            PreviewState::Active(active) => Some(active.core.playback_position().into()),
            PreviewState::Inactive
            | PreviewState::Starting(_)
            | PreviewState::Stopping { .. }
            | PreviewState::Finished
            | PreviewState::Error => None,
        }
    }

    #[rustfmt::skip]
    pub fn start_with_timing(
        &mut self,
        selected: Option<(&Path, &str)>,
        settings: &GuiSettings,
    ) -> String {
        if self.needs_frame_work() && !matches!(self.state, PreviewState::Starting(_)) {
            return ALREADY_ACTIVE_MESSAGE.to_string();
        }

        let Some((source_path, display_name)) = selected else {
            return NO_SELECTION_MESSAGE.to_string();
        };
        self.clear_pending_terminal_surface_refresh_for_preview_start();
        let display_name = display_name.to_string();
        let preview_settings = PreviewSettings::from_gui(settings);
        let spec = PreviewStartSpec {
            source_path: source_path.to_path_buf(),
            display_name: display_name.clone(),
            settings: preview_settings,
            sound_enabled: preview_settings.gui_sound_enabled(),
        };
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);
        self.preview_start_surface_refresh =
            PreviewStartSurfaceRefresh::AwaitingFirstSubmit { request_id };
        let receiver = spawn_preview_startup_worker(&spec);
        self.state = PreviewState::Starting(PreviewStartup {
            spec,
            request_id,
            receiver,
        });
        self.repaint_after = Some(PREVIEW_REPAINT_INTERVAL);
        format!(
            "Loading preview with {}: {display_name}.",
            settings.optimizer_profile.label()
        )
    }

    pub fn stop(&mut self) -> String {
        self.clear_preview_start_surface_refresh();
        let state = std::mem::replace(&mut self.state, PreviewState::Inactive);
        match state {
            PreviewState::Starting(_) => {
                self.repaint_after = None;
                "Preview stopped.".to_string()
            }
            PreviewState::Active(mut active) => {
                active.stop_sound();
                active.core.request_shutdown();
                self.repaint_after = Some(PREVIEW_REPAINT_INTERVAL);
                self.state = PreviewState::Stopping {
                    active,
                    reason: PreviewStopReason::Stopped,
                };
                "Preview stopped.".to_string()
            }
            PreviewState::Stopping { active, reason } => {
                self.state = PreviewState::Stopping { active, reason };
                self.repaint_after = Some(PREVIEW_REPAINT_INTERVAL);
                "Preview stopping.".to_string()
            }
            idle @ (PreviewState::Inactive | PreviewState::Finished | PreviewState::Error) => {
                self.state = idle;
                self.repaint_after = None;
                "No active Preview.".to_string()
            }
        }
    }

    pub fn toggle_play_pause(&mut self) -> String {
        match &mut self.state {
            PreviewState::Active(active) if active.paused => {
                active.core.resume_after_pause();
                if let Some(message) = active.rebase_sound_to_current_frame(true, false) {
                    self.pending_status = Some(message);
                }
                active.paused = false;
                self.repaint_after = Some(Duration::ZERO);
                PREVIEW_RESUMED_MESSAGE.to_string()
            }
            PreviewState::Active(active) => {
                active.paused = true;
                active.pause_sound();
                self.repaint_after = None;
                PREVIEW_PAUSED_MESSAGE.to_string()
            }
            PreviewState::Starting(_) => PREVIEW_LOADING_MESSAGE.to_string(),
            PreviewState::Inactive
            | PreviewState::Stopping { .. }
            | PreviewState::Finished
            | PreviewState::Error => "No active Preview.".to_string(),
        }
    }

    pub fn seek_to_frame_index(&mut self, frame_index: usize) -> String {
        match &mut self.state {
            PreviewState::Active(active) => match active.core.seek_to_frame_index(frame_index) {
                Ok(()) => {
                    active.seek_pending = true;
                    let frame_count = active.core.playback_position().frame_count;
                    let start_pending = !active.paused;
                    if let Some(message) = active.rebase_sound_to_frame(
                        frame_index,
                        frame_count,
                        start_pending,
                        start_pending,
                    ) {
                        self.pending_status = Some(message);
                    }
                    self.repaint_after = Some(Duration::ZERO);
                    format!(
                        "Preview seek requested: frame {}.",
                        frame_index.saturating_add(1)
                    )
                }
                Err(error) => format!("Preview seek failed: {error}"),
            },
            PreviewState::Starting(_) => PREVIEW_LOADING_MESSAGE.to_string(),
            PreviewState::Inactive
            | PreviewState::Stopping { .. }
            | PreviewState::Finished
            | PreviewState::Error => "No active Preview.".to_string(),
        }
    }

    pub fn clear_if_source(&mut self, source_path: &Path) -> Option<String> {
        if self
            .current_source_path()
            .is_none_or(|active_path| active_path != source_path)
        {
            return None;
        }

        self.clear_preview_start_surface_refresh();
        let state = std::mem::replace(&mut self.state, PreviewState::Inactive);
        match state {
            PreviewState::Active(mut active) => {
                active.stop_sound();
                active.core.request_shutdown();
                self.state = PreviewState::Stopping {
                    active,
                    reason: PreviewStopReason::Stopped,
                };
                self.repaint_after = Some(PREVIEW_REPAINT_INTERVAL);
            }
            PreviewState::Stopping { active, reason } => {
                self.state = PreviewState::Stopping { active, reason };
                self.repaint_after = Some(PREVIEW_REPAINT_INTERVAL);
            }
            PreviewState::Starting(_) => {
                self.repaint_after = None;
            }
            idle @ (PreviewState::Inactive | PreviewState::Finished | PreviewState::Error) => {
                self.state = idle;
                self.repaint_after = None;
            }
        }
        Some("Preview stopped.".to_string())
    }

    pub fn clear_starting_if_selection_changed(
        &mut self,
        selected_source_path: Option<&Path>,
    ) -> Option<String> {
        let PreviewState::Starting(startup) = &self.state else {
            return None;
        };
        if selected_source_path == Some(startup.spec.source_path.as_path()) {
            return None;
        }

        self.clear_preview_start_surface_refresh();
        self.state = PreviewState::Inactive;
        self.repaint_after = None;
        Some("Preview stopped.".to_string())
    }

    pub fn current_source_path(&self) -> Option<&Path> {
        match &self.state {
            PreviewState::Starting(startup) => Some(startup.spec.source_path.as_path()),
            PreviewState::Active(active) => Some(active.spec.source_path.as_path()),
            PreviewState::Stopping { active, .. } => Some(active.spec.source_path.as_path()),
            PreviewState::Inactive | PreviewState::Finished | PreviewState::Error => None,
        }
    }

    pub fn take_status_message(&mut self) -> Option<String> {
        self.pending_status.take()
    }

    // Starting audio after the matching video submit prevents seek or resume
    // audio from leading the first frame that can actually be displayed.
    pub fn after_surface_submit(&mut self, video_advanced: bool) -> Option<String> {
        let PreviewState::Active(active) = &mut self.state else {
            return None;
        };
        active.after_surface_submit(video_advanced)
    }

    #[allow(clippy::let_and_return)]
    #[rustfmt::skip]
    pub fn render_frame(
        &mut self,
        frame: RenderFrameContext<'_>,
        target: Option<EmbeddedDisplayPreviewTarget>,
        adapter_info: wgpu::AdapterInfo,
        adapter_features: wgpu::Features,
        advance: bool,
    ) -> PreviewFrameStatus {
        let state = std::mem::replace(&mut self.state, PreviewState::Inactive);
        let status = match state {
            idle @ (PreviewState::Inactive | PreviewState::Finished | PreviewState::Error) => {
                self.state = idle;
                PreviewFrameStatus::Idle
            }
            PreviewState::Starting(startup) => self.poll_starting_preview(
                frame,
                target,
                PreviewAdapterContext {
                    info: adapter_info,
                    features: adapter_features,
                },
                advance,
                startup,
            ),
            PreviewState::Active(mut active) => {
                let Some(target) = target else {
                    self.enter_error();
                    return PreviewFrameStatus::Error("Preview target unavailable.".to_string());
                };
                if active.target != target {
                    active.core.resize(target);
                    active.target = target;
                }
                if active.seek_pending || (advance && !active.paused) {
                    self.render_active_frame(
                        frame,
                        active,
                    )
                } else {
                    self.present_active_frame(
                        frame,
                        active,
                    )
                }
            }
            PreviewState::Stopping { active, reason } => self.poll_stopping_preview(active, reason),
        };
        status
    }

    #[rustfmt::skip]
    fn poll_starting_preview(
        &mut self,
        frame: RenderFrameContext<'_>,
        target: Option<EmbeddedDisplayPreviewTarget>,
        adapter: PreviewAdapterContext,
        advance: bool,
        startup: PreviewStartup,
    ) -> PreviewFrameStatus {
        match startup.receiver.try_recv() {
            Ok(message) => {
                let bundle = match message.result {
                    Ok(bundle) => bundle,
                    Err(error) => {
                        self.enter_error();
                        return PreviewFrameStatus::Error(format!("Preview failed: {error}"));
                    }
                };
                let PreviewPreparedBundle {
                    prepared,
                    sound_plan,
                    sound_warning,
                } = bundle;
                let Some(target) = target else {
                    self.enter_error();
                    return PreviewFrameStatus::Error("Preview target unavailable.".to_string());
                };
                let core = {
                        EmbeddedDisplayPreview::from_prepared(
                            prepared,
                            adapter.info,
                            adapter.features,
                            frame.device.clone(),
                            frame.queue.clone(),
                            frame.surface_format,
                            target,
                        )
                        .map(|core| (core, ()))
                };
                match core {
                    Ok((core, _create_timings)) => {
                        if let Some(message) = sound_warning {
                            self.pending_status = Some(message);
                        }
                        let sound = sound_plan
                            .and_then(|plan| match DisplaySoundSession::from_plan(plan) {
                                Ok(sound) => Some(sound),
                                Err(error) => {
                                    self.pending_status =
                                        Some(preview_audio_failed_message(&error));
                                    None
                                }
                            });
                        let sound_start_pending = sound.is_some();
                        let active = Box::new(ActivePreview {
                            core,
                            spec: startup.spec,
                            target,
                            request_id: startup.request_id,
                            paused: false,
                            seek_pending: false,
                            sound,
                            sound_start_pending,
                            sound_start_requires_video_advance: sound_start_pending,
                        });
                        if advance {
                            self.render_active_frame(
                                frame,
                                active
                            )
                        } else {
                            self.repaint_after = Some(skipped_starting_preview_repaint_after());
                            self.state = PreviewState::Active(active);
                            PreviewFrameStatus::Continue {
                                video_advanced: false,
                            }
                        }
                    }
                    Err(error) => {
                        let message = format!("Preview failed: {error}");
                        self.enter_error();
                        PreviewFrameStatus::Error(message)
                    }
                }
            }
            Err(TryRecvError::Empty) => {
                self.repaint_after = Some(PREVIEW_REPAINT_INTERVAL);
                self.state = PreviewState::Starting(startup);
                PreviewFrameStatus::Continue {
                    video_advanced: false,
                }
            }
            Err(TryRecvError::Disconnected) => {
                self.enter_error();
                PreviewFrameStatus::Error("Preview failed: startup worker stopped.".to_string())
            }
        }
    }

    fn present_active_frame(
        &mut self,
        frame: RenderFrameContext<'_>,
        mut active: Box<ActivePreview>,
    ) -> PreviewFrameStatus {
        match active
            .core
            .present_current_frame_with_timings(frame, active.target)
        {
            Ok(frame_result) => {
                let video_advanced = frame_result.timings.new_video_frame;
                self.handle_active_progress(active, frame_result.progress, video_advanced)
            }
            Err(error) => self.preview_error(error),
        }
    }

    fn render_active_frame(
        &mut self,
        frame: RenderFrameContext<'_>,
        mut active: Box<ActivePreview>,
    ) -> PreviewFrameStatus {
        match active
            .core
            .render_next_frame_with_timings(frame, active.target)
        {
            Ok(frame_result) => {
                let video_advanced = frame_result.timings.new_video_frame;
                self.handle_active_progress(active, frame_result.progress, video_advanced)
            }
            Err(error) => self.preview_error(error),
        }
    }

    fn handle_active_progress(
        &mut self,
        mut active: Box<ActivePreview>,
        progress: EmbeddedDisplayPreviewProgress,
        video_advanced: bool,
    ) -> PreviewFrameStatus {
        match progress {
            EmbeddedDisplayPreviewProgress::Continue { repaint_after } => {
                if video_advanced {
                    active.seek_pending = false;
                }
                self.repaint_after =
                    (!active.paused || active.seek_pending).then_some(repaint_after);
                self.state = PreviewState::Active(active);
                PreviewFrameStatus::Continue { video_advanced }
            }
            EmbeddedDisplayPreviewProgress::Finished => {
                active.stop_sound();
                active.core.request_shutdown();
                self.poll_stopping_preview(active, PreviewStopReason::Finished)
            }
        }
    }

    fn preview_error(&mut self, error: impl std::fmt::Display) -> PreviewFrameStatus {
        self.enter_error();
        PreviewFrameStatus::Error(format!("Preview failed: {error}"))
    }

    fn poll_stopping_preview(
        &mut self,
        mut active: Box<ActivePreview>,
        reason: PreviewStopReason,
    ) -> PreviewFrameStatus {
        active.stop_sound();
        match active.core.poll_shutdown() {
            Ok(true) => self.complete_stopping_preview(reason),
            Ok(false) => {
                self.repaint_after = Some(PREVIEW_REPAINT_INTERVAL);
                self.state = PreviewState::Stopping { active, reason };
                PreviewFrameStatus::Continue {
                    video_advanced: false,
                }
            }
            Err(error) => {
                self.enter_error();
                PreviewFrameStatus::Error(format!("Preview cleanup failed: {error}"))
            }
        }
    }
}

fn skipped_starting_preview_repaint_after() -> Duration {
    PREVIEW_EVENT_SKIP_RETRY_INTERVAL
}

fn active_vsync_cadence_from_parts(
    vsync: DisplayCliVsync,
    present_mode: wgpu::PresentMode,
    source_frame_duration: Option<Duration>,
) -> Option<ActiveVsyncPreviewCadence> {
    (vsync == DisplayCliVsync::Vsync && present_mode == PREVIEW_VSYNC_PRESENT_MODE)
        .then_some(source_frame_duration)
        .flatten()
        .filter(|duration| !duration.is_zero())
        .map(|frame_duration| ActiveVsyncPreviewCadence {
            frame_duration,
            present_mode,
        })
}

pub fn physical_preview_target(
    rect: PreviewLogicalRect,
    pixels_per_point: f32,
    window_size: WindowSize,
) -> Option<EmbeddedDisplayPreviewTarget> {
    if window_size.drawable_width == 0 || window_size.drawable_height == 0 {
        return None;
    }

    let scale = pixels_per_point.max(1.0);
    let max_x = window_size.drawable_width as f32;
    let max_y = window_size.drawable_height as f32;
    let x0 = (rect.x * scale).floor().clamp(0.0, max_x);
    let y0 = (rect.y * scale).floor().clamp(0.0, max_y);
    let x1 = ((rect.x + rect.width) * scale).ceil().clamp(0.0, max_x);
    let y1 = ((rect.y + rect.height) * scale).ceil().clamp(0.0, max_y);
    let width = (x1 - x0).max(0.0) as u32;
    let height = (y1 - y0).max(0.0) as u32;

    EmbeddedDisplayPreviewTarget::new(x0 as u32, y0 as u32, width, height)
}

pub fn fullscreen_preview_target(window_size: WindowSize) -> Option<EmbeddedDisplayPreviewTarget> {
    EmbeddedDisplayPreviewTarget::new(
        0,
        0,
        window_size.drawable_width,
        window_size.drawable_height,
    )
}

pub fn clamp_preview_target_to_render_target(
    target: EmbeddedDisplayPreviewTarget,
    render_size: WindowSize,
) -> Option<EmbeddedDisplayPreviewTarget> {
    if render_size.drawable_width == 0 || render_size.drawable_height == 0 {
        return None;
    }
    if target.origin_x >= render_size.drawable_width
        || target.origin_y >= render_size.drawable_height
    {
        return None;
    }

    let x1 = target
        .origin_x
        .saturating_add(target.width)
        .min(render_size.drawable_width);
    let y1 = target
        .origin_y
        .saturating_add(target.height)
        .min(render_size.drawable_height);
    EmbeddedDisplayPreviewTarget::new(
        target.origin_x,
        target.origin_y,
        x1 - target.origin_x,
        y1 - target.origin_y,
    )
}
