use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use egui::{
    Align, Align2, CentralPanel, ColorImage, Context, Event as EguiEvent, FontId, Frame, Image,
    Layout, Modifiers, MouseWheelUnit, OutputCommand, PlatformOutput, PointerButton, Pos2,
    RawInput, Rect, RichText, Sense, Stroke, StrokeKind, TextureHandle, TextureOptions, Vec2,
    ViewportId, ViewportInfo, pos2, vec2,
};
use egui_wgpu::{Renderer, ScreenDescriptor};
use mcraw4vulkan_preflight::{
    PreflightOptions, collect_system_ram, format_system_ram_line, run_preflight,
};
use mcraw4vulkan_sdl2_wgpu_surface::{
    RenderFrameContext, RenderFrameStatus, Sdl2WgpuSurface, Sdl2WgpuSurfaceConfig,
    Sdl2WgpuSurfaceError, WindowSize,
};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::{Keycode, Mod};
use sdl2::mouse::{MouseButton, MouseState, MouseWheelDirection};

use crate::dng_actions::{
    DngActionKind, DngCliFlags, DngCommandSpec, DngProcessController, DngProcessEvent,
    resolve_mcraw4vulkan_binary,
};
use crate::file_chooser::{self, FileChooserOutcome, FileChooserPoll, FileChooserStart};
use crate::gui_settings::{DecodeMode, GuiSettings, OptimizerProfile, PreviewTiming};
use crate::lazy_loop::{LazyFrameDecision, LazyRepaintState, WaitMode, timeout_millis};
use crate::main_view;
use crate::optimizer_actions::{
    GuiOptimizer, OPTIMIZER_REPAINT_INTERVAL, OptimizerAnswer, OptimizerCancelRequest,
    OptimizerCommandSpec, OptimizerProcessEvent, OptimizerTerminalResult,
    optimizer_success_terminal_result,
};
use crate::pipe_example::{self, PipeExamplePanel};
use crate::playlist::{
    DesiredMountState, LiveMountState, Playlist, PlaylistAddSummary, PlaylistEntry,
};
use crate::playlist_store::{PlaylistLoadOutcome, PlaylistSaveOutcome, PlaylistStore};
use crate::preflight_view::{LineStatus, PREFLIGHT_PANEL_TITLE, PreflightViewModel};
use crate::preview::{
    self, GuiPreview, PreviewFrameStatus, PreviewLogicalRect, PreviewPlaybackControl,
    PreviewPlaybackPosition,
};
use crate::style;

const WINDOW_TITLE: &str = "mcraw4vulkan";
const WINDOW_WIDTH: u32 = 1600;
const WINDOW_HEIGHT: u32 = 900;
const SUBTITLE: &str = "GPU decoding of MotionCam RAW files";
const SPLASH_IMAGE_BYTES: &[u8] = include_bytes!("../../../packaging/splash.png");
const SPLASH_IMAGE_WIDTH: usize = 1600;
const SPLASH_IMAGE_HEIGHT: usize = 347;
const SPLASH_VERSION_LABEL: &str = concat!("Version ", env!("CARGO_PKG_VERSION"));
const MAIN_MIN_COLUMN_HEIGHT: f32 = 520.0;
const PREFLIGHT_SPLASH_MIN_VISIBLE_MS: u64 = 5_000;
const FILE_CHOOSER_POLL_INTERVAL: Duration = Duration::from_millis(125);
const DNG_PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(500);
const PIPE_EXAMPLE_SCROLL_ID: &str = "pipe-example-scroll";
const PLAYLIST_SCROLL_ID: &str = "playlist-scroll";
const SHARED_CONTROLS_SCROLL_ID: &str = "shared-controls-scroll";
const PLAYLIST_STATUS_RESERVED_LINES: usize = 3;
const HEADER_QUIT_BASE_BUTTON_WIDTH: f32 = 56.0;
const MACOS_MACFUSE_VFS_MOUNT_LIMIT: usize = 64;
const MACOS_MOUNT_LIMIT_WARNING_TITLE: &str = "macOS DNG mount limit";
const MACOS_MOUNT_LIMIT_WARNING_BUTTON: &str = "OK";
const OPTIMIZER_SHORT_INPUT_MESSAGE: &str =
    "Optimization testing requires at least 600 frames for accuracy.";

#[rustfmt::skip]
pub fn run() -> Result<(), GuiError> {
    run_with_options(
    )
}

#[rustfmt::skip]
pub fn run_with_options(
) -> Result<(), GuiError> {
    // The surface, SDL event pump, and all frame submissions stay on this thread.
    // Resize and reconfigure requests are handled by the same event loop.
    let mut surface = Sdl2WgpuSurface::new(default_surface_config())?;
    let mut event_pump = surface.event_pump()?;

    println!("SDL video driver: {}", surface.current_video_driver());
    let adapter_info = surface.adapter_info().clone();
    println!(
        "wgpu adapter: {} ({:?})",
        adapter_info.name, adapter_info.backend
    );
    let gui_gpu_model = gui_gpu_model_from_adapter_info(&adapter_info);

    let context = Context::default();
    style::apply_project_style(&context);
    let splash_texture = load_splash_texture(&context)?;

    let mut renderer = Renderer::new(surface.device(), surface.surface_format(), None, 1, false);
    let start = Instant::now();
    let mut input = EguiInputState::new(start);
    let mut scheduler = LazyRepaintState::new(start);
    let ram_line = format_system_ram_line(&collect_system_ram());
    let mut app = GuiApp::new(ram_line);
    apply_pending_splash_window_size(&mut app, &mut surface)?;
    app.splash_texture = Some(splash_texture);
    app.load_playlist_from_store();

    render_splash_frame(
        &mut surface,
        &mut renderer,
        &context,
        &mut input,
        &mut app,
        &mut scheduler,
        start,
    )?;

    let report = run_preflight(PreflightOptions::gui_blocking_policy());
    app.set_preflight_result(
        PreflightViewModel::from_report_with_gui_gpu(&report, gui_gpu_model.as_deref()),
        Instant::now(),
    );
    scheduler.mark_dirty();
    render_splash_frame(
        &mut surface,
        &mut renderer,
        &context,
        &mut input,
        &mut app,
        &mut scheduler,
        start,
    )?;

    while !app.quit_requested() {
        let now = Instant::now();
        scheduler.mark_elapsed_repaint_deadline(now);

        let wait_mode = scheduler.wait_mode(now);
        match wait_mode {
            WaitMode::RenderNow => {
                if drain_ready_events(
                    &mut event_pump,
                    &mut surface,
                    &mut input,
                    &mut app,
                    &mut scheduler,
                )? {
                    break;
                }
                render_splash_frame(
                    &mut surface,
                    &mut renderer,
                    &context,
                    &mut input,
                    &mut app,
                    &mut scheduler,
                    start,
                )?;
            }
            WaitMode::WaitIndefinitely => {
                let event = event_pump.wait_event();
                if handle_event(event, &mut surface, &mut input, &mut app, &mut scheduler)? {
                    break;
                }
                if drain_ready_events(
                    &mut event_pump,
                    &mut surface,
                    &mut input,
                    &mut app,
                    &mut scheduler,
                )? {
                    break;
                }
            }
            WaitMode::WaitTimeout(timeout) => {
                if let Some(event) = event_pump.wait_event_timeout(timeout_millis(timeout)) {
                    if handle_event(event, &mut surface, &mut input, &mut app, &mut scheduler)? {
                        break;
                    }
                    if drain_ready_events(
                        &mut event_pump,
                        &mut surface,
                        &mut input,
                        &mut app,
                        &mut scheduler,
                    )? {
                        break;
                    }
                } else {
                    scheduler.mark_elapsed_repaint_deadline(Instant::now());
                }
            }
        }
    }

    app.cleanup_optimizer_on_exit();
    app.cleanup_dng_on_exit();

    Ok(())
}

#[derive(Debug)]
pub enum GuiError {
    Surface(Sdl2WgpuSurfaceError),
    SplashImage(String),
}

impl fmt::Display for GuiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Surface(error) => write!(formatter, "{error}"),
            Self::SplashImage(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for GuiError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Surface(error) => Some(error),
            Self::SplashImage(_) => None,
        }
    }
}

impl From<Sdl2WgpuSurfaceError> for GuiError {
    fn from(error: Sdl2WgpuSurfaceError) -> Self {
        Self::Surface(error)
    }
}

struct GuiApp {
    phase: AppPhase,
    preflight: PreflightViewModel,
    splash_texture: Option<TextureHandle>,
    ram_line: String,
    playlist: Playlist,
    playlist_store: PlaylistStore,
    settings: GuiSettings,
    preview: GuiPreview,
    dng_processes: DngProcessController,
    mount_all_dngs_batch: Option<MountAllDngsBatch>,
    macos_mount_limit_warning: Option<MacosMountLimitWarning>,
    dng_binary: PathBuf,
    optimizer: GuiOptimizer,
    optimizer_binary: PathBuf,
    preview_area: Option<PreviewLogicalRect>,
    preview_fullscreen: bool,
    preview_fullscreen_exit_pending: bool,
    splash_window_size_pending: bool,
    main_window_maximize_pending: bool,
    defer_preview_advance_once: bool,
    more_options_visible: bool,
    pipe_example: Option<PipeExamplePanel>,
    right_pane_owner: RightPaneOwner,
    right_pane_frame_owner: RightPaneOwner,
    pending_right_pane_request: Option<RightPaneRequest>,
    status_message: Option<String>,
    drop_in_progress: bool,
    pending_drop_files: Vec<PathBuf>,
    pending_file_chooser: Option<file_chooser::PendingFileChooser>,
    quit_requested: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct PreviewRenderOutcome {
    dirty: bool,
    video_advanced: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DngMountRequest {
    entry_id: u64,
    source_path: PathBuf,
    display_name: String,
}

impl DngMountRequest {
    fn from_entry(entry: &PlaylistEntry) -> Self {
        Self {
            entry_id: entry.id,
            source_path: entry.source_path.clone(),
            display_name: entry.display_name.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DngMountFailure {
    entry_id: u64,
    display_name: String,
    reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MacosMountLimitWarning {
    playlist_count: usize,
}

fn should_show_macos_mount_limit_warning(is_macos: bool, playlist_count: usize) -> bool {
    is_macos && playlist_count >= MACOS_MACFUSE_VFS_MOUNT_LIMIT
}

fn macos_mount_limit_warning_body(playlist_count: usize) -> String {
    format!(
        "macFUSE VFS supports a maximum of \
         {MACOS_MACFUSE_VFS_MOUNT_LIMIT} simultaneous mounts system-wide.\n\n\
         This Playlist contains {playlist_count} clips. Other macFUSE volumes use the same \
         mount slots, so some DNG folders may fail to mount."
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MountAllDngsPhase {
    WaitingForUnmountAll,
    WaitingForMount,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MountAllDngsBatch {
    phase: MountAllDngsPhase,
    pending: VecDeque<DngMountRequest>,
    current_entry_id: Option<u64>,
    total: usize,
    succeeded: usize,
    failures: Vec<DngMountFailure>,
    dng_flags: DngCliFlags,
}

impl MountAllDngsBatch {
    fn new(pending: VecDeque<DngMountRequest>, dng_flags: DngCliFlags) -> Self {
        let total = pending.len();
        Self {
            phase: MountAllDngsPhase::WaitingForUnmountAll,
            pending,
            current_entry_id: None,
            total,
            succeeded: 0,
            failures: Vec::new(),
            dng_flags,
        }
    }

    fn next_mount_ordinal(&self) -> usize {
        self.total
            .saturating_sub(self.pending.len())
            .saturating_add(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnmountAllDngsOrigin {
    Standalone,
    MountAllPreparation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DngMountStartError {
    MissingEntry,
    MissingSource,
    AlreadyMounted,
    DuplicateLiveSource,
    ProcessStart(String),
}

impl DngMountStartError {
    fn selected_status(&self) -> String {
        match self {
            Self::MissingEntry => "Selected playlist file is no longer available.".to_string(),
            Self::MissingSource => "Selected file no longer exists.".to_string(),
            Self::AlreadyMounted => "DNG is already mounted for this row.".to_string(),
            Self::DuplicateLiveSource => {
                "This source is already mounted in this GUI session. Use Unmount all before mounting duplicate sources."
                    .to_string()
            }
            Self::ProcessStart(error) => format!("DNG action failed: {error}"),
        }
    }

    fn batch_reason(&self) -> String {
        match self {
            Self::MissingEntry => "playlist entry is no longer available".to_string(),
            Self::MissingSource => "source file no longer exists".to_string(),
            Self::AlreadyMounted => "DNG is already mounted for this row".to_string(),
            Self::DuplicateLiveSource => {
                "source is already mounted in this GUI session".to_string()
            }
            Self::ProcessStart(_) => "DNG action failed before the mount became active".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppPhase {
    PreflightStarting,
    PreflightReadyVisible { transition_at: Instant },
    PreflightNotReady,
    MainSkeleton,
}

// Only one functional owner may use the right pane at a time. A replacement
// waits until the current owner's renderer or child process has finished teardown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RightPaneOwner {
    Idle,
    Preview,
    PipeExample,
    Optimizer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RightPaneRequest {
    StartPreview(PreparedPreviewRequest),
    ShowPipeExample(PreparedPipeRequest),
    StartOptimizer(PreparedOptimizerRequest),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedPreviewRequest {
    entry_id: u64,
    source_path: PathBuf,
    display_name: String,
    settings: GuiSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedPipeRequest {
    panel: PipeExamplePanel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedOptimizerRequest {
    entry_id: u64,
    source_path: PathBuf,
    display_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RightPaneActivation {
    Immediate,
    AfterTeardown,
}

impl GuiApp {
    fn new(ram_line: String) -> Self {
        Self {
            phase: AppPhase::PreflightStarting,
            preflight: PreflightViewModel::starting(),
            splash_texture: None,
            ram_line,
            playlist: Playlist::new(),
            playlist_store: default_playlist_store(),
            settings: GuiSettings::default(),
            preview: GuiPreview::default(),
            dng_processes: DngProcessController::new(),
            mount_all_dngs_batch: None,
            macos_mount_limit_warning: None,
            dng_binary: resolve_mcraw4vulkan_binary(),
            optimizer: GuiOptimizer::default(),
            optimizer_binary: resolve_mcraw4vulkan_binary(),
            preview_area: None,
            preview_fullscreen: false,
            preview_fullscreen_exit_pending: false,
            splash_window_size_pending: true,
            main_window_maximize_pending: false,
            defer_preview_advance_once: false,
            more_options_visible: false,
            pipe_example: None,
            right_pane_owner: RightPaneOwner::Idle,
            right_pane_frame_owner: RightPaneOwner::Idle,
            pending_right_pane_request: None,
            status_message: None,
            drop_in_progress: false,
            pending_drop_files: Vec::new(),
            pending_file_chooser: None,
            quit_requested: false,
        }
    }

    fn set_preflight_result(&mut self, preflight: PreflightViewModel, now: Instant) {
        self.phase = phase_for_preflight(&preflight, now);
        self.preflight = preflight;
    }

    fn advance_startup_phase(&mut self, now: Instant) {
        if let AppPhase::PreflightReadyVisible { transition_at } = self.phase {
            if now >= transition_at {
                self.phase = AppPhase::MainSkeleton;
                self.main_window_maximize_pending = true;
            }
        }
    }

    fn preflight_repaint_after(&self, now: Instant) -> Option<Duration> {
        match self.phase {
            AppPhase::PreflightReadyVisible { transition_at } if transition_at > now => {
                Some(transition_at.duration_since(now))
            }
            _ => None,
        }
    }

    fn quit_requested(&self) -> bool {
        self.quit_requested
    }

    fn main_layout_mode(&self) -> main_view::MainLayoutMode {
        if self.more_options_visible {
            main_view::MainLayoutMode::Expanded
        } else {
            main_view::MainLayoutMode::Condensed
        }
    }

    fn toggle_more_options(&mut self) {
        self.more_options_visible = !self.more_options_visible;
    }

    fn load_playlist_from_store(&mut self) {
        match self.playlist_store.load() {
            PlaylistLoadOutcome::Missing => {}
            PlaylistLoadOutcome::Loaded { playlist } => {
                let mounted_count = playlist
                    .entries()
                    .iter()
                    .filter(|entry| entry.desired_mount_state == DesiredMountState::Mounted)
                    .count();
                self.playlist = playlist;
                if mounted_count == 0 {
                    self.set_status("Playlist loaded.");
                } else {
                    self.set_status(
                        "Playlist loaded. Mounted-intent rows are not live until mounted this session.",
                    );
                }
            }
            PlaylistLoadOutcome::Unavailable { message }
            | PlaylistLoadOutcome::Invalid { message } => self.set_status(message),
        }
    }

    fn save_playlist_after_mutation(&mut self) -> bool {
        match self.playlist_store.save(&self.playlist) {
            PlaylistSaveOutcome::Saved { .. } => true,
            PlaylistSaveOutcome::Unavailable { message }
            | PlaylistSaveOutcome::Failed { message } => {
                self.set_status(message);
                false
            }
        }
    }

    #[cfg(test)]
    fn preflight_needs_startup_run(&self) -> bool {
        self.phase == AppPhase::PreflightStarting
    }

    fn ui(&mut self, context: &Context, now: Instant) {
        if let Some(repaint_after) = self.preflight_repaint_after(now) {
            context.request_repaint_after(repaint_after);
        }
        if self.pending_file_chooser.is_some() {
            context.request_repaint_after(FILE_CHOOSER_POLL_INTERVAL);
        }
        if let Some(repaint_after) = self.preview.repaint_after() {
            context.request_repaint_after(repaint_after);
        }
        if self.dng_processes.has_active_work() {
            context.request_repaint_after(DNG_PROCESS_POLL_INTERVAL);
        }
        if self.optimizer.is_running() {
            context.request_repaint_after(OPTIMIZER_REPAINT_INTERVAL);
        }
        if self.advance_right_pane_transition() {
            context.request_repaint();
        }
        self.right_pane_frame_owner = self.right_pane_owner;
        self.preview_area = None;

        if self.preview_fullscreen {
            CentralPanel::default()
                .frame(Frame::new().fill(style::background()))
                .show(context, |_ui| {});
            return;
        }

        CentralPanel::default()
            .frame(Frame::new().fill(style::background()))
            .show(context, |ui| match self.phase {
                AppPhase::MainSkeleton => self.main_gui(ui),
                AppPhase::PreflightStarting
                | AppPhase::PreflightReadyVisible { .. }
                | AppPhase::PreflightNotReady => self.preflight_splash(ui),
            });
        self.draw_macos_mount_limit_warning(context);
    }

    fn main_gui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(style::OUTER_MARGIN);
        ui.horizontal(|ui| {
            ui.add_space(style::OUTER_MARGIN);
            ui.vertical(|ui| {
                ui.add_space(2.0);
                ui.label(
                    RichText::new(WINDOW_TITLE)
                        .size(style::HEADING_FONT_SIZE)
                        .color(style::header_text()),
                );
                ui.label(
                    RichText::new(SUBTITLE)
                        .size(style::BODY_FONT_SIZE)
                        .color(style::body_text()),
                );
            });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add_space(style::OUTER_MARGIN);
                if ui
                    .add_sized(
                        vec2(header_quit_button_width(), ui.spacing().interact_size.y),
                        egui::Button::new("Quit"),
                    )
                    .clicked()
                {
                    self.quit_requested = true;
                }
            });
        });
        ui.add_space(14.0);
        draw_divider(ui);
        self.main_skeleton(ui);
    }

    fn preflight_splash(&self, ui: &mut egui::Ui) {
        self.splash_image(ui);
        ui.vertical_centered(|ui| {
            let width = ui.available_width().clamp(320.0, style::SPLASH_PANEL_WIDTH);
            ui.set_width(width);
            Frame::new()
                .fill(style::panel())
                .stroke(Stroke::new(1.0, style::divider()))
                .corner_radius(4)
                .inner_margin(24)
                .show(ui, |ui| {
                    self.preflight_panel(ui);
                });
        });
    }

    fn splash_image(&self, ui: &mut egui::Ui) {
        let size = splash_image_display_size(ui.available_width());
        if let Some(texture) = &self.splash_texture {
            ui.add(Image::from_texture(texture).fit_to_exact_size(size));
        } else {
            ui.add_space(size.y);
        }
    }

    fn preflight_panel(&self, ui: &mut egui::Ui) {
        ui.label(
            RichText::new(PREFLIGHT_PANEL_TITLE)
                .size(style::PANEL_TITLE_FONT_SIZE)
                .color(style::header_text()),
        );
        ui.add_space(6.0);
        draw_divider(ui);
        ui.add_space(14.0);

        let status_color = if self.preflight.ready {
            style::bright_blue()
        } else {
            style::header_text()
        };
        ui.label(
            RichText::new(&self.preflight.status_line)
                .size(style::STATUS_FONT_SIZE)
                .color(status_color),
        );
        ui.label(
            RichText::new(SPLASH_VERSION_LABEL)
                .size(style::SMALL_FONT_SIZE)
                .color(style::body_text()),
        );
        ui.label(
            RichText::new(&self.preflight.detail_line)
                .size(style::DETAIL_FONT_SIZE)
                .color(style::muted_text()),
        );
        ui.add_space(16.0);
        draw_detail_line(ui, &self.ram_line, LineStatus::Informational);
        for line in &self.preflight.lines {
            draw_detail_line(ui, &line.text, line.status);
        }
    }

    fn main_skeleton(&mut self, ui: &mut egui::Ui) {
        ui.add_space(18.0);
        let layout = main_view::target_main_layout(self.main_layout_mode(), ui.available_width());
        let column_height =
            (ui.available_height() - style::OUTER_MARGIN).max(MAIN_MIN_COLUMN_HEIGHT);

        ui.with_layout(Layout::left_to_right(Align::Min), |ui| {
            ui.add_space(layout.outer_margin);
            if layout.mode == main_view::MainLayoutMode::Expanded {
                self.draw_expanded_controls_region(ui, layout, column_height);
                ui.add_space(layout.column_gap);
            } else {
                self.draw_condensed_controls_region(ui, layout, column_height);
                ui.add_space(layout.column_gap);
            }
            draw_column_frame(
                ui,
                layout.display_width,
                column_height,
                |ui, content_height| {
                    self.draw_display_playback_region(ui, content_height);
                },
            );
            ui.add_space(layout.outer_margin);
        });
    }

    fn draw_condensed_controls_region(
        &mut self,
        ui: &mut egui::Ui,
        layout: main_view::MainLayout,
        column_height: f32,
    ) {
        let content_height = main_view::column_content_height(column_height);
        let row_layout = main_view::shared_controls_row_layout(content_height);
        if row_layout.outer_scroll_required {
            self.draw_scrolled_primary_controls(ui, layout.primary_controls_width, column_height);
        } else {
            self.draw_primary_controls_frame(ui, layout.primary_controls_width, column_height);
        }
    }

    fn draw_expanded_controls_region(
        &mut self,
        ui: &mut egui::Ui,
        layout: main_view::MainLayout,
        column_height: f32,
    ) {
        let content_height = main_view::column_content_height(column_height);
        let row_layout = main_view::shared_controls_row_layout(content_height);
        let controls_width =
            layout.primary_controls_width + layout.column_gap + layout.more_options_width;

        if row_layout.outer_scroll_required {
            ui.allocate_ui_with_layout(
                vec2(controls_width, column_height),
                Layout::top_down(Align::Min),
                |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt(SHARED_CONTROLS_SCROLL_ID)
                        .auto_shrink([false, false])
                        .max_height(column_height)
                        .show(ui, |ui| {
                            ui.set_width(controls_width);
                            ui.with_layout(Layout::left_to_right(Align::Min), |ui| {
                                self.draw_primary_controls_frame_for_rows(
                                    ui,
                                    layout.primary_controls_width,
                                    row_layout,
                                );
                                ui.add_space(layout.column_gap);
                                self.draw_more_options_frame_for_rows(
                                    ui,
                                    layout.more_options_width,
                                    row_layout,
                                );
                            });
                        });
                },
            );
        } else {
            self.draw_primary_controls_frame(ui, layout.primary_controls_width, column_height);
            ui.add_space(layout.column_gap);
            self.draw_more_options_frame(ui, layout.more_options_width, column_height);
        }
    }

    fn draw_scrolled_primary_controls(
        &mut self,
        ui: &mut egui::Ui,
        width: f32,
        column_height: f32,
    ) {
        let content_height = main_view::column_content_height(column_height);
        let row_layout = main_view::shared_controls_row_layout(content_height);
        ui.allocate_ui_with_layout(
            vec2(width, column_height),
            Layout::top_down(Align::Min),
            |ui| {
                egui::ScrollArea::vertical()
                    .id_salt(SHARED_CONTROLS_SCROLL_ID)
                    .auto_shrink([false, false])
                    .max_height(column_height)
                    .show(ui, |ui| {
                        self.draw_primary_controls_frame_for_rows(ui, width, row_layout);
                    });
            },
        );
    }

    fn draw_primary_controls_frame(&mut self, ui: &mut egui::Ui, width: f32, height: f32) {
        draw_column_frame(ui, width, height, |ui, content_height| {
            let row_layout = main_view::shared_controls_row_layout(content_height);
            self.draw_primary_controls_column(ui, row_layout);
        });
    }

    fn draw_primary_controls_frame_for_rows(
        &mut self,
        ui: &mut egui::Ui,
        width: f32,
        row_layout: main_view::SharedControlsRowLayout,
    ) {
        let frame_height =
            row_layout.total_used_height + main_view::COLUMN_FRAME_INNER_MARGIN * 2.0;
        draw_column_frame(ui, width, frame_height, |ui, _content_height| {
            self.draw_primary_controls_column(ui, row_layout);
        });
    }

    fn draw_more_options_frame(&mut self, ui: &mut egui::Ui, width: f32, height: f32) {
        draw_column_frame(ui, width, height, |ui, content_height| {
            let row_layout = main_view::shared_controls_row_layout(content_height);
            self.draw_more_options_column(ui, row_layout);
        });
    }

    fn draw_more_options_frame_for_rows(
        &mut self,
        ui: &mut egui::Ui,
        width: f32,
        row_layout: main_view::SharedControlsRowLayout,
    ) {
        let frame_height =
            row_layout.total_used_height + main_view::COLUMN_FRAME_INNER_MARGIN * 2.0;
        draw_column_frame(ui, width, frame_height, |ui, _content_height| {
            self.draw_more_options_column(ui, row_layout);
        });
    }

    fn draw_display_playback_region(&mut self, ui: &mut egui::Ui, content_height: f32) {
        let owner = self.right_pane_frame_owner;
        let response = draw_display_playback_column(
            ui,
            content_height,
            DisplayPlaybackState {
                owner,
                pipe_example: self.pipe_example.as_ref(),
                optimizer: &self.optimizer,
                playback_control: self.preview.playback_control(),
                preview_running: self.preview.is_running(),
                playback_position: self.preview.playback_position(),
            },
        );
        self.preview_area = response.preview_rect;
        if response.copy_clicked {
            self.request_pipe_command_copy(ui.ctx());
        }
        if response.play_pause_clicked {
            self.toggle_preview_play_pause();
            ui.ctx().request_repaint();
        }
        if response.stop_clicked {
            self.stop_preview();
            ui.ctx().request_repaint();
        }
        if let Some(frame_index) = response.seek_target {
            self.seek_preview_to_frame(frame_index);
            ui.ctx().request_repaint();
        }
        if response.pipe_close_clicked {
            self.close_pipe_example();
            ui.ctx().request_repaint();
        }
        if response.optimizer_stop_clicked {
            self.stop_optimizer();
            ui.ctx().request_repaint();
        }
        if response.optimizer_close_clicked {
            self.close_optimizer_result();
            ui.ctx().request_repaint();
        }
        if let Some(answer) = response.optimizer_answer {
            self.send_optimizer_answer(answer);
            ui.ctx().request_repaint();
        }
    }
    fn show_pipe_example(&mut self) {
        self.request_pipe_owner();
    }

    #[cfg(test)]
    fn show_pipe_example_with_facts_source(
        &mut self,
        facts_source: impl FnOnce(
            &std::path::Path,
        ) -> Result<
            mcraw4vulkan::PipeExampleFacts,
            pipe_example::PipeExampleError,
        >,
    ) {
        if let Some(request) = self.prepare_pipe_request_with_facts_source(facts_source) {
            self.request_right_pane_owner(RightPaneRequest::ShowPipeExample(request));
        }
    }

    fn set_status(&mut self, message: impl Into<String>) {
        self.status_message = Some(message.into());
    }

    fn prepare_preview_request(&mut self) -> Option<PreparedPreviewRequest> {
        if self.mount_all_dngs_active() {
            self.set_status("Wait for Mount all DNGs to finish.");
            return None;
        }
        let Some(entry) = self.playlist.selected_entry() else {
            self.set_status(preview::NO_SELECTION_MESSAGE);
            return None;
        };
        Some(PreparedPreviewRequest {
            entry_id: entry.id,
            source_path: entry.source_path.clone(),
            display_name: entry.display_name.clone(),
            settings: self.settings,
        })
    }

    fn prepare_pipe_request(&mut self) -> Option<PreparedPipeRequest> {
        let Some(entry) = self.playlist.selected_entry() else {
            self.set_status(pipe_example::NO_SELECTED_FILE_MESSAGE);
            return None;
        };
        let panel = pipe_example::example_for_selected_file(
            Some(entry.source_path.as_path()),
            self.settings.decode_mode(),
            self.settings.optimizer_profile,
            self.settings.dng_vignette,
        );
        if let PipeExamplePanel::Error(message) = &panel {
            self.set_status(message.clone());
            return None;
        }
        Some(PreparedPipeRequest { panel })
    }

    #[cfg(test)]
    fn prepare_pipe_request_with_facts_source(
        &mut self,
        facts_source: impl FnOnce(
            &std::path::Path,
        ) -> Result<
            mcraw4vulkan::PipeExampleFacts,
            pipe_example::PipeExampleError,
        >,
    ) -> Option<PreparedPipeRequest> {
        let Some(entry) = self.playlist.selected_entry() else {
            self.set_status(pipe_example::NO_SELECTED_FILE_MESSAGE);
            return None;
        };
        let panel = pipe_example::example_for_selected_file_with_facts(
            Some(entry.source_path.as_path()),
            self.settings.decode_mode(),
            self.settings.optimizer_profile,
            self.settings.dng_vignette,
            facts_source,
        );
        if let PipeExamplePanel::Error(message) = &panel {
            self.set_status(message.clone());
            return None;
        }
        Some(PreparedPipeRequest { panel })
    }

    fn prepare_optimizer_request(&mut self) -> Option<PreparedOptimizerRequest> {
        if self.optimizer.is_running() {
            self.set_status("Optimizer is already running.");
            return None;
        }
        let Some(entry) = self.playlist.selected_entry() else {
            self.set_status("Select a playlist file to optimize.");
            return None;
        };
        if !entry.source_path.exists() {
            self.set_status("Selected file no longer exists.");
            return None;
        }
        Some(PreparedOptimizerRequest {
            entry_id: entry.id,
            source_path: entry.source_path.clone(),
            display_name: entry.display_name.clone(),
        })
    }

    fn request_preview_owner(&mut self) {
        let Some(request) = self.prepare_preview_request() else {
            return;
        };
        if self.right_pane_owner == RightPaneOwner::Preview
            && self.pending_right_pane_request.is_none()
        {
            let _ = self.activate_right_pane_request(
                RightPaneRequest::StartPreview(request),
                RightPaneActivation::Immediate,
            );
            return;
        }
        self.request_right_pane_owner(RightPaneRequest::StartPreview(request));
    }

    fn request_pipe_owner(&mut self) {
        let Some(request) = self.prepare_pipe_request() else {
            return;
        };
        self.request_right_pane_owner(RightPaneRequest::ShowPipeExample(request));
    }

    fn request_optimizer_owner(&mut self) {
        let Some(request) = self.prepare_optimizer_request() else {
            return;
        };
        self.request_right_pane_owner(RightPaneRequest::StartOptimizer(request));
    }

    fn request_right_pane_owner(&mut self, request: RightPaneRequest) {
        match self.right_pane_owner {
            RightPaneOwner::Preview if !self.preview.right_pane_handoff_ready() => {
                self.pending_right_pane_request = Some(request);
                if self.preview.needs_frame_work() {
                    let _ = self.preview.stop();
                    self.request_preview_fullscreen_exit();
                }
                self.set_status(preview_transition_status(
                    self.pending_right_pane_request.as_ref(),
                ));
            }
            RightPaneOwner::Optimizer if self.optimizer.choice_controls_visible() => {
                self.pending_right_pane_request = Some(request);
                match self.optimizer.send_answer(OptimizerAnswer::No) {
                    Ok(_) => self
                        .set_status("Recommendations not applied. Switching right-pane function."),
                    Err(error) => {
                        self.pending_right_pane_request = None;
                        self.set_status(error);
                    }
                }
            }
            RightPaneOwner::Optimizer if self.optimizer.is_running() => {
                self.pending_right_pane_request = Some(request);
                match self.optimizer.request_cancel() {
                    Ok(OptimizerCancelRequest::Requested)
                    | Ok(OptimizerCancelRequest::AlreadyCancelling) => {
                        self.set_status(optimizer_transition_status(
                            self.pending_right_pane_request.as_ref(),
                        ));
                    }
                    Ok(OptimizerCancelRequest::AlreadyExited) => {
                        self.set_status(
                            "Optimizer already finished. Switching right-pane function.",
                        );
                    }
                    Err(error) => self.set_status(error),
                }
            }
            _ => {
                self.pending_right_pane_request = None;
                let _ = self.activate_right_pane_request(request, RightPaneActivation::Immediate);
            }
        }
    }

    fn activate_right_pane_request(
        &mut self,
        request: RightPaneRequest,
        activation: RightPaneActivation,
    ) -> bool {
        match request {
            RightPaneRequest::StartPreview(request) => {
                self.start_prepared_preview(request, activation)
            }
            RightPaneRequest::ShowPipeExample(request) => {
                self.pipe_example = Some(request.panel);
                self.optimizer.release_panel();
                self.right_pane_owner = RightPaneOwner::PipeExample;
                self.set_status("Pipe Example ready.");
                true
            }
            RightPaneRequest::StartOptimizer(request) => {
                self.start_prepared_optimizer(request, activation)
            }
        }
    }

    fn start_prepared_preview(
        &mut self,
        request: PreparedPreviewRequest,
        activation: RightPaneActivation,
    ) -> bool {
        debug_assert!(request.entry_id > 0);
        let message = self.preview.start_with_timing(
            Some((request.source_path.as_path(), request.display_name.as_str())),
            &request.settings,
        );
        if self.preview.is_running() {
            self.pipe_example = None;
            self.optimizer.release_panel();
            self.right_pane_owner = RightPaneOwner::Preview;
            self.set_status(message);
            true
        } else {
            if activation == RightPaneActivation::AfterTeardown {
                self.right_pane_owner = RightPaneOwner::Idle;
            }
            self.set_status(message);
            false
        }
    }

    fn start_prepared_optimizer(
        &mut self,
        request: PreparedOptimizerRequest,
        activation: RightPaneActivation,
    ) -> bool {
        debug_assert!(request.entry_id > 0);
        let spec = OptimizerCommandSpec::run(self.optimizer_binary.clone(), &request.source_path);
        match self.optimizer.start(&spec) {
            Ok(()) => {
                self.pipe_example = None;
                self.right_pane_owner = RightPaneOwner::Optimizer;
                self.set_status(format!("Optimizer running: {}", request.display_name));
                true
            }
            Err(error) => {
                if activation == RightPaneActivation::AfterTeardown
                    || self.right_pane_owner != RightPaneOwner::PipeExample
                {
                    self.pipe_example = None;
                    self.optimizer
                        .set_terminal_result(OptimizerTerminalResult::Failed(
                            "Optimizer failed to start.".to_string(),
                        ));
                    self.right_pane_owner = RightPaneOwner::Optimizer;
                }
                let _ = error;
                self.set_status("Optimizer failed to start.");
                false
            }
        }
    }

    fn advance_right_pane_transition(&mut self) -> bool {
        if self.right_pane_owner == RightPaneOwner::Preview
            && self.preview.right_pane_handoff_ready()
        {
            if let Some(request) = self.pending_right_pane_request.take() {
                return self
                    .activate_right_pane_request(request, RightPaneActivation::AfterTeardown);
            }
            self.right_pane_owner = RightPaneOwner::Idle;
            return true;
        }

        if self.right_pane_owner == RightPaneOwner::Optimizer
            && !self.optimizer.is_running()
            && let Some(request) = self.pending_right_pane_request.take()
        {
            return self.activate_right_pane_request(request, RightPaneActivation::AfterTeardown);
        }

        false
    }

    fn close_pipe_example(&mut self) {
        self.clear_pipe_example();
        self.right_pane_owner = RightPaneOwner::Idle;
        self.set_status("Pipe Example closed.");
    }

    fn clear_pipe_example(&mut self) {
        self.pipe_example = None;
        if self.right_pane_owner == RightPaneOwner::PipeExample {
            self.right_pane_owner = RightPaneOwner::Idle;
        }
    }

    fn close_optimizer_result(&mut self) {
        self.optimizer.release_panel();
        self.right_pane_owner = RightPaneOwner::Idle;
        self.set_status("Optimizer result closed.");
    }

    fn stop_optimizer(&mut self) {
        if !self.optimizer.is_running() {
            self.set_status("Optimizer is not running.");
            return;
        }
        match self.optimizer.request_cancel() {
            Ok(OptimizerCancelRequest::Requested) => self.set_status("Stopping Optimizer..."),
            Ok(OptimizerCancelRequest::AlreadyCancelling) => {
                self.set_status("Optimizer is stopping.")
            }
            Ok(OptimizerCancelRequest::AlreadyExited) => {
                self.set_status("Optimizer already finished.")
            }
            Err(error) => self.set_status(error),
        }
    }

    fn begin_drop(&mut self) {
        self.drop_in_progress = true;
        self.pending_drop_files.clear();
    }

    fn queue_drop_file(&mut self, filename: String) -> bool {
        let path = PathBuf::from(filename);
        if self.drop_in_progress {
            self.pending_drop_files.push(path);
            false
        } else {
            self.add_dropped_paths([path])
        }
    }

    fn complete_drop(&mut self) -> bool {
        self.drop_in_progress = false;
        let dropped = std::mem::take(&mut self.pending_drop_files);
        self.add_dropped_paths(dropped)
    }

    fn add_dropped_paths<I, P>(&mut self, paths: I) -> bool
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        if self.mount_all_dngs_active() {
            self.set_status("Wait for Mount all DNGs to finish.");
            return false;
        }
        let summary = self.playlist.add_dropped_paths(paths);
        self.set_drop_summary_status(summary);
        if summary.accepted > 0 {
            self.save_playlist_after_mutation();
        }
        summary.accepted > 0 || summary.rejected > 0
    }

    fn set_drop_summary_status(&mut self, summary: PlaylistAddSummary) {
        match (summary.accepted, summary.rejected) {
            (0, 0) => {}
            (accepted, 0) => {
                self.set_status(format!("Added {accepted} file(s) to playlist."));
            }
            (0, rejected) => {
                self.set_status(format!("Ignored {rejected} non-mcraw file(s)."));
            }
            (accepted, rejected) => {
                self.set_status(format!(
                    "Added {accepted} file(s) to playlist. Ignored {rejected} non-mcraw file(s)."
                ));
            }
        }
    }

    fn remove_selected_playlist_entry(&mut self) {
        if self.mount_all_dngs_active() {
            self.set_status("Wait for Mount all DNGs to finish.");
            return;
        }
        let Some(entry) = self.playlist.selected_entry() else {
            self.set_status("Select a playlist file first.");
            return;
        };
        if matches!(
            entry.live_mount_state,
            LiveMountState::Mounting
                | LiveMountState::MountedThisSession
                | LiveMountState::Unmounting
        ) {
            self.set_status("Unmount DNG before removing a mounted playlist entry.");
            return;
        }

        match self.playlist.remove_selected() {
            Some(entry) => {
                self.clear_pipe_example();
                self.save_playlist_after_mutation();
                if let Some(preview_status) = self.preview.clear_if_source(&entry.source_path) {
                    self.request_preview_fullscreen_exit();
                    self.set_status(format!("Removed {}. {preview_status}", entry.display_name));
                } else {
                    self.set_status(format!("Removed {}.", entry.display_name));
                }
            }
            None => self.set_status("Select a playlist file first."),
        }
    }

    fn remove_all_playlist_entries(&mut self) {
        if self.mount_all_dngs_active() {
            self.set_status("Wait for Mount all DNGs to finish.");
            return;
        }
        if self.playlist.is_empty() {
            return;
        }
        if self.preview.is_running() {
            self.set_status("Stop Preview before removing all playlist entries.");
            return;
        }
        if self
            .playlist
            .entries()
            .iter()
            .any(playlist_entry_protected_from_remove_all)
        {
            self.set_status("Unmount DNGs before removing all playlist entries.");
            return;
        }

        let removed = self.playlist.clear_entries();
        self.clear_pipe_example();
        if self.save_playlist_after_mutation() {
            self.set_status(format!("Removed {removed} playlist file(s)."));
        }
    }

    fn dng_flags(&self) -> DngCliFlags {
        DngCliFlags {
            decode: self.settings.decode_mode(),
            vignette: self.settings.dng_vignette,
            optimizer_profile: self.settings.optimizer_profile,
        }
    }

    fn selected_dng_entry(&self) -> Option<DngMountRequest> {
        self.playlist
            .selected_entry()
            .map(DngMountRequest::from_entry)
    }

    fn stop_preview_before_dng_action(&mut self) {
        if self.preview.is_running() {
            let _ = self.preview.stop();
            self.request_preview_fullscreen_exit();
        }
    }

    fn mount_all_dngs_active(&self) -> bool {
        self.mount_all_dngs_batch.is_some()
    }

    fn playlist_mutation_controls_enabled(&self) -> bool {
        !self.mount_all_dngs_active()
    }

    fn preview_start_enabled(&self) -> bool {
        !self.mount_all_dngs_active()
    }

    fn dng_action_controls_enabled(&self) -> bool {
        !self.mount_all_dngs_active()
    }

    fn mount_all_dngs_button_enabled(&self) -> bool {
        !self.playlist.is_empty()
            && !self.mount_all_dngs_active()
            && !self.dng_processes.has_active_transient_action()
            && !self.quit_requested
    }

    fn mount_selected_dng(&mut self) {
        if self.mount_all_dngs_active() {
            self.set_status("Wait for Mount all DNGs to finish.");
            return;
        }

        let Some(request) = self.selected_dng_entry() else {
            self.set_status("Select a playlist file to mount DNG.");
            return;
        };

        let display_name = request.display_name.clone();
        match self.start_dng_mount(request, self.dng_flags()) {
            Ok(()) => self.set_status(format!("Mounting DNG: {display_name}")),
            Err(error) => self.set_status(error.selected_status()),
        }
    }

    fn unmount_selected_dng(&mut self) {
        if self.mount_all_dngs_active() {
            self.set_status("Wait for Mount all DNGs to finish.");
            return;
        }

        let Some(request) = self.selected_dng_entry() else {
            self.set_status("Select a playlist file to unmount DNG.");
            return;
        };

        self.stop_preview_before_dng_action();
        if let Some(entry) = self.playlist.entry_mut_by_id(request.entry_id) {
            entry.desired_mount_state = DesiredMountState::Unmounted;
            entry.live_mount_state = LiveMountState::Unmounting;
        }
        self.save_playlist_after_mutation();

        let spec = DngCommandSpec::unmount_file(self.dng_binary.clone(), &request.source_path);
        match self.dng_processes.start_unmount_file(
            request.entry_id,
            request.display_name.clone(),
            &spec,
        ) {
            Ok(()) => self.set_status(format!("Unmounting DNG: {}", request.display_name)),
            Err(error) => {
                if let Some(entry) = self.playlist.entry_mut_by_id(request.entry_id) {
                    entry.live_mount_state = LiveMountState::Error;
                }
                self.set_status(format!("DNG action failed: {error}"));
            }
        }
    }

    fn unmount_all_dngs(&mut self) {
        if self.mount_all_dngs_active() {
            self.set_status("Wait for Mount all DNGs to finish.");
            return;
        }
        let _ = self.start_unmount_all_dngs(UnmountAllDngsOrigin::Standalone);
    }

    fn start_dng_mount(
        &mut self,
        request: DngMountRequest,
        flags: DngCliFlags,
    ) -> Result<(), DngMountStartError> {
        if !request.source_path.exists() {
            return Err(DngMountStartError::MissingSource);
        }
        if self.playlist.entry_by_id(request.entry_id).is_none() {
            return Err(DngMountStartError::MissingEntry);
        }
        if self.dng_processes.has_mount_for_entry(request.entry_id)
            || self
                .playlist
                .entry_by_id(request.entry_id)
                .is_some_and(|entry| entry.live_mount_state == LiveMountState::MountedThisSession)
        {
            return Err(DngMountStartError::AlreadyMounted);
        }
        if self.duplicate_live_source_exists(request.entry_id, &request.source_path) {
            return Err(DngMountStartError::DuplicateLiveSource);
        }

        self.stop_preview_before_dng_action();
        if let Some(entry) = self.playlist.entry_mut_by_id(request.entry_id) {
            entry.desired_mount_state = DesiredMountState::Mounted;
            entry.live_mount_state = LiveMountState::Mounting;
        }
        self.save_playlist_after_mutation();

        let spec = DngCommandSpec::mount(self.dng_binary.clone(), flags, &request.source_path);
        match self
            .dng_processes
            .start_mount(request.entry_id, request.display_name.clone(), &spec)
        {
            Ok(()) => Ok(()),
            Err(error) => {
                if let Some(entry) = self.playlist.entry_mut_by_id(request.entry_id) {
                    entry.live_mount_state = LiveMountState::Error;
                }
                Err(DngMountStartError::ProcessStart(error))
            }
        }
    }

    fn start_unmount_all_dngs(&mut self, origin: UnmountAllDngsOrigin) -> Result<(), String> {
        self.stop_preview_before_dng_action();
        self.playlist
            .set_all_desired_mount_state(DesiredMountState::Unmounted);
        for entry in self.playlist.entries().to_vec() {
            if entry.desired_mount_state == DesiredMountState::Mounted
                || entry.live_mount_state != LiveMountState::NotLive
            {
                let _ = self
                    .playlist
                    .set_live_mount_state(entry.id, LiveMountState::Unmounting);
            }
        }
        self.save_playlist_after_mutation();

        let spec = DngCommandSpec::unmount_all(self.dng_binary.clone());
        match self.dng_processes.start_unmount_all(&spec) {
            Ok(()) => {
                match origin {
                    UnmountAllDngsOrigin::Standalone => self.set_status("Unmounting all DNGs."),
                    UnmountAllDngsOrigin::MountAllPreparation => {
                        self.set_status("Preparing to mount all DNGs...")
                    }
                }
                Ok(())
            }
            Err(error) => {
                self.playlist
                    .set_all_live_mount_state(LiveMountState::Error);
                self.set_status(format!("DNG action failed: {error}"));
                Err(error)
            }
        }
    }

    fn mount_all_dngs(&mut self) {
        if self.mount_all_dngs_active() {
            self.set_status("Mount all DNGs is already running.");
            return;
        }
        if self.playlist.is_empty() {
            self.set_status("Add playlist files before mounting all DNGs.");
            return;
        }
        if self.dng_processes.has_active_transient_action() {
            self.set_status("Wait for the current DNG action to finish before mounting all DNGs.");
            return;
        }

        let pending = self
            .playlist
            .entries()
            .iter()
            .map(DngMountRequest::from_entry)
            .collect::<VecDeque<_>>();
        let playlist_count = pending.len();
        let dng_flags = self.dng_flags();
        self.mount_all_dngs_batch = Some(MountAllDngsBatch::new(pending, dng_flags));

        if self
            .start_unmount_all_dngs(UnmountAllDngsOrigin::MountAllPreparation)
            .is_err()
        {
            self.mount_all_dngs_batch = None;
            self.set_status("Mount all DNGs stopped because Unmount all DNGs failed.");
            return;
        }

        self.record_macos_mount_limit_warning(cfg!(target_os = "macos"), playlist_count);
    }

    fn record_macos_mount_limit_warning(&mut self, is_macos: bool, playlist_count: usize) {
        if should_show_macos_mount_limit_warning(is_macos, playlist_count) {
            self.macos_mount_limit_warning = Some(MacosMountLimitWarning { playlist_count });
        }
    }

    fn dismiss_macos_mount_limit_warning(&mut self) {
        self.macos_mount_limit_warning = None;
    }

    fn draw_macos_mount_limit_warning(&mut self, context: &Context) {
        let Some(warning) = self.macos_mount_limit_warning else {
            return;
        };

        let mut open = true;
        let mut ok_clicked = false;
        egui::Window::new(MACOS_MOUNT_LIMIT_WARNING_TITLE)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .default_width(420.0)
            .resizable(false)
            .collapsible(false)
            .movable(false)
            .open(&mut open)
            .show(context, |ui| {
                draw_body_line(ui, &macos_mount_limit_warning_body(warning.playlist_count));
                ui.add_space(8.0);
                ui.vertical_centered(|ui| {
                    ok_clicked = ui.button(MACOS_MOUNT_LIMIT_WARNING_BUTTON).clicked();
                });
            });

        if ok_clicked || !open {
            self.dismiss_macos_mount_limit_warning();
        }
    }

    fn mount_all_waiting_for_unmount_all(&self) -> bool {
        self.mount_all_dngs_batch
            .as_ref()
            .is_some_and(|batch| batch.phase == MountAllDngsPhase::WaitingForUnmountAll)
    }

    #[cfg(test)]
    fn mount_all_current_entry_id(&self) -> Option<u64> {
        self.mount_all_dngs_batch
            .as_ref()
            .and_then(|batch| batch.current_entry_id)
    }

    fn complete_mount_all_unmount_success(&mut self) {
        if let Some(batch) = &mut self.mount_all_dngs_batch {
            batch.phase = MountAllDngsPhase::WaitingForMount;
        }
        self.start_next_mount_all_dng();
    }

    fn abort_mount_all_after_unmount_failure(&mut self) {
        if self.mount_all_dngs_batch.take().is_some() {
            self.set_status("Mount all DNGs stopped because Unmount all DNGs failed.");
        }
    }

    fn start_next_mount_all_dng(&mut self) {
        loop {
            let Some((request, flags, ordinal, total)) = self.take_next_mount_all_request() else {
                self.finish_mount_all_dngs_batch();
                return;
            };
            self.set_status(format!("Mounting DNG {ordinal} of {total}..."));
            match self.start_dng_mount(request.clone(), flags) {
                Ok(()) => {
                    if let Some(batch) = &mut self.mount_all_dngs_batch {
                        batch.current_entry_id = Some(request.entry_id);
                    }
                    return;
                }
                Err(error) => {
                    self.record_mount_all_failure(request, error.batch_reason());
                }
            }
        }
    }

    fn take_next_mount_all_request(
        &mut self,
    ) -> Option<(DngMountRequest, DngCliFlags, usize, usize)> {
        let batch = self.mount_all_dngs_batch.as_mut()?;
        batch.current_entry_id = None;
        let ordinal = batch.next_mount_ordinal();
        let total = batch.total;
        let flags = batch.dng_flags;
        let request = batch.pending.pop_front()?;
        Some((request, flags, ordinal, total))
    }

    fn record_mount_all_success(&mut self, entry_id: u64) -> bool {
        let Some(batch) = &mut self.mount_all_dngs_batch else {
            return false;
        };
        if batch.current_entry_id != Some(entry_id) {
            return false;
        }
        batch.succeeded += 1;
        batch.current_entry_id = None;
        true
    }

    fn record_mount_all_failure(&mut self, request: DngMountRequest, reason: String) {
        if let Some(batch) = &mut self.mount_all_dngs_batch {
            batch.failures.push(DngMountFailure {
                entry_id: request.entry_id,
                display_name: request.display_name,
                reason,
            });
            batch.current_entry_id = None;
        }
    }

    fn record_current_mount_all_failure(
        &mut self,
        entry_id: u64,
        display_name: String,
        reason: String,
    ) -> bool {
        let Some(batch) = &mut self.mount_all_dngs_batch else {
            return false;
        };
        if batch.current_entry_id != Some(entry_id) {
            return false;
        }
        batch.failures.push(DngMountFailure {
            entry_id,
            display_name,
            reason,
        });
        batch.current_entry_id = None;
        true
    }

    fn finish_mount_all_dngs_batch(&mut self) {
        let Some(batch) = self.mount_all_dngs_batch.take() else {
            return;
        };
        let succeeded = batch.succeeded;
        let failed = batch.failures.len();
        let total = batch.total;
        if succeeded == total {
            self.set_status(format!("Mounted all {total} DNGs."));
        } else if succeeded == 0 {
            self.set_status(format!("Could not mount any of {total} DNGs."));
        } else {
            self.set_status(format!(
                "Mounted {succeeded} of {total} DNGs; {failed} failed."
            ));
        }
    }

    fn duplicate_live_source_exists(&self, selected_id: u64, selected_path: &Path) -> bool {
        let selected_identity = source_identity(selected_path);
        self.playlist.entries().iter().any(|entry| {
            entry.id != selected_id
                && entry.live_mount_state == LiveMountState::MountedThisSession
                && source_identity(&entry.source_path) == selected_identity
        })
    }

    fn poll_dng_processes(&mut self) -> bool {
        let events = self.dng_processes.poll();
        let changed = !events.is_empty();
        for event in events {
            self.handle_dng_process_event(event);
        }
        changed
    }

    fn handle_dng_process_event(&mut self, event: DngProcessEvent) {
        match event {
            DngProcessEvent::MountBecameActive {
                entry_id,
                display_name,
                mount_path,
            } => {
                let _ = self
                    .playlist
                    .set_live_mount_state(entry_id, LiveMountState::MountedThisSession);
                if self.record_mount_all_success(entry_id) {
                    self.start_next_mount_all_dng();
                    return;
                }
                if let Some(mount_path) = mount_path {
                    self.set_status(format!(
                        "Mounted DNG: {display_name} at {}",
                        mount_path.display()
                    ));
                } else {
                    self.set_status(format!("Mounted DNG: {display_name}"));
                }
            }
            DngProcessEvent::MountExited {
                entry_id,
                display_name,
                active,
                stopping,
                success,
                status,
            } => {
                if stopping {
                    let _ = self
                        .playlist
                        .set_live_mount_state(entry_id, LiveMountState::NotLive);
                } else if success && active {
                    let _ = self
                        .playlist
                        .set_live_mount_state(entry_id, LiveMountState::NotLive);
                    self.set_status(format!("DNG mount ended: {display_name}"));
                } else {
                    let _ = self
                        .playlist
                        .set_live_mount_state(entry_id, LiveMountState::Error);
                    if self.record_current_mount_all_failure(
                        entry_id,
                        display_name.clone(),
                        "mount ended before it became active".to_string(),
                    ) {
                        self.start_next_mount_all_dng();
                        return;
                    }
                    self.set_status(format!(
                        "DNG action failed: mount for {display_name} ended before it became active, exited with {status}"
                    ));
                }
            }
            DngProcessEvent::ActionExited {
                kind:
                    DngActionKind::UnmountFile {
                        entry_id,
                        display_name,
                    },
                success,
                status,
            } => {
                if success {
                    let _ = self
                        .playlist
                        .set_live_mount_state(entry_id, LiveMountState::NotLive);
                    self.set_status(format!("DNG unmounted: {display_name}"));
                } else {
                    let _ = self
                        .playlist
                        .set_live_mount_state(entry_id, LiveMountState::Error);
                    self.set_status(format!(
                        "DNG unmount failed: {status}. Use Unmount all DNGs if duplicate mounts exist."
                    ));
                }
            }
            DngProcessEvent::ActionExited {
                kind: DngActionKind::UnmountAll,
                success,
                status,
            } => {
                let waiting_for_mount_all_unmount = self.mount_all_waiting_for_unmount_all();
                if success {
                    self.playlist
                        .set_all_live_mount_state(LiveMountState::NotLive);
                    if waiting_for_mount_all_unmount {
                        self.complete_mount_all_unmount_success();
                    } else {
                        self.set_status("DNG unmounted.");
                    }
                } else if waiting_for_mount_all_unmount {
                    self.abort_mount_all_after_unmount_failure();
                } else {
                    self.set_status(format!("DNG unmount all failed: {status}"));
                }
            }
        }
    }

    fn cleanup_dng_on_exit(&mut self) {
        let spec = DngCommandSpec::unmount_all(self.dng_binary.clone());
        self.dng_processes.cleanup_on_exit(&spec);
        self.playlist
            .set_all_live_mount_state(LiveMountState::NotLive);
    }

    fn poll_optimizer(&mut self) -> bool {
        let events = self.optimizer.poll();
        let changed = !events.is_empty();
        for event in events {
            match event {
                OptimizerProcessEvent::Output(_) => {}
                OptimizerProcessEvent::Exited {
                    success,
                    status,
                    output,
                    answer,
                    cancelled,
                } => {
                    if self.pending_right_pane_request.is_some() {
                        if cancelled {
                            self.set_status("Optimizer stopped to switch right-pane function.");
                        } else {
                            self.set_status("Switching right-pane function.");
                        }
                    } else {
                        let result =
                            optimizer_terminal_result(success, &status, &output, answer, cancelled);
                        let message = result.message().to_string();
                        self.optimizer.set_terminal_result(result);
                        self.right_pane_owner = RightPaneOwner::Optimizer;
                        self.set_status(message);
                    }
                }
            }
        }
        changed
    }

    fn cleanup_optimizer_on_exit(&mut self) {
        self.optimizer.cleanup_on_exit();
    }

    fn handle_optimizer_answer_key(&mut self, keycode: Option<Keycode>, repeat: bool) -> bool {
        if repeat || !self.optimizer.is_visible() || !self.optimizer.choice_controls_visible() {
            return false;
        }
        let answer = match keycode {
            Some(Keycode::Y) => OptimizerAnswer::Yes,
            Some(Keycode::N) => OptimizerAnswer::No,
            _ => return false,
        };
        self.send_optimizer_answer(answer);
        true
    }

    fn send_optimizer_answer(&mut self, answer: OptimizerAnswer) {
        match self.optimizer.send_answer(answer) {
            Ok(message) => self.set_status(message),
            Err(error) => self.set_status(error),
        }
    }

    #[rustfmt::skip]
    fn start_preview(&mut self) {
        self.request_preview_owner();
    }

    fn stop_preview(&mut self) {
        let message = self.preview.stop();
        self.request_preview_fullscreen_exit();
        self.set_status(message);
    }

    fn toggle_preview_play_pause(&mut self) {
        let message = self.preview.toggle_play_pause();
        self.set_status(message);
    }

    fn seek_preview_to_frame(&mut self, frame_index: usize) {
        let message = self.preview.seek_to_frame_index(frame_index);
        self.set_status(message);
    }

    fn handle_optimizer_profile_changed(&mut self, profile: OptimizerProfile) {
        if self.preview.is_running() {
            let _ = self.preview.stop();
            self.request_preview_fullscreen_exit();
            self.set_status(format!(
                "Preview stopped. Click Preview again to use {}.",
                profile.label()
            ));
        } else {
            self.set_status(format!("{} selected.", profile.label()));
        }
    }

    fn preview_fullscreen_active(&self) -> bool {
        self.preview_fullscreen
    }

    fn next_preview_fullscreen_state(&self) -> Option<bool> {
        if self.preview_fullscreen {
            Some(false)
        } else {
            self.preview.is_running().then_some(true)
        }
    }

    fn set_preview_fullscreen_window(
        &mut self,
        surface: &mut Sdl2WgpuSurface,
        enabled: bool,
    ) -> Result<bool, GuiError> {
        if self.preview_fullscreen == enabled {
            return Ok(false);
        }

        surface.set_fullscreen_desktop(enabled)?;
        self.preview_fullscreen = enabled;
        self.preview_fullscreen_exit_pending = false;
        self.defer_preview_advance();
        Ok(true)
    }

    fn toggle_preview_fullscreen(
        &mut self,
        surface: &mut Sdl2WgpuSurface,
    ) -> Result<bool, GuiError> {
        let Some(enabled) = self.next_preview_fullscreen_state() else {
            return Ok(false);
        };

        let changed = self.set_preview_fullscreen_window(surface, enabled)?;
        if changed {
            if enabled {
                self.clear_pipe_example();
                self.set_status("Preview fullscreen.");
            } else {
                self.set_status("Preview embedded.");
            }
        }
        Ok(changed)
    }

    fn exit_preview_fullscreen(&mut self, surface: &mut Sdl2WgpuSurface) -> Result<bool, GuiError> {
        self.set_preview_fullscreen_window(surface, false)
    }

    fn request_preview_fullscreen_exit(&mut self) {
        if self.preview_fullscreen {
            self.preview_fullscreen = false;
            self.preview_fullscreen_exit_pending = true;
            self.defer_preview_advance();
        }
    }

    fn take_preview_fullscreen_exit_pending(&mut self) -> bool {
        let pending = self.preview_fullscreen_exit_pending;
        self.preview_fullscreen_exit_pending = false;
        pending
    }

    fn take_main_window_maximize_request(&mut self) -> bool {
        let pending = self.main_window_maximize_pending;
        self.main_window_maximize_pending = false;
        pending
    }

    fn take_splash_window_size_request(&mut self) -> Option<(u32, u32)> {
        if !self.splash_window_size_pending {
            return None;
        }
        self.splash_window_size_pending = false;
        if matches!(self.phase, AppPhase::MainSkeleton) {
            return None;
        }
        Some((WINDOW_WIDTH, WINDOW_HEIGHT))
    }

    fn defer_preview_advance(&mut self) {
        if self.preview.needs_frame_work() {
            self.defer_preview_advance_once = true;
        }
    }

    fn take_preview_advance(&mut self) -> bool {
        let advance = !self.defer_preview_advance_once;
        self.defer_preview_advance_once = false;
        advance
    }

    fn add_files_from_chooser(&mut self) -> bool {
        if self.pending_file_chooser.is_some() {
            self.set_status("File chooser is already open.");
            return true;
        }
        self.apply_file_chooser_start(file_chooser::start_mcraw_file_chooser())
    }

    fn apply_file_chooser_start(&mut self, start: FileChooserStart) -> bool {
        match start {
            FileChooserStart::Pending(pending) => {
                self.pending_file_chooser = Some(pending);
                self.set_status("File chooser open...");
                true
            }
            FileChooserStart::Ready(outcome) => self.apply_file_chooser_outcome(outcome),
        }
    }

    fn poll_file_chooser(&mut self) -> bool {
        let Some(pending) = self.pending_file_chooser.as_mut() else {
            return false;
        };

        match pending.poll() {
            FileChooserPoll::Pending => false,
            FileChooserPoll::Ready(outcome) => {
                self.pending_file_chooser = None;
                self.apply_file_chooser_outcome(outcome)
            }
        }
    }

    fn apply_file_chooser_outcome(&mut self, outcome: FileChooserOutcome) -> bool {
        match outcome {
            FileChooserOutcome::Selected(paths) => self.add_dropped_paths(paths),
            FileChooserOutcome::Cancelled => {
                self.set_status("File chooser cancelled.");
                true
            }
            FileChooserOutcome::Unavailable(message) => {
                self.set_status(message);
                true
            }
            FileChooserOutcome::Failed(message) => {
                self.set_status(message);
                true
            }
        }
    }

    fn request_pipe_command_copy(&mut self, context: &Context) {
        match pipe_command_for_copy(self.pipe_example.as_ref()) {
            Some(command) => context.copy_text(command.to_string()),
            None => self.set_status("No pipe command to copy."),
        }
    }

    fn preview_target_for_surface(
        &self,
        render_size: WindowSize,
        pixels_per_point: f32,
    ) -> Option<mcraw4vulkan::display_window::EmbeddedDisplayPreviewTarget> {
        if self.right_pane_frame_owner != RightPaneOwner::Preview {
            return None;
        }
        let target = if self.preview_fullscreen {
            preview::fullscreen_preview_target(render_size)
        } else {
            self.preview_area.and_then(|rect| {
                preview::physical_preview_target(rect, pixels_per_point, render_size)
            })
        };

        target
            .and_then(|target| preview::clamp_preview_target_to_render_target(target, render_size))
    }

    fn preview_frame_work_allowed_for_frame(&self) -> bool {
        self.right_pane_frame_owner == RightPaneOwner::Preview
    }

    #[rustfmt::skip]
    fn render_preview_frame(
        &mut self,
        frame: RenderFrameContext<'_>,
        target: Option<mcraw4vulkan::display_window::EmbeddedDisplayPreviewTarget>,
        adapter_info: wgpu::AdapterInfo,
        adapter_features: wgpu::Features,
        advance: bool,
    ) -> PreviewRenderOutcome {
        let selected_source_path = self.playlist.selected_path().map(PathBuf::from);
        if let Some(message) = self
            .preview
            .clear_starting_if_selection_changed(selected_source_path.as_deref())
        {
            self.set_status(message);
            return PreviewRenderOutcome {
                dirty: true,
                video_advanced: false,
            };
        }
        let status = self.preview.render_frame(
            frame,
            target,
            adapter_info,
            adapter_features,
            advance,
        );
        let mut outcome = match status {
            PreviewFrameStatus::Idle => PreviewRenderOutcome::default(),
            PreviewFrameStatus::Continue { video_advanced } => PreviewRenderOutcome {
                dirty: false,
                video_advanced,
            },
            PreviewFrameStatus::Finished(message) | PreviewFrameStatus::Error(message) => {
                self.set_status(message);
                PreviewRenderOutcome {
                    dirty: true,
                    video_advanced: false,
                }
            }
        };
        if let Some(message) = self.preview.take_status_message() {
            self.set_status(message);
            outcome.dirty = true;
        }
        outcome
    }

    #[cfg(test)]
    fn apply_file_chooser_start_for_test(&mut self, start: FileChooserStart) -> bool {
        self.apply_file_chooser_start(start)
    }

    #[cfg(test)]
    fn copy_pipe_command_with(
        &mut self,
        copy: impl FnOnce(&str) -> Result<(), String>,
    ) -> Option<String> {
        let Some(command) = pipe_command_for_copy(self.pipe_example.as_ref()) else {
            self.set_status("No pipe command to copy.");
            return None;
        };
        let copied = command.to_string();
        match copy(command) {
            Ok(()) => self.set_status("Pipe command copied."),
            Err(error) => self.set_status(format!("Clipboard copy failed: {error}")),
        }
        Some(copied)
    }

    fn draw_primary_controls_column(
        &mut self,
        ui: &mut egui::Ui,
        row_layout: main_view::SharedControlsRowLayout,
    ) {
        draw_controls_row(ui, row_layout.top_height, |ui| {
            self.draw_playlist_section(ui, row_layout.playlist_list_height);
        });
        draw_shared_row_gap(ui);
        draw_padded_controls_row(
            ui,
            row_layout.quick_preview_height,
            main_view::CONTROL_ROW_TOP_PADDING,
            main_view::CONTROL_ROW_BOTTOM_PADDING,
            |ui| {
                self.draw_quick_preview_primary_section(ui);
            },
        );
        draw_shared_row_gap(ui);
        draw_padded_controls_row(
            ui,
            row_layout.dng_height,
            main_view::CONTROL_ROW_TOP_PADDING,
            main_view::CONTROL_ROW_BOTTOM_PADDING,
            |ui| {
                self.draw_dng_primary_section(ui);
            },
        );
        draw_shared_row_gap(ui);
        draw_controls_row(ui, row_layout.status_height, |ui| {
            self.draw_status_section(ui);
        });
        draw_shared_row_gap(ui);
        draw_padded_controls_row(
            ui,
            row_layout.footer_height,
            main_view::FOOTER_ROW_TOP_PADDING,
            main_view::FOOTER_BOTTOM_INSET,
            |ui| {
                self.draw_more_options_footer(ui);
            },
        );
    }

    fn draw_playlist_section(&mut self, ui: &mut egui::Ui, playlist_height: f32) {
        draw_section_title(ui, "Playlist");
        ui.add_space(8.0);
        let response = draw_playlist_action_row(
            ui,
            self.pending_file_chooser.is_none() && self.playlist_mutation_controls_enabled(),
            self.playlist_mutation_controls_enabled(),
            !self.playlist.is_empty() && self.playlist_mutation_controls_enabled(),
        );
        if response.add_files_clicked && self.add_files_from_chooser() {
            ui.ctx().request_repaint();
        }
        if response.remove_file_clicked {
            self.remove_selected_playlist_entry();
            ui.ctx().request_repaint();
        }
        if response.remove_all_clicked {
            self.remove_all_playlist_entries();
            ui.ctx().request_repaint();
        }
        ui.add_space(12.0);
        draw_divider(ui);
        ui.add_space(12.0);
        draw_playlist_box(ui, playlist_height, &mut self.playlist);
    }

    fn draw_quick_preview_primary_section(&mut self, ui: &mut egui::Ui) {
        draw_section_title(ui, "Quick Preview");
        draw_body_line(
            ui,
            "8 bit medium-quality preview. Toggle F key for full screen preview.",
        );
        ui.add_space(8.0);
        let response = draw_quick_preview_primary_row(
            ui,
            self.preview.is_running(),
            self.settings.quick_preview_vignette,
            self.preview_start_enabled(),
        );
        if response.preview_clicked {
            self.start_preview();
            ui.ctx().request_repaint();
        }
        if response.stop_clicked {
            self.stop_preview();
            ui.ctx().request_repaint();
        }
        if response.vignette_clicked {
            self.settings.toggle_quick_preview_vignette();
            ui.ctx().request_repaint();
        }
    }

    fn draw_dng_primary_section(&mut self, ui: &mut egui::Ui) {
        draw_section_title(ui, "DNG");
        draw_body_line(ui, "Mount raw video as a full-quality virtual DNG folder");
        draw_body_line(
            ui,
            "DNG vignette correction avoids the magenta shift of Quick Preview",
        );
        ui.add_space(8.0);
        let dng_actions_enabled = self.dng_action_controls_enabled();
        let response = draw_dng_primary_grid(
            ui,
            self.settings.dng_vignette,
            dng_actions_enabled,
            self.mount_all_dngs_button_enabled(),
        );
        if response.mount_clicked {
            self.mount_selected_dng();
            ui.ctx().request_repaint();
        }
        if response.unmount_clicked {
            self.unmount_selected_dng();
            ui.ctx().request_repaint();
        }
        if response.vignette_clicked {
            self.settings.toggle_dng_vignette();
            ui.ctx().request_repaint();
        }
        if response.mount_all_clicked {
            self.mount_all_dngs();
            ui.ctx().request_repaint();
        }
        if response.unmount_all_clicked {
            self.unmount_all_dngs();
            ui.ctx().request_repaint();
        }
    }

    fn draw_status_section(&mut self, ui: &mut egui::Ui) {
        draw_playlist_status_area(ui, self.status_message.as_deref());
    }

    fn draw_more_options_footer(&mut self, ui: &mut egui::Ui) {
        if draw_full_width_selected_button(ui, "More Options", self.more_options_visible) {
            self.toggle_more_options();
            ui.ctx().request_repaint();
        }
    }

    fn draw_more_options_column(
        &mut self,
        ui: &mut egui::Ui,
        row_layout: main_view::SharedControlsRowLayout,
    ) {
        draw_controls_row(ui, row_layout.top_height, |ui| {
            let response =
                draw_more_options_top_settings(ui, &mut self.settings, self.optimizer.is_active());
            self.handle_more_options_response(ui, response);
        });
        draw_shared_row_gap(ui);
        draw_padded_controls_row(
            ui,
            row_layout.quick_preview_height,
            main_view::CONTROL_ROW_TOP_PADDING,
            main_view::CONTROL_ROW_BOTTOM_PADDING,
            |ui| {
                let response = draw_quick_preview_options(ui, &mut self.settings);
                self.handle_more_options_response(ui, response);
            },
        );
        draw_shared_row_gap(ui);
        draw_empty_controls_row(ui, row_layout.dng_height);
        draw_shared_row_gap(ui);
        draw_empty_controls_row(ui, row_layout.status_height);
        draw_shared_row_gap(ui);
        draw_padded_controls_row(
            ui,
            row_layout.footer_height,
            main_view::FOOTER_ROW_TOP_PADDING,
            main_view::FOOTER_BOTTOM_INSET,
            |ui| {
                let response = draw_pipe_example_footer(ui);
                self.handle_more_options_response(ui, response);
            },
        );
    }

    fn handle_more_options_response(&mut self, ui: &egui::Ui, response: MoreOptionsResponse) {
        if let Some(message) = response.status {
            self.set_status(message);
        }
        if let Some(profile) = response.optimizer_profile_changed {
            self.handle_optimizer_profile_changed(profile);
        }
        if response.run_optimizer_clicked {
            self.request_optimizer_owner();
            ui.ctx().request_repaint();
        }
        if response.pipe_example_clicked {
            self.show_pipe_example();
            ui.ctx().request_repaint();
        }
    }
}

#[cfg(not(test))]
fn default_playlist_store() -> PlaylistStore {
    PlaylistStore::from_default_config()
}

#[cfg(test)]
fn default_playlist_store() -> PlaylistStore {
    PlaylistStore::in_memory()
}

fn phase_for_preflight(preflight: &PreflightViewModel, now: Instant) -> AppPhase {
    if preflight.ready {
        AppPhase::PreflightReadyVisible {
            transition_at: now + preflight_splash_min_visible(),
        }
    } else {
        AppPhase::PreflightNotReady
    }
}

fn load_splash_texture(context: &Context) -> Result<TextureHandle, GuiError> {
    let image = decode_splash_png(SPLASH_IMAGE_BYTES).map_err(GuiError::SplashImage)?;
    Ok(context.load_texture("packaging/splash.png", image, TextureOptions::LINEAR))
}

fn decode_splash_png(bytes: &[u8]) -> Result<ColorImage, String> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|error| format!("failed to read packaging/splash.png: {error}"))?;
    let output_size = reader
        .output_buffer_size()
        .ok_or_else(|| "packaging/splash.png output size exceeds decoder limits".to_string())?;
    let mut rgba = vec![0; output_size];
    let info = reader
        .next_frame(&mut rgba)
        .map_err(|error| format!("failed to decode packaging/splash.png: {error}"))?;

    if info.width as usize != SPLASH_IMAGE_WIDTH || info.height as usize != SPLASH_IMAGE_HEIGHT {
        return Err(format!(
            "packaging/splash.png is {}x{}; expected {}x{}",
            info.width, info.height, SPLASH_IMAGE_WIDTH, SPLASH_IMAGE_HEIGHT
        ));
    }
    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "packaging/splash.png is {:?} {:?}; expected 8-bit RGBA",
            info.color_type, info.bit_depth
        ));
    }

    Ok(ColorImage::from_rgba_unmultiplied(
        [SPLASH_IMAGE_WIDTH, SPLASH_IMAGE_HEIGHT],
        &rgba[..info.buffer_size()],
    ))
}

fn default_surface_config() -> Sdl2WgpuSurfaceConfig {
    Sdl2WgpuSurfaceConfig::new(WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT)
        .with_preferred_present_mode(preview::PREVIEW_NORMAL_PRESENT_MODE)
}

fn preflight_splash_min_visible() -> Duration {
    Duration::from_millis(PREFLIGHT_SPLASH_MIN_VISIBLE_MS)
}

fn consume_main_window_maximize_request(
    app: &mut GuiApp,
    maximize: impl FnOnce() -> Result<(), Sdl2WgpuSurfaceError>,
) -> bool {
    if !app.take_main_window_maximize_request() {
        return false;
    }
    maximize().is_ok()
}

fn apply_pending_splash_window_size(
    app: &mut GuiApp,
    surface: &mut Sdl2WgpuSurface,
) -> Result<bool, GuiError> {
    let Some((width, height)) = app.take_splash_window_size_request() else {
        return Ok(false);
    };
    surface.set_window_size(width, height)?;
    Ok(true)
}

fn gui_gpu_model_from_adapter_info(adapter_info: &wgpu::AdapterInfo) -> Option<String> {
    let name = adapter_info.name.trim();
    if name.is_empty() || name.eq_ignore_ascii_case("unknown") {
        return None;
    }

    Some(format!("{name} ({})", adapter_info.backend))
}

fn splash_image_display_size(available_width: f32) -> Vec2 {
    let width = available_width.max(0.0);
    let height = width * SPLASH_IMAGE_HEIGHT as f32 / SPLASH_IMAGE_WIDTH as f32;
    vec2(width, height)
}

fn pipe_command_for_copy(panel: Option<&PipeExamplePanel>) -> Option<&str> {
    match panel {
        Some(PipeExamplePanel::Example(example)) => Some(&example.command),
        _ => None,
    }
}

fn source_identity(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn playlist_entry_protected_from_remove_all(entry: &PlaylistEntry) -> bool {
    entry.desired_mount_state == DesiredMountState::Mounted
        || matches!(
            entry.live_mount_state,
            LiveMountState::Mounting
                | LiveMountState::MountedThisSession
                | LiveMountState::Unmounting
        )
}

fn draw_column_frame(
    ui: &mut egui::Ui,
    width: f32,
    height: f32,
    add_contents: impl FnOnce(&mut egui::Ui, f32),
) {
    let _ = draw_column_frame_with_insets(
        ui,
        width,
        height,
        standard_column_frame_insets(),
        |ui, content_height, _geometry| {
            add_contents(ui, content_height);
        },
    );
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ColumnFrameInsets {
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
}

impl ColumnFrameInsets {
    fn symmetric(value: f32) -> Self {
        Self {
            left: value,
            right: value,
            top: value,
            bottom: value,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ColumnFrameGeometry {
    outer_rect: Rect,
    inner_rect: Rect,
    clip_rect: Rect,
    insets: ColumnFrameInsets,
}

fn standard_column_frame_insets() -> ColumnFrameInsets {
    ColumnFrameInsets::symmetric(main_view::COLUMN_FRAME_INNER_MARGIN)
}

fn draw_column_frame_with_insets<T>(
    ui: &mut egui::Ui,
    width: f32,
    height: f32,
    insets: ColumnFrameInsets,
    add_contents: impl FnOnce(&mut egui::Ui, f32, ColumnFrameGeometry) -> T,
) -> (ColumnFrameGeometry, T) {
    let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    ui.painter().rect_filled(rect, 4, style::panel());
    ui.painter().rect_stroke(
        rect,
        4,
        Stroke::new(1.0, style::divider()),
        StrokeKind::Inside,
    );

    let content_rect = inset_rect_by(rect, insets);
    let content_height = content_rect.height();
    let geometry = ColumnFrameGeometry {
        outer_rect: rect,
        inner_rect: content_rect,
        clip_rect: content_rect,
        insets,
    };
    let mut child_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(content_rect)
            .layout(Layout::top_down(Align::Min)),
    );
    child_ui.set_clip_rect(content_rect);
    child_ui.set_width(content_rect.width());
    child_ui.set_height(content_height);
    child_ui.set_width_range(content_rect.width()..=content_rect.width());
    child_ui.set_height_range(content_height..=content_height);
    let output = add_contents(&mut child_ui, content_height, geometry);
    (geometry, output)
}

fn inset_rect(rect: Rect, margin: f32) -> Rect {
    inset_rect_by(rect, ColumnFrameInsets::symmetric(margin))
}

fn inset_rect_by(rect: Rect, insets: ColumnFrameInsets) -> Rect {
    let left = insets.left.max(0.0);
    let right = insets.right.max(0.0);
    let top = insets.top.max(0.0);
    let bottom = insets.bottom.max(0.0);
    Rect::from_min_size(
        rect.min + vec2(left, top),
        vec2(
            (rect.width() - left - right).max(0.0),
            (rect.height() - top - bottom).max(0.0),
        ),
    )
}

struct DisplayPlaybackColumnResponse {
    copy_clicked: bool,
    play_pause_clicked: bool,
    stop_clicked: bool,
    seek_target: Option<usize>,
    pipe_close_clicked: bool,
    optimizer_stop_clicked: bool,
    optimizer_close_clicked: bool,
    optimizer_answer: Option<OptimizerAnswer>,
    preview_rect: Option<PreviewLogicalRect>,
}

struct DisplayPlaybackState<'a> {
    owner: RightPaneOwner,
    pipe_example: Option<&'a PipeExamplePanel>,
    optimizer: &'a GuiOptimizer,
    playback_control: PreviewPlaybackControl,
    preview_running: bool,
    playback_position: Option<PreviewPlaybackPosition>,
}

fn draw_display_playback_column(
    ui: &mut egui::Ui,
    content_height: f32,
    state: DisplayPlaybackState<'_>,
) -> DisplayPlaybackColumnResponse {
    let display_height = (content_height - 74.0).max(260.0);
    let (copy_clicked, preview_rect) = draw_display_area(
        ui,
        display_height,
        state.owner,
        state.pipe_example,
        state.optimizer,
    );
    ui.add_space(14.0);
    let footer = draw_right_pane_footer(
        ui,
        state.owner,
        state.optimizer,
        state.playback_control,
        state.preview_running,
        state.playback_position,
    );
    DisplayPlaybackColumnResponse {
        copy_clicked,
        play_pause_clicked: footer.play_pause_clicked,
        stop_clicked: footer.stop_clicked,
        seek_target: footer.seek_target,
        pipe_close_clicked: footer.pipe_close_clicked,
        optimizer_stop_clicked: footer.optimizer_stop_clicked,
        optimizer_close_clicked: footer.optimizer_close_clicked,
        optimizer_answer: footer.optimizer_answer,
        preview_rect,
    }
}

fn draw_display_area(
    ui: &mut egui::Ui,
    height: f32,
    owner: RightPaneOwner,
    pipe_example: Option<&PipeExamplePanel>,
    optimizer: &GuiOptimizer,
) -> (bool, Option<PreviewLogicalRect>) {
    let size = vec2(ui.available_width(), height);
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    let preview_rect = inset_rect(rect, 1.0);
    let preview_rect = PreviewLogicalRect::new(
        preview_rect.min.x,
        preview_rect.min.y,
        preview_rect.width(),
        preview_rect.height(),
    );
    ui.painter().rect_filled(rect, 4, style::background());
    ui.painter().rect_stroke(
        rect,
        4,
        Stroke::new(1.0, style::divider()),
        StrokeKind::Inside,
    );
    match owner {
        RightPaneOwner::Idle => {
            draw_idle_display_area(ui, rect);
            (false, None)
        }
        RightPaneOwner::Preview => (false, preview_rect),
        RightPaneOwner::PipeExample => {
            let copy_clicked = pipe_example
                .map(|panel| draw_pipe_example_panel(ui, rect, panel))
                .unwrap_or_else(|| {
                    draw_idle_display_area(ui, rect);
                    false
                });
            (copy_clicked, None)
        }
        RightPaneOwner::Optimizer => {
            draw_optimizer_panel(ui, rect, optimizer);
            (false, None)
        }
    }
}

fn draw_idle_display_area(ui: &mut egui::Ui, rect: Rect) {
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        "No clip playing",
        FontId::proportional(style::STATUS_FONT_SIZE),
        style::muted_text(),
    );
}

fn draw_optimizer_panel(ui: &mut egui::Ui, rect: Rect, optimizer: &GuiOptimizer) {
    let content_rect = inset_rect(rect, 24.0);
    let mut child_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(content_rect)
            .layout(Layout::top_down(Align::Min)),
    );
    child_ui.set_clip_rect(content_rect);
    child_ui.set_width(content_rect.width());
    child_ui.set_height(content_rect.height());

    child_ui.label(
        RichText::new("mcraw4vulkan Optimizer")
            .size(style::STATUS_FONT_SIZE)
            .color(style::header_text()),
    );
    child_ui.add_space(8.0);
    child_ui.add(
        egui::Label::new(
            RichText::new(OPTIMIZER_PANEL_BODY)
                .size(style::LINE_FONT_SIZE)
                .color(style::body_text()),
        )
        .wrap(),
    );
    child_ui.add_space(12.0);
    draw_divider(&mut child_ui);
    child_ui.add_space(10.0);

    if let Some(result) = optimizer.terminal_result() {
        child_ui.add(
            egui::Label::new(
                RichText::new(result.message())
                    .size(style::STATUS_FONT_SIZE)
                    .color(style::header_text()),
            )
            .wrap(),
        );
        child_ui.add_space(10.0);
        draw_divider(&mut child_ui);
        child_ui.add_space(10.0);
        return;
    }

    let width = child_ui.available_width().max(0.0);
    draw_optimizer_progress_bar(&mut child_ui, width, optimizer.progress());
    child_ui.add_space(10.0);
    draw_divider(&mut child_ui);
    child_ui.add_space(10.0);

    let output_height = child_ui.available_height().max(80.0);
    child_ui.allocate_ui_with_layout(
        vec2(content_rect.width(), output_height),
        Layout::top_down(Align::Min),
        |ui| {
            egui::ScrollArea::vertical()
                .id_salt("optimizer-output-scroll")
                .auto_shrink([false, false])
                .stick_to_bottom(optimizer.is_running())
                .max_height(output_height)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.add(
                        egui::Label::new(
                            RichText::new(optimizer.output())
                                .size(style::LINE_FONT_SIZE)
                                .color(style::body_text()),
                        )
                        .selectable(true)
                        .wrap(),
                    );
                });
        },
    );
}

fn draw_optimizer_progress_bar(ui: &mut egui::Ui, width: f32, progress: f32) {
    let width = width.max(48.0);
    let (rect, _) = ui.allocate_exact_size(vec2(width, 24.0), Sense::hover());
    ui.painter().rect_filled(rect, 4, style::button_dark());
    let fill_width = rect.width() * progress.clamp(0.0, 1.0);
    if fill_width > 0.0 {
        let fill = Rect::from_min_size(rect.min, vec2(fill_width, rect.height()));
        ui.painter().rect_filled(fill, 4, style::bright_blue());
    }
    ui.painter().rect_stroke(
        rect,
        4,
        Stroke::new(1.0, style::divider()),
        StrokeKind::Inside,
    );
}

const OPTIMIZER_PANEL_BODY: &str = "The optimizer compares the Default and Offset payload profiles for this system using Display / Quick Preview and PIPE throughput.\n\nTo run the optimizer, select a typical file from the playlist that is at least 600 frames. Resolution and storage location affect the measurements.";

fn draw_pipe_example_panel(ui: &mut egui::Ui, rect: Rect, panel: &PipeExamplePanel) -> bool {
    match panel {
        PipeExamplePanel::Message(message) => {
            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                message,
                FontId::proportional(style::STATUS_FONT_SIZE),
                style::muted_text(),
            );
            false
        }
        PipeExamplePanel::Error(message) => {
            let content_rect = inset_rect(rect, 24.0);
            draw_pipe_example_text_panel(ui, content_rect, Some(message), None)
        }
        PipeExamplePanel::Example(example) => {
            let content_rect = inset_rect(rect, 24.0);
            draw_pipe_example_text_panel(ui, content_rect, None, Some(example))
        }
    }
}

fn draw_pipe_example_text_panel(
    ui: &mut egui::Ui,
    content_rect: Rect,
    error: Option<&str>,
    example: Option<&pipe_example::PipeExample>,
) -> bool {
    let mut child_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(content_rect)
            .layout(Layout::top_down(Align::Min)),
    );
    child_ui.set_clip_rect(content_rect);
    child_ui.set_width(content_rect.width());
    child_ui.set_height(content_rect.height());

    let mut copy_clicked = false;

    egui::ScrollArea::vertical()
        .id_salt(PIPE_EXAMPLE_SCROLL_ID)
        .auto_shrink([false, false])
        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
        .max_height(content_rect.height())
        .show(&mut child_ui, |ui| {
            ui.label(
                RichText::new(pipe_example::HEADER)
                    .size(style::STATUS_FONT_SIZE)
                    .color(style::header_text()),
            );
            ui.add_space(16.0);
            draw_pipe_example_paragraph(ui, pipe_example::BODY);

            if let Some(error) = error {
                ui.add_space(16.0);
                ui.add(
                    egui::Label::new(
                        RichText::new(error)
                            .size(style::LINE_FONT_SIZE)
                            .color(style::header_text()),
                    )
                    .wrap(),
                );
            }

            if let Some(example) = example {
                ui.add_space(32.0);
                draw_pipe_example_paragraph(ui, example.target.terminal_label());
                ui.add_space(8.0);
                draw_pipe_example_simple_command(ui, &example.simple_command);
                ui.add_space(32.0);
                draw_pipe_example_paragraph(ui, pipe_example::COMPLICATED_EXAMPLE_LABEL);
                ui.add_space(8.0);
                draw_pipe_example_paragraph(ui, example.target.hardware_text());
                if let Some(prerequisite_text) = example.target.ffmpeg_prerequisite_text() {
                    ui.add_space(16.0);
                    draw_pipe_example_paragraph(ui, prerequisite_text);
                }
                ui.add_space(8.0);
                draw_pipe_example_paragraph(ui, pipe_example::USEFUL_TEXT);
                ui.add_space(32.0);
                draw_pipe_example_paragraph(ui, pipe_example::COPY_COMMAND_TEXT);
                ui.add_space(8.0);
                if ui
                    .button(
                        RichText::new("Copy command")
                            .size(style::BUTTON_FONT_SIZE)
                            .color(style::header_text()),
                    )
                    .clicked()
                {
                    copy_clicked = true;
                    ui.ctx().request_repaint();
                }
                ui.add_space(8.0);
                draw_pipe_example_command_text(ui, &example.command);
            }
        });

    copy_clicked
}

fn draw_pipe_example_paragraph(ui: &mut egui::Ui, text: &str) {
    ui.add(
        egui::Label::new(
            RichText::new(text)
                .size(style::LINE_FONT_SIZE)
                .color(style::body_text()),
        )
        .wrap(),
    );
}

fn draw_pipe_example_command_text(ui: &mut egui::Ui, command: &str) {
    ui.add(
        egui::Label::new(
            RichText::new(command)
                .size(style::LINE_FONT_SIZE)
                .color(style::header_text()),
        )
        .selectable(true)
        .wrap(),
    );
}

fn pipe_example_simple_command_layout(command: &str) -> egui::text::LayoutJob {
    let mut layout = egui::text::LayoutJob::default();
    let font_id = FontId::proportional(style::LINE_FONT_SIZE);
    let remainder = if let Some(remainder) = command.strip_prefix(pipe_example::SIMPLE_COMMAND_PIPE)
    {
        layout.append(
            pipe_example::SIMPLE_COMMAND_PIPE,
            0.0,
            egui::TextFormat {
                font_id: font_id.clone(),
                color: style::bright_blue(),
                ..Default::default()
            },
        );
        remainder
    } else {
        command
    };
    layout.append(
        remainder,
        0.0,
        egui::TextFormat {
            font_id,
            color: style::header_text(),
            ..Default::default()
        },
    );
    layout
}

fn draw_pipe_example_simple_command(ui: &mut egui::Ui, command: &str) {
    ui.add(
        egui::Label::new(pipe_example_simple_command_layout(command))
            .selectable(true)
            .extend(),
    );
}

#[derive(Default)]
struct RightPaneFooterResponse {
    play_pause_clicked: bool,
    stop_clicked: bool,
    seek_target: Option<usize>,
    pipe_close_clicked: bool,
    optimizer_stop_clicked: bool,
    optimizer_close_clicked: bool,
    optimizer_answer: Option<OptimizerAnswer>,
}

fn draw_right_pane_footer(
    ui: &mut egui::Ui,
    owner: RightPaneOwner,
    optimizer: &GuiOptimizer,
    playback_control: PreviewPlaybackControl,
    preview_running: bool,
    playback_position: Option<PreviewPlaybackPosition>,
) -> RightPaneFooterResponse {
    match owner {
        RightPaneOwner::Idle => {
            draw_blank_footer_row(ui);
            RightPaneFooterResponse::default()
        }
        RightPaneOwner::Preview => {
            let transport =
                draw_transport_row(ui, playback_control, preview_running, playback_position);
            RightPaneFooterResponse {
                play_pause_clicked: transport.play_pause_clicked,
                stop_clicked: transport.stop_clicked,
                seek_target: transport.seek_target,
                ..RightPaneFooterResponse::default()
            }
        }
        RightPaneOwner::PipeExample => {
            let close_clicked = draw_footer_button_row(ui, [FooterButton::new("Close")]);
            RightPaneFooterResponse {
                pipe_close_clicked: close_clicked[0],
                ..RightPaneFooterResponse::default()
            }
        }
        RightPaneOwner::Optimizer => draw_optimizer_footer(ui, optimizer),
    }
}

fn draw_optimizer_footer(ui: &mut egui::Ui, optimizer: &GuiOptimizer) -> RightPaneFooterResponse {
    if optimizer.is_cancelling() {
        draw_footer_status_row(ui, "Stopping Optimizer...");
        return RightPaneFooterResponse::default();
    }
    if optimizer.choice_controls_visible() {
        let clicked =
            draw_footer_button_row(ui, [FooterButton::new("Yes"), FooterButton::new("No")]);
        return RightPaneFooterResponse {
            optimizer_answer: if clicked[0] {
                Some(OptimizerAnswer::Yes)
            } else if clicked[1] {
                Some(OptimizerAnswer::No)
            } else {
                None
            },
            ..RightPaneFooterResponse::default()
        };
    }
    if optimizer.finishing_after_answer() {
        draw_footer_status_row(ui, "Finishing Optimizer...");
        return RightPaneFooterResponse::default();
    }
    if optimizer.is_running() {
        let clicked = draw_footer_button_row(ui, [FooterButton::wide("Stop Optimizer")]);
        return RightPaneFooterResponse {
            optimizer_stop_clicked: clicked[0],
            ..RightPaneFooterResponse::default()
        };
    }
    if optimizer.terminal_result().is_some() {
        let clicked = draw_footer_button_row(ui, [FooterButton::new("Close")]);
        return RightPaneFooterResponse {
            optimizer_close_clicked: clicked[0],
            ..RightPaneFooterResponse::default()
        };
    }

    draw_blank_footer_row(ui);
    RightPaneFooterResponse::default()
}

#[derive(Clone, Copy)]
struct FooterButton<'a> {
    label: &'a str,
    span: usize,
}

impl<'a> FooterButton<'a> {
    fn new(label: &'a str) -> Self {
        Self { label, span: 1 }
    }

    fn wide(label: &'a str) -> Self {
        Self { label, span: 2 }
    }
}

fn draw_footer_button_row<const N: usize>(
    ui: &mut egui::Ui,
    buttons: [FooterButton<'_>; N],
) -> [bool; N] {
    let row_width = ui.available_width().max(0.0);
    let (row_rect, _) =
        ui.allocate_exact_size(vec2(row_width, TRANSPORT_ROW_HEIGHT), Sense::hover());
    let geometry = transport_row_geometry(row_width);
    let mut clicked = [false; N];
    let mut left = geometry.play_left;

    for (index, button) in buttons.into_iter().enumerate() {
        let right = if button.span >= 2 {
            geometry.stop_right
        } else {
            left + transport_button_width()
        };
        clicked[index] = transport_button_in_rect(
            ui,
            row_rect,
            transport_row_rect(row_rect, left, right, TRANSPORT_BUTTON_HEIGHT),
            button.label,
            true,
        );
        left = right + TRANSPORT_ROW_GAP;
    }

    clicked
}

fn draw_footer_status_row(ui: &mut egui::Ui, label: &str) {
    let row_width = ui.available_width().max(0.0);
    let (row_rect, _) =
        ui.allocate_exact_size(vec2(row_width, TRANSPORT_ROW_HEIGHT), Sense::hover());
    let geometry = transport_row_geometry(row_width);
    let rect = transport_row_rect(
        row_rect,
        geometry.play_left,
        geometry.frame_label_right,
        TRANSPORT_ROW_HEIGHT,
    );
    let mut child_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(Layout::left_to_right(Align::Center)),
    );
    child_ui.set_clip_rect(transport_widget_clip_rect(row_rect, rect));
    child_ui.set_width(rect.width());
    child_ui.set_height(rect.height());
    child_ui.label(
        RichText::new(label)
            .size(style::LINE_FONT_SIZE)
            .color(style::body_text()),
    );
}

fn draw_blank_footer_row(ui: &mut egui::Ui) {
    let row_width = ui.available_width().max(0.0);
    let _ = ui.allocate_exact_size(vec2(row_width, TRANSPORT_ROW_HEIGHT), Sense::hover());
}

#[derive(Default)]
struct TransportRowResponse {
    play_pause_clicked: bool,
    stop_clicked: bool,
    seek_target: Option<usize>,
}

const TRANSPORT_ROW_SIDE_MARGIN: f32 = 4.0;
const TRANSPORT_ROW_GAP: f32 = 8.0;
const TRANSPORT_FRAME_LABEL_WIDTH: f32 = 224.0;
const TRANSPORT_BASE_BUTTON_WIDTH: f32 = 112.0;
#[cfg(test)]
const TRANSPORT_BUTTON_WIDTH: f32 =
    TRANSPORT_BASE_BUTTON_WIDTH + main_view::TRANSPORT_BUTTON_EXTRA_WIDTH;
const TRANSPORT_BUTTON_HEIGHT: f32 = 32.0;
const TRANSPORT_SCRUBBER_HEIGHT: f32 = TRANSPORT_BUTTON_HEIGHT;
const TRANSPORT_ROW_VERTICAL_PADDING: f32 = 4.0;
const TRANSPORT_ROW_HEIGHT: f32 = TRANSPORT_BUTTON_HEIGHT + TRANSPORT_ROW_VERTICAL_PADDING * 2.0;

#[derive(Debug, Clone, Copy, PartialEq)]
struct TransportRowGeometry {
    play_left: f32,
    play_right: f32,
    stop_left: f32,
    stop_right: f32,
    scrubber_left: f32,
    scrubber_right: f32,
    frame_label_left: f32,
    frame_label_right: f32,
}

impl TransportRowGeometry {
    fn scrubber_width(self) -> f32 {
        (self.scrubber_right - self.scrubber_left).max(0.0)
    }
}

fn transport_row_geometry(row_width: f32) -> TransportRowGeometry {
    let row_width = row_width.max(0.0);
    let transport_button_width = transport_button_width();
    let play_left = TRANSPORT_ROW_SIDE_MARGIN;
    let play_right = play_left + transport_button_width;
    let stop_left = play_right + TRANSPORT_ROW_GAP;
    let stop_right = stop_left + transport_button_width;
    let scrubber_left = stop_right + TRANSPORT_ROW_GAP;
    let frame_label_right = (row_width - TRANSPORT_ROW_SIDE_MARGIN).max(0.0);
    let frame_label_left = (frame_label_right - TRANSPORT_FRAME_LABEL_WIDTH).max(0.0);
    let scrubber_right = (frame_label_left - TRANSPORT_ROW_GAP).max(scrubber_left);
    TransportRowGeometry {
        play_left,
        play_right,
        stop_left,
        stop_right,
        scrubber_left,
        scrubber_right,
        frame_label_left,
        frame_label_right,
    }
}

fn draw_transport_row(
    ui: &mut egui::Ui,
    playback_control: PreviewPlaybackControl,
    preview_running: bool,
    playback_position: Option<PreviewPlaybackPosition>,
) -> TransportRowResponse {
    let row_width = ui.available_width().max(0.0);
    let (row_rect, _) =
        ui.allocate_exact_size(vec2(row_width, TRANSPORT_ROW_HEIGHT), Sense::hover());
    let geometry = transport_row_geometry(row_width);
    debug_assert!(geometry.scrubber_left >= geometry.stop_right + TRANSPORT_ROW_GAP);
    debug_assert!(geometry.frame_label_left <= geometry.frame_label_right);
    debug_assert!(geometry.frame_label_right <= row_width.max(0.0));

    let play_pause_clicked = transport_button_in_rect(
        ui,
        row_rect,
        transport_row_rect(
            row_rect,
            geometry.play_left,
            geometry.play_right,
            TRANSPORT_BUTTON_HEIGHT,
        ),
        playback_control.label(),
        playback_control.enabled(),
    );
    let stop_clicked = transport_button_in_rect(
        ui,
        row_rect,
        transport_row_rect(
            row_rect,
            geometry.stop_left,
            geometry.stop_right,
            TRANSPORT_BUTTON_HEIGHT,
        ),
        "Stop",
        preview_running,
    );
    draw_transport_frame_label(
        ui,
        transport_row_rect(
            row_rect,
            geometry.frame_label_left,
            geometry.frame_label_right,
            TRANSPORT_ROW_HEIGHT,
        ),
        &frame_counter_label(playback_position),
    );

    let mut seek_target = None;
    let scrubber_width = geometry.scrubber_width();
    if scrubber_width > 0.0 {
        let scrubber_rect = transport_row_rect(
            row_rect,
            geometry.scrubber_left,
            geometry.scrubber_right,
            TRANSPORT_SCRUBBER_HEIGHT,
        );
        if let Some(position) = playback_position.filter(|position| position.frame_count > 0) {
            let mut scrubber_value = scrubber_value_for_position(position);
            let slider = egui::Slider::new(&mut scrubber_value, 1.0..=position.frame_count as f64)
                .show_value(false);
            let response = draw_transport_scrubber(ui, row_rect, scrubber_rect, slider, true);
            if response.clicked() || response.drag_stopped() {
                seek_target = scrubber_seek_index_from_value(scrubber_value, position.frame_count);
            }
        } else {
            let mut progress = 0.0;
            let slider = egui::Slider::new(&mut progress, 0.0..=1.0).show_value(false);
            draw_transport_scrubber(ui, row_rect, scrubber_rect, slider, false);
        }
    }

    TransportRowResponse {
        play_pause_clicked,
        stop_clicked,
        seek_target,
    }
}

fn transport_row_rect(row_rect: Rect, left: f32, right: f32, height: f32) -> Rect {
    let width = (right - left).max(0.0);
    let top = row_rect.center().y - height.max(0.0) * 0.5;
    Rect::from_min_size(
        pos2(row_rect.min.x + left, top),
        vec2(width, height.max(0.0)),
    )
}

fn transport_widget_clip_rect(row_rect: Rect, widget_rect: Rect) -> Rect {
    Rect::from_min_max(
        pos2(widget_rect.min.x, row_rect.min.y),
        pos2(widget_rect.max.x, row_rect.max.y),
    )
}

fn draw_transport_frame_label(ui: &mut egui::Ui, rect: Rect, frame_label: &str) {
    let mut child_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(Layout::right_to_left(Align::Center)),
    );
    child_ui.set_clip_rect(rect);
    child_ui.set_width(rect.width());
    child_ui.set_height(rect.height());
    child_ui.label(
        RichText::new(frame_label)
            .size(style::LINE_FONT_SIZE)
            .color(style::body_text()),
    );
}

fn draw_transport_scrubber(
    ui: &mut egui::Ui,
    row_rect: Rect,
    rect: Rect,
    slider: egui::Slider<'_>,
    enabled: bool,
) -> egui::Response {
    let scrubber_width = rect.width().max(1.0);
    let mut child_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(Layout::left_to_right(Align::Center)),
    );
    child_ui.set_clip_rect(transport_widget_clip_rect(row_rect, rect));
    child_ui.set_width(rect.width());
    child_ui.set_height(rect.height());
    child_ui.spacing_mut().slider_width = scrubber_width;
    child_ui.add_enabled(enabled, slider)
}

fn transport_button_in_rect(
    ui: &mut egui::Ui,
    row_rect: Rect,
    rect: Rect,
    label: &str,
    enabled: bool,
) -> bool {
    let mut child_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(Layout::left_to_right(Align::Center)),
    );
    child_ui.set_clip_rect(transport_widget_clip_rect(row_rect, rect));
    child_ui.set_width(rect.width());
    child_ui.set_height(rect.height());
    transport_button(&mut child_ui, label, enabled)
}

fn frame_counter_label(position: Option<PreviewPlaybackPosition>) -> String {
    let Some(position) = position else {
        return "Frame 0".to_string();
    };
    if position.frame_count == 0 {
        return "Frame 0".to_string();
    }
    let current = position
        .current_frame_index
        .map(|index| index.saturating_add(1).min(position.frame_count))
        .unwrap_or(0);
    format!("Frame {current} / {}", position.frame_count)
}

fn scrubber_value_for_position(position: PreviewPlaybackPosition) -> f64 {
    position
        .current_frame_index
        .map(|index| index.saturating_add(1).min(position.frame_count.max(1)) as f64)
        .unwrap_or(1.0)
}

fn scrubber_seek_index_from_value(value: f64, frame_count: usize) -> Option<usize> {
    if frame_count == 0 {
        return None;
    }
    let frame_number = value.round().clamp(1.0, frame_count as f64) as usize;
    Some(frame_number - 1)
}

#[derive(Default)]
struct MoreOptionsResponse {
    status: Option<&'static str>,
    run_optimizer_clicked: bool,
    optimizer_profile_changed: Option<OptimizerProfile>,
    pipe_example_clicked: bool,
}

fn draw_more_options_top_settings(
    ui: &mut egui::Ui,
    settings: &mut GuiSettings,
    optimizer_active: bool,
) -> MoreOptionsResponse {
    let mut response = MoreOptionsResponse::default();

    draw_section_title(ui, "Decoding Settings");
    ui.add_space(8.0);
    let decode_mode = settings.decode_mode();
    draw_option_pair(
        ui,
        VisualOption::new(
            "GPU Decoding",
            decode_mode == DecodeMode::Gpu,
            DecodeMode::Gpu,
        ),
        VisualOption::new(
            "CPU Decoding",
            decode_mode == DecodeMode::Cpu,
            DecodeMode::Cpu,
        ),
        |selected| settings.select_decode_mode(selected),
    );

    ui.add_space(14.0);
    draw_section_title(ui, "Optimized Decode Settings");
    draw_body_line(
        ui,
        "Compares Default and Offset payload profiles for this system",
    );
    ui.add_space(8.0);
    if draw_single_option_row(
        ui,
        "Run optimizer now",
        run_optimizer_button_selected(optimizer_active),
    ) {
        response.run_optimizer_clicked = true;
    }
    ui.add_space(6.0);
    let optimizer_profile = settings.optimizer_profile;
    draw_option_pair(
        ui,
        VisualOption::new(
            "Default Settings",
            optimizer_profile == OptimizerProfile::Default,
            OptimizerProfile::Default,
        ),
        VisualOption::new(
            "Optimized Settings",
            optimizer_profile == OptimizerProfile::Optimized,
            OptimizerProfile::Optimized,
        ),
        |selected| {
            if settings.optimizer_profile != selected {
                settings.select_optimizer_profile(selected);
                response.optimizer_profile_changed = Some(selected);
            }
        },
    );

    response
}

fn draw_quick_preview_options(
    ui: &mut egui::Ui,
    settings: &mut GuiSettings,
) -> MoreOptionsResponse {
    let response = MoreOptionsResponse::default();

    draw_section_title(ui, "Quick Preview Options");
    ui.add_space(8.0);
    let quick_timing = settings.quick_preview_timing;
    draw_option_pair(
        ui,
        VisualOption::new(
            "Vsync On",
            quick_timing == PreviewTiming::VsyncOn,
            PreviewTiming::VsyncOn,
        ),
        VisualOption::new(
            "Max speed",
            quick_timing == PreviewTiming::MaxSpeed,
            PreviewTiming::MaxSpeed,
        ),
        |selected| settings.select_quick_preview_timing(selected),
    );
    ui.add_space(6.0);
    if draw_single_option_row(ui, "FPS overlay", settings.quick_preview_fps_overlay) {
        settings.toggle_quick_preview_fps_overlay();
    }

    response
}

fn draw_pipe_example_footer(ui: &mut egui::Ui) -> MoreOptionsResponse {
    let mut response = MoreOptionsResponse::default();
    if draw_single_option_row(ui, "Pipe Example", false) {
        response.pipe_example_clicked = true;
    }
    response
}

fn draw_playlist_box(ui: &mut egui::Ui, height: f32, playlist: &mut Playlist) {
    let (rect, _) =
        ui.allocate_exact_size(vec2(ui.available_width(), height.max(0.0)), Sense::hover());
    ui.painter().rect_filled(rect, 4, style::background());
    ui.painter().rect_stroke(
        rect,
        4,
        Stroke::new(1.0, style::divider()),
        StrokeKind::Inside,
    );

    let content_rect = inset_rect(rect, 16.0);
    let mut child_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(content_rect)
            .layout(Layout::top_down(Align::Min)),
    );
    child_ui.set_clip_rect(content_rect);
    child_ui.set_width(content_rect.width());
    child_ui.set_height(content_rect.height());
    child_ui.set_width_range(content_rect.width()..=content_rect.width());
    child_ui.set_height_range(content_rect.height()..=content_rect.height());

    egui::ScrollArea::vertical()
        .id_salt(PLAYLIST_SCROLL_ID)
        .auto_shrink([false, false])
        .max_height(content_rect.height())
        .show(&mut child_ui, |ui| {
            if playlist.is_empty() {
                ui.label(
                    RichText::new("No files in playlist.")
                        .size(style::LINE_FONT_SIZE)
                        .color(style::muted_text()),
                );
            } else {
                let mut clicked_index = None;
                for (index, entry) in playlist.entries().iter().enumerate() {
                    let selected = playlist.selected_index() == Some(index);
                    let response = draw_playlist_entry_row(
                        ui,
                        &entry.display_name,
                        selected,
                        entry.visual_state(),
                    );
                    if response.clicked() {
                        clicked_index = Some(index);
                    }
                    ui.add_space(4.0);
                }

                if let Some(index) = clicked_index {
                    if playlist.select(index) {
                        ui.ctx().request_repaint();
                    }
                }
            }
        });
}

fn draw_section_title(ui: &mut egui::Ui, text: &str) {
    ui.label(
        RichText::new(text)
            .size(style::STATUS_FONT_SIZE)
            .color(style::header_text()),
    );
}

fn draw_body_line(ui: &mut egui::Ui, text: &str) {
    ui.add(
        egui::Label::new(
            RichText::new(text)
                .size(style::LINE_FONT_SIZE)
                .color(style::body_text()),
        )
        .wrap(),
    );
}

struct VisualOption<'a, T> {
    label: &'a str,
    selected: bool,
    value: T,
}

impl<'a, T> VisualOption<'a, T> {
    fn new(label: &'a str, selected: bool, value: T) -> Self {
        Self {
            label,
            selected,
            value,
        }
    }
}

fn draw_option_pair<T: Copy>(
    ui: &mut egui::Ui,
    left: VisualOption<'_, T>,
    right: VisualOption<'_, T>,
    mut select: impl FnMut(T),
) {
    ui.horizontal(|ui| {
        let [left_width, right_width] = equal_button_widths(ui);
        if draw_clickable_visual_button(ui, left.label, left.selected, left_width).clicked() {
            select(left.value);
            ui.ctx().request_repaint();
        }
        if draw_clickable_visual_button(ui, right.label, right.selected, right_width).clicked() {
            select(right.value);
            ui.ctx().request_repaint();
        }
    });
}

fn draw_single_option_row(ui: &mut egui::Ui, label: &str, selected: bool) -> bool {
    ui.horizontal(|ui| {
        let [button_width, empty_width] = equal_button_widths(ui);
        let response = draw_clickable_visual_button(ui, label, selected, button_width);
        ui.add_space(empty_width);
        if response.clicked() {
            ui.ctx().request_repaint();
        }
        response.clicked()
    })
    .inner
}

fn run_optimizer_button_selected(optimizer_active: bool) -> bool {
    optimizer_active
}

fn preview_transition_status(request: Option<&RightPaneRequest>) -> &'static str {
    match request {
        Some(RightPaneRequest::ShowPipeExample(_)) => "Stopping Preview to show Pipe Example...",
        Some(RightPaneRequest::StartOptimizer(_)) => "Stopping Preview to run Optimizer...",
        Some(RightPaneRequest::StartPreview(_)) => "Stopping Preview to start Preview...",
        None => "Stopping Preview...",
    }
}

fn optimizer_transition_status(request: Option<&RightPaneRequest>) -> &'static str {
    match request {
        Some(RightPaneRequest::StartPreview(_)) => "Stopping Optimizer to start Preview...",
        Some(RightPaneRequest::ShowPipeExample(_)) => "Stopping Optimizer to show Pipe Example...",
        Some(RightPaneRequest::StartOptimizer(_)) => "Stopping Optimizer...",
        None => "Stopping Optimizer...",
    }
}

fn optimizer_terminal_result(
    success: bool,
    status: &str,
    output: &str,
    answer: Option<OptimizerAnswer>,
    cancelled: bool,
) -> OptimizerTerminalResult {
    if cancelled {
        return OptimizerTerminalResult::Cancelled;
    }
    if !success {
        return OptimizerTerminalResult::Failed(optimizer_failure_message(status, output));
    }
    optimizer_success_terminal_result(output, answer)
}

fn optimizer_failure_message(status: &str, output: &str) -> String {
    if output.contains(OPTIMIZER_SHORT_INPUT_MESSAGE) {
        OPTIMIZER_SHORT_INPUT_MESSAGE.to_string()
    } else {
        format!("Optimizer failed: {status}")
    }
}

#[derive(Default)]
struct PlaylistActionRowResponse {
    add_files_clicked: bool,
    remove_file_clicked: bool,
    remove_all_clicked: bool,
}

fn draw_playlist_action_row(
    ui: &mut egui::Ui,
    add_files_enabled: bool,
    remove_file_enabled: bool,
    remove_all_enabled: bool,
) -> PlaylistActionRowResponse {
    let responses = draw_three_button_row(
        ui,
        [
            ButtonCell::new("Add files", false, add_files_enabled),
            ButtonCell::new("Remove file", false, remove_file_enabled),
            ButtonCell::new("Remove all", false, remove_all_enabled),
        ],
    );

    PlaylistActionRowResponse {
        add_files_clicked: responses[0],
        remove_file_clicked: responses[1],
        remove_all_clicked: responses[2],
    }
}

#[derive(Default)]
struct QuickPreviewPrimaryRowResponse {
    preview_clicked: bool,
    stop_clicked: bool,
    vignette_clicked: bool,
}

fn draw_quick_preview_primary_row(
    ui: &mut egui::Ui,
    preview_running: bool,
    vignette_selected: bool,
    preview_enabled: bool,
) -> QuickPreviewPrimaryRowResponse {
    let responses = draw_three_button_row(
        ui,
        [
            ButtonCell::new("Preview", false, preview_enabled).orange_emphasis(),
            ButtonCell::new("Stop", false, preview_running),
            ButtonCell::new("Vignette Correction", vignette_selected, true),
        ],
    );

    QuickPreviewPrimaryRowResponse {
        preview_clicked: responses[0],
        stop_clicked: responses[1],
        vignette_clicked: responses[2],
    }
}

#[derive(Default)]
struct DngPrimaryGridResponse {
    mount_clicked: bool,
    unmount_clicked: bool,
    vignette_clicked: bool,
    mount_all_clicked: bool,
    unmount_all_clicked: bool,
}

fn draw_dng_primary_grid(
    ui: &mut egui::Ui,
    vignette_selected: bool,
    dng_actions_enabled: bool,
    mount_all_enabled: bool,
) -> DngPrimaryGridResponse {
    let top = draw_three_button_row(
        ui,
        [
            ButtonCell::new("Mount DNG", false, dng_actions_enabled).orange_emphasis(),
            ButtonCell::new("Unmount DNG", false, dng_actions_enabled),
            ButtonCell::new("Vignette Correction", vignette_selected, true),
        ],
    );
    ui.add_space(6.0);
    let bottom = draw_dng_batch_action_row(ui, mount_all_enabled, dng_actions_enabled);

    DngPrimaryGridResponse {
        mount_clicked: top[0],
        unmount_clicked: top[1],
        vignette_clicked: top[2],
        mount_all_clicked: bottom[0],
        unmount_all_clicked: bottom[1],
    }
}

fn draw_dng_batch_action_row(
    ui: &mut egui::Ui,
    mount_all_enabled: bool,
    unmount_all_enabled: bool,
) -> [bool; 2] {
    ui.horizontal(|ui| {
        let widths = equal_three_button_widths(ui);
        let mount_all_clicked = draw_visual_button_with_sense(
            ui,
            "Mount all DNGs",
            false,
            widths[0],
            if mount_all_enabled {
                Sense::click()
            } else {
                Sense::hover()
            },
        )
        .clicked();
        let unmount_all_clicked = draw_visual_button_with_sense(
            ui,
            "Unmount all DNGs",
            false,
            widths[1],
            if unmount_all_enabled {
                Sense::click()
            } else {
                Sense::hover()
            },
        )
        .clicked();
        draw_blank_button_cell(ui, widths[2]);
        [mount_all_clicked, unmount_all_clicked]
    })
    .inner
}

fn draw_blank_button_cell(ui: &mut egui::Ui, width: f32) {
    let _ = ui.allocate_exact_size(
        vec2(width.max(0.0), main_view::ORDINARY_BUTTON_HEIGHT),
        Sense::empty(),
    );
}

#[cfg(test)]
fn dng_grid_column_positions(available_width: f32, item_spacing: f32) -> [f32; 3] {
    let widths = main_view::dng_grid_columns(available_width, item_spacing);
    [
        0.0,
        widths[0] + item_spacing.max(0.0),
        widths[0] + item_spacing.max(0.0) + widths[1] + item_spacing.max(0.0),
    ]
}

#[cfg(test)]
fn dng_grid_top_row_labels() -> [&'static str; 3] {
    ["Mount DNG", "Unmount DNG", "Vignette Correction"]
}

#[cfg(test)]
fn dng_grid_second_row_labels() -> [Option<&'static str>; 3] {
    [Some("Mount all DNGs"), Some("Unmount all DNGs"), None]
}

#[cfg(test)]
fn dng_blank_cell_is_interactive() -> bool {
    false
}

#[cfg(test)]
fn mount_all_dngs_control_exists() -> bool {
    true
}

fn header_quit_button_width() -> f32 {
    ordinary_button_width(HEADER_QUIT_BASE_BUTTON_WIDTH)
}

fn ordinary_button_width(base_width: f32) -> f32 {
    base_width.max(0.0) + main_view::BUTTON_THREE_CHARACTER_EXTRA_WIDTH
}

fn transport_button_width() -> f32 {
    ordinary_button_width(TRANSPORT_BASE_BUTTON_WIDTH)
}

fn ordinary_button_height() -> f32 {
    main_view::ORDINARY_BUTTON_HEIGHT
}

#[derive(Clone, Copy)]
struct ButtonCell<'a> {
    label: &'a str,
    selected: bool,
    enabled: bool,
    stroke_style: VisualButtonStrokeStyle,
}

impl<'a> ButtonCell<'a> {
    fn new(label: &'a str, selected: bool, enabled: bool) -> Self {
        Self {
            label,
            selected,
            enabled,
            stroke_style: VisualButtonStrokeStyle::Normal,
        }
    }

    fn orange_emphasis(mut self) -> Self {
        self.stroke_style = VisualButtonStrokeStyle::OrangeEmphasis;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VisualButtonStrokeStyle {
    Normal,
    OrangeEmphasis,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct VisualButtonPaint {
    fill: egui::Color32,
    text_color: egui::Color32,
    stroke: Stroke,
}

fn draw_three_button_row(ui: &mut egui::Ui, cells: [ButtonCell<'_>; 3]) -> [bool; 3] {
    ui.horizontal(|ui| {
        let widths = equal_three_button_widths(ui);
        let mut clicked = [false; 3];
        for (index, cell) in cells.into_iter().enumerate() {
            let sense = if cell.enabled {
                Sense::click()
            } else {
                Sense::hover()
            };
            clicked[index] = draw_button_cell_with_sense(ui, cell, widths[index], sense).clicked();
        }
        clicked
    })
    .inner
}

fn draw_full_width_selected_button(ui: &mut egui::Ui, label: &str, selected: bool) -> bool {
    draw_clickable_visual_button(ui, label, selected, ui.available_width()).clicked()
}

fn equal_three_button_widths(ui: &egui::Ui) -> [f32; 3] {
    main_view::equal_three_button_widths(ui.available_width(), ui.spacing().item_spacing.x)
}

fn equal_button_widths(ui: &egui::Ui) -> [f32; 2] {
    main_view::equal_button_widths(ui.available_width(), ui.spacing().item_spacing.x)
}

fn draw_clickable_visual_button(
    ui: &mut egui::Ui,
    label: &str,
    selected: bool,
    width: f32,
) -> egui::Response {
    draw_visual_button_with_sense(ui, label, selected, width, Sense::click())
}

fn draw_button_cell_with_sense(
    ui: &mut egui::Ui,
    cell: ButtonCell<'_>,
    width: f32,
    sense: Sense,
) -> egui::Response {
    match cell.stroke_style {
        VisualButtonStrokeStyle::Normal => {
            draw_visual_button_with_sense(ui, cell.label, cell.selected, width, sense)
        }
        VisualButtonStrokeStyle::OrangeEmphasis => {
            draw_orange_emphasis_button_with_sense(ui, cell.label, cell.selected, width, sense)
        }
    }
}

fn draw_visual_button_with_sense(
    ui: &mut egui::Ui,
    label: &str,
    selected: bool,
    width: f32,
    sense: Sense,
) -> egui::Response {
    draw_visual_button_with_stroke_style(
        ui,
        label,
        selected,
        width,
        sense,
        VisualButtonStrokeStyle::Normal,
    )
}

fn draw_orange_emphasis_button_with_sense(
    ui: &mut egui::Ui,
    label: &str,
    selected: bool,
    width: f32,
    sense: Sense,
) -> egui::Response {
    style::with_orange_emphasis_button_style(ui, |ui| {
        draw_visual_button_with_stroke_style(
            ui,
            label,
            selected,
            width,
            sense,
            VisualButtonStrokeStyle::OrangeEmphasis,
        )
    })
}

fn draw_visual_button_with_stroke_style(
    ui: &mut egui::Ui,
    label: &str,
    selected: bool,
    width: f32,
    sense: Sense,
    stroke_style: VisualButtonStrokeStyle,
) -> egui::Response {
    let height = ordinary_button_height();
    let (rect, response) = ui.allocate_exact_size(vec2(width, height), sense);
    let paint = visual_button_paint(selected, stroke_style);

    ui.painter().rect_filled(rect, 3, paint.fill);
    ui.painter()
        .rect_stroke(rect, 3, paint.stroke, StrokeKind::Inside);
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(style::BUTTON_FONT_SIZE),
        paint.text_color,
    );

    response
}

fn visual_button_paint(selected: bool, stroke_style: VisualButtonStrokeStyle) -> VisualButtonPaint {
    let fill = if selected {
        style::bright_blue()
    } else {
        style::button_dark()
    };
    let text_color = if selected {
        style::header_text()
    } else {
        style::body_text()
    };
    let normal_stroke_color = if selected {
        style::bright_blue()
    } else {
        style::divider()
    };
    let stroke_color = match stroke_style {
        VisualButtonStrokeStyle::Normal => normal_stroke_color,
        VisualButtonStrokeStyle::OrangeEmphasis => style::orange_emphasis(),
    };

    VisualButtonPaint {
        fill,
        text_color,
        stroke: Stroke::new(1.0, stroke_color),
    }
}

fn playlist_status_reserved_line_count() -> usize {
    PLAYLIST_STATUS_RESERVED_LINES
}

fn playlist_status_reserved_height(ui: &egui::Ui) -> f32 {
    playlist_status_reserved_height_for_line_height(ui.text_style_height(&egui::TextStyle::Small))
}

fn playlist_status_reserved_height_for_line_height(line_height: f32) -> f32 {
    line_height.max(0.0) * playlist_status_reserved_line_count() as f32
}

fn playlist_status_color() -> egui::Color32 {
    style::bright_blue()
}

fn draw_controls_row(ui: &mut egui::Ui, height: f32, add_contents: impl FnOnce(&mut egui::Ui)) {
    let height = height.max(0.0);
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
    draw_controls_row_contents(ui, rect, add_contents);
}

fn draw_padded_controls_row(
    ui: &mut egui::Ui,
    height: f32,
    top_padding: f32,
    bottom_padding: f32,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    let height = height.max(0.0);
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
    let content_rect = controls_row_content_rect(rect, top_padding, bottom_padding);
    draw_controls_row_contents(ui, content_rect, add_contents);
}

fn controls_row_content_rect(rect: Rect, top_padding: f32, bottom_padding: f32) -> Rect {
    let top = top_padding.max(0.0);
    let bottom = bottom_padding.max(0.0);
    let min_y = rect.min.y + top;
    let max_y = (rect.max.y - bottom).max(min_y);
    Rect::from_min_max(pos2(rect.min.x, min_y), pos2(rect.max.x, max_y))
}

fn draw_controls_row_contents(
    ui: &mut egui::Ui,
    rect: Rect,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    let mut child_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(Layout::top_down(Align::Min)),
    );
    child_ui.set_clip_rect(rect);
    child_ui.set_width(rect.width());
    child_ui.set_height(rect.height());
    child_ui.set_width_range(rect.width()..=rect.width());
    child_ui.set_height_range(rect.height()..=rect.height());
    add_contents(&mut child_ui);
}

fn draw_empty_controls_row(ui: &mut egui::Ui, height: f32) {
    let _ = ui.allocate_exact_size(vec2(ui.available_width(), height.max(0.0)), Sense::hover());
}

fn draw_shared_row_gap(ui: &mut egui::Ui) {
    let implicit_spacing = ui.spacing().item_spacing.y.max(0.0);
    let gap_height = (main_view::SHARED_CONTROL_ROW_GAP - implicit_spacing * 2.0).max(0.0);
    let divider_top_padding = (12.0 - implicit_spacing).max(0.0);
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), gap_height), Sense::hover());
    ui.painter().hline(
        rect.left()..=rect.right(),
        rect.top() + divider_top_padding,
        Stroke::new(1.0, style::divider()),
    );
}

fn draw_playlist_status_area(ui: &mut egui::Ui, text: Option<&str>) {
    let height = playlist_status_reserved_height(ui);
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
    let Some(text) = text.filter(|text| !text.is_empty()) else {
        return;
    };
    let mut child_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(Layout::top_down(Align::Min)),
    );
    child_ui.set_clip_rect(rect);
    child_ui.set_width(rect.width());
    child_ui.set_height(rect.height());
    child_ui.add(
        egui::Label::new(
            RichText::new(text)
                .size(style::SMALL_FONT_SIZE)
                .color(playlist_status_color()),
        )
        .wrap(),
    );
}

fn draw_playlist_entry_row(
    ui: &mut egui::Ui,
    display_name: &str,
    selected: bool,
    visual_state: main_view::PlaylistEntryVisualState,
) -> egui::Response {
    let height = 32.0;
    let (rect, response) =
        ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::click());
    let fill = if selected {
        style::bright_blue()
    } else if response.hovered() {
        style::button_dark()
    } else {
        style::background()
    };
    let text_color = if selected {
        style::header_text()
    } else {
        playlist_text_color(main_view::playlist_entry_text_tone(visual_state))
    };

    ui.painter().rect_filled(rect, 3, fill);
    if selected || response.hovered() {
        ui.painter().rect_stroke(
            rect,
            3,
            Stroke::new(1.0, style::divider()),
            StrokeKind::Inside,
        );
    }
    ui.painter().text(
        rect.left_center() + vec2(8.0, 0.0),
        Align2::LEFT_CENTER,
        display_name,
        FontId::proportional(style::LINE_FONT_SIZE),
        text_color,
    );

    response
}

fn playlist_text_color(tone: main_view::PlaylistEntryTextTone) -> egui::Color32 {
    match tone {
        main_view::PlaylistEntryTextTone::Gray => style::muted_text(),
        main_view::PlaylistEntryTextTone::White => style::header_text(),
        main_view::PlaylistEntryTextTone::BrightBlue => style::bright_blue(),
    }
}

fn transport_button(ui: &mut egui::Ui, label: &str, enabled: bool) -> bool {
    ui.add_enabled(
        enabled,
        egui::Button::new(
            RichText::new(label)
                .size(style::BUTTON_FONT_SIZE)
                .color(style::body_text()),
        )
        .min_size(vec2(transport_button_width(), TRANSPORT_BUTTON_HEIGHT)),
    )
    .clicked()
}

struct EguiInputState {
    start: Instant,
    events: Vec<EguiEvent>,
    modifiers: Modifiers,
    focused: bool,
    minimized: bool,
}

impl EguiInputState {
    fn new(start: Instant) -> Self {
        Self {
            start,
            events: Vec::new(),
            modifiers: Modifiers::NONE,
            focused: true,
            minimized: false,
        }
    }

    fn push_event(&mut self, event: EguiEvent) {
        self.events.push(event);
    }

    fn set_modifiers(&mut self, keymod: Mod) {
        self.modifiers = modifiers_from_sdl(keymod);
    }

    fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        self.events.push(EguiEvent::WindowFocused(focused));
    }

    fn set_minimized(&mut self, minimized: bool) {
        self.minimized = minimized;
    }

    fn raw_input(&mut self, surface: &Sdl2WgpuSurface, now: Instant) -> RawInput {
        let size = surface.size();
        let points_per_pixel = pixels_per_point(size);
        let screen_rect = egui::Rect::from_min_size(
            Pos2::ZERO,
            Vec2::new(
                size.drawable_width as f32 / points_per_pixel,
                size.drawable_height as f32 / points_per_pixel,
            ),
        );

        let viewport = ViewportInfo {
            title: Some(WINDOW_TITLE.to_string()),
            native_pixels_per_point: Some(points_per_pixel),
            inner_rect: Some(screen_rect),
            outer_rect: Some(screen_rect),
            minimized: Some(self.minimized),
            focused: Some(self.focused),
            ..ViewportInfo::default()
        };

        let mut input = RawInput {
            screen_rect: Some(screen_rect),
            time: Some(now.duration_since(self.start).as_secs_f64()),
            modifiers: self.modifiers,
            events: std::mem::take(&mut self.events),
            focused: self.focused,
            system_theme: Some(egui::Theme::Dark),
            ..RawInput::default()
        };
        input.viewports.insert(ViewportId::ROOT, viewport);
        input
    }
}

#[rustfmt::skip]
fn render_splash_frame(
    surface: &mut Sdl2WgpuSurface,
    renderer: &mut Renderer,
    context: &Context,
    input: &mut EguiInputState,
    app: &mut GuiApp,
    scheduler: &mut LazyRepaintState,
    start: Instant,
) -> Result<(), GuiError> {
    let now = Instant::now();
    app.advance_startup_phase(now);
    if consume_main_window_maximize_request(app, || surface.maximize_window()) {
        scheduler.mark_resize();
    }
    if app.poll_file_chooser() {
        scheduler.mark_dirty();
    }
    if app.poll_dng_processes() {
        scheduler.mark_dirty();
    }
    if app.poll_optimizer() {
        scheduler.mark_dirty();
    }
    if app.take_preview_fullscreen_exit_pending() {
        surface.set_fullscreen_desktop(false)?;
        scheduler.mark_resize();
    }
    let raw_input = input.raw_input(surface, now);
    let full_output = context.run(raw_input, |context| app.ui(context, now));
    let egui::FullOutput {
        platform_output,
        textures_delta,
        shapes,
        pixels_per_point,
        viewport_output,
    } = full_output;
    let viewport_repaint_after = viewport_output
        .get(&ViewportId::ROOT)
        .map_or(Duration::MAX, |output| output.repaint_delay);
    handle_platform_output(surface, app, scheduler, platform_output);
    let paint_jobs = context.tessellate(shapes, pixels_per_point);
    let adapter_info = surface.adapter_info().clone();
    let adapter_features = surface.adapter().features();
    let scheduler_decision = scheduler.frame_decision(now);
    let event_allows_preview_advance = app.take_preview_advance();
    let advance_preview =
        preview_advance_allowed_for_frame(event_allows_preview_advance, scheduler_decision);
    let mut preview_outcome = PreviewRenderOutcome::default();
    let mut preview_frame_work_ran = false;
    let present_mode = app.preview.surface_present_mode();
    let mut post_stop_refresh_follow_up = false;

    reconfigure_surface_present_mode(surface, present_mode)?;
    if let Some(request) = app.preview.take_terminal_surface_refresh_request() {
        // Request one surface reconfigure before rendering after Preview terminates.
        let _reconfigure_status = surface.reconfigure()?;
        post_stop_refresh_follow_up = request.follow_up_after_suboptimal();
    }
    if let Some(_request_id) = app
        .preview
        .take_preview_start_surface_refresh_request()
    {
        // A submitted-but-suboptimal start frame is valid and must be presented before this
        // one-shot refresh. Reconfigure before the following acquire so later frames use the
        // refreshed surface without introducing a retry loop or per-frame configuration.
        let _reconfigure_status = surface.reconfigure()?;
    }

    let surface_result = surface.render_frame(|mut frame| {
        let render_size = frame.size;
        let screen_descriptor = screen_descriptor(render_size, pixels_per_point);
        let preview_target = app.preview_target_for_surface(render_size, pixels_per_point);
        render_egui_frame(
            renderer,
            &mut frame,
            &textures_delta.set,
            &paint_jobs,
            &screen_descriptor,
        );
        if app.preview_frame_work_allowed_for_frame() && app.preview.needs_frame_work() {
            preview_frame_work_ran = true;
            preview_outcome = app.render_preview_frame(
                frame,
                preview_target,
                adapter_info,
                adapter_features,
                advance_preview,
            );
        }
    });
    let status = match surface_result {
        Ok(status) => status,
        Err(error) => {
            return Err(error.into());
        }
    };

    let preview_fullscreen_should_exit =
        app.preview_fullscreen_active() && !app.preview.needs_frame_work();
    if preview_fullscreen_should_exit && app.exit_preview_fullscreen(surface)? {
        scheduler.mark_resize();
    }

    for texture_id in &textures_delta.free {
        renderer.free_texture(texture_id);
    }

    let preview_repaint_after = app.preview.repaint_after();
    let active_vsync_cadence = app.preview.active_vsync_cadence();
    let repaint_after =
        submitted_frame_repaint_after(viewport_repaint_after, preview_repaint_after);

    if let Some(suboptimal) = preview_start_surface_submit_suboptimal(status) {
        if app
            .preview
            .observe_preview_start_surface_submit(suboptimal)
        {
            preview_outcome.dirty = true;
        }
    }

    match status {
        RenderFrameStatus::Submitted { .. } => {
            if let Some(message) = app
                .preview
                .after_surface_submit(preview_outcome.video_advanced)
            {
                app.set_status(message);
                preview_outcome.dirty = true;
            }
            if stop_surface_refresh_follow_up_needed(status, post_stop_refresh_follow_up) {
                app.preview.request_post_stop_follow_up_surface_refresh();
                preview_outcome.dirty = true;
            }
            if stop_surface_refresh_can_advance_after_submit(status, preview_frame_work_ran)
                && app
                    .preview
                    .advance_post_stop_surface_refresh(preview_frame_work_ran)
            {
                preview_outcome.dirty = true;
            }
            if app.advance_right_pane_transition() {
                preview_outcome.dirty = true;
            }
            scheduler.after_frame_with_video_cadence(
                now,
                repaint_after,
                active_vsync_cadence.map(|cadence| cadence.frame_duration),
                preview_outcome.video_advanced,
            );
        }
        RenderFrameStatus::SkippedZeroSize | RenderFrameStatus::Timeout => {
            scheduler.after_frame(start, Duration::MAX);
        }
        RenderFrameStatus::SurfaceChanged => scheduler.mark_dirty(),
    }
    if preview_outcome.dirty {
        scheduler.mark_dirty();
    }

    Ok(())
}

fn submitted_frame_repaint_after(
    viewport_repaint_after: Duration,
    preview_repaint_after: Option<Duration>,
) -> Duration {
    preview_repaint_after.map_or(viewport_repaint_after, |preview| {
        viewport_repaint_after.min(preview)
    })
}

fn stop_surface_refresh_can_advance_after_submit(
    status: RenderFrameStatus,
    preview_frame_work_ran: bool,
) -> bool {
    matches!(status, RenderFrameStatus::Submitted { .. }) && !preview_frame_work_ran
}

fn preview_start_surface_submit_suboptimal(status: RenderFrameStatus) -> Option<bool> {
    match status {
        RenderFrameStatus::Submitted { suboptimal } => Some(suboptimal),
        RenderFrameStatus::SkippedZeroSize
        | RenderFrameStatus::SurfaceChanged
        | RenderFrameStatus::Timeout => None,
    }
}

fn stop_surface_refresh_follow_up_needed(
    status: RenderFrameStatus,
    post_stop_refresh_follow_up: bool,
) -> bool {
    matches!(status, RenderFrameStatus::Submitted { suboptimal: true })
        && post_stop_refresh_follow_up
}

fn preview_advance_allowed_for_frame(
    input_allows_preview_advance: bool,
    scheduler_decision: LazyFrameDecision,
) -> bool {
    if !scheduler_decision.video_advance_allowed {
        return false;
    }
    input_allows_preview_advance
        || (scheduler_decision.active_vsync_fixed_cadence && scheduler_decision.video_tick_due)
}

fn reconfigure_surface_present_mode(
    surface: &mut Sdl2WgpuSurface,
    present_mode: wgpu::PresentMode,
) -> Result<(), GuiError> {
    if surface.current_present_mode() != present_mode {
        surface.set_preferred_present_mode(present_mode)?;
    }
    Ok(())
}

fn render_egui_frame(
    renderer: &mut Renderer,
    frame: &mut RenderFrameContext<'_>,
    texture_updates: &[(egui::TextureId, egui::epaint::ImageDelta)],
    paint_jobs: &[egui::ClippedPrimitive],
    screen_descriptor: &ScreenDescriptor,
) {
    for (texture_id, image_delta) in texture_updates {
        renderer.update_texture(frame.device, frame.queue, *texture_id, image_delta);
    }

    let command_buffers = renderer.update_buffers(
        frame.device,
        frame.queue,
        frame.encoder,
        paint_jobs,
        screen_descriptor,
    );
    if !command_buffers.is_empty() {
        frame.queue.submit(command_buffers);
    }

    let render_pass = frame
        .encoder
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mcraw4vulkan gui egui render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear_color()),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    renderer.render(
        &mut render_pass.forget_lifetime(),
        paint_jobs,
        screen_descriptor,
    );
}

fn handle_platform_output(
    surface: &Sdl2WgpuSurface,
    app: &mut GuiApp,
    scheduler: &mut LazyRepaintState,
    platform_output: PlatformOutput,
) {
    for command in platform_output.commands {
        if let OutputCommand::CopyText(text) = command {
            match surface.set_clipboard_text(&text) {
                Ok(()) => app.set_status("Pipe command copied."),
                Err(error) => app.set_status(format!("Clipboard copy failed: {error}")),
            }
            scheduler.mark_dirty();
        }
    }
}

fn screen_descriptor(size: WindowSize, pixels_per_point: f32) -> ScreenDescriptor {
    ScreenDescriptor {
        size_in_pixels: [size.drawable_width, size.drawable_height],
        pixels_per_point: pixels_per_point.max(1.0),
    }
}

fn pixels_per_point(size: WindowSize) -> f32 {
    size.scale_factor().max(1.0) as f32
}

fn clear_color() -> wgpu::Color {
    let [red, green, blue] = style::BACKGROUND_RGB;
    wgpu::Color {
        r: f64::from(red) / 255.0,
        g: f64::from(green) / 255.0,
        b: f64::from(blue) / 255.0,
        a: 1.0,
    }
}

fn draw_divider(ui: &mut egui::Ui) {
    let rect = ui.available_rect_before_wrap();
    let y = ui.cursor().top();
    ui.painter().hline(
        rect.left()..=rect.right(),
        y,
        Stroke::new(1.0, style::divider()),
    );
    ui.add_space(1.0);
}

fn drain_ready_events(
    event_pump: &mut sdl2::EventPump,
    surface: &mut Sdl2WgpuSurface,
    input: &mut EguiInputState,
    app: &mut GuiApp,
    scheduler: &mut LazyRepaintState,
) -> Result<bool, GuiError> {
    for event in event_pump.poll_iter() {
        if handle_event(event, surface, input, app, scheduler)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn handle_event(
    event: Event,
    surface: &mut Sdl2WgpuSurface,
    input: &mut EguiInputState,
    app: &mut GuiApp,
    scheduler: &mut LazyRepaintState,
) -> Result<bool, GuiError> {
    let window_id = surface.window_id();
    if event_targets_window(&event, window_id) {
        app.defer_preview_advance();
    }
    match event {
        Event::Quit { .. } => return Ok(true),
        Event::Window {
            window_id: event_window_id,
            win_event,
            ..
        } if event_window_id == window_id => {
            let result = handle_window_event(win_event, surface, input, scheduler);
            return result;
        }
        Event::KeyDown {
            window_id: event_window_id,
            keycode,
            keymod,
            repeat,
            ..
        } if event_window_id == window_id => {
            input.set_modifiers(keymod);
            if keycode == Some(Keycode::Escape) {
                return Ok(true);
            }
            if app.handle_optimizer_answer_key(keycode, repeat) {
                scheduler.mark_sdl_event();
                return Ok(false);
            }
            if preview_fullscreen_toggle_key(keycode, repeat) {
                if app.toggle_preview_fullscreen(surface)? {
                    scheduler.mark_resize();
                }
                return Ok(false);
            }
            if let Some(key) = key_from_sdl(keycode) {
                input.push_event(EguiEvent::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat,
                    modifiers: input.modifiers,
                });
                scheduler.mark_sdl_event();
            }
        }
        Event::KeyUp {
            window_id: event_window_id,
            keycode,
            keymod,
            ..
        } if event_window_id == window_id => {
            input.set_modifiers(keymod);
            if let Some(key) = key_from_sdl(keycode) {
                input.push_event(EguiEvent::Key {
                    key,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers: input.modifiers,
                });
                scheduler.mark_sdl_event();
            }
        }
        Event::TextInput {
            window_id: event_window_id,
            text,
            ..
        } if event_window_id == window_id => {
            input.push_event(EguiEvent::Text(text));
            scheduler.mark_sdl_event();
        }
        Event::MouseMotion {
            window_id: event_window_id,
            mousestate,
            x,
            y,
            ..
        } if event_window_id == window_id => {
            input.push_event(EguiEvent::PointerMoved(pos2(x as f32, y as f32)));
            if mouse_motion_requests_immediate_repaint(
                app.preview.active_vsync_cadence().is_some(),
                mousestate,
            ) {
                scheduler.mark_sdl_event();
            }
        }
        Event::MouseButtonDown {
            window_id: event_window_id,
            mouse_btn,
            x,
            y,
            ..
        } if event_window_id == window_id => {
            if let Some(button) = pointer_button_from_sdl(mouse_btn) {
                input.push_event(EguiEvent::PointerButton {
                    pos: pos2(x as f32, y as f32),
                    button,
                    pressed: true,
                    modifiers: input.modifiers,
                });
                scheduler.mark_sdl_event();
            }
        }
        Event::MouseButtonUp {
            window_id: event_window_id,
            mouse_btn,
            x,
            y,
            ..
        } if event_window_id == window_id => {
            if let Some(button) = pointer_button_from_sdl(mouse_btn) {
                input.push_event(EguiEvent::PointerButton {
                    pos: pos2(x as f32, y as f32),
                    button,
                    pressed: false,
                    modifiers: input.modifiers,
                });
                scheduler.mark_sdl_event();
            }
        }
        Event::MouseWheel {
            window_id: event_window_id,
            x,
            y,
            precise_x,
            precise_y,
            direction,
            ..
        } if event_window_id == window_id => {
            input.push_event(EguiEvent::MouseWheel {
                unit: MouseWheelUnit::Point,
                delta: wheel_delta(x, y, precise_x, precise_y, direction),
                modifiers: input.modifiers,
            });
            scheduler.mark_sdl_event();
        }
        Event::DropBegin {
            window_id: event_window_id,
            ..
        } if event_window_id == window_id => {
            app.begin_drop();
        }
        Event::DropFile {
            window_id: event_window_id,
            filename,
            ..
        } if event_window_id == window_id => {
            let playlist_changed = app.queue_drop_file(filename);
            if playlist_changed {
                scheduler.mark_sdl_event();
            }
        }
        Event::DropComplete {
            window_id: event_window_id,
            ..
        } if event_window_id == window_id => {
            let playlist_changed = app.complete_drop();
            if playlist_changed {
                scheduler.mark_sdl_event();
            }
        }
        _ => {}
    }

    Ok(false)
}

fn preview_fullscreen_toggle_key(keycode: Option<Keycode>, repeat: bool) -> bool {
    !repeat && keycode == Some(Keycode::F)
}

fn event_targets_window(event: &Event, window_id: u32) -> bool {
    match event {
        Event::Window {
            window_id: event_window_id,
            ..
        }
        | Event::KeyDown {
            window_id: event_window_id,
            ..
        }
        | Event::KeyUp {
            window_id: event_window_id,
            ..
        }
        | Event::TextInput {
            window_id: event_window_id,
            ..
        }
        | Event::MouseMotion {
            window_id: event_window_id,
            ..
        }
        | Event::MouseButtonDown {
            window_id: event_window_id,
            ..
        }
        | Event::MouseButtonUp {
            window_id: event_window_id,
            ..
        }
        | Event::MouseWheel {
            window_id: event_window_id,
            ..
        }
        | Event::DropBegin {
            window_id: event_window_id,
            ..
        }
        | Event::DropFile {
            window_id: event_window_id,
            ..
        }
        | Event::DropComplete {
            window_id: event_window_id,
            ..
        } => *event_window_id == window_id,
        _ => false,
    }
}

fn handle_window_event(
    event: WindowEvent,
    surface: &mut Sdl2WgpuSurface,
    input: &mut EguiInputState,
    scheduler: &mut LazyRepaintState,
) -> Result<bool, GuiError> {
    match event {
        WindowEvent::Close => return Ok(true),
        WindowEvent::Resized(_, _) | WindowEvent::SizeChanged(_, _) | WindowEvent::Restored => {
            input.set_minimized(false);
            surface.reconfigure()?;
            scheduler.mark_resize();
        }
        WindowEvent::Exposed | WindowEvent::Shown | WindowEvent::Maximized => {
            input.set_minimized(false);
            scheduler.mark_sdl_event();
        }
        WindowEvent::Minimized | WindowEvent::Hidden => {
            input.set_minimized(true);
            scheduler.mark_sdl_event();
        }
        WindowEvent::FocusGained => {
            input.set_focused(true);
            scheduler.mark_sdl_event();
        }
        WindowEvent::FocusLost => {
            input.set_focused(false);
            scheduler.mark_sdl_event();
        }
        WindowEvent::Leave => {
            input.push_event(EguiEvent::PointerGone);
            scheduler.mark_sdl_event();
        }
        _ => {}
    }

    Ok(false)
}

fn key_from_sdl(keycode: Option<Keycode>) -> Option<egui::Key> {
    match keycode {
        Some(Keycode::Escape) => Some(egui::Key::Escape),
        Some(Keycode::KpEnter) | Some(Keycode::Return) => Some(egui::Key::Enter),
        Some(Keycode::Tab) => Some(egui::Key::Tab),
        Some(Keycode::Backspace) => Some(egui::Key::Backspace),
        Some(Keycode::Delete) => Some(egui::Key::Delete),
        Some(Keycode::Home) => Some(egui::Key::Home),
        Some(Keycode::End) => Some(egui::Key::End),
        Some(Keycode::PageUp) => Some(egui::Key::PageUp),
        Some(Keycode::PageDown) => Some(egui::Key::PageDown),
        Some(Keycode::Up) => Some(egui::Key::ArrowUp),
        Some(Keycode::Down) => Some(egui::Key::ArrowDown),
        Some(Keycode::Left) => Some(egui::Key::ArrowLeft),
        Some(Keycode::Right) => Some(egui::Key::ArrowRight),
        _ => None,
    }
}

fn pointer_button_from_sdl(button: MouseButton) -> Option<PointerButton> {
    match button {
        MouseButton::Left => Some(PointerButton::Primary),
        MouseButton::Right => Some(PointerButton::Secondary),
        MouseButton::Middle => Some(PointerButton::Middle),
        MouseButton::X1 => Some(PointerButton::Extra1),
        MouseButton::X2 => Some(PointerButton::Extra2),
        _ => None,
    }
}

fn mouse_motion_requests_immediate_repaint(
    active_vsync_preview: bool,
    mousestate: MouseState,
) -> bool {
    !active_vsync_preview || mouse_buttons_pressed(mousestate)
}

fn mouse_buttons_pressed(mousestate: MouseState) -> bool {
    mousestate.left()
        || mousestate.middle()
        || mousestate.right()
        || mousestate.x1()
        || mousestate.x2()
}

fn wheel_delta(
    x: i32,
    y: i32,
    precise_x: f32,
    precise_y: f32,
    direction: MouseWheelDirection,
) -> Vec2 {
    let mut x = if precise_x.abs() > f32::EPSILON {
        precise_x
    } else {
        x as f32
    };
    let mut y = if precise_y.abs() > f32::EPSILON {
        precise_y
    } else {
        y as f32
    };

    if matches!(direction, MouseWheelDirection::Flipped) {
        x = -x;
        y = -y;
    }

    vec2(x * 40.0, -y * 40.0)
}

fn modifiers_from_sdl(keymod: Mod) -> Modifiers {
    let ctrl = keymod.intersects(Mod::LCTRLMOD | Mod::RCTRLMOD);
    Modifiers {
        alt: keymod.intersects(Mod::LALTMOD | Mod::RALTMOD),
        ctrl,
        shift: keymod.intersects(Mod::LSHIFTMOD | Mod::RSHIFTMOD),
        mac_cmd: false,
        command: ctrl,
    }
}

fn line_color(status: LineStatus) -> egui::Color32 {
    match status {
        LineStatus::Available => style::bright_blue(),
        LineStatus::Missing => style::header_text(),
        LineStatus::Unknown => style::body_text(),
        LineStatus::Informational => style::body_text(),
    }
}

fn draw_detail_line(ui: &mut egui::Ui, text: &str, status: LineStatus) {
    ui.horizontal_wrapped(|ui| {
        ui.label(
            RichText::new(text)
                .size(style::LINE_FONT_SIZE)
                .color(line_color(status)),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "mcraw4vulkan-gui-app-{name}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn app_with_playlist_store(name: &str) -> (GuiApp, PathBuf) {
        let path = temp_path(name).join("config").join("playlist.json");
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.playlist_store = PlaylistStore::from_path(path.clone());
        (app, path)
    }

    fn existing_mcraw_file(name: &str, file_name: &str) -> PathBuf {
        let path = temp_path(name).join(file_name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("temp parent");
        }
        std::fs::write(&path, b"mcraw").expect("temp mcraw");
        path
    }

    fn existing_mcraw_files(name: &str, file_names: &[&str]) -> Vec<PathBuf> {
        file_names
            .iter()
            .enumerate()
            .map(|(index, file_name)| existing_mcraw_file(&format!("{name}-{index}"), file_name))
            .collect()
    }

    fn pipe_example_facts(width: u32) -> mcraw4vulkan::PipeExampleFacts {
        use mcraw4vulkan::{PipeAspectRatio, PipeExampleFacts, PipeMovCadence};

        let sample_aspect_ratio = PipeAspectRatio::square_pixels();
        let display_aspect_ratio =
            PipeAspectRatio::display_for_frame(width, 2160, sample_aspect_ratio)
                .expect("valid test display aspect ratio");
        PipeExampleFacts {
            width,
            height: 2160,
            cadence: PipeMovCadence::from_source_rate(30_000, 1_001).expect("valid test cadence"),
            sample_aspect_ratio,
            display_aspect_ratio,
        }
    }

    #[derive(Debug)]
    struct ButtonLabelAllocation {
        context: &'static str,
        label: &'static str,
        width: f32,
    }

    fn test_context_with_project_style() -> Context {
        let context = Context::default();
        style::apply_project_style(&context);
        let _ = context.run(RawInput::default(), |_context| {});
        context
    }

    fn measured_button_text_width(context: &Context, label: &str) -> f32 {
        context.fonts(|fonts| {
            fonts
                .layout_no_wrap(
                    label.to_string(),
                    FontId::proportional(style::BUTTON_FONT_SIZE),
                    style::body_text(),
                )
                .rect
                .width()
        })
    }

    fn project_button_horizontal_padding(context: &Context) -> f32 {
        context.style().spacing.button_padding.x * 2.0
    }

    #[derive(Debug)]
    struct RenderedTestRow<T> {
        row_rect: Rect,
        content_rect: Rect,
        payload: T,
    }

    #[derive(Debug)]
    struct QuickPreviewOptionsRects {
        fps_overlay: Rect,
    }

    #[derive(Debug)]
    struct DngGridRects {
        top_row: [Rect; 3],
        mount_all: Rect,
        unmount_all: Rect,
    }

    #[derive(Debug)]
    struct FooterButtonRects {
        button: Rect,
    }

    #[derive(Debug, Clone, Copy)]
    enum FooterProbeKind {
        Primary,
        Middle,
    }

    #[derive(Debug)]
    struct ProductionFooterGeometry {
        frame: ColumnFrameGeometry,
        row_layout: main_view::SharedControlsRowLayout,
        footer_row_rect: Rect,
        footer_content_rect: Rect,
        button_rect: Rect,
    }

    fn test_raw_input(width: f32, height: f32) -> RawInput {
        RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(width, height))),
            ..RawInput::default()
        }
    }

    fn render_test_controls_row<T>(
        width: f32,
        height: f32,
        top_padding: f32,
        bottom_padding: f32,
        render: impl FnOnce(&mut egui::Ui) -> T,
    ) -> RenderedTestRow<T> {
        let context = test_context_with_project_style();
        let mut captured = None;
        let mut render = Some(render);
        let screen_size = vec2(width.max(1.0) + 64.0, height.max(1.0) + 64.0);

        let _ = context.run(test_raw_input(screen_size.x, screen_size.y), |context| {
            egui::Area::new("row-geometry-test".into())
                .fixed_pos(Pos2::ZERO)
                .show(context, |ui| {
                    ui.set_width(width);
                    ui.set_height(height);
                    let (row_rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
                    let content_rect =
                        controls_row_content_rect(row_rect, top_padding, bottom_padding);
                    let mut child_ui = ui.new_child(
                        egui::UiBuilder::new()
                            .max_rect(content_rect)
                            .layout(Layout::top_down(Align::Min)),
                    );
                    child_ui.set_clip_rect(content_rect);
                    child_ui.set_width(content_rect.width());
                    child_ui.set_height(content_rect.height());
                    child_ui.set_width_range(content_rect.width()..=content_rect.width());
                    child_ui.set_height_range(content_rect.height()..=content_rect.height());
                    let payload = render
                        .take()
                        .expect("row geometry renderer is consumed once")(
                        &mut child_ui
                    );
                    captured = Some(RenderedTestRow {
                        row_rect,
                        content_rect,
                        payload,
                    });
                });
        });

        captured.expect("row geometry captured")
    }

    fn draw_single_option_row_response_rect(
        ui: &mut egui::Ui,
        label: &str,
        selected: bool,
    ) -> Rect {
        ui.horizontal(|ui| {
            let [button_width, empty_width] = equal_button_widths(ui);
            let response = draw_clickable_visual_button(ui, label, selected, button_width);
            ui.add_space(empty_width);
            response.rect
        })
        .inner
    }

    fn draw_three_button_row_response_rects(
        ui: &mut egui::Ui,
        cells: [ButtonCell<'_>; 3],
    ) -> [Rect; 3] {
        ui.horizontal(|ui| {
            let widths = equal_three_button_widths(ui);
            [
                draw_visual_button_with_sense(
                    ui,
                    cells[0].label,
                    cells[0].selected,
                    widths[0],
                    Sense::click(),
                )
                .rect,
                draw_visual_button_with_sense(
                    ui,
                    cells[1].label,
                    cells[1].selected,
                    widths[1],
                    Sense::click(),
                )
                .rect,
                draw_visual_button_with_sense(
                    ui,
                    cells[2].label,
                    cells[2].selected,
                    widths[2],
                    Sense::click(),
                )
                .rect,
            ]
        })
        .inner
    }

    fn draw_dng_batch_action_response_rects(ui: &mut egui::Ui) -> [Rect; 2] {
        ui.horizontal(|ui| {
            let widths = equal_three_button_widths(ui);
            let mount_all =
                draw_clickable_visual_button(ui, "Mount all DNGs", false, widths[0]).rect;
            let unmount_all =
                draw_clickable_visual_button(ui, "Unmount all DNGs", false, widths[1]).rect;
            draw_blank_button_cell(ui, widths[2]);
            [mount_all, unmount_all]
        })
        .inner
    }

    fn render_quick_preview_options_rects(
        height: f32,
        top_padding: f32,
        bottom_padding: f32,
    ) -> RenderedTestRow<QuickPreviewOptionsRects> {
        render_test_controls_row(
            main_view::MORE_OPTIONS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0,
            height,
            top_padding,
            bottom_padding,
            |ui| {
                draw_section_title(ui, "Quick Preview Options");
                ui.add_space(8.0);
                draw_option_pair(
                    ui,
                    VisualOption::new("Vsync On", true, PreviewTiming::VsyncOn),
                    VisualOption::new("Max speed", false, PreviewTiming::MaxSpeed),
                    |_| {},
                );
                ui.add_space(6.0);
                QuickPreviewOptionsRects {
                    fps_overlay: draw_single_option_row_response_rect(ui, "FPS overlay", false),
                }
            },
        )
    }

    fn render_dng_grid_rects(
        height: f32,
        top_padding: f32,
        bottom_padding: f32,
    ) -> RenderedTestRow<DngGridRects> {
        render_test_controls_row(
            main_view::PRIMARY_CONTROLS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0,
            height,
            top_padding,
            bottom_padding,
            |ui| {
                draw_section_title(ui, "DNG");
                draw_body_line(ui, "Mount raw video as a full-quality virtual DNG folder");
                draw_body_line(
                    ui,
                    "DNG vignette correction avoids the magenta shift of Quick Preview",
                );
                ui.add_space(8.0);
                let top_row = draw_three_button_row_response_rects(
                    ui,
                    [
                        ButtonCell::new("Mount DNG", false, true),
                        ButtonCell::new("Unmount DNG", false, true),
                        ButtonCell::new("Vignette Correction", false, true),
                    ],
                );
                ui.add_space(6.0);
                let batch_row = draw_dng_batch_action_response_rects(ui);
                DngGridRects {
                    top_row,
                    mount_all: batch_row[0],
                    unmount_all: batch_row[1],
                }
            },
        )
    }

    fn render_more_options_footer_rects(
        height: f32,
        top_padding: f32,
        bottom_padding: f32,
    ) -> RenderedTestRow<FooterButtonRects> {
        render_test_controls_row(
            main_view::PRIMARY_CONTROLS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0,
            height,
            top_padding,
            bottom_padding,
            |ui| FooterButtonRects {
                button: draw_clickable_visual_button(
                    ui,
                    "More Options",
                    false,
                    ui.available_width(),
                )
                .rect,
            },
        )
    }

    fn render_pipe_example_footer_rects(
        height: f32,
        top_padding: f32,
        bottom_padding: f32,
    ) -> RenderedTestRow<FooterButtonRects> {
        render_test_controls_row(
            main_view::MORE_OPTIONS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0,
            height,
            top_padding,
            bottom_padding,
            |ui| FooterButtonRects {
                button: draw_single_option_row_response_rect(ui, "Pipe Example", false),
            },
        )
    }

    fn render_production_footer_geometry(
        width: f32,
        outer_height: f32,
        kind: FooterProbeKind,
    ) -> ProductionFooterGeometry {
        let content_height = main_view::column_content_height(outer_height);
        let row_layout = main_view::shared_controls_row_layout(content_height);
        render_production_footer_geometry_with_rows(width, row_layout, kind)
    }

    fn render_production_scroll_footer_geometry(
        width: f32,
        viewport_outer_height: f32,
        kind: FooterProbeKind,
    ) -> ProductionFooterGeometry {
        let viewport_content_height = main_view::column_content_height(viewport_outer_height);
        let row_layout = main_view::shared_controls_row_layout(viewport_content_height);
        assert!(row_layout.outer_scroll_required);
        render_production_footer_geometry_with_rows(width, row_layout, kind)
    }

    fn render_production_footer_geometry_with_rows(
        width: f32,
        row_layout: main_view::SharedControlsRowLayout,
        kind: FooterProbeKind,
    ) -> ProductionFooterGeometry {
        let outer_height =
            row_layout.total_used_height + main_view::COLUMN_FRAME_INNER_MARGIN * 2.0;
        let context = test_context_with_project_style();
        let mut captured = None;
        let screen_size = vec2(width.max(1.0) + 64.0, outer_height.max(1.0) + 64.0);

        let _ = context.run(test_raw_input(screen_size.x, screen_size.y), |context| {
            egui::Area::new("production-footer-geometry-test".into())
                .fixed_pos(Pos2::ZERO)
                .show(context, |ui| {
                    ui.set_width(width);
                    ui.set_height(outer_height);
                    let (geometry, _output) = draw_column_frame_with_insets(
                        ui,
                        width,
                        outer_height,
                        standard_column_frame_insets(),
                        |ui, content_height, frame| {
                            assert_eq!(content_height, row_layout.total_used_height);
                            let footer_bounds =
                                row_layout.row_bounds(main_view::ControlRow::Footer);
                            let footer_row_rect = Rect::from_min_size(
                                frame.inner_rect.min + vec2(0.0, footer_bounds.top),
                                vec2(frame.inner_rect.width(), footer_bounds.height),
                            );
                            let footer_content_rect = controls_row_content_rect(
                                footer_row_rect,
                                main_view::FOOTER_ROW_TOP_PADDING,
                                main_view::FOOTER_BOTTOM_INSET,
                            );

                            draw_empty_controls_row(ui, row_layout.top_height);
                            draw_shared_row_gap(ui);
                            draw_empty_controls_row(ui, row_layout.quick_preview_height);
                            draw_shared_row_gap(ui);
                            draw_empty_controls_row(ui, row_layout.dng_height);
                            draw_shared_row_gap(ui);
                            draw_empty_controls_row(ui, row_layout.status_height);
                            draw_shared_row_gap(ui);
                            let mut button_rect = None;
                            draw_padded_controls_row(
                                ui,
                                row_layout.footer_height,
                                main_view::FOOTER_ROW_TOP_PADDING,
                                main_view::FOOTER_BOTTOM_INSET,
                                |ui| {
                                    button_rect = Some(match kind {
                                        FooterProbeKind::Primary => {
                                            draw_clickable_visual_button(
                                                ui,
                                                "More Options",
                                                false,
                                                ui.available_width(),
                                            )
                                            .rect
                                        }
                                        FooterProbeKind::Middle => {
                                            draw_single_option_row_response_rect(
                                                ui,
                                                "Pipe Example",
                                                false,
                                            )
                                        }
                                    });
                                },
                            );

                            captured = Some(ProductionFooterGeometry {
                                frame,
                                row_layout,
                                footer_row_rect,
                                footer_content_rect,
                                button_rect: button_rect.expect("footer button response"),
                            });
                        },
                    );
                    assert_eq!(geometry.insets, standard_column_frame_insets());
                });
        });

        captured.expect("production footer geometry captured")
    }

    fn assert_rect_within(label: &str, rect: Rect, container: Rect) {
        assert!(
            rect.min.y + f32::EPSILON >= container.min.y,
            "{label} min {} is above container min {}",
            rect.min.y,
            container.min.y
        );
        assert!(
            rect.max.y <= container.max.y + f32::EPSILON,
            "{label} max {} exceeds container max {}",
            rect.max.y,
            container.max.y
        );
    }

    fn assert_rect_fully_inside(label: &str, rect: Rect, container: Rect) {
        assert!(
            rect.min.x + f32::EPSILON >= container.min.x
                && rect.max.x <= container.max.x + f32::EPSILON
                && rect.min.y + f32::EPSILON >= container.min.y
                && rect.max.y <= container.max.y + f32::EPSILON,
            "{label} rect {:?} is not inside {:?}",
            rect,
            container
        );
    }

    fn assert_no_vertical_intersection(label: &str, rect: Rect, divider: Rect) {
        assert!(
            rect.max.y <= divider.min.y || rect.min.y >= divider.max.y,
            "{label} intersects divider: rect {:?}, divider {:?}",
            rect,
            divider
        );
    }

    fn following_divider_rect(row: Rect) -> Rect {
        Rect::from_min_size(pos2(row.min.x, row.max.y + 12.0), vec2(row.width(), 1.0))
    }

    fn ordinary_button_label_allocations() -> Vec<ButtonLabelAllocation> {
        let item_spacing = main_view::STANDARD_ITEM_SPACING;
        let primary_width = main_view::equal_three_button_widths(
            main_view::PRIMARY_CONTROLS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0,
            item_spacing,
        )[0];
        let more_options_width = main_view::equal_button_widths(
            main_view::MORE_OPTIONS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0,
            item_spacing,
        )[0];
        vec![
            ButtonLabelAllocation {
                context: "Header Quit",
                label: "Quit",
                width: header_quit_button_width(),
            },
            ButtonLabelAllocation {
                context: "Playlist Add files",
                label: "Add files",
                width: primary_width,
            },
            ButtonLabelAllocation {
                context: "Playlist Remove file",
                label: "Remove file",
                width: primary_width,
            },
            ButtonLabelAllocation {
                context: "Playlist Remove all",
                label: "Remove all",
                width: primary_width,
            },
            ButtonLabelAllocation {
                context: "Quick Preview Preview",
                label: "Preview",
                width: primary_width,
            },
            ButtonLabelAllocation {
                context: "Quick Preview Stop",
                label: "Stop",
                width: primary_width,
            },
            ButtonLabelAllocation {
                context: "Quick Preview Vignette Correction",
                label: "Vignette Correction",
                width: primary_width,
            },
            ButtonLabelAllocation {
                context: "DNG Mount DNG",
                label: "Mount DNG",
                width: primary_width,
            },
            ButtonLabelAllocation {
                context: "DNG Unmount DNG",
                label: "Unmount DNG",
                width: primary_width,
            },
            ButtonLabelAllocation {
                context: "DNG Vignette Correction",
                label: "Vignette Correction",
                width: primary_width,
            },
            ButtonLabelAllocation {
                context: "DNG Mount all DNGs",
                label: "Mount all DNGs",
                width: primary_width,
            },
            ButtonLabelAllocation {
                context: "DNG Unmount all DNGs",
                label: "Unmount all DNGs",
                width: primary_width,
            },
            ButtonLabelAllocation {
                context: "Display Play",
                label: "Play",
                width: transport_button_width(),
            },
            ButtonLabelAllocation {
                context: "Display Pause",
                label: "Pause",
                width: transport_button_width(),
            },
            ButtonLabelAllocation {
                context: "Display Stop",
                label: "Stop",
                width: transport_button_width(),
            },
            ButtonLabelAllocation {
                context: "Shared GPU Decoding",
                label: "GPU Decoding",
                width: more_options_width,
            },
            ButtonLabelAllocation {
                context: "Shared CPU Decoding",
                label: "CPU Decoding",
                width: more_options_width,
            },
            ButtonLabelAllocation {
                context: "Run optimizer now",
                label: "Run optimizer now",
                width: more_options_width,
            },
            ButtonLabelAllocation {
                context: "Default Settings",
                label: "Default Settings",
                width: more_options_width,
            },
            ButtonLabelAllocation {
                context: "Optimized Settings",
                label: "Optimized Settings",
                width: more_options_width,
            },
            ButtonLabelAllocation {
                context: "Vsync On",
                label: "Vsync On",
                width: more_options_width,
            },
            ButtonLabelAllocation {
                context: "Max speed",
                label: "Max speed",
                width: more_options_width,
            },
            ButtonLabelAllocation {
                context: "FPS overlay",
                label: "FPS overlay",
                width: more_options_width,
            },
            ButtonLabelAllocation {
                context: "Pipe Example",
                label: "Pipe Example",
                width: more_options_width,
            },
        ]
    }

    #[test]
    fn default_gui_window_config_is_1600_by_900() {
        let config = default_surface_config();

        assert_eq!(WINDOW_WIDTH, 1600);
        assert_eq!(WINDOW_HEIGHT, 900);
        assert_eq!(config.width, 1600);
        assert_eq!(config.height, 900);
        assert_eq!(
            config.preferred_present_mode,
            Some(preview::PREVIEW_NORMAL_PRESENT_MODE)
        );
    }

    #[test]
    fn splash_and_main_gui_share_default_window_config() {
        let config = default_surface_config();

        assert_eq!(config.width, WINDOW_WIDTH);
        assert_eq!(config.height, WINDOW_HEIGHT);
    }

    #[test]
    fn splash_window_size_request_is_1600_by_900_and_consumed_once() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert_eq!(app.take_splash_window_size_request(), Some((1600, 900)));
        assert_eq!(app.take_splash_window_size_request(), None);
    }

    #[test]
    fn splash_window_size_request_cannot_run_after_main_phase() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.phase = AppPhase::MainSkeleton;

        assert_eq!(app.take_splash_window_size_request(), None);
        assert_eq!(app.take_splash_window_size_request(), None);
    }

    #[test]
    fn splash_png_asset_decodes_to_native_top_area_size() {
        let image = decode_splash_png(SPLASH_IMAGE_BYTES).expect("splash PNG should decode");

        assert_eq!(image.size, [SPLASH_IMAGE_WIDTH, SPLASH_IMAGE_HEIGHT]);
    }

    #[test]
    fn splash_image_display_size_matches_native_window_width() {
        assert_eq!(splash_image_display_size(1600.0), vec2(1600.0, 347.0));
    }

    #[test]
    fn splash_image_display_size_preserves_proportions_in_logical_points() {
        assert_eq!(splash_image_display_size(800.0), vec2(800.0, 173.5));
    }

    #[test]
    fn gui_gpu_model_from_adapter_info_uses_name_and_backend() {
        let adapter_info = wgpu::AdapterInfo {
            name: "AMD Radeon RX 7900 XTX".to_string(),
            vendor: 0x1002,
            device: 0x744c,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };

        assert_eq!(
            gui_gpu_model_from_adapter_info(&adapter_info).as_deref(),
            Some("AMD Radeon RX 7900 XTX (vulkan)")
        );
    }

    #[test]
    fn gui_gpu_model_from_adapter_info_ignores_unknown_names() {
        let adapter_info = wgpu::AdapterInfo {
            name: "Unknown".to_string(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::Other,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };

        assert_eq!(gui_gpu_model_from_adapter_info(&adapter_info), None);
    }

    #[test]
    fn active_gui_default_is_not_doubled() {
        assert_ne!((WINDOW_WIDTH, WINDOW_HEIGHT), (3200, 1800));
    }

    #[test]
    fn button_width_mmm_measurement_is_covered_by_shared_increment() {
        let context = test_context_with_project_style();
        let measured = measured_button_text_width(&context, "MMM");

        println!("MMM button text width: {measured}");
        assert_eq!(style::BUTTON_FONT_SIZE, 20.0);
        assert!(measured > 0.0);
        assert!(main_view::BUTTON_THREE_CHARACTER_EXTRA_WIDTH >= measured.ceil());
    }

    #[test]
    fn ordinary_button_labels_fit_allocated_widths() {
        let context = test_context_with_project_style();
        let horizontal_padding = project_button_horizontal_padding(&context);
        let allocations = ordinary_button_label_allocations();

        assert_eq!(allocations.len(), 24);
        for allocation in allocations {
            let text_width = measured_button_text_width(&context, allocation.label);
            assert!(
                text_width + horizontal_padding <= allocation.width,
                "{} label '{}' needs {} plus padding {}, allocated {}",
                allocation.context,
                allocation.label,
                text_width,
                horizontal_padding,
                allocation.width
            );
        }
    }

    #[test]
    fn primary_vignette_and_unmount_all_labels_fit_wider_buttons() {
        let context = test_context_with_project_style();
        let horizontal_padding = project_button_horizontal_padding(&context);
        let primary_width = main_view::equal_three_button_widths(
            main_view::PRIMARY_CONTROLS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0,
            main_view::STANDARD_ITEM_SPACING,
        )[0];

        for label in ["Vignette Correction", "Mount all DNGs", "Unmount all DNGs"] {
            let text_width = measured_button_text_width(&context, label);
            assert!(
                text_width + horizontal_padding <= primary_width,
                "{label} needs {} plus padding {}, allocated {}",
                text_width,
                horizontal_padding,
                primary_width
            );
        }
    }

    #[test]
    fn more_options_footer_fits_full_primary_width() {
        let context = test_context_with_project_style();
        let horizontal_padding = project_button_horizontal_padding(&context);
        let text_width = measured_button_text_width(&context, "More Options");
        let footer_width =
            main_view::PRIMARY_CONTROLS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0;

        assert!(text_width + horizontal_padding <= footer_width);
    }

    #[test]
    fn button_heights_fonts_and_padding_remain_unchanged() {
        let context = test_context_with_project_style();
        let style = context.style();

        assert_eq!(ordinary_button_height(), 34.0);
        assert_eq!(TRANSPORT_BUTTON_HEIGHT, 32.0);
        assert_eq!(style::BUTTON_FONT_SIZE, 20.0);
        assert_eq!(style.spacing.button_padding.x, 12.0);
        assert_eq!(style.spacing.button_padding.y, 6.0);
    }

    #[test]
    fn normal_visual_button_paint_matches_existing_palette() {
        let idle = visual_button_paint(false, VisualButtonStrokeStyle::Normal);
        let selected = visual_button_paint(true, VisualButtonStrokeStyle::Normal);

        assert_eq!(idle.fill, style::button_dark());
        assert_eq!(idle.text_color, style::body_text());
        assert_eq!(idle.stroke, Stroke::new(1.0, style::divider()));
        assert_eq!(selected.fill, style::bright_blue());
        assert_eq!(selected.text_color, style::header_text());
        assert_eq!(selected.stroke, Stroke::new(1.0, style::bright_blue()));
    }

    #[test]
    fn orange_emphasis_visual_button_paint_changes_only_outline_color() {
        for selected in [false, true] {
            let normal = visual_button_paint(selected, VisualButtonStrokeStyle::Normal);
            let emphasized = visual_button_paint(selected, VisualButtonStrokeStyle::OrangeEmphasis);

            assert_eq!(emphasized.fill, normal.fill);
            assert_eq!(emphasized.text_color, normal.text_color);
            assert_eq!(emphasized.stroke.width, normal.stroke.width);
            assert_eq!(emphasized.stroke.color, style::orange_emphasis());
        }
    }

    #[test]
    fn orange_emphasis_button_cells_preserve_label_enablement_and_selection() {
        let preview = ButtonCell::new("Preview", false, false).orange_emphasis();
        let mount_dng = ButtonCell::new("Mount DNG", false, true).orange_emphasis();

        assert_eq!(preview.label, "Preview");
        assert!(!preview.selected);
        assert!(!preview.enabled);
        assert_eq!(
            preview.stroke_style,
            VisualButtonStrokeStyle::OrangeEmphasis
        );
        assert_eq!(mount_dng.label, "Mount DNG");
        assert!(!mount_dng.selected);
        assert!(mount_dng.enabled);
        assert_eq!(
            mount_dng.stroke_style,
            VisualButtonStrokeStyle::OrangeEmphasis
        );
    }

    #[test]
    fn row_geometry_undersized_rows_reproduce_fps_and_unmount_all_overflow() {
        let quick = render_quick_preview_options_rects(
            main_view::UNDERSIZED_QUICK_PREVIEW_ROW_HEIGHT,
            0.0,
            0.0,
        );
        let dng = render_dng_grid_rects(main_view::UNDERSIZED_DNG_ROW_HEIGHT, 0.0, 0.0);
        let more_options =
            render_more_options_footer_rects(main_view::UNDERSIZED_FOOTER_ROW_HEIGHT, 0.0, 0.0);
        let pipe_example =
            render_pipe_example_footer_rects(main_view::UNDERSIZED_FOOTER_ROW_HEIGHT, 0.0, 0.0);

        assert!(
            quick.payload.fps_overlay.max.y > quick.row_rect.max.y,
            "undersized FPS overlay should exceed row bottom: {:?} vs {:?}",
            quick.payload.fps_overlay,
            quick.row_rect
        );
        assert!(
            dng.payload.unmount_all.max.y > dng.row_rect.max.y,
            "undersized Unmount all DNGs should exceed row bottom: {:?} vs {:?}",
            dng.payload.unmount_all,
            dng.row_rect
        );
        assert!(
            more_options.row_rect.max.y - more_options.payload.button.max.y
                < main_view::FOOTER_BOTTOM_INSET
        );
        assert!(
            pipe_example.row_rect.max.y - pipe_example.payload.button.max.y
                < main_view::FOOTER_BOTTOM_INSET
        );
    }

    #[test]
    fn quick_preview_fps_overlay_row_geometry_fits_corrected_content_rect() {
        let quick = render_quick_preview_options_rects(
            main_view::SHARED_QUICK_PREVIEW_ROW_HEIGHT,
            main_view::CONTROL_ROW_TOP_PADDING,
            main_view::CONTROL_ROW_BOTTOM_PADDING,
        );

        assert_rect_within("FPS overlay", quick.payload.fps_overlay, quick.content_rect);
        assert_rect_within("FPS overlay row", quick.payload.fps_overlay, quick.row_rect);
        assert_no_vertical_intersection(
            "FPS overlay",
            quick.payload.fps_overlay,
            following_divider_rect(quick.row_rect),
        );
    }

    #[test]
    fn dng_grid_batch_action_row_geometry_fits_corrected_content_rect() {
        let dng = render_dng_grid_rects(
            main_view::SHARED_DNG_ROW_HEIGHT,
            main_view::CONTROL_ROW_TOP_PADDING,
            main_view::CONTROL_ROW_BOTTOM_PADDING,
        );

        for (index, rect) in dng.payload.top_row.into_iter().enumerate() {
            assert_rect_within(&format!("DNG top button {index}"), rect, dng.content_rect);
            assert_rect_within(&format!("DNG top button {index} row"), rect, dng.row_rect);
        }
        assert_rect_within("Mount all DNGs", dng.payload.mount_all, dng.content_rect);
        assert_rect_within("Mount all DNGs row", dng.payload.mount_all, dng.row_rect);
        assert_no_vertical_intersection(
            "Mount all DNGs",
            dng.payload.mount_all,
            following_divider_rect(dng.row_rect),
        );
        assert_rect_within(
            "Unmount all DNGs",
            dng.payload.unmount_all,
            dng.content_rect,
        );
        assert_rect_within(
            "Unmount all DNGs row",
            dng.payload.unmount_all,
            dng.row_rect,
        );
        assert_no_vertical_intersection(
            "Unmount all DNGs",
            dng.payload.unmount_all,
            following_divider_rect(dng.row_rect),
        );
    }

    #[test]
    fn footer_more_options_and_pipe_example_geometry_respects_panel_inset() {
        let more_options = render_more_options_footer_rects(
            main_view::SHARED_FOOTER_ROW_HEIGHT,
            main_view::FOOTER_ROW_TOP_PADDING,
            main_view::FOOTER_BOTTOM_INSET,
        );
        let pipe_example = render_pipe_example_footer_rects(
            main_view::SHARED_FOOTER_ROW_HEIGHT,
            main_view::FOOTER_ROW_TOP_PADDING,
            main_view::FOOTER_BOTTOM_INSET,
        );

        assert_rect_within(
            "More Options",
            more_options.payload.button,
            more_options.content_rect,
        );
        assert_rect_within(
            "Pipe Example",
            pipe_example.payload.button,
            pipe_example.content_rect,
        );
        assert_eq!(
            more_options.payload.button.top(),
            pipe_example.payload.button.top()
        );
        assert_eq!(
            more_options.payload.button.bottom(),
            pipe_example.payload.button.bottom()
        );
        assert_eq!(
            more_options.payload.button.center().y,
            pipe_example.payload.button.center().y
        );

        let layout = main_view::shared_controls_row_layout(792.0);
        let footer_top = layout.row_top(main_view::ControlRow::Footer);
        let more_options_panel_inner = Rect::from_min_size(
            Pos2::ZERO,
            vec2(
                main_view::PRIMARY_CONTROLS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0,
                layout.total_used_height,
            ),
        );
        let pipe_example_panel_inner = Rect::from_min_size(
            Pos2::ZERO,
            vec2(
                main_view::MORE_OPTIONS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0,
                layout.total_used_height,
            ),
        );
        let more_options_abs = more_options.payload.button.translate(vec2(0.0, footer_top));
        let pipe_example_abs = pipe_example.payload.button.translate(vec2(0.0, footer_top));

        assert_rect_within(
            "More Options panel",
            more_options_abs,
            more_options_panel_inner,
        );
        assert_rect_within(
            "Pipe Example panel",
            pipe_example_abs,
            pipe_example_panel_inner,
        );
        assert!(
            more_options_panel_inner.max.y - more_options_abs.max.y
                >= main_view::FOOTER_BOTTOM_INSET
        );
        assert!(
            pipe_example_panel_inner.max.y - pipe_example_abs.max.y
                >= main_view::FOOTER_BOTTOM_INSET
        );
        assert_no_vertical_intersection(
            "More Options",
            more_options.payload.button,
            following_divider_rect(more_options.row_rect),
        );
        assert_no_vertical_intersection(
            "Pipe Example",
            pipe_example.payload.button,
            following_divider_rect(pipe_example.row_rect),
        );
    }

    #[test]
    fn footer_geometry_uses_actual_frame_inner_rect() {
        let outer_height = 792.0 + main_view::COLUMN_FRAME_INNER_MARGIN * 2.0;
        let primary = render_production_footer_geometry(
            main_view::PRIMARY_CONTROLS_WIDTH,
            outer_height,
            FooterProbeKind::Primary,
        );
        let middle = render_production_footer_geometry(
            main_view::MORE_OPTIONS_WIDTH,
            outer_height,
            FooterProbeKind::Middle,
        );
        let actual_inner_height = main_view::column_content_height(outer_height);

        assert_eq!(outer_height, 828.0);
        assert_eq!(actual_inner_height, 792.0);
        assert_eq!(primary.frame.insets.left, 18.0);
        assert_eq!(primary.frame.insets.right, 18.0);
        assert_eq!(primary.frame.insets.top, 18.0);
        assert_eq!(primary.frame.insets.bottom, 18.0);
        assert_eq!(primary.frame.inner_rect.height(), actual_inner_height);
        assert_eq!(
            middle.frame.inner_rect.height(),
            primary.frame.inner_rect.height()
        );
        assert_eq!(
            primary.row_layout.available_height,
            primary.frame.inner_rect.height()
        );
        assert_eq!(
            middle.row_layout.available_height,
            middle.frame.inner_rect.height()
        );
        assert_eq!(
            primary.footer_row_rect.top(),
            primary.frame.inner_rect.top()
                + primary.row_layout.row_top(main_view::ControlRow::Footer)
        );
        assert_eq!(
            middle.footer_row_rect.top(),
            middle.frame.inner_rect.top()
                + middle.row_layout.row_top(main_view::ControlRow::Footer)
        );
        assert_eq!(primary.button_rect.top(), primary.footer_content_rect.top());
        assert_eq!(middle.button_rect.top(), middle.footer_content_rect.top());
    }

    #[test]
    fn production_footer_buttons_fit_real_primary_and_middle_inner_rects() {
        let outer_height = 792.0 + main_view::COLUMN_FRAME_INNER_MARGIN * 2.0;
        let primary = render_production_footer_geometry(
            main_view::PRIMARY_CONTROLS_WIDTH,
            outer_height,
            FooterProbeKind::Primary,
        );
        let middle = render_production_footer_geometry(
            main_view::MORE_OPTIONS_WIDTH,
            outer_height,
            FooterProbeKind::Middle,
        );

        assert_rect_fully_inside(
            "More Options frame inner",
            primary.button_rect,
            primary.frame.inner_rect,
        );
        assert_rect_fully_inside(
            "Pipe Example frame inner",
            middle.button_rect,
            middle.frame.inner_rect,
        );
        assert_rect_fully_inside(
            "More Options footer row",
            primary.button_rect,
            primary.footer_row_rect,
        );
        assert_rect_fully_inside(
            "Pipe Example footer row",
            middle.button_rect,
            middle.footer_row_rect,
        );
        assert_rect_fully_inside(
            "More Options footer content",
            primary.button_rect,
            primary.footer_content_rect,
        );
        assert_rect_fully_inside(
            "Pipe Example footer content",
            middle.button_rect,
            middle.footer_content_rect,
        );
        assert_eq!(
            primary.frame.inner_rect.max.y - primary.button_rect.max.y,
            main_view::FOOTER_BOTTOM_INSET
        );
        assert_eq!(
            middle.frame.inner_rect.max.y - middle.button_rect.max.y,
            main_view::FOOTER_BOTTOM_INSET
        );
        assert!(
            primary.frame.outer_rect.max.y - primary.button_rect.max.y
                > main_view::COLUMN_FRAME_INNER_MARGIN
        );
        assert!(
            middle.frame.outer_rect.max.y - middle.button_rect.max.y
                > main_view::COLUMN_FRAME_INNER_MARGIN
        );
        assert_eq!(
            primary.frame.clip_rect, primary.frame.inner_rect,
            "clip boundary should be the actual inner rect"
        );
        assert_eq!(middle.frame.clip_rect, middle.frame.inner_rect);
    }

    #[test]
    fn production_footer_pair_alignment_is_shared_after_inner_rect_fix() {
        let outer_height = 792.0 + main_view::COLUMN_FRAME_INNER_MARGIN * 2.0;
        let primary = render_production_footer_geometry(
            main_view::PRIMARY_CONTROLS_WIDTH,
            outer_height,
            FooterProbeKind::Primary,
        );
        let middle = render_production_footer_geometry(
            main_view::MORE_OPTIONS_WIDTH,
            outer_height,
            FooterProbeKind::Middle,
        );

        assert_eq!(primary.footer_row_rect.top(), middle.footer_row_rect.top());
        assert_eq!(
            primary.footer_row_rect.bottom(),
            middle.footer_row_rect.bottom()
        );
        assert_eq!(primary.button_rect.top(), middle.button_rect.top());
        assert_eq!(primary.button_rect.bottom(), middle.button_rect.bottom());
        assert_eq!(
            primary.button_rect.center().y,
            middle.button_rect.center().y
        );
        assert_eq!(
            primary.button_rect.width(),
            main_view::PRIMARY_CONTROLS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0
        );
        assert_eq!(
            middle.button_rect.width(),
            main_view::equal_button_widths(
                main_view::MORE_OPTIONS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0,
                main_view::STANDARD_ITEM_SPACING,
            )[0]
        );
    }

    #[test]
    fn production_footer_reference_heights_remain_contained() {
        for content_height in [792.0, 900.0, 1260.0, 1980.0] {
            let outer_height = content_height + main_view::COLUMN_FRAME_INNER_MARGIN * 2.0;
            let primary = render_production_footer_geometry(
                main_view::PRIMARY_CONTROLS_WIDTH,
                outer_height,
                FooterProbeKind::Primary,
            );
            let middle = render_production_footer_geometry(
                main_view::MORE_OPTIONS_WIDTH,
                outer_height,
                FooterProbeKind::Middle,
            );

            assert!(!primary.row_layout.outer_scroll_required);
            assert_eq!(primary.row_layout.quick_preview_height, 142.0);
            assert_eq!(primary.row_layout.dng_height, 206.0);
            assert_eq!(primary.row_layout.footer_height, 50.0);
            assert_rect_fully_inside(
                "More Options",
                primary.button_rect,
                primary.frame.inner_rect,
            );
            assert_rect_fully_inside("Pipe Example", middle.button_rect, middle.frame.inner_rect);
            assert!(
                primary.frame.inner_rect.max.y - primary.button_rect.max.y
                    >= main_view::STANDARD_ITEM_SPACING
            );
            assert!(
                middle.frame.inner_rect.max.y - middle.button_rect.max.y
                    >= main_view::STANDARD_ITEM_SPACING
            );
        }

        let compact_outer_height = 580.0 + main_view::COLUMN_FRAME_INNER_MARGIN * 2.0;
        let primary_scroll = render_production_scroll_footer_geometry(
            main_view::PRIMARY_CONTROLS_WIDTH,
            compact_outer_height,
            FooterProbeKind::Primary,
        );
        let middle_scroll = render_production_scroll_footer_geometry(
            main_view::MORE_OPTIONS_WIDTH,
            compact_outer_height,
            FooterProbeKind::Middle,
        );

        assert!(main_view::shared_controls_row_layout(580.0).outer_scroll_required);
        assert_rect_fully_inside(
            "More Options scroll content",
            primary_scroll.button_rect,
            primary_scroll.frame.inner_rect,
        );
        assert_rect_fully_inside(
            "Pipe Example scroll content",
            middle_scroll.button_rect,
            middle_scroll.frame.inner_rect,
        );
        assert_eq!(
            primary_scroll.footer_row_rect.top(),
            middle_scroll.footer_row_rect.top()
        );
        assert_eq!(
            primary_scroll.button_rect.bottom(),
            middle_scroll.button_rect.bottom()
        );
    }

    #[test]
    fn row_geometry_logical_size_references_preserve_alignment_and_shared_scroll() {
        let corrected_fit = main_view::shared_controls_row_layout(792.0);
        let reference_1080 = main_view::shared_controls_row_layout(900.0);
        let reference_1440 = main_view::shared_controls_row_layout(1260.0);
        let reference_2160 = main_view::shared_controls_row_layout(1980.0);
        let constrained_768 = main_view::shared_controls_row_layout(768.0);
        let small_1366x768 = main_view::shared_controls_row_layout(580.0);

        assert!(!corrected_fit.outer_scroll_required);
        assert!(!reference_1080.outer_scroll_required);
        assert!(!reference_1440.outer_scroll_required);
        assert!(!reference_2160.outer_scroll_required);
        assert!(constrained_768.outer_scroll_required);
        assert!(small_1366x768.outer_scroll_required);

        assert_eq!(
            corrected_fit.quick_preview_height,
            reference_1440.quick_preview_height
        );
        assert_eq!(corrected_fit.dng_height, reference_2160.dng_height);
        assert_eq!(corrected_fit.status_height, reference_2160.status_height);
        assert_eq!(corrected_fit.footer_height, reference_2160.footer_height);
        assert!(reference_1440.playlist_list_height > corrected_fit.playlist_list_height);
        assert!(reference_2160.playlist_list_height > reference_1440.playlist_list_height);
        assert_eq!(
            corrected_fit.row_top(main_view::ControlRow::Footer),
            corrected_fit.row_bottom(main_view::ControlRow::Status)
                + main_view::SHARED_CONTROL_ROW_GAP
        );
        assert!(constrained_768.total_used_height > constrained_768.available_height);
        assert!(small_1366x768.total_used_height > small_1366x768.available_height);
    }

    #[test]
    fn dng_grid_top_and_second_rows_match_required_order() {
        assert_eq!(
            dng_grid_top_row_labels(),
            ["Mount DNG", "Unmount DNG", "Vignette Correction"]
        );
        assert_eq!(
            dng_grid_second_row_labels(),
            [Some("Mount all DNGs"), Some("Unmount all DNGs"), None]
        );
    }

    #[test]
    fn dng_grid_aligns_mount_all_and_unmount_all_under_selected_actions() {
        let content_width =
            main_view::PRIMARY_CONTROLS_WIDTH - main_view::COLUMN_FRAME_INNER_MARGIN * 2.0;
        let positions = dng_grid_column_positions(content_width, main_view::STANDARD_ITEM_SPACING);
        let columns = main_view::dng_grid_columns(content_width, main_view::STANDARD_ITEM_SPACING);

        assert_eq!(positions[0], 0.0);
        assert_eq!(positions[1], columns[0] + main_view::STANDARD_ITEM_SPACING);
        assert_eq!(columns[1], columns[0]);
        assert_eq!(columns[2], columns[0]);
    }

    #[test]
    fn dng_grid_third_second_row_cell_is_blank_and_mount_all_exists() {
        assert!(!dng_blank_cell_is_interactive());
        assert!(mount_all_dngs_control_exists());
    }

    #[test]
    fn run_optimizer_button_is_selected_while_optimizer_is_active() {
        assert!(run_optimizer_button_selected(true));
        assert!(!run_optimizer_button_selected(false));
    }

    #[test]
    fn optimizer_terminal_result_distinguishes_all_terminal_states() {
        assert_eq!(
            optimizer_terminal_result(
                true,
                "exit status: 0",
                "",
                Some(OptimizerAnswer::Yes),
                false
            ),
            OptimizerTerminalResult::Applied
        );
        assert_eq!(
            optimizer_terminal_result(true, "exit status: 0", "", Some(OptimizerAnswer::No), false),
            OptimizerTerminalResult::Declined
        );
        assert_eq!(
            optimizer_terminal_result(true, "exit status: 0", "", None, false),
            OptimizerTerminalResult::NoChanges
        );
        assert_eq!(
            optimizer_terminal_result(
                false,
                "exit status: 1",
                "unrelated stderr",
                Some(OptimizerAnswer::Yes),
                false
            ),
            OptimizerTerminalResult::Failed("Optimizer failed: exit status: 1".to_string())
        );
        assert_eq!(
            optimizer_terminal_result(false, "signal: 9", "anything", None, true),
            OptimizerTerminalResult::Cancelled
        );
    }

    #[test]
    fn optimizer_terminal_result_uses_cli_reason_without_recalculating() {
        let throughput = "Display ratio: 0.100\nPIPE ratio: 0.100\nrecommendation_reason: offset_throughput_gain\n";
        let low_ram =
            "Total physical RAM: 128.00 GiB\nrecommendation_reason: offset_low_ram_no_regression\n";

        assert_eq!(
            optimizer_terminal_result(
                true,
                "exit status: 0",
                throughput,
                Some(OptimizerAnswer::Yes),
                false,
            ),
            OptimizerTerminalResult::AppliedThroughputGain
        );
        assert_eq!(
            optimizer_terminal_result(
                true,
                "exit status: 0",
                low_ram,
                Some(OptimizerAnswer::No),
                false,
            ),
            OptimizerTerminalResult::DeclinedLowRamNoRegression
        );
    }

    #[test]
    fn optimizer_short_input_diagnostic_maps_to_required_gui_message() {
        assert_eq!(
            optimizer_failure_message("exit status: 1", OPTIMIZER_SHORT_INPUT_MESSAGE),
            OPTIMIZER_SHORT_INPUT_MESSAGE
        );
    }

    #[test]
    fn optimizer_short_input_diagnostic_maps_with_prefix_and_newlines() {
        let prefixed =
            format!("error: optimizer shell-out runner failed: {OPTIMIZER_SHORT_INPUT_MESSAGE}");
        let surrounded = format!("\n\n{OPTIMIZER_SHORT_INPUT_MESSAGE}\n");

        assert_eq!(
            optimizer_failure_message("exit status: 1", &prefixed),
            OPTIMIZER_SHORT_INPUT_MESSAGE
        );
        assert_eq!(
            optimizer_failure_message("exit status: 1", &surrounded),
            OPTIMIZER_SHORT_INPUT_MESSAGE
        );
    }

    #[test]
    fn optimizer_short_input_diagnostic_from_captured_stream_maps_correctly() {
        let output = format!(
            "optimizer started\n{OPTIMIZER_SHORT_INPUT_MESSAGE}\noptimizer exited with exit status: 1\n"
        );

        assert_eq!(
            optimizer_terminal_result(false, "exit status: 1", &output, None, false),
            OptimizerTerminalResult::Failed(OPTIMIZER_SHORT_INPUT_MESSAGE.to_string())
        );
    }

    #[test]
    fn optimizer_unknown_failure_keeps_generic_exit_status() {
        assert_eq!(
            optimizer_failure_message("exit status: 1", "optimizer failed for another reason"),
            "Optimizer failed: exit status: 1"
        );
    }

    #[test]
    fn optimizer_failure_classifier_does_not_match_600_or_frames_alone() {
        assert_eq!(
            optimizer_failure_message("exit status: 1", "error: 600"),
            "Optimizer failed: exit status: 1"
        );
        assert_eq!(
            optimizer_failure_message("exit status: 1", "error: frames"),
            "Optimizer failed: exit status: 1"
        );
    }

    #[test]
    fn optimizer_short_input_gui_message_contains_no_path_or_exit_status() {
        let output = format!("private-source/clip.mcraw: {OPTIMIZER_SHORT_INPUT_MESSAGE}");
        let message = optimizer_failure_message("exit status: 1", &output);

        assert_eq!(message, OPTIMIZER_SHORT_INPUT_MESSAGE);
        assert!(!message.contains("private-source"));
        assert!(!message.contains("exit status: 1"));
    }

    #[test]
    fn optimizer_short_input_result_is_persistent_until_close() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let result = optimizer_terminal_result(
            false,
            "exit status: 1",
            OPTIMIZER_SHORT_INPUT_MESSAGE,
            None,
            false,
        );
        app.right_pane_owner = RightPaneOwner::Optimizer;
        app.optimizer.set_terminal_result(result);

        assert_eq!(app.right_pane_owner, RightPaneOwner::Optimizer);
        assert_eq!(
            app.optimizer
                .terminal_result()
                .map(OptimizerTerminalResult::message),
            Some(OPTIMIZER_SHORT_INPUT_MESSAGE)
        );

        app.close_optimizer_result();

        assert_eq!(app.right_pane_owner, RightPaneOwner::Idle);
        assert!(app.optimizer.terminal_result().is_none());
    }

    #[test]
    fn right_pane_owner_gates_preview_target_branch() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.preview_area = PreviewLogicalRect::new(20.0, 40.0, 300.0, 200.0);

        for owner in [
            RightPaneOwner::Idle,
            RightPaneOwner::PipeExample,
            RightPaneOwner::Optimizer,
        ] {
            app.right_pane_frame_owner = owner;
            assert!(
                app.preview_target_for_surface(WindowSize::new(800, 600, 1600, 1200), 2.0)
                    .is_none()
            );
        }

        app.right_pane_frame_owner = RightPaneOwner::Preview;
        assert!(
            app.preview_target_for_surface(WindowSize::new(800, 600, 1600, 1200), 2.0)
                .is_some()
        );
    }

    #[test]
    fn passing_preflight_splash_visible_dwell_is_five_seconds() {
        assert_eq!(preflight_splash_min_visible(), Duration::from_secs(5));
    }

    #[test]
    fn splash_version_label_uses_cargo_package_version() {
        assert_eq!(
            SPLASH_VERSION_LABEL.strip_prefix("Version "),
            Some(env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn splash_app_preserves_ram_line() {
        let ram_line = "Installed RAM 128 GB | Available RAM 84 GB".to_string();
        let app = GuiApp::new(ram_line.clone());

        assert_eq!(app.ram_line, ram_line);
    }

    #[test]
    fn playlist_status_reserves_exactly_three_lines() {
        assert_eq!(playlist_status_reserved_line_count(), 3);
    }

    #[test]
    fn playlist_status_reserved_height_depends_only_on_line_height() {
        let line_height = 18.0;
        let heights = [
            "",
            "Playlist loaded.",
            "A long playlist status message that would normally wrap and must stay constrained",
        ]
        .map(|_| playlist_status_reserved_height_for_line_height(line_height));

        assert_eq!(heights, [line_height * 3.0; 3]);
    }

    #[test]
    fn playlist_status_text_uses_project_blue() {
        assert_eq!(playlist_status_color(), style::bright_blue());
    }

    fn source_between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("start marker");
        let tail = &source[start_index..];
        let end_index = tail.find(end).expect("end marker");
        &tail[..end_index]
    }

    #[test]
    fn expanded_controls_use_one_shared_outer_scroll_area() {
        let source = include_str!("app.rs");
        let old_primary_scroll = ["primary", "-controls-scroll"].concat();
        let old_options_scroll_suffix = ["-options", "-scroll"].concat();

        assert!(source.contains("SHARED_CONTROLS_SCROLL_ID"));
        assert!(source.contains("shared-controls-scroll"));
        assert!(!source.contains(&old_primary_scroll));
        assert!(!source.contains(&old_options_scroll_suffix));
    }

    #[test]
    fn more_options_render_exactly_one_decode_selector() {
        let source = include_str!("app.rs");
        let top_settings = source_between(
            source,
            "fn draw_more_options_top_settings",
            "fn draw_quick_preview_options",
        );
        let quick_options = source_between(
            source,
            "fn draw_quick_preview_options",
            "fn draw_pipe_example_footer",
        );
        let dng_settings_title = ["DNG", " Decode Settings"].concat();

        assert_eq!(top_settings.matches("\"GPU Decoding\"").count(), 1);
        assert_eq!(top_settings.matches("\"CPU Decoding\"").count(), 1);
        assert_eq!(quick_options.matches("\"GPU Decoding\"").count(), 0);
        assert_eq!(quick_options.matches("\"CPU Decoding\"").count(), 0);
        assert!(!source.contains(&dng_settings_title));
    }

    #[test]
    fn dng_and_status_middle_rows_are_empty_matching_slots() {
        let source = include_str!("app.rs");
        let more_options_column = source_between(
            source,
            "fn draw_more_options_column",
            "fn handle_more_options_response",
        );

        assert!(more_options_column.contains("draw_empty_controls_row(ui, row_layout.dng_height)"));
        assert!(
            more_options_column.contains("draw_empty_controls_row(ui, row_layout.status_height)")
        );
    }

    #[test]
    fn footer_row_pairs_more_options_with_pipe_example() {
        let source = include_str!("app.rs");
        let primary_column = source_between(
            source,
            "fn draw_primary_controls_column",
            "fn draw_playlist_section",
        );
        let more_options_column = source_between(
            source,
            "fn draw_more_options_column",
            "fn handle_more_options_response",
        );

        assert!(primary_column.contains("self.draw_more_options_footer(ui)"));
        assert!(more_options_column.contains("let response = draw_pipe_example_footer(ui)"));
        assert!(primary_column.contains("row_layout.footer_height"));
        assert!(more_options_column.contains("row_layout.footer_height"));
    }

    #[test]
    fn active_preview_event_defers_one_advance() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let path = Path::new("clips/first.mcraw");
        let _ = app
            .preview
            .start(Some((path, "first")), &GuiSettings::default());

        assert!(app.take_preview_advance());

        app.defer_preview_advance();

        assert!(!app.take_preview_advance());
        assert!(app.take_preview_advance());
    }

    #[test]
    fn active_vsync_input_deferral_before_deadline_keeps_repaint_ui_only() {
        assert!(!preview_advance_allowed_for_frame(
            false,
            LazyFrameDecision {
                active_vsync_fixed_cadence: true,
                source_frame_duration: Some(Duration::from_millis(33)),
                video_tick_due: false,
                video_advance_allowed: false,
                ui_only_repaint: true,
            },
        ));
    }

    #[test]
    fn active_vsync_due_video_tick_ignores_ordinary_input_deferral() {
        assert!(preview_advance_allowed_for_frame(
            false,
            LazyFrameDecision {
                active_vsync_fixed_cadence: true,
                source_frame_duration: Some(Duration::from_millis(33)),
                video_tick_due: true,
                video_advance_allowed: true,
                ui_only_repaint: false,
            },
        ));
    }

    #[test]
    fn repeated_hover_deferrals_do_not_starve_due_active_vsync_ticks() {
        for _ in 0..120 {
            assert!(preview_advance_allowed_for_frame(
                false,
                LazyFrameDecision {
                    active_vsync_fixed_cadence: true,
                    source_frame_duration: Some(Duration::from_millis(33)),
                    video_tick_due: true,
                    video_advance_allowed: true,
                    ui_only_repaint: false,
                },
            ));
        }
    }

    #[test]
    fn no_vsync_preserves_existing_input_deferral_behavior() {
        assert!(!preview_advance_allowed_for_frame(
            false,
            LazyFrameDecision {
                active_vsync_fixed_cadence: false,
                source_frame_duration: None,
                video_tick_due: true,
                video_advance_allowed: true,
                ui_only_repaint: false,
            },
        ));
        assert!(preview_advance_allowed_for_frame(
            true,
            LazyFrameDecision {
                active_vsync_fixed_cadence: false,
                source_frame_duration: None,
                video_tick_due: true,
                video_advance_allowed: true,
                ui_only_repaint: false,
            },
        ));
    }

    #[test]
    fn stop_surface_refresh_advances_only_after_submitted_gui_only_frame() {
        assert!(stop_surface_refresh_can_advance_after_submit(
            RenderFrameStatus::Submitted { suboptimal: true },
            false,
        ));
        assert!(!stop_surface_refresh_can_advance_after_submit(
            RenderFrameStatus::Submitted { suboptimal: false },
            true,
        ));
        assert!(!stop_surface_refresh_can_advance_after_submit(
            RenderFrameStatus::SurfaceChanged,
            false,
        ));
        assert!(!stop_surface_refresh_can_advance_after_submit(
            RenderFrameStatus::Timeout,
            false,
        ));
        assert!(!stop_surface_refresh_can_advance_after_submit(
            RenderFrameStatus::SkippedZeroSize,
            false,
        ));
    }

    #[test]
    fn preview_start_surface_refresh_observes_only_submitted_frames() {
        assert_eq!(
            preview_start_surface_submit_suboptimal(RenderFrameStatus::Submitted {
                suboptimal: false,
            }),
            Some(false)
        );
        assert_eq!(
            preview_start_surface_submit_suboptimal(RenderFrameStatus::Submitted {
                suboptimal: true,
            }),
            Some(true)
        );
        assert_eq!(
            preview_start_surface_submit_suboptimal(RenderFrameStatus::SurfaceChanged),
            None
        );
        assert_eq!(
            preview_start_surface_submit_suboptimal(RenderFrameStatus::Timeout),
            None
        );
        assert_eq!(
            preview_start_surface_submit_suboptimal(RenderFrameStatus::SkippedZeroSize),
            None
        );
    }

    #[test]
    fn stop_surface_refresh_follow_up_requires_stop_owned_suboptimal_submit() {
        assert!(stop_surface_refresh_follow_up_needed(
            RenderFrameStatus::Submitted { suboptimal: true },
            true,
        ));
        assert!(!stop_surface_refresh_follow_up_needed(
            RenderFrameStatus::Submitted { suboptimal: false },
            true,
        ));
        assert!(!stop_surface_refresh_follow_up_needed(
            RenderFrameStatus::Submitted { suboptimal: true },
            false,
        ));
        assert!(!stop_surface_refresh_follow_up_needed(
            RenderFrameStatus::SurfaceChanged,
            true,
        ));
        assert!(!stop_surface_refresh_follow_up_needed(
            RenderFrameStatus::Timeout,
            true,
        ));
    }

    #[test]
    fn passive_pointer_motion_coalesces_during_active_vsync_preview() {
        assert!(!mouse_motion_requests_immediate_repaint(
            true,
            MouseState::from_sdl_state(0),
        ));
    }

    #[test]
    fn passive_pointer_motion_remains_immediate_without_active_vsync_preview() {
        assert!(mouse_motion_requests_immediate_repaint(
            false,
            MouseState::from_sdl_state(0),
        ));
    }

    #[test]
    fn pointer_drag_motion_remains_immediate_during_active_vsync_preview() {
        assert!(mouse_motion_requests_immediate_repaint(
            true,
            MouseState::from_sdl_state(1),
        ));
    }

    #[test]
    fn repeated_passive_pointer_motion_does_not_force_extra_active_vsync_repaints() {
        for _ in 0..120 {
            assert!(!mouse_motion_requests_immediate_repaint(
                true,
                MouseState::from_sdl_state(0),
            ));
        }
    }

    #[test]
    fn transport_row_geometry_reserves_frame_label_at_right_edge() {
        let geometry = transport_row_geometry(720.0);

        assert_eq!(
            geometry.frame_label_right,
            720.0 - TRANSPORT_ROW_SIDE_MARGIN
        );
        assert_eq!(
            geometry.frame_label_left,
            720.0 - TRANSPORT_ROW_SIDE_MARGIN - TRANSPORT_FRAME_LABEL_WIDTH
        );
    }

    #[test]
    fn transport_row_geometry_reserves_stop_gap_before_scrubber() {
        let geometry = transport_row_geometry(720.0);

        assert_eq!(geometry.play_left, TRANSPORT_ROW_SIDE_MARGIN);
        assert_eq!(
            geometry.play_right - geometry.play_left,
            TRANSPORT_BUTTON_WIDTH
        );
        assert_eq!(
            geometry.stop_right - geometry.stop_left,
            TRANSPORT_BUTTON_WIDTH
        );
        assert_eq!(
            geometry.scrubber_left,
            geometry.stop_right + TRANSPORT_ROW_GAP
        );
        assert!(geometry.scrubber_width() > 0.0);
    }

    #[test]
    fn transport_row_geometry_grows_scrubber_with_row_width() {
        let normal = transport_row_geometry(720.0);
        let wide = transport_row_geometry(1020.0);

        assert_eq!(normal.scrubber_width(), 128.0);
        assert_eq!(wide.scrubber_width(), 428.0);
        assert_eq!(wide.scrubber_width() - normal.scrubber_width(), 300.0);
    }

    #[test]
    fn transport_row_geometry_reserves_room_for_six_digit_frame_counter() {
        let geometry = transport_row_geometry(720.0);

        assert_eq!(
            geometry.frame_label_right - geometry.frame_label_left,
            TRANSPORT_FRAME_LABEL_WIDTH
        );
        assert_eq!(
            frame_counter_label(Some(PreviewPlaybackPosition {
                current_frame_index: Some(999_998),
                frame_count: 999_999,
            })),
            "Frame 999999 / 999999"
        );
        assert!(geometry.scrubber_right <= geometry.frame_label_left - TRANSPORT_ROW_GAP);
    }

    #[test]
    fn transport_row_geometry_clamps_tiny_widths_without_scrubber_overlap() {
        let geometry = transport_row_geometry(260.0);

        assert_eq!(
            geometry.frame_label_right,
            260.0 - TRANSPORT_ROW_SIDE_MARGIN
        );
        assert_eq!(geometry.scrubber_width(), 0.0);
        assert!(geometry.scrubber_left >= geometry.stop_right + TRANSPORT_ROW_GAP);
        assert!(geometry.scrubber_right >= geometry.scrubber_left);
    }

    #[test]
    fn transport_row_height_reserves_vertical_padding_for_controls() {
        let row_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(720.0, TRANSPORT_ROW_HEIGHT));
        let control_rect = transport_row_rect(
            row_rect,
            TRANSPORT_ROW_SIDE_MARGIN,
            TRANSPORT_ROW_SIDE_MARGIN + TRANSPORT_BUTTON_WIDTH,
            TRANSPORT_BUTTON_HEIGHT,
        );

        assert_eq!(
            row_rect.height() - control_rect.height(),
            TRANSPORT_ROW_VERTICAL_PADDING * 2.0
        );
        assert_eq!(TRANSPORT_SCRUBBER_HEIGHT, TRANSPORT_BUTTON_HEIGHT);
    }

    #[test]
    fn transport_widget_rects_fit_inside_padded_row_vertically() {
        let row_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(720.0, TRANSPORT_ROW_HEIGHT));
        let geometry = transport_row_geometry(row_rect.width());
        let play_rect = transport_row_rect(
            row_rect,
            geometry.play_left,
            geometry.play_right,
            TRANSPORT_BUTTON_HEIGHT,
        );
        let stop_rect = transport_row_rect(
            row_rect,
            geometry.stop_left,
            geometry.stop_right,
            TRANSPORT_BUTTON_HEIGHT,
        );
        let scrubber_rect = transport_row_rect(
            row_rect,
            geometry.scrubber_left,
            geometry.scrubber_right,
            TRANSPORT_SCRUBBER_HEIGHT,
        );

        for rect in [play_rect, stop_rect, scrubber_rect] {
            assert_eq!(rect.min.y - row_rect.min.y, TRANSPORT_ROW_VERTICAL_PADDING);
            assert_eq!(row_rect.max.y - rect.max.y, TRANSPORT_ROW_VERTICAL_PADDING);
        }
    }

    #[test]
    fn transport_widget_clip_keeps_horizontal_bounds_and_row_vertical_bounds() {
        let row_rect = Rect::from_min_size(pos2(10.0, 20.0), vec2(720.0, TRANSPORT_ROW_HEIGHT));
        let geometry = transport_row_geometry(row_rect.width());
        let scrubber_rect = transport_row_rect(
            row_rect,
            geometry.scrubber_left,
            geometry.scrubber_right,
            TRANSPORT_SCRUBBER_HEIGHT,
        );
        let clip_rect = transport_widget_clip_rect(row_rect, scrubber_rect);

        assert_eq!(clip_rect.min.x, scrubber_rect.min.x);
        assert_eq!(clip_rect.max.x, scrubber_rect.max.x);
        assert_eq!(clip_rect.min.y, row_rect.min.y);
        assert_eq!(clip_rect.max.y, row_rect.max.y);
    }

    #[test]
    fn transport_button_width_includes_shared_extra_characters() {
        assert_eq!(TRANSPORT_BASE_BUTTON_WIDTH, 112.0);
        assert_eq!(
            TRANSPORT_BUTTON_WIDTH,
            TRANSPORT_BASE_BUTTON_WIDTH + main_view::BUTTON_THREE_CHARACTER_EXTRA_WIDTH
        );
        assert_eq!(TRANSPORT_BUTTON_WIDTH, 168.0);
        assert_eq!(TRANSPORT_BUTTON_HEIGHT, 32.0);
    }

    #[test]
    fn frame_counter_label_without_active_clip_is_frame_zero() {
        assert_eq!(frame_counter_label(None), "Frame 0");
        assert_eq!(
            frame_counter_label(Some(PreviewPlaybackPosition {
                current_frame_index: None,
                frame_count: 0,
            })),
            "Frame 0"
        );
    }

    #[test]
    fn frame_counter_label_is_one_based_with_total_frames() {
        assert_eq!(
            frame_counter_label(Some(PreviewPlaybackPosition {
                current_frame_index: Some(0),
                frame_count: 100,
            })),
            "Frame 1 / 100"
        );
        assert_eq!(
            frame_counter_label(Some(PreviewPlaybackPosition {
                current_frame_index: Some(99),
                frame_count: 100,
            })),
            "Frame 100 / 100"
        );
    }

    #[test]
    fn frame_counter_label_clamps_to_last_user_frame() {
        assert_eq!(
            frame_counter_label(Some(PreviewPlaybackPosition {
                current_frame_index: Some(150),
                frame_count: 100,
            })),
            "Frame 100 / 100"
        );
    }

    #[test]
    fn scrubber_value_uses_first_frame_before_any_frame_is_displayed() {
        assert_eq!(
            scrubber_value_for_position(PreviewPlaybackPosition {
                current_frame_index: None,
                frame_count: 100,
            }),
            1.0
        );
    }

    #[test]
    fn scrubber_seek_index_maps_user_frames_to_zero_based_indices() {
        assert_eq!(scrubber_seek_index_from_value(1.0, 100), Some(0));
        assert_eq!(scrubber_seek_index_from_value(100.0, 100), Some(99));
        assert_eq!(scrubber_seek_index_from_value(0.0, 100), Some(0));
        assert_eq!(scrubber_seek_index_from_value(250.0, 100), Some(99));
        assert_eq!(scrubber_seek_index_from_value(50.4, 100), Some(49));
        assert_eq!(scrubber_seek_index_from_value(50.5, 100), Some(50));
    }

    #[test]
    fn scrubber_seek_index_ignores_empty_clips() {
        assert_eq!(scrubber_seek_index_from_value(1.0, 0), None);
    }

    #[test]
    fn submitted_frame_repaint_uses_post_render_preview_deadline_when_sooner() {
        assert_eq!(
            submitted_frame_repaint_after(
                Duration::from_millis(33),
                Some(Duration::from_millis(5))
            ),
            Duration::from_millis(5)
        );
    }

    #[test]
    fn submitted_frame_repaint_keeps_viewport_deadline_when_sooner() {
        assert_eq!(
            submitted_frame_repaint_after(
                Duration::from_millis(5),
                Some(Duration::from_millis(33))
            ),
            Duration::from_millis(5)
        );
    }

    #[test]
    fn submitted_frame_repaint_uses_immediate_preview_retry() {
        assert_eq!(
            submitted_frame_repaint_after(Duration::from_millis(16), Some(Duration::ZERO)),
            Duration::ZERO
        );
    }

    #[test]
    fn submitted_frame_repaint_uses_viewport_when_preview_idle() {
        assert_eq!(
            submitted_frame_repaint_after(Duration::from_millis(250), None),
            Duration::from_millis(250)
        );
    }

    #[test]
    fn inactive_preview_has_no_fullscreen_toggle_target_state() {
        let app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert_eq!(app.next_preview_fullscreen_state(), None);
        assert!(!app.preview_fullscreen_active());
    }

    #[test]
    fn active_preview_fullscreen_state_toggles_on_and_off() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let path = Path::new("clips/first.mcraw");
        let _ = app
            .preview
            .start(Some((path, "first")), &GuiSettings::default());

        assert_eq!(app.next_preview_fullscreen_state(), Some(true));

        app.preview_fullscreen = true;

        assert_eq!(app.next_preview_fullscreen_state(), Some(false));
    }

    #[test]
    fn stop_preview_clears_fullscreen_state() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let path = Path::new("clips/first.mcraw");
        let _ = app
            .preview
            .start(Some((path, "first")), &GuiSettings::default());
        app.preview_fullscreen = true;

        app.stop_preview();

        assert!(!app.preview_fullscreen_active());
        assert!(app.take_preview_fullscreen_exit_pending());
        assert!(!app.preview.is_running());
    }

    #[test]
    fn optimizer_profile_change_stops_active_preview() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let path = Path::new("clips/first.mcraw");
        let _ = app
            .preview
            .start(Some((path, "first")), &GuiSettings::default());
        app.preview_fullscreen = true;

        app.handle_optimizer_profile_changed(OptimizerProfile::Optimized);

        assert!(!app.preview.is_running());
        assert!(!app.preview_fullscreen_active());
        assert!(app.take_preview_fullscreen_exit_pending());
        assert_eq!(
            app.status_message.as_deref(),
            Some("Preview stopped. Click Preview again to use Optimized Settings.")
        );
    }

    #[test]
    fn optimizer_profile_change_while_preview_inactive_does_not_start_preview() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        app.handle_optimizer_profile_changed(OptimizerProfile::Default);

        assert!(!app.preview.is_running());
        assert_eq!(
            app.status_message.as_deref(),
            Some("Default Settings selected.")
        );
    }

    #[test]
    fn removing_active_preview_source_clears_fullscreen_state() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let path = PathBuf::from("clips/first.mcraw");
        app.add_dropped_paths([path.clone()]);
        app.playlist.select(0);
        let _ = app
            .preview
            .start(Some((path.as_path(), "first")), &GuiSettings::default());
        app.preview_fullscreen = true;

        app.remove_selected_playlist_entry();

        assert!(!app.preview_fullscreen_active());
        assert!(app.take_preview_fullscreen_exit_pending());
        assert_eq!(app.playlist.selected_index(), None);
    }

    #[test]
    fn normal_preview_target_uses_display_rect() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.preview_area = PreviewLogicalRect::new(20.0, 40.0, 300.0, 200.0);
        app.right_pane_frame_owner = RightPaneOwner::Preview;

        let target = app
            .preview_target_for_surface(WindowSize::new(800, 600, 1600, 1200), 2.0)
            .expect("normal target");

        assert_eq!(target.origin_x, 40);
        assert_eq!(target.origin_y, 80);
        assert_eq!(target.width, 600);
        assert_eq!(target.height, 400);
    }

    #[test]
    fn fullscreen_preview_target_uses_full_drawable_rect() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.preview_area = PreviewLogicalRect::new(20.0, 40.0, 300.0, 200.0);
        app.preview_fullscreen = true;
        app.right_pane_frame_owner = RightPaneOwner::Preview;

        let target = app
            .preview_target_for_surface(WindowSize::new(1600, 1080, 3200, 2160), 2.0)
            .expect("fullscreen target");

        assert_eq!(target.origin_x, 0);
        assert_eq!(target.origin_y, 0);
        assert_eq!(target.width, 3200);
        assert_eq!(target.height, 2160);
    }

    #[test]
    fn f_key_toggles_preview_fullscreen_and_repeats_do_not() {
        assert!(preview_fullscreen_toggle_key(Some(Keycode::F), false));
        assert!(!preview_fullscreen_toggle_key(Some(Keycode::F), true));
        assert!(!preview_fullscreen_toggle_key(Some(Keycode::G), false));
        assert!(!preview_fullscreen_toggle_key(None, false));
    }

    #[test]
    fn app_starts_on_preflight_splash() {
        let app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert_eq!(app.phase, AppPhase::PreflightStarting);
        assert!(app.preflight_needs_startup_run());
        assert!(!matches!(app.phase, AppPhase::MainSkeleton));
    }

    #[test]
    fn splash_starts_with_no_main_window_maximize_request() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert!(!app.take_main_window_maximize_request());
    }

    #[test]
    fn splash_frames_do_not_arm_main_window_maximize_before_main() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let now = Instant::now();

        app.set_preflight_result(
            PreflightViewModel {
                status_line: "Ready to launch".to_string(),
                detail_line: String::new(),
                ready: true,
                lines: Vec::new(),
            },
            now,
        );
        app.advance_startup_phase(now + preflight_splash_min_visible() - Duration::from_millis(1));

        assert!(!matches!(app.phase, AppPhase::MainSkeleton));
        assert!(!app.take_main_window_maximize_request());
    }

    #[test]
    fn main_skeleton_is_not_shown_before_preflight_result_exists() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.advance_startup_phase(Instant::now() + Duration::from_secs(60));

        assert_eq!(app.phase, AppPhase::PreflightStarting);
        assert!(!matches!(app.phase, AppPhase::MainSkeleton));
    }

    #[test]
    fn preflight_pass_does_not_skip_visible_splash() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let now = Instant::now();

        app.set_preflight_result(
            PreflightViewModel {
                status_line: "Ready to launch".to_string(),
                detail_line: String::new(),
                ready: true,
                lines: Vec::new(),
            },
            now,
        );

        assert_eq!(
            app.phase,
            AppPhase::PreflightReadyVisible {
                transition_at: now + preflight_splash_min_visible()
            }
        );
        assert!(!app.preflight_needs_startup_run());
        assert_eq!(
            app.preflight_repaint_after(now),
            Some(preflight_splash_min_visible())
        );
        assert!(!matches!(app.phase, AppPhase::MainSkeleton));
    }

    #[test]
    fn preflight_pass_transitions_to_main_after_visible_dwell() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let now = Instant::now();

        app.set_preflight_result(
            PreflightViewModel {
                status_line: "Ready to launch".to_string(),
                detail_line: String::new(),
                ready: true,
                lines: Vec::new(),
            },
            now,
        );
        app.advance_startup_phase(now + preflight_splash_min_visible() - Duration::from_millis(1));
        assert!(!matches!(app.phase, AppPhase::MainSkeleton));

        app.advance_startup_phase(now + preflight_splash_min_visible());
        assert_eq!(app.phase, AppPhase::MainSkeleton);
        assert!(app.take_main_window_maximize_request());
        assert!(!app.take_main_window_maximize_request());
        assert_eq!(
            app.preflight_repaint_after(now + preflight_splash_min_visible()),
            None
        );
    }

    #[test]
    fn remaining_in_main_does_not_rearm_main_window_maximize() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let now = Instant::now();

        app.set_preflight_result(
            PreflightViewModel {
                status_line: "Ready to launch".to_string(),
                detail_line: String::new(),
                ready: true,
                lines: Vec::new(),
            },
            now,
        );
        app.advance_startup_phase(now + preflight_splash_min_visible());
        assert!(app.take_main_window_maximize_request());

        app.advance_startup_phase(now + preflight_splash_min_visible() + Duration::from_secs(1));

        assert_eq!(app.phase, AppPhase::MainSkeleton);
        assert!(!app.take_main_window_maximize_request());
    }

    #[test]
    fn restoring_main_window_does_not_rearm_startup_requests() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let now = Instant::now();

        assert_eq!(app.take_splash_window_size_request(), Some((1600, 900)));
        app.set_preflight_result(
            PreflightViewModel {
                status_line: "Ready to launch".to_string(),
                detail_line: String::new(),
                ready: true,
                lines: Vec::new(),
            },
            now,
        );
        app.advance_startup_phase(now + preflight_splash_min_visible());
        assert!(app.take_main_window_maximize_request());

        app.advance_startup_phase(now + preflight_splash_min_visible() + Duration::from_secs(1));

        assert_eq!(app.phase, AppPhase::MainSkeleton);
        assert_eq!(app.take_splash_window_size_request(), None);
        assert!(!app.take_main_window_maximize_request());
    }

    #[test]
    fn right_pane_transitions_do_not_arm_startup_window_requests() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert_eq!(app.take_splash_window_size_request(), Some((1600, 900)));
        app.right_pane_owner = RightPaneOwner::PipeExample;
        app.pipe_example = Some(PipeExamplePanel::Message("example".to_string()));
        app.close_pipe_example();
        app.right_pane_owner = RightPaneOwner::Optimizer;
        app.optimizer
            .set_terminal_result(OptimizerTerminalResult::NoChanges);
        app.close_optimizer_result();

        assert_eq!(app.take_splash_window_size_request(), None);
        assert!(!app.take_main_window_maximize_request());
    }

    #[test]
    fn main_window_maximize_request_consumes_once() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let now = Instant::now();

        app.set_preflight_result(
            PreflightViewModel {
                status_line: "Ready to launch".to_string(),
                detail_line: String::new(),
                ready: true,
                lines: Vec::new(),
            },
            now,
        );
        app.advance_startup_phase(now + preflight_splash_min_visible());

        assert!(consume_main_window_maximize_request(&mut app, || Ok(())));
        assert!(!consume_main_window_maximize_request(&mut app, || Ok(())));
    }

    #[test]
    fn main_window_maximize_failure_is_nonfatal_and_not_retried() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let now = Instant::now();

        app.set_preflight_result(
            PreflightViewModel {
                status_line: "Ready to launch".to_string(),
                detail_line: String::new(),
                ready: true,
                lines: Vec::new(),
            },
            now,
        );
        app.advance_startup_phase(now + preflight_splash_min_visible());

        assert!(!consume_main_window_maximize_request(&mut app, || Err(
            Sdl2WgpuSurfaceError::WindowBuild("ignored by window manager".to_string())
        )));
        assert!(!app.take_main_window_maximize_request());
    }

    #[test]
    fn preflight_failure_keeps_not_ready_splash() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let now = Instant::now();

        app.set_preflight_result(
            PreflightViewModel {
                status_line: "Not Ready".to_string(),
                detail_line: String::new(),
                ready: false,
                lines: Vec::new(),
            },
            now,
        );
        app.advance_startup_phase(now + Duration::from_secs(60));

        assert_eq!(app.phase, AppPhase::PreflightNotReady);
        assert!(!app.preflight_needs_startup_run());
        assert!(!matches!(app.phase, AppPhase::MainSkeleton));
    }

    #[test]
    fn preflight_result_is_marked_run_once_across_frames() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let now = Instant::now();

        assert!(app.preflight_needs_startup_run());
        app.set_preflight_result(
            PreflightViewModel {
                status_line: "Ready to launch".to_string(),
                detail_line: String::new(),
                ready: true,
                lines: Vec::new(),
            },
            now,
        );
        assert!(!app.preflight_needs_startup_run());
        app.advance_startup_phase(now + Duration::from_millis(1));
        assert!(!app.preflight_needs_startup_run());
        app.advance_startup_phase(now + preflight_splash_min_visible());
        assert!(!app.preflight_needs_startup_run());
    }

    #[test]
    fn new_app_construction_resets_to_preflight_splash() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let now = Instant::now();
        app.set_preflight_result(
            PreflightViewModel {
                status_line: "Ready to launch".to_string(),
                detail_line: String::new(),
                ready: true,
                lines: Vec::new(),
            },
            now,
        );
        app.advance_startup_phase(now + preflight_splash_min_visible());
        assert_eq!(app.phase, AppPhase::MainSkeleton);

        let fresh_app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        assert_eq!(fresh_app.phase, AppPhase::PreflightStarting);
        assert!(fresh_app.preflight_needs_startup_run());
    }

    #[test]
    fn pipe_example_state_is_not_active_on_startup() {
        let app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert!(app.pipe_example.is_none());
    }

    #[test]
    fn right_pane_owner_defaults_to_idle_and_is_session_only() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert_eq!(app.right_pane_owner, RightPaneOwner::Idle);
        assert_eq!(app.right_pane_frame_owner, RightPaneOwner::Idle);
        assert!(app.pending_right_pane_request.is_none());

        app.right_pane_owner = RightPaneOwner::PipeExample;
        let fresh_app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert_eq!(fresh_app.right_pane_owner, RightPaneOwner::Idle);
        assert!(fresh_app.pending_right_pane_request.is_none());
    }

    #[test]
    fn right_pane_frame_owner_snapshot_is_stable_for_preview_target() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.preview_area = PreviewLogicalRect::new(20.0, 40.0, 300.0, 200.0);
        app.right_pane_owner = RightPaneOwner::Preview;
        app.right_pane_frame_owner = RightPaneOwner::Idle;

        assert!(
            app.preview_target_for_surface(WindowSize::new(800, 600, 1600, 1200), 2.0)
                .is_none()
        );

        app.right_pane_frame_owner = app.right_pane_owner;

        assert!(
            app.preview_target_for_surface(WindowSize::new(800, 600, 1600, 1200), 2.0)
                .is_some()
        );
    }

    #[test]
    fn preview_request_snapshots_selected_entry_and_settings() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/second.mcraw"),
        ]);
        app.playlist.select(0);
        app.settings.select_decode_mode(DecodeMode::Cpu);
        app.settings
            .select_optimizer_profile(OptimizerProfile::Optimized);

        let request = app.prepare_preview_request().expect("preview request");

        app.playlist.select(1);
        app.settings.select_decode_mode(DecodeMode::Gpu);
        app.settings
            .select_optimizer_profile(OptimizerProfile::Default);

        assert_eq!(request.source_path, PathBuf::from("clips/first.mcraw"));
        assert_eq!(request.display_name, "first");
        assert_eq!(request.settings.decode_mode(), DecodeMode::Cpu);
        assert_eq!(
            request.settings.optimizer_profile,
            OptimizerProfile::Optimized
        );
    }

    #[test]
    fn optimizer_request_snapshots_selected_entry() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let files = existing_mcraw_files(
            "optimizer-request-snapshot",
            &["first.mcraw", "second.mcraw"],
        );
        let first = files[0].clone();
        app.add_dropped_paths(files);
        app.playlist.select(0);

        let request = app.prepare_optimizer_request().expect("optimizer request");

        app.playlist.select(1);

        assert_eq!(request.source_path, first);
        assert_eq!(request.display_name, "first");
    }

    #[test]
    fn invalid_pipe_request_does_not_replace_existing_pending_request() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let preview_request = PreparedPreviewRequest {
            entry_id: 1,
            source_path: PathBuf::from("clips/first.mcraw"),
            display_name: "first".to_string(),
            settings: GuiSettings::default(),
        };
        app.pending_right_pane_request =
            Some(RightPaneRequest::StartPreview(preview_request.clone()));
        app.set_status("previous");

        app.request_pipe_owner();

        assert_eq!(
            app.pending_right_pane_request,
            Some(RightPaneRequest::StartPreview(preview_request))
        );
        assert_eq!(
            app.status_message.as_deref(),
            Some(pipe_example::NO_SELECTED_FILE_MESSAGE)
        );
    }

    #[test]
    fn latest_valid_request_replaces_pending_preview_handoff_request() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let _ = app.preview.start(
            Some((Path::new("clips/active.mcraw"), "active")),
            &app.settings,
        );
        app.right_pane_owner = RightPaneOwner::Preview;
        app.pending_right_pane_request =
            Some(RightPaneRequest::ShowPipeExample(PreparedPipeRequest {
                panel: PipeExamplePanel::Message("old pipe".to_string()),
            }));
        app.add_dropped_paths([PathBuf::from("clips/replacement.mcraw")]);
        app.playlist.select(0);

        app.request_preview_owner();

        let Some(RightPaneRequest::StartPreview(request)) = &app.pending_right_pane_request else {
            panic!("expected replacement preview request");
        };
        assert_eq!(
            request.source_path,
            PathBuf::from("clips/replacement.mcraw")
        );
        assert!(!app.preview.needs_frame_work());
    }

    #[test]
    fn pipe_request_carries_prepared_static_panel_snapshot() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/selected.mcraw")]);
        app.playlist.select(0);
        app.settings.select_decode_mode(DecodeMode::Cpu);

        let request = app
            .prepare_pipe_request_with_facts_source(|_| Ok(pipe_example_facts(4096)))
            .expect("pipe request");

        app.settings.select_decode_mode(DecodeMode::Gpu);

        let PipeExamplePanel::Example(example) = request.panel else {
            panic!("expected prepared example");
        };
        assert_eq!(example.decode_mode, DecodeMode::Cpu);
        assert!(example.command.contains("--cpu"));
    }

    #[test]
    fn pipe_owner_retains_snapshot_until_close_or_successful_replacement() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/selected.mcraw")]);
        app.playlist.select(0);

        app.show_pipe_example_with_facts_source(|_| Ok(pipe_example_facts(4096)));
        let original = app.pipe_example.clone();

        app.toggle_more_options();
        app.handle_optimizer_profile_changed(OptimizerProfile::Optimized);

        assert_eq!(app.right_pane_owner, RightPaneOwner::PipeExample);
        assert_eq!(app.pipe_example, original);

        app.close_pipe_example();

        assert_eq!(app.right_pane_owner, RightPaneOwner::Idle);
        assert!(app.pipe_example.is_none());
    }

    #[test]
    fn pipe_to_preview_keeps_pipe_visible_when_preview_request_is_invalid() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.pipe_example = Some(PipeExamplePanel::Message("example".to_string()));
        app.right_pane_owner = RightPaneOwner::PipeExample;

        app.request_preview_owner();

        assert_eq!(app.right_pane_owner, RightPaneOwner::PipeExample);
        assert_eq!(
            app.pipe_example,
            Some(PipeExamplePanel::Message("example".to_string()))
        );
        assert_eq!(
            app.status_message.as_deref(),
            Some(preview::NO_SELECTION_MESSAGE)
        );
    }

    #[test]
    fn pipe_to_preview_clears_pipe_only_after_preview_start_is_accepted() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.pipe_example = Some(PipeExamplePanel::Message("example".to_string()));
        app.right_pane_owner = RightPaneOwner::PipeExample;
        app.add_dropped_paths([PathBuf::from("clips/selected.mcraw")]);
        app.playlist.select(0);

        app.request_preview_owner();

        assert_eq!(app.right_pane_owner, RightPaneOwner::Preview);
        assert!(app.pipe_example.is_none());
        assert!(app.preview.is_running());
    }

    #[test]
    fn manual_preview_stop_releases_owner_only_after_handoff_ready() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/selected.mcraw")]);
        app.playlist.select(0);
        app.request_preview_owner();
        assert_eq!(app.right_pane_owner, RightPaneOwner::Preview);

        app.stop_preview();

        assert_eq!(app.right_pane_owner, RightPaneOwner::Preview);
        assert!(app.advance_right_pane_transition());
        assert_eq!(app.right_pane_owner, RightPaneOwner::Idle);
    }

    #[test]
    fn pipe_to_optimizer_launch_failure_preserves_pipe_panel() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.pipe_example = Some(PipeExamplePanel::Message("example".to_string()));
        app.right_pane_owner = RightPaneOwner::PipeExample;
        let path = existing_mcraw_file("pipe-to-optimizer-failure", "selected.mcraw");
        app.add_dropped_paths([path]);
        app.playlist.select(0);
        app.optimizer_binary = temp_path("missing-optimizer-binary");

        app.request_optimizer_owner();

        assert_eq!(app.right_pane_owner, RightPaneOwner::PipeExample);
        assert_eq!(
            app.pipe_example,
            Some(PipeExamplePanel::Message("example".to_string()))
        );
        assert_eq!(
            app.status_message.as_deref(),
            Some("Optimizer failed to start.")
        );
    }

    #[test]
    fn terminal_optimizer_close_returns_to_idle() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.right_pane_owner = RightPaneOwner::Optimizer;
        app.optimizer
            .set_terminal_result(OptimizerTerminalResult::NoChanges);

        app.close_optimizer_result();

        assert_eq!(app.right_pane_owner, RightPaneOwner::Idle);
        assert!(app.optimizer.terminal_result().is_none());
    }

    #[test]
    fn more_options_starts_collapsed_and_maps_to_condensed_layout() {
        let app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert!(!app.more_options_visible);
        assert_eq!(app.main_layout_mode(), main_view::MainLayoutMode::Condensed);
    }

    #[test]
    fn more_options_toggle_expands_and_collapses_layout_mode() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        app.toggle_more_options();
        assert!(app.more_options_visible);
        assert_eq!(app.main_layout_mode(), main_view::MainLayoutMode::Expanded);

        app.toggle_more_options();
        assert!(!app.more_options_visible);
        assert_eq!(app.main_layout_mode(), main_view::MainLayoutMode::Condensed);
    }

    #[test]
    fn more_options_toggle_does_not_change_processing_settings_or_playlist() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/second.mcraw"),
        ]);
        app.playlist.select(1);
        app.settings.select_decode_mode(DecodeMode::Cpu);
        app.settings
            .select_optimizer_profile(OptimizerProfile::Optimized);
        let settings = app.settings;
        let playlist = app.playlist.clone();

        app.toggle_more_options();
        app.toggle_more_options();

        assert_eq!(app.settings, settings);
        assert_eq!(app.playlist, playlist);
    }

    #[test]
    fn more_options_toggle_does_not_stop_active_preview() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let path = Path::new("clips/first.mcraw");
        let _ = app
            .preview
            .start(Some((path, "first")), &GuiSettings::default());

        app.toggle_more_options();

        assert!(app.preview.is_running());
        assert_eq!(app.main_layout_mode(), main_view::MainLayoutMode::Expanded);
    }

    #[test]
    fn more_options_toggle_does_not_reset_pipe_example_state() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.pipe_example = Some(PipeExamplePanel::Message("example".to_string()));

        app.toggle_more_options();
        app.toggle_more_options();

        assert_eq!(
            app.pipe_example,
            Some(PipeExamplePanel::Message("example".to_string()))
        );
    }

    #[test]
    fn more_options_state_is_session_only() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.toggle_more_options();

        let fresh_app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert!(app.more_options_visible);
        assert!(!fresh_app.more_options_visible);
        assert_eq!(
            fresh_app.main_layout_mode(),
            main_view::MainLayoutMode::Condensed
        );
    }

    #[test]
    fn gui_playlist_starts_empty_on_startup() {
        let app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert!(app.playlist.is_empty());
        assert_eq!(app.playlist.selected_index(), None);
    }

    #[test]
    fn add_files_saves_playlist_json_with_desired_unmounted() {
        let (mut app, path) = app_with_playlist_store("add-save");

        assert!(app.add_dropped_paths([PathBuf::from("clips/first.mcraw")]));

        let text = std::fs::read_to_string(path).expect("playlist saved");
        assert!(text.contains("\"source_path\": \"clips/first.mcraw\""));
        assert!(text.contains("\"desired_mount_state\": \"unmounted\""));
    }

    #[test]
    fn drag_drop_saves_playlist_json_with_desired_unmounted() {
        let (mut app, path) = app_with_playlist_store("drop-save");

        app.begin_drop();
        assert!(!app.queue_drop_file("clips/first.mcraw".to_string()));
        assert!(app.complete_drop());

        let text = std::fs::read_to_string(path).expect("playlist saved");
        assert!(text.contains("\"desired_mount_state\": \"unmounted\""));
    }

    #[test]
    fn startup_desired_mounted_rows_are_white_not_live() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let mut entry = crate::playlist::PlaylistEntry::new(1, PathBuf::from("clips/first.mcraw"));
        entry.desired_mount_state = DesiredMountState::Mounted;
        app.playlist = Playlist::from_entries(vec![entry]);

        let entry = &app.playlist.entries()[0];
        assert_eq!(entry.live_mount_state, LiveMountState::NotLive);
        assert_eq!(
            main_view::playlist_entry_text_tone(entry.visual_state()),
            main_view::PlaylistEntryTextTone::White
        );
    }

    #[test]
    fn mounted_this_session_rows_are_blue() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/first.mcraw")]);
        let id = app.playlist.entries()[0].id;
        assert!(
            app.playlist
                .set_live_mount_state(id, LiveMountState::MountedThisSession)
        );

        assert_eq!(
            main_view::playlist_entry_text_tone(app.playlist.entries()[0].visual_state()),
            main_view::PlaylistEntryTextTone::BrightBlue
        );
    }

    #[test]
    fn duplicate_live_source_policy_rejects_same_source_only() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/second.mcraw"),
        ]);
        let live_id = app.playlist.entries()[0].id;
        assert!(
            app.playlist
                .set_live_mount_state(live_id, LiveMountState::MountedThisSession)
        );

        assert!(app.duplicate_live_source_exists(
            app.playlist.entries()[1].id,
            &app.playlist.entries()[1].source_path
        ));
        assert!(!app.duplicate_live_source_exists(
            app.playlist.entries()[2].id,
            &app.playlist.entries()[2].source_path
        ));
    }

    #[test]
    fn remove_file_rejects_live_mounted_row() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/first.mcraw")]);
        app.playlist.select(0);
        let id = app.playlist.selected_id().expect("selected id");
        assert!(
            app.playlist
                .set_live_mount_state(id, LiveMountState::MountedThisSession)
        );

        app.remove_selected_playlist_entry();

        assert_eq!(app.playlist.len(), 1);
        assert_eq!(
            app.status_message.as_deref(),
            Some("Unmount DNG before removing a mounted playlist entry.")
        );
    }

    #[test]
    fn normal_exit_cleanup_preserves_desired_state() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.dng_binary = PathBuf::from("true");
        app.add_dropped_paths([PathBuf::from("clips/first.mcraw")]);
        app.playlist.select(0);
        assert!(
            app.playlist
                .set_selected_desired_mount_state(DesiredMountState::Mounted)
        );

        app.cleanup_dng_on_exit();

        assert_eq!(
            app.playlist.entries()[0].desired_mount_state,
            DesiredMountState::Mounted
        );
        assert_eq!(
            app.playlist.entries()[0].live_mount_state,
            LiveMountState::NotLive
        );
    }

    #[test]
    fn gui_drop_staging_adds_mcraw_files_in_order() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        app.begin_drop();
        assert!(!app.queue_drop_file("clips/first.mcraw".to_string()));
        assert!(!app.queue_drop_file("clips/second.MCRAW".to_string()));
        assert!(app.complete_drop());

        assert_eq!(app.playlist.len(), 2);
        assert_eq!(app.playlist.entries()[0].display_name, "first");
        assert_eq!(app.playlist.entries()[1].display_name, "second");
        assert_eq!(
            app.status_message.as_deref(),
            Some("Added 2 file(s) to playlist.")
        );
    }

    #[test]
    fn gui_drop_rejects_non_mcraw_with_status_message() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert!(app.add_dropped_paths([
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/not-a-clip.txt"),
        ]));

        assert_eq!(app.playlist.len(), 1);
        assert_eq!(
            app.status_message.as_deref(),
            Some("Added 1 file(s) to playlist. Ignored 1 non-mcraw file(s).")
        );
    }

    #[test]
    fn remove_file_removes_selected_playlist_entry_only() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/second.mcraw"),
        ]);
        app.playlist.select(0);
        app.pipe_example = Some(PipeExamplePanel::Message("example".to_string()));

        app.remove_selected_playlist_entry();

        assert_eq!(app.playlist.len(), 1);
        assert_eq!(app.playlist.entries()[0].display_name, "second");
        assert_eq!(app.playlist.selected_index(), None);
        assert!(app.pipe_example.is_none());
        assert_eq!(app.status_message.as_deref(), Some("Removed first."));
    }

    #[test]
    fn remove_all_empty_playlist_is_noop_without_persistence_or_error() {
        let (mut app, path) = app_with_playlist_store("remove-all-empty");
        app.set_status("Previous status.");

        app.remove_all_playlist_entries();

        assert!(app.playlist.is_empty());
        assert_eq!(app.playlist.selected_index(), None);
        assert!(!path.exists());
        assert_eq!(app.status_message.as_deref(), Some("Previous status."));
    }

    #[test]
    fn remove_all_clears_removable_entries_selection_and_persists_empty_playlist() {
        let (mut app, path) = app_with_playlist_store("remove-all-success");
        app.add_dropped_paths([
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/second.mcraw"),
        ]);
        app.playlist.select(1);
        app.pipe_example = Some(PipeExamplePanel::Message("example".to_string()));

        app.remove_all_playlist_entries();

        assert!(app.playlist.is_empty());
        assert_eq!(app.playlist.selected_index(), None);
        assert!(app.pipe_example.is_none());
        let text = std::fs::read_to_string(path).expect("empty playlist saved");
        assert!(text.contains("\"entries\": []"));
        assert_eq!(
            app.status_message.as_deref(),
            Some("Removed 2 playlist file(s).")
        );
    }

    #[test]
    fn remove_all_rejects_desired_mounted_entry_without_mutation_or_persistence() {
        let (mut app, path) = app_with_playlist_store("remove-all-desired-mounted");
        app.add_dropped_paths([
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/second.mcraw"),
        ]);
        app.playlist.select(1);
        assert!(
            app.playlist
                .set_selected_desired_mount_state(DesiredMountState::Mounted)
        );
        assert!(app.save_playlist_after_mutation());
        let before = std::fs::read_to_string(&path).expect("playlist saved");

        app.remove_all_playlist_entries();

        let after = std::fs::read_to_string(path).expect("playlist still saved");
        assert_eq!(app.playlist.len(), 2);
        assert_eq!(app.playlist.selected_index(), Some(1));
        assert_eq!(after, before);
        assert_eq!(
            app.status_message.as_deref(),
            Some("Unmount DNGs before removing all playlist entries.")
        );
    }

    #[test]
    fn remove_all_rejects_live_mounting_or_unmounting_without_partial_mutation() {
        for live_state in [LiveMountState::Mounting, LiveMountState::Unmounting] {
            let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
            app.add_dropped_paths([
                PathBuf::from("clips/first.mcraw"),
                PathBuf::from("clips/second.mcraw"),
            ]);
            let id = app.playlist.entries()[0].id;
            assert!(app.playlist.set_live_mount_state(id, live_state));

            app.remove_all_playlist_entries();

            assert_eq!(app.playlist.len(), 2);
            assert_eq!(app.playlist.entries()[0].live_mount_state, live_state);
            assert_eq!(
                app.status_message.as_deref(),
                Some("Unmount DNGs before removing all playlist entries.")
            );
        }
    }

    #[test]
    fn remove_all_rejects_mounted_this_session_entry_without_partial_mutation() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/second.mcraw"),
        ]);
        let id = app.playlist.entries()[0].id;
        assert!(
            app.playlist
                .set_live_mount_state(id, LiveMountState::MountedThisSession)
        );

        app.remove_all_playlist_entries();

        assert_eq!(app.playlist.len(), 2);
        assert_eq!(
            app.playlist.entries()[0].live_mount_state,
            LiveMountState::MountedThisSession
        );
        assert_eq!(
            app.status_message.as_deref(),
            Some("Unmount DNGs before removing all playlist entries.")
        );
    }

    #[test]
    fn remove_all_rejects_active_preview_without_mutation() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        let path = PathBuf::from("clips/first.mcraw");
        app.add_dropped_paths([path.clone()]);
        app.playlist.select(0);
        let _ = app
            .preview
            .start(Some((path.as_path(), "first")), &GuiSettings::default());

        app.remove_all_playlist_entries();

        assert_eq!(app.playlist.len(), 1);
        assert_eq!(app.playlist.selected_index(), Some(0));
        assert_eq!(
            app.status_message.as_deref(),
            Some("Stop Preview before removing all playlist entries.")
        );
    }

    #[test]
    fn remove_file_without_selection_shows_status_only() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/first.mcraw")]);

        app.remove_selected_playlist_entry();

        assert_eq!(app.playlist.len(), 1);
        assert_eq!(
            app.status_message.as_deref(),
            Some("Select a playlist file first.")
        );
    }

    #[test]
    fn pipe_example_uses_selected_playlist_full_path() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/selected file.mcraw")]);
        app.playlist.select(0);

        app.show_pipe_example_with_facts_source(|path| {
            assert_eq!(path, Path::new("clips/selected file.mcraw"));
            Ok(pipe_example_facts(4096))
        });

        let Some(PipeExamplePanel::Example(example)) = &app.pipe_example else {
            panic!("expected generated pipe example");
        };
        assert_eq!(
            example.input_path,
            PathBuf::from("clips/selected file.mcraw")
        );
        assert_eq!(
            example.output_file_name,
            "selected file_prores4444_bt2020_linear.mov"
        );
        assert_eq!(example.decode_mode, DecodeMode::Gpu);
        assert!(example.command.contains("--gpu"));
    }

    #[test]
    fn dng_flags_use_shared_decode_mode() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert_eq!(app.dng_flags().decode, DecodeMode::Gpu);

        app.settings.select_decode_mode(DecodeMode::Cpu);

        assert_eq!(app.dng_flags().decode, DecodeMode::Cpu);
    }

    #[test]
    fn macos_mount_limit_warning_threshold_is_exact_and_platform_gated() {
        assert_eq!(MACOS_MACFUSE_VFS_MOUNT_LIMIT, 64);
        assert!(!should_show_macos_mount_limit_warning(true, 63));
        assert!(should_show_macos_mount_limit_warning(true, 64));
        assert!(should_show_macos_mount_limit_warning(true, 65));
        assert!(should_show_macos_mount_limit_warning(true, 102));
        assert!(!should_show_macos_mount_limit_warning(false, 102));
    }

    #[test]
    fn macos_mount_limit_warning_text_is_exact_and_uses_captured_count() {
        let body = macos_mount_limit_warning_body(102);

        assert_eq!(MACOS_MOUNT_LIMIT_WARNING_TITLE, "macOS DNG mount limit");
        assert_eq!(MACOS_MOUNT_LIMIT_WARNING_BUTTON, "OK");
        assert_eq!(
            body,
            "macFUSE VFS supports a maximum of 64 simultaneous mounts system-wide.\n\n\
             This Playlist contains 102 clips. Other macFUSE volumes use the same mount slots, \
             so some DNG folders may fail to mount."
        );
        assert!(body.contains("macFUSE VFS"));
        assert!(body.contains("64"));
        assert!(body.contains("102"));
        let alternate_backend_name = ["FS", "Kit"].concat();
        assert!(!body.contains(&alternate_backend_name));
    }

    #[test]
    fn macos_mount_limit_warning_state_is_session_only_dismissible_and_repeatable() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert!(app.macos_mount_limit_warning.is_none());

        app.record_macos_mount_limit_warning(true, 64);
        assert_eq!(
            app.macos_mount_limit_warning,
            Some(MacosMountLimitWarning { playlist_count: 64 })
        );

        app.record_macos_mount_limit_warning(false, 102);
        assert_eq!(
            app.macos_mount_limit_warning,
            Some(MacosMountLimitWarning { playlist_count: 64 })
        );

        app.dismiss_macos_mount_limit_warning();
        assert!(app.macos_mount_limit_warning.is_none());

        app.record_macos_mount_limit_warning(true, 65);
        assert_eq!(
            app.macos_mount_limit_warning,
            Some(MacosMountLimitWarning { playlist_count: 65 })
        );
    }

    #[test]
    fn macos_mount_limit_warning_keeps_the_request_time_playlist_count() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths(
            (1..=64).map(|ordinal| PathBuf::from(format!("clips/clip-{ordinal:03}.mcraw"))),
        );
        app.record_macos_mount_limit_warning(true, app.playlist.len());

        app.add_dropped_paths([PathBuf::from("clips/clip-065.mcraw")]);

        let warning = app.macos_mount_limit_warning.expect("warning");
        assert_eq!(warning.playlist_count, 64);
        assert!(
            macos_mount_limit_warning_body(warning.playlist_count)
                .contains("This Playlist contains 64 clips.")
        );
    }

    #[test]
    fn macos_mount_limit_warning_is_not_saved_with_playlist_json() {
        let (mut app, playlist_path) = app_with_playlist_store("mount-limit-session-only");
        app.add_dropped_paths([PathBuf::from("clips/first.mcraw")]);
        app.record_macos_mount_limit_warning(true, 64);

        assert!(app.save_playlist_after_mutation());

        let saved = std::fs::read_to_string(playlist_path).expect("playlist saved");
        assert!(!saved.contains(MACOS_MOUNT_LIMIT_WARNING_TITLE));
        assert!(!saved.contains("macos_mount_limit_warning"));
        assert_eq!(
            app.macos_mount_limit_warning,
            Some(MacosMountLimitWarning { playlist_count: 64 })
        );
    }

    #[test]
    fn qualifying_mount_all_warning_is_informational_and_does_not_cap_the_queue() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.dng_binary = PathBuf::from("true");
        app.add_dropped_paths(
            (1..=102).map(|ordinal| PathBuf::from(format!("clips/clip-{ordinal:03}.mcraw"))),
        );
        app.playlist.select(64);
        app.settings.select_decode_mode(DecodeMode::Cpu);
        app.settings.toggle_dng_vignette();
        app.settings
            .select_optimizer_profile(OptimizerProfile::Optimized);
        let expected_order = app
            .playlist
            .entries()
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>();

        app.mount_all_dngs();

        let batch_before_dismiss = app.mount_all_dngs_batch.clone().expect("batch");
        assert_eq!(
            batch_before_dismiss.phase,
            MountAllDngsPhase::WaitingForUnmountAll
        );
        assert_eq!(batch_before_dismiss.current_entry_id, None);
        assert_eq!(batch_before_dismiss.total, 102);
        assert_eq!(batch_before_dismiss.pending.len(), 102);
        assert_eq!(
            batch_before_dismiss
                .pending
                .iter()
                .map(|request| request.entry_id)
                .collect::<Vec<_>>(),
            expected_order
        );
        assert_eq!(
            batch_before_dismiss.dng_flags,
            DngCliFlags {
                decode: DecodeMode::Cpu,
                vignette: false,
                optimizer_profile: OptimizerProfile::Optimized,
            }
        );
        assert_eq!(app.playlist.selected_index(), Some(64));
        assert!(app.dng_processes.has_active_transient_action());
        assert_eq!(
            app.macos_mount_limit_warning,
            cfg!(target_os = "macos").then_some(MacosMountLimitWarning {
                playlist_count: 102,
            })
        );

        app.dismiss_macos_mount_limit_warning();

        assert_eq!(
            app.mount_all_dngs_batch.as_ref(),
            Some(&batch_before_dismiss)
        );
        assert_eq!(app.playlist.selected_index(), Some(64));
        assert!(app.dng_processes.has_active_transient_action());

        app.record_macos_mount_limit_warning(true, 102);
        assert_eq!(
            app.macos_mount_limit_warning,
            Some(MacosMountLimitWarning {
                playlist_count: 102,
            })
        );
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn dng_mount_paths_remain_singular_and_share_the_per_entry_primitive() {
        let source = include_str!("app.rs");
        let mount_all_definition = ["fn mount_all_dngs", "(&mut self)"].concat();
        let per_entry_definition = ["fn start_dng_mount", "("].concat();
        let selected_mount =
            source_between(source, "fn mount_selected_dng", "fn unmount_selected_dng");
        let mount_all = source_between(
            source,
            &mount_all_definition,
            "fn record_macos_mount_limit_warning",
        );
        let next_mount = source_between(
            source,
            "fn start_next_mount_all_dng",
            "fn take_next_mount_all_request",
        );
        let warning_window = source_between(
            source,
            "fn draw_macos_mount_limit_warning",
            "fn mount_all_waiting_for_unmount_all",
        );
        let per_entry_mount = source_between(
            source,
            "fn start_dng_mount",
            "fn duplicate_live_source_exists",
        );

        assert_eq!(source.matches(&mount_all_definition).count(), 1);
        assert_eq!(source.matches(&per_entry_definition).count(), 1);
        assert_eq!(selected_mount.matches("self.start_dng_mount(").count(), 1);
        assert_eq!(next_mount.matches("self.start_dng_mount(").count(), 1);
        assert_eq!(
            mount_all
                .matches("start_unmount_all_dngs(UnmountAllDngsOrigin::MountAllPreparation)")
                .count(),
            1
        );
        assert_eq!(
            mount_all
                .matches("self.record_macos_mount_limit_warning(")
                .count(),
            1
        );
        assert!(!mount_all.contains(".take("));
        assert!(!mount_all.contains(".truncate("));
        assert_eq!(per_entry_mount.matches("DngCommandSpec::mount(").count(), 1);
        assert_eq!(per_entry_mount.matches(".start_mount(").count(), 1);
        assert!(warning_window.contains(".anchor(Align2::CENTER_CENTER, Vec2::ZERO)"));
        assert!(warning_window.contains(".resizable(false)"));
        assert!(warning_window.contains(".collapsible(false)"));
        assert!(warning_window.contains(".open(&mut open)"));
        assert!(warning_window.contains("ui.button(MACOS_MOUNT_LIMIT_WARNING_BUTTON)"));
        let alternate_backend_name = ["FS", "Kit"].concat();
        let shared_root_name = ["Shared", "Root"].concat();
        assert!(!source.contains(&alternate_backend_name));
        assert!(!source.contains(&shared_root_name));
    }

    #[test]
    fn selected_mount_dng_uses_reusable_per_entry_primitive() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.dng_binary = PathBuf::from("true");
        let path = existing_mcraw_file("selected-mount", "selected.mcraw");
        app.add_dropped_paths([path]);
        app.playlist.select(0);
        let entry_id = app.playlist.selected_id().expect("selected id");

        app.mount_selected_dng();

        let entry = app.playlist.entry_by_id(entry_id).expect("entry retained");
        assert_eq!(entry.desired_mount_state, DesiredMountState::Mounted);
        assert_eq!(entry.live_mount_state, LiveMountState::Mounting);
        assert!(app.dng_processes.has_mount_for_entry(entry_id));
        assert_eq!(
            app.status_message.as_deref(),
            Some("Mounting DNG: selected")
        );
        assert!(app.macos_mount_limit_warning.is_none());
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn selected_mount_dng_preserves_missing_source_validation() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/missing.mcraw")]);
        app.playlist.select(0);

        app.mount_selected_dng();

        assert_eq!(
            app.playlist.entries()[0].desired_mount_state,
            DesiredMountState::Unmounted
        );
        assert_eq!(
            app.status_message.as_deref(),
            Some("Selected file no longer exists.")
        );
    }

    #[test]
    fn selected_mount_dng_preserves_already_mounted_and_in_progress_validation() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.dng_binary = PathBuf::from("true");
        let path = existing_mcraw_file("selected-mount-validation", "selected.mcraw");
        app.add_dropped_paths([path]);
        app.playlist.select(0);
        let entry_id = app.playlist.selected_id().expect("selected id");

        assert!(
            app.playlist
                .set_live_mount_state(entry_id, LiveMountState::MountedThisSession)
        );
        app.mount_selected_dng();
        assert_eq!(
            app.status_message.as_deref(),
            Some("DNG is already mounted for this row.")
        );

        assert!(
            app.playlist
                .set_live_mount_state(entry_id, LiveMountState::NotLive)
        );
        app.mount_selected_dng();
        app.mount_selected_dng();
        assert_eq!(
            app.status_message.as_deref(),
            Some("DNG is already mounted for this row.")
        );
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn selected_unmount_dng_preserves_existing_state_and_process_start() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.dng_binary = PathBuf::from("true");
        let path = existing_mcraw_file("selected-unmount", "selected.mcraw");
        app.add_dropped_paths([path]);
        app.playlist.select(0);
        let entry_id = app.playlist.selected_id().expect("selected id");
        assert!(
            app.playlist
                .set_selected_desired_mount_state(DesiredMountState::Mounted)
        );
        assert!(
            app.playlist
                .set_live_mount_state(entry_id, LiveMountState::MountedThisSession)
        );

        app.unmount_selected_dng();

        let entry = app.playlist.entry_by_id(entry_id).expect("entry retained");
        assert_eq!(entry.desired_mount_state, DesiredMountState::Unmounted);
        assert_eq!(entry.live_mount_state, LiveMountState::Unmounting);
        assert_eq!(
            app.status_message.as_deref(),
            Some("Unmounting DNG: selected")
        );
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn standalone_unmount_all_reuses_shared_boundary_and_persists_unmounted_intent() {
        let (mut app, playlist_path) = app_with_playlist_store("standalone-unmount-all");
        app.dng_binary = PathBuf::from("true");
        let path = existing_mcraw_file("standalone-unmount-all-file", "first.mcraw");
        app.add_dropped_paths([path]);
        let entry_id = app.playlist.entries()[0].id;
        app.playlist.select(0);
        assert!(
            app.playlist
                .set_selected_desired_mount_state(DesiredMountState::Mounted)
        );
        assert!(
            app.playlist
                .set_live_mount_state(entry_id, LiveMountState::MountedThisSession)
        );

        app.unmount_all_dngs();

        let entry = app.playlist.entry_by_id(entry_id).expect("entry retained");
        assert_eq!(entry.desired_mount_state, DesiredMountState::Unmounted);
        assert_eq!(entry.live_mount_state, LiveMountState::Unmounting);
        assert_eq!(app.status_message.as_deref(), Some("Unmounting all DNGs."));
        let saved = std::fs::read_to_string(playlist_path).expect("playlist saved");
        assert!(saved.contains("\"desired_mount_state\": \"unmounted\""));
        assert!(app.mount_all_dngs_batch.is_none());
        assert!(app.macos_mount_limit_warning.is_none());
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn mount_all_dngs_rejects_empty_playlist_without_batch() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        app.mount_all_dngs();

        assert!(app.mount_all_dngs_batch.is_none());
        assert!(app.macos_mount_limit_warning.is_none());
        assert_eq!(
            app.status_message.as_deref(),
            Some("Add playlist files before mounting all DNGs.")
        );
    }

    #[test]
    fn mount_all_dngs_snapshots_playlist_order_selection_and_dng_flags() {
        let (mut app, playlist_path) = app_with_playlist_store("mount-all-snapshot");
        app.dng_binary = PathBuf::from("true");
        let files = existing_mcraw_files(
            "mount-all-snapshot-files",
            &["first.mcraw", "second.mcraw", "third.mcraw"],
        );
        app.add_dropped_paths(files);
        app.playlist.select(1);
        app.settings.select_decode_mode(DecodeMode::Cpu);
        app.settings.toggle_dng_vignette();
        app.settings
            .select_optimizer_profile(OptimizerProfile::Optimized);

        app.mount_all_dngs();

        assert_eq!(app.playlist.selected_index(), Some(1));
        let batch = app.mount_all_dngs_batch.as_ref().expect("batch");
        assert_eq!(batch.phase, MountAllDngsPhase::WaitingForUnmountAll);
        assert_eq!(batch.current_entry_id, None);
        assert_eq!(batch.total, 3);
        assert_eq!(
            batch
                .pending
                .iter()
                .map(|request| request.display_name.as_str())
                .collect::<Vec<_>>(),
            ["first", "second", "third"]
        );
        assert_eq!(
            batch.dng_flags,
            DngCliFlags {
                decode: DecodeMode::Cpu,
                vignette: false,
                optimizer_profile: OptimizerProfile::Optimized,
            }
        );
        assert_eq!(
            app.status_message.as_deref(),
            Some("Preparing to mount all DNGs...")
        );
        let saved = std::fs::read_to_string(playlist_path).expect("playlist saved");
        assert!(saved.contains("\"desired_mount_state\": \"unmounted\""));
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn mount_all_dngs_duplicate_request_is_rejected_while_active() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.dng_binary = PathBuf::from("true");
        let path = existing_mcraw_file("mount-all-duplicate", "first.mcraw");
        app.add_dropped_paths([path]);
        app.mount_all_dngs();

        app.mount_all_dngs();

        assert!(app.mount_all_dngs_batch.is_some());
        assert_eq!(
            app.status_message.as_deref(),
            Some("Mount all DNGs is already running.")
        );
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn mount_all_dngs_unmount_failure_aborts_without_starting_mounts() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.dng_binary = PathBuf::from("true");
        let files =
            existing_mcraw_files("mount-all-unmount-fail", &["first.mcraw", "second.mcraw"]);
        app.add_dropped_paths(files);
        let first_id = app.playlist.entries()[0].id;

        app.mount_all_dngs();
        app.handle_dng_process_event(DngProcessEvent::ActionExited {
            kind: DngActionKind::UnmountAll,
            success: false,
            status: "forced failure".to_string(),
        });

        assert!(app.mount_all_dngs_batch.is_none());
        assert!(!app.dng_processes.has_mount_for_entry(first_id));
        assert!(
            app.playlist
                .entries()
                .iter()
                .all(|entry| entry.desired_mount_state == DesiredMountState::Unmounted)
        );
        assert_eq!(
            app.status_message.as_deref(),
            Some("Mount all DNGs stopped because Unmount all DNGs failed.")
        );
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn mount_all_dngs_waits_for_unmount_success_before_first_mount() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.dng_binary = PathBuf::from("true");
        let files = existing_mcraw_files("mount-all-waits", &["first.mcraw", "second.mcraw"]);
        app.add_dropped_paths(files);
        let first_id = app.playlist.entries()[0].id;

        app.mount_all_dngs();
        assert_eq!(app.mount_all_current_entry_id(), None);
        assert!(!app.dng_processes.has_mount_for_entry(first_id));

        app.handle_dng_process_event(DngProcessEvent::ActionExited {
            kind: DngActionKind::UnmountAll,
            success: true,
            status: "ok".to_string(),
        });

        assert_eq!(app.mount_all_current_entry_id(), Some(first_id));
        assert!(app.dng_processes.has_mount_for_entry(first_id));
        assert_eq!(
            app.status_message.as_deref(),
            Some("Mounting DNG 1 of 2...")
        );
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn mount_all_dngs_continues_after_entry_failure_and_preserves_selection() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.dng_binary = PathBuf::from("true");
        let files = existing_mcraw_files(
            "mount-all-mixed",
            &["first.mcraw", "second.mcraw", "third.mcraw"],
        );
        app.add_dropped_paths(files);
        app.playlist.select(2);
        let first_id = app.playlist.entries()[0].id;
        let second_id = app.playlist.entries()[1].id;
        let third_id = app.playlist.entries()[2].id;

        app.mount_all_dngs();
        app.handle_dng_process_event(DngProcessEvent::ActionExited {
            kind: DngActionKind::UnmountAll,
            success: true,
            status: "ok".to_string(),
        });
        assert_eq!(app.mount_all_current_entry_id(), Some(first_id));

        app.handle_dng_process_event(DngProcessEvent::MountBecameActive {
            entry_id: first_id,
            display_name: "first".to_string(),
            mount_path: None,
        });
        assert_eq!(app.mount_all_current_entry_id(), Some(second_id));

        app.handle_dng_process_event(DngProcessEvent::MountExited {
            entry_id: second_id,
            display_name: "second".to_string(),
            active: false,
            stopping: false,
            success: false,
            status: "exit status: 1".to_string(),
        });
        assert_eq!(app.mount_all_current_entry_id(), Some(third_id));

        app.handle_dng_process_event(DngProcessEvent::MountBecameActive {
            entry_id: third_id,
            display_name: "third".to_string(),
            mount_path: None,
        });

        assert!(app.mount_all_dngs_batch.is_none());
        assert_eq!(app.playlist.selected_index(), Some(2));
        assert_eq!(
            app.status_message.as_deref(),
            Some("Mounted 2 of 3 DNGs; 1 failed.")
        );

        app.handle_dng_process_event(DngProcessEvent::MountExited {
            entry_id: first_id,
            display_name: "first".to_string(),
            active: true,
            stopping: false,
            success: true,
            status: "exit status: 0".to_string(),
        });
        assert!(app.mount_all_dngs_batch.is_none());
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn mount_all_dngs_all_success_and_all_failed_summaries_are_correct() {
        let mut success_app =
            GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        success_app.dng_binary = PathBuf::from("true");
        let files = existing_mcraw_files("mount-all-success", &["first.mcraw", "second.mcraw"]);
        success_app.add_dropped_paths(files);
        let first_id = success_app.playlist.entries()[0].id;
        let second_id = success_app.playlist.entries()[1].id;
        success_app.mount_all_dngs();
        success_app.handle_dng_process_event(DngProcessEvent::ActionExited {
            kind: DngActionKind::UnmountAll,
            success: true,
            status: "ok".to_string(),
        });
        success_app.handle_dng_process_event(DngProcessEvent::MountBecameActive {
            entry_id: first_id,
            display_name: "first".to_string(),
            mount_path: None,
        });
        success_app.handle_dng_process_event(DngProcessEvent::MountBecameActive {
            entry_id: second_id,
            display_name: "second".to_string(),
            mount_path: None,
        });
        assert_eq!(
            success_app.status_message.as_deref(),
            Some("Mounted all 2 DNGs.")
        );

        let mut failed_app =
            GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        failed_app.dng_binary = PathBuf::from("true");
        failed_app.add_dropped_paths([
            PathBuf::from("clips/missing-one.mcraw"),
            PathBuf::from("clips/missing-two.mcraw"),
        ]);
        failed_app.mount_all_dngs();
        failed_app.handle_dng_process_event(DngProcessEvent::ActionExited {
            kind: DngActionKind::UnmountAll,
            success: true,
            status: "ok".to_string(),
        });
        assert_eq!(
            failed_app.status_message.as_deref(),
            Some("Could not mount any of 2 DNGs.")
        );
        assert!(
            failed_app
                .playlist
                .entries()
                .iter()
                .all(|entry| entry.desired_mount_state == DesiredMountState::Unmounted)
        );
        let _ = success_app.dng_processes.poll();
        let _ = failed_app.dng_processes.poll();
    }

    #[test]
    fn mount_all_dngs_missing_source_failure_continues_without_fake_mount_write() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.dng_binary = PathBuf::from("true");
        let valid = existing_mcraw_file("mount-all-missing-valid", "valid.mcraw");
        app.add_dropped_paths([PathBuf::from("clips/missing.mcraw"), valid]);
        let missing_id = app.playlist.entries()[0].id;
        let valid_id = app.playlist.entries()[1].id;

        app.mount_all_dngs();
        app.handle_dng_process_event(DngProcessEvent::ActionExited {
            kind: DngActionKind::UnmountAll,
            success: true,
            status: "ok".to_string(),
        });

        assert_eq!(app.mount_all_current_entry_id(), Some(valid_id));
        assert_eq!(
            app.playlist
                .entry_by_id(missing_id)
                .unwrap()
                .desired_mount_state,
            DesiredMountState::Unmounted
        );
        assert_eq!(
            app.playlist
                .entry_by_id(valid_id)
                .unwrap()
                .desired_mount_state,
            DesiredMountState::Mounted
        );
        let batch = app.mount_all_dngs_batch.as_ref().expect("batch");
        assert_eq!(batch.failures.len(), 1);
        assert_eq!(batch.failures[0].entry_id, missing_id);
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn mount_all_dngs_running_batch_uses_settings_snapshot() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.dng_binary = PathBuf::from("true");
        let files = existing_mcraw_files("mount-all-settings", &["first.mcraw", "second.mcraw"]);
        app.add_dropped_paths(files);
        app.settings.select_decode_mode(DecodeMode::Cpu);
        app.settings.toggle_dng_vignette();
        app.settings
            .select_optimizer_profile(OptimizerProfile::Optimized);

        app.mount_all_dngs();
        app.settings.select_decode_mode(DecodeMode::Gpu);
        app.settings.toggle_dng_vignette();
        app.settings
            .select_optimizer_profile(OptimizerProfile::Default);

        assert_eq!(
            app.mount_all_dngs_batch.as_ref().unwrap().dng_flags,
            DngCliFlags {
                decode: DecodeMode::Cpu,
                vignette: false,
                optimizer_profile: OptimizerProfile::Optimized,
            }
        );
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn mount_all_dngs_disables_conflicting_controls_but_keeps_more_options_available() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.dng_binary = PathBuf::from("true");
        let path = existing_mcraw_file("mount-all-controls", "first.mcraw");
        app.add_dropped_paths([path]);

        assert!(app.playlist_mutation_controls_enabled());
        assert!(app.preview_start_enabled());
        assert!(app.dng_action_controls_enabled());
        assert!(app.mount_all_dngs_button_enabled());

        app.mount_all_dngs();

        assert!(!app.playlist_mutation_controls_enabled());
        assert!(!app.preview_start_enabled());
        assert!(!app.dng_action_controls_enabled());
        assert!(!app.mount_all_dngs_button_enabled());
        app.toggle_more_options();
        assert!(app.more_options_visible);
        app.remove_all_playlist_entries();
        assert_eq!(app.playlist.len(), 1);
        assert_eq!(
            app.status_message.as_deref(),
            Some("Wait for Mount all DNGs to finish.")
        );
        let _ = app.dng_processes.poll();
    }

    #[test]
    fn pipe_example_uses_shared_cpu_decode_mode() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/selected file.mcraw")]);
        app.playlist.select(0);
        app.settings.select_decode_mode(DecodeMode::Cpu);

        app.show_pipe_example_with_facts_source(|_| Ok(pipe_example_facts(4096)));

        let Some(PipeExamplePanel::Example(example)) = &app.pipe_example else {
            panic!("expected generated pipe example");
        };
        assert_eq!(example.decode_mode, DecodeMode::Cpu);
        assert!(example.command.contains("--cpu"));
        assert!(!example.command.contains("--gpu"));
    }

    #[test]
    fn pipe_example_and_playlist_scroll_ids_are_distinct() {
        assert_ne!(PIPE_EXAMPLE_SCROLL_ID, PLAYLIST_SCROLL_ID);
    }

    #[test]
    fn pipe_example_uses_dng_vignette_setting() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/selected file.mcraw")]);
        app.playlist.select(0);
        app.settings.toggle_dng_vignette();

        app.show_pipe_example_with_facts_source(|_| Ok(pipe_example_facts(4096)));

        let Some(PipeExamplePanel::Example(example)) = &app.pipe_example else {
            panic!("expected generated pipe example");
        };
        assert!(!example.vignette_correction);
        assert!(example.command.contains("--no-vig-correction"));
        assert!(!example.command.contains("--with-vig-correction"));
    }

    #[test]
    fn pipe_example_without_selection_preserves_current_pane_and_reports_status() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.pipe_example = Some(PipeExamplePanel::Message("existing".to_string()));
        app.right_pane_owner = RightPaneOwner::PipeExample;

        app.show_pipe_example_with_facts_source(|_| {
            panic!("facts should not be requested without selection")
        });

        assert_eq!(
            app.pipe_example,
            Some(PipeExamplePanel::Message("existing".to_string()))
        );
        assert_eq!(app.right_pane_owner, RightPaneOwner::PipeExample);
        assert_eq!(
            app.status_message.as_deref(),
            Some(pipe_example::NO_SELECTED_FILE_MESSAGE)
        );
    }

    #[test]
    fn add_files_chooser_selection_uses_playlist_filtering() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert!(
            app.apply_file_chooser_outcome(FileChooserOutcome::Selected(vec![
                PathBuf::from("clips/chooser one.mcraw"),
                PathBuf::from("clips/reject.txt"),
                PathBuf::from("clips/chooser two.MCRAW"),
            ]))
        );

        assert_eq!(app.playlist.len(), 2);
        assert_eq!(app.playlist.entries()[0].display_name, "chooser one");
        assert_eq!(app.playlist.entries()[1].display_name, "chooser two");
        assert_eq!(
            app.status_message.as_deref(),
            Some("Added 2 file(s) to playlist. Ignored 1 non-mcraw file(s).")
        );
    }

    #[test]
    fn add_files_pending_chooser_starts_without_mutating_playlist() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert!(
            app.apply_file_chooser_start_for_test(FileChooserStart::Pending(
                file_chooser::PendingFileChooser::test_pending()
            ))
        );

        assert!(app.pending_file_chooser.is_some());
        assert!(app.playlist.is_empty());
        assert_eq!(app.status_message.as_deref(), Some("File chooser open..."));
    }

    #[test]
    fn add_files_click_while_pending_reports_already_open() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.apply_file_chooser_start_for_test(FileChooserStart::Pending(
            file_chooser::PendingFileChooser::test_pending(),
        ));

        assert!(app.add_files_from_chooser());

        assert!(app.pending_file_chooser.is_some());
        assert!(app.playlist.is_empty());
        assert_eq!(
            app.status_message.as_deref(),
            Some("File chooser is already open.")
        );
    }

    #[test]
    fn pending_chooser_poll_does_not_mutate_playlist_until_ready() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.apply_file_chooser_start_for_test(FileChooserStart::Pending(
            file_chooser::PendingFileChooser::test_pending(),
        ));

        assert!(!app.poll_file_chooser());

        assert!(app.pending_file_chooser.is_some());
        assert!(app.playlist.is_empty());
        assert_eq!(app.status_message.as_deref(), Some("File chooser open..."));
    }

    #[test]
    fn completed_chooser_selection_preserves_order() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert!(
            app.apply_file_chooser_start_for_test(FileChooserStart::Ready(
                FileChooserOutcome::Selected(vec![
                    PathBuf::from("clips/first.mcraw"),
                    PathBuf::from("clips/second.mcraw"),
                    PathBuf::from("clips/third.MCRAW"),
                ])
            ))
        );

        let names: Vec<&str> = app
            .playlist
            .entries()
            .iter()
            .map(|entry| entry.display_name.as_str())
            .collect();
        assert_eq!(names, vec!["first", "second", "third"]);
        assert_eq!(
            app.status_message.as_deref(),
            Some("Added 3 file(s) to playlist.")
        );
    }

    #[test]
    fn add_files_cancel_leaves_playlist_unchanged() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/existing.mcraw")]);

        assert!(app.apply_file_chooser_outcome(FileChooserOutcome::Cancelled));

        assert_eq!(app.playlist.len(), 1);
        assert_eq!(app.playlist.entries()[0].display_name, "existing");
        assert_eq!(
            app.status_message.as_deref(),
            Some("File chooser cancelled.")
        );
    }

    #[test]
    fn add_files_unavailable_reports_drag_drop_fallback() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        assert!(
            app.apply_file_chooser_outcome(FileChooserOutcome::Unavailable(
                "File chooser unavailable. Use drag and drop.".to_string()
            ))
        );

        assert!(app.playlist.is_empty());
        assert_eq!(
            app.status_message.as_deref(),
            Some("File chooser unavailable. Use drag and drop.")
        );
    }

    #[test]
    fn pipe_example_simple_command_layout_uses_blue_pipe_and_white_paths() {
        let layout = pipe_example_simple_command_layout(pipe_example::SIMPLE_COMMAND);
        let pipe_end = pipe_example::SIMPLE_COMMAND_PIPE.len();

        assert_eq!(layout.text, pipe_example::SIMPLE_COMMAND);
        assert_eq!(layout.sections.len(), 2);
        assert_eq!(layout.sections[0].byte_range, 0..pipe_end);
        assert_eq!(layout.sections[0].format.color, style::bright_blue());
        assert_eq!(
            layout.sections[1].byte_range,
            pipe_end..pipe_example::SIMPLE_COMMAND.len()
        );
        assert_eq!(layout.sections[1].format.color, style::header_text());
    }

    #[test]
    fn pipe_example_command_copy_contains_exact_one_line_command() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/copy me.mcraw")]);
        app.playlist.select(0);
        app.show_pipe_example_with_facts_source(|_| Ok(pipe_example_facts(4096)));
        let Some(PipeExamplePanel::Example(example)) = &app.pipe_example else {
            panic!("expected generated pipe example");
        };
        let expected_command = example.command.clone();
        let simple_command = example.simple_command.clone();
        let pipe_invocation = if cfg!(target_os = "windows") {
            "mcraw4vulkan.exe pipe"
        } else {
            "mcraw4vulkan pipe"
        };

        let copied = app
            .copy_pipe_command_with(|command| {
                assert!(!command.contains('\n'));
                assert!(!command.contains('\r'));
                assert!(command.contains(pipe_invocation));
                assert!(!command.contains(pipe_example::BODY));
                assert!(!command.contains(&simple_command));
                assert_eq!(command, expected_command);
                Ok(())
            })
            .expect("command copied");

        assert!(!copied.contains('\n'));
        assert!(!copied.contains('\r'));
        assert!(!copied.contains(pipe_example::HEADER));
        assert!(!copied.contains(pipe_example::BODY));
        assert!(!copied.contains(&simple_command));
        assert_eq!(copied, expected_command);
        assert_eq!(app.status_message.as_deref(), Some("Pipe command copied."));
    }

    #[test]
    fn copy_command_without_generated_command_reports_status() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());

        let copied = app.copy_pipe_command_with(|_| {
            panic!("copy must not be attempted without a generated command")
        });

        assert_eq!(copied, None);
        assert_eq!(
            app.status_message.as_deref(),
            Some("No pipe command to copy.")
        );
    }

    #[test]
    fn copy_command_failure_reports_status() {
        let mut app = GuiApp::new("Installed RAM unknown | Available RAM unknown".to_string());
        app.add_dropped_paths([PathBuf::from("clips/copy me.mcraw")]);
        app.playlist.select(0);
        app.show_pipe_example_with_facts_source(|_| Ok(pipe_example_facts(4096)));

        let copied = app.copy_pipe_command_with(|_| Err("clipboard unavailable".to_string()));

        assert!(copied.is_some());
        assert_eq!(
            app.status_message.as_deref(),
            Some("Clipboard copy failed: clipboard unavailable")
        );
    }

    #[test]
    fn splash_detail_colors_match_b10_rules() {
        assert_eq!(line_color(LineStatus::Available), style::bright_blue());
        assert_eq!(line_color(LineStatus::Missing), style::header_text());
        assert_eq!(line_color(LineStatus::Informational), style::body_text());
    }

    #[test]
    fn sdl_modifiers_map_ctrl_to_command_on_linux() {
        let modifiers = modifiers_from_sdl(Mod::LCTRLMOD | Mod::RSHIFTMOD);

        assert!(modifiers.ctrl);
        assert!(modifiers.command);
        assert!(modifiers.shift);
        assert!(!modifiers.mac_cmd);
    }

    #[test]
    fn mouse_buttons_map_to_egui_buttons() {
        assert_eq!(
            pointer_button_from_sdl(MouseButton::Left),
            Some(PointerButton::Primary)
        );
        assert_eq!(
            pointer_button_from_sdl(MouseButton::Right),
            Some(PointerButton::Secondary)
        );
    }

    #[test]
    fn wheel_delta_uses_precise_values_when_available() {
        assert_eq!(
            wheel_delta(0, 0, 0.5, 1.0, MouseWheelDirection::Normal),
            vec2(20.0, -40.0)
        );
    }
}
