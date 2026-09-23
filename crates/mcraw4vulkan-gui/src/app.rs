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

#[cfg_attr(test, allow(unreachable_code))]
fn default_playlist_store() -> PlaylistStore {
    PlaylistStore::from_default_config()
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
            .wrap(),
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
