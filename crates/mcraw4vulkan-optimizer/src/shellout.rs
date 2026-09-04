use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::decision::{
    LOW_RAM_THRESHOLD_BYTES, LowRamGate, PayloadProfileDecision, RecommendationReason, ScoredSink,
    ThroughputMedians, recommend_payload_profile,
};
use crate::optimized_state::{
    OptimizedState, OptimizedStateError, PayloadProfile, save_optimized_state,
};
use serde_json::Value;

pub const OPTIMIZER_DECISION_FRAMES: usize = 600;
pub const OPTIMIZER_GPU_WARMUP_FRAMES: usize = 600;
pub const OPTIMIZER_REPETITIONS: usize = 3;
pub const OPTIMIZER_HARD_TIMEOUT: Duration = Duration::from_secs(600);
pub const OPTIMIZER_MINIMUM_FRAMES_MESSAGE: &str =
    "Optimization testing requires at least 600 frames for accuracy.";
const DISPLAY_WARMUP_FRAMES: usize = 30;
const KEEP_REPORTS_ENV: &str = "MCRAW4VULKAN_KEEP_OPTIMIZER_REPORTS";

pub type OptimizerResult<T> = Result<T, OptimizerError>;

#[derive(Debug)]
pub enum OptimizerError {
    Io(io::Error),
    Json(serde_json::Error),
    InsufficientFrames,
    ExecutableNotFound(PathBuf),
    InvalidReport {
        measurement_id: String,
        message: String,
    },
    State(OptimizedStateError),
}

impl fmt::Display for OptimizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::InsufficientFrames => write!(formatter, "{OPTIMIZER_MINIMUM_FRAMES_MESSAGE}"),
            Self::ExecutableNotFound(path) => {
                write!(
                    formatter,
                    "mcraw4vulkan executable not found: {}",
                    path.display()
                )
            }
            Self::InvalidReport {
                measurement_id,
                message,
            } => {
                write!(
                    formatter,
                    "optimizer measurement report {measurement_id} is invalid: {message}"
                )
            }
            Self::State(error) => write!(formatter, "{error}"),
        }
    }
}

pub fn validate_optimizer_frame_count(frame_count: usize) -> OptimizerResult<()> {
    if frame_count < OPTIMIZER_DECISION_FRAMES {
        Err(OptimizerError::InsufficientFrames)
    } else {
        Ok(())
    }
}

impl Error for OptimizerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::State(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for OptimizerError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for OptimizerError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<OptimizedStateError> for OptimizerError {
    fn from(error: OptimizedStateError) -> Self {
        Self::State(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizerRunConfig {
    pub input_path: PathBuf,
    pub mcraw4vulkan_exe: PathBuf,
    pub frames: usize,
    pub keep_reports: bool,
    pub hard_timeout: Duration,
    pub total_ram_bytes: Option<u64>,
}

impl OptimizerRunConfig {
    pub fn user_facing(
        input_path: PathBuf,
        mcraw4vulkan_exe: PathBuf,
        keep_reports: bool,
        total_ram_bytes: Option<u64>,
    ) -> Self {
        Self {
            input_path,
            mcraw4vulkan_exe,
            frames: OPTIMIZER_DECISION_FRAMES,
            keep_reports,
            hard_timeout: OPTIMIZER_HARD_TIMEOUT,
            total_ram_bytes,
        }
    }

    pub fn keep_reports_from_env() -> bool {
        std::env::var(KEEP_REPORTS_ENV)
            .map(|value| value == "1")
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecommendedPayloadProfile {
    DefaultChunked64,
    OffsetPrefetch,
}

impl RecommendedPayloadProfile {
    pub fn summary_label(self) -> &'static str {
        match self {
            Self::DefaultChunked64 => "default_chunked64",
            Self::OffsetPrefetch => "offset_prefetch",
        }
    }

    fn result_label(self) -> &'static str {
        match self {
            Self::DefaultChunked64 => "Default payload profile remains selected",
            Self::OffsetPrefetch => "Offset payload profile is recommended",
        }
    }

    fn state_profile(self) -> PayloadProfile {
        match self {
            Self::DefaultChunked64 => PayloadProfile::DefaultChunked64,
            Self::OffsetPrefetch => PayloadProfile::OffsetPrefetch,
        }
    }

    fn from_state_profile(profile: PayloadProfile) -> Self {
        match profile {
            PayloadProfile::DefaultChunked64 => Self::DefaultChunked64,
            PayloadProfile::OffsetPrefetch => Self::OffsetPrefetch,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizerRecommendation {
    pub payload_profile: RecommendedPayloadProfile,
    pub decision_quality_ok: bool,
    pub settings_written: bool,
    pub reason: RecommendationReason,
    pub reason_detail: Option<String>,
}

impl OptimizerRecommendation {
    pub fn is_built_in_default(&self) -> bool {
        self.payload_profile == RecommendedPayloadProfile::DefaultChunked64
    }

    pub fn optimized_state(&self) -> OptimizedState {
        OptimizedState::new(self.payload_profile.state_profile())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerRunOutcome {
    pub input_basename: String,
    pub frames: usize,
    pub progress_total_steps: usize,
    pub measurements: OptimizerMeasurementSet,
    pub recommendation: OptimizerRecommendation,
    pub selection: SelectionSummary,
}

pub fn run_optimizer(config: OptimizerRunConfig) -> OptimizerResult<OptimizerRecommendation> {
    let outcome = OptimizerShelloutRunner::new(config).run()?;
    OptimizerSummaryPrinter::print(&outcome);
    prompt_and_maybe_save_recommendation(&outcome.recommendation)
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerShelloutRunner {
    config: OptimizerRunConfig,
}

impl OptimizerShelloutRunner {
    pub fn new(config: OptimizerRunConfig) -> Self {
        Self { config }
    }

    pub fn run(&self) -> OptimizerResult<OptimizerRunOutcome> {
        if !self.config.mcraw4vulkan_exe.exists() {
            return Err(OptimizerError::ExecutableNotFound(
                self.config.mcraw4vulkan_exe.clone(),
            ));
        }
        let report_dir = create_optimizer_temp_report_dir()?;
        match self.run_with_report_dir(&report_dir) {
            Ok(outcome) => {
                finish_report_dir(&report_dir, self.config.keep_reports, true)?;
                Ok(outcome)
            }
            Err(error) => {
                let _ = finish_report_dir(&report_dir, self.config.keep_reports, false);
                Err(error)
            }
        }
    }

    fn run_with_report_dir(&self, report_dir: &Path) -> OptimizerResult<OptimizerRunOutcome> {
        // One deadline covers warmup and every scored row, so no later child can
        // acquire a fresh timeout after earlier work consumed the run budget.
        let deadline = Instant::now() + self.config.hard_timeout;
        let commands =
            optimizer_measurement_commands(&self.config.input_path, report_dir, self.config.frames);
        let progress_total_steps = optimizer_progress_total_steps(&commands);

        let warmup = optimizer_gpu_warmup_command(&self.config.input_path, report_dir);
        let _ = self.run_child_measurement_report(&warmup, deadline);
        let _ = fs::remove_file(&warmup.report_path);
        print_progress_step(1, progress_total_steps, "gpu warmup complete");

        let mut rows = Vec::with_capacity(commands.len());
        for (index, command) in commands.iter().enumerate() {
            if Instant::now() >= deadline {
                rows.push(command.timeout_row("optimizer deadline reached before row started"));
                break;
            }

            rows.push(match self.run_child_measurement_report(command, deadline) {
                Ok(report) => CandidateMeasurementRow::from_report(command, &report),
                Err(error) => command.failure_row(error.to_string()),
            });
            print_progress_step(index + 2, progress_total_steps, command.id.label());
        }

        let measurements = OptimizerMeasurementSet::from_rows(rows);
        let selection =
            OptimizerRecommendationCalculator::select(&measurements, self.config.total_ram_bytes);
        let recommendation = selection.recommendation();
        Ok(OptimizerRunOutcome {
            input_basename: input_basename(&self.config.input_path),
            frames: self.config.frames,
            progress_total_steps,
            measurements,
            recommendation,
            selection,
        })
    }

    fn run_child_measurement_report(
        &self,
        command: &ChildMeasurementCommand,
        deadline: Instant,
    ) -> OptimizerResult<MeasurementReport> {
        run_child_measurement(
            &self.config.mcraw4vulkan_exe,
            command,
            &self.config.input_path,
            deadline,
        )?;
        let report = MeasurementReport::from_file(&command.report_path)
            .map_err(|error| invalid_report(command.id.label(), error.to_string()))?;
        validate_report_for_spec(&report, command)?;
        Ok(report)
    }
}

fn run_child_measurement(
    exe: &Path,
    command: &ChildMeasurementCommand,
    input_path: &Path,
    deadline: Instant,
) -> OptimizerResult<()> {
    let mut child_command = Command::new(exe);
    child_command
        .args(&command.argv)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(temp_dir) = command.report_path.parent() {
        child_command
            .env("TMPDIR", temp_dir)
            .env("TEMP", temp_dir)
            .env("TMP", temp_dir);
    }
    let mut child = child_command.spawn()?;
    loop {
        if child.try_wait()?.is_some() {
            let output = child.wait_with_output()?;
            if output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if stderr_mentions_warning(&stderr) {
                    eprintln!(
                        "optimizer measurement {} warning output:\n{}",
                        command.id.label(),
                        stderr_tail(&sanitize_for_input(&stderr, input_path))
                    );
                }
                return Ok(());
            }
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(invalid_report(
                command.id.label(),
                format!(
                    "child failed with status {:?}: {}",
                    output.status.code(),
                    stderr_tail(&sanitize_for_input(&stderr, input_path))
                ),
            ));
        }

        if Instant::now() >= deadline {
            // The runner owns exactly one child here: kill requests termination,
            // then wait_with_output reaps it before the timeout error escapes.
            let _ = child.kill();
            let output = child.wait_with_output()?;
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(invalid_report(
                command.id.label(),
                format!(
                    "timed out at {} seconds: {}",
                    OPTIMIZER_HARD_TIMEOUT.as_secs(),
                    stderr_tail(&sanitize_for_input(&stderr, input_path))
                ),
            ));
        }
        thread::sleep(Duration::from_millis(20));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeasurementSink {
    Display,
    Pipe,
}

impl MeasurementSink {
    fn command_label(self) -> &'static str {
        match self {
            Self::Display => "display",
            Self::Pipe => "pipe",
        }
    }

    fn summary_label(self) -> &'static str {
        match self {
            Self::Display => "Display",
            Self::Pipe => "PIPE",
        }
    }

    fn report_stem(self) -> &'static str {
        match self {
            Self::Display => "display",
            Self::Pipe => "pipe",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeasurementId {
    GpuWarmupUnscored,
    Candidate {
        sink: MeasurementSink,
        profile: RecommendedPayloadProfile,
        repetition: usize,
    },
}

impl MeasurementId {
    pub fn label(self) -> String {
        match self {
            Self::GpuWarmupUnscored => "gpu_warmup_unscored".to_string(),
            Self::Candidate {
                sink,
                profile,
                repetition,
            } => format!(
                "{}_{}_rep{}",
                sink.report_stem(),
                profile.summary_label(),
                repetition
            ),
        }
    }

    fn report_file(self) -> String {
        match self {
            Self::GpuWarmupUnscored => "gpu-warmup.json".to_string(),
            Self::Candidate {
                sink,
                profile,
                repetition,
            } => format!(
                "{}-{}-rep{}.json",
                sink.report_stem(),
                profile.summary_label().replace('_', "-"),
                repetition
            ),
        }
    }
}

pub fn optimizer_progress_line(step: usize, total: usize, label: &str) -> String {
    format!("optimizer progress: step {step}/{total} {label}")
}

fn print_progress_step(step: usize, total: usize, label: impl AsRef<str>) {
    println!("{}", optimizer_progress_line(step, total, label.as_ref()));
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildMeasurementCommand {
    pub id: MeasurementId,
    pub argv: Vec<String>,
    pub report_path: PathBuf,
    sink: MeasurementSink,
    profile: RecommendedPayloadProfile,
    repetition: usize,
}

impl ChildMeasurementCommand {
    fn failure_row(&self, reason: String) -> CandidateMeasurementRow {
        CandidateMeasurementRow {
            sink: self.sink,
            profile: self.profile,
            repetition: self.repetition,
            frames_measured: 0,
            fps: 0.0,
            success: false,
            failure_reason: Some(reason),
        }
    }

    fn timeout_row(&self, reason: &str) -> CandidateMeasurementRow {
        self.failure_row(reason.to_string())
    }
}

pub fn optimizer_gpu_warmup_command(
    input_path: &Path,
    report_dir: &Path,
) -> ChildMeasurementCommand {
    display_command_with_warmup(
        MeasurementId::GpuWarmupUnscored,
        RecommendedPayloadProfile::DefaultChunked64,
        0,
        input_path,
        report_dir,
        OPTIMIZER_GPU_WARMUP_FRAMES,
        0,
    )
}

pub fn optimizer_measurement_commands(
    input_path: &Path,
    report_dir: &Path,
    frames: usize,
) -> Vec<ChildMeasurementCommand> {
    let mut commands = Vec::with_capacity(2 * 2 * OPTIMIZER_REPETITIONS);
    for repetition in 1..=OPTIMIZER_REPETITIONS {
        // The middle repetition reverses sink and profile order so one fixed
        // command position is not always associated with the same candidate.
        let reverse = repetition == 2;
        let sinks = if reverse {
            [MeasurementSink::Pipe, MeasurementSink::Display]
        } else {
            [MeasurementSink::Display, MeasurementSink::Pipe]
        };
        let profiles = if reverse {
            [
                RecommendedPayloadProfile::OffsetPrefetch,
                RecommendedPayloadProfile::DefaultChunked64,
            ]
        } else {
            [
                RecommendedPayloadProfile::DefaultChunked64,
                RecommendedPayloadProfile::OffsetPrefetch,
            ]
        };
        for sink in sinks {
            for profile in profiles {
                let id = MeasurementId::Candidate {
                    sink,
                    profile,
                    repetition,
                };
                commands.push(match sink {
                    MeasurementSink::Display => {
                        display_command(id, profile, repetition, input_path, report_dir, frames)
                    }
                    MeasurementSink::Pipe => {
                        pipe_command(id, profile, repetition, input_path, report_dir, frames)
                    }
                });
            }
        }
    }
    commands
}

pub fn optimizer_progress_total_steps(commands: &[ChildMeasurementCommand]) -> usize {
    commands.len() + 1
}

fn display_command(
    id: MeasurementId,
    profile: RecommendedPayloadProfile,
    repetition: usize,
    input_path: &Path,
    report_dir: &Path,
    frames: usize,
) -> ChildMeasurementCommand {
    display_command_with_warmup(
        id,
        profile,
        repetition,
        input_path,
        report_dir,
        frames,
        DISPLAY_WARMUP_FRAMES,
    )
}

fn display_command_with_warmup(
    id: MeasurementId,
    profile: RecommendedPayloadProfile,
    repetition: usize,
    input_path: &Path,
    report_dir: &Path,
    frames: usize,
    display_warmup_frames: usize,
) -> ChildMeasurementCommand {
    let report_path = report_dir.join(id.report_file());
    ChildMeasurementCommand {
        id,
        argv: vec![
            "display".to_string(),
            "--gpu".to_string(),
            "--no-vig-correction".to_string(),
            "--no-vsync".to_string(),
            "--default".to_string(),
            "--internal-measure".to_string(),
            "--internal-measure-report".to_string(),
            report_path.display().to_string(),
            "--internal-measure-frames".to_string(),
            frames.to_string(),
            "--internal-measure-warmup-frames".to_string(),
            display_warmup_frames.to_string(),
            "--internal-payload-profile".to_string(),
            profile.summary_label().to_string(),
            input_path.display().to_string(),
        ],
        report_path,
        sink: MeasurementSink::Display,
        profile,
        repetition,
    }
}

fn pipe_command(
    id: MeasurementId,
    profile: RecommendedPayloadProfile,
    repetition: usize,
    input_path: &Path,
    report_dir: &Path,
    frames: usize,
) -> ChildMeasurementCommand {
    let report_path = report_dir.join(id.report_file());
    ChildMeasurementCommand {
        id,
        argv: vec![
            "pipe".to_string(),
            "--gpu".to_string(),
            "--with-vig-correction".to_string(),
            "--default".to_string(),
            "--internal-measure".to_string(),
            "--internal-measure-report".to_string(),
            report_path.display().to_string(),
            "--internal-measure-frames".to_string(),
            frames.to_string(),
            "--internal-payload-profile".to_string(),
            profile.summary_label().to_string(),
            input_path.display().to_string(),
        ],
        report_path,
        sink: MeasurementSink::Pipe,
        profile,
        repetition,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateMeasurementRow {
    pub sink: MeasurementSink,
    pub profile: RecommendedPayloadProfile,
    pub repetition: usize,
    pub frames_measured: usize,
    pub fps: f64,
    pub success: bool,
    pub failure_reason: Option<String>,
}

impl CandidateMeasurementRow {
    fn from_report(command: &ChildMeasurementCommand, report: &MeasurementReport) -> Self {
        Self {
            sink: command.sink,
            profile: command.profile,
            repetition: command.repetition,
            frames_measured: report.frames_measured,
            fps: report.fps,
            success: true,
            failure_reason: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerMeasurementSet {
    pub rows: Vec<CandidateMeasurementRow>,
}

impl OptimizerMeasurementSet {
    pub fn from_rows(rows: Vec<CandidateMeasurementRow>) -> Self {
        Self { rows }
    }

    fn rows_for(
        &self,
        sink: MeasurementSink,
        profile: RecommendedPayloadProfile,
    ) -> Vec<&CandidateMeasurementRow> {
        self.rows
            .iter()
            .filter(|row| row.sink == sink && row.profile == profile)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementReport {
    pub measurement_report_version: u64,
    pub command: String,
    pub sink_name: String,
    pub status: String,
    pub error_message: Option<String>,
    pub input_basename: String,
    pub input_stem: String,
    pub input_path_recorded: bool,
    pub frames_requested: usize,
    pub frames_available: usize,
    pub frames_measured: usize,
    pub measurement_truncated_by_clip_length: bool,
    pub decision_quality_min_frames: usize,
    pub decision_quality_ok: bool,
    pub run_s: f64,
    pub fps: f64,
    pub total_fps: f64,
    pub payload_profile_requested: Option<String>,
    pub payload_profile_effective: bool,
    pub notes: Vec<String>,
    pub display: Option<DisplayReport>,
    pub pipe: Option<PipeReport>,
}

impl MeasurementReport {
    pub fn from_file(path: &Path) -> OptimizerResult<Self> {
        let text = fs::read_to_string(path)?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> OptimizerResult<Self> {
        let value: Value = serde_json::from_str(text)?;
        let object = value
            .as_object()
            .ok_or_else(|| parse_error("report root must be an object"))?;
        let display = match object.get("display") {
            Some(value) => Some(DisplayReport::parse(value)?),
            None => None,
        };
        let pipe = match object.get("pipe") {
            Some(value) => Some(PipeReport::parse(value)?),
            None => None,
        };
        Ok(Self {
            measurement_report_version: required_u64(object, "measurement_report_version")?,
            command: required_string(object, "command")?,
            sink_name: required_string(object, "sink_name")?,
            status: required_string(object, "status")?,
            error_message: optional_string(object, "error_message")?,
            input_basename: required_string(object, "input_basename")?,
            input_stem: required_string(object, "input_stem")?,
            input_path_recorded: required_bool(object, "input_path_recorded")?,
            frames_requested: required_usize(object, "frames_requested")?,
            frames_available: required_usize(object, "frames_available")?,
            frames_measured: required_usize(object, "frames_measured")?,
            measurement_truncated_by_clip_length: required_bool(
                object,
                "measurement_truncated_by_clip_length",
            )?,
            decision_quality_min_frames: required_usize(object, "decision_quality_min_frames")?,
            decision_quality_ok: required_bool(object, "decision_quality_ok")?,
            run_s: required_f64(object, "run_s")?,
            fps: required_f64(object, "fps")?,
            total_fps: required_f64(object, "total_fps")?,
            payload_profile_requested: optional_string(object, "payload_profile_requested")?,
            payload_profile_effective: required_bool(object, "payload_profile_effective")?,
            notes: optional_string_array(object, "notes")?,
            display,
            pipe,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DisplayReport {
    pub preview_width: Option<usize>,
    pub preview_height: Option<usize>,
    pub output_bytes: u64,
    pub full_frame_readback_performed: bool,
    pub live_window: bool,
}

impl DisplayReport {
    fn parse(value: &Value) -> OptimizerResult<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| parse_error("display object must be an object"))?;
        Ok(Self {
            preview_width: optional_usize(object, "preview_width")?,
            preview_height: optional_usize(object, "preview_height")?,
            output_bytes: required_u64(object, "output_bytes")?,
            full_frame_readback_performed: required_bool(object, "full_frame_readback_performed")?,
            live_window: required_bool(object, "live_window")?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PipeReport {
    pub pixel_format: String,
    pub plane_order: Vec<String>,
    pub endianness: String,
    pub bits_per_channel: usize,
    pub storage_bits_per_sample: usize,
    pub meaningful_bits_per_sample: usize,
    pub bit_alignment: String,
    pub upper_four_bits_zero: bool,
    pub sample_range: String,
    pub color_range: String,
    pub range_is_limited: bool,
    pub color_primaries: String,
    pub color_transfer: String,
    pub matrix_coefficients: String,
    pub chroma_sampling: String,
    pub alpha: String,
    pub final_code_bounds: Vec<u64>,
    pub linear_signal_scale_num: u64,
    pub linear_signal_scale_den: u64,
    pub linear_signal_scale_stops: i64,
    pub scale_application_point: String,
    pub expected_total_video_bytes: u64,
    pub output_target: String,
    pub stdout_used: bool,
    pub byte_clean_stdout: bool,
    pub render_mode: String,
    pub byte_identity_status: String,
    pub byte_identity_output_byte_count_ok: bool,
    pub byte_identity_frame_byte_count_ok: bool,
    pub byte_identity_plane_order_ok: bool,
    pub byte_identity_little_endian_ok: bool,
    pub byte_identity_meaningful_low_12_bits_ok: bool,
    pub byte_identity_code_bounds_ok: bool,
    pub byte_identity_mismatches: u64,
    pub pipe_full_byte_consume_in_run: bool,
    pub bytes_per_frame: u64,
    pub frame_width: usize,
    pub frame_height: usize,
}

impl PipeReport {
    fn parse(value: &Value) -> OptimizerResult<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| parse_error("pipe object must be an object"))?;
        Ok(Self {
            pixel_format: required_string(object, "pixel_format")?,
            plane_order: required_string_array(object, "plane_order")?,
            endianness: required_string(object, "endianness")?,
            bits_per_channel: required_usize(object, "bits_per_channel")?,
            storage_bits_per_sample: required_usize(object, "storage_bits_per_sample")?,
            meaningful_bits_per_sample: required_usize(object, "meaningful_bits_per_sample")?,
            bit_alignment: required_string(object, "bit_alignment")?,
            upper_four_bits_zero: required_bool(object, "upper_four_bits_zero")?,
            sample_range: required_string(object, "sample_range")?,
            color_range: required_string(object, "color_range")?,
            range_is_limited: required_bool(object, "range_is_limited")?,
            color_primaries: required_string(object, "color_primaries")?,
            color_transfer: required_string(object, "color_transfer")?,
            matrix_coefficients: required_string(object, "matrix_coefficients")?,
            chroma_sampling: required_string(object, "chroma_sampling")?,
            alpha: required_string(object, "alpha")?,
            final_code_bounds: required_u64_array(object, "final_code_bounds")?,
            linear_signal_scale_num: required_u64(object, "linear_signal_scale_num")?,
            linear_signal_scale_den: required_u64(object, "linear_signal_scale_den")?,
            linear_signal_scale_stops: required_i64(object, "linear_signal_scale_stops")?,
            scale_application_point: required_string(object, "scale_application_point")?,
            expected_total_video_bytes: required_u64(object, "expected_total_video_bytes")?,
            output_target: required_string(object, "output_target")?,
            stdout_used: required_bool(object, "stdout_used")?,
            byte_clean_stdout: required_bool(object, "byte_clean_stdout")?,
            render_mode: required_string(object, "render_mode")?,
            byte_identity_status: required_string(object, "byte_identity_status")?,
            byte_identity_output_byte_count_ok: required_bool(
                object,
                "byte_identity_output_byte_count_ok",
            )?,
            byte_identity_frame_byte_count_ok: required_bool(
                object,
                "byte_identity_frame_byte_count_ok",
            )?,
            byte_identity_plane_order_ok: required_bool(object, "byte_identity_plane_order_ok")?,
            byte_identity_little_endian_ok: required_bool(
                object,
                "byte_identity_little_endian_ok",
            )?,
            byte_identity_meaningful_low_12_bits_ok: required_bool(
                object,
                "byte_identity_meaningful_low_12_bits_ok",
            )?,
            byte_identity_code_bounds_ok: required_bool(object, "byte_identity_code_bounds_ok")?,
            byte_identity_mismatches: required_u64(object, "byte_identity_mismatches")?,
            pipe_full_byte_consume_in_run: required_bool(object, "pipe_full_byte_consume_in_run")?,
            bytes_per_frame: required_u64(object, "bytes_per_frame")?,
            frame_width: required_usize(object, "frame_width")?,
            frame_height: required_usize(object, "frame_height")?,
        })
    }
}

fn validate_report_for_spec(
    report: &MeasurementReport,
    command: &ChildMeasurementCommand,
) -> OptimizerResult<()> {
    // Child JSON is an untrusted measurement boundary. Policy sees a row only after
    // its report version and status, frame-count floor, payload profile, and
    // sink-specific output contract are validated.
    let measurement_id = command.id.label();
    if report.measurement_report_version != 1 {
        return Err(invalid_report(measurement_id, "unsupported report version"));
    }
    if report.command != command.sink.command_label() {
        return Err(invalid_report(measurement_id, "report command mismatch"));
    }
    if report.input_path_recorded {
        return Err(invalid_report(
            measurement_id,
            "report recorded an input path",
        ));
    }
    if report.status != "ok" {
        return Err(invalid_report(
            measurement_id,
            format!(
                "status is {}; {}",
                report.status,
                report
                    .error_message
                    .as_deref()
                    .unwrap_or("no error message")
            ),
        ));
    }
    if !report.decision_quality_ok
        || report.decision_quality_min_frames != OPTIMIZER_DECISION_FRAMES
        || report.frames_measured < OPTIMIZER_DECISION_FRAMES
        || report.measurement_truncated_by_clip_length
    {
        return Err(invalid_report(
            measurement_id,
            OPTIMIZER_MINIMUM_FRAMES_MESSAGE,
        ));
    }
    if report.payload_profile_requested.as_deref() != Some(command.profile.summary_label())
        || !report.payload_profile_effective
    {
        return Err(invalid_report(
            measurement_id,
            "payload profile did not take effect",
        ));
    }

    match command.sink {
        MeasurementSink::Display => validate_display_report(report, measurement_id),
        MeasurementSink::Pipe => validate_pipe_report(report, measurement_id),
    }
}

fn validate_display_report(
    report: &MeasurementReport,
    measurement_id: String,
) -> OptimizerResult<()> {
    let display = report
        .display
        .as_ref()
        .ok_or_else(|| invalid_report(measurement_id.clone(), "missing display object"))?;
    if display.full_frame_readback_performed {
        return Err(invalid_report(
            measurement_id,
            "display report performed full-frame readback",
        ));
    }
    if display.output_bytes != 0 {
        return Err(invalid_report(
            measurement_id,
            "display report output_bytes must be 0",
        ));
    }
    if display.live_window {
        return Err(invalid_report(
            measurement_id,
            "display measurement opened a live window",
        ));
    }
    Ok(())
}

fn validate_pipe_report(report: &MeasurementReport, measurement_id: String) -> OptimizerResult<()> {
    let pipe = report
        .pipe
        .as_ref()
        .ok_or_else(|| invalid_report(measurement_id.clone(), "missing pipe object"))?;
    if report.sink_name != "gpu_gpu_vignette_pipe_yuv444p12le_raw"
        || pipe.pixel_format != "yuv444p12le"
        || pipe.plane_order != ["Y", "Cb", "Cr"]
        || pipe.endianness != "little"
        || pipe.bits_per_channel != 12
        || pipe.storage_bits_per_sample != 16
        || pipe.meaningful_bits_per_sample != 12
        || pipe.bit_alignment != "lsb"
        || !pipe.upper_four_bits_zero
        || pipe.sample_range != "video-data-12bit"
        || pipe.color_range != "tv"
        || !pipe.range_is_limited
        || pipe.color_primaries != "bt2020"
        || pipe.color_transfer != "linear"
        || pipe.matrix_coefficients != "bt2020nc"
        || pipe.chroma_sampling != "4:4:4"
        || pipe.alpha != "none"
        || pipe.final_code_bounds != [16, 4079]
        || pipe.linear_signal_scale_num != 1
        || pipe.linear_signal_scale_den != 2
        || pipe.linear_signal_scale_stops != -1
        || pipe.scale_application_point != "linear-signal-before-ycbcr-tv-offsets"
        || pipe.output_target != "discard"
        || pipe.stdout_used
        || !pipe.byte_clean_stdout
        || pipe.render_mode != "vignette"
        || pipe.byte_identity_status != "OK"
        || !pipe.byte_identity_output_byte_count_ok
        || !pipe.byte_identity_frame_byte_count_ok
        || !pipe.byte_identity_plane_order_ok
        || !pipe.byte_identity_little_endian_ok
        || !pipe.byte_identity_meaningful_low_12_bits_ok
        || !pipe.byte_identity_code_bounds_ok
        || pipe.byte_identity_mismatches != 0
        || pipe.pipe_full_byte_consume_in_run
    {
        return Err(invalid_report(
            measurement_id.clone(),
            "PIPE report does not match the canonical direct yuv444p12le discard contract",
        ));
    }

    let expected_frame_bytes = u64::try_from(pipe.frame_width)
        .ok()
        .and_then(|width| {
            u64::try_from(pipe.frame_height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(6))
        .ok_or_else(|| {
            invalid_report(
                measurement_id.clone(),
                "PIPE frame-byte arithmetic overflow",
            )
        })?;
    if pipe.bytes_per_frame != expected_frame_bytes {
        return Err(invalid_report(
            measurement_id.clone(),
            "PIPE bytes_per_frame does not equal width * height * 6",
        ));
    }
    let frames_measured = u64::try_from(report.frames_measured).map_err(|_| {
        invalid_report(
            measurement_id.clone(),
            "PIPE measured-frame count does not fit u64",
        )
    })?;
    let expected_total_bytes = expected_frame_bytes
        .checked_mul(frames_measured)
        .ok_or_else(|| {
            invalid_report(
                measurement_id.clone(),
                "PIPE total-byte arithmetic overflow",
            )
        })?;
    if pipe.expected_total_video_bytes != expected_total_bytes {
        return Err(invalid_report(
            measurement_id,
            "PIPE expected_total_video_bytes contradicts frame geometry/count",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct SinkProfileSummary {
    pub default_median_fps: f64,
    pub offset_median_fps: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectionSummary {
    pub display: Option<SinkProfileSummary>,
    pub pipe: Option<SinkProfileSummary>,
    pub total_ram_bytes: Option<u64>,
    pub decision: PayloadProfileDecision,
    pub measurement_failure_detail: Option<String>,
}

impl SelectionSummary {
    fn recommendation(&self) -> OptimizerRecommendation {
        OptimizerRecommendation {
            payload_profile: RecommendedPayloadProfile::from_state_profile(self.decision.selected),
            decision_quality_ok: self.decision.reason.measurements_are_valid(),
            settings_written: false,
            reason: self.decision.reason,
            reason_detail: self.measurement_failure_detail.clone(),
        }
    }
}

pub struct OptimizerRecommendationCalculator;

impl OptimizerRecommendationCalculator {
    pub fn select(
        measurements: &OptimizerMeasurementSet,
        total_ram_bytes: Option<u64>,
    ) -> SelectionSummary {
        let display = summarize_sink(measurements, MeasurementSink::Display);
        let pipe = summarize_sink(measurements, MeasurementSink::Pipe);
        let mut failures = row_failures(&measurements.rows);
        if display.is_none() {
            failures.push("Display rows are incomplete".to_string());
        }
        if pipe.is_none() {
            failures.push("PIPE rows are incomplete".to_string());
        }
        let medians = match (&display, &pipe) {
            (Some(display), Some(pipe)) if failures.is_empty() => Some(ThroughputMedians {
                default_display_fps: display.default_median_fps,
                default_pipe_fps: pipe.default_median_fps,
                offset_display_fps: display.offset_median_fps,
                offset_pipe_fps: pipe.offset_median_fps,
            }),
            _ => None,
        };
        let decision = recommend_payload_profile(medians, total_ram_bytes);
        SelectionSummary {
            display,
            pipe,
            total_ram_bytes,
            decision,
            measurement_failure_detail: (!failures.is_empty()).then(|| failures.join("; ")),
        }
    }
}

fn summarize_sink(
    measurements: &OptimizerMeasurementSet,
    sink: MeasurementSink,
) -> Option<SinkProfileSummary> {
    let default_rows = complete_success_rows(
        measurements,
        sink,
        RecommendedPayloadProfile::DefaultChunked64,
    )?;
    let offset_rows = complete_success_rows(
        measurements,
        sink,
        RecommendedPayloadProfile::OffsetPrefetch,
    )?;
    let default_median_fps = measured_median_fps(&default_rows);
    let offset_median_fps = measured_median_fps(&offset_rows);
    Some(SinkProfileSummary {
        default_median_fps,
        offset_median_fps,
    })
}

fn measured_median_fps(rows: &[&CandidateMeasurementRow]) -> f64 {
    if let Some(invalid) = rows
        .iter()
        .map(|row| row.fps)
        .find(|fps| !fps.is_finite() || *fps <= 0.0)
    {
        invalid
    } else {
        median(rows.iter().map(|row| row.fps).collect())
    }
}

fn complete_success_rows(
    measurements: &OptimizerMeasurementSet,
    sink: MeasurementSink,
    profile: RecommendedPayloadProfile,
) -> Option<Vec<&CandidateMeasurementRow>> {
    let rows = measurements.rows_for(sink, profile);
    if rows.len() != OPTIMIZER_REPETITIONS {
        return None;
    }
    let mut repetitions = HashSet::new();
    for row in &rows {
        if !row.success
            || row.frames_measured < OPTIMIZER_DECISION_FRAMES
            || !repetitions.insert(row.repetition)
        {
            return None;
        }
    }
    (repetitions == HashSet::from([1usize, 2, 3])).then_some(rows)
}

fn row_failures(rows: &[CandidateMeasurementRow]) -> Vec<String> {
    rows.iter()
        .filter(|row| !row.success)
        .map(|row| {
            format!(
                "{} {} repetition {} failed: {}",
                row.sink.summary_label(),
                row.profile.summary_label(),
                row.repetition,
                row.failure_reason
                    .as_deref()
                    .unwrap_or("measurement failed")
            )
        })
        .collect()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    values[values.len() / 2]
}

pub fn prompt_and_maybe_save_recommendation(
    recommendation: &OptimizerRecommendation,
) -> OptimizerResult<OptimizerRecommendation> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = stdout.lock();
    prompt_and_maybe_save_recommendation_with_io(
        recommendation,
        &mut reader,
        &mut writer,
        save_optimized_state,
    )
}

pub fn prompt_and_maybe_save_recommendation_with_io<R, W, F>(
    recommendation: &OptimizerRecommendation,
    reader: &mut R,
    writer: &mut W,
    save: F,
) -> OptimizerResult<OptimizerRecommendation>
where
    R: BufRead,
    W: Write,
    F: FnOnce(&OptimizedState) -> Result<PathBuf, OptimizedStateError>,
{
    let mut result = recommendation.clone();

    if !recommendation.decision_quality_ok {
        writeln!(
            writer,
            "Optimizer benchmark did not complete all required rows. Default settings remain active. No settings were written."
        )?;
        return Ok(result);
    }

    if recommendation.is_built_in_default() {
        writeln!(
            writer,
            "The optimizer policy recommends default_chunked64 for this run. No optimized settings file was written. Use `mcraw4vulkan optimizer --restore-defaults` to remove an existing optimized override."
        )?;
        return Ok(result);
    }

    writeln!(writer, "Save optimized payload profile? (n) no / (y) yes")?;
    writer.flush()?;
    let mut answer = String::new();
    let bytes_read = reader.read_line(&mut answer)?;
    if bytes_read == 0 || !answer.trim().eq_ignore_ascii_case("y") {
        writeln!(writer, "No settings were written.")?;
        return Ok(result);
    }

    let state = recommendation.optimized_state();
    // Saving is a user-owned transition and occurs only for a complete,
    // non-default recommendation; policy evaluation never persists by itself.
    let path = save(&state)?;
    result.settings_written = true;
    writeln!(writer, "optimized settings written: {}", path.display())?;
    Ok(result)
}

pub struct OptimizerSummaryPrinter;

impl OptimizerSummaryPrinter {
    pub fn print(outcome: &OptimizerRunOutcome) {
        println!("{}", Self::format(outcome));
    }

    pub fn format(outcome: &OptimizerRunOutcome) -> String {
        let display = sink_summary_line(
            outcome.selection.display.as_ref(),
            outcome.selection.decision.display_ratio,
        );
        let pipe = sink_summary_line(
            outcome.selection.pipe.as_ref(),
            outcome.selection.decision.pipe_ratio,
        );
        let combined_score = combined_score_line(&outcome.selection);
        let gate_lines = selection_gate_lines(&outcome.selection);
        let reason = recommendation_reason_text(&outcome.selection);
        let settings_file = if outcome.recommendation.is_built_in_default() {
            "not written"
        } else {
            "pending user choice"
        };

        format!(
            "mcraw4vulkan optimizer\n\nInput: {}\nDecision frames: {}\nGPU warmup frames: {} unscored\nMeasured repetitions: {}\nProgress rows: {}\nHard timeout: {} seconds\nTotal physical RAM: {}\n\nMeasurement note: the optimizer compares the complete default_chunked64 and offset_prefetch payload profiles using only Display and PIPE throughput.\n\nPayload profile measurements:\n  Display GPU no-vig:\n{}\n  PIPE GPU+vig:\n{}\n\nCombined throughput:\n{}\n\nDecision gates:\n{}\n\nRecommendation: {}\nrecommendation_reason: {}\nReason: {}\nResult: {}\nsettings_file: {}",
            outcome.input_basename,
            outcome.frames,
            OPTIMIZER_GPU_WARMUP_FRAMES,
            OPTIMIZER_REPETITIONS,
            outcome.progress_total_steps,
            OPTIMIZER_HARD_TIMEOUT.as_secs(),
            total_ram_text(outcome.selection.total_ram_bytes),
            display,
            pipe,
            combined_score,
            gate_lines,
            outcome.recommendation.payload_profile.summary_label(),
            outcome.recommendation.reason.token(),
            reason,
            outcome.recommendation.payload_profile.result_label(),
            settings_file,
        )
    }
}

fn sink_summary_line(summary: Option<&SinkProfileSummary>, ratio: Option<f64>) -> String {
    let Some(summary) = summary else {
        return "    required rows incomplete".to_string();
    };
    format!(
        "    default_chunked64 median: {:.2} fps\n    offset_prefetch median:   {:.2} fps\n    throughput ratio: {}",
        summary.default_median_fps,
        summary.offset_median_fps,
        ratio
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "unavailable".to_string())
    )
}

fn combined_score_line(selection: &SelectionSummary) -> String {
    match selection.decision.combined_ratio {
        Some(ratio) => format!(
            "  Geometric Display/PIPE ratio: {:.3}\n  Combined improvement: {:.1}%",
            ratio,
            (ratio - 1.0) * 100.0
        ),
        None => "  Geometric Display/PIPE ratio: unavailable\n  Combined improvement: unavailable"
            .to_string(),
    }
}

fn selection_gate_lines(selection: &SelectionSummary) -> String {
    format!(
        "  Neither sink loses more than 5%: {}\n  Combined throughput gain greater than 10%: {}\n  Total RAM below {} bytes (17 GiB): {}",
        optional_gate_status(selection.decision.no_loss_over_five_percent),
        optional_gate_status(selection.decision.throughput_gain_over_ten_percent),
        LOW_RAM_THRESHOLD_BYTES,
        low_ram_gate_status(selection.decision.low_ram_gate),
    )
}

fn optional_gate_status(passed: Option<bool>) -> &'static str {
    match passed {
        Some(passed) => boolean_gate_status(passed),
        None => "unavailable",
    }
}

fn low_ram_gate_status(gate: LowRamGate) -> &'static str {
    match gate {
        LowRamGate::Pass => boolean_gate_status(true),
        LowRamGate::Fail => boolean_gate_status(false),
        LowRamGate::Unavailable => "unavailable",
    }
}

fn boolean_gate_status(value: bool) -> &'static str {
    if value { "TRUE" } else { "FALSE" }
}

fn total_ram_text(total_ram_bytes: Option<u64>) -> String {
    total_ram_bytes.map_or_else(
        || "unavailable".to_string(),
        |bytes| {
            format!(
                "{bytes} bytes ({:.2} GiB)",
                bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            )
        },
    )
}

fn recommendation_reason_text(selection: &SelectionSummary) -> String {
    match selection.decision.reason {
        RecommendationReason::OffsetThroughputGain => "combined Display/PIPE throughput gain exceeded 10%, and neither sink regressed by more than 5%.".to_string(),
        RecommendationReason::OffsetLowRamNoRegression => "total system RAM is below 17 GiB, and neither Display nor PIPE regressed by more than 5%.".to_string(),
        RecommendationReason::DefaultSinkRegression(sink) => {
            let (label, ratio) = match sink {
                ScoredSink::Display => ("Display", selection.decision.display_ratio),
                ScoredSink::Pipe => ("PIPE", selection.decision.pipe_ratio),
            };
            ratio.map_or_else(
                || format!("offset_prefetch regressed {label} by more than 5%."),
                |ratio| format!("offset_prefetch regressed {label} by more than 5% (ratio {ratio:.3})."),
            )
        }
        RecommendationReason::DefaultRamNotLow => "combined Display/PIPE throughput gain did not exceed 10%, and total system RAM is not below 17 GiB.".to_string(),
        RecommendationReason::DefaultRamUnavailable => "combined Display/PIPE throughput gain did not exceed 10%, and total physical RAM was unavailable, so the low-RAM rule could not pass.".to_string(),
        RecommendationReason::DefaultInvalidMeasurements => "required Display and PIPE median FPS values were not all finite and positive.".to_string(),
        RecommendationReason::DefaultMeasurementFailure => selection
            .measurement_failure_detail
            .as_deref()
            .map_or_else(
                || "a required Display or PIPE measurement was missing or failed.".to_string(),
                |detail| format!("a required Display or PIPE measurement failed: {detail}"),
            ),
    }
}

pub fn create_optimizer_temp_report_dir() -> OptimizerResult<PathBuf> {
    let root = std::env::temp_dir().join("mcraw4vulkan-optimizer");
    fs::create_dir_all(&root)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let dir = root.join(format!("run-{}-{timestamp}", std::process::id()));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn finish_report_dir(report_dir: &Path, keep_reports: bool, success: bool) -> OptimizerResult<()> {
    if !keep_reports {
        fs::remove_dir_all(report_dir)?;
    } else {
        let reason = if success {
            "optimizer reports retained"
        } else {
            "optimizer reports retained after failure"
        };
        eprintln!("{reason}: {}", report_dir.display());
    }
    Ok(())
}

fn input_basename(input: &Path) -> String {
    input
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown.mcraw".to_string())
}

fn stderr_tail(text: &str) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    let start = lines.len().saturating_sub(20);
    lines[start..].join("\n")
}

fn stderr_mentions_warning(text: &str) -> bool {
    text.lines()
        .any(|line| line.to_ascii_lowercase().contains("warn"))
}

fn sanitize_for_input(text: &str, input_path: &Path) -> String {
    let input = input_path.to_string_lossy();
    if input.is_empty() {
        text.to_string()
    } else {
        text.replace(input.as_ref(), "<input>")
    }
}

fn invalid_report(measurement_id: impl Into<String>, message: impl Into<String>) -> OptimizerError {
    OptimizerError::InvalidReport {
        measurement_id: measurement_id.into(),
        message: message.into(),
    }
}

fn parse_error(message: impl Into<String>) -> OptimizerError {
    OptimizerError::InvalidReport {
        measurement_id: "unknown".to_string(),
        message: message.into(),
    }
}

fn required_value<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
) -> OptimizerResult<&'a Value> {
    object
        .get(key)
        .ok_or_else(|| parse_error(format!("missing required field {key}")))
}

fn required_string(object: &serde_json::Map<String, Value>, key: &str) -> OptimizerResult<String> {
    required_value(object, key)?
        .as_str()
        .map(ToString::to_string)
        .ok_or_else(|| parse_error(format!("{key} must be a string")))
}

fn optional_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> OptimizerResult<Option<String>> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(|value| Some(value.to_string()))
            .ok_or_else(|| parse_error(format!("{key} must be a string or null"))),
    }
}

fn required_bool(object: &serde_json::Map<String, Value>, key: &str) -> OptimizerResult<bool> {
    required_value(object, key)?
        .as_bool()
        .ok_or_else(|| parse_error(format!("{key} must be a bool")))
}

fn required_u64(object: &serde_json::Map<String, Value>, key: &str) -> OptimizerResult<u64> {
    required_value(object, key)?
        .as_u64()
        .ok_or_else(|| parse_error(format!("{key} must be an unsigned integer")))
}

fn required_i64(object: &serde_json::Map<String, Value>, key: &str) -> OptimizerResult<i64> {
    required_value(object, key)?
        .as_i64()
        .ok_or_else(|| parse_error(format!("{key} must be a signed integer")))
}

fn optional_u64(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> OptimizerResult<Option<u64>> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| parse_error(format!("{key} must be an unsigned integer or null"))),
    }
}

fn required_usize(object: &serde_json::Map<String, Value>, key: &str) -> OptimizerResult<usize> {
    let value = required_u64(object, key)?;
    usize::try_from(value).map_err(|_| parse_error(format!("{key} is too large")))
}

fn optional_usize(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> OptimizerResult<Option<usize>> {
    optional_u64(object, key)?
        .map(|value| usize::try_from(value).map_err(|_| parse_error(format!("{key} is too large"))))
        .transpose()
}

fn required_f64(object: &serde_json::Map<String, Value>, key: &str) -> OptimizerResult<f64> {
    let value = required_value(object, key)?
        .as_f64()
        .ok_or_else(|| parse_error(format!("{key} must be a number")))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(parse_error(format!("{key} must be finite")))
    }
}

fn required_string_array(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> OptimizerResult<Vec<String>> {
    let array = required_value(object, key)?
        .as_array()
        .ok_or_else(|| parse_error(format!("{key} must be an array")))?;
    array
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToString::to_string)
                .ok_or_else(|| parse_error(format!("{key} must contain strings")))
        })
        .collect()
}

fn required_u64_array(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> OptimizerResult<Vec<u64>> {
    let array = required_value(object, key)?
        .as_array()
        .ok_or_else(|| parse_error(format!("{key} must be an array")))?;
    array
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| parse_error(format!("{key} must contain unsigned integers")))
        })
        .collect()
}

fn optional_string_array(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> OptimizerResult<Vec<String>> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(value) => {
            let array = value
                .as_array()
                .ok_or_else(|| parse_error(format!("{key} must be an array")))?;
            array
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(ToString::to_string)
                        .ok_or_else(|| parse_error(format!("{key} must contain strings")))
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn commands() -> Vec<ChildMeasurementCommand> {
        optimizer_measurement_commands(
            Path::new("private/source/clip.mcraw"),
            Path::new("tmp/reports"),
            OPTIMIZER_DECISION_FRAMES,
        )
    }

    fn display_report(fps: f64, payload: &str, p95: f64) -> Value {
        json!({
            "measurement_report_version": 1,
            "created_by": "mcraw4vulkan",
            "mcraw4vulkan_version": "0.1.0",
            "mcraw4vulkan_commit": null,
            "command": "display",
            "sink_name": "gpu_display_no_vsync_proxy",
            "status": "ok",
            "error_message": null,
            "input_basename": "clip.mcraw",
            "input_stem": "clip",
            "input_path_recorded": false,
            "frames_requested": 600,
            "frames_available": 600,
            "frames_selected": 600,
            "frames_measured": 600,
            "warmup_frames_requested": 30,
            "warmup_frames_processed": 30,
            "measurement_truncated_by_clip_length": false,
            "decision_quality_min_frames": 600,
            "decision_quality_ok": true,
            "start_frame": 0,
            "stride": 1,
            "settings_source": "default",
            "backend": "gpu",
            "backend_effective": "gpu",
            "vignette_correction": "none",
            "payload_profile_requested": payload,
            "payload_profile_effective": true,
            "setup_s": 1.0,
            "warmup_s": 1.0,
            "run_s": 10.0,
            "flush_s": 0.1,
            "total_s": 12.1,
            "fps": fps,
            "total_fps": fps * 0.9,
            "service_p50_s": p95 * 0.8,
            "service_p95_s": p95,
            "service_p99_s": p95 * 1.2,
            "bytes_output_logical": 0,
            "bytes_written_physical": 0,
            "full_frame_readback_performed": false,
            "notes": [],
            "display": {
                "preview_width": 1280,
                "preview_height": 720,
                "output_bytes": 0,
                "full_frame_readback_performed": false,
                "overlay_applicable": false,
                "live_window": false,
                "vsync_applicable": false,
                "warmup_frames_processed": 30,
                "measured_frames_processed": 600
            }
        })
    }

    fn pipe_report(fps: f64, payload: &str, p95: f64) -> Value {
        let pipe = json!({
            "pixel_format": "yuv444p12le",
            "plane_order": ["Y", "Cb", "Cr"],
            "endianness": "little",
            "bits_per_channel": 12,
            "storage_bits_per_sample": 16,
            "meaningful_bits_per_sample": 12,
            "bit_alignment": "lsb",
            "upper_four_bits_zero": true,
            "sample_range": "video-data-12bit",
            "color_range": "tv",
            "range_is_limited": true,
            "color_primaries": "bt2020",
            "color_transfer": "linear",
            "matrix_coefficients": "bt2020nc",
            "chroma_sampling": "4:4:4",
            "alpha": "none",
            "final_code_bounds": [16, 4079],
            "linear_signal_scale_num": 1,
            "linear_signal_scale_den": 2,
            "linear_signal_scale_stops": -1,
            "scale_application_point": "linear-signal-before-ycbcr-tv-offsets",
            "frame_width": 1920,
            "frame_height": 1080,
            "bytes_per_frame": 12441600,
            "expected_total_video_bytes": 7464960000u64,
            "output_target": "discard",
            "stdout_used": false,
            "byte_clean_stdout": true,
            "render_mode": "vignette",
            "producer_fps": fps,
            "writer_s": 0.0,
            "writer_flush_s": 0.0,
            "mapped_consume_s": 0.0,
            "pipe_full_byte_consume_in_run": false,
            "byte_identity_status": "OK",
            "byte_identity_frames_checked": 600,
            "byte_identity_validation_frames_checked": 3,
            "byte_identity_validation_bytes_sampled": 12288,
            "byte_identity_output_byte_count_ok": true,
            "byte_identity_frame_byte_count_ok": true,
            "byte_identity_plane_order_ok": true,
            "byte_identity_little_endian_ok": true,
            "byte_identity_meaningful_low_12_bits_ok": true,
            "byte_identity_code_bounds_ok": true,
            "byte_identity_mismatches": 0,
            "byte_identity_first_mismatch_offset": null,
            "byte_identity_rows": []
        });
        json!({
            "measurement_report_version": 1,
            "created_by": "mcraw4vulkan",
            "mcraw4vulkan_version": "0.1.0",
            "mcraw4vulkan_commit": null,
            "command": "pipe",
            "sink_name": "gpu_gpu_vignette_pipe_yuv444p12le_raw",
            "status": "ok",
            "error_message": null,
            "input_basename": "clip.mcraw",
            "input_stem": "clip",
            "input_path_recorded": false,
            "frames_requested": 600,
            "frames_available": 600,
            "frames_selected": 600,
            "frames_measured": 600,
            "warmup_frames_requested": 0,
            "warmup_frames_processed": 0,
            "measurement_truncated_by_clip_length": false,
            "decision_quality_min_frames": 600,
            "decision_quality_ok": true,
            "start_frame": 0,
            "stride": 1,
            "settings_source": "default",
            "backend": "gpu",
            "backend_effective": "gpu",
            "vignette_correction": "with",
            "payload_profile_requested": payload,
            "payload_profile_effective": true,
            "setup_s": 1.0,
            "warmup_s": 0.0,
            "run_s": 10.0,
            "flush_s": 0.1,
            "total_s": 11.1,
            "fps": fps,
            "total_fps": fps * 0.9,
            "service_p50_s": p95 * 0.8,
            "service_p95_s": p95,
            "service_p99_s": p95 * 1.2,
            "bytes_output_logical": 7464960000u64,
            "bytes_written_physical": 0,
            "full_frame_readback_performed": true,
            "notes": [],
            "pipe": pipe
        })
    }

    fn parse_report(value: Value) -> MeasurementReport {
        MeasurementReport::parse(&serde_json::to_string(&value).unwrap()).unwrap()
    }

    fn row(
        sink: MeasurementSink,
        profile: RecommendedPayloadProfile,
        repetition: usize,
        fps: f64,
    ) -> CandidateMeasurementRow {
        CandidateMeasurementRow {
            sink,
            profile,
            repetition,
            frames_measured: OPTIMIZER_DECISION_FRAMES,
            fps,
            success: true,
            failure_reason: None,
        }
    }

    fn winning_rows() -> OptimizerMeasurementSet {
        let mut rows = Vec::new();
        for repetition in 1..=3 {
            rows.push(row(
                MeasurementSink::Display,
                RecommendedPayloadProfile::DefaultChunked64,
                repetition,
                100.0 + repetition as f64,
            ));
            rows.push(row(
                MeasurementSink::Display,
                RecommendedPayloadProfile::OffsetPrefetch,
                repetition,
                112.0 + repetition as f64,
            ));
            rows.push(row(
                MeasurementSink::Pipe,
                RecommendedPayloadProfile::DefaultChunked64,
                repetition,
                100.0 + repetition as f64,
            ));
            rows.push(row(
                MeasurementSink::Pipe,
                RecommendedPayloadProfile::OffsetPrefetch,
                repetition,
                112.0 + repetition as f64,
            ));
        }
        OptimizerMeasurementSet::from_rows(rows)
    }

    #[test]
    fn minimum_frame_gate_rejects_599_and_accepts_600_and_601() {
        let error = validate_optimizer_frame_count(599).expect_err("599 frames must be rejected");
        assert_eq!(error.to_string(), OPTIMIZER_MINIMUM_FRAMES_MESSAGE);
        assert_eq!(
            error.to_string(),
            "Optimization testing requires at least 600 frames for accuracy."
        );
        assert!(validate_optimizer_frame_count(600).is_ok());
        assert!(validate_optimizer_frame_count(601).is_ok());
    }

    #[test]
    fn builds_balanced_payload_only_child_command_vectors() {
        let commands = commands();
        assert_eq!(commands.len(), 12);
        assert_eq!(optimizer_progress_total_steps(&commands), 13);
        assert_eq!(
            [
                RecommendedPayloadProfile::DefaultChunked64.summary_label(),
                RecommendedPayloadProfile::OffsetPrefetch.summary_label(),
            ],
            ["default_chunked64", "offset_prefetch"]
        );
        assert_eq!(
            commands
                .iter()
                .map(|command| command.profile)
                .collect::<HashSet<_>>(),
            HashSet::from([
                RecommendedPayloadProfile::DefaultChunked64,
                RecommendedPayloadProfile::OffsetPrefetch,
            ])
        );
        let labels = commands
            .iter()
            .map(|command| command.id.label())
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            vec![
                "display_default_chunked64_rep1",
                "display_offset_prefetch_rep1",
                "pipe_default_chunked64_rep1",
                "pipe_offset_prefetch_rep1",
                "pipe_offset_prefetch_rep2",
                "pipe_default_chunked64_rep2",
                "display_offset_prefetch_rep2",
                "display_default_chunked64_rep2",
                "display_default_chunked64_rep3",
                "display_offset_prefetch_rep3",
                "pipe_default_chunked64_rep3",
                "pipe_offset_prefetch_rep3",
            ]
        );
        for sink in [MeasurementSink::Display, MeasurementSink::Pipe] {
            for profile in [
                RecommendedPayloadProfile::DefaultChunked64,
                RecommendedPayloadProfile::OffsetPrefetch,
            ] {
                assert_eq!(
                    commands
                        .iter()
                        .filter(|command| command.sink == sink && command.profile == profile)
                        .count(),
                    OPTIMIZER_REPETITIONS
                );
            }
        }
        assert!(commands.iter().all(|command| {
            matches!(command.argv[0].as_str(), "display" | "pipe")
                && command.argv.iter().any(|arg| arg == "--gpu")
        }));
        for command in &commands {
            for retired in [
                "dng",
                "--cpu",
                "chunk32",
                "chunk128",
                "--output",
                "pipe_raw_file_delete",
            ] {
                assert!(
                    !command.argv.iter().any(|arg| arg == retired),
                    "retired optimizer argument {retired:?} appeared in {:?}",
                    command.argv
                );
            }
        }
    }

    #[test]
    fn gpu_warmup_command_uses_unscored_default_display_path() {
        let command = optimizer_gpu_warmup_command(
            Path::new("private/source/clip.mcraw"),
            Path::new("tmp/reports"),
        );

        assert_eq!(command.id, MeasurementId::GpuWarmupUnscored);
        assert_eq!(command.argv[0], "display");
        assert!(command.argv.contains(&"--gpu".to_string()));
        assert!(command.argv.contains(&"--no-vig-correction".to_string()));
        assert!(command.argv.contains(&"--no-vsync".to_string()));
        assert!(command.argv.contains(&"default_chunked64".to_string()));
        assert_eq!(
            value_after_flag(&command.argv, "--internal-measure-frames"),
            Some("600")
        );
        assert_eq!(
            value_after_flag(&command.argv, "--internal-measure-warmup-frames"),
            Some("0")
        );
        assert!(command.report_path.ends_with("gpu-warmup.json"));
    }

    #[test]
    fn progress_line_uses_dynamic_total() {
        assert_eq!(
            optimizer_progress_line(13, 13, "pipe_offset_prefetch_rep3"),
            "optimizer progress: step 13/13 pipe_offset_prefetch_rep3"
        );
    }

    #[test]
    fn child_commands_use_internal_measurement_flags_and_600_frames() {
        for command in commands() {
            assert!(command.argv.contains(&"--internal-measure".to_string()));
            assert!(
                command
                    .argv
                    .contains(&"--internal-measure-report".to_string())
            );
            assert!(
                command
                    .argv
                    .contains(&"--internal-measure-frames".to_string())
            );
            assert!(command.argv.contains(&"600".to_string()));
        }
    }

    #[test]
    fn display_commands_use_warmup_and_payload_profiles() {
        let display = commands()
            .into_iter()
            .filter(|command| command.argv[0] == "display")
            .collect::<Vec<_>>();
        assert_eq!(display.len(), 6);
        for command in &display {
            assert!(
                command
                    .argv
                    .contains(&"--internal-measure-warmup-frames".to_string())
            );
            assert!(command.argv.contains(&"30".to_string()));
        }
    }

    #[test]
    fn pipe_commands_use_payload_profiles_and_no_output_flag() {
        let pipe = commands()
            .into_iter()
            .filter(|command| command.argv[0] == "pipe")
            .collect::<Vec<_>>();
        assert_eq!(pipe.len(), 6);
        for command in pipe {
            assert!(!command.argv.contains(&"--output".to_string()));
        }
    }

    fn value_after_flag<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
        argv.windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].as_str())
    }

    #[test]
    fn report_cleanup_removes_run_dir_after_success_or_failure() {
        for success in [true, false] {
            let dir = std::env::temp_dir().join(format!(
                "mcraw4vulkan-optimizer-cleanup-test-{}-{success}",
                std::process::id()
            ));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("report.json"), b"{}").unwrap();

            finish_report_dir(&dir, false, success).unwrap();

            assert!(!dir.exists());
        }
    }

    #[test]
    fn report_parser_accepts_valid_sink_reports() {
        assert_eq!(
            parse_report(display_report(100.0, "default_chunked64", 0.010)).command,
            "display"
        );
        assert_eq!(
            parse_report(pipe_report(80.0, "default_chunked64", 0.020)).command,
            "pipe"
        );
    }

    #[test]
    fn report_parser_rejects_missing_fps() {
        let mut value = display_report(100.0, "default_chunked64", 0.010);
        value.as_object_mut().unwrap().remove("fps");
        assert!(MeasurementReport::parse(&serde_json::to_string(&value).unwrap()).is_err());
    }

    #[test]
    fn report_validation_rejects_status_not_ok() {
        let mut value = display_report(100.0, "default_chunked64", 0.010);
        value["status"] = json!("failed");
        let report = parse_report(value);
        let command = &commands()[0];
        assert!(validate_report_for_spec(&report, command).is_err());
    }

    #[test]
    fn report_validation_rejects_less_than_600_frames() {
        let mut value = display_report(100.0, "default_chunked64", 0.010);
        value["frames_measured"] = json!(599);
        value["decision_quality_ok"] = json!(false);
        let report = parse_report(value);
        let command = &commands()[0];
        let error = validate_report_for_spec(&report, command)
            .unwrap_err()
            .to_string();
        assert!(error.ends_with(OPTIMIZER_MINIMUM_FRAMES_MESSAGE));
        assert!(!error.contains("Test video must"));
    }

    #[test]
    fn report_validation_rejects_display_readback() {
        let mut value = display_report(100.0, "default_chunked64", 0.010);
        value["display"]["full_frame_readback_performed"] = json!(true);
        let report = parse_report(value);
        let command = &commands()[0];
        assert!(validate_report_for_spec(&report, command).is_err());
    }

    #[test]
    fn report_validation_rejects_pipe_stdout() {
        let mut value = pipe_report(80.0, "default_chunked64", 0.020);
        value["pipe"]["stdout_used"] = json!(true);
        let report = parse_report(value);
        let command = commands()
            .into_iter()
            .find(|command| command.sink == MeasurementSink::Pipe)
            .unwrap();
        assert!(validate_report_for_spec(&report, &command).is_err());
    }

    #[test]
    fn report_validation_accepts_canonical_direct_yuv_pipe_contract() {
        let report = parse_report(pipe_report(80.0, "default_chunked64", 0.020));
        let command = commands()
            .into_iter()
            .find(|command| {
                command.sink == MeasurementSink::Pipe
                    && command.profile == RecommendedPayloadProfile::DefaultChunked64
            })
            .unwrap();
        validate_report_for_spec(&report, &command).unwrap();
    }

    #[test]
    fn report_validation_rejects_stale_or_contradictory_pipe_contract_fields() {
        for (field, stale) in [
            ("pixel_format", json!("gbrp16le")),
            ("plane_order", json!(["G", "B", "R"])),
            ("bits_per_channel", json!(16)),
            ("storage_bits_per_sample", json!(12)),
            ("meaningful_bits_per_sample", json!(16)),
            ("bit_alignment", json!("msb")),
            ("upper_four_bits_zero", json!(false)),
            ("sample_range", json!("full")),
            ("color_range", json!("pc")),
            ("range_is_limited", json!(false)),
            ("color_primaries", json!("bt709")),
            ("color_transfer", json!("srgb")),
            ("matrix_coefficients", json!("bt709")),
            ("chroma_sampling", json!("4:2:2")),
            ("alpha", json!("straight")),
            ("final_code_bounds", json!([0, 4095])),
            ("linear_signal_scale_num", json!(2)),
            ("linear_signal_scale_den", json!(1)),
            ("linear_signal_scale_stops", json!(0)),
            ("scale_application_point", json!("post-quantization")),
            ("render_mode", json!("no_vignette")),
            ("byte_identity_meaningful_low_12_bits_ok", json!(false)),
            ("byte_identity_code_bounds_ok", json!(false)),
        ] {
            let mut value = pipe_report(80.0, "default_chunked64", 0.020);
            value["pipe"][field] = stale;
            let report = parse_report(value);
            let command = commands()
                .into_iter()
                .find(|command| command.sink == MeasurementSink::Pipe)
                .unwrap();
            assert!(
                validate_report_for_spec(&report, &command).is_err(),
                "stale or contradictory PIPE field {field} must be rejected",
            );
        }
    }

    #[test]
    fn report_validation_rejects_pipe_byte_arithmetic_contradictions() {
        let command = commands()
            .into_iter()
            .find(|command| command.sink == MeasurementSink::Pipe)
            .unwrap();
        for (field, contradiction) in [
            ("bytes_per_frame", json!(12_441_599_u64)),
            ("expected_total_video_bytes", json!(7_464_959_999_u64)),
        ] {
            let mut value = pipe_report(80.0, "default_chunked64", 0.020);
            value["pipe"][field] = contradiction;
            let report = parse_report(value);
            assert!(
                validate_report_for_spec(&report, &command).is_err(),
                "PIPE byte arithmetic contradiction {field} must be rejected",
            );
        }
    }

    fn select_rows(
        rows: &OptimizerMeasurementSet,
        total_ram_bytes: Option<u64>,
    ) -> SelectionSummary {
        OptimizerRecommendationCalculator::select(rows, total_ram_bytes)
    }

    #[test]
    fn timeout_uses_one_global_deadline_and_exact_owned_child_cleanup() {
        let source = include_str!("shellout.rs");
        assert!(source.contains("let deadline = Instant::now() + self.config.hard_timeout;"));
        assert!(source.contains("run_child_measurement_report(&warmup, deadline)"));
        assert!(source.contains("run_child_measurement_report(command, deadline)"));
        assert!(source.contains("optimizer deadline reached before row started"));
        assert!(source.contains("let _ = child.kill();"));
        assert!(source.contains("child.wait_with_output()?"));
    }

    #[test]
    fn complete_rows_feed_the_pure_throughput_decision() {
        let selection = select_rows(&winning_rows(), Some(64 * 1024 * 1024 * 1024));
        assert_eq!(selection.decision.selected, PayloadProfile::OffsetPrefetch);
        assert_eq!(
            selection.decision.reason,
            RecommendationReason::OffsetThroughputGain
        );
        assert!(selection.decision.throughput_gain_over_ten_percent.unwrap());
        assert_eq!(selection.decision.no_loss_over_five_percent, Some(true));
    }

    #[test]
    fn low_ram_rule_is_used_after_a_valid_near_tie() {
        let mut rows = winning_rows();
        for row in &mut rows.rows {
            row.fps = 100.0;
        }
        let selection = select_rows(&rows, Some(16 * 1024 * 1024 * 1024));
        assert_eq!(
            selection.decision.reason,
            RecommendationReason::OffsetLowRamNoRegression
        );
    }

    #[test]
    fn generic_p95_report_changes_have_no_optimizer_decision_influence() {
        let command = commands()
            .into_iter()
            .find(|command| {
                command.sink == MeasurementSink::Display
                    && command.profile == RecommendedPayloadProfile::DefaultChunked64
            })
            .unwrap();
        let fast_p95 = parse_report(display_report(100.0, "default_chunked64", 0.000_001));
        let slow_p95 = parse_report(display_report(100.0, "default_chunked64", 10_000.0));

        assert_eq!(
            CandidateMeasurementRow::from_report(&command, &fast_p95),
            CandidateMeasurementRow::from_report(&command, &slow_p95)
        );
        assert_eq!(
            select_rows(&winning_rows(), Some(64 * 1024 * 1024 * 1024))
                .decision
                .reason,
            RecommendationReason::OffsetThroughputGain
        );
    }

    #[test]
    fn child_failure_timeout_and_partial_rows_keep_default_without_quality() {
        for failure in ["child failed", "timed out"] {
            let mut set = winning_rows();
            set.rows[0].success = false;
            set.rows[0].failure_reason = Some(failure.to_string());
            let selection = select_rows(&set, Some(8 * 1024 * 1024 * 1024));
            assert_eq!(
                selection.decision.selected,
                PayloadProfile::DefaultChunked64
            );
            assert_eq!(
                selection.decision.reason,
                RecommendationReason::DefaultMeasurementFailure
            );
            assert!(
                selection
                    .measurement_failure_detail
                    .as_deref()
                    .unwrap()
                    .contains(failure)
            );
            assert!(!selection.recommendation().decision_quality_ok);
        }

        let mut partial = winning_rows();
        partial.rows.pop();
        let selection = select_rows(&partial, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(
            selection.decision.reason,
            RecommendationReason::DefaultMeasurementFailure
        );
        assert!(!selection.recommendation().decision_quality_ok);
    }

    #[test]
    fn incomplete_frame_count_keeps_default() {
        let mut set = winning_rows();
        set.rows[0].frames_measured = OPTIMIZER_DECISION_FRAMES - 1;
        let selection = select_rows(&set, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(
            selection.decision.reason,
            RecommendationReason::DefaultMeasurementFailure
        );
    }

    #[test]
    fn invalid_complete_medians_are_not_treated_as_missing() {
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut set = winning_rows();
            for row in set.rows.iter_mut().filter(|row| {
                row.sink == MeasurementSink::Pipe
                    && row.profile == RecommendedPayloadProfile::OffsetPrefetch
            }) {
                row.fps = invalid;
            }
            let selection = select_rows(&set, Some(8 * 1024 * 1024 * 1024));
            assert_eq!(
                selection.decision.reason,
                RecommendationReason::DefaultInvalidMeasurements
            );
            assert!(!selection.recommendation().decision_quality_ok);
        }
    }

    #[test]
    fn one_invalid_repetition_cannot_be_hidden_by_a_valid_median() {
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut set = winning_rows();
            let row = set
                .rows
                .iter_mut()
                .find(|row| {
                    row.sink == MeasurementSink::Display
                        && row.profile == RecommendedPayloadProfile::DefaultChunked64
                        && row.repetition == 1
                })
                .unwrap();
            row.fps = invalid;

            let selection = select_rows(&set, Some(8 * 1024 * 1024 * 1024));
            assert_eq!(
                selection.decision.reason,
                RecommendationReason::DefaultInvalidMeasurements
            );
            assert!(!selection.recommendation().decision_quality_ok);
        }
    }

    #[test]
    fn row_order_does_not_change_medians_or_selection() {
        let mut set = winning_rows();
        let original = select_rows(&set, Some(64 * 1024 * 1024 * 1024));
        set.rows.reverse();
        let reversed = select_rows(&set, Some(64 * 1024 * 1024 * 1024));
        assert_eq!(original, reversed);
    }

    #[test]
    fn exact_three_repetitions_are_required() {
        let mut set = winning_rows();
        set.rows.push(row(
            MeasurementSink::Display,
            RecommendedPayloadProfile::DefaultChunked64,
            4,
            100.0,
        ));
        assert_eq!(
            select_rows(&set, Some(8 * 1024 * 1024 * 1024))
                .decision
                .reason,
            RecommendationReason::DefaultMeasurementFailure
        );
    }

    #[test]
    fn summary_reports_ram_medians_ratios_gates_and_typed_reason() {
        let selection = select_rows(&winning_rows(), Some(64 * 1024 * 1024 * 1024));
        let outcome = OptimizerRunOutcome {
            input_basename: "clip.mcraw".to_string(),
            frames: OPTIMIZER_DECISION_FRAMES,
            progress_total_steps: 13,
            measurements: winning_rows(),
            recommendation: selection.recommendation(),
            selection,
        };
        let text = OptimizerSummaryPrinter::format(&outcome);
        assert!(text.contains("Total physical RAM: 68719476736 bytes (64.00 GiB)"));
        assert!(text.contains("default_chunked64 median:"));
        assert!(text.contains("offset_prefetch median:"));
        assert!(text.contains("throughput ratio:"));
        assert!(text.contains("Geometric Display/PIPE ratio:"));
        assert!(text.contains("Neither sink loses more than 5%: TRUE"));
        assert!(text.contains("Combined throughput gain greater than 10%: TRUE"));
        assert!(text.contains("Total RAM below 18253611008 bytes (17 GiB): FALSE"));
        assert!(text.contains("Recommendation: offset_prefetch"));
        assert!(text.contains("recommendation_reason: offset_throughput_gain"));
        assert!(text.contains("Reason: combined Display/PIPE throughput gain exceeded 10%"));
        assert!(text.contains("settings_file: pending user choice"));
        assert!(text.contains("Decision frames: 600\nGPU warmup frames: 600 unscored"));
        assert!(text.contains("Measured repetitions: 3"));
        assert!(!text.contains("private/source"));
        assert!(!text.to_ascii_lowercase().contains("p95"));
    }

    #[test]
    fn decision_gate_lines_render_true_false_and_unavailable_exactly() {
        let high_ram_win =
            selection_gate_lines(&select_rows(&winning_rows(), Some(64 * 1024 * 1024 * 1024)));
        assert!(high_ram_win.contains("Neither sink loses more than 5%: TRUE"));
        assert!(high_ram_win.contains("Combined throughput gain greater than 10%: TRUE"));
        assert!(high_ram_win.contains("Total RAM below 18253611008 bytes (17 GiB): FALSE"));

        let mut equal_rows = winning_rows();
        for row in &mut equal_rows.rows {
            row.fps = 100.0;
        }
        let low_ram_tie =
            selection_gate_lines(&select_rows(&equal_rows, Some(16 * 1024 * 1024 * 1024)));
        assert!(low_ram_tie.contains("Neither sink loses more than 5%: TRUE"));
        assert!(low_ram_tie.contains("Combined throughput gain greater than 10%: FALSE"));
        assert!(low_ram_tie.contains("Total RAM below 18253611008 bytes (17 GiB): TRUE"));

        let mut regression_rows = equal_rows.clone();
        for row in &mut regression_rows.rows {
            if row.profile == RecommendedPayloadProfile::OffsetPrefetch
                && row.sink == MeasurementSink::Display
            {
                row.fps = 94.0;
            }
        }
        let regression = selection_gate_lines(&select_rows(
            &regression_rows,
            Some(16 * 1024 * 1024 * 1024),
        ));
        assert!(regression.contains("Neither sink loses more than 5%: FALSE"));

        let unavailable = selection_gate_lines(&select_rows(&equal_rows, None));
        assert!(unavailable.contains("Total RAM below 18253611008 bytes (17 GiB): unavailable"));

        for gate_lines in [high_ram_win, low_ram_tie, regression, unavailable] {
            assert!(!gate_lines.contains("PASS"));
            assert!(!gate_lines.contains("FAIL"));
        }
    }

    #[test]
    fn default_summary_has_no_pending_choice_marker() {
        let mut rows = winning_rows();
        for row in &mut rows.rows {
            row.fps = 100.0;
        }
        let selection = select_rows(&rows, Some(64 * 1024 * 1024 * 1024));
        let outcome = OptimizerRunOutcome {
            input_basename: "clip.mcraw".to_string(),
            frames: OPTIMIZER_DECISION_FRAMES,
            progress_total_steps: 13,
            measurements: rows,
            recommendation: selection.recommendation(),
            selection,
        };
        let text = OptimizerSummaryPrinter::format(&outcome);
        assert!(text.contains("Recommendation: default_chunked64"));
        assert!(text.contains("settings_file: not written"));
        assert!(!text.contains("settings_file: pending user choice"));
    }

    fn non_default_recommendation() -> OptimizerRecommendation {
        OptimizerRecommendation {
            payload_profile: RecommendedPayloadProfile::OffsetPrefetch,
            decision_quality_ok: true,
            settings_written: false,
            reason: RecommendationReason::OffsetThroughputGain,
            reason_detail: None,
        }
    }

    fn default_recommendation() -> OptimizerRecommendation {
        OptimizerRecommendation {
            payload_profile: RecommendedPayloadProfile::DefaultChunked64,
            decision_quality_ok: true,
            settings_written: false,
            reason: RecommendationReason::DefaultRamNotLow,
            reason_detail: None,
        }
    }

    #[test]
    fn prompt_yes_writes_optimized_state_and_reports_path() {
        for input in ["y\n", "Y\n"] {
            let recommendation = non_default_recommendation();
            let mut reader = io::Cursor::new(input.as_bytes());
            let mut writer = Vec::new();
            let result = prompt_and_maybe_save_recommendation_with_io(
                &recommendation,
                &mut reader,
                &mut writer,
                |state| {
                    assert_eq!(state.payload_profile, PayloadProfile::OffsetPrefetch);
                    Ok(PathBuf::from("tmp/mcraw4vulkan/optimized-state.json"))
                },
            )
            .unwrap();
            let text = String::from_utf8(writer).unwrap();

            assert!(result.settings_written);
            assert!(text.contains("Save optimized payload profile?"));
            assert!(text.contains("optimized settings written:"));
            assert!(text.contains("optimized-state.json"));
        }
    }

    #[test]
    fn prompt_non_yes_or_eof_does_not_write_state() {
        for input in ["\n", "n\n", "anything else\n", ""] {
            let recommendation = non_default_recommendation();
            let mut reader = io::Cursor::new(input.as_bytes());
            let mut writer = Vec::new();
            let result = prompt_and_maybe_save_recommendation_with_io(
                &recommendation,
                &mut reader,
                &mut writer,
                |_| panic!("state must not be saved for non-yes input"),
            )
            .unwrap();
            let text = String::from_utf8(writer).unwrap();

            assert!(!result.settings_written);
            assert!(text.contains("No settings were written."));
        }
    }

    #[test]
    fn keep_default_recommendation_does_not_write_default_state_file() {
        let recommendation = default_recommendation();
        let mut reader = io::Cursor::new("y\n".as_bytes());
        let mut writer = Vec::new();
        let result = prompt_and_maybe_save_recommendation_with_io(
            &recommendation,
            &mut reader,
            &mut writer,
            |_| panic!("default recommendation must not be saved"),
        )
        .unwrap();
        let text = String::from_utf8(writer).unwrap();

        assert!(!result.settings_written);
        assert!(text.contains("policy recommends default_chunked64"));
        assert!(text.contains("No optimized settings file was written"));
    }

    #[test]
    fn non_decision_quality_recommendation_does_not_write_state() {
        let mut recommendation = non_default_recommendation();
        recommendation.decision_quality_ok = false;
        let mut reader = io::Cursor::new("y\n".as_bytes());
        let mut writer = Vec::new();
        let result = prompt_and_maybe_save_recommendation_with_io(
            &recommendation,
            &mut reader,
            &mut writer,
            |_| panic!("non-decision-quality recommendation must not be saved"),
        )
        .unwrap();
        let text = String::from_utf8(writer).unwrap();

        assert!(!result.settings_written);
        assert!(text.contains("did not complete all required rows"));
    }
}
