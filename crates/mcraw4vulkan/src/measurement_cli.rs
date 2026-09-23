use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use mcraw4vulkan_core::FrameNumber;
use mcraw4vulkan_gpu::GpuBackendPreference;
use mcraw4vulkan_mcrawcontainer::McrawContainer;
use mcraw4vulkan_vignette::PipeF32BayerCorrectionMode;

use crate::cli::{
    ComputeBackendChoice, DisplayCommand, PipeCommand, SettingsSourceChoice,
    VignetteCorrectionChoice, VsyncChoice,
};
use crate::measurement::display::DisplayMeasurementRunner;
use crate::measurement::pipe::PipeProducerMeasurementRunner;
use crate::measurement::types::{
    DisplayGpuExecutionProfile, DisplayMeasurementMode, DisplayMeasurementPolicy,
    DisplayMeasurementRequest, DisplayMeasurementRunSpec, PayloadReadPolicy, PayloadReadPolicyMode,
    PipeProducerMeasurementPolicy, PipeProducerMeasurementRequest, PipeProducerMeasurementRunSpec,
};
use crate::measurement_report::{
    MeasurementReport, ReportCommand, ReportCommon, ReportInput, ReportStatus,
};

pub const INTERNAL_MEASUREMENT_DEFAULT_FRAMES: usize = 600;
pub const OPTIMIZER_DECISION_MIN_FRAMES: usize = 600;
const DISPLAY_INTERNAL_WARMUP_DEFAULT_FRAMES: usize = 30;
const SINK_INTERNAL_WARMUP_DEFAULT_FRAMES: usize = 0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalMeasurementConfig {
    pub report_path: PathBuf,
    pub frames: usize,
    pub warmup_frames: usize,
    pub payload_profile: Option<InternalPayloadProfile>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InternalPayloadProfile {
    DefaultChunked64,
    OffsetPrefetch,
}

impl InternalPayloadProfile {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "default_chunked64" => Ok(Self::DefaultChunked64),
            "offset_prefetch" => Ok(Self::OffsetPrefetch),
            _ => bail!(
                "invalid --internal-payload-profile value {value:?}; expected default_chunked64 or offset_prefetch"
            ),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::DefaultChunked64 => "default_chunked64",
            Self::OffsetPrefetch => "offset_prefetch",
        }
    }

    pub fn payload_read_policy(self) -> PayloadReadPolicy {
        match self {
            Self::DefaultChunked64 => PayloadReadPolicy {
                mode: PayloadReadPolicyMode::ChunkedOffsetPrefetch,
                chunk_mib: 64,
            },
            Self::OffsetPrefetch => PayloadReadPolicy {
                mode: PayloadReadPolicyMode::OffsetPrefetch,
                chunk_mib: 64,
            },
        }
    }

    fn from_optimizer_payload_profile(profile: mcraw4vulkan_optimizer::PayloadProfile) -> Self {
        match profile {
            mcraw4vulkan_optimizer::PayloadProfile::DefaultChunked64 => Self::DefaultChunked64,
            mcraw4vulkan_optimizer::PayloadProfile::OffsetPrefetch => Self::OffsetPrefetch,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InternalMeasurementCommand {
    Display,
    Pipe,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InternalMeasurementFlagState {
    measure: bool,
    report_path: Option<PathBuf>,
    frames: Option<usize>,
    warmup_frames: Option<usize>,
    payload_profile: Option<InternalPayloadProfile>,
}

impl InternalMeasurementFlagState {
    fn any(&self) -> bool {
        self.measure
            || self.report_path.is_some()
            || self.frames.is_some()
            || self.warmup_frames.is_some()
            || self.payload_profile.is_some()
    }
}

pub fn is_internal_measurement_option(value: &str) -> bool {
    let flag = value.split_once('=').map_or(value, |(flag, _)| flag);
    matches!(
        flag,
        "--internal-measure"
            | "--internal-measure-report"
            | "--internal-measure-frames"
            | "--internal-measure-warmup-frames"
            | "--internal-payload-profile"
    )
}

pub fn consume_internal_measurement_flag(
    args: &[String],
    index: usize,
    state: &mut InternalMeasurementFlagState,
) -> Result<usize> {
    let value = &args[index];
    if value == "--internal-measure" {
        if state.measure {
            bail!("--internal-measure was supplied more than once");
        }
        state.measure = true;
        return Ok(index);
    }
    if let Some(raw) = value.strip_prefix("--internal-measure-report=") {
        set_once_path("--internal-measure-report", &mut state.report_path, raw)?;
        return Ok(index);
    }
    if value == "--internal-measure-report" {
        let next_index = index.saturating_add(1);
        let raw = next_internal_value(args, next_index, "--internal-measure-report")?;
        set_once_path("--internal-measure-report", &mut state.report_path, raw)?;
        return Ok(next_index);
    }
    if let Some(raw) = value.strip_prefix("--internal-measure-frames=") {
        set_once_usize(
            "--internal-measure-frames",
            &mut state.frames,
            parse_positive_usize(raw, "--internal-measure-frames")?,
        )?;
        return Ok(index);
    }
    if value == "--internal-measure-frames" {
        let next_index = index.saturating_add(1);
        let raw = next_internal_value(args, next_index, "--internal-measure-frames")?;
        set_once_usize(
            "--internal-measure-frames",
            &mut state.frames,
            parse_positive_usize(raw, "--internal-measure-frames")?,
        )?;
        return Ok(next_index);
    }
    if let Some(raw) = value.strip_prefix("--internal-measure-warmup-frames=") {
        set_once_usize(
            "--internal-measure-warmup-frames",
            &mut state.warmup_frames,
            parse_nonnegative_usize(raw, "--internal-measure-warmup-frames")?,
        )?;
        return Ok(index);
    }
    if value == "--internal-measure-warmup-frames" {
        let next_index = index.saturating_add(1);
        let raw = next_internal_value(args, next_index, "--internal-measure-warmup-frames")?;
        set_once_usize(
            "--internal-measure-warmup-frames",
            &mut state.warmup_frames,
            parse_nonnegative_usize(raw, "--internal-measure-warmup-frames")?,
        )?;
        return Ok(next_index);
    }
    if let Some(raw) = value.strip_prefix("--internal-payload-profile=") {
        set_once_payload_profile(raw, state)?;
        return Ok(index);
    }
    if value == "--internal-payload-profile" {
        let next_index = index.saturating_add(1);
        let raw = next_internal_value(args, next_index, "--internal-payload-profile")?;
        set_once_payload_profile(raw, state)?;
        return Ok(next_index);
    }
    bail!("unknown internal measurement flag {value:?}")
}

pub fn finalize_internal_measurement(
    command: InternalMeasurementCommand,
    state: InternalMeasurementFlagState,
) -> Result<Option<InternalMeasurementConfig>> {
    if !state.any() {
        return Ok(None);
    }
    if !state.measure {
        bail!("hidden internal measurement flags require --internal-measure");
    }
    let report_path = state
        .report_path
        .ok_or_else(|| anyhow!("--internal-measure requires --internal-measure-report FILE"))?;
    let frames = state.frames.unwrap_or(INTERNAL_MEASUREMENT_DEFAULT_FRAMES);
    let warmup_frames = state.warmup_frames.unwrap_or(match command {
        InternalMeasurementCommand::Display => DISPLAY_INTERNAL_WARMUP_DEFAULT_FRAMES,
        InternalMeasurementCommand::Pipe => SINK_INTERNAL_WARMUP_DEFAULT_FRAMES,
    });

    Ok(Some(InternalMeasurementConfig {
        report_path,
        frames,
        warmup_frames,
        payload_profile: state.payload_profile,
    }))
}

pub fn run_display_internal_measurement(command: DisplayCommand) -> Result<()> {
    let config = command
        .internal_measurement
        .clone()
        .context("display internal measurement config missing")?;
    let payload_profile = effective_internal_payload_profile(command.settings, &config);

    let selection = match select_internal_frames(&command.input, config.frames) {
        Ok(selection) => selection,
        Err(error) => {
            let message = sanitize_error_message(&error, &command.input);
            return write_display_error_report(
                &command,
                &config,
                ReportStatus::Failed,
                &message,
                None,
            );
        }
    };
    let mode = match (command.backend, command.vignette) {
        (ComputeBackendChoice::Cpu, _) => DisplayMeasurementMode::Cpu,
        (ComputeBackendChoice::Gpu, VignetteCorrectionChoice::NoCorrection) => {
            DisplayMeasurementMode::Gpu
        }
        (ComputeBackendChoice::Gpu, VignetteCorrectionChoice::WithCorrection) => {
            DisplayMeasurementMode::GpuVignette
        }
    };
    let request = DisplayMeasurementRequest {
        run: DisplayMeasurementRunSpec {
            input_path: command.input.clone(),
            start_frame: 0,
            stride: 1,
            frames_requested: config.frames,
            selected_frames: selection.selected_frames.clone(),
            display_warmup_frames: config.warmup_frames,
        },
        payload: payload_profile.payload_read_policy(),
        display: DisplayMeasurementPolicy::new(mode),
        gpu: crate::measurement::types::GpuMeasurementPolicy {
            backend_preference: match command.backend {
                ComputeBackendChoice::Gpu => GpuBackendPreference::VulkanOnly,
                ComputeBackendChoice::Cpu => GpuBackendPreference::Auto,
            },
            execution_profile: DisplayGpuExecutionProfile::Current,
        },
    };

    match DisplayMeasurementRunner::new().measure(request) {
        Ok(result) => {
            let mut common = display_common_from_result(&command, &config, &selection, &result);
            common.payload_profile_requested = Some(payload_profile.label().to_string());
            let report = MeasurementReport::display_success(common, &result);
            report.write_to_path(&config.report_path)?;
            if result.status == crate::measurement::types::MeasurementStatus::Ok {
                Ok(())
            } else {
                bail!(
                    "display internal measurement failed: {}",
                    result
                        .failure_stage
                        .as_deref()
                        .unwrap_or("measurement runner reported failure")
                )
            }
        }
        Err(error) => {
            let message = sanitize_error_message(&error, &command.input);
            write_display_error_report(
                &command,
                &config,
                ReportStatus::Failed,
                &message,
                Some(&selection),
            )
        }
    }
}

pub fn run_pipe_internal_measurement(command: PipeCommand) -> Result<()> {
    let config = command
        .internal_measurement
        .clone()
        .context("pipe internal measurement config missing")?;
    let payload_profile = effective_internal_payload_profile(command.settings, &config);
    if command.backend == ComputeBackendChoice::Cpu {
        return write_pipe_error_report(
            &command,
            &config,
            ReportStatus::Unsupported,
            "CPU PIPE output is not implemented; use --gpu for internal PIPE measurement",
            None,
        );
    }

    let selection = match select_internal_frames(&command.input, config.frames) {
        Ok(selection) => selection,
        Err(error) => {
            let message = sanitize_error_message(&error, &command.input);
            return write_pipe_error_report(
                &command,
                &config,
                ReportStatus::Failed,
                &message,
                None,
            );
        }
    };
    let mut policy =
        PipeProducerMeasurementPolicy::canonical_discard(GpuBackendPreference::VulkanOnly);
    policy.correction_mode = match command.vignette {
        VignetteCorrectionChoice::NoCorrection => PipeF32BayerCorrectionMode::IdentitySpatialGain,
        VignetteCorrectionChoice::WithCorrection => PipeF32BayerCorrectionMode::MotionCamSpatial,
    };
    let request = PipeProducerMeasurementRequest {
        run: PipeProducerMeasurementRunSpec {
            input_path: command.input.clone(),
            start_frame: 0,
            stride: 1,
            warmup_frames: config.warmup_frames,
            frames_requested: config.frames,
            selected_frames: selection.selected_frames.clone(),
        },
        payload: payload_profile.payload_read_policy(),
        policy,
    };

    match PipeProducerMeasurementRunner::new().measure(request) {
        Ok(result) => {
            let mut common = pipe_common_from_result(&command, &config, &selection, &result);
            common.payload_profile_requested = Some(payload_profile.label().to_string());
            let report = MeasurementReport::pipe_success(common, &result);
            report.write_to_path(&config.report_path)?;
            if result.status == crate::measurement::types::MeasurementStatus::Ok {
                Ok(())
            } else {
                bail!(
                    "PIPE internal measurement failed: {}",
                    result
                        .failure_stage
                        .as_deref()
                        .unwrap_or("measurement runner reported failure")
                )
            }
        }
        Err(error) => {
            let message = sanitize_error_message(&error, &command.input);
            write_pipe_error_report(
                &command,
                &config,
                ReportStatus::Failed,
                &message,
                Some(&selection),
            )
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InternalFrameSelection {
    frames_available: usize,
    selected_frames: Vec<FrameNumber>,
    truncated_by_clip_length: bool,
}

impl InternalFrameSelection {
    fn frames_selected(&self) -> usize {
        self.selected_frames.len()
    }
}

fn select_internal_frames(input: &Path, frames_requested: usize) -> Result<InternalFrameSelection> {
    let container = McrawContainer::open(input)
        .with_context(|| "failed to open input while preparing internal measurement frames")?;
    let frames_available = container.frame_count();
    if frames_available == 0 {
        bail!("internal measurement requires at least one available frame; clip has zero frames");
    }
    let frames_selected = frames_requested.min(frames_available);
    let selected_frames = (0..frames_selected)
        .map(|frame| FrameNumber(frame as u32))
        .collect();
    Ok(InternalFrameSelection {
        frames_available,
        selected_frames,
        truncated_by_clip_length: frames_available < frames_requested,
    })
}

fn display_common_from_result(
    command: &DisplayCommand,
    config: &InternalMeasurementConfig,
    selection: &InternalFrameSelection,
    result: &crate::measurement::types::MeasurementResult,
) -> ReportCommon {
    let mut common = common_base(
        ReportCommand::Display,
        result.sink.label().to_string(),
        command.input.as_path(),
        command.settings,
        command.backend,
        command.vignette,
        config,
    );
    common.status = report_status_from_measurement(result.status);
    common.frames_available = Some(selection.frames_available);
    common.frames_selected = Some(selection.frames_selected());
    common.frames_measured = Some(result.frames.frames_processed);
    common.warmup_frames_processed = Some(result.display_warmup.frames_processed);
    common.measurement_truncated_by_clip_length = Some(selection.truncated_by_clip_length);
    apply_decision_quality(&mut common, selection, result.frames.frames_processed);
    common.backend_effective = Some(backend_label(command.backend).to_string());
    common.payload_profile_effective = true;
    common.setup_s = Some(result.timing.setup_s());
    common.warmup_s = Some(result.display_warmup.duration_s());
    common.run_s = Some(result.timing.run_s());
    common.flush_s = Some(result.timing.flush_s());
    let total_s = result
        .timing
        .setup
        .saturating_add(result.display_warmup.duration)
        .saturating_add(result.timing.run)
        .saturating_add(result.timing.flush)
        .as_secs_f64();
    common.total_s = Some(total_s);
    common.fps = Some(fps(result.frames.frames_processed, result.timing.run_s()));
    common.total_fps = Some(fps(result.frames.frames_processed, total_s));
    common.bytes_output_logical = Some(result.display.output_bytes);
    common.bytes_written_physical = Some(0);
    common.full_frame_readback_performed = Some(false);
    common.notes.extend(result.notes.clone());
    if command.vsync == VsyncChoice::Vsync {
        common
            .notes
            .push("display internal measurement uses a no-window runner; real surface vsync is not presented".to_string());
    }
    common
}

fn pipe_common_from_result(
    command: &PipeCommand,
    config: &InternalMeasurementConfig,
    selection: &InternalFrameSelection,
    result: &crate::measurement::types::PipeProducerMeasurementResult,
) -> ReportCommon {
    let mut common = common_base(
        ReportCommand::Pipe,
        result.sink.label().to_string(),
        command.input.as_path(),
        command.settings,
        command.backend,
        command.vignette,
        config,
    );
    common.status = report_status_from_measurement(result.status);
    common.frames_available = Some(selection.frames_available);
    common.frames_selected = Some(selection.frames_selected());
    common.frames_measured = Some(result.frames.frames_processed);
    common.warmup_frames_processed = Some(result.metrics.warmup_frames_processed);
    common.measurement_truncated_by_clip_length = Some(selection.truncated_by_clip_length);
    apply_decision_quality(&mut common, selection, result.frames.frames_processed);
    common.backend_effective = Some("gpu".to_string());
    common.payload_profile_effective = true;
    common.setup_s = Some(result.timing.setup_s());
    common.warmup_s = Some(result.metrics.warmup_s);
    common.run_s = Some(result.timing.run_s());
    common.flush_s = Some(result.timing.flush_s());
    common.total_s = Some(result.timing.total_s() + result.metrics.warmup_s);
    common.fps = Some(fps(result.frames.frames_processed, result.timing.run_s()));
    common.total_fps = Some(fps(
        result.frames.frames_processed,
        result.timing.total_s() + result.metrics.warmup_s,
    ));
    common.bytes_output_logical = Some(result.metrics.output_bytes_expected);
    common.bytes_written_physical = Some(0);
    common.full_frame_readback_performed = Some(true);
    common.notes.extend(result.notes.clone());
    common
}

fn common_base(
    command: ReportCommand,
    sink_name: String,
    input: &Path,
    settings: SettingsSourceChoice,
    backend: ComputeBackendChoice,
    vignette: VignetteCorrectionChoice,
    config: &InternalMeasurementConfig,
) -> ReportCommon {
    ReportCommon {
        command,
        sink_name,
        status: ReportStatus::Failed,
        error_message: None,
        input: ReportInput::from_path(input),
        frames_requested: config.frames,
        frames_available: None,
        frames_selected: None,
        frames_measured: None,
        warmup_frames_requested: config.warmup_frames,
        warmup_frames_processed: None,
        measurement_truncated_by_clip_length: None,
        decision_quality_min_frames: OPTIMIZER_DECISION_MIN_FRAMES,
        decision_quality_ok: false,
        start_frame: 0,
        stride: 1,
        settings_source: settings_source_label(settings).to_string(),
        backend: backend_label(backend).to_string(),
        backend_effective: None,
        vignette_correction: vignette_label(vignette).to_string(),
        payload_profile_requested: config
            .payload_profile
            .map(|profile| profile.label().to_string()),
        payload_profile_effective: false,
        setup_s: None,
        warmup_s: None,
        run_s: None,
        flush_s: None,
        total_s: None,
        fps: None,
        total_fps: None,
        bytes_output_logical: None,
        bytes_written_physical: None,
        full_frame_readback_performed: None,
        notes: Vec::new(),
    }
}

// The decision gate includes runs exactly at the minimum frame count. Reports
// below that threshold cannot drive optimizer policy.
fn apply_decision_quality(
    common: &mut ReportCommon,
    selection: &InternalFrameSelection,
    frames_measured: usize,
) {
    common.decision_quality_ok = frames_measured >= common.decision_quality_min_frames;
    if selection.truncated_by_clip_length {
        common.notes.push(format!(
            "warning: clip has fewer frames than requested; requested={} available={} selected={}",
            common.frames_requested,
            selection.frames_available,
            selection.frames_selected()
        ));
    }
    if frames_measured < common.decision_quality_min_frames {
        common.notes.push(format!(
            "warning: measurement used fewer than {} frames; this report does not meet the optimizer decision-quality gate",
            common.decision_quality_min_frames
        ));
    }
}

fn write_display_error_report(
    command: &DisplayCommand,
    config: &InternalMeasurementConfig,
    status: ReportStatus,
    message: &str,
    selection: Option<&InternalFrameSelection>,
) -> Result<()> {
    let mut common = common_base(
        ReportCommand::Display,
        display_sink_name(command).to_string(),
        command.input.as_path(),
        command.settings,
        command.backend,
        command.vignette,
        config,
    );
    apply_error_common(&mut common, status, message, selection);
    common.backend_effective = Some(backend_label(command.backend).to_string());
    common.payload_profile_effective = true;
    let report = MeasurementReport::display_error(common);
    report.write_to_path(&config.report_path)?;
    bail!("{message}")
}

fn write_pipe_error_report(
    command: &PipeCommand,
    config: &InternalMeasurementConfig,
    status: ReportStatus,
    message: &str,
    selection: Option<&InternalFrameSelection>,
) -> Result<()> {
    let mut common = common_base(
        ReportCommand::Pipe,
        pipe_sink_name(command).to_string(),
        command.input.as_path(),
        command.settings,
        command.backend,
        command.vignette,
        config,
    );
    apply_error_common(&mut common, status, message, selection);
    common.backend_effective = if command.backend == ComputeBackendChoice::Gpu {
        Some("gpu".to_string())
    } else {
        None
    };
    common.payload_profile_effective = command.backend == ComputeBackendChoice::Gpu;
    let report = MeasurementReport::pipe_error(common);
    report.write_to_path(&config.report_path)?;
    bail!("{message}")
}

fn apply_error_common(
    common: &mut ReportCommon,
    status: ReportStatus,
    message: &str,
    selection: Option<&InternalFrameSelection>,
) {
    common.status = status;
    common.error_message = Some(message.to_string());
    common.warmup_frames_processed = Some(0);
    common.frames_measured = Some(0);
    common.decision_quality_ok = false;
    common.setup_s = Some(0.0);
    common.warmup_s = Some(0.0);
    common.run_s = Some(0.0);
    common.flush_s = Some(0.0);
    common.total_s = Some(0.0);
    common.fps = Some(0.0);
    common.total_fps = Some(0.0);
    common.bytes_output_logical = Some(0);
    common.bytes_written_physical = Some(0);
    common.full_frame_readback_performed = Some(false);
    if let Some(selection) = selection {
        common.frames_available = Some(selection.frames_available);
        common.frames_selected = Some(selection.frames_selected());
        common.measurement_truncated_by_clip_length = Some(selection.truncated_by_clip_length);
        apply_decision_quality(common, selection, 0);
    }
    common.notes.push(message.to_string());
}

fn report_status_from_measurement(
    status: crate::measurement::types::MeasurementStatus,
) -> ReportStatus {
    match status {
        crate::measurement::types::MeasurementStatus::Ok => ReportStatus::Ok,
        crate::measurement::types::MeasurementStatus::Failed => ReportStatus::Failed,
    }
}

fn next_internal_value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str> {
    let Some(value) = args.get(index) else {
        bail!("{flag} requires a value")
    };
    if value.starts_with('-') {
        bail!("{flag} requires a value, got option {value:?}")
    }
    Ok(value)
}

fn set_once_path(flag: &str, target: &mut Option<PathBuf>, raw: &str) -> Result<()> {
    if target.is_some() {
        bail!("{flag} was supplied more than once");
    }
    if raw.is_empty() {
        bail!("{flag} requires a path");
    }
    *target = Some(PathBuf::from(raw));
    Ok(())
}

fn set_once_usize(flag: &str, target: &mut Option<usize>, value: usize) -> Result<()> {
    if target.is_some() {
        bail!("{flag} was supplied more than once");
    }
    *target = Some(value);
    Ok(())
}

fn set_once_payload_profile(raw: &str, state: &mut InternalMeasurementFlagState) -> Result<()> {
    if state.payload_profile.is_some() {
        bail!("--internal-payload-profile was supplied more than once");
    }
    state.payload_profile = Some(InternalPayloadProfile::parse(raw)?);
    Ok(())
}

fn parse_positive_usize(raw: &str, flag: &str) -> Result<usize> {
    let value = parse_nonnegative_usize(raw, flag)?;
    if value == 0 {
        bail!("{flag} must be positive");
    }
    Ok(value)
}

fn parse_nonnegative_usize(raw: &str, flag: &str) -> Result<usize> {
    raw.parse::<usize>()
        .with_context(|| format!("{flag} requires an integer value"))
}

fn settings_source_label(settings: SettingsSourceChoice) -> &'static str {
    match settings {
        SettingsSourceChoice::Default => "default",
        SettingsSourceChoice::Optimized => "optimized",
    }
}

fn effective_internal_payload_profile(
    settings: SettingsSourceChoice,
    config: &InternalMeasurementConfig,
) -> InternalPayloadProfile {
    if let Some(profile) = config.payload_profile {
        return profile;
    }
    let selection = match settings {
        SettingsSourceChoice::Default => mcraw4vulkan_optimizer::SettingsSourceSelection::Default,
        SettingsSourceChoice::Optimized => {
            mcraw4vulkan_optimizer::SettingsSourceSelection::Optimized
        }
    };
    let effective = mcraw4vulkan_optimizer::resolve_effective_settings(selection);
    if let Some(warning) = effective.warning() {
        eprintln!("warning: {warning}");
    }
    InternalPayloadProfile::from_optimizer_payload_profile(effective.payload_profile)
}

fn backend_label(backend: ComputeBackendChoice) -> &'static str {
    match backend {
        ComputeBackendChoice::Gpu => "gpu",
        ComputeBackendChoice::Cpu => "cpu",
    }
}

fn vignette_label(vignette: VignetteCorrectionChoice) -> &'static str {
    match vignette {
        VignetteCorrectionChoice::NoCorrection => "none",
        VignetteCorrectionChoice::WithCorrection => "with",
    }
}

fn display_sink_name(command: &DisplayCommand) -> &'static str {
    match (command.backend, command.vignette) {
        (ComputeBackendChoice::Cpu, _) => "cpu_display_no_vsync_proxy",
        (ComputeBackendChoice::Gpu, VignetteCorrectionChoice::NoCorrection) => {
            "gpu_display_no_vsync_proxy"
        }
        (ComputeBackendChoice::Gpu, VignetteCorrectionChoice::WithCorrection) => {
            "gpu_gpu_vignette_display_no_vsync_proxy"
        }
    }
}

fn pipe_sink_name(command: &PipeCommand) -> &'static str {
    match command.vignette {
        VignetteCorrectionChoice::NoCorrection => "gpu_pipe_yuv444p12le_raw_no_vignette",
        VignetteCorrectionChoice::WithCorrection => "gpu_gpu_vignette_pipe_yuv444p12le_raw",
    }
}

fn sanitize_error_message(error: &anyhow::Error, input: &Path) -> String {
    let mut message = error.to_string();
    let display_path = input.to_string_lossy();
    if !display_path.is_empty() {
        message = message.replace(display_path.as_ref(), "<input>");
    }
    let debug_path = format!("{input:?}");
    if !debug_path.is_empty() {
        message = message.replace(&debug_path, "\"<input>\"");
    }
    message
}

fn fps(frames: usize, seconds: f64) -> f64 {
    if frames == 0 || seconds <= 0.0 {
        0.0
    } else {
        frames as f64 / seconds
    }
}
