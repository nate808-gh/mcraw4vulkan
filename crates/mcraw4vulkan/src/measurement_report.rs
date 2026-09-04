use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Result;
use mcraw4vulkan_render::Yuv444p12lePackPolicy;
use mcraw4vulkan_vignette::PipeF32BayerCorrectionMode;
use serde_json::{Map, Value, json};

use crate::measurement::types::{
    MeasurementResult, PipeByteIdentityRow, PipeProducerMeasurementResult,
};
use crate::pipe_contract::{
    PIPE_ALPHA, PIPE_BIT_ALIGNMENT, PIPE_CHROMA_SAMPLING, PIPE_COLOR_PRIMARIES, PIPE_COLOR_RANGE,
    PIPE_COLOR_TRANSFER, PIPE_ENDIANNESS_LABEL, PIPE_MATRIX_COEFFICIENTS, PIPE_PIXEL_FORMAT,
    PIPE_SAMPLE_RANGE, PIPE_SCALE_APPLICATION_POINT,
};

pub const MEASUREMENT_REPORT_VERSION: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportCommand {
    Display,
    Pipe,
}

impl ReportCommand {
    fn label(self) -> &'static str {
        match self {
            Self::Display => "display",
            Self::Pipe => "pipe",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportStatus {
    Ok,
    Failed,
    Unsupported,
}

impl ReportStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Failed => "failed",
            Self::Unsupported => "unsupported",
        }
    }
}

// The report's input identity keeps only display names, omitting the source
// directory from the structured input fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportInput {
    pub input_basename: String,
    pub input_stem: String,
}

impl ReportInput {
    pub fn from_path(path: &Path) -> Self {
        let input_basename = path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "unknown".to_string());
        let input_stem = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .filter(|stem| !stem.is_empty())
            .unwrap_or_else(|| input_basename.clone());
        Self {
            input_basename,
            input_stem,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReportCommon {
    pub command: ReportCommand,
    pub sink_name: String,
    pub status: ReportStatus,
    pub error_message: Option<String>,
    pub input: ReportInput,
    pub frames_requested: usize,
    pub frames_available: Option<usize>,
    pub frames_selected: Option<usize>,
    pub frames_measured: Option<usize>,
    pub warmup_frames_requested: usize,
    pub warmup_frames_processed: Option<usize>,
    pub measurement_truncated_by_clip_length: Option<bool>,
    pub decision_quality_min_frames: usize,
    pub decision_quality_ok: bool,
    pub start_frame: usize,
    pub stride: usize,
    pub settings_source: String,
    pub backend: String,
    pub backend_effective: Option<String>,
    pub vignette_correction: String,
    pub payload_profile_requested: Option<String>,
    pub payload_profile_effective: bool,
    pub setup_s: Option<f64>,
    pub warmup_s: Option<f64>,
    pub run_s: Option<f64>,
    pub flush_s: Option<f64>,
    pub total_s: Option<f64>,
    pub fps: Option<f64>,
    pub total_fps: Option<f64>,
    pub bytes_output_logical: Option<u64>,
    pub bytes_written_physical: Option<u64>,
    pub full_frame_readback_performed: Option<bool>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementReport {
    value: Value,
}

impl MeasurementReport {
    pub fn display_success(common: ReportCommon, result: &MeasurementResult) -> Self {
        let mut root = common_root(&common);
        root.insert(
            "display".to_string(),
            display_object(
                Some(result.display.preview_width),
                Some(result.display.preview_height),
                result.display.output_bytes,
                false,
                common.warmup_frames_processed,
                common.frames_measured,
            ),
        );
        Self {
            value: Value::Object(root),
        }
    }

    pub fn display_error(common: ReportCommon) -> Self {
        let mut root = common_root(&common);
        root.insert(
            "display".to_string(),
            display_object(
                None,
                None,
                0,
                false,
                common.warmup_frames_processed,
                common.frames_measured,
            ),
        );
        Self {
            value: Value::Object(root),
        }
    }

    pub fn pipe_success(common: ReportCommon, result: &PipeProducerMeasurementResult) -> Self {
        let mut root = common_root(&common);
        let mut pipe =
            pipe_contract_fields(Some(result.metrics.byte_identity.meaningful_low_12_bits_ok));
        pipe.extend(pipe_success_runtime_fields(result));
        root.insert("pipe".to_string(), Value::Object(pipe));
        Self {
            value: Value::Object(root),
        }
    }

    pub fn pipe_error(common: ReportCommon) -> Self {
        let mut root = common_root(&common);
        let mut pipe = pipe_contract_fields(None);
        pipe.extend(pipe_error_runtime_fields());
        root.insert("pipe".to_string(), Value::Object(pipe));
        Self {
            value: Value::Object(root),
        }
    }

    pub fn write_to_path(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &self.value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(())
    }

    #[cfg(test)]
    fn value(&self) -> &Value {
        &self.value
    }
}

fn json_object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(object) => object,
        _ => unreachable!("json! object literal produced a non-object"),
    }
}

fn pipe_contract_fields(upper_four_bits_zero: Option<bool>) -> Map<String, Value> {
    json_object(json!({
        "pixel_format": PIPE_PIXEL_FORMAT,
        "plane_order": Yuv444p12lePackPolicy::PLANE_ORDER,
        "endianness": PIPE_ENDIANNESS_LABEL,
        "bits_per_channel": Yuv444p12lePackPolicy::MEANINGFUL_BITS_PER_SAMPLE,
        "storage_bits_per_sample": Yuv444p12lePackPolicy::STORAGE_BITS_PER_SAMPLE,
        "meaningful_bits_per_sample": Yuv444p12lePackPolicy::MEANINGFUL_BITS_PER_SAMPLE,
        "bit_alignment": PIPE_BIT_ALIGNMENT,
        "upper_four_bits_zero": option_bool(upper_four_bits_zero),
        "sample_range": PIPE_SAMPLE_RANGE,
        "color_range": PIPE_COLOR_RANGE,
        "range_is_limited": true,
        "color_primaries": PIPE_COLOR_PRIMARIES,
        "color_transfer": PIPE_COLOR_TRANSFER,
        "matrix_coefficients": PIPE_MATRIX_COEFFICIENTS,
        "chroma_sampling": PIPE_CHROMA_SAMPLING,
        "alpha": PIPE_ALPHA,
        "final_code_bounds": [Yuv444p12lePackPolicy::MIN_CODE, Yuv444p12lePackPolicy::MAX_CODE],
        "linear_signal_scale_num": Yuv444p12lePackPolicy::LINEAR_SIGNAL_SCALE_NUMERATOR,
        "linear_signal_scale_den": Yuv444p12lePackPolicy::LINEAR_SIGNAL_SCALE_DENOMINATOR,
        "linear_signal_scale_stops": Yuv444p12lePackPolicy::LINEAR_SIGNAL_SCALE_STOPS,
        "scale_application_point": PIPE_SCALE_APPLICATION_POINT,
    }))
}

fn pipe_success_runtime_fields(result: &PipeProducerMeasurementResult) -> Map<String, Value> {
    let mut fields = json_object(json!({
        "frame_width": result.input.frame_width,
        "frame_height": result.input.frame_height,
        "bytes_per_frame": result.metrics.bytes_per_frame_expected,
        "expected_total_video_bytes": result.metrics.output_bytes_expected,
        "output_target": "discard",
        "stdout_used": false,
        "byte_clean_stdout": true,
        "render_mode": pipe_render_mode_label(result.metrics.correction_mode),
        "producer_fps": result.metrics.fps_no_write,
        "writer_s": result.metrics.writer_s,
        "writer_flush_s": result.metrics.writer_flush_s,
        "mapped_consume_s": result.metrics.mapped_consume_s,
        "pipe_full_byte_consume_in_run": false,
    }));
    fields.extend(json_object(json!({
        "byte_identity_status": result.metrics.byte_identity.status(),
        "byte_identity_frames_checked": result.metrics.byte_identity.frames_checked,
        "byte_identity_validation_frames_checked": result.metrics.byte_identity.validation_frames_checked,
        "byte_identity_validation_bytes_sampled": result.metrics.byte_identity.validation_bytes_sampled,
        "byte_identity_output_byte_count_ok": result.metrics.byte_identity.output_byte_count_ok,
        "byte_identity_frame_byte_count_ok": result.metrics.byte_identity.frame_byte_count_ok,
        "byte_identity_plane_order_ok": result.metrics.byte_identity.plane_order_ok,
        "byte_identity_little_endian_ok": result.metrics.byte_identity.little_endian_ok,
        "byte_identity_meaningful_low_12_bits_ok": result.metrics.byte_identity.meaningful_low_12_bits_ok,
        "byte_identity_code_bounds_ok": result.metrics.byte_identity.code_bounds_ok,
        "byte_identity_mismatches": result.metrics.byte_identity.mismatches,
        "byte_identity_first_mismatch_offset": option_usize(result.metrics.byte_identity.first_mismatch_offset),
        "byte_identity_rows": pipe_byte_identity_rows_value(&result.byte_identity_rows),
    })));
    fields
}

fn pipe_error_runtime_fields() -> Map<String, Value> {
    let mut fields = json_object(json!({
        "frame_width": null,
        "frame_height": null,
        "bytes_per_frame": null,
        "expected_total_video_bytes": null,
        "output_target": "discard",
        "stdout_used": false,
        "byte_clean_stdout": true,
        "render_mode": null,
        "producer_fps": null,
        "writer_s": null,
        "writer_flush_s": null,
        "mapped_consume_s": null,
        "pipe_full_byte_consume_in_run": false,
    }));
    fields.extend(json_object(json!({
        "byte_identity_status": null,
        "byte_identity_frames_checked": null,
        "byte_identity_validation_frames_checked": null,
        "byte_identity_validation_bytes_sampled": null,
        "byte_identity_output_byte_count_ok": null,
        "byte_identity_frame_byte_count_ok": null,
        "byte_identity_plane_order_ok": null,
        "byte_identity_little_endian_ok": null,
        "byte_identity_meaningful_low_12_bits_ok": null,
        "byte_identity_code_bounds_ok": null,
        "byte_identity_mismatches": null,
        "byte_identity_first_mismatch_offset": null,
        "byte_identity_rows": [],
    })));
    fields
}

fn pipe_render_mode_label(correction_mode: PipeF32BayerCorrectionMode) -> &'static str {
    match correction_mode {
        PipeF32BayerCorrectionMode::IdentitySpatialGain => "no_vignette",
        PipeF32BayerCorrectionMode::MotionCamSpatial => "vignette",
    }
}

fn common_root(common: &ReportCommon) -> Map<String, Value> {
    let mut root = Map::new();
    root.insert(
        "measurement_report_version".to_string(),
        json!(MEASUREMENT_REPORT_VERSION),
    );
    root.insert("created_by".to_string(), json!("mcraw4vulkan"));
    root.insert(
        "mcraw4vulkan_version".to_string(),
        json!(env!("CARGO_PKG_VERSION")),
    );
    root.insert(
        "mcraw4vulkan_commit".to_string(),
        option_string(option_env!("MCRAW4VULKAN_COMMIT").map(str::to_string)),
    );
    root.insert("command".to_string(), json!(common.command.label()));
    root.insert("sink_name".to_string(), json!(common.sink_name));
    root.insert("status".to_string(), json!(common.status.label()));
    root.insert(
        "error_message".to_string(),
        option_string(common.error_message.clone()),
    );
    root.insert(
        "input_basename".to_string(),
        json!(common.input.input_basename),
    );
    root.insert("input_stem".to_string(), json!(common.input.input_stem));
    root.insert("input_path_recorded".to_string(), json!(false));
    root.insert(
        "frames_requested".to_string(),
        json!(common.frames_requested),
    );
    root.insert(
        "frames_available".to_string(),
        option_usize(common.frames_available),
    );
    root.insert(
        "frames_selected".to_string(),
        option_usize(common.frames_selected),
    );
    root.insert(
        "frames_measured".to_string(),
        option_usize(common.frames_measured),
    );
    root.insert(
        "warmup_frames_requested".to_string(),
        json!(common.warmup_frames_requested),
    );
    root.insert(
        "warmup_frames_processed".to_string(),
        option_usize(common.warmup_frames_processed),
    );
    root.insert(
        "measurement_truncated_by_clip_length".to_string(),
        option_bool(common.measurement_truncated_by_clip_length),
    );
    root.insert(
        "decision_quality_min_frames".to_string(),
        json!(common.decision_quality_min_frames),
    );
    root.insert(
        "decision_quality_ok".to_string(),
        json!(common.decision_quality_ok),
    );
    root.insert("start_frame".to_string(), json!(common.start_frame));
    root.insert("stride".to_string(), json!(common.stride));
    root.insert("settings_source".to_string(), json!(common.settings_source));
    root.insert("backend".to_string(), json!(common.backend));
    root.insert(
        "backend_effective".to_string(),
        option_string(common.backend_effective.clone()),
    );
    root.insert(
        "vignette_correction".to_string(),
        json!(common.vignette_correction),
    );
    root.insert(
        "payload_profile_requested".to_string(),
        option_string(common.payload_profile_requested.clone()),
    );
    root.insert(
        "payload_profile_effective".to_string(),
        json!(common.payload_profile_effective),
    );
    root.insert("setup_s".to_string(), option_f64(common.setup_s));
    root.insert("warmup_s".to_string(), option_f64(common.warmup_s));
    root.insert("run_s".to_string(), option_f64(common.run_s));
    root.insert("flush_s".to_string(), option_f64(common.flush_s));
    root.insert("total_s".to_string(), option_f64(common.total_s));
    root.insert("fps".to_string(), option_f64(common.fps));
    root.insert("total_fps".to_string(), option_f64(common.total_fps));
    root.insert(
        "bytes_output_logical".to_string(),
        option_u64(common.bytes_output_logical),
    );
    root.insert(
        "bytes_written_physical".to_string(),
        option_u64(common.bytes_written_physical),
    );
    root.insert(
        "full_frame_readback_performed".to_string(),
        option_bool(common.full_frame_readback_performed),
    );
    root.insert("notes".to_string(), json!(common.notes));
    root
}

fn display_object(
    preview_width: Option<u32>,
    preview_height: Option<u32>,
    output_bytes: u64,
    full_frame_readback_performed: bool,
    warmup_frames_processed: Option<usize>,
    measured_frames_processed: Option<usize>,
) -> Value {
    json!({
        "preview_width": option_u32(preview_width),
        "preview_height": option_u32(preview_height),
        "output_bytes": output_bytes,
        "full_frame_readback_performed": full_frame_readback_performed,
        "overlay_applicable": false,
        "live_window": false,
        "vsync_applicable": false,
        "warmup_frames_processed": option_usize(warmup_frames_processed),
        "measured_frames_processed": option_usize(measured_frames_processed),
    })
}

fn pipe_byte_identity_rows_value(rows: &[PipeByteIdentityRow]) -> Value {
    Value::Array(
        rows.iter()
            .map(|row| {
                json!({
                    "frame_index": option_usize(row.frame_index),
                    "check_kind": row.check_kind.clone(),
                    "status": row.status.clone(),
                    "expected_frame_bytes": option_u64(row.expected_frame_bytes),
                    "actual_frame_bytes": option_u64(row.actual_frame_bytes),
                    "expected_total_bytes": option_u64(row.expected_total_bytes),
                    "actual_total_bytes": option_u64(row.actual_total_bytes),
                    "plane_order_expected": row.plane_order_expected.clone(),
                    "little_endian_expected": row.little_endian_expected.clone(),
                    "output_target": row.output_target.clone(),
                    "first_mismatch_offset": option_usize(row.first_mismatch_offset),
                    "digest64": option_u64(row.digest64),
                    "notes": row.notes.clone(),
                })
            })
            .collect(),
    )
}

fn option_string(value: Option<String>) -> Value {
    value.map_or(Value::Null, Value::String)
}

fn option_bool(value: Option<bool>) -> Value {
    value.map_or(Value::Null, Value::Bool)
}

fn option_u32(value: Option<u32>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

fn option_usize(value: Option<usize>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

fn option_u64(value: Option<u64>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

fn option_f64(value: Option<f64>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::measurement::types::{
        DisplayMetrics, DisplayWarmupMetrics, DisplayWarmupStatus, FrameCountMetrics,
        GpuStageMetrics, InputClipMetrics, MeasuredSink, MeasurementStatus, MeasurementTiming,
        PayloadMetrics, PipeByteIdentityMetrics, PipeProducerMetrics,
    };

    fn base_common(command: ReportCommand) -> ReportCommon {
        ReportCommon {
            command,
            sink_name: match command {
                ReportCommand::Display => "gpu_display_no_vsync_proxy",
                ReportCommand::Pipe => "gpu_gpu_vignette_pipe_yuv444p12le_raw",
            }
            .to_string(),
            status: ReportStatus::Ok,
            error_message: None,
            input: ReportInput::from_path(Path::new("private/source/clip.mcraw")),
            frames_requested: 600,
            frames_available: Some(600),
            frames_selected: Some(600),
            frames_measured: Some(600),
            warmup_frames_requested: 0,
            warmup_frames_processed: Some(0),
            measurement_truncated_by_clip_length: Some(false),
            decision_quality_min_frames: 600,
            decision_quality_ok: true,
            start_frame: 0,
            stride: 1,
            settings_source: "default".to_string(),
            backend: "gpu".to_string(),
            backend_effective: Some("gpu".to_string()),
            vignette_correction: "with".to_string(),
            payload_profile_requested: None,
            payload_profile_effective: false,
            setup_s: Some(1.0),
            warmup_s: Some(0.0),
            run_s: Some(10.0),
            flush_s: Some(1.0),
            total_s: Some(12.0),
            fps: Some(60.0),
            total_fps: Some(50.0),
            bytes_output_logical: Some(0),
            bytes_written_physical: Some(0),
            full_frame_readback_performed: Some(false),
            notes: Vec::new(),
        }
    }

    fn fake_display_result() -> MeasurementResult {
        MeasurementResult {
            sink: MeasuredSink::GpuDisplayNoVsyncProxy,
            status: MeasurementStatus::Ok,
            failure_stage: None,
            timing: MeasurementTiming {
                setup: Duration::from_secs(1),
                run: Duration::from_secs(10),
                flush: Duration::from_secs(1),
            },
            frames: FrameCountMetrics {
                frames_requested: 600,
                frames_processed: 600,
            },
            input: InputClipMetrics {
                input_basename: "clip.mcraw".to_string(),
                frame_width: 1920,
                frame_height: 1080,
            },
            payload: PayloadMetrics::default(),
            gpu: GpuStageMetrics::default(),
            display: DisplayMetrics::no_readback(1280, 720, 4),
            display_warmup: DisplayWarmupMetrics {
                frames_requested: 30,
                frames_processed: 30,
                duration: Duration::from_secs(1),
                status: DisplayWarmupStatus::Ok,
            },
            notes: Vec::new(),
        }
    }

    fn fake_pipe_result() -> PipeProducerMeasurementResult {
        let expected_frame = 1920u64 * 1080u64 * 6;
        PipeProducerMeasurementResult {
            sink: MeasuredSink::GpuGpuVignettePipeYuv444p12LeRaw,
            status: MeasurementStatus::Ok,
            failure_stage: None,
            timing: MeasurementTiming {
                setup: Duration::from_secs(1),
                run: Duration::from_secs(10),
                flush: Duration::from_secs(1),
            },
            frames: FrameCountMetrics {
                frames_requested: 600,
                frames_processed: 600,
            },
            input: InputClipMetrics {
                input_basename: "clip.mcraw".to_string(),
                frame_width: 1920,
                frame_height: 1080,
            },
            payload: PayloadMetrics::default(),
            gpu: GpuStageMetrics::default(),
            metrics: PipeProducerMetrics {
                correction_mode: PipeF32BayerCorrectionMode::MotionCamSpatial,
                warmup_frames_requested: 0,
                warmup_frames_processed: 0,
                warmup_s: 0.0,
                bytes_per_frame_expected: expected_frame,
                output_bytes_expected: expected_frame * 600,
                output_bytes: expected_frame * 600,
                writer_bytes: 0,
                wall_no_write_s: 10.0,
                wall_with_writer_s: 10.0,
                fps_no_write: 60.0,
                fps_incl_write: 60.0,
                writer_s: 0.0,
                writer_flush_s: 0.0,
                render_encode_s: 0.0,
                readback_s: 0.0,
                pipe_pack_s: 0.0,
                mapped_consume_s: 0.0,
                histogram_s: 0.0,
                byte_identity_s: 0.0,
                validation_bytes_sampled: 0,
                byte_identity: PipeByteIdentityMetrics {
                    frames_checked: 600,
                    validation_frames_checked: 3,
                    validation_bytes_sampled: 0,
                    output_byte_count_ok: true,
                    frame_byte_count_ok: true,
                    plane_order_ok: true,
                    little_endian_ok: true,
                    meaningful_low_12_bits_ok: true,
                    code_bounds_ok: true,
                    mismatches: 0,
                    first_mismatch_offset: None,
                },
            },
            byte_identity_rows: vec![PipeByteIdentityRow {
                frame_index: Some(0),
                check_kind: "bounded_sample_digest".to_string(),
                status: "OK".to_string(),
                expected_frame_bytes: None,
                actual_frame_bytes: Some(16),
                expected_total_bytes: None,
                actual_total_bytes: None,
                plane_order_expected: "Y,Cb,Cr".to_string(),
                little_endian_expected: "little_endian_low_12".to_string(),
                output_target: "discard".to_string(),
                first_mismatch_offset: None,
                digest64: Some(123),
                notes: "deterministic bounded sample digest".to_string(),
            }],
            notes: Vec::new(),
        }
    }

    #[test]
    fn report_writer_emits_deterministic_valid_json() {
        let common = base_common(ReportCommand::Display);
        let result = fake_display_result();
        let first = MeasurementReport::display_success(common.clone(), &result);
        let second = MeasurementReport::display_success(common, &result);

        let first_text = serde_json::to_string_pretty(first.value()).unwrap();
        let second_text = serde_json::to_string_pretty(second.value()).unwrap();
        assert_eq!(first_text, second_text);
        let reparsed: Value = serde_json::from_str(&first_text).unwrap();
        assert_eq!(reparsed["measurement_report_version"], 1);
    }

    #[test]
    fn report_includes_top_level_schema_and_path_privacy() {
        let report = MeasurementReport::display_success(
            base_common(ReportCommand::Display),
            &fake_display_result(),
        );
        let value = report.value();

        assert_eq!(value["command"], "display");
        assert_eq!(value["sink_name"], "gpu_display_no_vsync_proxy");
        assert_eq!(value["status"], "ok");
        assert_eq!(value["input_basename"], "clip.mcraw");
        assert_eq!(value["input_stem"], "clip");
        assert_eq!(value["input_path_recorded"], false);
        assert!(
            !serde_json::to_string(value)
                .unwrap()
                .contains("private/source")
        );
    }

    #[test]
    fn report_includes_decision_quality_fields() {
        let report = MeasurementReport::display_success(
            base_common(ReportCommand::Display),
            &fake_display_result(),
        );
        let value = report.value();

        assert_eq!(value["frames_requested"], 600);
        assert_eq!(value["frames_available"], 600);
        assert_eq!(value["frames_selected"], 600);
        assert_eq!(value["frames_measured"], 600);
        assert_eq!(value["measurement_truncated_by_clip_length"], false);
        assert_eq!(value["decision_quality_min_frames"], 600);
        assert_eq!(value["decision_quality_ok"], true);
    }

    #[test]
    fn report_allows_short_smoke_but_marks_decision_quality_false() {
        let mut common = base_common(ReportCommand::Display);
        common.frames_requested = 1;
        common.frames_available = Some(600);
        common.frames_selected = Some(1);
        common.frames_measured = Some(1);
        common.decision_quality_ok = false;
        common
            .notes
            .push("warning: measurement used fewer than 600 frames".to_string());
        let mut result = fake_display_result();
        result.frames.frames_requested = 1;
        result.frames.frames_processed = 1;
        let report = MeasurementReport::display_success(common, &result);
        let value = report.value();

        assert_eq!(value["decision_quality_ok"], false);
        assert!(
            value["notes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|note| note.as_str().unwrap().contains("fewer than 600"))
        );
    }

    #[test]
    fn report_marks_truncated_short_clip() {
        let mut common = base_common(ReportCommand::Display);
        common.frames_available = Some(42);
        common.frames_selected = Some(42);
        common.frames_measured = Some(42);
        common.measurement_truncated_by_clip_length = Some(true);
        common.decision_quality_ok = false;
        common
            .notes
            .push("warning: clip has fewer frames than requested".to_string());
        let mut result = fake_display_result();
        result.frames.frames_processed = 42;
        let report = MeasurementReport::display_success(common, &result);
        let value = report.value();

        assert_eq!(value["measurement_truncated_by_clip_length"], true);
        assert_eq!(value["decision_quality_ok"], false);
    }

    #[test]
    fn display_object_records_no_output_and_no_readback() {
        let report = MeasurementReport::display_success(
            base_common(ReportCommand::Display),
            &fake_display_result(),
        );
        let display = &report.value()["display"];

        assert_eq!(display["output_bytes"], 0);
        assert_eq!(display["full_frame_readback_performed"], false);
        assert_eq!(display["overlay_applicable"], false);
        assert_eq!(display["live_window"], false);
    }

    #[test]
    fn pipe_object_includes_canonical_format_and_expected_bytes() {
        let mut common = base_common(ReportCommand::Pipe);
        common.payload_profile_requested = Some("default_chunked64".to_string());
        common.payload_profile_effective = true;
        common.full_frame_readback_performed = Some(true);
        let result = fake_pipe_result();
        let report = MeasurementReport::pipe_success(common, &result);
        let pipe = &report.value()["pipe"];

        assert_eq!(pipe["pixel_format"], "yuv444p12le");
        assert_eq!(pipe["plane_order"], json!(["Y", "Cb", "Cr"]));
        assert_eq!(pipe["endianness"], "little");
        assert_eq!(pipe["sample_range"], "video-data-12bit");
        assert_eq!(pipe["color_range"], "tv");
        assert_eq!(pipe["range_is_limited"], true);
        assert_eq!(pipe["meaningful_bits_per_sample"], 12);
        assert_eq!(pipe["upper_four_bits_zero"], true);
        assert_eq!(pipe["final_code_bounds"], json!([16, 4079]));
        assert_eq!(pipe["color_primaries"], "bt2020");
        assert_eq!(pipe["color_transfer"], "linear");
        assert_eq!(pipe["matrix_coefficients"], "bt2020nc");
        assert_eq!(pipe["linear_signal_scale_num"], 1);
        assert_eq!(pipe["linear_signal_scale_den"], 2);
        assert_eq!(
            pipe["expected_total_video_bytes"],
            1920u64 * 1080u64 * 6u64 * 600u64
        );
        assert_eq!(pipe["output_target"], "discard");
        assert_eq!(pipe["stdout_used"], false);
        assert_eq!(pipe["byte_identity_status"], "OK");
        assert_eq!(pipe["byte_identity_frames_checked"], 600);
        assert_eq!(pipe["byte_identity_validation_frames_checked"], 3);
        assert_eq!(pipe["byte_identity_rows"][0]["digest64"], 123);
        assert!(
            !serde_json::to_string(pipe)
                .unwrap()
                .contains("inflight_2_default")
        );
    }

    #[test]
    fn unsupported_cpu_pipe_report_has_error_status() {
        let mut common = base_common(ReportCommand::Pipe);
        common.status = ReportStatus::Unsupported;
        common.error_message = Some("CPU PIPE output is not implemented".to_string());
        common.backend = "cpu".to_string();
        common.backend_effective = None;
        common.decision_quality_ok = false;
        let report = MeasurementReport::pipe_error(common);
        let value = report.value();

        assert_eq!(value["status"], "unsupported");
        assert!(
            value["error_message"]
                .as_str()
                .unwrap()
                .contains("CPU PIPE")
        );
    }

    #[test]
    fn report_contains_no_external_tool_wording() {
        let report = MeasurementReport::display_success(
            base_common(ReportCommand::Display),
            &fake_display_result(),
        );
        let text = serde_json::to_string(report.value()).unwrap();
        let forbidden = [
            ["ff", "mpeg"].concat(),
            ["FF", "mpeg"].concat(),
            ["lib", "av"].concat(),
            ["Pro", "Res"].concat(),
            ["HE", "VC"].concat(),
            ["NV", "ENC"].concat(),
            ["lib", "placebo"].concat(),
            ["x", "264"].concat(),
            ["x", "265"].concat(),
        ];

        for word in forbidden {
            assert!(!text.contains(&word), "report contains {word}");
        }
    }
}
