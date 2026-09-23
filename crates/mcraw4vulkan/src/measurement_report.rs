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
        "terminal_exposure_scale_num": Yuv444p12lePackPolicy::TERMINAL_EXPOSURE_SCALE_NUMERATOR,
        "terminal_exposure_scale_den": Yuv444p12lePackPolicy::TERMINAL_EXPOSURE_SCALE_DENOMINATOR,
        "terminal_exposure_scale_stops": Yuv444p12lePackPolicy::TERMINAL_EXPOSURE_SCALE_STOPS,
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
