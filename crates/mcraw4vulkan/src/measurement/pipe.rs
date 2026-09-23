use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use mcraw4vulkan_core::FrameNumber;
use mcraw4vulkan_mcrawcontainer::{McrawContainer, payload_reader::PayloadReadPlan};
use mcraw4vulkan_vignette::PipeF32BayerCorrectionMode;

use crate::direct_yuv12_pipeline::{
    DirectYuv12FrameIdentity, DirectYuv12FrameSink, DirectYuv12PipelineStats,
    DirectYuv12PublicationTiming,
};
use crate::pipe_cli::{
    PipeCliBackend, PipeFramePreflight, preflight_pipe_frames, stream_pipe_frames,
};
use crate::pipe_contract::{
    PIPE_BYTE_ORDER_LABEL, PIPE_PLANE_ORDER_LABEL, checked_pipe_bytes_per_frame,
    checked_pipe_total_bytes,
};

use super::types::{
    FrameCountMetrics, GpuStageMetrics, InputClipMetrics, MeasurementStatus, MeasurementTiming,
    PayloadMetrics, PipeByteIdentityMetrics, PipeByteIdentityRow, PipeProducerMeasurementRequest,
    PipeProducerMeasurementResult, PipeProducerMetrics, pipe_digest64,
    pipe_validation_frame_indices, pipe_validation_sample_ranges, validate_pipe_frame_byte_count,
};

const PIPE_MEASUREMENT_OUTPUT_TARGET: &str = "discard";
pub struct PipeProducerMeasurementRunner;

impl PipeProducerMeasurementRunner {
    pub fn new() -> Self {
        Self
    }

    pub fn measure(
        &self,
        request: PipeProducerMeasurementRequest,
    ) -> Result<PipeProducerMeasurementResult> {
        if request.run.selected_frames.is_empty() {
            bail!("PIPE producer measurement requires at least one selected frame");
        }
        validate_direct_yuv_measurement_selection(&request)?;

        let preflight_start = Instant::now();
        let container = McrawContainer::open(&request.run.input_path)?;
        let preflight = preflight_pipe_frames(
            &request.run.input_path,
            PipeCliBackend::Gpu,
            &container,
            request.run.selected_frames.len(),
        )?;
        let input = InputClipMetrics {
            input_basename: request
                .run
                .input_path
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            frame_width: preflight.dimensions.width,
            frame_height: preflight.dimensions.height,
        };
        let metadata_preflight = preflight_start.elapsed();

        let mut warmup_frames_processed = 0usize;
        let mut warmup_s = 0.0f64;
        if request.run.warmup_frames > 0 {
            let warmup_count = request
                .run
                .warmup_frames
                .min(request.run.selected_frames.len());
            let warmup_selected = &request.run.selected_frames[..warmup_count];
            let mut warmup_output = PipeDiscardWriter::default();
            let mut warmup_byte_rows = Vec::new();
            let mut warmup_validation = PipeValidationCollector::new(Vec::new());
            let warmup_start = Instant::now();
            let warmup = run_direct_yuv_measurement_sequence(
                &request,
                &container,
                &preflight,
                warmup_selected,
                &mut warmup_output,
                &mut warmup_byte_rows,
                &mut warmup_validation,
            )?;
            warmup_frames_processed = warmup.frames_processed;
            let expected_warmup_bytes = checked_pipe_total_bytes(
                input.frame_width,
                input.frame_height,
                u64::try_from(warmup_frames_processed)
                    .context("PIPE warm-up frame count overflowed")?,
            )
            .context("PIPE warm-up expected total byte count overflowed")?;
            let _ = warmup_output.finish(expected_warmup_bytes);
            warmup_s = warmup_start.elapsed().as_secs_f64();
        }

        let mut output = PipeDiscardWriter::default();
        let mut byte_rows = Vec::new();
        let mut validation_collector = PipeValidationCollector::new(pipe_validation_frame_indices(
            &request.run.selected_frames,
        ));
        let expected_frame_bytes =
            checked_pipe_bytes_per_frame(input.frame_width, input.frame_height)
                .context("PIPE expected frame byte count overflowed")?;
        let measured = run_direct_yuv_measurement_sequence(
            &request,
            &container,
            &preflight,
            &request.run.selected_frames,
            &mut output,
            &mut byte_rows,
            &mut validation_collector,
        )?;
        let frames_processed = measured.frames_processed;

        let expected_total_bytes = checked_pipe_total_bytes(
            input.frame_width,
            input.frame_height,
            u64::try_from(frames_processed).context("PIPE measured frame count overflowed")?,
        )
        .context("PIPE expected total byte count overflowed")?;
        let writer_byte_count_ok = output.finish(expected_total_bytes);
        let wall_with_writer = measured.run;
        let writer_total = measured.writer_s;
        let wall_no_write = wall_with_writer.saturating_sub(writer_total);

        let output_byte_count_ok =
            measured.output_bytes == expected_total_bytes && writer_byte_count_ok;
        let mut mismatches = 0usize;
        let mut first_mismatch_offset = measured.first_mismatch_offset;
        if !output_byte_count_ok {
            mismatches = mismatches.saturating_add(1);
            first_mismatch_offset.get_or_insert(0);
        }
        if !measured.frame_byte_count_ok {
            mismatches = mismatches.saturating_add(1);
        }

        let byte_identity_start = Instant::now();
        let validation_summary = validation_collector.finish_rows(&mut byte_rows);
        byte_rows.push(output_count_row(
            expected_total_bytes,
            measured.output_bytes,
        ));
        byte_rows.push(static_contract_row(
            "plane_order",
            PIPE_PLANE_ORDER_LABEL,
            "canonical planar order",
        ));
        byte_rows.push(static_contract_row(
            "little_endian",
            PIPE_BYTE_ORDER_LABEL,
            "canonical 16-bit little-endian storage",
        ));
        byte_rows.push(static_contract_row(
            "meaningful_low_12_bits",
            "bits 12..15 zero",
            "canonical low-12-bit word alignment",
        ));
        let byte_identity_s = byte_identity_start.elapsed().as_secs_f64();

        if !validation_summary.meaningful_low_12_bits_ok {
            mismatches = mismatches.saturating_add(1);
            first_mismatch_offset.get_or_insert(0);
        }
        if !validation_summary.code_bounds_ok {
            mismatches = mismatches.saturating_add(1);
            first_mismatch_offset.get_or_insert(0);
        }

        let byte_identity = PipeByteIdentityMetrics {
            frames_checked: frames_processed,
            validation_frames_checked: validation_summary.frames_checked,
            validation_bytes_sampled: validation_summary.bytes_sampled,
            output_byte_count_ok,
            frame_byte_count_ok: measured.frame_byte_count_ok,
            plane_order_ok: true,
            little_endian_ok: true,
            meaningful_low_12_bits_ok: validation_summary.meaningful_low_12_bits_ok,
            code_bounds_ok: validation_summary.code_bounds_ok,
            mismatches,
            first_mismatch_offset,
        };
        let status = if byte_identity.status() == "OK" {
            MeasurementStatus::Ok
        } else {
            MeasurementStatus::Failed
        };
        let failure_stage = if status == MeasurementStatus::Ok {
            None
        } else {
            Some("pipe_byte_identity_mismatch".to_string())
        };

        Ok(PipeProducerMeasurementResult {
            sink: measured_sink(request.policy.correction_mode),
            status,
            failure_stage,
            timing: MeasurementTiming {
                setup: metadata_preflight.saturating_add(measured.setup),
                run: wall_no_write,
                flush: writer_total,
            },
            frames: FrameCountMetrics {
                frames_requested: request.run.frames_requested,
                frames_processed,
            },
            input,
            payload: measured.payload,
            gpu: GpuStageMetrics {
                dispatch_s: measured.stats.total_queue_encode_submit.as_secs_f64(),
                wait_s: measured.stats.total_map_wait.as_secs_f64(),
                full_frame_readback_s: measured.stats.total_map_wait.as_secs_f64(),
            },
            metrics: PipeProducerMetrics {
                correction_mode: request.policy.correction_mode,
                warmup_frames_requested: request.run.warmup_frames,
                warmup_frames_processed,
                warmup_s,
                bytes_per_frame_expected: expected_frame_bytes,
                output_bytes_expected: expected_total_bytes,
                output_bytes: measured.output_bytes,
                writer_bytes: measured.writer_bytes,
                wall_no_write_s: wall_no_write.as_secs_f64(),
                wall_with_writer_s: wall_with_writer.as_secs_f64(),
                fps_no_write: fps(frames_processed, wall_no_write.as_secs_f64()),
                fps_incl_write: fps(frames_processed, wall_with_writer.as_secs_f64()),
                writer_s: measured.writer_s.as_secs_f64(),
                writer_flush_s: 0.0,
                render_encode_s: measured.stats.total_queue_encode_submit.as_secs_f64(),
                readback_s: measured.stats.total_map_wait.as_secs_f64(),
                pipe_pack_s: measured.stats.total_queue_encode_submit.as_secs_f64(),
                mapped_consume_s: measured.stats.total_direct_mapped_write.as_secs_f64(),
                histogram_s: 0.0,
                byte_identity_s,
                validation_bytes_sampled: validation_summary.bytes_sampled,
                byte_identity,
            },
            byte_identity_rows: byte_rows,
            notes: vec![
                measured.payload_note,
                "canonical direct yuv444p12le original Apple Log / BT.2020 path, unity terminal exposure scale".to_string(),
                "native GPU payload feeder; zero CPU Bayer predecode and no validation oracle"
                    .to_string(),
                if uses_vignette(request.policy.correction_mode) {
                    "MotionCamSpatial correction before signed demosaic".to_string()
                } else {
                    "IdentitySpatialGain retains non-spatial correction".to_string()
                },
                format!(
                    "scheduler_depth_max={} decode_work_plan_reused_frames={} first_output_latency_s={}",
                    measured.stats.maximum_pending_frames,
                    measured.stats.decode_work_plan_reused_frames,
                    measured.stats.first_output_latency.map_or_else(
                        || "none".to_string(),
                        |value| value.as_secs_f64().to_string()
                    ),
                ),
                format!("pipe_output_target={PIPE_MEASUREMENT_OUTPUT_TARGET}"),
            ],
        })
    }
}

fn validate_direct_yuv_measurement_selection(
    request: &PipeProducerMeasurementRequest,
) -> Result<()> {
    if request.run.start_frame != 0 || request.run.stride != 1 {
        bail!("internal PIPE measurement requires a first-N contiguous frame selection");
    }
    for (index, frame) in request.run.selected_frames.iter().copied().enumerate() {
        let expected = u32::try_from(index).context("PIPE measurement frame index overflow")?;
        if frame != FrameNumber(expected) {
            bail!(
                "internal PIPE measurement selection is not first-N contiguous at index {index}: got frame {}",
                frame.0,
            );
        }
    }
    Ok(())
}

fn uses_vignette(correction_mode: PipeF32BayerCorrectionMode) -> bool {
    matches!(
        correction_mode,
        PipeF32BayerCorrectionMode::MotionCamSpatial
    )
}

fn measured_sink(correction_mode: PipeF32BayerCorrectionMode) -> super::types::MeasuredSink {
    match correction_mode {
        PipeF32BayerCorrectionMode::IdentitySpatialGain => {
            super::types::MeasuredSink::GpuPipeYuv444p12LeRawNoVignette
        }
        PipeF32BayerCorrectionMode::MotionCamSpatial => {
            super::types::MeasuredSink::GpuGpuVignettePipeYuv444p12LeRaw
        }
    }
}

struct DirectYuv12MeasurementSequenceResult {
    setup: Duration,
    run: Duration,
    frames_processed: usize,
    output_bytes: u64,
    writer_bytes: u64,
    writer_s: Duration,
    frame_byte_count_ok: bool,
    first_mismatch_offset: Option<usize>,
    payload: PayloadMetrics,
    payload_note: String,
    stats: DirectYuv12PipelineStats,
}

#[allow(clippy::too_many_arguments)]
fn run_direct_yuv_measurement_sequence(
    request: &PipeProducerMeasurementRequest,
    container: &McrawContainer,
    preflight: &PipeFramePreflight,
    selected_frames: &[FrameNumber],
    output: &mut PipeDiscardWriter,
    byte_rows: &mut Vec<PipeByteIdentityRow>,
    validation_collector: &mut PipeValidationCollector,
) -> Result<DirectYuv12MeasurementSequenceResult> {
    let payload_plan = PayloadReadPlan::from_core_frame_numbers(container, selected_frames)?;
    let resolved_payload = request.payload.resolve(&payload_plan)?;
    let expected_frame_bytes =
        checked_pipe_bytes_per_frame(preflight.dimensions.width, preflight.dimensions.height)
            .context("internal PIPE expected frame byte count overflowed")?;
    let mut sink = DirectYuv12MeasurementSink {
        expected_frame_bytes,
        next_sequence: 0,
        output,
        byte_rows,
        validation_collector,
        output_bytes: 0,
        writer_bytes: 0,
        writer_s: Duration::ZERO,
        frame_byte_count_ok: true,
        first_mismatch_offset: None,
    };
    let streamed = stream_pipe_frames(
        &request.run.input_path,
        PipeCliBackend::Gpu,
        request.policy.correction_mode,
        request.policy.backend_preference,
        resolved_payload.feeder_options,
        payload_plan,
        container,
        preflight,
        &mut sink,
    )?;

    Ok(DirectYuv12MeasurementSequenceResult {
        setup: streamed.setup,
        run: streamed.stream,
        frames_processed: streamed.frames_processed,
        output_bytes: sink.output_bytes,
        writer_bytes: sink.writer_bytes,
        writer_s: sink.writer_s,
        frame_byte_count_ok: sink.frame_byte_count_ok,
        first_mismatch_offset: sink.first_mismatch_offset,
        payload: PayloadMetrics::from_feeder_stats(streamed.payload),
        payload_note: resolved_payload.note,
        stats: streamed.scheduler,
    })
}

struct DirectYuv12MeasurementSink<'a> {
    expected_frame_bytes: u64,
    next_sequence: u64,
    output: &'a mut PipeDiscardWriter,
    byte_rows: &'a mut Vec<PipeByteIdentityRow>,
    validation_collector: &'a mut PipeValidationCollector,
    output_bytes: u64,
    writer_bytes: u64,
    writer_s: Duration,
    frame_byte_count_ok: bool,
    first_mismatch_offset: Option<usize>,
}

impl DirectYuv12FrameSink for DirectYuv12MeasurementSink<'_> {
    fn publish_mapped_frame(
        &mut self,
        identity: DirectYuv12FrameIdentity,
        planar_bytes: &[u8],
    ) -> Result<(), String> {
        if identity.sequence != self.next_sequence
            || identity.source_frame_index != self.next_sequence
        {
            return Err(format!(
                "internal PIPE publication order mismatch: expected {}, got sequence={} source={}",
                self.next_sequence, identity.sequence, identity.source_frame_index,
            ));
        }
        let frame_index = usize::try_from(identity.source_frame_index)
            .map_err(|_| "internal PIPE source frame index overflow".to_string())?;
        let actual_frame_bytes = u64::try_from(planar_bytes.len()).unwrap_or(u64::MAX);
        let status = validate_pipe_frame_byte_count(self.expected_frame_bytes, actual_frame_bytes);
        let frame_byte_count_ok = status == "OK";
        let writer_s = self.output.write(planar_bytes);
        self.validation_collector.capture(frame_index, planar_bytes);
        self.byte_rows.push(PipeByteIdentityRow {
            frame_index: Some(frame_index),
            check_kind: "frame_byte_count".to_string(),
            status: status.to_string(),
            expected_frame_bytes: Some(self.expected_frame_bytes),
            actual_frame_bytes: Some(actual_frame_bytes),
            expected_total_bytes: None,
            actual_total_bytes: None,
            plane_order_expected: PIPE_PLANE_ORDER_LABEL.to_string(),
            little_endian_expected: PIPE_BYTE_ORDER_LABEL.to_string(),
            output_target: PIPE_MEASUREMENT_OUTPUT_TARGET.to_string(),
            first_mismatch_offset: if frame_byte_count_ok { None } else { Some(0) },
            digest64: None,
            notes: "mapped direct-YUV scheduler readback frame byte count; bounded samples are checked separately".to_string(),
        });
        self.output_bytes = self.output_bytes.saturating_add(actual_frame_bytes);
        self.writer_bytes = self.writer_bytes.saturating_add(actual_frame_bytes);
        self.writer_s = self.writer_s.saturating_add(writer_s);
        if !frame_byte_count_ok {
            self.frame_byte_count_ok = false;
            self.first_mismatch_offset.get_or_insert(0);
        }
        self.next_sequence = self.next_sequence.saturating_add(1);
        Ok(())
    }

    fn record_publication_timing(
        &mut self,
        _identity: DirectYuv12FrameIdentity,
        _timing: DirectYuv12PublicationTiming,
    ) {
    }
}

impl Default for PipeProducerMeasurementRunner {
    fn default() -> Self {
        Self::new()
    }
}

fn fps(frames: usize, seconds: f64) -> f64 {
    if frames == 0 || seconds <= 0.0 {
        0.0
    } else {
        frames as f64 / seconds
    }
}

#[derive(Debug, Clone)]
struct PipeValidationSample {
    frame_index: usize,
    bytes: Vec<u8>,
    sampled_bytes: u64,
    meaningful_low_12_bits_ok: bool,
    code_bounds_ok: bool,
}

#[derive(Debug, Clone, Copy)]
struct PipeValidationSummary {
    frames_checked: usize,
    bytes_sampled: u64,
    meaningful_low_12_bits_ok: bool,
    code_bounds_ok: bool,
}

#[derive(Debug, Clone)]
struct PipeValidationCollector {
    frames: Vec<usize>,
    samples: Vec<PipeValidationSample>,
    bytes_sampled: u64,
}

impl PipeValidationCollector {
    fn new(frames: Vec<usize>) -> Self {
        Self {
            frames,
            samples: Vec::new(),
            bytes_sampled: 0,
        }
    }

    fn capture(&mut self, frame_index: usize, mapped: &[u8]) {
        if !self.frames.contains(&frame_index)
            || self
                .samples
                .iter()
                .any(|sample| sample.frame_index == frame_index)
        {
            return;
        }
        let mut sampled = Vec::new();
        for (offset, len) in pipe_validation_sample_ranges(mapped.len()) {
            sampled.extend_from_slice(&mapped[offset..offset + len]);
        }
        let sampled_bytes = u64::try_from(sampled.len()).unwrap_or(u64::MAX);
        let mut meaningful_low_12_bits_ok = true;
        let mut code_bounds_ok = true;
        for bytes in sampled.chunks_exact(2) {
            let code = u16::from_le_bytes([bytes[0], bytes[1]]);
            meaningful_low_12_bits_ok &= code & 0xf000 == 0;
            code_bounds_ok &= (16..=4079).contains(&code);
        }
        self.bytes_sampled = self.bytes_sampled.saturating_add(sampled_bytes);
        self.samples.push(PipeValidationSample {
            frame_index,
            bytes: sampled,
            sampled_bytes,
            meaningful_low_12_bits_ok,
            code_bounds_ok,
        });
    }

    fn finish_rows(self, byte_rows: &mut Vec<PipeByteIdentityRow>) -> PipeValidationSummary {
        let frames_checked = self.samples.len();
        let meaningful_low_12_bits_ok = self
            .samples
            .iter()
            .all(|sample| sample.meaningful_low_12_bits_ok);
        let code_bounds_ok = self.samples.iter().all(|sample| sample.code_bounds_ok);
        for sample in self.samples {
            byte_rows.push(PipeByteIdentityRow {
                frame_index: Some(sample.frame_index),
                check_kind: "bounded_sample_digest".to_string(),
                status: if sample.meaningful_low_12_bits_ok && sample.code_bounds_ok {
                    "OK".to_string()
                } else {
                    "SAMPLE_CONTRACT_MISMATCH".to_string()
                },
                expected_frame_bytes: None,
                actual_frame_bytes: Some(sample.sampled_bytes),
                expected_total_bytes: None,
                actual_total_bytes: None,
                plane_order_expected: PIPE_PLANE_ORDER_LABEL.to_string(),
                little_endian_expected: PIPE_BYTE_ORDER_LABEL.to_string(),
                output_target: PIPE_MEASUREMENT_OUTPUT_TARGET.to_string(),
                first_mismatch_offset: None,
                digest64: Some(pipe_digest64(&sample.bytes)),
                notes: format!(
                    "deterministic bounded planar sample digest; high_bits_zero={} code_bounds_16_4079={}",
                    sample.meaningful_low_12_bits_ok, sample.code_bounds_ok,
                ),
            });
        }
        PipeValidationSummary {
            frames_checked,
            bytes_sampled: self.bytes_sampled,
            meaningful_low_12_bits_ok,
            code_bounds_ok,
        }
    }
}

fn output_count_row(expected_total_bytes: u64, actual_total_bytes: u64) -> PipeByteIdentityRow {
    let status = validate_pipe_frame_byte_count(expected_total_bytes, actual_total_bytes);
    PipeByteIdentityRow {
        frame_index: None,
        check_kind: "output_byte_count".to_string(),
        status: status.to_string(),
        expected_frame_bytes: None,
        actual_frame_bytes: None,
        expected_total_bytes: Some(expected_total_bytes),
        actual_total_bytes: Some(actual_total_bytes),
        plane_order_expected: PIPE_PLANE_ORDER_LABEL.to_string(),
        little_endian_expected: PIPE_BYTE_ORDER_LABEL.to_string(),
        output_target: PIPE_MEASUREMENT_OUTPUT_TARGET.to_string(),
        first_mismatch_offset: if status == "OK" { None } else { Some(0) },
        digest64: None,
        notes: "total producer byte count".to_string(),
    }
}

fn static_contract_row(check_kind: &str, expected: &str, notes: &str) -> PipeByteIdentityRow {
    PipeByteIdentityRow {
        frame_index: None,
        check_kind: check_kind.to_string(),
        status: "OK".to_string(),
        expected_frame_bytes: None,
        actual_frame_bytes: None,
        expected_total_bytes: None,
        actual_total_bytes: None,
        plane_order_expected: PIPE_PLANE_ORDER_LABEL.to_string(),
        little_endian_expected: PIPE_BYTE_ORDER_LABEL.to_string(),
        output_target: PIPE_MEASUREMENT_OUTPUT_TARGET.to_string(),
        first_mismatch_offset: None,
        digest64: None,
        notes: format!("{notes}: {expected}"),
    }
}

#[derive(Debug, Default)]
struct PipeDiscardWriter {
    bytes: u64,
}

impl PipeDiscardWriter {
    fn write(&mut self, bytes: &[u8]) -> Duration {
        let start = Instant::now();
        let writer_bytes = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        self.bytes = self.bytes.saturating_add(writer_bytes);
        start.elapsed()
    }

    fn finish(self, expected_total_bytes: u64) -> bool {
        self.bytes == expected_total_bytes
    }
}
