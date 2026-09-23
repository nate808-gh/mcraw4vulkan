use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use mcraw4vulkan_core::{FrameDimensions, FrameNumber};
use mcraw4vulkan_display::{
    DEFAULT_DISPLAY_WINDOW_HEIGHT, DEFAULT_DISPLAY_WINDOW_WIDTH, fit_aspect_preserving,
};
use mcraw4vulkan_gpu::GpuBackendPreference;
use mcraw4vulkan_mcrawcontainer::payload_reader::{
    PRODUCTION_PAYLOAD_CHUNK_GAP_THRESHOLD_BYTES, PRODUCTION_PAYLOAD_CHUNK_MAX_PAYLOADS,
    PRODUCTION_PAYLOAD_PREFETCH_DEPTH, PRODUCTION_PAYLOAD_REORDER_WINDOW, PayloadAdaptiveOptions,
    PayloadAdaptivePolicy, PayloadChunkOptions, PayloadFeederOptions, PayloadFeederStats,
    PayloadReadMode, PayloadReadModeRequest, PayloadReadPlan,
};
use mcraw4vulkan_render::{PreviewScaleMode, PreviewTextureFormat, PreviewTransferMode};
use mcraw4vulkan_vignette::PipeF32BayerCorrectionMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasuredSink {
    CpuDisplayNoVsyncProxy,
    GpuDisplayNoVsyncProxy,
    GpuGpuVignetteDisplayNoVsyncProxy,
    GpuGpuVignettePipeYuv444p12LeRaw,
    GpuPipeYuv444p12LeRawNoVignette,
}

impl MeasuredSink {
    pub fn label(self) -> &'static str {
        match self {
            Self::CpuDisplayNoVsyncProxy => "cpu_display_no_vsync_proxy",
            Self::GpuDisplayNoVsyncProxy => "gpu_display_no_vsync_proxy",
            Self::GpuGpuVignetteDisplayNoVsyncProxy => "gpu_gpu_vignette_display_no_vsync_proxy",
            Self::GpuGpuVignettePipeYuv444p12LeRaw => "gpu_gpu_vignette_pipe_yuv444p12le_raw",
            Self::GpuPipeYuv444p12LeRawNoVignette => "gpu_pipe_yuv444p12le_raw_no_vignette",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementStatus {
    Ok,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurementTiming {
    pub setup: Duration,
    pub run: Duration,
    pub flush: Duration,
}

impl MeasurementTiming {
    pub fn total(self) -> Duration {
        self.setup
            .saturating_add(self.run)
            .saturating_add(self.flush)
    }

    pub fn setup_s(self) -> f64 {
        self.setup.as_secs_f64()
    }

    pub fn run_s(self) -> f64 {
        self.run.as_secs_f64()
    }

    pub fn flush_s(self) -> f64 {
        self.flush.as_secs_f64()
    }

    pub fn total_s(self) -> f64 {
        self.total().as_secs_f64()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameCountMetrics {
    pub frames_requested: usize,
    pub frames_processed: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputClipMetrics {
    pub input_basename: String,
    pub frame_width: u32,
    pub frame_height: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PayloadMetrics {
    pub reader_read_calls: usize,
    pub reader_seeks: usize,
    pub reader_bytes: u64,
    pub payload_bytes_returned: u64,
    pub chunk_overread_ratio: f64,
    pub retained_chunk_bytes_estimate: u64,
}

impl PayloadMetrics {
    pub fn from_feeder_stats(stats: PayloadFeederStats) -> Self {
        Self {
            reader_read_calls: stats.read_calls,
            reader_seeks: stats.seeks,
            reader_bytes: stats.bytes_read,
            payload_bytes_returned: stats.payload_bytes_returned,
            chunk_overread_ratio: stats.chunk_overread_ratio(),
            retained_chunk_bytes_estimate: stats.retained_chunk_bytes_estimate,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuStageMetrics {
    pub dispatch_s: f64,
    pub wait_s: f64,
    pub full_frame_readback_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadbackKind {
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputBytesKind {
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayMetrics {
    pub preview_width: u32,
    pub preview_height: u32,
    pub preview_target_bytes_estimate: u64,
    pub full_frame_gpu_readback_performed: bool,
    pub readback_kind: ReadbackKind,
    pub output_bytes: u64,
    pub output_bytes_kind: OutputBytesKind,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DisplayWarmupMetrics {
    pub frames_requested: usize,
    pub frames_processed: usize,
    pub duration: Duration,
    pub status: DisplayWarmupStatus,
}

impl DisplayWarmupMetrics {
    pub fn not_requested() -> Self {
        Self::default()
    }

    pub fn duration_s(self) -> f64 {
        self.duration.as_secs_f64()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DisplayWarmupStatus {
    #[default]
    NotRequested,
    Ok,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementResult {
    pub sink: MeasuredSink,
    pub status: MeasurementStatus,
    pub failure_stage: Option<String>,
    pub timing: MeasurementTiming,
    pub frames: FrameCountMetrics,
    pub input: InputClipMetrics,
    pub payload: PayloadMetrics,
    pub gpu: GpuStageMetrics,
    pub display: DisplayMetrics,
    pub display_warmup: DisplayWarmupMetrics,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayMeasurementRunSpec {
    pub input_path: PathBuf,
    pub start_frame: usize,
    pub stride: usize,
    pub frames_requested: usize,
    pub selected_frames: Vec<FrameNumber>,
    pub display_warmup_frames: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadReadPolicyMode {
    OffsetPrefetch,
    ChunkedOffsetPrefetch,
}

impl PayloadReadPolicyMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::OffsetPrefetch => "offset_prefetch",
            Self::ChunkedOffsetPrefetch => "chunked_offset_prefetch",
        }
    }

    pub fn request(self) -> PayloadReadModeRequest {
        match self {
            Self::OffsetPrefetch => PayloadReadModeRequest::OffsetPrefetch,
            Self::ChunkedOffsetPrefetch => PayloadReadModeRequest::ChunkedOffsetPrefetch,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadReadPolicy {
    pub mode: PayloadReadPolicyMode,
    pub chunk_mib: u32,
}

impl PayloadReadPolicy {
    pub fn chunk_bytes(self) -> u64 {
        u64::from(self.chunk_mib) * 1024 * 1024
    }

    pub fn resolve(self, read_plan: &PayloadReadPlan) -> Result<ResolvedPayloadReadPolicy> {
        let chunk_options = PayloadChunkOptions {
            gap_threshold_bytes: PRODUCTION_PAYLOAD_CHUNK_GAP_THRESHOLD_BYTES,
            max_chunk_bytes: self.chunk_bytes(),
            max_payloads_per_chunk: PRODUCTION_PAYLOAD_CHUNK_MAX_PAYLOADS,
        };
        let base_options = PayloadFeederOptions {
            mode: PayloadReadMode::Current,
            prefetch_depth: PRODUCTION_PAYLOAD_PREFETCH_DEPTH,
            reorder_window: PRODUCTION_PAYLOAD_REORDER_WINDOW,
            chunk_options,
        };
        let decision =
            mcraw4vulkan_mcrawcontainer::payload_reader::PayloadAdaptiveDecision::resolve(
                self.mode.request(),
                read_plan,
                base_options,
                PayloadAdaptiveOptions {
                    policy: PayloadAdaptivePolicy::Balanced,
                    ..PayloadAdaptiveOptions::default()
                },
            )
            .map_err(|error| anyhow::anyhow!("failed to resolve payload policy: {error}"))?;

        Ok(ResolvedPayloadReadPolicy {
            feeder_options: PayloadFeederOptions {
                mode: decision.selected,
                ..base_options
            },
            note: format!(
                "payload_factor={} selected_payload_mode={} reason={}",
                self.mode.label(),
                decision.selected.label(),
                decision.reason.label()
            ),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPayloadReadPolicy {
    pub feeder_options: PayloadFeederOptions,
    pub note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMeasurementMode {
    Cpu,
    Gpu,
    GpuVignette,
}

impl DisplayMeasurementMode {
    pub fn sink(self) -> MeasuredSink {
        match self {
            Self::Cpu => MeasuredSink::CpuDisplayNoVsyncProxy,
            Self::Gpu => MeasuredSink::GpuDisplayNoVsyncProxy,
            Self::GpuVignette => MeasuredSink::GpuGpuVignetteDisplayNoVsyncProxy,
        }
    }

    pub fn uses_gpu_vignette(self) -> bool {
        matches!(self, Self::GpuVignette)
    }

    pub fn uses_cpu_decode(self) -> bool {
        matches!(self, Self::Cpu)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayProxyPolicy {
    WindowAspectFit { width: u32, height: u32 },
}

impl Default for DisplayProxyPolicy {
    fn default() -> Self {
        Self::WindowAspectFit {
            width: DEFAULT_DISPLAY_WINDOW_WIDTH,
            height: DEFAULT_DISPLAY_WINDOW_HEIGHT,
        }
    }
}

impl DisplayProxyPolicy {
    pub fn preview_scale_mode(self, input: FrameDimensions) -> PreviewScaleMode {
        match self {
            Self::WindowAspectFit { width, height } => {
                let viewport = fit_aspect_preserving(width, height, input.width, input.height);
                PreviewScaleMode::Explicit {
                    width: viewport.content_width.max(1),
                    height: viewport.content_height.max(1),
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayMeasurementPolicy {
    pub mode: DisplayMeasurementMode,
    pub proxy: DisplayProxyPolicy,
    pub texture_format: PreviewTextureFormat,
    pub transfer_mode: PreviewTransferMode,
}

impl DisplayMeasurementPolicy {
    pub fn new(mode: DisplayMeasurementMode) -> Self {
        Self {
            mode,
            proxy: DisplayProxyPolicy::default(),
            texture_format: PreviewTextureFormat::Rgba8Unorm,
            transfer_mode: PreviewTransferMode::ShaderSrgb,
        }
    }

    pub fn validate(self) -> Result<()> {
        match self.proxy {
            DisplayProxyPolicy::WindowAspectFit { .. } => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayGpuExecutionProfile {
    Current,
    WorkPlanScratchReuse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMeasurementPolicy {
    pub backend_preference: GpuBackendPreference,
    pub execution_profile: DisplayGpuExecutionProfile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayMeasurementRequest {
    pub run: DisplayMeasurementRunSpec,
    pub payload: PayloadReadPolicy,
    pub display: DisplayMeasurementPolicy,
    pub gpu: GpuMeasurementPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipeProducerMeasurementRunSpec {
    pub input_path: PathBuf,
    pub start_frame: usize,
    pub stride: usize,
    pub warmup_frames: usize,
    pub frames_requested: usize,
    pub selected_frames: Vec<FrameNumber>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipeProducerMeasurementPolicy {
    pub correction_mode: PipeF32BayerCorrectionMode,
    pub backend_preference: GpuBackendPreference,
    pub execution_profile: DisplayGpuExecutionProfile,
}

impl PipeProducerMeasurementPolicy {
    pub fn canonical_discard(backend_preference: GpuBackendPreference) -> Self {
        Self {
            correction_mode: PipeF32BayerCorrectionMode::MotionCamSpatial,
            backend_preference,
            execution_profile: DisplayGpuExecutionProfile::Current,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipeProducerMeasurementRequest {
    pub run: PipeProducerMeasurementRunSpec,
    pub payload: PayloadReadPolicy,
    pub policy: PipeProducerMeasurementPolicy,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PipeByteIdentityMetrics {
    pub frames_checked: usize,
    pub validation_frames_checked: usize,
    pub validation_bytes_sampled: u64,
    pub output_byte_count_ok: bool,
    pub frame_byte_count_ok: bool,
    pub plane_order_ok: bool,
    pub little_endian_ok: bool,
    pub meaningful_low_12_bits_ok: bool,
    pub code_bounds_ok: bool,
    pub mismatches: usize,
    pub first_mismatch_offset: Option<usize>,
}

impl PipeByteIdentityMetrics {
    pub fn status(self) -> &'static str {
        if self.mismatches == 0
            && self.output_byte_count_ok
            && self.frame_byte_count_ok
            && self.plane_order_ok
            && self.little_endian_ok
            && self.meaningful_low_12_bits_ok
            && self.code_bounds_ok
        {
            "OK"
        } else {
            "LENGTH_MISMATCH"
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipeByteIdentityRow {
    pub frame_index: Option<usize>,
    pub check_kind: String,
    pub status: String,
    pub expected_frame_bytes: Option<u64>,
    pub actual_frame_bytes: Option<u64>,
    pub expected_total_bytes: Option<u64>,
    pub actual_total_bytes: Option<u64>,
    pub plane_order_expected: String,
    pub little_endian_expected: String,
    pub output_target: String,
    pub first_mismatch_offset: Option<usize>,
    pub digest64: Option<u64>,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PipeProducerMetrics {
    pub correction_mode: PipeF32BayerCorrectionMode,
    pub warmup_frames_requested: usize,
    pub warmup_frames_processed: usize,
    pub warmup_s: f64,
    pub bytes_per_frame_expected: u64,
    pub output_bytes_expected: u64,
    pub output_bytes: u64,
    pub writer_bytes: u64,
    pub wall_no_write_s: f64,
    pub wall_with_writer_s: f64,
    pub fps_no_write: f64,
    pub fps_incl_write: f64,
    pub writer_s: f64,
    pub writer_flush_s: f64,
    pub render_encode_s: f64,
    pub readback_s: f64,
    pub pipe_pack_s: f64,
    pub mapped_consume_s: f64,
    pub histogram_s: f64,
    pub byte_identity_s: f64,
    pub validation_bytes_sampled: u64,
    pub byte_identity: PipeByteIdentityMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PipeProducerMeasurementResult {
    pub sink: MeasuredSink,
    pub status: MeasurementStatus,
    pub failure_stage: Option<String>,
    pub timing: MeasurementTiming,
    pub frames: FrameCountMetrics,
    pub input: InputClipMetrics,
    pub payload: PayloadMetrics,
    pub gpu: GpuStageMetrics,
    pub metrics: PipeProducerMetrics,
    pub byte_identity_rows: Vec<PipeByteIdentityRow>,
    pub notes: Vec<String>,
}

pub fn pipe_digest64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub fn validate_pipe_frame_byte_count(expected: u64, actual: u64) -> &'static str {
    if expected == actual {
        "OK"
    } else {
        "LENGTH_MISMATCH"
    }
}

pub fn pipe_validation_frame_indices(selected_frames: &[FrameNumber]) -> Vec<usize> {
    let mut frames = selected_frames
        .iter()
        .map(|frame| frame.0 as usize)
        .collect::<Vec<_>>();
    frames.dedup();
    if frames.len() <= 5 {
        return frames;
    }
    let first = frames[0];
    let middle = frames[frames.len() / 2];
    let last = frames[frames.len() - 1];
    let mut selected = vec![first, middle, last];
    selected.dedup();
    selected
}

pub fn pipe_validation_sample_ranges(byte_len: usize) -> Vec<(usize, usize)> {
    if byte_len == 0 {
        return Vec::new();
    }
    const SAMPLE_BYTES: usize = 64;
    let mut starts = vec![0usize];
    if byte_len > SAMPLE_BYTES && byte_len.is_multiple_of(3) {
        let plane_bytes = byte_len / 3;
        starts.push(plane_bytes);
        starts.push(plane_bytes.saturating_mul(2));
    } else if byte_len > SAMPLE_BYTES {
        starts.push(byte_len / 2);
        starts.push(byte_len.saturating_sub(SAMPLE_BYTES));
    }
    for start in &mut starts {
        *start -= *start % 2;
    }
    starts.sort_unstable();
    starts.dedup();
    starts
        .into_iter()
        .map(|start| {
            let end = start.saturating_add(SAMPLE_BYTES).min(byte_len);
            (start, (end - start) & !1)
        })
        .filter(|(_, len)| *len > 0)
        .collect()
}
