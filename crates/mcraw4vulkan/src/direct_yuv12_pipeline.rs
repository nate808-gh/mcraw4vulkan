//! Direct-YUV frame scheduler.
//!
//! One shared decoder slot, Bayer-correction stage, and direct-YUV stage are reused
//! in queue order. Two independent mapped output/context slots bound readback
//! and permit one later frame to be submitted while the earlier frame is
//! retired and written.

use std::error::Error;
use std::fmt;
use std::io::Write;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use mcraw4vulkan_core::{BayerPattern, FrameDimensions, FramePayloadLayout};
use mcraw4vulkan_gpu::{
    GpuDecodeBackend, GpuDecodeScratch, GpuDecodedPackedU16BufferView, GpuNoReadbackSlotAllocation,
    GpuPreparedType6WorkPlan, GpuPreparedType7WorkPlan, GpuSubmittedNoReadbackGpuStageOutput,
};
use mcraw4vulkan_render::{
    DIRECT_YUV12_NONFINITE_CAMERA, DIRECT_YUV12_NONFINITE_MAPPED, DIRECT_YUV12_NONFINITE_NCL,
    DIRECT_YUV12_STATUS_BYTE_LEN, DirectYuv12ColorTransform, GpuDirectYuv12DispatchStats,
    GpuDirectYuv12EncodeInput, GpuDirectYuv12Stage, Yuv444p12lePackPolicy,
};
use mcraw4vulkan_vignette::{
    FixedPointVignetteInputFacts, GpuPipeF32BayerDispatchStats, GpuPipeF32BayerPrepareInput,
    GpuPipeF32BayerStage, PipeF32BayerCorrectionFingerprint, PipeF32BayerCorrectionMode,
    PipeF32BayerNumericDomain,
};

use crate::strict_motioncam_color::{
    ClipSourceSha256, ColorContextFingerprintFacts, ColorContextFingerprintV2,
    VerifiedStrictPipeColorContextV2,
};

/// Exact retained readback-ring depth used by production scheduling.
pub const DIRECT_YUV12_READBACK_SLOTS: usize = 2;
const SHARED_DECODER_SLOT: usize = 0;
const MAP_TIMEOUT: Duration = Duration::from_secs(30);
const BAYER_CORRECTION_PARAMS_BYTES: u64 = 128;
const BAYER_CORRECTION_IDENTITY_GAIN_BYTES: u64 = 8;
const DIRECT_YUV12_PARAMS_BYTES: u64 = 64;
const DIRECT_YUV12_STATUS_RESET_BYTES: u64 = 16;
const DIRECT_YUV12_DEVICE_STATUS_BYTES: u64 = 16;
// Preserve the established fixed scheduler-memory admission budget. The
// reserved bytes cover queue-ordered stage bookkeeping outside visible output.
const DIRECT_YUV12_RESERVED_STAGE_AUX_BYTES: u64 = 96;
const DIRECT_YUV12_FIXED_STAGE_AUX_BYTES: u64 = BAYER_CORRECTION_PARAMS_BYTES
    + BAYER_CORRECTION_IDENTITY_GAIN_BYTES
    + DIRECT_YUV12_PARAMS_BYTES
    + DIRECT_YUV12_STATUS_RESET_BYTES
    + DIRECT_YUV12_DEVICE_STATUS_BYTES
    + DIRECT_YUV12_RESERVED_STAGE_AUX_BYTES;
const BUFFER_COPY_ALIGNMENT: u64 = wgpu::COPY_BUFFER_ALIGNMENT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectYuv12SourceFrameRange {
    pub first_frame_index: u64,
    pub frame_count: u64,
}

fn apply_linear_signal_scale(mut matrix: [f32; 9], factor: f32) -> [f32; 9] {
    for value in &mut matrix {
        *value *= factor;
    }
    matrix
}

#[derive(Clone, Copy)]
pub enum DirectYuv12FrameFeeder<'a> {
    NativePayload {
        raw_payload: &'a [u8],
        payload_layout: FramePayloadLayout,
    },
    CpuDecodedPackedU16 {
        decoded_pixel_bytes_le: &'a [u8],
    },
}

#[derive(Clone, Copy)]
enum PreparedNativeWorkPlan<'a> {
    None,
    Type6(GpuPreparedType6WorkPlan<'a>),
    Type7(GpuPreparedType7WorkPlan<'a>),
}

pub struct DirectYuv12FrameInput<'a> {
    pub feeder: DirectYuv12FrameFeeder<'a>,
    pub dimensions: FrameDimensions,
    pub correction_facts: &'a FixedPointVignetteInputFacts<'a>,
    pub correction_mode: PipeF32BayerCorrectionMode,
    pub verified_color: &'a VerifiedStrictPipeColorContextV2,
    pub source_sha256: ClipSourceSha256,
    pub source_frame_index: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectYuv12SubmissionToken {
    pub sequence: u64,
    pub source_frame_index: u64,
}

/// Ordered mapped-frame publication boundary.
///
/// The scheduler invokes this only after status and completion-context checks
/// pass. Implementations must not retain `planar_bytes`, which borrows a mapped
/// readback and is invalid after the callback returns.
pub trait DirectYuv12FrameSink {
    fn publish_mapped_frame(
        &mut self,
        identity: DirectYuv12FrameIdentity,
        planar_bytes: &[u8],
    ) -> Result<(), String>;

    fn record_publication_timing(
        &mut self,
        _identity: DirectYuv12FrameIdentity,
        _timing: DirectYuv12PublicationTiming,
    ) {
    }
}

pub struct DirectYuv12WriteSink<'a, W: Write> {
    writer: &'a mut W,
}

impl<'a, W: Write> DirectYuv12WriteSink<'a, W> {
    pub fn new(writer: &'a mut W) -> Self {
        Self { writer }
    }
}

impl<W: Write> DirectYuv12FrameSink for DirectYuv12WriteSink<'_, W> {
    fn publish_mapped_frame(
        &mut self,
        _identity: DirectYuv12FrameIdentity,
        planar_bytes: &[u8],
    ) -> Result<(), String> {
        self.writer
            .write_all(planar_bytes)
            .map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectYuv12FrameIdentity {
    pub sequence: u64,
    pub source_frame_index: u64,
    pub source_sha256: [u8; 32],
    pub dimensions: FrameDimensions,
    pub bayer_pattern: BayerPattern,
    pub color_context_fingerprint: ColorContextFingerprintV2,
    pub correction_fingerprint: PipeF32BayerCorrectionFingerprint,
    pub correction_mode: PipeF32BayerCorrectionMode,
}

#[derive(Debug, Clone, Copy)]
struct DirectYuv12FrameIdentitySeed {
    sequence: u64,
    source_frame_index: u64,
    source_sha256: [u8; 32],
    color_context_fingerprint: ColorContextFingerprintV2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectYuv12FrameGpuStats {
    pub identity: DirectYuv12FrameIdentity,
    pub bayer_correction: GpuPipeF32BayerDispatchStats,
    pub direct_yuv12: GpuDirectYuv12DispatchStats,
    pub visible_output_bytes: u64,
    pub status_offset: u64,
    pub composite_readback_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectYuv12PublicationTiming {
    pub queue_encode_submit: Duration,
    pub queue_submit_call: Duration,
    pub map_wait: Duration,
    pub direct_mapped_write: Duration,
    pub elapsed_since_scheduler_start: Duration,
    pub interval_since_previous_publication: Option<Duration>,
    pub wait_reason: DirectYuv12WaitReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectYuv12WaitReason {
    RingFull,
    FinalDrain,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DirectYuv12PipelineStats {
    pub submitted_frames: u64,
    pub published_frames: u64,
    pub ring_full_waits: u64,
    pub final_drain_waits: u64,
    pub readback_allocation_count: u64,
    pub readback_reuse_count: u64,
    pub maximum_pending_frames: u32,
    pub maximum_readback_bytes_per_slot: u64,
    /// Maximum aggregate explicit GPU bytes admitted by the complete
    /// capacity-aware scheduler budget formula during this run.
    pub maximum_required_explicit_memory_bytes: u64,
    pub total_queue_encode_submit: Duration,
    pub total_queue_submit_call: Duration,
    pub total_map_wait: Duration,
    pub total_direct_mapped_write: Duration,
    pub first_output_latency: Option<Duration>,
    pub final_drain_duration: Duration,
    pub decode_work_plan_reused_frames: u64,
    pub decode_work_plan_scratch_grow_count: u64,
    pub bayer_correction_output_allocation_count: u64,
    pub bayer_correction_output_reuse_count: u64,
    pub bayer_correction_bind_group_allocation_count: u64,
    pub bayer_correction_bind_group_reuse_count: u64,
    pub direct_yuv12_output_allocation_count: u64,
    pub direct_yuv12_output_reuse_count: u64,
    pub direct_yuv12_bind_group_allocation_count: u64,
    pub direct_yuv12_bind_group_reuse_count: u64,
}

struct PendingDirectYuv12Frame {
    submitted: GpuSubmittedNoReadbackGpuStageOutput<DirectYuv12FrameGpuStats>,
    expected_identity: DirectYuv12FrameIdentity,
}

#[derive(Default)]
struct DirectYuv12ReadbackSlot {
    buffer: Option<wgpu::Buffer>,
    allocated_bytes: u64,
    pending: Option<PendingDirectYuv12Frame>,
    map_active: bool,
}

/// One queue-ordered decoder/compute set with exactly two retained readbacks.
///
/// Decoder slot zero, Bayer-correction, and direct-YUV storage are reused only
/// after each frame has been submitted to the same ordered queue. Each submit
/// copies output and status into a distinct retained readback before later
/// commands can overwrite the shared resources. Runtime decoder timestamps
/// are prohibited because their recorder storage is not independently ringed.
pub struct OneSharedComputeTwoReadbackDirectYuv12Scheduler {
    backend: GpuDecodeBackend,
    bayer_correction: GpuPipeF32BayerStage,
    direct_yuv12: GpuDirectYuv12Stage,
    decode_scratch: GpuDecodeScratch,
    readback_slots: [DirectYuv12ReadbackSlot; DIRECT_YUV12_READBACK_SLOTS],
    effective_readback_slots: usize,
    source_range: DirectYuv12SourceFrameRange,
    next_source_frame_index: u64,
    expected_source_end_exclusive: u64,
    configured_memory_budget_bytes: u64,
    retained_compact_gain_bytes: u64,
    next_sequence: u64,
    next_to_publish: u64,
    stats: DirectYuv12PipelineStats,
    terminal_error: Option<String>,
    finished: bool,
    started_at: Instant,
    last_publication_at: Option<Instant>,
    map_callback_sender: mpsc::SyncSender<(u64, Result<(), wgpu::BufferAsyncError>)>,
    map_callback_receiver: mpsc::Receiver<(u64, Result<(), wgpu::BufferAsyncError>)>,
    linear_signal_scale_factor: f32,
}

impl OneSharedComputeTwoReadbackDirectYuv12Scheduler {
    /// Construct the production scheduler with the fixed 1/2 scene-linear
    /// signal scale and all validation-only controls disabled.
    pub fn new(
        backend: GpuDecodeBackend,
        source_range: DirectYuv12SourceFrameRange,
        configured_memory_budget_bytes: u64,
    ) -> Result<Self, DirectYuv12PipelineError> {
        if backend.gpu_timestamp_support().enabled {
            return Err(DirectYuv12PipelineError::SharedDecoderTimestampsEnabled);
        }
        Self::new_with_effective_depth(
            backend,
            source_range,
            configured_memory_budget_bytes,
            DIRECT_YUV12_READBACK_SLOTS,
            Yuv444p12lePackPolicy::LINEAR_SIGNAL_SCALE_FACTOR_F32,
        )
    }

    fn new_with_effective_depth(
        backend: GpuDecodeBackend,
        source_range: DirectYuv12SourceFrameRange,
        configured_memory_budget_bytes: u64,
        effective_readback_slots: usize,
        linear_signal_scale_factor: f32,
    ) -> Result<Self, DirectYuv12PipelineError> {
        if !(1..=DIRECT_YUV12_READBACK_SLOTS).contains(&effective_readback_slots) {
            return Err(DirectYuv12PipelineError::InvalidReadbackDepth {
                effective_readback_slots,
            });
        }
        if let Some((slot_index, allocation)) = backend.first_nonpristine_decoder_allocation() {
            return Err(DirectYuv12PipelineError::DecoderBackendNotPristine {
                slot_index,
                allocation: Box::new(allocation),
            });
        }
        if source_range.frame_count == 0 {
            return Err(DirectYuv12PipelineError::InvalidSourceRange(source_range));
        }
        let expected_source_end_exclusive = source_range
            .first_frame_index
            .checked_add(source_range.frame_count)
            .ok_or(DirectYuv12PipelineError::InvalidSourceRange(source_range))?;
        if configured_memory_budget_bytes == 0 {
            return Err(DirectYuv12PipelineError::InvalidMemoryBudget {
                configured_bytes: configured_memory_budget_bytes,
            });
        }
        let fixed_required_bytes = DIRECT_YUV12_FIXED_STAGE_AUX_BYTES;
        if configured_memory_budget_bytes < fixed_required_bytes {
            return Err(DirectYuv12PipelineError::MinimumMemoryBudget {
                configured_bytes: configured_memory_budget_bytes,
                fixed_required_bytes,
            });
        }
        let bayer_correction = GpuPipeF32BayerStage::new(backend.device(), backend.queue())
            .map_err(|error| DirectYuv12PipelineError::BayerCorrection(error.to_string()))?;
        let direct_yuv12 = GpuDirectYuv12Stage::new(backend.device(), backend.queue())
            .map_err(|error| DirectYuv12PipelineError::DirectYuv12(error.to_string()))?;
        let (map_callback_sender, map_callback_receiver) = mpsc::sync_channel(1);
        Ok(Self {
            backend,
            bayer_correction,
            direct_yuv12,
            decode_scratch: GpuDecodeScratch::new(),
            readback_slots: std::array::from_fn(|_| DirectYuv12ReadbackSlot::default()),
            effective_readback_slots,
            source_range,
            next_source_frame_index: source_range.first_frame_index,
            expected_source_end_exclusive,
            configured_memory_budget_bytes,
            retained_compact_gain_bytes: 0,
            next_sequence: 0,
            next_to_publish: 0,
            stats: DirectYuv12PipelineStats::default(),
            terminal_error: None,
            finished: false,
            started_at: Instant::now(),
            last_publication_at: None,
            map_callback_sender,
            map_callback_receiver,
            linear_signal_scale_factor,
        })
    }

    pub fn pending_frame_count(&self) -> usize {
        self.readback_slots
            .iter()
            .filter(|slot| slot.pending.is_some())
            .count()
    }

    pub fn submit_frame<S: DirectYuv12FrameSink>(
        &mut self,
        frame: DirectYuv12FrameInput<'_>,
        sink: &mut S,
    ) -> Result<DirectYuv12SubmissionToken, DirectYuv12PipelineError> {
        self.ensure_running()?;
        match self.submit_frame_inner(frame, sink) {
            Ok(token) => Ok(token),
            Err(error) => {
                self.poison(&error);
                Err(error)
            }
        }
    }

    fn submit_frame_inner<S: DirectYuv12FrameSink>(
        &mut self,
        frame: DirectYuv12FrameInput<'_>,
        sink: &mut S,
    ) -> Result<DirectYuv12SubmissionToken, DirectYuv12PipelineError> {
        let mut decode_scratch = std::mem::take(&mut self.decode_scratch);
        let result = self.submit_frame_inner_with_scratch(frame, sink, &mut decode_scratch);
        self.decode_scratch = decode_scratch;
        result
    }

    fn submit_frame_inner_with_scratch<S: DirectYuv12FrameSink>(
        &mut self,
        frame: DirectYuv12FrameInput<'_>,
        sink: &mut S,
        decode_scratch: &mut GpuDecodeScratch,
    ) -> Result<DirectYuv12SubmissionToken, DirectYuv12PipelineError> {
        self.validate_frame_input(&frame)?;
        let sequence = self.next_sequence;
        if sequence >= self.source_range.frame_count {
            return Err(DirectYuv12PipelineError::TooManyFrames {
                expected: self.source_range.frame_count,
                attempted_sequence: sequence,
            });
        }
        if frame.source_frame_index != self.next_source_frame_index {
            return Err(DirectYuv12PipelineError::SourceFrameOrder {
                expected: self.next_source_frame_index,
                actual: frame.source_frame_index,
            });
        }
        let next_sequence = sequence
            .checked_add(1)
            .ok_or(DirectYuv12PipelineError::SequenceOverflow)?;
        let next_source_frame_index = frame.source_frame_index.checked_add(1).ok_or(
            DirectYuv12PipelineError::SourceFrameIndexOverflow {
                source_frame_index: frame.source_frame_index,
            },
        )?;
        let readback_slot_index = usize::try_from(
            frame.source_frame_index % u64::try_from(self.effective_readback_slots).unwrap_or(2),
        )
        .unwrap_or(0);

        if self.readback_slots[readback_slot_index].pending.is_some() {
            if let Err(error) =
                self.complete_slot(readback_slot_index, DirectYuv12WaitReason::RingFull, sink)
            {
                self.discard_pending();
                return Err(error);
            }
        }

        let pixel_count =
            frame
                .dimensions
                .pixel_count()
                .ok_or(DirectYuv12PipelineError::InvalidDimensions(
                    frame.dimensions,
                ))?;
        let pixel_count_u64 = u64::try_from(pixel_count)
            .map_err(|_| DirectYuv12PipelineError::InvalidDimensions(frame.dimensions))?;
        let visible_output_bytes =
            pixel_count_u64
                .checked_mul(6)
                .ok_or(DirectYuv12PipelineError::InvalidDimensions(
                    frame.dimensions,
                ))?;
        let status_end = visible_output_bytes
            .checked_add(DIRECT_YUV12_STATUS_BYTE_LEN)
            .ok_or(DirectYuv12PipelineError::InvalidDimensions(
                frame.dimensions,
            ))?;
        let composite_end = status_end;
        let composite_readback_bytes = align_up_checked(composite_end, BUFFER_COPY_ALIGNMENT)
            .ok_or(DirectYuv12PipelineError::InvalidDimensions(
                frame.dimensions,
            ))?;
        let prepared_work_plan = match frame.feeder {
            DirectYuv12FrameFeeder::NativePayload {
                raw_payload,
                payload_layout: FramePayloadLayout::CompressedRawcodecType7,
            } => PreparedNativeWorkPlan::Type7(
                decode_scratch
                    .prepare_type7(raw_payload, frame.dimensions)
                    .map_err(|error| DirectYuv12PipelineError::Decode(error.to_string()))?,
            ),
            DirectYuv12FrameFeeder::NativePayload {
                raw_payload,
                payload_layout: FramePayloadLayout::BinnedRaw16Type6 { row_stride },
            } => PreparedNativeWorkPlan::Type6(
                decode_scratch
                    .prepare_type6(raw_payload, frame.dimensions, row_stride)
                    .map_err(|error| DirectYuv12PipelineError::Decode(error.to_string()))?,
            ),
            DirectYuv12FrameFeeder::CpuDecodedPackedU16 { .. } => PreparedNativeWorkPlan::None,
        };
        let decoder_allocation = match frame.feeder {
            DirectYuv12FrameFeeder::NativePayload {
                raw_payload,
                payload_layout: FramePayloadLayout::CompressedRawcodecType7,
            } => self
                .backend
                .prospective_prepared_type7_no_readback_slot_allocation(
                    SHARED_DECODER_SLOT,
                    raw_payload,
                    frame.dimensions,
                    match prepared_work_plan {
                        PreparedNativeWorkPlan::Type7(prepared) => prepared,
                        _ => unreachable!("type-7 feeder has a prepared plan"),
                    },
                ),
            DirectYuv12FrameFeeder::NativePayload {
                raw_payload,
                payload_layout: FramePayloadLayout::BinnedRaw16Type6 { row_stride },
            } => self
                .backend
                .prospective_prepared_type6_no_readback_slot_allocation(
                    SHARED_DECODER_SLOT,
                    raw_payload,
                    frame.dimensions,
                    row_stride,
                    match prepared_work_plan {
                        PreparedNativeWorkPlan::Type6(prepared) => prepared,
                        _ => unreachable!("type-6 feeder has a prepared plan"),
                    },
                ),
            DirectYuv12FrameFeeder::CpuDecodedPackedU16 { .. } => self
                .backend
                .prospective_cpu_upload_no_readback_slot_allocation(
                    SHARED_DECODER_SLOT,
                    frame.dimensions,
                ),
        }
        .map_err(|error| DirectYuv12PipelineError::Decode(error.to_string()))?;
        validate_decoder_allocation_limits(decoder_allocation, self.backend.limits())?;
        let prospective_compact_gain_bytes = self.validate_memory_budget(
            pixel_count_u64,
            composite_readback_bytes,
            decoder_allocation,
            frame.correction_facts,
            frame.correction_mode,
        )?;
        self.ensure_readback_slot(readback_slot_index, composite_readback_bytes)?;

        let expected_correction_fingerprint = PipeF32BayerCorrectionFingerprint::from_fixed_facts(
            frame.correction_facts,
            frame.correction_mode,
        )
        .map_err(|error| DirectYuv12PipelineError::BayerCorrection(error.to_string()))?;
        let expected_identity = DirectYuv12FrameIdentity {
            sequence,
            source_frame_index: frame.source_frame_index,
            source_sha256: frame.source_sha256.bytes(),
            dimensions: frame.dimensions,
            bayer_pattern: frame.correction_facts.bayer_pattern,
            color_context_fingerprint: frame.verified_color.fingerprint(),
            correction_fingerprint: expected_correction_fingerprint,
            correction_mode: frame.correction_mode,
        };

        let identity_seed = DirectYuv12FrameIdentitySeed {
            sequence,
            source_frame_index: frame.source_frame_index,
            source_sha256: frame.source_sha256.bytes(),
            color_context_fingerprint: frame.verified_color.fingerprint(),
        };
        let readback_buffer = self.readback_slots[readback_slot_index]
            .buffer
            .as_ref()
            .expect("readback slot was ensured before encoding");
        let correction_facts = frame.correction_facts;
        let correction_mode = frame.correction_mode;
        let frame_color = frame.verified_color.resolved();
        let color_transform = DirectYuv12ColorTransform {
            camera_to_normalized_ncl: apply_linear_signal_scale(
                frame_color.camera_to_normalized_ncl_f32(),
                self.linear_signal_scale_factor,
            ),
        };

        let bayer_correction = &mut self.bayer_correction;
        let direct_yuv12 = &mut self.direct_yuv12;
        let encode = |device: &wgpu::Device,
                      queue: &wgpu::Queue,
                      encoder: &mut wgpu::CommandEncoder,
                      decoded: GpuDecodedPackedU16BufferView<'_>|
         -> anyhow::Result<DirectYuv12FrameGpuStats> {
            let corrected = bayer_correction
                .prepare_pipe_f32_bayer(GpuPipeF32BayerPrepareInput {
                    device,
                    queue,
                    encoder,
                    input_buffer: decoded.buffer,
                    input_buffer_bytes: decoded.byte_len,
                    facts: correction_facts,
                    correction_mode,
                })
                .map_err(|error| anyhow::anyhow!("Bayer correction failed: {error}"))?;
            frame_color
                .validate_demosaiced_component_bound(
                    corrected.view.conservative_max_abs_four_tap_sum(),
                )
                .map_err(|error| {
                    anyhow::anyhow!("strict color bound validation failed: {error}")
                })?;
            let correction_fingerprint = corrected.view.correction_fingerprint();
            let direct = direct_yuv12
                .encode_direct_yuv12(GpuDirectYuv12EncodeInput {
                    device,
                    queue,
                    encoder,
                    bayer: &corrected.view,
                    color_transform,
                    pack_policy: Yuv444p12lePackPolicy::adopted(),
                })
                .map_err(|error| anyhow::anyhow!("direct YUV12 encode failed: {error}"))?;
            encoder.copy_buffer_to_buffer(
                direct.view.output_buffer(),
                0,
                readback_buffer,
                0,
                direct.view.visible_byte_len(),
            );
            encoder.copy_buffer_to_buffer(
                direct.view.status_buffer(),
                0,
                readback_buffer,
                direct.view.visible_byte_len(),
                direct.view.status_byte_len(),
            );
            Ok(DirectYuv12FrameGpuStats {
                identity: DirectYuv12FrameIdentity {
                    sequence: identity_seed.sequence,
                    source_frame_index: identity_seed.source_frame_index,
                    source_sha256: identity_seed.source_sha256,
                    dimensions: corrected.view.dimensions(),
                    bayer_pattern: corrected.view.bayer_pattern(),
                    color_context_fingerprint: identity_seed.color_context_fingerprint,
                    correction_fingerprint,
                    correction_mode: corrected.view.correction_mode(),
                },
                bayer_correction: corrected.stats,
                direct_yuv12: direct.stats,
                visible_output_bytes: direct.view.visible_byte_len(),
                status_offset: direct.view.visible_byte_len(),
                composite_readback_bytes,
            })
        };

        let submitted = match frame.feeder {
            DirectYuv12FrameFeeder::NativePayload {
                raw_payload,
                payload_layout: FramePayloadLayout::CompressedRawcodecType7,
            } => self
                .backend
                .submit_prepared_raw_payload_packed_u16_gpu_stage_no_readback_slot_with_vignette(
                    SHARED_DECODER_SLOT,
                    raw_payload,
                    frame.dimensions,
                    match prepared_work_plan {
                        PreparedNativeWorkPlan::Type7(prepared) => prepared,
                        _ => unreachable!("type-7 feeder has a prepared plan"),
                    },
                    None,
                    encode,
                ),
            DirectYuv12FrameFeeder::NativePayload {
                raw_payload,
                payload_layout: FramePayloadLayout::BinnedRaw16Type6 { row_stride },
            } => self
                .backend
                .submit_prepared_legacy_raw16_payload_packed_u16_gpu_stage_no_readback_slot_with_vignette(
                    SHARED_DECODER_SLOT,
                    raw_payload,
                    frame.dimensions,
                    row_stride,
                    match prepared_work_plan {
                        PreparedNativeWorkPlan::Type6(prepared) => prepared,
                        _ => unreachable!("type-6 feeder has a prepared plan"),
                    },
                    None,
                    encode,
                ),
            DirectYuv12FrameFeeder::CpuDecodedPackedU16 {
                decoded_pixel_bytes_le,
            } => self
                .backend
                .submit_cpu_decoded_bayer_u16_gpu_stage_no_readback_slot_with_vignette(
                    SHARED_DECODER_SLOT,
                    decoded_pixel_bytes_le,
                    frame.dimensions,
                    None,
                    encode,
                ),
        }
        .map_err(|error| DirectYuv12PipelineError::Decode(error.to_string()))?;

        let identity = submitted.stage.identity;
        if submitted.timings.work_plan_reused_scratch {
            self.stats.decode_work_plan_reused_frames = self
                .stats
                .decode_work_plan_reused_frames
                .checked_add(1)
                .ok_or(DirectYuv12PipelineError::SequenceOverflow)?;
        }
        self.stats.decode_work_plan_scratch_grow_count = self
            .stats
            .decode_work_plan_scratch_grow_count
            .checked_add(submitted.timings.work_plan_scratch_grow_count)
            .ok_or(DirectYuv12PipelineError::SequenceOverflow)?;
        self.stats.bayer_correction_bind_group_allocation_count =
            submitted.stage.bayer_correction.bind_group_allocation_count;
        self.stats.bayer_correction_bind_group_reuse_count =
            submitted.stage.bayer_correction.bind_group_reuse_count;
        self.stats.bayer_correction_output_allocation_count =
            submitted.stage.bayer_correction.output_allocation_count;
        self.stats.bayer_correction_output_reuse_count =
            submitted.stage.bayer_correction.output_reuse_count;
        self.stats.direct_yuv12_bind_group_allocation_count =
            submitted.stage.direct_yuv12.bind_group_allocation_count;
        self.stats.direct_yuv12_bind_group_reuse_count =
            submitted.stage.direct_yuv12.bind_group_reuse_count;
        self.stats.direct_yuv12_output_allocation_count =
            submitted.stage.direct_yuv12.output_allocation_count;
        self.stats.direct_yuv12_output_reuse_count =
            submitted.stage.direct_yuv12.output_reuse_count;
        self.stats.total_queue_encode_submit = self
            .stats
            .total_queue_encode_submit
            .saturating_add(submitted.timings.encode_submit);
        self.stats.total_queue_submit_call = self
            .stats
            .total_queue_submit_call
            .saturating_add(submitted.timings.command_finish_submit);
        self.readback_slots[readback_slot_index].pending = Some(PendingDirectYuv12Frame {
            submitted,
            expected_identity,
        });
        self.next_sequence = next_sequence;
        self.next_source_frame_index = next_source_frame_index;
        self.retained_compact_gain_bytes = prospective_compact_gain_bytes;
        self.stats.submitted_frames = self
            .stats
            .submitted_frames
            .checked_add(1)
            .ok_or(DirectYuv12PipelineError::SequenceOverflow)?;
        self.stats.maximum_pending_frames = self
            .stats
            .maximum_pending_frames
            .max(u32::try_from(self.pending_frame_count()).unwrap_or(u32::MAX));
        Ok(DirectYuv12SubmissionToken {
            sequence: identity.sequence,
            source_frame_index: identity.source_frame_index,
        })
    }

    pub fn finish<S: DirectYuv12FrameSink>(
        &mut self,
        sink: &mut S,
    ) -> Result<DirectYuv12PipelineStats, DirectYuv12PipelineError> {
        self.ensure_running()?;
        let drain_start = Instant::now();
        match self.finish_inner(sink) {
            Ok(_) => {
                self.stats.final_drain_duration = drain_start.elapsed();
                self.finished = true;
                Ok(self.stats)
            }
            Err(error) => {
                self.poison(&error);
                Err(error)
            }
        }
    }

    fn finish_inner<S: DirectYuv12FrameSink>(
        &mut self,
        sink: &mut S,
    ) -> Result<DirectYuv12PipelineStats, DirectYuv12PipelineError> {
        if self.next_sequence != self.source_range.frame_count
            || self.next_source_frame_index != self.expected_source_end_exclusive
        {
            return Err(DirectYuv12PipelineError::IncompleteSourceRange {
                first_frame_index: self.source_range.first_frame_index,
                expected_frame_count: self.source_range.frame_count,
                submitted_frames: self.next_sequence,
                next_expected_source_frame_index: self.next_source_frame_index,
            });
        }
        while self.next_to_publish < self.next_sequence {
            let slot_index = self
                .readback_slots
                .iter()
                .position(|slot| {
                    slot.pending.as_ref().is_some_and(|pending| {
                        pending.submitted.stage.identity.sequence == self.next_to_publish
                    })
                })
                .ok_or(DirectYuv12PipelineError::MissingSequence {
                    expected: self.next_to_publish,
                })?;
            if let Err(error) =
                self.complete_slot(slot_index, DirectYuv12WaitReason::FinalDrain, sink)
            {
                self.discard_pending();
                return Err(error);
            }
        }
        if self.pending_frame_count() != 0 || self.stats.published_frames != self.next_sequence {
            return Err(DirectYuv12PipelineError::IncompleteDrain {
                submitted: self.next_sequence,
                published: self.stats.published_frames,
                pending: self.pending_frame_count(),
            });
        }
        Ok(self.stats)
    }

    fn validate_frame_input(
        &self,
        frame: &DirectYuv12FrameInput<'_>,
    ) -> Result<(), DirectYuv12PipelineError> {
        let color_facts = frame.verified_color.fingerprint_facts();
        if frame.dimensions.width == 0
            || frame.dimensions.height == 0
            || frame.dimensions.width & 1 != 0
            || frame.correction_facts.frame_dimensions != frame.dimensions
            || frame.verified_color.resolved().source_frame_index() != frame.source_frame_index
            || frame.verified_color.resolved().provenance().source_sha256()
                != frame.source_sha256.bytes()
        {
            return Err(DirectYuv12PipelineError::InvalidFrameContext {
                dimensions: frame.dimensions,
                correction_dimensions: frame.correction_facts.frame_dimensions,
                source_frame_index: frame.source_frame_index,
                resolved_frame_index: frame.verified_color.resolved().source_frame_index(),
            });
        }
        validate_verified_color_facts(
            color_facts,
            frame.dimensions,
            frame.correction_facts.bayer_pattern,
            frame.correction_mode,
            frame.source_sha256,
            frame.source_frame_index,
        )?;
        if let DirectYuv12FrameFeeder::CpuDecodedPackedU16 {
            decoded_pixel_bytes_le,
        } = &frame.feeder
        {
            let expected_bytes = frame
                .dimensions
                .pixel_count()
                .and_then(|pixels| pixels.checked_mul(std::mem::size_of::<u16>()))
                .ok_or(DirectYuv12PipelineError::InvalidDimensions(
                    frame.dimensions,
                ))?;
            if decoded_pixel_bytes_le.len() != expected_bytes {
                return Err(DirectYuv12PipelineError::InvalidCpuFeederLength {
                    expected_bytes,
                    actual_bytes: decoded_pixel_bytes_le.len(),
                });
            }
        }
        Ok(())
    }

    fn ensure_running(&self) -> Result<(), DirectYuv12PipelineError> {
        if let Some(first_error) = self.terminal_error.as_ref() {
            return Err(DirectYuv12PipelineError::Poisoned {
                first_error: first_error.clone(),
            });
        }
        if self.finished {
            return Err(DirectYuv12PipelineError::AlreadyFinished);
        }
        Ok(())
    }

    fn poison(&mut self, error: &DirectYuv12PipelineError) {
        if self.terminal_error.is_none() {
            self.terminal_error = Some(error.to_string());
        }
        self.discard_pending();
    }

    fn validate_memory_budget(
        &mut self,
        pixel_count: u64,
        required_composite_readback_bytes: u64,
        decoder_allocation: GpuNoReadbackSlotAllocation,
        correction_facts: &FixedPointVignetteInputFacts<'_>,
        correction_mode: PipeF32BayerCorrectionMode,
    ) -> Result<u64, DirectYuv12PipelineError> {
        let requested_corrected_f32_bytes = pixel_count
            .checked_mul(4)
            .ok_or(DirectYuv12PipelineError::MemoryArithmeticOverflow)?;
        let requested_visible_yuv_bytes = pixel_count
            .checked_mul(6)
            .ok_or(DirectYuv12PipelineError::MemoryArithmeticOverflow)?;
        let packed_input_capacity_bytes = decoder_allocation.packed_output_bytes;
        let corrected_f32_capacity_bytes = self
            .bayer_correction
            .allocated_output_bytes()
            .max(requested_corrected_f32_bytes);
        let yuv_output_capacity_bytes = self
            .direct_yuv12
            .allocated_output_bytes()
            .max(requested_visible_yuv_bytes);
        let retained_readback_capacity = self
            .readback_slots
            .iter()
            .map(|slot| slot.allocated_bytes)
            .max()
            .unwrap_or(0);
        let composite_readback_capacity_bytes_per_slot = retained_readback_capacity
            .max(required_composite_readback_bytes)
            .checked_add(0)
            .ok_or(DirectYuv12PipelineError::MemoryArithmeticOverflow)?;
        let retained_readbacks_bytes = composite_readback_capacity_bytes_per_slot
            .checked_mul(u64::try_from(self.effective_readback_slots).unwrap_or(2))
            .ok_or(DirectYuv12PipelineError::MemoryArithmeticOverflow)?;
        let requested_compact_gain_bytes =
            compact_gain_binding_bytes(correction_facts, correction_mode)?;
        let compact_gain_capacity_bytes = self
            .retained_compact_gain_bytes
            .max(requested_compact_gain_bytes);
        let decoder_extra_bytes = decoder_allocation
            .decoder_extra_gpu_bytes()
            .map_err(|_| DirectYuv12PipelineError::MemoryArithmeticOverflow)?;
        let visible_principal_bytes = packed_input_capacity_bytes
            .checked_add(corrected_f32_capacity_bytes)
            .and_then(|value| value.checked_add(yuv_output_capacity_bytes))
            .and_then(|value| {
                yuv_output_capacity_bytes
                    .checked_mul(u64::try_from(self.effective_readback_slots).unwrap_or(2))
                    .and_then(|readbacks| value.checked_add(readbacks))
            })
            .ok_or(DirectYuv12PipelineError::MemoryArithmeticOverflow)?;
        let required_bytes = packed_input_capacity_bytes
            .checked_add(corrected_f32_capacity_bytes)
            .and_then(|value| value.checked_add(yuv_output_capacity_bytes))
            .and_then(|value| value.checked_add(retained_readbacks_bytes))
            .and_then(|value| value.checked_add(decoder_extra_bytes))
            .and_then(|value| value.checked_add(compact_gain_capacity_bytes))
            .and_then(|value| value.checked_add(DIRECT_YUV12_FIXED_STAGE_AUX_BYTES))
            .ok_or(DirectYuv12PipelineError::MemoryArithmeticOverflow)?;
        if required_bytes > self.configured_memory_budget_bytes {
            return Err(DirectYuv12PipelineError::ConfiguredMemoryBudgetExceeded(
                Box::new(DirectYuv12MemoryBudgetDetails {
                    pixel_count,
                    packed_input_capacity_bytes,
                    corrected_f32_capacity_bytes,
                    yuv_output_capacity_bytes,
                    composite_readback_capacity_bytes_per_slot,
                    readback_slot_count: self.effective_readback_slots,
                    decoder_extra_bytes,
                    compact_gain_capacity_bytes,
                    fixed_stage_aux_bytes: DIRECT_YUV12_FIXED_STAGE_AUX_BYTES,
                    visible_principal_bytes,
                    required_bytes,
                    configured_bytes: self.configured_memory_budget_bytes,
                }),
            ));
        }
        self.stats.maximum_required_explicit_memory_bytes = self
            .stats
            .maximum_required_explicit_memory_bytes
            .max(required_bytes);
        Ok(compact_gain_capacity_bytes)
    }

    fn ensure_readback_slot(
        &mut self,
        slot_index: usize,
        required_bytes: u64,
    ) -> Result<(), DirectYuv12PipelineError> {
        let max_buffer_size = self.backend.limits().max_buffer_size;
        if required_bytes > max_buffer_size {
            return Err(DirectYuv12PipelineError::ReadbackTooLarge {
                required_bytes,
                max_buffer_size,
            });
        }
        let slot = &mut self.readback_slots[slot_index];
        if slot.allocated_bytes >= required_bytes {
            self.stats.readback_reuse_count = self.stats.readback_reuse_count.saturating_add(1);
            return Ok(());
        }
        slot.buffer = Some(
            self.backend
                .device()
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mcraw4vulkan direct YUV12 output/status readback slot"),
                    size: required_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
        );
        slot.allocated_bytes = required_bytes;
        self.stats.readback_allocation_count =
            self.stats.readback_allocation_count.saturating_add(1);
        self.stats.maximum_readback_bytes_per_slot = self
            .stats
            .maximum_readback_bytes_per_slot
            .max(required_bytes);
        Ok(())
    }

    fn complete_slot<S: DirectYuv12FrameSink>(
        &mut self,
        slot_index: usize,
        wait_reason: DirectYuv12WaitReason,
        sink: &mut S,
    ) -> Result<(), DirectYuv12PipelineError> {
        {
            let pending = self.readback_slots[slot_index]
                .pending
                .as_ref()
                .ok_or(DirectYuv12PipelineError::EmptySlot { slot_index })?;
            validate_completion_identity(
                pending.expected_identity,
                pending.submitted.stage.identity,
            )?;
            if pending.submitted.stage.identity.sequence != self.next_to_publish {
                return Err(DirectYuv12PipelineError::PublicationOrder {
                    expected: self.next_to_publish,
                    actual: pending.submitted.stage.identity.sequence,
                });
            }
        }
        if self.readback_slots[slot_index].buffer.is_none() {
            return Err(DirectYuv12PipelineError::EmptySlot { slot_index });
        }
        let pending = self.readback_slots[slot_index]
            .pending
            .take()
            .ok_or(DirectYuv12PipelineError::EmptySlot { slot_index })?;
        let queue_encode_submit = pending.submitted.timings.encode_submit;
        let queue_submit_call = pending.submitted.timings.command_finish_submit;
        let map_wait_start = Instant::now();
        let stage = pending.submitted.stage;
        match wait_reason {
            DirectYuv12WaitReason::RingFull => {
                self.stats.ring_full_waits = self
                    .stats
                    .ring_full_waits
                    .checked_add(1)
                    .ok_or(DirectYuv12PipelineError::SequenceOverflow)?
            }
            DirectYuv12WaitReason::FinalDrain => {
                self.stats.final_drain_waits = self
                    .stats
                    .final_drain_waits
                    .checked_add(1)
                    .ok_or(DirectYuv12PipelineError::SequenceOverflow)?
            }
        }

        self.readback_slots[slot_index].map_active = true;
        let buffer = self.readback_slots[slot_index]
            .buffer
            .as_ref()
            .ok_or(DirectYuv12PipelineError::EmptySlot { slot_index })?;
        let slice = buffer.slice(..stage.composite_readback_bytes);
        // Exactly one callback is armed at a time. The channel is allocated
        // once with capacity one, so callback state cannot grow with clip
        // length and retirement performs no per-frame channel allocation.
        let sender = self.map_callback_sender.clone();
        let callback_sequence = stage.identity.sequence;
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send((callback_sequence, result));
        });
        // Arm the map before waiting for this submission. The one blocking
        // poll both retires the independently retained copy and drives its
        // map callback; no wait occurs on the shared compute resources before
        // the readback ring applies backpressure or the final drain begins.
        self.backend.device().poll(wgpu::Maintain::wait_for(
            pending.submitted.submission_index.clone(),
        ));
        let map_result = match self.map_callback_receiver.recv_timeout(MAP_TIMEOUT) {
            Ok((sequence, result)) => {
                validate_map_callback(stage.identity.sequence, sequence, result)
            }
            Err(_) => Err(DirectYuv12PipelineError::MapTimeout {
                sequence: stage.identity.sequence,
            }),
        };
        if let Err(error) = map_result {
            buffer.unmap();
            self.readback_slots[slot_index].map_active = false;
            return Err(error);
        }
        let map_wait = map_wait_start.elapsed();

        let direct_write_start = Instant::now();
        let write_result = {
            let mapped = slice.get_mapped_range();
            (|| {
                let status_start = usize::try_from(stage.status_offset)
                    .map_err(|_| DirectYuv12PipelineError::InvalidReadbackLayout)?;
                let status_end = status_start
                    .checked_add(usize::try_from(DIRECT_YUV12_STATUS_BYTE_LEN).unwrap_or(16))
                    .ok_or(DirectYuv12PipelineError::InvalidReadbackLayout)?;
                if mapped.len()
                    != usize::try_from(stage.composite_readback_bytes).unwrap_or(usize::MAX)
                    || status_end > mapped.len()
                {
                    Err(DirectYuv12PipelineError::InvalidReadbackLayout)
                } else {
                    validate_nonfinite_status(
                        &mapped[status_start..status_end],
                        stage.identity.sequence,
                        stage.identity.source_frame_index,
                    )?;
                    publish_validated_mapped_frame(&mapped[..status_start], stage.identity, sink)?;
                    Ok(())
                }
            })()
        };
        buffer.unmap();
        self.readback_slots[slot_index].map_active = false;
        let direct_mapped_write = direct_write_start.elapsed();
        self.stats.total_map_wait = self.stats.total_map_wait.saturating_add(map_wait);
        self.stats.total_direct_mapped_write = self
            .stats
            .total_direct_mapped_write
            .saturating_add(direct_mapped_write);
        write_result?;
        let publication_at = Instant::now();
        let elapsed_since_scheduler_start = publication_at.duration_since(self.started_at);
        let interval_since_previous_publication = self
            .last_publication_at
            .map(|previous| publication_at.duration_since(previous));
        let timing = DirectYuv12PublicationTiming {
            queue_encode_submit,
            queue_submit_call,
            map_wait,
            direct_mapped_write,
            elapsed_since_scheduler_start,
            interval_since_previous_publication,
            wait_reason,
        };
        sink.record_publication_timing(stage.identity, timing);
        self.last_publication_at = Some(publication_at);
        if self.stats.first_output_latency.is_none() {
            self.stats.first_output_latency = Some(elapsed_since_scheduler_start);
        }

        self.next_to_publish = self
            .next_to_publish
            .checked_add(1)
            .ok_or(DirectYuv12PipelineError::SequenceOverflow)?;
        self.stats.published_frames = self
            .stats
            .published_frames
            .checked_add(1)
            .ok_or(DirectYuv12PipelineError::SequenceOverflow)?;
        Ok(())
    }

    fn discard_pending(&mut self) {
        for slot in &mut self.readback_slots {
            if let Some(pending) = slot.pending.take() {
                self.backend
                    .device()
                    .poll(wgpu::Maintain::wait_for(pending.submitted.submission_index));
            }
            if slot.map_active
                && let Some(buffer) = slot.buffer.as_ref()
            {
                buffer.unmap();
                slot.map_active = false;
            }
        }
    }
}

fn validate_map_callback(
    expected_sequence: u64,
    received_sequence: u64,
    result: Result<(), wgpu::BufferAsyncError>,
) -> Result<(), DirectYuv12PipelineError> {
    if received_sequence != expected_sequence {
        return Err(DirectYuv12PipelineError::Map {
            sequence: expected_sequence,
            detail: format!("stale map callback for sequence {received_sequence}"),
        });
    }
    result.map_err(|error| DirectYuv12PipelineError::Map {
        sequence: expected_sequence,
        detail: format!("{error:?}"),
    })
}

fn validate_verified_color_facts(
    color_facts: ColorContextFingerprintFacts,
    dimensions: FrameDimensions,
    bayer_pattern: BayerPattern,
    correction_mode: PipeF32BayerCorrectionMode,
    source_sha256: ClipSourceSha256,
    source_frame_index: u64,
) -> Result<(), DirectYuv12PipelineError> {
    for (matches, reason) in [
        (
            color_facts.numeric_domain == PipeF32BayerNumericDomain::RelativeLinearCorrectedCodeV1,
            "numeric domain",
        ),
        (color_facts.dimensions == dimensions, "dimensions"),
        (color_facts.bayer_pattern == bayer_pattern, "CFA pattern"),
        (
            color_facts.correction_mode == correction_mode,
            "correction mode",
        ),
        (color_facts.source_sha256 == source_sha256, "source SHA-256"),
        (
            color_facts.source_frame_index == source_frame_index,
            "source frame index",
        ),
    ] {
        if !matches {
            return Err(DirectYuv12PipelineError::VerifiedColorContextMismatch { reason });
        }
    }
    Ok(())
}

fn align_up_checked(value: u64, alignment: u64) -> Option<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return None;
    }
    value
        .checked_add(alignment.checked_sub(1)?)
        .map(|sum| sum & !(alignment - 1))
}

fn validate_decoder_allocation_limits(
    allocation: GpuNoReadbackSlotAllocation,
    limits: &wgpu::Limits,
) -> Result<(), DirectYuv12PipelineError> {
    let max_buffer_size = limits.max_buffer_size;
    for (buffer, required_bytes) in [
        ("raw payload", allocation.raw_payload_bytes),
        ("work items", allocation.work_item_bytes),
        ("packed output", allocation.packed_output_bytes),
        ("params", allocation.params_bytes),
        ("retained readback", allocation.retained_readback_bytes),
        (
            "retained mappable output",
            allocation.retained_mappable_output_bytes,
        ),
        ("timestamp buffers", allocation.timestamp_buffer_bytes),
    ] {
        validate_decoder_buffer_limit(buffer, "max_buffer_size", required_bytes, max_buffer_size)?;
    }
    let storage_limit = u64::from(limits.max_storage_buffer_binding_size);
    for (buffer, required_bytes) in [
        ("raw payload", allocation.raw_payload_bytes),
        ("work items", allocation.work_item_bytes),
        ("packed output", allocation.packed_output_bytes),
    ] {
        validate_decoder_buffer_limit(
            buffer,
            "max_storage_buffer_binding_size",
            required_bytes,
            storage_limit,
        )?;
    }
    validate_decoder_buffer_limit(
        "params",
        "max_uniform_buffer_binding_size",
        allocation.params_bytes,
        u64::from(limits.max_uniform_buffer_binding_size),
    )?;
    Ok(())
}

fn validate_decoder_buffer_limit(
    buffer: &'static str,
    limit_kind: &'static str,
    required_bytes: u64,
    limit_bytes: u64,
) -> Result<(), DirectYuv12PipelineError> {
    if required_bytes > limit_bytes {
        return Err(DirectYuv12PipelineError::DecoderBufferLimit {
            buffer,
            limit_kind,
            required_bytes,
            limit_bytes,
        });
    }
    Ok(())
}

fn compact_gain_binding_bytes(
    facts: &FixedPointVignetteInputFacts<'_>,
    mode: PipeF32BayerCorrectionMode,
) -> Result<u64, DirectYuv12PipelineError> {
    match mode {
        PipeF32BayerCorrectionMode::IdentitySpatialGain => Ok(0),
        PipeF32BayerCorrectionMode::MotionCamSpatial => {
            let map = facts.lens_shading_map.as_ref().ok_or_else(|| {
                DirectYuv12PipelineError::BayerCorrection(
                    "MotionCam spatial correction requires a compact gain map".to_owned(),
                )
            })?;
            u64::try_from(map.width())
                .ok()
                .and_then(|width| {
                    u64::try_from(map.height())
                        .ok()
                        .and_then(|height| width.checked_mul(height))
                })
                .and_then(|samples| {
                    u64::try_from(map.plane_count())
                        .ok()
                        .and_then(|planes| samples.checked_mul(planes))
                })
                .and_then(|samples| samples.checked_mul(8))
                .ok_or(DirectYuv12PipelineError::MemoryArithmeticOverflow)
        }
    }
}

fn validate_nonfinite_status(
    bytes: &[u8],
    sequence: u64,
    source_frame_index: u64,
) -> Result<(), DirectYuv12PipelineError> {
    if bytes.len() != usize::try_from(DIRECT_YUV12_STATUS_BYTE_LEN).unwrap_or(16) {
        return Err(DirectYuv12PipelineError::InvalidReadbackLayout);
    }
    let mut words = [0_u32; 4];
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        words[index] = u32::from_le_bytes(chunk.try_into().unwrap_or([0; 4]));
    }
    if words != [0, 0, u32::MAX, 0] {
        let flags = words[0];
        return Err(DirectYuv12PipelineError::NonfiniteStatus {
            sequence,
            source_frame_index,
            categories: DirectYuv12NonfiniteCategories {
                camera: flags & DIRECT_YUV12_NONFINITE_CAMERA != 0,
                ncl: flags & DIRECT_YUV12_NONFINITE_NCL != 0,
                mapped: flags & DIRECT_YUV12_NONFINITE_MAPPED != 0,
                unknown_bits: flags
                    & !(DIRECT_YUV12_NONFINITE_CAMERA
                        | DIRECT_YUV12_NONFINITE_NCL
                        | DIRECT_YUV12_NONFINITE_MAPPED),
            },
            words,
        });
    }
    Ok(())
}

fn publish_validated_mapped_frame<S: DirectYuv12FrameSink>(
    planar_bytes: &[u8],
    identity: DirectYuv12FrameIdentity,
    sink: &mut S,
) -> Result<(), DirectYuv12PipelineError> {
    sink.publish_mapped_frame(identity, planar_bytes)
        .map_err(|detail| DirectYuv12PipelineError::Writer {
            sequence: identity.sequence,
            source_frame_index: identity.source_frame_index,
            detail,
        })
}

fn validate_completion_identity(
    expected: DirectYuv12FrameIdentity,
    actual: DirectYuv12FrameIdentity,
) -> Result<(), DirectYuv12PipelineError> {
    if expected != actual {
        return Err(DirectYuv12PipelineError::CompletionContextMismatch(
            Box::new(DirectYuv12CompletionContextMismatch { expected, actual }),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectYuv12MemoryBudgetDetails {
    pub pixel_count: u64,
    pub packed_input_capacity_bytes: u64,
    pub corrected_f32_capacity_bytes: u64,
    pub yuv_output_capacity_bytes: u64,
    pub composite_readback_capacity_bytes_per_slot: u64,
    pub readback_slot_count: usize,
    pub decoder_extra_bytes: u64,
    pub compact_gain_capacity_bytes: u64,
    pub fixed_stage_aux_bytes: u64,
    pub visible_principal_bytes: u64,
    pub required_bytes: u64,
    pub configured_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectYuv12CompletionContextMismatch {
    pub expected: DirectYuv12FrameIdentity,
    pub actual: DirectYuv12FrameIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectYuv12NonfiniteCategories {
    pub camera: bool,
    pub ncl: bool,
    pub mapped: bool,
    pub unknown_bits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectYuv12PipelineError {
    SharedDecoderTimestampsEnabled,
    InvalidReadbackDepth {
        effective_readback_slots: usize,
    },
    InvalidSourceRange(DirectYuv12SourceFrameRange),
    InvalidMemoryBudget {
        configured_bytes: u64,
    },
    MinimumMemoryBudget {
        configured_bytes: u64,
        fixed_required_bytes: u64,
    },
    DecoderBackendNotPristine {
        slot_index: Option<usize>,
        allocation: Box<GpuNoReadbackSlotAllocation>,
    },
    DecoderBufferLimit {
        buffer: &'static str,
        limit_kind: &'static str,
        required_bytes: u64,
        limit_bytes: u64,
    },
    MemoryArithmeticOverflow,
    ConfiguredMemoryBudgetExceeded(Box<DirectYuv12MemoryBudgetDetails>),
    SourceFrameOrder {
        expected: u64,
        actual: u64,
    },
    SourceFrameIndexOverflow {
        source_frame_index: u64,
    },
    SequenceOverflow,
    TooManyFrames {
        expected: u64,
        attempted_sequence: u64,
    },
    IncompleteSourceRange {
        first_frame_index: u64,
        expected_frame_count: u64,
        submitted_frames: u64,
        next_expected_source_frame_index: u64,
    },
    Poisoned {
        first_error: String,
    },
    AlreadyFinished,
    InvalidDimensions(FrameDimensions),
    InvalidCpuFeederLength {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    InvalidFrameContext {
        dimensions: FrameDimensions,
        correction_dimensions: FrameDimensions,
        source_frame_index: u64,
        resolved_frame_index: u64,
    },
    VerifiedColorContextMismatch {
        reason: &'static str,
    },
    BayerCorrection(String),
    DirectYuv12(String),
    Decode(String),
    ReadbackTooLarge {
        required_bytes: u64,
        max_buffer_size: u64,
    },
    EmptySlot {
        slot_index: usize,
    },
    MissingSequence {
        expected: u64,
    },
    PublicationOrder {
        expected: u64,
        actual: u64,
    },
    MapTimeout {
        sequence: u64,
    },
    Map {
        sequence: u64,
        detail: String,
    },
    InvalidReadbackLayout,
    NonfiniteStatus {
        sequence: u64,
        source_frame_index: u64,
        categories: DirectYuv12NonfiniteCategories,
        words: [u32; 4],
    },
    Writer {
        sequence: u64,
        source_frame_index: u64,
        detail: String,
    },
    IncompleteDrain {
        submitted: u64,
        published: u64,
        pending: usize,
    },
    CompletionContextMismatch(Box<DirectYuv12CompletionContextMismatch>),
}

impl fmt::Display for DirectYuv12PipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SharedDecoderTimestampsEnabled => write!(
                formatter,
                "shared decoder scheduling requires runtime backend timestamps disabled"
            ),
            Self::InvalidReadbackDepth {
                effective_readback_slots,
            } => write!(
                formatter,
                "direct YUV12 readback depth must be one or two, got {effective_readback_slots}"
            ),
            Self::InvalidSourceRange(range) => write!(
                formatter,
                "invalid direct YUV12 source range: first={} count={} (count must be positive and exclusive end must fit u64)",
                range.first_frame_index, range.frame_count
            ),
            Self::InvalidMemoryBudget { configured_bytes } => write!(
                formatter,
                "invalid direct YUV12 configured memory budget {configured_bytes} bytes"
            ),
            Self::MinimumMemoryBudget {
                configured_bytes,
                fixed_required_bytes,
            } => write!(
                formatter,
                "direct YUV12 configured memory budget {configured_bytes} bytes is below fixed pre-allocation requirement {fixed_required_bytes} bytes"
            ),
            Self::DecoderBackendNotPristine {
                slot_index,
                allocation,
            } => write!(
                formatter,
                "direct YUV12 scheduler requires a pristine decoder backend; retained allocation in {} is {allocation:?}",
                slot_index.map_or_else(
                    || "legacy decoder set".to_owned(),
                    |index| format!("decoder slot {index}")
                )
            ),
            Self::DecoderBufferLimit {
                buffer,
                limit_kind,
                required_bytes,
                limit_bytes,
            } => write!(
                formatter,
                "direct YUV12 decoder {buffer} requires {required_bytes} bytes, exceeding {limit_kind} limit {limit_bytes} bytes"
            ),
            Self::MemoryArithmeticOverflow => {
                write!(
                    formatter,
                    "direct YUV12 aggregate memory arithmetic overflow"
                )
            }
            Self::ConfiguredMemoryBudgetExceeded(details) => write!(
                formatter,
                "direct YUV12 aggregate memory requires {} bytes (visible-principal={} packed-capacity={} corrected-f32-capacity={} yuv-capacity={} {} readbacks of {} bytes decoder-extra={} compact-gain={} fixed-stage-aux={}), configured budget is {} bytes",
                details.required_bytes,
                details.visible_principal_bytes,
                details.packed_input_capacity_bytes,
                details.corrected_f32_capacity_bytes,
                details.yuv_output_capacity_bytes,
                details.readback_slot_count,
                details.composite_readback_capacity_bytes_per_slot,
                details.decoder_extra_bytes,
                details.compact_gain_capacity_bytes,
                details.fixed_stage_aux_bytes,
                details.configured_bytes
            ),
            Self::SourceFrameOrder { expected, actual } => write!(
                formatter,
                "direct YUV12 source-frame order mismatch: expected logical frame {expected}, got {actual}"
            ),
            Self::SourceFrameIndexOverflow { source_frame_index } => write!(
                formatter,
                "direct YUV12 source-frame index {source_frame_index} cannot be incremented"
            ),
            Self::SequenceOverflow => write!(formatter, "direct YUV12 sequence overflow"),
            Self::TooManyFrames {
                expected,
                attempted_sequence,
            } => write!(
                formatter,
                "direct YUV12 source range expects {expected} frames; attempted sequence {attempted_sequence}"
            ),
            Self::IncompleteSourceRange {
                first_frame_index,
                expected_frame_count,
                submitted_frames,
                next_expected_source_frame_index,
            } => write!(
                formatter,
                "direct YUV12 source range incomplete: first={first_frame_index} expected_count={expected_frame_count} submitted={submitted_frames} next_expected_source={next_expected_source_frame_index}"
            ),
            Self::Poisoned { first_error } => write!(
                formatter,
                "direct YUV12 scheduler is permanently stopped after: {first_error}"
            ),
            Self::AlreadyFinished => write!(formatter, "direct YUV12 scheduler already finished"),
            Self::InvalidDimensions(dimensions) => write!(
                formatter,
                "direct YUV12 scheduler dimensions {}x{} overflow",
                dimensions.width, dimensions.height
            ),
            Self::InvalidCpuFeederLength {
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "CPU-decoded packed-U16 feeder requires exactly {expected_bytes} bytes, got {actual_bytes}"
            ),
            Self::InvalidFrameContext {
                dimensions,
                correction_dimensions,
                source_frame_index,
                resolved_frame_index,
            } => write!(
                formatter,
                "direct YUV12 frame context mismatch: frame={}x{} correction={}x{} source_index={} resolved_index={} (positive even visible width required)",
                dimensions.width,
                dimensions.height,
                correction_dimensions.width,
                correction_dimensions.height,
                source_frame_index,
                resolved_frame_index
            ),
            Self::VerifiedColorContextMismatch { reason } => write!(
                formatter,
                "verified strict color context does not match scheduled frame {reason}"
            ),
            Self::BayerCorrection(detail) => {
                write!(formatter, "Bayer-correction stage failed: {detail}")
            }
            Self::DirectYuv12(detail) => write!(formatter, "direct YUV12 stage failed: {detail}"),
            Self::Decode(detail) => write!(formatter, "GPU feeder submission failed: {detail}"),
            Self::ReadbackTooLarge {
                required_bytes,
                max_buffer_size,
            } => write!(
                formatter,
                "direct YUV12 composite readback requires {required_bytes} bytes, adapter max buffer size is {max_buffer_size}"
            ),
            Self::EmptySlot { slot_index } => {
                write!(
                    formatter,
                    "direct YUV12 readback slot {slot_index} is empty"
                )
            }
            Self::MissingSequence { expected } => {
                write!(
                    formatter,
                    "direct YUV12 sequence {expected} is missing during drain"
                )
            }
            Self::PublicationOrder { expected, actual } => write!(
                formatter,
                "direct YUV12 publication order mismatch: expected sequence {expected}, got {actual}"
            ),
            Self::MapTimeout { sequence } => {
                write!(
                    formatter,
                    "timed out mapping direct YUV12 sequence {sequence}"
                )
            }
            Self::Map { sequence, detail } => {
                write!(
                    formatter,
                    "failed to map direct YUV12 sequence {sequence}: {detail}"
                )
            }
            Self::InvalidReadbackLayout => {
                write!(formatter, "invalid direct YUV12 readback layout")
            }
            Self::NonfiniteStatus {
                sequence,
                source_frame_index,
                categories,
                words,
            } => write!(
                formatter,
                "direct YUV12 sequence {sequence} source frame {source_frame_index} reported nonfinite categories {categories:?}, raw status {words:?}"
            ),
            Self::Writer {
                sequence,
                source_frame_index,
                detail,
            } => write!(
                formatter,
                "failed to publish direct YUV12 sequence {sequence} source frame {source_frame_index}: {detail}"
            ),
            Self::IncompleteDrain {
                submitted,
                published,
                pending,
            } => write!(
                formatter,
                "direct YUV12 drain incomplete: submitted={submitted} published={published} pending={pending}"
            ),
            Self::CompletionContextMismatch(details) => write!(
                formatter,
                "direct YUV12 completion context mismatch: expected={:?} actual={:?}",
                details.expected, details.actual
            ),
        }
    }
}

impl Error for DirectYuv12PipelineError {}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use mcraw4vulkan_gpu::{GpuBackendPreference, GpuDecodeConfig};

    fn test_backend() -> GpuDecodeBackend {
        GpuDecodeBackend::new_blocking(GpuDecodeConfig {
            backend_preference: GpuBackendPreference::VulkanOnly,
            ..GpuDecodeConfig::default()
        })
        .expect("scheduler test Vulkan backend")
    }

    fn test_source_sha() -> ClipSourceSha256 {
        ClipSourceSha256::read_once(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
            .expect("test source SHA")
    }

    #[test]
    fn accepts_clean_status_and_rejects_every_changed_word() {
        let clean = [0_u32, 0, u32::MAX, 0];
        let bytes: Vec<_> = clean.iter().flat_map(|word| word.to_le_bytes()).collect();
        assert_eq!(validate_nonfinite_status(&bytes, 0, 11), Ok(()));
        for index in 0..4 {
            let mut changed = clean;
            changed[index] ^= 1;
            let bytes: Vec<_> = changed.iter().flat_map(|word| word.to_le_bytes()).collect();
            assert!(matches!(
                validate_nonfinite_status(&bytes, 7, 19),
                Err(DirectYuv12PipelineError::NonfiniteStatus {
                    sequence: 7,
                    source_frame_index: 19,
                    ..
                })
            ));
        }
    }

    #[test]
    fn selected_scheduler_depth_is_two_with_one_decoder_slot() {
        assert_eq!(DIRECT_YUV12_READBACK_SLOTS, 2);
        assert_eq!(SHARED_DECODER_SLOT, 0);
    }

    #[test]
    fn production_constructor_freezes_half_scale_depth_and_timestamp_policy() {
        let scheduler = OneSharedComputeTwoReadbackDirectYuv12Scheduler::new(
            test_backend(),
            DirectYuv12SourceFrameRange {
                first_frame_index: 0,
                frame_count: 1,
            },
            8 * 1024 * 1024,
        )
        .expect("production scheduler");

        assert_eq!(Yuv444p12lePackPolicy::LINEAR_SIGNAL_SCALE_NUMERATOR, 1);
        assert_eq!(Yuv444p12lePackPolicy::LINEAR_SIGNAL_SCALE_DENOMINATOR, 2);
        assert_eq!(
            scheduler.linear_signal_scale_factor.to_bits(),
            Yuv444p12lePackPolicy::LINEAR_SIGNAL_SCALE_FACTOR_F32.to_bits()
        );
        assert_eq!(
            scheduler.effective_readback_slots,
            DIRECT_YUV12_READBACK_SLOTS
        );
        assert!(!scheduler.backend.gpu_timestamp_support().enabled);
    }

    #[test]
    fn decoder_allocation_preflight_checks_buffer_and_binding_limits() {
        let limits = wgpu::Limits {
            max_buffer_size: 1024,
            max_storage_buffer_binding_size: 768,
            max_uniform_buffer_binding_size: 256,
            ..wgpu::Limits::default()
        };
        let admitted = GpuNoReadbackSlotAllocation {
            raw_payload_bytes: 512,
            work_item_bytes: 256,
            packed_output_bytes: 768,
            params_bytes: 256,
            retained_readback_bytes: 1024,
            retained_mappable_output_bytes: 1024,
            timestamp_buffer_bytes: 16,
            ..GpuNoReadbackSlotAllocation::default()
        };
        assert_eq!(
            validate_decoder_allocation_limits(admitted, &limits),
            Ok(())
        );

        let mut too_large = admitted;
        too_large.retained_readback_bytes = 1025;
        assert!(matches!(
            validate_decoder_allocation_limits(too_large, &limits),
            Err(DirectYuv12PipelineError::DecoderBufferLimit {
                buffer: "retained readback",
                limit_kind: "max_buffer_size",
                ..
            })
        ));
        let mut storage = admitted;
        storage.raw_payload_bytes = 769;
        assert!(matches!(
            validate_decoder_allocation_limits(storage, &limits),
            Err(DirectYuv12PipelineError::DecoderBufferLimit {
                buffer: "raw payload",
                limit_kind: "max_storage_buffer_binding_size",
                ..
            })
        ));
        let mut uniform = admitted;
        uniform.params_bytes = 257;
        assert!(matches!(
            validate_decoder_allocation_limits(uniform, &limits),
            Err(DirectYuv12PipelineError::DecoderBufferLimit {
                buffer: "params",
                limit_kind: "max_uniform_buffer_binding_size",
                ..
            })
        ));
    }

    #[test]
    fn nonfinite_status_carries_source_and_categories() {
        let bytes: Vec<_> = [
            DIRECT_YUV12_NONFINITE_CAMERA | DIRECT_YUV12_NONFINITE_MAPPED,
            2,
            3,
            0,
        ]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
        assert!(matches!(
            validate_nonfinite_status(&bytes, 9, 42),
            Err(DirectYuv12PipelineError::NonfiniteStatus {
                sequence: 9,
                source_frame_index: 42,
                categories: DirectYuv12NonfiniteCategories {
                    camera: true,
                    ncl: false,
                    mapped: true,
                    unknown_bits: 0,
                },
                ..
            })
        ));
    }

    #[test]
    fn stale_or_future_map_callback_is_rejected_before_publication() {
        assert_eq!(validate_map_callback(7, 7, Ok(())), Ok(()));
        for received in [6, 8] {
            assert!(matches!(
                validate_map_callback(7, received, Ok(())),
                Err(DirectYuv12PipelineError::Map { sequence: 7, .. })
            ));
        }
    }

    #[test]
    fn verified_color_fact_check_rejects_each_single_field_mismatch() {
        let source = test_source_sha();
        let expected = ColorContextFingerprintFacts {
            numeric_domain: PipeF32BayerNumericDomain::RelativeLinearCorrectedCodeV1,
            dimensions: FrameDimensions {
                width: 4,
                height: 3,
            },
            bayer_pattern: BayerPattern::Rggb,
            correction_mode: PipeF32BayerCorrectionMode::IdentitySpatialGain,
            source_sha256: source,
            source_frame_index: 7,
        };
        assert_eq!(
            validate_verified_color_facts(
                expected,
                expected.dimensions,
                expected.bayer_pattern,
                expected.correction_mode,
                expected.source_sha256,
                expected.source_frame_index,
            ),
            Ok(())
        );
        let mismatches = [
            (
                ColorContextFingerprintFacts {
                    numeric_domain: PipeF32BayerNumericDomain::RelativeLinearCorrectedCodeV1,
                    dimensions: FrameDimensions {
                        width: 6,
                        height: 3,
                    },
                    ..expected
                },
                "dimensions",
            ),
            (
                ColorContextFingerprintFacts {
                    bayer_pattern: BayerPattern::Bggr,
                    ..expected
                },
                "CFA pattern",
            ),
            (
                ColorContextFingerprintFacts {
                    correction_mode: PipeF32BayerCorrectionMode::MotionCamSpatial,
                    ..expected
                },
                "correction mode",
            ),
            (
                ColorContextFingerprintFacts {
                    source_sha256: test_source_sha(),
                    ..expected
                },
                "source SHA-256",
            ),
            (
                ColorContextFingerprintFacts {
                    source_frame_index: 8,
                    ..expected
                },
                "source frame index",
            ),
        ];
        for (actual, reason) in mismatches {
            // The SHA case uses the same file above; replace it with a
            // different real digest without exposing a digest constructor.
            let actual = if reason == "source SHA-256" {
                ColorContextFingerprintFacts {
                    source_sha256: ClipSourceSha256::read_once(
                        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"),
                    )
                    .expect("second test source SHA"),
                    ..actual
                }
            } else {
                actual
            };
            assert_eq!(
                validate_verified_color_facts(
                    actual,
                    expected.dimensions,
                    expected.bayer_pattern,
                    expected.correction_mode,
                    expected.source_sha256,
                    expected.source_frame_index,
                ),
                Err(DirectYuv12PipelineError::VerifiedColorContextMismatch { reason })
            );
        }
    }
}
