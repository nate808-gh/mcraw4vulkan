use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::McrawVulkanBlockWorkItem;
use crate::MCRAW_DECODE_BLOCKS_PACKED_U16_WGSL;
use crate::MCRAW_DECODE_LEGACY_RAW16_PACKED_U16_WGSL;
use crate::MCRAW_DECODE_WORKGROUP_SIZE;
use crate::MCRAW_DESCRIPTORS_PER_MACROBLOCK;
use crate::{
    build_vulkan_work_plan, build_vulkan_work_plan_into, McrawVulkanWorkPlan,
    McrawVulkanWorkPlanRef, McrawVulkanWorkPlanScratch,
};
use anyhow::{anyhow, Context, Result};
use mcraw4vulkan_core::FrameDimensions;
use mcraw4vulkan_vignette::{
    FixedPointVignetteInputFacts, GpuUploadedFullResolutionGainMap, GpuVignetteCorrectionParams,
    GpuVignetteCorrectionStats, GpuVignetteCorrector, GpuVignetteGainMapUpload,
    GpuVignettePackedU16DispatchInput, PreparedFullResolutionFixedGainMap,
};

fn has_complete_macroblocks(work_item_count: usize) -> bool {
    let descriptors_per_macroblock = MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
    (work_item_count / descriptors_per_macroblock) * descriptors_per_macroblock == work_item_count
}

// Backend selection preference for the reusable GPU decode backend.
//
// Auto enumerates every enabled wgpu backend and ranks its adapters. VulkanOnly
// constrains both instance creation and adapter enumeration to Vulkan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendPreference {
    Auto,
    VulkanOnly,
}

// Configuration for creating a reusable GPU decode backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuDecodeConfig {
    pub backend_preference: GpuBackendPreference,
    pub enable_gpu_timestamps: bool,
}

impl Default for GpuDecodeConfig {
    fn default() -> Self {
        Self {
            backend_preference: GpuBackendPreference::Auto,
            enable_gpu_timestamps: false,
        }
    }
}

/// Exact buffer limits required by one known decode/render job.
///
/// The ordinary backend constructor retains the established downlevel limits.
/// Callers that know their complete GPU job may use the requirements-aware
/// constructor so the logical device requests only the additional buffer
/// capacity that job needs and the selected adapter actually supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuDecodeRequiredLimits {
    dimensions: FrameDimensions,
    storage_buffer_resource: &'static str,
    storage_buffer_binding_bytes: u64,
    buffer_resource: &'static str,
    buffer_bytes: u64,
}

impl GpuDecodeRequiredLimits {
    pub const fn new(
        dimensions: FrameDimensions,
        storage_buffer_resource: &'static str,
        storage_buffer_binding_bytes: u64,
        buffer_resource: &'static str,
        buffer_bytes: u64,
    ) -> Self {
        Self {
            dimensions,
            storage_buffer_resource,
            storage_buffer_binding_bytes,
            buffer_resource,
            buffer_bytes,
        }
    }

    pub const fn dimensions(self) -> FrameDimensions {
        self.dimensions
    }

    pub const fn storage_buffer_resource(self) -> &'static str {
        self.storage_buffer_resource
    }

    pub const fn storage_buffer_binding_bytes(self) -> u64 {
        self.storage_buffer_binding_bytes
    }

    pub const fn buffer_resource(self) -> &'static str {
        self.buffer_resource
    }

    pub const fn buffer_bytes(self) -> u64 {
        self.buffer_bytes
    }
}

pub struct GpuDecodeBackend {
    adapter_info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    timestamp_support: GpuTimestampSupport,
    // GPU child resources precede queue/device because Rust drops fields in
    // declaration order. Buffers and pipelines therefore release before their
    // device when the backend is torn down.
    buffers: ReusableGpuBuffers,
    pipeline_slots: Vec<ReusableGpuBuffers>,
    packed_u16_pipeline: wgpu::ComputePipeline,
    legacy_raw16_packed_u16_pipeline: wgpu::ComputePipeline,
    queue: wgpu::Queue,
    device: wgpu::Device,
}

/// Capacity-aware explicit allocation plan for one no-readback decoder slot.
///
/// The packed output is reported separately because the direct-YUV scheduler's
/// principal formula already counts it. Other GPU buffers are decoder extras.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuNoReadbackSlotAllocation {
    pub raw_payload_bytes: u64,
    pub work_item_bytes: u64,
    pub packed_output_bytes: u64,
    pub params_bytes: u64,
    pub retained_readback_bytes: u64,
    pub retained_mappable_output_bytes: u64,
    pub timestamp_buffer_bytes: u64,
    pub host_raw_upload_scratch_capacity: u64,
    pub host_work_item_scratch_capacity: u64,
}

impl GpuNoReadbackSlotAllocation {
    pub fn decoder_extra_gpu_bytes(self) -> Result<u64> {
        self.raw_payload_bytes
            .checked_add(self.work_item_bytes)
            .and_then(|value| value.checked_add(self.params_bytes))
            .and_then(|value| value.checked_add(self.retained_readback_bytes))
            .and_then(|value| value.checked_add(self.retained_mappable_output_bytes))
            .and_then(|value| value.checked_add(self.timestamp_buffer_bytes))
            .context("decoder extra GPU allocation byte count overflow")
    }

    pub fn total_explicit_gpu_bytes(self) -> Result<u64> {
        self.decoder_extra_gpu_bytes()?
            .checked_add(self.packed_output_bytes)
            .context("decoder total explicit GPU allocation byte count overflow")
    }
}

// Output from mapped packed-u16 readback callbacks.
//
// The GPU backend owns the mapped staging buffer and unmaps it before returning.
// The caller-provided closure consumes the mapped little-endian pixel bytes while
// they are valid and returns whatever output it built, such as final DNG bytes.
#[derive(Debug, Clone)]
pub struct GpuMappedPackedU16Output<T> {
    pub value: T,
    pub dimensions: FrameDimensions,
    pub work_item_count: usize,
    pub macroblock_count: usize,
    pub expected_invocations: usize,
    pub timings: GpuDecodeTimings,
}

// Optional vignette postprocess request for the mapped packed-u16 decode path.
//
// The request carries only already-prepared GPU vignette resources. Metadata
// parsing, lens-map preparation, and gain-map upload stay outside the per-frame
// decode hot path.
pub struct OptionalGpuVignetteCorrection<'a> {
    pub corrector: &'a mut GpuVignetteCorrector,
    pub uploaded_gain_map: &'a GpuUploadedFullResolutionGainMap,
    pub params: GpuVignetteCorrectionParams,
}

// Optional vignette correction for one ring-mapped packed-u16 decode frame.
//
// The ring backend owns per-slot mutable output resources. This input carries
// only immutable uploaded gain-map resources plus per-frame correction params.
pub struct GpuMappedRingVignetteCorrection {
    pub uploaded_gain_map: GpuUploadedFullResolutionGainMap,
    pub params: GpuVignetteCorrectionParams,
}

// Output from mapped packed-u16 decode with optional vignette correction. Its
// callback has the same map-consume-unmap lifetime as the uncorrected path.
#[derive(Debug, Clone)]
pub struct GpuMappedPackedU16WithVignetteOutput<T> {
    pub value: T,
    pub dimensions: FrameDimensions,
    pub work_item_count: usize,
    pub macroblock_count: usize,
    pub expected_invocations: usize,
    pub timings: GpuDecodeTimings,
    pub vignette_stats: Option<GpuVignetteCorrectionStats>,
}

// Neutral view of a decoded/corrected packed-u16 Bayer GPU buffer.
//
// Downstream composition layers can encode additional GPU work against this
// buffer without making the raw decode crate depend on render, preview, pipe,
// DNG, or sink policy.
#[derive(Debug, Clone)]
pub struct GpuDecodedPackedU16BufferView<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub byte_len: u64,
    pub dimensions: FrameDimensions,
    pub pixel_count: usize,
}

// Downstream GPU work may ask the decode backend to map any readback buffer it
// encoded into the same command stream. The buffer is owned as a neutral wgpu
// handle, so this type does not mention render or pipe formats.
#[derive(Debug, Clone)]
pub struct GpuExternalReadbackBuffer {
    pub buffer: wgpu::Buffer,
    pub mapped_byte_len: u64,
    pub valid_byte_len: u64,
}

// Output from the opt-in neutral GPU-stage decode path. The stage payload is
// caller-defined opaque data returned by the downstream encoder.
#[derive(Debug, Clone)]
pub struct GpuMappedGpuStageOutput<T, S> {
    pub value: T,
    pub stage: S,
    pub dimensions: FrameDimensions,
    pub work_item_count: usize,
    pub macroblock_count: usize,
    pub expected_invocations: usize,
    pub timings: GpuDecodeTimings,
    pub vignette_stats: Option<GpuVignetteCorrectionStats>,
}

// Output from the neutral GPU-stage decode path that does not require any
// downstream readback buffer. This is the preview/display composition hook:
// decode, optional vignette, and caller-owned GPU work share one command stream,
// then the backend waits for completion without mapping full-frame bytes.
#[derive(Debug, Clone)]
pub struct GpuNoReadbackGpuStageOutput<S> {
    pub stage: S,
    pub dimensions: FrameDimensions,
    pub work_item_count: usize,
    pub macroblock_count: usize,
    pub expected_invocations: usize,
    pub timings: GpuDecodeTimings,
    pub vignette_stats: Option<GpuVignetteCorrectionStats>,
}

// Output from a no-readback GPU stage that remains submitted on return. The
// submission index is its completion token; the selected backend slot must not
// be reused until that submission completes.
#[derive(Debug, Clone)]
pub struct GpuSubmittedNoReadbackGpuStageOutput<S> {
    pub stage: S,
    pub submission_index: wgpu::SubmissionIndex,
    pub dimensions: FrameDimensions,
    pub work_item_count: usize,
    pub macroblock_count: usize,
    pub expected_invocations: usize,
    pub timings: GpuDecodeTimings,
    pub vignette_stats: Option<GpuVignetteCorrectionStats>,
    pub gpu_timestamps: GpuSubmittedTimestampProfile,
}

// Lightweight descriptor for timestamp data resolved as part of a submitted
// no-readback GPU-stage frame. The query resources themselves remain owned by
// the backend slot so callers can retire frames without taking GPU resources out
// of the backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuSubmittedTimestampProfile {
    pub support: GpuTimestampSupport,
    pub output_clear_stage: bool,
    pub vignette_stage: bool,
    pub preview_render_stage: bool,
}

impl GpuSubmittedTimestampProfile {
    fn unavailable(support: GpuTimestampSupport) -> Self {
        Self {
            support,
            output_clear_stage: false,
            vignette_stage: false,
            preview_render_stage: false,
        }
    }

    fn from_stage_mask(support: GpuTimestampSupport, stages: GpuTimestampStageMask) -> Self {
        Self {
            support,
            output_clear_stage: stages.output_clear,
            vignette_stage: stages.vignette,
            preview_render_stage: true,
        }
    }

    fn stage_mask(self) -> GpuTimestampStageMask {
        GpuTimestampStageMask {
            output_clear: self.output_clear_stage,
            vignette: self.vignette_stage,
        }
    }
}

// Input for an in-flight mapped packed-u16 batch decode.
//
// Each item borrows one raw MCRAW frame payload that must remain valid until
// the batch method returns. The output order matches the input order.
#[derive(Debug, Clone, Copy)]
pub struct GpuMappedBatchInput<'a> {
    pub frame_index: usize,
    pub raw_payload: &'a [u8],
    pub visible_dimensions: FrameDimensions,
}

// The ring owns each raw payload until it has been copied into slot-owned GPU
// storage, so later submissions never retain a borrow of an earlier frame.
#[derive(Debug, Clone)]
pub struct GpuMappedRingFrame {
    pub frame_index: usize,
    pub raw_payload: Vec<u8>,
    pub visible_dimensions: FrameDimensions,
}

// Summary counters for the profiling-only streaming mapped readback ring.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuMappedRingStats {
    pub submitted_frames: usize,
    pub completed_frames: usize,
    pub max_observed_depth: usize,
    pub slot_count: usize,
}

// Output from the profiling-only streaming mapped readback ring.
//
// Values are returned in input order. The backend maps each readback buffer,
// lets the caller consume the mapped bytes, unmaps the slot, and then reuses the
// slot for later submissions.
#[derive(Debug, Clone)]
pub struct GpuMappedRingOutput<T> {
    pub values: Vec<T>,
    pub timings: Vec<GpuDecodeTimings>,
    pub vignette_stats: Vec<Option<GpuVignetteCorrectionStats>>,
    pub stats: GpuMappedRingStats,
}
#[derive(Debug, Default)]
pub struct GpuDecodeScratch {
    work_plan: McrawVulkanWorkPlanScratch,
    legacy_raw16_work_plan: LegacyRaw16GpuWorkPlan,
}

impl GpuDecodeScratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build one checked type-7 work plan into caller-owned reusable storage.
    pub fn prepare_type7<'a>(
        &'a mut self,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
    ) -> Result<GpuPreparedType7WorkPlan<'a>> {
        let started = Instant::now();
        let work_plan =
            build_vulkan_work_plan_into(raw_payload, visible_dimensions, &mut self.work_plan)
                .context("failed to build reusable submitted type-7 GPU work plan")?;
        Ok(GpuPreparedType7WorkPlan {
            work_plan,
            build_elapsed: started.elapsed(),
        })
    }

    /// Build one checked legacy/type-6 work plan into caller-owned storage.
    pub fn prepare_type6<'a>(
        &'a mut self,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        row_stride: u32,
    ) -> Result<GpuPreparedType6WorkPlan<'a>> {
        let started = Instant::now();
        let build_stats = build_legacy_raw16_gpu_work_plan_into(
            raw_payload,
            visible_dimensions,
            row_stride,
            &mut self.legacy_raw16_work_plan,
        )
        .context("failed to build reusable submitted type-6 GPU work plan")?;
        Ok(GpuPreparedType6WorkPlan {
            blocks: &self.legacy_raw16_work_plan.blocks,
            chunk_count: self.legacy_raw16_work_plan.chunk_count,
            blocks_capacity: self.legacy_raw16_work_plan.blocks.capacity(),
            reused_scratch: build_stats.reused_scratch,
            scratch_grow_count: build_stats.scratch_grow_count,
            build_elapsed: started.elapsed(),
            visible_dimensions,
            row_stride,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct LegacyRaw16WorkPlanBuildStats {
    reused_scratch: bool,
    scratch_grow_count: u64,
}

/// Borrowed checked type-6 work plan retained in [`GpuDecodeScratch`].
///
/// Allocation preflight and submitted decode consume this same descriptor
/// slice, so the scheduling path performs one plan build per source frame.
#[derive(Debug, Clone, Copy)]
pub struct GpuPreparedType6WorkPlan<'a> {
    blocks: &'a [McrawVulkanBlockWorkItem],
    chunk_count: usize,
    blocks_capacity: usize,
    reused_scratch: bool,
    scratch_grow_count: u64,
    build_elapsed: Duration,
    visible_dimensions: FrameDimensions,
    row_stride: u32,
}

impl GpuPreparedType6WorkPlan<'_> {
    pub fn block_count(self) -> usize {
        self.blocks.len()
    }

    pub fn block_capacity(self) -> usize {
        self.blocks_capacity
    }

    pub fn reused_scratch(self) -> bool {
        self.reused_scratch
    }

    pub fn scratch_grow_count(self) -> u64 {
        self.scratch_grow_count
    }

    pub fn build_elapsed(self) -> Duration {
        self.build_elapsed
    }

    fn allocation_stats(self) -> GpuWorkPlanAllocationStats {
        let blocks_len = usize_to_u64_saturating(self.blocks.len());
        let blocks_capacity = usize_to_u64_saturating(self.blocks_capacity);
        GpuWorkPlanAllocationStats {
            blocks_len,
            blocks_capacity,
            work_items_len: blocks_len,
            work_items_capacity: blocks_capacity,
            work_items_bytes: blocks_len.saturating_mul(PACKED_WORK_ITEM_BYTES),
            reused_scratch: self.reused_scratch,
            scratch_grow_count: self.scratch_grow_count,
        }
    }
}

/// Borrowed checked type-7 work plan retained in [`GpuDecodeScratch`].
///
/// Allocation preflight and submitted decode consume the same descriptor
/// slice, preventing a second per-frame plan build on the scheduling path.
#[derive(Debug, Clone, Copy)]
pub struct GpuPreparedType7WorkPlan<'a> {
    work_plan: McrawVulkanWorkPlanRef<'a>,
    build_elapsed: Duration,
}

impl GpuPreparedType7WorkPlan<'_> {
    pub fn block_count(self) -> usize {
        self.work_plan.block_count()
    }

    pub fn block_capacity(self) -> usize {
        self.work_plan.blocks_capacity
    }

    pub fn reused_scratch(self) -> bool {
        self.work_plan.build_stats.reused_scratch
    }

    pub fn scratch_grow_count(self) -> u64 {
        self.work_plan.build_stats.grow_count
    }

    pub fn build_elapsed(self) -> Duration {
        self.build_elapsed
    }
}

// CPU-side timing and byte-count diagnostics for one GPU decode call.
//
// Durations are wall-clock spans around wgpu preparation, submission, mapping,
// and callback work, not GPU timestamp queries. Some spans enclose finer
// buckets, so the duration fields are not all additive.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuDecodeTimings {
    pub raw_payload_bytes: u64,
    pub work_item_count: u64,
    pub work_item_bytes: u64,
    pub params_bytes: u64,
    pub output_bytes: u64,
    pub readback_bytes: u64,
    pub work_plan_reused_scratch: bool,
    pub work_plan_scratch_grow_count: u64,
    pub work_plan: Duration,
    pub cpu_prepare: Duration,
    pub buffer_ensure: Duration,
    pub cpu_staging_serialize: Duration,
    pub upload: Duration,
    pub raw_payload_upload: Duration,
    pub work_items_upload: Duration,
    pub params_upload: Duration,
    pub encode_submit: Duration,
    pub bind_group: Duration,
    pub command_encoder_create: Duration,
    pub output_clear_encode: Duration,
    pub decode_pass_encode: Duration,
    pub vignette_encode: Duration,
    pub copy_to_readback_encode: Duration,
    pub command_finish_submit: Duration,
    pub wait_map: Duration,
    pub mapped_consumer: Duration,
    pub readback_convert: Duration,
    pub dispatch_readback: Duration,
    pub total: Duration,
    pub gpu_timestamps: GpuTimestampProfile,
}

impl GpuDecodeTimings {
    fn from_dispatch(work_plan: Duration, dispatch: GpuDispatchTimings, total: Duration) -> Self {
        Self {
            raw_payload_bytes: dispatch.raw_payload_bytes,
            work_item_count: dispatch.work_item_count,
            work_item_bytes: dispatch.work_item_bytes,
            params_bytes: dispatch.params_bytes,
            output_bytes: dispatch.output_bytes,
            readback_bytes: dispatch.readback_bytes,
            work_plan_reused_scratch: false,
            work_plan_scratch_grow_count: 0,
            work_plan,
            cpu_prepare: dispatch.cpu_prepare,
            buffer_ensure: dispatch.buffer_ensure,
            cpu_staging_serialize: dispatch.cpu_staging_serialize,
            upload: dispatch.upload,
            raw_payload_upload: dispatch.raw_payload_upload,
            work_items_upload: dispatch.work_items_upload,
            params_upload: dispatch.params_upload,
            encode_submit: dispatch.encode_submit,
            bind_group: dispatch.bind_group,
            command_encoder_create: dispatch.command_encoder_create,
            output_clear_encode: dispatch.output_clear_encode,
            decode_pass_encode: dispatch.decode_pass_encode,
            vignette_encode: dispatch.vignette_encode,
            copy_to_readback_encode: dispatch.copy_to_readback_encode,
            command_finish_submit: dispatch.command_finish_submit,
            wait_map: dispatch.wait_map,
            mapped_consumer: dispatch.mapped_consumer,
            readback_convert: dispatch.readback_convert,
            dispatch_readback: dispatch.total,
            total,
            gpu_timestamps: dispatch.gpu_timestamps,
        }
    }

    fn from_work_plan_dispatch(
        work_plan: &McrawVulkanWorkPlan,
        work_plan_elapsed: Duration,
        dispatch: GpuDispatchTimings,
        total: Duration,
    ) -> Self {
        let mut timings = Self::from_dispatch(work_plan_elapsed, dispatch, total);
        timings.record_work_plan_allocation(GpuWorkPlanAllocationStats::from_work_plan(work_plan));
        timings
    }

    fn from_work_plan_ref_dispatch(
        work_plan: McrawVulkanWorkPlanRef<'_>,
        work_plan_elapsed: Duration,
        dispatch: GpuDispatchTimings,
        total: Duration,
    ) -> Self {
        let mut timings = Self::from_dispatch(work_plan_elapsed, dispatch, total);
        timings
            .record_work_plan_allocation(GpuWorkPlanAllocationStats::from_work_plan_ref(work_plan));
        timings
    }

    fn from_work_plan_stats_dispatch(
        work_plan_stats: GpuWorkPlanAllocationStats,
        work_plan: Duration,
        dispatch: GpuDispatchTimings,
        total: Duration,
    ) -> Self {
        let mut timings = Self::from_dispatch(work_plan, dispatch, total);
        timings.record_work_plan_allocation(work_plan_stats);
        timings
    }

    fn record_work_plan_allocation(&mut self, stats: GpuWorkPlanAllocationStats) {
        let _ = stats;
        self.work_plan_reused_scratch = stats.reused_scratch;
        self.work_plan_scratch_grow_count = stats.scratch_grow_count;
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct GpuWorkPlanAllocationStats {
    blocks_len: u64,
    blocks_capacity: u64,
    work_items_len: u64,
    work_items_capacity: u64,
    work_items_bytes: u64,
    reused_scratch: bool,
    scratch_grow_count: u64,
}

impl GpuWorkPlanAllocationStats {
    fn from_work_plan(work_plan: &McrawVulkanWorkPlan) -> Self {
        Self::from_work_plan_ref(work_plan.as_ref())
    }

    fn from_work_plan_ref(work_plan: McrawVulkanWorkPlanRef<'_>) -> Self {
        let blocks_len = usize_to_u64_saturating(work_plan.blocks.len());
        let blocks_capacity = usize_to_u64_saturating(work_plan.blocks_capacity);
        Self {
            blocks_len,
            blocks_capacity,
            work_items_len: blocks_len,
            work_items_capacity: blocks_capacity,
            work_items_bytes: blocks_len.saturating_mul(PACKED_WORK_ITEM_BYTES),
            reused_scratch: work_plan.build_stats.reused_scratch,
            scratch_grow_count: work_plan.build_stats.grow_count,
        }
    }
}

fn usize_to_u64_saturating(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

// Adapter/device timestamp-query capability for the reusable decode backend.
//
// Timestamp profiling is opt-in. When requested but unsupported, the backend is
// still created without timestamp features and decode output behavior is
// unchanged.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuTimestampSupport {
    pub requested: bool,
    pub supported: bool,
    pub enabled: bool,
    pub period_ps: u64,
}

impl GpuTimestampSupport {
    pub fn period_ns(self) -> Option<f64> {
        (self.period_ps != 0).then(|| self.period_ps as f64 / 1_000.0)
    }
}

// GPU-side timestamp-query profile for one packed-u16 decode submission.
//
// These durations are derived from GPU timestamp query deltas. They are separate
// from CPU-side encode/submit/map timings and are only populated when timestamp
// profiling was requested, supported, and enabled for the device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuTimestampProfile {
    pub requested: bool,
    pub supported: bool,
    pub enabled: bool,
    pub period_ps: u64,
    pub samples_valid: bool,
    pub output_clear: Option<Duration>,
    pub decode_dispatch: Option<Duration>,
    pub vignette_dispatch: Option<Duration>,
    pub copy_to_readback: Option<Duration>,
    pub total_command: Option<Duration>,
    pub query_resolve_readback: Duration,
}

impl GpuTimestampProfile {
    fn unavailable(support: GpuTimestampSupport) -> Self {
        Self {
            requested: support.requested,
            supported: support.supported,
            enabled: support.enabled,
            period_ps: support.period_ps,
            samples_valid: false,
            output_clear: None,
            decode_dispatch: None,
            vignette_dispatch: None,
            copy_to_readback: None,
            total_command: None,
            query_resolve_readback: Duration::ZERO,
        }
    }

    pub fn period_ns(self) -> Option<f64> {
        (self.period_ps != 0).then(|| self.period_ps as f64 / 1_000.0)
    }
}

// Reusable GPU buffers owned by one GpuDecodeBackend.
//
// These buffers grow when needed and are then reused across later frames. This is
// intentionally private to the GPU crate so callers cannot observe or depend on
// backend implementation details.
#[derive(Default)]
struct ReusableGpuBuffers {
    raw_payload: Option<SizedGpuBuffer>,
    work_items: Option<SizedGpuBuffer>,
    output: Option<SizedGpuBuffer>,
    readback: Option<SizedGpuBuffer>,
    params: Option<SizedGpuBuffer>,
    bind_group: Option<wgpu::BindGroup>,
    vignette_corrector: Option<GpuVignetteCorrector>,
    timestamp_recorder: Option<GpuTimestampRecorder>,

    // CPU-side upload staging reused across frames.
    //
    // GPU buffers above are already reusable. These Vecs avoid rebuilding new
    // temporary upload byte buffers for each frame when raw payload padding or
    // work-item serialization is needed.
    raw_payload_upload_scratch: Vec<u8>,
    work_item_upload_scratch: Vec<u8>,
}

// A wgpu buffer plus the allocated byte capacity we track ourselves.
//
// wgpu::Buffer does not expose a simple public capacity API, so the backend
// records the size used at creation time.
struct SizedGpuBuffer {
    buffer: wgpu::Buffer,
    size: u64,
}

impl ReusableGpuBuffers {
    fn allocation_snapshot(&self) -> GpuNoReadbackSlotAllocation {
        #[allow(unused_variables)]
        let retained_mappable_output_bytes = 0;
        GpuNoReadbackSlotAllocation {
            raw_payload_bytes: self.raw_payload.as_ref().map_or(0, |buffer| buffer.size),
            work_item_bytes: self.work_items.as_ref().map_or(0, |buffer| buffer.size),
            packed_output_bytes: self.output.as_ref().map_or(0, |buffer| buffer.size),
            params_bytes: self.params.as_ref().map_or(0, |buffer| buffer.size),
            retained_readback_bytes: self.readback.as_ref().map_or(0, |buffer| buffer.size),
            retained_mappable_output_bytes,
            timestamp_buffer_bytes: if self.timestamp_recorder.is_some() {
                GPU_TIMESTAMP_READBACK_BYTES.saturating_mul(2)
            } else {
                0
            },
            host_raw_upload_scratch_capacity: u64::try_from(
                self.raw_payload_upload_scratch.capacity(),
            )
            .unwrap_or(u64::MAX),
            host_work_item_scratch_capacity: u64::try_from(
                self.work_item_upload_scratch.capacity(),
            )
            .unwrap_or(u64::MAX),
        }
    }

    fn prospective_allocation(
        &self,
        raw_payload_bytes: u64,
        work_item_bytes: u64,
        packed_output_bytes: u64,
        params_bytes: u64,
    ) -> GpuNoReadbackSlotAllocation {
        let current = self.allocation_snapshot();
        GpuNoReadbackSlotAllocation {
            raw_payload_bytes: current.raw_payload_bytes.max(raw_payload_bytes),
            work_item_bytes: current.work_item_bytes.max(work_item_bytes),
            packed_output_bytes: current.packed_output_bytes.max(packed_output_bytes),
            params_bytes: current.params_bytes.max(params_bytes),
            ..current
        }
    }

    // Ensure all per-frame GPU buffers are large enough for the current decode.
    //
    // Returns true if any bind-group-visible buffer was replaced. The readback
    // buffer is not part of the bind group, so growing it does not by itself
    // require bind group recreation.
    fn ensure_for_decode(
        &mut self,
        device: &wgpu::Device,
        raw_payload_size: u64,
        work_items_size: u64,
        output_size: u64,
        params_size: u64,
    ) -> bool {
        let mut bind_group_buffers_changed = false;

        bind_group_buffers_changed |= ensure_buffer_slot(
            device,
            &mut self.raw_payload,
            "MCRAW reusable raw payload storage buffer",
            raw_payload_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        bind_group_buffers_changed |= ensure_buffer_slot(
            device,
            &mut self.work_items,
            "MCRAW reusable work item storage buffer",
            work_items_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        bind_group_buffers_changed |= ensure_buffer_slot(
            device,
            &mut self.output,
            "MCRAW reusable GPU output pixel storage buffer",
            output_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        ensure_buffer_slot(
            device,
            &mut self.readback,
            "MCRAW reusable GPU output readback buffer",
            output_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        bind_group_buffers_changed |= ensure_buffer_slot(
            device,
            &mut self.params,
            "MCRAW reusable params uniform buffer",
            params_size,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        if bind_group_buffers_changed {
            self.bind_group = None;
        }

        bind_group_buffers_changed
    }

    fn ensure_for_cpu_decoded_upload(&mut self, device: &wgpu::Device, output_size: u64) -> bool {
        let output_changed = ensure_buffer_slot(
            device,
            &mut self.output,
            "MCRAW reusable CPU-decoded Bayer U16 GPU-stage buffer",
            output_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        if output_changed {
            self.bind_group = None;
        }

        output_changed
    }
    // Ensure only the bind-group-visible decode buffers are large enough.
    //
    // The production GPU-stage path intentionally keeps decoded Bayer data on
    // the GPU and therefore does not allocate the normal MAP_READ staging
    // buffer.
    fn ensure_for_decode_without_readback(
        &mut self,
        device: &wgpu::Device,
        raw_payload_size: u64,
        work_items_size: u64,
        output_size: u64,
        params_size: u64,
    ) -> bool {
        let mut bind_group_buffers_changed = false;

        bind_group_buffers_changed |= ensure_buffer_slot(
            device,
            &mut self.raw_payload,
            "MCRAW reusable raw payload storage buffer",
            raw_payload_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        bind_group_buffers_changed |= ensure_buffer_slot(
            device,
            &mut self.work_items,
            "MCRAW reusable work item storage buffer",
            work_items_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        bind_group_buffers_changed |= ensure_buffer_slot(
            device,
            &mut self.output,
            "MCRAW reusable GPU output pixel storage buffer",
            output_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        bind_group_buffers_changed |= ensure_buffer_slot(
            device,
            &mut self.params,
            "MCRAW reusable params uniform buffer",
            params_size,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        if bind_group_buffers_changed {
            self.bind_group = None;
        }

        bind_group_buffers_changed
    }
}

// Fine-grained timing for one GPU dispatch/readback operation.
//
// This is intentionally internal. The public GpuDecodeTimings type exposes the
// same timing buckets after they are combined with work-plan and total timings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct GpuDispatchTimings {
    raw_payload_bytes: u64,
    work_item_count: u64,
    work_item_bytes: u64,
    params_bytes: u64,
    output_bytes: u64,
    readback_bytes: u64,
    work_item_buffer_reused: bool,
    work_item_buffer_capacity: u64,
    work_item_buffer_reallocs: u64,
    write_buffer_with_work_items: bool,
    write_buffer_with_params: bool,
    direct_staging_fallback: bool,
    queue_write_buffer_calls: u64,
    queue_write_buffer_with_calls: u64,
    cpu_prepare: Duration,
    buffer_ensure: Duration,
    cpu_staging_serialize: Duration,
    upload: Duration,
    raw_payload_upload: Duration,
    work_items_upload: Duration,
    params_upload: Duration,
    encode_submit: Duration,
    bind_group: Duration,
    command_encoder_create: Duration,
    output_clear_encode: Duration,
    decode_pass_encode: Duration,
    vignette_encode: Duration,
    copy_to_readback_encode: Duration,
    command_finish_submit: Duration,
    wait_map: Duration,
    mapped_consumer: Duration,
    readback_convert: Duration,
    total: Duration,
    gpu_timestamps: GpuTimestampProfile,
}

// Result from one GPU dispatch/readback operation that lets the caller consume
// the mapped readback bytes before the staging buffer is unmapped.
struct GpuMappedDispatchResult<T> {
    value: T,
    timings: GpuDispatchTimings,
}

// Result from the opt-in mapped dispatch path that may insert vignette
// correction before readback.
struct GpuMappedDispatchWithVignetteResult<T> {
    value: T,
    timings: GpuDispatchTimings,
    vignette_stats: Option<GpuVignetteCorrectionStats>,
}

struct GpuMappedGpuStageDispatchWithVignetteResult<T, S> {
    value: T,
    stage: S,
    timings: GpuDispatchTimings,
    vignette_stats: Option<GpuVignetteCorrectionStats>,
}

struct GpuNoReadbackGpuStageDispatchWithVignetteResult<S> {
    stage: S,
    timings: GpuDispatchTimings,
    vignette_stats: Option<GpuVignetteCorrectionStats>,
}

struct PendingNoReadbackGpuStageDispatchWithVignetteResult<S> {
    stage: S,
    submission_index: wgpu::SubmissionIndex,
    dispatch_start: Instant,
    gpu_timestamps: GpuSubmittedTimestampProfile,
    timings: GpuDispatchTimings,
    vignette_stats: Option<GpuVignetteCorrectionStats>,
}

const GPU_TIMESTAMP_QUERY_COUNT: u32 = 10;
const GPU_TIMESTAMP_QUERY_COUNT_USIZE: usize = GPU_TIMESTAMP_QUERY_COUNT as usize;
const GPU_TIMESTAMP_READBACK_BYTES: u64 =
    GPU_TIMESTAMP_QUERY_COUNT as u64 * wgpu::QUERY_SIZE as u64;

#[derive(Clone, Copy)]
#[repr(u32)]
enum GpuTimestampQuery {
    TotalStart = 0,
    OutputClearStart = 1,
    OutputClearEnd = 2,
    DecodeDispatchStart = 3,
    DecodeDispatchEnd = 4,
    VignetteDispatchStart = 5,
    VignetteDispatchEnd = 6,
    CopyToReadbackStart = 7,
    CopyToReadbackEnd = 8,
    TotalEnd = 9,
}

#[derive(Clone, Copy)]
struct GpuTimestampStageMask {
    output_clear: bool,
    vignette: bool,
}

struct GpuTimestampRecorder {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    support: GpuTimestampSupport,
}

impl GpuTimestampRecorder {
    fn new(
        device: &wgpu::Device,
        support: GpuTimestampSupport,
        label: &'static str,
    ) -> Option<Self> {
        if !support.enabled {
            return None;
        }

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some(label),
            ty: wgpu::QueryType::Timestamp,
            count: GPU_TIMESTAMP_QUERY_COUNT,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MCRAW GPU timestamp resolve buffer"),
            size: GPU_TIMESTAMP_READBACK_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MCRAW GPU timestamp readback buffer"),
            size: GPU_TIMESTAMP_READBACK_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Some(Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            support,
        })
    }

    fn write(&self, encoder: &mut wgpu::CommandEncoder, query: GpuTimestampQuery) {
        encoder.write_timestamp(&self.query_set, query as u32);
    }

    fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.query_set,
            0..GPU_TIMESTAMP_QUERY_COUNT,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            GPU_TIMESTAMP_READBACK_BYTES,
        );
    }

    fn read_profile(
        &self,
        device: &wgpu::Device,
        submission_index: wgpu::SubmissionIndex,
        stages: GpuTimestampStageMask,
    ) -> Result<GpuTimestampProfile> {
        let query_readback_start = Instant::now();
        let timestamp_slice = self.readback_buffer.slice(0..GPU_TIMESTAMP_READBACK_BYTES);

        let (sender, receiver) = mpsc::channel();
        timestamp_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        device.poll(wgpu::Maintain::wait_for(submission_index.clone()));

        let map_result = receiver
            .recv_timeout(Duration::from_secs(30))
            .context("timed out waiting for MCRAW GPU timestamp mapping callback")?;

        map_result.map_err(|error| anyhow!("failed to map MCRAW timestamp buffer: {error:?}"))?;

        let mut samples = [0u64; GPU_TIMESTAMP_QUERY_COUNT_USIZE];
        {
            let mapped = timestamp_slice.get_mapped_range();
            if mapped.len() != GPU_TIMESTAMP_READBACK_BYTES as usize {
                anyhow::bail!(
                    "GPU timestamp readback byte length mismatch: expected={} actual={}",
                    GPU_TIMESTAMP_READBACK_BYTES,
                    mapped.len()
                );
            }
            for (index, chunk) in mapped.chunks_exact(8).enumerate() {
                samples[index] = u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
            }
        }

        self.readback_buffer.unmap();

        let mut profile = gpu_timestamp_profile_from_samples(self.support, &samples, stages);
        profile.query_resolve_readback = query_readback_start.elapsed();
        Ok(profile)
    }
}

fn ensure_timestamp_recorder<'a>(
    device: &wgpu::Device,
    recorder: &'a mut Option<GpuTimestampRecorder>,
    support: GpuTimestampSupport,
    label: &'static str,
) -> Option<&'a GpuTimestampRecorder> {
    if !support.enabled {
        return None;
    }
    if recorder.is_none() {
        *recorder = GpuTimestampRecorder::new(device, support, label);
    }
    recorder.as_ref()
}

fn read_submitted_timestamp_profile(
    device: &wgpu::Device,
    recorder: Option<&GpuTimestampRecorder>,
    submission_index: wgpu::SubmissionIndex,
    submitted: GpuSubmittedTimestampProfile,
) -> Result<GpuTimestampProfile> {
    if !submitted.support.enabled {
        return Ok(GpuTimestampProfile::unavailable(submitted.support));
    }
    let recorder = recorder.context("GPU timestamp recorder was not allocated for submission")?;
    recorder.read_profile(device, submission_index, submitted.stage_mask())
}

fn gpu_timestamp_required_features() -> wgpu::Features {
    wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
}

fn timestamp_period_ps(period_ns: f32) -> u64 {
    (f64::from(period_ns) * 1_000.0).round() as u64
}

fn gpu_timestamp_profile_from_samples(
    support: GpuTimestampSupport,
    samples: &[u64; GPU_TIMESTAMP_QUERY_COUNT_USIZE],
    stages: GpuTimestampStageMask,
) -> GpuTimestampProfile {
    let total_command = timestamp_delta_duration(
        samples[GpuTimestampQuery::TotalStart as usize],
        samples[GpuTimestampQuery::TotalEnd as usize],
        support.period_ps,
    );
    let output_clear = stages.output_clear.then(|| {
        timestamp_delta_duration(
            samples[GpuTimestampQuery::OutputClearStart as usize],
            samples[GpuTimestampQuery::OutputClearEnd as usize],
            support.period_ps,
        )
    });
    let decode_dispatch = timestamp_delta_duration(
        samples[GpuTimestampQuery::DecodeDispatchStart as usize],
        samples[GpuTimestampQuery::DecodeDispatchEnd as usize],
        support.period_ps,
    );
    let vignette_dispatch = stages.vignette.then(|| {
        timestamp_delta_duration(
            samples[GpuTimestampQuery::VignetteDispatchStart as usize],
            samples[GpuTimestampQuery::VignetteDispatchEnd as usize],
            support.period_ps,
        )
    });
    let copy_to_readback = timestamp_delta_duration(
        samples[GpuTimestampQuery::CopyToReadbackStart as usize],
        samples[GpuTimestampQuery::CopyToReadbackEnd as usize],
        support.period_ps,
    );

    let output_clear = output_clear.flatten();
    let vignette_dispatch = vignette_dispatch.flatten();
    let samples_valid = total_command.is_some()
        && decode_dispatch.is_some()
        && copy_to_readback.is_some()
        && (!stages.output_clear || output_clear.is_some())
        && (!stages.vignette || vignette_dispatch.is_some());

    GpuTimestampProfile {
        requested: support.requested,
        supported: support.supported,
        enabled: support.enabled,
        period_ps: support.period_ps,
        samples_valid,
        output_clear,
        decode_dispatch,
        vignette_dispatch,
        copy_to_readback,
        total_command,
        query_resolve_readback: Duration::ZERO,
    }
}

fn timestamp_delta_duration(start: u64, end: u64, period_ps: u64) -> Option<Duration> {
    if period_ps == 0 || end < start {
        return None;
    }

    let delta_ticks = u128::from(end - start);
    let total_ps = delta_ticks.checked_mul(u128::from(period_ps))?;
    let nanos = (total_ps + 500) / 1_000;
    let nanos = u64::try_from(nanos).ok()?;
    Some(Duration::from_nanos(nanos))
}

// GPU output layout selected by a decode path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuOutputLayout {
    PackedU16Samples,
}

impl GpuOutputLayout {
    fn output_clear_enabled(self) -> bool {
        match self {
            Self::PackedU16Samples => true,
        }
    }
}

// Compact GPU upload descriptor for one CPU-built MCRAW block work item.
//
// The rich McrawVulkanBlockWorkItem remains the CPU work-plan representation.
// This four-word representation is the storage-buffer contract consumed by WGSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
struct McrawVulkanPackedBlockWorkItem {
    word0: u32,
    word1: u32,
    word2: u32,
    word3: u32,
}

const PACKED_WORK_ITEM_BYTES: u64 = std::mem::size_of::<McrawVulkanPackedBlockWorkItem>() as u64;
const LEGACY_RAW16_BLOCK_SAMPLES: usize = 16;
const LEGACY_RAW16_ENCODING_BLOCK: usize = LEGACY_RAW16_BLOCK_SAMPLES * 2;
const LEGACY_RAW16_WORKGROUP_SIZE: usize = 128;
const LEGACY_RAW16_HEADER_LENGTH: usize = 2;
const LEGACY_RAW16_BLOCK_LENGTHS: [usize; 17] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 32, 32, 32, 32, 32, 32,
];

#[derive(Debug, Default)]
struct LegacyRaw16GpuWorkPlan {
    blocks: Vec<McrawVulkanBlockWorkItem>,
    chunk_count: usize,
}

impl LegacyRaw16GpuWorkPlan {
    fn allocation_stats(&self) -> GpuWorkPlanAllocationStats {
        let blocks_len = usize_to_u64_saturating(self.blocks.len());
        let blocks_capacity = usize_to_u64_saturating(self.blocks.capacity());
        GpuWorkPlanAllocationStats {
            blocks_len,
            blocks_capacity,
            work_items_len: blocks_len,
            work_items_capacity: blocks_capacity,
            work_items_bytes: blocks_len.saturating_mul(PACKED_WORK_ITEM_BYTES),
            reused_scratch: false,
            scratch_grow_count: 0,
        }
    }
}

fn requested_device_limits(
    adapter_limits: &wgpu::Limits,
    required_limits: Option<&GpuDecodeRequiredLimits>,
) -> Result<wgpu::Limits> {
    let mut requested = wgpu::Limits::downlevel_defaults().using_resolution(adapter_limits.clone());
    let Some(required) = required_limits else {
        return Ok(requested);
    };
    if required.dimensions.width == 0 || required.dimensions.height == 0 {
        anyhow::bail!(
            "GPU resource requirements have invalid dimensions {}x{}",
            required.dimensions.width,
            required.dimensions.height,
        );
    }
    if required.storage_buffer_binding_bytes == 0 {
        anyhow::bail!(
            "GPU resource {:?} for {}x{} has invalid required max_storage_buffer_binding_size=0 bytes",
            required.storage_buffer_resource,
            required.dimensions.width,
            required.dimensions.height,
        );
    }
    if required.buffer_bytes == 0 {
        anyhow::bail!(
            "GPU resource {:?} for {}x{} has invalid required max_buffer_size=0 bytes",
            required.buffer_resource,
            required.dimensions.width,
            required.dimensions.height,
        );
    }

    let requested_storage_bytes = u64::from(requested.max_storage_buffer_binding_size)
        .max(required.storage_buffer_binding_bytes);
    validate_requested_device_limit(
        required.dimensions,
        required.storage_buffer_resource,
        "max_storage_buffer_binding_size",
        required.storage_buffer_binding_bytes,
        u64::from(adapter_limits.max_storage_buffer_binding_size),
        requested_storage_bytes,
    )?;
    requested.max_storage_buffer_binding_size =
        u32::try_from(requested_storage_bytes).map_err(|_| {
            anyhow!(
                "GPU resource {:?} for {}x{} requires max_storage_buffer_binding_size={} bytes, which does not fit the wgpu u32 limit domain",
                required.storage_buffer_resource,
                required.dimensions.width,
                required.dimensions.height,
                required.storage_buffer_binding_bytes,
            )
        })?;

    let (buffer_resource, required_buffer_bytes) =
        if required.storage_buffer_binding_bytes > required.buffer_bytes {
            (
                required.storage_buffer_resource,
                required.storage_buffer_binding_bytes,
            )
        } else {
            (required.buffer_resource, required.buffer_bytes)
        };
    let requested_buffer_bytes = requested.max_buffer_size.max(required_buffer_bytes);
    validate_requested_device_limit(
        required.dimensions,
        buffer_resource,
        "max_buffer_size",
        required_buffer_bytes,
        adapter_limits.max_buffer_size,
        requested_buffer_bytes,
    )?;
    requested.max_buffer_size = requested_buffer_bytes;

    Ok(requested)
}

fn validate_requested_device_limit(
    dimensions: FrameDimensions,
    resource: &'static str,
    limit: &'static str,
    required_bytes: u64,
    adapter_supported_bytes: u64,
    requested_bytes: u64,
) -> Result<()> {
    if requested_bytes > adapter_supported_bytes {
        anyhow::bail!(
            "GPU resource {:?} for {}x{} requires {limit}={required_bytes} bytes; adapter-supported={adapter_supported_bytes} bytes; logical-device-requested={requested_bytes} bytes",
            resource,
            dimensions.width,
            dimensions.height,
        );
    }
    Ok(())
}

impl GpuDecodeBackend {
    // Create a reusable GPU backend using async wgpu initialization.
    pub async fn new(config: GpuDecodeConfig) -> Result<Self> {
        Self::new_inner(config, None).await
    }

    /// Create a reusable GPU backend with exact buffer requirements for the
    /// caller's known job. Other device limits retain their downlevel defaults.
    pub async fn new_with_required_limits(
        config: GpuDecodeConfig,
        required_limits: GpuDecodeRequiredLimits,
    ) -> Result<Self> {
        Self::new_inner(config, Some(required_limits)).await
    }

    async fn new_inner(
        config: GpuDecodeConfig,
        required_limits: Option<GpuDecodeRequiredLimits>,
    ) -> Result<Self> {
        let instance = create_instance(config.backend_preference);
        let adapters = enumerate_adapters(&instance, config.backend_preference).await;
        let selected = select_best_adapter(adapters).context("no suitable wgpu adapter found")?;

        let adapter_info = selected.info;
        let adapter_limits = selected.adapter.limits();
        let requested_limits = requested_device_limits(&adapter_limits, required_limits.as_ref())?;
        let adapter_features = selected.adapter.features();
        let timestamp_features = gpu_timestamp_required_features();
        let timestamp_supported = adapter_features.contains(timestamp_features);
        let timestamp_enabled = config.enable_gpu_timestamps && timestamp_supported;
        let mut required_features = wgpu::Features::empty();
        if timestamp_enabled {
            required_features |= timestamp_features;
        }
        let (device, queue) = selected
            .adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("mcraw4vulkan GPU decode device"),
                    required_features,
                    required_limits: requested_limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("failed to create wgpu device and queue")?;
        let timestamp_support = GpuTimestampSupport {
            requested: config.enable_gpu_timestamps,
            supported: timestamp_supported,
            enabled: timestamp_enabled,
            period_ps: if timestamp_enabled {
                timestamp_period_ps(queue.get_timestamp_period())
            } else {
                0
            },
        };
        let decode_bind_group_layout = create_decode_bind_group_layout(&device);
        let packed_u16_pipeline = create_decode_pipeline(
            &device,
            "mcraw4vulkan MCRAW packed u16 block decode WGSL shader",
            "mcraw4vulkan MCRAW packed u16 block decode compute pipeline",
            MCRAW_DECODE_BLOCKS_PACKED_U16_WGSL,
            &decode_bind_group_layout,
        );
        let legacy_raw16_packed_u16_pipeline = create_decode_pipeline(
            &device,
            "mcraw4vulkan legacy raw16 packed u16 decode WGSL shader",
            "mcraw4vulkan legacy raw16 packed u16 decode compute pipeline",
            MCRAW_DECODE_LEGACY_RAW16_PACKED_U16_WGSL,
            &decode_bind_group_layout,
        );

        let limits = device.limits();

        Ok(Self {
            adapter_info,
            limits,
            timestamp_support,
            buffers: ReusableGpuBuffers::default(),
            pipeline_slots: Vec::new(),
            packed_u16_pipeline,
            legacy_raw16_packed_u16_pipeline,
            queue,
            device,
        })
    }

    pub fn new_blocking(config: GpuDecodeConfig) -> Result<Self> {
        pollster::block_on(Self::new(config))
    }

    pub fn new_blocking_with_required_limits(
        config: GpuDecodeConfig,
        required_limits: GpuDecodeRequiredLimits,
    ) -> Result<Self> {
        pollster::block_on(Self::new_with_required_limits(config, required_limits))
    }

    pub fn from_wgpu_device(
        adapter_info: wgpu::AdapterInfo,
        adapter_features: wgpu::Features,
        device: wgpu::Device,
        queue: wgpu::Queue,
        config: GpuDecodeConfig,
    ) -> Result<Self> {
        let timestamp_features = gpu_timestamp_required_features();
        let timestamp_supported = adapter_features.contains(timestamp_features);
        let timestamp_enabled = config.enable_gpu_timestamps && timestamp_supported;
        if config.enable_gpu_timestamps && !timestamp_supported {
            anyhow::bail!(
                "GPU timestamps requested but adapter does not support timestamp query features"
            );
        }
        if timestamp_enabled && !device.features().contains(timestamp_features) {
            anyhow::bail!(
                "GPU timestamps requested but device was not created with timestamp query features"
            );
        }

        let decode_bind_group_layout = create_decode_bind_group_layout(&device);
        let packed_u16_pipeline = create_decode_pipeline(
            &device,
            "mcraw4vulkan MCRAW packed u16 block decode WGSL shader",
            "mcraw4vulkan MCRAW packed u16 block decode compute pipeline",
            MCRAW_DECODE_BLOCKS_PACKED_U16_WGSL,
            &decode_bind_group_layout,
        );
        let legacy_raw16_packed_u16_pipeline = create_decode_pipeline(
            &device,
            "mcraw4vulkan legacy raw16 packed u16 decode WGSL shader",
            "mcraw4vulkan legacy raw16 packed u16 decode compute pipeline",
            MCRAW_DECODE_LEGACY_RAW16_PACKED_U16_WGSL,
            &decode_bind_group_layout,
        );

        let limits = device.limits();
        let timestamp_support = GpuTimestampSupport {
            requested: config.enable_gpu_timestamps,
            supported: timestamp_supported,
            enabled: timestamp_enabled,
            period_ps: if timestamp_enabled {
                timestamp_period_ps(queue.get_timestamp_period())
            } else {
                0
            },
        };
        Ok(Self {
            adapter_info,
            limits,
            timestamp_support,
            buffers: ReusableGpuBuffers::default(),
            pipeline_slots: Vec::new(),
            packed_u16_pipeline,
            legacy_raw16_packed_u16_pipeline,
            queue,
            device,
        })
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn no_readback_slot_allocation_snapshot(
        &self,
        slot_index: usize,
    ) -> GpuNoReadbackSlotAllocation {
        self.pipeline_slots
            .get(slot_index)
            .map(ReusableGpuBuffers::allocation_snapshot)
            .unwrap_or_default()
    }

    /// Returns the first retained decoder allocation that would violate a
    /// caller's requirement to take ownership of a pristine backend.
    ///
    /// `None` identifies the legacy reusable decoder set; `Some(index)`
    /// identifies a submitted/no-readback pipeline slot.
    pub fn first_nonpristine_decoder_allocation(
        &self,
    ) -> Option<(Option<usize>, GpuNoReadbackSlotAllocation)> {
        let legacy = self.buffers.allocation_snapshot();
        if legacy != GpuNoReadbackSlotAllocation::default() {
            return Some((None, legacy));
        }
        self.pipeline_slots
            .iter()
            .enumerate()
            .find_map(|(index, slot)| {
                let allocation = slot.allocation_snapshot();
                (allocation != GpuNoReadbackSlotAllocation::default())
                    .then_some((Some(index), allocation))
            })
    }

    pub fn prospective_type7_no_readback_slot_allocation(
        &self,
        slot_index: usize,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
    ) -> Result<GpuNoReadbackSlotAllocation> {
        let work_plan = build_vulkan_work_plan(raw_payload, visible_dimensions)
            .context("failed to build type-7 allocation preflight work plan")?;
        self.prospective_no_readback_slot_allocation(
            slot_index,
            raw_payload,
            work_plan.blocks.len(),
            visible_dimensions,
        )
    }

    pub fn prospective_prepared_type7_no_readback_slot_allocation(
        &self,
        slot_index: usize,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        prepared: GpuPreparedType7WorkPlan<'_>,
    ) -> Result<GpuNoReadbackSlotAllocation> {
        anyhow::ensure!(
            prepared.work_plan.visible_dimensions == visible_dimensions,
            "prepared type-7 visible dimensions do not match allocation request"
        );
        self.prospective_no_readback_slot_allocation(
            slot_index,
            raw_payload,
            prepared.work_plan.blocks.len(),
            visible_dimensions,
        )
    }

    pub fn prospective_type6_no_readback_slot_allocation(
        &self,
        slot_index: usize,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        row_stride: u32,
    ) -> Result<GpuNoReadbackSlotAllocation> {
        let work_plan =
            build_legacy_raw16_gpu_work_plan(raw_payload, visible_dimensions, row_stride)
                .context("failed to build type-6 allocation preflight work plan")?;
        self.prospective_no_readback_slot_allocation(
            slot_index,
            raw_payload,
            work_plan.blocks.len(),
            visible_dimensions,
        )
    }

    pub fn prospective_prepared_type6_no_readback_slot_allocation(
        &self,
        slot_index: usize,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        row_stride: u32,
        prepared: GpuPreparedType6WorkPlan<'_>,
    ) -> Result<GpuNoReadbackSlotAllocation> {
        anyhow::ensure!(
            prepared.visible_dimensions == visible_dimensions,
            "prepared type-6 visible dimensions do not match allocation request"
        );
        anyhow::ensure!(
            prepared.row_stride == row_stride,
            "prepared type-6 row stride does not match allocation request"
        );
        self.prospective_no_readback_slot_allocation(
            slot_index,
            raw_payload,
            prepared.blocks.len(),
            visible_dimensions,
        )
    }

    pub fn prospective_cpu_upload_no_readback_slot_allocation(
        &self,
        slot_index: usize,
        visible_dimensions: FrameDimensions,
    ) -> Result<GpuNoReadbackSlotAllocation> {
        let pixel_count = visible_dimensions
            .pixel_count()
            .context("CPU-upload allocation preflight dimensions overflow")?;
        let packed_output_bytes =
            byte_len_for_output_layout(pixel_count, GpuOutputLayout::PackedU16Samples)?;
        let buffers = self.pipeline_slots.get(slot_index);
        Ok(buffers.map_or(
            GpuNoReadbackSlotAllocation {
                packed_output_bytes,
                ..GpuNoReadbackSlotAllocation::default()
            },
            |slot| slot.prospective_allocation(0, 0, packed_output_bytes, 0),
        ))
    }

    fn prospective_no_readback_slot_allocation(
        &self,
        slot_index: usize,
        raw_payload: &[u8],
        work_item_count: usize,
        visible_dimensions: FrameDimensions,
    ) -> Result<GpuNoReadbackSlotAllocation> {
        let pixel_count = visible_dimensions
            .pixel_count()
            .context("decoder allocation preflight dimensions overflow")?;
        let raw_payload_bytes = u64::try_from(padded_u32_byte_len(raw_payload.len())?)
            .context("decoder allocation raw payload bytes overflow")?;
        let work_item_bytes = u64::try_from(work_item_byte_len(work_item_count)?)
            .context("decoder allocation work-item bytes overflow")?;
        let packed_output_bytes =
            byte_len_for_output_layout(pixel_count, GpuOutputLayout::PackedU16Samples)?;
        let params_bytes = u64::try_from(mcraw_params_to_le_bytes(0, 0, 0, 0, 0).len())
            .context("decoder allocation params bytes overflow")?;
        Ok(self.pipeline_slots.get(slot_index).map_or(
            GpuNoReadbackSlotAllocation {
                raw_payload_bytes,
                work_item_bytes,
                packed_output_bytes,
                params_bytes,
                ..GpuNoReadbackSlotAllocation::default()
            },
            |slot| {
                slot.prospective_allocation(
                    raw_payload_bytes,
                    work_item_bytes,
                    packed_output_bytes,
                    params_bytes,
                )
            },
        ))
    }

    pub fn gpu_timestamp_support(&self) -> GpuTimestampSupport {
        self.timestamp_support
    }

    // Create an isolated vignette GPU corrector on this backend's device.
    //
    // The corrector is owned by the caller so it can be reused across frames
    // and kept disabled until a higher-level path explicitly opts in.
    pub fn create_vignette_corrector(&self) -> Result<GpuVignetteCorrector> {
        GpuVignetteCorrector::new(&self.device)
            .map_err(|error| anyhow!("failed to create GPU vignette corrector: {error}"))
    }

    // Upload a prepared full-resolution gain map once for reuse by opt-in
    // vignette decode calls. This method does not parse metadata or build the
    // gain map; callers provide the already-prepared typed representation.
    pub fn upload_vignette_gain_map(
        &self,
        corrector: &GpuVignetteCorrector,
        gain_map: &PreparedFullResolutionFixedGainMap,
    ) -> Result<GpuUploadedFullResolutionGainMap> {
        corrector
            .upload_full_resolution_gain_map(&self.device, &self.queue, gain_map)
            .map_err(|error| anyhow!("failed to upload GPU vignette gain map: {error}"))
    }

    pub fn upload_vignette_gain_map_with_timings(
        &self,
        corrector: &GpuVignetteCorrector,
        gain_map: &PreparedFullResolutionFixedGainMap,
    ) -> Result<GpuVignetteGainMapUpload> {
        corrector
            .upload_full_resolution_gain_map_with_timings(&self.device, &self.queue, gain_map)
            .map_err(|error| anyhow!("failed to upload GPU vignette gain map: {error}"))
    }

    pub fn upload_compact_vignette_gain_map(
        &self,
        corrector: &mut GpuVignetteCorrector,
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<GpuUploadedFullResolutionGainMap> {
        corrector
            .upload_compact_gain_map_from_fixed_facts(&self.device, &self.queue, facts)
            .map_err(|error| anyhow!("failed to upload compact GPU vignette gain map: {error}"))
    }

    // Decode through the canonical packed-u16 GPU path and expose mapped
    // little-endian Bayer bytes for sink production.
    //
    // DNG/FUSE sinks need readback bytes, but they should still enter through the
    // same packed-output decode policy as preview/RGB production. The closure
    // must not retain the mapped slice after it returns.
    pub fn decode_raw_payload_to_canonical_bayer_u16_mapped<T, F>(
        &mut self,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        consume_mapped_pixels: F,
    ) -> Result<GpuMappedPackedU16Output<T>>
    where
        F: FnOnce(&[u8]) -> Result<T>,
    {
        let work_plan_start = Instant::now();
        let work_plan = build_vulkan_work_plan(raw_payload, visible_dimensions)
            .context("failed to build mapped packed Vulkan GPU work plan")?;
        let work_plan_elapsed = work_plan_start.elapsed();

        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        let packed_output_layout = GpuOutputLayout::PackedU16Samples;
        let packed_u16_pipeline = &self.packed_u16_pipeline;

        let dispatch_result = run_mcraw_gpu_dispatch_packed_u16_mapped(
            &self.device,
            &self.queue,
            packed_u16_pipeline,
            &mut self.buffers,
            self.timestamp_support,
            raw_payload,
            &work_plan.blocks,
            pixel_count,
            visible_dimensions.width,
            visible_dimensions.height,
            self.limits.max_compute_workgroups_per_dimension,
            packed_output_layout,
            consume_mapped_pixels,
        )?;

        let macroblock_count = work_plan.blocks.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
        let expected_invocations = expected_invocations_for_packed_output(macroblock_count)?;

        let gpu_total_without_consumer =
            work_plan_elapsed.saturating_add(dispatch_result.timings.total);

        Ok(GpuMappedPackedU16Output {
            value: dispatch_result.value,
            dimensions: visible_dimensions,
            work_item_count: work_plan.blocks.len(),
            macroblock_count,
            expected_invocations,
            timings: GpuDecodeTimings::from_work_plan_dispatch(
                &work_plan,
                work_plan_elapsed,
                dispatch_result.timings,
                gpu_total_without_consumer,
            ),
        })
    }

    // Decode through the canonical packed-u16 GPU path, optionally run the
    // full-resolution GPU vignette correction stage, then expose mapped
    // little-endian Bayer bytes for sink production.
    pub fn decode_raw_payload_to_canonical_bayer_u16_mapped_with_vignette<T, F>(
        &mut self,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        consume_mapped_pixels: F,
    ) -> Result<GpuMappedPackedU16WithVignetteOutput<T>>
    where
        F: FnOnce(&[u8]) -> Result<T>,
    {
        let work_plan_start = Instant::now();
        let work_plan = build_vulkan_work_plan(raw_payload, visible_dimensions)
            .context("failed to build mapped packed Vulkan GPU work plan")?;
        let work_plan_elapsed = work_plan_start.elapsed();

        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        let packed_output_layout = GpuOutputLayout::PackedU16Samples;
        let packed_u16_pipeline = &self.packed_u16_pipeline;

        let dispatch_result = run_mcraw_gpu_dispatch_packed_u16_mapped_with_vignette(
            &self.device,
            &self.queue,
            packed_u16_pipeline,
            &mut self.buffers,
            self.timestamp_support,
            raw_payload,
            &work_plan.blocks,
            pixel_count,
            visible_dimensions.width,
            visible_dimensions.height,
            self.limits.max_compute_workgroups_per_dimension,
            packed_output_layout,
            vignette_correction,
            consume_mapped_pixels,
        )?;

        let macroblock_count = work_plan.blocks.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
        let expected_invocations = expected_invocations_for_packed_output(macroblock_count)?;

        let gpu_total_without_consumer =
            work_plan_elapsed.saturating_add(dispatch_result.timings.total);

        Ok(GpuMappedPackedU16WithVignetteOutput {
            value: dispatch_result.value,
            dimensions: visible_dimensions,
            work_item_count: work_plan.blocks.len(),
            macroblock_count,
            expected_invocations,
            timings: GpuDecodeTimings::from_work_plan_dispatch(
                &work_plan,
                work_plan_elapsed,
                dispatch_result.timings,
                gpu_total_without_consumer,
            ),
            vignette_stats: dispatch_result.vignette_stats,
        })
    }

    // Decode legacy compressionType 6 raw16 binned payload blocks directly on
    // the GPU into the same canonical packed-u16 Bayer sink contract as type 7.
    pub fn decode_legacy_raw16_payload_to_canonical_bayer_u16_mapped<T, F>(
        &mut self,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        row_stride: u32,
        consume_mapped_pixels: F,
    ) -> Result<GpuMappedPackedU16Output<T>>
    where
        F: FnOnce(&[u8]) -> Result<T>,
    {
        let output = self.decode_legacy_raw16_payload_to_canonical_bayer_u16_mapped_with_vignette(
            raw_payload,
            visible_dimensions,
            row_stride,
            None,
            consume_mapped_pixels,
        )?;

        Ok(GpuMappedPackedU16Output {
            value: output.value,
            dimensions: output.dimensions,
            work_item_count: output.work_item_count,
            macroblock_count: output.macroblock_count,
            expected_invocations: output.expected_invocations,
            timings: output.timings,
        })
    }

    // Decode legacy compressionType 6 raw16 binned payload blocks directly on
    // the GPU, optionally apply the common full-resolution vignette stage, then
    // expose mapped little-endian Bayer bytes to sink builders.
    pub fn decode_legacy_raw16_payload_to_canonical_bayer_u16_mapped_with_vignette<T, F>(
        &mut self,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        row_stride: u32,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        consume_mapped_pixels: F,
    ) -> Result<GpuMappedPackedU16WithVignetteOutput<T>>
    where
        F: FnOnce(&[u8]) -> Result<T>,
    {
        let work_plan_start = Instant::now();
        let work_plan =
            build_legacy_raw16_gpu_work_plan(raw_payload, visible_dimensions, row_stride)
                .context("failed to build legacy raw16 GPU work plan")?;
        let work_plan_elapsed = work_plan_start.elapsed();

        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        let output_layout = legacy_raw16_gpu_output_layout();

        let dispatch_result = run_legacy_raw16_gpu_dispatch_mapped_with_vignette(
            &self.device,
            &self.queue,
            &self.legacy_raw16_packed_u16_pipeline,
            &mut self.buffers,
            self.timestamp_support,
            raw_payload,
            &work_plan.blocks,
            work_plan.chunk_count,
            pixel_count,
            visible_dimensions.width,
            visible_dimensions.height,
            self.limits.max_compute_workgroups_per_dimension,
            output_layout,
            vignette_correction,
            consume_mapped_pixels,
        )?;

        let expected_invocations = legacy_raw16_expected_invocations(
            work_plan.chunk_count,
            self.limits.max_compute_workgroups_per_dimension,
        )?;
        let gpu_total_without_consumer =
            work_plan_elapsed.saturating_add(dispatch_result.timings.total);

        Ok(GpuMappedPackedU16WithVignetteOutput {
            value: dispatch_result.value,
            dimensions: visible_dimensions,
            work_item_count: work_plan.blocks.len(),
            macroblock_count: work_plan.chunk_count,
            expected_invocations,
            timings: GpuDecodeTimings::from_work_plan_stats_dispatch(
                work_plan.allocation_stats(),
                work_plan_elapsed,
                dispatch_result.timings,
                gpu_total_without_consumer,
            ),
            vignette_stats: dispatch_result.vignette_stats,
        })
    }

    // Apply the canonical GPU vignette stage to already-decoded little-endian
    // packed-u16 Bayer bytes and expose mapped corrected bytes to the caller.
    //
    // This is the CPU-fallback sink bridge: CPU decode owns only decoding, while
    // the correction stage remains the same full-resolution GPU LumaPlane0 path
    // used by GPU-decoded DNG/preview outputs.
    pub fn correct_decoded_bayer_u16_mapped_with_vignette<T, F>(
        &mut self,
        decoded_pixel_bytes_le: &[u8],
        visible_dimensions: FrameDimensions,
        vignette_correction: OptionalGpuVignetteCorrection<'_>,
        consume_mapped_pixels: F,
    ) -> Result<GpuMappedPackedU16WithVignetteOutput<T>>
    where
        F: FnOnce(&[u8]) -> Result<T>,
    {
        let total_start = Instant::now();
        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        anyhow::ensure!(
            pixel_count > 0,
            "visible frame dimensions contain zero pixels"
        );

        let exact_pixel_byte_len = pixel_count
            .checked_mul(std::mem::size_of::<u16>())
            .context("decoded Bayer U16 byte length overflow")?;
        anyhow::ensure!(
            decoded_pixel_bytes_le.len() >= exact_pixel_byte_len,
            "decoded Bayer U16 bytes are too short: expected at least {}, got {}",
            exact_pixel_byte_len,
            decoded_pixel_bytes_le.len()
        );

        let output_layout = GpuOutputLayout::PackedU16Samples;
        let output_byte_len = byte_len_for_output_layout(pixel_count, output_layout)?;
        let output_byte_len_usize =
            usize::try_from(output_byte_len).context("GPU output byte length overflows usize")?;

        let input_bytes = if output_byte_len_usize == exact_pixel_byte_len {
            Cow::Borrowed(&decoded_pixel_bytes_le[..exact_pixel_byte_len])
        } else {
            let mut padded = vec![0_u8; output_byte_len_usize];
            padded[..exact_pixel_byte_len]
                .copy_from_slice(&decoded_pixel_bytes_le[..exact_pixel_byte_len]);
            Cow::Owned(padded)
        };

        let buffer_ensure_start = Instant::now();
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MCRAW CPU fallback decoded Bayer U16 GPU vignette input"),
            size: output_byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MCRAW CPU fallback GPU vignette readback"),
            size: output_byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let buffer_ensure_elapsed = buffer_ensure_start.elapsed();

        let upload_start = Instant::now();
        self.queue
            .write_buffer(&input_buffer, 0, input_bytes.as_ref());
        let upload_elapsed = upload_start.elapsed();

        let encode_submit_start = Instant::now();
        let command_encoder_create_start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MCRAW CPU fallback GPU vignette command encoder"),
            });
        let command_encoder_create_elapsed = command_encoder_create_start.elapsed();

        let vignette_encode_start = Instant::now();
        let dispatch = vignette_correction
            .corrector
            .dispatch_packed_u16_with_output_clear(GpuVignettePackedU16DispatchInput {
                device: &self.device,
                queue: &self.queue,
                encoder: &mut encoder,
                input_buffer: &input_buffer,
                input_buffer_bytes: output_byte_len,
                uploaded_gain_map: vignette_correction.uploaded_gain_map,
                params: vignette_correction.params,
            })
            .context("failed to dispatch GPU vignette correction for CPU fallback frame")?;
        let vignette_encode_elapsed = vignette_encode_start.elapsed();
        let vignette_stats = dispatch.stats;

        let copy_to_readback_encode_start = Instant::now();
        encoder.copy_buffer_to_buffer(
            dispatch.output.buffer(),
            0,
            &readback_buffer,
            0,
            output_byte_len,
        );
        let copy_to_readback_encode_elapsed = copy_to_readback_encode_start.elapsed();

        let command_finish_submit_start = Instant::now();
        let submission_index = self.queue.submit(Some(encoder.finish()));
        let command_finish_submit_elapsed = command_finish_submit_start.elapsed();
        let encode_submit_elapsed = encode_submit_start.elapsed();

        let readback_slice = readback_buffer.slice(0..output_byte_len);
        let wait_map_start = Instant::now();
        let (sender, receiver) = mpsc::channel();
        readback_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device
            .poll(wgpu::Maintain::wait_for(submission_index.clone()));

        let map_result = receiver
            .recv_timeout(Duration::from_secs(30))
            .context("timed out waiting for CPU fallback GPU vignette readback mapping callback")?;

        map_result.map_err(|error| {
            anyhow!("failed to map CPU fallback GPU vignette readback: {error:?}")
        })?;

        let wait_map_elapsed = wait_map_start.elapsed();
        let dispatch_readback_elapsed = total_start.elapsed();

        let mapped_consumer_start = Instant::now();
        let consume_result = {
            let mapped = readback_slice.get_mapped_range();
            let value = consume_mapped_pixels(&mapped);
            drop(mapped);
            value
        };
        let mapped_consumer_elapsed = mapped_consumer_start.elapsed();

        readback_buffer.unmap();

        let value = consume_result?;
        let timings = GpuDecodeTimings {
            output_bytes: output_byte_len,
            readback_bytes: output_byte_len,
            cpu_prepare: buffer_ensure_elapsed,
            buffer_ensure: buffer_ensure_elapsed,
            upload: upload_elapsed,
            command_encoder_create: command_encoder_create_elapsed,
            vignette_encode: vignette_encode_elapsed,
            copy_to_readback_encode: copy_to_readback_encode_elapsed,
            command_finish_submit: command_finish_submit_elapsed,
            encode_submit: encode_submit_elapsed,
            wait_map: wait_map_elapsed,
            mapped_consumer: mapped_consumer_elapsed,
            dispatch_readback: dispatch_readback_elapsed,
            total: total_start.elapsed(),
            ..GpuDecodeTimings::default()
        };

        Ok(GpuMappedPackedU16WithVignetteOutput {
            value,
            dimensions: visible_dimensions,
            work_item_count: 0,
            macroblock_count: 0,
            expected_invocations: 0,
            timings,
            vignette_stats: Some(vignette_stats),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_raw_payload_packed_u16_gpu_stage_mapped_with_vignette<T, S, E, C>(
        &mut self,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        encode_gpu_stage: E,
        consume_mapped_readback: C,
    ) -> Result<GpuMappedGpuStageOutput<T, S>>
    where
        E: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            GpuDecodedPackedU16BufferView<'_>,
        ) -> Result<(S, GpuExternalReadbackBuffer)>,
        C: FnOnce(&[u8]) -> Result<T>,
    {
        let work_plan_start = Instant::now();
        let work_plan = build_vulkan_work_plan(raw_payload, visible_dimensions)
            .context("failed to build GPU-stage packed Vulkan GPU work plan")?;
        let work_plan_elapsed = work_plan_start.elapsed();

        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        let packed_output_layout = GpuOutputLayout::PackedU16Samples;
        let packed_u16_pipeline = &self.packed_u16_pipeline;

        let dispatch_result = run_mcraw_gpu_dispatch_packed_u16_mapped_with_gpu_stage(
            &self.device,
            &self.queue,
            packed_u16_pipeline,
            &mut self.buffers,
            self.timestamp_support,
            raw_payload,
            &work_plan.blocks,
            pixel_count,
            visible_dimensions.width,
            visible_dimensions.height,
            self.limits.max_compute_workgroups_per_dimension,
            packed_output_layout,
            vignette_correction,
            visible_dimensions,
            encode_gpu_stage,
            consume_mapped_readback,
        )?;

        let macroblock_count = work_plan.blocks.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
        let expected_invocations = expected_invocations_for_packed_output(macroblock_count)?;

        let gpu_total_without_consumer =
            work_plan_elapsed.saturating_add(dispatch_result.timings.total);

        Ok(GpuMappedGpuStageOutput {
            value: dispatch_result.value,
            stage: dispatch_result.stage,
            dimensions: visible_dimensions,
            work_item_count: work_plan.blocks.len(),
            macroblock_count,
            expected_invocations,
            timings: GpuDecodeTimings::from_work_plan_dispatch(
                &work_plan,
                work_plan_elapsed,
                dispatch_result.timings,
                gpu_total_without_consumer,
            ),
            vignette_stats: dispatch_result.vignette_stats,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_raw_payload_packed_u16_gpu_stage_mapped_with_vignette_with_scratch<T, S, E, C>(
        &mut self,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        scratch: &mut GpuDecodeScratch,
        encode_gpu_stage: E,
        consume_mapped_readback: C,
    ) -> Result<GpuMappedGpuStageOutput<T, S>>
    where
        E: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            GpuDecodedPackedU16BufferView<'_>,
        ) -> Result<(S, GpuExternalReadbackBuffer)>,
        C: FnOnce(&[u8]) -> Result<T>,
    {
        let work_plan_start = Instant::now();
        let work_plan =
            build_vulkan_work_plan_into(raw_payload, visible_dimensions, &mut scratch.work_plan)
                .context("failed to build scratch GPU-stage packed Vulkan GPU work plan")?;
        let work_plan_elapsed = work_plan_start.elapsed();

        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        let packed_output_layout = GpuOutputLayout::PackedU16Samples;
        let packed_u16_pipeline = &self.packed_u16_pipeline;

        let dispatch_result = run_mcraw_gpu_dispatch_packed_u16_mapped_with_gpu_stage(
            &self.device,
            &self.queue,
            packed_u16_pipeline,
            &mut self.buffers,
            self.timestamp_support,
            raw_payload,
            work_plan.blocks,
            pixel_count,
            visible_dimensions.width,
            visible_dimensions.height,
            self.limits.max_compute_workgroups_per_dimension,
            packed_output_layout,
            vignette_correction,
            visible_dimensions,
            encode_gpu_stage,
            consume_mapped_readback,
        )?;

        let macroblock_count = work_plan.blocks.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
        let expected_invocations = expected_invocations_for_packed_output(macroblock_count)?;

        let gpu_total_without_consumer =
            work_plan_elapsed.saturating_add(dispatch_result.timings.total);

        Ok(GpuMappedGpuStageOutput {
            value: dispatch_result.value,
            stage: dispatch_result.stage,
            dimensions: visible_dimensions,
            work_item_count: work_plan.blocks.len(),
            macroblock_count,
            expected_invocations,
            timings: GpuDecodeTimings::from_work_plan_ref_dispatch(
                work_plan,
                work_plan_elapsed,
                dispatch_result.timings,
                gpu_total_without_consumer,
            ),
            vignette_stats: dispatch_result.vignette_stats,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_raw_payload_packed_u16_gpu_stage_no_readback_with_vignette<S, E>(
        &mut self,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        encode_gpu_stage: E,
    ) -> Result<GpuNoReadbackGpuStageOutput<S>>
    where
        E: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            GpuDecodedPackedU16BufferView<'_>,
        ) -> Result<S>,
    {
        let work_plan_start = Instant::now();
        let work_plan = build_vulkan_work_plan(raw_payload, visible_dimensions)
            .context("failed to build no-readback GPU-stage packed Vulkan GPU work plan")?;
        let work_plan_elapsed = work_plan_start.elapsed();

        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        let packed_output_layout = GpuOutputLayout::PackedU16Samples;
        let packed_u16_pipeline = &self.packed_u16_pipeline;

        let dispatch_result = run_mcraw_gpu_dispatch_packed_u16_no_readback_with_gpu_stage(
            &self.device,
            &self.queue,
            packed_u16_pipeline,
            &mut self.buffers,
            self.timestamp_support,
            raw_payload,
            &work_plan.blocks,
            pixel_count,
            visible_dimensions.width,
            visible_dimensions.height,
            self.limits.max_compute_workgroups_per_dimension,
            packed_output_layout,
            vignette_correction,
            visible_dimensions,
            encode_gpu_stage,
        )?;

        let macroblock_count = work_plan.blocks.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
        let expected_invocations = expected_invocations_for_packed_output(macroblock_count)?;

        let gpu_total_without_consumer =
            work_plan_elapsed.saturating_add(dispatch_result.timings.total);

        Ok(GpuNoReadbackGpuStageOutput {
            stage: dispatch_result.stage,
            dimensions: visible_dimensions,
            work_item_count: work_plan.blocks.len(),
            macroblock_count,
            expected_invocations,
            timings: GpuDecodeTimings::from_work_plan_dispatch(
                &work_plan,
                work_plan_elapsed,
                dispatch_result.timings,
                gpu_total_without_consumer,
            ),
            vignette_stats: dispatch_result.vignette_stats,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_legacy_raw16_payload_packed_u16_gpu_stage_no_readback_with_vignette<S, E>(
        &mut self,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        row_stride: u32,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        encode_gpu_stage: E,
    ) -> Result<GpuNoReadbackGpuStageOutput<S>>
    where
        E: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            GpuDecodedPackedU16BufferView<'_>,
        ) -> Result<S>,
    {
        let work_plan_start = Instant::now();
        let work_plan =
            build_legacy_raw16_gpu_work_plan(raw_payload, visible_dimensions, row_stride)
                .context("failed to build no-readback legacy raw16 GPU work plan")?;
        let work_plan_elapsed = work_plan_start.elapsed();

        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        let output_layout = legacy_raw16_gpu_output_layout();

        let dispatch_result = run_legacy_raw16_gpu_dispatch_no_readback_with_gpu_stage(
            &self.device,
            &self.queue,
            &self.legacy_raw16_packed_u16_pipeline,
            &mut self.buffers,
            self.timestamp_support,
            raw_payload,
            &work_plan.blocks,
            work_plan.chunk_count,
            pixel_count,
            visible_dimensions.width,
            visible_dimensions.height,
            self.limits.max_compute_workgroups_per_dimension,
            output_layout,
            vignette_correction,
            visible_dimensions,
            encode_gpu_stage,
        )?;

        let expected_invocations = legacy_raw16_expected_invocations(
            work_plan.chunk_count,
            self.limits.max_compute_workgroups_per_dimension,
        )?;
        let gpu_total_without_consumer =
            work_plan_elapsed.saturating_add(dispatch_result.timings.total);

        Ok(GpuNoReadbackGpuStageOutput {
            stage: dispatch_result.stage,
            dimensions: visible_dimensions,
            work_item_count: work_plan.blocks.len(),
            macroblock_count: work_plan.chunk_count,
            expected_invocations,
            timings: GpuDecodeTimings::from_work_plan_stats_dispatch(
                work_plan.allocation_stats(),
                work_plan_elapsed,
                dispatch_result.timings,
                gpu_total_without_consumer,
            ),
            vignette_stats: dispatch_result.vignette_stats,
        })
    }

    /// Submit one legacy/type-6 RAW16 frame into a reusable backend slot.
    ///
    /// This is the nonblocking counterpart of
    /// [`Self::decode_legacy_raw16_payload_packed_u16_gpu_stage_no_readback_with_vignette`].
    /// With timestamps disabled, the same slot may be reused immediately after
    /// submit only when every later command is submitted to this same ordered
    /// queue and the consumer has copied its result into independently retained
    /// storage. Otherwise the selected slot must not be reused until
    /// `submission_index` completes.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_legacy_raw16_payload_packed_u16_gpu_stage_no_readback_slot_with_vignette<S, E>(
        &mut self,
        slot_index: usize,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        row_stride: u32,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        encode_gpu_stage: E,
    ) -> Result<GpuSubmittedNoReadbackGpuStageOutput<S>>
    where
        E: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            GpuDecodedPackedU16BufferView<'_>,
        ) -> Result<S>,
    {
        let mut scratch = GpuDecodeScratch::new();
        let prepared = scratch
            .prepare_type6(raw_payload, visible_dimensions, row_stride)
            .context("failed to prepare submitted no-readback legacy raw16 GPU work plan")?;
        self.submit_prepared_legacy_raw16_payload_packed_u16_gpu_stage_no_readback_slot_with_vignette(
            slot_index,
            raw_payload,
            visible_dimensions,
            row_stride,
            prepared,
            vignette_correction,
            encode_gpu_stage,
        )
    }

    /// Submit one type-6 frame using a caller-owned checked descriptor plan.
    ///
    /// The same [`GpuPreparedType6WorkPlan`] may first be passed to
    /// [`Self::prospective_prepared_type6_no_readback_slot_allocation`], so
    /// allocation preflight and submission never rebuild the descriptor Vec.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_prepared_legacy_raw16_payload_packed_u16_gpu_stage_no_readback_slot_with_vignette<
        S,
        E,
    >(
        &mut self,
        slot_index: usize,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        row_stride: u32,
        prepared: GpuPreparedType6WorkPlan<'_>,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        encode_gpu_stage: E,
    ) -> Result<GpuSubmittedNoReadbackGpuStageOutput<S>>
    where
        E: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            GpuDecodedPackedU16BufferView<'_>,
        ) -> Result<S>,
    {
        anyhow::ensure!(
            prepared.visible_dimensions == visible_dimensions,
            "prepared type-6 visible dimensions do not match submitted frame"
        );
        anyhow::ensure!(
            prepared.row_stride == row_stride,
            "prepared type-6 row stride does not match submitted frame"
        );

        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        let output_layout = legacy_raw16_gpu_output_layout();
        ensure_pipeline_slot_count(&mut self.pipeline_slots, slot_index + 1);
        let slot_buffers = self
            .pipeline_slots
            .get_mut(slot_index)
            .context("legacy raw16 GPU no-readback slot was not allocated")?;

        let dispatch_result = submit_legacy_raw16_gpu_dispatch_no_readback_with_gpu_stage(
            &self.device,
            &self.queue,
            &self.legacy_raw16_packed_u16_pipeline,
            slot_buffers,
            self.timestamp_support,
            raw_payload,
            prepared.blocks,
            prepared.chunk_count,
            pixel_count,
            visible_dimensions.width,
            visible_dimensions.height,
            self.limits.max_compute_workgroups_per_dimension,
            output_layout,
            vignette_correction,
            visible_dimensions,
            encode_gpu_stage,
        )?;

        let expected_invocations = legacy_raw16_expected_invocations(
            prepared.chunk_count,
            self.limits.max_compute_workgroups_per_dimension,
        )?;
        let gpu_total_without_wait = prepared
            .build_elapsed
            .saturating_add(dispatch_result.timings.total);

        Ok(GpuSubmittedNoReadbackGpuStageOutput {
            stage: dispatch_result.stage,
            submission_index: dispatch_result.submission_index,
            dimensions: visible_dimensions,
            work_item_count: prepared.blocks.len(),
            macroblock_count: prepared.chunk_count,
            expected_invocations,
            timings: GpuDecodeTimings::from_work_plan_stats_dispatch(
                prepared.allocation_stats(),
                prepared.build_elapsed,
                dispatch_result.timings,
                gpu_total_without_wait,
            ),
            vignette_stats: dispatch_result.vignette_stats,
            gpu_timestamps: dispatch_result.gpu_timestamps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cpu_decoded_bayer_u16_gpu_stage_no_readback_with_vignette<S, E>(
        &mut self,
        decoded_pixel_bytes_le: &[u8],
        visible_dimensions: FrameDimensions,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        encode_gpu_stage: E,
    ) -> Result<GpuNoReadbackGpuStageOutput<S>>
    where
        E: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            GpuDecodedPackedU16BufferView<'_>,
        ) -> Result<S>,
    {
        let total_start = Instant::now();
        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        anyhow::ensure!(
            pixel_count > 0,
            "visible frame dimensions contain zero pixels"
        );
        let exact_pixel_byte_len = pixel_count
            .checked_mul(std::mem::size_of::<u16>())
            .context("CPU-decoded Bayer U16 byte length overflow")?;
        anyhow::ensure!(
            decoded_pixel_bytes_le.len() >= exact_pixel_byte_len,
            "CPU-decoded Bayer U16 bytes are too short: expected at least {}, got {}",
            exact_pixel_byte_len,
            decoded_pixel_bytes_le.len()
        );

        let output_layout = GpuOutputLayout::PackedU16Samples;
        let output_byte_len = byte_len_for_output_layout(pixel_count, output_layout)?;
        let output_byte_len_usize =
            usize::try_from(output_byte_len).context("GPU output byte length overflows usize")?;

        let input_bytes = if output_byte_len_usize == exact_pixel_byte_len {
            Cow::Borrowed(&decoded_pixel_bytes_le[..exact_pixel_byte_len])
        } else {
            let mut padded = vec![0_u8; output_byte_len_usize];
            padded[..exact_pixel_byte_len]
                .copy_from_slice(&decoded_pixel_bytes_le[..exact_pixel_byte_len]);
            Cow::Owned(padded)
        };

        let buffer_ensure_start = Instant::now();
        self.buffers
            .ensure_for_cpu_decoded_upload(&self.device, output_byte_len);
        let buffer_ensure_elapsed = buffer_ensure_start.elapsed();

        let upload_start = Instant::now();
        let output_buffer = &self
            .buffers
            .output
            .as_ref()
            .context("CPU-decoded Bayer U16 GPU-stage buffer was not allocated")?
            .buffer;
        self.queue
            .write_buffer(output_buffer, 0, input_bytes.as_ref());
        let upload_elapsed = upload_start.elapsed();

        let encode_submit_start = Instant::now();
        let command_encoder_create_start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MCRAW CPU-decoded Bayer U16 GPU-stage command encoder"),
            });
        let command_encoder_create_elapsed = command_encoder_create_start.elapsed();

        let mut vignette_stats = None;
        let mut vignette_encode_elapsed = Duration::ZERO;
        let stage = if let Some(correction) = vignette_correction {
            let vignette_encode_start = Instant::now();
            let dispatch = correction
                .corrector
                .dispatch_packed_u16(GpuVignettePackedU16DispatchInput {
                    device: &self.device,
                    queue: &self.queue,
                    encoder: &mut encoder,
                    input_buffer: output_buffer,
                    input_buffer_bytes: output_byte_len,
                    uploaded_gain_map: correction.uploaded_gain_map,
                    params: correction.params,
                })
                .context(
                    "failed to dispatch GPU vignette correction for CPU-decoded Bayer U16 frame",
                )?;
            vignette_encode_elapsed = vignette_encode_start.elapsed();
            vignette_stats = Some(dispatch.stats);
            encode_gpu_stage(
                &self.device,
                &self.queue,
                &mut encoder,
                GpuDecodedPackedU16BufferView {
                    buffer: dispatch.output.buffer(),
                    byte_len: output_byte_len,
                    dimensions: visible_dimensions,
                    pixel_count,
                },
            )?
        } else {
            encode_gpu_stage(
                &self.device,
                &self.queue,
                &mut encoder,
                GpuDecodedPackedU16BufferView {
                    buffer: output_buffer,
                    byte_len: output_byte_len,
                    dimensions: visible_dimensions,
                    pixel_count,
                },
            )?
        };

        let command_finish_submit_start = Instant::now();
        self.queue.submit(Some(encoder.finish()));
        let command_finish_submit_elapsed = command_finish_submit_start.elapsed();
        let encode_submit_elapsed = encode_submit_start.elapsed();
        self.device.poll(wgpu::Maintain::Wait);

        let timings = GpuDecodeTimings {
            raw_payload_bytes: u64::try_from(exact_pixel_byte_len).unwrap_or(u64::MAX),
            output_bytes: output_byte_len,
            cpu_prepare: buffer_ensure_elapsed,
            buffer_ensure: buffer_ensure_elapsed,
            upload: upload_elapsed,
            command_encoder_create: command_encoder_create_elapsed,
            vignette_encode: vignette_encode_elapsed,
            command_finish_submit: command_finish_submit_elapsed,
            encode_submit: encode_submit_elapsed,
            total: total_start.elapsed(),
            ..GpuDecodeTimings::default()
        };

        Ok(GpuNoReadbackGpuStageOutput {
            stage,
            dimensions: visible_dimensions,
            work_item_count: 0,
            macroblock_count: 0,
            expected_invocations: 0,
            timings,
            vignette_stats,
        })
    }

    /// Upload a CPU-decoded Bayer-U16 frame and submit a neutral downstream GPU
    /// stage without waiting. Queue order permits slot zero to be reused after
    /// each submit; callers remain responsible for retaining downstream output
    /// resources until the returned submission completes.
    pub fn submit_cpu_decoded_bayer_u16_gpu_stage_no_readback_slot_with_vignette<S, E>(
        &mut self,
        slot_index: usize,
        decoded_pixel_bytes_le: &[u8],
        visible_dimensions: FrameDimensions,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        encode_gpu_stage: E,
    ) -> Result<GpuSubmittedNoReadbackGpuStageOutput<S>>
    where
        E: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            GpuDecodedPackedU16BufferView<'_>,
        ) -> Result<S>,
    {
        let total_start = Instant::now();
        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        anyhow::ensure!(
            pixel_count > 0,
            "visible frame dimensions contain zero pixels"
        );
        let exact_pixel_byte_len = pixel_count
            .checked_mul(std::mem::size_of::<u16>())
            .context("CPU-decoded Bayer U16 byte length overflow")?;
        anyhow::ensure!(
            decoded_pixel_bytes_le.len() >= exact_pixel_byte_len,
            "CPU-decoded Bayer U16 bytes are too short: expected at least {}, got {}",
            exact_pixel_byte_len,
            decoded_pixel_bytes_le.len()
        );
        let output_byte_len =
            byte_len_for_output_layout(pixel_count, GpuOutputLayout::PackedU16Samples)?;
        let output_byte_len_usize =
            usize::try_from(output_byte_len).context("GPU output byte length overflows usize")?;
        let input_bytes = if output_byte_len_usize == exact_pixel_byte_len {
            Cow::Borrowed(&decoded_pixel_bytes_le[..exact_pixel_byte_len])
        } else {
            let mut padded = vec![0_u8; output_byte_len_usize];
            padded[..exact_pixel_byte_len]
                .copy_from_slice(&decoded_pixel_bytes_le[..exact_pixel_byte_len]);
            Cow::Owned(padded)
        };

        ensure_pipeline_slot_count(&mut self.pipeline_slots, slot_index + 1);
        let slot_buffers = self
            .pipeline_slots
            .get_mut(slot_index)
            .context("CPU-upload GPU no-readback slot was not allocated")?;
        let buffer_ensure_start = Instant::now();
        slot_buffers.ensure_for_cpu_decoded_upload(&self.device, output_byte_len);
        let buffer_ensure_elapsed = buffer_ensure_start.elapsed();
        let output_buffer = &slot_buffers
            .output
            .as_ref()
            .context("CPU-decoded Bayer U16 GPU-stage buffer was not allocated")?
            .buffer;
        let upload_start = Instant::now();
        self.queue
            .write_buffer(output_buffer, 0, input_bytes.as_ref());
        let upload_elapsed = upload_start.elapsed();

        let encode_submit_start = Instant::now();
        let command_encoder_create_start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MCRAW submitted CPU-decoded Bayer U16 GPU-stage encoder"),
            });
        let command_encoder_create_elapsed = command_encoder_create_start.elapsed();
        let mut vignette_stats = None;
        let mut vignette_encode_elapsed = Duration::ZERO;
        let stage = if let Some(correction) = vignette_correction {
            let vignette_encode_start = Instant::now();
            let dispatch = correction
                .corrector
                .dispatch_packed_u16(GpuVignettePackedU16DispatchInput {
                    device: &self.device,
                    queue: &self.queue,
                    encoder: &mut encoder,
                    input_buffer: output_buffer,
                    input_buffer_bytes: output_byte_len,
                    uploaded_gain_map: correction.uploaded_gain_map,
                    params: correction.params,
                })
                .context(
                    "failed to dispatch GPU vignette correction for submitted CPU Bayer frame",
                )?;
            vignette_encode_elapsed = vignette_encode_start.elapsed();
            vignette_stats = Some(dispatch.stats);
            encode_gpu_stage(
                &self.device,
                &self.queue,
                &mut encoder,
                GpuDecodedPackedU16BufferView {
                    buffer: dispatch.output.buffer(),
                    byte_len: output_byte_len,
                    dimensions: visible_dimensions,
                    pixel_count,
                },
            )?
        } else {
            encode_gpu_stage(
                &self.device,
                &self.queue,
                &mut encoder,
                GpuDecodedPackedU16BufferView {
                    buffer: output_buffer,
                    byte_len: output_byte_len,
                    dimensions: visible_dimensions,
                    pixel_count,
                },
            )?
        };
        let command_finish_submit_start = Instant::now();
        let submission_index = self.queue.submit(Some(encoder.finish()));
        let command_finish_submit_elapsed = command_finish_submit_start.elapsed();
        let encode_submit_elapsed = encode_submit_start.elapsed();
        let timings = GpuDecodeTimings {
            raw_payload_bytes: u64::try_from(exact_pixel_byte_len).unwrap_or(u64::MAX),
            output_bytes: output_byte_len,
            cpu_prepare: buffer_ensure_elapsed,
            buffer_ensure: buffer_ensure_elapsed,
            upload: upload_elapsed,
            command_encoder_create: command_encoder_create_elapsed,
            vignette_encode: vignette_encode_elapsed,
            command_finish_submit: command_finish_submit_elapsed,
            encode_submit: encode_submit_elapsed,
            total: total_start.elapsed(),
            ..GpuDecodeTimings::default()
        };
        Ok(GpuSubmittedNoReadbackGpuStageOutput {
            stage,
            submission_index,
            dimensions: visible_dimensions,
            work_item_count: 0,
            macroblock_count: 0,
            expected_invocations: 0,
            timings,
            vignette_stats,
            gpu_timestamps: GpuSubmittedTimestampProfile::unavailable(self.timestamp_support),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_raw_payload_packed_u16_gpu_stage_no_readback_with_vignette_with_scratch<S, E>(
        &mut self,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        scratch: &mut GpuDecodeScratch,
        encode_gpu_stage: E,
    ) -> Result<GpuNoReadbackGpuStageOutput<S>>
    where
        E: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            GpuDecodedPackedU16BufferView<'_>,
        ) -> Result<S>,
    {
        let work_plan_start = Instant::now();
        let work_plan =
            build_vulkan_work_plan_into(raw_payload, visible_dimensions, &mut scratch.work_plan)
                .context(
                    "failed to build scratch no-readback GPU-stage packed Vulkan GPU work plan",
                )?;
        let work_plan_elapsed = work_plan_start.elapsed();

        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        let packed_output_layout = GpuOutputLayout::PackedU16Samples;
        let packed_u16_pipeline = &self.packed_u16_pipeline;

        let dispatch_result = run_mcraw_gpu_dispatch_packed_u16_no_readback_with_gpu_stage(
            &self.device,
            &self.queue,
            packed_u16_pipeline,
            &mut self.buffers,
            self.timestamp_support,
            raw_payload,
            work_plan.blocks,
            pixel_count,
            visible_dimensions.width,
            visible_dimensions.height,
            self.limits.max_compute_workgroups_per_dimension,
            packed_output_layout,
            vignette_correction,
            visible_dimensions,
            encode_gpu_stage,
        )?;

        let macroblock_count = work_plan.blocks.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
        let expected_invocations = expected_invocations_for_packed_output(macroblock_count)?;

        let gpu_total_without_consumer =
            work_plan_elapsed.saturating_add(dispatch_result.timings.total);

        Ok(GpuNoReadbackGpuStageOutput {
            stage: dispatch_result.stage,
            dimensions: visible_dimensions,
            work_item_count: work_plan.blocks.len(),
            macroblock_count,
            expected_invocations,
            timings: GpuDecodeTimings::from_work_plan_ref_dispatch(
                work_plan,
                work_plan_elapsed,
                dispatch_result.timings,
                gpu_total_without_consumer,
            ),
            vignette_stats: dispatch_result.vignette_stats,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_legacy_raw16_payload_packed_u16_gpu_stage_no_readback_with_vignette_with_scratch<
        S,
        E,
    >(
        &mut self,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        row_stride: u32,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        _scratch: &mut GpuDecodeScratch,
        encode_gpu_stage: E,
    ) -> Result<GpuNoReadbackGpuStageOutput<S>>
    where
        E: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            GpuDecodedPackedU16BufferView<'_>,
        ) -> Result<S>,
    {
        self.decode_legacy_raw16_payload_packed_u16_gpu_stage_no_readback_with_vignette(
            raw_payload,
            visible_dimensions,
            row_stride,
            vignette_correction,
            encode_gpu_stage,
        )
    }

    /// Submit one type-7 frame into a reusable backend slot.
    ///
    /// With timestamps disabled, the same slot may be reused immediately after
    /// submit only when every later command is submitted to this same ordered
    /// queue and the consumer has copied its result into independently retained
    /// storage. Otherwise the selected slot must not be reused until
    /// `submission_index` completes.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_raw_payload_packed_u16_gpu_stage_no_readback_slot_with_vignette<S, E>(
        &mut self,
        slot_index: usize,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        encode_gpu_stage: E,
    ) -> Result<GpuSubmittedNoReadbackGpuStageOutput<S>>
    where
        E: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            GpuDecodedPackedU16BufferView<'_>,
        ) -> Result<S>,
    {
        let work_plan_start = Instant::now();
        let work_plan = build_vulkan_work_plan(raw_payload, visible_dimensions).context(
            "failed to build submitted no-readback GPU-stage packed Vulkan GPU work plan",
        )?;
        let work_plan_elapsed = work_plan_start.elapsed();

        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        let packed_output_layout = GpuOutputLayout::PackedU16Samples;
        let packed_u16_pipeline = &self.packed_u16_pipeline;
        ensure_pipeline_slot_count(&mut self.pipeline_slots, slot_index + 1);
        let slot_buffers = self
            .pipeline_slots
            .get_mut(slot_index)
            .context("GPU no-readback slot was not allocated")?;

        let dispatch_result = submit_mcraw_gpu_dispatch_packed_u16_no_readback_with_gpu_stage(
            &self.device,
            &self.queue,
            packed_u16_pipeline,
            slot_buffers,
            self.timestamp_support,
            raw_payload,
            &work_plan.blocks,
            pixel_count,
            visible_dimensions.width,
            visible_dimensions.height,
            self.limits.max_compute_workgroups_per_dimension,
            packed_output_layout,
            vignette_correction,
            visible_dimensions,
            encode_gpu_stage,
        )?;

        let macroblock_count = work_plan.blocks.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
        let expected_invocations = expected_invocations_for_packed_output(macroblock_count)?;
        let gpu_total_without_wait =
            work_plan_elapsed.saturating_add(dispatch_result.timings.total);

        Ok(GpuSubmittedNoReadbackGpuStageOutput {
            stage: dispatch_result.stage,
            submission_index: dispatch_result.submission_index,
            dimensions: visible_dimensions,
            work_item_count: work_plan.blocks.len(),
            macroblock_count,
            expected_invocations,
            timings: GpuDecodeTimings::from_work_plan_dispatch(
                &work_plan,
                work_plan_elapsed,
                dispatch_result.timings,
                gpu_total_without_wait,
            ),
            vignette_stats: dispatch_result.vignette_stats,
            gpu_timestamps: dispatch_result.gpu_timestamps,
        })
    }

    /// Submit one type-7 frame using a caller-owned checked descriptor plan.
    ///
    /// The same [`GpuPreparedType7WorkPlan`] may first be passed to
    /// [`Self::prospective_prepared_type7_no_readback_slot_allocation`]. This
    /// avoids rebuilding or reallocating the CPU descriptor vector between
    /// allocation preflight and queue submission.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_prepared_raw_payload_packed_u16_gpu_stage_no_readback_slot_with_vignette<S, E>(
        &mut self,
        slot_index: usize,
        raw_payload: &[u8],
        visible_dimensions: FrameDimensions,
        prepared: GpuPreparedType7WorkPlan<'_>,
        vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
        encode_gpu_stage: E,
    ) -> Result<GpuSubmittedNoReadbackGpuStageOutput<S>>
    where
        E: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            GpuDecodedPackedU16BufferView<'_>,
        ) -> Result<S>,
    {
        anyhow::ensure!(
            prepared.work_plan.visible_dimensions == visible_dimensions,
            "prepared type-7 visible dimensions do not match submitted frame"
        );
        let pixel_count = visible_dimensions
            .pixel_count()
            .context("visible frame dimensions do not fit in usize")?;
        let packed_output_layout = GpuOutputLayout::PackedU16Samples;
        let packed_u16_pipeline = &self.packed_u16_pipeline;
        ensure_pipeline_slot_count(&mut self.pipeline_slots, slot_index + 1);
        let slot_buffers = self
            .pipeline_slots
            .get_mut(slot_index)
            .context("GPU no-readback slot was not allocated")?;

        let dispatch_result = submit_mcraw_gpu_dispatch_packed_u16_no_readback_with_gpu_stage(
            &self.device,
            &self.queue,
            packed_u16_pipeline,
            slot_buffers,
            self.timestamp_support,
            raw_payload,
            prepared.work_plan.blocks,
            pixel_count,
            visible_dimensions.width,
            visible_dimensions.height,
            self.limits.max_compute_workgroups_per_dimension,
            packed_output_layout,
            vignette_correction,
            visible_dimensions,
            encode_gpu_stage,
        )?;

        let macroblock_count =
            prepared.work_plan.blocks.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
        let expected_invocations = expected_invocations_for_packed_output(macroblock_count)?;
        let gpu_total_without_wait = prepared
            .build_elapsed
            .saturating_add(dispatch_result.timings.total);

        Ok(GpuSubmittedNoReadbackGpuStageOutput {
            stage: dispatch_result.stage,
            submission_index: dispatch_result.submission_index,
            dimensions: visible_dimensions,
            work_item_count: prepared.work_plan.blocks.len(),
            macroblock_count,
            expected_invocations,
            timings: GpuDecodeTimings::from_work_plan_ref_dispatch(
                prepared.work_plan,
                prepared.build_elapsed,
                dispatch_result.timings,
                gpu_total_without_wait,
            ),
            vignette_stats: dispatch_result.vignette_stats,
            gpu_timestamps: dispatch_result.gpu_timestamps,
        })
    }

    pub fn read_submitted_no_readback_slot_timestamps<S>(
        &self,
        slot_index: usize,
        submitted: &GpuSubmittedNoReadbackGpuStageOutput<S>,
    ) -> Result<GpuTimestampProfile> {
        let slot_buffers = self
            .pipeline_slots
            .get(slot_index)
            .context("GPU no-readback timestamp slot was not allocated")?;
        read_submitted_timestamp_profile(
            &self.device,
            slot_buffers.timestamp_recorder.as_ref(),
            submitted.submission_index.clone(),
            submitted.gpu_timestamps,
        )
    }

    // Streaming ring over the normal mapped packed-u16 readback
    // model: device-local output buffer, copy to readback buffer, map, consume,
    // unmap, then reuse the slot.
    //
    // This is intentionally separate from production decode APIs. It keeps a
    // bounded number of independent GPU/readback slots in flight and drains the
    // oldest slot as soon as the window is full, preserving input order while
    // avoiding full-output Vec allocation in the caller.
    pub fn decode_raw_payloads_packed_u16_ring_mapped<T, P, F>(
        &mut self,
        frame_count: usize,
        window: usize,
        mut next_frame: P,
        consume_mapped_pixels: F,
    ) -> Result<GpuMappedRingOutput<T>>
    where
        P: FnMut(usize) -> Result<GpuMappedRingFrame>,
        F: FnMut(usize, &[u8]) -> Result<T>,
    {
        self.decode_raw_payloads_packed_u16_ring_mapped_internal(
            frame_count,
            window,
            |index, _backend| next_frame(index),
            |frame| Ok((frame, None)),
            consume_mapped_pixels,
        )
    }

    pub fn decode_raw_payloads_packed_u16_ring_mapped_with_vignette<T, P, F>(
        &mut self,
        frame_count: usize,
        window: usize,
        next_frame: P,
        consume_mapped_pixels: F,
    ) -> Result<GpuMappedRingOutput<T>>
    where
        P: FnMut(
            usize,
            &GpuDecodeBackend,
        ) -> Result<(GpuMappedRingFrame, GpuMappedRingVignetteCorrection)>,
        F: FnMut(usize, &[u8]) -> Result<T>,
    {
        self.decode_raw_payloads_packed_u16_ring_mapped_internal(
            frame_count,
            window,
            next_frame,
            |(frame, vignette)| Ok((frame, Some(vignette))),
            consume_mapped_pixels,
        )
    }

    fn decode_raw_payloads_packed_u16_ring_mapped_internal<T, P, M, I, F>(
        &mut self,
        frame_count: usize,
        window: usize,
        mut next_input: P,
        mut map_input: M,
        mut consume_mapped_pixels: F,
    ) -> Result<GpuMappedRingOutput<T>>
    where
        P: FnMut(usize, &GpuDecodeBackend) -> Result<I>,
        M: FnMut(I) -> Result<(GpuMappedRingFrame, Option<GpuMappedRingVignetteCorrection>)>,
        F: FnMut(usize, &[u8]) -> Result<T>,
    {
        if frame_count == 0 {
            return Ok(GpuMappedRingOutput {
                values: Vec::new(),
                timings: Vec::new(),
                vignette_stats: Vec::new(),
                stats: GpuMappedRingStats::default(),
            });
        }

        if window == 0 {
            anyhow::bail!("GPU mapped readback ring window must be at least 1");
        }

        let slot_count = window.min(frame_count);
        ensure_pipeline_slot_count(&mut self.pipeline_slots, slot_count);

        let mut idle_slots: VecDeque<usize> = (0..slot_count).collect();
        let mut pending: VecDeque<PendingMappedDispatch> = VecDeque::new();
        let mut values = Vec::with_capacity(frame_count);
        let mut timings = Vec::with_capacity(frame_count);
        let mut vignette_stats = Vec::with_capacity(frame_count);
        let mut stats = GpuMappedRingStats {
            slot_count,
            ..GpuMappedRingStats::default()
        };

        for input_index in 0..frame_count {
            if idle_slots.is_empty() {
                let oldest = pending
                    .pop_front()
                    .context("ring-mapped queue was empty while no slots were idle")?;
                let slot_index = oldest.slot_index;
                let output = finish_mcraw_gpu_dispatch_mapped_in_flight(
                    &self.device,
                    &mut self.pipeline_slots[slot_index],
                    oldest,
                    &mut consume_mapped_pixels,
                )?;
                let (output, output_vignette_stats) = output;
                timings.push(output.timings);
                vignette_stats.push(output_vignette_stats);
                values.push(output.value);
                stats.completed_frames += 1;
                idle_slots.push_back(slot_index);
            }

            let input = next_input(input_index, self)
                .with_context(|| format!("failed to prepare ring-mapped frame {input_index}"))?;
            let (frame, vignette_correction) = map_input(input)?;
            let slot_index = idle_slots
                .pop_front()
                .context("no idle GPU readback ring slot was available")?;

            let work_plan_start = Instant::now();
            let work_plan = build_vulkan_work_plan(&frame.raw_payload, frame.visible_dimensions)
                .context("failed to build ring-mapped packed Vulkan GPU work plan")?;
            let work_plan_elapsed = work_plan_start.elapsed();

            let pixel_count = frame
                .visible_dimensions
                .pixel_count()
                .context("visible frame dimensions do not fit in usize")?;

            let macroblock_count =
                work_plan.blocks.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
            let packed_output_layout = GpuOutputLayout::PackedU16Samples;
            let packed_u16_pipeline = &self.packed_u16_pipeline;
            let expected_invocations = expected_invocations_for_packed_output(macroblock_count)?;

            let pending_dispatch = submit_mcraw_gpu_dispatch_mapped_in_flight(
                &self.device,
                &self.queue,
                packed_u16_pipeline,
                &mut self.pipeline_slots[slot_index],
                self.timestamp_support,
                slot_index,
                input_index,
                frame.frame_index,
                &frame.raw_payload,
                &work_plan.blocks,
                pixel_count,
                frame.visible_dimensions.width,
                frame.visible_dimensions.height,
                self.limits.max_compute_workgroups_per_dimension,
                packed_output_layout,
                work_plan_elapsed,
                GpuWorkPlanAllocationStats::from_work_plan(&work_plan),
                frame.visible_dimensions,
                work_plan.blocks.len(),
                macroblock_count,
                expected_invocations,
                vignette_correction,
            )?;

            pending.push_back(pending_dispatch);
            stats.submitted_frames += 1;
            stats.max_observed_depth = stats.max_observed_depth.max(pending.len());
        }

        while let Some(oldest) = pending.pop_front() {
            let slot_index = oldest.slot_index;
            let output = finish_mcraw_gpu_dispatch_mapped_in_flight(
                &self.device,
                &mut self.pipeline_slots[slot_index],
                oldest,
                &mut consume_mapped_pixels,
            )?;
            let (output, output_vignette_stats) = output;
            timings.push(output.timings);
            vignette_stats.push(output_vignette_stats);
            values.push(output.value);
            stats.completed_frames += 1;
            idle_slots.push_back(slot_index);
        }

        Ok(GpuMappedRingOutput {
            values,
            timings,
            vignette_stats,
            stats,
        })
    }
    // Decode a batch through a small in-flight slot ring.
    //
    // Each in-flight slot owns independent GPU buffers so later frame
    // submissions cannot overwrite resources still needed by older submissions.
    // The callback consumes mapped little-endian packed-u16 bytes before each
    // slot is unmapped and returned to the idle pool.
    pub fn decode_raw_payloads_packed_u16_mapped_in_flight<T, F>(
        &mut self,
        inputs: &[GpuMappedBatchInput<'_>],
        pipeline_depth: usize,
        mut consume_mapped_pixels: F,
    ) -> Result<Vec<GpuMappedPackedU16Output<T>>>
    where
        F: FnMut(usize, &[u8]) -> Result<T>,
    {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let slot_count = pipeline_depth.max(1).min(inputs.len());
        ensure_pipeline_slot_count(&mut self.pipeline_slots, slot_count);

        let mut idle_slots: VecDeque<usize> = (0..slot_count).collect();
        let mut pending: VecDeque<PendingMappedDispatch> = VecDeque::new();
        let mut completed: Vec<(usize, GpuMappedPackedU16Output<T>)> =
            Vec::with_capacity(inputs.len());

        for (input_index, input) in inputs.iter().enumerate() {
            if idle_slots.is_empty() {
                let oldest = pending
                    .pop_front()
                    .context("in-flight queue was empty while no slots were idle")?;
                let slot_index = oldest.slot_index;
                let oldest_input_index = oldest.input_index;
                let output = finish_mcraw_gpu_dispatch_mapped_in_flight(
                    &self.device,
                    &mut self.pipeline_slots[slot_index],
                    oldest,
                    &mut consume_mapped_pixels,
                )?;
                let (output, _) = output;
                completed.push((oldest_input_index, output));
                idle_slots.push_back(slot_index);
            }

            let slot_index = idle_slots
                .pop_front()
                .context("no idle GPU pipeline slot was available")?;

            let work_plan_start = Instant::now();
            let work_plan = build_vulkan_work_plan(input.raw_payload, input.visible_dimensions)
                .context("failed to build in-flight mapped packed Vulkan GPU work plan")?;
            let work_plan_elapsed = work_plan_start.elapsed();

            let pixel_count = input
                .visible_dimensions
                .pixel_count()
                .context("visible frame dimensions do not fit in usize")?;

            let macroblock_count =
                work_plan.blocks.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
            let packed_output_layout = GpuOutputLayout::PackedU16Samples;
            let packed_u16_pipeline = &self.packed_u16_pipeline;
            let expected_invocations = expected_invocations_for_packed_output(macroblock_count)?;

            let pending_dispatch = submit_mcraw_gpu_dispatch_mapped_in_flight(
                &self.device,
                &self.queue,
                packed_u16_pipeline,
                &mut self.pipeline_slots[slot_index],
                self.timestamp_support,
                slot_index,
                input_index,
                input.frame_index,
                input.raw_payload,
                &work_plan.blocks,
                pixel_count,
                input.visible_dimensions.width,
                input.visible_dimensions.height,
                self.limits.max_compute_workgroups_per_dimension,
                packed_output_layout,
                work_plan_elapsed,
                GpuWorkPlanAllocationStats::from_work_plan(&work_plan),
                input.visible_dimensions,
                work_plan.blocks.len(),
                macroblock_count,
                expected_invocations,
                None,
            )?;

            pending.push_back(pending_dispatch);
        }

        while let Some(oldest) = pending.pop_front() {
            let slot_index = oldest.slot_index;
            let input_index = oldest.input_index;
            let output = finish_mcraw_gpu_dispatch_mapped_in_flight(
                &self.device,
                &mut self.pipeline_slots[slot_index],
                oldest,
                &mut consume_mapped_pixels,
            )?;
            let (output, _) = output;
            completed.push((input_index, output));
            idle_slots.push_back(slot_index);
        }

        completed.sort_by_key(|(input_index, _)| *input_index);

        Ok(completed.into_iter().map(|(_, output)| output).collect())
    }
}

struct AdapterCandidate {
    adapter: wgpu::Adapter,
    info: wgpu::AdapterInfo,
    score: i32,
}

fn create_instance(preference: GpuBackendPreference) -> wgpu::Instance {
    let instance_descriptor = wgpu::InstanceDescriptor {
        backends: backends_for_preference(preference),
        ..Default::default()
    };
    wgpu::Instance::new(&instance_descriptor)
}

async fn enumerate_adapters(
    instance: &wgpu::Instance,
    preference: GpuBackendPreference,
) -> Vec<AdapterCandidate> {
    instance
        .enumerate_adapters(backends_for_preference(preference))
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            let score = score_adapter(&info);

            AdapterCandidate {
                adapter,
                info,
                score,
            }
        })
        .collect()
}

fn backends_for_preference(preference: GpuBackendPreference) -> wgpu::Backends {
    match preference {
        GpuBackendPreference::Auto => wgpu::Backends::all(),
        GpuBackendPreference::VulkanOnly => wgpu::Backends::VULKAN,
    }
}

fn score_adapter(info: &wgpu::AdapterInfo) -> i32 {
    let backend_score = match info.backend {
        wgpu::Backend::Vulkan => 10_000,
        wgpu::Backend::Metal | wgpu::Backend::Dx12 => 8_000,
        wgpu::Backend::Gl => 4_000,
        wgpu::Backend::BrowserWebGpu => 2_000,
        _ => 0,
    };

    let device_score = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 1_000,
        wgpu::DeviceType::IntegratedGpu => 700,
        wgpu::DeviceType::VirtualGpu => 400,
        wgpu::DeviceType::Cpu => 100,
        wgpu::DeviceType::Other => 0,
    };

    backend_score + device_score
}

fn select_best_adapter(mut adapters: Vec<AdapterCandidate>) -> Option<AdapterCandidate> {
    adapters.sort_by_key(|candidate| candidate.score);
    adapters.pop()
}

fn create_decode_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mcraw4vulkan MCRAW decode bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_decode_pipeline(
    device: &wgpu::Device,
    shader_label: &'static str,
    pipeline_label: &'static str,
    shader_source: &'static str,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(shader_label),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(pipeline_label),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(pipeline_label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_mcraw_gpu_dispatch_packed_u16_mapped<T, F>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: &mut ReusableGpuBuffers,
    timestamp_support: GpuTimestampSupport,
    raw_payload: &[u8],
    work_items: &[McrawVulkanBlockWorkItem],
    pixel_count: usize,
    visible_width: u32,
    visible_height: u32,
    max_compute_workgroups_per_dimension: u32,
    output_layout: GpuOutputLayout,
    consume_mapped_pixels: F,
) -> Result<GpuMappedDispatchResult<T>>
where
    F: FnOnce(&[u8]) -> Result<T>,
{
    run_mcraw_gpu_dispatch_mapped(
        device,
        queue,
        pipeline,
        buffers,
        timestamp_support,
        raw_payload,
        work_items,
        pixel_count,
        visible_width,
        visible_height,
        max_compute_workgroups_per_dimension,
        output_layout,
        consume_mapped_pixels,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_mcraw_gpu_dispatch_packed_u16_mapped_with_vignette<T, F>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: &mut ReusableGpuBuffers,
    timestamp_support: GpuTimestampSupport,
    raw_payload: &[u8],
    work_items: &[McrawVulkanBlockWorkItem],
    pixel_count: usize,
    visible_width: u32,
    visible_height: u32,
    max_compute_workgroups_per_dimension: u32,
    output_layout: GpuOutputLayout,
    vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
    consume_mapped_pixels: F,
) -> Result<GpuMappedDispatchWithVignetteResult<T>>
where
    F: FnOnce(&[u8]) -> Result<T>,
{
    run_mcraw_gpu_dispatch_mapped_with_vignette(
        device,
        queue,
        pipeline,
        buffers,
        timestamp_support,
        raw_payload,
        work_items,
        pixel_count,
        visible_width,
        visible_height,
        max_compute_workgroups_per_dimension,
        output_layout,
        vignette_correction,
        consume_mapped_pixels,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_mcraw_gpu_dispatch_mapped<T, F>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: &mut ReusableGpuBuffers,
    timestamp_support: GpuTimestampSupport,
    raw_payload: &[u8],
    work_items: &[McrawVulkanBlockWorkItem],
    pixel_count: usize,
    visible_width: u32,
    visible_height: u32,
    max_compute_workgroups_per_dimension: u32,
    output_layout: GpuOutputLayout,
    consume_mapped_pixels: F,
) -> Result<GpuMappedDispatchResult<T>>
where
    F: FnOnce(&[u8]) -> Result<T>,
{
    let result = run_mcraw_gpu_dispatch_mapped_with_vignette(
        device,
        queue,
        pipeline,
        buffers,
        timestamp_support,
        raw_payload,
        work_items,
        pixel_count,
        visible_width,
        visible_height,
        max_compute_workgroups_per_dimension,
        output_layout,
        None,
        consume_mapped_pixels,
    )?;

    Ok(GpuMappedDispatchResult {
        value: result.value,
        timings: result.timings,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_mcraw_gpu_dispatch_mapped_with_vignette<T, F>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: &mut ReusableGpuBuffers,
    timestamp_support: GpuTimestampSupport,
    raw_payload: &[u8],
    work_items: &[McrawVulkanBlockWorkItem],
    pixel_count: usize,
    visible_width: u32,
    visible_height: u32,
    max_compute_workgroups_per_dimension: u32,
    output_layout: GpuOutputLayout,
    vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
    consume_mapped_pixels: F,
) -> Result<GpuMappedDispatchWithVignetteResult<T>>
where
    F: FnOnce(&[u8]) -> Result<T>,
{
    let total_start = Instant::now();

    let cpu_prepare_start = Instant::now();

    let cpu_staging_serialize_start = Instant::now();
    let upload_byte_counts = prepare_upload_staging(raw_payload, work_items, buffers)?;
    let cpu_staging_serialize_elapsed = cpu_staging_serialize_start.elapsed();

    let raw_payload_byte_len = upload_byte_counts.raw_payload_byte_len;
    let work_item_byte_len = upload_byte_counts.work_item_byte_len;
    let output_byte_len = byte_len_for_output_layout(pixel_count, output_layout)?;

    if !has_complete_macroblocks(work_items.len()) {
        anyhow::bail!(
            "work item count is not divisible by descriptors per macroblock: work_items={}, descriptors_per_macroblock={}",
            work_items.len(),
            MCRAW_DESCRIPTORS_PER_MACROBLOCK
        );
    }

    let macroblock_count = work_items.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
    let (dispatch_width, dispatch_height) =
        dispatch_grid(macroblock_count, max_compute_workgroups_per_dimension)?;

    let params_bytes = mcraw_params_to_le_bytes(
        visible_width,
        visible_height,
        u32::try_from(work_items.len()).context("work item count does not fit in u32")?,
        u32::try_from(macroblock_count).context("macroblock count does not fit in u32")?,
        dispatch_width,
    );

    let params_byte_len =
        u64::try_from(params_bytes.len()).context("params byte length overflow")?;

    let buffer_ensure_start = Instant::now();
    let bind_group_buffers_changed = buffers.ensure_for_decode(
        device,
        raw_payload_byte_len,
        work_item_byte_len,
        output_byte_len,
        params_byte_len,
    );
    let buffer_ensure_elapsed = buffer_ensure_start.elapsed();

    let cpu_prepare_elapsed = cpu_prepare_start.elapsed();

    let upload_timings = upload_decode_inputs(queue, buffers, raw_payload, &params_bytes)?;

    let encode_submit_start = Instant::now();

    let bind_group_start = Instant::now();
    if bind_group_buffers_changed || buffers.bind_group.is_none() {
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let raw_payload_buffer = &buffers
            .raw_payload
            .as_ref()
            .context("raw payload GPU buffer was not allocated")?
            .buffer;
        let work_items_buffer = &buffers
            .work_items
            .as_ref()
            .context("work item GPU buffer was not allocated")?
            .buffer;
        let output_buffer = &buffers
            .output
            .as_ref()
            .context("output GPU buffer was not allocated")?
            .buffer;
        let params_buffer = &buffers
            .params
            .as_ref()
            .context("params GPU buffer was not allocated")?
            .buffer;

        buffers.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MCRAW reusable GPU decode bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: raw_payload_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: work_items_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        }));
    }
    let bind_group_elapsed = bind_group_start.elapsed();

    let bind_group = buffers
        .bind_group
        .as_ref()
        .context("GPU bind group was not allocated")?;
    let output_buffer = &buffers
        .output
        .as_ref()
        .context("output GPU buffer was not allocated")?
        .buffer;
    let readback_buffer = &buffers
        .readback
        .as_ref()
        .context("readback GPU buffer was not allocated")?
        .buffer;

    let command_encoder_create_start = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MCRAW GPU decode command encoder"),
    });
    let command_encoder_create_elapsed = command_encoder_create_start.elapsed();
    let timestamp_recorder =
        GpuTimestampRecorder::new(device, timestamp_support, "MCRAW GPU decode timestamps");
    let output_clear_enabled = output_layout.output_clear_enabled();
    let mut timestamp_stages = GpuTimestampStageMask {
        output_clear: output_clear_enabled,
        vignette: false,
    };

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::TotalStart);
        timestamps.write(&mut encoder, GpuTimestampQuery::OutputClearStart);
    }

    // The shader assembles each packed word with atomicOr, which cannot clear
    // bits retained by a reused output buffer. Zero it before every dispatch.
    let mut output_clear_encode_elapsed = Duration::ZERO;
    if output_clear_enabled {
        let output_clear_encode_start = Instant::now();
        encoder.clear_buffer(output_buffer, 0, Some(output_byte_len));
        output_clear_encode_elapsed = output_clear_encode_start.elapsed();
    }

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::OutputClearEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::DecodeDispatchStart);
    }

    let decode_pass_encode_start = Instant::now();
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MCRAW GPU decode compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
    }
    let decode_pass_encode_elapsed = decode_pass_encode_start.elapsed();

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::DecodeDispatchEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchStart);
    }

    let mut vignette_stats = None;
    let mut vignette_encode_elapsed = Duration::ZERO;
    let copy_to_readback_encode_elapsed;
    if let Some(correction) = vignette_correction {
        let vignette_encode_start = Instant::now();
        let dispatch = correction
            .corrector
            .dispatch_packed_u16(GpuVignettePackedU16DispatchInput {
                device,
                queue,
                encoder: &mut encoder,
                input_buffer: output_buffer,
                input_buffer_bytes: output_byte_len,
                uploaded_gain_map: correction.uploaded_gain_map,
                params: correction.params,
            })
            .context("failed to dispatch GPU vignette correction")?;
        vignette_encode_elapsed = vignette_encode_start.elapsed();
        timestamp_stages.vignette = dispatch.stats.dispatch_submitted;
        if let Some(timestamps) = timestamp_recorder.as_ref() {
            timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchEnd);
            timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackStart);
        }
        let copy_to_readback_encode_start = Instant::now();
        encoder.copy_buffer_to_buffer(
            dispatch.output.buffer(),
            0,
            readback_buffer,
            0,
            output_byte_len,
        );
        copy_to_readback_encode_elapsed = copy_to_readback_encode_start.elapsed();
        vignette_stats = Some(dispatch.stats);
    } else {
        if let Some(timestamps) = timestamp_recorder.as_ref() {
            timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchEnd);
            timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackStart);
        }
        let copy_to_readback_encode_start = Instant::now();
        encoder.copy_buffer_to_buffer(output_buffer, 0, readback_buffer, 0, output_byte_len);
        copy_to_readback_encode_elapsed = copy_to_readback_encode_start.elapsed();
    }

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::TotalEnd);
        timestamps.resolve(&mut encoder);
    }

    let command_finish_submit_start = Instant::now();
    let submission_index = queue.submit(Some(encoder.finish()));
    let command_finish_submit_elapsed = command_finish_submit_start.elapsed();
    let readback_slice = readback_buffer.slice(0..output_byte_len);

    let encode_submit_elapsed = encode_submit_start.elapsed();

    let wait_map_start = Instant::now();

    let (sender, receiver) = mpsc::channel();
    readback_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device.poll(wgpu::Maintain::wait_for(submission_index.clone()));

    let map_result = receiver
        .recv_timeout(Duration::from_secs(30))
        .context("timed out waiting for MCRAW readback buffer mapping callback")?;

    map_result.map_err(|error| anyhow!("failed to map MCRAW readback buffer: {error:?}"))?;

    let wait_map_elapsed = wait_map_start.elapsed();

    // The dispatch/readback timing stops when the mapped bytes are available.
    // The caller may spend additional time consuming those bytes, such as
    // building a DNG, and should account for that separately.
    let dispatch_readback_elapsed = total_start.elapsed();

    let mapped_consumer_start = Instant::now();
    let consume_result = {
        let mapped = readback_slice.get_mapped_range();
        let value = consume_mapped_pixels(&mapped);
        drop(mapped);
        value
    };
    let mapped_consumer_elapsed = mapped_consumer_start.elapsed();

    readback_buffer.unmap();

    let value = consume_result?;
    let gpu_timestamps = if let Some(timestamps) = timestamp_recorder {
        timestamps.read_profile(device, submission_index, timestamp_stages)?
    } else {
        GpuTimestampProfile::unavailable(timestamp_support)
    };

    Ok(GpuMappedDispatchWithVignetteResult {
        value,
        timings: GpuDispatchTimings {
            raw_payload_bytes: raw_payload_byte_len,
            work_item_count: work_items.len() as u64,
            work_item_bytes: work_item_byte_len,
            params_bytes: params_byte_len,
            output_bytes: output_byte_len,
            readback_bytes: output_byte_len,
            work_item_buffer_reused: upload_byte_counts.work_item_buffer_reused,
            work_item_buffer_capacity: upload_byte_counts.work_item_buffer_capacity,
            work_item_buffer_reallocs: upload_byte_counts.work_item_buffer_reallocs,
            write_buffer_with_work_items: upload_timings.write_buffer_with_work_items,
            write_buffer_with_params: upload_timings.write_buffer_with_params,
            direct_staging_fallback: upload_timings.direct_staging_fallback,
            queue_write_buffer_calls: upload_timings.queue_write_buffer_calls,
            queue_write_buffer_with_calls: upload_timings.queue_write_buffer_with_calls,
            cpu_prepare: cpu_prepare_elapsed,
            buffer_ensure: buffer_ensure_elapsed,
            cpu_staging_serialize: cpu_staging_serialize_elapsed,
            upload: upload_timings.total,
            raw_payload_upload: upload_timings.raw_payload_upload,
            work_items_upload: upload_timings.work_items_upload,
            params_upload: upload_timings.params_upload,
            encode_submit: encode_submit_elapsed,
            bind_group: bind_group_elapsed,
            command_encoder_create: command_encoder_create_elapsed,
            output_clear_encode: output_clear_encode_elapsed,
            decode_pass_encode: decode_pass_encode_elapsed,
            vignette_encode: vignette_encode_elapsed,
            copy_to_readback_encode: copy_to_readback_encode_elapsed,
            command_finish_submit: command_finish_submit_elapsed,
            wait_map: wait_map_elapsed,
            mapped_consumer: mapped_consumer_elapsed,
            readback_convert: Duration::ZERO,
            total: dispatch_readback_elapsed,
            gpu_timestamps,
        },
        vignette_stats,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_legacy_raw16_gpu_dispatch_mapped_with_vignette<T, F>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: &mut ReusableGpuBuffers,
    timestamp_support: GpuTimestampSupport,
    raw_payload: &[u8],
    work_items: &[McrawVulkanBlockWorkItem],
    chunk_count: usize,
    pixel_count: usize,
    visible_width: u32,
    visible_height: u32,
    max_compute_workgroups_per_dimension: u32,
    output_layout: GpuOutputLayout,
    vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
    consume_mapped_pixels: F,
) -> Result<GpuMappedDispatchWithVignetteResult<T>>
where
    F: FnOnce(&[u8]) -> Result<T>,
{
    let total_start = Instant::now();

    let cpu_prepare_start = Instant::now();
    let cpu_staging_serialize_start = Instant::now();
    let upload_byte_counts = prepare_upload_staging(raw_payload, work_items, buffers)?;
    let cpu_staging_serialize_elapsed = cpu_staging_serialize_start.elapsed();

    let raw_payload_byte_len = upload_byte_counts.raw_payload_byte_len;
    let work_item_byte_len = upload_byte_counts.work_item_byte_len;
    let output_byte_len = byte_len_for_output_layout(pixel_count, output_layout)?;

    anyhow::ensure!(
        work_items.len() == chunk_count.saturating_mul(2),
        "legacy raw16 work item/chunk mismatch: work_items={} chunk_count={}",
        work_items.len(),
        chunk_count
    );

    let (dispatch_width, dispatch_height) =
        legacy_raw16_dispatch_grid(chunk_count, max_compute_workgroups_per_dimension)?;
    let params_bytes = legacy_raw16_params_to_le_bytes(
        visible_width,
        visible_height,
        u32::try_from(work_items.len())
            .context("legacy raw16 work item count does not fit in u32")?,
        u32::try_from(chunk_count).context("legacy raw16 chunk count does not fit in u32")?,
        dispatch_width,
    );
    let params_byte_len =
        u64::try_from(params_bytes.len()).context("params byte length overflow")?;

    let buffer_ensure_start = Instant::now();
    let bind_group_buffers_changed = buffers.ensure_for_decode(
        device,
        raw_payload_byte_len,
        work_item_byte_len,
        output_byte_len,
        params_byte_len,
    );
    let buffer_ensure_elapsed = buffer_ensure_start.elapsed();
    let cpu_prepare_elapsed = cpu_prepare_start.elapsed();

    let upload_timings = upload_decode_inputs(queue, buffers, raw_payload, &params_bytes)?;

    let encode_submit_start = Instant::now();

    let bind_group_start = Instant::now();
    if bind_group_buffers_changed || buffers.bind_group.is_none() {
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let raw_payload_buffer = &buffers
            .raw_payload
            .as_ref()
            .context("raw payload GPU buffer was not allocated")?
            .buffer;
        let work_items_buffer = &buffers
            .work_items
            .as_ref()
            .context("work item GPU buffer was not allocated")?
            .buffer;
        let output_buffer = &buffers
            .output
            .as_ref()
            .context("output GPU buffer was not allocated")?
            .buffer;
        let params_buffer = &buffers
            .params
            .as_ref()
            .context("params GPU buffer was not allocated")?
            .buffer;

        buffers.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MCRAW legacy raw16 GPU decode bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: raw_payload_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: work_items_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        }));
    }
    let bind_group_elapsed = bind_group_start.elapsed();

    let bind_group = buffers
        .bind_group
        .as_ref()
        .context("GPU bind group was not allocated")?;
    let output_buffer = &buffers
        .output
        .as_ref()
        .context("output GPU buffer was not allocated")?
        .buffer;
    let readback_buffer = &buffers
        .readback
        .as_ref()
        .context("readback GPU buffer was not allocated")?
        .buffer;

    let command_encoder_create_start = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MCRAW legacy raw16 GPU decode command encoder"),
    });
    let command_encoder_create_elapsed = command_encoder_create_start.elapsed();
    let timestamp_recorder = GpuTimestampRecorder::new(
        device,
        timestamp_support,
        "MCRAW legacy raw16 GPU decode timestamps",
    );
    let output_clear_enabled = output_layout.output_clear_enabled();
    let mut timestamp_stages = GpuTimestampStageMask {
        output_clear: output_clear_enabled,
        vignette: false,
    };

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::TotalStart);
        timestamps.write(&mut encoder, GpuTimestampQuery::OutputClearStart);
    }

    let mut output_clear_encode_elapsed = Duration::ZERO;
    if output_clear_enabled {
        let output_clear_encode_start = Instant::now();
        encoder.clear_buffer(output_buffer, 0, Some(output_byte_len));
        output_clear_encode_elapsed = output_clear_encode_start.elapsed();
    }

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::OutputClearEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::DecodeDispatchStart);
    }

    let decode_pass_encode_start = Instant::now();
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MCRAW legacy raw16 GPU decode compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
    }
    let decode_pass_encode_elapsed = decode_pass_encode_start.elapsed();

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::DecodeDispatchEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchStart);
    }

    let mut vignette_stats = None;
    let mut vignette_encode_elapsed = Duration::ZERO;
    let copy_to_readback_encode_elapsed;
    if let Some(correction) = vignette_correction {
        let vignette_encode_start = Instant::now();
        let dispatch = correction
            .corrector
            .dispatch_packed_u16(GpuVignettePackedU16DispatchInput {
                device,
                queue,
                encoder: &mut encoder,
                input_buffer: output_buffer,
                input_buffer_bytes: output_byte_len,
                uploaded_gain_map: correction.uploaded_gain_map,
                params: correction.params,
            })
            .context("failed to dispatch GPU vignette correction for legacy raw16")?;
        vignette_encode_elapsed = vignette_encode_start.elapsed();
        timestamp_stages.vignette = dispatch.stats.dispatch_submitted;
        if let Some(timestamps) = timestamp_recorder.as_ref() {
            timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchEnd);
            timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackStart);
        }
        let copy_to_readback_encode_start = Instant::now();
        encoder.copy_buffer_to_buffer(
            dispatch.output.buffer(),
            0,
            readback_buffer,
            0,
            output_byte_len,
        );
        copy_to_readback_encode_elapsed = copy_to_readback_encode_start.elapsed();
        vignette_stats = Some(dispatch.stats);
    } else {
        if let Some(timestamps) = timestamp_recorder.as_ref() {
            timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchEnd);
            timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackStart);
        }
        let copy_to_readback_encode_start = Instant::now();
        encoder.copy_buffer_to_buffer(output_buffer, 0, readback_buffer, 0, output_byte_len);
        copy_to_readback_encode_elapsed = copy_to_readback_encode_start.elapsed();
    }

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::TotalEnd);
        timestamps.resolve(&mut encoder);
    }

    let command_finish_submit_start = Instant::now();
    let submission_index = queue.submit(Some(encoder.finish()));
    let command_finish_submit_elapsed = command_finish_submit_start.elapsed();
    let readback_slice = readback_buffer.slice(0..output_byte_len);
    let encode_submit_elapsed = encode_submit_start.elapsed();

    let wait_map_start = Instant::now();
    let (sender, receiver) = mpsc::channel();
    readback_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::wait_for(submission_index.clone()));
    let map_result = receiver
        .recv_timeout(Duration::from_secs(30))
        .context("timed out waiting for legacy raw16 GPU readback mapping callback")?;
    map_result.map_err(|error| anyhow!("failed to map legacy raw16 GPU readback: {error:?}"))?;
    let wait_map_elapsed = wait_map_start.elapsed();
    let dispatch_readback_elapsed = total_start.elapsed();

    let mapped_consumer_start = Instant::now();
    let consume_result = {
        let mapped = readback_slice.get_mapped_range();
        let value = consume_mapped_pixels(&mapped);
        drop(mapped);
        value
    };
    let mapped_consumer_elapsed = mapped_consumer_start.elapsed();

    readback_buffer.unmap();

    let value = consume_result?;
    let gpu_timestamps = if let Some(timestamps) = timestamp_recorder {
        timestamps.read_profile(device, submission_index, timestamp_stages)?
    } else {
        GpuTimestampProfile::unavailable(timestamp_support)
    };

    Ok(GpuMappedDispatchWithVignetteResult {
        value,
        timings: GpuDispatchTimings {
            raw_payload_bytes: raw_payload_byte_len,
            work_item_count: work_items.len() as u64,
            work_item_bytes: work_item_byte_len,
            params_bytes: params_byte_len,
            output_bytes: output_byte_len,
            readback_bytes: output_byte_len,
            work_item_buffer_reused: upload_byte_counts.work_item_buffer_reused,
            work_item_buffer_capacity: upload_byte_counts.work_item_buffer_capacity,
            work_item_buffer_reallocs: upload_byte_counts.work_item_buffer_reallocs,
            write_buffer_with_work_items: upload_timings.write_buffer_with_work_items,
            write_buffer_with_params: upload_timings.write_buffer_with_params,
            direct_staging_fallback: upload_timings.direct_staging_fallback,
            queue_write_buffer_calls: upload_timings.queue_write_buffer_calls,
            queue_write_buffer_with_calls: upload_timings.queue_write_buffer_with_calls,
            cpu_prepare: cpu_prepare_elapsed,
            buffer_ensure: buffer_ensure_elapsed,
            cpu_staging_serialize: cpu_staging_serialize_elapsed,
            upload: upload_timings.total,
            raw_payload_upload: upload_timings.raw_payload_upload,
            work_items_upload: upload_timings.work_items_upload,
            params_upload: upload_timings.params_upload,
            encode_submit: encode_submit_elapsed,
            bind_group: bind_group_elapsed,
            command_encoder_create: command_encoder_create_elapsed,
            output_clear_encode: output_clear_encode_elapsed,
            decode_pass_encode: decode_pass_encode_elapsed,
            vignette_encode: vignette_encode_elapsed,
            copy_to_readback_encode: copy_to_readback_encode_elapsed,
            command_finish_submit: command_finish_submit_elapsed,
            wait_map: wait_map_elapsed,
            mapped_consumer: mapped_consumer_elapsed,
            readback_convert: Duration::ZERO,
            total: dispatch_readback_elapsed,
            gpu_timestamps,
        },
        vignette_stats,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_mcraw_gpu_dispatch_packed_u16_mapped_with_gpu_stage<T, S, E, C>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: &mut ReusableGpuBuffers,
    timestamp_support: GpuTimestampSupport,
    raw_payload: &[u8],
    work_items: &[McrawVulkanBlockWorkItem],
    pixel_count: usize,
    visible_width: u32,
    visible_height: u32,
    max_compute_workgroups_per_dimension: u32,
    output_layout: GpuOutputLayout,
    vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
    visible_dimensions: FrameDimensions,
    encode_gpu_stage: E,
    consume_mapped_readback: C,
) -> Result<GpuMappedGpuStageDispatchWithVignetteResult<T, S>>
where
    E: FnOnce(
        &wgpu::Device,
        &wgpu::Queue,
        &mut wgpu::CommandEncoder,
        GpuDecodedPackedU16BufferView<'_>,
    ) -> Result<(S, GpuExternalReadbackBuffer)>,
    C: FnOnce(&[u8]) -> Result<T>,
{
    let total_start = Instant::now();

    let cpu_prepare_start = Instant::now();

    let cpu_staging_serialize_start = Instant::now();
    let upload_byte_counts = prepare_upload_staging(raw_payload, work_items, buffers)?;
    let cpu_staging_serialize_elapsed = cpu_staging_serialize_start.elapsed();

    let raw_payload_byte_len = upload_byte_counts.raw_payload_byte_len;
    let work_item_byte_len = upload_byte_counts.work_item_byte_len;
    let decode_output_byte_len = byte_len_for_output_layout(pixel_count, output_layout)?;

    if !has_complete_macroblocks(work_items.len()) {
        anyhow::bail!(
            "work item count is not divisible by descriptors per macroblock: work_items={}, descriptors_per_macroblock={}",
            work_items.len(),
            MCRAW_DESCRIPTORS_PER_MACROBLOCK
        );
    }

    let macroblock_count = work_items.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
    let (dispatch_width, dispatch_height) =
        dispatch_grid(macroblock_count, max_compute_workgroups_per_dimension)?;

    let params_bytes = mcraw_params_to_le_bytes(
        visible_width,
        visible_height,
        u32::try_from(work_items.len()).context("work item count does not fit in u32")?,
        u32::try_from(macroblock_count).context("macroblock count does not fit in u32")?,
        dispatch_width,
    );

    let params_byte_len =
        u64::try_from(params_bytes.len()).context("params byte length overflow")?;

    let buffer_ensure_start = Instant::now();
    let bind_group_buffers_changed = buffers.ensure_for_decode(
        device,
        raw_payload_byte_len,
        work_item_byte_len,
        decode_output_byte_len,
        params_byte_len,
    );
    let buffer_ensure_elapsed = buffer_ensure_start.elapsed();

    let cpu_prepare_elapsed = cpu_prepare_start.elapsed();

    let upload_timings = upload_decode_inputs(queue, buffers, raw_payload, &params_bytes)?;

    let encode_submit_start = Instant::now();

    let bind_group_start = Instant::now();
    if bind_group_buffers_changed || buffers.bind_group.is_none() {
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let raw_payload_buffer = &buffers
            .raw_payload
            .as_ref()
            .context("raw payload GPU buffer was not allocated")?
            .buffer;
        let work_items_buffer = &buffers
            .work_items
            .as_ref()
            .context("work item GPU buffer was not allocated")?
            .buffer;
        let output_buffer = &buffers
            .output
            .as_ref()
            .context("output GPU buffer was not allocated")?
            .buffer;
        let params_buffer = &buffers
            .params
            .as_ref()
            .context("params GPU buffer was not allocated")?
            .buffer;

        buffers.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MCRAW GPU-stage decode bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: raw_payload_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: work_items_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        }));
    }
    let bind_group_elapsed = bind_group_start.elapsed();

    let bind_group = buffers
        .bind_group
        .as_ref()
        .context("GPU bind group was not allocated")?;
    let output_buffer = &buffers
        .output
        .as_ref()
        .context("output GPU buffer was not allocated")?
        .buffer;

    let command_encoder_create_start = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MCRAW GPU decode/stage command encoder"),
    });
    let command_encoder_create_elapsed = command_encoder_create_start.elapsed();
    let timestamp_recorder = GpuTimestampRecorder::new(
        device,
        timestamp_support,
        "MCRAW GPU decode/stage timestamps",
    );
    let output_clear_enabled = output_layout.output_clear_enabled();
    let mut timestamp_stages = GpuTimestampStageMask {
        output_clear: output_clear_enabled,
        vignette: false,
    };

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::TotalStart);
        timestamps.write(&mut encoder, GpuTimestampQuery::OutputClearStart);
    }

    let mut decode_output_clear_encode_elapsed = Duration::ZERO;
    if output_clear_enabled {
        let output_clear_encode_start = Instant::now();
        encoder.clear_buffer(output_buffer, 0, Some(decode_output_byte_len));
        decode_output_clear_encode_elapsed = output_clear_encode_start.elapsed();
    }

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::OutputClearEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::DecodeDispatchStart);
    }

    let decode_pass_encode_start = Instant::now();
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MCRAW GPU-stage decode compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
    }
    let decode_pass_encode_elapsed = decode_pass_encode_start.elapsed();

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::DecodeDispatchEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchStart);
    }

    let mut vignette_stats = None;
    let mut vignette_encode_elapsed = Duration::ZERO;
    let (stage, stage_readback) = if let Some(correction) = vignette_correction {
        let vignette_encode_start = Instant::now();
        let dispatch = correction
            .corrector
            .dispatch_packed_u16(GpuVignettePackedU16DispatchInput {
                device,
                queue,
                encoder: &mut encoder,
                input_buffer: output_buffer,
                input_buffer_bytes: decode_output_byte_len,
                uploaded_gain_map: correction.uploaded_gain_map,
                params: correction.params,
            })
            .context("failed to dispatch GPU vignette correction before downstream GPU stage")?;
        vignette_encode_elapsed = vignette_encode_start.elapsed();
        timestamp_stages.vignette = dispatch.stats.dispatch_submitted;
        if let Some(timestamps) = timestamp_recorder.as_ref() {
            timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchEnd);
            timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackStart);
        }
        vignette_stats = Some(dispatch.stats);
        encode_gpu_stage(
            device,
            queue,
            &mut encoder,
            GpuDecodedPackedU16BufferView {
                buffer: dispatch.output.buffer(),
                byte_len: decode_output_byte_len,
                dimensions: visible_dimensions,
                pixel_count,
            },
        )?
    } else {
        if let Some(timestamps) = timestamp_recorder.as_ref() {
            timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchEnd);
            timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackStart);
        }
        encode_gpu_stage(
            device,
            queue,
            &mut encoder,
            GpuDecodedPackedU16BufferView {
                buffer: output_buffer,
                byte_len: decode_output_byte_len,
                dimensions: visible_dimensions,
                pixel_count,
            },
        )?
    };

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::TotalEnd);
        timestamps.resolve(&mut encoder);
    }

    let command_finish_submit_start = Instant::now();
    let submission_index = queue.submit(Some(encoder.finish()));
    let command_finish_submit_elapsed = command_finish_submit_start.elapsed();
    let encode_submit_elapsed = encode_submit_start.elapsed();

    if stage_readback.valid_byte_len > stage_readback.mapped_byte_len {
        anyhow::bail!(
            "GPU-stage readback valid byte length {} exceeds mapped byte length {}",
            stage_readback.valid_byte_len,
            stage_readback.mapped_byte_len
        );
    }
    let readback_slice = stage_readback
        .buffer
        .slice(0..stage_readback.mapped_byte_len);

    let wait_map_start = Instant::now();
    let (sender, receiver) = mpsc::channel();
    readback_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device.poll(wgpu::Maintain::wait_for(submission_index.clone()));

    let map_result = receiver
        .recv_timeout(Duration::from_secs(30))
        .context("timed out waiting for MCRAW GPU-stage readback buffer mapping callback")?;

    map_result
        .map_err(|error| anyhow!("failed to map MCRAW GPU-stage readback buffer: {error:?}"))?;
    let wait_map_elapsed = wait_map_start.elapsed();

    let dispatch_readback_elapsed = total_start.elapsed();

    let mapped_consumer_start = Instant::now();
    let mapped_len = usize::try_from(stage_readback.valid_byte_len)
        .context("mapped valid byte length does not fit usize")?;
    let consume_result = {
        let mapped = readback_slice.get_mapped_range();
        let value = consume_mapped_readback(&mapped[..mapped_len]);
        drop(mapped);
        value
    };
    let mapped_consumer_elapsed = mapped_consumer_start.elapsed();

    stage_readback.buffer.unmap();

    let value = consume_result?;
    let gpu_timestamps = if let Some(timestamps) = timestamp_recorder {
        timestamps.read_profile(device, submission_index, timestamp_stages)?
    } else {
        GpuTimestampProfile::unavailable(timestamp_support)
    };

    Ok(GpuMappedGpuStageDispatchWithVignetteResult {
        value,
        stage,
        timings: GpuDispatchTimings {
            raw_payload_bytes: raw_payload_byte_len,
            work_item_count: work_items.len() as u64,
            work_item_bytes: work_item_byte_len,
            params_bytes: params_byte_len,
            output_bytes: stage_readback.valid_byte_len,
            readback_bytes: stage_readback.mapped_byte_len,
            work_item_buffer_reused: upload_byte_counts.work_item_buffer_reused,
            work_item_buffer_capacity: upload_byte_counts.work_item_buffer_capacity,
            work_item_buffer_reallocs: upload_byte_counts.work_item_buffer_reallocs,
            write_buffer_with_work_items: upload_timings.write_buffer_with_work_items,
            write_buffer_with_params: upload_timings.write_buffer_with_params,
            direct_staging_fallback: upload_timings.direct_staging_fallback,
            queue_write_buffer_calls: upload_timings.queue_write_buffer_calls,
            queue_write_buffer_with_calls: upload_timings.queue_write_buffer_with_calls,
            cpu_prepare: cpu_prepare_elapsed,
            buffer_ensure: buffer_ensure_elapsed,
            cpu_staging_serialize: cpu_staging_serialize_elapsed,
            upload: upload_timings.total,
            raw_payload_upload: upload_timings.raw_payload_upload,
            work_items_upload: upload_timings.work_items_upload,
            params_upload: upload_timings.params_upload,
            encode_submit: encode_submit_elapsed,
            bind_group: bind_group_elapsed,
            command_encoder_create: command_encoder_create_elapsed,
            output_clear_encode: decode_output_clear_encode_elapsed,
            decode_pass_encode: decode_pass_encode_elapsed,
            vignette_encode: vignette_encode_elapsed,
            copy_to_readback_encode: Duration::ZERO,
            command_finish_submit: command_finish_submit_elapsed,
            wait_map: wait_map_elapsed,
            mapped_consumer: mapped_consumer_elapsed,
            readback_convert: Duration::ZERO,
            total: dispatch_readback_elapsed,
            gpu_timestamps,
        },
        vignette_stats,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_mcraw_gpu_dispatch_packed_u16_no_readback_with_gpu_stage<S, E>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: &mut ReusableGpuBuffers,
    timestamp_support: GpuTimestampSupport,
    raw_payload: &[u8],
    work_items: &[McrawVulkanBlockWorkItem],
    pixel_count: usize,
    visible_width: u32,
    visible_height: u32,
    max_compute_workgroups_per_dimension: u32,
    output_layout: GpuOutputLayout,
    vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
    visible_dimensions: FrameDimensions,
    encode_gpu_stage: E,
) -> Result<GpuNoReadbackGpuStageDispatchWithVignetteResult<S>>
where
    E: FnOnce(
        &wgpu::Device,
        &wgpu::Queue,
        &mut wgpu::CommandEncoder,
        GpuDecodedPackedU16BufferView<'_>,
    ) -> Result<S>,
{
    let pending = submit_mcraw_gpu_dispatch_packed_u16_no_readback_with_gpu_stage(
        device,
        queue,
        pipeline,
        buffers,
        timestamp_support,
        raw_payload,
        work_items,
        pixel_count,
        visible_width,
        visible_height,
        max_compute_workgroups_per_dimension,
        output_layout,
        vignette_correction,
        visible_dimensions,
        encode_gpu_stage,
    )?;
    wait_mcraw_gpu_dispatch_packed_u16_no_readback_with_gpu_stage(device, buffers, pending)
}

#[allow(clippy::too_many_arguments)]
fn run_legacy_raw16_gpu_dispatch_no_readback_with_gpu_stage<S, E>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: &mut ReusableGpuBuffers,
    timestamp_support: GpuTimestampSupport,
    raw_payload: &[u8],
    work_items: &[McrawVulkanBlockWorkItem],
    chunk_count: usize,
    pixel_count: usize,
    visible_width: u32,
    visible_height: u32,
    max_compute_workgroups_per_dimension: u32,
    output_layout: GpuOutputLayout,
    vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
    visible_dimensions: FrameDimensions,
    encode_gpu_stage: E,
) -> Result<GpuNoReadbackGpuStageDispatchWithVignetteResult<S>>
where
    E: FnOnce(
        &wgpu::Device,
        &wgpu::Queue,
        &mut wgpu::CommandEncoder,
        GpuDecodedPackedU16BufferView<'_>,
    ) -> Result<S>,
{
    let pending = submit_legacy_raw16_gpu_dispatch_no_readback_with_gpu_stage(
        device,
        queue,
        pipeline,
        buffers,
        timestamp_support,
        raw_payload,
        work_items,
        chunk_count,
        pixel_count,
        visible_width,
        visible_height,
        max_compute_workgroups_per_dimension,
        output_layout,
        vignette_correction,
        visible_dimensions,
        encode_gpu_stage,
    )?;
    wait_mcraw_gpu_dispatch_packed_u16_no_readback_with_gpu_stage(device, buffers, pending)
}

#[allow(clippy::too_many_arguments)]
fn submit_legacy_raw16_gpu_dispatch_no_readback_with_gpu_stage<S, E>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: &mut ReusableGpuBuffers,
    timestamp_support: GpuTimestampSupport,
    raw_payload: &[u8],
    work_items: &[McrawVulkanBlockWorkItem],
    chunk_count: usize,
    pixel_count: usize,
    visible_width: u32,
    visible_height: u32,
    max_compute_workgroups_per_dimension: u32,
    output_layout: GpuOutputLayout,
    vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
    visible_dimensions: FrameDimensions,
    encode_gpu_stage: E,
) -> Result<PendingNoReadbackGpuStageDispatchWithVignetteResult<S>>
where
    E: FnOnce(
        &wgpu::Device,
        &wgpu::Queue,
        &mut wgpu::CommandEncoder,
        GpuDecodedPackedU16BufferView<'_>,
    ) -> Result<S>,
{
    let total_start = Instant::now();

    let cpu_prepare_start = Instant::now();
    let cpu_staging_serialize_start = Instant::now();
    let upload_byte_counts = prepare_upload_staging(raw_payload, work_items, buffers)?;
    let cpu_staging_serialize_elapsed = cpu_staging_serialize_start.elapsed();

    let raw_payload_byte_len = upload_byte_counts.raw_payload_byte_len;
    let work_item_byte_len = upload_byte_counts.work_item_byte_len;
    let decode_output_byte_len = byte_len_for_output_layout(pixel_count, output_layout)?;

    anyhow::ensure!(
        work_items.len() == chunk_count.saturating_mul(2),
        "legacy raw16 work item/chunk mismatch: work_items={} chunk_count={}",
        work_items.len(),
        chunk_count
    );

    let (dispatch_width, dispatch_height) =
        legacy_raw16_dispatch_grid(chunk_count, max_compute_workgroups_per_dimension)?;
    let params_bytes = legacy_raw16_params_to_le_bytes(
        visible_width,
        visible_height,
        u32::try_from(work_items.len())
            .context("legacy raw16 work item count does not fit in u32")?,
        u32::try_from(chunk_count).context("legacy raw16 chunk count does not fit in u32")?,
        dispatch_width,
    );
    let params_byte_len =
        u64::try_from(params_bytes.len()).context("params byte length overflow")?;

    let buffer_ensure_start = Instant::now();
    let bind_group_buffers_changed = buffers.ensure_for_decode_without_readback(
        device,
        raw_payload_byte_len,
        work_item_byte_len,
        decode_output_byte_len,
        params_byte_len,
    );
    let buffer_ensure_elapsed = buffer_ensure_start.elapsed();
    let cpu_prepare_elapsed = cpu_prepare_start.elapsed();

    let upload_timings = upload_decode_inputs(queue, buffers, raw_payload, &params_bytes)?;

    let encode_submit_start = Instant::now();
    let bind_group_start = Instant::now();
    if bind_group_buffers_changed || buffers.bind_group.is_none() {
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let raw_payload_buffer = &buffers
            .raw_payload
            .as_ref()
            .context("raw payload GPU buffer was not allocated")?
            .buffer;
        let work_items_buffer = &buffers
            .work_items
            .as_ref()
            .context("work item GPU buffer was not allocated")?
            .buffer;
        let output_buffer = &buffers
            .output
            .as_ref()
            .context("output GPU buffer was not allocated")?
            .buffer;
        let params_buffer = &buffers
            .params
            .as_ref()
            .context("params GPU buffer was not allocated")?
            .buffer;

        buffers.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MCRAW legacy raw16 no-readback decode bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: raw_payload_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: work_items_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        }));
    }
    let bind_group_elapsed = bind_group_start.elapsed();

    let bind_group = buffers
        .bind_group
        .as_ref()
        .context("GPU bind group was not allocated")?;
    let output_buffer = &buffers
        .output
        .as_ref()
        .context("output GPU buffer was not allocated")?
        .buffer;

    let command_encoder_create_start = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MCRAW legacy raw16 GPU decode/no-readback-stage command encoder"),
    });
    let command_encoder_create_elapsed = command_encoder_create_start.elapsed();
    let timestamp_recorder = ensure_timestamp_recorder(
        device,
        &mut buffers.timestamp_recorder,
        timestamp_support,
        "MCRAW legacy raw16 no-readback decode timestamps",
    );
    let output_clear_enabled = output_layout.output_clear_enabled();
    let mut timestamp_stages = GpuTimestampStageMask {
        output_clear: output_clear_enabled,
        vignette: false,
    };

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::TotalStart);
        timestamps.write(&mut encoder, GpuTimestampQuery::OutputClearStart);
    }

    let mut decode_output_clear_encode_elapsed = Duration::ZERO;
    if output_clear_enabled {
        let output_clear_encode_start = Instant::now();
        encoder.clear_buffer(output_buffer, 0, Some(decode_output_byte_len));
        decode_output_clear_encode_elapsed = output_clear_encode_start.elapsed();
    }

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::OutputClearEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::DecodeDispatchStart);
    }

    let decode_pass_encode_start = Instant::now();
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MCRAW legacy raw16 no-readback decode compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
    }
    let decode_pass_encode_elapsed = decode_pass_encode_start.elapsed();

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::DecodeDispatchEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchStart);
    }

    let mut vignette_stats = None;
    let mut vignette_encode_elapsed = Duration::ZERO;
    let stage = if let Some(correction) = vignette_correction {
        let vignette_encode_start = Instant::now();
        let dispatch = correction
            .corrector
            .dispatch_packed_u16(GpuVignettePackedU16DispatchInput {
                device,
                queue,
                encoder: &mut encoder,
                input_buffer: output_buffer,
                input_buffer_bytes: decode_output_byte_len,
                uploaded_gain_map: correction.uploaded_gain_map,
                params: correction.params,
            })
            .context("failed to dispatch GPU vignette correction before legacy raw16 GPU stage")?;
        vignette_encode_elapsed = vignette_encode_start.elapsed();
        timestamp_stages.vignette = dispatch.stats.dispatch_submitted;
        if let Some(timestamps) = timestamp_recorder.as_ref() {
            timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchEnd);
            timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackStart);
        }
        vignette_stats = Some(dispatch.stats);
        encode_gpu_stage(
            device,
            queue,
            &mut encoder,
            GpuDecodedPackedU16BufferView {
                buffer: dispatch.output.buffer(),
                byte_len: decode_output_byte_len,
                dimensions: visible_dimensions,
                pixel_count,
            },
        )?
    } else {
        if let Some(timestamps) = timestamp_recorder.as_ref() {
            timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchEnd);
            timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackStart);
        }
        encode_gpu_stage(
            device,
            queue,
            &mut encoder,
            GpuDecodedPackedU16BufferView {
                buffer: output_buffer,
                byte_len: decode_output_byte_len,
                dimensions: visible_dimensions,
                pixel_count,
            },
        )?
    };

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::TotalEnd);
        timestamps.resolve(&mut encoder);
    }

    let command_finish_submit_start = Instant::now();
    let submission_index = queue.submit(Some(encoder.finish()));
    let command_finish_submit_elapsed = command_finish_submit_start.elapsed();
    let encode_submit_elapsed = encode_submit_start.elapsed();
    let dispatch_submit_elapsed = total_start.elapsed();
    let submitted_timestamps = timestamp_recorder
        .map(|_| GpuSubmittedTimestampProfile::from_stage_mask(timestamp_support, timestamp_stages))
        .unwrap_or_else(|| GpuSubmittedTimestampProfile::unavailable(timestamp_support));

    Ok(PendingNoReadbackGpuStageDispatchWithVignetteResult {
        stage,
        submission_index,
        dispatch_start: total_start,
        gpu_timestamps: submitted_timestamps,
        timings: GpuDispatchTimings {
            raw_payload_bytes: raw_payload_byte_len,
            work_item_count: work_items.len() as u64,
            work_item_bytes: work_item_byte_len,
            params_bytes: params_byte_len,
            output_bytes: decode_output_byte_len,
            readback_bytes: 0,
            work_item_buffer_reused: upload_byte_counts.work_item_buffer_reused,
            work_item_buffer_capacity: upload_byte_counts.work_item_buffer_capacity,
            work_item_buffer_reallocs: upload_byte_counts.work_item_buffer_reallocs,
            write_buffer_with_work_items: upload_timings.write_buffer_with_work_items,
            write_buffer_with_params: upload_timings.write_buffer_with_params,
            direct_staging_fallback: upload_timings.direct_staging_fallback,
            queue_write_buffer_calls: upload_timings.queue_write_buffer_calls,
            queue_write_buffer_with_calls: upload_timings.queue_write_buffer_with_calls,
            cpu_prepare: cpu_prepare_elapsed,
            buffer_ensure: buffer_ensure_elapsed,
            cpu_staging_serialize: cpu_staging_serialize_elapsed,
            upload: upload_timings.total,
            raw_payload_upload: upload_timings.raw_payload_upload,
            work_items_upload: upload_timings.work_items_upload,
            params_upload: upload_timings.params_upload,
            encode_submit: encode_submit_elapsed,
            bind_group: bind_group_elapsed,
            command_encoder_create: command_encoder_create_elapsed,
            output_clear_encode: decode_output_clear_encode_elapsed,
            decode_pass_encode: decode_pass_encode_elapsed,
            vignette_encode: vignette_encode_elapsed,
            copy_to_readback_encode: Duration::ZERO,
            command_finish_submit: command_finish_submit_elapsed,
            wait_map: Duration::ZERO,
            mapped_consumer: Duration::ZERO,
            readback_convert: Duration::ZERO,
            total: dispatch_submit_elapsed,
            gpu_timestamps: GpuTimestampProfile::unavailable(timestamp_support),
        },
        vignette_stats,
    })
}

#[allow(clippy::too_many_arguments)]
fn submit_mcraw_gpu_dispatch_packed_u16_no_readback_with_gpu_stage<S, E>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: &mut ReusableGpuBuffers,
    timestamp_support: GpuTimestampSupport,
    raw_payload: &[u8],
    work_items: &[McrawVulkanBlockWorkItem],
    pixel_count: usize,
    visible_width: u32,
    visible_height: u32,
    max_compute_workgroups_per_dimension: u32,
    output_layout: GpuOutputLayout,
    vignette_correction: Option<OptionalGpuVignetteCorrection<'_>>,
    visible_dimensions: FrameDimensions,
    encode_gpu_stage: E,
) -> Result<PendingNoReadbackGpuStageDispatchWithVignetteResult<S>>
where
    E: FnOnce(
        &wgpu::Device,
        &wgpu::Queue,
        &mut wgpu::CommandEncoder,
        GpuDecodedPackedU16BufferView<'_>,
    ) -> Result<S>,
{
    let total_start = Instant::now();

    let cpu_prepare_start = Instant::now();

    let cpu_staging_serialize_start = Instant::now();
    let upload_byte_counts = prepare_upload_staging(raw_payload, work_items, buffers)?;
    let cpu_staging_serialize_elapsed = cpu_staging_serialize_start.elapsed();

    let raw_payload_byte_len = upload_byte_counts.raw_payload_byte_len;
    let work_item_byte_len = upload_byte_counts.work_item_byte_len;
    let decode_output_byte_len = byte_len_for_output_layout(pixel_count, output_layout)?;

    if !has_complete_macroblocks(work_items.len()) {
        anyhow::bail!(
            "work item count is not divisible by descriptors per macroblock: work_items={}, descriptors_per_macroblock={}",
            work_items.len(),
            MCRAW_DESCRIPTORS_PER_MACROBLOCK
        );
    }

    let macroblock_count = work_items.len() / MCRAW_DESCRIPTORS_PER_MACROBLOCK as usize;
    let (dispatch_width, dispatch_height) =
        dispatch_grid(macroblock_count, max_compute_workgroups_per_dimension)?;

    let params_bytes = mcraw_params_to_le_bytes(
        visible_width,
        visible_height,
        u32::try_from(work_items.len()).context("work item count does not fit in u32")?,
        u32::try_from(macroblock_count).context("macroblock count does not fit in u32")?,
        dispatch_width,
    );

    let params_byte_len =
        u64::try_from(params_bytes.len()).context("params byte length overflow")?;

    let buffer_ensure_start = Instant::now();
    let bind_group_buffers_changed = buffers.ensure_for_decode_without_readback(
        device,
        raw_payload_byte_len,
        work_item_byte_len,
        decode_output_byte_len,
        params_byte_len,
    );
    let buffer_ensure_elapsed = buffer_ensure_start.elapsed();

    let cpu_prepare_elapsed = cpu_prepare_start.elapsed();

    let upload_timings = upload_decode_inputs(queue, buffers, raw_payload, &params_bytes)?;

    let encode_submit_start = Instant::now();

    let bind_group_start = Instant::now();
    if bind_group_buffers_changed || buffers.bind_group.is_none() {
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let raw_payload_buffer = &buffers
            .raw_payload
            .as_ref()
            .context("raw payload GPU buffer was not allocated")?
            .buffer;
        let work_items_buffer = &buffers
            .work_items
            .as_ref()
            .context("work item GPU buffer was not allocated")?
            .buffer;
        let output_buffer = &buffers
            .output
            .as_ref()
            .context("output GPU buffer was not allocated")?
            .buffer;
        let params_buffer = &buffers
            .params
            .as_ref()
            .context("params GPU buffer was not allocated")?
            .buffer;

        buffers.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MCRAW no-readback GPU-stage decode bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: raw_payload_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: work_items_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        }));
    }
    let bind_group_elapsed = bind_group_start.elapsed();

    let bind_group = buffers
        .bind_group
        .as_ref()
        .context("GPU bind group was not allocated")?;
    let output_buffer = &buffers
        .output
        .as_ref()
        .context("output GPU buffer was not allocated")?
        .buffer;

    let command_encoder_create_start = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MCRAW GPU decode/no-readback-stage command encoder"),
    });
    let command_encoder_create_elapsed = command_encoder_create_start.elapsed();
    let timestamp_recorder = ensure_timestamp_recorder(
        device,
        &mut buffers.timestamp_recorder,
        timestamp_support,
        "MCRAW GPU decode/no-readback-stage timestamps",
    );
    let output_clear_enabled = output_layout.output_clear_enabled();
    let mut timestamp_stages = GpuTimestampStageMask {
        output_clear: output_clear_enabled,
        vignette: false,
    };

    if let Some(timestamps) = timestamp_recorder {
        timestamps.write(&mut encoder, GpuTimestampQuery::TotalStart);
        timestamps.write(&mut encoder, GpuTimestampQuery::OutputClearStart);
    }

    let mut decode_output_clear_encode_elapsed = Duration::ZERO;
    if output_clear_enabled {
        let output_clear_encode_start = Instant::now();
        encoder.clear_buffer(output_buffer, 0, Some(decode_output_byte_len));
        decode_output_clear_encode_elapsed = output_clear_encode_start.elapsed();
    }

    if let Some(timestamps) = timestamp_recorder {
        timestamps.write(&mut encoder, GpuTimestampQuery::OutputClearEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::DecodeDispatchStart);
    }

    let decode_pass_encode_start = Instant::now();
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MCRAW no-readback GPU-stage decode compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
    }
    let decode_pass_encode_elapsed = decode_pass_encode_start.elapsed();

    if let Some(timestamps) = timestamp_recorder {
        timestamps.write(&mut encoder, GpuTimestampQuery::DecodeDispatchEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchStart);
    }

    let mut vignette_stats = None;
    let mut vignette_encode_elapsed = Duration::ZERO;
    let stage = if let Some(correction) = vignette_correction {
        let vignette_encode_start = Instant::now();
        let dispatch = correction
            .corrector
            .dispatch_packed_u16(GpuVignettePackedU16DispatchInput {
                device,
                queue,
                encoder: &mut encoder,
                input_buffer: output_buffer,
                input_buffer_bytes: decode_output_byte_len,
                uploaded_gain_map: correction.uploaded_gain_map,
                params: correction.params,
            })
            .context("failed to dispatch GPU vignette correction before no-readback GPU stage")?;
        vignette_encode_elapsed = vignette_encode_start.elapsed();
        timestamp_stages.vignette = dispatch.stats.dispatch_submitted;
        if let Some(timestamps) = timestamp_recorder {
            timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchEnd);
            timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackStart);
        }
        vignette_stats = Some(dispatch.stats);
        encode_gpu_stage(
            device,
            queue,
            &mut encoder,
            GpuDecodedPackedU16BufferView {
                buffer: dispatch.output.buffer(),
                byte_len: decode_output_byte_len,
                dimensions: visible_dimensions,
                pixel_count,
            },
        )?
    } else {
        if let Some(timestamps) = timestamp_recorder {
            timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchEnd);
            timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackStart);
        }
        encode_gpu_stage(
            device,
            queue,
            &mut encoder,
            GpuDecodedPackedU16BufferView {
                buffer: output_buffer,
                byte_len: decode_output_byte_len,
                dimensions: visible_dimensions,
                pixel_count,
            },
        )?
    };

    if let Some(timestamps) = timestamp_recorder {
        timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::TotalEnd);
        timestamps.resolve(&mut encoder);
    }

    let command_finish_submit_start = Instant::now();
    let submission_index = queue.submit(Some(encoder.finish()));
    let command_finish_submit_elapsed = command_finish_submit_start.elapsed();
    let encode_submit_elapsed = encode_submit_start.elapsed();

    let dispatch_submit_elapsed = total_start.elapsed();
    let submitted_timestamps = timestamp_recorder
        .map(|_| GpuSubmittedTimestampProfile::from_stage_mask(timestamp_support, timestamp_stages))
        .unwrap_or_else(|| GpuSubmittedTimestampProfile::unavailable(timestamp_support));

    Ok(PendingNoReadbackGpuStageDispatchWithVignetteResult {
        stage,
        submission_index,
        dispatch_start: total_start,
        gpu_timestamps: submitted_timestamps,
        timings: GpuDispatchTimings {
            raw_payload_bytes: raw_payload_byte_len,
            work_item_count: work_items.len() as u64,
            work_item_bytes: work_item_byte_len,
            params_bytes: params_byte_len,
            output_bytes: decode_output_byte_len,
            readback_bytes: 0,
            work_item_buffer_reused: upload_byte_counts.work_item_buffer_reused,
            work_item_buffer_capacity: upload_byte_counts.work_item_buffer_capacity,
            work_item_buffer_reallocs: upload_byte_counts.work_item_buffer_reallocs,
            write_buffer_with_work_items: upload_timings.write_buffer_with_work_items,
            write_buffer_with_params: upload_timings.write_buffer_with_params,
            direct_staging_fallback: upload_timings.direct_staging_fallback,
            queue_write_buffer_calls: upload_timings.queue_write_buffer_calls,
            queue_write_buffer_with_calls: upload_timings.queue_write_buffer_with_calls,
            cpu_prepare: cpu_prepare_elapsed,
            buffer_ensure: buffer_ensure_elapsed,
            cpu_staging_serialize: cpu_staging_serialize_elapsed,
            upload: upload_timings.total,
            raw_payload_upload: upload_timings.raw_payload_upload,
            work_items_upload: upload_timings.work_items_upload,
            params_upload: upload_timings.params_upload,
            encode_submit: encode_submit_elapsed,
            bind_group: bind_group_elapsed,
            command_encoder_create: command_encoder_create_elapsed,
            output_clear_encode: decode_output_clear_encode_elapsed,
            decode_pass_encode: decode_pass_encode_elapsed,
            vignette_encode: vignette_encode_elapsed,
            copy_to_readback_encode: Duration::ZERO,
            command_finish_submit: command_finish_submit_elapsed,
            wait_map: Duration::ZERO,
            mapped_consumer: Duration::ZERO,
            readback_convert: Duration::ZERO,
            total: dispatch_submit_elapsed,
            gpu_timestamps: GpuTimestampProfile::unavailable(timestamp_support),
        },
        vignette_stats,
    })
}

fn wait_mcraw_gpu_dispatch_packed_u16_no_readback_with_gpu_stage<S>(
    device: &wgpu::Device,
    buffers: &ReusableGpuBuffers,
    pending: PendingNoReadbackGpuStageDispatchWithVignetteResult<S>,
) -> Result<GpuNoReadbackGpuStageDispatchWithVignetteResult<S>> {
    let wait_start = Instant::now();
    device.poll(wgpu::Maintain::wait_for(pending.submission_index.clone()));
    let wait_elapsed = wait_start.elapsed();

    let dispatch_elapsed = pending.dispatch_start.elapsed();
    let gpu_timestamps = read_submitted_timestamp_profile(
        device,
        buffers.timestamp_recorder.as_ref(),
        pending.submission_index,
        pending.gpu_timestamps,
    )?;
    let mut timings = pending.timings;
    timings.wait_map = wait_elapsed;
    timings.total = dispatch_elapsed;
    timings.gpu_timestamps = gpu_timestamps;

    Ok(GpuNoReadbackGpuStageDispatchWithVignetteResult {
        stage: pending.stage,
        timings,
        vignette_stats: pending.vignette_stats,
    })
}

// One submitted in-flight mapped dispatch.
//
// The slot_index identifies the independent GPU/readback buffers that must not
// be reused until this pending dispatch has been mapped, consumed, unmapped, and
// returned to the idle slot pool.
struct PendingMappedDispatch {
    slot_index: usize,
    input_index: usize,
    frame_index: usize,
    output_byte_len: u64,
    submission_index: wgpu::SubmissionIndex,
    work_plan_elapsed: Duration,
    work_plan_stats: GpuWorkPlanAllocationStats,
    dimensions: FrameDimensions,
    work_item_count: usize,
    macroblock_count: usize,
    expected_invocations: usize,
    dispatch_timings: GpuDispatchTimings,
    dispatch_start: Instant,
    timestamp_support: GpuTimestampSupport,
    timestamp_recorder: Option<GpuTimestampRecorder>,
    timestamp_stages: GpuTimestampStageMask,
    vignette_stats: Option<GpuVignetteCorrectionStats>,
}

// Slots own independent buffers while work is in flight. Retaining completed
// slots reuses allocations without aliasing a pending submission.
fn ensure_pipeline_slot_count(slots: &mut Vec<ReusableGpuBuffers>, slot_count: usize) {
    while slots.len() < slot_count {
        slots.push(ReusableGpuBuffers::default());
    }
}

#[allow(clippy::too_many_arguments)]
fn submit_mcraw_gpu_dispatch_mapped_in_flight(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    buffers: &mut ReusableGpuBuffers,
    timestamp_support: GpuTimestampSupport,
    slot_index: usize,
    input_index: usize,
    frame_index: usize,
    raw_payload: &[u8],
    work_items: &[McrawVulkanBlockWorkItem],
    pixel_count: usize,
    visible_width: u32,
    visible_height: u32,
    max_compute_workgroups_per_dimension: u32,
    output_layout: GpuOutputLayout,
    work_plan_elapsed: Duration,
    work_plan_stats: GpuWorkPlanAllocationStats,
    dimensions: FrameDimensions,
    work_item_count: usize,
    macroblock_count: usize,
    expected_invocations: usize,
    vignette_correction: Option<GpuMappedRingVignetteCorrection>,
) -> Result<PendingMappedDispatch> {
    let dispatch_start = Instant::now();

    let cpu_prepare_start = Instant::now();

    let cpu_staging_serialize_start = Instant::now();
    let upload_byte_counts = prepare_upload_staging(raw_payload, work_items, buffers)?;
    let cpu_staging_serialize_elapsed = cpu_staging_serialize_start.elapsed();

    let raw_payload_byte_len = upload_byte_counts.raw_payload_byte_len;
    let work_item_byte_len = upload_byte_counts.work_item_byte_len;
    let output_byte_len = byte_len_for_output_layout(pixel_count, output_layout)?;

    if !has_complete_macroblocks(work_items.len()) {
        anyhow::bail!(
            "work item count is not divisible by descriptors per macroblock: work_items={}, descriptors_per_macroblock={}",
            work_items.len(),
            MCRAW_DESCRIPTORS_PER_MACROBLOCK
        );
    }

    let (dispatch_width, dispatch_height) =
        dispatch_grid(macroblock_count, max_compute_workgroups_per_dimension)?;

    let params_bytes = mcraw_params_to_le_bytes(
        visible_width,
        visible_height,
        u32::try_from(work_items.len()).context("work item count does not fit in u32")?,
        u32::try_from(macroblock_count).context("macroblock count does not fit in u32")?,
        dispatch_width,
    );

    let params_byte_len =
        u64::try_from(params_bytes.len()).context("params byte length overflow")?;

    let buffer_ensure_start = Instant::now();
    let bind_group_buffers_changed = buffers.ensure_for_decode(
        device,
        raw_payload_byte_len,
        work_item_byte_len,
        output_byte_len,
        params_byte_len,
    );
    let buffer_ensure_elapsed = buffer_ensure_start.elapsed();

    let cpu_prepare_elapsed = cpu_prepare_start.elapsed();

    let upload_timings = upload_decode_inputs(queue, buffers, raw_payload, &params_bytes)?;

    let encode_submit_start = Instant::now();

    let bind_group_start = Instant::now();
    if bind_group_buffers_changed || buffers.bind_group.is_none() {
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let raw_payload_buffer = &buffers
            .raw_payload
            .as_ref()
            .context("raw payload GPU buffer was not allocated")?
            .buffer;
        let work_items_buffer = &buffers
            .work_items
            .as_ref()
            .context("work item GPU buffer was not allocated")?
            .buffer;
        let output_buffer = &buffers
            .output
            .as_ref()
            .context("output GPU buffer was not allocated")?
            .buffer;
        let params_buffer = &buffers
            .params
            .as_ref()
            .context("params GPU buffer was not allocated")?
            .buffer;

        buffers.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MCRAW in-flight GPU decode bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: raw_payload_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: work_items_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        }));
    }
    let bind_group_elapsed = bind_group_start.elapsed();

    let bind_group = buffers
        .bind_group
        .as_ref()
        .context("GPU bind group was not allocated")?;
    let output_buffer = &buffers
        .output
        .as_ref()
        .context("output GPU buffer was not allocated")?
        .buffer;
    let readback_buffer = &buffers
        .readback
        .as_ref()
        .context("readback GPU buffer was not allocated")?
        .buffer;

    let command_encoder_create_start = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MCRAW in-flight GPU decode command encoder"),
    });
    let command_encoder_create_elapsed = command_encoder_create_start.elapsed();
    let timestamp_recorder = GpuTimestampRecorder::new(
        device,
        timestamp_support,
        "MCRAW in-flight GPU decode timestamps",
    );
    let output_clear_enabled = output_layout.output_clear_enabled();
    let mut timestamp_stages = GpuTimestampStageMask {
        output_clear: output_clear_enabled,
        vignette: false,
    };

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::TotalStart);
        timestamps.write(&mut encoder, GpuTimestampQuery::OutputClearStart);
    }

    // Ring slots retain their output allocation between owners, so clear it
    // before atomic packed writes can inherit bits from the previous owner.
    let mut output_clear_encode_elapsed = Duration::ZERO;
    if output_clear_enabled {
        let output_clear_encode_start = Instant::now();
        encoder.clear_buffer(output_buffer, 0, Some(output_byte_len));
        output_clear_encode_elapsed = output_clear_encode_start.elapsed();
    }

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::OutputClearEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::DecodeDispatchStart);
    }

    let decode_pass_encode_start = Instant::now();
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MCRAW in-flight GPU decode compute pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
    }
    let decode_pass_encode_elapsed = decode_pass_encode_start.elapsed();

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::DecodeDispatchEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchStart);
    }

    let mut vignette_stats = None;
    let mut vignette_encode_elapsed = Duration::ZERO;
    let copy_to_readback_encode_elapsed;
    if let Some(correction) = vignette_correction {
        if buffers.vignette_corrector.is_none() {
            buffers.vignette_corrector =
                Some(GpuVignetteCorrector::new(device).map_err(|error| {
                    anyhow!("failed to create ring slot GPU vignette corrector: {error}")
                })?);
        }
        let corrector = buffers
            .vignette_corrector
            .as_mut()
            .context("ring slot GPU vignette corrector was not allocated")?;

        let vignette_encode_start = Instant::now();
        let dispatch = corrector
            .dispatch_packed_u16(GpuVignettePackedU16DispatchInput {
                device,
                queue,
                encoder: &mut encoder,
                input_buffer: output_buffer,
                input_buffer_bytes: output_byte_len,
                uploaded_gain_map: &correction.uploaded_gain_map,
                params: correction.params,
            })
            .context("failed to dispatch ring-mapped GPU vignette correction")?;
        vignette_encode_elapsed = vignette_encode_start.elapsed();
        timestamp_stages.vignette = dispatch.stats.dispatch_submitted;

        if let Some(timestamps) = timestamp_recorder.as_ref() {
            timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchEnd);
            timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackStart);
        }

        let copy_to_readback_encode_start = Instant::now();
        encoder.copy_buffer_to_buffer(
            dispatch.output.buffer(),
            0,
            readback_buffer,
            0,
            output_byte_len,
        );
        copy_to_readback_encode_elapsed = copy_to_readback_encode_start.elapsed();
        vignette_stats = Some(dispatch.stats);
    } else {
        if let Some(timestamps) = timestamp_recorder.as_ref() {
            timestamps.write(&mut encoder, GpuTimestampQuery::VignetteDispatchEnd);
            timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackStart);
        }

        let copy_to_readback_encode_start = Instant::now();
        encoder.copy_buffer_to_buffer(output_buffer, 0, readback_buffer, 0, output_byte_len);
        copy_to_readback_encode_elapsed = copy_to_readback_encode_start.elapsed();
    }

    if let Some(timestamps) = timestamp_recorder.as_ref() {
        timestamps.write(&mut encoder, GpuTimestampQuery::CopyToReadbackEnd);
        timestamps.write(&mut encoder, GpuTimestampQuery::TotalEnd);
        timestamps.resolve(&mut encoder);
    }

    let command_finish_submit_start = Instant::now();
    let submission_index = queue.submit(Some(encoder.finish()));
    let command_finish_submit_elapsed = command_finish_submit_start.elapsed();
    let encode_submit_elapsed = encode_submit_start.elapsed();

    Ok(PendingMappedDispatch {
        slot_index,
        input_index,
        frame_index,
        output_byte_len,
        submission_index,
        work_plan_elapsed,
        work_plan_stats,
        dimensions,
        work_item_count,
        macroblock_count,
        expected_invocations,
        dispatch_timings: GpuDispatchTimings {
            raw_payload_bytes: raw_payload_byte_len,
            work_item_count: work_items.len() as u64,
            work_item_bytes: work_item_byte_len,
            params_bytes: params_byte_len,
            output_bytes: output_byte_len,
            readback_bytes: output_byte_len,
            work_item_buffer_reused: upload_byte_counts.work_item_buffer_reused,
            work_item_buffer_capacity: upload_byte_counts.work_item_buffer_capacity,
            work_item_buffer_reallocs: upload_byte_counts.work_item_buffer_reallocs,
            write_buffer_with_work_items: upload_timings.write_buffer_with_work_items,
            write_buffer_with_params: upload_timings.write_buffer_with_params,
            direct_staging_fallback: upload_timings.direct_staging_fallback,
            queue_write_buffer_calls: upload_timings.queue_write_buffer_calls,
            queue_write_buffer_with_calls: upload_timings.queue_write_buffer_with_calls,
            cpu_prepare: cpu_prepare_elapsed,
            buffer_ensure: buffer_ensure_elapsed,
            cpu_staging_serialize: cpu_staging_serialize_elapsed,
            upload: upload_timings.total,
            raw_payload_upload: upload_timings.raw_payload_upload,
            work_items_upload: upload_timings.work_items_upload,
            params_upload: upload_timings.params_upload,
            encode_submit: encode_submit_elapsed,
            bind_group: bind_group_elapsed,
            command_encoder_create: command_encoder_create_elapsed,
            output_clear_encode: output_clear_encode_elapsed,
            decode_pass_encode: decode_pass_encode_elapsed,
            vignette_encode: vignette_encode_elapsed,
            copy_to_readback_encode: copy_to_readback_encode_elapsed,
            command_finish_submit: command_finish_submit_elapsed,
            wait_map: Duration::ZERO,
            mapped_consumer: Duration::ZERO,
            readback_convert: Duration::ZERO,
            total: Duration::ZERO,
            gpu_timestamps: GpuTimestampProfile::unavailable(timestamp_support),
        },
        dispatch_start,
        timestamp_support,
        timestamp_recorder,
        timestamp_stages,
        vignette_stats,
    })
}

fn finish_mcraw_gpu_dispatch_mapped_in_flight<T, F>(
    device: &wgpu::Device,
    buffers: &mut ReusableGpuBuffers,
    pending: PendingMappedDispatch,
    consume_mapped_pixels: &mut F,
) -> Result<(
    GpuMappedPackedU16Output<T>,
    Option<GpuVignetteCorrectionStats>,
)>
where
    F: FnMut(usize, &[u8]) -> Result<T>,
{
    let readback_buffer = &buffers
        .readback
        .as_ref()
        .context("readback GPU buffer was not allocated")?
        .buffer;
    let readback_slice = readback_buffer.slice(0..pending.output_byte_len);

    let wait_map_start = Instant::now();

    let (sender, receiver) = mpsc::channel();
    readback_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device.poll(wgpu::Maintain::wait_for(pending.submission_index.clone()));

    let map_result = receiver
        .recv_timeout(Duration::from_secs(30))
        .context("timed out waiting for in-flight MCRAW readback buffer mapping callback")?;

    map_result
        .map_err(|error| anyhow!("failed to map in-flight MCRAW readback buffer: {error:?}"))?;

    let wait_map_elapsed = wait_map_start.elapsed();
    let dispatch_readback_elapsed = pending.dispatch_start.elapsed();

    let mapped_consumer_start = Instant::now();
    let consume_result = {
        let mapped = readback_slice.get_mapped_range();
        let value = consume_mapped_pixels(pending.frame_index, &mapped);
        drop(mapped);
        value
    };
    let mapped_consumer_elapsed = mapped_consumer_start.elapsed();

    readback_buffer.unmap();

    let value = consume_result?;
    let gpu_timestamps = if let Some(timestamps) = pending.timestamp_recorder {
        timestamps.read_profile(device, pending.submission_index, pending.timestamp_stages)?
    } else {
        GpuTimestampProfile::unavailable(pending.timestamp_support)
    };
    let gpu_total_without_consumer = pending
        .work_plan_elapsed
        .saturating_add(dispatch_readback_elapsed);
    let mut dispatch_timings = pending.dispatch_timings;
    dispatch_timings.wait_map = wait_map_elapsed;
    dispatch_timings.mapped_consumer = mapped_consumer_elapsed;
    dispatch_timings.total = dispatch_readback_elapsed;
    dispatch_timings.gpu_timestamps = gpu_timestamps;

    Ok((
        GpuMappedPackedU16Output {
            value,
            dimensions: pending.dimensions,
            work_item_count: pending.work_item_count,
            macroblock_count: pending.macroblock_count,
            expected_invocations: pending.expected_invocations,
            timings: GpuDecodeTimings::from_work_plan_stats_dispatch(
                pending.work_plan_stats,
                pending.work_plan_elapsed,
                dispatch_timings,
                gpu_total_without_consumer,
            ),
        },
        pending.vignette_stats,
    ))
}

fn ensure_buffer_slot(
    device: &wgpu::Device,
    slot: &mut Option<SizedGpuBuffer>,
    label: &'static str,
    required_size: u64,
    usage: wgpu::BufferUsages,
) -> bool {
    let needs_allocation = slot
        .as_ref()
        .map(|existing| existing.size < required_size)
        .unwrap_or(true);

    if needs_allocation {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: required_size,
            usage,
            mapped_at_creation: false,
        });

        *slot = Some(SizedGpuBuffer {
            buffer,
            size: required_size,
        });
    }

    needs_allocation
}

struct UploadByteCounts {
    raw_payload_byte_len: u64,
    work_item_byte_len: u64,
    work_item_buffer_reused: bool,
    work_item_buffer_capacity: u64,
    work_item_buffer_reallocs: u64,
}

struct UploadTimings {
    raw_payload_upload: Duration,
    work_items_upload: Duration,
    params_upload: Duration,
    total: Duration,
    write_buffer_with_work_items: bool,
    write_buffer_with_params: bool,
    direct_staging_fallback: bool,
    queue_write_buffer_calls: u64,
    queue_write_buffer_with_calls: u64,
}

struct WorkItemSerializationStats {
    byte_len: usize,
    buffer_reused: bool,
    buffer_capacity: usize,
    reallocs: u64,
}

fn build_legacy_raw16_gpu_work_plan(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    row_stride: u32,
) -> Result<LegacyRaw16GpuWorkPlan> {
    let mut work_plan = LegacyRaw16GpuWorkPlan::default();
    build_legacy_raw16_gpu_work_plan_into(
        raw_payload,
        visible_dimensions,
        row_stride,
        &mut work_plan,
    )?;
    Ok(work_plan)
}

fn build_legacy_raw16_gpu_work_plan_into(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    row_stride: u32,
    work_plan: &mut LegacyRaw16GpuWorkPlan,
) -> Result<LegacyRaw16WorkPlanBuildStats> {
    let visible_row_bytes = u64::from(visible_dimensions.width)
        .checked_mul(2)
        .context("legacy raw16 visible row byte length overflow")?;
    anyhow::ensure!(
        u64::from(row_stride) >= visible_row_bytes,
        "raw16 rowStride is too short: rowStride={}, minimum={}",
        row_stride,
        visible_row_bytes
    );

    let padded_width = legacy_raw16_padded_width(visible_dimensions.width)?;
    let height = usize::try_from(visible_dimensions.height)
        .context("legacy raw16 visible height does not fit in usize")?;
    let chunk_count = padded_width
        .checked_div(LEGACY_RAW16_ENCODING_BLOCK)
        .and_then(|chunks_per_row| chunks_per_row.checked_mul(height))
        .context("legacy raw16 chunk count overflow")?;
    anyhow::ensure!(chunk_count > 0, "legacy raw16 frame has no chunks");

    let work_item_count = chunk_count
        .checked_mul(2)
        .context("legacy raw16 work item count overflow")?;
    let capacity_before = work_plan.blocks.capacity();
    work_plan.blocks.clear();
    if capacity_before < work_item_count {
        work_plan.blocks.reserve(work_item_count);
    }
    let capacity_after = work_plan.blocks.capacity();
    let mut offset = 0usize;

    for row in 0..height {
        let row_u32 = u32::try_from(row).context("legacy raw16 row does not fit in u32")?;
        let mut x = 0usize;
        while x < padded_width {
            let x_u32 = u32::try_from(x).context("legacy raw16 x offset does not fit in u32")?;
            let (first, consumed_first) =
                legacy_raw16_work_item(raw_payload, offset, x_u32, row_u32, 0)?;
            offset = offset
                .checked_add(consumed_first)
                .context("legacy raw16 payload offset overflow")?;
            let (second, consumed_second) =
                legacy_raw16_work_item(raw_payload, offset, x_u32, row_u32, 1)?;
            offset = offset
                .checked_add(consumed_second)
                .context("legacy raw16 payload offset overflow")?;

            work_plan.blocks.push(first);
            work_plan.blocks.push(second);
            x = x
                .checked_add(LEGACY_RAW16_ENCODING_BLOCK)
                .context("legacy raw16 x offset overflow")?;
        }
    }

    work_plan.chunk_count = chunk_count;
    Ok(LegacyRaw16WorkPlanBuildStats {
        reused_scratch: capacity_before >= work_item_count,
        scratch_grow_count: u64::from(capacity_after > capacity_before),
    })
}

fn legacy_raw16_work_item(
    raw_payload: &[u8],
    offset: usize,
    macroblock_x: u32,
    macroblock_y: u32,
    lane_index: u32,
) -> Result<(McrawVulkanBlockWorkItem, usize)> {
    let header_end = offset
        .checked_add(LEGACY_RAW16_HEADER_LENGTH)
        .context("legacy raw16 header offset overflow")?;
    anyhow::ensure!(
        header_end <= raw_payload.len(),
        "legacy raw16 payload is too short: required at least {} bytes, got {}",
        header_end,
        raw_payload.len()
    );

    let header0 = raw_payload[offset];
    let bits = u32::from((header0 >> 4) & 0x0f);
    let reference_value = (u32::from(header0 & 0x0f) << 8) | u32::from(raw_payload[offset + 1]);
    let payload_len = LEGACY_RAW16_BLOCK_LENGTHS[bits as usize];
    let payload_end = header_end
        .checked_add(payload_len)
        .context("legacy raw16 payload length overflow")?;
    anyhow::ensure!(
        payload_end <= raw_payload.len(),
        "legacy raw16 payload is too short: required at least {} bytes, got {}",
        payload_end,
        raw_payload.len()
    );

    Ok((
        McrawVulkanBlockWorkItem {
            payload_offset: u32::try_from(header_end)
                .context("legacy raw16 payload offset does not fit in u32")?,
            payload_len: u32::try_from(payload_len)
                .context("legacy raw16 payload length does not fit in u32")?,
            raw_encoding: bits,
            reference_value,
            macroblock_x,
            macroblock_y,
            lane_index,
            reserved: 0,
        },
        LEGACY_RAW16_HEADER_LENGTH + payload_len,
    ))
}

fn legacy_raw16_padded_width(width: u32) -> Result<usize> {
    let width = usize::try_from(width).context("legacy raw16 width does not fit in usize")?;
    width
        .checked_add(LEGACY_RAW16_ENCODING_BLOCK - 1)
        .map(|value| value / LEGACY_RAW16_ENCODING_BLOCK * LEGACY_RAW16_ENCODING_BLOCK)
        .context("legacy raw16 padded width overflow")
}

fn legacy_raw16_gpu_output_layout() -> GpuOutputLayout {
    GpuOutputLayout::PackedU16Samples
}

fn legacy_raw16_dispatch_grid(
    chunk_count: usize,
    max_compute_workgroups_per_dimension: u32,
) -> Result<(u32, u32)> {
    let invocation_count = chunk_count
        .checked_mul(LEGACY_RAW16_BLOCK_SAMPLES)
        .context("legacy raw16 invocation count overflow")?;
    let workgroup_count = invocation_count
        .div_ceil(LEGACY_RAW16_WORKGROUP_SIZE)
        .max(1);
    dispatch_grid(workgroup_count, max_compute_workgroups_per_dimension)
}

fn legacy_raw16_expected_invocations(
    chunk_count: usize,
    max_compute_workgroups_per_dimension: u32,
) -> Result<usize> {
    let (dispatch_width, dispatch_height) =
        legacy_raw16_dispatch_grid(chunk_count, max_compute_workgroups_per_dimension)?;
    usize::try_from(dispatch_width)
        .ok()
        .and_then(|width| {
            usize::try_from(dispatch_height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|workgroups| workgroups.checked_mul(LEGACY_RAW16_WORKGROUP_SIZE))
        .context("legacy raw16 expected invocation count overflow")
}

fn legacy_raw16_params_to_le_bytes(
    visible_width: u32,
    visible_height: u32,
    work_item_count: u32,
    chunk_count: u32,
    dispatch_width: u32,
) -> [u8; 32] {
    mcraw_params_to_le_bytes(
        visible_width,
        visible_height,
        work_item_count,
        chunk_count,
        dispatch_width,
    )
}

fn upload_decode_inputs(
    queue: &wgpu::Queue,
    buffers: &mut ReusableGpuBuffers,
    raw_payload: &[u8],
    params_bytes: &[u8],
) -> Result<UploadTimings> {
    let upload_start = Instant::now();
    let mut queue_write_buffer_calls = 0u64;

    let raw_payload_buffer = buffers
        .raw_payload
        .as_ref()
        .context("raw payload GPU buffer was not allocated")?
        .buffer
        .clone();
    let work_items_buffer = buffers
        .work_items
        .as_ref()
        .context("work item GPU buffer was not allocated")?
        .buffer
        .clone();
    let params_buffer = buffers
        .params
        .as_ref()
        .context("params GPU buffer was not allocated")?
        .buffer
        .clone();

    let raw_payload_upload_start = Instant::now();
    queue.write_buffer(
        &raw_payload_buffer,
        0,
        raw_payload_upload_bytes(raw_payload, &buffers.raw_payload_upload_scratch),
    );
    queue_write_buffer_calls += 1;
    let raw_payload_upload = raw_payload_upload_start.elapsed();

    let work_items_upload_start = Instant::now();
    queue.write_buffer(&work_items_buffer, 0, &buffers.work_item_upload_scratch);
    queue_write_buffer_calls += 1;
    let work_items_upload = work_items_upload_start.elapsed();

    let params_upload_start = Instant::now();
    queue.write_buffer(&params_buffer, 0, params_bytes);
    queue_write_buffer_calls += 1;
    let params_upload = params_upload_start.elapsed();

    Ok(UploadTimings {
        raw_payload_upload,
        work_items_upload,
        params_upload,
        total: upload_start.elapsed(),
        write_buffer_with_work_items: false,
        write_buffer_with_params: false,
        direct_staging_fallback: false,
        queue_write_buffer_calls,
        queue_write_buffer_with_calls: 0,
    })
}

// Prepare CPU-side upload byte slices for one GPU dispatch.
//
// The GPU storage buffers themselves are persistent and grow-on-demand. This
// helper makes the CPU upload staging match that design: work-item bytes always
// reuse a backend-owned Vec, and raw payload bytes use the original slice when
// already 4-byte aligned or a backend-owned padded scratch Vec only when needed.
fn prepare_upload_staging(
    raw_payload: &[u8],
    work_items: &[McrawVulkanBlockWorkItem],
    buffers: &mut ReusableGpuBuffers,
) -> Result<UploadByteCounts> {
    let raw_payload_byte_len_usize = padded_u32_byte_len(raw_payload.len())?;

    if raw_payload_byte_len_usize != raw_payload.len() {
        fill_padded_bytes_to_u32_multiple(raw_payload, &mut buffers.raw_payload_upload_scratch)?;
    }

    let work_item_stats =
        work_items_to_le_bytes_into(work_items, &mut buffers.work_item_upload_scratch)?;

    Ok(UploadByteCounts {
        raw_payload_byte_len: u64::try_from(raw_payload_byte_len_usize)
            .context("raw payload byte length overflow")?,
        work_item_byte_len: u64::try_from(work_item_stats.byte_len)
            .context("work item byte length overflow")?,
        work_item_buffer_reused: work_item_stats.buffer_reused,
        work_item_buffer_capacity: u64::try_from(work_item_stats.buffer_capacity)
            .context("work item buffer capacity overflow")?,
        work_item_buffer_reallocs: work_item_stats.reallocs,
    })
}

// Return the upload bytes for raw payload data.
//
// If the raw payload is already 4-byte aligned, upload directly from the caller's
// payload slice and avoid any CPU-side copy. Otherwise, use the padded scratch
// buffer filled by prepare_upload_staging().
fn raw_payload_upload_bytes<'a>(raw_payload: &'a [u8], padded_scratch: &'a [u8]) -> &'a [u8] {
    if raw_payload.len() & 3 == 0 {
        raw_payload
    } else {
        padded_scratch
    }
}

fn fill_padded_bytes_to_u32_multiple(bytes: &[u8], output: &mut Vec<u8>) -> Result<()> {
    let padded_len = padded_u32_byte_len(bytes.len())?;

    output.clear();
    output.reserve(padded_len);
    output.extend_from_slice(bytes);
    output.resize(padded_len, 0);

    Ok(())
}

fn padded_u32_byte_len(byte_len: usize) -> Result<usize> {
    let remainder = byte_len % 4;

    if remainder == 0 {
        Ok(byte_len)
    } else {
        byte_len
            .checked_add(4 - remainder)
            .context("padded raw payload byte length overflow")
    }
}

fn work_items_to_le_bytes_into(
    work_items: &[McrawVulkanBlockWorkItem],
    output: &mut Vec<u8>,
) -> Result<WorkItemSerializationStats> {
    let byte_len = work_item_byte_len(work_items.len())?;
    let before_capacity = output.capacity();

    output.clear();
    output.reserve(byte_len);
    output.resize(byte_len, 0);
    write_work_items_to_slice(work_items, output)?;

    let after_capacity = output.capacity();
    Ok(WorkItemSerializationStats {
        byte_len,
        buffer_reused: before_capacity >= byte_len && before_capacity > 0,
        buffer_capacity: after_capacity,
        reallocs: u64::from(after_capacity > before_capacity),
    })
}

fn write_work_items_to_slice(
    work_items: &[McrawVulkanBlockWorkItem],
    output: &mut [u8],
) -> Result<()> {
    let expected_byte_len = work_item_byte_len(work_items.len())?;
    if output.len() != expected_byte_len {
        anyhow::bail!(
            "work item output slice length mismatch: got {}, expected {}",
            output.len(),
            expected_byte_len
        );
    }

    let mut offset = 0usize;
    for (index, item) in work_items.iter().enumerate() {
        let packed = pack_gpu_work_item(item)
            .with_context(|| format!("failed to pack GPU work item {index}"))?;
        write_u32_to_slice(output, &mut offset, packed.word0)?;
        write_u32_to_slice(output, &mut offset, packed.word1)?;
        write_u32_to_slice(output, &mut offset, packed.word2)?;
        write_u32_to_slice(output, &mut offset, packed.word3)?;
    }

    Ok(())
}

fn write_u32_to_slice(output: &mut [u8], offset: &mut usize, value: u32) -> Result<()> {
    let end = offset
        .checked_add(4)
        .context("work item output slice offset overflow")?;
    let destination = output
        .get_mut(*offset..end)
        .context("work item output slice overflow")?;
    destination.copy_from_slice(&value.to_le_bytes());
    *offset = end;
    Ok(())
}

fn pack_gpu_work_item(item: &McrawVulkanBlockWorkItem) -> Result<McrawVulkanPackedBlockWorkItem> {
    let reference_value = checked_packed_field(item, "reference_value", item.reference_value, 16)?;
    let raw_encoding = checked_packed_field(item, "raw_encoding", item.raw_encoding, 8)?;
    let payload_len = checked_packed_field(item, "payload_len", item.payload_len, 8)?;
    let macroblock_x = checked_packed_field(item, "macroblock_x", item.macroblock_x, 16)?;
    let macroblock_y = checked_packed_field(item, "macroblock_y", item.macroblock_y, 16)?;
    let lane_index = checked_packed_field(item, "lane_index", item.lane_index, 8)?;

    if item.reserved != 0 {
        anyhow::bail!(
            "reserved={} must be zero while packing compact GPU work item: {}",
            item.reserved,
            work_item_context(item)
        );
    }

    Ok(McrawVulkanPackedBlockWorkItem {
        word0: item.payload_offset,
        word1: reference_value | (raw_encoding << 16) | (payload_len << 24),
        word2: macroblock_x | (macroblock_y << 16),
        word3: lane_index,
    })
}

fn checked_packed_field(
    item: &McrawVulkanBlockWorkItem,
    field_name: &'static str,
    value: u32,
    bits: u32,
) -> Result<u32> {
    let max = (1u32 << bits) - 1;
    if value > max {
        anyhow::bail!(
            "{field_name}={value} exceeds compact GPU work item {bits}-bit range max={max}: {}",
            work_item_context(item)
        );
    }

    Ok(value)
}

fn work_item_context(item: &McrawVulkanBlockWorkItem) -> String {
    format!(
        "payload_offset={} payload_len={} raw_encoding={} reference_value={} macroblock=({}, {}) lane_index={} reserved={}",
        item.payload_offset,
        item.payload_len,
        item.raw_encoding,
        item.reference_value,
        item.macroblock_x,
        item.macroblock_y,
        item.lane_index,
        item.reserved
    )
}

fn work_item_byte_len(item_count: usize) -> Result<usize> {
    item_count
        .checked_mul(std::mem::size_of::<McrawVulkanPackedBlockWorkItem>())
        .context("work item byte length overflow")
}

fn mcraw_params_to_le_bytes(
    visible_width: u32,
    visible_height: u32,
    work_item_count: u32,
    macroblock_count: u32,
    dispatch_width: u32,
) -> [u8; 32] {
    let mut bytes = [0u8; 32];

    bytes[0..4].copy_from_slice(&visible_width.to_le_bytes());
    bytes[4..8].copy_from_slice(&visible_height.to_le_bytes());
    bytes[8..12].copy_from_slice(&work_item_count.to_le_bytes());
    bytes[12..16].copy_from_slice(&macroblock_count.to_le_bytes());
    bytes[16..20].copy_from_slice(&dispatch_width.to_le_bytes());
    bytes[20..24].copy_from_slice(&0u32.to_le_bytes());
    bytes[24..28].copy_from_slice(&0u32.to_le_bytes());
    bytes[28..32].copy_from_slice(&0u32.to_le_bytes());

    bytes
}

fn byte_len_u32_count(count: usize) -> Result<u64> {
    let bytes = count.checked_mul(4).context("u32 byte length overflow")?;
    u64::try_from(bytes).context("u32 byte length does not fit in u64")
}

fn byte_len_for_output_layout(pixel_count: usize, output_layout: GpuOutputLayout) -> Result<u64> {
    match output_layout {
        GpuOutputLayout::PackedU16Samples => byte_len_u32_count(packed_u16_word_count(pixel_count)),
    }
}

fn packed_u16_word_count(pixel_count: usize) -> usize {
    pixel_count.div_ceil(2)
}

fn expected_invocations_for_packed_output(macroblock_count: usize) -> Result<usize> {
    macroblock_count
        .checked_mul(MCRAW_DECODE_WORKGROUP_SIZE as usize)
        .context("expected invocation count overflow")
}

fn dispatch_grid(macroblock_count: usize, max_workgroups_per_dimension: u32) -> Result<(u32, u32)> {
    let max_x = max_workgroups_per_dimension.max(1) as usize;
    let dispatch_width = macroblock_count.min(max_x).max(1);
    let dispatch_height = macroblock_count.div_ceil(dispatch_width);

    if dispatch_height > max_x {
        anyhow::bail!(
            "dispatch grid exceeds device limit: macroblocks={}, max_per_dimension={}",
            macroblock_count,
            max_workgroups_per_dimension
        );
    }

    Ok((dispatch_width as u32, dispatch_height as u32))
}
