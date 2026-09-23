use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use mcraw4vulkan_core::FrameNumber;
use thiserror::Error;

use crate::McrawContainer;
use crate::error::McrawContainerError;

pub const NEAR_CONTIGUOUS_THRESHOLD_BYTES: u64 = 1024 * 1024;
// Chunked reads trade bounded gap overread for fewer storage reads. Frame views
// share Arc-backed chunk storage until the last view is dropped.
pub const PRODUCTION_PAYLOAD_READ_MODE: PayloadReadMode = PayloadReadMode::ChunkedOffsetPrefetch;
pub const PRODUCTION_PAYLOAD_PREFETCH_DEPTH: usize = 4;
pub const PRODUCTION_PAYLOAD_REORDER_WINDOW: usize = 128;
pub const PRODUCTION_PAYLOAD_CHUNK_GAP_THRESHOLD_BYTES: u64 = NEAR_CONTIGUOUS_THRESHOLD_BYTES;
pub const PRODUCTION_PAYLOAD_CHUNK_MAX_BYTES: u64 = 64 * 1024 * 1024;
pub const PRODUCTION_PAYLOAD_CHUNK_MAX_PAYLOADS: usize = 16;

#[derive(Debug, Error)]
pub enum PayloadFeederError {
    #[error("container error: {0}")]
    Container(#[from] McrawContainerError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid payload plan: {0}")]
    InvalidPlan(String),

    #[error("payload feeder worker failed: {0}")]
    Worker(String),

    #[error("payload feeder worker panicked")]
    WorkerPanicked,

    #[error("payload feeder channel disconnected")]
    ChannelDisconnected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadReadMode {
    Current,
    OffsetPrefetch,
    ChunkedOffsetPrefetch,
}

impl PayloadReadMode {
    pub fn parse_label(value: &str) -> Option<Self> {
        match value {
            "current" => Some(Self::Current),
            "offset-prefetch" => Some(Self::OffsetPrefetch),
            "chunked-offset-prefetch" => Some(Self::ChunkedOffsetPrefetch),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::OffsetPrefetch => "offset-prefetch",
            Self::ChunkedOffsetPrefetch => "chunked-offset-prefetch",
        }
    }

    pub fn uses_resequence(self) -> bool {
        matches!(self, Self::OffsetPrefetch | Self::ChunkedOffsetPrefetch)
    }

    pub fn requires_monotonic(self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadReadModeRequest {
    Current,
    OffsetPrefetch,
    ChunkedOffsetPrefetch,
    Adaptive,
}

impl PayloadReadModeRequest {
    pub fn parse_label(value: &str) -> Option<Self> {
        match value {
            "current" => Some(Self::Current),
            "offset-prefetch" => Some(Self::OffsetPrefetch),
            "chunked-offset-prefetch" => Some(Self::ChunkedOffsetPrefetch),
            "adaptive" => Some(Self::Adaptive),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::OffsetPrefetch => "offset-prefetch",
            Self::ChunkedOffsetPrefetch => "chunked-offset-prefetch",
            Self::Adaptive => "adaptive",
        }
    }

    pub fn concrete_mode(self) -> Option<PayloadReadMode> {
        match self {
            Self::Current => Some(PayloadReadMode::Current),
            Self::OffsetPrefetch => Some(PayloadReadMode::OffsetPrefetch),
            Self::ChunkedOffsetPrefetch => Some(PayloadReadMode::ChunkedOffsetPrefetch),
            Self::Adaptive => None,
        }
    }

    pub fn uses_resequence(self) -> bool {
        match self {
            Self::Current => false,
            Self::OffsetPrefetch | Self::ChunkedOffsetPrefetch | Self::Adaptive => true,
        }
    }

    pub fn requires_monotonic(self) -> bool {
        false
    }
}

impl From<PayloadReadMode> for PayloadReadModeRequest {
    fn from(mode: PayloadReadMode) -> Self {
        match mode {
            PayloadReadMode::Current => Self::Current,
            PayloadReadMode::OffsetPrefetch => Self::OffsetPrefetch,
            PayloadReadMode::ChunkedOffsetPrefetch => Self::ChunkedOffsetPrefetch,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadAdaptivePolicy {
    Throughput,
    Balanced,
    FewestReads,
}

impl PayloadAdaptivePolicy {
    pub fn parse_label(value: &str) -> Option<Self> {
        match value {
            "throughput" => Some(Self::Throughput),
            "balanced" => Some(Self::Balanced),
            "fewest-reads" | "fewest_reads" => Some(Self::FewestReads),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Throughput => "throughput",
            Self::Balanced => "balanced",
            Self::FewestReads => "fewest-reads",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PayloadAdaptiveOptions {
    pub policy: PayloadAdaptivePolicy,
    pub max_overread_ratio: f64,
    pub min_read_reduction: f64,
    pub max_retained_chunk_bytes: u64,
    pub min_frames_for_threaded_prefetch: usize,
}

impl Default for PayloadAdaptiveOptions {
    fn default() -> Self {
        Self {
            policy: PayloadAdaptivePolicy::Throughput,
            max_overread_ratio: 1.02,
            min_read_reduction: 2.0,
            max_retained_chunk_bytes: 1024 * 1024 * 1024,
            min_frames_for_threaded_prefetch: 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadAdaptiveReason {
    ExplicitMode,
    EmptyPlan,
    SingleFrame,
    ReorderWindowTooSmall,
    TinyRangeThroughputCurrent,
    ThroughputChunkedStats,
    ThroughputOffsetDefault,
    BalancedChunked,
    BalancedOffsetFallback,
    FewestReadsChunked,
    FewestReadsOffsetFallback,
    CurrentFallback,
}

impl PayloadAdaptiveReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::ExplicitMode => "explicit_mode",
            Self::EmptyPlan => "empty_plan",
            Self::SingleFrame => "single_frame",
            Self::ReorderWindowTooSmall => "reorder_window_too_small",
            Self::TinyRangeThroughputCurrent => "tiny_range_throughput_current",
            Self::ThroughputChunkedStats => "throughput_chunked_stats",
            Self::ThroughputOffsetDefault => "throughput_offset_default",
            Self::BalancedChunked => "balanced_chunked",
            Self::BalancedOffsetFallback => "balanced_offset_fallback",
            Self::FewestReadsChunked => "fewest_reads_chunked",
            Self::FewestReadsOffsetFallback => "fewest_reads_offset_fallback",
            Self::CurrentFallback => "current_fallback",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PayloadAdaptiveDecision {
    pub requested: PayloadReadModeRequest,
    pub selected: PayloadReadMode,
    pub policy: PayloadAdaptivePolicy,
    pub reason: PayloadAdaptiveReason,
    pub candidate_current_allowed: bool,
    pub candidate_offset_allowed: bool,
    pub candidate_chunked_allowed: bool,
    pub selected_frames: usize,
    pub estimated_current_reads: usize,
    pub estimated_offset_reads: usize,
    pub estimated_chunked_reads: usize,
    pub estimated_read_reduction: f64,
    pub chunk_overread_ratio: Option<f64>,
    pub estimated_retained_chunk_bytes: u64,
    pub max_reorder_distance_frames: usize,
    pub reorder_window: usize,
    pub max_overread_ratio: f64,
    pub min_read_reduction: f64,
    pub max_retained_chunk_bytes: u64,
    pub min_frames_for_threaded_prefetch: usize,
}

impl PayloadAdaptiveDecision {
    pub fn resolve(
        requested: PayloadReadModeRequest,
        plan: &PayloadReadPlan,
        feeder_options: PayloadFeederOptions,
        adaptive_options: PayloadAdaptiveOptions,
    ) -> Result<Self, PayloadFeederError> {
        if let Some(selected) = requested.concrete_mode() {
            return Self::explicit(requested, selected, plan, feeder_options, adaptive_options);
        }

        let selected_frames = plan.selected_frames();
        let chunk_stats = plan.chunk_layout_stats(feeder_options.chunk_options)?;
        let common = AdaptiveCommon::new(plan, feeder_options, adaptive_options, chunk_stats);

        if selected_frames == 0 {
            return Ok(common.decision(
                requested,
                PayloadReadMode::Current,
                PayloadAdaptiveReason::EmptyPlan,
            ));
        }
        if selected_frames == 1 {
            return Ok(common.decision(
                requested,
                PayloadReadMode::Current,
                PayloadAdaptiveReason::SingleFrame,
            ));
        }
        if !common.candidate_offset_allowed {
            return Ok(common.decision(
                requested,
                PayloadReadMode::Current,
                PayloadAdaptiveReason::ReorderWindowTooSmall,
            ));
        }

        let decision = match adaptive_options.policy {
            PayloadAdaptivePolicy::Throughput => {
                if selected_frames <= adaptive_options.min_frames_for_threaded_prefetch {
                    common.decision(
                        requested,
                        PayloadReadMode::Current,
                        PayloadAdaptiveReason::TinyRangeThroughputCurrent,
                    )
                } else if common.candidate_chunked_allowed
                    && common.estimated_read_reduction >= 8.0
                    && common.chunk_overread_ratio <= 1.002
                    && selected_frames >= 120
                {
                    common.decision(
                        requested,
                        PayloadReadMode::ChunkedOffsetPrefetch,
                        PayloadAdaptiveReason::ThroughputChunkedStats,
                    )
                } else {
                    common.decision(
                        requested,
                        PayloadReadMode::OffsetPrefetch,
                        PayloadAdaptiveReason::ThroughputOffsetDefault,
                    )
                }
            }
            PayloadAdaptivePolicy::Balanced => {
                if common.candidate_chunked_allowed
                    && common.estimated_read_reduction >= adaptive_options.min_read_reduction
                {
                    common.decision(
                        requested,
                        PayloadReadMode::ChunkedOffsetPrefetch,
                        PayloadAdaptiveReason::BalancedChunked,
                    )
                } else {
                    common.decision(
                        requested,
                        PayloadReadMode::OffsetPrefetch,
                        PayloadAdaptiveReason::BalancedOffsetFallback,
                    )
                }
            }
            PayloadAdaptivePolicy::FewestReads => {
                if common.candidate_chunked_allowed {
                    common.decision(
                        requested,
                        PayloadReadMode::ChunkedOffsetPrefetch,
                        PayloadAdaptiveReason::FewestReadsChunked,
                    )
                } else {
                    common.decision(
                        requested,
                        PayloadReadMode::OffsetPrefetch,
                        PayloadAdaptiveReason::FewestReadsOffsetFallback,
                    )
                }
            }
        };
        Ok(decision)
    }

    fn explicit(
        requested: PayloadReadModeRequest,
        selected: PayloadReadMode,
        plan: &PayloadReadPlan,
        feeder_options: PayloadFeederOptions,
        adaptive_options: PayloadAdaptiveOptions,
    ) -> Result<Self, PayloadFeederError> {
        let chunk_stats = plan.chunk_layout_stats(feeder_options.chunk_options)?;
        let common = AdaptiveCommon::new(plan, feeder_options, adaptive_options, chunk_stats);
        Ok(common.decision(requested, selected, PayloadAdaptiveReason::ExplicitMode))
    }
}

struct AdaptiveCommon {
    selected_frames: usize,
    estimated_current_reads: usize,
    estimated_offset_reads: usize,
    estimated_chunked_reads: usize,
    estimated_read_reduction: f64,
    chunk_overread_ratio: f64,
    estimated_retained_chunk_bytes: u64,
    max_reorder_distance_frames: usize,
    reorder_window: usize,
    candidate_offset_allowed: bool,
    candidate_chunked_allowed: bool,
    adaptive_options: PayloadAdaptiveOptions,
}

impl AdaptiveCommon {
    fn new(
        plan: &PayloadReadPlan,
        feeder_options: PayloadFeederOptions,
        adaptive_options: PayloadAdaptiveOptions,
        chunk_stats: PayloadChunkLayoutStats,
    ) -> Self {
        let selected_frames = plan.selected_frames();
        let estimated_chunked_reads = chunk_stats.chunk_count;
        let estimated_read_reduction = if estimated_chunked_reads == 0 {
            0.0
        } else {
            selected_frames as f64 / estimated_chunked_reads as f64
        };
        let chunk_overread_ratio = chunk_stats.chunk_overread_ratio();
        let estimated_retained_chunk_bytes = chunk_stats
            .chunk_max_bytes_observed
            .saturating_mul(feeder_options.prefetch_depth as u64);
        let candidate_offset_allowed =
            plan.max_reorder_distance_frames <= feeder_options.reorder_window;
        let candidate_chunked_allowed = candidate_offset_allowed
            && estimated_chunked_reads > 0
            && chunk_overread_ratio <= adaptive_options.max_overread_ratio
            && estimated_retained_chunk_bytes <= adaptive_options.max_retained_chunk_bytes;
        Self {
            selected_frames,
            estimated_current_reads: selected_frames,
            estimated_offset_reads: selected_frames,
            estimated_chunked_reads,
            estimated_read_reduction,
            chunk_overread_ratio,
            estimated_retained_chunk_bytes,
            max_reorder_distance_frames: plan.max_reorder_distance_frames,
            reorder_window: feeder_options.reorder_window,
            candidate_offset_allowed,
            candidate_chunked_allowed,
            adaptive_options,
        }
    }

    fn decision(
        &self,
        requested: PayloadReadModeRequest,
        selected: PayloadReadMode,
        reason: PayloadAdaptiveReason,
    ) -> PayloadAdaptiveDecision {
        PayloadAdaptiveDecision {
            requested,
            selected,
            policy: self.adaptive_options.policy,
            reason,
            candidate_current_allowed: true,
            candidate_offset_allowed: self.candidate_offset_allowed,
            candidate_chunked_allowed: self.candidate_chunked_allowed,
            selected_frames: self.selected_frames,
            estimated_current_reads: self.estimated_current_reads,
            estimated_offset_reads: self.estimated_offset_reads,
            estimated_chunked_reads: self.estimated_chunked_reads,
            estimated_read_reduction: self.estimated_read_reduction,
            chunk_overread_ratio: if self.estimated_chunked_reads == 0 {
                None
            } else {
                Some(self.chunk_overread_ratio)
            },
            estimated_retained_chunk_bytes: self.estimated_retained_chunk_bytes,
            max_reorder_distance_frames: self.max_reorder_distance_frames,
            reorder_window: self.reorder_window,
            max_overread_ratio: self.adaptive_options.max_overread_ratio,
            min_read_reduction: self.adaptive_options.min_read_reduction,
            max_retained_chunk_bytes: self.adaptive_options.max_retained_chunk_bytes,
            min_frames_for_threaded_prefetch: self
                .adaptive_options
                .min_frames_for_threaded_prefetch,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadChunkOptions {
    pub gap_threshold_bytes: u64,
    pub max_chunk_bytes: u64,
    pub max_payloads_per_chunk: usize,
}

impl PayloadChunkOptions {
    pub const fn new(
        gap_threshold_bytes: u64,
        max_chunk_bytes: u64,
        max_payloads_per_chunk: usize,
    ) -> Self {
        Self {
            gap_threshold_bytes,
            max_chunk_bytes,
            max_payloads_per_chunk,
        }
    }

    pub const fn production_default() -> Self {
        Self {
            gap_threshold_bytes: PRODUCTION_PAYLOAD_CHUNK_GAP_THRESHOLD_BYTES,
            max_chunk_bytes: PRODUCTION_PAYLOAD_CHUNK_MAX_BYTES,
            max_payloads_per_chunk: PRODUCTION_PAYLOAD_CHUNK_MAX_PAYLOADS,
        }
    }
}

impl Default for PayloadChunkOptions {
    fn default() -> Self {
        Self::production_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadFeederOptions {
    pub mode: PayloadReadMode,
    pub prefetch_depth: usize,
    pub reorder_window: usize,
    pub chunk_options: PayloadChunkOptions,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PayloadFeederSpawnPhaseTiming {
    pub start_offset: Duration,
    pub duration: Duration,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PayloadFeederSpawnTimings {
    pub total: Duration,
    pub chunk_plan_from_plan: Option<PayloadFeederSpawnPhaseTiming>,
}

impl PayloadFeederOptions {
    pub const fn production_default() -> Self {
        Self {
            mode: PRODUCTION_PAYLOAD_READ_MODE,
            prefetch_depth: PRODUCTION_PAYLOAD_PREFETCH_DEPTH,
            reorder_window: PRODUCTION_PAYLOAD_REORDER_WINDOW,
            chunk_options: PayloadChunkOptions::production_default(),
        }
    }
}

impl Default for PayloadFeederOptions {
    fn default() -> Self {
        Self::production_default()
    }
}

#[derive(Debug, Clone)]
pub struct PayloadReadPlan {
    pub entries: Vec<PayloadFrameRequest>,
    pub frames: Vec<PayloadFrameRequest>,
    pub stats: PayloadLayoutStats,
    pub total_payload_bytes: u64,
    pub monotonic_offsets: bool,
    pub total_forward_gap_bytes: u64,
    pub backward_seek_count: usize,
    pub non_monotonic_pairs: usize,
    pub max_backward_distance_bytes: u64,
    pub contiguous_pair_count: usize,
    pub near_contiguous_pair_count: usize,
    pub near_contiguous_threshold_bytes: u64,
    pub max_gap_bytes: u64,
    pub max_payload_bytes: u64,
    pub first_offset: Option<u64>,
    pub last_end_offset: Option<u64>,
    pub estimated_span_bytes: u64,
    pub physical_span_bytes: u64,
    pub physical_sorted_total_forward_gap_bytes: u64,
    pub offset_sorted_contiguous_pair_count: usize,
    pub offset_sorted_near_contiguous_pair_count: usize,
    pub max_reorder_distance_frames: usize,
    pub max_reorder_window_observed: usize,
}

impl PayloadReadPlan {
    pub fn empty() -> Self {
        Self::from_requests(Vec::new()).expect("empty payload read plan is valid")
    }

    pub fn from_frame_numbers(
        container: &McrawContainer,
        frame_numbers: &[usize],
    ) -> Result<Self, PayloadFeederError> {
        let mut entries = Vec::with_capacity(frame_numbers.len());
        for (playback_index, &frame_number) in frame_numbers.iter().enumerate() {
            let frame_u32 = u32::try_from(frame_number).map_err(|_| {
                PayloadFeederError::InvalidPlan(format!(
                    "frame number {frame_number} does not fit in u32"
                ))
            })?;
            let span = container.video_payload_span(FrameNumber(frame_u32))?;
            let payload_len = usize::try_from(span.len).map_err(|_| {
                PayloadFeederError::InvalidPlan(format!(
                    "payload length for frame {frame_number} overflows usize"
                ))
            })?;
            entries.push(PayloadFrameRequest {
                playback_index,
                frame_number,
                payload_offset: span.offset,
                payload_len,
            });
        }
        Self::from_requests(entries)
    }

    pub fn from_core_frame_numbers(
        container: &McrawContainer,
        frame_numbers: &[FrameNumber],
    ) -> Result<Self, PayloadFeederError> {
        let indices: Vec<usize> = frame_numbers.iter().map(|frame| frame.0 as usize).collect();
        Self::from_frame_numbers(container, &indices)
    }

    pub fn selected_frames(&self) -> usize {
        self.entries.len()
    }

    pub fn entries_sorted_by_offset(&self) -> Vec<PayloadFrameRequest> {
        let mut entries = self.entries.clone();
        entries.sort_by_key(|entry| {
            (
                entry.payload_offset,
                entry.payload_len,
                entry.playback_index,
            )
        });
        entries
    }

    pub fn chunk_layout_stats(
        &self,
        options: PayloadChunkOptions,
    ) -> Result<PayloadChunkLayoutStats, PayloadFeederError> {
        Ok(PayloadChunkPlan::from_plan(self, options)?.stats)
    }

    fn from_requests(entries: Vec<PayloadFrameRequest>) -> Result<Self, PayloadFeederError> {
        let stats = PayloadLayoutStats::from_entries(&entries)?;
        Ok(Self {
            entries: entries.clone(),
            frames: entries,
            total_payload_bytes: stats.total_payload_bytes,
            monotonic_offsets: stats.monotonic_offsets,
            total_forward_gap_bytes: stats.total_forward_gap_bytes,
            backward_seek_count: stats.backward_seek_count,
            non_monotonic_pairs: stats.non_monotonic_pairs,
            max_backward_distance_bytes: stats.max_backward_distance_bytes,
            contiguous_pair_count: stats.contiguous_pair_count,
            near_contiguous_pair_count: stats.near_contiguous_pair_count,
            near_contiguous_threshold_bytes: stats.near_contiguous_threshold_bytes,
            max_gap_bytes: stats.max_gap_bytes,
            max_payload_bytes: stats.max_payload_bytes,
            first_offset: stats.first_offset,
            last_end_offset: stats.last_end_offset,
            estimated_span_bytes: stats.estimated_span_bytes,
            physical_span_bytes: stats.physical_span_bytes,
            physical_sorted_total_forward_gap_bytes: stats.physical_sorted_total_forward_gap_bytes,
            offset_sorted_contiguous_pair_count: stats.offset_sorted_contiguous_pair_count,
            offset_sorted_near_contiguous_pair_count: stats
                .offset_sorted_near_contiguous_pair_count,
            max_reorder_distance_frames: stats.max_reorder_distance_frames,
            max_reorder_window_observed: stats.max_reorder_window_observed,
            stats,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadFrameRequest {
    pub playback_index: usize,
    pub frame_number: usize,
    pub payload_offset: u64,
    pub payload_len: usize,
}

impl PayloadFrameRequest {
    pub fn end_offset(self) -> Result<u64, PayloadFeederError> {
        self.payload_offset
            .checked_add(self.payload_len as u64)
            .ok_or_else(|| PayloadFeederError::InvalidPlan("payload end offset overflow".into()))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PayloadLayoutStats {
    pub selected_frames: usize,
    pub total_payload_bytes: u64,
    pub monotonic_offsets: bool,
    pub total_forward_gap_bytes: u64,
    pub backward_seek_count: usize,
    pub non_monotonic_pairs: usize,
    pub max_backward_distance_bytes: u64,
    pub contiguous_pair_count: usize,
    pub near_contiguous_pair_count: usize,
    pub near_contiguous_threshold_bytes: u64,
    pub max_gap_bytes: u64,
    pub max_payload_bytes: u64,
    pub first_offset: Option<u64>,
    pub last_end_offset: Option<u64>,
    pub estimated_span_bytes: u64,
    pub physical_span_bytes: u64,
    pub physical_sorted_total_forward_gap_bytes: u64,
    pub offset_sorted_contiguous_pair_count: usize,
    pub offset_sorted_near_contiguous_pair_count: usize,
    pub max_reorder_distance_frames: usize,
    pub max_reorder_window_observed: usize,
}

impl PayloadLayoutStats {
    fn from_entries(entries: &[PayloadFrameRequest]) -> Result<Self, PayloadFeederError> {
        let mut stats = Self {
            selected_frames: entries.len(),
            monotonic_offsets: true,
            near_contiguous_threshold_bytes: NEAR_CONTIGUOUS_THRESHOLD_BYTES,
            ..Self::default()
        };

        let mut previous_end = None;
        for entry in entries {
            let end = entry.end_offset()?;
            stats.total_payload_bytes = stats
                .total_payload_bytes
                .checked_add(entry.payload_len as u64)
                .ok_or_else(|| {
                    PayloadFeederError::InvalidPlan("payload total byte count overflow".into())
                })?;
            stats.max_payload_bytes = stats.max_payload_bytes.max(entry.payload_len as u64);

            if let Some(previous_end) = previous_end {
                if entry.payload_offset < previous_end {
                    stats.monotonic_offsets = false;
                    stats.backward_seek_count += 1;
                    stats.non_monotonic_pairs += 1;
                    stats.max_backward_distance_bytes = stats
                        .max_backward_distance_bytes
                        .max(previous_end - entry.payload_offset);
                } else {
                    let gap = entry.payload_offset - previous_end;
                    stats.total_forward_gap_bytes = stats
                        .total_forward_gap_bytes
                        .checked_add(gap)
                        .ok_or_else(|| {
                            PayloadFeederError::InvalidPlan(
                                "payload forward gap byte count overflow".into(),
                            )
                        })?;
                    stats.max_gap_bytes = stats.max_gap_bytes.max(gap);
                    if gap == 0 {
                        stats.contiguous_pair_count += 1;
                    }
                    if gap <= NEAR_CONTIGUOUS_THRESHOLD_BYTES {
                        stats.near_contiguous_pair_count += 1;
                    }
                }
            }

            previous_end = Some(end);
        }

        stats.first_offset = entries.first().map(|entry| entry.payload_offset);
        stats.last_end_offset = entries.last().map(|entry| entry.end_offset()).transpose()?;
        stats.estimated_span_bytes = match (stats.first_offset, stats.last_end_offset) {
            (Some(first), Some(last)) if last >= first => last - first,
            _ => 0,
        };

        let mut physical_entries = entries.to_vec();
        physical_entries.sort_by_key(|entry| {
            (
                entry.payload_offset,
                entry.payload_len,
                entry.playback_index,
            )
        });

        let mut previous_physical_end = None;
        for entry in &physical_entries {
            let end = entry.end_offset()?;
            if let Some(previous_end) = previous_physical_end {
                if entry.payload_offset >= previous_end {
                    let gap = entry.payload_offset - previous_end;
                    stats.physical_sorted_total_forward_gap_bytes = stats
                        .physical_sorted_total_forward_gap_bytes
                        .checked_add(gap)
                        .ok_or_else(|| {
                            PayloadFeederError::InvalidPlan(
                                "physical payload gap byte count overflow".into(),
                            )
                        })?;
                    if gap == 0 {
                        stats.offset_sorted_contiguous_pair_count += 1;
                    }
                    if gap <= NEAR_CONTIGUOUS_THRESHOLD_BYTES {
                        stats.offset_sorted_near_contiguous_pair_count += 1;
                    }
                }
            }
            previous_physical_end = Some(end);
        }

        let physical_first_offset = physical_entries.first().map(|entry| entry.payload_offset);
        let physical_last_end_offset = physical_entries
            .last()
            .map(|entry| entry.end_offset())
            .transpose()?;
        stats.physical_span_bytes = match (physical_first_offset, physical_last_end_offset) {
            (Some(first), Some(last)) if last >= first => last - first,
            _ => 0,
        };
        stats.max_reorder_distance_frames = physical_entries
            .iter()
            .enumerate()
            .map(|(physical_index, entry)| physical_index.abs_diff(entry.playback_index))
            .max()
            .unwrap_or(0);
        stats.max_reorder_window_observed = stats.max_reorder_distance_frames;

        Ok(stats)
    }
}

// Chunk slices own their backing Arc, so returned frames remain valid after the
// feeder advances or drops. into_owned copies only the shared-slice variant.
#[derive(Debug, Clone)]
pub enum PayloadFrameData {
    Owned(Vec<u8>),
    ChunkSlice {
        chunk: Arc<[u8]>,
        range: Range<usize>,
    },
}

impl PayloadFrameData {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes.as_slice(),
            Self::ChunkSlice { chunk, range } => &chunk[range.clone()],
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn copy_into(&self, dst: &mut Vec<u8>) {
        dst.clear();
        dst.extend_from_slice(self.as_slice());
    }

    pub fn into_owned(self) -> Vec<u8> {
        match self {
            Self::Owned(bytes) => bytes,
            Self::ChunkSlice { chunk, range } => chunk[range].to_vec(),
        }
    }

    pub fn backing_kind(&self) -> &'static str {
        match self {
            Self::Owned(_) => "owned",
            Self::ChunkSlice { .. } => "chunk-slice",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PayloadFrame {
    pub playback_index: usize,
    pub frame_number: usize,
    pub data: PayloadFrameData,
}

impl PayloadFrame {
    pub fn frame_number_core(&self) -> Result<FrameNumber, PayloadFeederError> {
        let frame = u32::try_from(self.frame_number).map_err(|_| {
            PayloadFeederError::InvalidPlan(format!(
                "frame number {} does not fit in u32",
                self.frame_number
            ))
        })?;
        Ok(FrameNumber(frame))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PayloadChunkLayoutStats {
    pub chunk_count: usize,
    pub chunk_total_bytes: u64,
    pub chunk_payload_bytes: u64,
    pub chunk_gap_bytes_read: u64,
    pub chunk_max_bytes_observed: u64,
    pub chunk_max_payloads_observed: usize,
    pub chunk_single_payload_count: usize,
}

impl PayloadChunkLayoutStats {
    pub fn chunk_overread_ratio(self) -> f64 {
        if self.chunk_payload_bytes == 0 {
            1.0
        } else {
            self.chunk_total_bytes as f64 / self.chunk_payload_bytes as f64
        }
    }
}

#[derive(Debug, Clone)]
pub struct PayloadFeederStats {
    pub mode: PayloadReadMode,
    pub selected_frames: usize,
    pub seeks: usize,
    pub read_calls: usize,
    pub bytes_read: u64,
    pub producer_read_time: Duration,
    pub receiver_wait_time: Duration,
    pub max_depth_observed: usize,
    pub messages_received: usize,
    pub physical_read_order: bool,
    pub output_playback_order: bool,
    pub resequence_buffer_max_len: usize,
    pub resequence_window_exceeded: bool,
    pub chunked_reads: bool,
    pub chunk_gap_threshold_bytes: u64,
    pub chunk_max_bytes_configured: u64,
    pub chunk_max_payloads_configured: usize,
    pub chunk_count: usize,
    pub chunk_total_bytes: u64,
    pub chunk_payload_bytes: u64,
    pub chunk_gap_bytes_read: u64,
    pub chunk_max_bytes_observed: u64,
    pub chunk_max_payloads_observed: usize,
    pub chunk_single_payload_count: usize,
    pub producer_chunk_read_calls: usize,
    pub producer_payload_messages_sent: usize,
    pub producer_payload_bytes: u64,
    pub producer_gap_bytes_read: u64,
    pub payload_bytes_returned: u64,
    pub copies_to_owned: u64,
    pub retained_chunk_bytes_estimate: u64,
    pub owned_frames: usize,
    pub chunk_slice_frames: usize,
}

impl PayloadFeederStats {
    fn new(
        mode: PayloadReadMode,
        selected_frames: usize,
        chunk_options: PayloadChunkOptions,
    ) -> Self {
        Self {
            mode,
            selected_frames,
            physical_read_order: mode != PayloadReadMode::Current,
            output_playback_order: true,
            chunk_gap_threshold_bytes: chunk_options.gap_threshold_bytes,
            chunk_max_bytes_configured: chunk_options.max_chunk_bytes,
            chunk_max_payloads_configured: chunk_options.max_payloads_per_chunk,
            ..Self::default()
        }
    }

    pub fn chunk_overread_ratio(&self) -> f64 {
        if self.chunk_payload_bytes == 0 {
            1.0
        } else {
            self.chunk_total_bytes as f64 / self.chunk_payload_bytes as f64
        }
    }

    pub fn frame_data_backing(&self) -> &'static str {
        match (self.owned_frames > 0, self.chunk_slice_frames > 0) {
            (false, false) => "none",
            (true, false) => "owned",
            (false, true) => "chunk-slice",
            (true, true) => "mixed",
        }
    }

    fn merge_worker_done(&mut self, done: Self) {
        self.seeks = done.seeks;
        self.read_calls = done.read_calls;
        self.bytes_read = done.bytes_read;
        self.producer_read_time = self.producer_read_time.max(done.producer_read_time);
        self.max_depth_observed = self.max_depth_observed.max(done.max_depth_observed);
        self.physical_read_order = done.physical_read_order;
        self.output_playback_order &= done.output_playback_order;
        self.chunked_reads |= done.chunked_reads;
        self.chunk_count = self.chunk_count.max(done.chunk_count);
        self.chunk_total_bytes = self.chunk_total_bytes.max(done.chunk_total_bytes);
        self.chunk_payload_bytes = self.chunk_payload_bytes.max(done.chunk_payload_bytes);
        self.chunk_gap_bytes_read = self.chunk_gap_bytes_read.max(done.chunk_gap_bytes_read);
        self.chunk_max_bytes_observed = self
            .chunk_max_bytes_observed
            .max(done.chunk_max_bytes_observed);
        self.chunk_max_payloads_observed = self
            .chunk_max_payloads_observed
            .max(done.chunk_max_payloads_observed);
        self.chunk_single_payload_count = self
            .chunk_single_payload_count
            .max(done.chunk_single_payload_count);
        self.producer_chunk_read_calls = self
            .producer_chunk_read_calls
            .max(done.producer_chunk_read_calls);
        self.producer_payload_messages_sent = self
            .producer_payload_messages_sent
            .max(done.producer_payload_messages_sent);
        self.producer_payload_bytes = self.producer_payload_bytes.max(done.producer_payload_bytes);
        self.producer_gap_bytes_read = self
            .producer_gap_bytes_read
            .max(done.producer_gap_bytes_read);
        self.retained_chunk_bytes_estimate = self
            .retained_chunk_bytes_estimate
            .max(done.retained_chunk_bytes_estimate);
    }
}

impl Default for PayloadFeederStats {
    fn default() -> Self {
        Self {
            mode: PayloadReadMode::Current,
            selected_frames: 0,
            seeks: 0,
            read_calls: 0,
            bytes_read: 0,
            producer_read_time: Duration::ZERO,
            receiver_wait_time: Duration::ZERO,
            max_depth_observed: 0,
            messages_received: 0,
            physical_read_order: false,
            output_playback_order: true,
            resequence_buffer_max_len: 0,
            resequence_window_exceeded: false,
            chunked_reads: false,
            chunk_gap_threshold_bytes: 0,
            chunk_max_bytes_configured: 0,
            chunk_max_payloads_configured: 0,
            chunk_count: 0,
            chunk_total_bytes: 0,
            chunk_payload_bytes: 0,
            chunk_gap_bytes_read: 0,
            chunk_max_bytes_observed: 0,
            chunk_max_payloads_observed: 0,
            chunk_single_payload_count: 0,
            producer_chunk_read_calls: 0,
            producer_payload_messages_sent: 0,
            producer_payload_bytes: 0,
            producer_gap_bytes_read: 0,
            payload_bytes_returned: 0,
            copies_to_owned: 0,
            retained_chunk_bytes_estimate: 0,
            owned_frames: 0,
            chunk_slice_frames: 0,
        }
    }
}

// Owns either a synchronous reader or a background worker. Offset modes read in
// physical order, then a bounded resequence buffer restores playback order;
// exceeding that bound is an error instead of unbounded buffering.
pub struct PayloadFeeder {
    current: Option<CurrentPayloadReader>,
    receiver: Option<Receiver<PayloadFeederMessage>>,
    join_handle: Option<JoinHandle<()>>,
    queued_count: Option<Arc<AtomicUsize>>,
    next_playback_index: usize,
    plan_len: usize,
    reorder_window: usize,
    resequence_buffer: BTreeMap<usize, PayloadFrame>,
    stats: PayloadFeederStats,
    completed_stats: Option<PayloadFeederStats>,
}

#[derive(Debug)]
pub enum PayloadFeederPoll {
    Ready(Option<PayloadFrame>),
    Pending,
}

impl PayloadFeeder {
    pub fn spawn(
        path: impl AsRef<Path>,
        plan: PayloadReadPlan,
        options: PayloadFeederOptions,
    ) -> Result<Self, PayloadFeederError> {
        Self::spawn_inner(path, plan, options, None)
    }

    pub fn spawn_with_timing(
        path: impl AsRef<Path>,
        plan: PayloadReadPlan,
        options: PayloadFeederOptions,
        timings: &mut PayloadFeederSpawnTimings,
    ) -> Result<Self, PayloadFeederError> {
        *timings = PayloadFeederSpawnTimings::default();
        Self::spawn_inner(path, plan, options, Some(timings))
    }

    fn spawn_inner(
        path: impl AsRef<Path>,
        plan: PayloadReadPlan,
        options: PayloadFeederOptions,
        mut timings: Option<&mut PayloadFeederSpawnTimings>,
    ) -> Result<Self, PayloadFeederError> {
        let total_start = timings.as_ref().map(|_| Instant::now());
        if options.prefetch_depth == 0 {
            return Err(PayloadFeederError::InvalidPlan(
                "payload prefetch depth must be at least 1".into(),
            ));
        }
        if options.reorder_window == 0 {
            return Err(PayloadFeederError::InvalidPlan(
                "payload reorder window must be at least 1".into(),
            ));
        }
        if options.mode.uses_resequence() && options.reorder_window < options.prefetch_depth {
            return Err(PayloadFeederError::InvalidPlan(
                "payload reorder window must be >= prefetch depth".into(),
            ));
        }
        if options.chunk_options.max_chunk_bytes == 0 {
            return Err(PayloadFeederError::InvalidPlan(
                "payload max chunk bytes must be at least 1".into(),
            ));
        }
        if options.chunk_options.max_payloads_per_chunk == 0 {
            return Err(PayloadFeederError::InvalidPlan(
                "payload max payloads per chunk must be at least 1".into(),
            ));
        }

        let path = path.as_ref().to_path_buf();
        let result = match options.mode {
            PayloadReadMode::Current => Self::new_current(path, plan, options),
            PayloadReadMode::OffsetPrefetch => Self::spawn_offset(path, plan, options, None, false),
            PayloadReadMode::ChunkedOffsetPrefetch => {
                let chunk_plan_start = total_start.map(|_| Instant::now());
                let chunk_plan = PayloadChunkPlan::from_plan(&plan, options.chunk_options)?;
                if let (Some(total_start), Some(chunk_plan_start), Some(timings)) =
                    (total_start, chunk_plan_start, timings.as_deref_mut())
                {
                    timings.chunk_plan_from_plan = Some(PayloadFeederSpawnPhaseTiming {
                        start_offset: chunk_plan_start.duration_since(total_start),
                        duration: chunk_plan_start.elapsed(),
                    });
                }
                Self::spawn_offset(path, plan, options, Some(chunk_plan), true)
            }
        };
        if let (Some(total_start), Some(timings)) = (total_start, timings) {
            timings.total = total_start.elapsed();
        }
        result
    }

    pub fn next_frame(&mut self) -> Result<Option<PayloadFrame>, PayloadFeederError> {
        if let Some(current) = self.current.as_mut() {
            let frame = current.next_frame(&mut self.stats)?;
            if let Some(frame) = &frame {
                self.next_playback_index = frame.playback_index + 1;
            }
            return Ok(frame);
        }

        if self.next_playback_index >= self.plan_len {
            return Ok(None);
        }

        let wait_start = Instant::now();
        let frame = loop {
            if let Some(frame) = self.resequence_buffer.remove(&self.next_playback_index) {
                break frame;
            }

            match self.recv_message()? {
                PayloadFeederMessage::Frame(frame) => {
                    if frame.playback_index < self.next_playback_index {
                        return Err(PayloadFeederError::Worker(format!(
                            "payload feeder produced stale playback index {}",
                            frame.playback_index
                        )));
                    }
                    if frame.playback_index == self.next_playback_index {
                        break frame;
                    }
                    if self.resequence_buffer.contains_key(&frame.playback_index) {
                        return Err(PayloadFeederError::Worker(format!(
                            "payload feeder produced duplicate playback index {}",
                            frame.playback_index
                        )));
                    }
                    if self.resequence_buffer.len() >= self.reorder_window {
                        self.stats.resequence_window_exceeded = true;
                        return Err(PayloadFeederError::Worker(format!(
                            "payload feeder resequence window exceeded: window={} incoming_playback_index={}",
                            self.reorder_window, frame.playback_index
                        )));
                    }
                    self.resequence_buffer.insert(frame.playback_index, frame);
                    self.stats.resequence_buffer_max_len = self
                        .stats
                        .resequence_buffer_max_len
                        .max(self.resequence_buffer.len());
                }
                PayloadFeederMessage::Error(error) => {
                    return Err(PayloadFeederError::Worker(error));
                }
                PayloadFeederMessage::Done(done) => {
                    self.completed_stats = Some(done);
                    return Err(PayloadFeederError::Worker(format!(
                        "payload feeder ended before playback index {}",
                        self.next_playback_index
                    )));
                }
            }
        };

        self.stats.receiver_wait_time += wait_start.elapsed();
        self.stats.messages_received += 1;
        self.record_frame_backing(&frame);
        self.next_playback_index += 1;
        Ok(Some(frame))
    }

    pub fn try_next_frame(&mut self) -> Result<PayloadFeederPoll, PayloadFeederError> {
        if self.current.is_some() {
            return self.next_frame().map(PayloadFeederPoll::Ready);
        }

        if self.next_playback_index >= self.plan_len {
            return Ok(PayloadFeederPoll::Ready(None));
        }

        let wait_start = Instant::now();
        loop {
            if let Some(frame) = self.resequence_buffer.remove(&self.next_playback_index) {
                return Ok(PayloadFeederPoll::Ready(Some(
                    self.complete_ready_frame(frame, wait_start),
                )));
            }

            let Some(message) = self.try_recv_message()? else {
                return Ok(PayloadFeederPoll::Pending);
            };

            match message {
                PayloadFeederMessage::Frame(frame) => {
                    if frame.playback_index < self.next_playback_index {
                        return Err(PayloadFeederError::Worker(format!(
                            "payload feeder produced stale playback index {}",
                            frame.playback_index
                        )));
                    }
                    if frame.playback_index == self.next_playback_index {
                        return Ok(PayloadFeederPoll::Ready(Some(
                            self.complete_ready_frame(frame, wait_start),
                        )));
                    }
                    if self.resequence_buffer.contains_key(&frame.playback_index) {
                        return Err(PayloadFeederError::Worker(format!(
                            "payload feeder produced duplicate playback index {}",
                            frame.playback_index
                        )));
                    }
                    if self.resequence_buffer.len() >= self.reorder_window {
                        self.stats.resequence_window_exceeded = true;
                        return Err(PayloadFeederError::Worker(format!(
                            "payload feeder resequence window exceeded: window={} incoming_playback_index={}",
                            self.reorder_window, frame.playback_index
                        )));
                    }
                    self.resequence_buffer.insert(frame.playback_index, frame);
                    self.stats.resequence_buffer_max_len = self
                        .stats
                        .resequence_buffer_max_len
                        .max(self.resequence_buffer.len());
                }
                PayloadFeederMessage::Error(error) => {
                    return Err(PayloadFeederError::Worker(error));
                }
                PayloadFeederMessage::Done(done) => {
                    self.completed_stats = Some(done);
                    return Err(PayloadFeederError::Worker(format!(
                        "payload feeder ended before playback index {}",
                        self.next_playback_index
                    )));
                }
            }
        }
    }

    // Dropping the receiver is the cancellation signal: a worker blocked on the
    // bounded channel wakes with send failure and exits at its next message.
    pub fn request_stop(&mut self) {
        self.current = None;
        self.receiver = None;
    }

    pub fn try_join_after_stop(&mut self) -> Result<bool, PayloadFeederError> {
        self.request_stop();

        let Some(join_handle) = self.join_handle.as_ref() else {
            return Ok(true);
        };
        if !join_handle.is_finished() {
            return Ok(false);
        }

        let join_handle = self
            .join_handle
            .take()
            .expect("join handle was checked above");
        join_handle
            .join()
            .map_err(|_| PayloadFeederError::WorkerPanicked)?;
        Ok(true)
    }

    pub fn stats(&self) -> PayloadFeederStats {
        self.stats.clone()
    }

    pub fn finish(mut self) -> Result<PayloadFeederStats, PayloadFeederError> {
        if self.next_playback_index != self.plan_len {
            return Err(PayloadFeederError::Worker(format!(
                "payload feeder consumed {} frames but plan has {} frames",
                self.next_playback_index, self.plan_len
            )));
        }

        if self.current.is_none() && self.completed_stats.is_none() {
            match self.recv_message()? {
                PayloadFeederMessage::Done(done) => {
                    self.completed_stats = Some(done);
                }
                PayloadFeederMessage::Error(error) => {
                    return Err(PayloadFeederError::Worker(error));
                }
                PayloadFeederMessage::Frame(frame) => {
                    return Err(PayloadFeederError::Worker(format!(
                        "payload feeder produced extra frame at playback index {}",
                        frame.playback_index
                    )));
                }
            }
        }

        drop(self.receiver.take());
        if let Some(join_handle) = self.join_handle.take() {
            join_handle
                .join()
                .map_err(|_| PayloadFeederError::WorkerPanicked)?;
        }
        if let Some(done) = self.completed_stats.take() {
            self.stats.merge_worker_done(done);
        }
        Ok(self.stats.clone())
    }

    fn new_current(
        path: PathBuf,
        plan: PayloadReadPlan,
        options: PayloadFeederOptions,
    ) -> Result<Self, PayloadFeederError> {
        let file = File::open(path)?;
        let plan_len = plan.selected_frames();
        let mut stats =
            PayloadFeederStats::new(options.mode, plan.entries.len(), options.chunk_options);
        stats.physical_read_order = plan.monotonic_offsets;
        Ok(Self {
            current: Some(CurrentPayloadReader {
                file,
                current_offset: 0,
                entries: plan.entries,
                next_index: 0,
            }),
            receiver: None,
            join_handle: None,
            queued_count: None,
            next_playback_index: 0,
            plan_len,
            reorder_window: options.reorder_window,
            resequence_buffer: BTreeMap::new(),
            stats,
            completed_stats: None,
        })
    }

    fn spawn_offset(
        path: PathBuf,
        plan: PayloadReadPlan,
        options: PayloadFeederOptions,
        chunk_plan: Option<PayloadChunkPlan>,
        chunked: bool,
    ) -> Result<Self, PayloadFeederError> {
        let plan_len = plan.entries.len();
        let initial_stats = initial_worker_stats(options, plan_len, chunk_plan.as_ref());
        let (sender, receiver) = mpsc::sync_channel(options.prefetch_depth);
        let queued_count = Arc::new(AtomicUsize::new(0));
        let max_depth_observed = Arc::new(AtomicUsize::new(0));
        let worker_queued = queued_count.clone();
        let worker_max_depth = max_depth_observed.clone();
        let entries = if chunked {
            Vec::new()
        } else {
            plan.entries_sorted_by_offset()
        };
        let depth = options.prefetch_depth;
        let join_handle = thread::Builder::new()
            .name(if chunked {
                "mcraw4vulkan-payload-chunked-offset-prefetch".to_string()
            } else {
                "mcraw4vulkan-payload-offset-prefetch".to_string()
            })
            .spawn(move || {
                run_offset_worker(OffsetWorkerJob {
                    path,
                    entries,
                    chunk_plan,
                    stats: initial_stats,
                    sender,
                    queued_count: worker_queued,
                    max_depth_observed: worker_max_depth,
                    depth,
                });
            })
            .map_err(PayloadFeederError::Io)?;

        Ok(Self {
            current: None,
            receiver: Some(receiver),
            join_handle: Some(join_handle),
            queued_count: Some(queued_count),
            next_playback_index: 0,
            plan_len,
            reorder_window: options.reorder_window,
            resequence_buffer: BTreeMap::new(),
            stats: PayloadFeederStats::new(options.mode, plan_len, options.chunk_options),
            completed_stats: None,
        })
    }

    fn recv_message(&mut self) -> Result<PayloadFeederMessage, PayloadFeederError> {
        let receiver = self
            .receiver
            .as_ref()
            .ok_or(PayloadFeederError::ChannelDisconnected)?;
        let message = receiver
            .recv()
            .map_err(|_| PayloadFeederError::ChannelDisconnected)?;
        if let Some(queued_count) = &self.queued_count {
            queued_count
                .fetch_update(AtomicOrdering::AcqRel, AtomicOrdering::Acquire, |value| {
                    Some(value.saturating_sub(1))
                })
                .ok();
        }
        Ok(message)
    }

    fn try_recv_message(&mut self) -> Result<Option<PayloadFeederMessage>, PayloadFeederError> {
        let receiver = self
            .receiver
            .as_ref()
            .ok_or(PayloadFeederError::ChannelDisconnected)?;
        let message = match receiver.try_recv() {
            Ok(message) => message,
            Err(TryRecvError::Empty) => return Ok(None),
            Err(TryRecvError::Disconnected) => return Err(PayloadFeederError::ChannelDisconnected),
        };
        if let Some(queued_count) = &self.queued_count {
            queued_count
                .fetch_update(AtomicOrdering::AcqRel, AtomicOrdering::Acquire, |value| {
                    Some(value.saturating_sub(1))
                })
                .ok();
        }
        Ok(Some(message))
    }

    fn complete_ready_frame(&mut self, frame: PayloadFrame, wait_start: Instant) -> PayloadFrame {
        self.stats.receiver_wait_time += wait_start.elapsed();
        self.stats.messages_received += 1;
        self.record_frame_backing(&frame);
        self.next_playback_index += 1;
        frame
    }

    fn record_frame_backing(&mut self, frame: &PayloadFrame) {
        self.stats.payload_bytes_returned = self
            .stats
            .payload_bytes_returned
            .saturating_add(frame.data.len() as u64);
        match frame.data {
            PayloadFrameData::Owned(_) => self.stats.owned_frames += 1,
            PayloadFrameData::ChunkSlice { .. } => self.stats.chunk_slice_frames += 1,
        }
    }
}

impl Drop for PayloadFeeder {
    fn drop(&mut self) {
        drop(self.receiver.take());
        if let Some(join_handle) = self.join_handle.take() {
            let _ = join_handle.join();
        }
    }
}

struct CurrentPayloadReader {
    file: File,
    current_offset: u64,
    entries: Vec<PayloadFrameRequest>,
    next_index: usize,
}

impl CurrentPayloadReader {
    fn next_frame(
        &mut self,
        stats: &mut PayloadFeederStats,
    ) -> Result<Option<PayloadFrame>, PayloadFeederError> {
        let Some(entry) = self.entries.get(self.next_index).copied() else {
            return Ok(None);
        };
        let mut payload = Vec::new();
        let elapsed = read_payload_entry_from_file(
            &mut self.file,
            &mut self.current_offset,
            entry,
            &mut payload,
            stats,
        )?;
        stats.receiver_wait_time += elapsed;
        stats.messages_received += 1;
        stats.owned_frames += 1;
        stats.payload_bytes_returned = stats
            .payload_bytes_returned
            .saturating_add(payload.len() as u64);
        self.next_index += 1;
        Ok(Some(PayloadFrame {
            playback_index: entry.playback_index,
            frame_number: entry.frame_number,
            data: PayloadFrameData::Owned(payload),
        }))
    }
}

enum PayloadFeederMessage {
    Frame(PayloadFrame),
    Error(String),
    Done(PayloadFeederStats),
}

#[derive(Debug, Clone)]
struct PayloadChunkPlan {
    chunks: Vec<PayloadChunk>,
    stats: PayloadChunkLayoutStats,
}

impl PayloadChunkPlan {
    fn from_plan(
        plan: &PayloadReadPlan,
        options: PayloadChunkOptions,
    ) -> Result<Self, PayloadFeederError> {
        let physical_entries = plan.entries_sorted_by_offset();
        let mut chunks = Vec::new();
        let mut current: Option<PayloadChunkBuilder> = None;

        for (physical_read_index, entry) in physical_entries.into_iter().enumerate() {
            let job = PayloadReadJob {
                physical_read_index,
                playback_index: entry.playback_index,
                frame_number: entry.frame_number,
                offset: entry.payload_offset,
                byte_len: entry.payload_len,
            };
            if let Some(builder) = current.as_mut() {
                if builder.should_split(&job, options)? {
                    chunks.push(
                        current
                            .take()
                            .ok_or_else(|| {
                                PayloadFeederError::InvalidPlan(
                                    "payload chunk builder missing".into(),
                                )
                            })?
                            .finish(chunks.len())?,
                    );
                    current = Some(PayloadChunkBuilder::new(job)?);
                } else {
                    builder.push(job)?;
                }
            } else {
                current = Some(PayloadChunkBuilder::new(job)?);
            }
        }

        if let Some(builder) = current {
            chunks.push(builder.finish(chunks.len())?);
        }

        let stats = PayloadChunkLayoutStats::from_chunks(&chunks)?;
        Ok(Self { chunks, stats })
    }
}

impl PayloadChunkLayoutStats {
    fn from_chunks(chunks: &[PayloadChunk]) -> Result<Self, PayloadFeederError> {
        let mut stats = Self {
            chunk_count: chunks.len(),
            ..Self::default()
        };
        for chunk in chunks {
            let chunk_bytes = chunk.byte_len()?;
            stats.chunk_total_bytes = stats
                .chunk_total_bytes
                .checked_add(chunk_bytes)
                .ok_or_else(|| {
                    PayloadFeederError::InvalidPlan("payload chunk total byte overflow".into())
                })?;
            stats.chunk_payload_bytes = stats
                .chunk_payload_bytes
                .checked_add(chunk.payload_bytes)
                .ok_or_else(|| {
                    PayloadFeederError::InvalidPlan("payload chunk payload byte overflow".into())
                })?;
            stats.chunk_gap_bytes_read = stats
                .chunk_gap_bytes_read
                .checked_add(chunk.gap_bytes)
                .ok_or_else(|| {
                    PayloadFeederError::InvalidPlan("payload chunk gap byte overflow".into())
                })?;
            stats.chunk_max_bytes_observed = stats.chunk_max_bytes_observed.max(chunk_bytes);
            stats.chunk_max_payloads_observed =
                stats.chunk_max_payloads_observed.max(chunk.jobs.len());
            if chunk.jobs.len() == 1 {
                stats.chunk_single_payload_count += 1;
            }
        }
        Ok(stats)
    }
}

#[derive(Debug, Clone)]
struct PayloadChunk {
    physical_chunk_index: usize,
    start_offset: u64,
    end_offset: u64,
    jobs: Vec<PayloadChunkJob>,
    payload_bytes: u64,
    gap_bytes: u64,
}

impl PayloadChunk {
    fn byte_len(&self) -> Result<u64, PayloadFeederError> {
        self.end_offset
            .checked_sub(self.start_offset)
            .ok_or_else(|| {
                PayloadFeederError::InvalidPlan("payload chunk end precedes start".into())
            })
    }
}

#[derive(Debug, Clone, Copy)]
struct PayloadChunkJob {
    physical_read_index: usize,
    playback_index: usize,
    frame_number: usize,
    byte_len: usize,
    offset_within_chunk: usize,
}

#[derive(Debug, Clone, Copy)]
struct PayloadReadJob {
    physical_read_index: usize,
    playback_index: usize,
    frame_number: usize,
    offset: u64,
    byte_len: usize,
}

impl PayloadReadJob {
    fn end_offset(self) -> Result<u64, PayloadFeederError> {
        self.offset
            .checked_add(self.byte_len as u64)
            .ok_or_else(|| PayloadFeederError::InvalidPlan("payload read job end overflow".into()))
    }
}

struct PayloadChunkBuilder {
    start_offset: u64,
    end_offset: u64,
    jobs: Vec<PayloadChunkJob>,
    payload_bytes: u64,
}

impl PayloadChunkBuilder {
    fn new(job: PayloadReadJob) -> Result<Self, PayloadFeederError> {
        let end_offset = job.end_offset()?;
        Ok(Self {
            start_offset: job.offset,
            end_offset,
            jobs: vec![PayloadChunkJob {
                physical_read_index: job.physical_read_index,
                playback_index: job.playback_index,
                frame_number: job.frame_number,
                byte_len: job.byte_len,
                offset_within_chunk: 0,
            }],
            payload_bytes: job.byte_len as u64,
        })
    }

    fn should_split(
        &self,
        job: &PayloadReadJob,
        options: PayloadChunkOptions,
    ) -> Result<bool, PayloadFeederError> {
        if self.jobs.len() >= options.max_payloads_per_chunk {
            return Ok(true);
        }
        if job.offset > self.end_offset
            && job.offset - self.end_offset > options.gap_threshold_bytes
        {
            return Ok(true);
        }
        let proposed_end = self.end_offset.max(job.end_offset()?);
        let proposed_bytes = proposed_end.checked_sub(self.start_offset).ok_or_else(|| {
            PayloadFeederError::InvalidPlan("payload chunk proposed span underflow".into())
        })?;
        Ok(proposed_bytes > options.max_chunk_bytes)
    }

    fn push(&mut self, job: PayloadReadJob) -> Result<(), PayloadFeederError> {
        let end_offset = job.end_offset()?;
        let offset_within_chunk_u64 =
            job.offset.checked_sub(self.start_offset).ok_or_else(|| {
                PayloadFeederError::InvalidPlan("payload chunk job starts before chunk".into())
            })?;
        let offset_within_chunk = usize::try_from(offset_within_chunk_u64).map_err(|_| {
            PayloadFeederError::InvalidPlan("payload offset within chunk overflows usize".into())
        })?;
        self.end_offset = self.end_offset.max(end_offset);
        self.payload_bytes = self
            .payload_bytes
            .checked_add(job.byte_len as u64)
            .ok_or_else(|| {
                PayloadFeederError::InvalidPlan("payload chunk payload byte overflow".into())
            })?;
        self.jobs.push(PayloadChunkJob {
            physical_read_index: job.physical_read_index,
            playback_index: job.playback_index,
            frame_number: job.frame_number,
            byte_len: job.byte_len,
            offset_within_chunk,
        });
        Ok(())
    }

    fn finish(self, physical_chunk_index: usize) -> Result<PayloadChunk, PayloadFeederError> {
        let span_bytes = self
            .end_offset
            .checked_sub(self.start_offset)
            .ok_or_else(|| {
                PayloadFeederError::InvalidPlan("payload chunk span underflow".into())
            })?;
        let gap_bytes = span_bytes.checked_sub(self.payload_bytes).ok_or_else(|| {
            PayloadFeederError::InvalidPlan("payload chunk payload bytes exceed span".into())
        })?;
        Ok(PayloadChunk {
            physical_chunk_index,
            start_offset: self.start_offset,
            end_offset: self.end_offset,
            jobs: self.jobs,
            payload_bytes: self.payload_bytes,
            gap_bytes,
        })
    }
}

fn initial_worker_stats(
    options: PayloadFeederOptions,
    selected_frames: usize,
    chunk_plan: Option<&PayloadChunkPlan>,
) -> PayloadFeederStats {
    let mut stats = PayloadFeederStats::new(options.mode, selected_frames, options.chunk_options);
    stats.physical_read_order = true;
    stats.output_playback_order = true;
    if let Some(chunk_plan) = chunk_plan {
        stats.chunked_reads = true;
        stats.chunk_count = chunk_plan.stats.chunk_count;
        stats.chunk_total_bytes = chunk_plan.stats.chunk_total_bytes;
        stats.chunk_payload_bytes = chunk_plan.stats.chunk_payload_bytes;
        stats.chunk_gap_bytes_read = chunk_plan.stats.chunk_gap_bytes_read;
        stats.chunk_max_bytes_observed = chunk_plan.stats.chunk_max_bytes_observed;
        stats.chunk_max_payloads_observed = chunk_plan.stats.chunk_max_payloads_observed;
        stats.chunk_single_payload_count = chunk_plan.stats.chunk_single_payload_count;
        stats.retained_chunk_bytes_estimate = chunk_plan
            .stats
            .chunk_max_bytes_observed
            .saturating_mul(options.prefetch_depth as u64);
    }
    stats
}

struct OffsetWorkerJob {
    path: PathBuf,
    entries: Vec<PayloadFrameRequest>,
    chunk_plan: Option<PayloadChunkPlan>,
    stats: PayloadFeederStats,
    sender: mpsc::SyncSender<PayloadFeederMessage>,
    queued_count: Arc<AtomicUsize>,
    max_depth_observed: Arc<AtomicUsize>,
    depth: usize,
}

fn run_offset_worker(job: OffsetWorkerJob) {
    let OffsetWorkerJob {
        path,
        entries,
        chunk_plan,
        mut stats,
        sender,
        queued_count,
        max_depth_observed,
        depth,
    } = job;

    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) => {
            let _ = send_message(
                &sender,
                PayloadFeederMessage::Error(error.to_string()),
                &queued_count,
                &max_depth_observed,
                depth,
            );
            return;
        }
    };
    let mut current_offset = 0u64;

    if let Some(chunk_plan) = chunk_plan {
        for chunk in chunk_plan.chunks {
            let chunk_start = Instant::now();
            let did_seek = current_offset != chunk.start_offset;
            match read_payload_chunk_from_file(&mut file, &mut current_offset, &chunk) {
                Ok(chunk_bytes) => {
                    let elapsed = chunk_start.elapsed();
                    if did_seek {
                        stats.seeks += 1;
                    }
                    stats.producer_read_time += elapsed;
                    stats.read_calls += 1;
                    stats.producer_chunk_read_calls += 1;
                    let chunk_len = chunk_bytes.len() as u64;
                    stats.bytes_read = stats.bytes_read.saturating_add(chunk_len);
                    let chunk_arc: Arc<[u8]> = Arc::from(chunk_bytes.into_boxed_slice());
                    for job in chunk.jobs {
                        let end = match job.offset_within_chunk.checked_add(job.byte_len) {
                            Some(end) => end,
                            None => {
                                let _ = send_message(
                                    &sender,
                                    PayloadFeederMessage::Error(format!(
                                        "chunk payload slice end overflows for frame {}",
                                        job.frame_number
                                    )),
                                    &queued_count,
                                    &max_depth_observed,
                                    depth,
                                );
                                return;
                            }
                        };
                        if end > chunk_arc.len() {
                            let _ = send_message(
                                &sender,
                                PayloadFeederMessage::Error(format!(
                                    "chunk payload slice out of bounds for frame {} physical_read_index={} chunk_index={} start={} end={} chunk_len={}",
                                    job.frame_number,
                                    job.physical_read_index,
                                    chunk.physical_chunk_index,
                                    job.offset_within_chunk,
                                    end,
                                    chunk_arc.len()
                                )),
                                &queued_count,
                                &max_depth_observed,
                                depth,
                            );
                            return;
                        }
                        stats.producer_payload_messages_sent += 1;
                        stats.producer_payload_bytes = stats
                            .producer_payload_bytes
                            .saturating_add(job.byte_len as u64);
                        let frame = PayloadFrame {
                            playback_index: job.playback_index,
                            frame_number: job.frame_number,
                            data: PayloadFrameData::ChunkSlice {
                                chunk: chunk_arc.clone(),
                                range: job.offset_within_chunk..end,
                            },
                        };
                        if !send_message(
                            &sender,
                            PayloadFeederMessage::Frame(frame),
                            &queued_count,
                            &max_depth_observed,
                            depth,
                        ) {
                            return;
                        }
                    }
                }
                Err(error) => {
                    let _ = send_message(
                        &sender,
                        PayloadFeederMessage::Error(error.to_string()),
                        &queued_count,
                        &max_depth_observed,
                        depth,
                    );
                    return;
                }
            }
        }
    } else {
        for entry in entries {
            let mut payload = Vec::new();
            match read_payload_entry_from_file(
                &mut file,
                &mut current_offset,
                entry,
                &mut payload,
                &mut stats,
            ) {
                Ok(_) => {
                    stats.producer_payload_messages_sent += 1;
                    stats.producer_payload_bytes = stats
                        .producer_payload_bytes
                        .saturating_add(payload.len() as u64);
                    let frame = PayloadFrame {
                        playback_index: entry.playback_index,
                        frame_number: entry.frame_number,
                        data: PayloadFrameData::Owned(payload),
                    };
                    if !send_message(
                        &sender,
                        PayloadFeederMessage::Frame(frame),
                        &queued_count,
                        &max_depth_observed,
                        depth,
                    ) {
                        return;
                    }
                }
                Err(error) => {
                    let _ = send_message(
                        &sender,
                        PayloadFeederMessage::Error(error.to_string()),
                        &queued_count,
                        &max_depth_observed,
                        depth,
                    );
                    return;
                }
            }
        }
    }

    stats.max_depth_observed = max_depth_observed.load(AtomicOrdering::Relaxed).min(depth);
    let _ = send_message(
        &sender,
        PayloadFeederMessage::Done(stats),
        &queued_count,
        &max_depth_observed,
        depth,
    );
}

fn read_payload_entry_from_file(
    file: &mut File,
    current_offset: &mut u64,
    entry: PayloadFrameRequest,
    payload: &mut Vec<u8>,
    stats: &mut PayloadFeederStats,
) -> Result<Duration, PayloadFeederError> {
    let read_start = Instant::now();
    if *current_offset != entry.payload_offset {
        file.seek(SeekFrom::Start(entry.payload_offset))?;
        stats.seeks += 1;
        *current_offset = entry.payload_offset;
    }
    payload.clear();
    payload.resize(entry.payload_len, 0);
    file.read_exact(payload)?;
    *current_offset = entry.end_offset()?;
    stats.read_calls += 1;
    stats.bytes_read = stats.bytes_read.saturating_add(entry.payload_len as u64);
    let elapsed = read_start.elapsed();
    stats.producer_read_time += elapsed;
    Ok(elapsed)
}

fn read_payload_chunk_from_file(
    file: &mut File,
    current_offset: &mut u64,
    chunk: &PayloadChunk,
) -> Result<Vec<u8>, PayloadFeederError> {
    if *current_offset != chunk.start_offset {
        file.seek(SeekFrom::Start(chunk.start_offset))?;
        *current_offset = chunk.start_offset;
    }
    let chunk_len = usize::try_from(chunk.byte_len()?).map_err(|_| {
        PayloadFeederError::InvalidPlan("payload chunk length overflows usize".into())
    })?;
    let mut chunk_buffer = vec![0u8; chunk_len];
    file.read_exact(&mut chunk_buffer)?;
    *current_offset = chunk.end_offset;
    Ok(chunk_buffer)
}

fn send_message(
    sender: &mpsc::SyncSender<PayloadFeederMessage>,
    message: PayloadFeederMessage,
    queued_count: &AtomicUsize,
    max_depth_observed: &AtomicUsize,
    depth: usize,
) -> bool {
    let queued = queued_count.fetch_add(1, AtomicOrdering::AcqRel) + 1;
    let observed = queued.min(depth);
    let mut current = max_depth_observed.load(AtomicOrdering::Relaxed);
    while observed > current {
        match max_depth_observed.compare_exchange_weak(
            current,
            observed,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => break,
            Err(next) => current = next,
        }
    }

    match sender.send(message) {
        Ok(()) => true,
        Err(_) => {
            queued_count
                .fetch_update(AtomicOrdering::AcqRel, AtomicOrdering::Acquire, |value| {
                    Some(value.saturating_sub(1))
                })
                .ok();
            false
        }
    }
}
