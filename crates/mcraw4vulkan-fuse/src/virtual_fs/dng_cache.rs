use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::virtual_fs::{
    DngFrameByteLenCalculator, DngFrameGenerator, DngGenerationBackend, DngGenerationConfig,
    DngGenerationExecutionPolicy, DngGenerationTimings, GeneratedDngFrame,
};

const NO_FRAME_INDEX: usize = usize::MAX;
const MIB: u64 = 1024 * 1024;
const NOMINAL_4K_DNG_FRAME_BYTES: u64 = 16 * MIB;
const HOT_BYTE_HEADROOM_BYTES: u64 = 48 * MIB;
const DEFAULT_HOT_GRACE_FPS: u32 = 30;
const DEFAULT_HOT_GRACE_MULTIPLIER_NUMERATOR: u32 = 1;
const DEFAULT_HOT_GRACE_MULTIPLIER_DENOMINATOR: u32 = 1;
pub const DEFAULT_DNG_HOT_FRAME_CAPACITY: usize = 21;
pub const DEFAULT_DNG_HOT_PREFETCH_TARGET: usize = 4;
pub const DEFAULT_DNG_COLD_FRAME_CAPACITY: usize = 1;
pub const DEFAULT_DNG_HOT_BYTE_CAPACITY: u64 = 2 * 1024 * MIB;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DngReadDirection {
    Forward,
    Backward,
}

impl DngReadDirection {
    fn from_adjacent(previous: usize, current: usize) -> Option<Self> {
        if current == previous.saturating_add(1) {
            Some(Self::Forward)
        } else if previous == current.saturating_add(1) {
            Some(Self::Backward)
        } else {
            None
        }
    }

    fn next_frame(self, frame_index: usize, frame_count: usize) -> Option<usize> {
        match self {
            Self::Forward => frame_index
                .checked_add(1)
                .filter(|&frame| frame < frame_count),
            Self::Backward => frame_index.checked_sub(1),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Backward => "backward",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DngCachePolicy {
    hot_frame_capacity: usize,
    hot_prefetch_target: usize,
    hot_history_capacity: usize,
    hot_byte_capacity: u64,
    cold_frame_capacity: usize,
    cold_byte_capacity: u64,
    promotion_adjacent_requests: usize,
    hot_grace: Duration,
}

impl DngCachePolicy {
    pub fn production_default() -> Self {
        Self::from_hot_window(
            DEFAULT_DNG_HOT_FRAME_CAPACITY,
            DEFAULT_DNG_HOT_PREFETCH_TARGET,
        )
    }

    pub fn from_hot_window(hot_frame_capacity: usize, hot_prefetch_target: usize) -> Self {
        let hot_frame_capacity = hot_frame_capacity.max(1);
        let hot_prefetch_target = hot_prefetch_target.min(hot_frame_capacity.saturating_sub(1));
        let hot_history_capacity = hot_frame_capacity
            .saturating_sub(hot_prefetch_target)
            .saturating_sub(1);
        let hot_byte_capacity = derived_hot_byte_capacity(hot_frame_capacity);
        let hot_grace = hot_grace_duration(hot_frame_capacity);

        Self {
            hot_frame_capacity,
            hot_prefetch_target,
            hot_history_capacity,
            hot_byte_capacity,
            cold_frame_capacity: DEFAULT_DNG_COLD_FRAME_CAPACITY,
            cold_byte_capacity: NOMINAL_4K_DNG_FRAME_BYTES,
            promotion_adjacent_requests: 3,
            hot_grace,
        }
        .with_private_overrides()
    }

    pub fn hot_frame_capacity(self) -> usize {
        self.hot_frame_capacity
    }

    pub fn hot_prefetch_target(self) -> usize {
        self.hot_prefetch_target
    }

    pub fn hot_history_capacity(self) -> usize {
        self.hot_history_capacity
    }

    pub fn hot_byte_capacity(self) -> u64 {
        self.hot_byte_capacity
    }

    pub fn cold_frame_capacity(self) -> usize {
        self.cold_frame_capacity
    }

    pub fn hot_grace(self) -> Duration {
        self.hot_grace
    }

    fn queue_capacity(self) -> usize {
        self.hot_prefetch_target.saturating_add(2).max(1)
    }

    #[allow(unused_mut)]
    fn with_private_overrides(mut self) -> Self {
        self
    }
}

fn derived_hot_byte_capacity(hot_frame_capacity: usize) -> u64 {
    if hot_frame_capacity == DEFAULT_DNG_HOT_FRAME_CAPACITY {
        return DEFAULT_DNG_HOT_BYTE_CAPACITY;
    }

    let frame_bytes = u64::try_from(hot_frame_capacity)
        .unwrap_or(u64::MAX / NOMINAL_4K_DNG_FRAME_BYTES)
        .saturating_mul(NOMINAL_4K_DNG_FRAME_BYTES);
    frame_bytes
        .saturating_add(HOT_BYTE_HEADROOM_BYTES)
        .max(NOMINAL_4K_DNG_FRAME_BYTES)
}

fn hot_grace_duration(hot_frame_capacity: usize) -> Duration {
    let numerator = u128::from(DEFAULT_HOT_GRACE_MULTIPLIER_NUMERATOR)
        .saturating_mul(u128::try_from(hot_frame_capacity).unwrap_or(u128::MAX))
        .saturating_mul(1_000_000_000);
    let denominator = u128::from(DEFAULT_HOT_GRACE_FPS)
        .saturating_mul(u128::from(DEFAULT_HOT_GRACE_MULTIPLIER_DENOMINATOR))
        .max(1);
    Duration::from_nanos(u64::try_from(numerator / denominator).unwrap_or(u64::MAX))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrefetchHint {
    anchor_frame: usize,
    direction: DngReadDirection,
    epoch: u64,
}

#[derive(Debug)]
pub struct CachedDngFrame {
    // Immutable shared bytes let open handles retain the exact allocation even
    // after the cache removes its own LRU reference.
    pub frame_index: usize,
    pub bytes: Arc<[u8]>,
    pub byte_len: u64,
    pub timings: DngGenerationTimings,
}

pub struct DngFrameByteCache {
    inner: Arc<DngFrameByteCacheInner>,
    prefetch_tx: Option<SyncSender<PrefetchHint>>,
    prefetch_worker: Option<JoinHandle<()>>,
}

struct DngFrameByteCacheInner {
    source_path: PathBuf,
    generation_config: DngGenerationConfig,
    generator: Mutex<Option<DngFrameGenerator>>,
    byte_len_calculator: Mutex<DngFrameByteLenCalculator>,
    state: Mutex<CacheState>,
    instrumentation: CacheInstrumentation,
    policy: DngCachePolicy,
    frame_count: usize,
    execution_policy: DngGenerationExecutionPolicy,
    shutdown_requested: AtomicBool,
}

struct CacheState {
    entries: HashMap<usize, Arc<CachedDngFrame>>,
    lru_order: VecDeque<usize>,
    total_cached_bytes: u64,
    temperature: DngCacheTemperature,
    demand_history: DemandHistory,
    generation_epoch: u64,
    active_demands: usize,
    active_prefetches: usize,
    last_demand_frame: Option<usize>,
}

impl Default for CacheState {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            total_cached_bytes: 0,
            temperature: DngCacheTemperature::Cold,
            demand_history: DemandHistory::default(),
            generation_epoch: 0,
            active_demands: 0,
            active_prefetches: 0,
            last_demand_frame: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DngCacheTemperature {
    Cold,
    Hot {
        direction: DngReadDirection,
        last_demand: Instant,
    },
}

impl DngCacheTemperature {
    fn is_hot(self) -> bool {
        matches!(self, Self::Hot { .. })
    }

    fn direction(self) -> Option<DngReadDirection> {
        match self {
            Self::Cold => None,
            Self::Hot { direction, .. } => Some(direction),
        }
    }

    fn last_demand(self) -> Option<Instant> {
        match self {
            Self::Cold => None,
            Self::Hot { last_demand, .. } => Some(last_demand),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct DemandHistory {
    last_frame: Option<usize>,
    direction: Option<DngReadDirection>,
    adjacent_run_len: usize,
}

impl DemandHistory {
    fn observe(&mut self, frame_index: usize) -> DemandObservation {
        let previous_frame = self.last_frame;
        let adjacent_direction = previous_frame
            .and_then(|previous| DngReadDirection::from_adjacent(previous, frame_index));

        let observation = match adjacent_direction {
            Some(direction) if self.direction == Some(direction) => {
                self.adjacent_run_len = self.adjacent_run_len.saturating_add(1);
                DemandObservation::Adjacent {
                    direction,
                    run_len: self.adjacent_run_len,
                }
            }
            Some(direction) => {
                self.direction = Some(direction);
                self.adjacent_run_len = 2;
                DemandObservation::Adjacent {
                    direction,
                    run_len: self.adjacent_run_len,
                }
            }
            None => {
                self.direction = None;
                self.adjacent_run_len = 1;
                DemandObservation::RandomOrIsolated
            }
        };

        self.last_frame = Some(frame_index);
        observation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DemandObservation {
    Adjacent {
        direction: DngReadDirection,
        run_len: usize,
    },
    RandomOrIsolated,
}

#[derive(Debug)]
struct CacheInstrumentation {
    cache_hits: AtomicU64,
    cache_hits_after_wait: AtomicU64,
    cache_misses: AtomicU64,
    metadata_len_queries: AtomicU64,
    metadata_len_cached_hits: AtomicU64,

    frames_generated: AtomicU64,
    generated_bytes_total: AtomicU64,
    generation_time_total_ns: AtomicU64,

    raw_payload_bytes_total: AtomicU64,
    decoded_pixel_bytes_total: AtomicU64,
    final_dng_bytes_total: AtomicU64,

    dng_description_time_total_ns: AtomicU64,
    payload_read_time_total_ns: AtomicU64,
    decode_time_total_ns: AtomicU64,
    dng_build_time_total_ns: AtomicU64,
    cache_insert_time_total_ns: AtomicU64,

    gpu_work_plan_time_total_ns: AtomicU64,
    gpu_cpu_prepare_time_total_ns: AtomicU64,
    gpu_upload_time_total_ns: AtomicU64,
    gpu_encode_submit_time_total_ns: AtomicU64,
    gpu_wait_map_time_total_ns: AtomicU64,
    gpu_readback_convert_time_total_ns: AtomicU64,
    gpu_dispatch_readback_time_total_ns: AtomicU64,

    frames_evicted: AtomicU64,
    evicted_bytes_total: AtomicU64,
    last_generated_frame: AtomicUsize,

    prefetch_requested: AtomicU64,
    prefetch_queued: AtomicU64,
    prefetch_queue_full: AtomicU64,
    prefetch_skipped_cached: AtomicU64,
    prefetch_generator_busy: AtomicU64,
    prefetch_generated: AtomicU64,
    prefetch_errors: AtomicU64,
    prefetch_stale: AtomicU64,

    hot_promotions: AtomicU64,
    hot_demotions: AtomicU64,
    generator_created: AtomicU64,
    generator_dropped: AtomicU64,
}

impl Default for CacheInstrumentation {
    fn default() -> Self {
        Self {
            cache_hits: AtomicU64::new(0),
            cache_hits_after_wait: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            metadata_len_queries: AtomicU64::new(0),
            metadata_len_cached_hits: AtomicU64::new(0),

            frames_generated: AtomicU64::new(0),
            generated_bytes_total: AtomicU64::new(0),
            generation_time_total_ns: AtomicU64::new(0),

            raw_payload_bytes_total: AtomicU64::new(0),
            decoded_pixel_bytes_total: AtomicU64::new(0),
            final_dng_bytes_total: AtomicU64::new(0),

            dng_description_time_total_ns: AtomicU64::new(0),
            payload_read_time_total_ns: AtomicU64::new(0),
            decode_time_total_ns: AtomicU64::new(0),
            dng_build_time_total_ns: AtomicU64::new(0),
            cache_insert_time_total_ns: AtomicU64::new(0),

            gpu_work_plan_time_total_ns: AtomicU64::new(0),
            gpu_cpu_prepare_time_total_ns: AtomicU64::new(0),
            gpu_upload_time_total_ns: AtomicU64::new(0),
            gpu_encode_submit_time_total_ns: AtomicU64::new(0),
            gpu_wait_map_time_total_ns: AtomicU64::new(0),
            gpu_readback_convert_time_total_ns: AtomicU64::new(0),
            gpu_dispatch_readback_time_total_ns: AtomicU64::new(0),

            frames_evicted: AtomicU64::new(0),
            evicted_bytes_total: AtomicU64::new(0),
            last_generated_frame: AtomicUsize::new(NO_FRAME_INDEX),

            prefetch_requested: AtomicU64::new(0),
            prefetch_queued: AtomicU64::new(0),
            prefetch_queue_full: AtomicU64::new(0),
            prefetch_skipped_cached: AtomicU64::new(0),
            prefetch_generator_busy: AtomicU64::new(0),
            prefetch_generated: AtomicU64::new(0),
            prefetch_errors: AtomicU64::new(0),
            prefetch_stale: AtomicU64::new(0),

            hot_promotions: AtomicU64::new(0),
            hot_demotions: AtomicU64::new(0),
            generator_created: AtomicU64::new(0),
            generator_dropped: AtomicU64::new(0),
        }
    }
}

impl DngFrameByteCache {
    pub fn open(path: &Path, backend: DngGenerationBackend, max_frames: usize) -> Result<Self> {
        Self::open_with_config(
            path,
            DngGenerationConfig {
                backend,
                vignette_mode: Default::default(),
                execution_policy: DngGenerationExecutionPolicy::Inflight2Default,
            },
            max_frames,
        )
    }

    pub fn open_with_config(
        path: &Path,
        config: DngGenerationConfig,
        max_frames: usize,
    ) -> Result<Self> {
        Self::open_with_hot_window(path, config, max_frames, DEFAULT_DNG_HOT_PREFETCH_TARGET)
    }

    pub fn open_with_hot_window(
        path: &Path,
        config: DngGenerationConfig,
        max_frames: usize,
        hot_prefetch_target: usize,
    ) -> Result<Self> {
        let byte_len_calculator = DngFrameByteLenCalculator::open_with_config(path, config)?;
        let frame_count = byte_len_calculator.frame_count();
        let policy = if max_frames == DEFAULT_DNG_HOT_FRAME_CAPACITY
            && hot_prefetch_target == DEFAULT_DNG_HOT_PREFETCH_TARGET
        {
            DngCachePolicy::production_default()
        } else {
            DngCachePolicy::from_hot_window(max_frames, hot_prefetch_target)
        };

        let inner = Arc::new(DngFrameByteCacheInner {
            source_path: path.to_path_buf(),
            generation_config: config,
            generator: Mutex::new(None),
            byte_len_calculator: Mutex::new(byte_len_calculator),
            state: Mutex::new(CacheState::default()),
            instrumentation: CacheInstrumentation::default(),
            policy,
            frame_count,
            execution_policy: config.execution_policy,
            shutdown_requested: AtomicBool::new(false),
        });

        let (prefetch_tx, prefetch_rx) = mpsc::sync_channel(policy.queue_capacity());
        let prefetch_worker = spawn_prefetch_worker(inner.clone(), prefetch_rx)?;

        Ok(Self {
            inner,
            prefetch_tx: Some(prefetch_tx),
            prefetch_worker: Some(prefetch_worker),
        })
    }

    pub fn frame_count(&self) -> Result<usize> {
        Ok(self.inner.frame_count)
    }

    pub fn get_or_generate(&self, frame_index: usize) -> Result<Arc<CachedDngFrame>> {
        self.get_or_generate_for_read(frame_index, true)
    }

    pub fn observe_logical_demand(&self, frame_index: usize) -> Result<()> {
        if frame_index >= self.inner.frame_count {
            anyhow::bail!(
                "frame index {} is out of range for clip with {} frames",
                frame_index,
                self.inner.frame_count
            );
        }

        self.inner.begin_demand(frame_index)?;
        let should_prefetch = self.inner.finish_demand(frame_index);
        if should_prefetch {
            self.queue_hot_prefetch(frame_index);
        }

        Ok(())
    }

    pub fn get_or_generate_for_read(
        &self,
        frame_index: usize,
        count_as_demand: bool,
    ) -> Result<Arc<CachedDngFrame>> {
        let _request_start = Instant::now();
        if frame_index >= self.inner.frame_count {
            anyhow::bail!(
                "frame index {} is out of range for clip with {} frames",
                frame_index,
                self.inner.frame_count
            );
        }

        if count_as_demand {
            self.inner.begin_demand(frame_index)?;
        }

        let result = (|| {
            if let Some(cached) = self.inner.get_cached(frame_index)? {
                self.inner
                    .instrumentation
                    .cache_hits
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(cached);
            }

            let mut generator_guard = self
                .inner
                .generator
                .lock()
                .map_err(|_| anyhow::anyhow!("DNG generator mutex was poisoned"))?;

            if let Some(cached) = self.inner.get_cached(frame_index)? {
                self.inner
                    .instrumentation
                    .cache_hits_after_wait
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(cached);
            }

            self.inner
                .instrumentation
                .cache_misses
                .fetch_add(1, Ordering::Relaxed);

            let generation_start = Instant::now();
            let generator = self.inner.ensure_generator_locked(&mut generator_guard)?;
            let mut generated_frames = generator
                .generate_frames_with_execution_policy(&[frame_index], self.inner.execution_policy)
                .with_context(|| format!("failed to generate DNG bytes for frame {frame_index}"))?;
            let generated = generated_frames.pop().with_context(|| {
                format!("DNG generator returned no frame for requested frame {frame_index}")
            })?;

            let generation_elapsed = generation_start.elapsed();

            self.inner
                .insert_generated(generated, generation_elapsed, Some(frame_index))
        })();

        let should_prefetch = if count_as_demand {
            self.inner.finish_demand(frame_index)
        } else {
            self.inner.finish_auxiliary_read()
        };
        if should_prefetch {
            self.queue_hot_prefetch(frame_index);
        }

        result
    }

    pub fn dng_byte_len(&self, frame_index: usize) -> Result<u64> {
        self.inner
            .instrumentation
            .metadata_len_queries
            .fetch_add(1, Ordering::Relaxed);

        if let Some(cached) = self.inner.get_cached(frame_index)? {
            self.inner
                .instrumentation
                .metadata_len_cached_hits
                .fetch_add(1, Ordering::Relaxed);
            return Ok(cached.byte_len);
        }

        let mut calculator = self
            .inner
            .byte_len_calculator
            .lock()
            .map_err(|_| anyhow::anyhow!("DNG byte-length calculator mutex was poisoned"))?;

        calculator
            .frame_byte_len(frame_index)
            .with_context(|| format!("failed to compute DNG byte length for frame {frame_index}"))
    }

    pub fn stats(&self) -> Result<DngFrameCacheStats> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("DNG cache mutex was poisoned"))?;

        let last_generated_frame = match self
            .inner
            .instrumentation
            .last_generated_frame
            .load(Ordering::Relaxed)
        {
            NO_FRAME_INDEX => None,
            frame_index => Some(frame_index),
        };

        Ok(DngFrameCacheStats {
            cached_frames: state.entries.len(),
            total_cached_bytes: state.total_cached_bytes,
            max_frames: self.inner.policy.hot_frame_capacity(),
            max_cached_bytes: self.inner.policy.hot_byte_capacity(),
            cold_frame_capacity: self.inner.policy.cold_frame_capacity(),
            hot_prefetch_target: self.inner.policy.hot_prefetch_target(),
            hot_history_capacity: self.inner.policy.hot_history_capacity(),
            hot_grace_ms: duration_ms_u64(self.inner.policy.hot_grace()),
            is_hot: state.temperature.is_hot(),
            hot_direction: state.temperature.direction().map(DngReadDirection::label),
            active_demands: state.active_demands,
            active_prefetches: state.active_prefetches,
            generation_epoch: state.generation_epoch,
            cache_hits: self
                .inner
                .instrumentation
                .cache_hits
                .load(Ordering::Relaxed),
            cache_hits_after_wait: self
                .inner
                .instrumentation
                .cache_hits_after_wait
                .load(Ordering::Relaxed),
            cache_misses: self
                .inner
                .instrumentation
                .cache_misses
                .load(Ordering::Relaxed),
            metadata_len_queries: self
                .inner
                .instrumentation
                .metadata_len_queries
                .load(Ordering::Relaxed),
            metadata_len_cached_hits: self
                .inner
                .instrumentation
                .metadata_len_cached_hits
                .load(Ordering::Relaxed),

            frames_generated: self
                .inner
                .instrumentation
                .frames_generated
                .load(Ordering::Relaxed),
            generated_bytes_total: self
                .inner
                .instrumentation
                .generated_bytes_total
                .load(Ordering::Relaxed),
            generation_time_total_ns: self
                .inner
                .instrumentation
                .generation_time_total_ns
                .load(Ordering::Relaxed),

            raw_payload_bytes_total: self
                .inner
                .instrumentation
                .raw_payload_bytes_total
                .load(Ordering::Relaxed),
            decoded_pixel_bytes_total: self
                .inner
                .instrumentation
                .decoded_pixel_bytes_total
                .load(Ordering::Relaxed),
            final_dng_bytes_total: self
                .inner
                .instrumentation
                .final_dng_bytes_total
                .load(Ordering::Relaxed),

            dng_description_time_total_ns: self
                .inner
                .instrumentation
                .dng_description_time_total_ns
                .load(Ordering::Relaxed),
            payload_read_time_total_ns: self
                .inner
                .instrumentation
                .payload_read_time_total_ns
                .load(Ordering::Relaxed),
            decode_time_total_ns: self
                .inner
                .instrumentation
                .decode_time_total_ns
                .load(Ordering::Relaxed),
            dng_build_time_total_ns: self
                .inner
                .instrumentation
                .dng_build_time_total_ns
                .load(Ordering::Relaxed),
            cache_insert_time_total_ns: self
                .inner
                .instrumentation
                .cache_insert_time_total_ns
                .load(Ordering::Relaxed),

            gpu_work_plan_time_total_ns: self
                .inner
                .instrumentation
                .gpu_work_plan_time_total_ns
                .load(Ordering::Relaxed),
            gpu_cpu_prepare_time_total_ns: self
                .inner
                .instrumentation
                .gpu_cpu_prepare_time_total_ns
                .load(Ordering::Relaxed),
            gpu_upload_time_total_ns: self
                .inner
                .instrumentation
                .gpu_upload_time_total_ns
                .load(Ordering::Relaxed),
            gpu_encode_submit_time_total_ns: self
                .inner
                .instrumentation
                .gpu_encode_submit_time_total_ns
                .load(Ordering::Relaxed),
            gpu_wait_map_time_total_ns: self
                .inner
                .instrumentation
                .gpu_wait_map_time_total_ns
                .load(Ordering::Relaxed),
            gpu_readback_convert_time_total_ns: self
                .inner
                .instrumentation
                .gpu_readback_convert_time_total_ns
                .load(Ordering::Relaxed),
            gpu_dispatch_readback_time_total_ns: self
                .inner
                .instrumentation
                .gpu_dispatch_readback_time_total_ns
                .load(Ordering::Relaxed),

            frames_evicted: self
                .inner
                .instrumentation
                .frames_evicted
                .load(Ordering::Relaxed),
            evicted_bytes_total: self
                .inner
                .instrumentation
                .evicted_bytes_total
                .load(Ordering::Relaxed),
            last_generated_frame,

            prefetch_requested: self
                .inner
                .instrumentation
                .prefetch_requested
                .load(Ordering::Relaxed),
            prefetch_queued: self
                .inner
                .instrumentation
                .prefetch_queued
                .load(Ordering::Relaxed),
            prefetch_queue_full: self
                .inner
                .instrumentation
                .prefetch_queue_full
                .load(Ordering::Relaxed),
            prefetch_skipped_cached: self
                .inner
                .instrumentation
                .prefetch_skipped_cached
                .load(Ordering::Relaxed),
            prefetch_generator_busy: self
                .inner
                .instrumentation
                .prefetch_generator_busy
                .load(Ordering::Relaxed),
            prefetch_generated: self
                .inner
                .instrumentation
                .prefetch_generated
                .load(Ordering::Relaxed),
            prefetch_errors: self
                .inner
                .instrumentation
                .prefetch_errors
                .load(Ordering::Relaxed),
            prefetch_stale: self
                .inner
                .instrumentation
                .prefetch_stale
                .load(Ordering::Relaxed),
            hot_promotions: self
                .inner
                .instrumentation
                .hot_promotions
                .load(Ordering::Relaxed),
            hot_demotions: self
                .inner
                .instrumentation
                .hot_demotions
                .load(Ordering::Relaxed),
            generator_created: self
                .inner
                .instrumentation
                .generator_created
                .load(Ordering::Relaxed),
            generator_dropped: self
                .inner
                .instrumentation
                .generator_dropped
                .load(Ordering::Relaxed),
        })
    }

    fn queue_hot_prefetch(&self, anchor_frame: usize) {
        if self.inner.shutdown_requested.load(Ordering::Acquire) {
            return;
        }

        let Some(prefetch_tx) = self.prefetch_tx.as_ref() else {
            self.inner
                .instrumentation
                .prefetch_errors
                .fetch_add(1, Ordering::Relaxed);
            return;
        };

        let Some(hint) = self.inner.prefetch_hint_for_anchor(anchor_frame) else {
            return;
        };

        self.inner.instrumentation.prefetch_requested.fetch_add(
            u64::try_from(self.inner.policy.hot_prefetch_target).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );

        match prefetch_tx.try_send(hint) {
            Ok(()) => {
                self.inner
                    .instrumentation
                    .prefetch_queued
                    .fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Full(_)) => {
                self.inner
                    .instrumentation
                    .prefetch_queue_full
                    .fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {
                self.inner
                    .instrumentation
                    .prefetch_errors
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for DngFrameByteCache {
    fn drop(&mut self) {
        // Closing the sender wakes the worker, and joining guarantees no
        // speculative generation still owns cache state when drop returns.
        self.inner.shutdown_requested.store(true, Ordering::Release);

        drop(self.prefetch_tx.take());

        if let Some(worker) = self.prefetch_worker.take() {
            if worker.join().is_err() {
                self.inner
                    .instrumentation
                    .prefetch_errors
                    .fetch_add(1, Ordering::Relaxed);
            }
        }

        // The generator is lazy, but once constructed it belongs to this cache.
        // Release it only after the worker can no longer use it.
        self.inner.drop_generator_for_shutdown();
    }
}

impl DngFrameByteCacheInner {
    // The generation epoch cancels queued prefetch when an active sequence changes
    // direction or a random access demotes the hot cache, preventing stale refills.
    fn begin_demand(&self, frame_index: usize) -> Result<()> {
        let now = Instant::now();
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("DNG cache mutex was poisoned"))?;
        state.active_demands = state.active_demands.saturating_add(1);
        state.last_demand_frame = Some(frame_index);

        let was_hot = state.temperature.is_hot();
        let old_direction = state.temperature.direction();
        let observation = state.demand_history.observe(frame_index);

        match observation {
            DemandObservation::Adjacent { direction, run_len } => {
                if was_hot {
                    if old_direction != Some(direction) {
                        state.generation_epoch = state.generation_epoch.saturating_add(1);
                    }
                    state.temperature = DngCacheTemperature::Hot {
                        direction,
                        last_demand: now,
                    };
                } else if run_len >= self.policy.promotion_adjacent_requests {
                    state.generation_epoch = state.generation_epoch.saturating_add(1);
                    state.temperature = DngCacheTemperature::Hot {
                        direction,
                        last_demand: now,
                    };
                    self.instrumentation
                        .hot_promotions
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
            DemandObservation::RandomOrIsolated => {
                if was_hot {
                    state.generation_epoch = state.generation_epoch.saturating_add(1);
                    state.temperature = DngCacheTemperature::Cold;
                    let (evicted_frames, evicted_bytes) =
                        shrink_to_cold_locked(&mut state, self.policy, Some(frame_index));
                    self.record_eviction(evicted_frames, evicted_bytes);
                }
            }
        }

        Ok(())
    }

    fn finish_demand(&self, frame_index: usize) -> bool {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(_) => return false,
        };

        state.active_demands = state.active_demands.saturating_sub(1);
        let should_queue_prefetch = state.temperature.is_hot();

        let (evicted_frames, evicted_bytes) = if should_queue_prefetch {
            evict_to_limits_locked(
                &mut state,
                self.policy.hot_frame_capacity,
                self.policy.hot_byte_capacity,
                Some(frame_index),
            )
        } else {
            shrink_to_cold_locked(&mut state, self.policy, Some(frame_index))
        };
        self.record_eviction(evicted_frames, evicted_bytes);
        drop(state);

        should_queue_prefetch
    }

    fn finish_auxiliary_read(&self) -> bool {
        false
    }

    fn get_cached(&self, frame_index: usize) -> Result<Option<Arc<CachedDngFrame>>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("DNG cache mutex was poisoned"))?;

        if let Some(cached) = state.entries.get(&frame_index).cloned() {
            touch_lru_key(&mut state.lru_order, frame_index);
            return Ok(Some(cached));
        }

        Ok(None)
    }

    fn insert_generated(
        &self,
        generated: GeneratedDngFrame,
        generation_elapsed: Duration,
        protected_frame: Option<usize>,
    ) -> Result<Arc<CachedDngFrame>> {
        let insert_start = Instant::now();

        let frame_index = generated.frame_index;
        let timings = generated.timings;
        let bytes: Arc<[u8]> = Arc::from(generated.bytes.into_boxed_slice());
        let byte_len = u64::try_from(bytes.len()).context("DNG byte length does not fit u64")?;

        let cached = Arc::new(CachedDngFrame {
            frame_index,
            bytes,
            byte_len,
            timings,
        });

        let (evicted_frames, evicted_bytes) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("DNG cache mutex was poisoned"))?;

            if let Some(old) = state.entries.remove(&frame_index) {
                state.total_cached_bytes = state.total_cached_bytes.saturating_sub(old.byte_len);
                remove_lru_key(&mut state.lru_order, frame_index);
            }

            state.total_cached_bytes = state.total_cached_bytes.saturating_add(byte_len);
            state.lru_order.push_back(frame_index);
            state.entries.insert(frame_index, cached.clone());

            let protected_frame = protected_frame.or(Some(frame_index));
            if state.temperature.is_hot() {
                evict_to_limits_locked(
                    &mut state,
                    self.policy.hot_frame_capacity,
                    self.policy.hot_byte_capacity,
                    protected_frame,
                )
            } else {
                shrink_to_cold_locked(&mut state, self.policy, protected_frame)
            }
        };
        let cache_insert_time = insert_start.elapsed();

        self.record_generation_metrics(byte_len, timings, generation_elapsed, cache_insert_time);
        self.record_eviction(evicted_frames, evicted_bytes);
        self.instrumentation
            .last_generated_frame
            .store(frame_index, Ordering::Relaxed);

        Ok(cached)
    }

    fn record_generation_metrics(
        &self,
        byte_len: u64,
        timings: DngGenerationTimings,
        generation_elapsed: Duration,
        cache_insert_time: Duration,
    ) {
        self.instrumentation
            .frames_generated
            .fetch_add(1, Ordering::Relaxed);
        self.instrumentation
            .generated_bytes_total
            .fetch_add(byte_len, Ordering::Relaxed);
        self.instrumentation
            .generation_time_total_ns
            .fetch_add(duration_ns_u64(generation_elapsed), Ordering::Relaxed);

        self.instrumentation
            .raw_payload_bytes_total
            .fetch_add(timings.raw_payload_bytes, Ordering::Relaxed);
        self.instrumentation
            .decoded_pixel_bytes_total
            .fetch_add(timings.decoded_pixel_bytes, Ordering::Relaxed);
        self.instrumentation
            .final_dng_bytes_total
            .fetch_add(timings.final_dng_bytes, Ordering::Relaxed);

        self.instrumentation
            .dng_description_time_total_ns
            .fetch_add(
                duration_ns_u64(timings.dng_description_time),
                Ordering::Relaxed,
            );
        self.instrumentation.payload_read_time_total_ns.fetch_add(
            duration_ns_u64(timings.payload_read_time),
            Ordering::Relaxed,
        );
        self.instrumentation
            .decode_time_total_ns
            .fetch_add(duration_ns_u64(timings.decode_time), Ordering::Relaxed);
        self.instrumentation
            .dng_build_time_total_ns
            .fetch_add(duration_ns_u64(timings.dng_build_time), Ordering::Relaxed);
        self.instrumentation
            .cache_insert_time_total_ns
            .fetch_add(duration_ns_u64(cache_insert_time), Ordering::Relaxed);

        self.instrumentation.gpu_work_plan_time_total_ns.fetch_add(
            duration_ns_u64(timings.gpu_work_plan_time),
            Ordering::Relaxed,
        );
        self.instrumentation
            .gpu_cpu_prepare_time_total_ns
            .fetch_add(
                duration_ns_u64(timings.gpu_cpu_prepare_time),
                Ordering::Relaxed,
            );
        self.instrumentation
            .gpu_upload_time_total_ns
            .fetch_add(duration_ns_u64(timings.gpu_upload_time), Ordering::Relaxed);
        self.instrumentation
            .gpu_encode_submit_time_total_ns
            .fetch_add(
                duration_ns_u64(timings.gpu_encode_submit_time),
                Ordering::Relaxed,
            );
        self.instrumentation.gpu_wait_map_time_total_ns.fetch_add(
            duration_ns_u64(timings.gpu_wait_map_time),
            Ordering::Relaxed,
        );
        self.instrumentation
            .gpu_readback_convert_time_total_ns
            .fetch_add(
                duration_ns_u64(timings.gpu_readback_convert_time),
                Ordering::Relaxed,
            );
        self.instrumentation
            .gpu_dispatch_readback_time_total_ns
            .fetch_add(
                duration_ns_u64(timings.gpu_dispatch_readback_time),
                Ordering::Relaxed,
            );
    }

    fn record_eviction(&self, evicted_frames: u64, evicted_bytes: u64) {
        if evicted_frames > 0 {
            self.instrumentation
                .frames_evicted
                .fetch_add(evicted_frames, Ordering::Relaxed);
            self.instrumentation
                .evicted_bytes_total
                .fetch_add(evicted_bytes, Ordering::Relaxed);
        }
    }

    fn ensure_generator_locked<'a>(
        &'a self,
        guard: &'a mut Option<DngFrameGenerator>,
    ) -> Result<&'a mut DngFrameGenerator> {
        let should_construct = guard.is_none();
        if should_construct {
            initialize_mount_lifetime_resource(guard, || {
                DngFrameGenerator::open_with_config(&self.source_path, self.generation_config)
            })?;
            self.instrumentation
                .generator_created
                .fetch_add(1, Ordering::Relaxed);
        }

        guard
            .as_mut()
            .context("DNG generator was not available after creation")
    }

    fn drop_generator_for_shutdown(&self) {
        let dropped = match self.generator.lock() {
            Ok(mut guard) => take_mount_lifetime_resource(&mut guard).is_some(),
            Err(_) => false,
        };

        if dropped {
            self.instrumentation
                .generator_dropped
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn prefetch_hint_for_anchor(&self, anchor_frame: usize) -> Option<PrefetchHint> {
        let state = self.state.lock().ok()?;
        let DngCacheTemperature::Hot { direction, .. } = state.temperature else {
            return None;
        };
        if self.policy.hot_prefetch_target == 0 {
            return None;
        }
        Some(PrefetchHint {
            anchor_frame,
            direction,
            epoch: state.generation_epoch,
        })
    }

    fn generate_prefetch_batch(&self, hints: &[PrefetchHint]) -> Result<()> {
        // Speculative work checks for active demand and uses try_lock for the
        // serialized generator, so it abandons the batch instead of waiting on
        // a generator already serving foreground work.
        if hints.is_empty() || self.shutdown_requested.load(Ordering::Acquire) {
            return Ok(());
        }

        let mut selected = None;
        for &hint in hints.iter().rev() {
            if self.prefetch_hint_is_current(hint) {
                selected = Some(hint);
                break;
            }
            self.instrumentation
                .prefetch_stale
                .fetch_add(1, Ordering::Relaxed);
        }

        let Some(hint) = selected else {
            return Ok(());
        };

        let target_frames = self.prefetch_targets(hint)?;
        if target_frames.is_empty() {
            return Ok(());
        }

        if !self.mark_prefetch_active(hint)? {
            return Ok(());
        }

        let batch_start = Instant::now();
        let mut generated_count = 0_usize;
        let result = (|| {
            if self.shutdown_requested.load(Ordering::Acquire)
                || !self.prefetch_hint_is_current(hint)
                || self.has_active_demand()
            {
                self.instrumentation
                    .prefetch_stale
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }

            let Ok(mut generator_guard) = self.generator.try_lock() else {
                self.instrumentation
                    .prefetch_generator_busy
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(());
            };

            let generator = self.ensure_generator_locked(&mut generator_guard)?;
            let generation_start = Instant::now();
            let generated_frames = submit_prefetch_batch(&target_frames, |batch| {
                generator
                    .generate_frames_for_prefetch(batch, self.execution_policy)
                    .with_context(|| {
                        format!(
                            "failed to prefetch DNG hot-window batch with {} frames",
                            batch.len()
                        )
                    })
            })?;
            let generation_elapsed = generation_start.elapsed();

            if !self.prefetch_hint_is_current(hint) {
                self.instrumentation
                    .prefetch_stale
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }

            let generated_len = generated_frames.len();
            let elapsed_divisor = u32::try_from(generated_len).unwrap_or(u32::MAX).max(1);
            let per_frame_generation_elapsed = generation_elapsed / elapsed_divisor;
            for generated in generated_frames {
                self.insert_generated(generated, per_frame_generation_elapsed, None)?;
                generated_count = generated_count.saturating_add(1);
            }
            drop(generator_guard);

            self.instrumentation.prefetch_generated.fetch_add(
                u64::try_from(generated_count).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );

            tracing::info!(
                "mcraw4vulkan DNG hot prefetch: direction={} epoch={} anchor={} requested={} generated={} wall_ms={:.2}",
                hint.direction.label(),
                hint.epoch,
                hint.anchor_frame,
                target_frames.len(),
                generated_count,
                duration_ms(batch_start.elapsed()),
            );

            Ok(())
        })();

        self.mark_prefetch_inactive();
        result
    }

    fn has_active_demand(&self) -> bool {
        let Ok(state) = self.state.lock() else {
            return true;
        };
        state.active_demands > 0
    }

    fn prefetch_targets(&self, hint: PrefetchHint) -> Result<Vec<usize>> {
        let mut targets = Vec::with_capacity(self.policy.hot_prefetch_target);
        let mut next = hint
            .direction
            .next_frame(hint.anchor_frame, self.frame_count);

        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("DNG cache mutex was poisoned"))?;

        while let Some(frame_index) = next {
            if targets.len() >= self.policy.hot_prefetch_target {
                break;
            }

            if state.entries.contains_key(&frame_index) {
                self.instrumentation
                    .prefetch_skipped_cached
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                targets.push(frame_index);
            }

            next = hint.direction.next_frame(frame_index, self.frame_count);
        }

        Ok(targets)
    }

    fn prefetch_hint_is_current(&self, hint: PrefetchHint) -> bool {
        let Ok(state) = self.state.lock() else {
            return false;
        };
        matches!(
            state.temperature,
            DngCacheTemperature::Hot { direction, .. }
                if direction == hint.direction && state.generation_epoch == hint.epoch
        )
    }

    fn mark_prefetch_active(&self, hint: PrefetchHint) -> Result<bool> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("DNG cache mutex was poisoned"))?;
        if !matches!(
            state.temperature,
            DngCacheTemperature::Hot { direction, .. }
                if direction == hint.direction && state.generation_epoch == hint.epoch
        ) {
            self.instrumentation
                .prefetch_stale
                .fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        state.active_prefetches = state.active_prefetches.saturating_add(1);
        Ok(true)
    }

    fn mark_prefetch_inactive(&self) {
        {
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            state.active_prefetches = state.active_prefetches.saturating_sub(1);
        };
    }

    fn demote_if_idle(&self, now: Instant) {
        let demoted = {
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            if !should_demote_locked(&state, self.policy, now) {
                return;
            }

            state.temperature = DngCacheTemperature::Cold;
            state.generation_epoch = state.generation_epoch.saturating_add(1);
            let preserve = state.last_demand_frame;
            let (evicted_frames, evicted_bytes) =
                shrink_to_cold_locked(&mut state, self.policy, preserve);
            self.record_eviction(evicted_frames, evicted_bytes);
            true
        };

        if demoted {
            self.instrumentation
                .hot_demotions
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn initialize_mount_lifetime_resource<T, E>(
    slot: &mut Option<T>,
    initialize: impl FnOnce() -> std::result::Result<T, E>,
) -> std::result::Result<bool, E> {
    if slot.is_some() {
        return Ok(false);
    }

    *slot = Some(initialize()?);
    Ok(true)
}

fn take_mount_lifetime_resource<T>(slot: &mut Option<T>) -> Option<T> {
    slot.take()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DngFrameCacheStats {
    pub cached_frames: usize,
    pub total_cached_bytes: u64,
    pub max_frames: usize,
    pub max_cached_bytes: u64,
    pub cold_frame_capacity: usize,
    pub hot_prefetch_target: usize,
    pub hot_history_capacity: usize,
    pub hot_grace_ms: u64,
    pub is_hot: bool,
    pub hot_direction: Option<&'static str>,
    pub active_demands: usize,
    pub active_prefetches: usize,
    pub generation_epoch: u64,
    pub cache_hits: u64,
    pub cache_hits_after_wait: u64,
    pub cache_misses: u64,
    pub metadata_len_queries: u64,
    pub metadata_len_cached_hits: u64,

    pub frames_generated: u64,
    pub generated_bytes_total: u64,
    pub generation_time_total_ns: u64,

    pub raw_payload_bytes_total: u64,
    pub decoded_pixel_bytes_total: u64,
    pub final_dng_bytes_total: u64,

    pub dng_description_time_total_ns: u64,
    pub payload_read_time_total_ns: u64,
    pub decode_time_total_ns: u64,
    pub dng_build_time_total_ns: u64,
    pub cache_insert_time_total_ns: u64,

    pub gpu_work_plan_time_total_ns: u64,
    pub gpu_cpu_prepare_time_total_ns: u64,
    pub gpu_upload_time_total_ns: u64,
    pub gpu_encode_submit_time_total_ns: u64,
    pub gpu_wait_map_time_total_ns: u64,
    pub gpu_readback_convert_time_total_ns: u64,
    pub gpu_dispatch_readback_time_total_ns: u64,

    pub frames_evicted: u64,
    pub evicted_bytes_total: u64,
    pub last_generated_frame: Option<usize>,

    pub prefetch_requested: u64,
    pub prefetch_queued: u64,
    pub prefetch_queue_full: u64,
    pub prefetch_skipped_cached: u64,
    pub prefetch_generator_busy: u64,
    pub prefetch_generated: u64,
    pub prefetch_errors: u64,
    pub prefetch_stale: u64,
    pub hot_promotions: u64,
    pub hot_demotions: u64,
    pub generator_created: u64,
    pub generator_dropped: u64,
}

impl DngFrameCacheStats {
    pub fn average_generation_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.generation_time_total_ns)
    }

    pub fn average_description_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.dng_description_time_total_ns)
    }

    pub fn average_payload_read_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.payload_read_time_total_ns)
    }

    pub fn average_decode_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.decode_time_total_ns)
    }

    pub fn average_dng_build_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.dng_build_time_total_ns)
    }

    pub fn average_cache_insert_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.cache_insert_time_total_ns)
    }

    pub fn average_gpu_work_plan_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.gpu_work_plan_time_total_ns)
    }

    pub fn average_gpu_cpu_prepare_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.gpu_cpu_prepare_time_total_ns)
    }

    pub fn average_gpu_upload_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.gpu_upload_time_total_ns)
    }

    pub fn average_gpu_encode_submit_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.gpu_encode_submit_time_total_ns)
    }

    pub fn average_gpu_wait_map_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.gpu_wait_map_time_total_ns)
    }

    pub fn average_gpu_readback_convert_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.gpu_readback_convert_time_total_ns)
    }

    pub fn average_gpu_dispatch_readback_time_ms(&self) -> f64 {
        self.average_ns_as_ms(self.gpu_dispatch_readback_time_total_ns)
    }

    pub fn average_raw_payload_mib(&self) -> f64 {
        self.average_bytes_as_mib(self.raw_payload_bytes_total)
    }

    pub fn average_decoded_pixels_mib(&self) -> f64 {
        self.average_bytes_as_mib(self.decoded_pixel_bytes_total)
    }

    pub fn average_final_dng_mib(&self) -> f64 {
        self.average_bytes_as_mib(self.final_dng_bytes_total)
    }

    fn average_ns_as_ms(&self, value: u64) -> f64 {
        if self.frames_generated == 0 {
            0.0
        } else {
            value as f64 / self.frames_generated as f64 / 1_000_000.0
        }
    }

    fn average_bytes_as_mib(&self, value: u64) -> f64 {
        if self.frames_generated == 0 {
            0.0
        } else {
            value as f64 / self.frames_generated as f64 / 1024.0 / 1024.0
        }
    }
}

fn duration_ns_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn duration_ms_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn spawn_prefetch_worker(
    inner: Arc<DngFrameByteCacheInner>,
    receiver: Receiver<PrefetchHint>,
) -> Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("mcraw4vulkan-dng-cache".to_string())
        .spawn(move || prefetch_worker_loop(inner, receiver))
        .context("failed to spawn DNG cache worker")
}

fn prefetch_worker_loop(inner: Arc<DngFrameByteCacheInner>, receiver: Receiver<PrefetchHint>) {
    loop {
        if inner.shutdown_requested.load(Ordering::Acquire) {
            break;
        }

        let timeout = demotion_timeout(&inner);
        let first_hint = match receiver.recv_timeout(timeout) {
            Ok(hint) => hint,
            Err(RecvTimeoutError::Timeout) => {
                inner.demote_if_idle(Instant::now());
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };

        if inner.shutdown_requested.load(Ordering::Acquire) {
            break;
        }

        let mut hints = Vec::with_capacity(inner.policy.queue_capacity());
        hints.push(first_hint);

        let mut disconnected = false;
        while hints.len() < inner.policy.queue_capacity() {
            match receiver.try_recv() {
                Ok(hint) => hints.push(hint),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        if inner.generate_prefetch_batch(&hints).is_err() {
            inner.instrumentation.prefetch_errors.fetch_add(
                u64::try_from(hints.len()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        }

        inner.demote_if_idle(Instant::now());

        if disconnected {
            break;
        }
    }
}

fn demotion_timeout(inner: &DngFrameByteCacheInner) -> Duration {
    let Ok(state) = inner.state.lock() else {
        return inner.policy.hot_grace;
    };

    let Some(last_demand) = state.temperature.last_demand() else {
        return inner.policy.hot_grace;
    };

    let elapsed = last_demand.elapsed();
    inner.policy.hot_grace.saturating_sub(elapsed)
}

fn should_demote_locked(state: &CacheState, policy: DngCachePolicy, now: Instant) -> bool {
    if state.active_demands != 0 || state.active_prefetches != 0 {
        return false;
    }

    match state.temperature {
        DngCacheTemperature::Cold => false,
        DngCacheTemperature::Hot { last_demand, .. } => {
            now.duration_since(last_demand) >= policy.hot_grace
        }
    }
}

fn submit_prefetch_batch<T>(
    target_frames: &[usize],
    generate: impl FnOnce(&[usize]) -> Result<Vec<T>>,
) -> Result<Vec<T>> {
    generate(target_frames)
}

fn touch_lru_key(order: &mut VecDeque<usize>, key: usize) {
    remove_lru_key(order, key);
    order.push_back(key);
}

fn remove_lru_key(order: &mut VecDeque<usize>, key: usize) {
    order.retain(|&candidate| candidate != key);
}

fn shrink_to_cold_locked(
    state: &mut CacheState,
    policy: DngCachePolicy,
    protected_frame: Option<usize>,
) -> (u64, u64) {
    evict_to_limits_locked(
        state,
        policy.cold_frame_capacity,
        policy.cold_byte_capacity,
        protected_frame,
    )
}

fn evict_to_limits_locked(
    state: &mut CacheState,
    max_frames: usize,
    max_bytes: u64,
    protected_frame: Option<usize>,
) -> (u64, u64) {
    let mut evicted_frames = 0u64;
    let mut evicted_bytes = 0u64;
    let max_frames = max_frames.max(1);
    let max_bytes = max_bytes.max(1);

    while cache_exceeds_limits(state, max_frames, max_bytes, protected_frame) {
        let Some(frame_index) = next_evictable_lru(&mut state.lru_order, protected_frame) else {
            break;
        };

        let Some(evicted) = state.entries.remove(&frame_index) else {
            continue;
        };

        state.total_cached_bytes = state.total_cached_bytes.saturating_sub(evicted.byte_len);
        evicted_frames = evicted_frames.saturating_add(1);
        evicted_bytes = evicted_bytes.saturating_add(evicted.byte_len);
    }

    (evicted_frames, evicted_bytes)
}

fn cache_exceeds_limits(
    state: &CacheState,
    max_frames: usize,
    max_bytes: u64,
    protected_frame: Option<usize>,
) -> bool {
    if state.entries.len() > max_frames {
        return true;
    }

    if state.total_cached_bytes <= max_bytes {
        return false;
    }

    state
        .entries
        .keys()
        .any(|&frame_index| Some(frame_index) != protected_frame)
}

fn next_evictable_lru(
    order: &mut VecDeque<usize>,
    protected_frame: Option<usize>,
) -> Option<usize> {
    let position = order
        .iter()
        .position(|&frame_index| Some(frame_index) != protected_frame)?;
    order.remove(position)
}
