use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use mcraw4vulkan_dngwriter::DngSinkVignetteMode;
use mcraw4vulkan_gpu::GpuBackendPreference;
use mcraw4vulkan_mcrawcontainer::McrawContainer;

use crate::virtual_fs::CachedDngFrame;
use crate::virtual_fs::DEFAULT_DNG_HOT_FRAME_CAPACITY;
use crate::virtual_fs::DEFAULT_DNG_HOT_PREFETCH_TARGET;
use crate::virtual_fs::DngFrameByteCache;
use crate::virtual_fs::DngFrameCacheStats;
use crate::virtual_fs::DngGenerationBackend;
use crate::virtual_fs::DngGenerationConfig;
use crate::virtual_fs::DngGenerationExecutionPolicy;
use crate::virtual_fs::InodeMap;
use crate::virtual_fs::LazyAudioWav;
use crate::virtual_fs::LazyAudioWavConfig;
use crate::virtual_fs::VirtualDirEntry;
use crate::virtual_fs::VirtualFileKind;
use crate::virtual_fs::VirtualFileMetadata;
use crate::virtual_fs::VirtualMetadataProvider;
use crate::virtual_fs::VirtualNode;
use crate::virtual_fs::VirtualRuntimeStats;
use crate::virtual_fs::VirtualRuntimeStatsSnapshot;
use crate::virtual_timestamp::virtual_timestamp_for_input_path;

pub const DEFAULT_DNG_CACHE_FRAME_CAPACITY: usize = DEFAULT_DNG_HOT_FRAME_CAPACITY;
pub const DEFAULT_PREFETCH_FORWARD_FRAMES: usize = DEFAULT_DNG_HOT_PREFETCH_TARGET;
// These named profiles predate the 21/4 production default and remain stable
// for explicit callers and cross-machine comparison.
pub const LOWER_MEMORY_DNG_CACHE_FRAME_CAPACITY: usize = 32;
pub const LOWER_MEMORY_PREFETCH_FORWARD_FRAMES: usize = 8;
pub const LOWEST_MEMORY_DNG_CACHE_FRAME_CAPACITY: usize = 16;
pub const LOWEST_MEMORY_PREFETCH_FORWARD_FRAMES: usize = 4;

// Platform-neutral virtual filesystem facade.
//
// This is the shared callback-shaped layer used by platform mount adapters. It
// owns virtual metadata, DNG byte cache access, runtime stats, and prefetch
// triggering. OS-specific layers should remain thin translation layers around
// these methods.
pub struct VirtualFileSystem {
    provider: VirtualMetadataProvider,
    cache: Arc<DngFrameByteCache>,
    runtime_stats: Arc<VirtualRuntimeStats>,
    open_dng_handles: Mutex<OpenDngHandleTable>,
    prefetch_forward_frames: usize,
}

// Platform-neutral read result for virtual file data.
//
// This owns the cached Arc<[u8]> so its slice remains valid while an adapter
// replies to the read. DNG reads share complete frame bytes; lazy audio reads
// own only the requested range.
pub struct VirtualReadData {
    bytes: Arc<[u8]>,
    start: usize,
    end: usize,
}

impl VirtualReadData {
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes[self.start..self.end]
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Default)]
struct OpenDngHandleTable {
    handles: HashMap<u64, OpenDngHandle>,
}

#[derive(Debug)]
struct OpenDngHandle {
    // After the first read, an open handle pins one immutable complete DNG until
    // release. Handles for the same frame share the Arc, so cache eviction cannot
    // change bytes partway through an open/read/release lifecycle.
    inode: u64,
    frame_index: usize,
    pinned: Option<Arc<CachedDngFrame>>,
    demand_recorded: bool,
}

impl OpenDngHandleTable {
    fn insert(&mut self, handle_id: u64, inode: u64, frame_index: usize) {
        self.handles.insert(
            handle_id,
            OpenDngHandle {
                inode,
                frame_index,
                pinned: None,
                demand_recorded: false,
            },
        );
    }

    fn pinned_for_frame_except(
        &self,
        frame_index: usize,
        excluded_handle_id: u64,
    ) -> Option<Arc<CachedDngFrame>> {
        self.handles
            .iter()
            .filter(|(handle_id, _)| **handle_id != excluded_handle_id)
            .find_map(|(_, handle)| {
                (handle.frame_index == frame_index)
                    .then(|| handle.pinned.as_ref().cloned())
                    .flatten()
            })
    }

    fn stats(&self) -> VirtualDngHandleStats {
        let mut unique_frames = Vec::new();
        let mut unique_pinned_bytes = 0_u64;

        for handle in self.handles.values() {
            let Some(pinned) = handle.pinned.as_ref() else {
                continue;
            };
            if unique_frames.contains(&handle.frame_index) {
                continue;
            }
            unique_frames.push(handle.frame_index);
            unique_pinned_bytes = unique_pinned_bytes.saturating_add(pinned.byte_len);
        }

        VirtualDngHandleStats {
            active_handles: self.handles.len(),
            unique_pinned_frames: unique_frames.len(),
            unique_pinned_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VirtualDngHandleStats {
    pub active_handles: usize,
    pub unique_pinned_frames: usize,
    pub unique_pinned_bytes: u64,
}

// Configuration for the platform-neutral virtual filesystem facade.
//
// GPU DNG generation is the default and CPU remains a selectable fallback. The
// cache frame and prefetch fields configure the shared hot-window policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualFileSystemConfig {
    pub dng_backend: DngGenerationBackend,
    pub dng_vignette_mode: DngSinkVignetteMode,
    pub dng_generation_execution_policy: DngGenerationExecutionPolicy,
    pub max_cached_dng_frames: usize,
    pub prefetch_forward_frames: usize,
}

impl Default for VirtualFileSystemConfig {
    fn default() -> Self {
        Self {
            dng_backend: DngGenerationBackend::Gpu {
                backend_preference: GpuBackendPreference::VulkanOnly,
            },
            dng_vignette_mode: DngSinkVignetteMode::default(),
            dng_generation_execution_policy: DngGenerationExecutionPolicy::Inflight2Default,
            max_cached_dng_frames: DEFAULT_DNG_CACHE_FRAME_CAPACITY,
            prefetch_forward_frames: DEFAULT_PREFETCH_FORWARD_FRAMES,
        }
    }
}

impl VirtualFileSystemConfig {
    pub fn lower_memory() -> Self {
        Self {
            max_cached_dng_frames: LOWER_MEMORY_DNG_CACHE_FRAME_CAPACITY,
            prefetch_forward_frames: LOWER_MEMORY_PREFETCH_FORWARD_FRAMES,
            ..Self::default()
        }
    }

    pub fn lowest_memory() -> Self {
        Self {
            max_cached_dng_frames: LOWEST_MEMORY_DNG_CACHE_FRAME_CAPACITY,
            prefetch_forward_frames: LOWEST_MEMORY_PREFETCH_FORWARD_FRAMES,
            ..Self::default()
        }
    }
}

impl VirtualFileSystem {
    // Open one .mcraw clip and build the virtual filesystem facade.
    //
    // This creates:
    // - deterministic two-level inode/name map
    // - bounded DNG byte cache
    // - stable lazy BW64 WAV range source from raw decoded MCRAW audio
    // - virtual metadata provider with capture-time or fallback timestamp
    // - runtime instrumentation counters for filesystem adapter analysis
    // - best-effort directional prefetch after sequential logical demand
    pub fn open_clip(path: &Path, config: VirtualFileSystemConfig) -> Result<Self> {
        let container = McrawContainer::open(path).with_context(|| {
            format!(
                "failed to open clip for virtual filesystem: {}",
                path.display()
            )
        })?;

        let clip_stem = clip_stem(path)?;
        let frame_count = container.frame_count();
        let timestamp = virtual_timestamp_for_input_path(path);
        let has_audio = container.audio_info().is_some();

        let inode_map = InodeMap::new_with_audio(clip_stem, frame_count, has_audio);

        let cache = Arc::new(DngFrameByteCache::open_with_hot_window(
            path,
            DngGenerationConfig {
                backend: config.dng_backend,
                vignette_mode: config.dng_vignette_mode,
                execution_policy: config.dng_generation_execution_policy,
            },
            config.max_cached_dng_frames,
            config.prefetch_forward_frames,
        )?);

        // Expose the mounted Resolve/export WAV from the raw AUDIO_DATA sample
        // span without materializing the full BW64 file in memory.
        let audio_wav = has_audio
            .then(|| LazyAudioWav::from_container(container, LazyAudioWavConfig::default()))
            .transpose()?;

        let provider = VirtualMetadataProvider::new_optional_audio_with_timestamp(
            inode_map,
            cache.clone(),
            audio_wav,
            timestamp,
        );

        Ok(Self {
            provider,
            cache,
            runtime_stats: Arc::new(VirtualRuntimeStats::new()),
            open_dng_handles: Mutex::new(OpenDngHandleTable::default()),
            prefetch_forward_frames: config.prefetch_forward_frames,
        })
    }

    pub fn provider(&self) -> &VirtualMetadataProvider {
        &self.provider
    }

    pub fn inode_map(&self) -> &InodeMap {
        self.provider.inode_map()
    }

    pub fn cache_stats(&self) -> Result<DngFrameCacheStats> {
        self.cache.stats()
    }

    pub fn stats_snapshot(&self) -> Result<VirtualFileSystemStatsSnapshot> {
        Ok(VirtualFileSystemStatsSnapshot {
            runtime: self.runtime_stats.snapshot(),
            cache: self.cache.stats()?,
            dng_handles: self.dng_handle_stats(),
        })
    }

    pub fn stats_summary_line(&self) -> Result<String> {
        let snapshot = self.stats_snapshot()?;
        let read = snapshot.runtime.read;
        let cache = snapshot.cache;

        Ok(format!(
            "reads={} served_mib={:.1} avg_req_kib={} min_req={} max_req={} offset0={} empty_reads={} lookup={} getattr={} readdir={} cache_hits={} cache_wait_hits={} cache_misses={} generated={} evicted={} cached_frames={} cached_mib={:.1} max_frames={} max_cached_mib={:.1} cold_frames={} hot={} hot_direction={:?} hot_prefetch={} hot_history={} hot_grace_ms={} epoch={} active_demands={} active_prefetches={} active_handles={} pinned_frames={} pinned_mib={:.1} avg_gen_ms={:.2} avg_desc_ms={:.2} avg_payload_ms={:.2} avg_decode_ms={:.2} avg_dng_ms={:.2} avg_insert_ms={:.2} avg_gpu_work_plan_ms={:.2} avg_gpu_cpu_prepare_ms={:.2} avg_gpu_upload_ms={:.2} avg_gpu_submit_ms={:.2} avg_gpu_wait_map_ms={:.2} avg_gpu_readback_convert_ms={:.2} avg_gpu_dispatch_readback_ms={:.2} avg_raw_mib={:.1} avg_decoded_mib={:.1} avg_final_dng_mib={:.1} last_frame={:?} configured_prefetch={} prefetch_req={} prefetch_queued={} prefetch_full={} prefetch_skip_cached={} prefetch_busy={} prefetch_gen={} prefetch_stale={} prefetch_err={} hot_promotions={} hot_demotions={} generator_created={} generator_dropped={}",
            read.read_count,
            read.read_bytes_total as f64 / 1024.0 / 1024.0,
            read.average_requested_size() / 1024,
            read.read_size_min,
            read.read_size_max,
            read.read_offset_zero_count,
            read.read_empty_count,
            snapshot.runtime.lookup_count,
            snapshot.runtime.getattr_count,
            snapshot.runtime.readdir_count,
            cache.cache_hits,
            cache.cache_hits_after_wait,
            cache.cache_misses,
            cache.frames_generated,
            cache.frames_evicted,
            cache.cached_frames,
            cache.total_cached_bytes as f64 / 1024.0 / 1024.0,
            cache.max_frames,
            cache.max_cached_bytes as f64 / 1024.0 / 1024.0,
            cache.cold_frame_capacity,
            cache.is_hot,
            cache.hot_direction,
            cache.hot_prefetch_target,
            cache.hot_history_capacity,
            cache.hot_grace_ms,
            cache.generation_epoch,
            cache.active_demands,
            cache.active_prefetches,
            snapshot.dng_handles.active_handles,
            snapshot.dng_handles.unique_pinned_frames,
            snapshot.dng_handles.unique_pinned_bytes as f64 / 1024.0 / 1024.0,
            cache.average_generation_time_ms(),
            cache.average_description_time_ms(),
            cache.average_payload_read_time_ms(),
            cache.average_decode_time_ms(),
            cache.average_dng_build_time_ms(),
            cache.average_cache_insert_time_ms(),
            cache.average_gpu_work_plan_time_ms(),
            cache.average_gpu_cpu_prepare_time_ms(),
            cache.average_gpu_upload_time_ms(),
            cache.average_gpu_encode_submit_time_ms(),
            cache.average_gpu_wait_map_time_ms(),
            cache.average_gpu_readback_convert_time_ms(),
            cache.average_gpu_dispatch_readback_time_ms(),
            cache.average_raw_payload_mib(),
            cache.average_decoded_pixels_mib(),
            cache.average_final_dng_mib(),
            cache.last_generated_frame,
            self.prefetch_forward_frames,
            cache.prefetch_requested,
            cache.prefetch_queued,
            cache.prefetch_queue_full,
            cache.prefetch_skipped_cached,
            cache.prefetch_generator_busy,
            cache.prefetch_generated,
            cache.prefetch_stale,
            cache.prefetch_errors,
            cache.hot_promotions,
            cache.hot_demotions,
            cache.generator_created,
            cache.generator_dropped,
        ))
    }

    pub fn record_open(&self) {
        self.runtime_stats.record_open();
    }

    pub fn record_opendir(&self) {
        self.runtime_stats.record_opendir();
    }

    pub fn open_file_handle(&self, inode: u64, handle_id: u64) -> Result<()> {
        let Some(VirtualNode::DngFrame { frame_index }) =
            self.provider.inode_map().node_for_inode(inode)
        else {
            return Ok(());
        };

        self.open_dng_handles
            .lock()
            .map_err(|_| anyhow::anyhow!("DNG open-handle table mutex was poisoned"))?
            .insert(handle_id, inode, frame_index);

        Ok(())
    }

    pub fn release_file_handle(&self, handle_id: u64) {
        let _released = self
            .open_dng_handles
            .lock()
            .ok()
            .and_then(|mut handles| handles.handles.remove(&handle_id));
    }

    pub fn dng_handle_stats(&self) -> VirtualDngHandleStats {
        self.open_dng_handles
            .lock()
            .map(|handles| handles.stats())
            .unwrap_or_default()
    }

    fn dng_frame_bytes_for_read(
        &self,
        inode: u64,
        frame_index: usize,
        handle_id: Option<u64>,
    ) -> Result<Arc<[u8]>> {
        let Some(handle_id) = handle_id else {
            let Some(file_bytes) = self.provider.bytes_for_inode_with_demand(inode, true)? else {
                anyhow::bail!("DNG inode {inode} did not return bytes");
            };
            return Ok(file_bytes.bytes);
        };

        let active_pin = {
            let mut handles = self
                .open_dng_handles
                .lock()
                .map_err(|_| anyhow::anyhow!("DNG open-handle table mutex was poisoned"))?;

            let needs_insert = handles
                .handles
                .get(&handle_id)
                .map(|handle| handle.inode != inode || handle.frame_index != frame_index)
                .unwrap_or(true);
            if needs_insert {
                handles.insert(handle_id, inode, frame_index);
            }

            if let Some(pinned) = handles
                .handles
                .get(&handle_id)
                .and_then(|handle| handle.pinned.as_ref().cloned())
            {
                return Ok(pinned.bytes.clone());
            }

            handles.pinned_for_frame_except(frame_index, handle_id)
        };

        if let Some(pinned) = active_pin {
            self.cache.observe_logical_demand(frame_index)?;
            let mut handles = self
                .open_dng_handles
                .lock()
                .map_err(|_| anyhow::anyhow!("DNG open-handle table mutex was poisoned"))?;
            if let Some(handle) = handles.handles.get_mut(&handle_id) {
                handle.pinned = Some(pinned.clone());
                handle.demand_recorded = true;
            }
            return Ok(pinned.bytes.clone());
        }

        // The first read on a logical handle is the single demand event for
        // that open virtual DNG. It may start at any byte offset.
        let cached = self.cache.get_or_generate_for_read(frame_index, true)?;
        let mut handles = self
            .open_dng_handles
            .lock()
            .map_err(|_| anyhow::anyhow!("DNG open-handle table mutex was poisoned"))?;
        let handle = handles
            .handles
            .entry(handle_id)
            .or_insert_with(|| OpenDngHandle {
                inode,
                frame_index,
                pinned: None,
                demand_recorded: false,
            });
        if handle.inode != inode || handle.frame_index != frame_index {
            *handle = OpenDngHandle {
                inode,
                frame_index,
                pinned: None,
                demand_recorded: false,
            };
        }
        handle.pinned = Some(cached.clone());
        handle.demand_recorded = true;

        Ok(cached.bytes.clone())
    }

    // Platform adapter lookup callback shape.
    //
    // Unknown parents or names return Ok(None), matching ordinary lookup miss
    // behavior. The root contains one clip directory; the clip directory contains
    // the DNG sequence and BW64 WAV file.
    pub fn lookup(&self, parent_inode: u64, name: &OsStr) -> Result<Option<VirtualFileMetadata>> {
        self.runtime_stats.record_lookup();
        self.provider.metadata_for_child_name(parent_inode, name)
    }

    // Platform adapter getattr callback shape.
    pub fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
        self.runtime_stats.record_getattr();
        self.provider.metadata_for_inode(inode)
    }

    // Platform adapter readdir callback shape.
    //
    // This virtual filesystem has two directory levels:
    //
    //   root -> clip directory
    //   clip directory -> DNG frames + BW64 WAV
    //
    // Non-directory inodes return Ok(None). The platform callback translates
    // that into the appropriate filesystem error.
    pub fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>> {
        self.runtime_stats.record_readdir();

        let Some(metadata) = self.provider.metadata_for_inode(inode)? else {
            return Ok(None);
        };

        if metadata.kind != VirtualFileKind::Directory {
            return Ok(None);
        }

        Ok(self.provider.directory_entries(inode))
    }

    pub fn parent_inode_for_directory(&self, inode: u64) -> Option<u64> {
        self.provider.parent_inode_for_directory(inode)
    }

    // Zero-extra-allocation read path for platform filesystem adapters.
    //
    // This returns an object that owns the cached Arc<Vec<u8>> and exposes the
    // requested byte range as a slice. Unlike read(), this avoids an extra copy
    // for cached DNG frame bytes. Lazy audio reads allocate only the requested
    // output range, not the complete WAV/BW64 file.
    pub fn read_data_for_handle(
        &self,
        inode: u64,
        handle_id: u64,
        offset: u64,
        size: u32,
    ) -> Result<Option<VirtualReadData>> {
        self.read_data_common(inode, Some(handle_id), offset, size)
    }

    pub fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<VirtualReadData>> {
        self.read_data_common(inode, None, offset, size)
    }

    fn read_data_common(
        &self,
        inode: u64,
        handle_id: Option<u64>,
        offset: u64,
        size: u32,
    ) -> Result<Option<VirtualReadData>> {
        if matches!(
            self.provider.inode_map().node_for_inode(inode),
            Some(VirtualNode::AudioWav)
        ) {
            let bytes = self
                .provider
                .read_range(inode, offset, size)?
                .context("audio inode did not return read bytes")?;
            let returned_len = bytes.len();
            self.runtime_stats
                .record_read_result(offset, size, returned_len);

            return Ok(Some(VirtualReadData {
                bytes: Arc::from(bytes.into_boxed_slice()),
                start: 0,
                end: returned_len,
            }));
        }

        let Some(node) = self.provider.inode_map().node_for_inode(inode) else {
            self.runtime_stats.record_read_none();
            return Ok(None);
        };

        let file_bytes = match node {
            VirtualNode::Root | VirtualNode::ClipDirectory => {
                self.runtime_stats.record_read_none();
                return Ok(None);
            }
            VirtualNode::AudioWav => unreachable!("audio reads are handled above"),
            VirtualNode::DngFrame { frame_index } => {
                self.dng_frame_bytes_for_read(inode, frame_index, handle_id)?
            }
        };

        let start = usize::try_from(offset).context("read offset does not fit usize")?;
        let requested = usize::try_from(size).context("read size does not fit usize")?;
        let bytes_len = file_bytes.len();

        let (start, end) = if start >= bytes_len {
            (bytes_len, bytes_len)
        } else {
            let end = start.saturating_add(requested).min(bytes_len);
            (start, end)
        };

        let returned_len = end.saturating_sub(start);

        self.runtime_stats
            .record_read_result(offset, size, returned_len);

        Ok(Some(VirtualReadData {
            bytes: file_bytes,
            start,
            end,
        }))
    }
}

// Snapshot combining virtual filesystem operation counters and DNG cache counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualFileSystemStatsSnapshot {
    pub runtime: VirtualRuntimeStatsSnapshot,
    pub cache: DngFrameCacheStats,
    pub dng_handles: VirtualDngHandleStats,
}

// Return the input filename stem, used as the virtual directory, DNG frame, and
// WAV prefix.
fn clip_stem(input_path: &Path) -> Result<String> {
    let stem = input_path
        .file_stem()
        .and_then(|value| value.to_str())
        .context("input path has no valid UTF-8 file stem")?;

    Ok(stem.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        CachedDngFrame, DEFAULT_DNG_CACHE_FRAME_CAPACITY, DEFAULT_PREFETCH_FORWARD_FRAMES,
        DngGenerationExecutionPolicy, LOWER_MEMORY_DNG_CACHE_FRAME_CAPACITY,
        LOWER_MEMORY_PREFETCH_FORWARD_FRAMES, LOWEST_MEMORY_DNG_CACHE_FRAME_CAPACITY,
        LOWEST_MEMORY_PREFETCH_FORWARD_FRAMES, OpenDngHandleTable, VirtualFileSystemConfig,
    };
    use crate::virtual_fs::DngGenerationTimings;

    fn dummy_cached_frame(frame_index: usize, bytes: &[u8]) -> Arc<CachedDngFrame> {
        Arc::new(CachedDngFrame {
            frame_index,
            bytes: Arc::from(bytes.to_vec().into_boxed_slice()),
            byte_len: u64::try_from(bytes.len()).unwrap(),
            timings: DngGenerationTimings::default(),
        })
    }

    #[test]
    fn production_default_is_selected_21_4() {
        assert_eq!(
            VirtualFileSystemConfig::default().max_cached_dng_frames,
            DEFAULT_DNG_CACHE_FRAME_CAPACITY
        );
        assert_eq!(
            VirtualFileSystemConfig::default().prefetch_forward_frames,
            DEFAULT_PREFETCH_FORWARD_FRAMES
        );
        assert_eq!(
            VirtualFileSystemConfig::default().dng_generation_execution_policy,
            DngGenerationExecutionPolicy::Inflight2Default
        );
        assert_eq!(DEFAULT_DNG_CACHE_FRAME_CAPACITY, 21);
        assert_eq!(DEFAULT_PREFETCH_FORWARD_FRAMES, 4);
    }

    #[test]
    fn lower_memory_candidate_values_are_stable() {
        let config = VirtualFileSystemConfig::lower_memory();
        assert_eq!(
            config.max_cached_dng_frames,
            LOWER_MEMORY_DNG_CACHE_FRAME_CAPACITY
        );
        assert_eq!(
            config.prefetch_forward_frames,
            LOWER_MEMORY_PREFETCH_FORWARD_FRAMES
        );
        assert_eq!(config.max_cached_dng_frames, 32);
        assert_eq!(config.prefetch_forward_frames, 8);
    }

    #[test]
    fn lowest_memory_candidate_values_are_stable() {
        let config = VirtualFileSystemConfig::lowest_memory();
        assert_eq!(
            config.max_cached_dng_frames,
            LOWEST_MEMORY_DNG_CACHE_FRAME_CAPACITY
        );
        assert_eq!(
            config.prefetch_forward_frames,
            LOWEST_MEMORY_PREFETCH_FORWARD_FRAMES
        );
        assert_eq!(config.max_cached_dng_frames, 16);
        assert_eq!(config.prefetch_forward_frames, 4);
    }

    #[test]
    fn no_flag_virtual_file_system_config_uses_21_4() {
        assert_eq!(VirtualFileSystemConfig::default().max_cached_dng_frames, 21);
        assert_eq!(
            VirtualFileSystemConfig::default().prefetch_forward_frames,
            4
        );
    }

    #[test]
    fn numeric_override_128_8_still_possible_for_private_tools() {
        let config = VirtualFileSystemConfig {
            max_cached_dng_frames: 128,
            prefetch_forward_frames: 8,
            ..VirtualFileSystemConfig::default()
        };
        assert_eq!(config.max_cached_dng_frames, 128);
        assert_eq!(config.prefetch_forward_frames, 8);
    }

    #[test]
    fn numeric_override_4_0_still_possible_for_private_tools() {
        let config = VirtualFileSystemConfig {
            max_cached_dng_frames: 4,
            prefetch_forward_frames: 0,
            ..VirtualFileSystemConfig::default()
        };
        assert_eq!(config.max_cached_dng_frames, 4);
        assert_eq!(config.prefetch_forward_frames, 0);
    }

    #[test]
    fn open_handle_table_pins_immutable_complete_dng_bytes() {
        let mut table = OpenDngHandleTable::default();
        let frame = dummy_cached_frame(7, b"complete-dng");

        table.insert(100, 1000, 7);
        table.handles.get_mut(&100).unwrap().pinned = Some(frame.clone());

        let pinned = table.handles.get(&100).unwrap().pinned.as_ref().unwrap();
        assert!(Arc::ptr_eq(&frame, pinned));
        assert!(Arc::ptr_eq(&frame.bytes, &pinned.bytes));
        assert_eq!(pinned.bytes.as_ref(), b"complete-dng");
    }

    #[test]
    fn cache_reference_can_drop_while_handle_keeps_bytes() {
        let mut table = OpenDngHandleTable::default();
        let frame = dummy_cached_frame(1, b"frame-a");
        table.insert(10, 100, 1);
        table.handles.get_mut(&10).unwrap().pinned = Some(frame.clone());

        drop(frame);

        let pinned = table.handles.get(&10).unwrap().pinned.as_ref().unwrap();
        assert_eq!(pinned.bytes.as_ref(), b"frame-a");
        assert_eq!(table.stats().active_handles, 1);
        assert_eq!(table.stats().unique_pinned_frames, 1);
        assert_eq!(table.stats().unique_pinned_bytes, 7);
    }

    #[test]
    fn same_frame_handles_share_one_allocation() {
        let mut table = OpenDngHandleTable::default();
        let frame = dummy_cached_frame(2, b"same-frame");

        table.insert(20, 200, 2);
        table.handles.get_mut(&20).unwrap().pinned = Some(frame.clone());
        table.insert(21, 200, 2);
        let reused = table
            .pinned_for_frame_except(2, 21)
            .expect("active pin is reusable");
        table.handles.get_mut(&21).unwrap().pinned = Some(reused.clone());

        let first = table.handles.get(&20).unwrap().pinned.as_ref().unwrap();
        let second = table.handles.get(&21).unwrap().pinned.as_ref().unwrap();
        assert!(Arc::ptr_eq(first, second));
        assert!(Arc::ptr_eq(&first.bytes, &second.bytes));
        assert_eq!(table.stats().active_handles, 2);
        assert_eq!(table.stats().unique_pinned_frames, 1);
        assert_eq!(table.stats().unique_pinned_bytes, 10);
    }

    #[test]
    fn release_last_handle_removes_active_pin_accounting() {
        let mut table = OpenDngHandleTable::default();
        let frame = dummy_cached_frame(3, b"release");

        table.insert(30, 300, 3);
        table.handles.get_mut(&30).unwrap().pinned = Some(frame);
        assert_eq!(table.stats().unique_pinned_frames, 1);

        table.handles.remove(&30);

        assert_eq!(table.stats().active_handles, 0);
        assert_eq!(table.stats().unique_pinned_frames, 0);
        assert_eq!(table.stats().unique_pinned_bytes, 0);
    }

    #[test]
    fn one_handle_records_one_logical_demand_flag() {
        let mut table = OpenDngHandleTable::default();
        table.insert(40, 400, 4);

        let handle = table.handles.get_mut(&40).unwrap();
        assert!(!handle.demand_recorded);
        handle.demand_recorded = true;

        let handle = table.handles.get(&40).unwrap();
        assert!(handle.demand_recorded);
        assert_eq!(handle.frame_index, 4);
    }
}
