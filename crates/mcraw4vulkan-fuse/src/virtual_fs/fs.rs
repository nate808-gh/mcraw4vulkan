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
// owns virtual metadata, DNG byte cache access, and prefetch
// triggering. OS-specific layers should remain thin translation layers around
// these methods.
pub struct VirtualFileSystem {
    provider: VirtualMetadataProvider,
    cache: Arc<DngFrameByteCache>,
    open_dng_handles: Mutex<OpenDngHandleTable>,
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
            open_dng_handles: Mutex::new(OpenDngHandleTable::default()),
        })
    }

    pub fn provider(&self) -> &VirtualMetadataProvider {
        &self.provider
    }

    pub fn inode_map(&self) -> &InodeMap {
        self.provider.inode_map()
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
        self.provider.metadata_for_child_name(parent_inode, name)
    }

    // Platform adapter getattr callback shape.
    pub fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
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

            return Ok(Some(VirtualReadData {
                bytes: Arc::from(bytes.into_boxed_slice()),
                start: 0,
                end: returned_len,
            }));
        }

        let Some(node) = self.provider.inode_map().node_for_inode(inode) else {
            return Ok(None);
        };

        let file_bytes = match node {
            VirtualNode::Root | VirtualNode::ClipDirectory => {
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

        Ok(Some(VirtualReadData {
            bytes: file_bytes,
            start,
            end,
        }))
    }
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
