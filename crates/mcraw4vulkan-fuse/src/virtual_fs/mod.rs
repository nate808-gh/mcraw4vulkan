// Platform-neutral virtual filesystem model for mcraw4vulkan.
//
// This module contains the shared filesystem behavior that should be reused by
// Linux FUSE3, macOS macFUSE, and Windows ProjFS adapters:
//
// - virtual inode/name identity
// - virtual file metadata
// - BW64 WAV byte generation
// - DNG byte generation
// - DNG byte caching
// - read-range objects for platform adapters
// - runtime stats and prefetch counters
//
// Platform-specific modules should remain thin callback adapters around this
// module. Linux-specific FUSE types, macFUSE types, ProjFS types, Unix-only file
// APIs, and Windows-only file APIs do not belong here.

mod audio_wav;
mod dng_cache;
mod dng_generator;
mod fs;
mod inode_map;
mod lazy_audio_wav;
mod stats;
mod virtual_metadata;

pub use audio_wav::AudioWavMetadata;
pub use dng_cache::{
    CachedDngFrame, DEFAULT_DNG_HOT_FRAME_CAPACITY, DEFAULT_DNG_HOT_PREFETCH_TARGET,
    DngFrameByteCache, DngFrameCacheStats,
};
pub use dng_generator::{
    DngFrameByteLenCalculator, DngFrameGenerator, DngGenerationBackend, DngGenerationConfig,
    DngGenerationExecutionPolicy, DngGenerationTimings, GeneratedDngFrame,
};
pub use fs::{
    DEFAULT_DNG_CACHE_FRAME_CAPACITY, DEFAULT_PREFETCH_FORWARD_FRAMES,
    LOWER_MEMORY_DNG_CACHE_FRAME_CAPACITY, LOWER_MEMORY_PREFETCH_FORWARD_FRAMES,
    LOWEST_MEMORY_DNG_CACHE_FRAME_CAPACITY, LOWEST_MEMORY_PREFETCH_FORWARD_FRAMES,
    VirtualDngHandleStats, VirtualFileSystem, VirtualFileSystemConfig,
    VirtualFileSystemStatsSnapshot, VirtualReadData,
};
pub use inode_map::{
    AUDIO_WAV_INODE, CLIP_DIR_INODE, FIRST_FRAME_INODE, InodeMap, ROOT_INODE, VirtualDirEntry,
    VirtualNode,
};
pub use lazy_audio_wav::{LazyAudioWav, LazyAudioWavConfig, LazyAudioWavSummary};
pub use stats::{ReadStatsSnapshot, VirtualRuntimeStats, VirtualRuntimeStatsSnapshot};
pub use virtual_metadata::{
    VirtualFileBytes, VirtualFileKind, VirtualFileMetadata, VirtualMetadataProvider,
};

pub use crate::virtual_timestamp::{
    VirtualTimestamp, parse_motioncam_capture_timestamp_from_stem, virtual_timestamp_for_input_path,
};
