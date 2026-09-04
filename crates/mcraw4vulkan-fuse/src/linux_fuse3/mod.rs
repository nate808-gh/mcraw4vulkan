// This module translates Linux fuser requests into the platform-neutral virtual
// filesystem. Decode, byte generation, caching, and virtual identity remain
// owned by the shared layer.

#[cfg(target_os = "linux")]
mod fuser_fs;
mod mount;

pub use crate::virtual_fs::{
    AUDIO_WAV_INODE, AudioWavMetadata, CLIP_DIR_INODE, CachedDngFrame,
    DEFAULT_DNG_CACHE_FRAME_CAPACITY, DEFAULT_PREFETCH_FORWARD_FRAMES, DngFrameByteCache,
    DngFrameCacheStats, DngFrameGenerator, DngGenerationBackend, DngGenerationConfig,
    DngGenerationTimings, FIRST_FRAME_INODE, GeneratedDngFrame, InodeMap,
    LOWER_MEMORY_DNG_CACHE_FRAME_CAPACITY, LOWER_MEMORY_PREFETCH_FORWARD_FRAMES,
    LOWEST_MEMORY_DNG_CACHE_FRAME_CAPACITY, LOWEST_MEMORY_PREFETCH_FORWARD_FRAMES, LazyAudioWav,
    LazyAudioWavConfig, ROOT_INODE, ReadStatsSnapshot, VirtualDirEntry, VirtualFileBytes,
    VirtualFileKind, VirtualFileMetadata, VirtualFileSystem, VirtualFileSystemConfig,
    VirtualFileSystemStatsSnapshot, VirtualMetadataProvider, VirtualNode, VirtualReadData,
    VirtualRuntimeStats, VirtualRuntimeStatsSnapshot,
};

pub use crate::virtual_timestamp::{
    VirtualTimestamp, parse_motioncam_capture_timestamp_from_stem, virtual_timestamp_for_input_path,
};

#[cfg(target_os = "linux")]
pub use fuser_fs::LinuxFuse3FileSystem;
#[cfg(target_os = "linux")]
#[allow(unused_imports)]
pub(crate) use fuser_fs::LinuxSharedRootFuse3FileSystem;
#[cfg(target_os = "linux")]
#[allow(unused_imports)]
pub(crate) use fuser_fs::LinuxSingleClipRootFuse3FileSystem;
pub use mount::{
    MountHandle, MountManager, MountRequest, MountpointUnmountStatus, SharedRootMountHandle,
    SharedRootMountRequest, SharedRootMountedClip, unmount_owned_mountpoint,
};
