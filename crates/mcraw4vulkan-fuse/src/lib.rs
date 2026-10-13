// Platform adapters translate native mount and callback lifecycles into the
// shared virtual filesystem operations. Decode, DNG and audio generation, cache
// ownership, inode identity, timestamps, and range reads remain platform-neutral.

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod filesystem_view;
pub mod platform_lifecycle;
pub(crate) mod shared_root;

#[cfg(target_os = "linux")]
pub mod linux_fuse3;
#[cfg(target_os = "macos")]
pub mod macos_macfuse;
pub mod virtual_fs;
pub mod virtual_timestamp;
#[cfg(target_os = "windows")]
pub mod windows_projfs;

pub use platform_lifecycle::{
    PlatformMountCapabilities, PlatformMountCapability, PlatformMountError, PlatformMountHandle,
    PlatformMountKind, PlatformMountRequest, PlatformMountStatus, PlatformUnmountOutcome,
};

pub use virtual_fs::{
    AUDIO_WAV_INODE, AudioWavMetadata, CachedDngFrame, DEFAULT_DNG_CACHE_FRAME_CAPACITY,
    DEFAULT_PREFETCH_FORWARD_FRAMES, DngFrameByteCache, DngFrameGenerator, DngGenerationBackend,
    DngGenerationConfig, DngGenerationExecutionPolicy, FIRST_FRAME_INODE, GeneratedDngFrame,
    InodeMap, LOWER_MEMORY_DNG_CACHE_FRAME_CAPACITY, LOWER_MEMORY_PREFETCH_FORWARD_FRAMES,
    LOWEST_MEMORY_DNG_CACHE_FRAME_CAPACITY, LOWEST_MEMORY_PREFETCH_FORWARD_FRAMES, LazyAudioWav,
    LazyAudioWavConfig, LazyAudioWavSummary, ROOT_INODE, VirtualDirEntry, VirtualFileBytes,
    VirtualFileKind, VirtualFileMetadata, VirtualFileSystem, VirtualFileSystemConfig,
    VirtualMetadataProvider, VirtualNode, VirtualReadData, VirtualTimestamp,
    parse_motioncam_capture_timestamp_from_stem, virtual_timestamp_for_input_path,
};

#[cfg(target_os = "linux")]
pub use linux_fuse3::{
    MountHandle, MountManager, MountRequest, MountpointUnmountStatus, unmount_owned_mountpoint,
};

#[cfg(target_os = "macos")]
pub use macos_macfuse::{
    MacosMountInfo, MacosMountpointClassification, MountHandle, MountManager, MountRequest,
    MountpointUnmountStatus, classify_macos_mountpoint, unmount_owned_mountpoint,
};

#[cfg(target_os = "windows")]
pub use windows_projfs::{
    MountHandle, MountManager, MountRequest, MountpointUnmountStatus,
    cleanup_stale_shared_root_mountpoint, cleanup_stale_single_clip_mountpoint,
    provider_process_is_live, unmount_owned_mountpoint,
};

#[cfg(target_os = "macos")]
pub use macos_macfuse::macos_mount_instance_id;

#[cfg(target_os = "macos")]
pub use macos_macfuse::unmount_registered_instance;
