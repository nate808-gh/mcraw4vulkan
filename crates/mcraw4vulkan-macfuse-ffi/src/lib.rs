//! Deferred macFUSE FSKit macOS FUSE3 callback boundary.
#[cfg(target_os = "macos")]
mod abi;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::{
    CallbackFileSystem, DirectoryEntry, FsResult, MacFuseSession, Metadata, MountDirectory,
    UnmountOutcome, mount_instance_id, mounted, reusable_mount_directory, unmount_instance,
};
