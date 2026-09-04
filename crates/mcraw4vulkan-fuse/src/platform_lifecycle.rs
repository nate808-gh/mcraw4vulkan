use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::virtual_fs::{VirtualFileSystemConfig, VirtualFileSystemStatsSnapshot};

const LINUX_FUSE3_NOTES: &[&str] = &[
    "Linux FUSE3/fuser adapter is implemented for Linux builds.",
    "Mount roots must be prepared as empty directories before mounting.",
];

const MACOS_MACFUSE_NOTES: &[&str] = &[
    "macFUSE adapter is implemented for macOS builds.",
    "Shared virtual filesystem state remains the source of truth for mounted DNG and sidecar bytes.",
];

const WINDOWS_PROJFS_NOTES: &[&str] = &[
    "Windows ProjFS adapter is implemented for Windows builds.",
    "Shared virtual filesystem state remains the source of truth; materialized placeholders must not own DNG bytes.",
];

const WINDOWS_PROJFS_PLACEHOLDER_NOTES: &[&str] = &[
    "ProjFS adapter is a placeholder until native virtualization is implemented.",
    "VirtualFileSystem remains the source of truth; materialized placeholders must not own DNG bytes.",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformMountKind {
    LinuxFuse3,
    MacosMacFuse,
    WindowsProjFs,
}

impl PlatformMountKind {
    pub fn adapter_module(self) -> &'static str {
        match self {
            Self::LinuxFuse3 => "linux_fuse3",
            Self::MacosMacFuse => "macos_macfuse",
            Self::WindowsProjFs => "windows_projfs",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::LinuxFuse3 => "Linux FUSE3",
            Self::MacosMacFuse => "macOS macFUSE",
            Self::WindowsProjFs => "Windows ProjFS",
        }
    }
}

impl fmt::Display for PlatformMountKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.display_name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformMountCapability {
    SupportsLiveMount,
    SupportsRequestUnmount,
    SupportsWaitUntilUnmounted,
    RequiresEmptyMountRoot,
    RequiresWritableMountRoot,
    RequiresNativeDriver,
    MayMaterializePlaceholders,
    MayNeedMountRootCleanup,
    SupportsVirtualReadRanges,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlatformMountCapabilities {
    // These flags expose native lifecycle differences: FUSE mounts require an
    // empty root, while ProjFS may materialize placeholders that need cleanup.
    pub kind: PlatformMountKind,
    pub supports_live_mount: bool,
    pub supports_request_unmount: bool,
    pub supports_wait_until_unmounted: bool,
    pub requires_empty_mount_root: bool,
    pub requires_writable_mount_root: bool,
    pub requires_native_driver: bool,
    pub may_materialize_placeholders: bool,
    pub may_need_mount_root_cleanup: bool,
    pub virtual_file_source_of_truth: bool,
    pub supports_virtual_read_ranges: bool,
    pub notes: &'static [&'static str],
}

impl PlatformMountCapabilities {
    pub fn linux_fuse3() -> Self {
        Self {
            kind: PlatformMountKind::LinuxFuse3,
            supports_live_mount: true,
            supports_request_unmount: true,
            supports_wait_until_unmounted: true,
            requires_empty_mount_root: true,
            requires_writable_mount_root: true,
            requires_native_driver: true,
            may_materialize_placeholders: false,
            may_need_mount_root_cleanup: false,
            virtual_file_source_of_truth: true,
            supports_virtual_read_ranges: true,
            notes: LINUX_FUSE3_NOTES,
        }
    }

    pub fn macos_macfuse() -> Self {
        Self {
            kind: PlatformMountKind::MacosMacFuse,
            supports_live_mount: true,
            supports_request_unmount: true,
            supports_wait_until_unmounted: true,
            requires_empty_mount_root: true,
            requires_writable_mount_root: true,
            requires_native_driver: true,
            may_materialize_placeholders: false,
            may_need_mount_root_cleanup: false,
            virtual_file_source_of_truth: true,
            supports_virtual_read_ranges: true,
            notes: MACOS_MACFUSE_NOTES,
        }
    }

    pub fn windows_projfs_placeholder() -> Self {
        Self {
            kind: PlatformMountKind::WindowsProjFs,
            supports_live_mount: false,
            supports_request_unmount: false,
            supports_wait_until_unmounted: false,
            requires_empty_mount_root: false,
            requires_writable_mount_root: true,
            requires_native_driver: true,
            may_materialize_placeholders: true,
            may_need_mount_root_cleanup: true,
            virtual_file_source_of_truth: true,
            supports_virtual_read_ranges: true,
            notes: WINDOWS_PROJFS_PLACEHOLDER_NOTES,
        }
    }

    pub fn windows_projfs() -> Self {
        Self {
            kind: PlatformMountKind::WindowsProjFs,
            supports_live_mount: true,
            supports_request_unmount: true,
            supports_wait_until_unmounted: true,
            requires_empty_mount_root: false,
            requires_writable_mount_root: true,
            requires_native_driver: true,
            may_materialize_placeholders: true,
            may_need_mount_root_cleanup: true,
            virtual_file_source_of_truth: true,
            supports_virtual_read_ranges: true,
            notes: WINDOWS_PROJFS_NOTES,
        }
    }

    pub fn supports(&self, capability: PlatformMountCapability) -> bool {
        match capability {
            PlatformMountCapability::SupportsLiveMount => self.supports_live_mount,
            PlatformMountCapability::SupportsRequestUnmount => self.supports_request_unmount,
            PlatformMountCapability::SupportsWaitUntilUnmounted => {
                self.supports_wait_until_unmounted
            }
            PlatformMountCapability::RequiresEmptyMountRoot => self.requires_empty_mount_root,
            PlatformMountCapability::RequiresWritableMountRoot => self.requires_writable_mount_root,
            PlatformMountCapability::RequiresNativeDriver => self.requires_native_driver,
            PlatformMountCapability::MayMaterializePlaceholders => {
                self.may_materialize_placeholders
            }
            PlatformMountCapability::MayNeedMountRootCleanup => self.may_need_mount_root_cleanup,
            PlatformMountCapability::SupportsVirtualReadRanges => self.supports_virtual_read_ranges,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformMountRequest {
    pub source_path: PathBuf,
    pub mountpoint: PathBuf,
    pub mount_name: String,
    pub config: VirtualFileSystemConfig,
    pub worker_threads: usize,
    pub owner_label: Option<String>,
}

impl PlatformMountRequest {
    pub fn new(source_path: PathBuf, mountpoint: PathBuf, mount_name: String) -> Self {
        Self {
            source_path,
            mountpoint,
            mount_name,
            config: VirtualFileSystemConfig::default(),
            worker_threads: 1,
            owner_label: None,
        }
    }

    pub fn with_config(mut self, config: VirtualFileSystemConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_worker_threads(mut self, worker_threads: usize) -> Self {
        self.worker_threads = worker_threads;
        self
    }

    pub fn with_owner_label(mut self, owner_label: impl Into<String>) -> Self {
        self.owner_label = Some(owner_label.into());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformMountStatus {
    Mounted,
    UnmountRequested,
    Unmounted,
    Unsupported,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformUnmountOutcome {
    Unmounted,
    NotMounted,
    Unsupported,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlatformMountError {
    Unsupported {
        platform: PlatformMountKind,
        message: String,
    },
    InvalidMountRoot {
        path: PathBuf,
        message: String,
    },
    DriverUnavailable {
        message: String,
    },
    Io {
        message: String,
    },
    Internal {
        message: String,
    },
}

impl PlatformMountError {
    pub fn unsupported(platform: PlatformMountKind, message: impl Into<String>) -> Self {
        Self::Unsupported {
            platform,
            message: message.into(),
        }
    }
}

impl fmt::Display for PlatformMountError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { platform, message } => {
                write!(formatter, "{platform} adapter is unsupported: {message}")
            }
            Self::InvalidMountRoot { path, message } => {
                write!(
                    formatter,
                    "invalid mount root {}: {message}",
                    path.display()
                )
            }
            Self::DriverUnavailable { message } => {
                write!(formatter, "platform mount driver is unavailable: {message}")
            }
            Self::Io { message } => write!(formatter, "platform mount IO error: {message}"),
            Self::Internal { message } => {
                write!(formatter, "platform mount internal error: {message}")
            }
        }
    }
}

impl std::error::Error for PlatformMountError {}

pub trait PlatformMountHandle {
    fn kind(&self) -> PlatformMountKind;
    fn mountpoint(&self) -> &Path;
    fn request_unmount(self) -> Result<PlatformUnmountOutcome>
    where
        Self: Sized;
    fn wait_until_unmounted_status(self) -> Result<PlatformMountStatus>
    where
        Self: Sized;
    fn status(&self) -> PlatformMountStatus;
    fn stats_snapshot(&self) -> Result<Option<VirtualFileSystemStatsSnapshot>>;
    fn capabilities(&self) -> PlatformMountCapabilities;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capability_text(capabilities: PlatformMountCapabilities) -> String {
        format!(
            "{:?} {:?} {}",
            capabilities.kind,
            capabilities.notes,
            capabilities.kind.display_name()
        )
    }

    #[test]
    fn linux_capabilities_report_live_mount_support() {
        let capabilities = PlatformMountCapabilities::linux_fuse3();

        assert_eq!(capabilities.kind, PlatformMountKind::LinuxFuse3);
        assert!(capabilities.supports_live_mount);
        assert!(capabilities.supports_request_unmount);
        assert!(capabilities.supports_wait_until_unmounted);
        assert!(capabilities.requires_empty_mount_root);
        assert!(capabilities.virtual_file_source_of_truth);
        assert!(capabilities.supports(PlatformMountCapability::SupportsVirtualReadRanges));
    }

    #[test]
    fn macos_capabilities_report_live_macfuse_mount_support() {
        let capabilities = PlatformMountCapabilities::macos_macfuse();

        assert_eq!(capabilities.kind, PlatformMountKind::MacosMacFuse);
        assert!(capabilities.supports_live_mount);
        assert!(capabilities.supports_request_unmount);
        assert!(capabilities.supports_wait_until_unmounted);
        assert!(capabilities.requires_empty_mount_root);
        assert!(capabilities.requires_writable_mount_root);
        assert!(!capabilities.may_materialize_placeholders);
        assert!(capabilities.virtual_file_source_of_truth);
    }

    #[test]
    fn windows_capabilities_report_live_projfs_cleanup_risk() {
        let capabilities = PlatformMountCapabilities::windows_projfs();

        assert_eq!(capabilities.kind, PlatformMountKind::WindowsProjFs);
        assert!(capabilities.supports_live_mount);
        assert!(capabilities.supports_request_unmount);
        assert!(capabilities.supports_wait_until_unmounted);
        assert!(capabilities.requires_writable_mount_root);
        assert!(capabilities.may_materialize_placeholders);
        assert!(capabilities.may_need_mount_root_cleanup);
        assert!(capabilities.virtual_file_source_of_truth);
    }

    #[test]
    fn capability_vocabulary_has_no_legacy_windows_fuse_wording() {
        let text = [
            capability_text(PlatformMountCapabilities::linux_fuse3()),
            capability_text(PlatformMountCapabilities::macos_macfuse()),
            capability_text(PlatformMountCapabilities::windows_projfs()),
        ]
        .join("\n");
        let legacy_name = ['W', 'i', 'n', 'F', 's', 'p'].iter().collect::<String>();
        let legacy_name_lowercase = legacy_name.to_ascii_lowercase();

        assert!(!text.contains(&legacy_name));
        assert!(!text.contains(&legacy_name_lowercase));
    }

    #[test]
    fn unsupported_errors_name_placeholder_adapters() {
        let macos_error = PlatformMountError::unsupported(
            PlatformMountKind::MacosMacFuse,
            "macFUSE DNG mount adapter is not implemented yet",
        )
        .to_string();
        let windows_error = PlatformMountError::unsupported(
            PlatformMountKind::WindowsProjFs,
            "ProjFS DNG mount adapter is not implemented yet",
        )
        .to_string();

        assert!(macos_error.contains("macFUSE"));
        assert!(windows_error.contains("ProjFS"));
        let legacy_name = ['W', 'i', 'n', 'F', 's', 'p'].iter().collect::<String>();
        assert!(!windows_error.contains(&legacy_name));
    }

    #[test]
    fn shared_mount_request_keeps_source_mountpoint_config_and_workers() {
        let source_path = PathBuf::from("tmp/source.mcraw");
        let mountpoint = PathBuf::from("tmp/source");
        let config = VirtualFileSystemConfig::lowest_memory();

        let request = PlatformMountRequest::new(
            source_path.clone(),
            mountpoint.clone(),
            "source".to_string(),
        )
        .with_config(config)
        .with_worker_threads(4)
        .with_owner_label("mcraw4vulkan");

        assert_eq!(request.source_path, source_path);
        assert_eq!(request.mountpoint, mountpoint);
        assert_eq!(request.mount_name, "source");
        assert_eq!(request.config, config);
        assert_eq!(request.worker_threads, 4);
        assert_eq!(request.owner_label.as_deref(), Some("mcraw4vulkan"));
    }
}
