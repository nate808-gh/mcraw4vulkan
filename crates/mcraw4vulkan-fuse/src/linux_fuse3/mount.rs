use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use mcraw4vulkan_dngwriter::DngSinkVignetteMode;

use crate::linux_fuse3::{
    DngGenerationBackend, VirtualFileSystem, VirtualFileSystemConfig,
    VirtualFileSystemStatsSnapshot,
};
use crate::platform_lifecycle::{
    PlatformMountCapabilities, PlatformMountHandle, PlatformMountKind, PlatformMountRequest,
    PlatformMountStatus, PlatformUnmountOutcome,
};
use crate::shared_root::{
    AddClipResult, SharedRootVirtualFileSystem, SingleClipRootVirtualFileSystem,
};
use crate::virtual_fs::DngGenerationExecutionPolicy;

// The request carries generation and cache policy into the shared VFS; the Linux
// adapter adds native mount configuration and background-session ownership.
#[derive(Debug, Clone)]
pub struct MountRequest {
    pub input_path: PathBuf,
    pub mount_point: PathBuf,
    pub mount_name: String,
    pub dng_backend: DngGenerationBackend,
    pub dng_vignette_mode: DngSinkVignetteMode,
    pub dng_generation_execution_policy: DngGenerationExecutionPolicy,
    pub max_cached_dng_frames: usize,
    pub prefetch_forward_frames: usize,
    pub worker_threads: usize,
}

impl MountRequest {
    pub fn new(input_path: PathBuf, mount_point: PathBuf, mount_name: String) -> Self {
        let default_config = VirtualFileSystemConfig::default();

        Self {
            input_path,
            mount_point,
            mount_name,
            dng_backend: default_config.dng_backend,
            dng_vignette_mode: default_config.dng_vignette_mode,
            dng_generation_execution_policy: default_config.dng_generation_execution_policy,
            max_cached_dng_frames: default_config.max_cached_dng_frames,
            prefetch_forward_frames: default_config.prefetch_forward_frames,
            worker_threads: 1,
        }
    }

    pub fn from_platform_request(request: PlatformMountRequest) -> Self {
        let config = request.config;

        Self {
            input_path: request.source_path,
            mount_point: request.mountpoint,
            mount_name: request.mount_name,
            dng_backend: config.dng_backend,
            dng_vignette_mode: config.dng_vignette_mode,
            dng_generation_execution_policy: config.dng_generation_execution_policy,
            max_cached_dng_frames: config.max_cached_dng_frames,
            prefetch_forward_frames: config.prefetch_forward_frames,
            worker_threads: request.worker_threads,
        }
    }

    pub fn to_platform_request(&self) -> PlatformMountRequest {
        PlatformMountRequest {
            source_path: self.input_path.clone(),
            mountpoint: self.mount_point.clone(),
            mount_name: self.mount_name.clone(),
            config: self.virtual_file_system_config(),
            worker_threads: self.worker_threads,
            owner_label: None,
        }
    }

    pub fn virtual_file_system_config(&self) -> VirtualFileSystemConfig {
        VirtualFileSystemConfig {
            dng_backend: self.dng_backend,
            dng_vignette_mode: self.dng_vignette_mode,
            dng_generation_execution_policy: self.dng_generation_execution_policy,
            max_cached_dng_frames: self.max_cached_dng_frames,
            prefetch_forward_frames: self.prefetch_forward_frames,
        }
    }
}

impl From<PlatformMountRequest> for MountRequest {
    fn from(request: PlatformMountRequest) -> Self {
        Self::from_platform_request(request)
    }
}

// Linux FUSE3 shared-root mount request.
//
// This is the foreground/session-owned multi-clip path. It deliberately does
// not describe a long-lived manager, control endpoint, or persistent registry;
// callers provide the full desired clip set at mount start.
#[derive(Debug, Clone)]
pub struct SharedRootMountRequest {
    pub input_paths: Vec<PathBuf>,
    pub mount_point: PathBuf,
    pub mount_name: String,
    pub dng_backend: DngGenerationBackend,
    pub dng_vignette_mode: DngSinkVignetteMode,
    pub dng_generation_execution_policy: DngGenerationExecutionPolicy,
    pub max_cached_dng_frames: usize,
    pub prefetch_forward_frames: usize,
    pub worker_threads: usize,
}

impl SharedRootMountRequest {
    pub fn new(input_paths: Vec<PathBuf>, mount_point: PathBuf, mount_name: String) -> Self {
        let default_config = VirtualFileSystemConfig::default();

        Self {
            input_paths,
            mount_point,
            mount_name,
            dng_backend: default_config.dng_backend,
            dng_vignette_mode: default_config.dng_vignette_mode,
            dng_generation_execution_policy: default_config.dng_generation_execution_policy,
            max_cached_dng_frames: default_config.max_cached_dng_frames,
            prefetch_forward_frames: default_config.prefetch_forward_frames,
            worker_threads: 1,
        }
    }

    pub fn with_config(mut self, config: VirtualFileSystemConfig) -> Self {
        self.dng_backend = config.dng_backend;
        self.dng_vignette_mode = config.dng_vignette_mode;
        self.dng_generation_execution_policy = config.dng_generation_execution_policy;
        self.max_cached_dng_frames = config.max_cached_dng_frames;
        self.prefetch_forward_frames = config.prefetch_forward_frames;
        self
    }

    pub fn with_worker_threads(mut self, worker_threads: usize) -> Self {
        self.worker_threads = worker_threads;
        self
    }

    pub fn virtual_file_system_config(&self) -> VirtualFileSystemConfig {
        VirtualFileSystemConfig {
            dng_backend: self.dng_backend,
            dng_vignette_mode: self.dng_vignette_mode,
            dng_generation_execution_policy: self.dng_generation_execution_policy,
            max_cached_dng_frames: self.max_cached_dng_frames,
            prefetch_forward_frames: self.prefetch_forward_frames,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedRootMountedClip {
    pub input_path: PathBuf,
    pub folder_name: String,
    pub already_mounted: bool,
}

// Each returned handle owns the background FUSE session and its VFS allocation.
// Consuming unmount stops and joins callbacks before those resources are dropped.
pub struct MountManager;

impl MountManager {
    pub fn new() -> Self {
        Self
    }

    pub fn capabilities(&self) -> PlatformMountCapabilities {
        PlatformMountCapabilities::linux_fuse3()
    }

    pub fn mount_platform(&self, request: PlatformMountRequest) -> Result<MountHandle> {
        self.mount(request.into())
    }

    #[cfg(target_os = "linux")]
    pub fn mount(&self, request: MountRequest) -> Result<MountHandle> {
        use fuser::{Config, MountOption};

        use crate::linux_fuse3::LinuxSingleClipRootFuse3FileSystem;

        let fs = Arc::new(
            VirtualFileSystem::open_clip(&request.input_path, request.virtual_file_system_config())
                .with_context(|| {
                    format!(
                        "failed to create Linux FUSE3 filesystem for {}",
                        request.input_path.display()
                    )
                })?,
        );

        let root_view = Arc::new(SingleClipRootVirtualFileSystem::new(fs.clone()));
        let filesystem = LinuxSingleClipRootFuse3FileSystem::new_single_clip_root(root_view);

        let mut config = Config::default();
        config.mount_options = vec![
            MountOption::FSName(format!("mcraw4vulkan:{}", request.mount_name)),
            MountOption::Subtype("mcraw4vulkan".to_string()),
            MountOption::RO,
            MountOption::NoDev,
            MountOption::NoSuid,
            MountOption::NoExec,
            MountOption::NoAtime,
            MountOption::DefaultPermissions,
        ];
        config.n_threads = Some(request.worker_threads.max(1));

        let session =
            fuser::spawn_mount2(filesystem, &request.mount_point, &config).with_context(|| {
                format!("failed to mount FUSE at {}", request.mount_point.display())
            })?;

        Ok(MountHandle {
            mount_name: request.mount_name,
            mount_point: request.mount_point,
            fs,
            session,
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn mount(&self, _request: MountRequest) -> Result<MountHandle> {
        anyhow::bail!("linux_fuse3::MountManager is only implemented on Linux")
    }

    #[cfg(target_os = "linux")]
    pub fn mount_shared_root(
        &self,
        request: SharedRootMountRequest,
    ) -> Result<SharedRootMountHandle> {
        use fuser::{Config, MountOption};

        use crate::linux_fuse3::LinuxSharedRootFuse3FileSystem;

        if request.input_paths.is_empty() {
            bail!("shared-root Linux FUSE3 mount requires at least one input clip");
        }

        let virtual_config = request.virtual_file_system_config();
        let mut shared_root = SharedRootVirtualFileSystem::new();
        let mut mounted_clips = Vec::with_capacity(request.input_paths.len());

        for input_path in &request.input_paths {
            let clip_fs = Arc::new(
                VirtualFileSystem::open_clip(input_path, virtual_config).with_context(|| {
                    format!(
                        "failed to create Linux FUSE3 filesystem for {}",
                        input_path.display()
                    )
                })?,
            );
            let result = shared_root
                .registry_mut()
                .add_virtual_file_system_from_path(input_path.clone(), clip_fs)
                .with_context(|| {
                    format!(
                        "failed to add {} to Linux shared-root filesystem",
                        input_path.display()
                    )
                })?;
            let already_mounted = matches!(result, AddClipResult::AlreadyMounted(_));
            mounted_clips.push(SharedRootMountedClip {
                input_path: input_path.clone(),
                folder_name: result.folder_name().to_string(),
                already_mounted,
            });
        }

        let fs = Arc::new(shared_root);
        let filesystem = LinuxSharedRootFuse3FileSystem::new_shared_root(fs.clone());

        let mut config = Config::default();
        config.mount_options = vec![
            MountOption::FSName(format!("mcraw4vulkan:{}", request.mount_name)),
            MountOption::Subtype("mcraw4vulkan".to_string()),
            MountOption::RO,
            MountOption::NoDev,
            MountOption::NoSuid,
            MountOption::NoExec,
            MountOption::NoAtime,
            MountOption::DefaultPermissions,
        ];
        config.n_threads = Some(request.worker_threads.max(1));

        let session =
            fuser::spawn_mount2(filesystem, &request.mount_point, &config).with_context(|| {
                format!("failed to mount FUSE at {}", request.mount_point.display())
            })?;

        Ok(SharedRootMountHandle {
            mount_name: request.mount_name,
            mount_point: request.mount_point,
            mounted_clips,
            fs,
            session,
        })
    }
}

impl Default for MountManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "linux")]
pub struct MountHandle {
    pub mount_name: String,
    pub mount_point: PathBuf,
    fs: Arc<VirtualFileSystem>,
    session: fuser::BackgroundSession,
}

#[cfg(target_os = "linux")]
impl MountHandle {
    // Read-only diagnostics for live validation; does not affect mounted bytes.

    pub fn stats_snapshot(&self) -> Result<VirtualFileSystemStatsSnapshot> {
        self.fs.stats_snapshot()
    }

    pub fn stats_summary_line(&self) -> Result<String> {
        self.fs.stats_summary_line()
    }

    pub fn kind(&self) -> PlatformMountKind {
        PlatformMountKind::LinuxFuse3
    }

    pub fn mountpoint(&self) -> &Path {
        &self.mount_point
    }

    pub fn capabilities(&self) -> PlatformMountCapabilities {
        PlatformMountCapabilities::linux_fuse3()
    }

    pub fn status(&self) -> PlatformMountStatus {
        PlatformMountStatus::Mounted
    }

    pub fn request_unmount(self) -> Result<PlatformUnmountOutcome> {
        self.unmount()?;
        Ok(PlatformUnmountOutcome::Unmounted)
    }

    pub fn wait_until_unmounted_status(self) -> Result<PlatformMountStatus> {
        self.wait_until_unmounted()?;
        Ok(PlatformMountStatus::Unmounted)
    }

    pub fn unmount(self) -> Result<()> {
        self.session
            .umount_and_join()
            .with_context(|| format!("failed to unmount {}", self.mount_point.display()))
    }

    pub fn wait_until_unmounted(self) -> Result<()> {
        self.session.join().with_context(|| {
            format!(
                "mount session ended with an error for {}",
                self.mount_point.display()
            )
        })
    }
}

#[cfg(target_os = "linux")]
impl PlatformMountHandle for MountHandle {
    fn kind(&self) -> PlatformMountKind {
        PlatformMountKind::LinuxFuse3
    }

    fn mountpoint(&self) -> &Path {
        &self.mount_point
    }

    fn request_unmount(self) -> Result<PlatformUnmountOutcome> {
        MountHandle::request_unmount(self)
    }

    fn wait_until_unmounted_status(self) -> Result<PlatformMountStatus> {
        MountHandle::wait_until_unmounted_status(self)
    }

    fn status(&self) -> PlatformMountStatus {
        PlatformMountStatus::Mounted
    }

    fn stats_snapshot(&self) -> Result<Option<VirtualFileSystemStatsSnapshot>> {
        MountHandle::stats_snapshot(self).map(Some)
    }

    fn capabilities(&self) -> PlatformMountCapabilities {
        PlatformMountCapabilities::linux_fuse3()
    }
}

#[cfg(target_os = "linux")]
pub struct SharedRootMountHandle {
    pub mount_name: String,
    pub mount_point: PathBuf,
    mounted_clips: Vec<SharedRootMountedClip>,
    fs: Arc<SharedRootVirtualFileSystem>,
    session: fuser::BackgroundSession,
}

#[cfg(target_os = "linux")]
impl SharedRootMountHandle {
    pub fn kind(&self) -> PlatformMountKind {
        PlatformMountKind::LinuxFuse3
    }

    pub fn mountpoint(&self) -> &Path {
        &self.mount_point
    }

    pub fn mounted_clips(&self) -> &[SharedRootMountedClip] {
        &self.mounted_clips
    }

    pub fn unique_clip_count(&self) -> usize {
        self.fs.registry().len()
    }

    pub fn capabilities(&self) -> PlatformMountCapabilities {
        PlatformMountCapabilities::linux_fuse3()
    }

    pub fn status(&self) -> PlatformMountStatus {
        PlatformMountStatus::Mounted
    }

    pub fn request_unmount(self) -> Result<PlatformUnmountOutcome> {
        self.unmount()?;
        Ok(PlatformUnmountOutcome::Unmounted)
    }

    pub fn wait_until_unmounted_status(self) -> Result<PlatformMountStatus> {
        self.wait_until_unmounted()?;
        Ok(PlatformMountStatus::Unmounted)
    }

    pub fn unmount(self) -> Result<()> {
        self.session
            .umount_and_join()
            .with_context(|| format!("failed to unmount {}", self.mount_point.display()))
    }

    pub fn wait_until_unmounted(self) -> Result<()> {
        self.session.join().with_context(|| {
            format!(
                "mount session ended with an error for {}",
                self.mount_point.display()
            )
        })
    }
}

#[cfg(not(target_os = "linux"))]
pub struct MountHandle {
    pub mount_name: String,
    pub mount_point: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountpointUnmountStatus {
    Unmounted,
    NotMounted,
}

impl From<MountpointUnmountStatus> for PlatformUnmountOutcome {
    fn from(status: MountpointUnmountStatus) -> Self {
        match status {
            MountpointUnmountStatus::Unmounted => Self::Unmounted,
            MountpointUnmountStatus::NotMounted => Self::NotMounted,
        }
    }
}

pub fn unmount_owned_mountpoint(mount_point: &Path) -> Result<MountpointUnmountStatus> {
    for (program, args) in [
        ("fusermount3", &["-u"][..]),
        ("fusermount3", &["-u", "-z"][..]),
        ("fusermount", &["-u"][..]),
        ("fusermount", &["-u", "-z"][..]),
        ("umount", &[][..]),
        ("umount", &["-l"][..]),
    ] {
        let output = Command::new(program).args(args).arg(mount_point).output();
        let Ok(output) = output else {
            continue;
        };

        if output.status.success() {
            return Ok(MountpointUnmountStatus::Unmounted);
        }

        let stderr = String::from_utf8_lossy(&output.stderr).to_ascii_lowercase();
        if stderr.contains("not mounted")
            || stderr.contains("not found")
            || stderr.contains("no such file")
            || stderr.contains("no mount point")
            || stderr.contains("not a mountpoint")
            || stderr.contains("not a mount point")
            || stderr.contains("bad mount point")
        {
            return Ok(MountpointUnmountStatus::NotMounted);
        }

        if stderr.contains("device or resource busy") || stderr.contains("target is busy") {
            continue;
        }

        bail!(
            "{program} failed to unmount {}: {}",
            mount_point.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    bail!(
        "no supported FUSE unmount command was found for {}",
        mount_point.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_request_converts_to_linux_mount_request() {
        let config = VirtualFileSystemConfig::lower_memory();
        let platform_request = PlatformMountRequest::new(
            PathBuf::from("tmp/clip.mcraw"),
            PathBuf::from("tmp/clip"),
            "clip".to_string(),
        )
        .with_config(config)
        .with_worker_threads(4)
        .with_owner_label("mcraw4vulkan");

        let request = MountRequest::from_platform_request(platform_request);

        assert_eq!(request.input_path, PathBuf::from("tmp/clip.mcraw"));
        assert_eq!(request.mount_point, PathBuf::from("tmp/clip"));
        assert_eq!(request.mount_name, "clip");
        assert_eq!(request.virtual_file_system_config(), config);
        assert_eq!(request.worker_threads, 4);
    }

    #[test]
    fn shared_root_request_preserves_multiple_inputs() {
        let config = VirtualFileSystemConfig::lower_memory();
        let request = SharedRootMountRequest::new(
            vec![
                PathBuf::from("tmp/first.mcraw"),
                PathBuf::from("tmp/second.mcraw"),
            ],
            PathBuf::from("tmp/mcraw4vulkan"),
            "mcraw4vulkan".to_string(),
        )
        .with_config(config)
        .with_worker_threads(4);

        assert_eq!(
            request.input_paths,
            vec![
                PathBuf::from("tmp/first.mcraw"),
                PathBuf::from("tmp/second.mcraw")
            ]
        );
        assert_eq!(request.mount_point, PathBuf::from("tmp/mcraw4vulkan"));
        assert_eq!(request.mount_name, "mcraw4vulkan");
        assert_eq!(request.virtual_file_system_config(), config);
        assert_eq!(request.worker_threads, 4);
    }

    #[test]
    fn linux_mount_manager_reports_shared_capabilities() {
        let manager = MountManager::new();
        let capabilities = manager.capabilities();

        assert_eq!(capabilities.kind, PlatformMountKind::LinuxFuse3);
        assert!(capabilities.supports_live_mount);
        assert!(capabilities.supports_request_unmount);
        assert!(capabilities.supports_wait_until_unmounted);
    }

    #[test]
    fn linux_unmount_status_maps_to_shared_outcome() {
        assert_eq!(
            PlatformUnmountOutcome::from(MountpointUnmountStatus::Unmounted),
            PlatformUnmountOutcome::Unmounted
        );
        assert_eq!(
            PlatformUnmountOutcome::from(MountpointUnmountStatus::NotMounted),
            PlatformUnmountOutcome::NotMounted
        );
    }
}
