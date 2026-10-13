use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use mcraw4vulkan_dngwriter::DngSinkVignetteMode;
use mcraw4vulkan_macfuse_ffi::MountDirectory;
pub use mcraw4vulkan_macfuse_ffi::reusable_mount_directory;
use mcraw4vulkan_macfuse_ffi::{MacFuseSession, UnmountOutcome};

mod callback_fs;
use mcraw4vulkan_core::macfuse_location;

use crate::platform_lifecycle::{
    PlatformMountCapabilities, PlatformMountHandle, PlatformMountKind, PlatformMountRequest,
    PlatformMountStatus, PlatformUnmountOutcome,
};
use crate::shared_root::SingleClipRootVirtualFileSystem;
use crate::virtual_fs::{
    DngGenerationBackend, DngGenerationExecutionPolicy, VirtualFileSystem, VirtualFileSystemConfig,
};

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
    pub cancellation: Arc<std::sync::atomic::AtomicBool>,
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
            cancellation: Arc::new(std::sync::atomic::AtomicBool::new(false)),
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
            cancellation: Arc::new(std::sync::atomic::AtomicBool::new(false)),
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

// Each returned handle owns the provider-owned FUSE3 callback session and its VFS
// allocation. A refused borrowed unmount keeps both available for retry;
// successful teardown joins callbacks before releasing native state.
pub struct MountManager;

impl MountManager {
    pub fn new() -> Self {
        Self
    }

    pub fn capabilities(&self) -> PlatformMountCapabilities {
        PlatformMountCapabilities::macos_macfuse()
    }

    pub fn mount_platform(&self, request: PlatformMountRequest) -> Result<MountHandle> {
        self.mount(request.into())
    }

    pub fn mount(&self, request: MountRequest) -> Result<MountHandle> {
        let directory = MountDirectory::prepare(&request.mount_point, &request.input_path)?;
        let fs = Arc::new(
            VirtualFileSystem::open_clip(&request.input_path, request.virtual_file_system_config())
                .with_context(|| {
                    format!(
                        "failed to prepare DNG filesystem for {}",
                        request.input_path.display()
                    )
                })?,
        );

        let root_view = Arc::new(SingleClipRootVirtualFileSystem::new(fs.clone()));
        let filesystem = Arc::new(callback_fs::CallbackView::new(root_view));
        let (session, actual_path) = spawn_macos_session(
            filesystem,
            directory,
            &request.mount_name,
            request.cancellation.clone(),
        )
        .with_context(|| {
            format!(
                "failed to mount DNG volume at {}",
                request.mount_point.display()
            )
        })?;

        Ok(MountHandle {
            mount_name: request.mount_name,
            mount_point: actual_path,
            session,
        })
    }
}

fn spawn_macos_session(
    filesystem: Arc<dyn mcraw4vulkan_macfuse_ffi::CallbackFileSystem>,
    directory: MountDirectory,
    name: &str,
    cancellation: Arc<std::sync::atomic::AtomicBool>,
) -> Result<(MacFuseSession, PathBuf)> {
    let runtime = macfuse_location::discover()?;
    let session = MacFuseSession::mount(filesystem, directory, runtime, name, cancellation)?;
    let actual = session.mountpoint().to_path_buf();
    Ok((session, actual))
}

fn platform_status_from_mount_observation<E>(observation: Result<bool, E>) -> PlatformMountStatus {
    match observation {
        Ok(true) => PlatformMountStatus::Mounted,
        Ok(false) => PlatformMountStatus::Unmounted,
        Err(_) => PlatformMountStatus::Failed,
    }
}

fn platform_unmount_outcome(outcome: UnmountOutcome) -> PlatformUnmountOutcome {
    match outcome {
        UnmountOutcome::Unmounted => PlatformUnmountOutcome::Unmounted,
        UnmountOutcome::AlreadyUnmounted => PlatformUnmountOutcome::NotMounted,
    }
}

impl Default for MountManager {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MountHandle {
    pub mount_name: String,
    pub mount_point: PathBuf,
    session: MacFuseSession,
}

impl MountHandle {
    pub fn kind(&self) -> PlatformMountKind {
        PlatformMountKind::MacosMacFuse
    }

    pub fn instance_id(&self) -> Result<String> {
        self.session.instance_id().map_err(Into::into)
    }

    pub fn mountpoint(&self) -> &Path {
        &self.mount_point
    }

    pub fn capabilities(&self) -> PlatformMountCapabilities {
        PlatformMountCapabilities::macos_macfuse()
    }

    pub fn status(&self) -> PlatformMountStatus {
        platform_status_from_mount_observation(self.session.is_mounted())
    }

    pub fn request_unmount(&mut self) -> Result<PlatformUnmountOutcome> {
        self.session
            .try_unmount()
            .map(platform_unmount_outcome)
            .with_context(|| format!("failed to unmount {}", self.mount_point.display()))
    }

    pub fn wait_until_unmounted_status(self) -> Result<PlatformMountStatus> {
        self.wait_until_unmounted()?;
        Ok(PlatformMountStatus::Unmounted)
    }

    pub fn unmount(&mut self) -> Result<()> {
        self.request_unmount().map(|_| ())
    }

    pub fn wait_until_unmounted(self) -> Result<()> {
        self.session.wait_until_unmounted().with_context(|| {
            format!(
                "mount session ended with an error for {}",
                self.mount_point.display()
            )
        })
    }
}

impl PlatformMountHandle for MountHandle {
    fn kind(&self) -> PlatformMountKind {
        PlatformMountKind::MacosMacFuse
    }

    fn mountpoint(&self) -> &Path {
        &self.mount_point
    }

    fn request_unmount(&mut self) -> Result<PlatformUnmountOutcome> {
        MountHandle::request_unmount(self)
    }

    fn wait_until_unmounted_status(self) -> Result<PlatformMountStatus> {
        MountHandle::wait_until_unmounted_status(self)
    }

    fn status(&self) -> PlatformMountStatus {
        MountHandle::status(self)
    }

    fn capabilities(&self) -> PlatformMountCapabilities {
        PlatformMountCapabilities::macos_macfuse()
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacosMountInfo {
    pub source: String,
    pub mount_point: PathBuf,
    pub filesystem_type: String,
    pub raw_line: String,
}

impl MacosMountInfo {
    pub fn is_fuse_like(&self) -> bool {
        let filesystem_type = self.filesystem_type.to_ascii_lowercase();
        filesystem_type.contains("fuse")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacosMountpointClassification {
    NotMounted,
    Mounted(MacosMountInfo),
}

pub fn classify_macos_mountpoint(path: &Path) -> Result<MacosMountpointClassification> {
    let output = macfuse_location::bounded_command("/sbin/mount", &[])
        .context("failed to inspect macOS mount table with mount")?;
    if !output.status.success() {
        bail!(
            "failed to inspect macOS mount table with mount: {}{}",
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let output = String::from_utf8_lossy(&output.stdout);
    Ok(classify_macos_mountpoint_from_output(&output, path))
}

pub fn unmount_owned_mountpoint(mount_point: &Path) -> Result<MountpointUnmountStatus> {
    if !mcraw4vulkan_macfuse_ffi::mounted(mount_point)? {
        return Ok(MountpointUnmountStatus::NotMounted);
    }
    bail!(
        "path occupancy does not establish DNG ownership; use the registered mount owner or Finder"
    )
}
pub fn unmount_registered_instance(
    mount_point: &Path,
    identity: &str,
) -> Result<PlatformUnmountOutcome> {
    mcraw4vulkan_macfuse_ffi::unmount_instance(mount_point, identity)
        .map(platform_unmount_outcome)
        .map_err(Into::into)
}

fn classify_macos_mountpoint_from_output(
    output: &str,
    path: &Path,
) -> MacosMountpointClassification {
    for line in output.lines() {
        let Some(info) = parse_macos_mount_line(line) else {
            continue;
        };
        if macos_mount_paths_match(&info.mount_point, path) {
            return MacosMountpointClassification::Mounted(info);
        }
    }

    MacosMountpointClassification::NotMounted
}

fn parse_macos_mount_line(line: &str) -> Option<MacosMountInfo> {
    let (left, options) = line.rsplit_once(" (")?;
    let (source, mount_point) = left.split_once(" on ")?;
    let filesystem_type = options
        .split([',', ')'])
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())?;

    Some(MacosMountInfo {
        source: source.to_string(),
        mount_point: PathBuf::from(mount_point),
        filesystem_type: filesystem_type.to_string(),
        raw_line: line.to_string(),
    })
}

fn macos_mount_paths_match(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }

    normalized_mount_path_string(left) == normalized_mount_path_string(right)
}

fn normalized_mount_path_string(path: &Path) -> String {
    let value = path.to_string_lossy();
    let trimmed = value.trim_end_matches('/');
    if trimmed.is_empty() {
        "/".to_string()
    } else {
        trimmed.to_string()
    }
}

pub fn macos_mount_instance_id(path: &Path) -> Result<Option<String>> {
    Ok(mcraw4vulkan_macfuse_ffi::mount_instance_id(path)?)
}
