// Windows Microsoft ProjFS adapter.
//
// The official ProjFS FFI boundary lives in the mcraw4vulkan-projfs-ffi
// crate. This adapter keeps the shared virtual filesystem as the source of
// truth and only translates ProjFS callbacks into virtual filesystem calls.

use std::collections::HashMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use mcraw4vulkan_dngwriter::DngSinkVignetteMode;
use mcraw4vulkan_projfs_ffi::{
    DirectoryEntry, PlaceholderInfo, ProjectionInstanceId, ProjectionProvider, ProviderError,
    ProviderResult,
};

use crate::platform_lifecycle::{
    PlatformMountCapabilities, PlatformMountError, PlatformMountHandle, PlatformMountKind,
    PlatformMountRequest, PlatformMountStatus, PlatformUnmountOutcome,
};
use crate::shared_root::{
    AddClipResult, SHARED_ROOT_INODE, SharedRootVirtualFileSystem, SingleClipRootVirtualFileSystem,
};
use crate::virtual_fs::{
    DngGenerationBackend, DngGenerationExecutionPolicy, VirtualDirEntry, VirtualFileKind,
    VirtualFileMetadata, VirtualFileSystem, VirtualFileSystemConfig,
    VirtualFileSystemStatsSnapshot, VirtualTimestamp,
};

const STOP_REQUEST_DIR_NAME: &str = "dng-stop-requests";
const STOP_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const STOP_REQUEST_POLL_INTERVAL: Duration = Duration::from_millis(250);
const WINDOWS_ERROR_ACCESS_DENIED: i32 = 5;
const WINDOWS_ERROR_SHARING_VIOLATION: i32 = 32;
const WINDOWS_ERROR_DIR_NOT_EMPTY: i32 = 145;
const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

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

#[derive(Debug, Default, Clone, Copy)]
pub struct MountManager;

impl MountManager {
    pub fn new() -> Self {
        Self
    }

    pub fn capabilities(&self) -> PlatformMountCapabilities {
        PlatformMountCapabilities::windows_projfs()
    }

    pub fn mount_platform(&self, request: PlatformMountRequest) -> Result<MountHandle> {
        self.mount(request.into())
    }

    pub fn mount(&self, request: MountRequest) -> Result<MountHandle> {
        let clip_fs = Arc::new(
            VirtualFileSystem::open_clip(&request.input_path, request.virtual_file_system_config())
                .with_context(|| {
                    format!(
                        "failed to create Windows ProjFS filesystem for {}",
                        request.input_path.display()
                    )
                })?,
        );
        let root_view = Arc::new(SingleClipRootVirtualFileSystem::new(clip_fs.clone()));
        let session = WindowsProjfsMountSession::start(
            request.mount_point.clone(),
            root_view,
            Some(clip_fs),
            1,
        )?;

        Ok(MountHandle {
            mount_name: request.mount_name,
            mount_point: request.mount_point,
            input_path: request.input_path,
            session: Some(session),
        })
    }

    pub fn mount_shared_root(
        &self,
        request: SharedRootMountRequest,
    ) -> Result<SharedRootMountHandle> {
        let build = build_shared_root_virtual_file_system(
            &request.input_paths,
            request.virtual_file_system_config(),
        )?;
        let unique_clip_count = build.fs.registry().len();
        let session = WindowsProjfsMountSession::start(
            request.mount_point.clone(),
            build.fs,
            None,
            unique_clip_count,
        )?;

        Ok(SharedRootMountHandle {
            mount_name: request.mount_name,
            mount_point: request.mount_point,
            mounted_clips: build.mounted_clips,
            session: Some(session),
        })
    }
}

struct SharedRootBuild {
    fs: Arc<SharedRootVirtualFileSystem>,
    mounted_clips: Vec<SharedRootMountedClip>,
}

fn build_shared_root_virtual_file_system(
    input_paths: &[PathBuf],
    virtual_config: VirtualFileSystemConfig,
) -> Result<SharedRootBuild> {
    if input_paths.is_empty() {
        bail!("shared-root Windows ProjFS mount requires at least one input clip");
    }

    let mut shared_root = SharedRootVirtualFileSystem::new();
    let mut mounted_clips = Vec::with_capacity(input_paths.len());

    for input_path in input_paths {
        let clip_fs = Arc::new(
            VirtualFileSystem::open_clip(input_path, virtual_config).with_context(|| {
                format!(
                    "failed to create Windows ProjFS filesystem for {}",
                    input_path.display()
                )
            })?,
        );
        let result = shared_root
            .registry_mut()
            .add_virtual_file_system_from_path(input_path.clone(), clip_fs)
            .with_context(|| {
                format!(
                    "failed to add {} to Windows shared-root filesystem",
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

    Ok(SharedRootBuild {
        fs: Arc::new(shared_root),
        mounted_clips,
    })
}

#[derive(Debug, Clone, Copy)]
struct StopRequestWaitConfig {
    timeout: Duration,
    poll_interval: Duration,
}

impl Default for StopRequestWaitConfig {
    fn default() -> Self {
        Self {
            timeout: STOP_REQUEST_TIMEOUT,
            poll_interval: STOP_REQUEST_POLL_INTERVAL,
        }
    }
}

#[derive(Debug, Clone)]
struct StopRequestPaths {
    // A non-owner process cannot stop the native projection directly. It asks
    // the foreground owner to stop, then waits for completion before treating
    // the projection root as removable.
    request_path: PathBuf,
    completion_path: PathBuf,
}

impl StopRequestPaths {
    fn for_mountpoint(mount_point: &Path) -> Result<Self> {
        Self::in_dir(stop_request_dir()?, mount_point)
    }

    fn in_dir(dir: PathBuf, mount_point: &Path) -> Result<Self> {
        let hash = mountpoint_hash(mount_point);
        Ok(Self {
            request_path: dir.join(format!("mount-{hash:016x}.request")),
            completion_path: dir.join(format!("mount-{hash:016x}.stopped")),
        })
    }

    fn request_stop(&self) -> Result<()> {
        self.ensure_parent_dir()?;
        let _ = remove_file_if_exists(&self.completion_path);
        fs::write(&self.request_path, b"stop\n").with_context(|| {
            format!(
                "failed to write Windows ProjFS stop request {}",
                self.request_path.display()
            )
        })
    }

    fn request_exists(&self) -> bool {
        self.request_path.exists()
    }

    fn completion_exists(&self) -> bool {
        self.completion_path.exists()
    }

    fn mark_completed(&self) -> Result<()> {
        self.ensure_parent_dir()?;
        let _ = remove_file_if_exists(&self.request_path);
        fs::write(&self.completion_path, b"stopped\n").with_context(|| {
            format!(
                "failed to write Windows ProjFS stop completion {}",
                self.completion_path.display()
            )
        })
    }

    fn cleanup(&self) -> Result<()> {
        remove_file_if_exists(&self.request_path)?;
        remove_file_if_exists(&self.completion_path)
    }

    fn ensure_parent_dir(&self) -> Result<()> {
        if let Some(parent) = self.request_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create Windows ProjFS stop request dir {}",
                    parent.display()
                )
            })?;
        }
        Ok(())
    }
}

#[derive(Debug)]
enum MountRootRemoveAttempt {
    Removed,
    Missing,
    Pending(io::Error),
}

fn stop_request_dir() -> Result<PathBuf> {
    if let Some(path) = env_path("LOCALAPPDATA") {
        return Ok(path.join("mcraw4vulkan").join(STOP_REQUEST_DIR_NAME));
    }
    if let Some(path) = env_path("TEMP") {
        return Ok(path.join("mcraw4vulkan").join(STOP_REQUEST_DIR_NAME));
    }
    bail!(
        "no safe runtime directory found for Windows ProjFS stop requests; set LOCALAPPDATA or TEMP"
    )
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn mountpoint_hash(mount_point: &Path) -> u64 {
    let identity = fs::canonicalize(mount_point).unwrap_or_else(|_| mount_point.to_path_buf());
    let mut hash = FNV1A64_OFFSET_BASIS;
    for code_unit in identity.as_os_str().encode_wide() {
        for byte in code_unit.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV1A64_PRIME);
        }
    }
    hash
}

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(PlatformMountError::Io {
            message: format!("failed to remove {}: {}", path.display(), error),
        }
        .into()),
    }
}

fn try_remove_inactive_mount_root(mount_point: &Path) -> Result<MountRootRemoveAttempt> {
    match fs::remove_dir(mount_point) {
        Ok(()) => Ok(MountRootRemoveAttempt::Removed),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            Ok(MountRootRemoveAttempt::Missing)
        }
        Err(error) if remove_error_may_be_active_projection(&error) => {
            Ok(MountRootRemoveAttempt::Pending(error))
        }
        Err(error) => Err(PlatformMountError::Io {
            message: format!(
                "failed to remove Windows ProjFS mount root {}: {}",
                mount_point.display(),
                error
            ),
        }
        .into()),
    }
}

fn remove_error_may_be_active_projection(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(WINDOWS_ERROR_ACCESS_DENIED)
            | Some(WINDOWS_ERROR_SHARING_VIOLATION)
            | Some(WINDOWS_ERROR_DIR_NOT_EMPTY)
    )
}

fn cleanup_known_projection_artifacts(
    mount_root: &Path,
    fs: &dyn WindowsProjFsVirtualFileSystem,
) -> Result<()> {
    // Cleanup is limited to names and kinds reproduced by the shared VFS. An
    // unrecognized entry aborts the walk instead of broadening deletion beyond the
    // known projection namespace.
    cleanup_known_virtual_directory(mount_root, fs, SHARED_ROOT_INODE)
}

fn cleanup_known_virtual_directory(
    directory: &Path,
    fs: &dyn WindowsProjFsVirtualFileSystem,
    inode: u64,
) -> Result<()> {
    let Some(entries) = collect_existing_dir_entries(directory)? else {
        return Ok(());
    };
    let metadata = expected_metadata(fs, inode)?;
    if metadata.kind != VirtualFileKind::Directory {
        return Err(cleanup_invalid_root_error(
            directory,
            "expected virtual cleanup root is not a directory",
        ));
    }
    let expected_entries = expected_entries_by_name(fs.readdir(inode)?.ok_or_else(|| {
        cleanup_invalid_root_error(directory, "virtual cleanup directory has no entries")
    })?)?;

    for entry in entries {
        let name = entry.file_name();
        let Some(expected) = expected_entries.get(&name) else {
            return Err(cleanup_invalid_root_error(
                &entry.path(),
                "refusing to remove unknown file inside virtual clip folder",
            ));
        };
        let metadata = expected_metadata(fs, expected.inode)?;
        match metadata.kind {
            VirtualFileKind::Directory => {
                cleanup_known_virtual_directory(&entry.path(), fs, expected.inode)?;
            }
            VirtualFileKind::RegularFile => {
                let file_type = entry.file_type().map_err(|source| PlatformMountError::Io {
                    message: format!("failed to inspect {}: {}", entry.path().display(), source),
                })?;
                if file_type.is_dir() {
                    return Err(cleanup_invalid_root_error(
                        &entry.path(),
                        "refusing to remove directory where virtual file was expected",
                    ));
                }
                remove_file_if_exists(&entry.path())?;
            }
        }
    }

    remove_empty_dir_if_exists(directory)
}

fn collect_existing_dir_entries(path: &Path) -> Result<Option<Vec<fs::DirEntry>>> {
    let read_dir = match fs::read_dir(path) {
        Ok(read_dir) => read_dir,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(PlatformMountError::Io {
                message: format!("failed to read directory {}: {}", path.display(), error),
            }
            .into());
        }
    };

    read_dir
        .collect::<io::Result<Vec<_>>>()
        .map(Some)
        .map_err(|source| {
            PlatformMountError::Io {
                message: format!(
                    "failed to enumerate directory {}: {}",
                    path.display(),
                    source
                ),
            }
            .into()
        })
}

fn expected_entries_by_name(
    entries: Vec<VirtualDirEntry>,
) -> Result<HashMap<OsString, VirtualDirEntry>> {
    let mut by_name = HashMap::with_capacity(entries.len());
    for entry in entries {
        if by_name.insert(entry.name.clone(), entry).is_some() {
            return Err(PlatformMountError::Internal {
                message: "virtual directory contains duplicate cleanup entry names".to_string(),
            }
            .into());
        }
    }
    Ok(by_name)
}

fn expected_metadata(
    fs: &dyn WindowsProjFsVirtualFileSystem,
    inode: u64,
) -> Result<VirtualFileMetadata> {
    fs.getattr(inode)?.ok_or_else(|| {
        PlatformMountError::Internal {
            message: format!("virtual cleanup inode {inode} has no metadata"),
        }
        .into()
    })
}

fn remove_empty_dir_if_exists(path: &Path) -> Result<()> {
    match fs::remove_dir(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(PlatformMountError::Io {
            message: format!("failed to remove directory {}: {}", path.display(), error),
        }
        .into()),
    }
}

fn cleanup_invalid_root_error(path: &Path, message: impl Into<String>) -> anyhow::Error {
    PlatformMountError::InvalidMountRoot {
        path: path.to_path_buf(),
        message: message.into(),
    }
    .into()
}

pub fn cleanup_stale_shared_root_mountpoint(
    mount_point: &Path,
    input_paths: &[PathBuf],
) -> Result<MountpointUnmountStatus> {
    match try_remove_inactive_mount_root(mount_point)? {
        MountRootRemoveAttempt::Removed => return Ok(MountpointUnmountStatus::Unmounted),
        MountRootRemoveAttempt::Missing => return Ok(MountpointUnmountStatus::NotMounted),
        MountRootRemoveAttempt::Pending(_) => {}
    }

    let build =
        build_shared_root_virtual_file_system(input_paths, VirtualFileSystemConfig::default())?;
    cleanup_known_projection_artifacts(mount_point, build.fs.as_ref())?;
    Ok(MountpointUnmountStatus::Unmounted)
}

pub fn cleanup_stale_single_clip_mountpoint(
    mount_point: &Path,
    input_path: &Path,
) -> Result<MountpointUnmountStatus> {
    match try_remove_inactive_mount_root(mount_point)? {
        MountRootRemoveAttempt::Removed => return Ok(MountpointUnmountStatus::Unmounted),
        MountRootRemoveAttempt::Missing => return Ok(MountpointUnmountStatus::NotMounted),
        MountRootRemoveAttempt::Pending(_) => {}
    }

    let clip_fs = Arc::new(VirtualFileSystem::open_clip(
        input_path,
        VirtualFileSystemConfig::default(),
    )?);
    let root_view = SingleClipRootVirtualFileSystem::new(clip_fs);
    cleanup_known_projection_artifacts(mount_point, &root_view)?;
    Ok(MountpointUnmountStatus::Unmounted)
}

pub fn provider_process_is_live(process_id: u32) -> bool {
    mcraw4vulkan_projfs_ffi::process_id_is_live(process_id)
}

fn unmount_owned_mountpoint_with_wait_config(
    mount_point: &Path,
    wait: StopRequestWaitConfig,
) -> Result<MountpointUnmountStatus> {
    match try_remove_inactive_mount_root(mount_point)? {
        MountRootRemoveAttempt::Removed => Ok(MountpointUnmountStatus::Unmounted),
        MountRootRemoveAttempt::Missing => Ok(MountpointUnmountStatus::NotMounted),
        MountRootRemoveAttempt::Pending(_) => {
            let paths = StopRequestPaths::for_mountpoint(mount_point)?;
            request_foreground_stop_and_wait(mount_point, &paths, wait)
        }
    }
}

fn request_foreground_stop_and_wait(
    mount_point: &Path,
    paths: &StopRequestPaths,
    wait: StopRequestWaitConfig,
) -> Result<MountpointUnmountStatus> {
    paths.request_stop()?;
    let start = Instant::now();

    loop {
        if paths.completion_exists() {
            match try_remove_inactive_mount_root(mount_point)? {
                MountRootRemoveAttempt::Removed | MountRootRemoveAttempt::Missing => {
                    let _ = paths.cleanup();
                    return Ok(MountpointUnmountStatus::Unmounted);
                }
                MountRootRemoveAttempt::Pending(_) => {}
            }
        }

        let elapsed = start.elapsed();
        if elapsed >= wait.timeout {
            let last_remove_error = match try_remove_inactive_mount_root(mount_point)? {
                MountRootRemoveAttempt::Removed | MountRootRemoveAttempt::Missing => {
                    let _ = paths.cleanup();
                    return Ok(MountpointUnmountStatus::Unmounted);
                }
                MountRootRemoveAttempt::Pending(error) => error.to_string(),
            };
            let _ = paths.cleanup();
            return Err(PlatformMountError::Internal {
                message: format!(
                    "timed out after {:.1}s waiting for foreground Windows ProjFS owner to stop {}; last root removal error: {}",
                    wait.timeout.as_secs_f32(),
                    mount_point.display(),
                    last_remove_error
                ),
            }
            .into());
        }

        let remaining = wait.timeout.saturating_sub(elapsed);
        let sleep_for = if wait.poll_interval < remaining {
            wait.poll_interval
        } else {
            remaining
        };
        if !sleep_for.is_zero() {
            thread::sleep(sleep_for);
        }
    }
}

// The session owns the native projection and every VFS allocation reachable by
// callbacks. Stopping completes callback teardown before those allocations are
// released or cleanup walks the known projected namespace.
struct WindowsProjfsMountSession {
    projection: mcraw4vulkan_projfs_ffi::Projection,
    fs: Arc<dyn WindowsProjFsVirtualFileSystem>,
    stats_fs: Option<Arc<VirtualFileSystem>>,
    unique_clip_count: usize,
    mount_root: PathBuf,
}

impl WindowsProjfsMountSession {
    fn start(
        mount_root: PathBuf,
        fs: Arc<dyn WindowsProjFsVirtualFileSystem>,
        stats_fs: Option<Arc<VirtualFileSystem>>,
        unique_clip_count: usize,
    ) -> Result<Self> {
        let provider = VirtualFileSystemProjectionProvider::new(fs.clone());
        let projection = mcraw4vulkan_projfs_ffi::start_projection(&mount_root, provider)
            .with_context(|| {
                format!(
                    "failed to start Windows ProjFS projection at {}",
                    mount_root.display()
                )
            })?;

        let session = Self {
            projection,
            fs,
            stats_fs,
            unique_clip_count,
            mount_root,
        };
        session.clear_stale_stop_request_files();
        Ok(session)
    }

    fn stop(&mut self) {
        self.projection.stop();
    }

    fn stop_and_cleanup_projection_root(&mut self) -> Result<()> {
        self.stop();
        cleanup_known_projection_artifacts(&self.mount_root, self.fs.as_ref())
    }

    fn wait_until_mount_root_removed(&mut self) -> Result<PlatformMountStatus> {
        let stop_request_paths = StopRequestPaths::for_mountpoint(&self.mount_root).ok();
        loop {
            if let Some(paths) = &stop_request_paths {
                if paths.request_exists() {
                    self.stop_and_cleanup_projection_root()?;
                    paths.mark_completed()?;
                    return Ok(PlatformMountStatus::Unmounted);
                }
            }

            match fs::metadata(&self.mount_root) {
                Ok(_) => thread::sleep(Duration::from_millis(250)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    self.stop();
                    return Ok(PlatformMountStatus::Unmounted);
                }
                Err(error) => {
                    return Err(PlatformMountError::Io {
                        message: format!(
                            "failed to inspect Windows ProjFS mount root {}: {}",
                            self.mount_root.display(),
                            error
                        ),
                    }
                    .into());
                }
            }
        }
    }

    fn stats_snapshot(&self) -> Result<Option<VirtualFileSystemStatsSnapshot>> {
        self.stats_fs
            .as_ref()
            .map(|fs| fs.stats_snapshot())
            .transpose()
    }

    fn unique_clip_count(&self) -> usize {
        self.unique_clip_count
    }

    fn instance_id(&self) -> ProjectionInstanceId {
        self.projection.instance_id()
    }

    fn clear_stale_stop_request_files(&self) {
        if let Ok(paths) = StopRequestPaths::for_mountpoint(&self.mount_root) {
            let _ = paths.cleanup();
        }
    }
}

impl Drop for WindowsProjfsMountSession {
    fn drop(&mut self) {
        self.stop();
    }
}

pub struct MountHandle {
    pub mount_name: String,
    pub mount_point: PathBuf,
    pub input_path: PathBuf,
    session: Option<WindowsProjfsMountSession>,
}

impl fmt::Debug for MountHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MountHandle")
            .field("mount_name", &self.mount_name)
            .field("mount_point", &self.mount_point)
            .field("input_path", &self.input_path)
            .field("status", &self.status())
            .finish()
    }
}

impl MountHandle {
    pub fn kind(&self) -> PlatformMountKind {
        PlatformMountKind::WindowsProjFs
    }

    pub fn mountpoint(&self) -> &Path {
        &self.mount_point
    }

    pub fn instance_id(&self) -> ProjectionInstanceId {
        self.session
            .as_ref()
            .map(WindowsProjfsMountSession::instance_id)
            .unwrap_or_else(|| ProjectionInstanceId::from_u128(0))
    }

    pub fn capabilities(&self) -> PlatformMountCapabilities {
        PlatformMountCapabilities::windows_projfs()
    }

    pub fn status(&self) -> PlatformMountStatus {
        if self.session.is_some() {
            PlatformMountStatus::Mounted
        } else {
            PlatformMountStatus::Unmounted
        }
    }

    pub fn request_unmount(mut self) -> Result<PlatformUnmountOutcome> {
        let Some(mut session) = self.session.take() else {
            return Ok(PlatformUnmountOutcome::NotMounted);
        };

        session.stop_and_cleanup_projection_root()?;
        Ok(PlatformUnmountOutcome::Unmounted)
    }

    pub fn wait_until_unmounted_status(mut self) -> Result<PlatformMountStatus> {
        let Some(mut session) = self.session.take() else {
            return Ok(PlatformMountStatus::Unmounted);
        };

        session.wait_until_mount_root_removed()
    }

    pub fn unmount(self) -> Result<()> {
        match self.request_unmount()? {
            PlatformUnmountOutcome::Unmounted | PlatformUnmountOutcome::NotMounted => Ok(()),
            PlatformUnmountOutcome::Unsupported => Err(PlatformMountError::unsupported(
                PlatformMountKind::WindowsProjFs,
                "Windows ProjFS DNG mounting is not implemented",
            )
            .into()),
            PlatformUnmountOutcome::Failed => Err(PlatformMountError::Internal {
                message: "Windows ProjFS unmount failed".to_string(),
            }
            .into()),
        }
    }

    pub fn wait_until_unmounted(self) -> Result<()> {
        match self.wait_until_unmounted_status()? {
            PlatformMountStatus::Unmounted => Ok(()),
            status => Err(PlatformMountError::Internal {
                message: format!("Windows ProjFS mount did not unmount cleanly: {status:?}"),
            }
            .into()),
        }
    }
}

impl Drop for MountHandle {
    fn drop(&mut self) {
        if let Some(session) = self.session.as_mut() {
            session.stop();
        }
    }
}

pub struct SharedRootMountHandle {
    pub mount_name: String,
    pub mount_point: PathBuf,
    mounted_clips: Vec<SharedRootMountedClip>,
    session: Option<WindowsProjfsMountSession>,
}

impl fmt::Debug for SharedRootMountHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SharedRootMountHandle")
            .field("mount_name", &self.mount_name)
            .field("mount_point", &self.mount_point)
            .field("mounted_clips", &self.mounted_clips)
            .field("status", &self.status())
            .finish()
    }
}

impl SharedRootMountHandle {
    pub fn kind(&self) -> PlatformMountKind {
        PlatformMountKind::WindowsProjFs
    }

    pub fn mountpoint(&self) -> &Path {
        &self.mount_point
    }

    pub fn mounted_clips(&self) -> &[SharedRootMountedClip] {
        &self.mounted_clips
    }

    pub fn unique_clip_count(&self) -> usize {
        self.session
            .as_ref()
            .map(WindowsProjfsMountSession::unique_clip_count)
            .unwrap_or(0)
    }

    pub fn capabilities(&self) -> PlatformMountCapabilities {
        PlatformMountCapabilities::windows_projfs()
    }

    pub fn status(&self) -> PlatformMountStatus {
        if self.session.is_some() {
            PlatformMountStatus::Mounted
        } else {
            PlatformMountStatus::Unmounted
        }
    }

    pub fn request_unmount(mut self) -> Result<PlatformUnmountOutcome> {
        let Some(mut session) = self.session.take() else {
            return Ok(PlatformUnmountOutcome::NotMounted);
        };

        session.stop_and_cleanup_projection_root()?;
        Ok(PlatformUnmountOutcome::Unmounted)
    }

    pub fn wait_until_unmounted_status(mut self) -> Result<PlatformMountStatus> {
        let Some(mut session) = self.session.take() else {
            return Ok(PlatformMountStatus::Unmounted);
        };

        session.wait_until_mount_root_removed()
    }

    pub fn unmount(self) -> Result<()> {
        match self.request_unmount()? {
            PlatformUnmountOutcome::Unmounted | PlatformUnmountOutcome::NotMounted => Ok(()),
            PlatformUnmountOutcome::Unsupported => Err(PlatformMountError::unsupported(
                PlatformMountKind::WindowsProjFs,
                "Windows ProjFS DNG mounting is not implemented",
            )
            .into()),
            PlatformUnmountOutcome::Failed => Err(PlatformMountError::Internal {
                message: "Windows ProjFS unmount failed".to_string(),
            }
            .into()),
        }
    }

    pub fn wait_until_unmounted(self) -> Result<()> {
        match self.wait_until_unmounted_status()? {
            PlatformMountStatus::Unmounted => Ok(()),
            status => Err(PlatformMountError::Internal {
                message: format!("Windows ProjFS mount did not unmount cleanly: {status:?}"),
            }
            .into()),
        }
    }
}

impl Drop for SharedRootMountHandle {
    fn drop(&mut self) {
        if let Some(session) = self.session.as_mut() {
            session.stop();
        }
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

pub fn unmount_owned_mountpoint(mount_point: &Path) -> Result<MountpointUnmountStatus> {
    unmount_owned_mountpoint_with_wait_config(mount_point, StopRequestWaitConfig::default())
}

impl PlatformMountHandle for MountHandle {
    fn kind(&self) -> PlatformMountKind {
        PlatformMountKind::WindowsProjFs
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
        MountHandle::status(self)
    }

    fn stats_snapshot(&self) -> Result<Option<VirtualFileSystemStatsSnapshot>> {
        self.session
            .as_ref()
            .map(WindowsProjfsMountSession::stats_snapshot)
            .unwrap_or(Ok(None))
    }

    fn capabilities(&self) -> PlatformMountCapabilities {
        MountHandle::capabilities(self)
    }
}

struct VirtualFileSystemProjectionProvider<V: ?Sized = VirtualFileSystem> {
    fs: Arc<V>,
}

impl<V: ?Sized> VirtualFileSystemProjectionProvider<V> {
    fn new(fs: Arc<V>) -> Self {
        Self { fs }
    }
}

impl<V> ProjectionProvider for VirtualFileSystemProjectionProvider<V>
where
    V: WindowsProjFsVirtualFileSystem + ?Sized,
{
    fn list_directory(&self, relative_path: &Path) -> ProviderResult<Vec<DirectoryEntry>> {
        let metadata = self.metadata_for_path(relative_path)?;
        if metadata.kind != VirtualFileKind::Directory {
            return Err(ProviderError::NotFound);
        }

        let entries = self
            .fs
            .readdir(metadata.inode)
            .map_err(provider_internal_error)?
            .ok_or(ProviderError::NotFound)?;

        entries
            .into_iter()
            .map(|entry| self.directory_entry_from_vfs_entry(entry))
            .collect()
    }

    fn placeholder_info(&self, relative_path: &Path) -> ProviderResult<PlaceholderInfo> {
        self.metadata_for_path(relative_path)
            .map(placeholder_info_from_vfs_metadata)
    }

    fn read_file(
        &self,
        relative_path: &Path,
        byte_offset: u64,
        length: u32,
    ) -> ProviderResult<Vec<u8>> {
        let metadata = self.metadata_for_path(relative_path)?;
        if metadata.kind != VirtualFileKind::RegularFile {
            return Err(ProviderError::NotAFile);
        }

        self.fs
            .read_data(metadata.inode, byte_offset, length)
            .map_err(provider_internal_error)?
            .ok_or(ProviderError::NotAFile)
    }
}

impl<V> VirtualFileSystemProjectionProvider<V>
where
    V: WindowsProjFsVirtualFileSystem + ?Sized,
{
    fn metadata_for_path(&self, relative_path: &Path) -> ProviderResult<VirtualFileMetadata> {
        let components = normalized_provider_components(relative_path)?;
        let mut metadata = self
            .fs
            .getattr(SHARED_ROOT_INODE)
            .map_err(provider_internal_error)?
            .ok_or(ProviderError::NotFound)?;

        for component in components {
            if metadata.kind != VirtualFileKind::Directory {
                return Err(ProviderError::NotFound);
            }
            metadata = self
                .fs
                .lookup(metadata.inode, component.as_os_str())
                .map_err(provider_internal_error)?
                .ok_or(ProviderError::NotFound)?;
        }

        Ok(metadata)
    }

    fn directory_entry_from_vfs_entry(
        &self,
        entry: VirtualDirEntry,
    ) -> ProviderResult<DirectoryEntry> {
        directory_entry_from_vfs_entry(self.fs.as_ref(), entry)
    }
}

trait WindowsProjFsVirtualFileSystem: Send + Sync {
    fn lookup(&self, parent_inode: u64, name: &OsStr) -> Result<Option<VirtualFileMetadata>>;
    fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>>;
    fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>>;
    fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Vec<u8>>>;
}

impl WindowsProjFsVirtualFileSystem for VirtualFileSystem {
    fn lookup(&self, parent_inode: u64, name: &OsStr) -> Result<Option<VirtualFileMetadata>> {
        VirtualFileSystem::lookup(self, parent_inode, name)
    }

    fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
        VirtualFileSystem::getattr(self, inode)
    }

    fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>> {
        VirtualFileSystem::readdir(self, inode)
    }

    fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Vec<u8>>> {
        Ok(VirtualFileSystem::read_data(self, inode, offset, size)?
            .map(|data| data.as_slice().to_vec()))
    }
}

impl WindowsProjFsVirtualFileSystem for SharedRootVirtualFileSystem {
    fn lookup(&self, parent_inode: u64, name: &OsStr) -> Result<Option<VirtualFileMetadata>> {
        SharedRootVirtualFileSystem::lookup(self, parent_inode, name)
    }

    fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
        SharedRootVirtualFileSystem::getattr(self, inode)
    }

    fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>> {
        SharedRootVirtualFileSystem::readdir(self, inode)
    }

    fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Vec<u8>>> {
        SharedRootVirtualFileSystem::read_data(self, inode, offset, size)
    }
}

impl WindowsProjFsVirtualFileSystem for SingleClipRootVirtualFileSystem {
    fn lookup(&self, parent_inode: u64, name: &OsStr) -> Result<Option<VirtualFileMetadata>> {
        SingleClipRootVirtualFileSystem::lookup(self, parent_inode, name)
    }

    fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
        SingleClipRootVirtualFileSystem::getattr(self, inode)
    }

    fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>> {
        SingleClipRootVirtualFileSystem::readdir(self, inode)
    }

    fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Vec<u8>>> {
        SingleClipRootVirtualFileSystem::read_data(self, inode, offset, size)
    }
}

fn directory_entry_from_vfs_entry<V>(
    fs: &V,
    entry: VirtualDirEntry,
) -> ProviderResult<DirectoryEntry>
where
    V: WindowsProjFsVirtualFileSystem + ?Sized,
{
    let metadata = fs
        .getattr(entry.inode)
        .map_err(provider_internal_error)?
        .ok_or_else(|| {
            ProviderError::Internal(format!(
                "VFS directory entry {} has no metadata",
                entry.inode
            ))
        })?;

    Ok(DirectoryEntry::new(
        entry.name,
        placeholder_info_from_vfs_metadata(metadata),
    ))
}

fn normalized_provider_components(relative_path: &Path) -> ProviderResult<Vec<OsString>> {
    let mut components = Vec::new();

    if relative_path.as_os_str().is_empty() {
        return Ok(components);
    }

    for component in relative_path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(name) => {
                if name.to_string_lossy().contains(':') {
                    return Err(ProviderError::InvalidPath(
                        "alternate data streams are not supported".to_string(),
                    ));
                }
                components.push(name.to_os_string());
            }
            Component::ParentDir => {
                return Err(ProviderError::InvalidPath(
                    "parent traversal is not supported".to_string(),
                ));
            }
            Component::Prefix(_) | Component::RootDir => {
                return Err(ProviderError::InvalidPath(
                    "absolute provider paths are not supported".to_string(),
                ));
            }
        }
    }

    Ok(components)
}

fn placeholder_info_from_vfs_metadata(metadata: VirtualFileMetadata) -> PlaceholderInfo {
    let info = match metadata.kind {
        VirtualFileKind::Directory => PlaceholderInfo::directory(),
        VirtualFileKind::RegularFile => PlaceholderInfo::regular_file(metadata.byte_len),
    };
    let filetime = windows_filetime_from_virtual_timestamp(metadata.timestamp);

    info.with_file_times(filetime, filetime, filetime, filetime)
}

fn windows_filetime_from_virtual_timestamp(timestamp: VirtualTimestamp) -> i64 {
    const WINDOWS_UNIX_EPOCH_OFFSET_SECONDS: i128 = 11_644_473_600;
    const WINDOWS_TICKS_PER_SECOND: i128 = 10_000_000;

    let seconds = i128::from(timestamp.seconds) + WINDOWS_UNIX_EPOCH_OFFSET_SECONDS;
    let ticks = seconds
        .saturating_mul(WINDOWS_TICKS_PER_SECOND)
        .saturating_add(i128::from(timestamp.nanos / 100));

    ticks.clamp(0, i128::from(i64::MAX)) as i64
}

fn provider_internal_error(error: anyhow::Error) -> ProviderError {
    ProviderError::Internal(error.to_string())
}
