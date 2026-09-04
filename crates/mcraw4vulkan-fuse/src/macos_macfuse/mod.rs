use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use mcraw4vulkan_dngwriter::DngSinkVignetteMode;

#[path = "../linux_fuse3/fuser_fs.rs"]
mod fuser_fs;

use crate::platform_lifecycle::{
    PlatformMountCapabilities, PlatformMountHandle, PlatformMountKind, PlatformMountRequest,
    PlatformMountStatus, PlatformUnmountOutcome,
};
use crate::shared_root::{
    AddClipResult, SharedRootVirtualFileSystem, SingleClipRootVirtualFileSystem,
};
use crate::virtual_fs::{
    DngGenerationBackend, DngGenerationExecutionPolicy, VirtualFileSystem, VirtualFileSystemConfig,
    VirtualFileSystemStatsSnapshot,
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

// Each returned handle owns the background macFUSE session and its VFS
// allocation. Consuming unmount joins callbacks before those resources drop.
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
        let fs = Arc::new(
            VirtualFileSystem::open_clip(&request.input_path, request.virtual_file_system_config())
                .with_context(|| {
                    format!(
                        "failed to create macFUSE filesystem for {}",
                        request.input_path.display()
                    )
                })?,
        );

        let root_view = Arc::new(SingleClipRootVirtualFileSystem::new(fs.clone()));
        let attr_owner = fuser_file_attr_owner_for_mountpoint(&request.mount_point)?;
        let filesystem =
            fuser_fs::LinuxSingleClipRootFuse3FileSystem::new_single_clip_root_with_owner(
                root_view, attr_owner,
            );
        let session = fuser::spawn_mount2(
            filesystem,
            &request.mount_point,
            &macos_fuser_config(&request.mount_name, attr_owner),
        )
        .with_context(|| {
            format!(
                "failed to mount macFUSE at {}",
                request.mount_point.display()
            )
        })?;

        Ok(MountHandle {
            mount_name: request.mount_name,
            mount_point: request.mount_point,
            fs,
            session,
        })
    }

    pub fn mount_shared_root(
        &self,
        request: SharedRootMountRequest,
    ) -> Result<SharedRootMountHandle> {
        if request.input_paths.is_empty() {
            bail!("shared-root macFUSE mount requires at least one input clip");
        }

        let virtual_config = request.virtual_file_system_config();
        let mut shared_root = SharedRootVirtualFileSystem::new();
        let mut mounted_clips = Vec::with_capacity(request.input_paths.len());

        for input_path in &request.input_paths {
            let clip_fs = Arc::new(
                VirtualFileSystem::open_clip(input_path, virtual_config).with_context(|| {
                    format!(
                        "failed to create macFUSE filesystem for {}",
                        input_path.display()
                    )
                })?,
            );
            let result = shared_root
                .registry_mut()
                .add_virtual_file_system_from_path(input_path.clone(), clip_fs)
                .with_context(|| {
                    format!(
                        "failed to add {} to macOS shared-root filesystem",
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
        let attr_owner = fuser_file_attr_owner_for_mountpoint(&request.mount_point)?;
        let filesystem = fuser_fs::LinuxSharedRootFuse3FileSystem::new_shared_root_with_owner(
            fs.clone(),
            attr_owner,
        );
        let session = fuser::spawn_mount2(
            filesystem,
            &request.mount_point,
            &macos_fuser_config(&request.mount_name, attr_owner),
        )
        .with_context(|| {
            format!(
                "failed to mount macFUSE at {}",
                request.mount_point.display()
            )
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

pub struct MountHandle {
    pub mount_name: String,
    pub mount_point: PathBuf,
    fs: Arc<VirtualFileSystem>,
    session: fuser::BackgroundSession,
}

impl MountHandle {
    pub fn stats_snapshot(&self) -> Result<VirtualFileSystemStatsSnapshot> {
        self.fs.stats_snapshot()
    }

    pub fn stats_summary_line(&self) -> Result<String> {
        self.fs.stats_summary_line()
    }

    pub fn kind(&self) -> PlatformMountKind {
        PlatformMountKind::MacosMacFuse
    }

    pub fn mountpoint(&self) -> &Path {
        &self.mount_point
    }

    pub fn capabilities(&self) -> PlatformMountCapabilities {
        PlatformMountCapabilities::macos_macfuse()
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

impl PlatformMountHandle for MountHandle {
    fn kind(&self) -> PlatformMountKind {
        PlatformMountKind::MacosMacFuse
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
        PlatformMountCapabilities::macos_macfuse()
    }
}

pub struct SharedRootMountHandle {
    pub mount_name: String,
    pub mount_point: PathBuf,
    mounted_clips: Vec<SharedRootMountedClip>,
    fs: Arc<SharedRootVirtualFileSystem>,
    session: fuser::BackgroundSession,
}

impl SharedRootMountHandle {
    pub fn kind(&self) -> PlatformMountKind {
        PlatformMountKind::MacosMacFuse
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
        PlatformMountCapabilities::macos_macfuse()
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
        filesystem_type.contains("fuse") || filesystem_type.contains("macfuse")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacosMountpointClassification {
    NotMounted,
    Mounted(MacosMountInfo),
}

pub fn classify_macos_mountpoint(path: &Path) -> Result<MacosMountpointClassification> {
    let output = Command::new("mount")
        .output()
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
    if mount_point.as_os_str().is_empty() {
        bail!("refusing to unmount an empty mountpoint path");
    }
    if !mount_point.exists() {
        return Ok(MountpointUnmountStatus::NotMounted);
    }

    let commands: &[(&str, &[&str])] = &[("umount", &[]), ("diskutil", &["unmount"])];
    let mut last_error = None::<String>;

    for (program, args) in commands {
        let output = Command::new(program).args(*args).arg(mount_point).output();
        let Ok(output) = output else {
            continue;
        };

        if output.status.success() {
            return Ok(MountpointUnmountStatus::Unmounted);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{stdout}\n{stderr}").to_ascii_lowercase();
        if is_not_mounted_message(&combined) {
            return Ok(MountpointUnmountStatus::NotMounted);
        }

        last_error = Some(format!(
            "{program} failed to unmount {}: {}{}",
            mount_point.display(),
            stdout.trim(),
            stderr.trim()
        ));
    }

    bail!(
        "{}",
        last_error.unwrap_or_else(|| format!(
            "no supported macOS unmount command was found for {}",
            mount_point.display()
        ))
    )
}

fn macos_fuser_config(mount_name: &str, attr_owner: fuser_fs::FuserFileAttrOwner) -> fuser::Config {
    use fuser::MountOption;

    let volume_name = macos_mount_option_value(mount_name);
    let mut config = fuser::Config::default();
    config.mount_options = vec![
        MountOption::FSName(format!("mcraw4vulkan:{volume_name}")),
        MountOption::RO,
        MountOption::CUSTOM("rdonly".to_string()),
        MountOption::CUSTOM(format!("user_id={}", attr_owner.uid)),
        MountOption::CUSTOM(format!("group_id={}", attr_owner.gid)),
        MountOption::CUSTOM("defer_permissions".to_string()),
        MountOption::CUSTOM("noappledouble".to_string()),
        MountOption::CUSTOM(format!("volname={volume_name}")),
        MountOption::NoDev,
        MountOption::NoSuid,
        MountOption::NoExec,
    ];
    config.n_threads = Some(1);
    config
}

fn fuser_file_attr_owner_for_mountpoint(path: &Path) -> Result<fuser_fs::FuserFileAttrOwner> {
    // macFUSE attributes follow the prepared mountpoint's owner rather than
    // assuming the provider process identity owns the virtual files.
    let metadata = fs::metadata(path).with_context(|| {
        format!(
            "failed to read owner metadata for macFUSE mountpoint {}",
            path.display()
        )
    })?;
    Ok(fuser_fs::FuserFileAttrOwner::new(
        metadata.uid(),
        metadata.gid(),
    ))
}

fn macos_mount_option_value(value: &str) -> String {
    value
        .chars()
        .map(|value| {
            if value == ',' || value == '\0' || value.is_control() {
                '_'
            } else {
                value
            }
        })
        .collect()
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

fn is_not_mounted_message(message: &str) -> bool {
    message.contains("not mounted")
        || message.contains("not currently mounted")
        || message.contains("not a mount point")
        || message.contains("not a mountpoint")
        || message.contains("no such file")
        || message.contains("not found")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_request_converts_to_macos_mount_request() {
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
    fn macos_mount_manager_reports_live_macfuse_capabilities() {
        let manager = MountManager::new();
        let capabilities = manager.capabilities();

        assert_eq!(capabilities.kind, PlatformMountKind::MacosMacFuse);
        assert!(capabilities.supports_live_mount);
        assert!(capabilities.supports_request_unmount);
        assert!(capabilities.supports_wait_until_unmounted);
        assert!(capabilities.requires_empty_mount_root);
    }

    #[test]
    fn macos_fuser_config_uses_conservative_mount_options() {
        let config = macos_fuser_config(
            "clip__1234567890",
            fuser_fs::FuserFileAttrOwner::new(501, 20),
        );
        let custom_options = config
            .mount_options
            .iter()
            .filter_map(|option| match option {
                fuser::MountOption::CUSTOM(value) => Some(value.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert!(config.mount_options.contains(&fuser::MountOption::RO));
        assert!(
            !config
                .mount_options
                .contains(&fuser::MountOption::DefaultPermissions)
        );
        assert!(config.mount_options.contains(&fuser::MountOption::NoExec));
        assert!(custom_options.contains(&"rdonly"));
        assert!(custom_options.contains(&"user_id=501"));
        assert!(custom_options.contains(&"group_id=20"));
        assert!(custom_options.contains(&"defer_permissions"));
        assert!(custom_options.contains(&"noappledouble"));
        assert!(custom_options.contains(&"volname=clip__1234567890"));
        assert!(!custom_options.contains(&"noapplexattr"));

        for forbidden in [
            "allow_other",
            "allow_root",
            "allow_recursion",
            "local",
            "backend=fskit",
            "nobrowse",
        ] {
            assert!(
                !custom_options.contains(&forbidden),
                "macFUSE options must not include {forbidden}"
            );
        }
    }

    #[test]
    fn macos_fuser_config_sanitizes_mount_option_values() {
        let config = macos_fuser_config(
            "clip,bad\u{0007}name",
            fuser_fs::FuserFileAttrOwner::new(1, 2),
        );
        let custom_options = config
            .mount_options
            .iter()
            .filter_map(|option| match option {
                fuser::MountOption::CUSTOM(value) => Some(value.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert!(config.mount_options.contains(&fuser::MountOption::FSName(
            "mcraw4vulkan:clip_bad_name".to_string()
        )));
        assert!(custom_options.contains(&"volname=clip_bad_name"));
    }

    fn macos_fixture_mount_point() -> String {
        ["/", "Users", "example", "mcraw4vulkan"].concat()
    }

    fn macos_fixture_system_volume() -> String {
        ["/", "System", "Volumes", "Data"].concat()
    }

    fn macos_fixture_disk() -> String {
        ["/", "dev", "disk3s5"].concat()
    }

    #[test]
    fn macos_mount_output_classifies_mounted_macfuse_path() {
        let mount_point = macos_fixture_mount_point();
        let system_volume = macos_fixture_system_volume();
        let disk = macos_fixture_disk();
        let output = format!(
            "mcraw4vulkan:clip on {mount_point} (macfuse, nodev, nosuid, read-only)\n{disk} on {system_volume} (apfs, local)\n"
        );

        let classification =
            classify_macos_mountpoint_from_output(&output, Path::new(&mount_point));

        let MacosMountpointClassification::Mounted(info) = classification else {
            panic!("expected mounted macFUSE classification");
        };
        assert_eq!(info.source, "mcraw4vulkan:clip");
        assert_eq!(info.mount_point, PathBuf::from(mount_point));
        assert_eq!(info.filesystem_type, "macfuse");
        assert!(info.is_fuse_like());
    }

    #[test]
    fn macos_mount_output_classifies_unmounted_path() {
        let mount_point = macos_fixture_mount_point();
        let system_volume = macos_fixture_system_volume();
        let disk = macos_fixture_disk();
        let output = format!("{disk} on {system_volume} (apfs, local)\n");

        let classification =
            classify_macos_mountpoint_from_output(&output, Path::new(&mount_point));

        assert_eq!(classification, MacosMountpointClassification::NotMounted);
    }

    #[test]
    fn macos_mount_output_matches_trailing_slash() {
        let mount_point = macos_fixture_mount_point();
        let output = format!("mcraw4vulkan:clip on {mount_point} (macfuse, nodev)\n");
        let mut path_with_trailing_slash = mount_point;
        path_with_trailing_slash.push('/');

        let classification =
            classify_macos_mountpoint_from_output(&output, Path::new(&path_with_trailing_slash));

        assert!(matches!(
            classification,
            MacosMountpointClassification::Mounted(_)
        ));
    }

    #[test]
    fn macos_unmount_status_maps_to_shared_outcome() {
        assert_eq!(
            PlatformUnmountOutcome::from(MountpointUnmountStatus::Unmounted),
            PlatformUnmountOutcome::Unmounted
        );
        assert_eq!(
            PlatformUnmountOutcome::from(MountpointUnmountStatus::NotMounted),
            PlatformUnmountOutcome::NotMounted
        );
    }

    #[test]
    fn not_mounted_messages_include_umount_and_diskutil_variants() {
        assert!(is_not_mounted_message("not currently mounted"));
        assert!(is_not_mounted_message("not a mount point"));
        assert!(is_not_mounted_message("no such file"));
    }
}
