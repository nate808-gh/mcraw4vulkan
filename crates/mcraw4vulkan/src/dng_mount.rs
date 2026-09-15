use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mcraw4vulkan_core::{
    CLIP_MOUNT_SUFFIX_EXPANSION_HEX_LENGTHS, MountClipIdentityInput,
    clip_mount_folder_name_with_suffix_len, format_clip_mount_hash_suffix,
};
use mcraw4vulkan_dngwriter::DngSinkVignetteMode;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use mcraw4vulkan_fuse::MountRequest;
use mcraw4vulkan_fuse::{
    DEFAULT_DNG_CACHE_FRAME_CAPACITY, DEFAULT_PREFETCH_FORWARD_FRAMES, DngGenerationBackend,
    DngGenerationExecutionPolicy, MountManager, PlatformUnmountOutcome, unmount_owned_mountpoint,
};
#[cfg(target_os = "macos")]
use mcraw4vulkan_fuse::{MacosMountInfo, MacosMountpointClassification, classify_macos_mountpoint};
#[cfg(target_os = "windows")]
use mcraw4vulkan_fuse::{
    MountRequest, cleanup_stale_shared_root_mountpoint as cleanup_stale_windows_projfs_shared_root,
    cleanup_stale_single_clip_mountpoint as cleanup_stale_windows_projfs_single_clip,
    provider_process_is_live as windows_projfs_provider_process_is_live,
};
use mcraw4vulkan_gpu::GpuBackendPreference;

use crate::app_policy::{DngAppBackendPolicy, DngAppPolicy, DngMountPolicy};
use crate::dng_mount_registry::{DngMountRegistry, DngMountRegistryRecord, canonical_source_path};

const DEFAULT_DNG_MOUNT_WORKER_THREADS: usize = 4;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DngMountRunConfig {
    pub input_paths: Vec<PathBuf>,
    pub backend: DngGenerationBackend,
    pub vignette_mode: DngSinkVignetteMode,
    pub mountpoint_path: PathBuf,
    pub cache_frame_capacity: usize,
    pub prefetch_forward_frames: usize,
    pub generation_execution_policy: DngGenerationExecutionPolicy,
    pub worker_threads: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DngUnmountRequest {
    File { input_path: PathBuf },
    All,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DngUnmountSummary {
    pub attempted: usize,
    pub unmounted: usize,
    pub already_absent: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DngUnmountRegistryOutcome {
    summary: DngUnmountSummary,
    candidate_record_count: usize,
    completed_record_count: usize,
}

pub trait DngUnmountAdapter {
    fn unmount_owned_mountpoint(&self, mountpoint: &Path) -> Result<PlatformUnmountOutcome>;

    fn cleanup_stale_shared_root_mountpoint(
        &self,
        mountpoint: &Path,
        records: &[DngMountRegistryRecord],
    ) -> Result<Option<PlatformUnmountOutcome>> {
        let _ = (mountpoint, records);
        Ok(None)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RealDngUnmountAdapter;

impl DngUnmountAdapter for RealDngUnmountAdapter {
    fn unmount_owned_mountpoint(&self, mountpoint: &Path) -> Result<PlatformUnmountOutcome> {
        Ok(unmount_owned_mountpoint(mountpoint)?.into())
    }

    #[cfg(target_os = "windows")]
    fn cleanup_stale_shared_root_mountpoint(
        &self,
        mountpoint: &Path,
        records: &[DngMountRegistryRecord],
    ) -> Result<Option<PlatformUnmountOutcome>> {
        if let [record] = records {
            match cleanup_stale_windows_projfs_single_clip(
                mountpoint,
                &record.source_path_canonical,
            ) {
                Ok(outcome) => return Ok(Some(outcome.into())),
                Err(single_clip_error) => {
                    let input_paths = vec![record.source_path_canonical.clone()];
                    return Ok(Some(
                        cleanup_stale_windows_projfs_shared_root(mountpoint, &input_paths)
                            .with_context(|| {
                                format!("single-clip ProjFS cleanup failed: {single_clip_error}")
                            })?
                            .into(),
                    ));
                }
            }
        }

        let input_paths = records
            .iter()
            .map(|record| record.source_path_canonical.clone())
            .collect::<Vec<_>>();
        if input_paths.is_empty() {
            return Ok(None);
        }

        Ok(Some(
            cleanup_stale_windows_projfs_shared_root(mountpoint, &input_paths)?.into(),
        ))
    }
}

impl DngMountRunConfig {
    pub fn from_app_policy_inputs(input_paths: &[PathBuf], policy: &DngAppPolicy) -> Result<Self> {
        Self::from_app_policy_inputs_with_home(input_paths, policy, default_dng_home_env())
    }

    fn from_app_policy_inputs_with_home(
        input_paths: &[PathBuf],
        policy: &DngAppPolicy,
        home: Option<OsString>,
    ) -> Result<Self> {
        if input_paths.is_empty() {
            bail!("DNG mount requires at least one input file");
        }
        if input_paths.len() > 1 {
            bail!("dng currently accepts one FILE per mount process; start one process per file");
        }

        let source_paths = input_paths
            .iter()
            .map(|input_path| canonical_source_path(input_path))
            .collect::<Result<Vec<_>>>()?;
        let first_source_path = source_paths
            .first()
            .expect("input_paths is checked non-empty above");
        let mountpoint_path =
            mountpoint_from_app_policy(first_source_path, &policy.mount, home.as_deref())?;

        Ok(Self {
            input_paths: source_paths,
            backend: dng_backend_from_app_policy(policy.backend),
            vignette_mode: policy.vignette,
            mountpoint_path,
            cache_frame_capacity: DEFAULT_DNG_CACHE_FRAME_CAPACITY,
            prefetch_forward_frames: DEFAULT_PREFETCH_FORWARD_FRAMES,
            generation_execution_policy: DngGenerationExecutionPolicy::Inflight2Default,
            worker_threads: DEFAULT_DNG_MOUNT_WORKER_THREADS,
        })
    }
}

pub fn run_dng_mount(config: DngMountRunConfig) -> Result<()> {
    let registry = DngMountRegistry::from_default_path()?;
    run_dng_mount_with_registry(config, &registry)
}

// The returned platform handle owns the mounted session through the foreground
// wait. Its registry row is added only after mount succeeds and removed only
// after unmount is confirmed.
pub fn run_dng_mount_with_registry(
    config: DngMountRunConfig,
    registry: &DngMountRegistry,
) -> Result<()> {
    ensure_default_generation_policy(config.generation_execution_policy)?;

    if config.input_paths.is_empty() {
        bail!("DNG mount requires at least one input file");
    }

    let source_paths = config
        .input_paths
        .iter()
        .map(|input_path| canonical_source_path(input_path))
        .collect::<Result<Vec<_>>>()?;
    if source_paths.len() > 1 {
        bail!("dng currently accepts one FILE per mount process; start one process per file");
    }
    let source_path = source_paths
        .first()
        .expect("input_paths is checked non-empty above")
        .clone();

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let _ = source_path;
        bail!("DNG/FUSE mount is not implemented on this platform");
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    {
        let all_records = registry.all_owned_records()?;
        #[cfg(target_os = "windows")]
        let all_records = prune_stale_windows_mount_records(
            registry,
            &config.mountpoint_path,
            &source_path,
            all_records,
        )?;
        ensure_common_parent_available(&config.mountpoint_path, &all_records)?;
        #[cfg(target_os = "macos")]
        ensure_macos_common_parent_not_active_mount(&config.mountpoint_path)?;
        let child_mountpoint =
            create_unique_child_mountpoint(&config.mountpoint_path, &source_path, &all_records)?;

        let request = MountRequest {
            input_path: source_path.clone(),
            mount_point: child_mountpoint.path.clone(),
            mount_name: child_mountpoint.folder_name.clone(),
            dng_backend: config.backend,
            dng_vignette_mode: config.vignette_mode,
            dng_generation_execution_policy: config.generation_execution_policy,
            max_cached_dng_frames: config.cache_frame_capacity,
            prefetch_forward_frames: config.prefetch_forward_frames,
            worker_threads: config.worker_threads,
        };

        let manager = MountManager::new();
        let handle = match manager.mount(request) {
            Ok(handle) => handle,
            Err(error) => {
                let _ = remove_empty_dir_if_exists(&child_mountpoint.path);
                return Err(error);
            }
        };
        #[cfg(target_os = "macos")]
        let mut handle = handle;

        let mount_index_start = registry.load_records()?.len().saturating_add(1);
        #[cfg(target_os = "windows")]
        let instance_id = Some(handle.instance_id().to_hyphenated_string());
        #[cfg(not(target_os = "windows"))]
        let instance_id = None;
        let record = DngMountRegistryRecord::new_with_instance_id(
            source_path.clone(),
            child_mountpoint.path.clone(),
            mount_index_start,
            instance_id,
        );
        if let Err(error) = registry.add_record(record.clone()) {
            let _ = handle.unmount();
            let _ = remove_empty_dir_if_exists(&child_mountpoint.path);
            return Err(error)
                .context("failed to register DNG mount after platform mount succeeded");
        }

        #[cfg(windows)]
        touch_projected_root_view_sidecars_for_shell_visibility(
            &child_mountpoint.path,
            &source_path,
        );
        let mount_start_message =
            format_mount_start_message(&source_path, &config.mountpoint_path, &child_mountpoint);
        for line in mount_start_message.lines() {
            eprintln!("{line}");
        }

        let wait_result = handle.wait_until_unmounted();
        if wait_result.is_ok() {
            let _ = registry.remove_mount_ids(std::slice::from_ref(&record.mount_id));
            let _ = remove_empty_dir_if_exists(&child_mountpoint.path);
            let _ = remove_empty_dir_if_exists(&config.mountpoint_path);
        }
        wait_result
    }
}

#[cfg(windows)]
fn touch_projected_root_view_sidecars_for_shell_visibility(mount_root: &Path, source_path: &Path) {
    let Some(source_stem) = source_path.file_stem() else {
        return;
    };
    let mut wav_name = source_stem.to_os_string();
    wav_name.push(".wav");
    let _ = fs::metadata(mount_root.join(wav_name));

    let mut first_dng_name = source_stem.to_os_string();
    first_dng_name.push("_000000.dng");
    let _ = fs::metadata(mount_root.join(first_dng_name));
}

fn format_mount_start_message(
    source_path: &Path,
    common_parent_path: &Path,
    child_mountpoint: &DngChildMountpoint,
) -> String {
    format!(
        "mounted {} at {}; serving in foreground\ncommon parent: {}\nclip folder: {}\npress Ctrl-C to stop\nor run mcraw4vulkan dng unmount FILE.mcraw from another terminal\nor run mcraw4vulkan dng unmount all from another terminal",
        source_path.display(),
        child_mountpoint.path.display(),
        common_parent_path.display(),
        child_mountpoint.folder_name
    )
}

pub fn run_dng_unmount_file(input: &Path) -> Result<DngUnmountSummary> {
    let registry = DngMountRegistry::from_default_path()?;
    let adapter = RealDngUnmountAdapter;
    run_dng_unmount_file_with_adapter(input, &registry, &adapter)
}

pub fn run_dng_unmount_all() -> Result<DngUnmountSummary> {
    let registry = DngMountRegistry::from_default_path()?;
    let adapter = RealDngUnmountAdapter;
    run_dng_unmount_all_with_adapter(&registry, &adapter)
}

pub fn run_dng_unmount_file_with_adapter(
    input: &Path,
    registry: &DngMountRegistry,
    adapter: &dyn DngUnmountAdapter,
) -> Result<DngUnmountSummary> {
    let source_path = canonical_source_path(input)?;
    let records = registry.records_for_source(&source_path)?;
    if records.is_empty() {
        bail!(
            "no mcraw4vulkan-owned DNG mount found for {}",
            source_path.display()
        );
    }
    if records.len() > 1 {
        bail!(
            "multiple DNG mounts exist for {}; use mcraw4vulkan dng unmount all",
            source_path.display()
        );
    }

    let outcome = unmount_registry_records(registry, adapter, &records)?;
    eprintln!("{}", format_unmount_file_message(&source_path, &outcome));
    Ok(outcome.summary)
}

pub fn run_dng_unmount_all_with_adapter(
    registry: &DngMountRegistry,
    adapter: &dyn DngUnmountAdapter,
) -> Result<DngUnmountSummary> {
    let records = registry.all_owned_records()?;
    if records.is_empty() {
        eprintln!("no mcraw4vulkan DNG mounts found");
        return Ok(DngUnmountSummary {
            attempted: 0,
            unmounted: 0,
            already_absent: 0,
        });
    }

    let outcome = unmount_registry_records(registry, adapter, &records)?;
    eprintln!("{}", format_unmount_all_message(&outcome));
    Ok(outcome.summary)
}

pub fn default_mountpoint_for_source(source_path: &Path) -> Result<PathBuf> {
    let _ = source_path;
    default_dng_shared_mount_root()
}

pub fn default_dng_shared_mount_root() -> Result<PathBuf> {
    let home = default_dng_home_env();
    default_dng_shared_mount_root_from_home_env(home.as_deref())
}

fn default_dng_home_env() -> Option<OsString> {
    if cfg!(target_os = "windows") {
        std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))
    } else {
        std::env::var_os("HOME")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DefaultDngSharedMountRootPlatform {
    Linux,
    Macos,
    Windows,
}

fn default_dng_shared_mount_root_platform() -> DefaultDngSharedMountRootPlatform {
    if cfg!(target_os = "macos") {
        DefaultDngSharedMountRootPlatform::Macos
    } else if cfg!(target_os = "windows") {
        DefaultDngSharedMountRootPlatform::Windows
    } else {
        DefaultDngSharedMountRootPlatform::Linux
    }
}

fn default_dng_shared_mount_root_from_home(home: &Path) -> PathBuf {
    home.join("mcraw4vulkan")
}

fn default_dng_shared_mount_root_from_home_env(home: Option<&OsStr>) -> Result<PathBuf> {
    default_dng_shared_mount_root_for_platform(default_dng_shared_mount_root_platform(), home)
}

fn default_dng_shared_mount_root_for_platform(
    platform: DefaultDngSharedMountRootPlatform,
    home: Option<&OsStr>,
) -> Result<PathBuf> {
    let Some(home) = home.filter(|home| !home.is_empty()) else {
        let message = match platform {
            DefaultDngSharedMountRootPlatform::Windows => {
                "DNG mount requires USERPROFILE or HOME to choose default shared mount root"
            }
            DefaultDngSharedMountRootPlatform::Linux | DefaultDngSharedMountRootPlatform::Macos => {
                "DNG mount requires HOME to choose default shared mount root $HOME/mcraw4vulkan"
            }
        };
        bail!("{message}");
    };
    Ok(default_dng_shared_mount_root_from_home(Path::new(home)))
}

pub fn dng_virtual_names_for_stem(stem: &str) -> (PathBuf, PathBuf, PathBuf) {
    let mounted_folder = PathBuf::from(stem);
    let first_dng_path = mounted_folder.join(format!("{stem}_000000.dng"));
    let audio_path = mounted_folder.join(format!("{stem}.wav"));
    (mounted_folder, first_dng_path, audio_path)
}

pub fn ensure_mountpoint_available(mountpoint: &Path) -> Result<()> {
    if mountpoint.exists() {
        if !mountpoint.is_dir() {
            bail!(
                "mountpoint {} exists but is not a directory",
                mountpoint.display()
            );
        }
        if fs::read_dir(mountpoint)
            .with_context(|| format!("failed to read mountpoint {}", mountpoint.display()))?
            .next()
            .is_some()
        {
            bail!(
                "mountpoint {} already exists and is not empty; refusing to mount over user data",
                mountpoint.display()
            );
        }
        return Ok(());
    }

    fs::create_dir_all(mountpoint).map_err(|error| {
        anyhow::anyhow!(
            "failed to create mountpoint {}: {error}",
            mountpoint.display()
        )
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DngChildMountpoint {
    folder_name: String,
    path: PathBuf,
}

fn ensure_common_parent_available(
    common_parent: &Path,
    records: &[DngMountRegistryRecord],
) -> Result<()> {
    if records
        .iter()
        .any(|record| record.mountpoint_path == common_parent)
    {
        bail!(
            "DNG common parent {} is recorded as an old shared-root mountpoint; run mcraw4vulkan dng unmount all before mounting new per-file DNG roots",
            common_parent.display()
        );
    }

    if common_parent.exists() {
        if !common_parent.is_dir() {
            bail!(
                "DNG common parent {} exists but is not a directory",
                common_parent.display()
            );
        }
        return Ok(());
    }

    fs::create_dir_all(common_parent).map_err(|error| {
        anyhow::anyhow!(
            "failed to create DNG common parent {}: {error}",
            common_parent.display()
        )
    })
}

#[cfg(target_os = "macos")]
fn ensure_macos_common_parent_not_active_mount(common_parent: &Path) -> Result<()> {
    if !common_parent.exists() {
        return Ok(());
    }

    ensure_macos_common_parent_mount_classification(
        common_parent,
        classify_macos_mountpoint(common_parent)?,
    )
}

#[cfg(target_os = "macos")]
fn ensure_macos_common_parent_mount_classification(
    common_parent: &Path,
    classification: MacosMountpointClassification,
) -> Result<()> {
    match classification {
        MacosMountpointClassification::NotMounted => Ok(()),
        MacosMountpointClassification::Mounted(info) if info.is_fuse_like() => {
            bail!(
                "{}",
                stale_macos_common_parent_mount_error(common_parent, &info)
            )
        }
        MacosMountpointClassification::Mounted(info) => {
            bail!(
                "DNG common parent {} is an active mountpoint (filesystem type {}) instead of an ordinary directory; refusing to create per-file DNG mountpoints inside it. mcraw4vulkan expects $HOME/mcraw4vulkan to be an ordinary directory. If this is stale macFUSE/FUSE state, run: mcraw4vulkan dng unmount all. If that does not clear it, manually unmount the stale volume with Finder or diskutil, then retry.",
                common_parent.display(),
                info.filesystem_type
            )
        }
    }
}

#[cfg(target_os = "macos")]
fn stale_macos_common_parent_mount_error(common_parent: &Path, info: &MacosMountInfo) -> String {
    format!(
        "DNG common parent {} appears to be an active macFUSE/FUSE mount (filesystem type {}) instead of an ordinary directory. This may be stale state from an older mcraw4vulkan build. Run: mcraw4vulkan dng unmount all. If that does not clear it, manually unmount the stale macFUSE volume with Finder or diskutil, then retry.",
        common_parent.display(),
        info.filesystem_type
    )
}

#[cfg(target_os = "windows")]
fn prune_stale_windows_mount_records(
    registry: &DngMountRegistry,
    common_parent: &Path,
    source_path: &Path,
    records: Vec<DngMountRegistryRecord>,
) -> Result<Vec<DngMountRegistryRecord>> {
    let mut kept = Vec::with_capacity(records.len());
    let mut remove_ids = Vec::new();
    let mut errors = Vec::new();

    for record in records {
        if record.platform != "windows" {
            kept.push(record);
            continue;
        }
        if windows_projfs_provider_process_is_live(record.process_id) {
            kept.push(record);
            continue;
        }
        if !windows_registered_child_mountpoint_is_owned(common_parent, &record.mountpoint_path) {
            kept.push(record);
            continue;
        }

        match cleanup_registered_stale_windows_root(common_parent, &record) {
            Ok(()) => remove_ids.push(record.mount_id.clone()),
            Err(error) => {
                if record.source_path_canonical == source_path {
                    errors.push(format!("{}: {error}", record.mountpoint_path.display()));
                } else {
                    kept.push(record);
                }
            }
        }
    }

    registry.remove_mount_ids(&remove_ids)?;
    if !errors.is_empty() {
        bail!(
            "failed to clean stale Windows ProjFS root(s) before mounting {}: {}",
            source_path.display(),
            errors.join("; ")
        );
    }

    Ok(kept)
}

#[cfg(target_os = "windows")]
fn cleanup_registered_stale_windows_root(
    common_parent: &Path,
    record: &DngMountRegistryRecord,
) -> Result<()> {
    if !windows_registered_child_mountpoint_is_owned(common_parent, &record.mountpoint_path) {
        bail!(
            "refusing stale Windows ProjFS cleanup outside common parent: {}",
            record.mountpoint_path.display()
        );
    }

    match cleanup_stale_windows_projfs_single_clip(
        &record.mountpoint_path,
        &record.source_path_canonical,
    ) {
        Ok(_) => Ok(()),
        Err(single_clip_error) => {
            remove_registered_stale_windows_child_root(&record.mountpoint_path).with_context(|| {
                format!(
                    "single-clip cleanup failed: {single_clip_error}; fallback root removal failed"
                )
            })
        }
    }
}

#[cfg(target_os = "windows")]
fn remove_registered_stale_windows_child_root(mountpoint: &Path) -> Result<()> {
    match fs::remove_dir_all(mountpoint) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(anyhow::anyhow!(
            "failed to remove registered stale Windows ProjFS root {}: {error}",
            mountpoint.display()
        )),
    }
}

#[cfg(target_os = "windows")]
fn windows_registered_child_mountpoint_is_owned(common_parent: &Path, mountpoint: &Path) -> bool {
    if windows_paths_equal(common_parent, mountpoint) {
        return false;
    }

    mountpoint
        .parent()
        .is_some_and(|parent| windows_paths_equal(parent, common_parent))
        && mountpoint.file_name().is_some()
}

#[cfg(target_os = "windows")]
fn windows_paths_equal(left: &Path, right: &Path) -> bool {
    left.to_string_lossy()
        .eq_ignore_ascii_case(&right.to_string_lossy())
}

fn create_unique_child_mountpoint(
    common_parent: &Path,
    source_path: &Path,
    records: &[DngMountRegistryRecord],
) -> Result<DngChildMountpoint> {
    let child = choose_unique_child_mountpoint(common_parent, source_path, records)?;
    fs::create_dir(&child.path).map_err(|error| {
        anyhow::anyhow!(
            "failed to create DNG child mountpoint {}: {error}",
            child.path.display()
        )
    })?;
    Ok(child)
}

// Candidates expand a stable identity suffix before numeric disambiguation,
// preserving readable names while avoiding case-insensitive collisions.
fn choose_unique_child_mountpoint(
    common_parent: &Path,
    source_path: &Path,
    records: &[DngMountRegistryRecord],
) -> Result<DngChildMountpoint> {
    let identity_input = MountClipIdentityInput::from_path(source_path).with_context(|| {
        format!(
            "failed to build DNG mount folder identity for {}",
            source_path.display()
        )
    })?;
    let candidates = CLIP_MOUNT_SUFFIX_EXPANSION_HEX_LENGTHS
        .into_iter()
        .map(|suffix_len| clip_mount_folder_name_with_suffix_len(&identity_input, suffix_len))
        .collect::<Vec<_>>();
    let Some(last_candidate) = candidates.last() else {
        bail!("DNG child mountpoint selection requires at least one folder-name candidate");
    };

    for candidate in &candidates {
        if !child_mountpoint_name_conflicts(common_parent, records, &candidate.visible)? {
            return dng_child_mountpoint(common_parent, &candidate.visible);
        }
    }

    let full_suffix = format_clip_mount_hash_suffix(last_candidate.identity, 32);
    for disambiguator in 2usize.. {
        let visible = format!(
            "{}__{}__{}",
            last_candidate.sanitized_stem, full_suffix, disambiguator
        );
        if !child_mountpoint_name_conflicts(common_parent, records, &visible)? {
            return dng_child_mountpoint(common_parent, &visible);
        }
    }

    bail!("DNG child mountpoint folder disambiguator exhausted")
}

fn dng_child_mountpoint(common_parent: &Path, folder_name: &str) -> Result<DngChildMountpoint> {
    validate_child_mountpoint_name(folder_name)?;
    Ok(DngChildMountpoint {
        folder_name: folder_name.to_string(),
        path: common_parent.join(folder_name),
    })
}

fn validate_child_mountpoint_name(folder_name: &str) -> Result<()> {
    let path = Path::new(folder_name);
    let mut components = path.components();
    let Some(std::path::Component::Normal(component)) = components.next() else {
        bail!("DNG child mountpoint folder name is not a normal path component");
    };
    if component.is_empty() || components.next().is_some() {
        bail!("DNG child mountpoint folder name must be one path component");
    }
    Ok(())
}

fn child_mountpoint_name_conflicts(
    common_parent: &Path,
    records: &[DngMountRegistryRecord],
    candidate: &str,
) -> Result<bool> {
    if common_parent.join(candidate).exists() {
        return Ok(true);
    }

    for entry in match fs::read_dir(common_parent) {
        Ok(read_dir) => read_dir,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(anyhow::anyhow!(
                "failed to read DNG common parent {}: {error}",
                common_parent.display()
            ));
        }
    } {
        let entry = entry.with_context(|| {
            format!(
                "failed to inspect child entry in DNG common parent {}",
                common_parent.display()
            )
        })?;
        if entry
            .file_name()
            .to_string_lossy()
            .eq_ignore_ascii_case(candidate)
        {
            return Ok(true);
        }
    }

    Ok(records.iter().any(|record| {
        record
            .mountpoint_path
            .parent()
            .is_some_and(|parent| parent == common_parent)
            && record.mountpoint_path.file_name().is_some_and(|file_name| {
                file_name.to_string_lossy().eq_ignore_ascii_case(candidate)
            })
    }))
}

fn remove_empty_dir_if_exists(path: &Path) -> Result<()> {
    match fs::remove_dir(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::DirectoryNotEmpty => Ok(()),
        Err(error) => Err(anyhow::anyhow!(
            "failed to remove empty directory {}: {error}",
            path.display()
        )),
    }
}

// Registry ownership is the authority for cleanup. Rows are removed only for a
// confirmed unmount or an already-absent mount; failures retain their records so
// a later cleanup attempt still knows what is owned.
fn unmount_registry_records(
    registry: &DngMountRegistry,
    adapter: &dyn DngUnmountAdapter,
    records: &[DngMountRegistryRecord],
) -> Result<DngUnmountRegistryOutcome> {
    let mut summary = DngUnmountSummary {
        attempted: 0,
        unmounted: 0,
        already_absent: 0,
    };
    let mut remove_ids = Vec::new();
    let mut errors = Vec::new();
    let groups = group_records_by_mountpoint(records);
    let candidate_record_count = groups.iter().map(|group| group.records.len()).sum();
    let mut completed_record_count = 0usize;
    let mut completed_mountpoints = Vec::new();

    for group in groups {
        summary.attempted += 1;
        let mountpoint_path = group.mountpoint_path;
        let records = group.records;
        let outcome = match adapter.unmount_owned_mountpoint(&mountpoint_path) {
            Ok(outcome) => outcome,
            Err(error) => {
                match adapter.cleanup_stale_shared_root_mountpoint(&mountpoint_path, &records) {
                    Ok(Some(outcome)) => outcome,
                    Ok(None) => {
                        errors.push(format!("{}: {error}", mountpoint_path.display()));
                        continue;
                    }
                    Err(cleanup_error) => {
                        errors.push(format!(
                            "{}: {error}; stale Windows ProjFS cleanup failed: {cleanup_error}",
                            mountpoint_path.display()
                        ));
                        continue;
                    }
                }
            }
        };

        match outcome {
            PlatformUnmountOutcome::Unmounted => {
                summary.unmounted += 1;
                completed_record_count += records.len();
                remove_ids.extend(records.iter().map(|record| record.mount_id.clone()));
                completed_mountpoints.push(mountpoint_path.clone());
            }
            PlatformUnmountOutcome::NotMounted => {
                summary.already_absent += 1;
                completed_record_count += records.len();
                remove_ids.extend(records.iter().map(|record| record.mount_id.clone()));
                completed_mountpoints.push(mountpoint_path.clone());
            }
            PlatformUnmountOutcome::Unsupported => {
                errors.push(format!(
                    "{}: platform unmount is unsupported",
                    mountpoint_path.display()
                ));
            }
            PlatformUnmountOutcome::Failed => {
                errors.push(format!(
                    "{}: platform unmount failed",
                    mountpoint_path.display()
                ));
            }
        }
    }

    registry.remove_mount_ids(&remove_ids)?;
    cleanup_completed_mountpoint_dirs(&completed_mountpoints);

    if !errors.is_empty() {
        bail!(
            "failed to unmount {} DNG mount(s): {}",
            errors.len(),
            errors.join("; ")
        );
    }

    Ok(DngUnmountRegistryOutcome {
        summary,
        candidate_record_count,
        completed_record_count,
    })
}

fn cleanup_completed_mountpoint_dirs(mountpoints: &[PathBuf]) {
    for mountpoint in mountpoints {
        if !is_dng_child_mountpoint_cleanup_candidate(mountpoint) {
            continue;
        }
        let _ = remove_empty_dir_if_exists(mountpoint);
        if let Some(parent) = mountpoint.parent() {
            let _ = remove_empty_dir_if_exists(parent);
        }
    }
}

fn is_dng_child_mountpoint_cleanup_candidate(mountpoint: &Path) -> bool {
    mountpoint.file_name().is_some()
        && mountpoint
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|name| name == OsStr::new("mcraw4vulkan"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MountpointRecordGroup {
    mountpoint_path: PathBuf,
    records: Vec<DngMountRegistryRecord>,
}

fn group_records_by_mountpoint(records: &[DngMountRegistryRecord]) -> Vec<MountpointRecordGroup> {
    let mut groups = Vec::<MountpointRecordGroup>::new();

    for record in records {
        if !record.is_mcraw4vulkan_owned_dng_mount() {
            continue;
        }
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.mountpoint_path == record.mountpoint_path)
        {
            group.records.push(record.clone());
        } else {
            groups.push(MountpointRecordGroup {
                mountpoint_path: record.mountpoint_path.clone(),
                records: vec![record.clone()],
            });
        }
    }

    groups
}

fn format_unmount_file_message(source_path: &Path, outcome: &DngUnmountRegistryOutcome) -> String {
    format!(
        "DNG unmount for {}: per-file mountpoint processed; mountpoints_attempted={}, clip_records_matched={}, clip_records_removed={}, unmounted={}, already_absent={}",
        source_path.display(),
        outcome.summary.attempted,
        outcome.candidate_record_count,
        outcome.completed_record_count,
        outcome.summary.unmounted,
        outcome.summary.already_absent
    )
}

fn format_unmount_all_message(outcome: &DngUnmountRegistryOutcome) -> String {
    format!(
        "DNG unmount all: per-file DNG mountpoint(s) processed; mountpoints_attempted={}, clip_records_matched={}, clip_records_removed={}, unmounted={}, already_absent={}",
        outcome.summary.attempted,
        outcome.candidate_record_count,
        outcome.completed_record_count,
        outcome.summary.unmounted,
        outcome.summary.already_absent
    )
}

fn dng_backend_from_app_policy(policy: DngAppBackendPolicy) -> DngGenerationBackend {
    match policy {
        DngAppBackendPolicy::Auto | DngAppBackendPolicy::Gpu => DngGenerationBackend::Gpu {
            backend_preference: GpuBackendPreference::VulkanOnly,
        },
        DngAppBackendPolicy::Cpu => DngGenerationBackend::Cpu,
    }
}

fn mountpoint_from_app_policy(
    source_path: &Path,
    policy: &DngMountPolicy,
    home: Option<&OsStr>,
) -> Result<PathBuf> {
    match policy {
        DngMountPolicy::AutoTemp => {
            let _ = source_path;
            default_dng_shared_mount_root_from_home_env(home)
        }
        DngMountPolicy::Explicit(path) => Ok(path.clone()),
        DngMountPolicy::UnsupportedOnThisPlatform => {
            bail!("DNG/FUSE mount is not implemented on this platform")
        }
    }
}

fn ensure_default_generation_policy(policy: DngGenerationExecutionPolicy) -> Result<()> {
    match policy {
        DngGenerationExecutionPolicy::Inflight2Default => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::io::Write;

    #[derive(Debug)]
    struct MockUnmountAdapter {
        status: PlatformUnmountOutcome,
        paths: RefCell<Vec<PathBuf>>,
    }

    impl MockUnmountAdapter {
        fn new(status: PlatformUnmountOutcome) -> Self {
            Self {
                status,
                paths: RefCell::new(Vec::new()),
            }
        }
    }

    impl DngUnmountAdapter for MockUnmountAdapter {
        fn unmount_owned_mountpoint(&self, mountpoint: &Path) -> Result<PlatformUnmountOutcome> {
            self.paths.borrow_mut().push(mountpoint.to_path_buf());
            Ok(self.status)
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum StaleCleanupResult {
        Outcome(PlatformUnmountOutcome),
        Error(&'static str),
    }

    #[derive(Debug)]
    struct StaleCleanupUnmountAdapter {
        cleanup_result: StaleCleanupResult,
        paths: RefCell<Vec<PathBuf>>,
        cleanup_paths: RefCell<Vec<PathBuf>>,
        cleanup_record_counts: RefCell<Vec<usize>>,
    }

    impl StaleCleanupUnmountAdapter {
        fn new(cleanup_result: StaleCleanupResult) -> Self {
            Self {
                cleanup_result,
                paths: RefCell::new(Vec::new()),
                cleanup_paths: RefCell::new(Vec::new()),
                cleanup_record_counts: RefCell::new(Vec::new()),
            }
        }
    }

    impl DngUnmountAdapter for StaleCleanupUnmountAdapter {
        fn unmount_owned_mountpoint(&self, mountpoint: &Path) -> Result<PlatformUnmountOutcome> {
            self.paths.borrow_mut().push(mountpoint.to_path_buf());
            Err(anyhow::anyhow!("foreground owner did not stop"))
        }

        fn cleanup_stale_shared_root_mountpoint(
            &self,
            mountpoint: &Path,
            records: &[DngMountRegistryRecord],
        ) -> Result<Option<PlatformUnmountOutcome>> {
            self.cleanup_paths
                .borrow_mut()
                .push(mountpoint.to_path_buf());
            self.cleanup_record_counts.borrow_mut().push(records.len());
            match self.cleanup_result {
                StaleCleanupResult::Outcome(outcome) => Ok(Some(outcome)),
                StaleCleanupResult::Error(message) => Err(anyhow::anyhow!(message)),
            }
        }
    }

    fn unique_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mcraw4vulkan-dng-mount-test-{}-{name}",
            std::process::id()
        ))
    }

    fn create_source_file(root: &Path, name: &str) -> PathBuf {
        fs::create_dir_all(root).unwrap();
        let path = root.join(name);
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(b"test").unwrap();
        path
    }

    fn fake_home(root: &Path) -> OsString {
        root.join("home").into_os_string()
    }

    fn expected_current_default_mount_root(home: &OsStr) -> PathBuf {
        default_dng_shared_mount_root_from_home_env(Some(home)).unwrap()
    }

    #[test]
    fn dng_mount_config_uses_production_defaults() {
        let root = unique_dir("config-defaults");
        let source = create_source_file(&root, "clip.mcraw");
        let policy = DngAppPolicy::production_default();

        let config = DngMountRunConfig::from_app_policy_inputs_with_home(
            std::slice::from_ref(&source),
            &policy,
            Some(fake_home(&root)),
        )
        .unwrap();

        assert_eq!(
            config.backend,
            dng_backend_from_app_policy(DngAppBackendPolicy::Gpu)
        );
        assert_eq!(config.vignette_mode, DngSinkVignetteMode::LumaPlane0);
        assert_eq!(
            config.cache_frame_capacity,
            DEFAULT_DNG_CACHE_FRAME_CAPACITY
        );
        assert_eq!(
            config.prefetch_forward_frames,
            DEFAULT_PREFETCH_FORWARD_FRAMES
        );
        assert_eq!(config.worker_threads, 4);
        assert_eq!(
            config.generation_execution_policy,
            DngGenerationExecutionPolicy::Inflight2Default
        );
        assert_eq!(config.input_paths, vec![fs::canonicalize(&source).unwrap()]);
        assert_eq!(
            config.mountpoint_path,
            expected_current_default_mount_root(&fake_home(&root))
        );
        let _ = fs::remove_file(source);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn dng_mount_config_rejects_multiple_inputs() {
        let root = unique_dir("config-multiple");
        let first = create_source_file(&root, "first.mcraw");
        let second = create_source_file(&root, "second.mcraw");
        let policy = DngAppPolicy::production_default();

        let error = DngMountRunConfig::from_app_policy_inputs_with_home(
            &[first.clone(), second.clone()],
            &policy,
            Some(fake_home(&root)),
        )
        .expect_err("multiple dng mount inputs should reject")
        .to_string();

        assert!(error.contains("one FILE per mount process"));
        let _ = fs::remove_file(first);
        let _ = fs::remove_file(second);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn dng_mount_config_rejects_empty_input_list() {
        let policy = DngAppPolicy::production_default();

        let error = DngMountRunConfig::from_app_policy_inputs(&[], &policy)
            .expect_err("empty input list should reject")
            .to_string();

        assert!(error.contains("at least one input"));
    }

    #[test]
    fn dng_mount_config_maps_cpu_and_no_vignette() {
        let root = unique_dir("config-cpu");
        let source = create_source_file(&root, "clip.mcraw");
        let policy = DngAppPolicy {
            backend: DngAppBackendPolicy::Cpu,
            vignette: DngSinkVignetteMode::None,
            mount: DngMountPolicy::Explicit(root.join("explicit-mount")),
        };

        let config =
            DngMountRunConfig::from_app_policy_inputs(std::slice::from_ref(&source), &policy)
                .unwrap();

        assert_eq!(config.backend, DngGenerationBackend::Cpu);
        assert_eq!(config.vignette_mode, DngSinkVignetteMode::None);
        assert_eq!(config.mountpoint_path, root.join("explicit-mount"));
        let _ = fs::remove_file(source);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn linux_default_dng_shared_mount_root_uses_home_mcraw4vulkan() {
        let home = Path::new("fake-home");
        assert_eq!(
            default_dng_shared_mount_root_for_platform(
                DefaultDngSharedMountRootPlatform::Linux,
                Some(home.as_os_str())
            )
            .unwrap(),
            PathBuf::from("fake-home/mcraw4vulkan")
        );
    }

    #[test]
    fn windows_default_dng_shared_mount_root_uses_userprofile_mcraw4vulkan() {
        let userprofile = Path::new("fake-profile");
        assert_eq!(
            default_dng_shared_mount_root_for_platform(
                DefaultDngSharedMountRootPlatform::Windows,
                Some(userprofile.as_os_str())
            )
            .unwrap(),
            PathBuf::from("fake-profile/mcraw4vulkan")
        );
    }

    #[test]
    fn macos_default_dng_shared_mount_root_uses_home_mcraw4vulkan() {
        let home = Path::new("fake-home");
        let mountpoint = default_dng_shared_mount_root_for_platform(
            DefaultDngSharedMountRootPlatform::Macos,
            Some(home.as_os_str()),
        )
        .unwrap();

        assert_eq!(mountpoint, PathBuf::from("fake-home/mcraw4vulkan"));
    }

    #[test]
    fn macos_default_dng_shared_mount_root_is_not_volumes_or_temp_dir() {
        let home = Path::new("fake-home");
        let mountpoint = default_dng_shared_mount_root_for_platform(
            DefaultDngSharedMountRootPlatform::Macos,
            Some(home.as_os_str()),
        )
        .unwrap();
        let old_volumes_root = PathBuf::from(std::path::MAIN_SEPARATOR.to_string())
            .join("Volumes")
            .join("mcraw4vulkan");

        assert_ne!(mountpoint, old_volumes_root);
        assert_ne!(mountpoint, std::env::temp_dir().join("mcraw4vulkan"));
    }

    #[test]
    fn macos_default_dng_shared_mount_root_rejects_missing_or_empty_home() {
        let missing = default_dng_shared_mount_root_for_platform(
            DefaultDngSharedMountRootPlatform::Macos,
            None,
        )
        .expect_err("missing HOME should reject")
        .to_string();
        assert!(missing.contains("HOME"));

        let empty = default_dng_shared_mount_root_for_platform(
            DefaultDngSharedMountRootPlatform::Macos,
            Some(OsStr::new("")),
        )
        .expect_err("empty HOME should reject")
        .to_string();
        assert!(empty.contains("HOME"));
    }

    #[test]
    fn linux_default_dng_shared_mount_root_rejects_missing_or_empty_home() {
        let missing = default_dng_shared_mount_root_for_platform(
            DefaultDngSharedMountRootPlatform::Linux,
            None,
        )
        .expect_err("missing HOME should reject")
        .to_string();
        assert!(missing.contains("HOME"));

        let empty = default_dng_shared_mount_root_for_platform(
            DefaultDngSharedMountRootPlatform::Linux,
            Some(OsStr::new("")),
        )
        .expect_err("empty HOME should reject")
        .to_string();
        assert!(empty.contains("HOME"));
    }

    #[test]
    fn windows_default_dng_shared_mount_root_rejects_missing_or_empty_userprofile() {
        let missing = default_dng_shared_mount_root_for_platform(
            DefaultDngSharedMountRootPlatform::Windows,
            None,
        )
        .expect_err("missing USERPROFILE should reject")
        .to_string();
        assert!(missing.contains("USERPROFILE"));

        let empty = default_dng_shared_mount_root_for_platform(
            DefaultDngSharedMountRootPlatform::Windows,
            Some(OsStr::new("")),
        )
        .expect_err("empty USERPROFILE should reject")
        .to_string();
        assert!(empty.contains("USERPROFILE"));
    }

    #[test]
    fn default_mountpoint_for_source_no_longer_uses_input_parent_and_stem() {
        let source = Path::new("tmp/260518_211252_VIDEO_25mm.mcraw");
        let mountpoint = mountpoint_from_app_policy(
            source,
            &DngMountPolicy::AutoTemp,
            Some(OsStr::new("fake-home")),
        )
        .unwrap();

        assert_eq!(
            mountpoint,
            default_dng_shared_mount_root_from_home_env(Some(OsStr::new("fake-home"))).unwrap()
        );
        assert_ne!(mountpoint, PathBuf::from("tmp/260518_211252_VIDEO_25mm"));
    }

    #[test]
    fn mount_start_message_uses_cross_platform_foreground_guidance() {
        let child = DngChildMountpoint {
            folder_name: "clip__1234567890".to_string(),
            path: PathBuf::from("home/mcraw4vulkan/clip__1234567890"),
        };
        let message = format_mount_start_message(
            Path::new("clip.mcraw"),
            Path::new("home/mcraw4vulkan"),
            &child,
        );

        for expected in [
            "serving in foreground",
            "common parent",
            "clip folder",
            "Ctrl-C",
            "mcraw4vulkan dng unmount FILE.mcraw",
            "mcraw4vulkan dng unmount all",
        ] {
            assert!(message.contains(expected), "message missing {expected}");
        }

        for forbidden in [
            "Linux",
            "linux",
            "macOS",
            "MacOS",
            "Windows",
            "windows",
            "FUSE",
            "fuse",
            "ProjFS",
            "macFUSE",
            "WinFsp",
            "XDG",
            "XDG_VIDEOS_DIR",
            "Videos",
            "Movies",
            "USERPROFILE",
        ] {
            assert!(
                !message.contains(forbidden),
                "mount-start message contains {forbidden}"
            );
        }
    }

    #[test]
    fn virtual_names_match_dng_mount_contract() {
        let (folder, first_dng, audio) = dng_virtual_names_for_stem("260518_211252_VIDEO_25mm");
        assert_eq!(folder, PathBuf::from("260518_211252_VIDEO_25mm"));
        assert_eq!(
            first_dng,
            PathBuf::from("260518_211252_VIDEO_25mm").join("260518_211252_VIDEO_25mm_000000.dng")
        );
        assert_eq!(
            audio,
            PathBuf::from("260518_211252_VIDEO_25mm").join("260518_211252_VIDEO_25mm.wav")
        );
    }

    #[test]
    fn ensure_mountpoint_accepts_empty_directory() {
        let root = unique_dir("empty-mountpoint");
        let mountpoint = root.join("clip");
        fs::create_dir_all(&mountpoint).unwrap();

        ensure_mountpoint_available(&mountpoint).unwrap();

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn ensure_mountpoint_rejects_non_empty_directory() {
        let root = unique_dir("non-empty-mountpoint");
        let mountpoint = root.join("clip");
        fs::create_dir_all(&mountpoint).unwrap();
        fs::write(mountpoint.join("existing.txt"), b"existing").unwrap();

        let error = ensure_mountpoint_available(&mountpoint)
            .expect_err("non-empty mountpoint should reject")
            .to_string();

        assert!(error.contains("not empty"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn ensure_mountpoint_create_error_includes_underlying_cause() {
        let root = unique_dir("mountpoint-create-error");
        fs::create_dir_all(&root).unwrap();
        let parent_file = root.join("not-a-directory");
        fs::write(&parent_file, b"existing").unwrap();
        let mountpoint = parent_file.join("clip");

        let error = ensure_mountpoint_available(&mountpoint)
            .expect_err("mountpoint creation below a file should reject")
            .to_string();
        let context_only = format!("failed to create mountpoint {}", mountpoint.display());

        assert!(
            error.starts_with(&format!("{context_only}: ")),
            "error should include source cause in top-level display: {error}"
        );
        assert_ne!(error, context_only);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn common_parent_accepts_existing_ordinary_directory() {
        let root = unique_dir("ordinary-common-parent");
        let common_parent = root.join("home").join("mcraw4vulkan");
        fs::create_dir_all(&common_parent).unwrap();
        fs::write(common_parent.join("notes.txt"), b"user data").unwrap();

        ensure_common_parent_available(&common_parent, &[]).unwrap();

        assert!(common_parent.join("notes.txt").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn common_parent_rejects_old_shared_root_record() {
        let root = unique_dir("old-shared-root-record");
        let source = create_source_file(&root, "clip.mcraw");
        let common_parent = root.join("home").join("mcraw4vulkan");
        let record = DngMountRegistryRecord::new(
            fs::canonicalize(&source).unwrap(),
            common_parent.clone(),
            1,
        );

        let error = ensure_common_parent_available(&common_parent, &[record])
            .expect_err("old shared-root record should reject")
            .to_string();

        assert!(error.contains("old shared-root mountpoint"));
        assert!(error.contains("dng unmount all"));
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_common_parent_rejects_active_fuse_mount_classification() {
        let common_parent = macos_fixture_common_parent();
        let classification = MacosMountpointClassification::Mounted(MacosMountInfo {
            source: "mcraw4vulkan:old-shared-root".to_string(),
            mount_point: common_parent.clone(),
            filesystem_type: "macfuse".to_string(),
            raw_line: format!(
                "mcraw4vulkan:old-shared-root on {} (macfuse, nodev)",
                common_parent.display()
            ),
        });

        let error = ensure_macos_common_parent_mount_classification(&common_parent, classification)
            .expect_err("active common-parent macFUSE mount should reject")
            .to_string();

        assert!(error.contains("active macFUSE/FUSE mount"));
        assert!(error.contains("stale state from an older mcraw4vulkan build"));
        assert!(error.contains("mcraw4vulkan dng unmount all"));
        assert!(error.contains("Finder or diskutil"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_common_parent_accepts_not_mounted_classification() {
        let common_parent = macos_fixture_common_parent();

        ensure_macos_common_parent_mount_classification(
            &common_parent,
            MacosMountpointClassification::NotMounted,
        )
        .unwrap();
    }

    #[cfg(target_os = "macos")]
    fn macos_fixture_common_parent() -> PathBuf {
        let mut path = PathBuf::from(std::path::MAIN_SEPARATOR.to_string());
        for part in ["Users", "example", "mcraw4vulkan"] {
            path.push(part);
        }
        path
    }

    #[test]
    fn child_mountpoint_uses_core_mount_folder_naming() {
        let root = unique_dir("child-mountpoint-core-name");
        let source = create_source_file(&root, "clip take.mcraw");
        let common_parent = root.join("home").join("mcraw4vulkan");
        fs::create_dir_all(&common_parent).unwrap();

        let child = choose_unique_child_mountpoint(&common_parent, &source, &[]).unwrap();

        assert_eq!(child.path, common_parent.join(&child.folder_name));
        assert!(child.folder_name.starts_with("clip take__"));
        assert!(!child.folder_name.contains('/'));
        assert!(!child.folder_name.contains('\\'));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn child_mountpoint_collision_uses_suffix_expansion() {
        let root = unique_dir("child-mountpoint-collision");
        let source = create_source_file(&root, "clip.mcraw");
        let common_parent = root.join("home").join("mcraw4vulkan");
        fs::create_dir_all(&common_parent).unwrap();
        let first = choose_unique_child_mountpoint(&common_parent, &source, &[]).unwrap();
        fs::create_dir(&first.path).unwrap();

        let second = choose_unique_child_mountpoint(&common_parent, &source, &[]).unwrap();

        assert_ne!(first.folder_name, second.folder_name);
        assert_eq!(first.folder_name.rsplit_once("__").unwrap().1.len(), 10);
        assert_eq!(second.folder_name.rsplit_once("__").unwrap().1.len(), 12);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn duplicate_source_record_creates_distinct_child_mountpoint() {
        let root = unique_dir("duplicate-source-child-mountpoint");
        let source = create_source_file(&root, "clip.mcraw");
        let common_parent = root.join("home").join("mcraw4vulkan");
        fs::create_dir_all(&common_parent).unwrap();
        let first = choose_unique_child_mountpoint(&common_parent, &source, &[]).unwrap();
        let record =
            DngMountRegistryRecord::new(fs::canonicalize(&source).unwrap(), first.path.clone(), 1);

        let second = choose_unique_child_mountpoint(&common_parent, &source, &[record]).unwrap();

        assert_ne!(first.path, second.path);
        assert!(second.path.starts_with(&common_parent));
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_stale_dead_pid_registry_row_is_pruned_before_mount() {
        let root = unique_dir("windows-stale-dead-pid-prune");
        let source = create_source_file(&root, "clip.mcraw");
        let source_canonical = canonical_source_path(&source).unwrap();
        let common_parent = root.join("home").join("mcraw4vulkan");
        let child = common_parent.join("clip__stale");
        fs::create_dir_all(&child).unwrap();
        fs::write(child.join("stale-placeholder.dng"), b"stale").unwrap();
        let registry = DngMountRegistry::new(root.join("registry.tsv"));
        let mut record = DngMountRegistryRecord::new(source_canonical.clone(), child.clone(), 1);
        record.process_id = 0;
        registry.write_records(&[record]).unwrap();

        let kept = prune_stale_windows_mount_records(
            &registry,
            &common_parent,
            &source_canonical,
            registry.all_owned_records().unwrap(),
        )
        .unwrap();

        assert!(kept.is_empty());
        assert!(registry.all_owned_records().unwrap().is_empty());
        assert!(!child.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_stale_cleanup_refuses_paths_outside_common_parent() {
        let root = unique_dir("windows-stale-outside-parent");
        let source = create_source_file(&root, "clip.mcraw");
        let source_canonical = canonical_source_path(&source).unwrap();
        let common_parent = root.join("home").join("mcraw4vulkan");
        let outside = root.join("outside-root");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("user-file.txt"), b"user").unwrap();
        let registry = DngMountRegistry::new(root.join("registry.tsv"));
        let mut record = DngMountRegistryRecord::new(source_canonical.clone(), outside.clone(), 1);
        record.process_id = 0;
        registry.write_records(&[record.clone()]).unwrap();

        let kept = prune_stale_windows_mount_records(
            &registry,
            &common_parent,
            &source_canonical,
            registry.all_owned_records().unwrap(),
        )
        .unwrap();

        assert_eq!(kept, vec![record.clone()]);
        assert_eq!(registry.all_owned_records().unwrap(), vec![record]);
        assert!(outside.join("user-file.txt").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_live_provider_pid_registry_row_is_preserved() {
        let root = unique_dir("windows-live-pid-preserved");
        let source = create_source_file(&root, "clip.mcraw");
        let source_canonical = canonical_source_path(&source).unwrap();
        let common_parent = root.join("home").join("mcraw4vulkan");
        let child = common_parent.join("clip__live");
        fs::create_dir_all(&child).unwrap();
        let registry = DngMountRegistry::new(root.join("registry.tsv"));
        let record = DngMountRegistryRecord::new(source_canonical.clone(), child.clone(), 1);
        registry
            .write_records(std::slice::from_ref(&record))
            .unwrap();

        let kept = prune_stale_windows_mount_records(
            &registry,
            &common_parent,
            &source_canonical,
            registry.all_owned_records().unwrap(),
        )
        .unwrap();

        assert_eq!(kept, vec![record.clone()]);
        assert_eq!(registry.all_owned_records().unwrap(), vec![record]);
        assert!(child.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unmount_file_uses_only_matching_registry_records() {
        let root = unique_dir("unmount-file");
        let source = create_source_file(&root, "clip.mcraw");
        let source_canonical = canonical_source_path(&source).unwrap();
        let registry_path = root.join("registry.tsv");
        let registry = DngMountRegistry::new(registry_path);
        let record = DngMountRegistryRecord::new(source_canonical.clone(), root.join("clip"), 1);
        let other = DngMountRegistryRecord::new(root.join("other.mcraw"), root.join("other"), 2);
        registry
            .write_records(&[record.clone(), other.clone()])
            .unwrap();
        let adapter = MockUnmountAdapter::new(PlatformUnmountOutcome::Unmounted);

        let summary = run_dng_unmount_file_with_adapter(&source, &registry, &adapter).unwrap();

        assert_eq!(
            adapter.paths.borrow().as_slice(),
            std::slice::from_ref(&record.mountpoint_path)
        );
        assert_eq!(
            summary,
            DngUnmountSummary {
                attempted: 1,
                unmounted: 1,
                already_absent: 0
            }
        );
        assert!(
            registry
                .records_for_source(&source_canonical)
                .unwrap()
                .is_empty()
        );
        assert_eq!(registry.all_owned_records().unwrap(), vec![other]);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unmount_file_does_not_remove_sibling_rows_at_same_mountpoint() {
        let root = unique_dir("unmount-file-same-mountpoint-sibling");
        let source = create_source_file(&root, "first.mcraw");
        let sibling_source = create_source_file(&root, "second.mcraw");
        let source_canonical = canonical_source_path(&source).unwrap();
        let sibling_canonical = canonical_source_path(&sibling_source).unwrap();
        let registry_path = root.join("registry.tsv");
        let registry = DngMountRegistry::new(registry_path);
        let shared_mountpoint = root.join("shared");
        let requested = DngMountRegistryRecord::new(source_canonical, shared_mountpoint.clone(), 1);
        let sibling = DngMountRegistryRecord::new(sibling_canonical, shared_mountpoint.clone(), 2);
        let other = DngMountRegistryRecord::new(root.join("other.mcraw"), root.join("other"), 3);
        registry
            .write_records(&[requested, sibling.clone(), other.clone()])
            .unwrap();
        let adapter = MockUnmountAdapter::new(PlatformUnmountOutcome::Unmounted);

        let summary = run_dng_unmount_file_with_adapter(&source, &registry, &adapter).unwrap();

        assert_eq!(adapter.paths.borrow().as_slice(), &[shared_mountpoint]);
        assert_eq!(
            summary,
            DngUnmountSummary {
                attempted: 1,
                unmounted: 1,
                already_absent: 0
            }
        );
        assert_eq!(registry.all_owned_records().unwrap(), vec![sibling, other]);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unmount_file_reports_duplicate_source_ambiguity() {
        let root = unique_dir("unmount-file-multiple-mountpoints");
        let source = create_source_file(&root, "clip.mcraw");
        let source_canonical = canonical_source_path(&source).unwrap();
        let registry_path = root.join("registry.tsv");
        let registry = DngMountRegistry::new(registry_path);
        let first_mountpoint = root.join("shared-a");
        let second_mountpoint = root.join("shared-b");
        let first =
            DngMountRegistryRecord::new(source_canonical.clone(), first_mountpoint.clone(), 1);
        let second = DngMountRegistryRecord::new(source_canonical, second_mountpoint.clone(), 2);
        registry
            .write_records(&[first.clone(), second.clone()])
            .unwrap();
        let adapter = MockUnmountAdapter::new(PlatformUnmountOutcome::Unmounted);

        let error = run_dng_unmount_file_with_adapter(&source, &registry, &adapter)
            .expect_err("duplicate source records should be ambiguous")
            .to_string();

        assert!(error.contains("multiple DNG mounts exist"));
        assert!(error.contains("dng unmount all"));
        assert!(adapter.paths.borrow().is_empty());
        assert_eq!(registry.all_owned_records().unwrap(), vec![first, second]);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unmount_file_not_mounted_source_stays_clear() {
        let root = unique_dir("unmount-file-not-mounted");
        let source = create_source_file(&root, "clip.mcraw");
        let registry_path = root.join("registry.tsv");
        let registry = DngMountRegistry::new(registry_path);
        let adapter = MockUnmountAdapter::new(PlatformUnmountOutcome::Unmounted);

        let error = run_dng_unmount_file_with_adapter(&source, &registry, &adapter)
            .expect_err("missing source record should reject")
            .to_string();

        assert!(error.contains("no mcraw4vulkan-owned DNG mount found"));
        assert!(adapter.paths.borrow().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn failed_unmount_file_keeps_shared_mountpoint_rows() {
        let root = unique_dir("unmount-file-failed");
        let source = create_source_file(&root, "first.mcraw");
        let sibling_source = create_source_file(&root, "second.mcraw");
        let source_canonical = canonical_source_path(&source).unwrap();
        let sibling_canonical = canonical_source_path(&sibling_source).unwrap();
        let registry_path = root.join("registry.tsv");
        let registry = DngMountRegistry::new(registry_path);
        let shared_mountpoint = root.join("shared");
        let requested = DngMountRegistryRecord::new(source_canonical, shared_mountpoint.clone(), 1);
        let sibling = DngMountRegistryRecord::new(sibling_canonical, shared_mountpoint.clone(), 2);
        registry
            .write_records(&[requested.clone(), sibling.clone()])
            .unwrap();
        let adapter = MockUnmountAdapter::new(PlatformUnmountOutcome::Failed);

        let error = run_dng_unmount_file_with_adapter(&source, &registry, &adapter)
            .expect_err("failed platform unmount should reject")
            .to_string();

        assert!(error.contains("failed to unmount"));
        assert_eq!(adapter.paths.borrow().as_slice(), &[shared_mountpoint]);
        assert_eq!(
            registry.all_owned_records().unwrap(),
            vec![requested, sibling]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn stale_cleanup_success_after_unmount_error_removes_shared_mountpoint_rows() {
        let root = unique_dir("unmount-file-stale-cleanup");
        let source = create_source_file(&root, "first.mcraw");
        let sibling_source = create_source_file(&root, "second.mcraw");
        let source_canonical = canonical_source_path(&source).unwrap();
        let sibling_canonical = canonical_source_path(&sibling_source).unwrap();
        let registry_path = root.join("registry.tsv");
        let registry = DngMountRegistry::new(registry_path);
        let shared_mountpoint = root.join("shared");
        let requested = DngMountRegistryRecord::new(source_canonical, shared_mountpoint.clone(), 1);
        let sibling = DngMountRegistryRecord::new(sibling_canonical, shared_mountpoint.clone(), 2);
        registry
            .write_records(&[requested, sibling.clone()])
            .unwrap();
        let adapter = StaleCleanupUnmountAdapter::new(StaleCleanupResult::Outcome(
            PlatformUnmountOutcome::Unmounted,
        ));

        let summary = run_dng_unmount_file_with_adapter(&source, &registry, &adapter).unwrap();

        assert_eq!(
            adapter.paths.borrow().as_slice(),
            std::slice::from_ref(&shared_mountpoint)
        );
        assert_eq!(
            adapter.cleanup_paths.borrow().as_slice(),
            &[shared_mountpoint]
        );
        assert_eq!(adapter.cleanup_record_counts.borrow().as_slice(), &[1]);
        assert_eq!(
            summary,
            DngUnmountSummary {
                attempted: 1,
                unmounted: 1,
                already_absent: 0
            }
        );
        assert_eq!(registry.all_owned_records().unwrap(), vec![sibling]);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn stale_cleanup_failure_preserves_shared_mountpoint_rows() {
        let root = unique_dir("unmount-file-stale-cleanup-failed");
        let source = create_source_file(&root, "first.mcraw");
        let sibling_source = create_source_file(&root, "second.mcraw");
        let source_canonical = canonical_source_path(&source).unwrap();
        let sibling_canonical = canonical_source_path(&sibling_source).unwrap();
        let registry_path = root.join("registry.tsv");
        let registry = DngMountRegistry::new(registry_path);
        let shared_mountpoint = root.join("shared");
        let requested = DngMountRegistryRecord::new(source_canonical, shared_mountpoint.clone(), 1);
        let sibling = DngMountRegistryRecord::new(sibling_canonical, shared_mountpoint.clone(), 2);
        registry
            .write_records(&[requested.clone(), sibling.clone()])
            .unwrap();
        let adapter =
            StaleCleanupUnmountAdapter::new(StaleCleanupResult::Error("unknown user file"));

        let error = run_dng_unmount_file_with_adapter(&source, &registry, &adapter)
            .expect_err("failed stale cleanup should reject")
            .to_string();

        assert!(error.contains("failed to unmount"));
        assert!(error.contains("stale Windows ProjFS cleanup failed"));
        assert_eq!(
            adapter.paths.borrow().as_slice(),
            std::slice::from_ref(&shared_mountpoint)
        );
        assert_eq!(
            adapter.cleanup_paths.borrow().as_slice(),
            &[shared_mountpoint]
        );
        assert_eq!(adapter.cleanup_record_counts.borrow().as_slice(), &[1]);
        assert_eq!(
            registry.all_owned_records().unwrap(),
            vec![requested, sibling]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unmount_all_touches_only_owned_registry_records() {
        let root = unique_dir("unmount-all");
        fs::create_dir_all(&root).unwrap();
        let registry_path = root.join("registry.tsv");
        let registry = DngMountRegistry::new(registry_path);
        let owned = DngMountRegistryRecord::new(root.join("clip.mcraw"), root.join("clip"), 1);
        let mut unrelated =
            DngMountRegistryRecord::new(root.join("other.mcraw"), root.join("other"), 2);
        unrelated.owned_by = "other-app".to_string();
        registry.write_records(&[owned.clone(), unrelated]).unwrap();
        let adapter = MockUnmountAdapter::new(PlatformUnmountOutcome::NotMounted);

        let summary = run_dng_unmount_all_with_adapter(&registry, &adapter).unwrap();

        assert_eq!(
            adapter.paths.borrow().as_slice(),
            std::slice::from_ref(&owned.mountpoint_path)
        );
        assert_eq!(
            summary,
            DngUnmountSummary {
                attempted: 1,
                unmounted: 0,
                already_absent: 1
            }
        );
        assert!(registry.all_owned_records().unwrap().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unmount_all_touches_each_mountpoint_once() {
        let root = unique_dir("unmount-all-multiple-mountpoints");
        fs::create_dir_all(&root).unwrap();
        let registry_path = root.join("registry.tsv");
        let registry = DngMountRegistry::new(registry_path);
        let shared_mountpoint = root.join("shared");
        let other_mountpoint = root.join("other");
        let first =
            DngMountRegistryRecord::new(root.join("first.mcraw"), shared_mountpoint.clone(), 1);
        let second =
            DngMountRegistryRecord::new(root.join("second.mcraw"), shared_mountpoint.clone(), 2);
        let third =
            DngMountRegistryRecord::new(root.join("third.mcraw"), other_mountpoint.clone(), 3);
        registry.write_records(&[first, second, third]).unwrap();
        let adapter = MockUnmountAdapter::new(PlatformUnmountOutcome::Unmounted);

        let summary = run_dng_unmount_all_with_adapter(&registry, &adapter).unwrap();

        assert_eq!(
            adapter.paths.borrow().as_slice(),
            &[shared_mountpoint, other_mountpoint]
        );
        assert_eq!(
            summary,
            DngUnmountSummary {
                attempted: 2,
                unmounted: 2,
                already_absent: 0
            }
        );
        assert!(registry.all_owned_records().unwrap().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unmount_all_cleans_stale_child_row_and_empty_child_directory() {
        let root = unique_dir("unmount-all-stale-empty-child");
        let common_parent = root.join("home").join("mcraw4vulkan");
        let child = common_parent.join("clip__1234567890");
        fs::create_dir_all(&child).unwrap();
        let registry = DngMountRegistry::new(root.join("registry.tsv"));
        let record = DngMountRegistryRecord::new(root.join("clip.mcraw"), child.clone(), 1);
        registry.write_records(&[record]).unwrap();
        let adapter = MockUnmountAdapter::new(PlatformUnmountOutcome::NotMounted);

        let summary = run_dng_unmount_all_with_adapter(&registry, &adapter).unwrap();

        assert_eq!(
            summary,
            DngUnmountSummary {
                attempted: 1,
                unmounted: 0,
                already_absent: 1
            }
        );
        assert!(registry.all_owned_records().unwrap().is_empty());
        assert!(!child.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unmount_all_does_not_remove_outside_registered_directory() {
        let root = unique_dir("unmount-all-outside-dir-preserved");
        let outside = root.join("outside-root");
        fs::create_dir_all(&outside).unwrap();
        let registry = DngMountRegistry::new(root.join("registry.tsv"));
        let record = DngMountRegistryRecord::new(root.join("clip.mcraw"), outside.clone(), 1);
        registry.write_records(&[record]).unwrap();
        let adapter = MockUnmountAdapter::new(PlatformUnmountOutcome::NotMounted);

        let summary = run_dng_unmount_all_with_adapter(&registry, &adapter).unwrap();

        assert_eq!(
            summary,
            DngUnmountSummary {
                attempted: 1,
                unmounted: 0,
                already_absent: 1
            }
        );
        assert!(registry.all_owned_records().unwrap().is_empty());
        assert!(outside.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unmount_all_does_not_remove_non_empty_child_directory() {
        let root = unique_dir("unmount-all-non-empty-child-preserved");
        let common_parent = root.join("home").join("mcraw4vulkan");
        let child = common_parent.join("clip__1234567890");
        fs::create_dir_all(&child).unwrap();
        fs::write(child.join("user-file.txt"), b"user").unwrap();
        let registry = DngMountRegistry::new(root.join("registry.tsv"));
        let record = DngMountRegistryRecord::new(root.join("clip.mcraw"), child.clone(), 1);
        registry.write_records(&[record]).unwrap();
        let adapter = MockUnmountAdapter::new(PlatformUnmountOutcome::NotMounted);

        let summary = run_dng_unmount_all_with_adapter(&registry, &adapter).unwrap();

        assert_eq!(
            summary,
            DngUnmountSummary {
                attempted: 1,
                unmounted: 0,
                already_absent: 1
            }
        );
        assert!(registry.all_owned_records().unwrap().is_empty());
        assert!(child.join("user-file.txt").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unmount_file_message_names_shared_root_behavior() {
        let outcome = DngUnmountRegistryOutcome {
            summary: DngUnmountSummary {
                attempted: 1,
                unmounted: 1,
                already_absent: 0,
            },
            candidate_record_count: 2,
            completed_record_count: 2,
        };

        let message = format_unmount_file_message(Path::new("clip.mcraw"), &outcome);

        assert!(message.contains("per-file mountpoint"));
        assert!(message.contains("clip_records_matched=2"));
        assert!(message.contains("clip_records_removed=2"));
    }

    #[test]
    fn unmount_all_touches_shared_mountpoint_once() {
        let root = unique_dir("unmount-all-shared-root");
        fs::create_dir_all(&root).unwrap();
        let registry_path = root.join("registry.tsv");
        let registry = DngMountRegistry::new(registry_path);
        let mountpoint = root.join("shared");
        let first = DngMountRegistryRecord::new(root.join("first.mcraw"), mountpoint.clone(), 1);
        let second = DngMountRegistryRecord::new(root.join("second.mcraw"), mountpoint.clone(), 2);
        registry.write_records(&[first, second]).unwrap();
        let adapter = MockUnmountAdapter::new(PlatformUnmountOutcome::Unmounted);

        let summary = run_dng_unmount_all_with_adapter(&registry, &adapter).unwrap();

        assert_eq!(adapter.paths.borrow().as_slice(), &[mountpoint]);
        assert_eq!(
            summary,
            DngUnmountSummary {
                attempted: 1,
                unmounted: 1,
                already_absent: 0
            }
        );
        assert!(registry.all_owned_records().unwrap().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unmount_all_uses_registered_macos_home_mountpoint_once() {
        let root = unique_dir("unmount-all-macos-home-shared-root");
        fs::create_dir_all(&root).unwrap();
        let registry_path = root.join("registry.tsv");
        let registry = DngMountRegistry::new(registry_path);
        let home = root.join("home");
        let mountpoint = default_dng_shared_mount_root_for_platform(
            DefaultDngSharedMountRootPlatform::Macos,
            Some(home.as_os_str()),
        )
        .unwrap();
        let first = DngMountRegistryRecord::new(root.join("first.mcraw"), mountpoint.clone(), 1);
        let second = DngMountRegistryRecord::new(root.join("second.mcraw"), mountpoint.clone(), 2);
        registry.write_records(&[first, second]).unwrap();
        let adapter = MockUnmountAdapter::new(PlatformUnmountOutcome::Unmounted);

        let summary = run_dng_unmount_all_with_adapter(&registry, &adapter).unwrap();

        assert_eq!(adapter.paths.borrow().as_slice(), &[mountpoint]);
        assert_eq!(
            summary,
            DngUnmountSummary {
                attempted: 1,
                unmounted: 1,
                already_absent: 0
            }
        );
        assert!(registry.all_owned_records().unwrap().is_empty());
        let _ = fs::remove_dir_all(root);
    }
}
