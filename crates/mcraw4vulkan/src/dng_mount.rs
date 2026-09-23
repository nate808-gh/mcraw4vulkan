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
