#[cfg(any(target_os = "linux", test))]
use std::env;
#[cfg(any(target_os = "linux", test))]
use std::path::Path;
#[cfg(any(target_os = "linux", target_os = "macos", test))]
use std::path::PathBuf;

use crate::{HardwareSummary, MountBackendKind, PlatformSummary, PreflightRequirementStatus};

#[cfg(any(target_os = "linux", test))]
pub(crate) mod linux;
#[cfg(any(target_os = "macos", test))]
pub(crate) mod macos;
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub(crate) mod unsupported;
#[cfg(any(target_os = "windows", test))]
pub(crate) mod windows;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformKind {
    Linux,
    Macos,
    Windows,
    Unsupported,
}

pub(crate) fn collect_platform_summary() -> (PlatformSummary, HardwareSummary) {
    #[cfg(target_os = "linux")]
    {
        linux::collect()
    }
    #[cfg(target_os = "macos")]
    {
        macos::collect()
    }
    #[cfg(target_os = "windows")]
    {
        windows::collect()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        unsupported::collect()
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn path_value() -> Option<String> {
    env::var_os("PATH").map(|value| value.to_string_lossy().into_owned())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) fn absolute_path(parts: &[&str]) -> PathBuf {
    let mut path = PathBuf::from(std::path::MAIN_SEPARATOR.to_string());
    for part in parts {
        path.push(part);
    }
    path
}

#[cfg(any(target_os = "linux", test))]
pub(crate) fn find_executable_in_path_list(
    program: &str,
    path_value: Option<&str>,
    extra_paths: &[&Path],
) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path_value) = path_value {
        candidates.extend(env::split_paths(path_value).map(|dir| dir.join(program)));
    }
    candidates.extend(extra_paths.iter().map(|path| path.to_path_buf()));

    candidates
        .into_iter()
        .find(|candidate| is_executable_file(candidate))
}

pub(crate) fn availability_detail(
    available: bool,
    available_detail: &str,
    missing_detail: &str,
) -> (PreflightRequirementStatus, String) {
    if available {
        (
            PreflightRequirementStatus::Available,
            available_detail.to_string(),
        )
    } else {
        (
            PreflightRequirementStatus::Missing,
            missing_detail.to_string(),
        )
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub(crate) fn unknown_hardware() -> HardwareSummary {
    HardwareSummary {
        cpu_model: None,
        gpu_model: None,
    }
}

pub(crate) fn platform_summary(
    kind: PlatformKind,
    os_summary: String,
    mount_backend: MountBackendKind,
    mount_backend_status: PreflightRequirementStatus,
    mount_backend_detail: String,
) -> PlatformSummary {
    PlatformSummary {
        kind,
        os_summary,
        mount_backend,
        mount_backend_status,
        mount_backend_detail,
    }
}

#[cfg(any(target_os = "linux", test))]
fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = path.metadata() else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    executable_metadata(&metadata)
}

#[cfg(all(any(target_os = "linux", test), unix))]
fn executable_metadata(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;

    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(all(any(target_os = "linux", test), not(unix)))]
fn executable_metadata(_metadata: &std::fs::Metadata) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn path_executable_finder_finds_fake_program() {
        let dir = env::temp_dir().join(format!(
            "mcraw4vulkan-preflight-path-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create test dir");
        let program = dir.join("fake-fusermount3");
        fs::write(&program, b"").expect("write fake executable");
        #[cfg(unix)]
        {
            let mut permissions = fs::metadata(&program).expect("metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&program, permissions).expect("chmod");
        }

        let found =
            find_executable_in_path_list("fake-fusermount3", Some(&dir.to_string_lossy()), &[]);

        assert_eq!(found.as_deref(), Some(program.as_path()));
        let _ = fs::remove_dir_all(&dir);
    }
}
