#[cfg(target_os = "windows")]
use std::env;
#[cfg(target_os = "windows")]
use std::path::{Path, PathBuf};

#[cfg(target_os = "windows")]
use crate::platform::{PlatformKind, availability_detail, platform_summary};
#[cfg(target_os = "windows")]
use crate::{HardwareSummary, MountBackendKind, PlatformSummary};

#[cfg(target_os = "windows")]
pub(crate) fn collect() -> (PlatformSummary, HardwareSummary) {
    let os_summary = windows_os_summary(env::var("OS").ok().as_deref());
    let cpu_model = parse_processor_identifier(env::var("PROCESSOR_IDENTIFIER").ok().as_deref());
    // Presence of the client library is the preflight availability fact; this
    // probe does not start a projection or take ownership of a mount root.
    let projfs_dll = projected_fs_library_path();
    let (mount_backend_status, mount_backend_detail) = availability_detail(
        projfs_dll.as_deref().is_some_and(Path::exists),
        "ProjectedFSLib.dll is present",
        "ProjectedFSLib.dll was not found",
    );

    (
        platform_summary(
            PlatformKind::Windows,
            os_summary,
            MountBackendKind::WindowsProjFs,
            mount_backend_status,
            mount_backend_detail,
        ),
        HardwareSummary {
            cpu_model,
            gpu_model: None,
        },
    )
}

#[cfg(target_os = "windows")]
pub(crate) fn windows_os_summary(os_env: Option<&str>) -> String {
    match os_env.map(str::trim).filter(|value| !value.is_empty()) {
        Some(value) => format!("Windows {value}"),
        None => "Windows unknown".to_string(),
    }
}

pub(crate) fn parse_processor_identifier(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

#[cfg(target_os = "windows")]
fn projected_fs_library_path() -> Option<PathBuf> {
    let system_root = env::var_os("SystemRoot")?;
    Some(
        PathBuf::from(system_root)
            .join("System32")
            .join("ProjectedFSLib.dll"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_processor_identifier_parser_reads_value() {
        assert_eq!(
            parse_processor_identifier(Some("Intel64 Family 6 Model 170")).as_deref(),
            Some("Intel64 Family 6 Model 170")
        );
    }

    #[test]
    fn windows_processor_identifier_parser_ignores_empty_value() {
        assert_eq!(parse_processor_identifier(Some("   ")), None);
    }
}
