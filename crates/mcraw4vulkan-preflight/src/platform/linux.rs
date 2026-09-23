#[cfg(target_os = "linux")]
use std::fs;
#[cfg(target_os = "linux")]
use std::path::{Path, PathBuf};

#[cfg(target_os = "linux")]
use crate::platform::{
    PlatformKind, absolute_path, availability_detail, path_value, platform_summary,
};
#[cfg(target_os = "linux")]
use crate::{HardwareSummary, MountBackendKind, PlatformSummary, PreflightRequirementStatus};

#[cfg(target_os = "linux")]
pub(crate) fn collect() -> (PlatformSummary, HardwareSummary) {
    let os_release = fs::read_to_string(absolute_path(&["etc", "os-release"])).ok();
    let kernel_release =
        fs::read_to_string(absolute_path(&["proc", "sys", "kernel", "osrelease"])).ok();
    let cpuinfo = fs::read_to_string(absolute_path(&["proc", "cpuinfo"])).ok();
    let device_tree_model =
        fs::read_to_string(absolute_path(&["proc", "device-tree", "model"])).ok();
    let filesystems = fs::read_to_string(absolute_path(&["proc", "filesystems"])).ok();

    let os_summary = linux_os_summary(os_release.as_deref(), kernel_release.as_deref());
    let cpu_model = parse_cpuinfo_model(cpuinfo.as_deref().unwrap_or(""))
        .or_else(|| parse_device_tree_model(device_tree_model.as_deref().unwrap_or("")));
    let (mount_backend_status, mount_backend_detail) = linux_fuse3_status(
        absolute_path(&["dev", "fuse"]).as_path(),
        filesystems.as_deref(),
        path_value().as_deref(),
    );

    (
        platform_summary(
            PlatformKind::Linux,
            os_summary,
            MountBackendKind::LinuxFuse3,
            mount_backend_status,
            mount_backend_detail,
        ),
        HardwareSummary {
            cpu_model,
            gpu_model: None,
        },
    )
}

#[cfg(target_os = "linux")]
pub(crate) fn linux_os_summary(os_release: Option<&str>, kernel_release: Option<&str>) -> String {
    let pretty_name = os_release
        .and_then(parse_os_release_pretty_name)
        .unwrap_or_else(|| "Linux".to_string());
    match kernel_release
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(kernel) => format!("{pretty_name} kernel {kernel}"),
        None => pretty_name,
    }
}

pub(crate) fn parse_os_release_pretty_name(input: &str) -> Option<String> {
    for line in input.lines() {
        let Some(value) = line.strip_prefix("PRETTY_NAME=") else {
            continue;
        };
        return parse_os_release_value(value);
    }
    None
}

pub(crate) fn parse_cpuinfo_model(input: &str) -> Option<String> {
    for key in ["model name", "Hardware", "Processor", "Model"] {
        for line in input.lines() {
            let Some((line_key, value)) = line.split_once(':') else {
                continue;
            };
            if line_key.trim() == key {
                let value = value.trim();
                if !value.is_empty() {
                    return Some(value.to_string());
                }
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
pub(crate) fn parse_device_tree_model(input: &str) -> Option<String> {
    let value = input.trim_matches(char::from(0)).trim();
    (!value.is_empty()).then(|| value.to_string())
}

pub(crate) fn parse_proc_filesystems_has_fuse(input: &str) -> bool {
    input.lines().any(|line| {
        let name = line.split_whitespace().last().unwrap_or("");
        matches!(name, "fuse" | "fuseblk")
    })
}

#[cfg(target_os = "linux")]
fn linux_fuse3_status(
    dev_fuse: &Path,
    filesystems: Option<&str>,
    path_value: Option<&str>,
) -> (PreflightRequirementStatus, String) {
    let dev_fuse_exists = dev_fuse.exists();
    // Failure to read the kernel filesystem list is not proof that support is
    // absent, but the FUSE device and an executable fusermount3 remain required.
    let proc_lists_fuse = filesystems.is_none_or(parse_proc_filesystems_has_fuse);
    let common_paths = [
        absolute_path(&["bin", "fusermount3"]),
        absolute_path(&["usr", "bin", "fusermount3"]),
        absolute_path(&["usr", "local", "bin", "fusermount3"]),
    ];
    let common_path_refs = common_paths
        .iter()
        .map(PathBuf::as_path)
        .collect::<Vec<_>>();
    let fusermount3 =
        super::find_executable_in_path_list("fusermount3", path_value, &common_path_refs);
    let available = dev_fuse_exists && proc_lists_fuse && fusermount3.is_some();

    availability_detail(
        available,
        "Linux FUSE3 appears available through the FUSE device and fusermount3",
        "Linux FUSE3 requires the FUSE device, kernel FUSE support, and fusermount3",
    )
}

fn parse_os_release_value(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let value = if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
        &value[1..value.len() - 1]
    } else {
        value
    };
    let value = value.to_string();
    (!value.trim().is_empty()).then_some(value)
}
