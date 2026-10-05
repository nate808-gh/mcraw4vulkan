#[cfg(target_os = "macos")]
use std::process::Command;

#[cfg(target_os = "macos")]
use crate::platform::{PlatformKind, platform_summary};
#[cfg(target_os = "macos")]
use crate::{HardwareSummary, MountBackendKind, PlatformSummary};

#[cfg(target_os = "macos")]
pub(crate) fn collect() -> (PlatformSummary, HardwareSummary) {
    let os_version = command_stdout("sw_vers", &["-productVersion"])
        .and_then(|output| parse_sw_vers_product_version(&output));
    let cpu_model = command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"])
        .and_then(|output| parse_sysctl_value(&output))
        .or_else(|| {
            command_stdout("sysctl", &["-n", "hw.model"])
                .and_then(|output| parse_sysctl_value(&output))
        });
    // Share the mount-side location policy without loading the native library.
    let (mount_backend_status, mount_backend_detail) =
        match mcraw4vulkan_core::macfuse_location::library_from_environment() {
            Ok(_) => (
                crate::PreflightRequirementStatus::Available,
                "macFUSE FUSE3 installation artifacts are present without attempting a mount"
                    .to_string(),
            ),
            Err(error) => (
                crate::PreflightRequirementStatus::Missing,
                error.to_string(),
            ),
        };

    (
        platform_summary(
            PlatformKind::Macos,
            macos_os_summary(os_version.as_deref()),
            MountBackendKind::MacosMacFuse,
            mount_backend_status,
            mount_backend_detail,
        ),
        HardwareSummary {
            cpu_model,
            gpu_model: None,
        },
    )
}

#[cfg(target_os = "macos")]
pub(crate) fn macos_os_summary(version: Option<&str>) -> String {
    match version.map(str::trim).filter(|value| !value.is_empty()) {
        Some(version) => format!("macOS {version}"),
        None => "macOS unknown".to_string(),
    }
}

pub(crate) fn parse_sw_vers_product_version(input: &str) -> Option<String> {
    parse_first_non_empty_line(input)
}

pub(crate) fn parse_sysctl_value(input: &str) -> Option<String> {
    parse_first_non_empty_line(input)
}

#[cfg(target_os = "macos")]
fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

fn parse_first_non_empty_line(input: &str) -> Option<String> {
    input
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_string)
}
