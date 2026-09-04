#[cfg(target_os = "macos")]
use std::process::Command;

#[cfg(target_os = "macos")]
use crate::platform::{PlatformKind, absolute_path, availability_detail, platform_summary};
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
    let macfuse_root = absolute_path(&["Library", "Filesystems", "macfuse.fs"]);
    let helper = macfuse_root.join("Contents/Resources/mount_macfuse");
    let libfuse = absolute_path(&["usr", "local", "lib", "libfuse.2.dylib"]);
    // This checks installation artifacts rather than attempting a mount. The
    // macFUSE bundle requires its helper; libfuse.2.dylib is an alternative signal.
    let available = (macfuse_root.exists() && helper.exists()) || libfuse.exists();
    let (mount_backend_status, mount_backend_detail) = availability_detail(
        available,
        "macFUSE is present without loading libfuse",
        "macFUSE bundle, mount helper, or libfuse.2.dylib was not found",
    );

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macos_sw_vers_parser_reads_product_version() {
        assert_eq!(
            parse_sw_vers_product_version("26.5.1\n").as_deref(),
            Some("26.5.1")
        );
    }

    #[test]
    fn macos_sysctl_parser_reads_cpu_value() {
        assert_eq!(
            parse_sysctl_value("Apple M4 Pro\n").as_deref(),
            Some("Apple M4 Pro")
        );
    }
}
