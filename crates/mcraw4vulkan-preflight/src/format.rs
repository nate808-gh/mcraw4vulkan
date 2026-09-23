use crate::{MountBackendKind, PlatformKind, PreflightRequirementStatus};

pub(crate) fn start_line() -> String {
    "Pre-flight Checklist in progress".to_string()
}

pub(crate) fn cpu_line(cpu_model: Option<&str>) -> String {
    prefixed_line("CPU", cpu_model)
}

pub(crate) fn gpu_line(gpu_model: Option<&str>) -> String {
    prefixed_line("GPU", gpu_model)
}

pub(crate) fn mount_backend_line(
    kind: MountBackendKind,
    status: PreflightRequirementStatus,
) -> String {
    match (kind, status) {
        (MountBackendKind::LinuxFuse3, PreflightRequirementStatus::Available) => {
            "FUSE3: found".to_string()
        }
        (MountBackendKind::LinuxFuse3, _) => "FUSE3 is required".to_string(),
        (MountBackendKind::MacosMacFuse, PreflightRequirementStatus::Available) => {
            "macFUSE: found".to_string()
        }
        (MountBackendKind::MacosMacFuse, _) => "macFUSE is required".to_string(),
        (MountBackendKind::WindowsProjFs, PreflightRequirementStatus::Available) => {
            "ProjFS: enabled".to_string()
        }
        (MountBackendKind::WindowsProjFs, _) => "ProjFS is required".to_string(),
        (MountBackendKind::Unsupported, _) => {
            "Mount backend not found. mcraw4vulkan requires a supported mount backend".to_string()
        }
    }
}

pub(crate) fn vulkan_line(platform: PlatformKind, status: PreflightRequirementStatus) -> String {
    match (platform, status) {
        (PlatformKind::Macos, PreflightRequirementStatus::Available) => {
            "Vulkan/MoltenVK: found".to_string()
        }
        (_, PreflightRequirementStatus::Available) => "Vulkan: found".to_string(),
        _ => "Vulkan libraries are required".to_string(),
    }
}

pub(crate) fn launching_line() -> String {
    "Launching mcraw4vulkan".to_string()
}

fn prefixed_line(prefix: &str, value: Option<&str>) -> String {
    let value = value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("unknown");
    format!("{prefix} {value}")
}
