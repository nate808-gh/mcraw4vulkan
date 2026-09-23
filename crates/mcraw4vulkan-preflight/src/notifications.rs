use crate::{
    MountBackendKind, PlatformKind, PreflightReport, PreflightRequirement,
    PreflightRequirementStatus,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequirementNotification {
    pub title: String,
    pub body: String,
}

impl RequirementNotification {
    pub fn stderr_message(&self) -> String {
        format!("{}\n{}", self.title, self.body)
    }
}

pub trait RequirementNotifier {
    fn notify(&mut self, notification: &RequirementNotification) -> Result<(), String>;
}

pub fn send_requirement_notification(
    notifier: &mut impl RequirementNotifier,
    notification: &RequirementNotification,
) -> Result<(), String> {
    notifier.notify(notification)
}

pub fn notification_for_report(report: &PreflightReport) -> Option<RequirementNotification> {
    // Notifications consume the report's blocking decisions rather than
    // repeating platform probes or independently deciding launch policy.
    let missing = report
        .checks
        .iter()
        .filter(|check| {
            check.blocking
                && matches!(
                    check.requirement_status,
                    Some(PreflightRequirementStatus::Missing | PreflightRequirementStatus::Unknown)
                )
        })
        .filter_map(|check| check.requirement)
        .map(|requirement| missing_requirement_notification(report.platform.kind, requirement))
        .collect::<Vec<_>>();

    match missing.as_slice() {
        [] => None,
        [notification] => Some(notification.clone()),
        _ => Some(RequirementNotification {
            title: "mcraw4vulkan requirements missing".to_string(),
            body: missing
                .iter()
                .map(RequirementNotification::stderr_message)
                .collect::<Vec<_>>()
                .join("\n\n"),
        }),
    }
}

pub fn missing_requirement_notification(
    platform: PlatformKind,
    requirement: PreflightRequirement,
) -> RequirementNotification {
    match requirement {
        PreflightRequirement::MountBackend(MountBackendKind::MacosMacFuse) => {
            RequirementNotification {
                title: "macFUSE is required".to_string(),
                body: "https://macfuse.github.io/".to_string(),
            }
        }
        PreflightRequirement::MountBackend(MountBackendKind::LinuxFuse3) => {
            RequirementNotification {
                title: "FUSE3 is required".to_string(),
                body: "Install fuse3 using your distribution's package manager".to_string(),
            }
        }
        PreflightRequirement::MountBackend(MountBackendKind::WindowsProjFs) => {
            RequirementNotification {
                title: "ProjFS is required".to_string(),
                body: "In Windows PowerShell enter: Enable-WindowsOptionalFeature -Online -FeatureName Client-ProjFS -NoRestart".to_string(),
            }
        }
        PreflightRequirement::MountBackend(MountBackendKind::Unsupported) => RequirementNotification {
            title: "A supported mount backend is required".to_string(),
            body: "Use macFUSE on macOS, FUSE3 on Linux, or ProjFS on Windows".to_string(),
        },
        PreflightRequirement::Vulkan => match platform {
            PlatformKind::Macos => RequirementNotification {
                title: "Vulkan libraries are required".to_string(),
                body: "Vulkan libraries are included with the official installation .dmg"
                    .to_string(),
            },
            PlatformKind::Linux => RequirementNotification {
                title: "Vulkan libraries are required".to_string(),
                body: "Install Mesa Vulkan packages or your GPU vendor's Vulkan driver using your distribution's package manager".to_string(),
            },
            PlatformKind::Windows => RequirementNotification {
                title: "Vulkan libraries are required".to_string(),
                body: "These are included with GPU drivers from NVIDIA, AMD, and Intel".to_string(),
            },
            PlatformKind::Unsupported => RequirementNotification {
                title: "Vulkan libraries are required".to_string(),
                body: "Install Vulkan libraries for your platform".to_string(),
            },
        },
    }
}
