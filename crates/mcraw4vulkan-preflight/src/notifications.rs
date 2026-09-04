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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        HardwareSummary, PlatformSummary, PreflightCheckKind, PreflightCheckResult,
        PreflightOptions, PreflightStatus, VulkanSummary,
    };

    #[derive(Default)]
    struct MockNotifier {
        sent: Vec<RequirementNotification>,
        fail: bool,
    }

    impl RequirementNotifier for MockNotifier {
        fn notify(&mut self, notification: &RequirementNotification) -> Result<(), String> {
            self.sent.push(notification.clone());
            if self.fail {
                Err("notification unavailable".to_string())
            } else {
                Ok(())
            }
        }
    }

    fn blocked_report(
        platform: PlatformKind,
        mount_backend: MountBackendKind,
        missing: Vec<PreflightRequirement>,
    ) -> PreflightReport {
        let checks = missing
            .into_iter()
            .map(|requirement| PreflightCheckResult {
                kind: match requirement {
                    PreflightRequirement::MountBackend(_) => PreflightCheckKind::MountBackend,
                    PreflightRequirement::Vulkan => PreflightCheckKind::Vulkan,
                },
                line: "missing".to_string(),
                requirement: Some(requirement),
                requirement_status: Some(PreflightRequirementStatus::Missing),
                blocking: true,
            })
            .collect();

        PreflightReport {
            status: PreflightStatus::Blocked,
            ready_to_launch: false,
            options: PreflightOptions::gui_blocking_policy(),
            platform: PlatformSummary {
                kind: platform,
                os_summary: "test os".to_string(),
                mount_backend,
                mount_backend_status: PreflightRequirementStatus::Missing,
                mount_backend_detail: "missing".to_string(),
            },
            hardware: HardwareSummary {
                cpu_model: None,
                gpu_model: None,
            },
            vulkan: VulkanSummary {
                status: PreflightRequirementStatus::Missing,
                adapter_name: None,
                detail: "missing".to_string(),
            },
            checks,
            events: Vec::new(),
        }
    }

    #[test]
    fn macos_missing_macfuse_notification_is_actionable() {
        let notification = missing_requirement_notification(
            PlatformKind::Macos,
            PreflightRequirement::MountBackend(MountBackendKind::MacosMacFuse),
        );

        assert_eq!(notification.title, "macFUSE is required");
        assert_eq!(notification.body, "https://macfuse.github.io/");
    }

    #[test]
    fn macos_missing_vulkan_notification_is_actionable() {
        let notification =
            missing_requirement_notification(PlatformKind::Macos, PreflightRequirement::Vulkan);

        assert_eq!(notification.title, "Vulkan libraries are required");
        assert_eq!(
            notification.body,
            "Vulkan libraries are included with the official installation .dmg"
        );
    }

    #[test]
    fn linux_missing_fuse3_notification_is_actionable() {
        let notification = missing_requirement_notification(
            PlatformKind::Linux,
            PreflightRequirement::MountBackend(MountBackendKind::LinuxFuse3),
        );

        assert_eq!(notification.title, "FUSE3 is required");
        assert_eq!(
            notification.body,
            "Install fuse3 using your distribution's package manager"
        );
    }

    #[test]
    fn linux_missing_vulkan_notification_is_actionable() {
        let notification =
            missing_requirement_notification(PlatformKind::Linux, PreflightRequirement::Vulkan);

        assert_eq!(notification.title, "Vulkan libraries are required");
        assert_eq!(
            notification.body,
            "Install Mesa Vulkan packages or your GPU vendor's Vulkan driver using your distribution's package manager"
        );
    }

    #[test]
    fn windows_missing_projfs_notification_is_actionable() {
        let notification = missing_requirement_notification(
            PlatformKind::Windows,
            PreflightRequirement::MountBackend(MountBackendKind::WindowsProjFs),
        );

        assert_eq!(notification.title, "ProjFS is required");
        assert_eq!(
            notification.body,
            "In Windows PowerShell enter: Enable-WindowsOptionalFeature -Online -FeatureName Client-ProjFS -NoRestart"
        );
    }

    #[test]
    fn windows_missing_vulkan_notification_is_actionable() {
        let notification =
            missing_requirement_notification(PlatformKind::Windows, PreflightRequirement::Vulkan);

        assert_eq!(notification.title, "Vulkan libraries are required");
        assert_eq!(
            notification.body,
            "These are included with GPU drivers from NVIDIA, AMD, and Intel"
        );
    }

    #[test]
    fn multiple_missing_dependencies_are_combined() {
        let report = blocked_report(
            PlatformKind::Macos,
            MountBackendKind::MacosMacFuse,
            vec![
                PreflightRequirement::MountBackend(MountBackendKind::MacosMacFuse),
                PreflightRequirement::Vulkan,
            ],
        );

        let notification = notification_for_report(&report).expect("blocked notification");

        assert_eq!(notification.title, "mcraw4vulkan requirements missing");
        assert!(notification.body.contains("macFUSE is required"));
        assert!(notification.body.contains("https://macfuse.github.io/"));
        assert!(notification.body.contains("Vulkan libraries are required"));
        assert!(
            notification
                .body
                .contains("Vulkan libraries are included with the official installation .dmg")
        );
    }

    #[test]
    fn mock_notifier_receives_notification_without_os_side_effects() {
        let notification =
            missing_requirement_notification(PlatformKind::Macos, PreflightRequirement::Vulkan);
        let mut notifier = MockNotifier::default();

        send_requirement_notification(&mut notifier, &notification).expect("mock notify");

        assert_eq!(notifier.sent, vec![notification]);
    }

    #[test]
    fn mock_notifier_failure_is_reported_without_launching_os_notification() {
        let notification =
            missing_requirement_notification(PlatformKind::Macos, PreflightRequirement::Vulkan);
        let mut notifier = MockNotifier {
            sent: Vec::new(),
            fail: true,
        };

        let error =
            send_requirement_notification(&mut notifier, &notification).expect_err("mock fails");

        assert_eq!(error, "notification unavailable");
        assert_eq!(notifier.sent, vec![notification]);
    }
}
