use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

use crate::{PreflightReport, RequirementNotification, notification_for_report};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LauncherConfig {
    pub gui_path: PathBuf,
    pub gui_args: Vec<OsString>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LauncherAction {
    Launch(LauncherConfig),
    Blocked(RequirementNotification),
}

pub fn parse_launcher_args<I>(args: I) -> Result<LauncherConfig, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut iter = args.into_iter();
    let mut gui_args = Vec::new();

    let Some(arg) = iter.next() else {
        return Err("--gui is required".to_string());
    };

    let gui_path = if arg == OsStr::new("--gui") {
        let Some(path) = iter.next().map(PathBuf::from) else {
            return Err("--gui requires a path".to_string());
        };
        gui_args.extend(iter);
        path
    } else {
        let text = arg.to_string_lossy();
        if let Some(value) = text.strip_prefix("--gui=") {
            if value.is_empty() {
                return Err("--gui requires a path".to_string());
            }
            gui_args.extend(iter);
            PathBuf::from(value)
        } else {
            return Err(format!("unknown argument: {}", arg.to_string_lossy()));
        }
    };

    Ok(LauncherConfig { gui_path, gui_args })
}

pub fn action_for_report(report: &PreflightReport, config: LauncherConfig) -> LauncherAction {
    // The report already owns blocking policy. The launcher either preserves
    // the exact GUI request or presents the report-derived requirement notice.
    if let Some(notification) = notification_for_report(report) {
        LauncherAction::Blocked(notification)
    } else {
        LauncherAction::Launch(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        HardwareSummary, MountBackendKind, PlatformKind, PlatformSummary, PreflightCheckKind,
        PreflightCheckResult, PreflightOptions, PreflightRequirement, PreflightRequirementStatus,
        PreflightStatus, VulkanSummary,
    };

    fn report(status: PreflightStatus) -> PreflightReport {
        let blocked = status == PreflightStatus::Blocked;
        PreflightReport {
            status,
            ready_to_launch: !blocked,
            options: PreflightOptions::gui_blocking_policy(),
            platform: PlatformSummary {
                kind: PlatformKind::Macos,
                os_summary: "macOS test".to_string(),
                mount_backend: MountBackendKind::MacosMacFuse,
                mount_backend_status: if blocked {
                    PreflightRequirementStatus::Missing
                } else {
                    PreflightRequirementStatus::Available
                },
                mount_backend_detail: "test".to_string(),
            },
            hardware: HardwareSummary {
                cpu_model: None,
                gpu_model: None,
            },
            vulkan: VulkanSummary {
                status: PreflightRequirementStatus::Available,
                adapter_name: None,
                detail: "test".to_string(),
            },
            checks: if blocked {
                vec![PreflightCheckResult {
                    kind: PreflightCheckKind::MountBackend,
                    line: "macFUSE is required".to_string(),
                    requirement: Some(PreflightRequirement::MountBackend(
                        MountBackendKind::MacosMacFuse,
                    )),
                    requirement_status: Some(PreflightRequirementStatus::Missing),
                    blocking: true,
                }]
            } else {
                Vec::new()
            },
            events: Vec::new(),
        }
    }

    #[test]
    fn parses_gui_path_and_remaining_gui_args() {
        let config = parse_launcher_args([
            OsString::from("--gui"),
            OsString::from("gui"),
            OsString::from("--playlist"),
            OsString::from("clip.mcraw"),
        ])
        .expect("parse args");

        assert_eq!(config.gui_path, PathBuf::from("gui"));
        assert_eq!(
            config.gui_args,
            vec![OsString::from("--playlist"), OsString::from("clip.mcraw")]
        );
    }

    #[test]
    fn launcher_does_not_launch_gui_when_preflight_fails() {
        let config = LauncherConfig {
            gui_path: PathBuf::from("gui"),
            gui_args: Vec::new(),
        };

        let action = action_for_report(&report(PreflightStatus::Blocked), config);

        assert!(matches!(action, LauncherAction::Blocked(_)));
    }

    #[test]
    fn launcher_chooses_provided_gui_path_when_preflight_passes() {
        let config = LauncherConfig {
            gui_path: PathBuf::from("gui"),
            gui_args: vec![OsString::from("--example")],
        };

        let action = action_for_report(&report(PreflightStatus::ReadyToLaunch), config.clone());

        assert_eq!(action, LauncherAction::Launch(config));
    }
}
