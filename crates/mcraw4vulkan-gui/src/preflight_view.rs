use mcraw4vulkan_preflight::{
    PreflightCheckKind, PreflightCheckResult, PreflightReport, PreflightRequirementStatus,
    PreflightStatus,
};

pub const PREFLIGHT_PANEL_TITLE: &str = "Preflight System Checks";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightViewModel {
    pub status_line: String,
    pub detail_line: String,
    pub ready: bool,
    pub lines: Vec<PreflightLine>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightLine {
    pub text: String,
    pub status: LineStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineStatus {
    Available,
    Missing,
    Unknown,
    Informational,
}

impl PreflightViewModel {
    pub fn from_report(report: &PreflightReport) -> Self {
        Self::from_report_with_gui_gpu(report, None)
    }

    // Blocking policy belongs to the preflight report; this adapter only maps
    // its facts into presentation and may replace the displayed GPU description.
    pub fn from_report_with_gui_gpu(report: &PreflightReport, gui_gpu_model: Option<&str>) -> Self {
        let ready = report.status == PreflightStatus::ReadyToLaunch && report.ready_to_launch;
        let mut lines = report
            .checks
            .iter()
            .filter_map(|check| line_from_check(check, gui_gpu_model))
            .collect::<Vec<_>>();

        if !ready
            && !lines
                .iter()
                .any(|line| matches!(line.status, LineStatus::Missing | LineStatus::Unknown))
        {
            lines.push(PreflightLine {
                text: "Preflight blocked, but no detailed missing requirement was reported."
                    .to_string(),
                status: LineStatus::Unknown,
            });
        }

        Self {
            status_line: if ready {
                "Ready to Launch".to_string()
            } else {
                "Not Ready".to_string()
            },
            detail_line: if ready {
                "System requirements were checked and all requirements were met.".to_string()
            } else {
                "Resolve the listed requirements before launching the full GUI.".to_string()
            },
            ready,
            lines,
        }
    }

    pub fn starting() -> Self {
        Self {
            status_line: "Checking system requirements...".to_string(),
            detail_line: "Running startup preflight once on the foreground thread.".to_string(),
            ready: false,
            lines: vec![PreflightLine {
                text: "Pre-flight Checklist in progress".to_string(),
                status: LineStatus::Informational,
            }],
        }
    }
}

fn line_from_check(
    check: &PreflightCheckResult,
    gui_gpu_model: Option<&str>,
) -> Option<PreflightLine> {
    if check.kind == PreflightCheckKind::Launch {
        return None;
    }

    let status = line_status(check);

    Some(PreflightLine {
        text: splash_line_text(check, gui_gpu_model),
        status,
    })
}

fn line_status(check: &PreflightCheckResult) -> LineStatus {
    if check.blocking {
        return LineStatus::Missing;
    }

    match check.requirement_status {
        Some(PreflightRequirementStatus::Available) => LineStatus::Available,
        Some(PreflightRequirementStatus::Missing | PreflightRequirementStatus::Unknown) => {
            LineStatus::Missing
        }
        None => LineStatus::Informational,
    }
}

fn splash_line_text(check: &PreflightCheckResult, gui_gpu_model: Option<&str>) -> String {
    match check.kind {
        PreflightCheckKind::Start => strip_status_prefix(&check.line).to_string(),
        PreflightCheckKind::OsSummary => prefixed_summary_line("OS", &check.line),
        PreflightCheckKind::CpuSummary => prefixed_summary_line("CPU", &check.line),
        PreflightCheckKind::GpuSummary => gui_gpu_summary_line(gui_gpu_model)
            .unwrap_or_else(|| prefixed_summary_line("GPU", &check.line)),
        PreflightCheckKind::MountBackend | PreflightCheckKind::Vulkan => {
            strip_status_prefix(&check.line).to_string()
        }
        PreflightCheckKind::Launch => String::new(),
    }
}

fn gui_gpu_summary_line(gui_gpu_model: Option<&str>) -> Option<String> {
    let text = gui_gpu_model
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    Some(prefixed_summary_line("GPU", text))
}

fn prefixed_summary_line(prefix: &str, line: &str) -> String {
    let text = strip_status_prefix(line);
    if text.is_empty() {
        return format!("{prefix} unknown");
    }

    if text == prefix
        || text
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with(' '))
    {
        text.to_string()
    } else {
        format!("{prefix} {text}")
    }
}

fn strip_status_prefix(line: &str) -> &str {
    let trimmed = line.trim();
    for prefix in ["INFO", "OK", "BLOCKED", "CHECK"] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            if rest.chars().next().is_some_and(char::is_whitespace) {
                return rest.trim_start();
            }
        }
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcraw4vulkan_preflight::{
        HardwareSummary, MountBackendKind, PlatformKind, PlatformSummary, PreflightCheckKind,
        PreflightOptions, PreflightRequirement, VulkanSummary,
    };

    fn report(status: PreflightStatus, checks: Vec<PreflightCheckResult>) -> PreflightReport {
        PreflightReport {
            status,
            ready_to_launch: status == PreflightStatus::ReadyToLaunch,
            options: PreflightOptions::gui_blocking_policy(),
            platform: PlatformSummary {
                kind: PlatformKind::Linux,
                os_summary: "Linux".to_string(),
                mount_backend: MountBackendKind::LinuxFuse3,
                mount_backend_status: PreflightRequirementStatus::Available,
                mount_backend_detail: "available".to_string(),
            },
            hardware: HardwareSummary {
                cpu_model: None,
                gpu_model: None,
            },
            vulkan: VulkanSummary {
                status: PreflightRequirementStatus::Available,
                adapter_name: Some("GPU".to_string()),
                detail: "available".to_string(),
            },
            checks,
            events: Vec::new(),
        }
    }

    #[test]
    fn ready_report_formats_ready_status() {
        let model = PreflightViewModel::from_report(&report(
            PreflightStatus::ReadyToLaunch,
            vec![
                PreflightCheckResult {
                    kind: PreflightCheckKind::Vulkan,
                    line: "Vulkan library available".to_string(),
                    requirement: Some(PreflightRequirement::Vulkan),
                    requirement_status: Some(PreflightRequirementStatus::Available),
                    blocking: false,
                },
                PreflightCheckResult {
                    kind: PreflightCheckKind::Launch,
                    line: "Launching mcraw4vulkan".to_string(),
                    requirement: None,
                    requirement_status: None,
                    blocking: false,
                },
            ],
        ));

        assert!(model.ready);
        assert_eq!(PREFLIGHT_PANEL_TITLE, "Preflight System Checks");
        assert_eq!(model.status_line, "Ready to Launch");
        assert_eq!(model.lines[0].status, LineStatus::Available);
        assert_eq!(model.lines[0].text, "Vulkan library available");
        assert!(
            !model
                .lines
                .iter()
                .any(|line| line.text == "Launching mcraw4vulkan")
        );
    }

    #[test]
    fn blocked_report_lists_missing_requirement() {
        let model = PreflightViewModel::from_report(&report(
            PreflightStatus::Blocked,
            vec![PreflightCheckResult {
                kind: PreflightCheckKind::Vulkan,
                line: "Vulkan library not found. mcraw4vulkan requires Vulkan library".to_string(),
                requirement: Some(PreflightRequirement::Vulkan),
                requirement_status: Some(PreflightRequirementStatus::Missing),
                blocking: true,
            }],
        ));

        assert!(!model.ready);
        assert_eq!(model.status_line, "Not Ready");
        assert_eq!(model.lines[0].status, LineStatus::Missing);
        assert_eq!(
            model.lines[0].text,
            "Vulkan library not found. mcraw4vulkan requires Vulkan library"
        );
    }

    #[test]
    fn blocked_without_missing_detail_gets_fallback_line() {
        let model = PreflightViewModel::from_report(&report(
            PreflightStatus::Blocked,
            vec![PreflightCheckResult {
                kind: PreflightCheckKind::CpuSummary,
                line: "CPU unknown".to_string(),
                requirement: None,
                requirement_status: None,
                blocking: false,
            }],
        ));

        assert!(
            model
                .lines
                .iter()
                .any(|line| line.text.contains("no detailed missing requirement"))
        );
    }

    #[test]
    fn summary_lines_remove_status_prefixes() {
        let model = PreflightViewModel::from_report(&report(
            PreflightStatus::ReadyToLaunch,
            vec![
                PreflightCheckResult {
                    kind: PreflightCheckKind::OsSummary,
                    line: "INFO Linux kernel 6.1".to_string(),
                    requirement: None,
                    requirement_status: None,
                    blocking: false,
                },
                PreflightCheckResult {
                    kind: PreflightCheckKind::CpuSummary,
                    line: "INFO CPU Ryzen".to_string(),
                    requirement: None,
                    requirement_status: None,
                    blocking: false,
                },
                PreflightCheckResult {
                    kind: PreflightCheckKind::GpuSummary,
                    line: "INFO GPU unknown".to_string(),
                    requirement: None,
                    requirement_status: None,
                    blocking: false,
                },
            ],
        ));

        let texts = model
            .lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>();

        assert_eq!(texts, ["OS Linux kernel 6.1", "CPU Ryzen", "GPU unknown"]);
        assert!(texts.iter().all(|text| !text.starts_with("INFO ")));
    }

    #[test]
    fn gui_adapter_info_replaces_unknown_gpu_summary() {
        let model = PreflightViewModel::from_report_with_gui_gpu(
            &report(
                PreflightStatus::ReadyToLaunch,
                vec![PreflightCheckResult {
                    kind: PreflightCheckKind::GpuSummary,
                    line: "INFO GPU unknown".to_string(),
                    requirement: None,
                    requirement_status: None,
                    blocking: false,
                }],
            ),
            Some("AMD Radeon RX 7900 XTX (vulkan)"),
        );

        assert_eq!(model.lines[0].text, "GPU AMD Radeon RX 7900 XTX (vulkan)");
        assert!(!model.lines.iter().any(|line| line.text == "GPU unknown"));
    }

    #[test]
    fn missing_gui_adapter_info_keeps_gpu_fallback() {
        let model = PreflightViewModel::from_report_with_gui_gpu(
            &report(
                PreflightStatus::ReadyToLaunch,
                vec![PreflightCheckResult {
                    kind: PreflightCheckKind::GpuSummary,
                    line: "INFO GPU unknown".to_string(),
                    requirement: None,
                    requirement_status: None,
                    blocking: false,
                }],
            ),
            None,
        );

        assert_eq!(model.lines[0].text, "GPU unknown");
    }

    #[test]
    fn dependency_labels_remain_platform_specific() {
        let linux = PreflightViewModel::from_report(&report(
            PreflightStatus::ReadyToLaunch,
            vec![
                PreflightCheckResult {
                    kind: PreflightCheckKind::MountBackend,
                    line: "OK FUSE3: found".to_string(),
                    requirement: Some(PreflightRequirement::MountBackend(
                        MountBackendKind::LinuxFuse3,
                    )),
                    requirement_status: Some(PreflightRequirementStatus::Available),
                    blocking: false,
                },
                PreflightCheckResult {
                    kind: PreflightCheckKind::Vulkan,
                    line: "OK Vulkan: found".to_string(),
                    requirement: Some(PreflightRequirement::Vulkan),
                    requirement_status: Some(PreflightRequirementStatus::Available),
                    blocking: false,
                },
            ],
        ));
        let macos = PreflightViewModel::from_report(&report(
            PreflightStatus::ReadyToLaunch,
            vec![
                PreflightCheckResult {
                    kind: PreflightCheckKind::MountBackend,
                    line: "OK macFUSE: found".to_string(),
                    requirement: Some(PreflightRequirement::MountBackend(
                        MountBackendKind::MacosMacFuse,
                    )),
                    requirement_status: Some(PreflightRequirementStatus::Available),
                    blocking: false,
                },
                PreflightCheckResult {
                    kind: PreflightCheckKind::Vulkan,
                    line: "OK Vulkan/MoltenVK: found".to_string(),
                    requirement: Some(PreflightRequirement::Vulkan),
                    requirement_status: Some(PreflightRequirementStatus::Available),
                    blocking: false,
                },
            ],
        ));
        let windows = PreflightViewModel::from_report(&report(
            PreflightStatus::ReadyToLaunch,
            vec![
                PreflightCheckResult {
                    kind: PreflightCheckKind::MountBackend,
                    line: "OK ProjFS: enabled".to_string(),
                    requirement: Some(PreflightRequirement::MountBackend(
                        MountBackendKind::WindowsProjFs,
                    )),
                    requirement_status: Some(PreflightRequirementStatus::Available),
                    blocking: false,
                },
                PreflightCheckResult {
                    kind: PreflightCheckKind::Vulkan,
                    line: "OK Vulkan: found".to_string(),
                    requirement: Some(PreflightRequirement::Vulkan),
                    requirement_status: Some(PreflightRequirementStatus::Available),
                    blocking: false,
                },
            ],
        ));

        assert_eq!(
            linux
                .lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            ["FUSE3: found", "Vulkan: found"]
        );
        assert_eq!(
            macos
                .lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            ["macFUSE: found", "Vulkan/MoltenVK: found"]
        );
        assert_eq!(
            windows
                .lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            ["ProjFS: enabled", "Vulkan: found"]
        );
    }

    #[test]
    fn dependency_rows_use_preflight_report_copy() {
        let passing = PreflightViewModel::from_report(&report(
            PreflightStatus::ReadyToLaunch,
            vec![PreflightCheckResult {
                kind: PreflightCheckKind::MountBackend,
                line: "OK macFUSE: found".to_string(),
                requirement: Some(PreflightRequirement::MountBackend(
                    MountBackendKind::MacosMacFuse,
                )),
                requirement_status: Some(PreflightRequirementStatus::Available),
                blocking: false,
            }],
        ));
        let failing = PreflightViewModel::from_report(&report(
            PreflightStatus::Blocked,
            vec![PreflightCheckResult {
                kind: PreflightCheckKind::MountBackend,
                line: "BLOCKED macFUSE is required".to_string(),
                requirement: Some(PreflightRequirement::MountBackend(
                    MountBackendKind::MacosMacFuse,
                )),
                requirement_status: Some(PreflightRequirementStatus::Missing),
                blocking: true,
            }],
        ));

        assert_eq!(passing.lines[0].text, "macFUSE: found");
        assert_eq!(passing.lines[0].status, LineStatus::Available);
        assert_eq!(failing.lines[0].text, "macFUSE is required");
        assert_eq!(failing.lines[0].status, LineStatus::Missing);
    }

    #[test]
    fn vulkan_rows_use_preflight_report_copy() {
        let passing = PreflightViewModel::from_report(&report(
            PreflightStatus::ReadyToLaunch,
            vec![PreflightCheckResult {
                kind: PreflightCheckKind::Vulkan,
                line: "OK Vulkan/MoltenVK: found".to_string(),
                requirement: Some(PreflightRequirement::Vulkan),
                requirement_status: Some(PreflightRequirementStatus::Available),
                blocking: false,
            }],
        ));
        let failing = PreflightViewModel::from_report(&report(
            PreflightStatus::Blocked,
            vec![PreflightCheckResult {
                kind: PreflightCheckKind::Vulkan,
                line: "BLOCKED Vulkan libraries are required".to_string(),
                requirement: Some(PreflightRequirement::Vulkan),
                requirement_status: Some(PreflightRequirementStatus::Missing),
                blocking: true,
            }],
        ));

        assert_eq!(passing.lines[0].text, "Vulkan/MoltenVK: found");
        assert_eq!(passing.lines[0].status, LineStatus::Available);
        assert_eq!(failing.lines[0].text, "Vulkan libraries are required");
        assert_eq!(failing.lines[0].status, LineStatus::Missing);
    }
}
