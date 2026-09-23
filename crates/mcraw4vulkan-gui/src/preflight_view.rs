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
