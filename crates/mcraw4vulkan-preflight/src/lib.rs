#[cfg(any(target_os = "macos", target_os = "windows"))]
mod child_process;
mod format;
pub mod launcher;
mod notifications;
mod platform;
mod ram;
mod vulkan;

use std::error::Error;
use std::fmt;

pub use notifications::{
    RequirementNotification, RequirementNotifier, missing_requirement_notification,
    notification_for_report, send_requirement_notification,
};
pub use platform::PlatformKind;
pub use ram::{SystemRamSnapshot, collect_system_ram, format_system_ram_line};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreflightOptions {
    pub require_mount_backend: bool,
    pub require_vulkan: bool,
    pub probe_gpu_model: bool,
    pub probe_vulkan: bool,
    pub include_launching_line: bool,
}

impl PreflightOptions {
    pub fn gui_blocking_policy() -> Self {
        Self {
            require_mount_backend: true,
            require_vulkan: true,
            probe_gpu_model: true,
            probe_vulkan: true,
            include_launching_line: true,
        }
    }

    pub fn cli_informational_policy() -> Self {
        Self {
            require_mount_backend: false,
            require_vulkan: false,
            probe_gpu_model: true,
            probe_vulkan: true,
            include_launching_line: true,
        }
    }
}

impl Default for PreflightOptions {
    fn default() -> Self {
        Self::gui_blocking_policy()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightEvent {
    pub kind: PreflightEventKind,
    pub check: PreflightCheckKind,
    pub line: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightEventKind {
    Started,
    CheckCompleted,
    ReadyToLaunch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightCheckKind {
    Start,
    OsSummary,
    CpuSummary,
    GpuSummary,
    MountBackend,
    Vulkan,
    Launch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightReport {
    pub status: PreflightStatus,
    pub ready_to_launch: bool,
    pub options: PreflightOptions,
    pub platform: PlatformSummary,
    pub hardware: HardwareSummary,
    pub vulkan: VulkanSummary,
    pub checks: Vec<PreflightCheckResult>,
    pub events: Vec<PreflightEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightStatus {
    ReadyToLaunch,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightCheckResult {
    pub kind: PreflightCheckKind,
    pub line: String,
    pub requirement: Option<PreflightRequirement>,
    pub requirement_status: Option<PreflightRequirementStatus>,
    pub blocking: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightRequirement {
    MountBackend(MountBackendKind),
    Vulkan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightRequirementStatus {
    // Unknown means the fact was unavailable or not probed; it establishes neither
    // availability nor absence.
    Available,
    Missing,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountBackendKind {
    LinuxFuse3,
    MacosMacFuse,
    WindowsProjFs,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformSummary {
    pub kind: PlatformKind,
    pub os_summary: String,
    pub mount_backend: MountBackendKind,
    pub mount_backend_status: PreflightRequirementStatus,
    pub mount_backend_detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareSummary {
    pub cpu_model: Option<String>,
    pub gpu_model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanSummary {
    pub status: PreflightRequirementStatus,
    pub adapter_name: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreflightError {
    ProbeFailed {
        check: PreflightCheckKind,
        message: String,
    },
}

impl fmt::Display for PreflightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProbeFailed { check, message } => {
                write!(formatter, "preflight probe {check:?} failed: {message}")
            }
        }
    }
}

impl Error for PreflightError {}

pub fn run_preflight(options: PreflightOptions) -> PreflightReport {
    run_preflight_with_event_sink(options, |_| {})
}

pub fn run_preflight_with_event_sink(
    options: PreflightOptions,
    mut sink: impl FnMut(&PreflightEvent),
) -> PreflightReport {
    let mut probe = ProbeSnapshot::collect(&options);
    if options.probe_gpu_model && probe.hardware.gpu_model.is_none() {
        probe
            .hardware
            .gpu_model
            .clone_from(&probe.vulkan.adapter_name);
    }
    report_from_probe(options, probe, &mut sink)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProbeSnapshot {
    platform: PlatformSummary,
    hardware: HardwareSummary,
    vulkan: VulkanSummary,
}

impl ProbeSnapshot {
    fn collect(options: &PreflightOptions) -> Self {
        let (platform, hardware) = platform::collect_platform_summary();
        let vulkan = vulkan::probe_vulkan(options.probe_vulkan);
        Self {
            platform,
            hardware,
            vulkan,
        }
    }
}

fn report_from_probe(
    options: PreflightOptions,
    probe: ProbeSnapshot,
    sink: &mut impl FnMut(&PreflightEvent),
) -> PreflightReport {
    let mut checks = Vec::new();
    let mut events = Vec::new();

    push_event(
        &mut events,
        sink,
        PreflightEventKind::Started,
        PreflightCheckKind::Start,
        format::start_line(),
    );

    push_check(
        &mut checks,
        &mut events,
        sink,
        PreflightCheckResult {
            kind: PreflightCheckKind::OsSummary,
            line: probe.platform.os_summary.clone(),
            requirement: None,
            requirement_status: None,
            blocking: false,
        },
    );

    push_check(
        &mut checks,
        &mut events,
        sink,
        PreflightCheckResult {
            kind: PreflightCheckKind::CpuSummary,
            line: format::cpu_line(probe.hardware.cpu_model.as_deref()),
            requirement: None,
            requirement_status: None,
            blocking: false,
        },
    );

    push_check(
        &mut checks,
        &mut events,
        sink,
        PreflightCheckResult {
            kind: PreflightCheckKind::GpuSummary,
            line: format::gpu_line(probe.hardware.gpu_model.as_deref()),
            requirement: None,
            requirement_status: None,
            blocking: false,
        },
    );

    // A required Unknown fact blocks just like Missing. Informational policies
    // still report the fact but do not make it a launch blocker.
    let mount_blocking = options.require_mount_backend
        && probe.platform.mount_backend_status != PreflightRequirementStatus::Available;
    push_check(
        &mut checks,
        &mut events,
        sink,
        PreflightCheckResult {
            kind: PreflightCheckKind::MountBackend,
            line: format::mount_backend_line(
                probe.platform.mount_backend,
                probe.platform.mount_backend_status,
            ),
            requirement: Some(PreflightRequirement::MountBackend(
                probe.platform.mount_backend,
            )),
            requirement_status: Some(probe.platform.mount_backend_status),
            blocking: mount_blocking,
        },
    );

    let vulkan_blocking =
        options.require_vulkan && probe.vulkan.status != PreflightRequirementStatus::Available;
    push_check(
        &mut checks,
        &mut events,
        sink,
        PreflightCheckResult {
            kind: PreflightCheckKind::Vulkan,
            line: format::vulkan_line(probe.platform.kind, probe.vulkan.status),
            requirement: Some(PreflightRequirement::Vulkan),
            requirement_status: Some(probe.vulkan.status),
            blocking: vulkan_blocking,
        },
    );

    let ready_to_launch = !mount_blocking && !vulkan_blocking;
    let status = if ready_to_launch {
        PreflightStatus::ReadyToLaunch
    } else {
        PreflightStatus::Blocked
    };

    if ready_to_launch && options.include_launching_line {
        let line = format::launching_line();
        checks.push(PreflightCheckResult {
            kind: PreflightCheckKind::Launch,
            line: line.clone(),
            requirement: None,
            requirement_status: None,
            blocking: false,
        });
        push_event(
            &mut events,
            sink,
            PreflightEventKind::ReadyToLaunch,
            PreflightCheckKind::Launch,
            line,
        );
    }

    PreflightReport {
        status,
        ready_to_launch,
        options,
        platform: probe.platform,
        hardware: probe.hardware,
        vulkan: probe.vulkan,
        checks,
        events,
    }
}

fn push_check(
    checks: &mut Vec<PreflightCheckResult>,
    events: &mut Vec<PreflightEvent>,
    sink: &mut impl FnMut(&PreflightEvent),
    result: PreflightCheckResult,
) {
    let kind = result.kind;
    let line = result.line.clone();
    checks.push(result);
    push_event(events, sink, PreflightEventKind::CheckCompleted, kind, line);
}

fn push_event(
    events: &mut Vec<PreflightEvent>,
    sink: &mut impl FnMut(&PreflightEvent),
    kind: PreflightEventKind,
    check: PreflightCheckKind,
    line: String,
) {
    events.push(PreflightEvent { kind, check, line });
    let event = events
        .last()
        .expect("event was pushed before notifying sink");
    sink(event);
}
