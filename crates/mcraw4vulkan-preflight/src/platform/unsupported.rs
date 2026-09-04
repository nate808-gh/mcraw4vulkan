use crate::platform::{PlatformKind, platform_summary, unknown_hardware};
use crate::{MountBackendKind, PlatformSummary, PreflightRequirementStatus};

pub(crate) fn collect() -> (PlatformSummary, crate::HardwareSummary) {
    (
        platform_summary(
            PlatformKind::Unsupported,
            format!("{} unsupported", std::env::consts::OS),
            MountBackendKind::Unsupported,
            PreflightRequirementStatus::Missing,
            "No supported mount backend is defined for this platform".to_string(),
        ),
        unknown_hardware(),
    )
}
