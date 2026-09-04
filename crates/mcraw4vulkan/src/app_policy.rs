use std::path::PathBuf;

use mcraw4vulkan_dngwriter::DngSinkVignetteMode;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DngAppPolicy {
    pub vignette: DngSinkVignetteMode,
    pub backend: DngAppBackendPolicy,
    pub mount: DngMountPolicy,
}

impl DngAppPolicy {
    pub fn production_default() -> Self {
        Self {
            vignette: DngSinkVignetteMode::LumaPlane0,
            backend: DngAppBackendPolicy::Auto,
            mount: DngMountPolicy::platform_default(),
        }
    }
}

impl Default for DngAppPolicy {
    fn default() -> Self {
        Self::production_default()
    }
}

// Auto follows the production GPU route. Cpu is an explicit manual fallback,
// not a backend selected by optimizer state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DngAppBackendPolicy {
    Auto,
    Gpu,
    Cpu,
}

impl DngAppBackendPolicy {
    pub fn parse_label(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "gpu" => Some(Self::Gpu),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }
}

// AutoTemp is a policy token; mount code derives the shared root from the
// platform's home-directory environment instead of storing a caller path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DngMountPolicy {
    AutoTemp,
    Explicit(PathBuf),
    UnsupportedOnThisPlatform,
}

impl DngMountPolicy {
    pub fn platform_default() -> Self {
        if cfg!(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "windows"
        )) {
            Self::AutoTemp
        } else {
            Self::UnsupportedOnThisPlatform
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_policy_default_dng_is_luma_plane_0() {
        let policy = DngAppPolicy::production_default();
        assert_eq!(policy.vignette, DngSinkVignetteMode::LumaPlane0);
        assert_eq!(policy.backend, DngAppBackendPolicy::Auto);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_app_policy_default_dng_mount_is_enabled() {
        let policy = DngAppPolicy::production_default();
        assert_eq!(policy.mount, DngMountPolicy::AutoTemp);
    }
}
