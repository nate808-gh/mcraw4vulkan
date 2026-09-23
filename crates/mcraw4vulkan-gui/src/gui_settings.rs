#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewTiming {
    VsyncOn,
    MaxSpeed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeMode {
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerProfile {
    Default,
    Optimized,
}

impl OptimizerProfile {
    pub fn cli_flag(self) -> &'static str {
        match self {
            Self::Default => "--default",
            Self::Optimized => "--optimized",
        }
    }

    pub fn display_settings(self) -> mcraw4vulkan::display_window::DisplayCliSettings {
        match self {
            Self::Default => mcraw4vulkan::display_window::DisplayCliSettings::Default,
            Self::Optimized => mcraw4vulkan::display_window::DisplayCliSettings::Optimized,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "Default Settings",
            Self::Optimized => "Optimized Settings",
        }
    }
}

// Preview, PIPE, and DNG read the same decode-mode selection, so changing it
// cannot leave sink-specific GPU/CPU choices out of sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuiSettings {
    pub quick_preview_timing: PreviewTiming,
    pub decode_mode: DecodeMode,
    pub quick_preview_vignette: bool,
    pub quick_preview_fps_overlay: bool,
    pub optimizer_profile: OptimizerProfile,
    pub dng_vignette: bool,
}

impl Default for GuiSettings {
    fn default() -> Self {
        Self {
            quick_preview_timing: PreviewTiming::VsyncOn,
            decode_mode: DecodeMode::Gpu,
            quick_preview_vignette: false,
            quick_preview_fps_overlay: true,
            optimizer_profile: OptimizerProfile::Default,
            dng_vignette: true,
        }
    }
}

impl GuiSettings {
    pub fn select_quick_preview_timing(&mut self, timing: PreviewTiming) {
        self.quick_preview_timing = timing;
    }

    pub fn decode_mode(&self) -> DecodeMode {
        self.decode_mode
    }

    pub fn select_decode_mode(&mut self, mode: DecodeMode) {
        self.decode_mode = mode;
    }

    pub fn toggle_quick_preview_vignette(&mut self) {
        self.quick_preview_vignette = !self.quick_preview_vignette;
    }

    pub fn toggle_quick_preview_fps_overlay(&mut self) {
        self.quick_preview_fps_overlay = !self.quick_preview_fps_overlay;
    }

    pub fn select_optimizer_profile(&mut self, profile: OptimizerProfile) {
        self.optimizer_profile = profile;
    }

    pub fn toggle_dng_vignette(&mut self) {
        self.dng_vignette = !self.dng_vignette;
    }
}
