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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_current_gui_selection() {
        let settings = GuiSettings::default();

        assert_eq!(settings.quick_preview_timing, PreviewTiming::VsyncOn);
        assert_eq!(settings.decode_mode, DecodeMode::Gpu);
        assert_eq!(settings.decode_mode(), DecodeMode::Gpu);
        assert!(!settings.quick_preview_vignette);
        assert!(settings.quick_preview_fps_overlay);
        assert_eq!(settings.optimizer_profile, OptimizerProfile::Default);
        assert!(settings.dng_vignette);
    }

    #[test]
    fn old_sink_owned_decode_state_is_absent() {
        let source = include_str!("gui_settings.rs");
        let preview_field = ["pub quick_preview", "_decode: DecodeMode"].concat();
        let preview_setter = ["select_quick_preview", "_decode"].concat();
        let dng_field = ["pub dng", "_decode: DecodeMode"].concat();
        let dng_setter = ["select_dng", "_decode"].concat();

        assert!(!source.contains(&preview_field));
        assert!(!source.contains(&preview_setter));
        assert!(!source.contains(&dng_field));
        assert!(!source.contains(&dng_setter));
    }

    #[test]
    fn quick_preview_vsync_and_max_speed_are_mutually_exclusive() {
        let mut settings = GuiSettings::default();

        settings.select_quick_preview_timing(PreviewTiming::MaxSpeed);
        assert_eq!(settings.quick_preview_timing, PreviewTiming::MaxSpeed);

        settings.select_quick_preview_timing(PreviewTiming::VsyncOn);
        assert_eq!(settings.quick_preview_timing, PreviewTiming::VsyncOn);
    }

    #[test]
    fn decode_mode_gpu_and_cpu_are_mutually_exclusive() {
        let mut settings = GuiSettings::default();

        settings.select_decode_mode(DecodeMode::Cpu);
        assert_eq!(settings.decode_mode(), DecodeMode::Cpu);

        settings.select_decode_mode(DecodeMode::Gpu);
        assert_eq!(settings.decode_mode(), DecodeMode::Gpu);
    }

    #[test]
    fn quick_preview_vignette_and_fps_overlay_toggles_update_state() {
        let mut settings = GuiSettings::default();

        settings.toggle_quick_preview_vignette();
        settings.toggle_quick_preview_fps_overlay();

        assert!(settings.quick_preview_vignette);
        assert!(!settings.quick_preview_fps_overlay);
    }

    #[test]
    fn optimizer_default_and_optimized_settings_are_mutually_exclusive() {
        let mut settings = GuiSettings::default();

        settings.select_optimizer_profile(OptimizerProfile::Optimized);
        assert_eq!(settings.optimizer_profile, OptimizerProfile::Optimized);

        settings.select_optimizer_profile(OptimizerProfile::Default);
        assert_eq!(settings.optimizer_profile, OptimizerProfile::Default);
    }

    #[test]
    fn optimizer_profile_maps_to_shared_cli_flags_and_display_settings() {
        assert_eq!(OptimizerProfile::Default.cli_flag(), "--default");
        assert_eq!(OptimizerProfile::Optimized.cli_flag(), "--optimized");
        assert_eq!(
            OptimizerProfile::Default.display_settings(),
            mcraw4vulkan::display_window::DisplayCliSettings::Default
        );
        assert_eq!(
            OptimizerProfile::Optimized.display_settings(),
            mcraw4vulkan::display_window::DisplayCliSettings::Optimized
        );
    }

    #[test]
    fn dng_uses_the_shared_decode_mode() {
        let mut settings = GuiSettings::default();

        settings.select_decode_mode(DecodeMode::Cpu);
        assert_eq!(settings.decode_mode(), DecodeMode::Cpu);

        settings.select_decode_mode(DecodeMode::Gpu);
        assert_eq!(settings.decode_mode(), DecodeMode::Gpu);
    }

    #[test]
    fn dng_vignette_toggle_updates_state() {
        let mut settings = GuiSettings::default();

        settings.toggle_dng_vignette();

        assert!(!settings.dng_vignette);
    }
}
