use crate::gui_settings::{DecodeMode, OptimizerProfile};
pub use mcraw4vulkan::pipe_example::*;
use std::path::Path;

pub fn export_options(
    mode: DecodeMode,
    profile: OptimizerProfile,
    vignette: bool,
) -> mcraw4vulkan::movie_export::ExportOptions {
    mcraw4vulkan::movie_export::ExportOptions {
        backend: match mode {
            DecodeMode::Gpu => mcraw4vulkan::cli::ComputeBackendChoice::Gpu,
            DecodeMode::Cpu => mcraw4vulkan::cli::ComputeBackendChoice::Cpu,
        },
        settings: match profile {
            OptimizerProfile::Default => mcraw4vulkan::cli::SettingsSourceChoice::Default,
            OptimizerProfile::Optimized => mcraw4vulkan::cli::SettingsSourceChoice::Optimized,
        },
        vignette,
    }
}

pub fn example_for_selected_file(
    selected: Option<&Path>,
    mode: DecodeMode,
    profile: OptimizerProfile,
    vignette: bool,
) -> PipeExamplePanel {
    let options = export_options(mode, profile, vignette);
    mcraw4vulkan::pipe_example::example_for_selected_file(
        selected,
        options.backend,
        options.settings,
        vignette,
    )
}
