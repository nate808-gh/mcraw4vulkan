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

// The existing example-facts preparation owns its bounded header scan off the UI thread.
pub struct PendingPipeExample {
    pub source: std::path::PathBuf,
    receiver: std::sync::mpsc::Receiver<PipeExamplePanel>,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl PendingPipeExample {
    pub fn start(
        source: std::path::PathBuf,
        mode: DecodeMode,
        profile: OptimizerProfile,
        vignette: bool,
    ) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let token = cancel.clone();
        let path = source.clone();
        let options = export_options(mode, profile, vignette);
        std::thread::spawn(move || {
            let panel = match mcraw4vulkan::pipe_example_facts_for_input_with_cancel(&path, || {
                token.load(std::sync::atomic::Ordering::Relaxed)
            }) {
                Ok(facts) => {
                    PipeExamplePanel::Example(mcraw4vulkan::pipe_example::build_pipe_example(
                        &path,
                        facts,
                        options.backend,
                        options.settings,
                        vignette,
                        current_command_target(),
                    ))
                }
                Err(error) => {
                    PipeExamplePanel::Error(format!("Could not read PIPE example facts: {error}"))
                }
            };
            let _ = sender.send(panel);
        });
        Self {
            source,
            receiver,
            cancel,
        }
    }

    pub fn poll(&self) -> Option<PipeExamplePanel> {
        match self.receiver.try_recv() {
            Ok(panel) => Some(panel),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Some(PipeExamplePanel::Error(
                "PIPE example preparation stopped.".into(),
            )),
        }
    }
}
impl Drop for PendingPipeExample {
    fn drop(&mut self) {
        self.cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}
