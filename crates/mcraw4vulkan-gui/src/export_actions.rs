use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use mcraw4vulkan::movie_export::{
    ExportHandle, ExportInputs, ExportOptions, ExportRequest, ExportResult, ExportSnapshot,
    ProducerLocation,
};

use crate::file_chooser::{
    self, FileChooserOutcome, FileChooserPoll, FileChooserStart, PendingFileChooser,
};

#[derive(Default)]
pub struct GuiMovieExport {
    handle: Option<ExportHandle>,
    request: Option<ExportRequest>,
    ffmpeg: Option<PathBuf>,
    chooser: Option<PendingFileChooser>,
    recovery: bool,
    chooser_request: Option<ExportRequest>,
    retry: Option<ExportRequest>,
    pub result: Option<ExportResult>,
    pub message: Option<String>,
}

impl GuiMovieExport {
    pub fn is_running(&self) -> bool {
        self.handle.is_some()
    }
    pub fn needs_poll(&self) -> bool {
        self.is_running() || self.chooser.is_some()
    }
    pub fn options_locked(&self) -> bool {
        self.needs_poll() || self.recovery || self.chooser_request.is_some() || self.retry.is_some()
    }
    pub fn can_start(&self) -> bool {
        !self.needs_poll() && self.chooser_request.is_none() && self.retry.is_none()
    }
    pub fn new_request(&self, paths: Vec<PathBuf>, options: ExportOptions) -> ExportRequest {
        ExportRequest {
            inputs: ExportInputs::Files(paths),
            options,
            producer: ProducerLocation::GuiCompanion,
            ffmpeg: self.ffmpeg.clone(),
        }
    }
    // GuiApp owns admission for both new requests and retries. This second guard
    // protects handle/request ownership against accidental reentrant callers.
    pub fn start_snapshot(&mut self, request: ExportRequest) -> Result<(), String> {
        if !self.can_start() {
            return Err("An export or executable choice is already pending.".into());
        }
        let handle = ExportHandle::start(request.clone(), Arc::new(AtomicBool::new(false)))
            .map_err(|e| e.to_string())?;
        self.handle = Some(handle);
        self.request = Some(request);
        self.result = None;
        self.message = None;
        self.recovery = false;
        Ok(())
    }
    pub fn take_retry(&mut self) -> Option<ExportRequest> {
        self.retry.take()
    }
    pub(crate) fn begin_choice(&mut self) -> bool {
        if !self.recovery || !self.can_start() {
            return false;
        }
        self.chooser_request = self.request.clone();
        self.chooser_request.is_some()
    }
    pub fn cancel(&mut self) {
        if let Some(handle) = &self.handle {
            handle.cancel();
        }
        self.recovery = false;
        if let Some(chooser) = &mut self.chooser {
            chooser.cancel();
        }
        self.chooser_request = None;
        self.retry = None;
        self.request = None;
    }
    pub fn snapshot(&self) -> ExportSnapshot {
        self.handle
            .as_ref()
            .map(ExportHandle::snapshot)
            .or_else(|| self.result.as_ref().map(|r| r.snapshot.clone()))
            .unwrap_or_default()
    }
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        if let Some(result) = self.handle.as_mut().and_then(ExportHandle::poll) {
            self.handle = None;
            self.finish_result(result);
            changed = true;
        }
        if let Some(chooser) = &mut self.chooser {
            match chooser.poll() {
                FileChooserPoll::Pending => {}
                FileChooserPoll::CleanupPending(message) => {
                    if self.message.as_ref() != Some(&message) {
                        self.message = Some(message);
                        changed = true;
                    }
                }
                FileChooserPoll::Ready(outcome) => {
                    self.chooser = None;
                    // Cleanup errors remain visible after retry authority is gone.
                    if let FileChooserOutcome::Failed(error) = &outcome {
                        self.message = Some(error.clone());
                    }
                    self.apply_choice(outcome);
                    changed = true;
                }
            }
        }
        changed
    }
    fn finish_result(&mut self, result: ExportResult) {
        if let Some(path) = &result.checked_ffmpeg {
            if self.request.is_some() {
                self.ffmpeg = Some(path.clone());
            }
        }
        self.recovery =
            self.request.is_some() && result.needs_ffmpeg && !result.cancellation_requested;
        self.message = result
            .problem
            .clone()
            .or_else(|| result.cancellation_message().map(str::to_owned));
        self.result = Some(result);
    }
    pub(crate) fn apply_choice(&mut self, outcome: FileChooserOutcome) {
        // A result has authority only over the snapshot taken when its chooser
        // opened. Cancel discards this authority, including same-frame results.
        let Some(mut request) = self.chooser_request.take() else {
            return;
        };
        match outcome {
            FileChooserOutcome::Selected(paths) if paths.len() == 1 => {
                request.ffmpeg = Some(paths[0].clone());
                self.retry = Some(request);
            }
            FileChooserOutcome::Failed(error) | FileChooserOutcome::Unavailable(error) => {
                self.message = Some(error)
            }
            FileChooserOutcome::Cancelled => self.cancel(),
            _ => self.message = Some("Select one FFmpeg executable.".into()),
        }
    }
    pub fn recovery_dialog(&mut self, context: &egui::Context) {
        if !self.recovery || self.is_running() {
            return;
        }
        egui::Window::new("FFmpeg required")
            .collapsible(false)
            .resizable(false)
            .show(context, |ui| {
                ui.label(
                    self.message
                        .as_deref()
                        .unwrap_or("Locate a separately installed compatible FFmpeg executable."),
                );
                ui.label("The checked executable is remembered for this GUI session only.");
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(self.can_start(), egui::Button::new("Locate FFmpeg…"))
                        .clicked()
                    {
                        if !self.begin_choice() {
                            return;
                        }
                        match file_chooser::start_ffmpeg_file_chooser() {
                            FileChooserStart::Pending(chooser) => self.chooser = Some(chooser),
                            FileChooserStart::Ready(outcome) => self.apply_choice(outcome),
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        self.cancel();
                    }
                });
            });
    }
}
