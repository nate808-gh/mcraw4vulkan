use std::ffi::OsString;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::gui_child_process::configure_gui_child_process;

pub const OPTIMIZER_REPAINT_INTERVAL: Duration = Duration::from_millis(250);
pub const OPTIMIZER_OUTPUT_LIMIT_BYTES: usize = 64 * 1024;
const RESULTS_DECISION_MARKER: &str = "Decision frames:";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizerCommandSpec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
}

impl OptimizerCommandSpec {
    pub fn run(program: PathBuf, input_path: &Path) -> Self {
        Self {
            program,
            args: vec![
                OsString::from("optimizer"),
                input_path.as_os_str().to_os_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerAnswer {
    Yes,
    No,
}

impl OptimizerAnswer {
    fn line(self) -> &'static [u8] {
        match self {
            Self::Yes => b"y\n",
            Self::No => b"n\n",
        }
    }

    pub fn status(self) -> &'static str {
        match self {
            Self::Yes => "Sent Yes to optimizer.",
            Self::No => "Sent No to optimizer.",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizerProgress {
    pub step: usize,
    pub total: usize,
    pub fraction: f32,
}

pub fn parse_optimizer_progress_line(line: &str) -> Option<OptimizerProgress> {
    let line = line.trim();
    let rest = line.strip_prefix("optimizer progress: step ")?;
    let (step, rest) = rest.split_once('/')?;
    let (total, _) = rest.split_once(' ')?;
    let step = step.parse::<usize>().ok()?;
    let total = total.parse::<usize>().ok()?;
    if step == 0 || total == 0 || step > total {
        return None;
    }
    Some(OptimizerProgress {
        step,
        total,
        fraction: step as f32 / total as f32,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizerProcessEvent {
    Output(String),
    Exited {
        success: bool,
        status: String,
        output: String,
        answer: Option<OptimizerAnswer>,
        cancelled: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizerTerminalResult {
    Applied,
    AppliedThroughputGain,
    AppliedLowRamNoRegression,
    Declined,
    DeclinedThroughputGain,
    DeclinedLowRamNoRegression,
    NoChanges,
    Failed(String),
    Cancelled,
}

impl OptimizerTerminalResult {
    pub fn message(&self) -> &str {
        match self {
            Self::Applied => "Optimized settings applied.",
            Self::AppliedThroughputGain => {
                "Optimized settings applied. offset_prefetch was recommended because combined Display/PIPE throughput improved by more than 10% without either sink regressing by more than 5%."
            }
            Self::AppliedLowRamNoRegression => {
                "Optimized settings applied. offset_prefetch was recommended because total system RAM is below 17 GiB and neither Display nor PIPE regressed by more than 5%."
            }
            Self::Declined => "Current settings retained.",
            Self::DeclinedThroughputGain => {
                "Current settings retained. offset_prefetch was recommended because combined Display/PIPE throughput improved by more than 10% without either sink regressing by more than 5%."
            }
            Self::DeclinedLowRamNoRegression => {
                "Current settings retained. offset_prefetch was recommended because total system RAM is below 17 GiB and neither Display nor PIPE regressed by more than 5%."
            }
            Self::NoChanges => "No optimizer setting changes are recommended.",
            Self::Failed(message) => message.as_str(),
            Self::Cancelled => "Optimizer stopped.",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerCancelRequest {
    Requested,
    AlreadyCancelling,
    AlreadyExited,
}

#[derive(Debug)]
enum ReaderMessage {
    Output(String),
}

// The GUI owns the optimizer child, its stdin, reader threads, and run directory
// as one lifecycle; exit and cancellation join readers before deleting the directory.
#[derive(Debug)]
pub struct GuiOptimizer {
    visible: bool,
    raw_output: String,
    visible_output: String,
    visible_output_started: bool,
    awaiting_choice: bool,
    choice_sent: bool,
    sent_answer: Option<OptimizerAnswer>,
    progress: f32,
    child: Option<OptimizerChild>,
    last_exit_success: Option<bool>,
    cancelling: bool,
    terminal_result: Option<OptimizerTerminalResult>,
}

#[derive(Debug)]
struct OptimizerChild {
    child: Child,
    stdin: Option<ChildStdin>,
    reader_threads: Vec<JoinHandle<()>>,
    rx: Receiver<ReaderMessage>,
    temp_dir: PathBuf,
}

impl Default for GuiOptimizer {
    fn default() -> Self {
        Self {
            visible: false,
            raw_output: String::new(),
            visible_output: String::new(),
            visible_output_started: false,
            awaiting_choice: false,
            choice_sent: false,
            sent_answer: None,
            progress: 0.0,
            child: None,
            last_exit_success: None,
            cancelling: false,
            terminal_result: None,
        }
    }
}

impl GuiOptimizer {
    pub fn show(&mut self) {
        self.visible = true;
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn is_running(&self) -> bool {
        self.child.is_some()
    }

    pub fn is_cancelling(&self) -> bool {
        self.cancelling && self.is_running()
    }

    pub fn finishing_after_answer(&self) -> bool {
        self.choice_sent && self.is_running()
    }

    pub fn is_active(&self) -> bool {
        self.visible || self.is_running() || self.terminal_result.is_some()
    }

    pub fn progress(&self) -> f32 {
        self.progress
    }

    pub fn output(&self) -> &str {
        &self.visible_output
    }

    pub fn awaiting_choice(&self) -> bool {
        self.awaiting_choice && !self.choice_sent
    }

    pub fn choice_controls_visible(&self) -> bool {
        choice_controls_visible(self.awaiting_choice, self.choice_sent, self.is_running())
    }

    pub fn terminal_result(&self) -> Option<&OptimizerTerminalResult> {
        self.terminal_result.as_ref()
    }

    pub fn start(&mut self, spec: &OptimizerCommandSpec) -> Result<(), String> {
        if self.is_running() {
            return Err("Optimizer is already running.".to_string());
        }
        self.visible = true;
        self.raw_output.clear();
        self.visible_output.clear();
        self.visible_output_started = false;
        self.awaiting_choice = false;
        self.choice_sent = false;
        self.sent_answer = None;
        self.progress = 0.0;
        self.last_exit_success = None;
        self.cancelling = false;
        self.terminal_result = None;

        let temp_dir = create_optimizer_temp_dir()?;
        let mut command = Command::new(&spec.program);
        configure_gui_child_process(&mut command);
        command
            .args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("TMPDIR", &temp_dir)
            .env("TEMP", &temp_dir)
            .env("TMP", &temp_dir);
        let mut child = command
            .spawn()
            .map_err(|error| format!("failed to run {}: {error}", spec.program.display()))?;

        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let (tx, rx) = mpsc::channel();
        let mut reader_threads = Vec::new();
        if let Some(stdout) = stdout {
            reader_threads.push(spawn_reader(stdout, tx.clone()));
        }
        if let Some(stderr) = stderr {
            reader_threads.push(spawn_reader(stderr, tx));
        }

        self.child = Some(OptimizerChild {
            child,
            stdin,
            reader_threads,
            rx,
            temp_dir,
        });
        self.handle_output_chunk("optimizer started\n");
        Ok(())
    }

    pub fn send_answer(&mut self, answer: OptimizerAnswer) -> Result<&'static str, String> {
        if !self.choice_controls_visible() {
            return Err("Optimizer is not accepting input.".to_string());
        }
        let Some(child) = self.child.as_mut() else {
            return Err("Optimizer is not accepting input.".to_string());
        };
        let Some(stdin) = child.stdin.as_mut() else {
            return Err("Optimizer is not accepting input.".to_string());
        };
        stdin
            .write_all(answer.line())
            .map_err(|error| format!("failed to send input to optimizer: {error}"))?;
        stdin
            .flush()
            .map_err(|error| format!("failed to flush optimizer input: {error}"))?;
        self.choice_sent = true;
        self.awaiting_choice = false;
        self.sent_answer = Some(answer);
        Ok(answer.status())
    }

    pub fn request_cancel(&mut self) -> Result<OptimizerCancelRequest, String> {
        if self.cancelling {
            return Ok(OptimizerCancelRequest::AlreadyCancelling);
        }
        let Some(child) = self.child.as_mut() else {
            return Ok(OptimizerCancelRequest::AlreadyExited);
        };
        match child.child.try_wait() {
            Ok(Some(_)) => return Ok(OptimizerCancelRequest::AlreadyExited),
            Ok(None) => {}
            Err(error) => {
                return Err(format!(
                    "failed to check optimizer before stopping: {error}"
                ));
            }
        }
        child.stdin = None;
        child
            .child
            .kill()
            .map_err(|error| format!("failed to stop optimizer: {error}"))?;
        self.cancelling = true;
        self.handle_output_chunk("optimizer stopping\n");
        Ok(OptimizerCancelRequest::Requested)
    }

    pub fn poll(&mut self) -> Vec<OptimizerProcessEvent> {
        let mut events = Vec::new();
        let mut exited = None;
        let mut chunks = Vec::new();

        if let Some(child) = self.child.as_mut() {
            chunks.extend(drain_reader_messages(&child.rx, &mut events));

            match child.child.try_wait() {
                Ok(Some(status)) => {
                    exited = Some((status.success(), status.to_string()));
                }
                Ok(None) => {}
                Err(error) => {
                    exited = Some((false, error.to_string()));
                }
            }
        }
        for chunk in chunks {
            self.handle_output_chunk(&chunk);
        }

        if let Some((success, status)) = exited {
            let cancelled = self.cancelling;
            let mut child = self.child.take().expect("child exists after exit");
            join_readers(&mut child.reader_threads);
            for chunk in drain_reader_messages(&child.rx, &mut events) {
                self.handle_output_chunk(&chunk);
            }
            cleanup_temp_dir(&child.temp_dir);
            self.cancelling = false;
            self.last_exit_success = Some(success);
            if success {
                self.progress = 1.0;
            } else if !self.visible_output_started {
                self.reveal_hidden_output();
            }
            let line = format!("optimizer exited with {status}\n");
            self.handle_output_chunk(&line);
            let output = self.raw_output.clone();
            events.push(OptimizerProcessEvent::Exited {
                success,
                status,
                output,
                answer: self.sent_answer,
                cancelled,
            });
        }

        events
    }

    pub fn release_panel(&mut self) {
        self.visible = false;
        self.awaiting_choice = false;
        self.choice_sent = false;
        self.sent_answer = None;
        self.cancelling = false;
        self.terminal_result = None;
    }

    pub fn set_terminal_result(&mut self, result: OptimizerTerminalResult) {
        self.visible = true;
        self.awaiting_choice = false;
        self.choice_sent = false;
        self.sent_answer = None;
        self.cancelling = false;
        self.terminal_result = Some(result);
    }

    pub fn cleanup_on_exit(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.child.kill();
            let _ = child.child.wait();
            join_readers(&mut child.reader_threads);
            cleanup_temp_dir(&child.temp_dir);
        }
        self.release_panel();
    }

    fn handle_output_chunk(&mut self, text: &str) {
        update_progress_from_text(&mut self.progress, text);
        append_bounded_text(&mut self.raw_output, text, OPTIMIZER_OUTPUT_LIMIT_BYTES);
        if detect_pending_choice(&self.raw_output) && !self.choice_sent {
            self.awaiting_choice = true;
        }

        if self.visible_output_started {
            append_bounded_text(&mut self.visible_output, text, OPTIMIZER_OUTPUT_LIMIT_BYTES);
        } else if let Some(start) = results_block_start(&self.raw_output) {
            self.visible_output_started = true;
            self.visible_output.clear();
            append_bounded_text(
                &mut self.visible_output,
                &self.raw_output[start..],
                OPTIMIZER_OUTPUT_LIMIT_BYTES,
            );
        }
    }

    fn reveal_hidden_output(&mut self) {
        self.visible_output_started = true;
        self.visible_output.clear();
        append_bounded_text(
            &mut self.visible_output,
            &self.raw_output,
            OPTIMIZER_OUTPUT_LIMIT_BYTES,
        );
    }
}

impl Drop for GuiOptimizer {
    fn drop(&mut self) {
        self.cleanup_on_exit();
    }
}

fn drain_reader_messages(
    rx: &Receiver<ReaderMessage>,
    events: &mut Vec<OptimizerProcessEvent>,
) -> Vec<String> {
    let mut chunks = Vec::new();
    while let Ok(message) = rx.try_recv() {
        match message {
            ReaderMessage::Output(line) => {
                chunks.push(line.clone());
                events.push(OptimizerProcessEvent::Output(line));
            }
        }
    }
    chunks
}

fn update_progress_from_text(progress: &mut f32, text: &str) {
    for line in text.lines() {
        if let Some(parsed_progress) = parse_optimizer_progress_line(line) {
            *progress = parsed_progress.fraction;
        }
    }
}

pub fn detect_pending_choice(text: &str) -> bool {
    text.lines()
        .any(|line| line.trim() == "settings_file: pending user choice")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptimizerRecommendationReason {
    OffsetThroughputGain,
    OffsetLowRamNoRegression,
}

// These stable CLI tokens carry the decision reason into presentation. The GUI
// deliberately does not recompute threshold policy from measurement text.
fn parse_optimizer_recommendation_reason(text: &str) -> Option<OptimizerRecommendationReason> {
    text.lines().find_map(|line| match line.trim() {
        "recommendation_reason: offset_throughput_gain" => {
            Some(OptimizerRecommendationReason::OffsetThroughputGain)
        }
        "recommendation_reason: offset_low_ram_no_regression" => {
            Some(OptimizerRecommendationReason::OffsetLowRamNoRegression)
        }
        _ => None,
    })
}

pub(crate) fn optimizer_success_terminal_result(
    output: &str,
    answer: Option<OptimizerAnswer>,
) -> OptimizerTerminalResult {
    let reason = parse_optimizer_recommendation_reason(output);
    match (answer, reason) {
        (Some(OptimizerAnswer::Yes), Some(OptimizerRecommendationReason::OffsetThroughputGain)) => {
            OptimizerTerminalResult::AppliedThroughputGain
        }
        (
            Some(OptimizerAnswer::Yes),
            Some(OptimizerRecommendationReason::OffsetLowRamNoRegression),
        ) => OptimizerTerminalResult::AppliedLowRamNoRegression,
        (Some(OptimizerAnswer::No), Some(OptimizerRecommendationReason::OffsetThroughputGain)) => {
            OptimizerTerminalResult::DeclinedThroughputGain
        }
        (
            Some(OptimizerAnswer::No),
            Some(OptimizerRecommendationReason::OffsetLowRamNoRegression),
        ) => OptimizerTerminalResult::DeclinedLowRamNoRegression,
        (Some(OptimizerAnswer::Yes), None) => OptimizerTerminalResult::Applied,
        (Some(OptimizerAnswer::No), None) => OptimizerTerminalResult::Declined,
        (None, _) => OptimizerTerminalResult::NoChanges,
    }
}

pub fn choice_controls_visible(
    awaiting_choice: bool,
    choice_sent: bool,
    optimizer_running: bool,
) -> bool {
    awaiting_choice && !choice_sent && optimizer_running
}

fn results_block_start(text: &str) -> Option<usize> {
    let mut search_start = 0;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        let is_input_line = trimmed.starts_with("Input ") || trimmed.starts_with("Input:");
        if is_input_line && text[search_start..].contains(RESULTS_DECISION_MARKER) {
            return Some(search_start);
        }
        search_start += line.len();
    }
    let trimmed = text[search_start..].trim_start();
    let is_input_line = trimmed.starts_with("Input ") || trimmed.starts_with("Input:");
    (is_input_line && text[search_start..].contains(RESULTS_DECISION_MARKER))
        .then_some(search_start)
}

pub fn append_bounded_output(buffer: &mut String, text: &str, limit: usize) {
    buffer.push_str(text);
    if !text.ends_with('\n') {
        buffer.push('\n');
    }
    if buffer.len() > limit {
        let keep_from = buffer.len().saturating_sub(limit);
        let keep_from = buffer[keep_from..]
            .find('\n')
            .map_or(keep_from, |offset| keep_from + offset + 1);
        buffer.replace_range(..keep_from, "");
    }
}

fn append_bounded_text(buffer: &mut String, text: &str, limit: usize) {
    buffer.push_str(text);
    if buffer.len() > limit {
        let keep_from = buffer.len().saturating_sub(limit);
        let keep_from = buffer[keep_from..]
            .find('\n')
            .map_or(keep_from, |offset| keep_from + offset + 1);
        buffer.replace_range(..keep_from, "");
    }
}

fn spawn_reader<R>(reader: R, tx: Sender<ReaderMessage>) -> JoinHandle<()>
where
    R: std::io::Read + Send + 'static,
{
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let _ = tx.send(ReaderMessage::Output(line.clone()));
                }
                Err(error) => {
                    let _ = tx.send(ReaderMessage::Output(format!(
                        "optimizer output read failed: {error}\n"
                    )));
                    break;
                }
            }
        }
    })
}

fn join_readers(readers: &mut Vec<JoinHandle<()>>) {
    for reader in readers.drain(..) {
        let _ = reader.join();
    }
}

fn create_optimizer_temp_dir() -> Result<PathBuf, String> {
    let root = std::env::temp_dir().join("mcraw4vulkan-gui-optimizer");
    fs::create_dir_all(&root).map_err(|error| {
        format!(
            "failed to create optimizer temp root {}: {error}",
            root.display()
        )
    })?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let dir = root.join(format!("run-{}-{timestamp}", std::process::id()));
    fs::create_dir_all(&dir).map_err(|error| {
        format!(
            "failed to create optimizer temp dir {}: {error}",
            dir.display()
        )
    })?;
    Ok(dir)
}

fn cleanup_temp_dir(path: &Path) {
    let _ = fs::remove_dir_all(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(spec: &OptimizerCommandSpec) -> Vec<String> {
        spec.args
            .iter()
            .map(|arg| arg.to_string_lossy().to_string())
            .collect()
    }

    #[test]
    fn optimizer_command_builder_uses_structured_argv() {
        let spec = OptimizerCommandSpec::run(
            PathBuf::from("mcraw4vulkan"),
            Path::new("clips/clip one.mcraw"),
        );

        assert_eq!(spec.program, PathBuf::from("mcraw4vulkan"));
        assert_eq!(strings(&spec), vec!["optimizer", "clips/clip one.mcraw"]);
        assert!(!strings(&spec).iter().any(|arg| arg.contains("sh -c")));
        assert!(!strings(&spec).iter().any(|arg| arg.contains("PowerShell")));
        assert!(!strings(&spec).iter().any(|arg| arg == "--default"));
        assert!(!strings(&spec).iter().any(|arg| arg == "--optimized"));
    }

    #[test]
    fn progress_parser_maps_thirteen_steps_to_fractions() {
        for step in 1..=13 {
            let parsed =
                parse_optimizer_progress_line(&format!("optimizer progress: step {step}/13 label"))
                    .expect("progress parses");
            assert_eq!(parsed.step, step);
            assert_eq!(parsed.total, 13);
            assert!((parsed.fraction - step as f32 / 13.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn progress_parser_ignores_unknown_lines() {
        assert_eq!(parse_optimizer_progress_line("hello"), None);
        assert_eq!(
            parse_optimizer_progress_line("optimizer progress: step 14/13 bad"),
            None
        );
        assert_eq!(
            parse_optimizer_progress_line("optimizer progress: step 0/13 bad"),
            None
        );
    }

    #[test]
    fn bounded_output_keeps_recent_text() {
        let mut output = String::new();
        append_bounded_output(&mut output, "first line\n", 24);
        append_bounded_output(&mut output, "second line\n", 24);
        append_bounded_output(&mut output, "third line\n", 24);

        assert!(!output.contains("first line"));
        assert!(output.contains("second line") || output.contains("third line"));
        assert!(output.len() <= 24);
    }

    #[test]
    fn answer_statuses_match_keyboard_intent() {
        assert_eq!(OptimizerAnswer::Yes.status(), "Sent Yes to optimizer.");
        assert_eq!(OptimizerAnswer::No.status(), "Sent No to optimizer.");
        assert_eq!(OptimizerAnswer::Yes.line(), b"y\n");
        assert_eq!(OptimizerAnswer::No.line(), b"n\n");
    }

    #[test]
    fn output_filter_hides_progress_and_early_noise_until_results() {
        let mut optimizer = GuiOptimizer::default();

        optimizer.handle_output_chunk("wgpu_hal noisy warning\n");
        optimizer.handle_output_chunk("optimizer progress: step 1/13 gpu warmup complete\n");

        assert_eq!(optimizer.output(), "");
        assert!((optimizer.progress() - (1.0 / 13.0)).abs() < f32::EPSILON);

        optimizer.handle_output_chunk("Input: clip.mcraw\nDecision frames: 600\nresult line\n");

        assert!(!optimizer.output().contains("wgpu_hal noisy warning"));
        assert!(!optimizer.output().contains("optimizer progress"));
        assert!(optimizer.output().contains("Input: clip.mcraw"));
        assert!(optimizer.output().contains("Decision frames: 600"));
        assert!(optimizer.output().contains("result line"));
    }

    #[test]
    fn output_filter_preserves_two_profile_result_text() {
        let mut optimizer = GuiOptimizer::default();

        optimizer.handle_output_chunk(
            "Input: clip.mcraw\nDecision frames: 600\nTotal physical RAM: 17179869184 bytes (16.00 GiB)\nDefault Display median: 100.00 fps\nOffset Display median: 101.00 fps\nDisplay ratio: 1.010\nDefault PIPE median: 90.00 fps\nOffset PIPE median: 91.00 fps\nPIPE ratio: 1.011\nCombined geometric throughput ratio: 1.010\nRecommendation: offset_prefetch\nrecommendation_reason: offset_low_ram_no_regression\nReason: total system RAM is below 17 GiB, and neither Display nor PIPE regressed by more than 5%.\n",
        );

        assert!(
            optimizer
                .output()
                .contains("Total physical RAM: 17179869184 bytes (16.00 GiB)")
        );
        assert!(
            optimizer
                .output()
                .contains("Combined geometric throughput ratio: 1.010")
        );
        assert!(
            optimizer
                .output()
                .contains("Recommendation: offset_prefetch")
        );
        assert!(
            optimizer
                .output()
                .contains("recommendation_reason: offset_low_ram_no_regression")
        );
    }

    #[test]
    fn hidden_output_is_revealed_on_failure_before_results() {
        let mut optimizer = GuiOptimizer::default();

        optimizer.handle_output_chunk("early stderr before results\n");
        optimizer.reveal_hidden_output();

        assert!(optimizer.output().contains("early stderr before results"));
    }

    #[test]
    fn pending_choice_detection_accepts_only_stable_marker() {
        assert!(detect_pending_choice(
            "  settings_file: pending user choice\n"
        ));
        assert!(!detect_pending_choice(
            "settings_file: pending user choice later"
        ));
        assert!(!detect_pending_choice(
            "recommendation_reason: offset_throughput_gain"
        ));

        let mut optimizer = GuiOptimizer::default();
        optimizer.handle_output_chunk("settings_file: pending ");
        assert!(!optimizer.awaiting_choice());
        optimizer.handle_output_chunk("user choice");
        assert!(optimizer.awaiting_choice());
    }

    #[test]
    fn reason_parser_uses_only_stable_tokens_and_never_recalculates_policy() {
        let contradictory_throughput = "Display ratio: 0.100\nPIPE ratio: 0.100\nTotal physical RAM: 128.00 GiB\nrecommendation_reason: offset_throughput_gain\n";
        assert_eq!(
            parse_optimizer_recommendation_reason(contradictory_throughput),
            Some(OptimizerRecommendationReason::OffsetThroughputGain)
        );

        let contradictory_low_ram = "Display ratio: 2.000\nPIPE ratio: 2.000\nTotal physical RAM: 128.00 GiB\nrecommendation_reason: offset_low_ram_no_regression\n";
        assert_eq!(
            parse_optimizer_recommendation_reason(contradictory_low_ram),
            Some(OptimizerRecommendationReason::OffsetLowRamNoRegression)
        );

        assert_eq!(
            parse_optimizer_recommendation_reason(
                "Reason: offset won on throughput\ncombined ratio: 9.000\n"
            ),
            None
        );
        assert_eq!(
            parse_optimizer_recommendation_reason("recommendation_reason: unknown_future_reason\n"),
            None
        );
    }

    #[test]
    fn choice_buttons_are_visible_only_while_awaiting_running_and_unsent() {
        assert!(choice_controls_visible(true, false, true));
        assert!(!choice_controls_visible(false, false, true));
        assert!(!choice_controls_visible(true, true, true));
        assert!(!choice_controls_visible(true, false, false));
    }

    #[test]
    fn terminal_result_messages_are_persistent_user_facing_states() {
        assert_eq!(
            OptimizerTerminalResult::Applied.message(),
            "Optimized settings applied."
        );
        assert_eq!(
            OptimizerTerminalResult::AppliedThroughputGain.message(),
            "Optimized settings applied. offset_prefetch was recommended because combined Display/PIPE throughput improved by more than 10% without either sink regressing by more than 5%."
        );
        assert_eq!(
            OptimizerTerminalResult::AppliedLowRamNoRegression.message(),
            "Optimized settings applied. offset_prefetch was recommended because total system RAM is below 17 GiB and neither Display nor PIPE regressed by more than 5%."
        );
        assert_eq!(
            OptimizerTerminalResult::Declined.message(),
            "Current settings retained."
        );
        assert_eq!(
            OptimizerTerminalResult::DeclinedThroughputGain.message(),
            "Current settings retained. offset_prefetch was recommended because combined Display/PIPE throughput improved by more than 10% without either sink regressing by more than 5%."
        );
        assert_eq!(
            OptimizerTerminalResult::DeclinedLowRamNoRegression.message(),
            "Current settings retained. offset_prefetch was recommended because total system RAM is below 17 GiB and neither Display nor PIPE regressed by more than 5%."
        );
        assert_eq!(
            OptimizerTerminalResult::NoChanges.message(),
            "No optimizer setting changes are recommended."
        );
        assert_eq!(
            OptimizerTerminalResult::Cancelled.message(),
            "Optimizer stopped."
        );
        assert_eq!(
            OptimizerTerminalResult::Failed("Optimizer failed: exit status: 1".to_string())
                .message(),
            "Optimizer failed: exit status: 1"
        );
    }

    #[test]
    fn success_terminal_result_distinguishes_cli_reason_and_preserves_fallbacks() {
        let throughput = "recommendation_reason: offset_throughput_gain\n";
        let low_ram = "recommendation_reason: offset_low_ram_no_regression\n";

        assert_eq!(
            optimizer_success_terminal_result(throughput, Some(OptimizerAnswer::Yes)),
            OptimizerTerminalResult::AppliedThroughputGain
        );
        assert_eq!(
            optimizer_success_terminal_result(low_ram, Some(OptimizerAnswer::Yes)),
            OptimizerTerminalResult::AppliedLowRamNoRegression
        );
        assert_eq!(
            optimizer_success_terminal_result(throughput, Some(OptimizerAnswer::No)),
            OptimizerTerminalResult::DeclinedThroughputGain
        );
        assert_eq!(
            optimizer_success_terminal_result(low_ram, Some(OptimizerAnswer::No)),
            OptimizerTerminalResult::DeclinedLowRamNoRegression
        );
        assert_eq!(
            optimizer_success_terminal_result("", Some(OptimizerAnswer::Yes)),
            OptimizerTerminalResult::Applied
        );
        assert_eq!(
            optimizer_success_terminal_result("", Some(OptimizerAnswer::No)),
            OptimizerTerminalResult::Declined
        );
        assert_eq!(
            optimizer_success_terminal_result(throughput, None),
            OptimizerTerminalResult::NoChanges
        );
    }

    #[test]
    fn terminal_result_keeps_optimizer_active_until_release() {
        let mut optimizer = GuiOptimizer::default();

        optimizer.set_terminal_result(OptimizerTerminalResult::NoChanges);

        assert!(optimizer.is_active());
        assert!(optimizer.is_visible());
        assert_eq!(
            optimizer.terminal_result(),
            Some(&OptimizerTerminalResult::NoChanges)
        );

        optimizer.release_panel();

        assert!(!optimizer.is_active());
        assert!(optimizer.terminal_result().is_none());
    }

    #[test]
    fn cancelling_without_child_reports_already_exited_without_state_change() {
        let mut optimizer = GuiOptimizer::default();

        assert_eq!(
            optimizer.request_cancel(),
            Ok(OptimizerCancelRequest::AlreadyExited)
        );
        assert!(!optimizer.is_cancelling());
        assert!(!optimizer.is_running());
    }

    #[test]
    fn release_panel_clears_active_and_choice_state() {
        let mut optimizer = GuiOptimizer::default();
        optimizer.show();
        optimizer.handle_output_chunk("settings_file: pending user choice\n");

        assert!(optimizer.is_active());
        assert!(optimizer.awaiting_choice());

        optimizer.release_panel();

        assert!(!optimizer.is_active());
        assert!(!optimizer.is_visible());
        assert!(!optimizer.awaiting_choice());
        assert!(!optimizer.choice_controls_visible());
    }
}
