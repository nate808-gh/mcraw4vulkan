//! A bounded, shared owner for external PIPE -> FFmpeg movie exports.
//! The producer still owns every pixel, audio sample, source hash and sidecar.
use std::collections::{HashMap, VecDeque};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child as ProcessHandle, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::cli::{ComputeBackendChoice, SettingsSourceChoice};
use crate::pipe_example::{self, CommandTarget, ProresEncoder};
use crate::{PIPE_PRORES_FILE_SUFFIX, PIPE_PRORES_SIDECAR_SUFFIX, PipeExampleFacts};

pub const FFMPEG_PRORES_UNAVAILABLE: &str = "FFmpeg ProRes Encoding not found on this system. \"Export Movie\" will not function without FFmpeg Vulkan or VideoToolbox Encoding.";

const LOG_LIMIT: usize = 64 * 1024;
const RECORD_LIMIT: usize = 8 * 1024;
const BATCH_LOG_LIMIT: usize = 1024 * 1024;
const POLL: Duration = Duration::from_millis(50);
static ATTEMPT: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportOptions {
    pub backend: ComputeBackendChoice,
    pub settings: SettingsSourceChoice,
    pub vignette: bool,
}

#[derive(Debug, Clone)]
pub enum ExportInputs {
    Files(Vec<PathBuf>),
    Folder(PathBuf),
}

#[derive(Debug, Clone, Copy)]
pub enum ProducerLocation {
    ThisCli,
    GuiCompanion,
}

#[derive(Debug, Clone)]
pub struct ExportRequest {
    pub inputs: ExportInputs,
    pub options: ExportOptions,
    pub producer: ProducerLocation,
    /// A GUI session choice, never a new persistent setting or CLI option.
    pub ffmpeg: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Succeeded,
    Failed,
    Skipped,
    Canceled,
}

#[derive(Debug, Clone)]
pub struct JobResult {
    pub row: usize,
    pub encode_fps: Option<f64>,
    pub source: PathBuf,
    pub outcome: Outcome,
    pub reason: String,
}

impl Outcome {
    pub fn label(self) -> &'static str {
        match self {
            Self::Succeeded => "Succeeded",
            Self::Failed => "Failed",
            Self::Skipped => "Skipped",
            Self::Canceled => "Canceled",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExportRow {
    pub source: PathBuf,
    /// Last valid fps supplied by this video's encoder progress stream only.
    pub encode_fps: Option<f64>,
    pub started: bool,
    pub outcome: Option<Outcome>,
    pub reason: String,
    pub duplicate_selections: usize,
}
impl ExportRow {
    pub fn rate_text(&self) -> String {
        self.encode_fps
            .map(|fps| format!("{fps:.1} encode fps"))
            .unwrap_or_else(|| "— encode fps".into())
    }
    pub fn visible(&self) -> bool {
        self.started || self.outcome.is_some()
    }
}

/// Compact presentation facts; independent of slot admission and detailed diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExportActivity {
    pub encoding: bool,
    pub remuxing: bool,
}
impl ExportActivity {
    pub const VIDEO_LABEL: &'static str = "Video Processing";
    pub const FINALIZER_LABEL: &'static str = "Finalizer";
    pub fn video_word(self) -> &'static str {
        if self.encoding { "encoding" } else { "idle" }
    }
    pub fn finalizer_word(self) -> &'static str {
        if self.remuxing { "remuxing" } else { "idle" }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExportSnapshot {
    pub activity: ExportActivity,
    pub destination: PathBuf,
    pub rows: Vec<ExportRow>,
    pub video: Option<String>,
    pub finalizing: Option<String>,
    pub succeeded: usize,
    pub failed: usize,
    pub skipped: usize,
    pub canceled: usize,
    pub total: usize,
}

impl ExportSnapshot {
    pub fn stages_text(&self) -> String {
        format!(
            "{} [{}]  {} [{}]",
            ExportActivity::VIDEO_LABEL,
            self.activity.video_word(),
            ExportActivity::FINALIZER_LABEL,
            self.activity.finalizer_word()
        )
    }
    pub fn counts_text(&self) -> String {
        format!(
            "Succeeded [{}], failed [{}], skipped [{}], canceled [{}]",
            self.succeeded, self.failed, self.skipped, self.canceled
        )
    }
    pub fn status_text(&self) -> String {
        let mut text = String::new();
        if !self.destination.as_os_str().is_empty() {
            text.push_str(&format!("Exported to {}\n\n", self.destination.display()));
        }
        text.push_str(&format!("{}\n{}", self.stages_text(), self.counts_text()));
        if let Some(row) = self.rows.iter().rev().find(|row| row.started) {
            text.push_str(&format!(
                "\n\n{}  {}",
                row.source.display(),
                row.rate_text()
            ));
        }
        text
    }
    fn update_activity(&mut self, video: Option<&VideoSlot>, finalizer: Option<&FinalizerSlot>) {
        self.activity = ExportActivity {
            encoding: video
                .and_then(|slot| slot.encoder.as_ref())
                .is_some_and(ExportProcess::stage_active),
            remuxing: finalizer
                .and_then(|slot| slot.process.as_ref())
                .is_some_and(ExportProcess::stage_active),
        };
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExportResult {
    pub snapshot: ExportSnapshot,
    pub jobs: Vec<JobResult>,
    pub problem: Option<String>,
    pub needs_ffmpeg: bool,
    pub checked_ffmpeg: Option<PathBuf>,
    pub cancellation_requested: bool,
}

impl ExportResult {
    pub fn cancellation_message(&self) -> Option<&'static str> {
        (self.cancellation_requested
            && self.snapshot.failed == 0
            && self
                .problem
                .as_deref()
                .is_none_or(|problem| problem == "Export canceled."))
        .then_some("Export canceled.")
    }
    pub fn is_success(&self) -> bool {
        self.problem.is_none()
            && !self.cancellation_requested
            && self.snapshot.succeeded > 0
            && self.snapshot.failed == 0
            && self.snapshot.skipped == 0
            && self.snapshot.canceled == 0
    }
    fn add_source(&mut self, source: PathBuf) -> usize {
        let index = self.snapshot.rows.len();
        self.snapshot.rows.push(ExportRow {
            source,
            encode_fps: None,
            started: false,
            outcome: None,
            reason: String::new(),
            duplicate_selections: 0,
        });
        index
    }
    fn record(&mut self, row: usize, outcome: Outcome, reason: String) {
        let reason = self.record_result(row, outcome, reason);
        self.snapshot.rows[row].outcome = Some(outcome);
        self.snapshot.rows[row].reason = reason;
    }
    fn record_result(&mut self, row: usize, outcome: Outcome, reason: String) -> String {
        match outcome {
            Outcome::Succeeded => self.snapshot.succeeded += 1,
            Outcome::Failed => self.snapshot.failed += 1,
            Outcome::Skipped => self.snapshot.skipped += 1,
            Outcome::Canceled => self.snapshot.canceled += 1,
        }
        let used: usize = self.jobs.iter().map(|j| j.reason.len()).sum();
        let limit = BATCH_LOG_LIMIT.saturating_sub(used).min(LOG_LIMIT);
        let reason = truncate_text(&reason, limit.max(128));
        let progress = &self.snapshot.rows[row];
        self.jobs.push(JobResult {
            row,
            source: progress.source.clone(),
            encode_fps: progress.encode_fps,
            outcome,
            reason: reason.clone(),
        });
        reason
    }
    fn update_encoder(&mut self, slot: &VideoSlot) {
        let row = &mut self.snapshot.rows[slot.files.job.row];
        row.started = true;
        if let Some(fps) = slot.encode_fps() {
            row.encode_fps = Some(fps);
        }
    }
}

pub struct ExportHandle {
    cancel: Arc<AtomicBool>,
    snapshot: Arc<Mutex<ExportSnapshot>>,
    worker: Option<JoinHandle<ExportResult>>,
}

impl ExportHandle {
    pub fn start(request: ExportRequest, cancel: Arc<AtomicBool>) -> Result<Self> {
        let snapshot = Arc::new(Mutex::new(ExportSnapshot::default()));
        let state = Arc::clone(&snapshot);
        let token = Arc::clone(&cancel);
        let worker = thread::Builder::new()
            .name("movie-export".into())
            .spawn(move || run_batch(request, token, state))
            .context("could not start export worker")?;
        Ok(Self {
            cancel,
            snapshot,
            worker: Some(worker),
        })
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    pub fn snapshot(&self) -> ExportSnapshot {
        lock(&self.snapshot).clone()
    }
    pub fn poll(&mut self) -> Option<ExportResult> {
        if !self.worker.as_ref()?.is_finished() {
            return None;
        }
        Some(match self.worker.take()?.join() {
            Ok(result) => result,
            Err(_) => ExportResult {
                problem: Some("Export worker panicked; inspect owned temporary output.".into()),
                ..Default::default()
            },
        })
    }
}

impl Drop for ExportHandle {
    fn drop(&mut self) {
        // The GUI normally polls the close/cancel acknowledgment. This also owns
        // teardown on exceptional host errors instead of detaching the worker.
        self.cancel();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn lock<T>(value: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    value
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn truncate_text(text: &str, limit: usize) -> String {
    let mut end = text.len().min(limit);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}

pub fn is_mcraw(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("mcraw"))
}

fn discover(inputs: ExportInputs, cancel: &AtomicBool) -> Result<Vec<PathBuf>> {
    match inputs {
        ExportInputs::Files(paths) => Ok(paths),
        ExportInputs::Folder(folder) => {
            let mut paths = Vec::new();
            for entry in fs::read_dir(&folder)
                .with_context(|| format!("cannot read folder {}", folder.display()))?
            {
                ensure!(!cancel.load(Ordering::Relaxed), "Export canceled.");
                let entry = entry?;
                // Direct regular entries only: no symlink directory traversal.
                if entry.file_type()?.is_file() && is_mcraw(&entry.path()) {
                    paths.push(entry.path());
                }
            }
            paths.sort();
            Ok(paths)
        }
    }
}

#[derive(Debug)]
struct Job {
    row: usize,
    source: PathBuf,
    facts: PipeExampleFacts,
    frames: u64,
    has_audio: bool,
    movie: PathBuf,
    json: PathBuf,
}

fn absolute(path: &Path, cwd: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        cwd.join(path)
    }
}

fn plan(
    paths: Vec<PathBuf>,
    cwd: &Path,
    dest: &Path,
    result: &mut ExportResult,
    cancel: &AtomicBool,
) -> VecDeque<Job> {
    let mut seen = HashMap::<PathBuf, usize>::new();
    let mut candidates = Vec::new();
    let mut names = HashMap::<String, usize>::new();
    for path in paths {
        let source = absolute(&path, cwd);
        let identity = source.canonicalize().unwrap_or_else(|_| source.clone());
        if let Some(&row) = seen.get(&identity) {
            result.snapshot.rows[row].duplicate_selections += 1;
            result.record_result(row, Outcome::Skipped, "Duplicate source selection.".into());
            continue;
        }
        let row = result.add_source(source.clone());
        seen.insert(identity, row);
        if cancel.load(Ordering::Relaxed) {
            result.record(
                row,
                Outcome::Canceled,
                "Not exported; batch canceled.".into(),
            );
            continue;
        }
        let candidate = (|| -> Result<Option<Job>> {
            ensure!(is_mcraw(&source), "expected a .mcraw file");
            ensure!(
                fs::metadata(&source)?.is_file(),
                "source is not a regular file"
            );
            // Keep the requested basename; canonical identity is deduplication only.
            let stem = source
                .file_stem()
                .and_then(|s| s.to_str())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow!("source filename must have a nonempty UTF-8 stem"))?;
            ensure!(
                source.to_str().is_some(),
                "producer CLI requires a UTF-8 source path"
            );
            let movie = dest.join(format!("{stem}{PIPE_PRORES_FILE_SUFFIX}"));
            let json = dest.join(format!("{stem}{PIPE_PRORES_SIDECAR_SUFFIX}"));
            let (facts, frames, has_audio) = crate::pipe_cli::pipe_export_facts_for_input(&source)?;
            *names.entry(stem.to_uppercase()).or_default() += 1;
            Ok(Some(Job {
                row,
                source: source.clone(),
                facts,
                frames,
                has_audio,
                movie,
                json,
            }))
        })();
        match candidate {
            Ok(Some(job)) => candidates.push(job),
            Ok(None) => unreachable!("deduplication precedes facts"),
            Err(error) => result.record(row, Outcome::Failed, format!("{error:#}")),
        }
    }
    candidates
        .into_iter()
        .filter_map(|job| {
            let key = job
                .source
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_uppercase();
            let conflict = if names[&key] > 1 {
                Some("distinct sources have colliding output stems".to_owned())
            } else {
                [job.movie.as_path(), job.json.as_path()]
                    .iter()
                    .find_map(|path| match fs::symlink_metadata(path) {
                        Ok(_) => Some(format!("output already exists: {}", path.display())),
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                        Err(e) => Some(format!("cannot inspect output {}: {e}", path.display())),
                    })
            };
            if let Some(reason) = conflict {
                result.record(job.row, Outcome::Skipped, reason);
                None
            } else {
                Some(job)
            }
        })
        .collect()
}

fn producer_path(location: ProducerLocation) -> Result<PathBuf> {
    let exe = std::env::current_exe().context("cannot locate running application")?;
    let path = match location {
        ProducerLocation::ThisCli => exe,
        ProducerLocation::GuiCompanion => exe
            .parent()
            .ok_or_else(|| anyhow!("GUI executable has no parent"))?
            .join(if cfg!(windows) {
                "mcraw4vulkan.exe"
            } else {
                "mcraw4vulkan"
            }),
    };
    ensure!(
        path.is_file(),
        "matching package-local CLI producer is missing: {} (install the complete application package)",
        path.display()
    );
    Ok(path)
}

fn discover_ffmpeg(choice: Option<&Path>, cwd: &Path) -> Result<PathBuf> {
    select_ffmpeg(
        choice,
        cwd,
        std::env::var_os("PATH").as_deref(),
        ffmpeg_fallbacks(pipe_example::current_command_target()),
        Path::is_file,
    )
}

fn ffmpeg_fallbacks(target: CommandTarget) -> &'static [&'static str] {
    match target {
        CommandTarget::Macos => &["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"],
        CommandTarget::Linux | CommandTarget::Windows => &[],
    }
}

fn select_ffmpeg(
    choice: Option<&Path>,
    cwd: &Path,
    search_path: Option<&OsStr>,
    fallbacks: &[&str],
    mut is_file: impl FnMut(&Path) -> bool,
) -> Result<PathBuf> {
    if let Some(path) = choice {
        let path = absolute(path, cwd);
        ensure!(
            is_file(&path),
            "FFmpeg executable does not exist: {}",
            path.display()
        );
        return Ok(path);
    }
    let name = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    for directory in std::env::split_paths(search_path.unwrap_or_default()) {
        if directory.as_os_str().is_empty() {
            continue;
        }
        let path = absolute(&directory, cwd).join(name);
        if is_file(&path) {
            return Ok(path);
        }
    }
    // A selected file still has to pass the launch and encoder-list query.
    for fallback in fallbacks {
        let path = absolute(Path::new(fallback), cwd);
        if is_file(&path) {
            return Ok(path);
        }
    }
    bail!(
        "FFmpeg was not found in the application's PATH. Install a compatible FFmpeg separately and relaunch; the GUI can locate an existing executable. No software is installed automatically."
    )
}

fn configure_process(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    #[cfg(not(windows))]
    let _ = command;
}

fn ffmpeg_command(path: &Path, producer: &Path) -> Command {
    let mut command = Command::new(path);
    configure_process(&mut command);
    // Keep independent FFmpeg out of an app bundle's relocated runtime. Preserve
    // unrelated user/system search entries; never mutate the host environment.
    let directory = producer.parent().unwrap_or(Path::new(""));
    let package = if directory.file_name().is_some_and(|s| s == "bin")
        && directory
            .parent()
            .is_some_and(|p| p.join("vulkan").is_dir())
    {
        directory.parent().unwrap_or(directory)
    } else {
        directory
    };
    for key in [
        "PATH",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "VK_ICD_FILENAMES",
        "VK_DRIVER_FILES",
    ] {
        if let Some(value) = std::env::var_os(key) {
            let paths: Vec<_> = std::env::split_paths(&value)
                .filter(|p| !p.starts_with(package))
                .collect();
            if paths.is_empty() {
                command.env_remove(key);
            } else if let Ok(value) = std::env::join_paths(paths) {
                command.env(key, value);
            }
        }
    }
    command.stdin(Stdio::null());
    command
}

#[derive(Default)]
struct Output {
    tail: VecDeque<u8>,
    bytes: usize,
    error: Option<String>,
    frames: Option<u64>,
    fps: Option<f64>,
}

struct Drain {
    state: Arc<Mutex<Output>>,
    thread: Option<JoinHandle<()>>,
}

impl Drain {
    fn start(mut reader: impl Read + Send + 'static, progress: bool) -> std::io::Result<Self> {
        let state = Arc::new(Mutex::new(Output::default()));
        let shared = Arc::clone(&state);
        let thread = thread::Builder::new()
            .name("export-drain".into())
            .spawn(move || {
                let mut buffer = [0u8; 4096];
                let mut record = Vec::new();
                let mut discard = false;
                loop {
                    match reader.read(&mut buffer) {
                        Ok(0) => {
                            if progress && !discard && !record.is_empty() {
                                parse_progress(&record, &mut lock(&shared));
                            }
                            break;
                        }
                        Ok(n) => {
                            let mut out = lock(&shared);
                            out.bytes = out.bytes.saturating_add(n);
                            for &byte in &buffer[..n] {
                                if out.tail.len() == LOG_LIMIT {
                                    out.tail.pop_front();
                                }
                                out.tail.push_back(byte);
                                if progress {
                                    if byte == b'\n' {
                                        if !discard {
                                            parse_progress(&record, &mut out);
                                        }
                                        record.clear();
                                        discard = false;
                                    } else if record.len() < RECORD_LIMIT && !discard {
                                        record.push(byte);
                                    } else {
                                        record.clear();
                                        discard = true;
                                    }
                                }
                            }
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(e) => {
                            lock(&shared).error = Some(e.to_string());
                            break;
                        }
                    }
                }
            })?;
        Ok(Self {
            state,
            thread: Some(thread),
        })
    }
    fn join(&mut self) -> Result<()> {
        if let Some(thread) = self.thread.take() {
            thread.join().map_err(|_| anyhow!("log reader panicked"))?;
        }
        if let Some(error) = &lock(&self.state).error {
            bail!("log reader failed: {error}");
        }
        Ok(())
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&lock(&self.state).tail.iter().copied().collect::<Vec<_>>())
            .into_owned()
    }
}

fn parse_progress(record: &[u8], out: &mut Output) {
    let Ok(text) = std::str::from_utf8(record) else {
        return;
    };
    let Some((key, value)) = text.trim().split_once('=') else {
        return;
    };
    match key {
        "frame" => {
            if let Ok(n) = value.trim().parse() {
                out.frames = Some(n);
            }
        }
        "fps" => {
            if let Some(fps) = value
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|n| n.is_finite() && *n >= 0.0)
            {
                out.fps = Some(fps);
            }
        }
        _ => {}
    }
}

struct ExportProcess {
    process: ProcessHandle,
    status: Option<ExitStatus>,
    stderr: Drain,
    stdout: Option<Drain>,
    finished: bool,
}

fn stop_uncaptured_process(process: &mut ProcessHandle) -> Result<()> {
    let kill = match process.try_wait() {
        Ok(Some(_)) => Ok(()),
        _ => process.kill(),
    };
    let wait = process.wait();
    kill.context("Could not stop the export process")?;
    wait.context("Could not confirm that the export process has stopped")?;
    Ok(())
}

impl ExportProcess {
    fn stage_active(&self) -> bool {
        // An observed exit can precede the last buffered progress/log records.
        // A completed encoder does not inherit the producer's remaining tail.
        self.status.is_none()
            || self
                .stderr
                .thread
                .as_ref()
                .is_some_and(|reader| !reader.is_finished())
            || self
                .stdout
                .as_ref()
                .and_then(|drain| drain.thread.as_ref())
                .is_some_and(|reader| !reader.is_finished())
    }
    fn capture(mut process: ProcessHandle, capture_stdout: bool, progress: bool) -> Result<Self> {
        let mut stderr = match Drain::start(
            process.stderr.take().expect("spawn configured stderr pipe"),
            false,
        ) {
            Ok(reader) => reader,
            Err(error) => {
                stop_uncaptured_process(&mut process)
                    .context(format!("could not start stderr drain: {error}"))?;
                return Err(error).context("could not start stderr drain");
            }
        };
        let stdout = if capture_stdout {
            match Drain::start(
                process.stdout.take().expect("spawn configured stdout pipe"),
                progress,
            ) {
                Ok(reader) => Some(reader),
                Err(error) => {
                    let stopped = stop_uncaptured_process(&mut process);
                    let drained = stderr.join();
                    stopped.context(format!("could not start stdout drain: {error}"))?;
                    drained.context(format!("could not start stdout drain: {error}"))?;
                    return Err(error).context("could not start stdout drain");
                }
            }
        } else {
            None
        };
        Ok(Self {
            process,
            status: None,
            stderr,
            stdout,
            finished: false,
        })
    }
    fn poll(&mut self) -> Result<Option<ExitStatus>> {
        if let Some(error) = &lock(&self.stderr.state).error {
            bail!("stderr reader failed: {error}");
        }
        if let Some(out) = &self.stdout {
            if let Some(error) = &lock(&out.state).error {
                bail!("stdout reader failed: {error}");
            }
        }
        if self.status.is_none() {
            self.status = self.process.try_wait()?;
        }
        Ok(self.status)
    }
    fn finish(&mut self, terminate: bool) -> Result<()> {
        let mut errors = Vec::new();
        if terminate && self.status.is_none() {
            match self.process.try_wait() {
                Ok(Some(status)) => self.status = Some(status),
                Ok(None) => {
                    if let Err(e) = self.process.kill() {
                        errors.push(format!("Could not stop the export process: {e}"));
                    }
                }
                Err(e) => {
                    errors.push(format!("Could not check the export process status: {e}"));
                    if let Err(e) = self.process.kill() {
                        errors.push(format!("Could not stop the export process: {e}"));
                    }
                }
            }
        }
        match self.process.wait() {
            Ok(status) => self.status = Some(status),
            Err(e) => errors.push(format!(
                "Could not confirm that the export process has stopped: {e}"
            )),
        }
        if let Err(e) = self.stderr.join() {
            errors.push(e.to_string());
        }
        if let Some(out) = &mut self.stdout {
            if let Err(e) = out.join() {
                errors.push(e.to_string());
            }
        }
        self.finished = errors.is_empty();
        ensure!(errors.is_empty(), "{}", errors.join("; "));
        Ok(())
    }
    fn diagnostics(&self) -> String {
        self.stderr.text()
    }
}

impl Drop for ExportProcess {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.finish(true);
        }
    }
}

fn probe(path: &Path, producer: &Path, args: &[&str], cancel: &AtomicBool) -> Result<String> {
    let mut command = ffmpeg_command(path, producer);
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let process = command
        .spawn()
        .with_context(|| format!("cannot launch FFmpeg {}", path.display()))?;
    drop(command);
    let mut process = ExportProcess::capture(process, true, false)?;
    let start = Instant::now();
    let status = loop {
        if cancel.load(Ordering::Relaxed) || start.elapsed() > Duration::from_secs(5) {
            process.finish(true)?;
            if cancel.load(Ordering::Relaxed) {
                bail!("Export canceled.");
            }
            bail!("FFmpeg readiness timed out");
        }
        match process.poll() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(error) => {
                process
                    .finish(true)
                    .context(format!("{error:#}; readiness teardown failed"))?;
                return Err(error);
            }
        }
        thread::sleep(POLL);
    };
    process.finish(false)?;
    ensure!(
        status.success(),
        "FFmpeg readiness query failed: {}",
        process.diagnostics()
    );
    let out = process.stdout.as_ref().expect("probe stdout");
    ensure!(
        lock(&out.state).bytes <= LOG_LIMIT && lock(&process.stderr.state).bytes <= LOG_LIMIT,
        "FFmpeg readiness output exceeded its bound"
    );
    Ok(format!("{}\n{}", out.text(), process.diagnostics()))
}

fn select_prores_encoder(output: &str) -> Result<ProresEncoder> {
    let mut vulkan = false;
    let mut videotoolbox = false;
    for line in output.lines() {
        let mut fields = line.split_whitespace();
        let Some(flags) = fields.next() else {
            continue;
        };
        // Identify an encoder-list entry, without imposing capability flags.
        if flags.len() != 6
            || !flags.starts_with('V')
            || !flags.bytes().all(|b| b == b'.' || b.is_ascii_uppercase())
        {
            continue;
        }
        match fields.next() {
            Some("prores_ks_vulkan") => vulkan = true,
            Some("prores_videotoolbox") => videotoolbox = true,
            _ => {}
        }
    }
    tracing::info!(
        prores_ks_vulkan = vulkan,
        prores_videotoolbox = videotoolbox,
        "FFmpeg ProRes encoder support"
    );
    if videotoolbox {
        Ok(ProresEncoder::VideoToolbox)
    } else if vulkan {
        Ok(ProresEncoder::Vulkan)
    } else {
        bail!("{FFMPEG_PRORES_UNAVAILABLE}")
    }
}

fn check_ffmpeg_with_query(
    mut query: impl FnMut(&[&str]) -> Result<String>,
) -> Result<ProresEncoder> {
    let output = query(&["-hide_banner", "-encoders"])?;
    select_prores_encoder(&output)
}

fn check_ffmpeg(path: &Path, producer: &Path, cancel: &AtomicBool) -> Result<ProresEncoder> {
    tracing::info!(ffmpeg = %path.display(), "Checking FFmpeg ProRes encoder list");
    check_ffmpeg_with_query(|args| probe(path, producer, args, cancel))
}

fn owned_directory(destination: &Path) -> Result<PathBuf> {
    for _ in 0..100 {
        let sequence = ATTEMPT.fetch_add(1, Ordering::Relaxed);
        let path = destination.join(format!(
            ".mcraw4vulkan-export-{}-{sequence}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e).context("cannot create owned export directory"),
        }
    }
    bail!("could not allocate an exclusive export directory")
}

fn prepare_destination(destination: &Path) -> Result<()> {
    for record in
        crate::dng_mount_registry::DngMountRegistry::from_default_path()?.all_owned_records()?
    {
        if record.mountpoint_path == destination
            || record
                .mountpoint_path
                .canonicalize()
                .ok()
                .zip(destination.canonicalize().ok())
                .is_some_and(|(a, b)| a == b)
        {
            bail!(
                "export destination is a registered clip mount: {}",
                destination.display()
            );
        }
    }
    if let Ok(meta) = fs::symlink_metadata(destination) {
        ensure!(
            meta.is_dir() && !meta.file_type().is_symlink(),
            "export destination must be an ordinary directory: {}",
            destination.display()
        );
    }
    fs::create_dir_all(destination).context("cannot create export destination")?;
    let directory = owned_directory(destination)?;
    let check = (|| -> Result<()> {
        let source = directory.join("publication-check-a");
        let second = directory.join("publication-check-b");
        let final_path = directory.join("publication-check-final");
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&source)?
            .write_all(b"export readiness")?;
        crate::pipe_cli::publish_movie_no_replace(&source, &final_path)
            .context("destination does not support required no-replace hard-link publication")?;
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&second)?
            .write_all(b"second")?;
        ensure!(
            crate::pipe_cli::publish_movie_no_replace(&second, &final_path).is_err(),
            "destination replaced an existing publication probe"
        );
        ensure!(
            fs::read(&final_path)? == b"export readiness"
                && fs::read(&second)? == b"second"
                && !source.try_exists()?,
            "destination failed no-replace publication check"
        );
        Ok(())
    })();
    fs::remove_dir_all(&directory).with_context(|| {
        format!(
            "cannot clean readiness directory {}; export stopped",
            directory.display()
        )
    })?;
    check
}

struct JobFiles {
    job: Job,
    directory: PathBuf,
    video: PathBuf,
    mux: PathBuf,
    wav: PathBuf,
    sidecar: PathBuf,
}

impl JobFiles {
    fn create(job: Job, destination: &Path) -> Result<Self> {
        let directory = owned_directory(destination)?;
        let paths = crate::pipe_cli::pipe_sidecar_paths(&job.source, &directory);
        match paths {
            Ok((wav, sidecar)) => Ok(Self {
                video: directory.join("video.mov"),
                mux: directory.join("mux.mov"),
                wav,
                sidecar,
                job,
                directory,
            }),
            Err(e) => {
                fs::remove_dir_all(&directory)
                    .context("failed to clean export directory after layout error")?;
                Err(e)
            }
        }
    }
    fn cleanup(&self) -> Result<()> {
        fs::remove_dir_all(&self.directory).with_context(|| {
            format!(
                "cleanup failed; owned files remain in {}",
                self.directory.display()
            )
        })
    }
    fn validate(&self, encoded_frames: Option<u64>) -> Result<()> {
        let value = serde_json::from_slice(
            &fs::read(&self.sidecar).context("producer JSON is missing or unreadable")?,
        )?;
        let contract = crate::validate_pipe_sidecar_v4(&value)?;
        ensure!(
            contract.frame_width == self.job.facts.width
                && contract.frame_height == self.job.facts.height
                && contract.cadence == self.job.facts.cadence
                && contract.sample_aspect_ratio == self.job.facts.sample_aspect_ratio
                && contract.display_aspect_ratio == self.job.facts.display_aspect_ratio
                && contract.frame_count == self.job.frames,
            "producer JSON does not match planned video facts/count"
        );
        if let Some(frames) = encoded_frames {
            ensure!(
                frames == self.job.frames,
                "encoder frame count {frames} differs from expected {}",
                self.job.frames
            );
        }
        ensure!(
            contract.audio.present == self.job.has_audio,
            "producer JSON audio presence differs from indexed source"
        );
        if contract.audio.present {
            let audio =
                fs::metadata(&self.wav).context("audio-bearing clip has no finalized WAV")?;
            ensure!(
                audio.is_file() && Some(audio.len()) == contract.audio.byte_len,
                "finalized WAV length contradicts producer JSON"
            );
        } else {
            ensure!(
                fs::symlink_metadata(&self.wav)
                    .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound),
                "silent clip unexpectedly has a WAV sidecar"
            );
        }
        ensure!(
            fs::metadata(&self.video)?.is_file() && fs::metadata(&self.video)?.len() > 0,
            "encoder did not produce a video MOV"
        );
        Ok(())
    }
    fn publish(&self) -> Result<()> {
        let movie = if self.job.has_audio {
            &self.mux
        } else {
            &self.video
        };
        ensure!(
            fs::metadata(movie)?.is_file() && fs::metadata(movie)?.len() > 0,
            "final movie is missing or empty"
        );
        // Once JSON belongs to this attempt, finish the short publication step
        // even if Cancel arrives. Two links are not an atomic pair. Never remove
        // either final name on failure: a competing writer must not be harmed.
        crate::pipe_cli::publish_movie_no_replace(&self.sidecar, &self.job.json)?;
        crate::pipe_cli::publish_movie_no_replace(movie, &self.job.movie).with_context(|| {
            format!(
                "JSON was published at {}; MOV publication failed, leaving an incomplete pair",
                self.job.json.display()
            )
        })?;
        Ok(())
    }
}

struct RunnerConfig {
    producer: PathBuf,
    ffmpeg: PathBuf,
    encoder: ProresEncoder,
    cwd: PathBuf,
    options: ExportOptions,
    overlap_remux: bool,
}

fn producer_args(source: &Path, options: ExportOptions) -> Vec<OsString> {
    vec![
        "pipe".into(),
        match options.backend {
            ComputeBackendChoice::Gpu => "--gpu",
            ComputeBackendChoice::Cpu => "--cpu",
        }
        .into(),
        pipe_example::optimizer_profile_flag(options.settings).into(),
        if options.vignette {
            "--with-vig-correction"
        } else {
            "--no-vig-correction"
        }
        .into(),
        // Source is absolute, so even a basename beginning with '-' is literal.
        source.as_os_str().to_owned(),
    ]
}

struct VideoSlot {
    files: JobFiles,
    producer: Option<ExportProcess>,
    encoder: Option<ExportProcess>,
    ready: bool,
}

impl VideoSlot {
    fn start(files: JobFiles, config: &RunnerConfig) -> Result<Self> {
        let mut slot = Self {
            files,
            producer: None,
            encoder: None,
            ready: false,
        };
        let started = (|| -> Result<()> {
            // Windows ChildStdout -> Stdio adds a std-owned raw-byte relay.
            // Pass synchronous pipe endpoints directly to the two processes.
            #[cfg(windows)]
            let (reader, writer) = std::io::pipe().context("could not create video pipe")?;
            let mut command = Command::new(&config.producer);
            configure_process(&mut command);
            command
                .args(producer_args(&slot.files.job.source, config.options))
                .current_dir(&slot.files.directory)
                .stdin(Stdio::null())
                .stderr(Stdio::piped());
            #[cfg(windows)]
            command.stdout(writer);
            #[cfg(not(windows))]
            command.stdout(Stdio::piped());
            // Resolve existing relative config/home paths before changing only
            // the producer process's CWD. No separate export settings mechanism.
            for key in [
                "HOME",
                "USERPROFILE",
                "XDG_CONFIG_HOME",
                "APPDATA",
                "LOCALAPPDATA",
                "XDG_RUNTIME_DIR",
            ] {
                if let Some(value) = std::env::var_os(key) {
                    command.env(key, absolute(Path::new(&value), &config.cwd));
                }
            }
            let process = command
                .spawn()
                .context("could not start matching PIPE producer")?;
            // Both Commands must release their configured endpoints after spawn:
            // a spare writer prevents EOF; a spare reader conceals a broken pipe.
            drop(command);
            slot.producer = Some(ExportProcess::capture(process, false, false)?);
            #[cfg(not(windows))]
            let reader = slot
                .producer
                .as_mut()
                .expect("captured producer")
                .process
                .stdout
                .take()
                .ok_or_else(|| anyhow!("producer stdout pipe missing"))?;
            let mut command = ffmpeg_command(&config.ffmpeg, &config.producer);
            command
                .args(pipe_example::encoder_args_for(
                    &slot.files.job.facts,
                    config.encoder,
                ))
                .args(["-progress", "pipe:1", "-nostats"])
                .arg(&slot.files.video)
                .stdin(reader)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let process = command.spawn();
            // Command owns its configured stdin handle, including on spawn failure.
            drop(command);
            slot.encoder = Some(ExportProcess::capture(
                process.context("could not start FFmpeg encoder")?,
                true,
                true,
            )?);
            Ok(())
        })();
        if let Err(error) = started {
            slot.stop()
                .context(format!("{error:#}; partial-start teardown failed"))?;
            slot.files
                .cleanup()
                .context(format!("{error:#}; partial-start cleanup failed"))?;
            return Err(error);
        }
        Ok(slot)
    }
    fn stop(&mut self) -> Result<()> {
        let mut errors = Vec::new();
        // Kill both before joining readers: encoder and producer can otherwise
        // wait on one another's pipes during partial failure.
        for process in [&mut self.producer, &mut self.encoder]
            .into_iter()
            .flatten()
        {
            if process.status.is_none() {
                match process.process.try_wait() {
                    Ok(Some(s)) => process.status = Some(s),
                    Ok(None) => {
                        if let Err(e) = process.process.kill() {
                            errors.push(e.to_string());
                        }
                    }
                    Err(e) => {
                        errors.push(e.to_string());
                        let _ = process.process.kill();
                    }
                }
            }
        }
        for process in [&mut self.producer, &mut self.encoder]
            .into_iter()
            .flatten()
        {
            if let Err(e) = process.finish(true) {
                errors.push(e.to_string());
            }
        }
        ensure!(
            errors.is_empty(),
            "process teardown failed: {}",
            errors.join("; ")
        );
        Ok(())
    }
    fn poll(&mut self) -> Result<bool> {
        if self.ready {
            return Ok(true);
        }
        let producer = self.producer.as_mut().expect("started producer");
        let encoder = self.encoder.as_mut().expect("started encoder");
        let p = producer.poll()?;
        let e = encoder.poll()?;
        if p.is_some_and(|s| !s.success()) || e.is_some_and(|s| !s.success()) {
            self.stop()?;
            bail!(
                "video stage failed (producer {p:?}, encoder {e:?})\n{}\n{}",
                self.producer.as_ref().unwrap().diagnostics(),
                self.encoder.as_ref().unwrap().diagnostics()
            );
        }
        if p.is_some() && e.is_some() {
            producer.finish(false)?;
            encoder.finish(false)?;
            let frames = encoder.stdout.as_ref().and_then(|d| lock(&d.state).frames);
            self.files.validate(frames)?;
            self.ready = true;
            return Ok(true);
        }
        Ok(false)
    }
    fn encode_fps(&self) -> Option<f64> {
        self.encoder
            .as_ref()
            .and_then(|e| e.stdout.as_ref())
            .and_then(|d| lock(&d.state).fps)
    }
    fn status_text(&self) -> String {
        if self.ready {
            "waiting for finalizer"
        } else {
            "encoding / producer completion"
        }
        .into()
    }
}

struct FinalizerSlot {
    files: JobFiles,
    process: Option<ExportProcess>,
}

impl FinalizerSlot {
    fn start(files: JobFiles, config: &RunnerConfig) -> Result<Self> {
        let mut slot = Self {
            files,
            process: None,
        };
        if slot.files.job.has_audio {
            let mut command = ffmpeg_command(&config.ffmpeg, &config.producer);
            command
                .args(pipe_example::remux_args(
                    &slot.files.job.facts,
                    &slot.files.video,
                    &slot.files.wav,
                ))
                .args(["-progress", "pipe:1", "-nostats"])
                .arg(&slot.files.mux)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            match command.spawn() {
                Ok(process) => match ExportProcess::capture(process, true, true) {
                    Ok(process) => slot.process = Some(process),
                    Err(error) => {
                        drop(command);
                        slot.files
                            .cleanup()
                            .context("remux reader startup cleanup failed")?;
                        return Err(error);
                    }
                },
                Err(error) => {
                    drop(command);
                    slot.files
                        .cleanup()
                        .context("remux startup cleanup failed")?;
                    return Err(error).context("cannot start FFmpeg remux");
                }
            }
        }
        Ok(slot)
    }
    fn stop(&mut self) -> Result<()> {
        if let Some(process) = &mut self.process {
            process.finish(true)?;
        }
        Ok(())
    }
    fn poll(&mut self, after_remux: impl FnOnce()) -> Result<bool> {
        if let Some(process) = &mut self.process {
            let Some(status) = process.poll()? else {
                return Ok(false);
            };
            process.finish(false)?;
            ensure!(
                status.success(),
                "FFmpeg remux failed: {status}\n{}",
                process.diagnostics()
            );
        }
        after_remux();
        self.files.publish()?;
        self.files.cleanup()?;
        Ok(true)
    }
}

fn batch_blocker(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|e| e.downcast_ref::<std::io::Error>())
        .any(|e| {
            matches!(
                e.kind(),
                std::io::ErrorKind::StorageFull | std::io::ErrorKind::PermissionDenied
            )
        })
        || [
            "cleanup failed",
            "teardown failed",
            "Could not stop",
            "Could not confirm",
            "No space left",
            "not enough space",
            "VK_ERROR_DEVICE_LOST",
            "cannot start FFmpeg",
            "could not start FFmpeg",
            "could not start matching",
        ]
        .iter()
        .any(|s| format!("{error:#}").contains(s))
}

fn run_batch(
    request: ExportRequest,
    cancel: Arc<AtomicBool>,
    state: Arc<Mutex<ExportSnapshot>>,
) -> ExportResult {
    let mut result = ExportResult::default();
    let mut pending_rows = Vec::new();
    let setup = (|| -> Result<(RunnerConfig, VecDeque<Job>)> {
        let cwd = std::env::current_dir()?;
        let destination = absolute(
            &crate::dng_mount::default_dng_shared_mount_root()?.join("exported_movies"),
            &cwd,
        );
        result.snapshot.destination = destination.clone();
        let paths = discover(request.inputs, &cancel)?;
        result.snapshot.total = paths.len();
        ensure!(!paths.is_empty(), "no .mcraw files to export");
        let jobs = plan(paths, &cwd, &destination, &mut result, &cancel);
        pending_rows = jobs.iter().map(|job| job.row).collect();
        result.snapshot.video = Some("Preparing export and checking FFmpeg".into());
        *lock(&state) = result.snapshot.clone();
        ensure!(!jobs.is_empty(), "no eligible clips to export");
        let producer = producer_path(request.producer)?;
        result.needs_ffmpeg = true;
        let ffmpeg = discover_ffmpeg(request.ffmpeg.as_deref(), &cwd)?;
        let encoder = check_ffmpeg(&ffmpeg, &producer, &cancel)?;
        result.needs_ffmpeg = false;
        result.checked_ffmpeg = Some(ffmpeg.clone());
        ensure!(!cancel.load(Ordering::Relaxed), "Export canceled.");
        prepare_destination(&destination)?;
        Ok((
            RunnerConfig {
                producer,
                ffmpeg,
                encoder,
                cwd,
                options: request.options,
                overlap_remux: true,
            },
            jobs,
        ))
    })();
    let (config, jobs) = match setup {
        Ok(value) => value,
        Err(error) => {
            result.cancellation_requested = cancel.load(Ordering::Relaxed);
            if result.needs_ffmpeg && !result.cancellation_requested {
                tracing::warn!(detail = %truncate_text(&format!("{error:#}"), LOG_LIMIT),
                    "FFmpeg export dependency unavailable");
                result.problem = Some(FFMPEG_PRORES_UNAVAILABLE.into());
            } else {
                result.problem = Some(format!("{error:#}"));
            }
            for row in pending_rows {
                result.record(
                    row,
                    if result.cancellation_requested {
                        Outcome::Canceled
                    } else {
                        Outcome::Skipped
                    },
                    if result.cancellation_requested {
                        "Not exported; batch canceled."
                    } else {
                        "Not exported; preparation did not complete."
                    }
                    .into(),
                );
            }
            result.snapshot.video = None;
            *lock(&state) = result.snapshot.clone();
            return result;
        }
    };
    run_schedule(config, jobs, result, cancel, state)
}

fn run_schedule(
    config: RunnerConfig,
    mut jobs: VecDeque<Job>,
    mut result: ExportResult,
    cancel: Arc<AtomicBool>,
    state: Arc<Mutex<ExportSnapshot>>,
) -> ExportResult {
    let mut video: Option<VideoSlot> = None;
    let mut finalizer: Option<FinalizerSlot> = None;
    loop {
        if cancel.load(Ordering::Relaxed) || result.problem.is_some() {
            break;
        }
        if !result.snapshot.destination.is_dir() {
            result.problem = Some("destination is no longer accessible".into());
            break;
        }
        if let Some(slot) = &mut finalizer {
            match slot.poll(|| {
                // Publication/cleanup still own this slot; the remux stage has ended.
                result.snapshot.activity.remuxing = false;
                *lock(&state) = result.snapshot.clone();
            }) {
                Ok(false) => {}
                Ok(true) => {
                    let slot = finalizer.take().unwrap();
                    result.record(
                        slot.files.job.row,
                        Outcome::Succeeded,
                        "MOV and JSON published; temporary files cleaned".into(),
                    );
                }
                Err(error) => {
                    let mut slot = finalizer.take().unwrap();
                    let teardown = slot.stop().and_then(|()| slot.files.cleanup());
                    result.snapshot.activity.remuxing = slot
                        .process
                        .as_ref()
                        .is_some_and(ExportProcess::stage_active);
                    let mut reason = format!("{error:#}");
                    if let Err(e) = teardown {
                        reason.push_str(&format!("; cleanup failed: {e:#}"));
                        result.problem = Some(reason.clone());
                    }
                    if batch_blocker(&error) {
                        result.problem = Some(reason.clone());
                    }
                    result.record(slot.files.job.row, Outcome::Failed, reason);
                }
            }
        }
        if let Some(slot) = &mut video {
            let completion = slot.poll();
            result.update_encoder(slot);
            match completion {
                Ok(_) => {}
                Err(error) => {
                    let mut slot = video.take().unwrap();
                    let teardown = slot.stop().and_then(|()| slot.files.cleanup());
                    result.snapshot.activity.encoding = slot
                        .encoder
                        .as_ref()
                        .is_some_and(ExportProcess::stage_active);
                    result.update_encoder(&slot);
                    let mut reason = format!("{error:#}");
                    if let Err(e) = teardown {
                        reason.push_str(&format!("; cleanup failed: {e:#}"));
                        result.problem = Some(reason.clone());
                    }
                    if batch_blocker(&error) {
                        result.problem = Some(reason.clone());
                    }
                    result.record(slot.files.job.row, Outcome::Failed, reason);
                }
            }
        }
        if result.problem.is_some() || cancel.load(Ordering::Relaxed) {
            continue;
        }
        if finalizer.is_none() && video.as_ref().is_some_and(|v| v.ready) {
            let slot = video.take().unwrap();
            let row = slot.files.job.row;
            match FinalizerSlot::start(slot.files, &config) {
                Ok(slot) => finalizer = Some(slot),
                Err(error) => {
                    result.problem = Some(format!("{error:#}"));
                    result.record(row, Outcome::Failed, format!("{error:#}"));
                }
            }
        }
        if video.is_none()
            && result.problem.is_none()
            && (config.overlap_remux || finalizer.is_none())
        {
            if let Some(job) = jobs.pop_front() {
                let row = job.row;
                result.snapshot.rows[row].started = true;
                match JobFiles::create(job, &result.snapshot.destination)
                    .and_then(|files| VideoSlot::start(files, &config))
                {
                    Ok(slot) => video = Some(slot),
                    Err(error) => {
                        if batch_blocker(&error) {
                            result.problem = Some(format!("{error:#}"));
                        }
                        result.record(row, Outcome::Failed, format!("{error:#}"));
                    }
                }
            }
        }
        result
            .snapshot
            .update_activity(video.as_ref(), finalizer.as_ref());
        result.snapshot.video = video.as_ref().map(VideoSlot::status_text);
        result.snapshot.finalizing = finalizer.as_ref().map(|_| "remuxing/finalizing".into());
        *lock(&state) = result.snapshot.clone();
        if video.is_none() && finalizer.is_none() && jobs.is_empty() {
            break;
        }
        thread::sleep(POLL);
    }
    result.cancellation_requested = cancel.load(Ordering::Relaxed);
    if let Some(mut slot) = video {
        let cleanup = slot.stop().and_then(|()| slot.files.cleanup());
        result.snapshot.activity.encoding = slot
            .encoder
            .as_ref()
            .is_some_and(ExportProcess::stage_active);
        result.update_encoder(&slot);
        let outcome = if cleanup.is_ok() {
            Outcome::Canceled
        } else {
            Outcome::Failed
        };
        let reason = cleanup.err().map(|e| format!("{e:#}")).unwrap_or_else(|| {
            if result.cancellation_requested {
                "Export canceled."
            } else {
                "Export stopped because the batch failed."
            }
            .into()
        });
        result.record(slot.files.job.row, outcome, reason);
    }
    if let Some(mut slot) = finalizer {
        let cleanup = slot.stop().and_then(|()| slot.files.cleanup());
        result.snapshot.activity.remuxing = slot
            .process
            .as_ref()
            .is_some_and(ExportProcess::stage_active);
        let outcome = if cleanup.is_ok() {
            Outcome::Canceled
        } else {
            Outcome::Failed
        };
        let reason = cleanup.err().map(|e| format!("{e:#}")).unwrap_or_else(|| {
            if result.cancellation_requested {
                "Export canceled."
            } else {
                "Export stopped because the batch failed."
            }
            .into()
        });
        result.record(slot.files.job.row, outcome, reason);
    }
    for job in jobs {
        result.record(
            job.row,
            Outcome::Canceled,
            if result.cancellation_requested {
                "Not exported; batch canceled."
            } else {
                "Not exported; batch stopped after an error."
            }
            .into(),
        );
    }
    result.snapshot.video = None;
    result.snapshot.finalizing = None;
    *lock(&state) = result.snapshot.clone();
    result
}
