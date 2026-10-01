use std::io::{self, Read};
use std::path::PathBuf;
use std::process::{Child as ProcessHandle, Command, ExitStatus, Stdio};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::gui_process::configure_gui_process;

const UNAVAILABLE_MESSAGE: &str = "File chooser unavailable. Use drag and drop.";
const WINDOWS_CANCEL_MARKER: &str = "__MCRAW4VULKAN_CHOOSER_CANCELLED__";
const MAX_SELECTION_BYTES: usize = 4 * 1024 * 1024;
const MAX_ERROR_BYTES: usize = 8 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileChooserOutcome {
    Selected(Vec<PathBuf>),
    Cancelled,
    Unavailable(String),
    Failed(String),
}

#[derive(Debug)]
pub enum FileChooserStart {
    Pending(PendingFileChooser),
    Ready(FileChooserOutcome),
}

#[derive(Debug)]
pub enum FileChooserPoll {
    Pending,
    CleanupPending(String),
    Ready(FileChooserOutcome),
}

// A pending chooser retains its platform helper process until an outcome is
// collected. Cancellation invalidates selection immediately, but the consumer
// must keep polling until Ready acknowledges process and reader completion.
#[derive(Debug)]
pub struct PendingFileChooser {
    inner: Option<PendingFileChooserInner>,
}

#[derive(Debug)]
enum PendingFileChooserInner {
    Running(RunningChooser),
}

#[derive(Debug)]
struct RunningChooser {
    kind: ChooserKind,
    process: ProcessHandle,
    literal_executable: bool,
    stdout: ChooserReader,
    stderr: ChooserReader,
    status: Option<ExitStatus>,
    stopping: bool,
    stop_sent: bool,
    failure: Option<String>,
    next_process_poll: Instant,
}

#[derive(Debug, Default)]
struct ChooserReader {
    handle: Option<JoinHandle<io::Result<CapturedOutput>>>,
    output: Option<CapturedOutput>,
}

#[derive(Debug, Default)]
struct CapturedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChooserKind {
    #[cfg(any(target_os = "linux", test))]
    Zenity,
    #[cfg(target_os = "linux")]
    Kdialog,
    #[cfg(any(target_os = "windows", test))]
    PowerShell,
    #[cfg(any(target_os = "macos", test))]
    Osascript,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChooserCommand {
    program: &'static str,
    args: Vec<&'static str>,
    kind: ChooserKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChooserOutput {
    success: bool,
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

pub fn start_mcraw_file_chooser() -> FileChooserStart {
    start_mcraw_file_chooser_with_spawner(&mut spawn_command)
}

/// Contextual recovery for a missing separately installed encoder. This is not
/// a playlist/folder chooser and does not persist export settings.
pub fn start_ffmpeg_file_chooser() -> FileChooserStart {
    let mut outcome = start_ffmpeg_platform_chooser();
    if let FileChooserStart::Pending(chooser) = &mut outcome {
        if let Some(PendingFileChooserInner::Running(running)) = &mut chooser.inner {
            running.literal_executable = true;
        }
    }
    outcome
}

fn start_ffmpeg_platform_chooser() -> FileChooserStart {
    #[cfg(target_os = "linux")]
    {
        let zenity = ChooserCommand {
            program: "zenity",
            args: vec!["--file-selection", "--title=Locate FFmpeg executable"],
            kind: ChooserKind::Zenity,
        };
        match start_one_chooser(&zenity, &mut spawn_command) {
            FileChooserStart::Ready(FileChooserOutcome::Unavailable(_)) => start_one_chooser(
                &ChooserCommand {
                    program: "kdialog",
                    args: vec![
                        "--getopenfilename",
                        ".",
                        "*",
                        "--title",
                        "Locate FFmpeg executable",
                    ],
                    kind: ChooserKind::Kdialog,
                },
                &mut spawn_command,
            ),
            outcome => outcome,
        }
    }
    #[cfg(target_os = "windows")]
    {
        start_one_chooser(
            &ChooserCommand {
                program: "powershell.exe",
                kind: ChooserKind::PowerShell,
                args: vec![
                    "-NoProfile",
                    "-STA",
                    "-Command",
                    "$ErrorActionPreference = 'Stop'; [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.OpenFileDialog; try { $d.Title = 'Locate FFmpeg executable'; $d.Filter = 'FFmpeg (ffmpeg.exe)|ffmpeg.exe|Executables (*.exe)|*.exe'; if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.WriteLine($d.FileName) } else { [Console]::Out.WriteLine('__MCRAW4VULKAN_CHOOSER_CANCELLED__') } } finally { $d.Dispose() }",
                ],
            },
            &mut spawn_command,
        )
    }
    #[cfg(target_os = "macos")]
    {
        start_one_chooser(
            &ChooserCommand {
                program: "osascript",
                kind: ChooserKind::Osascript,
                args: vec![
                    "-e",
                    "POSIX path of (choose file with prompt \"Locate FFmpeg executable\")",
                ],
            },
            &mut spawn_command,
        )
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        FileChooserStart::Ready(FileChooserOutcome::Unavailable(
            "Executable chooser unavailable on this platform.".into(),
        ))
    }
}

fn start_mcraw_file_chooser_with_spawner(
    spawner: &mut impl FnMut(&ChooserCommand) -> io::Result<ProcessHandle>,
) -> FileChooserStart {
    #[cfg(target_os = "linux")]
    {
        let zenity = zenity_command();
        match start_one_chooser(&zenity, spawner) {
            FileChooserStart::Ready(FileChooserOutcome::Unavailable(_)) => {
                let kdialog = kdialog_command();
                start_one_chooser(&kdialog, spawner)
            }
            outcome => outcome,
        }
    }

    #[cfg(target_os = "windows")]
    {
        start_one_chooser(&powershell_command(), spawner)
    }

    #[cfg(target_os = "macos")]
    {
        start_one_chooser(&osascript_command(), spawner)
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        let _ = spawner;
        FileChooserStart::Ready(FileChooserOutcome::Unavailable(
            UNAVAILABLE_MESSAGE.to_string(),
        ))
    }
}

fn start_one_chooser(
    command: &ChooserCommand,
    spawner: &mut impl FnMut(&ChooserCommand) -> io::Result<ProcessHandle>,
) -> FileChooserStart {
    match spawner(command) {
        Ok(process) => FileChooserStart::Pending(PendingFileChooser::new(command.kind, process)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => FileChooserStart::Ready(
            FileChooserOutcome::Unavailable(UNAVAILABLE_MESSAGE.to_string()),
        ),
        Err(error) => FileChooserStart::Ready(FileChooserOutcome::Failed(format!(
            "File chooser failed: {error}"
        ))),
    }
}

fn spawn_command(command: &ChooserCommand) -> io::Result<ProcessHandle> {
    let mut process_command = Command::new(command.program);
    configure_gui_process(&mut process_command);
    process_command
        .args(&command.args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}

impl PendingFileChooser {
    fn new(kind: ChooserKind, process: ProcessHandle) -> Self {
        let mut chooser = Self::own_process(kind, process);
        chooser.start_readers();
        chooser
    }

    fn own_process(kind: ChooserKind, process: ProcessHandle) -> Self {
        Self {
            inner: Some(PendingFileChooserInner::Running(RunningChooser {
                kind,
                process,
                literal_executable: false,
                stdout: ChooserReader::default(),
                stderr: ChooserReader::default(),
                status: None,
                stopping: false,
                stop_sent: false,
                failure: None,
                next_process_poll: Instant::now(),
            })),
        }
    }

    fn start_readers(&mut self) {
        let Some(PendingFileChooserInner::Running(running)) = &mut self.inner else {
            return;
        };
        // The process already has an owner before either fallible thread start.
        let stdout = running.process.stdout.take();
        match running.start_reader(stdout, MAX_SELECTION_BYTES) {
            Ok(reader) => running.stdout.handle = Some(reader),
            Err(error) => running.fail(format!("File chooser output reader failed: {error}")),
        }
        let stderr = running.process.stderr.take();
        if !running.stopping {
            match running.start_reader(stderr, MAX_ERROR_BYTES) {
                Ok(reader) => running.stderr.handle = Some(reader),
                Err(error) => running.fail(format!("File chooser error reader failed: {error}")),
            }
        }
        // Unused endpoints close here; started readers stay with this request.
    }

    pub fn cancel(&mut self) {
        if let Some(PendingFileChooserInner::Running(running)) = &mut self.inner {
            running.cancel();
        }
    }

    pub fn poll(&mut self) -> FileChooserPoll {
        let Some(PendingFileChooserInner::Running(running)) = &mut self.inner else {
            return FileChooserPoll::Pending;
        };

        running.poll_cleanup();
        if running.status.is_none()
            || running.stdout.handle.is_some()
            || running.stderr.handle.is_some()
        {
            return match &running.failure {
                Some(error) => {
                    FileChooserPoll::CleanupPending(format!("{error} Waiting for chooser cleanup."))
                }
                None => FileChooserPoll::Pending,
            };
        }
        let Some(PendingFileChooserInner::Running(running)) = self.inner.take() else {
            unreachable!()
        };
        FileChooserPoll::Ready(collect_finished_chooser(running))
    }
}

impl Drop for PendingFileChooser {
    fn drop(&mut self) {
        // Exceptional fallback only. Orderly callers retain and poll this owner.
        // No new worker or blocking wait is safe here. OS refusal or outstanding
        // reads cannot be guaranteed resolved by destruction; report that limit.
        self.cancel();
        if let Some(PendingFileChooserInner::Running(running)) = &mut self.inner {
            running.next_process_poll = Instant::now();
            running.poll_cleanup();
            if running.status.is_none()
                || running.stdout.handle.is_some()
                || running.stderr.handle.is_some()
            {
                eprintln!(
                    "File chooser dropped before cleanup completed: {:?}",
                    running.failure
                );
            }
        }
    }
}

impl RunningChooser {
    fn cancel(&mut self) {
        if !self.stopping {
            self.stopping = true;
            self.next_process_poll = Instant::now();
        }
    }

    fn fail(&mut self, message: String) {
        self.failure.get_or_insert(message);
        self.cancel();
    }

    fn poll_cleanup(&mut self) {
        // Check both workers even after an error/panic, including before exit.
        let mut failure = None;
        for (name, reader) in [("output", &mut self.stdout), ("error", &mut self.stderr)] {
            if reader.handle.as_ref().is_some_and(JoinHandle::is_finished) {
                let result = reader
                    .handle
                    .take()
                    .unwrap()
                    .join()
                    .map_err(|_| io::Error::other("reader panicked"))
                    .and_then(|result| result);
                match result {
                    Ok(output) => reader.output = Some(output),
                    Err(error) => {
                        failure.get_or_insert_with(|| {
                            format!("File chooser {name} reader failed: {error}")
                        });
                    }
                }
            }
        }
        if let Some(error) = failure {
            self.fail(error);
        }
        if self.status.is_some() || Instant::now() < self.next_process_poll {
            return;
        }
        // Bound retries after OS errors without imposing an interaction timeout.
        self.next_process_poll = Instant::now() + Duration::from_millis(100);
        match self.try_wait() {
            Ok(status) => self.status = status,
            Err(error) => self.fail(format!("File chooser status failed: {error}")),
        }
        if self.stopping && !self.stop_sent && self.status.is_none() {
            match self.stop_process() {
                Ok(()) => self.stop_sent = true,
                Err(stop_error) => {
                    // A stop racing exit is harmless only when exit is observed.
                    match self.try_wait() {
                        Ok(Some(status)) => self.status = Some(status),
                        _ => self.fail(format!("File chooser stop failed: {stop_error}")),
                    }
                }
            }
        }
        self.next_process_poll = Instant::now() + Duration::from_millis(100);
    }

    fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.process.try_wait()
    }

    fn stop_process(&mut self) -> io::Result<()> {
        self.process.kill()
    }

    fn start_reader<R: Read + Send + 'static>(
        &mut self,
        reader: Option<R>,
        limit: usize,
    ) -> io::Result<JoinHandle<io::Result<CapturedOutput>>> {
        let reader =
            reader.ok_or_else(|| io::Error::other("File chooser output pipe unavailable."))?;
        thread::Builder::new()
            .name("file-chooser-output".into())
            .spawn(move || read_bounded(reader, limit))
    }
}

fn read_bounded(mut reader: impl Read, limit: usize) -> io::Result<CapturedOutput> {
    let mut output = CapturedOutput {
        bytes: Vec::new(),
        truncated: false,
    };
    let mut buffer = [0; 8192];
    loop {
        let count = match reader.read(&mut buffer) {
            Ok(0) => return Ok(output),
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        let retained = count.min(limit.saturating_sub(output.bytes.len()));
        output.bytes.extend_from_slice(&buffer[..retained]);
        output.truncated |= retained != count;
        // Continue draining even after the retention limit, so the helper can exit.
    }
}

fn collect_finished_chooser(running: RunningChooser) -> FileChooserOutcome {
    if let Some(error) = running.failure {
        return FileChooserOutcome::Failed(error);
    }
    if running.stopping {
        return FileChooserOutcome::Cancelled;
    }
    let status = running.status.expect("collected process status");
    let stdout = running.stdout.output.expect("collected output");
    let stderr = running.stderr.output.expect("collected diagnostics");
    if stdout.truncated {
        return FileChooserOutcome::Failed("File chooser selection is too large.".into());
    }
    let Ok(stdout) = String::from_utf8(stdout.bytes) else {
        return FileChooserOutcome::Failed("File chooser returned invalid UTF-8.".into());
    };
    let outcome = classify_output(
        running.kind,
        ChooserOutput {
            success: status.success(),
            code: status.code(),
            stdout: stdout.clone(),
            stderr: String::from_utf8_lossy(&stderr.bytes).into_owned(),
        },
    );
    if running.literal_executable && matches!(outcome, FileChooserOutcome::Selected(_)) {
        literal_executable_path(stdout.as_bytes())
    } else {
        outcome
    }
}

fn literal_executable_path(bytes: &[u8]) -> FileChooserOutcome {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return FileChooserOutcome::Failed("Executable path is not valid UTF-8.".into());
    };
    // Remove only the chooser's line terminator, never filename spaces.
    let path = text
        .strip_suffix("\r\n")
        .or_else(|| text.strip_suffix('\n'))
        .unwrap_or(text);
    if path.is_empty() || path == WINDOWS_CANCEL_MARKER {
        FileChooserOutcome::Cancelled
    } else if path.contains(['\r', '\n']) {
        FileChooserOutcome::Failed("Select one executable with a single-line path.".into())
    } else {
        FileChooserOutcome::Selected(vec![PathBuf::from(path)])
    }
}

fn classify_output(kind: ChooserKind, output: ChooserOutput) -> FileChooserOutcome {
    if output.success {
        #[cfg(any(target_os = "windows", test))]
        if kind == ChooserKind::PowerShell {
            if output.stdout == format!("{WINDOWS_CANCEL_MARKER}\r\n")
                || output.stdout == format!("{WINDOWS_CANCEL_MARKER}\n")
            {
                return FileChooserOutcome::Cancelled;
            }
            if output.stdout.is_empty()
                || !output.stdout.ends_with('\n')
                || output.stdout.lines().any(|line| {
                    line.is_empty() || line == WINDOWS_CANCEL_MARKER || line.contains(['\r', '\0'])
                })
            {
                return FileChooserOutcome::Failed("File chooser returned invalid output.".into());
            }
        }
        let paths = parse_selected_paths(&output.stdout, kind);
        if paths.is_empty() {
            FileChooserOutcome::Cancelled
        } else {
            FileChooserOutcome::Selected(paths)
        }
    } else if chooser_cancelled(kind, &output) {
        FileChooserOutcome::Cancelled
    } else {
        let detail: String = output.stderr.trim().chars().take(240).collect();
        FileChooserOutcome::Failed(if detail.is_empty() {
            format!("File chooser failed (exit {:?}).", output.code)
        } else {
            format!("File chooser failed: {detail}")
        })
    }
}

fn chooser_cancelled(kind: ChooserKind, _output: &ChooserOutput) -> bool {
    match kind {
        #[cfg(any(target_os = "linux", test))]
        ChooserKind::Zenity => _output.code == Some(1),
        #[cfg(target_os = "linux")]
        ChooserKind::Kdialog => _output.code == Some(1),
        #[cfg(any(target_os = "windows", test))]
        ChooserKind::PowerShell => false,
        #[cfg(any(target_os = "macos", test))]
        ChooserKind::Osascript => {
            _output.code == Some(1)
                && _output
                    .stderr
                    .to_ascii_lowercase()
                    .contains("user canceled")
        }
    }
}

fn parse_selected_paths(output: &str, kind: ChooserKind) -> Vec<PathBuf> {
    match kind {
        #[cfg(any(target_os = "linux", test))]
        ChooserKind::Zenity => parse_newline_paths(output),
        #[cfg(target_os = "linux")]
        ChooserKind::Kdialog => parse_newline_paths(output),
        #[cfg(any(target_os = "windows", test))]
        ChooserKind::PowerShell => parse_newline_paths(output),
        #[cfg(any(target_os = "macos", test))]
        ChooserKind::Osascript => parse_newline_paths(output),
    }
}

fn parse_newline_paths(output: &str) -> Vec<PathBuf> {
    output
        .lines()
        .filter(|line| !line.is_empty())
        .filter(|line| *line != WINDOWS_CANCEL_MARKER)
        .map(PathBuf::from)
        .collect()
}

#[cfg(any(target_os = "linux", test))]
fn zenity_command() -> ChooserCommand {
    ChooserCommand {
        program: "zenity",
        args: vec![
            "--file-selection",
            "--multiple",
            "--separator=\n",
            "--title=Add MotionCam RAW files",
            "--file-filter=MotionCam RAW files | *.mcraw *.MCRAW",
            "--file-filter=All files | *",
        ],
        kind: ChooserKind::Zenity,
    }
}

#[cfg(target_os = "linux")]
fn kdialog_command() -> ChooserCommand {
    ChooserCommand {
        program: "kdialog",
        args: vec![
            "--getopenfilename",
            ".",
            "*.mcraw *.MCRAW|MotionCam RAW files",
            "--multiple",
            "--separate-output",
        ],
        kind: ChooserKind::Kdialog,
    }
}

#[cfg(target_os = "windows")]
fn powershell_command() -> ChooserCommand {
    const SCRIPT: &str = "$ErrorActionPreference = 'Stop'; [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); Add-Type -AssemblyName System.Windows.Forms; $dialog = New-Object System.Windows.Forms.OpenFileDialog; try { $dialog.Title = 'Add MotionCam RAW files'; $dialog.Filter = 'MotionCam RAW files (*.mcraw)|*.mcraw|All files (*.*)|*.*'; $dialog.Multiselect = $true; if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.FileNames | ForEach-Object { [Console]::Out.WriteLine($_) } } else { [Console]::Out.WriteLine('__MCRAW4VULKAN_CHOOSER_CANCELLED__') } } finally { $dialog.Dispose() }";
    ChooserCommand {
        program: "powershell.exe",
        args: vec!["-NoProfile", "-STA", "-Command", SCRIPT],
        kind: ChooserKind::PowerShell,
    }
}

#[cfg(target_os = "macos")]
fn osascript_command() -> ChooserCommand {
    ChooserCommand {
        program: "osascript",
        args: vec![
            "-e",
            "set chosenFiles to choose file with prompt \"Add MotionCam RAW files\" of type {\"mcraw\"} with multiple selections allowed",
            "-e",
            "set outputText to \"\"",
            "-e",
            "repeat with chosenFile in chosenFiles",
            "-e",
            "set outputText to outputText & POSIX path of chosenFile & linefeed",
            "-e",
            "end repeat",
            "-e",
            "return outputText",
        ],
        kind: ChooserKind::Osascript,
    }
}
