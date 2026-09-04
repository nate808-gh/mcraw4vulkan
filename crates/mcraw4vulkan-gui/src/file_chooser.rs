use std::io;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use crate::gui_child_process::configure_gui_child_process;

const UNAVAILABLE_MESSAGE: &str = "File chooser unavailable. Use drag and drop.";
const WINDOWS_CANCEL_MARKER: &str = "__MCRAW4VULKAN_CHOOSER_CANCELLED__";

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
    Ready(FileChooserOutcome),
}

// A pending chooser retains its platform helper process until an outcome is
// collected; dropping it attempts to kill and reap any remaining helper.
#[derive(Debug)]
pub struct PendingFileChooser {
    inner: Option<PendingFileChooserInner>,
}

#[derive(Debug)]
enum PendingFileChooserInner {
    Running(RunningChooser),
    #[cfg(test)]
    TestPending,
}

#[derive(Debug)]
struct RunningChooser {
    kind: ChooserKind,
    child: Child,
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

fn start_mcraw_file_chooser_with_spawner(
    spawner: &mut impl FnMut(&ChooserCommand) -> io::Result<Child>,
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
    spawner: &mut impl FnMut(&ChooserCommand) -> io::Result<Child>,
) -> FileChooserStart {
    match spawner(command) {
        Ok(child) => FileChooserStart::Pending(PendingFileChooser::new(command.kind, child)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => FileChooserStart::Ready(
            FileChooserOutcome::Unavailable(UNAVAILABLE_MESSAGE.to_string()),
        ),
        Err(error) => FileChooserStart::Ready(FileChooserOutcome::Failed(format!(
            "File chooser failed: {error}"
        ))),
    }
}

fn spawn_command(command: &ChooserCommand) -> io::Result<Child> {
    let mut child_command = Command::new(command.program);
    configure_gui_child_process(&mut child_command);
    child_command
        .args(&command.args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}

impl PendingFileChooser {
    fn new(kind: ChooserKind, child: Child) -> Self {
        Self {
            inner: Some(PendingFileChooserInner::Running(RunningChooser {
                kind,
                child,
            })),
        }
    }

    pub fn poll(&mut self) -> FileChooserPoll {
        let Some(PendingFileChooserInner::Running(running)) = &mut self.inner else {
            return FileChooserPoll::Pending;
        };

        match running.child.try_wait() {
            Ok(None) => FileChooserPoll::Pending,
            Ok(Some(_status)) => {
                let Some(PendingFileChooserInner::Running(running)) = self.inner.take() else {
                    return FileChooserPoll::Ready(FileChooserOutcome::Failed(
                        "File chooser failed.".to_string(),
                    ));
                };
                FileChooserPoll::Ready(collect_finished_chooser(running))
            }
            Err(error) => FileChooserPoll::Ready(FileChooserOutcome::Failed(format!(
                "File chooser failed: {error}"
            ))),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_pending() -> Self {
        Self {
            inner: Some(PendingFileChooserInner::TestPending),
        }
    }
}

impl Drop for PendingFileChooser {
    fn drop(&mut self) {
        if let Some(PendingFileChooserInner::Running(running)) = &mut self.inner {
            let _ = running.child.kill();
            let _ = running.child.wait();
        }
    }
}

fn collect_finished_chooser(running: RunningChooser) -> FileChooserOutcome {
    match running.child.wait_with_output() {
        Ok(output) => classify_output(
            running.kind,
            ChooserOutput {
                success: output.status.success(),
                code: output.status.code(),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            },
        ),
        Err(error) => FileChooserOutcome::Failed(format!("File chooser failed: {error}")),
    }
}

fn classify_output(kind: ChooserKind, output: ChooserOutput) -> FileChooserOutcome {
    if output.success {
        let paths = parse_selected_paths(&output.stdout, kind);
        if paths.is_empty() {
            FileChooserOutcome::Cancelled
        } else {
            FileChooserOutcome::Selected(paths)
        }
    } else if chooser_cancelled(kind, &output) {
        FileChooserOutcome::Cancelled
    } else {
        FileChooserOutcome::Failed("File chooser failed.".to_string())
    }
}

fn chooser_cancelled(kind: ChooserKind, output: &ChooserOutput) -> bool {
    match kind {
        #[cfg(any(target_os = "linux", test))]
        ChooserKind::Zenity => output.code == Some(1),
        #[cfg(target_os = "linux")]
        ChooserKind::Kdialog => output.code == Some(1),
        #[cfg(any(target_os = "windows", test))]
        ChooserKind::PowerShell => output
            .stdout
            .lines()
            .any(|line| line.trim() == WINDOWS_CANCEL_MARKER),
        #[cfg(any(target_os = "macos", test))]
        ChooserKind::Osascript => {
            output.code == Some(1) && output.stderr.to_ascii_lowercase().contains("user canceled")
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
        .map(str::trim)
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
    const SCRIPT: &str = "Add-Type -AssemblyName System.Windows.Forms; $dialog = New-Object System.Windows.Forms.OpenFileDialog; $dialog.Title = 'Add MotionCam RAW files'; $dialog.Filter = 'MotionCam RAW files (*.mcraw)|*.mcraw|All files (*.*)|*.*'; $dialog.Multiselect = $true; if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.FileNames | ForEach-Object { [Console]::Out.WriteLine($_) } } else { [Console]::Out.WriteLine('__MCRAW4VULKAN_CHOOSER_CANCELLED__') }";
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_output_parses_one_path() {
        assert_eq!(
            parse_newline_paths("clips/one.mcraw\n"),
            vec![PathBuf::from("clips/one.mcraw")]
        );
    }

    #[test]
    fn selected_output_parses_multiple_paths() {
        assert_eq!(
            parse_newline_paths("clips/one.mcraw\nclips/two.mcraw\n"),
            vec![
                PathBuf::from("clips/one.mcraw"),
                PathBuf::from("clips/two.mcraw")
            ]
        );
    }

    #[test]
    fn cancel_maps_to_cancelled() {
        let output = ChooserOutput {
            success: false,
            code: Some(1),
            stdout: String::new(),
            stderr: String::new(),
        };

        assert_eq!(
            classify_output(ChooserKind::Zenity, output),
            FileChooserOutcome::Cancelled
        );
    }

    #[test]
    fn powershell_cancel_marker_maps_to_cancelled() {
        let output = ChooserOutput {
            success: true,
            code: Some(0),
            stdout: format!("{WINDOWS_CANCEL_MARKER}\n"),
            stderr: String::new(),
        };

        assert_eq!(
            classify_output(ChooserKind::PowerShell, output),
            FileChooserOutcome::Cancelled
        );
    }

    #[test]
    fn osascript_user_cancel_maps_to_cancelled() {
        let output = ChooserOutput {
            success: false,
            code: Some(1),
            stdout: String::new(),
            stderr: "execution error: User canceled. (-128)".to_string(),
        };

        assert_eq!(
            classify_output(ChooserKind::Osascript, output),
            FileChooserOutcome::Cancelled
        );
    }

    #[test]
    fn unavailable_error_maps_to_unavailable() {
        let mut spawner =
            |_: &ChooserCommand| Err(io::Error::new(io::ErrorKind::NotFound, "chooser not found"));

        let start = start_one_chooser(&zenity_command(), &mut spawner);

        assert!(matches!(
            start,
            FileChooserStart::Ready(FileChooserOutcome::Unavailable(message))
                if message == UNAVAILABLE_MESSAGE
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_tries_kdialog_when_zenity_is_unavailable() {
        let mut attempted = Vec::new();
        let mut spawner = |command: &ChooserCommand| {
            attempted.push(command.program);
            Err(io::Error::new(io::ErrorKind::NotFound, "missing"))
        };

        let start = start_mcraw_file_chooser_with_spawner(&mut spawner);

        assert_eq!(attempted, vec!["zenity", "kdialog"]);
        assert!(matches!(
            start,
            FileChooserStart::Ready(FileChooserOutcome::Unavailable(message))
                if message == UNAVAILABLE_MESSAGE
        ));
    }
}
