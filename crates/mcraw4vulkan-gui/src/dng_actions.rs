use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crate::gui_child_process::configure_gui_child_process;
use crate::gui_settings::{DecodeMode, OptimizerProfile};

const EXIT_CLEANUP_TIMEOUT: Duration = Duration::from_secs(3);
const EXIT_CLEANUP_POLL_INTERVAL: Duration = Duration::from_millis(50);
const DNG_PROCESS_OUTPUT_TAIL_LINES: usize = 8;
const DNG_PROCESS_EXIT_DRAIN_TIMEOUT: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DngCliFlags {
    pub decode: DecodeMode,
    pub vignette: bool,
    pub optimizer_profile: OptimizerProfile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DngCommandSpec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
}

impl DngCommandSpec {
    pub fn mount(program: PathBuf, flags: DngCliFlags, input_path: &Path) -> Self {
        let mut args = Vec::with_capacity(5);
        args.push(OsString::from("dng"));
        args.push(OsString::from(match flags.decode {
            DecodeMode::Gpu => "--gpu",
            DecodeMode::Cpu => "--cpu",
        }));
        args.push(OsString::from(if flags.vignette {
            "--with-vig-correction"
        } else {
            "--no-vig-correction"
        }));
        args.push(OsString::from(flags.optimizer_profile.cli_flag()));
        args.push(input_path.as_os_str().to_os_string());
        Self { program, args }
    }

    pub fn unmount_file(program: PathBuf, input_path: &Path) -> Self {
        Self {
            program,
            args: vec![
                OsString::from("dng"),
                OsString::from("unmount"),
                input_path.as_os_str().to_os_string(),
            ],
        }
    }

    pub fn unmount_all(program: PathBuf) -> Self {
        Self {
            program,
            args: vec![
                OsString::from("dng"),
                OsString::from("unmount"),
                OsString::from("all"),
            ],
        }
    }

    pub fn spawn_silent(&self) -> Result<Child, String> {
        let mut command = Command::new(&self.program);
        configure_gui_child_process(&mut command);
        command
            .args(&self.args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("failed to run {}: {error}", self.program.display()))
    }

    pub fn spawn_capture(&self) -> Result<(Child, DngProcessOutputReader), String> {
        let mut command = Command::new(&self.program);
        configure_gui_child_process(&mut command);
        let mut child = command
            .args(&self.args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("failed to run {}: {error}", self.program.display()))?;
        let stdout = child.stdout.take().ok_or_else(|| {
            format!(
                "failed to capture DNG command stdout from {}",
                self.program.display()
            )
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            format!(
                "failed to capture DNG command stderr from {}",
                self.program.display()
            )
        })?;

        Ok((
            child,
            DngProcessOutputReader::from_stdout_stderr(stdout, stderr),
        ))
    }

    pub fn spawn_mount(&self) -> Result<(Child, DngProcessOutputReader), String> {
        let mut command = Command::new(&self.program);
        configure_gui_child_process(&mut command);
        let mut child = command
            .args(&self.args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("failed to run {}: {error}", self.program.display()))?;
        let stderr = child.stderr.take().ok_or_else(|| {
            format!(
                "failed to capture DNG mount output from {}",
                self.program.display()
            )
        })?;

        Ok((child, DngProcessOutputReader::from_stderr(stderr)))
    }
}

pub fn resolve_mcraw4vulkan_binary() -> PathBuf {
    let binary_name = if cfg!(target_os = "windows") {
        "mcraw4vulkan.exe"
    } else {
        "mcraw4vulkan"
    };

    if let Ok(current_exe) = std::env::current_exe() {
        return resolve_mcraw4vulkan_binary_from_current_exe(&current_exe, binary_name);
    }

    PathBuf::from(binary_name)
}

fn resolve_mcraw4vulkan_binary_from_current_exe(current_exe: &Path, binary_name: &str) -> PathBuf {
    if let Some(parent) = current_exe.parent() {
        let sibling = parent.join(binary_name);
        if sibling.is_file() {
            return sibling;
        }
    }

    PathBuf::from(binary_name)
}

// The controller retains mount and transient-action child handles so polling can
// turn process status and captured output into GUI lifecycle events.
#[derive(Debug)]
pub struct DngProcessController {
    mounts: Vec<DngMountChild>,
    actions: Vec<DngTransientAction>,
}

#[derive(Debug)]
struct DngMountChild {
    entry_id: u64,
    display_name: String,
    child: Child,
    output: DngProcessOutputReader,
    output_tail: Vec<String>,
    mount_path: Option<PathBuf>,
    active: bool,
    stopping: bool,
}

#[derive(Debug)]
pub struct DngProcessOutputReader {
    lines: mpsc::Receiver<String>,
}

impl DngProcessOutputReader {
    fn from_stderr(stderr: impl Read + Send + 'static) -> Self {
        let (sender, lines) = mpsc::channel();
        spawn_output_reader(stderr, sender);
        Self { lines }
    }

    fn from_stdout_stderr(
        stdout: impl Read + Send + 'static,
        stderr: impl Read + Send + 'static,
    ) -> Self {
        let (sender, lines) = mpsc::channel();
        spawn_output_reader(stdout, sender.clone());
        spawn_output_reader(stderr, sender);
        Self { lines }
    }

    fn drain(&self) -> Vec<String> {
        let mut lines = Vec::new();
        while let Ok(line) = self.lines.try_recv() {
            lines.push(line);
        }
        lines
    }

    fn drain_for(&self, timeout: Duration) -> Vec<String> {
        let deadline = Instant::now() + timeout;
        let mut lines = self.drain();

        loop {
            let now = Instant::now();
            if now >= deadline {
                return lines;
            }
            match self
                .lines
                .recv_timeout(deadline.saturating_duration_since(now))
            {
                Ok(line) => lines.push(line),
                Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => {
                    return lines;
                }
            }
        }
    }
}

fn spawn_output_reader(stream: impl Read + Send + 'static, sender: mpsc::Sender<String>) {
    thread::spawn(move || {
        let reader = BufReader::new(stream);
        for line in reader.lines().map_while(Result::ok) {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
}

#[derive(Debug)]
struct DngTransientAction {
    kind: DngActionKind,
    child: Child,
    output: DngProcessOutputReader,
    output_tail: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DngActionKind {
    UnmountFile { entry_id: u64, display_name: String },
    UnmountAll,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DngProcessEvent {
    MountBecameActive {
        entry_id: u64,
        display_name: String,
        mount_path: Option<PathBuf>,
    },
    MountExited {
        entry_id: u64,
        display_name: String,
        active: bool,
        stopping: bool,
        success: bool,
        status: String,
    },
    ActionExited {
        kind: DngActionKind,
        success: bool,
        status: String,
    },
}

impl Default for DngProcessController {
    fn default() -> Self {
        Self::new()
    }
}

impl DngProcessController {
    pub fn new() -> Self {
        Self {
            mounts: Vec::new(),
            actions: Vec::new(),
        }
    }

    pub fn has_active_work(&self) -> bool {
        !self.mounts.is_empty() || !self.actions.is_empty()
    }

    pub fn has_active_transient_action(&self) -> bool {
        !self.actions.is_empty()
    }

    pub fn has_mount_for_entry(&self, entry_id: u64) -> bool {
        self.mounts.iter().any(|mount| mount.entry_id == entry_id)
    }

    pub fn start_mount(
        &mut self,
        entry_id: u64,
        display_name: String,
        spec: &DngCommandSpec,
    ) -> Result<(), String> {
        if self.has_mount_for_entry(entry_id) {
            return Err("DNG is already mounted for this row.".to_string());
        }
        let (child, output) = spec.spawn_mount()?;
        self.mounts.push(DngMountChild {
            entry_id,
            display_name,
            child,
            output,
            output_tail: Vec::new(),
            mount_path: None,
            active: false,
            stopping: false,
        });
        Ok(())
    }

    pub fn start_unmount_file(
        &mut self,
        entry_id: u64,
        display_name: String,
        spec: &DngCommandSpec,
    ) -> Result<(), String> {
        let (child, output) = spec.spawn_capture()?;
        self.actions.push(DngTransientAction {
            kind: DngActionKind::UnmountFile {
                entry_id,
                display_name,
            },
            child,
            output,
            output_tail: Vec::new(),
        });
        self.mark_mount_stopping(entry_id);
        Ok(())
    }

    pub fn start_unmount_all(&mut self, spec: &DngCommandSpec) -> Result<(), String> {
        let (child, output) = spec.spawn_capture()?;
        self.actions.push(DngTransientAction {
            kind: DngActionKind::UnmountAll,
            child,
            output,
            output_tail: Vec::new(),
        });
        for mount in &mut self.mounts {
            mount.stopping = true;
        }
        Ok(())
    }

    pub fn mark_mount_stopping(&mut self, entry_id: u64) {
        for mount in &mut self.mounts {
            if mount.entry_id == entry_id {
                mount.stopping = true;
            }
        }
    }

    pub fn poll(&mut self) -> Vec<DngProcessEvent> {
        let mut events = Vec::new();

        let mut index = 0;
        while index < self.mounts.len() {
            for line in self.mounts[index].output.drain() {
                if let Some(mount_path) = parse_dng_mount_path_from_cli_line(&line) {
                    self.mounts[index].mount_path = Some(mount_path);
                }
                push_output_tail(&mut self.mounts[index].output_tail, line);
            }
            match self.mounts[index].child.try_wait() {
                Ok(None)
                    if !self.mounts[index].active
                        && !self.mounts[index].stopping
                        && self.mounts[index].mount_path.is_some() =>
                {
                    self.mounts[index].active = true;
                    events.push(DngProcessEvent::MountBecameActive {
                        entry_id: self.mounts[index].entry_id,
                        display_name: self.mounts[index].display_name.clone(),
                        mount_path: self.mounts[index].mount_path.clone(),
                    });
                    index += 1;
                }
                Ok(None) => {
                    index += 1;
                }
                Ok(Some(status)) => {
                    for line in self.mounts[index]
                        .output
                        .drain_for(DNG_PROCESS_EXIT_DRAIN_TIMEOUT)
                    {
                        push_output_tail(&mut self.mounts[index].output_tail, line);
                    }
                    let mount = self.mounts.remove(index);
                    let success = status.success();
                    let status = format_process_status(&status.to_string(), &mount.output_tail);
                    events.push(DngProcessEvent::MountExited {
                        entry_id: mount.entry_id,
                        display_name: mount.display_name,
                        active: mount.active,
                        stopping: mount.stopping,
                        success,
                        status,
                    });
                }
                Err(error) => {
                    for line in self.mounts[index]
                        .output
                        .drain_for(DNG_PROCESS_EXIT_DRAIN_TIMEOUT)
                    {
                        push_output_tail(&mut self.mounts[index].output_tail, line);
                    }
                    let mount = self.mounts.remove(index);
                    let status = format_process_status(&error.to_string(), &mount.output_tail);
                    events.push(DngProcessEvent::MountExited {
                        entry_id: mount.entry_id,
                        display_name: mount.display_name,
                        active: mount.active,
                        stopping: mount.stopping,
                        success: false,
                        status,
                    });
                }
            }
        }

        let mut action_index = 0;
        while action_index < self.actions.len() {
            for line in self.actions[action_index].output.drain() {
                push_output_tail(&mut self.actions[action_index].output_tail, line);
            }
            match self.actions[action_index].child.try_wait() {
                Ok(None) => action_index += 1,
                Ok(Some(status)) => {
                    for line in self.actions[action_index]
                        .output
                        .drain_for(DNG_PROCESS_EXIT_DRAIN_TIMEOUT)
                    {
                        push_output_tail(&mut self.actions[action_index].output_tail, line);
                    }
                    let action = self.actions.remove(action_index);
                    let success = status.success();
                    let status = format_process_status(&status.to_string(), &action.output_tail);
                    events.push(DngProcessEvent::ActionExited {
                        kind: action.kind,
                        success,
                        status,
                    });
                }
                Err(error) => {
                    for line in self.actions[action_index]
                        .output
                        .drain_for(DNG_PROCESS_EXIT_DRAIN_TIMEOUT)
                    {
                        push_output_tail(&mut self.actions[action_index].output_tail, line);
                    }
                    let action = self.actions.remove(action_index);
                    let status = format_process_status(&error.to_string(), &action.output_tail);
                    events.push(DngProcessEvent::ActionExited {
                        kind: action.kind,
                        success: false,
                        status,
                    });
                }
            }
        }

        events
    }

    // Give explicit unmount a bounded chance to finish, then attempt to kill and
    // reap each retained child before clearing the controller's ownership state.
    pub fn cleanup_on_exit(&mut self, unmount_all: &DngCommandSpec) {
        if let Ok(mut child) = unmount_all.spawn_silent() {
            if !wait_bounded(&mut child, EXIT_CLEANUP_TIMEOUT) {
                let _ = child.kill();
                let _ = child.wait();
            }
        }

        for mount in &mut self.mounts {
            let _ = mount.child.kill();
            let _ = mount.child.wait();
        }
        for action in &mut self.actions {
            let _ = action.child.kill();
            let _ = action.child.wait();
        }
        self.mounts.clear();
        self.actions.clear();
    }
}

fn push_output_tail(tail: &mut Vec<String>, line: String) {
    let line = line.trim().to_string();
    if line.is_empty() {
        return;
    }
    tail.push(line);
    if tail.len() > DNG_PROCESS_OUTPUT_TAIL_LINES {
        tail.remove(0);
    }
}

fn format_process_status(status: &str, output_tail: &[String]) -> String {
    if output_tail.is_empty() {
        return status.to_string();
    }

    format!("{status}; CLI output: {}", output_tail.join(" | "))
}

fn wait_bounded(child: &mut Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return true,
            Ok(None) if Instant::now() >= deadline => return false,
            Ok(None) => thread::sleep(EXIT_CLEANUP_POLL_INTERVAL),
            Err(_) => return true,
        }
    }
}

fn parse_dng_mount_path_from_cli_line(line: &str) -> Option<PathBuf> {
    let (before_suffix, _) = line.split_once("; serving in foreground")?;
    let (_, mount_path) = before_suffix.rsplit_once(" at ")?;
    (!mount_path.is_empty()).then(|| PathBuf::from(mount_path))
}

impl Drop for DngProcessController {
    fn drop(&mut self) {
        for mount in &mut self.mounts {
            let _ = mount.child.kill();
            let _ = mount.child.wait();
        }
        for action in &mut self.actions {
            let _ = action.child.kill();
            let _ = action.child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn strings(spec: &DngCommandSpec) -> Vec<String> {
        spec.args
            .iter()
            .map(|arg| arg.to_string_lossy().to_string())
            .collect()
    }

    fn windows_fixture_path(parts: &[&str]) -> String {
        parts.join("\\")
    }

    #[cfg(unix)]
    fn unix_fixture_path(parts: &[&str]) -> PathBuf {
        let mut path = PathBuf::from(std::path::MAIN_SEPARATOR.to_string());
        for part in parts {
            path.push(part);
        }
        path
    }

    #[test]
    fn mount_command_uses_dng_subcommand_and_one_file() {
        let spec = DngCommandSpec::mount(
            PathBuf::from("mcraw4vulkan"),
            DngCliFlags {
                decode: DecodeMode::Gpu,
                vignette: true,
                optimizer_profile: OptimizerProfile::Default,
            },
            Path::new("clips/a.mcraw"),
        );

        assert_eq!(
            strings(&spec),
            vec![
                "dng",
                "--gpu",
                "--with-vig-correction",
                "--default",
                "clips/a.mcraw"
            ]
        );
    }

    #[test]
    fn mount_command_maps_cpu_no_vig_and_optimized_flags() {
        let spec = DngCommandSpec::mount(
            PathBuf::from("mcraw4vulkan"),
            DngCliFlags {
                decode: DecodeMode::Cpu,
                vignette: false,
                optimizer_profile: OptimizerProfile::Optimized,
            },
            Path::new("clips/a.mcraw"),
        );

        assert_eq!(
            strings(&spec),
            vec![
                "dng",
                "--cpu",
                "--no-vig-correction",
                "--optimized",
                "clips/a.mcraw"
            ]
        );
    }

    #[test]
    fn unmount_file_command_uses_selected_file() {
        let spec =
            DngCommandSpec::unmount_file(PathBuf::from("mcraw4vulkan"), Path::new("clips/a.mcraw"));

        assert_eq!(strings(&spec), vec!["dng", "unmount", "clips/a.mcraw"]);
    }

    #[test]
    fn unmount_all_command_uses_all_literal() {
        let spec = DngCommandSpec::unmount_all(PathBuf::from("mcraw4vulkan"));

        assert_eq!(strings(&spec), vec!["dng", "unmount", "all"]);
    }

    #[cfg(unix)]
    #[test]
    fn transient_action_busy_query_tracks_unmount_action_only() {
        let mut controller = DngProcessController::new();
        let spec = DngCommandSpec {
            program: unix_fixture_path(&["bin", "sh"]),
            args: vec![OsString::from("-c"), OsString::from("exit 0")],
        };

        assert!(!controller.has_active_transient_action());
        controller.start_unmount_all(&spec).unwrap();
        assert!(controller.has_active_transient_action());

        let event = poll_until_event(&mut controller);
        assert!(matches!(
            event,
            DngProcessEvent::ActionExited {
                kind: DngActionKind::UnmountAll,
                success: true,
                ..
            }
        ));
        assert!(!controller.has_active_transient_action());
    }

    #[test]
    fn command_specs_are_structured_argv_not_shell_text() {
        let spec = DngCommandSpec::mount(
            PathBuf::from("mcraw4vulkan"),
            DngCliFlags {
                decode: DecodeMode::Gpu,
                vignette: true,
                optimizer_profile: OptimizerProfile::Default,
            },
            Path::new("clips/a file.mcraw"),
        );

        assert_eq!(spec.program, PathBuf::from("mcraw4vulkan"));
        assert!(!strings(&spec).iter().any(|arg| arg.contains("sh -c")));
        assert!(!strings(&spec).iter().any(|arg| arg.contains("PowerShell")));
        assert_eq!(strings(&spec).len(), 5);
    }

    #[test]
    fn parses_exact_child_mount_path_from_cli_output() {
        let input = windows_fixture_path(&["D:", "Media", "clip"]);
        let mount =
            windows_fixture_path(&["C:", "Users", "example", "mcraw4vulkan", "clip take__1234"]);
        let line = format!("mounted {input} at {mount}; serving in foreground");

        assert_eq!(
            parse_dng_mount_path_from_cli_line(&line),
            Some(PathBuf::from(mount))
        );
    }

    #[test]
    fn mount_path_parser_uses_last_at_separator() {
        let input = windows_fixture_path(&["D:", "Media", "clip at noon.mcraw"]);
        let mount = windows_fixture_path(&["C:", "Users", "example", "mcraw4vulkan", "clip__1234"]);
        let line = format!("mounted {input} at {mount}; serving in foreground");

        assert_eq!(
            parse_dng_mount_path_from_cli_line(&line),
            Some(PathBuf::from(mount))
        );
    }

    #[test]
    fn process_status_includes_bounded_output_tail() {
        let mut tail = Vec::new();
        for index in 0..10 {
            push_output_tail(&mut tail, format!("line {index}"));
        }

        let status = format_process_status("exit status: 1", &tail);

        assert!(!status.contains("line 0"));
        assert!(!status.contains("line 1"));
        assert!(status.contains("line 2"));
        assert!(status.contains("line 9"));
        assert!(status.contains("CLI output"));
    }

    #[test]
    fn resolve_binary_chooses_sibling_public_wrapper() {
        let root = std::env::temp_dir().join(format!(
            "mcraw4vulkan-gui-resolve-cli-{}",
            std::process::id()
        ));
        let bin = root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        let cli = bin.join(if cfg!(target_os = "windows") {
            "mcraw4vulkan.exe"
        } else {
            "mcraw4vulkan"
        });
        fs::write(&cli, b"").unwrap();
        let gui = bin.join(if cfg!(target_os = "windows") {
            "mcraw4vulkan-gui.exe"
        } else {
            "mcraw4vulkan-gui-macos"
        });

        let resolved = resolve_mcraw4vulkan_binary_from_current_exe(
            &gui,
            cli.file_name().unwrap().to_str().unwrap(),
        );

        assert_eq!(resolved, cli);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn failed_mount_event_includes_stderr_tail() {
        let mut controller = DngProcessController::new();
        let spec = DngCommandSpec {
            program: unix_fixture_path(&["bin", "sh"]),
            args: vec![
                OsString::from("-c"),
                OsString::from("echo stale parent mount >&2; exit 7"),
            ],
        };
        controller
            .start_mount(42, "clip".to_string(), &spec)
            .unwrap();

        let event = poll_until_event(&mut controller);

        let DngProcessEvent::MountExited {
            success, status, ..
        } = event
        else {
            panic!("expected mount exit event");
        };
        assert!(!success);
        assert!(status.contains("stale parent mount"));
    }

    #[cfg(unix)]
    #[test]
    fn failed_unmount_event_includes_stderr_tail() {
        let mut controller = DngProcessController::new();
        let spec = DngCommandSpec {
            program: unix_fixture_path(&["bin", "sh"]),
            args: vec![
                OsString::from("-c"),
                OsString::from("echo duplicate mount >&2; exit 8"),
            ],
        };
        controller.start_unmount_all(&spec).unwrap();

        let event = poll_until_event(&mut controller);

        let DngProcessEvent::ActionExited {
            success, status, ..
        } = event
        else {
            panic!("expected unmount action exit event");
        };
        assert!(!success);
        assert!(status.contains("duplicate mount"));
    }

    #[cfg(unix)]
    fn poll_until_event(controller: &mut DngProcessController) -> DngProcessEvent {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(event) = controller.poll().into_iter().next() {
                return event;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for DNG process event"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }
}
