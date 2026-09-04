#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]
#![forbid(unsafe_code)]

use std::env;
use std::process::{Command, exit};

use mcraw4vulkan_preflight::launcher::{LauncherAction, action_for_report, parse_launcher_args};
use mcraw4vulkan_preflight::{
    PreflightOptions, RequirementNotification, RequirementNotifier, run_preflight,
    send_requirement_notification,
};

struct DesktopNotifier;

impl RequirementNotifier for DesktopNotifier {
    fn notify(&mut self, notification: &RequirementNotification) -> Result<(), String> {
        notify_rust::Notification::new()
            .summary(&notification.title)
            .body(&notification.body)
            .show()
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

fn main() {
    let config = match parse_launcher_args(env::args_os().skip(1)) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("mcraw4vulkan-preflight-check: {error}");
            exit(2);
        }
    };

    let report = run_preflight(PreflightOptions::gui_blocking_policy());
    match action_for_report(&report, config) {
        LauncherAction::Blocked(notification) => {
            let mut notifier = DesktopNotifier;
            if let Err(error) = send_requirement_notification(&mut notifier, &notification) {
                eprintln!("OS notification failed: {error}");
            }
            eprintln!("{}", notification.stderr_message());
            exit(1);
        }
        LauncherAction::Launch(config) => launch_gui(config),
    }
}

#[cfg(unix)]
fn launch_gui(config: mcraw4vulkan_preflight::launcher::LauncherConfig) {
    use std::os::unix::process::CommandExt;

    // exec replaces the launcher process, so a successful preflight leaves no parent
    // helper; this branch returns only when process replacement fails.
    let error = Command::new(&config.gui_path).args(&config.gui_args).exec();
    eprintln!(
        "mcraw4vulkan-preflight-check: failed to launch {}: {error}",
        config.gui_path.display()
    );
    exit(1);
}

#[cfg(windows)]
fn launch_gui(config: mcraw4vulkan_preflight::launcher::LauncherConfig) {
    // Windows keeps the launcher as the parent so it can propagate the GUI's
    // eventual exit status to the process that invoked the preflight helper.
    match Command::new(&config.gui_path)
        .args(&config.gui_args)
        .status()
    {
        Ok(status) => exit(status.code().unwrap_or(1)),
        Err(error) => {
            eprintln!(
                "mcraw4vulkan-preflight-check: failed to launch {}: {error}",
                config.gui_path.display()
            );
            exit(1);
        }
    }
}

#[cfg(not(any(unix, windows)))]
fn launch_gui(config: mcraw4vulkan_preflight::launcher::LauncherConfig) {
    match Command::new(&config.gui_path)
        .args(&config.gui_args)
        .status()
    {
        Ok(status) => exit(status.code().unwrap_or(1)),
        Err(error) => {
            eprintln!(
                "mcraw4vulkan-preflight-check: failed to launch {}: {error}",
                config.gui_path.display()
            );
            exit(1);
        }
    }
}
