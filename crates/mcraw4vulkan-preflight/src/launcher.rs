use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

use crate::{PreflightReport, RequirementNotification, notification_for_report};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LauncherConfig {
    pub gui_path: PathBuf,
    pub gui_args: Vec<OsString>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LauncherAction {
    Launch(LauncherConfig),
    Blocked(RequirementNotification),
}

pub fn parse_launcher_args<I>(args: I) -> Result<LauncherConfig, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut iter = args.into_iter();
    let mut gui_args = Vec::new();

    let Some(arg) = iter.next() else {
        return Err("--gui is required".to_string());
    };

    let gui_path = if arg == OsStr::new("--gui") {
        let Some(path) = iter.next().map(PathBuf::from) else {
            return Err("--gui requires a path".to_string());
        };
        gui_args.extend(iter);
        path
    } else {
        let text = arg.to_string_lossy();
        if let Some(value) = text.strip_prefix("--gui=") {
            if value.is_empty() {
                return Err("--gui requires a path".to_string());
            }
            gui_args.extend(iter);
            PathBuf::from(value)
        } else {
            return Err(format!("unknown argument: {}", arg.to_string_lossy()));
        }
    };

    Ok(LauncherConfig { gui_path, gui_args })
}

pub fn action_for_report(report: &PreflightReport, config: LauncherConfig) -> LauncherAction {
    // The report already owns blocking policy. The launcher either preserves
    // the exact GUI request or presents the report-derived requirement notice.
    if let Some(notification) = notification_for_report(report) {
        LauncherAction::Blocked(notification)
    } else {
        LauncherAction::Launch(config)
    }
}
