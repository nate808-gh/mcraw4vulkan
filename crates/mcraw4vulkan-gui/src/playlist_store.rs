use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::playlist::{DesiredMountState, LiveMountState, Playlist, PlaylistEntry, display_stem};

pub const PLAYLIST_FILE_NAME: &str = "playlist.json";
const PLAYLIST_VERSION: u64 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaylistLoadOutcome {
    Missing,
    Loaded { playlist: Playlist },
    Unavailable { message: String },
    Invalid { message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaylistSaveOutcome {
    Saved { path: PathBuf },
    Unavailable { message: String },
    Failed { message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaylistStore {
    path: Option<PathBuf>,
    in_memory: bool,
}

impl PlaylistStore {
    pub fn from_default_config() -> Self {
        Self {
            path: default_playlist_path(),
            in_memory: false,
        }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn load(&self) -> PlaylistLoadOutcome {
        if self.in_memory {
            return PlaylistLoadOutcome::Missing;
        }

        let Some(path) = &self.path else {
            return PlaylistLoadOutcome::Unavailable {
                message: "Playlist path unavailable; playlist will be in memory only.".to_string(),
            };
        };

        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return PlaylistLoadOutcome::Missing;
            }
            Err(error) => {
                return PlaylistLoadOutcome::Invalid {
                    message: format!("Could not read playlist {}: {error}", path.display()),
                };
            }
        };

        match parse_playlist_json(&text) {
            Ok(playlist) => PlaylistLoadOutcome::Loaded { playlist },
            Err(message) => PlaylistLoadOutcome::Invalid {
                message: format!("Could not load playlist {}: {message}", path.display()),
            },
        }
    }

    pub fn save(&self, playlist: &Playlist) -> PlaylistSaveOutcome {
        if self.in_memory {
            return PlaylistSaveOutcome::Saved {
                path: PathBuf::from(PLAYLIST_FILE_NAME),
            };
        }

        let Some(path) = &self.path else {
            return PlaylistSaveOutcome::Unavailable {
                message: "Playlist path unavailable; playlist is in memory only.".to_string(),
            };
        };

        let Some(parent) = path.parent() else {
            return PlaylistSaveOutcome::Failed {
                message: format!("Playlist path has no parent: {}", path.display()),
            };
        };

        if let Err(error) = fs::create_dir_all(parent) {
            return PlaylistSaveOutcome::Failed {
                message: format!(
                    "Could not create playlist directory {}: {error}",
                    parent.display()
                ),
            };
        }

        let temp_path = parent.join(format!(
            ".{}.tmp-{}",
            PLAYLIST_FILE_NAME,
            std::process::id()
        ));
        let text = render_playlist_json(playlist);
        if let Err(error) = fs::write(&temp_path, text) {
            return PlaylistSaveOutcome::Failed {
                message: format!("Could not write playlist {}: {error}", temp_path.display()),
            };
        }
        if let Err(error) = fs::rename(&temp_path, path) {
            let _ = fs::remove_file(&temp_path);
            return PlaylistSaveOutcome::Failed {
                message: format!("Could not update playlist {}: {error}", path.display()),
            };
        }

        PlaylistSaveOutcome::Saved { path: path.clone() }
    }
}

pub fn default_playlist_path() -> Option<PathBuf> {
    if cfg!(target_os = "windows") {
        windows_playlist_path(env::var_os("APPDATA"), env::var_os("USERPROFILE"))
    } else if cfg!(target_os = "macos") {
        macos_playlist_path(env::var_os("HOME"))
    } else {
        linux_playlist_path(env::var_os("XDG_CONFIG_HOME"), env::var_os("HOME"))
    }
}

pub fn linux_playlist_path(
    xdg_config_home: Option<impl Into<PathBuf>>,
    home: Option<impl Into<PathBuf>>,
) -> Option<PathBuf> {
    if let Some(xdg_config_home) = xdg_config_home {
        let path = xdg_config_home.into();
        if !path.as_os_str().is_empty() {
            return Some(path.join("mcraw4vulkan").join(PLAYLIST_FILE_NAME));
        }
    }

    home.map(|home| {
        home.into()
            .join(".config")
            .join("mcraw4vulkan")
            .join(PLAYLIST_FILE_NAME)
    })
}

pub fn macos_playlist_path(home: Option<impl Into<PathBuf>>) -> Option<PathBuf> {
    home.map(|home| {
        home.into()
            .join("Library")
            .join("Application Support")
            .join("mcraw4vulkan")
            .join(PLAYLIST_FILE_NAME)
    })
}

pub fn windows_playlist_path(
    appdata: Option<impl Into<PathBuf>>,
    userprofile: Option<impl Into<PathBuf>>,
) -> Option<PathBuf> {
    if let Some(appdata) = appdata {
        let path = appdata.into();
        if !path.as_os_str().is_empty() {
            return Some(path.join("mcraw4vulkan").join(PLAYLIST_FILE_NAME));
        }
    }

    userprofile.map(|userprofile| {
        userprofile
            .into()
            .join("AppData")
            .join("Roaming")
            .join("mcraw4vulkan")
            .join(PLAYLIST_FILE_NAME)
    })
}

fn parse_playlist_json(text: &str) -> Result<Playlist, String> {
    let value: Value =
        serde_json::from_str(text).map_err(|error| format!("invalid JSON: {error}"))?;
    let version = value
        .get("version")
        .and_then(Value::as_u64)
        .ok_or_else(|| "missing version".to_string())?;
    if version != PLAYLIST_VERSION {
        return Err(format!("unsupported playlist version {version}"));
    }

    let entries = value
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing entries".to_string())?;

    let mut playlist = Playlist::new();
    for entry in entries {
        let Some(source_path) = entry
            .get("source_path")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        let source_path = PathBuf::from(source_path);
        let display_name = entry
            .get("display_name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map_or_else(|| display_stem(&source_path), ToOwned::to_owned);
        let desired_mount_state = entry
            .get("desired_mount_state")
            .and_then(Value::as_str)
            .and_then(DesiredMountState::parse_json_value)
            .unwrap_or_default();

        playlist.push_loaded_entry(PlaylistEntry {
            id: 0,
            source_path: source_path.clone(),
            display_name,
            clip_identity: string_field(entry, "clip_identity")
                .unwrap_or_else(|| source_path.to_string_lossy().to_string()),
            folder_name: string_field(entry, "folder_name")
                .unwrap_or_else(|| display_stem(&source_path)),
            last_seen_file_len: entry
                .get("last_seen_file_len")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            last_seen_modified_time: entry
                .get("last_seen_modified_time")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            desired_mount_state,
            live_mount_state: LiveMountState::NotLive,
        });
    }

    Ok(playlist)
}

fn string_field(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

// Persist desired state only. Session-local live state is not restored because
// the child handles that established it do not survive a GUI restart.
fn render_playlist_json(playlist: &Playlist) -> String {
    let entries = playlist
        .entries()
        .iter()
        .map(|entry| {
            json!({
                "source_path": entry.source_path.to_string_lossy(),
                "display_name": entry.display_name,
                "clip_identity": entry.clip_identity,
                "folder_name": entry.folder_name,
                "last_seen_file_len": entry.last_seen_file_len,
                "last_seen_modified_time": entry.last_seen_modified_time,
                "desired_mount_state": entry.desired_mount_state.as_str(),
            })
        })
        .collect::<Vec<_>>();

    serde_json::to_string_pretty(&json!({
        "version": PLAYLIST_VERSION,
        "entries": entries,
    }))
    .expect("playlist JSON values are serializable")
}
