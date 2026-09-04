use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::main_view;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DesiredMountState {
    Mounted,
    #[default]
    Unmounted,
}

impl DesiredMountState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mounted => "mounted",
            Self::Unmounted => "unmounted",
        }
    }

    pub fn parse_json_value(value: &str) -> Option<Self> {
        match value {
            "mounted" => Some(Self::Mounted),
            "unmounted" => Some(Self::Unmounted),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LiveMountState {
    #[default]
    NotLive,
    Mounting,
    MountedThisSession,
    Unmounting,
    Error,
}

// Desired mount state is persistent user intent; live mount state is session-local
// process status and is reset when entries are loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaylistEntry {
    pub id: u64,
    pub source_path: PathBuf,
    pub display_name: String,
    pub clip_identity: String,
    pub folder_name: String,
    pub last_seen_file_len: u64,
    pub last_seen_modified_time: u64,
    pub desired_mount_state: DesiredMountState,
    pub live_mount_state: LiveMountState,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaylistAddSummary {
    pub accepted: usize,
    pub rejected: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Playlist {
    entries: Vec<PlaylistEntry>,
    selected: Option<usize>,
    next_id: u64,
}

impl Playlist {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_entries(entries: Vec<PlaylistEntry>) -> Self {
        let next_id = entries
            .iter()
            .map(|entry| entry.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
        Self {
            entries,
            selected: None,
            next_id,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn entries(&self) -> &[PlaylistEntry] {
        &self.entries
    }

    pub fn selected_index(&self) -> Option<usize> {
        self.selected
    }

    pub fn selected_entry(&self) -> Option<&PlaylistEntry> {
        self.selected.and_then(|index| self.entries.get(index))
    }

    pub fn selected_path(&self) -> Option<&Path> {
        self.selected_entry()
            .map(|entry| entry.source_path.as_path())
    }

    pub fn selected_entry_mut(&mut self) -> Option<&mut PlaylistEntry> {
        let selected = self.selected?;
        self.entries.get_mut(selected)
    }

    pub fn entry_mut_by_id(&mut self, id: u64) -> Option<&mut PlaylistEntry> {
        self.entries.iter_mut().find(|entry| entry.id == id)
    }

    pub fn entry_by_id(&self, id: u64) -> Option<&PlaylistEntry> {
        self.entries.iter().find(|entry| entry.id == id)
    }

    pub fn add_dropped_paths<I, P>(&mut self, paths: I) -> PlaylistAddSummary
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        let mut summary = PlaylistAddSummary::default();
        for path in paths {
            if self.try_add_path(path.into()) {
                summary.accepted += 1;
            } else {
                summary.rejected += 1;
            }
        }
        summary
    }

    pub fn try_add_path(&mut self, path: PathBuf) -> bool {
        if !is_mcraw_path(&path) {
            return false;
        }

        let id = self.alloc_id();
        self.entries.push(PlaylistEntry::new(id, path));
        true
    }

    pub fn push_loaded_entry(&mut self, mut entry: PlaylistEntry) {
        if entry.id == 0 {
            entry.id = self.alloc_id();
        } else {
            self.next_id = self.next_id.max(entry.id.saturating_add(1));
        }
        entry.live_mount_state = LiveMountState::NotLive;
        self.entries.push(entry);
    }

    pub fn select(&mut self, index: usize) -> bool {
        if index >= self.entries.len() {
            return false;
        }
        self.selected = Some(index);
        true
    }

    pub fn clear_selection(&mut self) {
        self.selected = None;
    }

    pub fn clear_entries(&mut self) -> usize {
        let removed = self.entries.len();
        self.entries.clear();
        self.selected = None;
        removed
    }

    pub fn remove_selected(&mut self) -> Option<PlaylistEntry> {
        let selected = self.selected?;
        if selected >= self.entries.len() {
            self.selected = None;
            return None;
        }

        let removed = self.entries.remove(selected);
        self.selected = None;
        Some(removed)
    }

    pub fn set_selected_desired_mount_state(&mut self, state: DesiredMountState) -> bool {
        let Some(entry) = self.selected_entry_mut() else {
            return false;
        };
        entry.desired_mount_state = state;
        true
    }

    pub fn set_all_desired_mount_state(&mut self, state: DesiredMountState) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        for entry in &mut self.entries {
            entry.desired_mount_state = state;
        }
        true
    }

    pub fn set_live_mount_state(&mut self, id: u64, state: LiveMountState) -> bool {
        let Some(entry) = self.entry_mut_by_id(id) else {
            return false;
        };
        entry.live_mount_state = state;
        true
    }

    pub fn set_all_live_mount_state(&mut self, state: LiveMountState) {
        for entry in &mut self.entries {
            entry.live_mount_state = state;
        }
    }

    pub fn selected_id(&self) -> Option<u64> {
        self.selected_entry().map(|entry| entry.id)
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id.max(1);
        self.next_id = id.saturating_add(1);
        id
    }
}

impl Default for Playlist {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            selected: None,
            next_id: 1,
        }
    }
}

impl PlaylistEntry {
    pub fn new(id: u64, source_path: PathBuf) -> Self {
        let display_name = display_stem(&source_path);
        let (last_seen_file_len, last_seen_modified_time) = file_observation(&source_path);
        Self {
            id,
            source_path: source_path.clone(),
            display_name: display_name.clone(),
            clip_identity: source_path.to_string_lossy().to_string(),
            folder_name: display_name,
            last_seen_file_len,
            last_seen_modified_time,
            desired_mount_state: DesiredMountState::Unmounted,
            live_mount_state: LiveMountState::NotLive,
        }
    }

    pub fn visual_state(&self) -> main_view::PlaylistEntryVisualState {
        match self.live_mount_state {
            LiveMountState::MountedThisSession => main_view::PlaylistEntryVisualState::MountedByGui,
            LiveMountState::Mounting | LiveMountState::Unmounting => {
                main_view::PlaylistEntryVisualState::IntendedMounted
            }
            LiveMountState::NotLive | LiveMountState::Error => match self.desired_mount_state {
                DesiredMountState::Mounted => main_view::PlaylistEntryVisualState::IntendedMounted,
                DesiredMountState::Unmounted => main_view::PlaylistEntryVisualState::Normal,
            },
        }
    }
}

fn file_observation(path: &Path) -> (u64, u64) {
    let Ok(metadata) = fs::metadata(path) else {
        return (0, 0);
    };
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_secs());
    (metadata.len(), modified)
}

pub fn is_mcraw_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mcraw"))
}

pub fn display_stem(path: &Path) -> String {
    path.file_stem()
        .or_else(|| path.file_name())
        .map(|name| name.to_string_lossy().trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "untitled".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playlist_starts_empty() {
        let playlist = Playlist::new();

        assert!(playlist.is_empty());
        assert_eq!(playlist.len(), 0);
        assert_eq!(playlist.selected_index(), None);
    }

    #[test]
    fn adding_mcraw_path_stores_full_path_and_display_stem() {
        let mut playlist = Playlist::new();
        let path = PathBuf::from("clips/shot one.mcraw");

        assert!(playlist.try_add_path(path.clone()));

        assert_eq!(playlist.len(), 1);
        assert_eq!(playlist.entries()[0].source_path, path);
        assert_eq!(playlist.entries()[0].display_name, "shot one");
        assert_eq!(
            playlist.entries()[0].desired_mount_state,
            DesiredMountState::Unmounted
        );
        assert_eq!(
            playlist.entries()[0].live_mount_state,
            LiveMountState::NotLive
        );
    }

    #[test]
    fn non_mcraw_path_is_rejected() {
        let mut playlist = Playlist::new();

        assert!(!playlist.try_add_path(PathBuf::from("clips/notes.txt")));
        assert!(playlist.is_empty());
    }

    #[test]
    fn extension_check_is_case_insensitive() {
        let mut playlist = Playlist::new();

        assert!(playlist.try_add_path(PathBuf::from("clips/A.MCRAW")));
        assert!(playlist.try_add_path(PathBuf::from("clips/B.McRaW")));

        assert_eq!(playlist.len(), 2);
    }

    #[test]
    fn multiple_dropped_mcraw_files_preserve_order() {
        let mut playlist = Playlist::new();
        let summary = playlist.add_dropped_paths([
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/ignore.mov"),
            PathBuf::from("clips/second.mcraw"),
        ]);

        assert_eq!(
            summary,
            PlaylistAddSummary {
                accepted: 2,
                rejected: 1,
            }
        );
        assert_eq!(playlist.entries()[0].display_name, "first");
        assert_eq!(playlist.entries()[1].display_name, "second");
    }

    #[test]
    fn selecting_entry_marks_it_selected() {
        let mut playlist = Playlist::new();
        playlist.add_dropped_paths([
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/second.mcraw"),
        ]);

        assert!(playlist.select(1));

        assert_eq!(playlist.selected_index(), Some(1));
        assert_eq!(
            playlist.selected_path(),
            Some(Path::new("clips/second.mcraw"))
        );
    }

    #[test]
    fn selecting_missing_entry_is_ignored() {
        let mut playlist = Playlist::new();

        assert!(!playlist.select(0));
        assert_eq!(playlist.selected_index(), None);
    }

    #[test]
    fn removing_selected_entry_removes_only_that_entry_and_clears_selection() {
        let mut playlist = Playlist::new();
        playlist.add_dropped_paths([
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/second.mcraw"),
            PathBuf::from("clips/third.mcraw"),
        ]);
        playlist.select(1);

        let removed = playlist.remove_selected().expect("selected entry removed");

        assert_eq!(removed.display_name, "second");
        assert_eq!(playlist.len(), 2);
        assert_eq!(playlist.entries()[0].display_name, "first");
        assert_eq!(playlist.entries()[1].display_name, "third");
        assert_eq!(playlist.selected_index(), None);
    }

    #[test]
    fn clearing_entries_removes_all_entries_and_clears_selection() {
        let mut playlist = Playlist::new();
        playlist.add_dropped_paths([
            PathBuf::from("clips/first.mcraw"),
            PathBuf::from("clips/second.mcraw"),
        ]);
        playlist.select(1);

        assert_eq!(playlist.clear_entries(), 2);

        assert!(playlist.is_empty());
        assert_eq!(playlist.selected_index(), None);
    }

    #[test]
    fn display_stem_falls_back_to_untitled() {
        assert_eq!(display_stem(Path::new("")), "untitled");
    }

    #[test]
    fn desired_and_live_state_drive_visual_state() {
        let mut entry = PlaylistEntry::new(1, PathBuf::from("clips/first.mcraw"));

        assert_eq!(
            entry.visual_state(),
            main_view::PlaylistEntryVisualState::Normal
        );

        entry.desired_mount_state = DesiredMountState::Mounted;
        assert_eq!(
            entry.visual_state(),
            main_view::PlaylistEntryVisualState::IntendedMounted
        );

        entry.live_mount_state = LiveMountState::MountedThisSession;
        assert_eq!(
            entry.visual_state(),
            main_view::PlaylistEntryVisualState::MountedByGui
        );
    }

    #[test]
    fn loaded_entries_reset_live_state() {
        let mut playlist = Playlist::new();
        let mut entry = PlaylistEntry::new(0, PathBuf::from("clips/first.mcraw"));
        entry.desired_mount_state = DesiredMountState::Mounted;
        entry.live_mount_state = LiveMountState::MountedThisSession;

        playlist.push_loaded_entry(entry);

        assert_eq!(
            playlist.entries()[0].desired_mount_state,
            DesiredMountState::Mounted
        );
        assert_eq!(
            playlist.entries()[0].live_mount_state,
            LiveMountState::NotLive
        );
    }
}
