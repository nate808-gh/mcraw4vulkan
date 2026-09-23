use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use mcraw4vulkan_core::{DecodeBackendChoice, RegisteredClipId};

use crate::virtual_fs::{DEFAULT_DNG_CACHE_FRAME_CAPACITY, DEFAULT_PREFETCH_FORWARD_FRAMES};

// Future app-wide resource manager skeleton.
//
// This is intentionally not wired into the working Linux FUSE3 mount yet. The
// current one-clip VirtualFileSystem path remains the validated Resolve/FUSE
// implementation.
//
// This skeleton introduces the future ownership boundary:
//
//   RegisteredClipId -> clip path, backend choice, cache settings, mount/preview state
//
// Later, this manager can grow into:
//
// - shared GPU runtime/device/queue ownership
// - global DNG cache byte budget
// - per-clip soft quotas
// - active-clip priority
// - inactive clip eviction
// - bounded prefetch workers
// - get_dng_bytes(clip_id, frame_index)
//
// Keeping this outside linux_fuse3 avoids baking Linux-only FUSE assumptions into
// the multi-platform app design.
#[derive(Debug)]
pub struct AppResourceManager {
    next_clip_id: AtomicU64,
    state: Mutex<AppResourceManagerState>,
}

#[derive(Debug, Default)]
struct AppResourceManagerState {
    clips: HashMap<RegisteredClipId, RegisteredClipRecord>,
}

// Configuration used when registering a clip with the future app resource
// manager.
//
// This is lightweight. It does not open the decoder, create a DNG cache, create
// a GPU backend, or mount anything yet. Heavy resources should remain lazy.
#[derive(Debug, Clone)]
pub struct RegisteredClipConfig {
    pub path: PathBuf,
    pub display_name: String,
    pub backend_choice: DecodeBackendChoice,
    pub max_cached_dng_frames: usize,
    pub prefetch_forward_frames: usize,
}

impl RegisteredClipConfig {
    pub fn new(path: impl Into<PathBuf>, backend_choice: DecodeBackendChoice) -> Self {
        let path = path.into();
        let display_name = display_name_for_path(&path);

        Self {
            path,
            display_name,
            backend_choice,
            max_cached_dng_frames: DEFAULT_DNG_CACHE_FRAME_CAPACITY,
            prefetch_forward_frames: DEFAULT_PREFETCH_FORWARD_FRAMES,
        }
    }

    pub fn with_display_name(mut self, display_name: impl Into<String>) -> Self {
        self.display_name = display_name.into();
        self
    }

    pub fn with_cache_frames(mut self, max_cached_dng_frames: usize) -> Self {
        self.max_cached_dng_frames = max_cached_dng_frames.max(1);
        self
    }

    pub fn with_prefetch_frames(mut self, prefetch_forward_frames: usize) -> Self {
        self.prefetch_forward_frames = prefetch_forward_frames;
        self
    }
}

// Lightweight registered clip record.
//
// This should stay cheap enough that many playlist entries can exist without
// allocating large DNG caches or GPU resources.
#[derive(Debug, Clone)]
pub struct RegisteredClipRecord {
    pub clip_id: RegisteredClipId,
    pub path: PathBuf,
    pub display_name: String,
    pub backend_choice: DecodeBackendChoice,
    pub max_cached_dng_frames: usize,
    pub prefetch_forward_frames: usize,
    pub mount_state: ClipMountState,
    pub preview_state: ClipPreviewState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipMountState {
    Unmounted,
    Mounted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipPreviewState {
    Inactive,
    Selected,
    Playing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppResourceManagerStats {
    pub registered_clips: usize,
    pub next_clip_id_value: u64,
}

impl AppResourceManager {
    pub fn new() -> Self {
        Self {
            next_clip_id: AtomicU64::new(1),
            state: Mutex::new(AppResourceManagerState::default()),
        }
    }

    // Register a clip and return its application-level ID.
    //
    // This does not open the clip, decode frames, create a DNG cache, or mount a
    // filesystem. Those heavy operations should be added later as lazy manager
    // operations.
    pub fn register_clip(&self, config: RegisteredClipConfig) -> Result<RegisteredClipId> {
        if config.path.as_os_str().is_empty() {
            anyhow::bail!("cannot register clip with empty path");
        }

        let clip_id = self.allocate_clip_id();

        let record = RegisteredClipRecord {
            clip_id,
            path: config.path,
            display_name: config.display_name,
            backend_choice: config.backend_choice,
            max_cached_dng_frames: config.max_cached_dng_frames.max(1),
            prefetch_forward_frames: config.prefetch_forward_frames,
            mount_state: ClipMountState::Unmounted,
            preview_state: ClipPreviewState::Inactive,
        };

        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("app resource manager mutex was poisoned"))?;

        state.clips.insert(clip_id, record);

        Ok(clip_id)
    }

    pub fn unregister_clip(
        &self,
        clip_id: RegisteredClipId,
    ) -> Result<Option<RegisteredClipRecord>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("app resource manager mutex was poisoned"))?;

        Ok(state.clips.remove(&clip_id))
    }

    pub fn clip_record(&self, clip_id: RegisteredClipId) -> Result<Option<RegisteredClipRecord>> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("app resource manager mutex was poisoned"))?;

        Ok(state.clips.get(&clip_id).cloned())
    }

    pub fn registered_clip_ids(&self) -> Result<Vec<RegisteredClipId>> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("app resource manager mutex was poisoned"))?;

        let mut ids = state.clips.keys().copied().collect::<Vec<_>>();
        ids.sort();

        Ok(ids)
    }

    pub fn stats(&self) -> Result<AppResourceManagerStats> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("app resource manager mutex was poisoned"))?;

        Ok(AppResourceManagerStats {
            registered_clips: state.clips.len(),
            next_clip_id_value: self.next_clip_id.load(Ordering::Relaxed),
        })
    }

    pub fn set_mount_state(
        &self,
        clip_id: RegisteredClipId,
        mount_state: ClipMountState,
    ) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("app resource manager mutex was poisoned"))?;

        let record = state
            .clips
            .get_mut(&clip_id)
            .with_context(|| format!("unknown registered clip ID: {clip_id}"))?;

        record.mount_state = mount_state;

        Ok(())
    }

    pub fn set_preview_state(
        &self,
        clip_id: RegisteredClipId,
        preview_state: ClipPreviewState,
    ) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("app resource manager mutex was poisoned"))?;

        let record = state
            .clips
            .get_mut(&clip_id)
            .with_context(|| format!("unknown registered clip ID: {clip_id}"))?;

        record.preview_state = preview_state;

        Ok(())
    }

    fn allocate_clip_id(&self) -> RegisteredClipId {
        let value = self.next_clip_id.fetch_add(1, Ordering::Relaxed);
        RegisteredClipId::new(value)
    }
}

impl Default for AppResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

// Shared manager handle type for future GUI/FUSE/player integration.
pub type SharedAppResourceManager = Arc<AppResourceManager>;

fn display_name_for_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("untitled")
        .to_string()
}
