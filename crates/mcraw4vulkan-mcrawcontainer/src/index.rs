// Stores the derived clip index used to look up frame entries and optional
// audio track metadata after container parsing.
use mcraw4vulkan_core::frame_index::{AudioChunkEntry, FrameEntry};
use mcraw4vulkan_core::{AudioTrackInfo, McrawClipInfo};

#[derive(Debug, Clone)]
pub struct ClipIndex {
    pub clip_info: McrawClipInfo,
    pub frames: Vec<FrameEntry>,
    pub audio_chunks: Vec<AudioChunkEntry>,
}

impl ClipIndex {
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn audio_info(&self) -> Option<AudioTrackInfo> {
        self.clip_info.audio
    }
}
