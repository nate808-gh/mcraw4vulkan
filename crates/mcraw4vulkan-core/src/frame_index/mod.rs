// Defines compact frame and audio chunk index entries used by decoder sessions
// to locate payload bytes and expose clip timing metadata.
use crate::{FrameDimensions, FrameNumber};

#[derive(Debug, Clone)]
pub struct FrameEntry {
    pub frame_number: FrameNumber,
    pub byte_offset: u64,
    pub byte_len: u32,
    pub timestamp_us: u64,
    pub dimensions: FrameDimensions,
    pub is_key_frame: bool,
}

#[derive(Debug, Clone)]
pub struct AudioChunkEntry {
    pub chunk_index: u32,
    pub byte_offset: u64,
    pub byte_len: u32,

    // These are sample-frame units; each frame contains one interleaved scalar
    // sample per channel.
    pub start_sample: u64,
    pub sample_count: u32,

    // MotionCam stores a per-audio-chunk timestamp in AUDIO_DATA_METADATA.
    //
    // Keep this in nanoseconds to match the original C++ container structure and
    // frame timestamp math. Higher layers can convert to microseconds only when
    // they need display-friendly values.
    pub timestamp_ns: Option<i64>,
}
