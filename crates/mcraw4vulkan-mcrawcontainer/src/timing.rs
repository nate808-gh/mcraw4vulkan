use mcraw4vulkan_core::FrameRate;

// Video timestamp-derived frame-rate diagnostics.
//
// MotionCam frame timestamps are nanosecond-scale values. The creator's C++
// code derives video FPS from adjacent frame timestamp deltas, not from total
// decoded audio sample count.
#[derive(Debug, Clone, Copy)]
pub struct VideoFrameRateInfo {
    pub first_timestamp_ns: Option<u64>,
    pub last_timestamp_ns: Option<u64>,
    pub total_span_ns: Option<u64>,
    pub valid_duration_count: u32,
    pub median_frame_rate: Option<FrameRate>,
    pub average_frame_rate: Option<FrameRate>,
}

// C++-style audio/video start alignment diagnostics.
//
// drift_ns = first_audio_timestamp_ns - first_video_timestamp_ns.
// - negative drift means audio starts before video, so trim audio
// - positive drift means audio starts after video, so insert silence
#[derive(Debug, Clone, Copy)]
pub struct AudioSyncInfo {
    pub first_video_timestamp_ns: u64,
    pub first_audio_timestamp_ns: i64,
    pub drift_ns: i128,
    pub sample_frames_to_trim: u64,
    pub sample_frames_to_insert_silence: u64,
    pub sync_is_within_one_second: bool,
}

// Read-only summary of AUDIO_DATA chunk timing and sample ranges.
//
// Timestamps are nanoseconds; start and end positions count sample frames.
// This value owns no mutable decoder or alignment state.
#[derive(Debug, Clone, Copy)]
pub struct AudioChunkTimingInfo {
    pub chunk_count: usize,
    pub timestamped_chunk_count: usize,
    pub first_timestamp_ns: Option<i64>,
    pub last_timestamp_ns: Option<i64>,
    pub timestamp_span_ns: Option<i128>,
    pub min_timestamp_delta_ns: Option<i128>,
    pub max_timestamp_delta_ns: Option<i128>,
    pub first_start_sample: Option<u64>,
    pub last_end_sample: Option<u64>,
}
