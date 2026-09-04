// These byte and sample-frame totals answer metadata requests without
// materializing audio. Counts are u64 because BW64 extends RIFF's 32-bit chunk
// size limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioWavMetadata {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub sample_frames: u64,
    pub byte_len: u64,
}
