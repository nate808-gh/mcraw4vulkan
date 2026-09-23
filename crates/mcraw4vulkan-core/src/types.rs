// Defines the workspace-wide value types for clips, frames, audio, timing,
// backends, and MotionCam block encodings used across crate boundaries.

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClipId(pub String);

#[derive(Debug, Clone)]
pub struct McrawClipInfo {
    pub clip_id: ClipId,
    pub display_name: String,
    pub width: u32,
    pub height: u32,
    pub frame_count: u32,

    // Compatibility field for existing callers.
    //
    // This should mirror timing.playback_frame_rate. New code should prefer the
    // richer ClipTimingInfo structure so FUSE/Resolve and GPU preview can both
    // use audio-derived timing when available.
    pub frame_rate: FrameRate,

    // Compatibility field for existing callers.
    //
    // This should mirror timing.playback_duration_us. New code should prefer
    // timing.playback_duration_us so audio-truth timeline behavior is explicit.
    pub duration_us: u64,

    pub timing: ClipTimingInfo,
    pub audio: Option<AudioTrackInfo>,
    pub container_flavor: ContainerFlavor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameDimensions {
    pub width: u32,
    pub height: u32,
}

impl FrameDimensions {
    pub fn pixel_count(self) -> Option<usize> {
        let count = u64::from(self.width) * u64::from(self.height);
        usize::try_from(count).ok()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FramePayloadLayout {
    CompressedRawcodecType7,
    BinnedRaw16Type6 { row_stride: u32 },
}

impl FramePayloadLayout {
    pub const fn label(self) -> &'static str {
        match self {
            Self::CompressedRawcodecType7 => "compressed_rawcodec_type7",
            Self::BinnedRaw16Type6 { .. } => "binned_raw16_type6",
        }
    }

    pub const fn uses_compressed_rawcodec_work_plan(self) -> bool {
        matches!(self, Self::CompressedRawcodecType7)
    }

    pub const fn supports_native_gpu_decode(self) -> bool {
        matches!(
            self,
            Self::CompressedRawcodecType7 | Self::BinnedRaw16Type6 { .. }
        )
    }

    pub const fn unsupported_gpu_decode_message(self) -> Option<&'static str> {
        match self {
            Self::CompressedRawcodecType7 => None,
            Self::BinnedRaw16Type6 { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameNumber(pub u32);

// Rational frames-per-second value.
//
// Do not use f32/f64 as the source of truth for timeline FPS. MotionCam clips
// can use arbitrary/non-standard rates, and audio-derived timing may produce
// values such as 29.987 or 24.001545. Internally, preserve the exact rational:
//
//   fps = numerator / denominator
//
// Use f64 only for display/logging/approximate comparisons, and round to 3
// decimals only at Resolve-facing or UI boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRate {
    pub numerator: u64,
    pub denominator: u64,
}

impl FrameRate {
    pub const fn new(numerator: u64, denominator: u64) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    pub const fn integer(fps: u64) -> Self {
        Self {
            numerator: fps,
            denominator: 1,
        }
    }

    pub fn from_audio_timing(
        frame_count: u64,
        sample_rate_hz: u32,
        samples_per_channel: u64,
    ) -> Option<Self> {
        if frame_count == 0 || sample_rate_hz == 0 || samples_per_channel == 0 {
            return None;
        }

        let numerator = frame_count.checked_mul(u64::from(sample_rate_hz))?;
        Some(Self::new(numerator, samples_per_channel).reduced())
    }

    pub const fn is_valid(self) -> bool {
        self.denominator != 0
    }

    pub fn reduced(self) -> Self {
        if self.numerator == 0 || self.denominator == 0 {
            return self;
        }

        let divisor = gcd_u64(self.numerator, self.denominator);

        Self {
            numerator: self.numerator / divisor,
            denominator: self.denominator / divisor,
        }
    }

    pub fn as_f64(self) -> Option<f64> {
        if self.denominator == 0 {
            return None;
        }

        Some(self.numerator as f64 / self.denominator as f64)
    }

    pub fn rounded_millifps(self) -> Option<u64> {
        if self.denominator == 0 {
            return None;
        }

        let numerator = u128::from(self.numerator);
        let denominator = u128::from(self.denominator);
        let scaled = numerator.checked_mul(1_000)?;
        let rounded = scaled.checked_add(denominator / 2)? / denominator;

        u64::try_from(rounded).ok()
    }

    pub fn as_three_decimal_string(self) -> Option<String> {
        let millifps = self.rounded_millifps()?;

        Some(format!("{}.{:03}", millifps / 1_000, millifps % 1_000))
    }

    pub fn duration_us_for_frame_count(self, frame_count: u64) -> Option<u64> {
        if self.numerator == 0 || self.denominator == 0 {
            return None;
        }

        // duration seconds = frame_count / fps
        // duration us = frame_count * denominator * 1_000_000 / numerator
        let numerator = u128::from(frame_count)
            .checked_mul(u128::from(self.denominator))?
            .checked_mul(1_000_000)?;

        let duration_us = numerator / u128::from(self.numerator);

        u64::try_from(duration_us).ok()
    }
}

// Explains which timing source is authoritative for playback.
//
// Audio timing should win when available because DaVinci Resolve cDNG sequences
// and the mcraw4vulkan GPU preview path need to stay synchronized to the real
// audio duration, even when nominal video metadata reports a slightly different
// FPS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimelineFrameRateSource {
    AudioMetadata,
    AudioDerived,
    VideoMetadata,
    Fallback,
}

// Complete timing model for one clip.
//
// The decoder should preserve reported/derived timing separately, then select a
// playback frame rate using this priority:
//
// 1. explicit audio FPS metadata
// 2. audio-derived FPS from decoded audio duration + frame count
// 3. video-reported FPS
// 4. fallback only as a last resort
//
// FUSE/Resolve cDNG metadata and GPU preview playback should use
// playback_frame_rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipTimingInfo {
    pub video_reported_frame_rate: Option<FrameRate>,
    pub audio_reported_frame_rate: Option<FrameRate>,
    pub audio_derived_frame_rate: Option<FrameRate>,
    pub playback_frame_rate: FrameRate,
    pub playback_frame_rate_source: TimelineFrameRateSource,
    pub video_duration_us: u64,
    pub audio_duration_us: Option<u64>,
    pub playback_duration_us: u64,
}

impl ClipTimingInfo {
    pub fn from_sources(
        frame_count: u64,
        video_reported_frame_rate: Option<FrameRate>,
        audio_reported_frame_rate: Option<FrameRate>,
        audio_derived_frame_rate: Option<FrameRate>,
        audio_duration_us: Option<u64>,
        fallback_frame_rate: FrameRate,
    ) -> Self {
        let (playback_frame_rate, playback_frame_rate_source) =
            if let Some(frame_rate) = audio_reported_frame_rate {
                (frame_rate, TimelineFrameRateSource::AudioMetadata)
            } else if let Some(frame_rate) = audio_derived_frame_rate {
                (frame_rate, TimelineFrameRateSource::AudioDerived)
            } else if let Some(frame_rate) = video_reported_frame_rate {
                (frame_rate, TimelineFrameRateSource::VideoMetadata)
            } else {
                (fallback_frame_rate, TimelineFrameRateSource::Fallback)
            };

        let video_duration_us = video_reported_frame_rate
            .and_then(|frame_rate| frame_rate.duration_us_for_frame_count(frame_count))
            .or_else(|| playback_frame_rate.duration_us_for_frame_count(frame_count))
            .unwrap_or(0);

        let playback_duration_us = match playback_frame_rate_source {
            TimelineFrameRateSource::AudioMetadata | TimelineFrameRateSource::AudioDerived => {
                audio_duration_us
                    .or_else(|| playback_frame_rate.duration_us_for_frame_count(frame_count))
                    .unwrap_or(0)
            }
            TimelineFrameRateSource::VideoMetadata | TimelineFrameRateSource::Fallback => {
                playback_frame_rate
                    .duration_us_for_frame_count(frame_count)
                    .unwrap_or(0)
            }
        };

        Self {
            video_reported_frame_rate,
            audio_reported_frame_rate,
            audio_derived_frame_rate,
            playback_frame_rate,
            playback_frame_rate_source,
            video_duration_us,
            audio_duration_us,
            playback_duration_us,
        }
    }
}

// total_samples counts sample frames per channel, not interleaved scalar values;
// an interleaved buffer therefore contains total_samples * channels values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioTrackInfo {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub total_samples: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFlavor {
    Current,
    Legacy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayerPattern {
    Rggb,
    Bggr,
    Grbg,
    Gbrg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeBackend {
    Cpu,
    Vulkan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSampleRange {
    pub start_sample: u64,
    pub sample_count: u64,
}

// Discriminants are raw residual bit widths. Keeping this mapping explicit lets
// byte-length tables and CPU/GPU decode plans use the same encoded values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockEncoding {
    Zero = 0,
    Bits1 = 1,
    Bits2 = 2,
    Bits3 = 3,
    Bits4 = 4,
    Bits5 = 5,
    Bits6 = 6,
    Bits7 = 7,
    Bits8 = 8,
    Bits9 = 9,
    Bits10 = 10,
    Bits16 = 16,
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }

    left
}
