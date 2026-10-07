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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataHeader {
    pub encoded_width: u32,
    pub encoded_height: u32,
    pub bits_offset: u32,
    pub refs_offset: u32,
}

// Parse the fixed raw payload metadata header at the start of one compressed raw
// video frame payload.
pub fn read_metadata_header(input: &[u8]) -> Option<MetadataHeader> {
    if input.len() < 16 {
        return None;
    }

    Some(MetadataHeader {
        encoded_width: u32::from_le_bytes([input[0], input[1], input[2], input[3]]),
        encoded_height: u32::from_le_bytes([input[4], input[5], input[6], input[7]]),
        bits_offset: u32::from_le_bytes([input[8], input[9], input[10], input[11]]),
        refs_offset: u32::from_le_bytes([input[12], input[13], input[14], input[15]]),
    })
}

/// Source and output coordinates are distinct: reconciliation never rewrites JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedFrameGeometry {
    pub declared: FrameDimensions,
    pub encoded: Option<FrameDimensions>,
    pub effective: FrameDimensions,
    pub origin: [u32; 2],
    pub reason: Option<GeometryRecoveryReason>,
    pub type7: Option<Type7GeometryEvidence>,
    pub layout_guess: Option<GeometryLayoutGuess>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryRecoveryReason {
    IncompleteFinalFourRowGroup,
}

/// Compact evidence retained before an output contract is published.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Type7GeometryEvidence {
    pub header: MetadataHeader,
    pub payload_len: u32,
    pub counts: Option<[u32; 2]>,
}

/// One bounded analysis decision, shared by every sink. Scores are not probabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GeometryLayoutGuess {
    pub policy_version: u32,
    pub extraction: FrameDimensions,
    pub scene_pitch: u32,
    pub selected: bool,
    pub analysis_frame: u32,
    pub informative_sites: u32,
    pub baseline_score: u32,
    pub selected_score: u32,
    pub margin_basis_points: u32,
}

impl ResolvedFrameGeometry {
    pub fn extraction(self) -> FrameDimensions {
        self.layout_guess
            .map_or(self.effective, |guess| guess.extraction)
    }

    /// Reframes expose a checked contiguous prefix of the codec extraction.
    /// Never pass the effective width as the codec width for this mapping.
    pub fn output_sample_count(self) -> Option<usize> {
        let extraction = self.extraction();
        if self.origin != [0, 0]
            || self.effective.width == 0
            || self.effective.height == 0
            || self.effective.height != extraction.height
            || self.effective.width > extraction.width
            || self.layout_guess.is_some_and(|g| {
                g.policy_version != 1
                    || g.analysis_frame != 0
                    || g.scene_pitch != self.effective.width
                    || g.extraction.width != self.declared.width
                    || (!g.selected && self.effective != extraction)
            })
        {
            return None;
        }
        let count = self.effective.pixel_count()?;
        (count <= self.extraction().pixel_count()?).then_some(count)
    }

    pub fn recovery_message(self) -> Option<String> {
        if let Some(guess) = self.layout_guess {
            return Some(if guess.selected {
                format!(
                    "Metadata/payload geometry mismatch: using a best-guess {}x{} layout. Original samples may be missing or repeated.",
                    self.effective.width, self.effective.height
                )
            } else {
                format!(
                    "Metadata/payload geometry mismatch: retained the safe {}x{} metadata-based layout; alignment may remain imperfect.",
                    self.effective.width, self.effective.height
                )
            });
        }
        self.reason.map(|_| format!(
            "Metadata declares {}x{}; using the represented {}x{} image. The payload does not represent the final {} declared rows under this layout.",
            self.declared.width, self.declared.height, self.effective.width,
            self.effective.height, self.declared.height - self.effective.height))
    }

    /// Compare against the evidence frozen before publication; never resolve again.
    pub fn matches_payload(self, payload: &[u8]) -> bool {
        let Some(evidence) = self.type7 else {
            return true;
        };
        if payload.len() != evidence.payload_len as usize
            || read_metadata_header(payload) != Some(evidence.header)
        {
            return false;
        }
        evidence.counts.is_none_or(|counts| {
            [evidence.header.bits_offset, evidence.header.refs_offset]
                .into_iter()
                .zip(counts)
                .all(|(offset, count)| {
                    usize::try_from(offset)
                        .ok()
                        .and_then(|start| {
                            start.checked_add(4).and_then(|end| payload.get(start..end))
                        })
                        .is_some_and(|bytes| bytes == count.to_le_bytes())
                })
        })
    }
}

/// Bounded preparation evidence identity; final source publication retains its SHA-256 check.
pub fn geometry_evidence_identity(bytes: &[u8]) -> u128 {
    xxhash_rust::xxh3::xxh3_128(bytes)
}
