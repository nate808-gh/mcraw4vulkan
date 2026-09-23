//! Production PIPE format, timing, and sidecar-v4 contract.
//!
//! This module is deliberately independent of the CLI and GUI. The public
//! producer and every sidecar consumer use the same fixed direct-YUV policy
//! and the same bounded MOV cadence implementation.

#![deny(unreachable_pub)]

use std::error::Error;
use std::fmt;

use mcraw4vulkan_render::Yuv444p12lePackPolicy;
use serde_json::{Map, Value, json};

const PIPE_METADATA_VERSION: u64 = 4;
const PIPE_PRODUCER: &str = "mcraw4vulkan pipe";
const PIPE_OUTPUT_ALGORITHM_ID: &str = "mcraw-yuv444p12le-tv-bt2020-apple-log-original-ncl-v1";
pub(crate) const PIPE_PIXEL_FORMAT: &str = "yuv444p12le";
const PIPE_MIME_TYPE: &str = "application/octet-stream";
const PIPE_MAX_MOV_TIMESCALE_NUMERATOR: u64 = 99_999;
const PIPE_TERMINAL_EXPOSURE_SCALE_ID: &str = "unity-no-extra-exposure-v1";
pub(crate) const PIPE_SCALE_APPLICATION_POINT: &str = "scene-linear-bt2020-before-apple-log";
pub(crate) const PIPE_CORRECTION_POLICY_ID: &str = "motioncam-compatible-pixel-domain-v1";
pub(crate) const PIPE_CORRECTION_TERMINAL_ID: &str = "materialized-signed-f32-before-demosaic-v1";
const PIPE_CORRECTION_MODE_MOTIONCAM_SPATIAL: &str = "motioncam-spatial";
const PIPE_CORRECTION_MODE_IDENTITY_SPATIAL_GAIN: &str = "identity-spatial-gain";
pub(crate) const PIPE_COLOR_POLICY_ID: &str = "StrictMotionCamForwardMatrixColorV2";
const PIPE_SCENE_NORMALIZATION_ID: &str = "quantized-black-and-gain-divide-cw-v1";
const PIPE_ROUNDING_ID: &str = "clamp-then-floor-plus-half-v1";

pub(crate) const PIPE_PLANE_ORDER_LABEL: &str = "Y,Cb,Cr";
pub(crate) const PIPE_BYTE_ORDER_LABEL: &str = "little_endian_low_12";
pub(crate) const PIPE_ENDIANNESS_LABEL: &str = "little";
pub(crate) const PIPE_BIT_ALIGNMENT: &str = "lsb";
const PIPE_BYTE_ORDER: &str = "little-endian";
pub(crate) const PIPE_SAMPLE_RANGE: &str = "video-data-12bit";
pub(crate) const PIPE_COLOR_RANGE: &str = "tv";
pub(crate) const PIPE_COLOR_PRIMARIES: &str = "bt2020";
pub(crate) const PIPE_COLOR_TRANSFER: &str = "apple-log-original";
pub(crate) const PIPE_MATRIX_COEFFICIENTS: &str = "bt2020nc";
pub(crate) const PIPE_CHROMA_SAMPLING: &str = "4:4:4";
pub(crate) const PIPE_ALPHA: &str = "none";
pub(crate) const PIPE_STORAGE_BYTES_PER_PIXEL: u64 =
    Yuv444p12lePackPolicy::STORAGE_BYTES_PER_PIXEL as u64;

pub(crate) fn checked_pipe_bytes_per_frame(
    width: u32,
    height: u32,
) -> Result<u64, PipeContractError> {
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(PIPE_STORAGE_BYTES_PER_PIXEL))
        .ok_or_else(|| PipeContractError::new("PIPE bytes-per-frame overflow"))
}

pub(crate) fn checked_pipe_total_bytes(
    width: u32,
    height: u32,
    frame_count: u64,
) -> Result<u64, PipeContractError> {
    checked_pipe_bytes_per_frame(width, height)?
        .checked_mul(frame_count)
        .ok_or_else(|| PipeContractError::new("PIPE total-byte count overflow"))
}

// One naming owner for raw video, metadata, audio and GUI-generated ProRes.
macro_rules! pipe_output_suffixes {
    ($stem:literal) => {
        pub const PIPE_OUTPUT_STEM_SUFFIX: &str = $stem;
        pub const PIPE_METADATA_FILE_SUFFIX: &str = concat!($stem, ".json");
        pub const PIPE_AUDIO_FILE_SUFFIX: &str = concat!($stem, "-audio.wav");
        pub const PIPE_PRORES_FILE_SUFFIX: &str = "_prores4444_bt2020_applelog.mov";
        pub const PIPE_PRORES_SIDECAR_SUFFIX: &str = "_prores4444_bt2020_applelog.json";
    };
}
pipe_output_suffixes!("-BT2020-AppleLog");

fn apple_log_contract_fields() -> Value {
    json!({
        "white_point": "D65",
        "ffmpeg_color_trc": 2,
        "transfer_parameters": {
            "R0": -0.05641088, "Rt": 0.01, "c": 47.28711236,
            "beta": 0.00964052, "gamma": 0.08550479, "delta": 0.69336945
        },
        "negative_component_policy": "finite-below-R0-encodes-zero-inverse-R0",
        "upper_transfer_policy": "no-F-one-clamp-final-ycbcr-bounds-only",
        "nonfinite_policy": "frame-error-before-publication",
        "dither": false,
        "inverse_order": ["remove-tv-offsets-and-scales", "inverse-bt2020-ncl",
            "original-apple-log-component-inverse", "unity-exposure-scale"],
        "editor_input_assignment": "manual-original-Apple-Log-Rec2020-D65-video-levels-BT2020-NCL",
        "automatic_editor_recognition": false
    })
}

const LEGACY_FORMAT_FIELDS: &[&str] = &[
    "linear_signal_scale_num",
    "linear_signal_scale_den",
    "linear_signal_scale_stops",
    "linear_signal_scale_id",
    "bits_per_channel",
    "black_code_value",
    "camera_to_linear_srgb_3x3",
    "color_mode",
    "color_space",
    "endianness",
    "expected_total_video_bytes",
    "range_is_limited",
    "render_policy",
    "rgb_sink_transform",
    "sample_range",
    "srgb_oetf_applied",
    "transfer_function",
    "white_code_value",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipeContractError {
    detail: String,
}

impl PipeContractError {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

impl fmt::Display for PipeContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl Error for PipeContractError {}

/// The exact bounded-integer cadence used by the established Pipe Example.
///
/// The reduced FPS numerator is also the MOV video-track timescale, and the
/// inverse FPS is the sidecar timebase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipeMovCadence {
    pub fps_num: u64,
    pub fps_den: u64,
    pub timebase_num: u64,
    pub timebase_den: u64,
    pub video_track_timescale: u64,
}

/// The small, source-derived fact set needed to format a Pipe Example.
///
/// This deliberately excludes strict color/correction identities and every
/// sidecar-only field. Producing the command must not require the complete
/// production metadata preflight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipeExampleFacts {
    pub width: u32,
    pub height: u32,
    pub cadence: PipeMovCadence,
    pub sample_aspect_ratio: PipeAspectRatio,
    pub display_aspect_ratio: PipeAspectRatio,
}

impl PipeMovCadence {
    pub fn from_source_rate(numerator: u64, denominator: u64) -> Result<Self, PipeContractError> {
        let source = FrameRateRational::new(numerator, denominator)
            .ok_or_else(|| PipeContractError::new("PIPE frame rate must be positive"))?;
        Ok(Self::from_bounded_rational(
            source.approximate_with_max_numerator(PIPE_MAX_MOV_TIMESCALE_NUMERATOR),
        ))
    }

    pub fn from_source_rate_text(text: &str) -> Result<Self, PipeContractError> {
        let source = FrameRateRational::parse(text)
            .ok_or_else(|| PipeContractError::new("PIPE frame rate is not a positive rational"))?;
        Ok(Self::from_bounded_rational(
            source.approximate_with_max_numerator(PIPE_MAX_MOV_TIMESCALE_NUMERATOR),
        ))
    }

    fn from_sidecar_fields(
        fps_num: u64,
        fps_den: u64,
        timebase_num: u64,
        timebase_den: u64,
        video_track_timescale: u64,
    ) -> Result<Self, PipeContractError> {
        let bounded = FrameRateRational::new(fps_num, fps_den)
            .ok_or_else(|| PipeContractError::new("PIPE sidecar FPS must be positive"))?;
        if bounded.numerator != fps_num || bounded.denominator != fps_den {
            return Err(PipeContractError::new(
                "PIPE sidecar FPS must be an exact reduced rational",
            ));
        }
        if fps_num > PIPE_MAX_MOV_TIMESCALE_NUMERATOR {
            return Err(PipeContractError::new(format!(
                "PIPE sidecar FPS numerator {fps_num} exceeds the MOV timescale bound {PIPE_MAX_MOV_TIMESCALE_NUMERATOR}"
            )));
        }
        if timebase_num != fps_den || timebase_den != fps_num {
            return Err(PipeContractError::new(
                "PIPE sidecar timebase must be the exact inverse of FPS",
            ));
        }
        if video_track_timescale != fps_num {
            return Err(PipeContractError::new(
                "PIPE sidecar video-track timescale must equal the bounded FPS numerator",
            ));
        }
        Ok(Self::from_bounded_rational(bounded))
    }

    pub fn ffmpeg_framerate(self) -> String {
        format!("{}/{}", self.fps_num, self.fps_den)
    }

    fn from_bounded_rational(rate: FrameRateRational) -> Self {
        Self {
            fps_num: rate.numerator,
            fps_den: rate.denominator,
            timebase_num: rate.denominator,
            timebase_den: rate.numerator,
            video_track_timescale: rate.numerator,
        }
    }
}

// Keep the established Pipe Example integer candidate search and tie-breaking
// semantics in one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameRateRational {
    numerator: u64,
    denominator: u64,
}

impl FrameRateRational {
    fn new(numerator: u64, denominator: u64) -> Option<Self> {
        if numerator == 0 || denominator == 0 {
            return None;
        }
        let divisor = gcd(numerator, denominator);
        Some(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    fn parse(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        if let Some((num, den)) = trimmed.split_once('/') {
            return Self::new(num.trim().parse().ok()?, den.trim().parse().ok()?);
        }
        if let Ok(value) = trimmed.parse::<u64>() {
            return Self::new(value, 1);
        }
        parse_decimal_rate(trimmed)
    }

    fn approximate_with_max_numerator(self, max_numerator: u64) -> Self {
        if self.numerator <= max_numerator {
            return self;
        }

        let mut best: Option<Self> = None;
        let max_denominator = max_numerator;
        for denominator in 1..=max_denominator {
            let numerator = rounded_scaled_numerator(self.numerator, self.denominator, denominator);
            if numerator == 0 || numerator > max_numerator {
                continue;
            }
            let Some(candidate) = Self::new(numerator, denominator) else {
                continue;
            };
            if candidate.numerator > max_numerator {
                continue;
            }
            if best.is_none_or(|current| rational_candidate_is_better(self, candidate, current)) {
                best = Some(candidate);
            }
        }

        best.unwrap_or(Self {
            numerator: max_numerator,
            denominator: 1,
        })
    }
}

fn parse_decimal_rate(text: &str) -> Option<FrameRateRational> {
    let (whole, fraction) = text.split_once('.')?;
    let whole: u64 = whole.trim().parse().ok()?;
    let fraction = fraction.trim();
    if fraction.is_empty() || !fraction.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let scale = 10_u64.checked_pow(u32::try_from(fraction.len()).ok()?)?;
    let fraction: u64 = fraction.parse().ok()?;
    let numerator = whole.checked_mul(scale)?.checked_add(fraction)?;
    FrameRateRational::new(numerator, scale)
}

fn rounded_scaled_numerator(source_num: u64, source_den: u64, denominator: u64) -> u64 {
    let scaled = u128::from(source_num) * u128::from(denominator);
    let rounded = (scaled + u128::from(source_den / 2)) / u128::from(source_den);
    u64::try_from(rounded).unwrap_or(u64::MAX)
}

fn rational_candidate_is_better(
    source: FrameRateRational,
    candidate: FrameRateRational,
    current: FrameRateRational,
) -> bool {
    let candidate_error = rational_error(source, candidate);
    let current_error = rational_error(source, current);
    let candidate_den = u128::from(candidate.denominator);
    let current_den = u128::from(current.denominator);
    let candidate_scaled = candidate_error * current_den;
    let current_scaled = current_error * candidate_den;
    candidate_scaled < current_scaled
        || (candidate_scaled == current_scaled
            && (candidate.denominator, candidate.numerator)
                < (current.denominator, current.numerator))
}

fn rational_error(source: FrameRateRational, candidate: FrameRateRational) -> u128 {
    let left = u128::from(source.numerator) * u128::from(candidate.denominator);
    let right = u128::from(source.denominator) * u128::from(candidate.numerator);
    left.abs_diff(right)
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipeAspectRatio {
    pub numerator: u64,
    pub denominator: u64,
}

impl PipeAspectRatio {
    pub fn new(numerator: u64, denominator: u64) -> Result<Self, PipeContractError> {
        if numerator == 0 || denominator == 0 {
            return Err(PipeContractError::new(
                "PIPE aspect-ratio terms must be positive",
            ));
        }
        let divisor = gcd(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub fn square_pixels() -> Self {
        Self {
            numerator: 1,
            denominator: 1,
        }
    }

    pub fn display_for_frame(
        width: u32,
        height: u32,
        sample_aspect_ratio: Self,
    ) -> Result<Self, PipeContractError> {
        let numerator = u64::from(width)
            .checked_mul(sample_aspect_ratio.numerator)
            .ok_or_else(|| PipeContractError::new("PIPE display-aspect numerator overflow"))?;
        let denominator = u64::from(height)
            .checked_mul(sample_aspect_ratio.denominator)
            .ok_or_else(|| PipeContractError::new("PIPE display-aspect denominator overflow"))?;
        Self::new(numerator, denominator)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipeAudioContractV4 {
    pub present: bool,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u16>,
    pub sample_format: Option<String>,
    pub bits_per_sample: Option<u16>,
    pub sample_frames: Option<u64>,
    pub byte_len: Option<u64>,
}

impl PipeAudioContractV4 {
    pub fn absent() -> Self {
        Self {
            present: false,
            sample_rate_hz: None,
            channels: None,
            sample_format: None,
            bits_per_sample: None,
            sample_frames: None,
            byte_len: None,
        }
    }

    pub fn pcm_s16le(
        sample_rate_hz: u32,
        channels: u16,
        sample_frames: u64,
        byte_len: u64,
    ) -> Result<Self, PipeContractError> {
        let value = Self {
            present: true,
            sample_rate_hz: Some(sample_rate_hz),
            channels: Some(channels),
            sample_format: Some("s16le".to_string()),
            bits_per_sample: Some(16),
            sample_frames: Some(sample_frames),
            byte_len: Some(byte_len),
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), PipeContractError> {
        if !self.present {
            if self.sample_rate_hz.is_some()
                || self.channels.is_some()
                || self.sample_format.is_some()
                || self.bits_per_sample.is_some()
                || self.sample_frames.is_some()
                || self.byte_len.is_some()
            {
                return Err(PipeContractError::new(
                    "absent PIPE audio must not carry present-track fields",
                ));
            }
            return Ok(());
        }

        if self.sample_rate_hz.is_none_or(|value| value == 0)
            || self.channels.is_none_or(|value| value == 0)
            || self.sample_frames.is_none_or(|value| value == 0)
            || self.byte_len.is_none_or(|value| value == 0)
        {
            return Err(PipeContractError::new(
                "present PIPE audio requires positive rate/channels/frames/bytes",
            ));
        }
        if self.sample_format.as_deref() != Some("s16le") || self.bits_per_sample != Some(16) {
            return Err(PipeContractError::new(
                "present PIPE audio must describe the existing s16le/16-bit path",
            ));
        }
        Ok(())
    }

    fn to_value(&self) -> Value {
        json!({
            "present": self.present,
            // The production Pipe Example uses a temporary WAV only for the
            // second-pass mux, then removes it. Never publish a stale path in
            // the v3 sidecar that is moved next to the final MOV.
            "sidecar_path": Value::Null,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "sample_format": self.sample_format,
            "bits_per_sample": self.bits_per_sample,
            "sample_frames": self.sample_frames,
            "byte_len": self.byte_len,
        })
    }

    fn from_value(value: &Value) -> Result<Self, PipeContractError> {
        let object = required_object_value(value, "audio")?;
        if optional_string(object, "sidecar_path")?.is_some() {
            return Err(PipeContractError::new(
                "PIPE sidecar audio.sidecar_path must be null because the WAV is a temporary mux input",
            ));
        }
        let result = Self {
            present: required_bool(object, "present")?,
            sample_rate_hz: optional_u64(object, "sample_rate_hz")?
                .map(|value| {
                    u32::try_from(value).map_err(|_| {
                        PipeContractError::new("audio.sample_rate_hz does not fit u32")
                    })
                })
                .transpose()?,
            channels: optional_u64(object, "channels")?
                .map(|value| {
                    u16::try_from(value)
                        .map_err(|_| PipeContractError::new("audio.channels does not fit u16"))
                })
                .transpose()?,
            sample_format: optional_string(object, "sample_format")?,
            bits_per_sample: optional_u64(object, "bits_per_sample")?
                .map(|value| {
                    u16::try_from(value).map_err(|_| {
                        PipeContractError::new("audio.bits_per_sample does not fit u16")
                    })
                })
                .transpose()?,
            sample_frames: optional_u64(object, "sample_frames")?,
            byte_len: optional_u64(object, "byte_len")?,
        };
        result.validate()?;
        Ok(result)
    }
}

/// Typed representation of the strict version-3 public PIPE sidecar.
///
/// All format policy is fixed by `to_value`; only clip facts and validated
/// context identities vary.
#[derive(Debug, Clone, PartialEq)]
pub struct PipeSidecarV4 {
    pub frame_width: u32,
    pub frame_height: u32,
    pub frame_count: u64,
    pub cadence: PipeMovCadence,
    pub sample_aspect_ratio: PipeAspectRatio,
    pub display_aspect_ratio: PipeAspectRatio,
    pub bytes_per_frame: u64,
    pub expected_total_bytes: u64,
    pub correction_mode: String,
    pub source_payload_geometry_identity: Value,
    pub strict_color_context_identity: Value,
    pub correction_context_identity: Value,
    pub audio: PipeAudioContractV4,
}

impl PipeSidecarV4 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame_width: u32,
        frame_height: u32,
        frame_count: u64,
        cadence: PipeMovCadence,
        sample_aspect_ratio: PipeAspectRatio,
        display_aspect_ratio: PipeAspectRatio,
        correction_mode: String,
        source_payload_geometry_identity: Value,
        strict_color_context_identity: Value,
        correction_context_identity: Value,
        audio: PipeAudioContractV4,
    ) -> Result<Self, PipeContractError> {
        if frame_width == 0 || frame_height == 0 || frame_count == 0 {
            return Err(PipeContractError::new(
                "PIPE dimensions and frame count must be positive",
            ));
        }
        let cadence = PipeMovCadence::from_sidecar_fields(
            cadence.fps_num,
            cadence.fps_den,
            cadence.timebase_num,
            cadence.timebase_den,
            cadence.video_track_timescale,
        )?;
        let bytes_per_frame = checked_pipe_bytes_per_frame(frame_width, frame_height)?;
        let expected_total_bytes =
            checked_pipe_total_bytes(frame_width, frame_height, frame_count)?;
        let expected_dar =
            PipeAspectRatio::display_for_frame(frame_width, frame_height, sample_aspect_ratio)?;
        if display_aspect_ratio != expected_dar {
            return Err(PipeContractError::new(
                "PIPE display aspect ratio contradicts dimensions and sample aspect ratio",
            ));
        }
        if !matches!(
            correction_mode.as_str(),
            PIPE_CORRECTION_MODE_MOTIONCAM_SPATIAL | PIPE_CORRECTION_MODE_IDENTITY_SPATIAL_GAIN
        ) {
            return Err(PipeContractError::new(
                "PIPE correction_mode must be motioncam-spatial or identity-spatial-gain",
            ));
        }
        require_nonempty_identity_object(
            &source_payload_geometry_identity,
            "source_payload_geometry_identity",
        )?;
        require_nonempty_identity_object(
            &strict_color_context_identity,
            "strict_color_context_identity",
        )?;
        require_nonempty_identity_object(
            &correction_context_identity,
            "correction_context_identity",
        )?;
        audio.validate()?;

        Ok(Self {
            frame_width,
            frame_height,
            frame_count,
            cadence,
            sample_aspect_ratio,
            display_aspect_ratio,
            bytes_per_frame,
            expected_total_bytes,
            correction_mode,
            source_payload_geometry_identity,
            strict_color_context_identity,
            correction_context_identity,
            audio,
        })
    }

    pub fn to_value(&self) -> Value {
        let mut sidecar = json!({
            "metadata_version": PIPE_METADATA_VERSION,
            "producer": PIPE_PRODUCER,
            "output_algorithm_id": PIPE_OUTPUT_ALGORITHM_ID,
            "pixel_format": PIPE_PIXEL_FORMAT,
            "rawvideo_pixel_format": PIPE_PIXEL_FORMAT,
            "mime_type": PIPE_MIME_TYPE,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "frame_count": self.frame_count,
            "fps_num": self.cadence.fps_num,
            "fps_den": self.cadence.fps_den,
            "timebase_num": self.cadence.timebase_num,
            "timebase_den": self.cadence.timebase_den,
            "video_track_timescale": self.cadence.video_track_timescale,
            "sample_aspect_ratio_num": self.sample_aspect_ratio.numerator,
            "sample_aspect_ratio_den": self.sample_aspect_ratio.denominator,
            "display_aspect_ratio_num": self.display_aspect_ratio.numerator,
            "display_aspect_ratio_den": self.display_aspect_ratio.denominator,
            "bytes_per_frame": self.bytes_per_frame,
            "expected_total_bytes": self.expected_total_bytes,
        });
        let fixed_contract = json!({
            "plane_order": Yuv444p12lePackPolicy::PLANE_ORDER,
            "bytes_per_sample": Yuv444p12lePackPolicy::BYTES_PER_SAMPLE,
            "storage_bytes_per_pixel": Yuv444p12lePackPolicy::STORAGE_BYTES_PER_PIXEL,
            "storage_bits_per_sample": Yuv444p12lePackPolicy::STORAGE_BITS_PER_SAMPLE,
            "meaningful_bits_per_sample": Yuv444p12lePackPolicy::MEANINGFUL_BITS_PER_SAMPLE,
            "bit_alignment": PIPE_BIT_ALIGNMENT,
            "byte_order": PIPE_BYTE_ORDER,
            "chroma_sampling": PIPE_CHROMA_SAMPLING,
            "alpha": PIPE_ALPHA,
            "color_range": PIPE_COLOR_RANGE,
            "color_primaries": PIPE_COLOR_PRIMARIES,
            "color_transfer": PIPE_COLOR_TRANSFER,
            "matrix_coefficients": PIPE_MATRIX_COEFFICIENTS,
            "nominal_y_codes": [
                Yuv444p12lePackPolicy::NOMINAL_LUMA_MIN_CODE,
                Yuv444p12lePackPolicy::NOMINAL_LUMA_MAX_CODE,
            ],
            "nominal_chroma_codes": [
                Yuv444p12lePackPolicy::NOMINAL_CHROMA_MIN_CODE,
                Yuv444p12lePackPolicy::NOMINAL_CHROMA_MAX_CODE,
            ],
            "chroma_center_code": Yuv444p12lePackPolicy::CHROMA_OFFSET,
            "final_code_bounds": [
                Yuv444p12lePackPolicy::MIN_CODE,
                Yuv444p12lePackPolicy::MAX_CODE,
            ],
            "rounding_id": PIPE_ROUNDING_ID,
            "terminal_exposure_scale_num": Yuv444p12lePackPolicy::TERMINAL_EXPOSURE_SCALE_NUMERATOR,
            "terminal_exposure_scale_den": Yuv444p12lePackPolicy::TERMINAL_EXPOSURE_SCALE_DENOMINATOR,
            "terminal_exposure_scale_stops": Yuv444p12lePackPolicy::TERMINAL_EXPOSURE_SCALE_STOPS,
            "terminal_exposure_scale_id": PIPE_TERMINAL_EXPOSURE_SCALE_ID,
            "scale_application_point": PIPE_SCALE_APPLICATION_POINT,
            "display_ready": false,
            "correction_policy_id": PIPE_CORRECTION_POLICY_ID,
            "correction_terminal_id": PIPE_CORRECTION_TERMINAL_ID,
            "correction_mode": self.correction_mode,
            "color_policy_id": self.strict_color_context_identity.get("policy_id").and_then(Value::as_str).unwrap_or(PIPE_COLOR_POLICY_ID),
            "scene_normalization_id": PIPE_SCENE_NORMALIZATION_ID,
            "source_payload_geometry_identity": self.source_payload_geometry_identity,
            "strict_color_context_identity": self.strict_color_context_identity,
            "correction_context_identity": self.correction_context_identity,
            "audio": self.audio.to_value(),
        });
        sidecar
            .as_object_mut()
            .expect("PIPE sidecar JSON literal is an object")
            .extend(
                fixed_contract
                    .as_object()
                    .expect("PIPE fixed-contract JSON literal is an object")
                    .clone(),
            );
        sidecar.as_object_mut().expect("sidecar object").extend(
            apple_log_contract_fields()
                .as_object()
                .expect("transfer object")
                .clone(),
        );
        sidecar
    }

    fn from_value(value: &Value) -> Result<Self, PipeContractError> {
        let object = required_object_value(value, "PIPE sidecar")?;
        if object.get("metadata_version").and_then(Value::as_u64) == Some(3) {
            return Err(PipeContractError::new(
                "legacy PIPE version 3 is half-scale linear (0.9.0); it cannot be interpreted as original Apple Log version 4",
            ));
        }
        for field in LEGACY_FORMAT_FIELDS {
            if object.contains_key(*field) {
                return Err(PipeContractError::new(format!(
                    "PIPE sidecar contains stale legacy field {field}"
                )));
            }
        }

        require_u64_equal(object, "metadata_version", PIPE_METADATA_VERSION)?;
        for (field, expected) in apple_log_contract_fields()
            .as_object()
            .expect("transfer object")
        {
            if object.get(field) != Some(expected) {
                return Err(PipeContractError::new(format!(
                    "PIPE sidecar {field} contradicts original Apple Log contract"
                )));
            }
        }
        require_string_equal(object, "producer", PIPE_PRODUCER)?;
        require_string_equal(object, "output_algorithm_id", PIPE_OUTPUT_ALGORITHM_ID)?;
        require_string_equal(object, "pixel_format", PIPE_PIXEL_FORMAT)?;
        require_string_equal(object, "rawvideo_pixel_format", PIPE_PIXEL_FORMAT)?;
        require_string_equal(object, "mime_type", PIPE_MIME_TYPE)?;
        require_string_array_equal(object, "plane_order", &Yuv444p12lePackPolicy::PLANE_ORDER)?;
        require_u64_equal(
            object,
            "bytes_per_sample",
            u64::from(Yuv444p12lePackPolicy::BYTES_PER_SAMPLE),
        )?;
        require_u64_equal(
            object,
            "storage_bytes_per_pixel",
            u64::from(Yuv444p12lePackPolicy::STORAGE_BYTES_PER_PIXEL),
        )?;
        require_u64_equal(
            object,
            "storage_bits_per_sample",
            u64::from(Yuv444p12lePackPolicy::STORAGE_BITS_PER_SAMPLE),
        )?;
        require_u64_equal(
            object,
            "meaningful_bits_per_sample",
            u64::from(Yuv444p12lePackPolicy::MEANINGFUL_BITS_PER_SAMPLE),
        )?;
        require_string_equal(object, "bit_alignment", PIPE_BIT_ALIGNMENT)?;
        require_string_equal(object, "byte_order", PIPE_BYTE_ORDER)?;
        require_string_equal(object, "chroma_sampling", PIPE_CHROMA_SAMPLING)?;
        require_string_equal(object, "alpha", PIPE_ALPHA)?;
        require_string_equal(object, "color_range", PIPE_COLOR_RANGE)?;
        require_string_equal(object, "color_primaries", PIPE_COLOR_PRIMARIES)?;
        require_string_equal(object, "color_transfer", PIPE_COLOR_TRANSFER)?;
        require_string_equal(object, "matrix_coefficients", PIPE_MATRIX_COEFFICIENTS)?;
        require_u64_array_equal(
            object,
            "nominal_y_codes",
            &[
                u64::from(Yuv444p12lePackPolicy::NOMINAL_LUMA_MIN_CODE),
                u64::from(Yuv444p12lePackPolicy::NOMINAL_LUMA_MAX_CODE),
            ],
        )?;
        require_u64_array_equal(
            object,
            "nominal_chroma_codes",
            &[
                u64::from(Yuv444p12lePackPolicy::NOMINAL_CHROMA_MIN_CODE),
                u64::from(Yuv444p12lePackPolicy::NOMINAL_CHROMA_MAX_CODE),
            ],
        )?;
        require_u64_equal(
            object,
            "chroma_center_code",
            u64::from(Yuv444p12lePackPolicy::CHROMA_OFFSET),
        )?;
        require_u64_array_equal(
            object,
            "final_code_bounds",
            &[
                u64::from(Yuv444p12lePackPolicy::MIN_CODE),
                u64::from(Yuv444p12lePackPolicy::MAX_CODE),
            ],
        )?;
        require_string_equal(object, "rounding_id", PIPE_ROUNDING_ID)?;
        require_u64_equal(
            object,
            "terminal_exposure_scale_num",
            u64::from(Yuv444p12lePackPolicy::TERMINAL_EXPOSURE_SCALE_NUMERATOR),
        )?;
        require_u64_equal(
            object,
            "terminal_exposure_scale_den",
            u64::from(Yuv444p12lePackPolicy::TERMINAL_EXPOSURE_SCALE_DENOMINATOR),
        )?;
        require_i64_equal(
            object,
            "terminal_exposure_scale_stops",
            i64::from(Yuv444p12lePackPolicy::TERMINAL_EXPOSURE_SCALE_STOPS),
        )?;
        require_string_equal(
            object,
            "terminal_exposure_scale_id",
            PIPE_TERMINAL_EXPOSURE_SCALE_ID,
        )?;
        require_string_equal(
            object,
            "scale_application_point",
            PIPE_SCALE_APPLICATION_POINT,
        )?;
        require_bool_equal(object, "display_ready", false)?;
        require_string_equal(object, "correction_policy_id", PIPE_CORRECTION_POLICY_ID)?;
        require_string_equal(
            object,
            "correction_terminal_id",
            PIPE_CORRECTION_TERMINAL_ID,
        )?;
        match object.get("color_policy_id").and_then(Value::as_str) {
            Some(
                PIPE_COLOR_POLICY_ID
                | "StrictMotionCamColorMatrixColorV1"
                | "MixedMotionCamColorProfilesV1",
            ) => {}
            _ => return Err(PipeContractError::new("unsupported PIPE color policy")),
        }
        require_string_equal(
            object,
            "scene_normalization_id",
            PIPE_SCENE_NORMALIZATION_ID,
        )?;

        let frame_width = required_u32(object, "frame_width")?;
        let frame_height = required_u32(object, "frame_height")?;
        let frame_count = required_u64(object, "frame_count")?;
        let cadence = PipeMovCadence::from_sidecar_fields(
            required_u64(object, "fps_num")?,
            required_u64(object, "fps_den")?,
            required_u64(object, "timebase_num")?,
            required_u64(object, "timebase_den")?,
            required_u64(object, "video_track_timescale")?,
        )?;
        let sample_aspect_ratio = exact_reduced_aspect(
            required_u64(object, "sample_aspect_ratio_num")?,
            required_u64(object, "sample_aspect_ratio_den")?,
            "sample aspect ratio",
        )?;
        let display_aspect_ratio = exact_reduced_aspect(
            required_u64(object, "display_aspect_ratio_num")?,
            required_u64(object, "display_aspect_ratio_den")?,
            "display aspect ratio",
        )?;
        let source_payload_geometry_identity =
            required_value(object, "source_payload_geometry_identity")?.clone();
        let strict_color_context_identity =
            required_value(object, "strict_color_context_identity")?.clone();
        let correction_context_identity =
            required_value(object, "correction_context_identity")?.clone();
        let audio = PipeAudioContractV4::from_value(required_value(object, "audio")?)?;
        let result = Self::new(
            frame_width,
            frame_height,
            frame_count,
            cadence,
            sample_aspect_ratio,
            display_aspect_ratio,
            required_string(object, "correction_mode")?.to_string(),
            source_payload_geometry_identity,
            strict_color_context_identity,
            correction_context_identity,
            audio,
        )?;
        if required_u64(object, "bytes_per_frame")? != result.bytes_per_frame {
            return Err(PipeContractError::new(
                "PIPE sidecar bytes_per_frame contradicts dimensions and six-byte storage",
            ));
        }
        if required_u64(object, "expected_total_bytes")? != result.expected_total_bytes {
            return Err(PipeContractError::new(
                "PIPE sidecar expected_total_bytes contradicts frame geometry/count",
            ));
        }
        Ok(result)
    }
}

pub fn validate_pipe_sidecar_v4(value: &Value) -> Result<PipeSidecarV4, PipeContractError> {
    PipeSidecarV4::from_value(value)
}

fn exact_reduced_aspect(
    numerator: u64,
    denominator: u64,
    label: &str,
) -> Result<PipeAspectRatio, PipeContractError> {
    let reduced = PipeAspectRatio::new(numerator, denominator)?;
    if reduced.numerator != numerator || reduced.denominator != denominator {
        return Err(PipeContractError::new(format!(
            "PIPE {label} must be an exact reduced rational"
        )));
    }
    Ok(reduced)
}

fn require_nonempty_identity_object(value: &Value, field: &str) -> Result<(), PipeContractError> {
    let object = value
        .as_object()
        .ok_or_else(|| PipeContractError::new(format!("PIPE {field} must be a JSON object")))?;
    if object.is_empty() {
        return Err(PipeContractError::new(format!(
            "PIPE {field} must not be empty"
        )));
    }
    Ok(())
}

fn required_object_value<'a>(
    value: &'a Value,
    label: &str,
) -> Result<&'a Map<String, Value>, PipeContractError> {
    value
        .as_object()
        .ok_or_else(|| PipeContractError::new(format!("{label} must be a JSON object")))
}

fn required_value<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a Value, PipeContractError> {
    object
        .get(field)
        .ok_or_else(|| PipeContractError::new(format!("PIPE sidecar is missing {field}")))
}

fn required_u64(object: &Map<String, Value>, field: &str) -> Result<u64, PipeContractError> {
    required_value(object, field)?
        .as_u64()
        .ok_or_else(|| PipeContractError::new(format!("PIPE sidecar {field} must be a u64")))
}

fn required_u32(object: &Map<String, Value>, field: &str) -> Result<u32, PipeContractError> {
    u32::try_from(required_u64(object, field)?)
        .map_err(|_| PipeContractError::new(format!("PIPE sidecar {field} does not fit u32")))
}

fn optional_u64(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<u64>, PipeContractError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value.as_u64().map(Some).ok_or_else(|| {
            PipeContractError::new(format!("PIPE sidecar {field} must be a u64 or null"))
        }),
    }
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a str, PipeContractError> {
    required_value(object, field)?
        .as_str()
        .ok_or_else(|| PipeContractError::new(format!("PIPE sidecar {field} must be a string")))
}

fn optional_string(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<String>, PipeContractError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(PipeContractError::new(format!(
            "PIPE sidecar {field} must be a string or null"
        ))),
    }
}

fn required_bool(object: &Map<String, Value>, field: &str) -> Result<bool, PipeContractError> {
    required_value(object, field)?
        .as_bool()
        .ok_or_else(|| PipeContractError::new(format!("PIPE sidecar {field} must be a boolean")))
}

fn require_u64_equal(
    object: &Map<String, Value>,
    field: &str,
    expected: u64,
) -> Result<(), PipeContractError> {
    let actual = required_u64(object, field)?;
    if actual != expected {
        return Err(PipeContractError::new(format!(
            "PIPE sidecar {field} must equal {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn require_i64_equal(
    object: &Map<String, Value>,
    field: &str,
    expected: i64,
) -> Result<(), PipeContractError> {
    let actual = required_value(object, field)?
        .as_i64()
        .ok_or_else(|| PipeContractError::new(format!("PIPE sidecar {field} must be an i64")))?;
    if actual != expected {
        return Err(PipeContractError::new(format!(
            "PIPE sidecar {field} must equal {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn require_string_equal(
    object: &Map<String, Value>,
    field: &str,
    expected: &str,
) -> Result<(), PipeContractError> {
    let actual = required_string(object, field)?;
    if actual != expected {
        return Err(PipeContractError::new(format!(
            "PIPE sidecar {field} must equal {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn require_bool_equal(
    object: &Map<String, Value>,
    field: &str,
    expected: bool,
) -> Result<(), PipeContractError> {
    let actual = required_bool(object, field)?;
    if actual != expected {
        return Err(PipeContractError::new(format!(
            "PIPE sidecar {field} must equal {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn require_string_array_equal(
    object: &Map<String, Value>,
    field: &str,
    expected: &[&str],
) -> Result<(), PipeContractError> {
    let actual = required_value(object, field)?
        .as_array()
        .ok_or_else(|| PipeContractError::new(format!("PIPE sidecar {field} must be an array")))?;
    let matches = actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.as_str() == Some(*expected));
    if !matches {
        return Err(PipeContractError::new(format!(
            "PIPE sidecar {field} contradicts the canonical sequence"
        )));
    }
    Ok(())
}

fn require_u64_array_equal(
    object: &Map<String, Value>,
    field: &str,
    expected: &[u64],
) -> Result<(), PipeContractError> {
    let actual = required_value(object, field)?
        .as_array()
        .ok_or_else(|| PipeContractError::new(format!("PIPE sidecar {field} must be an array")))?;
    let matches = actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.as_u64() == Some(*expected));
    if !matches {
        return Err(PipeContractError::new(format!(
            "PIPE sidecar {field} contradicts the canonical code interval"
        )));
    }
    Ok(())
}
