// Parses MotionCam audio metadata into the small set of values needed by the CPU
// PCM decoder and timing diagnostics.
//
// The creator's C++ decoder reads audio sample rate and channel count from
// container metadata extraData fields. Some AUDIO_DATA_METADATA payloads are
// binary timestamp structs rather than JSON, so JSON audio metadata remains
// optional and non-fatal here.

use mcraw4vulkan_core::FrameRate;
use serde_json::Value;

use crate::error::McrawContainerError as DecodeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioDataMetadata {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub reported_frame_rate: Option<FrameRate>,
}

impl AudioDataMetadata {
    pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 48_000;
    pub const DEFAULT_CHANNELS: u16 = 2;
    pub const DEFAULT_BITS_PER_SAMPLE: u16 = 16;

    pub fn parse_sources(
        container_metadata_json: &str,
        json_values: &[String],
    ) -> Result<Self, DecodeError> {
        let mut metadata = Self::default();

        if let Ok(value) = serde_json::from_str::<Value>(container_metadata_json) {
            metadata.apply_json_value(&value)?;
        }

        for json in json_values {
            let Ok(value) = serde_json::from_str::<Value>(json) else {
                // Some clips contain AUDIO_DATA_METADATA payloads that are valid
                // UTF-8 but not JSON. Ignore those so real PCM decode can
                // proceed from AUDIO_DATA payload sizes and binary timestamps.
                continue;
            };

            metadata.apply_json_value(&value)?;
        }

        metadata.validate()?;
        Ok(metadata)
    }

    fn apply_json_value(&mut self, value: &Value) -> Result<(), DecodeError> {
        if let Some(sample_rate_hz) = find_u64_by_keys(
            value,
            &[
                "audioSampleRate",
                "audio_sample_rate",
                "audioSampleRateHz",
                "audio_sample_rate_hz",
                "sampleRate",
                "sample_rate",
                "sampleRateHz",
                "sample_rate_hz",
            ],
        ) {
            self.sample_rate_hz = u32::try_from(sample_rate_hz).map_err(|_| {
                DecodeError::UnsupportedFormat("audio sample rate overflows u32".to_string())
            })?;
        }

        if let Some(channels) = find_u64_by_keys(
            value,
            &[
                "audioChannels",
                "audio_channels",
                "channels",
                "channelCount",
                "channel_count",
                "numChannels",
                "num_channels",
            ],
        ) {
            self.channels = u16::try_from(channels).map_err(|_| {
                DecodeError::UnsupportedFormat("audio channel count overflows u16".to_string())
            })?;
        }

        if let Some(bits_per_sample) = find_u64_by_keys(
            value,
            &[
                "bitsPerSample",
                "bits_per_sample",
                "bitDepth",
                "bit_depth",
                "audioBitsPerSample",
                "audio_bits_per_sample",
            ],
        ) {
            self.bits_per_sample = u16::try_from(bits_per_sample).map_err(|_| {
                DecodeError::UnsupportedFormat("audio bit depth overflows u16".to_string())
            })?;
        }

        if self.reported_frame_rate.is_none() {
            self.reported_frame_rate = find_frame_rate_by_keys(
                value,
                &[
                    "fps",
                    "frameRate",
                    "frame_rate",
                    "audioFps",
                    "audio_fps",
                    "audioFrameRate",
                    "audio_frame_rate",
                ],
            );
        }

        Ok(())
    }

    fn validate(self) -> Result<(), DecodeError> {
        if self.sample_rate_hz == 0 {
            return Err(DecodeError::UnsupportedFormat(
                "audio sample rate must be non-zero".to_string(),
            ));
        }

        if self.channels == 0 {
            return Err(DecodeError::UnsupportedFormat(
                "audio channel count must be non-zero".to_string(),
            ));
        }

        if self.bits_per_sample != 16 {
            return Err(DecodeError::UnsupportedFormat(format!(
                "only 16-bit PCM audio is currently supported, found {} bits",
                self.bits_per_sample
            )));
        }

        Ok(())
    }
}

impl Default for AudioDataMetadata {
    fn default() -> Self {
        Self {
            sample_rate_hz: Self::DEFAULT_SAMPLE_RATE_HZ,
            channels: Self::DEFAULT_CHANNELS,
            bits_per_sample: Self::DEFAULT_BITS_PER_SAMPLE,
            reported_frame_rate: None,
        }
    }
}

fn find_u64_by_keys(value: &Value, keys: &[&str]) -> Option<u64> {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                if keys
                    .iter()
                    .any(|candidate| key.eq_ignore_ascii_case(candidate))
                {
                    if let Some(number) = value_to_u64(child) {
                        return Some(number);
                    }
                }
            }

            for child in map.values() {
                if let Some(number) = find_u64_by_keys(child, keys) {
                    return Some(number);
                }
            }

            None
        }
        Value::Array(items) => items.iter().find_map(|child| find_u64_by_keys(child, keys)),
        _ => None,
    }
}

fn find_frame_rate_by_keys(value: &Value, keys: &[&str]) -> Option<FrameRate> {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                if keys
                    .iter()
                    .any(|candidate| key.eq_ignore_ascii_case(candidate))
                {
                    if let Some(frame_rate) = value_to_frame_rate(child) {
                        return Some(frame_rate);
                    }
                }
            }

            for child in map.values() {
                if let Some(frame_rate) = find_frame_rate_by_keys(child, keys) {
                    return Some(frame_rate);
                }
            }

            None
        }
        Value::Array(items) => items
            .iter()
            .find_map(|child| find_frame_rate_by_keys(child, keys)),
        _ => None,
    }
}

fn value_to_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number
            .as_u64()
            .or_else(|| number.as_i64().and_then(|value| u64::try_from(value).ok())),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }
}

fn value_to_frame_rate(value: &Value) -> Option<FrameRate> {
    match value {
        Value::Number(number) => decimal_text_to_frame_rate(&number.to_string()),
        Value::String(text) => decimal_text_to_frame_rate(text.trim()),
        Value::Object(map) => {
            let numerator = map
                .get("numerator")
                .or_else(|| map.get("num"))
                .and_then(value_to_u64)?;

            let denominator = map
                .get("denominator")
                .or_else(|| map.get("den"))
                .and_then(value_to_u64)?;

            if numerator == 0 || denominator == 0 {
                return None;
            }

            Some(FrameRate::new(numerator, denominator).reduced())
        }
        _ => None,
    }
}

fn decimal_text_to_frame_rate(text: &str) -> Option<FrameRate> {
    let text = text.trim();
    if text.is_empty() || text.starts_with('-') {
        return None;
    }

    if let Some((left, right)) = text.split_once('/') {
        let numerator = left.trim().parse::<u64>().ok()?;
        let denominator = right.trim().parse::<u64>().ok()?;

        if numerator == 0 || denominator == 0 {
            return None;
        }

        return Some(FrameRate::new(numerator, denominator).reduced());
    }

    if let Some((whole, fraction)) = text.split_once('.') {
        let whole = whole.trim().parse::<u64>().ok()?;
        let fraction = fraction.trim();

        if fraction.is_empty() || !fraction.chars().all(|value| value.is_ascii_digit()) {
            return None;
        }

        let denominator = 10u64.checked_pow(u32::try_from(fraction.len()).ok()?)?;
        let fraction_value = fraction.parse::<u64>().ok()?;
        let numerator = whole
            .checked_mul(denominator)?
            .checked_add(fraction_value)?;

        if numerator == 0 {
            return None;
        }

        return Some(FrameRate::new(numerator, denominator).reduced());
    }

    let numerator = text.parse::<u64>().ok()?;
    if numerator == 0 {
        return None;
    }

    Some(FrameRate::new(numerator, 1))
}
