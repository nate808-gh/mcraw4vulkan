use std::error::Error;
use std::fmt;

use mcraw4vulkan_core::{FrameDimensions, FramePayloadLayout};
use serde_json::Value;

use crate::error::McrawContainerError as DecodeError;
use crate::lens_shading_map::LensShadingMap;

// Parsed per-frame metadata from the .mcraw frame metadata JSON.
//
// This struct is owned by mcraw4vulkan-mcrawcontainer because it represents
// MotionCam/.mcraw container metadata that accompanies each frame, not DNG tags
// directly. Output layers can map this typed metadata into their own formats.
#[derive(Debug, Clone)]
pub struct FrameMetadata {
    pub dimensions: FrameDimensions,
    pub original_width: Option<u32>,
    pub original_height: Option<u32>,
    pub row_stride: Option<u32>,
    pub pixel_format: Option<String>,
    pub compression_type: CompressionType,
    pub compression_type_raw: Option<String>,
    pub is_binned: Option<bool>,
    pub is_compressed: Option<bool>,
    pub need_remosaic: Option<bool>,
    pub dynamic_white_level: Option<f64>,
    pub dynamic_black_level: Option<[f64; 4]>,
    pub lens_shading_map: Option<LensShadingMap>,
    pub as_shot_neutral: Option<[f64; 3]>,
    pub color_overrides: crate::ColorMetadataOverrides,
}

// Finite known compression categories from the frame metadata.
//
// Unknown values are preserved rather than rejected so payload support decisions
// retain the source token instead of losing unrecognized metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionType {
    Lossless,
    Legacy,
    Uncompressed,
    MotionCamType6,
    MotionCamType7,
    Unknown(String),
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedFramePayloadLayout {
    pub compression_type: CompressionType,
    pub compression_type_raw: Option<String>,
    pub pixel_format: Option<String>,
    pub is_binned: Option<bool>,
    pub is_compressed: Option<bool>,
    pub need_remosaic: Option<bool>,
    pub row_stride: Option<u32>,
    pub width: u32,
    pub height: u32,
}

impl fmt::Display for UnsupportedFramePayloadLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unsupported MotionCam payload layout: compressionType {}, pixelFormat {}, isBinned {}, isCompressed {}, needRemosaic {}, rowStride {}, dimensions {}x{}",
            self.compression_type_label(),
            option_string(self.pixel_format.as_deref()),
            option_bool(self.is_binned),
            option_bool(self.is_compressed),
            option_bool(self.need_remosaic),
            option_u32(self.row_stride),
            self.width,
            self.height,
        )
    }
}

impl Error for UnsupportedFramePayloadLayout {}

impl FrameMetadata {
    // Parse the JSON metadata that follows each frame BUFFER item. Width and height
    // are required because they define visible output dimensions; optional fields
    // preserve absence for payload-layout and output-policy decisions.
    pub fn parse(frame_metadata_json: &str) -> Result<Self, DecodeError> {
        let value: Value = serde_json::from_str(frame_metadata_json).map_err(|err| {
            DecodeError::UnsupportedFormat(format!("invalid frame metadata json: {err}"))
        })?;

        let dimensions = parse_dimensions(&value)?;
        let original_width = parse_optional_u32(&value, "originalWidth")?;
        let original_height = parse_optional_u32(&value, "originalHeight")?;
        let row_stride = parse_optional_u32(&value, "rowStride")?;
        let pixel_format = parse_optional_string(&value, "pixelFormat")?;
        let compression_type = parse_compression_type(&value);
        let compression_type_raw = parse_raw_value_for_diagnostics(&value, "compressionType");
        let is_binned = parse_optional_bool(&value, "isBinned")?;
        let is_compressed = parse_optional_bool(&value, "isCompressed")?;
        let need_remosaic = parse_optional_bool(&value, "needRemosaic")?;
        let dynamic_white_level = parse_optional_f64(&value, "dynamicWhiteLevel")?;
        let dynamic_black_level = parse_level_values(&value, "dynamicBlackLevel")?;
        let lens_shading_map = parse_lens_shading_map(&value)?;
        let as_shot_neutral = parse_as_shot_neutral(&value)?;

        Ok(Self {
            dimensions,
            original_width,
            original_height,
            row_stride,
            pixel_format,
            compression_type,
            compression_type_raw,
            is_binned,
            is_compressed,
            need_remosaic,
            dynamic_white_level,
            dynamic_black_level,
            lens_shading_map,
            as_shot_neutral,
            color_overrides: crate::ColorMetadataOverrides::parse(&value)
                .map_err(|e| DecodeError::UnsupportedFormat(e.to_string()))?,
        })
    }

    pub fn payload_layout(&self) -> Result<FramePayloadLayout, UnsupportedFramePayloadLayout> {
        match self.compression_type {
            CompressionType::Lossless
            | CompressionType::Legacy
            | CompressionType::MotionCamType7 => Ok(FramePayloadLayout::CompressedRawcodecType7),
            CompressionType::MotionCamType6
                if self.pixel_format_is("raw16")
                    && self.is_binned == Some(true)
                    && self.need_remosaic != Some(true) =>
            {
                let Some(row_stride) = self.row_stride else {
                    return Err(self.unsupported_payload_layout());
                };
                Ok(FramePayloadLayout::BinnedRaw16Type6 { row_stride })
            }
            _ => Err(self.unsupported_payload_layout()),
        }
    }

    pub fn payload_layout_diagnostic(&self) -> UnsupportedFramePayloadLayout {
        self.unsupported_payload_layout()
    }

    fn pixel_format_is(&self, expected: &str) -> bool {
        self.pixel_format
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case(expected))
    }

    fn unsupported_payload_layout(&self) -> UnsupportedFramePayloadLayout {
        UnsupportedFramePayloadLayout {
            compression_type: self.compression_type.clone(),
            compression_type_raw: self.compression_type_raw.clone(),
            pixel_format: self.pixel_format.clone(),
            is_binned: self.is_binned,
            is_compressed: self.is_compressed,
            need_remosaic: self.need_remosaic,
            row_stride: self.row_stride,
            width: self.dimensions.width,
            height: self.dimensions.height,
        }
    }
}

impl UnsupportedFramePayloadLayout {
    fn compression_type_label(&self) -> String {
        self.compression_type
            .diagnostic_label(self.compression_type_raw.as_deref())
    }
}

impl CompressionType {
    pub fn diagnostic_label(&self, raw: Option<&str>) -> String {
        match self {
            Self::Lossless => "lossless".to_string(),
            Self::Legacy => "legacy".to_string(),
            Self::Uncompressed => "uncompressed".to_string(),
            Self::MotionCamType6 => "6".to_string(),
            Self::MotionCamType7 => "7".to_string(),
            Self::Unknown(value) => raw
                .map(str::to_string)
                .unwrap_or_else(|| format!("unknown:{value}")),
            Self::Missing => "(missing)".to_string(),
        }
    }
}

// Read the visible frame dimensions from the per-frame metadata.
//
// These dimensions can be smaller than the encoded raw payload dimensions.
// For example, the encoded row may be padded to a 64-pixel block boundary while
// the visible image width is cropped to the real active width.
fn parse_dimensions(value: &Value) -> Result<FrameDimensions, DecodeError> {
    let width = value.get("width").and_then(Value::as_u64).ok_or_else(|| {
        DecodeError::UnsupportedFormat("frame metadata missing width".to_string())
    })?;

    let height = value.get("height").and_then(Value::as_u64).ok_or_else(|| {
        DecodeError::UnsupportedFormat("frame metadata missing height".to_string())
    })?;

    let width = u32::try_from(width)
        .map_err(|_| DecodeError::UnsupportedFormat("frame metadata width overflow".to_string()))?;

    let height = u32::try_from(height).map_err(|_| {
        DecodeError::UnsupportedFormat("frame metadata height overflow".to_string())
    })?;

    if width == 0 || height == 0 {
        return Err(DecodeError::UnsupportedFormat(
            "frame metadata dimensions must be non-zero".to_string(),
        ));
    }

    Ok(FrameDimensions { width, height })
}

// Read the compression type as a typed enum while preserving unknown strings.
//
// The raw decoder validates the payload independently of this metadata
// classification, so an inaccurate label cannot bypass byte-layout checks.
fn parse_compression_type(value: &Value) -> CompressionType {
    let Some(raw_value) = value.get("compressionType") else {
        return CompressionType::Missing;
    };

    if let Some(number) = raw_value.as_u64() {
        return match number {
            6 => CompressionType::MotionCamType6,
            7 => CompressionType::MotionCamType7,
            other => CompressionType::Unknown(other.to_string()),
        };
    }

    let Some(raw) = raw_value.as_str() else {
        return CompressionType::Unknown(raw_value.to_string());
    };

    match raw {
        "lossless" => CompressionType::Lossless,
        "legacy" => CompressionType::Legacy,
        "uncompressed" => CompressionType::Uncompressed,
        "6" => CompressionType::MotionCamType6,
        "7" => CompressionType::MotionCamType7,
        other => CompressionType::Unknown(other.to_string()),
    }
}

// Preserve the optional three-component AsShotNeutral source vector. Decode does
// not require it; output layers own any format-specific mapping.
fn parse_as_shot_neutral(value: &Value) -> Result<Option<[f64; 3]>, DecodeError> {
    let Some(items) = value.get("asShotNeutral").and_then(Value::as_array) else {
        return Ok(None);
    };

    if items.len() != 3 {
        return Err(DecodeError::UnsupportedFormat(
            "asShotNeutral must contain exactly 3 values".to_string(),
        ));
    }

    let mut neutral = [0.0_f64; 3];

    for (index, item) in items.iter().enumerate() {
        neutral[index] = item.as_f64().ok_or_else(|| {
            DecodeError::UnsupportedFormat("asShotNeutral contains a non-number".to_string())
        })?;
    }

    Ok(Some(neutral))
}

// Parse black-level style metadata from a frame field such as dynamicBlackLevel.
//
// Accept the same scalar, one-value, or four-CFA-position shapes as container
// levels so both metadata scopes share one normalization contract.
fn parse_level_values(value: &Value, key: &str) -> Result<Option<[f64; 4]>, DecodeError> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };

    if let Some(level) = parse_f64_value(raw, key)? {
        return Ok(Some([level, level, level, level]));
    }

    let Some(items) = raw.as_array() else {
        return Err(DecodeError::UnsupportedFormat(format!(
            "{key} must be a number or an array"
        )));
    };

    if items.len() == 1 {
        let level = parse_required_f64_value(&items[0], key)?;
        return Ok(Some([level, level, level, level]));
    }

    if items.len() != 4 {
        return Err(DecodeError::UnsupportedFormat(format!(
            "{key} must contain either 1 or 4 values"
        )));
    }

    let mut values = [0.0_f64; 4];

    for (index, item) in items.iter().enumerate() {
        values[index] = parse_required_f64_value(item, key)?;
    }

    Ok(Some(values))
}

// Parse the optional frame lensShadingMap metadata using runtime dimensions.
//
// Other devices already use different map sizes, so this must validate against
// width * height from the same frame metadata instead of fixed constants.
fn parse_lens_shading_map(value: &Value) -> Result<Option<LensShadingMap>, DecodeError> {
    let width_present = value.get("lensShadingMapWidth").is_some();
    let height_present = value.get("lensShadingMapHeight").is_some();
    let map_present = value.get("lensShadingMap").is_some();

    if !width_present && !height_present && !map_present {
        return Ok(None);
    }

    if !(width_present && height_present && map_present) {
        return Err(DecodeError::UnsupportedFormat(
            "lensShadingMap metadata must include width, height, and map values".to_string(),
        ));
    }

    let width = parse_required_u32(value, "lensShadingMapWidth")?;
    let height = parse_required_u32(value, "lensShadingMapHeight")?;
    let expected_len = expected_lens_shading_plane_len(width, height)?;
    let raw_planes = value
        .get("lensShadingMap")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            DecodeError::UnsupportedFormat("lensShadingMap must be an array".to_string())
        })?;

    let mut planes = Vec::with_capacity(raw_planes.len());

    for (plane_index, raw_plane) in raw_planes.iter().enumerate() {
        let raw_values = raw_plane.as_array().ok_or_else(|| {
            DecodeError::UnsupportedFormat(format!(
                "lensShadingMap[{plane_index}] must be an array"
            ))
        })?;

        if raw_values.len() != expected_len {
            return Err(DecodeError::UnsupportedFormat(format!(
                "lensShadingMap[{plane_index}] len {} does not match width * height {expected_len}",
                raw_values.len()
            )));
        }

        let mut plane = Vec::with_capacity(raw_values.len());

        for (sample_index, raw_value) in raw_values.iter().enumerate() {
            let gain = parse_required_f32_value(raw_value, "lensShadingMap")?;

            if gain < 0.0 {
                return Err(DecodeError::UnsupportedFormat(format!(
                    "lensShadingMap[{plane_index}][{sample_index}] must not be negative"
                )));
            }

            plane.push(gain);
        }

        planes.push(plane);
    }

    LensShadingMap::new(width, height, planes)
        .map(Some)
        .map_err(|error| DecodeError::UnsupportedFormat(error.to_string()))
}

// Parse an optional JSON string field while rejecting non-string values.
fn parse_optional_string(value: &Value, key: &str) -> Result<Option<String>, DecodeError> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };

    let Some(string) = raw.as_str() else {
        return Err(DecodeError::UnsupportedFormat(format!(
            "{key} must be a string"
        )));
    };

    Ok(Some(string.to_string()))
}

// Parse an optional non-negative integer that fits frame dimension diagnostics.
fn parse_optional_u32(value: &Value, key: &str) -> Result<Option<u32>, DecodeError> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };

    let Some(number) = raw.as_u64() else {
        return Err(DecodeError::UnsupportedFormat(format!(
            "{key} must be a non-negative integer"
        )));
    };

    let number = u32::try_from(number)
        .map_err(|_| DecodeError::UnsupportedFormat(format!("{key} overflows u32")))?;

    Ok(Some(number))
}

fn parse_optional_bool(value: &Value, key: &str) -> Result<Option<bool>, DecodeError> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };

    raw.as_bool()
        .map(Some)
        .ok_or_else(|| DecodeError::UnsupportedFormat(format!("{key} must be a boolean")))
}

fn parse_required_u32(value: &Value, key: &str) -> Result<u32, DecodeError> {
    parse_optional_u32(value, key)?
        .ok_or_else(|| DecodeError::UnsupportedFormat(format!("{key} is required")))
}

fn parse_optional_f64(value: &Value, key: &str) -> Result<Option<f64>, DecodeError> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };

    parse_f64_value(raw, key)
}

// Preserve the raw compressionType token for diagnostics when it is numeric.
fn parse_raw_value_for_diagnostics(value: &Value, key: &str) -> Option<String> {
    let raw = value.get(key)?;

    match raw {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => Some(raw.to_string()),
    }
}

fn option_string(value: Option<&str>) -> String {
    value.unwrap_or("(missing)").to_string()
}

fn option_bool(value: Option<bool>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "(missing)".to_string())
}

fn option_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "(missing)".to_string())
}

// Parse one JSON numeric value into f64.
fn parse_required_f64_value(value: &Value, key: &str) -> Result<f64, DecodeError> {
    parse_f64_value(value, key)?
        .ok_or_else(|| DecodeError::UnsupportedFormat(format!("{key} must be a numeric value")))
}

fn parse_required_f32_value(value: &Value, key: &str) -> Result<f32, DecodeError> {
    let value = parse_required_f64_value(value, key)?;

    if value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(DecodeError::UnsupportedFormat(format!(
            "{key} value overflows f32"
        )));
    }

    let value = value as f32;

    if !value.is_finite() {
        return Err(DecodeError::UnsupportedFormat(format!(
            "{key} must be finite"
        )));
    }

    Ok(value)
}

// Parse one JSON value into finite f64 if it is numeric.
fn parse_f64_value(value: &Value, key: &str) -> Result<Option<f64>, DecodeError> {
    let Some(number) = value.as_f64() else {
        return Ok(None);
    };

    if !number.is_finite() {
        return Err(DecodeError::UnsupportedFormat(format!(
            "{key} must be finite"
        )));
    }

    Ok(Some(number))
}

fn expected_lens_shading_plane_len(width: u32, height: u32) -> Result<usize, DecodeError> {
    if width == 0 || height == 0 {
        return Err(DecodeError::UnsupportedFormat(
            "lensShadingMap dimensions must be non-zero".to_string(),
        ));
    }

    let len = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| {
            DecodeError::UnsupportedFormat("lensShadingMap dimensions overflow".to_string())
        })?;

    usize::try_from(len).map_err(|_| {
        DecodeError::UnsupportedFormat("lensShadingMap plane length overflows usize".to_string())
    })
}
