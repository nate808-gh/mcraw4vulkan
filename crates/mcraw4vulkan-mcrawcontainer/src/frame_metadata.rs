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
    /// No per-frame crop, origin, transform or CFA override requiring another mapping.
    pub origin_zero_mapping: bool,
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
    /// Read only geometry/layout facts. Color and lens-map interpretation stays lazy.
    pub(crate) fn parse_geometry(json: &str) -> Result<(Self, bool), DecodeError> {
        let value: Value =
            serde_json::from_str(json).map_err(|e| DecodeError::InvalidMetadata(e.to_string()))?;
        let understood_origin = origin_zero_mapping(&value);
        let mut projection = serde_json::Map::new();
        for key in [
            "width",
            "height",
            "rowStride",
            "pixelFormat",
            "compressionType",
            "isBinned",
            "isCompressed",
            "needRemosaic",
            "dynamicWhiteLevel",
            "dynamicBlackLevel",
        ] {
            if let Some(v) = value.get(key) {
                projection.insert(key.to_owned(), v.clone());
            }
        }
        Ok((
            Self::parse(&Value::Object(projection).to_string())?,
            understood_origin,
        ))
    }

    /// Sole geometry policy. Normal layouts retain their declared extent.
    pub fn resolve_output_geometry(
        &self,
        evidence: Option<mcraw4vulkan_core::Type7GeometryEvidence>,
        known_origin_zero_bayer: bool,
    ) -> Result<mcraw4vulkan_core::ResolvedFrameGeometry, DecodeError> {
        use mcraw4vulkan_core::{GeometryRecoveryReason, ResolvedFrameGeometry};
        let layout = self
            .payload_layout()
            .map_err(|e| DecodeError::UnsupportedFormat(e.to_string()))?;
        let declared = self.dimensions;
        let invalid = |message: &str| DecodeError::UnsupportedFormat(message.to_owned());
        if declared.width == 0 || declared.height == 0 {
            return Err(invalid("visible dimensions must be non-zero"));
        }
        let mut result = ResolvedFrameGeometry {
            layout_guess: None,
            declared,
            encoded: None,
            effective: declared,
            origin: [0, 0],
            reason: None,
            type7: None,
        };
        if matches!(layout, FramePayloadLayout::BinnedRaw16Type6 { .. }) {
            return Ok(result);
        }
        let evidence = evidence.ok_or_else(|| invalid("missing type-7 geometry header"))?;
        let h = evidence.header;
        let encoded = FrameDimensions {
            width: h.encoded_width,
            height: h.encoded_height,
        };
        if encoded.width == 0
            || encoded.height == 0
            || encoded.width & 63 != 0
            || encoded.height & 3 != 0
            || encoded.width > 16_384
            || encoded.height > 16_384
            || u64::from(declared.width) * u64::from(declared.height) > 100_000_000
            || h.bits_offset > evidence.payload_len
            || h.refs_offset > evidence.payload_len
        {
            return Err(invalid("invalid type-7 geometry or metadata offsets"));
        }
        result.encoded = Some(encoded);
        result.type7 = Some(evidence);
        if declared.width <= encoded.width && declared.height <= encoded.height {
            return Ok(result);
        }
        let rounded_width = declared.width.checked_add(63).map(|n| n / 64 * 64);
        if !known_origin_zero_bayer
            || !self.origin_zero_mapping
            || self.compression_type != CompressionType::MotionCamType7
            || !self.pixel_format_is("raw16")
            || self.need_remosaic != Some(false)
            || self.is_compressed == Some(false)
            || declared.height & 3 == 0
            || encoded.height != declared.height / 4 * 4
            || Some(encoded.width) != rounded_width
        {
            return Err(invalid(
                "visible dimensions exceed encoded dimensions; unsupported geometry mismatch",
            ));
        }
        let lanes = (encoded.width / 64)
            .checked_mul(encoded.height)
            .ok_or_else(|| invalid("type-7 lane count overflow"))?;
        let max_count = lanes
            .checked_add(63)
            .map(|n| n / 64 * 64)
            .ok_or_else(|| invalid("type-7 metadata count overflow"))?;
        let counts = evidence
            .counts
            .ok_or_else(|| invalid("missing recovery count evidence"))?;
        if h.bits_offset < 16
            || h.bits_offset
                .checked_add(4)
                .is_none_or(|end| end > h.refs_offset)
            || h.refs_offset
                .checked_add(4)
                .is_none_or(|end| end > evidence.payload_len)
            || counts
                .into_iter()
                .any(|count| count < lanes || count > max_count)
        {
            return Err(invalid(
                "invalid recovery section ordering or metadata counts",
            ));
        }
        result.effective.height = encoded.height;
        result.reason = Some(GeometryRecoveryReason::IncompleteFinalFourRowGroup);
        Ok(result)
    }

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
            origin_zero_mapping: origin_zero_mapping(&value),
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

fn origin_zero_mapping(value: &Value) -> bool {
    value.as_object().is_some_and(|object| {
        object.keys().all(|key| {
            let key = key.to_ascii_lowercase();
            !key.contains("crop")
                && !key.contains("transform")
                && (!key.contains("origin")
                    || matches!(key.as_str(), "originalwidth" | "originalheight"))
                && !matches!(
                    key.as_str(),
                    "offsetx" | "offsety" | "sensorarrangement" | "sensorarrangment"
                )
        })
    })
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

/// Policy v1: a finite set of views of ONE decoded encoded raster. This is a
/// heuristic about image coordinates, not a claim about original sensor samples.
pub(crate) fn guess_layout(
    encoded_pixels: &[u16],
    geometry: mcraw4vulkan_core::ResolvedFrameGeometry,
    black: [u16; 4],
    white: u16,
) -> Result<LayoutAnalysis, DecodeError> {
    let invalid = || DecodeError::InvalidMetadata("invalid layout analysis coverage".into());
    let encoded = geometry.encoded.ok_or_else(invalid)?;
    let extraction = geometry.effective;
    if geometry.reason.is_none()
        || extraction.width == 0
        || extraction.height == 0
        || encoded.width == 0
        || encoded.height == 0
        || encoded.pixel_count() != Some(encoded_pixels.len())
        || extraction.width > encoded.width
        || extraction.height > encoded.height
    {
        return Err(invalid());
    }
    let c = extraction.width;
    let h = extraction.height;
    let mut views = vec![LayoutView { width: c, pitch: c }];
    // Last codec block only; at most 32 rows * 64 comparisons. A repeated edge
    // is evidence for a candidate, never permission to delete repeated samples.
    let mut edge = encoded.width.saturating_sub(64);
    let mut replicated_rows = 0;
    for row in 0..32u32.min(h) {
        let y = row * (h - 1) / 31.min(h - 1).max(1);
        let base = y as usize * encoded.width as usize;
        let last = encoded_pixels[base + encoded.width as usize - 1];
        let mut start = encoded.width - 1;
        while start > encoded.width.saturating_sub(64)
            && encoded_pixels[base + start as usize - 1] == last
        {
            start -= 1;
        }
        edge = edge.max(start + 1);
        if start + 1 < c {
            replicated_rows += 1;
        }
    }
    let lower_two = c.saturating_sub(1) / 2 * 2;
    let lower_four = c.saturating_sub(1) / 4 * 4;
    for width in [
        Some(lower_two),
        Some(lower_four),
        (replicated_rows >= 24.min(h) && edge < c).then_some(edge),
    ]
    .into_iter()
    .flatten()
    {
        if width < 8 || h < 8 || width >= c {
            continue;
        }
        // Reframe then crop. A crop keeps pitch c; it cannot masquerade as a reframe.
        for pitch in [width, c] {
            let view = LayoutView { width, pitch };
            if !views.contains(&view) {
                views.push(view);
            }
        }
    }
    debug_assert!(views.len() <= 7);
    let mut scores = Vec::with_capacity(views.len());
    for view in views {
        let last = u64::from(h - 1)
            .checked_mul(u64::from(view.pitch))
            .and_then(|n| n.checked_add(u64::from(view.width - 1)))
            .ok_or_else(invalid)?;
        if last >= u64::from(c) * u64::from(h) {
            return Err(invalid());
        }
        scores.push(score_layout(
            encoded_pixels,
            encoded.width,
            extraction,
            view,
            edge,
            black,
            white,
        ));
    }
    let baseline = &scores[0];
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by_key(|i| (scores[*i].cost, *i));
    let best = order[0];
    let winner = &scores[best];
    let runner = &scores[*order.get(1).unwrap_or(&0)];
    let margin =
        |other: u32| other.saturating_sub(winner.cost).saturating_mul(10_000) / other.max(1);
    let improved_regions = (0..16)
        .filter(|r| {
            let b = baseline.regions[*r];
            let w = winner.regions[*r];
            b.1 >= 8 && w.1 >= 8 && w.0 * u64::from(b.1) * 100 < b.0 * u64::from(w.1) * 92
        })
        .collect::<Vec<_>>();
    let rows = improved_regions
        .iter()
        .fold(0u32, |bits, r| bits | (1 << (r / 4)));
    let cols = improved_regions
        .iter()
        .fold(0u32, |bits, r| bits | (1 << (r % 4)));
    let selected = best != 0
        && winner.informative >= 128
        && baseline.informative >= 128
        && improved_regions.len() >= 4
        && rows.count_ones() >= 2
        && cols.count_ones() >= 2
        && winner.informative * 4 >= baseline.informative * 3
        && winner.informative * 3 <= baseline.informative * 4
        && margin(baseline.cost) >= 500
        && margin(runner.cost) >= 400;
    let choice = if selected { winner } else { baseline };
    // Only a contiguous reframe can be executed without an extra per-frame copy.
    // Crops remain competing controls: if one wins, keep the safe baseline.
    let selected = selected && choice.view.pitch == choice.view.width;
    let choice = if selected { choice } else { baseline };
    let guess = mcraw4vulkan_core::GeometryLayoutGuess {
        policy_version: 1,
        extraction,
        scene_pitch: choice.view.pitch,
        selected,
        analysis_frame: 0,
        informative_sites: winner.informative,
        baseline_score: baseline.cost,
        selected_score: choice.cost,
        margin_basis_points: if selected {
            margin(runner.cost).min(margin(baseline.cost))
        } else {
            0
        },
    };
    Ok(LayoutAnalysis {
        guess,
        effective: FrameDimensions {
            width: choice.view.width,
            height: h,
        },
    })
}

pub(crate) struct LayoutAnalysis {
    pub guess: mcraw4vulkan_core::GeometryLayoutGuess,
    pub effective: FrameDimensions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LayoutView {
    width: u32,
    pitch: u32,
}

#[derive(Debug)]
pub(crate) struct LayoutScore {
    view: LayoutView,
    cost: u32,
    informative: u32,
    regions: [(u64, u32); 16],
}

fn score_layout(
    pixels: &[u16],
    encoded_width: u32,
    extraction: FrameDimensions,
    view: LayoutView,
    replicated_edge: u32,
    black: [u16; 4],
    white: u16,
) -> LayoutScore {
    let mut regions = [(0u64, 0u32); 16];
    let mut total = 0u64;
    let mut informative = 0u32;
    if view.width < 8 || extraction.height < 8 || black.iter().any(|b| *b >= white) {
        return LayoutScore {
            view,
            cost: u32::MAX / 2,
            informative,
            regions,
        };
    }
    let c = extraction.width;
    let h = extraction.height;
    let sample = |x: u32, y: u32| -> Option<i32> {
        let linear = u64::from(y) * u64::from(view.pitch) + u64::from(x);
        let sx = (linear % u64::from(c)) as u32;
        if sx >= replicated_edge && replicated_edge < c {
            return None;
        }
        let sy = (linear / u64::from(c)) as u32;
        let value =
            i32::from(pixels[(u64::from(sy) * u64::from(encoded_width) + u64::from(sx)) as usize]);
        let b = i32::from(black[((y & 1) * 2 + (x & 1)) as usize]);
        Some((value - b).max(0) * 4096 / (i32::from(white) - b))
    };
    // 128*96=12,288 sites, fixed integer positions/reduction order, all phases.
    // Offsets of two always remain within the same proposed Bayer phase.
    for gy in 0..96u32 {
        let y = 2 + gy * (h - 5) / 95;
        for gx in 0..128u32 {
            let x = 2 + gx * (view.width - 5) / 127;
            let Some([v, l, r, u, d]) = sample(x, y)
                .zip(sample(x - 2, y))
                .zip(sample(x + 2, y))
                .zip(sample(x, y - 2))
                .zip(sample(x, y + 2))
                .map(|((((v, l), r), u), d)| [v, l, r, u, d])
            else {
                continue;
            };
            let low = v.min(l).min(r).min(u).min(d);
            let high = v.max(l).max(r).max(u).max(d);
            // A small fixed normalized contrast floor rejects nearly constant neighborhoods;
            // flat/dark/saturated neighborhoods cannot earn a smoothness reward.
            if high < 48 || low > 3900 || high - low < 24 {
                continue;
            }
            let horizontal = (2 * v - l - r).unsigned_abs().min(512);
            let vertical = (2 * v - u - d).unsigned_abs().min(512);
            let row_discontinuity = if (r - l).abs() < 32 && (d - u).abs() > 96 {
                64
            } else {
                0
            };
            let cost = horizontal + vertical + row_discontinuity;
            let region = ((gy / 24) * 4 + gx / 32) as usize;
            regions[region].0 += u64::from(cost);
            regions[region].1 += 1;
            total += u64::from(cost);
            informative += 1;
        }
    }
    let change_penalty = if view.width == c {
        0
    } else {
        2 + (c - view.width) * 1000 / c
    };
    let cost = if informative == 0 {
        u32::MAX / 2
    } else {
        (total / u64::from(informative)) as u32 + change_penalty
    };
    LayoutScore {
        view,
        cost,
        informative,
        regions,
    }
}
