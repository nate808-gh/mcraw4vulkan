use mcraw4vulkan_core::BayerPattern;
use serde_json::Value;

use crate::error::McrawContainerError as DecodeError;

// Parsed source metadata that applies to the whole clip rather than one frame.
// Output layers map these values into their own format-specific representations.
#[derive(Debug, Clone)]
pub struct ContainerMetadata {
    pub black_level: Option<BlackLevel>,
    pub white_level: Option<WhiteLevel>,
    pub sensor_arrangement: SensorArrangement,
    pub sensor_orientation: Option<u32>,
    pub unique_camera_model: Option<String>,
    pub device_specific_profile: Option<DeviceSpecificProfile>,
    pub focal_lengths: Option<Vec<f64>>,
    pub apertures: Option<Vec<f64>>,
    pub color_matrix1: Option<ColorMatrix>,
    pub color_matrix2: Option<ColorMatrix>,
    pub forward_matrix1: Option<ColorMatrix>,
    pub forward_matrix2: Option<ColorMatrix>,
    pub color_illuminant1: Option<ColorIlluminant>,
    pub color_illuminant2: Option<ColorIlluminant>,
    pub calibration_matrix1: Option<ColorMatrix>,
    pub calibration_matrix2: Option<ColorMatrix>,
    pub analog_balance: Option<[f64; 3]>,
}

// Black-level metadata normalized to the four positions of a 2x2 Bayer pattern.
//
// Scalar inputs expand to four CFA positions so callers always receive one value
// per position in the 2x2 Bayer repeat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlackLevel {
    pub values: [f64; 4],
}

// White-level metadata normalized to the four positions of a 2x2 Bayer pattern.
//
// This intentionally mirrors BlackLevel because .mcraw metadata is expected to
// store blackLevel and whiteLevel using the same scalar-or-array shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhiteLevel {
    pub values: [f64; 4],
}

// A 3x3 color or forward matrix in row-major order.
//
// Values remain f64 source metadata; output serializers own any rational or
// fixed-width encoding required by their formats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorMatrix {
    pub values: [f64; 9],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColorIlluminant {
    StandardA,
    D65,
    D50,
    Other(String),
}

impl ColorIlluminant {
    pub fn from_source_token(token: &crate::RawIlluminantToken) -> Self {
        match token.dng_code() {
            Some(17) => Self::StandardA,
            Some(21) => Self::D65,
            Some(23) => Self::D50,
            _ => Self::Other(match token {
                crate::RawIlluminantToken::String(s) => s.clone(),
                crate::RawIlluminantToken::Integer(n) => format!("code:{n}"),
            }),
        }
    }

    pub const fn dng_code(&self) -> Option<u16> {
        match self {
            Self::StandardA => Some(17),
            Self::D65 => Some(21),
            Self::D50 => Some(23),
            Self::Other(_) => None,
        }
    }

    pub fn label(&self) -> &str {
        match self {
            Self::StandardA => "standard-a",
            Self::D65 => "d65",
            Self::D50 => "d50",
            Self::Other(value) => value.as_str(),
        }
    }
}

// Device-specific camera identity fields retained without interpreting their
// identifiers as output-format policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceSpecificProfile {
    pub camera_id: Option<String>,
    pub device_model: Option<String>,
    pub disable_shading_map: Option<bool>,
}

// Typed CFA/sensor arrangement from MotionCam metadata.
//
// The MotionCam metadata commonly uses the misspelled key "sensorArrangment".
// The parser also accepts the corrected "sensorArrangement" spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SensorArrangement {
    Rggb,
    Bggr,
    Grbg,
    Gbrg,
    Unknown(String),
    Missing,
}

impl SensorArrangement {
    /// Return the typed Bayer mosaic represented by a supported MotionCam CFA.
    ///
    /// Unknown and missing source declarations deliberately remain unresolved;
    /// output policy belongs to the caller and must not be invented here.
    pub fn bayer_pattern(&self) -> Option<BayerPattern> {
        match self {
            Self::Rggb => Some(BayerPattern::Rggb),
            Self::Bggr => Some(BayerPattern::Bggr),
            Self::Grbg => Some(BayerPattern::Grbg),
            Self::Gbrg => Some(BayerPattern::Gbrg),
            Self::Unknown(_) | Self::Missing => None,
        }
    }
}

impl ContainerMetadata {
    pub fn with_frame_color(&self, frame: &crate::FrameMetadata) -> Self {
        let mut out = self.clone();
        for (i, over) in frame.color_overrides.slots.iter().enumerate() {
            let (cm, fm, cc, light) = if i == 0 {
                (
                    &mut out.color_matrix1,
                    &mut out.forward_matrix1,
                    &mut out.calibration_matrix1,
                    &mut out.color_illuminant1,
                )
            } else {
                (
                    &mut out.color_matrix2,
                    &mut out.forward_matrix2,
                    &mut out.calibration_matrix2,
                    &mut out.color_illuminant2,
                )
            };
            if let Some(v) = over.color_matrix {
                *cm = Some(ColorMatrix { values: v.values });
            }
            if let Some(v) = over.forward_matrix {
                *fm = v.map(|v| ColorMatrix { values: v.values });
            }
            if let Some(v) = over.camera_calibration {
                *cc = v.map(|v| ColorMatrix { values: v.values });
            }
            if let Some(v) = &over.illuminant {
                *light = Some(ColorIlluminant::from_source_token(v));
            }
        }
        if let Some(v) = frame.color_overrides.analog_balance {
            out.analog_balance = Some(v);
        }
        out
    }

    // Parse the container-level JSON stored near the beginning of a .mcraw file.
    //
    // Missing optional fields remain absent rather than receiving policy defaults;
    // each output layer decides which source fields it requires.
    pub fn parse(container_metadata_json: &str) -> Result<Self, DecodeError> {
        let value: Value = serde_json::from_str(container_metadata_json).map_err(|err| {
            DecodeError::UnsupportedFormat(format!("invalid container metadata json: {err}"))
        })?;

        Ok(Self {
            black_level: parse_level_values(&value, "blackLevel")?
                .map(|values| BlackLevel { values }),
            white_level: parse_level_values(&value, "whiteLevel")?
                .map(|values| WhiteLevel { values }),
            sensor_arrangement: parse_sensor_arrangement(&value),
            sensor_orientation: parse_optional_u32(&value, "sensorOrientation")?,
            unique_camera_model: parse_optional_string(&value, "uniqueCameraModel")?,
            device_specific_profile: parse_device_specific_profile(&value)?,
            focal_lengths: parse_optional_f64_array(&value, "focalLengths")?,
            apertures: parse_optional_f64_array(&value, "apertures")?,
            color_matrix1: parse_optional_matrix(&value, "colorMatrix1")?,
            color_matrix2: parse_optional_matrix(&value, "colorMatrix2")?,
            forward_matrix1: parse_optional_matrix(&value, "forwardMatrix1")?,
            forward_matrix2: parse_optional_matrix(&value, "forwardMatrix2")?,
            color_illuminant1: parse_optional_illuminant(&value, "colorIlluminant1")?,
            color_illuminant2: parse_optional_illuminant(&value, "colorIlluminant2")?,
            calibration_matrix1: parse_optional_matrix(&value, "calibrationMatrix1")?,
            calibration_matrix2: parse_optional_matrix(&value, "calibrationMatrix2")?,
            analog_balance: crate::ColorMetadataOverrides::parse(&value)
                .map_err(|e| DecodeError::UnsupportedFormat(e.to_string()))?
                .analog_balance,
        })
    }
}

// Parse blackLevel or whiteLevel using the same metadata shape.
//
// Accepted forms:
// - scalar number: 4095
// - scalar float: 4095.0
// - one-value array: [4095]
// - four-value array: [4095, 4095, 4095, 4095]
//
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

// Parse the CFA/sensor arrangement string while preserving unknown variants.
//
// Unknown tokens remain explicit so callers can decide whether their operation
// can proceed without a recognized arrangement.
fn parse_sensor_arrangement(value: &Value) -> SensorArrangement {
    let raw = value
        .get("sensorArrangment")
        .or_else(|| value.get("sensorArrangement"))
        .and_then(Value::as_str);

    let Some(raw) = raw else {
        return SensorArrangement::Missing;
    };

    match raw.to_ascii_lowercase().as_str() {
        "rggb" => SensorArrangement::Rggb,
        "bggr" => SensorArrangement::Bggr,
        "grbg" => SensorArrangement::Grbg,
        "gbrg" => SensorArrangement::Gbrg,
        other => SensorArrangement::Unknown(other.to_string()),
    }
}

fn parse_device_specific_profile(
    value: &Value,
) -> Result<Option<DeviceSpecificProfile>, DecodeError> {
    let Some(raw) = value.get("deviceSpecificProfile") else {
        return Ok(None);
    };

    let Some(object) = raw.as_object() else {
        return Err(DecodeError::UnsupportedFormat(
            "deviceSpecificProfile must be an object".to_string(),
        ));
    };

    Ok(Some(DeviceSpecificProfile {
        camera_id: parse_optional_string_from_object(object, "cameraId")?,
        device_model: parse_optional_string_from_object(object, "deviceModel")?,
        disable_shading_map: object.get("disableShadingMap").and_then(Value::as_bool),
    }))
}

// Parse an optional 3x3 matrix.
//
// The parser accepts either a flat 9-number array or a nested 3x3 array. This is
// intentionally tolerant because metadata JSON layouts may vary slightly across
// MotionCam versions.
fn parse_optional_matrix(value: &Value, key: &str) -> Result<Option<ColorMatrix>, DecodeError> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };

    let optional_calibration = matches!(
        key,
        "forwardMatrix1" | "forwardMatrix2" | "calibrationMatrix1" | "calibrationMatrix2"
    );
    crate::strict_color::parse_source_matrix(raw, optional_calibration)
        .map(|values| values.map(|values| ColorMatrix { values }))
        .map_err(|detail| DecodeError::UnsupportedFormat(format!("{key}: {detail}")))
}

fn parse_optional_illuminant(
    value: &Value,
    key: &str,
) -> Result<Option<ColorIlluminant>, DecodeError> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };
    let token = if let Some(s) = raw.as_str() {
        crate::RawIlluminantToken::String(s.to_owned())
    } else if let Some(n) = raw.as_i64() {
        crate::RawIlluminantToken::Integer(n)
    } else {
        return Err(DecodeError::UnsupportedFormat(format!(
            "{key} must be a string or integer"
        )));
    };
    Ok(Some(ColorIlluminant::from_source_token(&token)))
}

// Parse one JSON numeric value into f64.
//
// This accepts JSON integers and floats. It rejects missing, non-numeric,
// infinite, and NaN values so invalid metadata fails clearly.
fn parse_required_f64_value(value: &Value, key: &str) -> Result<f64, DecodeError> {
    parse_f64_value(value, key)?
        .ok_or_else(|| DecodeError::UnsupportedFormat(format!("{key} must be a numeric value")))
}

// Centralize the finite-value check so level, matrix, and scalar parsers share
// one NaN/infinity rejection boundary.
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

fn parse_optional_string_from_object(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<String>, DecodeError> {
    let Some(raw) = object.get(key) else {
        return Ok(None);
    };

    let Some(string) = raw.as_str() else {
        return Err(DecodeError::UnsupportedFormat(format!(
            "deviceSpecificProfile.{key} must be a string"
        )));
    };

    Ok(Some(string.to_string()))
}

// Parse an optional non-negative integer field that must fit DNG-facing u32 logs.
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

// Parse optional numeric arrays such as focalLengths and apertures for diagnostics.
fn parse_optional_f64_array(value: &Value, key: &str) -> Result<Option<Vec<f64>>, DecodeError> {
    let Some(raw) = value.get(key) else {
        return Ok(None);
    };

    let Some(items) = raw.as_array() else {
        return Err(DecodeError::UnsupportedFormat(format!(
            "{key} must be an array"
        )));
    };

    let mut values = Vec::with_capacity(items.len());
    for item in items {
        values.push(parse_required_f64_value(item, key)?);
    }

    Ok(Some(values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_sensor_arrangements_have_exact_bayer_patterns() {
        assert_eq!(
            SensorArrangement::Rggb.bayer_pattern(),
            Some(BayerPattern::Rggb)
        );
        assert_eq!(
            SensorArrangement::Bggr.bayer_pattern(),
            Some(BayerPattern::Bggr)
        );
        assert_eq!(
            SensorArrangement::Grbg.bayer_pattern(),
            Some(BayerPattern::Grbg)
        );
        assert_eq!(
            SensorArrangement::Gbrg.bayer_pattern(),
            Some(BayerPattern::Gbrg)
        );
        assert_eq!(SensorArrangement::Unknown("x".into()).bayer_pattern(), None);
        assert_eq!(SensorArrangement::Missing.bayer_pattern(), None);
    }

    #[test]
    fn parses_color_illuminants_by_name_and_dng_code() {
        let metadata = ContainerMetadata::parse(
            r#"{
                "colorIlluminant1": "Standard A",
                "colorIlluminant2": 21
            }"#,
        )
        .expect("metadata parses");

        assert_eq!(metadata.color_illuminant1, Some(ColorIlluminant::StandardA));
        assert_eq!(metadata.color_illuminant2, Some(ColorIlluminant::D65));
    }

    #[test]
    fn parses_calibration_matrices_without_changing_color_matrix_fields() {
        let metadata = ContainerMetadata::parse(
            r#"{
                "calibrationMatrix1": [1,0,0,0,1,0,0,0,1],
                "calibrationMatrix2": [[2,0,0],[0,2,0],[0,0,2]]
            }"#,
        )
        .expect("metadata parses");

        assert_eq!(
            metadata.calibration_matrix1,
            Some(ColorMatrix {
                values: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            })
        );
        assert_eq!(
            metadata.calibration_matrix2,
            Some(ColorMatrix {
                values: [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]
            })
        );
        assert_eq!(metadata.color_matrix1, None);
        assert_eq!(metadata.color_matrix2, None);
    }
}
