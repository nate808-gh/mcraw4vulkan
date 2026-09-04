//! Strict, source-preserving Camera2 color facts for the rendered PIPE.
//!
//! This parser is deliberately independent of the existing DNG-facing metadata
//! adapter. In particular, it preserves raw illuminant token type/value and
//! does not use [`crate::ColorIlluminant`].

use serde_json::Value;
use thiserror::Error;

/// Identity tying color facts to the complete source container from which they
/// were extracted. The caller supplies the already-computed source SHA-256.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrictColorProfileProvenance {
    source_sha256: [u8; 32],
}

impl StrictColorProfileProvenance {
    pub const fn from_source_sha256(source_sha256: [u8; 32]) -> Self {
        Self { source_sha256 }
    }

    pub const fn source_sha256(self) -> [u8; 32] {
        self.source_sha256
    }
}

/// Exact decoded JSON scalar used by a Camera2 calibration slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawIlluminantToken {
    String(String),
    Integer(i64),
}

/// A raw row-major Camera2 3x3 matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawCamera2Matrix {
    pub values: [f64; 9],
}

/// Original Camera2 calibration-slot number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RawColorCalibrationSlotIndex {
    Slot1 = 1,
    Slot2 = 2,
}

impl RawColorCalibrationSlotIndex {
    pub const fn number(self) -> u8 {
        self as u8
    }
}

/// Source facts for one slot. Optional fields preserve incomplete presence so
/// the strict resolver can return the contract's precise failure class.
#[derive(Debug, Clone, PartialEq)]
pub struct RawColorCalibrationSlot {
    pub source_slot: RawColorCalibrationSlotIndex,
    pub illuminant: Option<RawIlluminantToken>,
    pub color_matrix: Option<RawCamera2Matrix>,
    pub camera_calibration: Option<RawCamera2Matrix>,
    pub forward_matrix: Option<RawCamera2Matrix>,
    pub provenance: StrictColorProfileProvenance,
}

/// Clip-level raw Camera2 profile facts in original source-slot order.
#[derive(Debug, Clone, PartialEq)]
pub struct RawCamera2ColorProfile {
    pub slots: Vec<RawColorCalibrationSlot>,
    pub analog_balance: Option<[f64; 3]>,
    pub provenance: StrictColorProfileProvenance,
}

/// Per-frame color facts. AsShotNeutral absence remains distinct from invalid
/// presence and is interpreted only by the strict PIPE resolver.
#[derive(Debug, Clone, PartialEq)]
pub struct RawCamera2FrameColor {
    pub source_frame_index: u64,
    pub as_shot_neutral: Option<[f64; 3]>,
    pub provenance: StrictColorProfileProvenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawCamera2MatrixKind {
    ColorMatrix,
    CameraCalibration,
    ForwardMatrix,
}

impl std::fmt::Display for RawCamera2MatrixKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ColorMatrix => "ColorMatrix",
            Self::CameraCalibration => "CameraCalibration",
            Self::ForwardMatrix => "ForwardMatrix",
        })
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RawCamera2ColorSourceError {
    #[error("invalid {scope} metadata JSON: {detail}")]
    InvalidJson { scope: &'static str, detail: String },
    #[error("color illuminant {slot} must be a JSON string or signed i64 integer")]
    InvalidIlluminantToken { slot: u8 },
    #[error("{kind}{slot} must be a finite 3x3 numeric matrix: {detail}")]
    InvalidMatrix {
        slot: u8,
        kind: RawCamera2MatrixKind,
        detail: String,
    },
    #[error("analogBalance must contain exactly three finite positive values")]
    InvalidAnalogBalance,
    #[error("asShotNeutral must contain exactly three finite positive values")]
    InvalidAsShotNeutral,
}

impl RawCamera2ColorProfile {
    /// Parse only the strict color facts. This does not alter or call the
    /// existing permissive/DNG-facing illuminant interpretation.
    pub fn parse(
        container_metadata_json: &str,
        provenance: StrictColorProfileProvenance,
    ) -> Result<Self, RawCamera2ColorSourceError> {
        let value: Value = serde_json::from_str(container_metadata_json).map_err(|error| {
            RawCamera2ColorSourceError::InvalidJson {
                scope: "container",
                detail: error.to_string(),
            }
        })?;

        let mut slots = Vec::with_capacity(2);
        for source_slot in [
            RawColorCalibrationSlotIndex::Slot1,
            RawColorCalibrationSlotIndex::Slot2,
        ] {
            if let Some(slot) = parse_slot(&value, source_slot, provenance)? {
                slots.push(slot);
            }
        }

        Ok(Self {
            slots,
            analog_balance: parse_optional_positive_triplet(
                &value,
                "analogBalance",
                RawCamera2ColorSourceError::InvalidAnalogBalance,
            )?,
            provenance,
        })
    }
}

impl RawCamera2FrameColor {
    pub fn parse(
        frame_metadata_json: &str,
        source_frame_index: u64,
        provenance: StrictColorProfileProvenance,
    ) -> Result<Self, RawCamera2ColorSourceError> {
        let value: Value = serde_json::from_str(frame_metadata_json).map_err(|error| {
            RawCamera2ColorSourceError::InvalidJson {
                scope: "frame",
                detail: error.to_string(),
            }
        })?;

        Ok(Self {
            source_frame_index,
            as_shot_neutral: parse_optional_positive_triplet(
                &value,
                "asShotNeutral",
                RawCamera2ColorSourceError::InvalidAsShotNeutral,
            )?,
            provenance,
        })
    }
}

fn parse_slot(
    value: &Value,
    source_slot: RawColorCalibrationSlotIndex,
    provenance: StrictColorProfileProvenance,
) -> Result<Option<RawColorCalibrationSlot>, RawCamera2ColorSourceError> {
    let number = source_slot.number();
    let illuminant_key = format!("colorIlluminant{number}");
    let color_matrix_key = format!("colorMatrix{number}");
    let calibration_matrix_key = format!("calibrationMatrix{number}");
    let forward_matrix_key = format!("forwardMatrix{number}");

    let present = [
        illuminant_key.as_str(),
        color_matrix_key.as_str(),
        calibration_matrix_key.as_str(),
        forward_matrix_key.as_str(),
    ]
    .iter()
    .any(|key| value.get(key).is_some());
    if !present {
        return Ok(None);
    }

    let illuminant = value
        .get(&illuminant_key)
        .map(|raw| parse_illuminant(raw, number))
        .transpose()?;
    let color_matrix = parse_optional_matrix(
        value,
        &color_matrix_key,
        number,
        RawCamera2MatrixKind::ColorMatrix,
    )?;
    let camera_calibration = parse_optional_matrix(
        value,
        &calibration_matrix_key,
        number,
        RawCamera2MatrixKind::CameraCalibration,
    )?;
    let forward_matrix = parse_optional_matrix(
        value,
        &forward_matrix_key,
        number,
        RawCamera2MatrixKind::ForwardMatrix,
    )?;

    Ok(Some(RawColorCalibrationSlot {
        source_slot,
        illuminant,
        color_matrix,
        camera_calibration,
        forward_matrix,
        provenance,
    }))
}

fn parse_illuminant(
    raw: &Value,
    slot: u8,
) -> Result<RawIlluminantToken, RawCamera2ColorSourceError> {
    if let Some(value) = raw.as_str() {
        return Ok(RawIlluminantToken::String(value.to_owned()));
    }
    if let Some(value) = raw.as_i64() {
        return Ok(RawIlluminantToken::Integer(value));
    }
    Err(RawCamera2ColorSourceError::InvalidIlluminantToken { slot })
}

fn parse_optional_matrix(
    object: &Value,
    key: &str,
    slot: u8,
    kind: RawCamera2MatrixKind,
) -> Result<Option<RawCamera2Matrix>, RawCamera2ColorSourceError> {
    let Some(raw) = object.get(key) else {
        return Ok(None);
    };
    parse_matrix(raw)
        .map(|values| Some(RawCamera2Matrix { values }))
        .map_err(|detail| RawCamera2ColorSourceError::InvalidMatrix { slot, kind, detail })
}

fn parse_matrix(raw: &Value) -> Result<[f64; 9], String> {
    let rows = raw
        .as_array()
        .ok_or_else(|| "value is not an array".to_owned())?;
    let mut values = [0.0; 9];
    if rows.len() == 9 {
        for (index, raw_value) in rows.iter().enumerate() {
            values[index] = finite_f64(raw_value)
                .ok_or_else(|| format!("element {index} is not finite numeric"))?;
        }
        return Ok(values);
    }
    if rows.len() == 3 {
        for (row_index, raw_row) in rows.iter().enumerate() {
            let row = raw_row
                .as_array()
                .filter(|row| row.len() == 3)
                .ok_or_else(|| format!("row {row_index} is not a three-value array"))?;
            for (column_index, raw_value) in row.iter().enumerate() {
                let index = row_index * 3 + column_index;
                values[index] = finite_f64(raw_value)
                    .ok_or_else(|| format!("element {index} is not finite numeric"))?;
            }
        }
        return Ok(values);
    }
    Err("value is neither a flat nine-value nor nested 3x3 array".to_owned())
}

fn parse_optional_positive_triplet(
    object: &Value,
    key: &str,
    error: RawCamera2ColorSourceError,
) -> Result<Option<[f64; 3]>, RawCamera2ColorSourceError> {
    let Some(raw) = object.get(key) else {
        return Ok(None);
    };
    let Some(items) = raw.as_array().filter(|items| items.len() == 3) else {
        return Err(error);
    };
    let mut values = [0.0; 3];
    for (index, item) in items.iter().enumerate() {
        let Some(value) = finite_f64(item).filter(|value| *value > 0.0) else {
            return Err(error);
        };
        values[index] = value;
    }
    Ok(Some(values))
}

fn finite_f64(raw: &Value) -> Option<f64> {
    raw.as_f64().filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROVENANCE: StrictColorProfileProvenance =
        StrictColorProfileProvenance::from_source_sha256([7; 32]);

    #[test]
    fn preserves_exact_illuminant_type_value_and_source_slot_order() {
        let profile = RawCamera2ColorProfile::parse(
            r#"{
                "colorIlluminant1":17,
                "colorMatrix1":[1,0,0,0,1,0,0,0,1],
                "forwardMatrix1":[1,0,0,0,1,0,0,0,1],
                "colorIlluminant2":"d65",
                "colorMatrix2":[[1,0,0],[0,1,0],[0,0,1]],
                "calibrationMatrix2":[1,0,0,0,1,0,0,0,1],
                "forwardMatrix2":[1,0,0,0,1,0,0,0,1]
            }"#,
            PROVENANCE,
        )
        .expect("strict profile");
        assert_eq!(profile.slots.len(), 2);
        assert_eq!(profile.slots[0].source_slot.number(), 1);
        assert_eq!(
            profile.slots[0].illuminant,
            Some(RawIlluminantToken::Integer(17))
        );
        assert_eq!(
            profile.slots[1].illuminant,
            Some(RawIlluminantToken::String("d65".to_owned()))
        );
        assert!(profile.slots[0].camera_calibration.is_none());
        assert!(profile.slots[1].camera_calibration.is_some());
    }

    #[test]
    fn does_not_trim_case_fold_or_alias_raw_string_tokens() {
        for token in ["standarda", "StandardA", " standardA ", "standard-a"] {
            let json = format!(r#"{{"colorIlluminant1":"{token}"}}"#);
            let profile = RawCamera2ColorProfile::parse(&json, PROVENANCE).unwrap();
            assert_eq!(
                profile.slots[0].illuminant,
                Some(RawIlluminantToken::String(token.to_owned()))
            );
        }
    }

    #[test]
    fn preserves_absence_and_rejects_invalid_presence() {
        let profile = RawCamera2ColorProfile::parse("{}", PROVENANCE).unwrap();
        assert!(profile.slots.is_empty());
        assert!(profile.analog_balance.is_none());

        assert!(matches!(
            RawCamera2ColorProfile::parse(r#"{"analogBalance":[1,0,1]}"#, PROVENANCE),
            Err(RawCamera2ColorSourceError::InvalidAnalogBalance)
        ));
        assert!(matches!(
            RawCamera2FrameColor::parse(r#"{"asShotNeutral":"missing"}"#, 4, PROVENANCE),
            Err(RawCamera2ColorSourceError::InvalidAsShotNeutral)
        ));
        assert!(matches!(
            RawCamera2ColorProfile::parse(r#"{"colorIlluminant1":17.0}"#, PROVENANCE),
            Err(RawCamera2ColorSourceError::InvalidIlluminantToken { slot: 1 })
        ));
    }

    #[test]
    fn frame_facts_retain_index_asn_and_provenance() {
        let frame =
            RawCamera2FrameColor::parse(r#"{"asShotNeutral":[0.5,1,0.75]}"#, 123, PROVENANCE)
                .unwrap();
        assert_eq!(frame.source_frame_index, 123);
        assert_eq!(frame.as_shot_neutral, Some([0.5, 1.0, 0.75]));
        assert_eq!(frame.provenance, PROVENANCE);
    }
}
