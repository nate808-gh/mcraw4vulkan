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

impl RawIlluminantToken {
    /// DNG/Exif light-source codes. Source tokens themselves remain unchanged.
    pub fn dng_code(&self) -> Option<u16> {
        match self {
            Self::Integer(17) => Some(17),
            Self::Integer(21) => Some(21),
            Self::Integer(23) => Some(23),
            Self::String(s) => match s
                .trim()
                .to_ascii_lowercase()
                .replace(['_', ' '], "-")
                .as_str()
            {
                "standarda" | "standard-a" | "standard-light-a" | "std-a" | "a" => Some(17),
                "d65" => Some(21),
                "d50" => Some(23),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Frame overrides retain the difference between an unspecified optional matrix
/// and an explicitly unavailable one. No mathematical defaults enter source facts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ColorSlotOverrides {
    pub illuminant: Option<RawIlluminantToken>,
    pub color_matrix: Option<RawCamera2Matrix>,
    pub camera_calibration: Option<Option<RawCamera2Matrix>>,
    pub forward_matrix: Option<Option<RawCamera2Matrix>>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ColorMetadataOverrides {
    pub slots: [ColorSlotOverrides; 2],
    pub analog_balance: Option<[f64; 3]>,
}

impl ColorMetadataOverrides {
    pub(crate) fn parse(value: &Value) -> Result<Self, RawCamera2ColorSourceError> {
        let mut result = Self::default();
        for (i, slot) in result.slots.iter_mut().enumerate() {
            let n = (i + 1) as u8;
            let key = format!("colorIlluminant{n}");
            slot.illuminant = value
                .get(&key)
                .map(|v| parse_illuminant(v, n))
                .transpose()?;
            slot.color_matrix = parse_optional_matrix(
                value,
                &format!("colorMatrix{n}"),
                n,
                RawCamera2MatrixKind::ColorMatrix,
            )?;
            let key = format!("calibrationMatrix{n}");
            if value.get(&key).is_some() {
                slot.camera_calibration = Some(parse_optional_matrix(
                    value,
                    &key,
                    n,
                    RawCamera2MatrixKind::CameraCalibration,
                )?);
            }
            let key = format!("forwardMatrix{n}");
            if value.get(&key).is_some() {
                slot.forward_matrix = Some(parse_optional_matrix(
                    value,
                    &key,
                    n,
                    RawCamera2MatrixKind::ForwardMatrix,
                )?);
            }
        }
        result.analog_balance = parse_optional_positive_triplet(
            value,
            "analogBalance",
            RawCamera2ColorSourceError::InvalidAnalogBalance,
        )?;
        Ok(result)
    }

    pub fn apply_to_profile(&self, source: &RawCamera2ColorProfile) -> RawCamera2ColorProfile {
        let mut profile = source.clone();
        for (i, over) in self.slots.iter().enumerate() {
            if *over == ColorSlotOverrides::default() {
                continue;
            }
            let source_slot = if i == 0 {
                RawColorCalibrationSlotIndex::Slot1
            } else {
                RawColorCalibrationSlotIndex::Slot2
            };
            if !profile.slots.iter().any(|s| s.source_slot == source_slot) {
                profile.slots.push(RawColorCalibrationSlot {
                    source_slot,
                    illuminant: None,
                    color_matrix: None,
                    camera_calibration: None,
                    forward_matrix: None,
                    provenance: source.provenance,
                });
            }
            let slot = profile
                .slots
                .iter_mut()
                .find(|s| s.source_slot == source_slot)
                .expect("slot inserted");
            if let Some(v) = &over.illuminant {
                slot.illuminant = Some(v.clone());
            }
            if let Some(v) = over.color_matrix {
                slot.color_matrix = Some(v);
            }
            if let Some(v) = over.camera_calibration {
                slot.camera_calibration = v;
            }
            if let Some(v) = over.forward_matrix {
                slot.forward_matrix = v;
            }
        }
        profile.slots.sort_by_key(|s| s.source_slot.number());
        if let Some(v) = self.analog_balance {
            profile.analog_balance = Some(v);
        }
        profile
    }
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
    parse_source_matrix(raw, kind != RawCamera2MatrixKind::ColorMatrix)
        .map(|values| values.map(|values| RawCamera2Matrix { values }))
        .map_err(|detail| RawCamera2ColorSourceError::InvalidMatrix { slot, kind, detail })
}

pub(crate) fn parse_source_matrix(
    raw: &Value,
    optional_calibration: bool,
) -> Result<Option<[f64; 9]>, String> {
    if optional_calibration && raw.as_array().is_some_and(Vec::is_empty) {
        return Ok(None);
    }
    parse_matrix(raw).map(Some)
}

pub(crate) fn parse_matrix(raw: &Value) -> Result<[f64; 9], String> {
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

#[cfg(test)]
mod shared_contract_tests {
    use super::*;
    use crate::{ContainerMetadata, FrameMetadata};
    use serde_json::json;
    const P: StrictColorProfileProvenance =
        StrictColorProfileProvenance::from_source_sha256([1; 32]);

    #[test]
    fn only_empty_optional_calibration_is_absent_in_both_parsers() {
        for key in [
            "forwardMatrix1",
            "forwardMatrix2",
            "calibrationMatrix1",
            "calibrationMatrix2",
        ] {
            let j = json!({key: []}).to_string();
            assert!(ContainerMetadata::parse(&j).is_ok(), "{key}");
            let p = RawCamera2ColorProfile::parse(&j, P).unwrap();
            assert!(
                p.slots
                    .iter()
                    .all(|s| s.forward_matrix.is_none() && s.camera_calibration.is_none())
            );
            for value in [
                json!(null),
                json!(""),
                json!([[], [], []]),
                json!([1, 2]),
                json!([[1, 2, 3], [], [1, 2, 3]]),
            ] {
                let j = json!({key: value}).to_string();
                assert!(ContainerMetadata::parse(&j).is_err());
                assert!(RawCamera2ColorProfile::parse(&j, P).is_err());
            }
        }
        for key in ["colorMatrix1", "colorMatrix2"] {
            let j = json!({key: []}).to_string();
            assert!(ContainerMetadata::parse(&j).is_err());
            assert!(RawCamera2ColorProfile::parse(&j, P).is_err());
        }
    }

    #[test]
    fn source_slots_and_explicit_absence_survive_frame_precedence() {
        let j = json!({"colorIlluminant1":17,"colorMatrix1":[1,0,0,0,1,0,0,0,1],"forwardMatrix1":[1,0,0,0,1,0,0,0,1]});
        let clip = ContainerMetadata::parse(&j.to_string()).unwrap();
        let source = RawCamera2ColorProfile::parse(&j.to_string(), P).unwrap();
        let frame = FrameMetadata::parse(&json!({"width":64,"height":4,"forwardMatrix1":[],"colorMatrix1":[2,0,0,0,1,0,0,0,1],"colorIlluminant1":"d50","asShotNeutral":[0.5,1,0.7]}).to_string()).unwrap();
        let raw = frame.color_overrides.apply_to_profile(&source);
        let typed = clip.with_frame_color(&frame);
        assert!(raw.slots[0].forward_matrix.is_none());
        assert!(typed.forward_matrix1.is_none());
        assert_eq!(
            raw.slots[0].color_matrix.unwrap().values,
            typed.color_matrix1.unwrap().values
        );
        assert_eq!(
            raw.slots[0].illuminant.as_ref().unwrap().dng_code(),
            Some(23)
        );
        assert_eq!(typed.color_illuminant1.unwrap().dng_code(), Some(23));
        assert_eq!(raw.provenance, P);
        assert!(source.slots[0].forward_matrix.is_some());
    }

    #[test]
    fn normative_numeric_and_string_illuminants_agree() {
        for (code, name) in [(17, "standarda"), (21, "d65"), (23, "d50")] {
            assert_eq!(
                RawIlluminantToken::Integer(code).dng_code(),
                Some(code as u16)
            );
            assert_eq!(
                RawIlluminantToken::String(name.into()).dng_code(),
                Some(code as u16)
            );
            let c = ContainerMetadata::parse(
                &json!({"colorIlluminant1":code,"colorIlluminant2":name}).to_string(),
            )
            .unwrap();
            assert_eq!(c.color_illuminant1, c.color_illuminant2);
        }
        assert_eq!(
            RawIlluminantToken::String("unknown".into()).dng_code(),
            None
        );
    }
}
