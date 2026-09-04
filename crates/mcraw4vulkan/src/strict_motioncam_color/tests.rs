use std::collections::BTreeMap;

use mcraw4vulkan_core::{BayerPattern, FrameDimensions};
use mcraw4vulkan_mcrawcontainer::{
    RawCamera2ColorProfile, RawCamera2FrameColor, RawCamera2Matrix, RawColorCalibrationSlot,
    RawColorCalibrationSlotIndex, RawIlluminantToken, StrictColorProfileProvenance,
};
use mcraw4vulkan_vignette::{PipeF32BayerCorrectionMode, PipeF32BayerNumericDomain};
use serde_json::Value;

use super::policy::{EXPECTED_POLICY_DIGEST, POLICY_RECORD_BYTE_LEN, policy_record};
use super::*;

const F64_VECTORS: &str = include_str!("testdata/f64-reference-vectors-v2.tsv");
const F32_VECTORS: &str = include_str!("testdata/f32-serialization-vectors-v2.tsv");
const POLICY_VECTOR: &str = include_str!("testdata/color-policy-vector-v2.tsv");

#[test]
fn mapped_slice_sha256_matches_fips_180_abc_vector() {
    assert_eq!(
        Sha256::digest(b"abc")
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[derive(Clone)]
struct TsvRow(BTreeMap<String, String>);

impl TsvRow {
    fn get(&self, name: &str) -> &str {
        self.0
            .get(name)
            .map(String::as_str)
            .unwrap_or_else(|| panic!("missing TSV column {name}"))
    }
}

fn parse_tsv(input: &str) -> Vec<TsvRow> {
    let mut lines = input.lines();
    let headers = lines
        .next()
        .expect("TSV header")
        .split('\t')
        .map(str::to_owned)
        .collect::<Vec<_>>();
    lines
        .filter(|line| !line.is_empty() && !line.starts_with("//"))
        .map(|line| {
            let fields = line.split('\t').map(unquote_tsv).collect::<Vec<_>>();
            assert_eq!(fields.len(), headers.len(), "malformed TSV row");
            TsvRow(headers.iter().cloned().zip(fields).collect())
        })
        .collect()
}

fn unquote_tsv(field: &str) -> String {
    if field.len() >= 2 && field.starts_with('"') && field.ends_with('"') {
        field[1..field.len() - 1].replace("\"\"", "\"")
    } else {
        field.to_owned()
    }
}

fn parse_f64_array<const N: usize>(value: &str) -> [f64; N] {
    let values = value
        .split(',')
        .map(|field| field.parse::<f64>().expect("f64 field"))
        .collect::<Vec<_>>();
    values
        .try_into()
        .unwrap_or_else(|values: Vec<f64>| panic!("expected {N} values, got {}", values.len()))
}

fn parse_matrix_groups(value: &str) -> Vec<Option<[f64; 9]>> {
    value
        .split('|')
        .map(|matrix| {
            if matrix == "NONE" {
                None
            } else {
                Some(parse_f64_array(matrix))
            }
        })
        .collect()
}

fn parse_f32_bits(value: &str) -> [u32; 9] {
    let bits = value
        .split(',')
        .map(|field| {
            u32::from_str_radix(field.strip_prefix("0x").expect("0x f32 bits"), 16)
                .expect("f32 bits")
        })
        .collect::<Vec<_>>();
    bits.try_into().expect("nine f32 words")
}

fn decode_hex(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0);
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).expect("hex byte"))
        .collect()
}

fn decode_digest(value: &str) -> [u8; 32] {
    decode_hex(value).try_into().expect("SHA-256 digest")
}

fn token_from_json(value: &Value) -> RawIlluminantToken {
    if let Some(string) = value.as_str() {
        RawIlluminantToken::String(string.to_owned())
    } else {
        RawIlluminantToken::Integer(value.as_i64().expect("integer token"))
    }
}

fn input_from_row(
    row: &TsvRow,
    provenance: StrictColorProfileProvenance,
    source_frame_index: u64,
) -> (RawCamera2ColorProfile, StrictMotionCamFrameColorInput) {
    let tokens: Vec<Value> =
        serde_json::from_str(row.get("raw_illuminant_tokens_json")).expect("illuminant token JSON");
    let color_matrices = parse_matrix_groups(row.get("input_color_matrices"));
    let calibration_matrices = parse_matrix_groups(row.get("input_calibration_matrices"));
    let forward_matrices = parse_matrix_groups(row.get("input_forward_matrices"));
    assert_eq!(tokens.len(), color_matrices.len());
    assert_eq!(tokens.len(), calibration_matrices.len());
    assert_eq!(tokens.len(), forward_matrices.len());
    let slots = (0..tokens.len())
        .map(|index| RawColorCalibrationSlot {
            source_slot: match index {
                0 => RawColorCalibrationSlotIndex::Slot1,
                1 => RawColorCalibrationSlotIndex::Slot2,
                _ => panic!("only one or two slots are supported"),
            },
            illuminant: Some(token_from_json(&tokens[index])),
            color_matrix: color_matrices[index].map(|values| RawCamera2Matrix { values }),
            camera_calibration: calibration_matrices[index]
                .map(|values| RawCamera2Matrix { values }),
            forward_matrix: forward_matrices[index].map(|values| RawCamera2Matrix { values }),
            provenance,
        })
        .collect();
    let analog_balance = match row.get("input_analog_balance") {
        "ABSENT" => None,
        values => Some(parse_f64_array(values)),
    };
    let profile = RawCamera2ColorProfile {
        slots,
        analog_balance,
        provenance,
    };
    let frame = StrictMotionCamFrameColorInput::from_raw(RawCamera2FrameColor {
        source_frame_index,
        as_shot_neutral: Some(parse_f64_array(row.get("input_as_shot_neutral"))),
        provenance,
    });
    (profile, frame)
}

fn vector_row(vector_id: &str) -> TsvRow {
    parse_tsv(F64_VECTORS)
        .into_iter()
        .find(|row| row.get("vector_id") == vector_id)
        .unwrap_or_else(|| panic!("missing vector {vector_id}"))
}

fn frozen_f32_map() -> BTreeMap<(String, String, usize, usize), u32> {
    parse_tsv(F32_VECTORS)
        .into_iter()
        .map(|row| {
            let bits =
                u32::from_str_radix(row.get("f32_bits").strip_prefix("0x").unwrap(), 16).unwrap();
            (
                (
                    row.get("vector_id").to_owned(),
                    row.get("matrix").to_owned(),
                    row.get("row").parse().unwrap(),
                    row.get("column").parse().unwrap(),
                ),
                bits,
            )
        })
        .collect()
}

fn portable_f32_serialization_bytes() -> Vec<u8> {
    let portable_vector_ids = parse_tsv(F64_VECTORS)
        .into_iter()
        .filter(|row| row.get("provenance") == "synthetic")
        .map(|row| row.get("vector_id").to_owned())
        .collect::<Vec<_>>();
    let mut bytes = Vec::new();
    for (index, line) in F32_VECTORS.lines().enumerate() {
        let vector_id = line.split('\t').next().expect("F32 TSV row");
        if index == 0 || portable_vector_ids.iter().any(|id| id == vector_id) {
            bytes.extend_from_slice(line.as_bytes());
            bytes.push(b'\n');
        }
    }
    bytes
}

fn assert_f32_matrix_rows(
    frozen: &BTreeMap<(String, String, usize, usize), u32>,
    vector_id: &str,
    matrix_name: &str,
    matrix: [f64; 9],
) -> usize {
    for (index, value) in matrix.into_iter().enumerate() {
        assert_eq!(
            (value as f32).to_bits(),
            frozen[&(
                vector_id.to_owned(),
                matrix_name.to_owned(),
                index / 3,
                index % 3,
            )],
            "{vector_id} {matrix_name} element {index}"
        );
    }
    9
}

fn assert_resolved_forward_matrix_vector(
    frozen_f32: &BTreeMap<(String, String, usize, usize), u32>,
    row: &TsvRow,
    resolved: &ResolvedStrictPipeColor,
) -> usize {
    let vector_id = row.get("vector_id");
    assert_eq!(
        resolved.convergence_iterations().to_string(),
        row.get("iterations"),
        "{vector_id} iteration count"
    );
    assert_matrix_bits_equal(resolved.t50(), row.get("t50"), vector_id, "T50");
    assert_matrix_bits_equal(
        resolved.camera_to_linear_bt2020(),
        row.get("camera_to_linear_bt2020"),
        vector_id,
        "linear BT.2020",
    );
    assert_matrix_bits_equal(
        resolved.camera_to_normalized_ncl(),
        row.get("camera_to_normalized_ncl"),
        vector_id,
        "normalized NCL",
    );
    assert_eq!(
        resolved.camera_to_normalized_ncl_f32_bits(),
        parse_f32_bits(row.get("ncl_f32_bits")),
        "{vector_id} exact final f32 bits"
    );
    assert_f32_matrix_rows(frozen_f32, vector_id, "t50", resolved.t50())
        + assert_f32_matrix_rows(
            frozen_f32,
            vector_id,
            "camera_to_linear_bt2020",
            resolved.camera_to_linear_bt2020(),
        )
        + assert_f32_matrix_rows(
            frozen_f32,
            vector_id,
            "camera_to_normalized_ncl",
            resolved.camera_to_normalized_ncl(),
        )
}

fn assert_independent_forward_matrix_vector(
    frozen_f32: &BTreeMap<(String, String, usize, usize), u32>,
    row: &TsvRow,
    result: &independent_reference::ResultMatrices,
) -> usize {
    let vector_id = row.get("vector_id");
    assert_f32_matrix_rows(frozen_f32, vector_id, "t50", result.t50)
        + assert_f32_matrix_rows(
            frozen_f32,
            vector_id,
            "camera_to_linear_bt2020",
            result.linear_bt2020,
        )
        + assert_f32_matrix_rows(
            frozen_f32,
            vector_id,
            "camera_to_normalized_ncl",
            result.normalized_ncl,
        )
}

#[test]
fn policy_record_is_exact_frozen_v2_stream() {
    let row = parse_tsv(POLICY_VECTOR).pop().expect("policy row");
    let expected_record = decode_hex(row.get("record_hex"));
    let record = policy_record();
    assert_eq!(record.len(), POLICY_RECORD_BYTE_LEN);
    assert_eq!(record, expected_record);
    assert_eq!(Sha256::digest(&record), EXPECTED_POLICY_DIGEST);
    assert_eq!(
        Sha256::digest(&portable_f32_serialization_bytes()),
        decode_digest("3d82ab053541682a31dade54a2dc9e49ce978935e638df3437d4ddf27bb2ec3a")
    );
    assert!(StrictMotionCamForwardMatrixColorV2::new().is_ok());
}

#[test]
fn production_reproduces_portable_forward_matrix_accepts_and_exact_ten_rejects() {
    let resolver = StrictMotionCamForwardMatrixColorV2::new().unwrap();
    let provenance = StrictColorProfileProvenance::from_source_sha256([7; 32]);
    let frozen_f32 = frozen_f32_map();
    let mut accepted = 0;
    let mut rejected = 0;
    let mut reference_only = 0;
    let mut serialized_rows = 0;
    for row in parse_tsv(F64_VECTORS)
        .into_iter()
        .filter(|row| row.get("provenance") == "synthetic")
    {
        let vector_id = row.get("vector_id");
        let (raw_profile, frame) = input_from_row(&row, provenance, 0);
        let profile = resolver.validate_profile(raw_profile);
        match row.get("accepted_resolver_class") {
            "FORWARD_MATRIX_T50" => {
                accepted += 1;
                let resolved = resolver
                    .resolve(&profile.unwrap(), &frame)
                    .unwrap_or_else(|error| panic!("{vector_id}: {error}"));
                serialized_rows +=
                    assert_resolved_forward_matrix_vector(&frozen_f32, &row, &resolved);
            }
            "COLORMATRIX_ADAPTED_T50" => {
                reference_only += 1;
                assert_eq!(
                    profile.unwrap_err().code(),
                    "UNSUPPORTED_COLOR_METADATA",
                    "{vector_id} is frozen reference-only, not in production subset"
                );
            }
            "REJECTED" => {
                rejected += 1;
                let error = match profile {
                    Ok(profile) => resolver.resolve(&profile, &frame).unwrap_err(),
                    Err(error) => error,
                };
                assert_eq!(error.code(), row.get("expected_failure"), "{vector_id}");
            }
            class => panic!("unknown resolver class {class}"),
        }
    }
    assert_eq!((accepted, reference_only, rejected), (12, 1, 10));
    assert_eq!(serialized_rows, 324);
}

fn assert_matrix_bits_equal(
    observed: [f64; 9],
    expected_text: &str,
    vector_id: &str,
    matrix: &str,
) {
    let expected = parse_f64_array::<9>(expected_text);
    assert_eq!(
        observed.map(f64::to_bits),
        expected.map(f64::to_bits),
        "{vector_id} exact f64 {matrix}"
    );
}

#[test]
fn absence_defaults_are_identity_but_remain_distinct_in_context_bytes() {
    let row = vector_row("identity_d65_single");
    let provenance = StrictColorProfileProvenance::from_source_sha256([9; 32]);
    let (mut absent, frame) = input_from_row(&row, provenance, 3);
    absent.analog_balance = None;
    absent.slots[0].camera_calibration = None;
    let mut explicit = absent.clone();
    explicit.analog_balance = Some([1.0; 3]);
    explicit.slots[0].camera_calibration = Some(RawCamera2Matrix { values: IDENTITY.0 });
    let resolver = StrictMotionCamForwardMatrixColorV2::new().unwrap();
    let absent = resolver.validate_profile(absent).unwrap();
    let explicit = resolver.validate_profile(explicit).unwrap();
    let absent_result = resolver.resolve(&absent, &frame).unwrap();
    let explicit_result = resolver.resolve(&explicit, &frame).unwrap();
    assert_eq!(absent_result.t50(), explicit_result.t50());
    let facts = ColorContextFingerprintFacts {
        numeric_domain: PipeF32BayerNumericDomain::RelativeLinearCorrectedCodeV1,
        dimensions: FrameDimensions {
            width: 2,
            height: 1,
        },
        bayer_pattern: BayerPattern::Rggb,
        correction_mode: PipeF32BayerCorrectionMode::IdentitySpatialGain,
        source_sha256: ClipSourceSha256::from_frozen_digest([9; 32]),
        source_frame_index: 3,
    };
    let absent_fingerprint = resolver
        .color_context_fingerprint(&absent, &frame, &absent_result, facts)
        .unwrap();
    let explicit_fingerprint = resolver
        .color_context_fingerprint(&explicit, &frame, &explicit_result, facts)
        .unwrap();
    assert_ne!(absent_fingerprint, explicit_fingerprint);
}

#[test]
fn camera_calibration_absence_is_per_slot_and_interpolated_numeric_gates_are_exact() {
    let resolver = StrictMotionCamForwardMatrixColorV2::new().unwrap();
    let provenance = StrictColorProfileProvenance::from_source_sha256([11; 32]);
    let dual = vector_row("dual_reciprocal_midpoint");
    let (mut raw_profile, frame) = input_from_row(&dual, provenance, 0);
    raw_profile.slots[0].camera_calibration = None;
    let profile = resolver.validate_profile(raw_profile).unwrap();
    assert!(resolver.resolve(&profile, &frame).is_ok());

    for (vector_id, expected) in [
        ("interpolated_singular_color_matrix", "SINGULAR_MATRIX"),
        (
            "interpolated_ill_conditioned_color_matrix",
            "ILL_CONDITIONED_MATRIX",
        ),
    ] {
        let row = vector_row(vector_id);
        let (raw_profile, frame) = input_from_row(&row, provenance, 0);
        let profile = resolver.validate_profile(raw_profile).unwrap();
        assert_eq!(
            resolver.resolve(&profile, &frame).unwrap_err().code(),
            expected
        );
    }
}

#[test]
fn missing_asn_and_provenance_or_identity_mismatch_fail_closed() {
    let resolver = StrictMotionCamForwardMatrixColorV2::new().unwrap();
    let row = vector_row("identity_d65_single");
    let p1 = StrictColorProfileProvenance::from_source_sha256([1; 32]);
    let p2 = StrictColorProfileProvenance::from_source_sha256([2; 32]);
    let (raw_profile, mut frame) = input_from_row(&row, p1, 1);
    let profile = resolver.validate_profile(raw_profile).unwrap();
    frame.raw.as_shot_neutral = None;
    assert_eq!(
        resolver.resolve(&profile, &frame).unwrap_err().code(),
        "MISSING_AS_SHOT_NEUTRAL"
    );
    frame.raw.as_shot_neutral = Some([0.5, 1.0, 0.7]);
    frame.raw.provenance = p2;
    assert_eq!(
        resolver.resolve(&profile, &frame).unwrap_err().code(),
        "COLOR_PROVENANCE_MISMATCH"
    );
}

#[test]
fn matrix_bound_checks_are_typed_and_do_not_clamp() {
    let resolver = StrictMotionCamForwardMatrixColorV2::new().unwrap();
    let provenance = StrictColorProfileProvenance::from_source_sha256([4; 32]);
    let row = vector_row("identity_d65_single");
    let (raw_profile, frame) = input_from_row(&row, provenance, 0);
    let profile = resolver.validate_profile(raw_profile).unwrap();
    let resolved = resolver.resolve(&profile, &frame).unwrap();
    resolved.validate_demosaiced_component_bound(1.0).unwrap();
    assert_eq!(
        resolved
            .validate_demosaiced_component_bound(f64::MAX)
            .unwrap_err()
            .code(),
        "UNSAFE_DEMOSAICED_MAGNITUDE"
    );
}

#[test]
fn independent_test_reference_reproduces_portable_accepts_and_exact_ten_rejects() {
    let frozen_f32 = frozen_f32_map();
    let mut accepted = 0;
    let mut rejected = 0;
    let mut serialized_rows = 0;
    for row in parse_tsv(F64_VECTORS)
        .into_iter()
        .filter(|row| row.get("provenance") == "synthetic")
    {
        let vector_id = row.get("vector_id");
        match independent_reference::resolve(&row) {
            Ok(result) => {
                accepted += 1;
                assert_ne!(
                    row.get("accepted_resolver_class"),
                    "REJECTED",
                    "{vector_id}"
                );
                serialized_rows +=
                    assert_independent_forward_matrix_vector(&frozen_f32, &row, &result);
            }
            Err(code) => {
                rejected += 1;
                assert_eq!(code, row.get("expected_failure"), "{vector_id}");
            }
        }
    }
    assert_eq!((accepted, rejected), (13, 10));
    assert_eq!(serialized_rows, 351);
}

/// Test-only second implementation of the hash-pinned strict-color equations. It does
/// not call the production resolver or matrix helpers, and deliberately keeps
/// the DNG-spec ColorMatrix fallback so the portable reference-only accept remains
/// independently verifiable while production rejects it.
mod independent_reference {
    use serde_json::Value;

    use super::{TsvRow, parse_f64_array, parse_matrix_groups};
    use crate::strict_motioncam_color::policy::{
        BRADFORD_CONE, D50_XY, D65_XY, LINEAR_BT2020_TO_NORMALIZED_NCL, MAX_COMPOSITE_ELEMENT,
        MAX_CONDITION_1, MAX_INPUT_ELEMENT, MIN_FORWARD_NORMALIZATION_ROW_SUM,
        MIN_RELATIVE_DETERMINANT, TEMPERATURE_TABLE, WHITE_MAX_ITERATIONS, WHITE_TOLERANCE_L1_XY,
        XYZ_D65_TO_LINEAR_BT2020,
    };

    type Matrix = [f64; 9];
    const IDENTITY: Matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    pub(super) struct ResultMatrices {
        pub(super) t50: Matrix,
        pub(super) linear_bt2020: Matrix,
        pub(super) normalized_ncl: Matrix,
    }

    #[derive(Clone, Copy)]
    struct Slot {
        illuminant: u8,
        temperature: f64,
        color_matrix: Matrix,
        calibration_matrix: Matrix,
        forward_matrix: Option<Matrix>,
    }

    #[derive(Clone, Copy)]
    struct Interpolated {
        color_matrix: Matrix,
        calibration_matrix: Matrix,
        forward_matrix: Option<Matrix>,
    }

    pub(super) fn resolve(row: &TsvRow) -> Result<ResultMatrices, &'static str> {
        let neutral = parse_f64_array::<3>(row.get("input_as_shot_neutral"));
        if neutral
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("INVALID_AS_SHOT_NEUTRAL");
        }
        let maximum = neutral.into_iter().fold(0.0_f64, f64::max);
        let camera_neutral = neutral.map(|value| value / maximum);
        let analog_balance = match row.get("input_analog_balance") {
            "ABSENT" => [1.0; 3],
            value => parse_f64_array(value),
        };
        if analog_balance
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("INVALID_ANALOG_BALANCE");
        }
        let tokens: Vec<Value> =
            serde_json::from_str(row.get("raw_illuminant_tokens_json")).unwrap();
        let color_matrices = parse_matrix_groups(row.get("input_color_matrices"));
        let calibration_matrices = parse_matrix_groups(row.get("input_calibration_matrices"));
        let forward_matrices = parse_matrix_groups(row.get("input_forward_matrices"));
        let mut slots = Vec::new();
        let mut forward_presence = Vec::new();
        for index in 0..tokens.len() {
            let (illuminant, temperature) = match &tokens[index] {
                Value::String(value) if value == "standarda" => (0, 2850.0),
                Value::String(value) if value == "d65" => (1, 6500.0),
                Value::Number(value) if value.as_i64() == Some(17) => (0, 2850.0),
                Value::Number(value) if value.as_i64() == Some(21) => (1, 6500.0),
                _ => return Err("UNKNOWN_ILLUMINANT"),
            };
            let color_matrix = color_matrices[index].ok_or("INCOMPLETE_COLOR_SLOT")?;
            validate_matrix(color_matrix)?;
            let calibration_matrix = calibration_matrices[index].unwrap_or(IDENTITY);
            validate_matrix(calibration_matrix)?;
            let forward_matrix = forward_matrices[index].map(normalize_forward).transpose()?;
            forward_presence.push(forward_matrix.is_some());
            slots.push(Slot {
                illuminant,
                temperature,
                color_matrix,
                calibration_matrix,
                forward_matrix,
            });
        }
        if slots.len() == 2 && slots[0].illuminant == slots[1].illuminant {
            return Err("DUPLICATE_ILLUMINANTS");
        }
        if forward_presence.iter().any(|value| *value)
            && !forward_presence.iter().all(|value| *value)
        {
            return Err("INCOMPLETE_FORWARD_MATRIX_SET");
        }
        slots.sort_by(|left, right| left.temperature.partial_cmp(&right.temperature).unwrap());

        let ab = diagonal(analog_balance);
        let mut white_xy = D50_XY;
        let mut converged = false;
        for _ in 0..WHITE_MAX_ITERATIONS {
            let facts = interpolate(&slots, white_xy)?;
            let xyz_to_camera =
                multiply(ab, multiply(facts.calibration_matrix, facts.color_matrix));
            validate_matrix(xyz_to_camera)?;
            let next_xy = xyz_to_xy(apply(inverse(xyz_to_camera)?, camera_neutral))?;
            let delta = (next_xy[0] - white_xy[0]).abs() + (next_xy[1] - white_xy[1]).abs();
            white_xy = next_xy;
            if delta < WHITE_TOLERANCE_L1_XY {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err("WHITE_SOLVE_NONCONVERGENT");
        }
        let facts = interpolate(&slots, white_xy)?;
        let abcc = multiply(ab, facts.calibration_matrix);
        let inverse_abcc = inverse(abcc)?;
        let reference_neutral = apply(inverse_abcc, camera_neutral);
        let t50 = if let Some(forward_matrix) = facts.forward_matrix {
            if reference_neutral
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err("NUMERICALLY_INVALID");
            }
            multiply(
                multiply(
                    forward_matrix,
                    diagonal(reference_neutral.map(|value| 1.0 / value)),
                ),
                inverse_abcc,
            )
        } else {
            let xyz_to_camera = multiply(abcc, facts.color_matrix);
            multiply(adaptation(white_xy, D50_XY)?, inverse(xyz_to_camera)?)
        };
        validate_composite(t50)?;
        let d50_to_d65 = adaptation(D50_XY, D65_XY)?;
        let linear_bt2020 = multiply(multiply(XYZ_D65_TO_LINEAR_BT2020, d50_to_d65), t50);
        let normalized_ncl = multiply(LINEAR_BT2020_TO_NORMALIZED_NCL, linear_bt2020);
        Ok(ResultMatrices {
            t50,
            linear_bt2020,
            normalized_ncl,
        })
    }

    fn interpolate(slots: &[Slot], xy: [f64; 2]) -> Result<Interpolated, &'static str> {
        let temperature = temperature_tint(xy)?.0;
        let weight = if slots.len() == 1 || temperature <= slots[0].temperature {
            1.0
        } else if temperature >= slots[1].temperature {
            0.0
        } else {
            ((1.0 / temperature) - (1.0 / slots[1].temperature))
                / ((1.0 / slots[0].temperature) - (1.0 / slots[1].temperature))
        };
        let low = slots[0];
        let high = slots[slots.len() - 1];
        let output = if slots.len() == 1 {
            Interpolated {
                color_matrix: low.color_matrix,
                calibration_matrix: low.calibration_matrix,
                forward_matrix: low.forward_matrix,
            }
        } else {
            Interpolated {
                color_matrix: lerp(low.color_matrix, high.color_matrix, weight),
                calibration_matrix: lerp(low.calibration_matrix, high.calibration_matrix, weight),
                forward_matrix: Some(lerp(
                    low.forward_matrix.unwrap(),
                    high.forward_matrix.unwrap(),
                    weight,
                )),
            }
        };
        validate_matrix(output.color_matrix)?;
        validate_matrix(output.calibration_matrix)?;
        if let Some(forward_matrix) = output.forward_matrix {
            validate_matrix(forward_matrix)?;
        }
        Ok(output)
    }

    fn normalize_forward(matrix: Matrix) -> Result<Matrix, &'static str> {
        validate_matrix(matrix)?;
        let sums = apply(matrix, [1.0; 3]);
        if sums
            .iter()
            .any(|value| value.abs() <= MIN_FORWARD_NORMALIZATION_ROW_SUM)
        {
            return Err("UNREASONABLE_MATRIX");
        }
        let pcs = xy_to_xyz(D50_XY)?;
        let output = multiply(
            diagonal([pcs[0] / sums[0], pcs[1] / sums[1], pcs[2] / sums[2]]),
            matrix,
        );
        validate_matrix(output)?;
        Ok(output)
    }

    fn validate_matrix(matrix: Matrix) -> Result<(), &'static str> {
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err("NONFINITE_MATRIX");
        }
        if matrix.iter().map(|value| value.abs()).fold(0.0, f64::max) > MAX_INPUT_ELEMENT {
            return Err("UNREASONABLE_MATRIX");
        }
        let _ = inverse(matrix)?;
        Ok(())
    }

    fn validate_composite(matrix: Matrix) -> Result<(), &'static str> {
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err("NONFINITE_COMPOSITE");
        }
        if matrix.iter().map(|value| value.abs()).fold(0.0, f64::max) > MAX_COMPOSITE_ELEMENT {
            return Err("UNREASONABLE_COMPOSITE");
        }
        let _ = inverse(matrix)?;
        Ok(())
    }

    fn inverse(matrix: Matrix) -> Result<Matrix, &'static str> {
        let determinant = determinant(matrix);
        let norm = norm1(matrix);
        let relative = if norm == 0.0 {
            0.0
        } else {
            determinant.abs() / norm.powi(3)
        };
        if relative <= MIN_RELATIVE_DETERMINANT {
            return Err("SINGULAR_MATRIX");
        }
        let m = matrix;
        let output = [
            (m[4] * m[8] - m[5] * m[7]) / determinant,
            (m[2] * m[7] - m[1] * m[8]) / determinant,
            (m[1] * m[5] - m[2] * m[4]) / determinant,
            (m[5] * m[6] - m[3] * m[8]) / determinant,
            (m[0] * m[8] - m[2] * m[6]) / determinant,
            (m[2] * m[3] - m[0] * m[5]) / determinant,
            (m[3] * m[7] - m[4] * m[6]) / determinant,
            (m[1] * m[6] - m[0] * m[7]) / determinant,
            (m[0] * m[4] - m[1] * m[3]) / determinant,
        ];
        if norm * norm1(output) > MAX_CONDITION_1 {
            return Err("ILL_CONDITIONED_MATRIX");
        }
        Ok(output)
    }

    fn determinant(m: Matrix) -> f64 {
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
    }

    fn norm1(matrix: Matrix) -> f64 {
        (0..3)
            .map(|column| {
                sum3([
                    matrix[column].abs(),
                    matrix[3 + column].abs(),
                    matrix[6 + column].abs(),
                ])
            })
            .fold(0.0, f64::max)
    }

    fn multiply(left: Matrix, right: Matrix) -> Matrix {
        let mut output = [0.0; 9];
        for row in 0..3 {
            for column in 0..3 {
                output[row * 3 + column] = sum3([
                    left[row * 3] * right[column],
                    left[row * 3 + 1] * right[3 + column],
                    left[row * 3 + 2] * right[6 + column],
                ]);
            }
        }
        output
    }

    fn apply(matrix: Matrix, vector: [f64; 3]) -> [f64; 3] {
        let mut output = [0.0; 3];
        for row in 0..3 {
            output[row] = sum3([
                matrix[row * 3] * vector[0],
                matrix[row * 3 + 1] * vector[1],
                matrix[row * 3 + 2] * vector[2],
            ]);
        }
        output
    }

    fn sum3(values: [f64; 3]) -> f64 {
        let mut high = 0.0_f64;
        let mut low = 0.0_f64;
        for value in values {
            let next = high + value;
            if high.abs() >= value.abs() {
                low += (high - next) + value;
            } else {
                low += (value - next) + high;
            }
            high = next;
        }
        high + low
    }

    fn diagonal(values: [f64; 3]) -> Matrix {
        [
            values[0], 0.0, 0.0, 0.0, values[1], 0.0, 0.0, 0.0, values[2],
        ]
    }

    fn lerp(left: Matrix, right: Matrix, weight: f64) -> Matrix {
        let mut output = [0.0; 9];
        for index in 0..9 {
            output[index] = weight * left[index] + (1.0 - weight) * right[index];
        }
        output
    }

    fn xy_to_xyz(xy: [f64; 2]) -> Result<[f64; 3], &'static str> {
        if xy[0] <= 0.0 || xy[1] <= 0.0 || xy[0] + xy[1] >= 1.0 {
            return Err("NUMERICALLY_INVALID");
        }
        Ok([xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]])
    }

    fn xyz_to_xy(xyz: [f64; 3]) -> Result<[f64; 2], &'static str> {
        let total = sum3(xyz);
        if total <= 0.0 || !total.is_finite() {
            return Err("NUMERICALLY_INVALID");
        }
        Ok([xyz[0] / total, xyz[1] / total])
    }

    fn adaptation(source: [f64; 2], target: [f64; 2]) -> Result<Matrix, &'static str> {
        let source_lms = apply(BRADFORD_CONE, xy_to_xyz(source)?);
        let target_lms = apply(BRADFORD_CONE, xy_to_xyz(target)?);
        Ok(multiply(
            multiply(
                inverse(BRADFORD_CONE)?,
                diagonal([
                    target_lms[0] / source_lms[0],
                    target_lms[1] / source_lms[1],
                    target_lms[2] / source_lms[2],
                ]),
            ),
            BRADFORD_CONE,
        ))
    }

    fn temperature_tint(xy: [f64; 2]) -> Result<(f64, f64), &'static str> {
        let denominator = 1.5 - xy[0] + 6.0 * xy[1];
        let u = 2.0 * xy[0] / denominator;
        let v = 3.0 * xy[1] / denominator;
        let mut last_dt = 0.0;
        let mut last_du = 0.0;
        let mut last_dv = 0.0;
        for index in 1..TEMPERATURE_TABLE.len() {
            let [_, table_u, table_v, slope] = TEMPERATURE_TABLE[index];
            let length = exact_hypot(1.0, slope);
            let mut du = 1.0 / length;
            let mut dv = slope / length;
            let uu = u - table_u;
            let vv = v - table_v;
            let mut dt = -uu * dv + vv * du;
            if dt <= 0.0 || index + 1 == TEMPERATURE_TABLE.len() {
                dt = (-dt).max(0.0);
                let factor = if index == 1 { 0.0 } else { dt / (last_dt + dt) };
                let reciprocal = TEMPERATURE_TABLE[index - 1][0] * factor
                    + TEMPERATURE_TABLE[index][0] * (1.0 - factor);
                let line_u = TEMPERATURE_TABLE[index - 1][1] * factor + table_u * (1.0 - factor);
                let line_v = TEMPERATURE_TABLE[index - 1][2] * factor + table_v * (1.0 - factor);
                let uu = u - line_u;
                let vv = v - line_v;
                du = du * (1.0 - factor) + last_du * factor;
                dv = dv * (1.0 - factor) + last_dv * factor;
                let length = exact_hypot(du, dv);
                du /= length;
                dv /= length;
                return Ok((1.0e6 / reciprocal, (uu * du + vv * dv) * -3000.0));
            }
            last_dt = dt;
            last_du = du;
            last_dv = dv;
        }
        Err("NUMERICALLY_INVALID")
    }

    fn exact_hypot(a: f64, b: f64) -> f64 {
        let a = a.abs();
        let b = b.abs();
        let p = a * a;
        let pe = a.mul_add(a, -p);
        let q = b * b;
        let qe = b.mul_add(b, -q);
        let sum = p + q;
        let virtual_q = sum - p;
        let sum_error = (p - (sum - virtual_q)) + (q - virtual_q);
        let low = (sum_error + pe) + qe;
        let root = sum.sqrt();
        let square = root * root;
        let square_error = root.mul_add(root, -square);
        root + ((sum - square) + (low - square_error)) / (2.0 * root)
    }
}
