//! Rendered-PIPE strict MotionCam Camera2 color resolver.
//!
//! Shared validated camera-color calculation for DISPLAY and direct-YUV PIPE.
//! Source DNG metadata remains independent of these derived rendering transforms.

mod fingerprint;
mod matrix;
mod policy;
mod sha256;

use std::error::Error;
use std::fmt;

use mcraw4vulkan_mcrawcontainer::{
    RawCamera2ColorProfile, RawCamera2ColorSourceError, RawCamera2FrameColor,
    RawColorCalibrationSlot, RawIlluminantToken, StrictColorProfileProvenance,
};

pub(crate) use fingerprint::DeferredColorContextFingerprintV2;
pub use fingerprint::{
    ClipSourceSha256, ColorContextFingerprintFacts, ColorContextFingerprintV2,
    VerifiedStrictPipeColorContextV2,
};
use matrix::{IDENTITY, Matrix3, diagonal, lerp, python_sum3};
use policy::{
    ASN_NORMALIZATION_MAX, BRADFORD_D50_TO_D65, D50_XY, D65_TEMPERATURE, EXPECTED_POLICY_DIGEST,
    LINEAR_BT2020_TO_NORMALIZED_NCL, MIN_FORWARD_NORMALIZATION_ROW_SUM, STANDARD_A_TEMPERATURE,
    TEMPERATURE_TABLE, WHITE_MAX_ITERATIONS, WHITE_TOLERANCE_L1_XY, XYZ_D65_TO_LINEAR_BT2020,
    policy_record,
};
use sha256::Sha256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrictColorPolicyDigestV2([u8; 32]);

impl StrictColorPolicyDigestV2 {
    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Illuminant {
    StandardA,
    D65,
    D50,
}

impl Illuminant {
    const fn temperature(self) -> f64 {
        match self {
            Self::StandardA => STANDARD_A_TEMPERATURE,
            Self::D65 => D65_TEMPERATURE,
            Self::D50 => 5000.0,
        }
    }
}

#[derive(Debug, Clone)]
struct ValidatedSlot {
    temperature: f64,
    color_matrix: Matrix3,
    camera_calibration: Matrix3,
    normalized_forward_matrix: Option<Matrix3>,
}

#[derive(Debug, Clone)]
pub struct StrictMotionCamColorProfile {
    raw: RawCamera2ColorProfile,
    analog_balance: [f64; 3],
    slots_by_temperature: Vec<ValidatedSlot>,
}

impl StrictMotionCamColorProfile {
    pub fn is_color_matrix_only(&self) -> bool {
        self.slots_by_temperature[0]
            .normalized_forward_matrix
            .is_none()
    }

    pub fn policy_name(&self) -> &'static str {
        if self.is_color_matrix_only() {
            "StrictMotionCamColorMatrixColorV1"
        } else {
            "StrictMotionCamForwardMatrixColorV2"
        }
    }

    pub const fn provenance(&self) -> StrictColorProfileProvenance {
        self.raw.provenance
    }
}

#[derive(Debug, Clone)]
pub struct StrictMotionCamFrameColorInput {
    raw: RawCamera2FrameColor,
}

impl StrictMotionCamFrameColorInput {
    pub fn from_raw(raw: RawCamera2FrameColor) -> Self {
        Self { raw }
    }

    pub const fn source_frame_index(&self) -> u64 {
        self.raw.source_frame_index
    }

    pub const fn provenance(&self) -> StrictColorProfileProvenance {
        self.raw.provenance
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedStrictPipeColor {
    provenance: StrictColorProfileProvenance,
    source_frame_index: u64,
    camera_neutral: [f64; 3],
    white_xy: [f64; 2],
    temperature: f64,
    tint: f64,
    low_temperature_weight: f64,
    t50: [f64; 9],
    #[allow(dead_code)] // retained for exact portable resolver-vector coverage
    camera_to_linear_bt2020: [f64; 9],
    #[allow(dead_code)] // retained for exact portable resolver-vector coverage
    camera_to_normalized_ncl: [f64; 9],
    camera_to_normalized_ncl_f32: [f32; 9],
    #[allow(dead_code)] // retained for exact portable convergence coverage
    convergence_iterations: u32,
}

impl ResolvedStrictPipeColor {
    pub const fn provenance(&self) -> StrictColorProfileProvenance {
        self.provenance
    }

    pub const fn source_frame_index(&self) -> u64 {
        self.source_frame_index
    }

    pub const fn camera_neutral(&self) -> [f64; 3] {
        self.camera_neutral
    }

    pub const fn white_xy(&self) -> [f64; 2] {
        self.white_xy
    }

    pub const fn temperature_kelvin(&self) -> f64 {
        self.temperature
    }

    pub const fn tint(&self) -> f64 {
        self.tint
    }

    pub const fn low_temperature_weight(&self) -> f64 {
        self.low_temperature_weight
    }

    pub const fn t50(&self) -> [f64; 9] {
        self.t50
    }

    #[allow(dead_code)] // retained for exact portable resolver-vector coverage
    pub const fn camera_to_linear_bt2020(&self) -> [f64; 9] {
        self.camera_to_linear_bt2020
    }

    #[allow(dead_code)] // retained for exact portable resolver-vector coverage
    pub const fn camera_to_normalized_ncl(&self) -> [f64; 9] {
        self.camera_to_normalized_ncl
    }

    pub const fn camera_to_normalized_ncl_f32(&self) -> [f32; 9] {
        self.camera_to_normalized_ncl_f32
    }

    #[allow(dead_code)] // retained for exact portable f32 serialization coverage
    pub fn camera_to_normalized_ncl_f32_bits(&self) -> [u32; 9] {
        self.camera_to_normalized_ncl_f32.map(f32::to_bits)
    }

    #[allow(dead_code)] // retained for exact portable convergence coverage
    pub const fn convergence_iterations(&self) -> u32 {
        self.convergence_iterations
    }

    pub fn validate_demosaiced_component_bound(
        &self,
        conservative_component_bound: f64,
    ) -> Result<(), StrictMotionCamColorError> {
        if !conservative_component_bound.is_finite() || conservative_component_bound < 0.0 {
            return Err(StrictMotionCamColorError::UnsafeDemosaicedMagnitude {
                bound: conservative_component_bound,
                maximum_row_sum: f64::NAN,
            });
        }
        let maximum_row_sum = self
            .camera_to_normalized_ncl_f32
            .chunks_exact(3)
            .map(|row| row.iter().map(|value| f64::from(*value).abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let product = conservative_component_bound * maximum_row_sum;
        if !product.is_finite() || product > f64::from(f32::MAX) / 2.0 {
            return Err(StrictMotionCamColorError::UnsafeDemosaicedMagnitude {
                bound: conservative_component_bound,
                maximum_row_sum,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StrictMotionCamColorError {
    MissingAsShotNeutral,
    InvalidAsShotNeutral,
    UnknownIlluminant {
        source_slot: u8,
    },
    DuplicateIlluminants,
    IncompleteColorSlot {
        source_slot: u8,
    },
    IncompleteForwardMatrixSet,
    UnsupportedColorMetadata,
    ColorProvenanceMismatch,
    InvalidAnalogBalance,
    NonfiniteMatrix {
        stage: &'static str,
    },
    SingularMatrix {
        stage: &'static str,
        relative_determinant: f64,
    },
    IllConditionedMatrix {
        stage: &'static str,
        condition: f64,
    },
    UnreasonableMatrix {
        stage: &'static str,
        maximum: f64,
    },
    WhiteSolveNonconvergent {
        iterations: u32,
    },
    NonfiniteComposite {
        stage: &'static str,
    },
    UnreasonableComposite {
        stage: &'static str,
        maximum: f64,
    },
    NumericallyInvalid {
        stage: &'static str,
    },
    PolicyRecordMismatch,
    ContextIdentityMismatch,
    SourceShaIo {
        detail: String,
    },
    UnsafeDemosaicedMagnitude {
        bound: f64,
        maximum_row_sum: f64,
    },
    SourceMetadata {
        detail: String,
    },
}

impl StrictMotionCamColorError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::MissingAsShotNeutral => "MISSING_AS_SHOT_NEUTRAL",
            Self::InvalidAsShotNeutral => "INVALID_AS_SHOT_NEUTRAL",
            Self::UnknownIlluminant { .. } => "UNKNOWN_ILLUMINANT",
            Self::DuplicateIlluminants => "DUPLICATE_ILLUMINANTS",
            Self::IncompleteColorSlot { .. } => "INCOMPLETE_COLOR_SLOT",
            Self::IncompleteForwardMatrixSet => "INCOMPLETE_FORWARD_MATRIX_SET",
            Self::UnsupportedColorMetadata => "UNSUPPORTED_COLOR_METADATA",
            Self::ColorProvenanceMismatch => "COLOR_PROVENANCE_MISMATCH",
            Self::InvalidAnalogBalance => "INVALID_ANALOG_BALANCE",
            Self::NonfiniteMatrix { .. } => "NONFINITE_MATRIX",
            Self::SingularMatrix { .. } => "SINGULAR_MATRIX",
            Self::IllConditionedMatrix { .. } => "ILL_CONDITIONED_MATRIX",
            Self::UnreasonableMatrix { .. } => "UNREASONABLE_MATRIX",
            Self::WhiteSolveNonconvergent { .. } => "WHITE_SOLVE_NONCONVERGENT",
            Self::NonfiniteComposite { .. } => "NONFINITE_COMPOSITE",
            Self::UnreasonableComposite { .. } => "UNREASONABLE_COMPOSITE",
            Self::NumericallyInvalid { .. } => "NUMERICALLY_INVALID",
            Self::PolicyRecordMismatch => "POLICY_RECORD_MISMATCH",
            Self::ContextIdentityMismatch => "CONTEXT_IDENTITY_MISMATCH",
            Self::SourceShaIo { .. } => "SOURCE_SHA_IO",
            Self::UnsafeDemosaicedMagnitude { .. } => "UNSAFE_DEMOSAICED_MAGNITUDE",
            Self::SourceMetadata { .. } => "SOURCE_METADATA",
        }
    }
}

impl fmt::Display for StrictMotionCamColorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: ", self.code())?;
        match self {
            Self::UnknownIlluminant { source_slot } | Self::IncompleteColorSlot { source_slot } => {
                write!(formatter, "source slot {source_slot}")
            }
            Self::SingularMatrix {
                stage,
                relative_determinant,
            } => {
                write!(
                    formatter,
                    "{stage} relative determinant {relative_determinant:.17}"
                )
            }
            Self::IllConditionedMatrix { stage, condition } => {
                write!(formatter, "{stage} condition-1 {condition:.17}")
            }
            Self::UnreasonableMatrix { stage, maximum }
            | Self::UnreasonableComposite { stage, maximum } => {
                write!(formatter, "{stage} maximum {maximum:.17}")
            }
            Self::WhiteSolveNonconvergent { iterations } => {
                write!(formatter, "no fixed point within {iterations} iterations")
            }
            Self::NonfiniteMatrix { stage }
            | Self::NonfiniteComposite { stage }
            | Self::NumericallyInvalid { stage } => formatter.write_str(stage),
            Self::SourceShaIo { detail } | Self::SourceMetadata { detail } => {
                formatter.write_str(detail)
            }
            Self::UnsafeDemosaicedMagnitude {
                bound,
                maximum_row_sum,
            } => write!(
                formatter,
                "component bound {bound:.17}, maximum absolute row sum {maximum_row_sum:.17}"
            ),
            _ => formatter.write_str("strict color contract rejected the input"),
        }
    }
}

impl Error for StrictMotionCamColorError {}

impl From<RawCamera2ColorSourceError> for StrictMotionCamColorError {
    fn from(error: RawCamera2ColorSourceError) -> Self {
        match error {
            RawCamera2ColorSourceError::InvalidAnalogBalance => Self::InvalidAnalogBalance,
            RawCamera2ColorSourceError::InvalidAsShotNeutral => Self::InvalidAsShotNeutral,
            other => Self::SourceMetadata {
                detail: other.to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct InterpolatedFacts {
    temperature: f64,
    tint: f64,
    weight_low: f64,
    color_matrix: Matrix3,
    camera_calibration: Matrix3,
    forward_matrix: Option<Matrix3>,
}

#[derive(Debug, Clone)]
pub struct StrictMotionCamForwardMatrixColorV2 {
    policy_digest: StrictColorPolicyDigestV2,
}

impl StrictMotionCamForwardMatrixColorV2 {
    pub fn new() -> Result<Self, StrictMotionCamColorError> {
        let record = policy_record();
        if record.len() != policy::POLICY_RECORD_BYTE_LEN {
            return Err(StrictMotionCamColorError::PolicyRecordMismatch);
        }
        let digest = Sha256::digest(&record);
        if digest != EXPECTED_POLICY_DIGEST {
            return Err(StrictMotionCamColorError::PolicyRecordMismatch);
        }
        Ok(Self {
            policy_digest: StrictColorPolicyDigestV2(digest),
        })
    }

    pub const fn policy_digest(&self) -> StrictColorPolicyDigestV2 {
        self.policy_digest
    }

    pub fn parse_and_validate_profile(
        &self,
        container_metadata_json: &str,
        provenance: StrictColorProfileProvenance,
    ) -> Result<StrictMotionCamColorProfile, StrictMotionCamColorError> {
        let raw = RawCamera2ColorProfile::parse(container_metadata_json, provenance)?;
        self.validate_profile(raw)
    }

    /// Select a validated profile by source calibration availability. The V2
    /// entry points retain their frozen ForwardMatrix-only acceptance contract.
    pub fn parse_supported_profile(
        &self,
        json: &str,
        provenance: StrictColorProfileProvenance,
    ) -> Result<StrictMotionCamColorProfile, StrictMotionCamColorError> {
        self.validate_supported_profile(RawCamera2ColorProfile::parse(json, provenance)?)
    }

    pub fn validate_supported_profile(
        &self,
        raw: RawCamera2ColorProfile,
    ) -> Result<StrictMotionCamColorProfile, StrictMotionCamColorError> {
        self.validate_profile_inner(raw, true)
    }

    pub fn effective_profile<'a>(
        &self,
        profile: &'a StrictMotionCamColorProfile,
        overrides: &mcraw4vulkan_mcrawcontainer::ColorMetadataOverrides,
    ) -> Result<std::borrow::Cow<'a, StrictMotionCamColorProfile>, StrictMotionCamColorError> {
        let raw = overrides.apply_to_profile(&profile.raw);
        if raw == profile.raw {
            Ok(std::borrow::Cow::Borrowed(profile))
        } else {
            Ok(std::borrow::Cow::Owned(
                self.validate_supported_profile(raw)?,
            ))
        }
    }

    pub fn policy_digest_for(
        &self,
        profile: &StrictMotionCamColorProfile,
    ) -> StrictColorPolicyDigestV2 {
        if profile.is_color_matrix_only() {
            let mut record = policy_record();
            record.extend_from_slice(b"StrictMotionCamColorMatrixColorV1:inverse(AB*CC*CM);Bradford-white-to-D50;D50=5000K");
            StrictColorPolicyDigestV2(Sha256::digest(&record))
        } else {
            self.policy_digest()
        }
    }

    pub fn parse_frame_input(
        &self,
        frame_metadata_json: &str,
        source_frame_index: u64,
        provenance: StrictColorProfileProvenance,
    ) -> Result<StrictMotionCamFrameColorInput, StrictMotionCamColorError> {
        Ok(StrictMotionCamFrameColorInput::from_raw(
            RawCamera2FrameColor::parse(frame_metadata_json, source_frame_index, provenance)?,
        ))
    }

    pub fn validate_profile(
        &self,
        raw: RawCamera2ColorProfile,
    ) -> Result<StrictMotionCamColorProfile, StrictMotionCamColorError> {
        self.validate_profile_inner(raw, false)
    }

    fn validate_profile_inner(
        &self,
        raw: RawCamera2ColorProfile,
        allow_color_matrix_only: bool,
    ) -> Result<StrictMotionCamColorProfile, StrictMotionCamColorError> {
        let analog_balance = raw.analog_balance.unwrap_or([1.0; 3]);
        if analog_balance
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(StrictMotionCamColorError::InvalidAnalogBalance);
        }
        if !(1..=2).contains(&raw.slots.len()) {
            return Err(StrictMotionCamColorError::UnsupportedColorMetadata);
        }
        if raw.slots.len() == 2 && raw.slots[0].source_slot == raw.slots[1].source_slot {
            return Err(StrictMotionCamColorError::UnsupportedColorMetadata);
        }

        let mut slots = Vec::with_capacity(raw.slots.len());
        let mut forward_presence = Vec::with_capacity(raw.slots.len());
        for slot in &raw.slots {
            if slot.provenance != raw.provenance {
                return Err(StrictMotionCamColorError::ColorProvenanceMismatch);
            }
            let source_slot = slot.source_slot.number();
            let illuminant = parse_illuminant(slot, allow_color_matrix_only)?;
            let Some(raw_cm) = slot.color_matrix else {
                return Err(StrictMotionCamColorError::IncompleteColorSlot { source_slot });
            };
            let color_matrix = Matrix3(raw_cm.values).validate_input("ColorMatrix")?;
            let camera_calibration = match slot.camera_calibration {
                Some(raw_cc) => Matrix3(raw_cc.values).validate_input("CameraCalibration")?,
                None => IDENTITY,
            };
            let forward_matrix = slot
                .forward_matrix
                .map(|raw_fm| normalize_forward(Matrix3(raw_fm.values)))
                .transpose()?;
            forward_presence.push(forward_matrix.is_some());
            slots.push((illuminant, color_matrix, camera_calibration, forward_matrix));
        }
        if slots.len() == 2 && slots[0].0 == slots[1].0 {
            return Err(StrictMotionCamColorError::DuplicateIlluminants);
        }
        if forward_presence.iter().any(|present| *present)
            && !forward_presence.iter().all(|present| *present)
        {
            return Err(StrictMotionCamColorError::IncompleteForwardMatrixSet);
        }
        if !allow_color_matrix_only && !forward_presence.iter().all(|present| *present) {
            return Err(StrictMotionCamColorError::UnsupportedColorMetadata);
        }

        let mut slots_by_temperature = slots
            .into_iter()
            .map(
                |(illuminant, color_matrix, camera_calibration, forward_matrix)| ValidatedSlot {
                    temperature: illuminant.temperature(),
                    color_matrix,
                    camera_calibration,
                    normalized_forward_matrix: forward_matrix,
                },
            )
            .collect::<Vec<_>>();
        slots_by_temperature.sort_by(|left, right| {
            left.temperature
                .partial_cmp(&right.temperature)
                .expect("registry temperatures are finite")
        });
        Ok(StrictMotionCamColorProfile {
            raw,
            analog_balance,
            slots_by_temperature,
        })
    }

    pub fn resolve(
        &self,
        profile: &StrictMotionCamColorProfile,
        frame: &StrictMotionCamFrameColorInput,
    ) -> Result<ResolvedStrictPipeColor, StrictMotionCamColorError> {
        if profile.provenance() != frame.provenance() {
            return Err(StrictMotionCamColorError::ColorProvenanceMismatch);
        }
        let as_shot_neutral = frame
            .raw
            .as_shot_neutral
            .ok_or(StrictMotionCamColorError::MissingAsShotNeutral)?;
        if as_shot_neutral
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(StrictMotionCamColorError::InvalidAsShotNeutral);
        }
        let maximum = as_shot_neutral.into_iter().fold(0.0_f64, f64::max);
        let camera_neutral = as_shot_neutral.map(|value| value / maximum * ASN_NORMALIZATION_MAX);
        if camera_neutral.into_iter().fold(0.0_f64, f64::max) != ASN_NORMALIZATION_MAX
            || camera_neutral.iter().any(|value| !value.is_finite())
        {
            return Err(StrictMotionCamColorError::InvalidAsShotNeutral);
        }

        let analog_balance_matrix = diagonal(profile.analog_balance);
        let mut white_xy = D50_XY;
        let mut converged_iteration = None;
        for iteration in 1..=WHITE_MAX_ITERATIONS {
            let facts = interpolate_for_xy(&profile.slots_by_temperature, white_xy)?;
            let xyz_to_camera = analog_balance_matrix
                .multiply(facts.camera_calibration.multiply(facts.color_matrix))
                .validate_input("XYZtoCamera")?;
            let next_xyz = xyz_to_camera.inverse("XYZtoCamera")?.apply(camera_neutral);
            let next_xy = xyz_to_xy(next_xyz)?;
            let delta = (next_xy[0] - white_xy[0]).abs() + (next_xy[1] - white_xy[1]).abs();
            white_xy = next_xy;
            if delta < WHITE_TOLERANCE_L1_XY {
                converged_iteration = Some(iteration);
                break;
            }
        }
        let convergence_iterations =
            converged_iteration.ok_or(StrictMotionCamColorError::WhiteSolveNonconvergent {
                iterations: WHITE_MAX_ITERATIONS,
            })?;
        let facts = interpolate_for_xy(&profile.slots_by_temperature, white_xy)?;
        let ab_cc = analog_balance_matrix.multiply(facts.camera_calibration);
        let t50 = if profile.is_color_matrix_only() {
            // DNG 1.7.1.0 pp. 101–103: unbalanced camera RGB -> XYZ at the
            // solved white -> Bradford adaptation to D50. Preserve luminance;
            // this is a rendering transform, never a source ForwardMatrix.
            let xyz_to_camera = ab_cc
                .multiply(facts.color_matrix)
                .validate_input("XYZtoCamera")?;
            chromatic_adaptation(white_xy, D50_XY)?
                .multiply(xyz_to_camera.inverse("XYZtoCamera")?)
                .validate_composite("ColorMatrix-only T50")?
        } else {
            let inverse_ab_cc = ab_cc.inverse("AB*CC")?;
            let reference_neutral = inverse_ab_cc.apply(camera_neutral);
            if reference_neutral
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(StrictMotionCamColorError::NumericallyInvalid {
                    stage: "ReferenceNeutral",
                });
            }
            let white_balance = diagonal(reference_neutral.map(|value| 1.0 / value));
            facts
                .forward_matrix
                .expect("complete ForwardMatrix profile")
                .multiply(white_balance)
                .multiply(inverse_ab_cc)
                .validate_composite("T50")?
        };
        let camera_to_linear_bt2020 = Matrix3(XYZ_D65_TO_LINEAR_BT2020)
            .multiply(Matrix3(BRADFORD_D50_TO_D65))
            .multiply(t50)
            .validate_composite("camera_to_linear_bt2020")?;
        let camera_to_normalized_ncl = Matrix3(LINEAR_BT2020_TO_NORMALIZED_NCL)
            .multiply(camera_to_linear_bt2020)
            .validate_composite("camera_to_normalized_ncl")?;
        let camera_to_normalized_ncl_f32 = camera_to_normalized_ncl.0.map(|value| value as f32);
        if camera_to_normalized_ncl_f32
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(StrictMotionCamColorError::NonfiniteComposite {
                stage: "camera_to_normalized_ncl_f32",
            });
        }

        Ok(ResolvedStrictPipeColor {
            provenance: profile.provenance(),
            source_frame_index: frame.source_frame_index(),
            camera_neutral,
            white_xy,
            temperature: facts.temperature,
            tint: facts.tint,
            low_temperature_weight: facts.weight_low,
            t50: t50.0,
            camera_to_linear_bt2020: camera_to_linear_bt2020.0,
            camera_to_normalized_ncl: camera_to_normalized_ncl.0,
            camera_to_normalized_ncl_f32,
            convergence_iterations,
        })
    }
}

fn parse_illuminant(
    slot: &RawColorCalibrationSlot,
    extended: bool,
) -> Result<Illuminant, StrictMotionCamColorError> {
    let source_slot = slot.source_slot.number();
    if !extended {
        match slot.illuminant.as_ref() {
            Some(RawIlluminantToken::String(v)) if v == "standarda" || v == "d65" => {}
            Some(RawIlluminantToken::Integer(17 | 21)) => {}
            None => return Err(StrictMotionCamColorError::IncompleteColorSlot { source_slot }),
            _ => return Err(StrictMotionCamColorError::UnknownIlluminant { source_slot }),
        }
    }
    match slot.illuminant.as_ref().map(RawIlluminantToken::dng_code) {
        Some(Some(17)) => Ok(Illuminant::StandardA),
        Some(Some(21)) => Ok(Illuminant::D65),
        Some(Some(23)) => Ok(Illuminant::D50),
        Some(_) => Err(StrictMotionCamColorError::UnknownIlluminant { source_slot }),
        None => Err(StrictMotionCamColorError::IncompleteColorSlot { source_slot }),
    }
}

fn chromatic_adaptation(
    source: [f64; 2],
    target: [f64; 2],
) -> Result<Matrix3, StrictMotionCamColorError> {
    let cone = Matrix3(policy::BRADFORD_CONE);
    let source_lms = cone.apply(xy_to_xyz(source)?);
    let target_lms = cone.apply(xy_to_xyz(target)?);
    if source_lms
        .iter()
        .chain(target_lms.iter())
        .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(StrictMotionCamColorError::NumericallyInvalid {
            stage: "Bradford white response",
        });
    }
    cone.inverse("Bradford cone")?
        .multiply(diagonal([
            target_lms[0] / source_lms[0],
            target_lms[1] / source_lms[1],
            target_lms[2] / source_lms[2],
        ]))
        .multiply(cone)
        .validate_composite("Bradford adaptation")
}

fn normalize_forward(matrix: Matrix3) -> Result<Matrix3, StrictMotionCamColorError> {
    matrix.validate_input("raw ForwardMatrix")?;
    let row_sums = matrix.apply([1.0; 3]);
    if row_sums
        .iter()
        .any(|value| !value.is_finite() || value.abs() <= MIN_FORWARD_NORMALIZATION_ROW_SUM)
    {
        return Err(StrictMotionCamColorError::UnreasonableMatrix {
            stage: "ForwardMatrix normalization row sum",
            maximum: row_sums.into_iter().map(f64::abs).fold(0.0, f64::max),
        });
    }
    let pcs = xy_to_xyz(D50_XY)?;
    diagonal([
        pcs[0] / row_sums[0],
        pcs[1] / row_sums[1],
        pcs[2] / row_sums[2],
    ])
    .multiply(matrix)
    .validate_input("normalized ForwardMatrix")
}

fn interpolate_for_xy(
    slots: &[ValidatedSlot],
    xy: [f64; 2],
) -> Result<InterpolatedFacts, StrictMotionCamColorError> {
    let (temperature, tint) = temperature_tint(xy)?;
    let weight_low = if slots.len() == 1 {
        1.0
    } else {
        interpolation_weight(temperature, slots[0].temperature, slots[1].temperature)?
    };
    let low = &slots[0];
    let high = &slots[slots.len() - 1];
    let facts = if slots.len() == 1 {
        InterpolatedFacts {
            temperature,
            tint,
            weight_low,
            color_matrix: low.color_matrix,
            camera_calibration: low.camera_calibration,
            forward_matrix: low.normalized_forward_matrix,
        }
    } else {
        InterpolatedFacts {
            temperature,
            tint,
            weight_low,
            color_matrix: lerp(low.color_matrix, high.color_matrix, weight_low),
            camera_calibration: lerp(low.camera_calibration, high.camera_calibration, weight_low),
            forward_matrix: low
                .normalized_forward_matrix
                .zip(high.normalized_forward_matrix)
                .map(|(low, high)| lerp(low, high, weight_low)),
        }
    };
    facts
        .color_matrix
        .validate_input("interpolated ColorMatrix")?;
    facts
        .camera_calibration
        .validate_input("interpolated CameraCalibration")?;
    if let Some(matrix) = facts.forward_matrix {
        matrix.validate_input("interpolated ForwardMatrix")?;
    }
    Ok(facts)
}

fn interpolation_weight(
    temperature: f64,
    low_temperature: f64,
    high_temperature: f64,
) -> Result<f64, StrictMotionCamColorError> {
    if !temperature.is_finite()
        || !low_temperature.is_finite()
        || !high_temperature.is_finite()
        || low_temperature >= high_temperature
    {
        return Err(StrictMotionCamColorError::DuplicateIlluminants);
    }
    if temperature <= low_temperature {
        return Ok(1.0);
    }
    if temperature >= high_temperature {
        return Ok(0.0);
    }
    Ok(((1.0 / temperature) - (1.0 / high_temperature))
        / ((1.0 / low_temperature) - (1.0 / high_temperature)))
}

fn xy_to_xyz(xy: [f64; 2]) -> Result<[f64; 3], StrictMotionCamColorError> {
    let [x, y] = xy;
    if !x.is_finite() || !y.is_finite() || x <= 0.0 || y <= 0.0 || x + y >= 1.0 {
        return Err(StrictMotionCamColorError::NumericallyInvalid { stage: "white xy" });
    }
    Ok([x / y, 1.0, (1.0 - x - y) / y])
}

fn xyz_to_xy(xyz: [f64; 3]) -> Result<[f64; 2], StrictMotionCamColorError> {
    let total = python_sum3(xyz);
    if xyz.iter().any(|value| !value.is_finite()) || total <= 0.0 {
        return Err(StrictMotionCamColorError::NumericallyInvalid { stage: "white XYZ" });
    }
    Ok([xyz[0] / total, xyz[1] / total])
}

fn temperature_tint(xy: [f64; 2]) -> Result<(f64, f64), StrictMotionCamColorError> {
    let [x, y] = xy;
    let denominator = 1.5 - x + 6.0 * y;
    if !x.is_finite() || !y.is_finite() || denominator == 0.0 {
        return Err(StrictMotionCamColorError::NumericallyInvalid {
            stage: "temperature/tint input",
        });
    }
    let u = 2.0 * x / denominator;
    let v = 3.0 * y / denominator;
    let mut last_dt = 0.0;
    let mut last_du = 0.0;
    let mut last_dv = 0.0;
    for index in 1..TEMPERATURE_TABLE.len() {
        let [_, table_u, table_v, slope] = TEMPERATURE_TABLE[index];
        let length = correctly_rounded_hypot2(1.0, slope);
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
            if !reciprocal.is_finite() || reciprocal <= 0.0 {
                return Err(StrictMotionCamColorError::NumericallyInvalid {
                    stage: "correlated color temperature",
                });
            }
            let temperature = 1.0e6 / reciprocal;
            let line_u = TEMPERATURE_TABLE[index - 1][1] * factor + table_u * (1.0 - factor);
            let line_v = TEMPERATURE_TABLE[index - 1][2] * factor + table_v * (1.0 - factor);
            let uu = u - line_u;
            let vv = v - line_v;
            du = du * (1.0 - factor) + last_du * factor;
            dv = dv * (1.0 - factor) + last_dv * factor;
            let length = correctly_rounded_hypot2(du, dv);
            du /= length;
            dv /= length;
            let tint = (uu * du + vv * dv) * -3000.0;
            if !temperature.is_finite() || !tint.is_finite() {
                return Err(StrictMotionCamColorError::NumericallyInvalid {
                    stage: "temperature/tint result",
                });
            }
            return Ok((temperature, tint));
        }
        last_dt = dt;
        last_du = du;
        last_dv = dv;
    }
    Err(StrictMotionCamColorError::NumericallyInvalid {
        stage: "temperature table traversal",
    })
}

/// Double-double refinement of sqrt(a*a + b*b). This reproduces the
/// correctly-rounded two-argument math.hypot used by the frozen Python 3.14
/// authority without FFI.
fn correctly_rounded_hypot2(a: f64, b: f64) -> f64 {
    let a = a.abs();
    let b = b.abs();
    let product_a = a * a;
    let product_a_error = a.mul_add(a, -product_a);
    let product_b = b * b;
    let product_b_error = b.mul_add(b, -product_b);
    let sum = product_a + product_b;
    let virtual_b = sum - product_a;
    let sum_error = (product_a - (sum - virtual_b)) + (product_b - virtual_b);
    let low = (sum_error + product_a_error) + product_b_error;
    let root = sum.sqrt();
    let root_square = root * root;
    let root_square_error = root.mul_add(root, -root_square);
    let residual = (sum - root_square) + (low - root_square_error);
    root + residual / (2.0 * root)
}

#[cfg(test)]
mod tests;
