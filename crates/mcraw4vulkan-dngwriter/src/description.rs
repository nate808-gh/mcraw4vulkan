use mcraw4vulkan_core::{BayerPattern, FrameDimensions, FrameNumber};
use mcraw4vulkan_mcrawcontainer::{
    BlackLevel, ColorMatrix, ContainerMetadata, FrameMetadata, SensorArrangement, WhiteLevel,
};
use thiserror::Error;

// Default DNG metadata values used by the original MotionCam Decoder.
const DEFAULT_CALIBRATION_ILLUMINANT1: u16 = 21;
const DEFAULT_CALIBRATION_ILLUMINANT2: u16 = 17;
const DEFAULT_DNG_VERSION: [u8; 4] = [1, 4, 0, 0];
const DEFAULT_DNG_BACKWARD_VERSION: [u8; 4] = [1, 1, 0, 0];
const DEFAULT_UNIQUE_CAMERA_MODEL: &str = "MotionCam";
const DYNAMIC_WHITE_INTEGER_EPSILON: f64 = 0.000_001;
const MOTIONCAM_DNG_SAMPLE_LEFT_SHIFT: u8 = 2;

// Errors from converting typed .mcraw metadata into DNG-facing metadata.
#[derive(Debug, Error)]
pub enum DngDescriptionError {
    #[error("unsupported DNG metadata: {0}")]
    UnsupportedMetadata(String),
}

// Neutral DNG-ready description for one decoded frame.
//
// This struct is the adapter layer between typed .mcraw metadata and the binary
// DNG writer. It does not own .mcraw parsing, raw decoding, or post-decode
// correction policy.
#[derive(Debug, Clone)]
pub struct DngFrameDescription {
    pub frame_number: FrameNumber,
    pub timestamp_us: u64,
    pub dimensions: FrameDimensions,
    pub default_crop_origin: [u32; 2],
    pub default_crop_size: [u32; 2],
    pub bits_per_sample: u16,
    pub samples_per_pixel: u16,
    pub compression: DngCompression,
    pub photometric_interpretation: DngPhotometricInterpretation,
    pub cfa_pattern: CfaPattern,
    pub black_level: BlackLevel,
    pub white_level: WhiteLevel,
    pub dng_sample_left_shift: u8,
    pub as_shot_neutral: Option<[f64; 3]>,
    pub color_matrix1: Option<ColorMatrix>,
    pub color_matrix2: Option<ColorMatrix>,
    pub forward_matrix1: Option<ColorMatrix>,
    pub forward_matrix2: Option<ColorMatrix>,
    pub camera_calibration1: Option<ColorMatrix>,
    pub camera_calibration2: Option<ColorMatrix>,
    pub analog_balance: Option<[f64; 3]>,
    pub calibration_illuminant1: u16,
    pub calibration_illuminant2: u16,
    pub dng_version: [u8; 4],
    pub dng_backward_version: [u8; 4],
    pub unique_camera_model: String,
}

// Optional caller-provided overrides for DNG output-range metadata.
//
// These values are already in the DNG output sample domain. They do not change
// how .mcraw metadata is parsed or how decoded pixels are produced.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DngFrameDescriptionOverrides {
    pub black_level: Option<[f64; 4]>,
    pub dng_sample_left_shift: Option<u8>,
}

impl DngFrameDescriptionOverrides {
    pub fn black_level_zero() -> Self {
        Self {
            black_level: Some([0.0; 4]),
            ..Self::default()
        }
    }

    pub fn corrected_pixel_domain() -> Self {
        Self {
            black_level: Some([0.0; 4]),
            dng_sample_left_shift: Some(0),
        }
    }
}

// The writer accepts only uncompressed CFA samples. Validation rejects
// mismatched sample layouts before any bytes are emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DngCompression {
    Uncompressed,
}

// DNG photometric interpretation for decoded MotionCam frames.
//
// MotionCam frames are Bayer CFA RAW frames, not rendered RGB images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DngPhotometricInterpretation {
    Cfa,
}

// DNG CFA pattern information for a 2x2 Bayer repeat pattern.
//
// DNG encodes CFA colors as plane indexes. The usual convention is:
// 0 = red, 1 = green, 2 = blue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CfaPattern {
    pub bayer_pattern: BayerPattern,
    pub repeat_pattern_dim: [u16; 2],
    pub cfa_plane_color: [u8; 3],
    pub pattern: [u8; 4],
}

impl DngFrameDescription {
    // Build a DNG-ready description from typed .mcraw metadata.
    //
    // Missing values that are required for a usable Bayer DNG fail here instead
    // of failing later inside the binary TIFF/DNG writer.
    pub fn from_metadata(
        container_metadata: &ContainerMetadata,
        frame_metadata: &FrameMetadata,
        frame_number: FrameNumber,
        timestamp_us: u64,
    ) -> Result<Self, DngDescriptionError> {
        let effective = container_metadata.with_frame_color(frame_metadata);
        let container_metadata = &effective;
        // Retain the accepted legacy ForwardMatrix writer policy. Profiles
        // without ForwardMatrices use source-associated DNG calibration facts.
        let source_calibration = container_metadata.forward_matrix1.is_none()
            && container_metadata.forward_matrix2.is_none()
            && (container_metadata.color_matrix1.is_some()
                || container_metadata.color_matrix2.is_some());
        let source_illuminant = |matrix: Option<ColorMatrix>,
                                 light: &Option<mcraw4vulkan_mcrawcontainer::ColorIlluminant>,
                                 default| {
            if !source_calibration {
                return Ok(default);
            }
            if matrix.is_none() {
                return Ok(0);
            }
            light.as_ref().and_then(|v| v.dng_code()).ok_or_else(|| {
                DngDescriptionError::UnsupportedMetadata(
                    "ColorMatrix requires its known source illuminant".to_owned(),
                )
            })
        };
        let calibration_illuminant1 = source_illuminant(
            container_metadata.color_matrix1,
            &container_metadata.color_illuminant1,
            DEFAULT_CALIBRATION_ILLUMINANT1,
        )?;
        let calibration_illuminant2 = source_illuminant(
            container_metadata.color_matrix2,
            &container_metadata.color_illuminant2,
            DEFAULT_CALIBRATION_ILLUMINANT2,
        )?;
        let container_black_level = container_metadata.black_level.ok_or_else(|| {
            DngDescriptionError::UnsupportedMetadata(
                "container metadata missing blackLevel".to_string(),
            )
        })?;

        let container_white_level = container_metadata.white_level.ok_or_else(|| {
            DngDescriptionError::UnsupportedMetadata(
                "container metadata missing whiteLevel".to_string(),
            )
        })?;
        let (white_level, dng_sample_left_shift) = derive_motioncam_dng_white_level_and_shift(
            frame_metadata.dynamic_white_level,
            container_white_level,
        );
        let black_level = derive_motioncam_dng_black_level(
            frame_metadata.dynamic_black_level,
            container_black_level,
            dng_sample_left_shift,
        );

        let cfa_pattern =
            CfaPattern::from_sensor_arrangement(&container_metadata.sensor_arrangement)?;

        Ok(Self {
            frame_number,
            timestamp_us,
            dimensions: frame_metadata.dimensions,
            default_crop_origin: [0, 0],
            default_crop_size: [
                frame_metadata.dimensions.width,
                frame_metadata.dimensions.height,
            ],
            bits_per_sample: 16,
            samples_per_pixel: 1,
            compression: DngCompression::Uncompressed,
            photometric_interpretation: DngPhotometricInterpretation::Cfa,
            cfa_pattern,
            black_level,
            white_level,
            dng_sample_left_shift,
            as_shot_neutral: frame_metadata.as_shot_neutral,
            color_matrix1: container_metadata.color_matrix1,
            color_matrix2: container_metadata.color_matrix2,
            forward_matrix1: container_metadata.forward_matrix1,
            forward_matrix2: container_metadata.forward_matrix2,
            camera_calibration1: source_calibration
                .then_some(container_metadata.calibration_matrix1)
                .flatten(),
            camera_calibration2: source_calibration
                .then_some(container_metadata.calibration_matrix2)
                .flatten(),
            analog_balance: source_calibration
                .then_some(container_metadata.analog_balance)
                .flatten(),
            calibration_illuminant1,
            calibration_illuminant2,
            dng_version: DEFAULT_DNG_VERSION,
            dng_backward_version: DEFAULT_DNG_BACKWARD_VERSION,
            unique_camera_model: DEFAULT_UNIQUE_CAMERA_MODEL.to_string(),
        })
    }

    pub fn with_output_overrides(&self, overrides: DngFrameDescriptionOverrides) -> Self {
        let mut description = self.clone();
        description.apply_output_overrides(overrides);
        description
    }

    pub fn apply_output_overrides(&mut self, overrides: DngFrameDescriptionOverrides) {
        if let Some(black_level) = overrides.black_level {
            self.black_level = BlackLevel {
                values: black_level,
            };
        }
        if let Some(dng_sample_left_shift) = overrides.dng_sample_left_shift {
            self.dng_sample_left_shift = dng_sample_left_shift;
        }
    }
}

// Derive DNG output range metadata from per-frame source dynamic range when
// available.
//
// MotionCam Android writes 16-bit DNGs whose WhiteLevel is the source white code
// shifted left by two bits with the low bits set. Model A expands the DNG samples
// by the same amount during serialization only. Model C also shifts the dynamic
// black level into the same output code range. If the per-frame white value is
// absent or not a safe integer-like source code, keep the older container
// metadata values and leave output samples unchanged.
fn derive_motioncam_dng_white_level_and_shift(
    dynamic_white_level: Option<f64>,
    fallback_white_level: WhiteLevel,
) -> (WhiteLevel, u8) {
    if let Some(white_level) = dynamic_white_level.and_then(motioncam_dng_white_level_from_dynamic)
    {
        (
            WhiteLevel {
                values: [f64::from(white_level); 4],
            },
            MOTIONCAM_DNG_SAMPLE_LEFT_SHIFT,
        )
    } else {
        (fallback_white_level, 0)
    }
}

fn motioncam_dng_white_level_from_dynamic(dynamic_white_level: f64) -> Option<u16> {
    let source_white_level = integer_like_u32(dynamic_white_level)?;
    let shifted = source_white_level.checked_mul(4)?.checked_add(3)?;

    u16::try_from(shifted).ok()
}

fn derive_motioncam_dng_black_level(
    dynamic_black_level: Option<[f64; 4]>,
    fallback_black_level: BlackLevel,
    dng_sample_left_shift: u8,
) -> BlackLevel {
    if dng_sample_left_shift == 0 {
        return fallback_black_level;
    }

    dynamic_black_level
        .and_then(|values| shifted_black_level(values, dng_sample_left_shift))
        .unwrap_or(fallback_black_level)
}

fn shifted_black_level(values: [f64; 4], dng_sample_left_shift: u8) -> Option<BlackLevel> {
    let multiplier = 1_u32.checked_shl(u32::from(dng_sample_left_shift))?;
    let multiplier = f64::from(multiplier);
    let mut shifted = [0.0; 4];

    for (index, value) in values.into_iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return None;
        }

        let shifted_value = value * multiplier;

        if !shifted_value.is_finite() || shifted_value < 0.0 || shifted_value > f64::from(u16::MAX)
        {
            return None;
        }

        shifted[index] = shifted_value;
    }

    Some(BlackLevel { values: shifted })
}

fn integer_like_u32(value: f64) -> Option<u32> {
    if !value.is_finite() || value < 0.0 || value > f64::from(u32::MAX) {
        return None;
    }

    let rounded = value.round();
    ((value - rounded).abs() <= DYNAMIC_WHITE_INTEGER_EPSILON).then_some(rounded as u32)
}

impl CfaPattern {
    // Convert MotionCam sensor arrangement metadata into DNG CFA pattern bytes.
    //
    // The pattern order is top-left, top-right, bottom-left, bottom-right for a
    // 2x2 Bayer repeat tile.
    pub fn from_sensor_arrangement(
        sensor_arrangement: &SensorArrangement,
    ) -> Result<Self, DngDescriptionError> {
        let (bayer_pattern, pattern) = match sensor_arrangement {
            SensorArrangement::Rggb => (BayerPattern::Rggb, [0, 1, 1, 2]),
            SensorArrangement::Bggr => (BayerPattern::Bggr, [2, 1, 1, 0]),
            SensorArrangement::Grbg => (BayerPattern::Grbg, [1, 0, 2, 1]),
            SensorArrangement::Gbrg => (BayerPattern::Gbrg, [1, 2, 0, 1]),
            SensorArrangement::Unknown(value) => {
                return Err(DngDescriptionError::UnsupportedMetadata(format!(
                    "unknown sensor arrangement: {value}"
                )));
            }
            SensorArrangement::Missing => {
                return Err(DngDescriptionError::UnsupportedMetadata(
                    "container metadata missing sensor arrangement".to_string(),
                ));
            }
        };

        Ok(Self {
            bayer_pattern,
            repeat_pattern_dim: [2, 2],
            cfa_plane_color: [0, 1, 2],
            pattern,
        })
    }
}
