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
            calibration_illuminant1: DEFAULT_CALIBRATION_ILLUMINANT1,
            calibration_illuminant2: DEFAULT_CALIBRATION_ILLUMINANT2,
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

#[cfg(test)]
mod tests {
    use mcraw4vulkan_core::{FrameDimensions, FrameNumber};
    use mcraw4vulkan_mcrawcontainer::{
        BlackLevel, CompressionType, ContainerMetadata, FrameMetadata, SensorArrangement,
        WhiteLevel,
    };

    use super::*;

    #[test]
    fn derives_motioncam_white_level_for_ten_bit_source() {
        let description = description_with_dynamic_range(Some(1023.0), Some([64.0; 4]));

        assert_eq!(description.default_crop_origin, [0, 0]);
        assert_eq!(description.default_crop_size, [3840, 2160]);
        assert_eq!(description.white_level.values, [4095.0; 4]);
        assert_eq!(description.dng_sample_left_shift, 2);
        assert_eq!(description.bits_per_sample, 16);
        assert_eq!(description.black_level.values, [256.0; 4]);
    }

    #[test]
    fn derives_motioncam_white_level_for_twelve_bit_source() {
        let description = description_with_dynamic_range(Some(4095.0), Some([256.0; 4]));

        assert_eq!(description.white_level.values, [16383.0; 4]);
        assert_eq!(description.dng_sample_left_shift, 2);
        assert_eq!(description.bits_per_sample, 16);
        assert_eq!(description.black_level.values, [1024.0; 4]);
    }

    #[test]
    fn missing_dynamic_white_level_falls_back_to_container_white_level() {
        let description = description_with_dynamic_range(None, Some([128.0; 4]));

        assert_eq!(description.white_level.values, [4095.0; 4]);
        assert_eq!(description.dng_sample_left_shift, 0);
        assert_eq!(description.black_level.values, [256.0; 4]);
    }

    #[test]
    fn invalid_dynamic_white_level_falls_back_to_container_white_level() {
        for dynamic_white_level in [-1.0, 1023.5, 16384.0] {
            let description =
                description_with_dynamic_range(Some(dynamic_white_level), Some([128.0; 4]));

            assert_eq!(description.white_level.values, [4095.0; 4]);
            assert_eq!(description.dng_sample_left_shift, 0);
            assert_eq!(description.black_level.values, [256.0; 4]);
        }
    }

    #[test]
    fn missing_dynamic_black_level_falls_back_to_container_black_level() {
        let description = description_with_dynamic_range(Some(4095.0), None);

        assert_eq!(description.white_level.values, [16383.0; 4]);
        assert_eq!(description.dng_sample_left_shift, 2);
        assert_eq!(description.black_level.values, [256.0; 4]);
    }

    #[test]
    fn invalid_dynamic_black_level_falls_back_to_container_black_level() {
        for dynamic_black_level in [
            [-1.0, 256.0, 256.0, 256.0],
            [f64::NAN, 256.0, 256.0, 256.0],
            [20000.0, 256.0, 256.0, 256.0],
        ] {
            let description =
                description_with_dynamic_range(Some(4095.0), Some(dynamic_black_level));

            assert_eq!(description.white_level.values, [16383.0; 4]);
            assert_eq!(description.dng_sample_left_shift, 2);
            assert_eq!(description.black_level.values, [256.0; 4]);
        }
    }

    #[test]
    fn empty_output_overrides_preserve_description() {
        let description = description_with_dynamic_range(Some(4095.0), Some([256.0; 4]));
        let overridden = description.with_output_overrides(DngFrameDescriptionOverrides::default());

        assert_same_description(&overridden, &description);
    }

    #[test]
    fn black_level_zero_override_changes_only_black_level() {
        let description = description_with_dynamic_range(Some(4095.0), Some([256.0; 4]));
        let overridden =
            description.with_output_overrides(DngFrameDescriptionOverrides::black_level_zero());

        assert_eq!(overridden.black_level.values, [0.0; 4]);
        assert_eq!(overridden.white_level, description.white_level);
        assert_eq!(
            overridden.dng_sample_left_shift,
            description.dng_sample_left_shift
        );
        assert_eq!(
            overridden.default_crop_origin,
            description.default_crop_origin
        );
        assert_eq!(overridden.default_crop_size, description.default_crop_size);
        assert_eq!(description.black_level.values, [1024.0; 4]);
    }

    #[test]
    fn corrected_pixel_domain_override_zeros_black_and_disables_sample_shift() {
        let description = description_with_dynamic_range(Some(1023.0), Some([64.0; 4]));
        assert_eq!(description.white_level.values, [4095.0; 4]);
        assert_eq!(description.dng_sample_left_shift, 2);

        let overridden = description
            .with_output_overrides(DngFrameDescriptionOverrides::corrected_pixel_domain());

        assert_eq!(overridden.black_level.values, [0.0; 4]);
        assert_eq!(overridden.white_level.values, [4095.0; 4]);
        assert_eq!(overridden.dng_sample_left_shift, 0);
    }

    #[test]
    fn apply_output_overrides_mutates_only_black_level() {
        let mut description = description_with_dynamic_range(Some(4095.0), Some([256.0; 4]));
        let original_white_level = description.white_level;
        let original_sample_left_shift = description.dng_sample_left_shift;

        description.apply_output_overrides(DngFrameDescriptionOverrides::black_level_zero());

        assert_eq!(description.black_level.values, [0.0; 4]);
        assert_eq!(description.white_level, original_white_level);
        assert_eq!(
            description.dng_sample_left_shift,
            original_sample_left_shift
        );
    }

    #[test]
    fn default_crop_uses_visible_dimensions_not_original_dimensions() {
        let description = description_with_dimensions_and_original(
            FrameDimensions {
                width: 3840,
                height: 2160,
            },
            Some(4080),
            Some(3072),
        );

        assert_eq!(description.default_crop_origin, [0, 0]);
        assert_eq!(description.default_crop_size, [3840, 2160]);
        assert_eq!(description.dimensions.width, 3840);
        assert_eq!(description.dimensions.height, 2160);
    }

    fn description_with_dynamic_range(
        dynamic_white_level: Option<f64>,
        dynamic_black_level: Option<[f64; 4]>,
    ) -> DngFrameDescription {
        description_with_metadata(
            FrameDimensions {
                width: 3840,
                height: 2160,
            },
            None,
            None,
            dynamic_white_level,
            dynamic_black_level,
        )
    }

    fn description_with_dimensions_and_original(
        dimensions: FrameDimensions,
        original_width: Option<u32>,
        original_height: Option<u32>,
    ) -> DngFrameDescription {
        description_with_metadata(
            dimensions,
            original_width,
            original_height,
            Some(4095.0),
            None,
        )
    }

    fn description_with_metadata(
        dimensions: FrameDimensions,
        original_width: Option<u32>,
        original_height: Option<u32>,
        dynamic_white_level: Option<f64>,
        dynamic_black_level: Option<[f64; 4]>,
    ) -> DngFrameDescription {
        let container_metadata = ContainerMetadata {
            black_level: Some(BlackLevel { values: [256.0; 4] }),
            white_level: Some(WhiteLevel {
                values: [4095.0; 4],
            }),
            sensor_arrangement: SensorArrangement::Gbrg,
            sensor_orientation: None,
            unique_camera_model: None,
            device_specific_profile: None,
            focal_lengths: None,
            apertures: None,
            color_matrix1: None,
            color_matrix2: None,
            forward_matrix1: None,
            forward_matrix2: None,
            color_illuminant1: None,
            color_illuminant2: None,
            calibration_matrix1: None,
            calibration_matrix2: None,
        };
        let frame_metadata = FrameMetadata {
            dimensions,
            original_width,
            original_height,
            row_stride: None,
            pixel_format: None,
            compression_type: CompressionType::Missing,
            compression_type_raw: None,
            is_binned: None,
            is_compressed: None,
            need_remosaic: None,
            dynamic_white_level,
            dynamic_black_level,
            lens_shading_map: None,
            as_shot_neutral: None,
        };

        DngFrameDescription::from_metadata(&container_metadata, &frame_metadata, FrameNumber(0), 0)
            .expect("valid test metadata should build a DNG description")
    }

    fn assert_same_description(left: &DngFrameDescription, right: &DngFrameDescription) {
        assert_eq!(left.frame_number, right.frame_number);
        assert_eq!(left.timestamp_us, right.timestamp_us);
        assert_eq!(left.dimensions, right.dimensions);
        assert_eq!(left.default_crop_origin, right.default_crop_origin);
        assert_eq!(left.default_crop_size, right.default_crop_size);
        assert_eq!(left.bits_per_sample, right.bits_per_sample);
        assert_eq!(left.samples_per_pixel, right.samples_per_pixel);
        assert_eq!(left.compression, right.compression);
        assert_eq!(
            left.photometric_interpretation,
            right.photometric_interpretation
        );
        assert_eq!(left.cfa_pattern, right.cfa_pattern);
        assert_eq!(left.black_level, right.black_level);
        assert_eq!(left.white_level, right.white_level);
        assert_eq!(left.dng_sample_left_shift, right.dng_sample_left_shift);
        assert_eq!(left.as_shot_neutral, right.as_shot_neutral);
        assert_eq!(left.color_matrix1, right.color_matrix1);
        assert_eq!(left.color_matrix2, right.color_matrix2);
        assert_eq!(left.forward_matrix1, right.forward_matrix1);
        assert_eq!(left.forward_matrix2, right.forward_matrix2);
        assert_eq!(left.calibration_illuminant1, right.calibration_illuminant1);
        assert_eq!(left.calibration_illuminant2, right.calibration_illuminant2);
        assert_eq!(left.dng_version, right.dng_version);
        assert_eq!(left.dng_backward_version, right.dng_backward_version);
        assert_eq!(left.unique_camera_model, right.unique_camera_model);
    }
}
