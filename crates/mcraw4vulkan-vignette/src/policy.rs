use mcraw4vulkan_core::BayerPattern;

use crate::BAYER_CFA_PLANE_COUNT;

pub const VIGNETTE_GAIN_FRACTIONAL_BITS: u32 = 16;
pub const VIGNETTE_GAIN_SCALE: i64 = 1_i64 << VIGNETTE_GAIN_FRACTIONAL_BITS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedPointVignettePolicy {
    pub fractional_bits: u32,
    pub gain_scale: i64,
}

impl Default for FixedPointVignettePolicy {
    fn default() -> Self {
        Self {
            fractional_bits: VIGNETTE_GAIN_FRACTIONAL_BITS,
            gain_scale: VIGNETTE_GAIN_SCALE,
        }
    }
}

pub const MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH: f32 = 0.977_092;
pub const MOTIONCAM_PIXEL_SAMPLE_LIMIT_MULTIPLIER: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VignetteCorrectionPolicy {
    #[default]
    MotionCamCompatiblePixelDomainV1,
    LumaPlane0,
}

impl VignetteCorrectionPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::MotionCamCompatiblePixelDomainV1 => "motioncam-compatible-pixel-v1",
            Self::LumaPlane0 => "luma-plane-0",
        }
    }

    pub fn from_label_for_private_cli(value: &str) -> Option<Self> {
        match value {
            "motioncam-compatible-pixel-v1" | "motioncam-compatible" => {
                Some(Self::MotionCamCompatiblePixelDomainV1)
            }
            "luma-plane-0" | "motioncam-compatible-luma" => Some(Self::LumaPlane0),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VignettePixelDomainFacts {
    pub source_black_storage: [f32; BAYER_CFA_PLANE_COUNT],
    pub source_white_storage: u16,
    pub corrected_white_tag: u16,
    pub source_to_corrected_scale: f32,
    pub sample_limit: u16,
}

impl VignettePixelDomainFacts {
    pub fn from_source_levels(
        source_black_storage: [f32; BAYER_CFA_PLANE_COUNT],
        source_white_storage: u16,
    ) -> Self {
        let corrected_white_tag = corrected_white_tag_for_source_white(source_white_storage);
        let source_black_average = source_black_storage.iter().sum::<f32>() * 0.25;
        let source_range = (f32::from(source_white_storage) - source_black_average).max(1.0);
        let sample_limit = sample_limit_for_corrected_white_tag(corrected_white_tag);

        Self {
            source_black_storage,
            source_white_storage,
            corrected_white_tag,
            source_to_corrected_scale: f32::from(corrected_white_tag) / source_range,
            sample_limit,
        }
    }
}

pub fn corrected_white_tag_for_source_white(source_white: u16) -> u16 {
    u32::from(source_white)
        .saturating_mul(4)
        .saturating_add(3)
        .min(u32::from(u16::MAX)) as u16
}

pub fn sample_limit_for_corrected_white_tag(corrected_white_tag: u16) -> u16 {
    u32::from(corrected_white_tag)
        .saturating_mul(MOTIONCAM_PIXEL_SAMPLE_LIMIT_MULTIPLIER)
        .saturating_add(MOTIONCAM_PIXEL_SAMPLE_LIMIT_MULTIPLIER - 1)
        .min(u32::from(u16::MAX)) as u16
}

// The mode travels with shared correction facts so CPU and GPU paths agree on
// passthrough versus correction and the resulting output black level.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VignetteCorrectionMode {
    Disabled,
    #[default]
    Enabled,
}

// Coordinate mapping from output pixels into the lens shading grid.
//
// VisibleFrame maps the first and last visible pixels to the first and last map
// samples. Keeping this explicit gives CPU and GPU interpolation the same
// coordinate endpoints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VignetteCoordinateMapping {
    #[default]
    VisibleFrame,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VignetteCorrectionOptions {
    pub mode: VignetteCorrectionMode,
    pub correction_policy: VignetteCorrectionPolicy,
    pub coordinate_mapping: VignetteCoordinateMapping,
    pub bayer_pattern: BayerPattern,
    pub input_black_level: [f32; BAYER_CFA_PLANE_COUNT],
    pub output_white_level: u16,
}

impl VignetteCorrectionOptions {
    pub fn disabled(
        bayer_pattern: BayerPattern,
        input_black_level: [f32; BAYER_CFA_PLANE_COUNT],
        output_white_level: u16,
    ) -> Self {
        Self {
            mode: VignetteCorrectionMode::Disabled,
            correction_policy: VignetteCorrectionPolicy::default(),
            coordinate_mapping: VignetteCoordinateMapping::VisibleFrame,
            bayer_pattern,
            input_black_level,
            output_white_level,
        }
    }

    pub fn enabled(
        bayer_pattern: BayerPattern,
        input_black_level: [f32; BAYER_CFA_PLANE_COUNT],
        output_white_level: u16,
    ) -> Self {
        Self {
            mode: VignetteCorrectionMode::Enabled,
            correction_policy: VignetteCorrectionPolicy::default(),
            coordinate_mapping: VignetteCoordinateMapping::VisibleFrame,
            bayer_pattern,
            input_black_level,
            output_white_level,
        }
    }

    pub fn enabled_with_policy(
        bayer_pattern: BayerPattern,
        input_black_level: [f32; BAYER_CFA_PLANE_COUNT],
        output_white_level: u16,
        correction_policy: VignetteCorrectionPolicy,
    ) -> Self {
        Self {
            mode: VignetteCorrectionMode::Enabled,
            correction_policy,
            coordinate_mapping: VignetteCoordinateMapping::VisibleFrame,
            bayer_pattern,
            input_black_level,
            output_white_level,
        }
    }

    pub fn with_policy(mut self, correction_policy: VignetteCorrectionPolicy) -> Self {
        self.correction_policy = correction_policy;
        self
    }

    pub fn output_black_level(self) -> [f32; BAYER_CFA_PLANE_COUNT] {
        match self.mode {
            VignetteCorrectionMode::Disabled => self.input_black_level,
            VignetteCorrectionMode::Enabled => [0.0; BAYER_CFA_PLANE_COUNT],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        VignetteCorrectionPolicy, VignettePixelDomainFacts, corrected_white_tag_for_source_white,
        sample_limit_for_corrected_white_tag,
    };

    #[test]
    fn default_policy_is_motioncam_compatible_pixel_domain_v1() {
        assert_eq!(
            VignetteCorrectionPolicy::default(),
            VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1
        );
    }

    #[test]
    fn source_domain_facts_expand_10bit_source_to_corrected_domain() {
        let facts = VignettePixelDomainFacts::from_source_levels([64.0; 4], 1023);

        assert_eq!(facts.source_black_storage, [64.0; 4]);
        assert_eq!(facts.source_white_storage, 1023);
        assert_eq!(facts.corrected_white_tag, 4095);
        assert_eq!(facts.sample_limit, 16383);
        assert!((facts.source_to_corrected_scale - (4095.0 / (1023.0 - 64.0))).abs() < 0.0001);
    }

    #[test]
    fn corrected_white_and_sample_limit_saturate_safely() {
        assert_eq!(corrected_white_tag_for_source_white(1023), 4095);
        assert_eq!(sample_limit_for_corrected_white_tag(4095), 16383);
        assert_eq!(corrected_white_tag_for_source_white(4095), 16383);
        assert_eq!(sample_limit_for_corrected_white_tag(16383), 65535);
        assert_eq!(corrected_white_tag_for_source_white(u16::MAX), u16::MAX);
        assert_eq!(sample_limit_for_corrected_white_tag(u16::MAX), u16::MAX);
    }
}
