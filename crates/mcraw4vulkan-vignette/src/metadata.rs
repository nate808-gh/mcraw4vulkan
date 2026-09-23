use mcraw4vulkan_core::{BayerPattern, FrameDimensions};
use mcraw4vulkan_mcrawcontainer::{ContainerMetadata, FrameMetadata};

use crate::lens_map::quantize_nonnegative_to_fixed;
use crate::math::validate_black_level;
use crate::{
    BAYER_CFA_PLANE_COUNT, PipeF32BayerCorrectionMode, PreparedFixedLensShadingMap,
    PreparedLensShadingMap, VignetteCoordinateMapping, VignetteCorrectionError,
    VignetteCorrectionMode, VignetteCorrectionOptions, VignetteCorrectionPolicy,
    VignettePixelDomainFacts,
};

/// Build the immutable MotionCam correction facts consumed by the PIPE f32
/// Bayer preparation stage for one indexed frame.
///
/// This resolves only typed source metadata. It may quantize a present lens
/// map, but it never reads or decodes Bayer payload bytes.
pub fn motioncam_pipe_f32_bayer_facts<'a>(
    container: &ContainerMetadata,
    frame: &'a FrameMetadata,
    bayer_pattern: BayerPattern,
    mode: PipeF32BayerCorrectionMode,
) -> Result<FixedPointVignetteInputFacts<'a>, VignetteCorrectionError> {
    let black = if let Some(values) = frame.dynamic_black_level {
        values.map(|value| value as f32)
    } else if let Some(level) = container.black_level {
        level.values.map(|value| value as f32)
    } else {
        return Err(VignetteCorrectionError::MissingSourceBlackLevel);
    };
    let white_f64 = if let Some(value) = frame.dynamic_white_level {
        value
    } else if let Some(level) = container.white_level {
        if !level
            .values
            .iter()
            .all(|value| (*value - level.values[0]).abs() <= 1e-6)
        {
            return Err(VignetteCorrectionError::NonuniformSourceWhiteLevel);
        }
        level.values[0]
    } else {
        return Err(VignetteCorrectionError::MissingSourceWhiteLevel);
    };
    if !white_f64.is_finite() || !(1.0..=f64::from(u16::MAX)).contains(&white_f64) {
        return Err(VignetteCorrectionError::InvalidSourceWhiteLevel { value: white_f64 });
    }
    let rounded_white = white_f64.round();
    if (rounded_white - white_f64).abs() > 1e-6 {
        return Err(VignetteCorrectionError::NonIntegerSourceWhiteLevel { value: white_f64 });
    }
    let white = rounded_white as u16;
    let input = VignetteCorrectionInputFacts::new(
        VignetteCorrectionMode::Enabled,
        VignetteCoordinateMapping::VisibleFrame,
        frame.dimensions,
        bayer_pattern,
        None,
        black,
        white,
    )?;
    let fixed_map = match mode {
        PipeF32BayerCorrectionMode::IdentitySpatialGain => None,
        PipeF32BayerCorrectionMode::MotionCamSpatial => {
            Some(PreparedFixedLensShadingMap::from_typed_map(
                frame
                    .lens_shading_map
                    .as_ref()
                    .ok_or(VignetteCorrectionError::MotionCamSpatialMissingLensShadingMap)?,
            )?)
        }
    };
    FixedPointVignetteInputFacts::from_input_facts_with_fixed_map(&input, fixed_map)
}

#[derive(Debug, Clone, Copy)]
pub struct VignetteCorrectionInputFacts<'a> {
    pub mode: VignetteCorrectionMode,
    pub correction_policy: VignetteCorrectionPolicy,
    pub coordinate_mapping: VignetteCoordinateMapping,
    pub frame_dimensions: FrameDimensions,
    pub bayer_pattern: BayerPattern,
    pub lens_shading_map: Option<&'a PreparedLensShadingMap<'a>>,
    pub input_black_level: [f32; BAYER_CFA_PLANE_COUNT],
    pub output_white_level: u16,
    pub pixel_domain: VignettePixelDomainFacts,
}

#[derive(Debug, Clone, Copy)]
pub struct VignetteCorrectionInputConfig<'a> {
    pub mode: VignetteCorrectionMode,
    pub correction_policy: VignetteCorrectionPolicy,
    pub coordinate_mapping: VignetteCoordinateMapping,
    pub frame_dimensions: FrameDimensions,
    pub bayer_pattern: BayerPattern,
    pub lens_shading_map: Option<&'a PreparedLensShadingMap<'a>>,
    pub input_black_level: [f32; BAYER_CFA_PLANE_COUNT],
    pub output_white_level: u16,
}

impl<'a> VignetteCorrectionInputFacts<'a> {
    pub fn new(
        mode: VignetteCorrectionMode,
        coordinate_mapping: VignetteCoordinateMapping,
        frame_dimensions: FrameDimensions,
        bayer_pattern: BayerPattern,
        lens_shading_map: Option<&'a PreparedLensShadingMap<'a>>,
        input_black_level: [f32; BAYER_CFA_PLANE_COUNT],
        output_white_level: u16,
    ) -> Result<Self, VignetteCorrectionError> {
        Self::new_with_policy(VignetteCorrectionInputConfig {
            mode,
            correction_policy: VignetteCorrectionPolicy::default(),
            coordinate_mapping,
            frame_dimensions,
            bayer_pattern,
            lens_shading_map,
            input_black_level,
            output_white_level,
        })
    }

    pub fn new_with_policy(
        config: VignetteCorrectionInputConfig<'a>,
    ) -> Result<Self, VignetteCorrectionError> {
        let VignetteCorrectionInputConfig {
            mode,
            correction_policy,
            coordinate_mapping,
            frame_dimensions,
            bayer_pattern,
            lens_shading_map,
            input_black_level,
            output_white_level,
        } = config;

        validate_black_level(input_black_level)?;

        let pixel_domain =
            VignettePixelDomainFacts::from_source_levels(input_black_level, output_white_level);

        Ok(Self {
            mode,
            correction_policy,
            coordinate_mapping,
            frame_dimensions,
            bayer_pattern,
            lens_shading_map,
            input_black_level,
            output_white_level,
            pixel_domain,
        })
    }

    pub fn from_options(
        frame_dimensions: FrameDimensions,
        lens_shading_map: Option<&'a PreparedLensShadingMap<'a>>,
        options: VignetteCorrectionOptions,
    ) -> Result<Self, VignetteCorrectionError> {
        Self::new_with_policy(VignetteCorrectionInputConfig {
            mode: options.mode,
            correction_policy: options.correction_policy,
            coordinate_mapping: options.coordinate_mapping,
            frame_dimensions,
            bayer_pattern: options.bayer_pattern,
            lens_shading_map,
            input_black_level: options.input_black_level,
            output_white_level: options.output_white_level,
        })
    }

    pub fn output_black_level(&self) -> [f32; BAYER_CFA_PLANE_COUNT] {
        match self.mode {
            VignetteCorrectionMode::Enabled => [0.0; BAYER_CFA_PLANE_COUNT],
            VignetteCorrectionMode::Disabled => self.input_black_level,
        }
    }
}

// Fixed facts own quantized map and black-level values, making them the common
// rounding inputs for CPU and GPU correction.
#[derive(Debug, Clone)]
pub struct FixedPointVignetteInputFacts<'a> {
    pub mode: VignetteCorrectionMode,
    pub correction_policy: VignetteCorrectionPolicy,
    pub coordinate_mapping: VignetteCoordinateMapping,
    pub frame_dimensions: FrameDimensions,
    pub bayer_pattern: BayerPattern,
    pub lens_shading_map: Option<PreparedFixedLensShadingMap<'a>>,
    pub input_black_level: [f32; BAYER_CFA_PLANE_COUNT],
    pub input_black_level_q: [i64; BAYER_CFA_PLANE_COUNT],
    pub output_white_level: u16,
    pub pixel_domain: VignettePixelDomainFacts,
}

impl<'a> FixedPointVignetteInputFacts<'a> {
    pub fn from_input_facts(
        facts: &VignetteCorrectionInputFacts<'a>,
    ) -> Result<Self, VignetteCorrectionError> {
        let lens_shading_map = facts
            .lens_shading_map
            .map(|lens_shading_map| {
                PreparedFixedLensShadingMap::from_prepared_map_with_policy(
                    *lens_shading_map,
                    facts.correction_policy,
                )
            })
            .transpose()?;

        Self::from_input_facts_with_fixed_map(facts, lens_shading_map)
    }

    pub fn from_input_facts_with_fixed_map(
        facts: &VignetteCorrectionInputFacts<'_>,
        lens_shading_map: Option<PreparedFixedLensShadingMap<'a>>,
    ) -> Result<Self, VignetteCorrectionError> {
        let mut input_black_level_q = [0_i64; BAYER_CFA_PLANE_COUNT];
        for (index, value) in facts.input_black_level.into_iter().enumerate() {
            input_black_level_q[index] = quantize_nonnegative_to_fixed(value)
                .ok_or(VignetteCorrectionError::FixedPointBlackLevelOverflow { index, value })?;
        }

        Ok(Self {
            mode: facts.mode,
            correction_policy: facts.correction_policy,
            coordinate_mapping: facts.coordinate_mapping,
            frame_dimensions: facts.frame_dimensions,
            bayer_pattern: facts.bayer_pattern,
            lens_shading_map,
            input_black_level: facts.input_black_level,
            input_black_level_q,
            output_white_level: facts.output_white_level,
            pixel_domain: facts.pixel_domain,
        })
    }

    pub fn output_black_level(&self) -> [f32; BAYER_CFA_PLANE_COUNT] {
        match self.mode {
            VignetteCorrectionMode::Enabled => [0.0; BAYER_CFA_PLANE_COUNT],
            VignetteCorrectionMode::Disabled => self.input_black_level,
        }
    }
}
