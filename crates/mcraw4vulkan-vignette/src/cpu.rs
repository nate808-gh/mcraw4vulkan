use mcraw4vulkan_core::DecodedBayerU16Frame;

use crate::math::{
    android_rggb_source_plane_index, cfa_position_plane_index, corrected_fixed_sample,
    corrected_sample, interpolated_fixed_gain, interpolated_prepared_gain,
};
use crate::{
    BAYER_CFA_PLANE_COUNT, BYTES_PER_U16_SAMPLE, FixedPointVignetteInputFacts, LensShadingMap,
    MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH, PreparedLensShadingMap, VIGNETTE_GAIN_SCALE,
    VignetteCorrectedFrameInfo, VignetteCorrectionError, VignetteCorrectionInputFacts,
    VignetteCorrectionMode, VignetteCorrectionOptions, VignetteCorrectionPolicy,
    VignetteCorrectionStats, lens_shading_map_for_policy,
};

#[derive(Debug, Clone, PartialEq)]
pub struct CpuVignetteCorrectionResult<'a> {
    pub frame: DecodedBayerU16Frame<'a>,
    pub output_black_level: [f32; BAYER_CFA_PLANE_COUNT],
    pub output_white_level: u16,
    pub applied: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct OptionalCpuVignetteCorrection<'facts, 'map> {
    pub facts: &'facts FixedPointVignetteInputFacts<'map>,
}

impl<'facts, 'map> OptionalCpuVignetteCorrection<'facts, 'map> {
    pub fn new(facts: &'facts FixedPointVignetteInputFacts<'map>) -> Self {
        Self { facts }
    }
}

// Passthrough borrows the original frame without touching output storage;
// corrected output instead borrows the caller-owned byte buffer.
#[derive(Debug, Clone, PartialEq)]
pub enum CpuPostprocessedFrame<'original, 'frame_bytes, 'corrected_bytes> {
    Original(&'original DecodedBayerU16Frame<'frame_bytes>),
    Corrected(DecodedBayerU16Frame<'corrected_bytes>),
}

impl<'original, 'frame_bytes, 'corrected_bytes>
    CpuPostprocessedFrame<'original, 'frame_bytes, 'corrected_bytes>
{
    pub fn dimensions(&self) -> mcraw4vulkan_core::FrameDimensions {
        match self {
            Self::Original(frame) => frame.dimensions(),
            Self::Corrected(frame) => frame.dimensions(),
        }
    }

    pub fn pixel_bytes_le(&self) -> &[u8] {
        match self {
            Self::Original(frame) => frame.pixel_bytes_le(),
            Self::Corrected(frame) => frame.pixel_bytes_le(),
        }
    }

    pub fn is_original(&self) -> bool {
        matches!(self, Self::Original(_))
    }

    pub fn is_corrected(&self) -> bool {
        matches!(self, Self::Corrected(_))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuVignettePostprocessResult<'original, 'frame_bytes, 'corrected_bytes> {
    pub frame: CpuPostprocessedFrame<'original, 'frame_bytes, 'corrected_bytes>,
    pub info: Option<VignetteCorrectedFrameInfo>,
    pub stats: VignetteCorrectionStats,
}

impl<'original, 'frame_bytes, 'corrected_bytes>
    CpuVignettePostprocessResult<'original, 'frame_bytes, 'corrected_bytes>
{
    pub fn applied(&self) -> bool {
        self.info.map(|info| info.applied).unwrap_or(false)
    }

    pub fn dimensions(&self) -> mcraw4vulkan_core::FrameDimensions {
        self.frame.dimensions()
    }

    pub fn pixel_bytes_le(&self) -> &[u8] {
        self.frame.pixel_bytes_le()
    }

    pub fn output_black_level(&self) -> Option<[f32; BAYER_CFA_PLANE_COUNT]> {
        self.info.map(|info| info.output_black_level)
    }

    pub fn output_white_level(&self) -> Option<u16> {
        self.info.map(|info| info.output_white_level)
    }
}

// Owned-return calls clone the reusable scratch so their frames outlive the
// corrector; borrowed-return calls expose only caller-owned output storage.
#[derive(Debug, Default)]
pub struct CpuFixedPointVignetteCorrector {
    output: Vec<u8>,
}

impl CpuFixedPointVignetteCorrector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn correct_fixed_to_owned(
        &mut self,
        frame: &DecodedBayerU16Frame<'_>,
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<CpuVignetteCorrectionResult<'static>, VignetteCorrectionError> {
        self.output.clear();
        correct_fixed_into_le_bytes(frame, facts, &mut self.output)?;
        let output = self.output.clone();
        let frame = DecodedBayerU16Frame::from_owned_le_bytes(frame.dimensions(), output).map_err(
            |error| VignetteCorrectionError::DecodedFrameValidationFailed(error.to_string()),
        )?;

        Ok(CpuVignetteCorrectionResult {
            frame,
            output_black_level: facts.output_black_level(),
            output_white_level: facts.output_white_level,
            applied: facts.mode == VignetteCorrectionMode::Enabled,
        })
    }

    pub fn correct_fixed_into<'a>(
        &mut self,
        frame: &DecodedBayerU16Frame<'_>,
        facts: &FixedPointVignetteInputFacts<'_>,
        output: &'a mut Vec<u8>,
    ) -> Result<CpuVignetteCorrectionResult<'a>, VignetteCorrectionError> {
        correct_fixed_into_le_bytes(frame, facts, output)?;
        let corrected_frame =
            DecodedBayerU16Frame::from_borrowed_le_bytes(frame.dimensions(), output.as_slice())
                .map_err(|error| {
                    VignetteCorrectionError::DecodedFrameValidationFailed(error.to_string())
                })?;

        Ok(CpuVignetteCorrectionResult {
            frame: corrected_frame,
            output_black_level: facts.output_black_level(),
            output_white_level: facts.output_white_level,
            applied: facts.mode == VignetteCorrectionMode::Enabled,
        })
    }

    pub fn postprocess_optional<'original, 'frame_bytes, 'corrected_bytes, 'facts, 'map>(
        &mut self,
        frame: &'original DecodedBayerU16Frame<'frame_bytes>,
        correction: Option<OptionalCpuVignetteCorrection<'facts, 'map>>,
        output: &'corrected_bytes mut Vec<u8>,
    ) -> Result<
        CpuVignettePostprocessResult<'original, 'frame_bytes, 'corrected_bytes>,
        VignetteCorrectionError,
    > {
        let Some(correction) = correction else {
            return Ok(CpuVignettePostprocessResult {
                frame: CpuPostprocessedFrame::Original(frame),
                info: None,
                stats: VignetteCorrectionStats {
                    cpu_passthrough_without_copy: true,
                    ..VignetteCorrectionStats::default()
                },
            });
        };

        let facts = correction.facts;
        validate_fixed_frame_dimensions(frame.dimensions(), facts)?;

        let info = VignetteCorrectedFrameInfo::from_correction_mode(
            facts.mode,
            facts.input_black_level,
            facts.output_white_level,
        );

        if facts.mode == VignetteCorrectionMode::Disabled {
            return Ok(CpuVignettePostprocessResult {
                frame: CpuPostprocessedFrame::Original(frame),
                info: Some(info),
                stats: cpu_postprocess_stats(facts, true, 0)?,
            });
        }

        let corrected = self.correct_fixed_into(frame, facts, output)?;
        let output_buffer_bytes =
            u64::try_from(corrected.frame.pixel_bytes_le().len()).map_err(|_| {
                VignetteCorrectionError::FrameDimensionsOverflow {
                    dimensions: facts.frame_dimensions,
                }
            })?;

        Ok(CpuVignettePostprocessResult {
            frame: CpuPostprocessedFrame::Corrected(corrected.frame),
            info: Some(info),
            stats: cpu_postprocess_stats(facts, false, output_buffer_bytes)?,
        })
    }
}

pub fn apply_cpu_vignette_correction<'a>(
    frame: DecodedBayerU16Frame<'a>,
    lens_shading_map: Option<&LensShadingMap>,
    options: VignetteCorrectionOptions,
) -> Result<CpuVignetteCorrectionResult<'a>, VignetteCorrectionError> {
    let policy_lens_shading_map = if options.mode == VignetteCorrectionMode::Enabled {
        Some(lens_shading_map_for_policy(
            lens_shading_map.ok_or(VignetteCorrectionError::MissingLensShadingMap)?,
            options.correction_policy,
        )?)
    } else {
        None
    };
    let prepared_lens_shading_map = if let Some(lens_shading_map) = policy_lens_shading_map.as_ref()
    {
        Some(PreparedLensShadingMap::from_typed_map(lens_shading_map)?)
    } else {
        None
    };
    let facts = VignetteCorrectionInputFacts::from_options(
        frame.dimensions(),
        prepared_lens_shading_map.as_ref(),
        options,
    )?;

    apply_cpu_vignette_correction_with_facts(frame, &facts)
}

pub fn apply_cpu_fixed_point_vignette_correction<'a>(
    frame: DecodedBayerU16Frame<'a>,
    facts: &FixedPointVignetteInputFacts<'_>,
) -> Result<CpuVignetteCorrectionResult<'a>, VignetteCorrectionError> {
    let dimensions = frame.dimensions();
    validate_fixed_frame_dimensions(dimensions, facts)?;

    if facts.mode == VignetteCorrectionMode::Disabled {
        return Ok(CpuVignetteCorrectionResult {
            frame,
            output_black_level: facts.output_black_level(),
            output_white_level: facts.output_white_level,
            applied: false,
        });
    }

    let mut output = Vec::with_capacity(frame.pixel_bytes_le().len());
    correct_fixed_into_le_bytes(&frame, facts, &mut output)?;
    let frame = DecodedBayerU16Frame::from_owned_le_bytes(dimensions, output).map_err(|error| {
        VignetteCorrectionError::DecodedFrameValidationFailed(error.to_string())
    })?;

    Ok(CpuVignetteCorrectionResult {
        frame,
        output_black_level: facts.output_black_level(),
        output_white_level: facts.output_white_level,
        applied: true,
    })
}

pub fn apply_cpu_vignette_correction_with_facts<'a>(
    frame: DecodedBayerU16Frame<'a>,
    facts: &VignetteCorrectionInputFacts<'_>,
) -> Result<CpuVignetteCorrectionResult<'a>, VignetteCorrectionError> {
    let dimensions = frame.dimensions();
    if dimensions != facts.frame_dimensions {
        return Err(VignetteCorrectionError::FrameDimensionsMismatch {
            frame_dimensions: dimensions,
            facts_dimensions: facts.frame_dimensions,
        });
    }

    if facts.mode == VignetteCorrectionMode::Disabled {
        return Ok(CpuVignetteCorrectionResult {
            frame,
            output_black_level: facts.output_black_level(),
            output_white_level: facts.output_white_level,
            applied: false,
        });
    }

    let lens_shading_map = facts
        .lens_shading_map
        .as_ref()
        .ok_or(VignetteCorrectionError::MissingLensShadingMap)?;

    let input = frame.pixel_bytes_le();
    let mut output = Vec::with_capacity(input.len());
    let width = usize::try_from(dimensions.width)
        .map_err(|_| VignetteCorrectionError::FrameDimensionsOverflow { dimensions })?;

    for (sample_index, sample_bytes) in input.chunks_exact(BYTES_PER_U16_SAMPLE).enumerate() {
        let x = sample_index % width;
        let y = sample_index / width;
        let plane_index = cfa_position_plane_index(facts.bayer_pattern, x, y);
        let gain_plane_index = match facts.correction_policy {
            VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1 => {
                android_rggb_source_plane_index(facts.bayer_pattern, x, y)
            }
            VignetteCorrectionPolicy::LumaPlane0 => 0,
        };
        let raw_sample = u16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
        let gain = interpolated_prepared_gain(
            lens_shading_map,
            gain_plane_index,
            x,
            y,
            dimensions,
            facts.coordinate_mapping,
        )?;
        let (black, gain, output_white_level) = if facts.correction_policy
            == VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1
        {
            let compressed = 1.0 + MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH * (gain - 1.0).max(0.0);
            (
                facts.pixel_domain.source_black_storage[plane_index],
                facts.pixel_domain.source_to_corrected_scale * compressed,
                facts.pixel_domain.sample_limit,
            )
        } else {
            (
                facts.input_black_level[plane_index],
                gain,
                facts.output_white_level,
            )
        };
        let corrected_sample = corrected_sample(raw_sample, black, gain, output_white_level);

        output.extend_from_slice(&corrected_sample.to_le_bytes());
    }

    let frame = DecodedBayerU16Frame::from_owned_le_bytes(dimensions, output).map_err(|error| {
        VignetteCorrectionError::DecodedFrameValidationFailed(error.to_string())
    })?;

    Ok(CpuVignetteCorrectionResult {
        frame,
        output_black_level: facts.output_black_level(),
        output_white_level: facts.output_white_level,
        applied: true,
    })
}

fn correct_fixed_into_le_bytes(
    frame: &DecodedBayerU16Frame<'_>,
    facts: &FixedPointVignetteInputFacts<'_>,
    output: &mut Vec<u8>,
) -> Result<(), VignetteCorrectionError> {
    let dimensions = frame.dimensions();
    validate_fixed_frame_dimensions(dimensions, facts)?;

    let input = frame.pixel_bytes_le();
    output.clear();
    output.resize(input.len(), 0);

    if facts.mode == VignetteCorrectionMode::Disabled {
        output.copy_from_slice(input);
        return Ok(());
    }

    let lens_shading_map = facts
        .lens_shading_map
        .as_ref()
        .ok_or(VignetteCorrectionError::MissingLensShadingMap)?;
    let width = usize::try_from(dimensions.width)
        .map_err(|_| VignetteCorrectionError::FrameDimensionsOverflow { dimensions })?;

    for (sample_index, (sample_bytes, output_bytes)) in input
        .chunks_exact(BYTES_PER_U16_SAMPLE)
        .zip(output.chunks_exact_mut(BYTES_PER_U16_SAMPLE))
        .enumerate()
    {
        let x = sample_index % width;
        let y = sample_index / width;
        let plane_index = cfa_position_plane_index(facts.bayer_pattern, x, y);
        let raw_sample = u16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
        let (black_q, gain_q, output_white_level) = if facts.correction_policy
            == VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1
        {
            let gain_plane_index = android_rggb_source_plane_index(facts.bayer_pattern, x, y);
            let raw_gain_q = interpolated_fixed_gain(
                lens_shading_map,
                gain_plane_index,
                x,
                y,
                dimensions,
                facts.coordinate_mapping,
            )?;
            let gain = raw_gain_q as f64 / VIGNETTE_GAIN_SCALE as f64;
            let compressed =
                1.0 + f64::from(MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH) * (gain - 1.0).max(0.0);
            let final_gain_q = (f64::from(facts.pixel_domain.source_to_corrected_scale)
                * compressed
                * VIGNETTE_GAIN_SCALE as f64)
                .round() as i64;
            let black_q = (f64::from(facts.pixel_domain.source_black_storage[plane_index])
                * VIGNETTE_GAIN_SCALE as f64)
                .round() as i64;
            (black_q, final_gain_q, facts.pixel_domain.sample_limit)
        } else {
            let gain_q = interpolated_fixed_gain(
                lens_shading_map,
                plane_index,
                x,
                y,
                dimensions,
                facts.coordinate_mapping,
            )?;
            (
                facts.input_black_level_q[plane_index],
                gain_q,
                facts.output_white_level,
            )
        };
        let corrected_sample =
            corrected_fixed_sample(raw_sample, black_q, gain_q, output_white_level)?;
        output_bytes.copy_from_slice(&corrected_sample.to_le_bytes());
    }

    Ok(())
}

fn validate_fixed_frame_dimensions(
    dimensions: mcraw4vulkan_core::FrameDimensions,
    facts: &FixedPointVignetteInputFacts<'_>,
) -> Result<(), VignetteCorrectionError> {
    if dimensions != facts.frame_dimensions {
        return Err(VignetteCorrectionError::FrameDimensionsMismatch {
            frame_dimensions: dimensions,
            facts_dimensions: facts.frame_dimensions,
        });
    }

    Ok(())
}

fn cpu_postprocess_stats(
    facts: &FixedPointVignetteInputFacts<'_>,
    passthrough_without_copy: bool,
    output_buffer_bytes: u64,
) -> Result<VignetteCorrectionStats, VignetteCorrectionError> {
    let corrected_pixel_count = if facts.mode == VignetteCorrectionMode::Enabled {
        u64::try_from(facts.frame_dimensions.pixel_count().ok_or(
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: facts.frame_dimensions,
            },
        )?)
        .map_err(|_| VignetteCorrectionError::FrameDimensionsOverflow {
            dimensions: facts.frame_dimensions,
        })?
    } else {
        0
    };
    let lens_shading_map_dimensions = facts.lens_shading_map.as_ref().and_then(|map| {
        Some((
            u32::try_from(map.width()).ok()?,
            u32::try_from(map.height()).ok()?,
        ))
    });
    let lens_shading_plane_count = facts.lens_shading_map.as_ref().map(|map| map.plane_count());

    Ok(VignetteCorrectionStats {
        applied: facts.mode == VignetteCorrectionMode::Enabled,
        corrected_pixel_count,
        cpu_passthrough_without_copy: passthrough_without_copy,
        cpu_output_buffer_bytes: Some(output_buffer_bytes),
        lens_shading_map_dimensions,
        lens_shading_plane_count,
        ..VignetteCorrectionStats::default()
    })
}

#[cfg(test)]
mod tests {
    use mcraw4vulkan_core::{BayerPattern, FrameDimensions};

    use super::*;
    use crate::{
        LensShadingMap, PreparedFixedLensShadingMap, VignetteCoordinateMapping,
        VignetteCorrectionInputFacts,
    };

    #[test]
    fn optional_cpu_correction_none_returns_original_without_touching_output() {
        let frame = frame_from_samples(2, 2, &[10, 20, 30, 40]);
        let original_bytes = frame.pixel_bytes_le().to_vec();
        let mut corrector = CpuFixedPointVignetteCorrector::new();
        let mut output = vec![9, 8, 7];

        {
            let result = corrector
                .postprocess_optional(&frame, None, &mut output)
                .expect("optional passthrough succeeds");

            assert!(result.frame.is_original());
            assert!(!result.applied());
            assert_eq!(result.info, None);
            assert!(result.stats.cpu_passthrough_without_copy);
            assert_eq!(result.pixel_bytes_le(), original_bytes.as_slice());
        }

        assert_eq!(output, vec![9, 8, 7]);
    }

    #[test]
    fn optional_cpu_disabled_returns_original_and_preserves_black_level() {
        let frame = frame_from_samples(2, 2, &[10, 20, 30, 40]);
        let facts = fixed_facts(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            VignetteCorrectionMode::Disabled,
            None,
            [3.0, 4.0, 5.0, 6.0],
            1023,
        );
        let correction = OptionalCpuVignetteCorrection::new(&facts);
        let mut corrector = CpuFixedPointVignetteCorrector::new();
        let mut output = vec![1, 2, 3, 4];

        {
            let result = corrector
                .postprocess_optional(&frame, Some(correction), &mut output)
                .expect("disabled optional correction succeeds");
            let info = result.info.expect("disabled correction reports info");

            assert!(result.frame.is_original());
            assert!(!result.applied());
            assert_eq!(info.output_black_level, [3.0, 4.0, 5.0, 6.0]);
            assert_eq!(info.output_white_level, 1023);
            assert!(result.stats.cpu_passthrough_without_copy);
            assert_eq!(result.stats.cpu_output_buffer_bytes, Some(0));
        }

        assert_eq!(output, vec![1, 2, 3, 4]);
    }

    #[test]
    fn optional_cpu_enabled_matches_existing_fixed_point_api() {
        let frame = frame_from_samples(2, 2, &[110, 120, 130, 140]);
        let map = constant_map(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let facts = fixed_facts(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            VignetteCorrectionMode::Enabled,
            Some(&map),
            [10.0, 20.0, 30.0, 40.0],
            1023,
        );
        let expected = apply_cpu_fixed_point_vignette_correction(frame.clone(), &facts)
            .expect("existing fixed correction succeeds");
        let mut corrector = CpuFixedPointVignetteCorrector::new();
        let mut output = Vec::new();

        let result = corrector
            .postprocess_optional(
                &frame,
                Some(OptionalCpuVignetteCorrection::new(&facts)),
                &mut output,
            )
            .expect("enabled optional correction succeeds");
        let info = result.info.expect("enabled correction reports info");

        assert!(result.frame.is_corrected());
        assert!(result.applied());
        assert_eq!(result.pixel_bytes_le(), expected.frame.pixel_bytes_le());
        assert_eq!(info.output_black_level, [0.0; BAYER_CFA_PLANE_COUNT]);
        assert_eq!(info.output_white_level, 1023);
        assert_eq!(result.stats.corrected_pixel_count, 4);
        assert_eq!(result.stats.cpu_output_buffer_bytes, Some(8));
        assert!(!result.stats.cpu_passthrough_without_copy);
    }

    #[test]
    fn optional_cpu_enabled_reuses_caller_output_buffer() {
        let frame = frame_from_samples(2, 2, &[110, 120, 130, 140]);
        let map = constant_map(2, 2, &[1.0; 4]);
        let facts = fixed_facts(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            VignetteCorrectionMode::Enabled,
            Some(&map),
            [10.0, 20.0, 30.0, 40.0],
            1023,
        );
        let mut corrector = CpuFixedPointVignetteCorrector::new();
        let mut output = Vec::with_capacity(64);
        let initial_capacity = output.capacity();
        let initial_ptr = output.as_ptr();

        {
            let result = corrector
                .postprocess_optional(
                    &frame,
                    Some(OptionalCpuVignetteCorrection::new(&facts)),
                    &mut output,
                )
                .expect("enabled optional correction succeeds");

            assert_eq!(
                samples_from_bytes(result.pixel_bytes_le()),
                vec![100, 100, 100, 100]
            );
        }

        assert_eq!(output.capacity(), initial_capacity);
        assert_eq!(output.as_ptr(), initial_ptr);
    }

    #[test]
    fn optional_cpu_enabled_leaves_original_frame_unchanged() {
        let frame = frame_from_samples(2, 2, &[110, 120, 130, 140]);
        let map = constant_map(2, 2, &[2.0; 4]);
        let facts = fixed_facts(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            VignetteCorrectionMode::Enabled,
            Some(&map),
            [10.0, 20.0, 30.0, 40.0],
            1023,
        );
        let mut corrector = CpuFixedPointVignetteCorrector::new();
        let mut output = Vec::new();

        {
            let result = corrector
                .postprocess_optional(
                    &frame,
                    Some(OptionalCpuVignetteCorrection::new(&facts)),
                    &mut output,
                )
                .expect("enabled optional correction succeeds");

            assert_eq!(
                samples_from_bytes(result.pixel_bytes_le()),
                vec![200, 200, 200, 200]
            );
        }

        assert_eq!(samples_from_frame(&frame), vec![110, 120, 130, 140]);
    }

    #[test]
    fn optional_cpu_hook_uses_typed_fixed_facts_without_raw_metadata() {
        let frame = frame_from_samples(1, 1, &[110]);
        let map = constant_map(1, 1, &[1.0; 4]);
        let facts = fixed_facts(
            FrameDimensions {
                width: 1,
                height: 1,
            },
            VignetteCorrectionMode::Enabled,
            Some(&map),
            [10.0; 4],
            1023,
        );
        let mut corrector = CpuFixedPointVignetteCorrector::new();
        let mut output = Vec::new();

        let result = corrector
            .postprocess_optional(
                &frame,
                Some(OptionalCpuVignetteCorrection::new(&facts)),
                &mut output,
            )
            .expect("typed facts are enough for optional correction");

        assert_eq!(samples_from_bytes(result.pixel_bytes_le()), vec![100]);
    }

    fn fixed_facts<'a>(
        dimensions: FrameDimensions,
        mode: VignetteCorrectionMode,
        map: Option<&'a LensShadingMap>,
        input_black_level: [f32; BAYER_CFA_PLANE_COUNT],
        output_white_level: u16,
    ) -> FixedPointVignetteInputFacts<'a> {
        let fixed_map = map.map(|map| {
            PreparedFixedLensShadingMap::from_typed_map_with_policy(
                map,
                VignetteCorrectionPolicy::LumaPlane0,
            )
            .expect("valid fixed map prepares")
        });
        let facts =
            VignetteCorrectionInputFacts::new_with_policy(crate::VignetteCorrectionInputConfig {
                mode,
                correction_policy: VignetteCorrectionPolicy::LumaPlane0,
                coordinate_mapping: VignetteCoordinateMapping::VisibleFrame,
                frame_dimensions: dimensions,
                bayer_pattern: BayerPattern::Rggb,
                lens_shading_map: None,
                input_black_level,
                output_white_level,
            })
            .expect("typed facts validate");

        FixedPointVignetteInputFacts::from_input_facts_with_fixed_map(&facts, fixed_map)
            .expect("fixed facts validate")
    }

    fn constant_map(width: u32, height: u32, gains: &[f32; 4]) -> LensShadingMap {
        let pixel_count = (width as usize) * (height as usize);
        LensShadingMap::new(
            width,
            height,
            gains.iter().map(|gain| vec![*gain; pixel_count]).collect(),
        )
        .expect("valid lens shading map")
    }

    fn frame_from_samples(
        width: u32,
        height: u32,
        samples: &[u16],
    ) -> DecodedBayerU16Frame<'static> {
        let mut bytes = Vec::with_capacity(samples.len() * BYTES_PER_U16_SAMPLE);
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        DecodedBayerU16Frame::from_owned_le_bytes(FrameDimensions { width, height }, bytes)
            .expect("valid decoded frame")
    }

    fn samples_from_frame(frame: &DecodedBayerU16Frame<'_>) -> Vec<u16> {
        samples_from_bytes(frame.pixel_bytes_le())
    }

    fn samples_from_bytes(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(BYTES_PER_U16_SAMPLE)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect()
    }
}
