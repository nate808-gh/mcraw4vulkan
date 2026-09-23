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
