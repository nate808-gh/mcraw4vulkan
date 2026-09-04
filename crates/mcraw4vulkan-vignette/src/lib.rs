mod cpu;
mod error;
mod gain_map;
mod gpu;
mod lens_map;
mod math;
mod metadata;
mod pipe_f32_bayer;
mod policy;
mod shader;
mod stats;

pub use cpu::{
    CpuFixedPointVignetteCorrector, CpuPostprocessedFrame, CpuVignetteCorrectionResult,
    CpuVignettePostprocessResult, OptionalCpuVignetteCorrection,
    apply_cpu_fixed_point_vignette_correction, apply_cpu_vignette_correction,
    apply_cpu_vignette_correction_with_facts,
};
pub use error::VignetteCorrectionError;
pub use gain_map::{
    CompactSpatialMapFingerprint, GainConversionFingerprint, PreparedFullResolutionFixedGainMap,
    PreparedFullResolutionGainMapCache, PreparedFullResolutionGainMapCacheStats,
    VignetteGainMapFingerprint,
};
pub use gpu::{
    GpuFullResolutionGainMapBinding, GpuLensShadingMapBinding, GpuUploadedFullResolutionGainMap,
    GpuVignetteCorrectionBackend, GpuVignetteCorrectionDispatch, GpuVignetteCorrectionInput,
    GpuVignetteCorrectionOutput, GpuVignetteCorrectionParams, GpuVignetteCorrectionResult,
    GpuVignetteCorrectionStats, GpuVignetteCorrector, GpuVignetteGainMapUpload,
    GpuVignetteGainMapUploadTimings, GpuVignettePackedU16DispatchInput,
    gpu_vignette_correction_result_for_mode,
};
pub use lens_map::{
    LensShadingMap, PreparedFixedLensShadingMap, PreparedLensShadingMap,
    lens_shading_map_for_policy, prepare_fixed_lens_shading_map, prepare_lens_shading_map,
    validate_bayer_lens_shading_map,
};
pub use math::{
    android_rggb_source_plane_index, bayer_site_label, cfa_position_plane_index,
    interpolated_fixed_gain, interpolated_gain, interpolated_prepared_gain,
};
pub use metadata::{
    FixedPointVignetteInputFacts, VignetteCorrectionInputConfig, VignetteCorrectionInputFacts,
    motioncam_pipe_f32_bayer_facts,
};
pub use pipe_f32_bayer::{
    GpuCorrectedF32BayerView, GpuPipeF32BayerDispatch, GpuPipeF32BayerDispatchStats,
    GpuPipeF32BayerPrepareInput, GpuPipeF32BayerStage, PipeF32BayerCorrectionFingerprint,
    PipeF32BayerCorrectionMode, PipeF32BayerError, PipeF32BayerNumericDomain,
};
pub use policy::{
    FixedPointVignettePolicy, MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH,
    MOTIONCAM_PIXEL_SAMPLE_LIMIT_MULTIPLIER, VIGNETTE_GAIN_FRACTIONAL_BITS, VIGNETTE_GAIN_SCALE,
    VignetteCoordinateMapping, VignetteCorrectionMode, VignetteCorrectionOptions,
    VignetteCorrectionPolicy, VignettePixelDomainFacts, corrected_white_tag_for_source_white,
    sample_limit_for_corrected_white_tag,
};
pub use shader::{
    PIPE_CORRECT_BAYER_F32_WGSL, PIPE_CORRECT_BAYER_F32_WORKGROUP_SIZE,
    VIGNETTE_CORRECT_PACKED_U16_WGSL, VIGNETTE_CORRECT_WORKGROUP_SIZE,
    VIGNETTE_PACKED_U16_BYTES_PER_WORD, VIGNETTE_PACKED_U16_SAMPLES_PER_WORD,
};
pub use stats::{VignetteCorrectedFrameInfo, VignetteCorrectionStats, VignetteCorrectionTimings};

pub(crate) const BAYER_CFA_PLANE_COUNT: usize = 4;
pub(crate) const BYTES_PER_U16_SAMPLE: usize = 2;

#[cfg(test)]
mod tests {
    use mcraw4vulkan_core::{BayerPattern, DecodedBayerU16Frame, FrameDimensions};

    use super::*;

    #[test]
    fn accepts_variable_map_dimensions() {
        assert_eq!(
            constant_map(17, 13, &[1.0, 1.0, 1.0, 1.0]).expected_plane_len(),
            221
        );
        assert_eq!(
            constant_map(33, 25, &[1.0, 1.0, 1.0, 1.0]).expected_plane_len(),
            825
        );
    }

    #[test]
    fn rejects_empty_map_dimensions() {
        let error =
            LensShadingMap::new(0, 2, vec![Vec::new()]).expect_err("empty dimensions are rejected");

        assert_eq!(
            error,
            mcraw4vulkan_mcrawcontainer::LensShadingMapValidationError::EmptyDimensions
        );
    }

    #[test]
    fn rejects_wrong_plane_length() {
        let error = LensShadingMap::new(3, 2, vec![vec![1.0; 6], vec![1.0; 5]])
            .expect_err("wrong plane length is rejected");

        assert_eq!(
            error,
            mcraw4vulkan_mcrawcontainer::LensShadingMapValidationError::PlaneLengthMismatch {
                plane_index: 1,
                expected_len: 6,
                actual_len: 5,
            }
        );
    }

    #[test]
    fn rejects_invalid_gain_values() {
        let non_finite = LensShadingMap::new(1, 1, vec![vec![f32::NAN]])
            .expect_err("non-finite gains are rejected");
        assert_eq!(
            non_finite,
            mcraw4vulkan_mcrawcontainer::LensShadingMapValidationError::NonFiniteGain {
                plane_index: 0,
                sample_index: 0,
            }
        );

        let negative =
            LensShadingMap::new(1, 1, vec![vec![-1.0]]).expect_err("negative gains are rejected");
        assert_eq!(
            negative,
            mcraw4vulkan_mcrawcontainer::LensShadingMapValidationError::NegativeGain {
                plane_index: 0,
                sample_index: 0,
                value: -1.0,
            }
        );
    }

    #[test]
    fn rejects_non_bayer_lens_shading_plane_count() {
        let map = LensShadingMap::new(1, 1, vec![vec![1.0], vec![1.0], vec![1.0]])
            .expect("three-plane map is structurally valid");

        let error = validate_bayer_lens_shading_map(&map)
            .expect_err("Bayer correction requires four planes");

        assert_eq!(
            error,
            VignetteCorrectionError::UnsupportedLensShadingPlaneCount {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn prepared_lens_shading_map_uses_typed_dimensions() {
        let map = constant_map(33, 25, &[1.0; 4]);
        let prepared =
            PreparedLensShadingMap::from_typed_map(&map).expect("valid Bayer map prepares");

        assert_eq!(prepared.width(), 33);
        assert_eq!(prepared.height(), 25);
        assert_eq!(prepared.plane_count(), 4);
        assert_eq!(prepared.typed_map().width(), 33);
        assert_eq!(prepared.typed_map().height(), 25);
    }

    #[test]
    fn prepared_lens_shading_map_supports_non_pixel_map_shape() {
        let map = constant_map(17, 13, &[1.0; 4]);
        let prepared =
            prepare_lens_shading_map(&map).expect("valid non-device-specific map prepares");

        assert_eq!(prepared.width(), 17);
        assert_eq!(prepared.height(), 13);
        assert_eq!(prepared.plane_count(), 4);
    }

    #[test]
    fn prepared_lens_shading_map_rejects_wrong_plane_count_at_construction() {
        let map = LensShadingMap::new(1, 1, vec![vec![1.0], vec![1.0], vec![1.0]])
            .expect("three-plane map is structurally valid");

        let error = PreparedLensShadingMap::from_typed_map(&map)
            .expect_err("Bayer preparation requires four planes");

        assert_eq!(
            error,
            VignetteCorrectionError::UnsupportedLensShadingPlaneCount {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn input_facts_use_typed_values_without_inference() {
        let map = constant_map(2, 2, &[1.0; 4]);
        let prepared =
            PreparedLensShadingMap::from_typed_map(&map).expect("valid Bayer map prepares");
        let facts = VignetteCorrectionInputFacts::new(
            VignetteCorrectionMode::Enabled,
            VignetteCoordinateMapping::VisibleFrame,
            FrameDimensions {
                width: 2,
                height: 2,
            },
            BayerPattern::Rggb,
            Some(&prepared),
            [1.0, 2.0, 3.0, 4.0],
            4095,
        )
        .expect("typed facts validate");

        assert_eq!(facts.frame_dimensions.width, 2);
        assert_eq!(facts.frame_dimensions.height, 2);
        assert_eq!(facts.input_black_level, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(facts.output_white_level, 4095);
        assert_eq!(
            facts
                .lens_shading_map
                .expect("prepared map retained")
                .width(),
            2
        );
    }

    #[test]
    fn prepared_fixed_lens_shading_map_quantizes_gains_once() {
        let map = constant_map(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let fixed =
            PreparedFixedLensShadingMap::from_typed_map(&map).expect("valid fixed map prepares");

        assert_eq!(fixed.width(), 2);
        assert_eq!(fixed.height(), 2);
        assert_eq!(fixed.plane_count(), 4);
        assert_eq!(fixed.policy(), FixedPointVignettePolicy::default());
        assert_eq!(fixed.base().typed_map().width(), 2);
    }

    #[test]
    fn prepared_fixed_lens_shading_map_rejects_unrepresentable_gain() {
        let map = LensShadingMap::new(1, 1, vec![vec![f32::MAX]; 4])
            .expect("large finite gains are structurally valid");

        let error = PreparedFixedLensShadingMap::from_typed_map(&map)
            .expect_err("fixed gain overflow is rejected");

        assert_eq!(
            error,
            VignetteCorrectionError::FixedPointGainOverflow {
                plane_index: 0,
                sample_index: 0,
                value: f32::MAX,
            }
        );
    }

    #[test]
    fn fixed_point_facts_quantize_black_levels_from_typed_facts() {
        let facts = VignetteCorrectionInputFacts::from_options(
            FrameDimensions {
                width: 1,
                height: 1,
            },
            None,
            VignetteCorrectionOptions::disabled(BayerPattern::Rggb, [1.0, 1.5, 2.0, 2.5], 4095),
        )
        .expect("typed facts validate");
        let fixed_facts =
            FixedPointVignetteInputFacts::from_input_facts(&facts).expect("facts quantize");

        assert_eq!(
            fixed_facts.input_black_level_q,
            [
                VIGNETTE_GAIN_SCALE,
                VIGNETTE_GAIN_SCALE + VIGNETTE_GAIN_SCALE / 2,
                VIGNETTE_GAIN_SCALE * 2,
                VIGNETTE_GAIN_SCALE * 2 + VIGNETTE_GAIN_SCALE / 2,
            ]
        );
        assert_eq!(fixed_facts.output_white_level, 4095);
    }

    #[test]
    fn fixed_point_facts_reject_unrepresentable_black_level() {
        let facts = VignetteCorrectionInputFacts::from_options(
            FrameDimensions {
                width: 1,
                height: 1,
            },
            None,
            VignetteCorrectionOptions::disabled(
                BayerPattern::Rggb,
                [f32::MAX, 0.0, 0.0, 0.0],
                4095,
            ),
        )
        .expect("typed facts allow finite non-negative black level");

        let error = FixedPointVignetteInputFacts::from_input_facts(&facts)
            .expect_err("fixed black level overflow is rejected");

        assert_eq!(
            error,
            VignetteCorrectionError::FixedPointBlackLevelOverflow {
                index: 0,
                value: f32::MAX,
            }
        );
    }

    #[test]
    fn disabled_mode_preserves_frame_and_metadata_black_level() {
        let frame = frame_from_samples(2, 2, &[10, 20, 30, 40]);
        let options =
            VignetteCorrectionOptions::disabled(BayerPattern::Rggb, [3.0, 4.0, 5.0, 6.0], 1023);

        let result =
            apply_cpu_vignette_correction(frame, None, options).expect("disabled succeeds");

        assert!(!result.applied);
        assert_eq!(result.output_black_level, [3.0, 4.0, 5.0, 6.0]);
        assert_eq!(samples_from_frame(&result.frame), vec![10, 20, 30, 40]);
    }

    #[test]
    fn fixed_point_disabled_mode_preserves_frame_and_metadata_black_level() {
        let frame = frame_from_samples(2, 2, &[10, 20, 30, 40]);
        let facts = VignetteCorrectionInputFacts::from_options(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            None,
            VignetteCorrectionOptions::disabled(BayerPattern::Rggb, [3.0, 4.0, 5.0, 6.0], 1023),
        )
        .expect("disabled facts validate");
        let fixed_facts =
            FixedPointVignetteInputFacts::from_input_facts(&facts).expect("facts quantize");

        let result = apply_cpu_fixed_point_vignette_correction(frame, &fixed_facts)
            .expect("disabled fixed correction succeeds");

        assert!(!result.applied);
        assert_eq!(result.output_black_level, [3.0, 4.0, 5.0, 6.0]);
        assert_eq!(samples_from_frame(&result.frame), vec![10, 20, 30, 40]);
    }

    #[test]
    fn disabled_facts_do_not_require_prepared_lens_shading_map() {
        let frame = frame_from_samples(2, 2, &[10, 20, 30, 40]);
        let facts = VignetteCorrectionInputFacts::from_options(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            None,
            VignetteCorrectionOptions::disabled(BayerPattern::Rggb, [3.0, 4.0, 5.0, 6.0], 1023),
        )
        .expect("disabled facts validate without a lens map");

        let result = apply_cpu_vignette_correction_with_facts(frame, &facts)
            .expect("disabled facts correction succeeds");

        assert!(!result.applied);
        assert_eq!(result.output_black_level, [3.0, 4.0, 5.0, 6.0]);
        assert_eq!(samples_from_frame(&result.frame), vec![10, 20, 30, 40]);
    }

    #[test]
    fn enabled_mode_requires_lens_shading_map() {
        let frame = frame_from_samples(1, 1, &[10]);
        let options = VignetteCorrectionOptions::enabled(BayerPattern::Rggb, [0.0; 4], 1023);

        let error = apply_cpu_vignette_correction(frame, None, options)
            .expect_err("enabled correction requires a map");

        assert_eq!(error, VignetteCorrectionError::MissingLensShadingMap);
    }

    #[test]
    fn rejects_invalid_black_level() {
        let frame = frame_from_samples(1, 1, &[10]);
        let map = constant_map(1, 1, &[1.0; 4]);
        let options =
            VignetteCorrectionOptions::enabled(BayerPattern::Rggb, [0.0, -1.0, 0.0, 0.0], 1023);

        let error = apply_cpu_vignette_correction(frame, Some(&map), options)
            .expect_err("negative black level is rejected");

        assert_eq!(
            error,
            VignetteCorrectionError::InvalidBlackLevel {
                index: 1,
                value: -1.0,
            }
        );
    }

    #[test]
    fn identity_gain_map_black_normalizes_enabled_output() {
        let frame = frame_from_samples(2, 2, &[110, 120, 130, 140]);
        let map = constant_map(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let options = VignetteCorrectionOptions::enabled_with_policy(
            BayerPattern::Rggb,
            [10.0, 20.0, 30.0, 40.0],
            1023,
            VignetteCorrectionPolicy::LumaPlane0,
        );

        let result =
            apply_cpu_vignette_correction(frame, Some(&map), options).expect("enabled succeeds");

        assert!(result.applied);
        assert_eq!(result.output_black_level, [0.0; 4]);
        assert_eq!(samples_from_frame(&result.frame), vec![100, 100, 100, 100]);
    }

    #[test]
    fn fixed_point_identity_gain_map_black_normalizes_enabled_output() {
        let frame = frame_from_samples(2, 2, &[110, 120, 130, 140]);
        let map = constant_map(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let options = VignetteCorrectionOptions::enabled_with_policy(
            BayerPattern::Rggb,
            [10.0, 20.0, 30.0, 40.0],
            1023,
            VignetteCorrectionPolicy::LumaPlane0,
        );
        let fixed_facts = fixed_facts_for_map(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            &map,
            options,
        );

        let result = apply_cpu_fixed_point_vignette_correction(frame, &fixed_facts)
            .expect("enabled fixed correction succeeds");

        assert!(result.applied);
        assert_eq!(result.output_black_level, [0.0; 4]);
        assert_eq!(samples_from_frame(&result.frame), vec![100, 100, 100, 100]);
    }

    #[test]
    fn prepared_facts_match_compatibility_wrapper_output() {
        let frame = frame_from_samples(2, 2, &[110, 120, 130, 140]);
        let map = constant_map(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let prepared =
            PreparedLensShadingMap::from_typed_map(&map).expect("valid Bayer map prepares");
        let options = VignetteCorrectionOptions::enabled_with_policy(
            BayerPattern::Rggb,
            [10.0, 20.0, 30.0, 40.0],
            1023,
            VignetteCorrectionPolicy::LumaPlane0,
        );
        let facts = VignetteCorrectionInputFacts::from_options(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            Some(&prepared),
            options,
        )
        .expect("facts validate");

        let prepared_result = apply_cpu_vignette_correction_with_facts(frame.clone(), &facts)
            .expect("prepared facts correction succeeds");
        let wrapper_result = apply_cpu_vignette_correction(frame, Some(&map), options)
            .expect("compatibility wrapper succeeds");

        assert_eq!(
            samples_from_frame(&prepared_result.frame),
            samples_from_frame(&wrapper_result.frame)
        );
        assert_eq!(
            prepared_result.output_black_level,
            wrapper_result.output_black_level
        );
        assert_eq!(
            prepared_result.output_white_level,
            wrapper_result.output_white_level
        );
        assert_eq!(prepared_result.applied, wrapper_result.applied);
    }

    #[test]
    fn fixed_point_output_matches_float_wrapper_for_exact_corner_gains() {
        let frame = frame_from_samples(2, 2, &[110, 120, 130, 140]);
        let map = constant_map(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let options =
            VignetteCorrectionOptions::enabled(BayerPattern::Rggb, [10.0, 20.0, 30.0, 40.0], 1023);
        let fixed_facts = fixed_facts_for_map(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            &map,
            options,
        );

        let fixed_result = apply_cpu_fixed_point_vignette_correction(frame.clone(), &fixed_facts)
            .expect("fixed correction succeeds");
        let float_result = apply_cpu_vignette_correction(frame, Some(&map), options)
            .expect("float correction succeeds");

        assert_eq!(
            samples_from_frame(&fixed_result.frame),
            samples_from_frame(&float_result.frame)
        );
    }

    #[test]
    fn prepared_facts_reject_frame_dimension_mismatch() {
        let frame = frame_from_samples(2, 2, &[110, 120, 130, 140]);
        let facts = VignetteCorrectionInputFacts::from_options(
            FrameDimensions {
                width: 1,
                height: 4,
            },
            None,
            VignetteCorrectionOptions::disabled(BayerPattern::Rggb, [0.0; 4], 1023),
        )
        .expect("facts validate");

        let error = apply_cpu_vignette_correction_with_facts(frame, &facts)
            .expect_err("facts dimensions must match frame dimensions");

        assert_eq!(
            error,
            VignetteCorrectionError::FrameDimensionsMismatch {
                frame_dimensions: FrameDimensions {
                    width: 2,
                    height: 2,
                },
                facts_dimensions: FrameDimensions {
                    width: 1,
                    height: 4,
                },
            }
        );
    }

    #[test]
    fn correction_uses_black_subtracted_signal_not_raw_sample() {
        let frame = frame_from_samples(1, 1, &[110]);
        let map = constant_map(1, 1, &[2.0, 2.0, 2.0, 2.0]);
        let options = VignetteCorrectionOptions::enabled_with_policy(
            BayerPattern::Rggb,
            [10.0; 4],
            1023,
            VignetteCorrectionPolicy::LumaPlane0,
        );

        let result =
            apply_cpu_vignette_correction(frame, Some(&map), options).expect("enabled succeeds");

        assert_eq!(samples_from_frame(&result.frame), vec![200]);
    }

    #[test]
    fn correction_clips_to_output_white_level() {
        let frame = frame_from_samples(1, 1, &[900]);
        let map = constant_map(1, 1, &[4.0, 4.0, 4.0, 4.0]);
        let options = VignetteCorrectionOptions::enabled_with_policy(
            BayerPattern::Rggb,
            [0.0; 4],
            1023,
            VignetteCorrectionPolicy::LumaPlane0,
        );

        let result =
            apply_cpu_vignette_correction(frame, Some(&map), options).expect("enabled succeeds");

        assert_eq!(samples_from_frame(&result.frame), vec![1023]);
    }

    #[test]
    fn fixed_point_correction_clips_to_output_white_level() {
        let frame = frame_from_samples(1, 1, &[900]);
        let map = constant_map(1, 1, &[4.0, 4.0, 4.0, 4.0]);
        let fixed_facts = fixed_facts_for_map(
            FrameDimensions {
                width: 1,
                height: 1,
            },
            &map,
            VignetteCorrectionOptions::enabled_with_policy(
                BayerPattern::Rggb,
                [0.0; 4],
                1023,
                VignetteCorrectionPolicy::LumaPlane0,
            ),
        );

        let result = apply_cpu_fixed_point_vignette_correction(frame, &fixed_facts)
            .expect("fixed correction succeeds");

        assert_eq!(samples_from_frame(&result.frame), vec![1023]);
    }

    #[test]
    fn luma_plane0_cpu_correction_uses_plane0_for_all_cfa_sites() {
        assert_eq!(cfa_position_plane_index(BayerPattern::Rggb, 0, 0), 0);
        assert_eq!(cfa_position_plane_index(BayerPattern::Rggb, 1, 0), 1);
        assert_eq!(cfa_position_plane_index(BayerPattern::Rggb, 0, 1), 2);
        assert_eq!(cfa_position_plane_index(BayerPattern::Rggb, 1, 1), 3);
        assert_eq!(cfa_position_plane_index(BayerPattern::Gbrg, 0, 0), 0);

        let frame = frame_from_samples(2, 2, &[10, 10, 10, 10]);
        let map = constant_map(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let options = VignetteCorrectionOptions::enabled_with_policy(
            BayerPattern::Rggb,
            [0.0; 4],
            1023,
            VignetteCorrectionPolicy::LumaPlane0,
        );

        let result =
            apply_cpu_vignette_correction(frame, Some(&map), options).expect("enabled succeeds");

        assert_eq!(samples_from_frame(&result.frame), vec![10, 10, 10, 10]);
    }

    #[test]
    fn fixed_point_luma_plane0_uses_plane0_for_all_cfa_sites() {
        let frame = frame_from_samples(2, 2, &[10, 10, 10, 10]);
        let map = constant_map(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let fixed_facts = fixed_facts_for_map_with_policy(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            &map,
            VignetteCorrectionOptions::enabled(BayerPattern::Rggb, [0.0; 4], 1023),
            VignetteCorrectionPolicy::LumaPlane0,
        );

        let result = apply_cpu_fixed_point_vignette_correction(frame, &fixed_facts)
            .expect("fixed correction succeeds");

        assert_eq!(samples_from_frame(&result.frame), vec![10, 10, 10, 10]);
    }

    #[test]
    fn correction_policy_default_is_motioncam_compatible_pixel_domain_v1() {
        assert_eq!(
            VignetteCorrectionPolicy::default(),
            VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1
        );
        assert_eq!(
            VignetteCorrectionOptions::enabled(BayerPattern::Rggb, [0.0; 4], 1023)
                .correction_policy,
            VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1
        );
    }

    #[test]
    fn luma_plane0_policy_copies_plane0_to_all_bayer_sites() {
        let map = constant_map(1, 1, &[1.0, 2.0, 3.0, 4.0]);
        let policy_map = lens_shading_map_for_policy(&map, VignetteCorrectionPolicy::LumaPlane0)
            .expect("policy map builds");

        for plane_index in 0..4 {
            assert_eq!(policy_map.plane(plane_index), Some(&[1.0][..]));
        }
    }

    #[test]
    fn luma_plane0_policy_replicates_first_plane() {
        let map = constant_map(1, 1, &[1.0, 2.0, 3.0, 4.0]);
        let policy_map = lens_shading_map_for_policy(&map, VignetteCorrectionPolicy::LumaPlane0)
            .expect("luma policy map builds");

        for plane_index in 0..4 {
            assert_eq!(policy_map.plane(plane_index), Some(&[1.0][..]));
        }
    }

    #[test]
    fn bilinear_interpolation_uses_runtime_map_dimensions() {
        let map = LensShadingMap::new(
            3,
            3,
            vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; 4],
        )
        .expect("valid map");
        let gain = interpolated_gain(
            &map,
            0,
            1,
            1,
            FrameDimensions {
                width: 3,
                height: 3,
            },
            VignetteCoordinateMapping::VisibleFrame,
        )
        .expect("gain interpolates");

        assert_eq!(gain, 5.0);
    }

    #[test]
    fn fixed_point_interpolation_uses_exact_corner_lookup() {
        let map = LensShadingMap::new(2, 2, vec![vec![1.0, 2.0, 3.0, 4.0]; 4]).expect("valid map");
        let fixed = PreparedFixedLensShadingMap::from_typed_map(&map).expect("fixed map prepares");
        let dimensions = FrameDimensions {
            width: 2,
            height: 2,
        };

        assert_eq!(
            interpolated_fixed_gain(
                &fixed,
                0,
                0,
                0,
                dimensions,
                VignetteCoordinateMapping::VisibleFrame,
            )
            .expect("gain interpolates"),
            VIGNETTE_GAIN_SCALE
        );
        assert_eq!(
            interpolated_fixed_gain(
                &fixed,
                0,
                1,
                1,
                dimensions,
                VignetteCoordinateMapping::VisibleFrame,
            )
            .expect("gain interpolates"),
            VIGNETTE_GAIN_SCALE * 4
        );
    }

    #[test]
    fn fixed_point_interpolation_uses_half_up_midpoint() {
        let map = LensShadingMap::new(2, 2, vec![vec![1.0, 3.0, 5.0, 7.0]; 4]).expect("valid map");
        let fixed = PreparedFixedLensShadingMap::from_typed_map(&map).expect("fixed map prepares");
        let gain = interpolated_fixed_gain(
            &fixed,
            0,
            1,
            1,
            FrameDimensions {
                width: 3,
                height: 3,
            },
            VignetteCoordinateMapping::VisibleFrame,
        )
        .expect("gain interpolates");

        assert_eq!(gain, VIGNETTE_GAIN_SCALE * 4);
    }

    #[test]
    fn fixed_point_interpolation_handles_one_pixel_wide_frame() {
        let map = LensShadingMap::new(2, 2, vec![vec![1.0, 3.0, 5.0, 7.0]; 4]).expect("valid map");
        let fixed = PreparedFixedLensShadingMap::from_typed_map(&map).expect("fixed map prepares");
        let gain = interpolated_fixed_gain(
            &fixed,
            0,
            0,
            1,
            FrameDimensions {
                width: 1,
                height: 2,
            },
            VignetteCoordinateMapping::VisibleFrame,
        )
        .expect("gain interpolates");

        assert_eq!(gain, VIGNETTE_GAIN_SCALE * 5);
    }

    #[test]
    fn fixed_point_interpolation_handles_one_pixel_high_frame() {
        let map = LensShadingMap::new(2, 2, vec![vec![1.0, 3.0, 5.0, 7.0]; 4]).expect("valid map");
        let fixed = PreparedFixedLensShadingMap::from_typed_map(&map).expect("fixed map prepares");
        let gain = interpolated_fixed_gain(
            &fixed,
            0,
            1,
            0,
            FrameDimensions {
                width: 2,
                height: 1,
            },
            VignetteCoordinateMapping::VisibleFrame,
        )
        .expect("gain interpolates");

        assert_eq!(gain, VIGNETTE_GAIN_SCALE * 3);
    }

    #[test]
    fn fixed_point_interpolation_handles_one_cell_map() {
        let map = constant_map(1, 1, &[2.0; 4]);
        let fixed = PreparedFixedLensShadingMap::from_typed_map(&map).expect("fixed map prepares");
        let gain = interpolated_fixed_gain(
            &fixed,
            0,
            3,
            5,
            FrameDimensions {
                width: 7,
                height: 9,
            },
            VignetteCoordinateMapping::VisibleFrame,
        )
        .expect("gain interpolates");

        assert_eq!(gain, VIGNETTE_GAIN_SCALE * 2);
    }

    #[test]
    fn fixed_point_correction_accepts_arbitrary_map_dimensions() {
        for (map_width, map_height) in [(17, 13), (33, 25)] {
            let frame = frame_from_samples(2, 2, &[20, 40, 60, 80]);
            let map = constant_map(map_width, map_height, &[2.0; 4]);
            let fixed_facts = fixed_facts_for_map(
                FrameDimensions {
                    width: 2,
                    height: 2,
                },
                &map,
                VignetteCorrectionOptions::enabled_with_policy(
                    BayerPattern::Rggb,
                    [10.0; 4],
                    4095,
                    VignetteCorrectionPolicy::LumaPlane0,
                ),
            );

            let result = apply_cpu_fixed_point_vignette_correction(frame, &fixed_facts)
                .expect("fixed correction succeeds");

            assert_eq!(samples_from_frame(&result.frame), vec![20, 60, 100, 140]);
        }
    }

    #[test]
    fn fixed_point_output_is_deterministic_across_repeated_calls() {
        let frame = frame_from_samples(3, 3, &[100, 110, 120, 130, 140, 150, 160, 170, 180]);
        let map =
            LensShadingMap::new(2, 2, vec![vec![1.0, 1.25, 1.5, 1.75]; 4]).expect("valid map");
        let fixed_facts = fixed_facts_for_map(
            FrameDimensions {
                width: 3,
                height: 3,
            },
            &map,
            VignetteCorrectionOptions::enabled_with_policy(
                BayerPattern::Rggb,
                [10.0; 4],
                4095,
                VignetteCorrectionPolicy::LumaPlane0,
            ),
        );

        let first = apply_cpu_fixed_point_vignette_correction(frame.clone(), &fixed_facts)
            .expect("first fixed correction succeeds");
        let second = apply_cpu_fixed_point_vignette_correction(frame, &fixed_facts)
            .expect("second fixed correction succeeds");

        assert_eq!(
            samples_from_frame(&first.frame),
            samples_from_frame(&second.frame)
        );
    }

    #[test]
    fn fixed_point_corrector_reuses_output_buffers() {
        let frame = frame_from_samples(2, 2, &[110, 120, 130, 140]);
        let map = constant_map(2, 2, &[1.0; 4]);
        let fixed_facts = fixed_facts_for_map(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            &map,
            VignetteCorrectionOptions::enabled_with_policy(
                BayerPattern::Rggb,
                [10.0, 20.0, 30.0, 40.0],
                1023,
                VignetteCorrectionPolicy::LumaPlane0,
            ),
        );
        let mut corrector = CpuFixedPointVignetteCorrector::new();
        let mut output = Vec::new();

        let borrowed_result = corrector
            .correct_fixed_into(&frame, &fixed_facts, &mut output)
            .expect("fixed correction into output succeeds");

        assert_eq!(
            samples_from_frame(&borrowed_result.frame),
            vec![100, 100, 100, 100]
        );

        let owned_result = corrector
            .correct_fixed_to_owned(&frame, &fixed_facts)
            .expect("fixed correction to owned output succeeds");

        assert_eq!(
            samples_from_frame(&owned_result.frame),
            vec![100, 100, 100, 100]
        );
    }

    #[test]
    fn prepared_bilinear_interpolation_uses_runtime_map_dimensions() {
        let map = LensShadingMap::new(
            3,
            3,
            vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; 4],
        )
        .expect("valid map");
        let prepared =
            PreparedLensShadingMap::from_typed_map(&map).expect("valid Bayer map prepares");
        let gain = interpolated_prepared_gain(
            &prepared,
            0,
            1,
            1,
            FrameDimensions {
                width: 3,
                height: 3,
            },
            VignetteCoordinateMapping::VisibleFrame,
        )
        .expect("prepared gain interpolates");

        assert_eq!(gain, 5.0);
    }

    #[test]
    fn correction_mode_default_is_enabled_for_future_app_policy() {
        assert_eq!(
            VignetteCorrectionMode::default(),
            VignetteCorrectionMode::Enabled
        );
    }

    #[test]
    fn stats_structs_are_public_gui_contracts() {
        let stats = VignetteCorrectionStats {
            applied: true,
            corrected_pixel_count: 4,
            lens_shading_map_dimensions: Some((2, 2)),
            lens_shading_plane_count: Some(4),
            ..VignetteCorrectionStats::default()
        };
        let timings = VignetteCorrectionTimings::default();

        assert!(stats.applied);
        assert_eq!(stats.corrected_pixel_count, 4);
        assert_eq!(timings.total, None);
    }

    #[test]
    fn gpu_api_distinguishes_gpu_correction_from_cpu_correction() {
        struct FakeGpuBackend;

        impl GpuVignetteCorrectionBackend for FakeGpuBackend {
            type FrameResource = u32;
            type LensShadingResource = u32;
            type Error = &'static str;

            fn apply_vignette_correction_gpu(
                &mut self,
                input: GpuVignetteCorrectionInput<
                    '_,
                    Self::FrameResource,
                    Self::LensShadingResource,
                >,
            ) -> Result<GpuVignetteCorrectionResult, Self::Error> {
                *input.output_frame = *input.input_frame + *input.lens_shading_map.resource;
                Ok(gpu_vignette_correction_result_for_mode(input.options))
            }
        }

        let map = constant_map(2, 2, &[1.0; 4]);
        let map_resource = 7_u32;
        let input_frame = 10_u32;
        let mut output_frame = 0_u32;
        let mut backend = FakeGpuBackend;
        let options = VignetteCorrectionOptions::enabled(BayerPattern::Rggb, [64.0; 4], 1023);
        let prepared =
            PreparedLensShadingMap::from_typed_map(&map).expect("valid Bayer map prepares");
        let binding =
            GpuLensShadingMapBinding::from_prepared_map_resource(&map_resource, &prepared);

        assert_eq!(binding.width, 2);
        assert_eq!(binding.height, 2);
        assert_eq!(binding.plane_count, 4);

        let fixed_facts = fixed_facts_for_map(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            &map,
            options,
        );
        let full_gain_map = PreparedFullResolutionFixedGainMap::from_fixed_facts(&fixed_facts)
            .expect("full gain map prepares");
        let full_map_resource = 11_u32;
        let full_binding =
            GpuFullResolutionGainMapBinding::from_prepared_full_resolution_gain_map_resource(
                &full_map_resource,
                &full_gain_map,
            );

        assert_eq!(full_binding.width, 2);
        assert_eq!(full_binding.height, 2);
        assert_eq!(full_binding.pixel_count, 4);
        assert_eq!(full_binding.memory_bytes, 16);
        assert_eq!(full_binding.fractional_bits, VIGNETTE_GAIN_FRACTIONAL_BITS);

        let result = backend
            .apply_vignette_correction_gpu(GpuVignetteCorrectionInput {
                input_frame: &input_frame,
                output_frame: &mut output_frame,
                lens_shading_map: binding,
                frame_dimensions: FrameDimensions {
                    width: 2,
                    height: 2,
                },
                options,
            })
            .expect("fake gpu backend succeeds");

        assert_eq!(output_frame, 17);
        assert!(result.applied);
        assert_eq!(result.output_black_level, [0.0; 4]);
    }

    fn constant_map(width: u32, height: u32, gains: &[f32; 4]) -> LensShadingMap {
        let pixel_count = (width as usize) * (height as usize);
        LensShadingMap::new(
            width,
            height,
            gains.iter().map(|gain| vec![*gain; pixel_count]).collect(),
        )
        .expect("valid constant map")
    }

    fn fixed_facts_for_map<'a>(
        dimensions: FrameDimensions,
        map: &'a LensShadingMap,
        options: VignetteCorrectionOptions,
    ) -> FixedPointVignetteInputFacts<'a> {
        fixed_facts_for_map_with_policy(dimensions, map, options, options.correction_policy)
    }

    fn fixed_facts_for_map_with_policy<'a>(
        dimensions: FrameDimensions,
        map: &'a LensShadingMap,
        options: VignetteCorrectionOptions,
        policy: VignetteCorrectionPolicy,
    ) -> FixedPointVignetteInputFacts<'a> {
        let fixed_map = PreparedFixedLensShadingMap::from_typed_map_with_policy(map, policy)
            .expect("valid fixed map prepares");
        let facts = VignetteCorrectionInputFacts::new_with_policy(VignetteCorrectionInputConfig {
            mode: options.mode,
            correction_policy: policy,
            coordinate_mapping: options.coordinate_mapping,
            frame_dimensions: dimensions,
            bayer_pattern: options.bayer_pattern,
            lens_shading_map: None,
            input_black_level: options.input_black_level,
            output_white_level: options.output_white_level,
        })
        .expect("facts");

        FixedPointVignetteInputFacts::from_input_facts_with_fixed_map(&facts, Some(fixed_map))
            .expect("fixed facts")
    }

    fn frame_from_samples(
        width: u32,
        height: u32,
        samples: &[u16],
    ) -> DecodedBayerU16Frame<'static> {
        let mut bytes = Vec::with_capacity(samples.len() * 2);

        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        DecodedBayerU16Frame::from_owned_le_bytes(FrameDimensions { width, height }, bytes)
            .expect("valid frame")
    }

    fn samples_from_frame(frame: &DecodedBayerU16Frame<'_>) -> Vec<u16> {
        frame
            .pixel_bytes_le()
            .chunks_exact(BYTES_PER_U16_SAMPLE)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect()
    }
}
