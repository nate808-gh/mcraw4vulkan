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
