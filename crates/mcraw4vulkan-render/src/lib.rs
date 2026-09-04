//! GPU rendering for mcraw4vulkan.
//!
//! This crate owns the reusable render shader code. It intentionally does
//! not depend on external encoders, DNG writing, CPU decode, FUSE, or application sinks.

pub mod direct_yuv12;
pub mod error;
pub mod gpu;
pub mod shader;
pub mod stats;

pub use direct_yuv12::{
    DIRECT_YUV12_NONFINITE_CAMERA, DIRECT_YUV12_NONFINITE_MAPPED, DIRECT_YUV12_NONFINITE_NCL,
    DIRECT_YUV12_STATUS_BYTE_LEN, DirectYuv12ColorTransform, DirectYuv12Error,
    GpuDirectYuv12Dispatch, GpuDirectYuv12DispatchStats, GpuDirectYuv12EncodeInput,
    GpuDirectYuv12Stage, GpuDirectYuv12View, Yuv444p12leNumericDomain, Yuv444p12lePackPolicy,
};
pub use error::DisplayRenderError;
pub use gpu::{
    BayerColor, DisplayRgbTonePolicy, GpuDisplayP999Histogram, GpuP999HistogramConfig,
    GpuP999HistogramInput, GpuP999HistogramParams, GpuP999HistogramResult,
    GpuPreviewRenderEncodeOutput, GpuPreviewTextureRenderer, GpuRenderCalibrationIlluminant,
    GpuRenderColorMatrixSource, GpuRenderColorMetadata, GpuRenderColorMode, GpuRenderColorParams,
    GpuRgbSinkGuard, GpuRgbSinkGuardMode, GpuRgbSinkHighlightDesat, PreviewRenderConfig,
    PreviewRenderOutputInfo, PreviewRgbSinkPolicy, PreviewScaleMode, PreviewTextureFormat,
    PreviewTransferMode, RenderHighlightPolicy, RenderInputCorrection, RenderSampleDomain,
    apply_rb_sum_guard_rgb, bayer_pattern_shader_value, bradford_d50_to_d65_matrix, cfa_color_at,
    hue_preserving_highlight_rolloff, motioncam_pixel_v1_display_sample_limit, p999_scale_for_luma,
    srgb_oetf, white_balance_from_as_shot_neutral, xyz_d65_to_linear_srgb_matrix,
};
pub use shader::{DIRECT_YUV12_TERMINAL_WGSL, SIGNED_BAYER_DEMOSAIC_WGSL};
pub use stats::PreviewRenderStats;
