use std::sync::mpsc;
use std::time::{Duration, Instant};

use mcraw4vulkan_core::{BayerPattern, FrameDimensions};

use crate::DisplayRenderError;
use crate::shader::DISPLAY_RGB_WGSL;
use crate::stats::PreviewRenderStats;

// This 176-byte host buffer mirrors WGSL Params: twenty scalar words occupy
// the first 80 bytes, followed by six vec4 rows aligned to 16 bytes.
const RENDER_PARAMS_BYTE_LEN_USIZE: usize = 176;
const RENDER_PARAMS_BYTE_LEN: u64 = RENDER_PARAMS_BYTE_LEN_USIZE as u64;
const P999_HISTOGRAM_WORKGROUP_THREADS: u64 = 256;
const P999_HISTOGRAM_DEFAULT_BINS: u32 = 65_536;
const P999_HISTOGRAM_STATS_WORDS: u64 = 1;
const P999_HISTOGRAM_LUMA_MIN: f32 = 0.0;
const P999_HISTOGRAM_LUMA_MAX: f32 = 8.0;
const P999_TARGET_PERCENTILE: f32 = 0.999;
const P999_TARGET_OUTPUT_LUMA: f32 = 0.96;
const P999_MIN_SCALE: f32 = 1.0 / 64.0;
const P999_MAX_SCALE: f32 = 8.0;
const P999_EPS: f32 = 1.0e-6;
const VIG_MINUS_HALF_EV_SCALE: f32 = std::f32::consts::FRAC_1_SQRT_2;
const DISPLAY_VIG_DESAT_START_SCALE: f32 = 0.50;
const DISPLAY_VIG_DESAT_END_SCALE: f32 = 1.00;
const DISPLAY_VIG_DESAT_STRENGTH: f32 = 0.60;
const IDENTITY_3X3: [f64; 9] = [
    1.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, //
    0.0, 0.0, 1.0,
];
const XYZ_D65_TO_LINEAR_SRGB: [f64; 9] = [
    3.2404542, -1.5371385, -0.4985314, //
    -0.9692660, 1.8760108, 0.0415560, //
    0.0556434, -0.2040259, 1.0572252,
];
const BRADFORD_D50_TO_D65: [f64; 9] = [
    0.9555766, -0.0230393, 0.0631636, //
    -0.0282895, 1.0099416, 0.0210077, //
    0.0122982, -0.0204830, 1.3299098,
];

/// Display-only tone policy used by the live preview histogram path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayRgbTonePolicy {
    pub no_vig_extra_ev: f32,
    pub vig_extra_ev: f32,
    pub vig_extra_scale: f32,
}

impl DisplayRgbTonePolicy {
    pub const fn canonical() -> Self {
        Self {
            no_vig_extra_ev: 0.0,
            vig_extra_ev: -0.5,
            vig_extra_scale: VIG_MINUS_HALF_EV_SCALE,
        }
    }

    pub fn extra_scale(self, vignette_enabled: bool) -> f32 {
        if vignette_enabled {
            self.vig_extra_scale
        } else {
            1.0
        }
    }

    pub fn render_guard(self, highlight_luma: f32) -> GpuRgbSinkGuard {
        GpuRgbSinkGuard {
            mode: GpuRgbSinkGuardMode::RbSumLimit,
            highlight_luma,
            r_ratio: 0.0,
            b_ratio: 0.0,
        }
    }

    pub fn highlight_desat(
        self,
        vignette_enabled: bool,
        luma_p999: f32,
        final_scale: f32,
    ) -> GpuRgbSinkHighlightDesat {
        if !vignette_enabled {
            return GpuRgbSinkHighlightDesat::default();
        }
        let threshold = (luma_p999 * final_scale).max(0.0);
        GpuRgbSinkHighlightDesat {
            enabled: true,
            start_luma: threshold * DISPLAY_VIG_DESAT_START_SCALE,
            end_luma: threshold * DISPLAY_VIG_DESAT_END_SCALE,
            strength: DISPLAY_VIG_DESAT_STRENGTH,
        }
    }

    pub fn vig_desat_start_scale(self) -> f32 {
        DISPLAY_VIG_DESAT_START_SCALE
    }

    pub fn vig_desat_end_scale(self) -> f32 {
        DISPLAY_VIG_DESAT_END_SCALE
    }

    pub fn vig_desat_strength(self) -> f32 {
        DISPLAY_VIG_DESAT_STRENGTH
    }
}

impl Default for DisplayRgbTonePolicy {
    fn default() -> Self {
        Self::canonical()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuP999HistogramConfig {
    pub bin_count: u32,
    pub luma_min: f32,
    pub luma_max: f32,
    pub target_percentile: f32,
    pub target_output_luma: f32,
    pub min_scale: f32,
    pub max_scale: f32,
}

impl GpuP999HistogramConfig {
    pub const fn canonical() -> Self {
        Self {
            bin_count: P999_HISTOGRAM_DEFAULT_BINS,
            luma_min: P999_HISTOGRAM_LUMA_MIN,
            luma_max: P999_HISTOGRAM_LUMA_MAX,
            target_percentile: P999_TARGET_PERCENTILE,
            target_output_luma: P999_TARGET_OUTPUT_LUMA,
            min_scale: P999_MIN_SCALE,
            max_scale: P999_MAX_SCALE,
        }
    }

    pub fn validate(self) -> Result<(), DisplayRenderError> {
        if self.bin_count == 0 {
            return Err(DisplayRenderError::InvalidParams(
                "p999 histogram bin count must be greater than zero".to_string(),
            ));
        }
        if !self.luma_min.is_finite()
            || !self.luma_max.is_finite()
            || self.luma_max <= self.luma_min
        {
            return Err(DisplayRenderError::InvalidParams(
                "p999 histogram luma range must be finite and increasing".to_string(),
            ));
        }
        if !self.target_percentile.is_finite()
            || self.target_percentile <= 0.0
            || self.target_percentile > 1.0
        {
            return Err(DisplayRenderError::InvalidParams(
                "p999 histogram percentile must be in (0, 1]".to_string(),
            ));
        }
        if !self.target_output_luma.is_finite() || self.target_output_luma <= 0.0 {
            return Err(DisplayRenderError::InvalidParams(
                "p999 target output luma must be finite and positive".to_string(),
            ));
        }
        if !self.min_scale.is_finite()
            || !self.max_scale.is_finite()
            || self.min_scale <= 0.0
            || self.max_scale < self.min_scale
        {
            return Err(DisplayRenderError::InvalidParams(
                "p999 scale range must be finite and increasing".to_string(),
            ));
        }
        Ok(())
    }

    pub fn histogram_readback_bytes(self) -> Result<usize, DisplayRenderError> {
        let bin_bytes = u64::from(self.bin_count).checked_mul(4).ok_or_else(|| {
            DisplayRenderError::InvalidParams("p999 bin byte count overflow".into())
        })?;
        let stats_bytes = P999_HISTOGRAM_STATS_WORDS.checked_mul(4).ok_or_else(|| {
            DisplayRenderError::InvalidParams("p999 stats byte count overflow".into())
        })?;
        usize::try_from(bin_bytes + stats_bytes).map_err(|_| {
            DisplayRenderError::InvalidParams("p999 readback byte count does not fit usize".into())
        })
    }
}

impl Default for GpuP999HistogramConfig {
    fn default() -> Self {
        Self::canonical()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuP999HistogramResult {
    pub luma_p999: f32,
    pub scale_p999: f32,
    pub final_scale: f32,
    pub total_samples: u32,
    pub overflow_count: u32,
    pub max_luma_observed: f32,
    pub histogram_bins: u32,
    pub readback_bytes: usize,
}

pub fn p999_scale_for_luma(luma_p999: f32, config: GpuP999HistogramConfig) -> f32 {
    let denom = luma_p999.max(P999_EPS);
    (config.target_output_luma / denom).clamp(config.min_scale, config.max_scale)
}

pub fn apply_rb_sum_guard_rgb(rgb: [f32; 3]) -> [f32; 3] {
    let rb_sum = rgb[0].max(0.0) + rgb[2].max(0.0);
    let limit = 2.0 * rgb[1].max(0.0);
    if rb_sum > limit && rb_sum > P999_EPS {
        let scale = limit / rb_sum;
        [rgb[0].max(0.0) * scale, rgb[1], rgb[2].max(0.0) * scale]
    } else {
        rgb
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GpuRgbSinkGuardMode {
    #[default]
    None,
    RbSumLimit,
}

impl GpuRgbSinkGuardMode {
    fn shader_value(self) -> u32 {
        match self {
            Self::None => 0,
            Self::RbSumLimit => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuRgbSinkGuard {
    pub mode: GpuRgbSinkGuardMode,
    pub highlight_luma: f32,
    pub r_ratio: f32,
    pub b_ratio: f32,
}

impl GpuRgbSinkGuard {
    pub fn none() -> Self {
        Self::default()
    }
}

impl Default for GpuRgbSinkGuard {
    fn default() -> Self {
        Self {
            mode: GpuRgbSinkGuardMode::None,
            highlight_luma: 0.0,
            r_ratio: 0.0,
            b_ratio: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuRgbSinkHighlightDesat {
    pub enabled: bool,
    pub start_luma: f32,
    pub end_luma: f32,
    pub strength: f32,
}

impl GpuRgbSinkHighlightDesat {
    pub fn none() -> Self {
        Self::default()
    }
}

impl Default for GpuRgbSinkHighlightDesat {
    fn default() -> Self {
        Self {
            enabled: false,
            start_luma: 0.0,
            end_luma: 0.0,
            strength: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PreviewTextureFormat {
    #[default]
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
}

impl PreviewTextureFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Rgba8Unorm => "rgba8unorm",
            Self::Rgba8UnormSrgb => "rgba8unorm-srgb",
            Self::Bgra8Unorm => "bgra8unorm",
            Self::Bgra8UnormSrgb => "bgra8unorm-srgb",
        }
    }

    pub fn parse_label(value: &str) -> Option<Self> {
        match value {
            "rgba8unorm" | "rgba8" => Some(Self::Rgba8Unorm),
            "rgba8unorm-srgb" | "rgba8-srgb" => Some(Self::Rgba8UnormSrgb),
            "bgra8unorm" | "bgra8" => Some(Self::Bgra8Unorm),
            "bgra8unorm-srgb" | "bgra8-srgb" => Some(Self::Bgra8UnormSrgb),
            _ => None,
        }
    }

    pub fn wgpu_format(self) -> wgpu::TextureFormat {
        match self {
            Self::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            Self::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            Self::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
            Self::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
        }
    }

    pub fn bytes_per_pixel(self) -> u64 {
        4
    }

    pub fn is_srgb(self) -> bool {
        matches!(self, Self::Rgba8UnormSrgb | Self::Bgra8UnormSrgb)
    }

    fn supports_compute_storage_write(self) -> bool {
        // This compute path writes only rgba8unorm storage textures. sRGB formats are
        // not storage targets, and BGRA writes require optional format support.
        matches!(self, Self::Rgba8Unorm)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PreviewTransferMode {
    #[default]
    ShaderSrgb,
    HardwareSrgb,
}

impl PreviewTransferMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::ShaderSrgb => "shader-srgb",
            Self::HardwareSrgb => "hardware-srgb",
        }
    }

    pub fn parse_label(value: &str) -> Option<Self> {
        match value {
            "shader-srgb" | "shader_srgb" => Some(Self::ShaderSrgb),
            "hardware-srgb" | "hardware_srgb" => Some(Self::HardwareSrgb),
            _ => None,
        }
    }

    pub fn applies_shader_srgb(self) -> bool {
        matches!(self, Self::ShaderSrgb)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PreviewScaleMode {
    #[default]
    FullResolution,
    Half,
    Quarter,
    Explicit {
        width: u32,
        height: u32,
    },
}

impl PreviewScaleMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::FullResolution => "full",
            Self::Half => "half",
            Self::Quarter => "quarter",
            Self::Explicit { .. } => "explicit",
        }
    }

    pub fn resolve(self, input: FrameDimensions) -> Result<FrameDimensions, DisplayRenderError> {
        let dimensions = match self {
            Self::FullResolution => input,
            Self::Half => FrameDimensions {
                width: (input.width / 2).max(1),
                height: (input.height / 2).max(1),
            },
            Self::Quarter => FrameDimensions {
                width: (input.width / 4).max(1),
                height: (input.height / 4).max(1),
            },
            Self::Explicit { width, height } => FrameDimensions { width, height },
        };
        if dimensions.width == 0 || dimensions.height == 0 {
            return Err(DisplayRenderError::InvalidDimensions { dimensions });
        }
        Ok(dimensions)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderInputCorrection {
    None,
    MotionCamCompatiblePixelDomainV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderHighlightPolicy {
    None,
    HuePreservingRolloff,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderSampleDomain {
    pub nominal_white_level: f32,
    pub sample_limit: f32,
    pub correction: RenderInputCorrection,
    pub highlight_policy: RenderHighlightPolicy,
}

impl RenderSampleDomain {
    pub fn raw(white_level: f32) -> Self {
        Self {
            nominal_white_level: white_level,
            sample_limit: white_level,
            correction: RenderInputCorrection::None,
            highlight_policy: RenderHighlightPolicy::None,
        }
    }

    pub fn motioncam_compatible_pixel_v1(nominal_white_level: f32) -> Self {
        Self {
            nominal_white_level,
            sample_limit: motioncam_pixel_v1_display_sample_limit(nominal_white_level),
            correction: RenderInputCorrection::MotionCamCompatiblePixelDomainV1,
            highlight_policy: RenderHighlightPolicy::HuePreservingRolloff,
        }
    }

    fn validate(self, preview_white_level: f32) -> Result<(), DisplayRenderError> {
        if !self.nominal_white_level.is_finite() || self.nominal_white_level <= 0.0 {
            return Err(DisplayRenderError::InvalidParams(
                "preview nominal white level must be finite and positive".to_string(),
            ));
        }
        if !self.sample_limit.is_finite() || self.sample_limit <= 0.0 {
            return Err(DisplayRenderError::InvalidParams(
                "preview sample limit must be finite and positive".to_string(),
            ));
        }
        if self.sample_limit + f32::EPSILON < self.nominal_white_level {
            return Err(DisplayRenderError::InvalidParams(
                "preview sample limit must be at least nominal white level".to_string(),
            ));
        }
        if (self.nominal_white_level - preview_white_level).abs() > 0.5 {
            return Err(DisplayRenderError::InvalidParams(format!(
                "preview sample-domain nominal white {} does not match preview white level {}",
                self.nominal_white_level, preview_white_level
            )));
        }
        Ok(())
    }

    fn preview_highlight_mode(self) -> u32 {
        match (self.correction, self.highlight_policy) {
            (
                RenderInputCorrection::MotionCamCompatiblePixelDomainV1,
                RenderHighlightPolicy::HuePreservingRolloff,
            ) => 1,
            _ => 0,
        }
    }

    fn sample_limit_word(self) -> u32 {
        self.sample_limit.round().clamp(0.0, u32::MAX as f32) as u32
    }
}

pub fn motioncam_pixel_v1_display_sample_limit(nominal_white_level: f32) -> f32 {
    (nominal_white_level * 4.0 + 3.0).clamp(nominal_white_level.max(1.0), f32::from(u16::MAX))
}

pub fn hue_preserving_highlight_rolloff(rgb: [f32; 3]) -> [f32; 3] {
    let non_negative = [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)];
    let max_channel = non_negative
        .iter()
        .copied()
        .fold(0.0_f32, |acc, value| acc.max(value));
    if max_channel <= 1.0 {
        return non_negative;
    }
    let scale = 1.0 / max_channel;
    [
        non_negative[0] * scale,
        non_negative[1] * scale,
        non_negative[2] * scale,
    ]
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GpuRenderColorMode {
    #[default]
    MetadataSrgb,
}

impl GpuRenderColorMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::MetadataSrgb => "metadata-srgb",
        }
    }

    pub fn parse_label(value: &str) -> Option<Self> {
        match value {
            "metadata-srgb" => Some(Self::MetadataSrgb),
            _ => None,
        }
    }

    fn shader_value(self) -> u32 {
        match self {
            Self::MetadataSrgb => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRenderCalibrationIlluminant {
    StandardA,
    D65,
    Other,
}

impl GpuRenderCalibrationIlluminant {
    pub fn label(self) -> &'static str {
        match self {
            Self::StandardA => "standard-a",
            Self::D65 => "d65",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRenderColorMatrixSource {
    ForwardMatrix1,
    ForwardMatrix2,
    InverseColorMatrix1,
    InverseColorMatrix2,
    ColorMatrixAdapted,
    IdentityFallback,
}

impl GpuRenderColorMatrixSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::ForwardMatrix1 => "forward-matrix-1",
            Self::ForwardMatrix2 => "forward-matrix-2",
            Self::InverseColorMatrix1 => "inverse-color-matrix-1",
            Self::InverseColorMatrix2 => "inverse-color-matrix-2",
            Self::IdentityFallback => "identity-fallback",
            Self::ColorMatrixAdapted => "colormatrix-adapted-d50",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuRenderColorMetadata {
    pub color_matrix1: Option<[f64; 9]>,
    pub color_matrix2: Option<[f64; 9]>,
    pub forward_matrix1: Option<[f64; 9]>,
    pub forward_matrix2: Option<[f64; 9]>,
    pub illuminant1: Option<GpuRenderCalibrationIlluminant>,
    pub illuminant2: Option<GpuRenderCalibrationIlluminant>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuRenderColorParams {
    pub mode: GpuRenderColorMode,
    pub white_balance_rgb: [f32; 3],
    pub camera_to_srgb: [f32; 9],
    pub apply_srgb_transfer: bool,
    pub matrix_source: GpuRenderColorMatrixSource,
    pub selected_illuminant: Option<GpuRenderCalibrationIlluminant>,
    pub camera_to_xyz_d50: [f64; 9],
}

impl GpuRenderColorParams {
    /// Validated DNG ColorMatrix-only transform, including white adaptation.
    /// The shader receives unbalanced camera RGB, so no second neutral division.
    pub fn from_camera_to_xyz_d50(camera_to_xyz_d50: [f64; 9]) -> Result<Self, DisplayRenderError> {
        let matrix = multiply_3x3(
            XYZ_D65_TO_LINEAR_SRGB,
            multiply_3x3(BRADFORD_D50_TO_D65, camera_to_xyz_d50),
        );
        if matrix
            .iter()
            .any(|v| !v.is_finite() || !(*v as f32).is_finite())
        {
            return Err(DisplayRenderError::InvalidParams(
                "camera to sRGB matrix must be finite".to_owned(),
            ));
        }
        Ok(Self {
            mode: GpuRenderColorMode::MetadataSrgb,
            white_balance_rgb: [1.0; 3],
            camera_to_srgb: f64_matrix_to_f32(matrix),
            apply_srgb_transfer: true,
            matrix_source: GpuRenderColorMatrixSource::ColorMatrixAdapted,
            selected_illuminant: None,
            camera_to_xyz_d50,
        })
    }

    pub fn from_metadata(
        mode: GpuRenderColorMode,
        as_shot_neutral: Option<[f64; 3]>,
        metadata: GpuRenderColorMetadata,
    ) -> Result<Self, DisplayRenderError> {
        let selection = select_camera_to_xyz_d50(metadata)?;
        // DNG ForwardMatrix is camera RGB -> XYZ D50 after white balance. DNG
        // ColorMatrix has the opposite direction, so the fallback inverts it
        // before this display chain:
        // XYZ_D65_to_sRGB * Bradford(D50->D65) * camera_to_XYZ_D50.
        let camera_to_xyz_for_srgb = multiply_3x3(BRADFORD_D50_TO_D65, selection.camera_to_xyz_d50);
        let camera_to_srgb = multiply_3x3(XYZ_D65_TO_LINEAR_SRGB, camera_to_xyz_for_srgb);

        if !camera_to_srgb.iter().all(|value| value.is_finite()) {
            return Err(DisplayRenderError::InvalidParams(
                "camera to sRGB matrix must be finite".to_string(),
            ));
        }

        Ok(Self {
            mode,
            white_balance_rgb: white_balance_from_as_shot_neutral(as_shot_neutral),
            camera_to_srgb: f64_matrix_to_f32(camera_to_srgb),
            apply_srgb_transfer: true,
            matrix_source: selection.source,
            selected_illuminant: selection.illuminant,
            camera_to_xyz_d50: selection.camera_to_xyz_d50,
        })
    }
}

impl Default for GpuRenderColorParams {
    fn default() -> Self {
        Self::from_metadata(
            GpuRenderColorMode::MetadataSrgb,
            None,
            GpuRenderColorMetadata::default(),
        )
        .expect("identity fallback color params build")
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CameraToXyzSelection {
    camera_to_xyz_d50: [f64; 9],
    source: GpuRenderColorMatrixSource,
    illuminant: Option<GpuRenderCalibrationIlluminant>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuP999HistogramParams {
    pub dimensions: FrameDimensions,
    pub bayer_pattern: BayerPattern,
    pub source_bits: u16,
    pub black_level: [f32; 4],
    pub white_level: f32,
    pub sample_limit: f32,
    pub color: GpuRenderColorParams,
}

impl GpuP999HistogramParams {
    fn validate(self) -> Result<usize, DisplayRenderError> {
        if self.dimensions.width == 0 || self.dimensions.height == 0 {
            return Err(DisplayRenderError::InvalidDimensions {
                dimensions: self.dimensions,
            });
        }
        let pixel_count =
            self.dimensions
                .pixel_count()
                .ok_or(DisplayRenderError::OutputByteCountOverflow {
                    dimensions: self.dimensions,
                })?;
        if pixel_count % 2 != 0 {
            return Err(DisplayRenderError::OddPixelCount { pixel_count });
        }
        if !self.white_level.is_finite() || self.white_level <= 0.0 {
            return Err(DisplayRenderError::InvalidParams(
                "white level must be finite and positive".to_string(),
            ));
        }
        if !self.sample_limit.is_finite() || self.sample_limit <= 0.0 {
            return Err(DisplayRenderError::InvalidParams(
                "render sample limit must be finite and positive".to_string(),
            ));
        }
        if self.sample_limit + f32::EPSILON < self.white_level {
            return Err(DisplayRenderError::InvalidParams(
                "render sample limit must be at least white level".to_string(),
            ));
        }
        for (index, value) in self.black_level.into_iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(DisplayRenderError::InvalidParams(format!(
                    "black level {index} must be finite and non-negative"
                )));
            }
        }
        for (index, value) in self.color.white_balance_rgb.into_iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(DisplayRenderError::InvalidParams(format!(
                    "white balance component {index} must be finite and non-negative"
                )));
            }
        }
        for (index, value) in self.color.camera_to_srgb.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(DisplayRenderError::InvalidParams(format!(
                    "camera to sRGB matrix component {index} must be finite"
                )));
            }
        }
        if self.color.mode != GpuRenderColorMode::MetadataSrgb {
            return Err(DisplayRenderError::InvalidParams(
                "display p999 histogram requires metadata-srgb color mode".to_string(),
            ));
        }
        Ok(pixel_count)
    }

    fn to_uniform_bytes(
        self,
        pixel_count: usize,
        dispatch_groups_x: u64,
    ) -> Result<[u8; RENDER_PARAMS_BYTE_LEN_USIZE], DisplayRenderError> {
        let mut bytes = [0u8; RENDER_PARAMS_BYTE_LEN_USIZE];
        let visible_bytes = u64::try_from(pixel_count)
            .ok()
            .and_then(|pixels| pixels.checked_mul(6))
            .ok_or(DisplayRenderError::OutputByteCountOverflow {
                dimensions: self.dimensions,
            })?;
        let aligned_word_count = align_to_copy_bytes(visible_bytes) / 4;
        let words = [
            self.dimensions.width,
            self.dimensions.height,
            u32::try_from(pixel_count).map_err(|_| {
                DisplayRenderError::OutputByteCountOverflow {
                    dimensions: self.dimensions,
                }
            })?,
            u32::try_from(aligned_word_count).map_err(|_| {
                DisplayRenderError::OutputByteCountOverflow {
                    dimensions: self.dimensions,
                }
            })?,
            0,
            bayer_pattern_to_shader_value(self.bayer_pattern),
            u32::from(self.source_bits),
            u32::try_from(dispatch_groups_x).map_err(|_| {
                DisplayRenderError::OutputByteCountOverflow {
                    dimensions: self.dimensions,
                }
            })?,
            u32::try_from(pixel_count).map_err(|_| {
                DisplayRenderError::OutputByteCountOverflow {
                    dimensions: self.dimensions,
                }
            })?,
            self.color.mode.shader_value(),
            u32::from(self.color.apply_srgb_transfer),
            1 | (2 << 8),
        ];
        for (index, word) in words.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }

        let floats = [
            self.black_level[0],
            self.black_level[1],
            self.black_level[2],
            self.black_level[3],
            self.white_level,
            self.color.white_balance_rgb[0],
            self.color.white_balance_rgb[1],
            self.color.white_balance_rgb[2],
            self.color.camera_to_srgb[0],
            self.color.camera_to_srgb[1],
            self.color.camera_to_srgb[2],
            self.sample_limit,
            self.color.camera_to_srgb[3],
            self.color.camera_to_srgb[4],
            self.color.camera_to_srgb[5],
            0.0,
            self.color.camera_to_srgb[6],
            self.color.camera_to_srgb[7],
            self.color.camera_to_srgb[8],
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        for (index, value) in floats.into_iter().enumerate() {
            let byte_index = 48 + index * 4;
            bytes[byte_index..byte_index + 4].copy_from_slice(&value.to_le_bytes());
        }

        Ok(bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreviewRgbSinkPolicy {
    pub rgb_sink_tone_scale: f32,
    pub rgb_sink_guard: GpuRgbSinkGuard,
    pub rgb_sink_desat: GpuRgbSinkHighlightDesat,
}

impl PreviewRgbSinkPolicy {
    pub fn none() -> Self {
        Self::default()
    }

    fn validate(self, color_mode: GpuRenderColorMode) -> Result<(), DisplayRenderError> {
        if !self.rgb_sink_tone_scale.is_finite() || self.rgb_sink_tone_scale < 0.0 {
            return Err(DisplayRenderError::InvalidParams(
                "preview RGB sink tone scale must be finite and non-negative".to_string(),
            ));
        }
        if self.rgb_sink_guard.mode != GpuRgbSinkGuardMode::None {
            if color_mode != GpuRenderColorMode::MetadataSrgb {
                return Err(DisplayRenderError::InvalidParams(
                    "preview RGB sink guard requires metadata-srgb color mode".to_string(),
                ));
            }
            for (label, value) in [
                ("highlight luma", self.rgb_sink_guard.highlight_luma),
                ("red ratio", self.rgb_sink_guard.r_ratio),
                ("blue ratio", self.rgb_sink_guard.b_ratio),
            ] {
                if !value.is_finite() || value < 0.0 {
                    return Err(DisplayRenderError::InvalidParams(format!(
                        "preview RGB sink guard {label} must be finite and non-negative"
                    )));
                }
            }
        }
        if self.rgb_sink_desat.enabled {
            if color_mode != GpuRenderColorMode::MetadataSrgb {
                return Err(DisplayRenderError::InvalidParams(
                    "preview RGB sink highlight desat requires metadata-srgb color mode"
                        .to_string(),
                ));
            }
            for (label, value) in [
                ("start luma", self.rgb_sink_desat.start_luma),
                ("end luma", self.rgb_sink_desat.end_luma),
                ("strength", self.rgb_sink_desat.strength),
            ] {
                if !value.is_finite() || value < 0.0 {
                    return Err(DisplayRenderError::InvalidParams(format!(
                        "preview RGB sink highlight desat {label} must be finite and non-negative"
                    )));
                }
            }
            if self.rgb_sink_desat.end_luma + f32::EPSILON < self.rgb_sink_desat.start_luma {
                return Err(DisplayRenderError::InvalidParams(
                    "preview RGB sink highlight desat end luma must be >= start luma".to_string(),
                ));
            }
            if self.rgb_sink_desat.strength > 1.0 {
                return Err(DisplayRenderError::InvalidParams(
                    "preview RGB sink highlight desat strength must be <= 1".to_string(),
                ));
            }
        }
        Ok(())
    }
}

impl Default for PreviewRgbSinkPolicy {
    fn default() -> Self {
        Self {
            rgb_sink_tone_scale: 1.0,
            rgb_sink_guard: GpuRgbSinkGuard::none(),
            rgb_sink_desat: GpuRgbSinkHighlightDesat::none(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreviewRenderConfig {
    pub dimensions: FrameDimensions,
    pub bayer_pattern: BayerPattern,
    pub source_bits: u16,
    pub black_level: [f32; 4],
    pub white_level: f32,
    pub sample_domain: RenderSampleDomain,
    pub color: GpuRenderColorParams,
    pub texture_format: PreviewTextureFormat,
    pub transfer_mode: PreviewTransferMode,
    pub scale_mode: PreviewScaleMode,
}

impl PreviewRenderConfig {
    pub fn output_dimensions(self) -> Result<FrameDimensions, DisplayRenderError> {
        self.scale_mode.resolve(self.dimensions)
    }

    #[cfg(test)]
    fn validate(self) -> Result<PreviewRenderOutputInfo, DisplayRenderError> {
        self.validate_with_rgb_sink_policy(PreviewRgbSinkPolicy::default())
    }

    fn validate_with_rgb_sink_policy(
        self,
        rgb_sink_policy: PreviewRgbSinkPolicy,
    ) -> Result<PreviewRenderOutputInfo, DisplayRenderError> {
        if self.dimensions.width == 0 || self.dimensions.height == 0 {
            return Err(DisplayRenderError::InvalidDimensions {
                dimensions: self.dimensions,
            });
        }
        if !self.texture_format.supports_compute_storage_write() {
            return Err(DisplayRenderError::InvalidParams(format!(
                "{} preview textures are parsed for future display/surface selection, but GPU-DISPLAY-B compute output currently implements rgba8unorm storage textures only",
                self.texture_format.label()
            )));
        }
        if !self.white_level.is_finite() || self.white_level <= 0.0 {
            return Err(DisplayRenderError::InvalidParams(
                "white level must be finite and positive".to_string(),
            ));
        }
        self.sample_domain.validate(self.white_level)?;
        for (index, value) in self.black_level.into_iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(DisplayRenderError::InvalidParams(format!(
                    "black level {index} must be finite and non-negative"
                )));
            }
        }
        for (index, value) in self.color.white_balance_rgb.into_iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(DisplayRenderError::InvalidParams(format!(
                    "white balance component {index} must be finite and non-negative"
                )));
            }
        }
        for (index, value) in self.color.camera_to_srgb.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(DisplayRenderError::InvalidParams(format!(
                    "camera to sRGB matrix component {index} must be finite"
                )));
            }
        }
        rgb_sink_policy.validate(self.color.mode)?;
        let input_pixel_count =
            self.dimensions
                .pixel_count()
                .ok_or(DisplayRenderError::OutputByteCountOverflow {
                    dimensions: self.dimensions,
                })?;
        let output_dimensions = self.output_dimensions()?;
        let output_pixel_count =
            output_dimensions
                .pixel_count()
                .ok_or(DisplayRenderError::OutputByteCountOverflow {
                    dimensions: output_dimensions,
                })?;
        let texture_bytes = u64::try_from(output_pixel_count)
            .map_err(|_| DisplayRenderError::OutputByteCountOverflow {
                dimensions: output_dimensions,
            })?
            .checked_mul(self.texture_format.bytes_per_pixel())
            .ok_or(DisplayRenderError::OutputByteCountOverflow {
                dimensions: output_dimensions,
            })?;
        Ok(PreviewRenderOutputInfo {
            input_dimensions: self.dimensions,
            output_dimensions,
            texture_format: self.texture_format,
            transfer_mode: self.transfer_mode,
            input_pixel_count,
            output_pixel_count,
            texture_bytes,
            no_full_frame_readback: true,
        })
    }

    fn shader_applies_srgb_transfer(self) -> bool {
        self.color.apply_srgb_transfer && self.transfer_mode.applies_shader_srgb()
    }

    #[cfg(test)]
    fn to_uniform_bytes(
        self,
        info: PreviewRenderOutputInfo,
    ) -> Result<[u8; RENDER_PARAMS_BYTE_LEN_USIZE], DisplayRenderError> {
        self.to_uniform_bytes_with_rgb_sink_policy(info, PreviewRgbSinkPolicy::default())
    }

    fn to_uniform_bytes_with_rgb_sink_policy(
        self,
        info: PreviewRenderOutputInfo,
        rgb_sink_policy: PreviewRgbSinkPolicy,
    ) -> Result<[u8; RENDER_PARAMS_BYTE_LEN_USIZE], DisplayRenderError> {
        let mut bytes = [0u8; RENDER_PARAMS_BYTE_LEN_USIZE];
        let words = [
            self.dimensions.width,
            self.dimensions.height,
            u32::try_from(info.input_pixel_count).map_err(|_| {
                DisplayRenderError::OutputByteCountOverflow {
                    dimensions: self.dimensions,
                }
            })?,
            info.output_dimensions.width,
            0,
            bayer_pattern_to_shader_value(self.bayer_pattern),
            u32::from(self.source_bits),
            self.sample_domain.preview_highlight_mode(),
            self.sample_domain.sample_limit_word(),
            self.color.mode.shader_value(),
            u32::from(self.shader_applies_srgb_transfer()),
            info.output_dimensions.height,
        ];
        for (index, word) in words.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }

        let floats = [
            self.black_level[0],
            self.black_level[1],
            self.black_level[2],
            self.black_level[3],
            self.white_level,
            self.color.white_balance_rgb[0],
            self.color.white_balance_rgb[1],
            self.color.white_balance_rgb[2],
            self.color.camera_to_srgb[0],
            self.color.camera_to_srgb[1],
            self.color.camera_to_srgb[2],
            0.0,
            self.color.camera_to_srgb[3],
            self.color.camera_to_srgb[4],
            self.color.camera_to_srgb[5],
            0.0,
            self.color.camera_to_srgb[6],
            self.color.camera_to_srgb[7],
            self.color.camera_to_srgb[8],
            0.0,
            rgb_sink_policy.rgb_sink_tone_scale,
            0.0,
            0.0,
            0.0,
            rgb_sink_policy.rgb_sink_guard.mode.shader_value() as f32,
            rgb_sink_policy.rgb_sink_guard.highlight_luma,
            rgb_sink_policy.rgb_sink_guard.r_ratio,
            rgb_sink_policy.rgb_sink_guard.b_ratio,
            if rgb_sink_policy.rgb_sink_desat.enabled {
                1.0
            } else {
                0.0
            },
            rgb_sink_policy.rgb_sink_desat.start_luma,
            rgb_sink_policy.rgb_sink_desat.end_luma,
            rgb_sink_policy.rgb_sink_desat.strength,
        ];
        for (index, value) in floats.into_iter().enumerate() {
            let byte_index = 48 + index * 4;
            bytes[byte_index..byte_index + 4].copy_from_slice(&value.to_le_bytes());
        }

        Ok(bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreviewRenderOutputInfo {
    pub input_dimensions: FrameDimensions,
    pub output_dimensions: FrameDimensions,
    pub texture_format: PreviewTextureFormat,
    pub transfer_mode: PreviewTransferMode,
    pub input_pixel_count: usize,
    pub output_pixel_count: usize,
    pub texture_bytes: u64,
    pub no_full_frame_readback: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuPreviewRenderEncodeOutput {
    pub info: PreviewRenderOutputInfo,
    pub stats: PreviewRenderStats,
}

pub struct GpuP999HistogramInput<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub input_buffer: &'a wgpu::Buffer,
    pub input_buffer_bytes: u64,
    pub params: GpuP999HistogramParams,
    pub config: GpuP999HistogramConfig,
    pub extra_scale: f32,
}

/// Display-only p99.9 luminance histogram stage used by live preview tone
/// policy. It owns no rawvideo output or full-frame RGB readback path.
pub struct GpuDisplayP999Histogram {
    p999_histogram_pipeline: wgpu::ComputePipeline,
    p999_histogram_bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: Option<GpuSizedBuffer>,
    p999_histogram_buffer: Option<GpuSizedBuffer>,
    p999_histogram_readback_buffer: Option<GpuSizedBuffer>,
    max_storage_buffer_binding_size: u64,
    min_storage_buffer_offset_alignment: u64,
    max_compute_workgroups_per_dimension: u64,
}

impl GpuDisplayP999Histogram {
    pub fn new(device: &wgpu::Device) -> Result<Self, DisplayRenderError> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mcraw4vulkan display RGB shader"),
            source: wgpu::ShaderSource::Wgsl(DISPLAY_RGB_WGSL.into()),
        });
        let p999_histogram_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mcraw4vulkan display p999 luminance histogram pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("p999_histogram_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let p999_histogram_bind_group_layout = p999_histogram_pipeline.get_bind_group_layout(0);

        let limits = device.limits();

        Ok(Self {
            p999_histogram_pipeline,
            p999_histogram_bind_group_layout,
            params_buffer: None,
            p999_histogram_buffer: None,
            p999_histogram_readback_buffer: None,
            max_storage_buffer_binding_size: u64::from(limits.max_storage_buffer_binding_size),
            min_storage_buffer_offset_alignment: u64::from(
                limits.min_storage_buffer_offset_alignment.max(1),
            ),
            max_compute_workgroups_per_dimension: u64::from(
                limits.max_compute_workgroups_per_dimension.max(1),
            ),
        })
    }

    pub fn compute_p999_luminance_histogram(
        &mut self,
        input: GpuP999HistogramInput<'_>,
    ) -> Result<GpuP999HistogramResult, DisplayRenderError> {
        let GpuP999HistogramInput {
            device,
            queue,
            input_buffer,
            input_buffer_bytes,
            params,
            config,
            extra_scale,
        } = input;

        config.validate()?;
        if !extra_scale.is_finite() || extra_scale < 0.0 {
            return Err(DisplayRenderError::InvalidParams(
                "p999 extra exposure scale must be finite and non-negative".to_string(),
            ));
        }
        if (config.luma_min - P999_HISTOGRAM_LUMA_MIN).abs() > f32::EPSILON
            || (config.luma_max - P999_HISTOGRAM_LUMA_MAX).abs() > f32::EPSILON
        {
            return Err(DisplayRenderError::InvalidParams(
                "p999 histogram shader currently requires the canonical 0..8 luma range"
                    .to_string(),
            ));
        }
        let pixel_count = params.validate()?;
        let required_input_bytes = packed_u16_input_byte_len(pixel_count)?;
        if input_buffer_bytes < required_input_bytes {
            return Err(DisplayRenderError::GpuBufferSizeMismatch {
                buffer: "packed Bayer u16 p999 histogram input",
                required_bytes: required_input_bytes,
                actual_bytes: input_buffer_bytes,
            });
        }
        if required_input_bytes > self.max_storage_buffer_binding_size {
            return Err(DisplayRenderError::Gpu(format!(
                "packed Bayer p999 histogram input binding requires {required_input_bytes} bytes, exceeding max_storage_buffer_binding_size={}",
                self.max_storage_buffer_binding_size
            )));
        }

        let bin_bytes = u64::from(config.bin_count).checked_mul(4).ok_or_else(|| {
            DisplayRenderError::InvalidParams("p999 histogram bin byte count overflow".to_string())
        })?;
        let stats_offset = align_to(bin_bytes, self.min_storage_buffer_offset_alignment.max(4));
        let stats_bytes = P999_HISTOGRAM_STATS_WORDS.checked_mul(4).ok_or_else(|| {
            DisplayRenderError::InvalidParams(
                "p999 histogram stats byte count overflow".to_string(),
            )
        })?;
        let histogram_readback_bytes = stats_offset.checked_add(stats_bytes).ok_or_else(|| {
            DisplayRenderError::InvalidParams(
                "p999 histogram readback byte count overflow".to_string(),
            )
        })?;
        let readback_bytes_usize = usize::try_from(histogram_readback_bytes).map_err(|_| {
            DisplayRenderError::InvalidParams(
                "p999 histogram readback byte count does not fit usize".to_string(),
            )
        })?;
        if histogram_readback_bytes > self.max_storage_buffer_binding_size {
            return Err(DisplayRenderError::Gpu(format!(
                "p999 histogram binding requires {histogram_readback_bytes} bytes, exceeding max_storage_buffer_binding_size={}",
                self.max_storage_buffer_binding_size
            )));
        }

        ensure_sized_buffer(
            device,
            &mut self.params_buffer,
            "mcraw4vulkan display p999 histogram params",
            RENDER_PARAMS_BYTE_LEN,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        ensure_sized_buffer(
            device,
            &mut self.p999_histogram_buffer,
            "mcraw4vulkan display p999 luminance histogram",
            histogram_readback_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        ensure_sized_buffer(
            device,
            &mut self.p999_histogram_readback_buffer,
            "mcraw4vulkan display p999 luminance histogram readback",
            histogram_readback_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let params_buffer = &self
            .params_buffer
            .as_ref()
            .expect("render params buffer was just ensured")
            .buffer;
        let histogram_buffer = &self
            .p999_histogram_buffer
            .as_ref()
            .expect("p999 histogram buffer was just ensured")
            .buffer;
        let histogram_readback_buffer = &self
            .p999_histogram_readback_buffer
            .as_ref()
            .expect("p999 histogram readback buffer was just ensured")
            .buffer;

        let total_workgroups = u64::try_from(pixel_count)
            .map_err(|_| DisplayRenderError::OutputByteCountOverflow {
                dimensions: params.dimensions,
            })?
            .div_ceil(P999_HISTOGRAM_WORKGROUP_THREADS);
        let dispatch_x = total_workgroups
            .min(self.max_compute_workgroups_per_dimension)
            .max(1);
        let dispatch_y = total_workgroups.div_ceil(dispatch_x).max(1);
        let uniform_bytes = params.to_uniform_bytes(pixel_count, dispatch_x)?;
        queue.write_buffer(params_buffer, 0, &uniform_bytes);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mcraw4vulkan display p999 luminance histogram bind group"),
            layout: &self.p999_histogram_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_binding(input_buffer, 0, required_input_bytes)?,
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffer_binding(params_buffer, 0, RENDER_PARAMS_BYTE_LEN)?,
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffer_binding(histogram_buffer, 0, bin_bytes)?,
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffer_binding(histogram_buffer, stats_offset, stats_bytes)?,
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mcraw4vulkan display p999 luminance histogram encoder"),
        });
        encoder.clear_buffer(histogram_buffer, 0, Some(histogram_readback_bytes));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mcraw4vulkan display p999 luminance histogram compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p999_histogram_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = u32::try_from(dispatch_x).map_err(|_| {
                DisplayRenderError::OutputByteCountOverflow {
                    dimensions: params.dimensions,
                }
            })?;
            let workgroups_y = u32::try_from(dispatch_y).map_err(|_| {
                DisplayRenderError::OutputByteCountOverflow {
                    dimensions: params.dimensions,
                }
            })?;
            if dispatch_y > self.max_compute_workgroups_per_dimension {
                return Err(DisplayRenderError::OutputByteCountOverflow {
                    dimensions: params.dimensions,
                });
            }
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        encoder.copy_buffer_to_buffer(
            histogram_buffer,
            0,
            histogram_readback_buffer,
            0,
            histogram_readback_bytes,
        );
        let submission_index = queue.submit(Some(encoder.finish()));

        let readback_slice = histogram_readback_buffer.slice(0..histogram_readback_bytes);
        let (sender, receiver) = mpsc::channel();
        readback_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::wait_for(submission_index));
        let map_result = receiver
            .recv_timeout(Duration::from_secs(30))
            .map_err(|error| DisplayRenderError::MapFailed(error.to_string()))?;
        map_result.map_err(|error| DisplayRenderError::MapFailed(format!("{error:?}")))?;

        let mapped = readback_slice.get_mapped_range();
        let result = p999_histogram_result_from_mapped(
            &mapped[..readback_bytes_usize],
            config,
            stats_offset,
            extra_scale,
        )?;
        drop(mapped);
        histogram_readback_buffer.unmap();
        Ok(result)
    }
}

pub struct GpuPreviewTextureRenderer {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    output_texture: Option<GpuSizedTexture>,
    params_buffer: Option<GpuSizedBuffer>,
}

impl GpuPreviewTextureRenderer {
    pub fn new(device: &wgpu::Device) -> Result<Self, DisplayRenderError> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mcraw4vulkan Bayer preview texture shader"),
            source: wgpu::ShaderSource::Wgsl(DISPLAY_RGB_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mcraw4vulkan Bayer preview texture pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("preview_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        Ok(Self {
            pipeline,
            bind_group_layout,
            output_texture: None,
            params_buffer: None,
        })
    }

    pub fn encode_render_to_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input_buffer: &wgpu::Buffer,
        input_buffer_bytes: u64,
        config: PreviewRenderConfig,
    ) -> Result<GpuPreviewRenderEncodeOutput, DisplayRenderError> {
        self.encode_render_to_texture_with_rgb_sink_policy(
            device,
            queue,
            encoder,
            input_buffer,
            input_buffer_bytes,
            config,
            PreviewRgbSinkPolicy::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_render_to_texture_with_rgb_sink_policy(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input_buffer: &wgpu::Buffer,
        input_buffer_bytes: u64,
        config: PreviewRenderConfig,
        rgb_sink_policy: PreviewRgbSinkPolicy,
    ) -> Result<GpuPreviewRenderEncodeOutput, DisplayRenderError> {
        let info = config.validate_with_rgb_sink_policy(rgb_sink_policy)?;
        let required_input_bytes = packed_u16_input_byte_len(info.input_pixel_count)?;
        if input_buffer_bytes < required_input_bytes {
            return Err(DisplayRenderError::GpuBufferSizeMismatch {
                buffer: "packed Bayer u16 preview input",
                required_bytes: required_input_bytes,
                actual_bytes: input_buffer_bytes,
            });
        }

        let texture_create_start = Instant::now();
        ensure_sized_texture(
            device,
            &mut self.output_texture,
            "mcraw4vulkan rendered preview texture",
            info.output_dimensions,
            info.texture_format,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let texture_create = texture_create_start.elapsed();

        let params_upload_start = Instant::now();
        ensure_sized_buffer(
            device,
            &mut self.params_buffer,
            "mcraw4vulkan preview render params",
            RENDER_PARAMS_BYTE_LEN,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let params_buffer = &self
            .params_buffer
            .as_ref()
            .expect("preview params buffer was just ensured")
            .buffer;
        let uniform_bytes = config.to_uniform_bytes_with_rgb_sink_policy(info, rgb_sink_policy)?;
        queue.write_buffer(params_buffer, 0, &uniform_bytes);
        let params_upload = params_upload_start.elapsed();

        let output_view = &self
            .output_texture
            .as_ref()
            .expect("preview texture was just ensured")
            .view;

        let bind_group_start = Instant::now();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mcraw4vulkan Bayer preview texture bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_binding(input_buffer, 0, required_input_bytes)?,
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffer_binding(params_buffer, 0, RENDER_PARAMS_BYTE_LEN)?,
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
            ],
        });
        let bind_group_elapsed = bind_group_start.elapsed();

        let dispatch_encode_start = Instant::now();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mcraw4vulkan Bayer preview texture compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                info.output_dimensions.width.div_ceil(16),
                info.output_dimensions.height.div_ceil(16),
                1,
            );
        }
        let dispatch_encode = dispatch_encode_start.elapsed();

        Ok(GpuPreviewRenderEncodeOutput {
            info,
            stats: PreviewRenderStats {
                input_bytes: required_input_bytes,
                texture_bytes: info.texture_bytes,
                tile_count: 1,
                max_tile_pixels: u64::try_from(info.output_pixel_count).unwrap_or(u64::MAX),
                texture_create,
                params_upload,
                bind_group: bind_group_elapsed,
                dispatch_encode,
            },
        })
    }

    pub fn texture(&self) -> Result<&wgpu::Texture, DisplayRenderError> {
        self.output_texture
            .as_ref()
            .map(|texture| &texture.texture)
            .ok_or_else(|| DisplayRenderError::Gpu("preview texture is not allocated".to_string()))
    }

    pub fn texture_view(&self) -> Result<&wgpu::TextureView, DisplayRenderError> {
        self.output_texture
            .as_ref()
            .map(|texture| &texture.view)
            .ok_or_else(|| {
                DisplayRenderError::Gpu("preview texture view is not allocated".to_string())
            })
    }
}

struct GpuSizedTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    dimensions: FrameDimensions,
    format: PreviewTextureFormat,
}

struct GpuSizedBuffer {
    buffer: wgpu::Buffer,
    size: u64,
}

fn ensure_sized_buffer(
    device: &wgpu::Device,
    slot: &mut Option<GpuSizedBuffer>,
    label: &'static str,
    required_size: u64,
    usage: wgpu::BufferUsages,
) {
    let needs_allocation = slot
        .as_ref()
        .map(|existing| existing.size < required_size)
        .unwrap_or(true);

    if needs_allocation {
        *slot = Some(GpuSizedBuffer {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: required_size.max(1),
                usage,
                mapped_at_creation: false,
            }),
            size: required_size.max(1),
        });
    }
}

fn ensure_sized_texture(
    device: &wgpu::Device,
    slot: &mut Option<GpuSizedTexture>,
    label: &'static str,
    dimensions: FrameDimensions,
    format: PreviewTextureFormat,
    usage: wgpu::TextureUsages,
) {
    let needs_allocation = slot
        .as_ref()
        .map(|existing| existing.dimensions != dimensions || existing.format != format)
        .unwrap_or(true);

    if needs_allocation {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: dimensions.width,
                height: dimensions.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.wgpu_format(),
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        *slot = Some(GpuSizedTexture {
            texture,
            view,
            dimensions,
            format,
        });
    }
}

fn packed_u16_input_byte_len(pixel_count: usize) -> Result<u64, DisplayRenderError> {
    let word_count = pixel_count.div_ceil(2);
    let bytes = word_count.checked_mul(std::mem::size_of::<u32>()).ok_or(
        DisplayRenderError::OutputByteCountOverflow {
            dimensions: FrameDimensions {
                width: u32::MAX,
                height: u32::MAX,
            },
        },
    )?;
    u64::try_from(bytes).map_err(|_| DisplayRenderError::OutputByteCountOverflow {
        dimensions: FrameDimensions {
            width: u32::MAX,
            height: u32::MAX,
        },
    })
}

fn align_to_copy_bytes(value: u64) -> u64 {
    let alignment = wgpu::COPY_BUFFER_ALIGNMENT;
    value.div_ceil(alignment) * alignment
}

fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

fn buffer_binding<'a>(
    buffer: &'a wgpu::Buffer,
    offset: u64,
    size: u64,
) -> Result<wgpu::BindingResource<'a>, DisplayRenderError> {
    let size = wgpu::BufferSize::new(size).ok_or_else(|| {
        DisplayRenderError::Gpu("render buffer binding size must be non-zero".to_string())
    })?;
    Ok(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset,
        size: Some(size),
    }))
}

pub fn bayer_pattern_shader_value(pattern: BayerPattern) -> u32 {
    match pattern {
        BayerPattern::Rggb => 0,
        BayerPattern::Bggr => 1,
        BayerPattern::Grbg => 2,
        BayerPattern::Gbrg => 3,
    }
}

fn bayer_pattern_to_shader_value(pattern: BayerPattern) -> u32 {
    bayer_pattern_shader_value(pattern)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayerColor {
    Red,
    Green,
    Blue,
}

pub fn cfa_color_at(pattern: BayerPattern, x: u32, y: u32) -> BayerColor {
    let position = ((y & 1) * 2) + (x & 1);
    match pattern {
        BayerPattern::Rggb => match position {
            0 => BayerColor::Red,
            3 => BayerColor::Blue,
            _ => BayerColor::Green,
        },
        BayerPattern::Bggr => match position {
            0 => BayerColor::Blue,
            3 => BayerColor::Red,
            _ => BayerColor::Green,
        },
        BayerPattern::Grbg => match position {
            1 => BayerColor::Red,
            2 => BayerColor::Blue,
            _ => BayerColor::Green,
        },
        BayerPattern::Gbrg => match position {
            1 => BayerColor::Blue,
            2 => BayerColor::Red,
            _ => BayerColor::Green,
        },
    }
}

pub fn white_balance_from_as_shot_neutral(as_shot_neutral: Option<[f64; 3]>) -> [f32; 3] {
    let Some(neutral) = as_shot_neutral else {
        return [1.0, 1.0, 1.0];
    };
    if neutral
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return [1.0, 1.0, 1.0];
    }
    [
        (neutral[1] / neutral[0]) as f32,
        1.0,
        (neutral[1] / neutral[2]) as f32,
    ]
}

pub fn srgb_oetf(linear: f32) -> f32 {
    let linear = linear.clamp(0.0, 1.0);
    if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

pub fn bradford_d50_to_d65_matrix() -> [f64; 9] {
    BRADFORD_D50_TO_D65
}

pub fn xyz_d65_to_linear_srgb_matrix() -> [f64; 9] {
    XYZ_D65_TO_LINEAR_SRGB
}

fn select_camera_to_xyz_d50(
    metadata: GpuRenderColorMetadata,
) -> Result<CameraToXyzSelection, DisplayRenderError> {
    if let Some((matrix, source, illuminant)) = select_forward_matrix(metadata) {
        return Ok(CameraToXyzSelection {
            camera_to_xyz_d50: matrix,
            source,
            illuminant,
        });
    }

    if let Some((matrix, source, illuminant)) = select_color_matrix(metadata) {
        let camera_to_xyz_d50 = invert_3x3(matrix).ok_or_else(|| {
            DisplayRenderError::InvalidParams(format!(
                "{} is singular; ColorMatrix direction is XYZ to camera and must be inverted",
                source.label()
            ))
        })?;
        return Ok(CameraToXyzSelection {
            camera_to_xyz_d50,
            source,
            illuminant,
        });
    }

    Ok(CameraToXyzSelection {
        camera_to_xyz_d50: IDENTITY_3X3,
        source: GpuRenderColorMatrixSource::IdentityFallback,
        illuminant: None,
    })
}

fn select_forward_matrix(
    metadata: GpuRenderColorMetadata,
) -> Option<(
    [f64; 9],
    GpuRenderColorMatrixSource,
    Option<GpuRenderCalibrationIlluminant>,
)> {
    select_d65_matrix(
        metadata.forward_matrix1,
        GpuRenderColorMatrixSource::ForwardMatrix1,
        metadata.illuminant1,
        metadata.forward_matrix2,
        GpuRenderColorMatrixSource::ForwardMatrix2,
        metadata.illuminant2,
    )
    .or_else(|| {
        metadata
            .forward_matrix2
            .map(|matrix| {
                (
                    matrix,
                    GpuRenderColorMatrixSource::ForwardMatrix2,
                    metadata.illuminant2,
                )
            })
            .or_else(|| {
                metadata.forward_matrix1.map(|matrix| {
                    (
                        matrix,
                        GpuRenderColorMatrixSource::ForwardMatrix1,
                        metadata.illuminant1,
                    )
                })
            })
    })
}

fn select_color_matrix(
    metadata: GpuRenderColorMetadata,
) -> Option<(
    [f64; 9],
    GpuRenderColorMatrixSource,
    Option<GpuRenderCalibrationIlluminant>,
)> {
    select_d65_matrix(
        metadata.color_matrix1,
        GpuRenderColorMatrixSource::InverseColorMatrix1,
        metadata.illuminant1,
        metadata.color_matrix2,
        GpuRenderColorMatrixSource::InverseColorMatrix2,
        metadata.illuminant2,
    )
    .or_else(|| {
        metadata
            .color_matrix2
            .map(|matrix| {
                (
                    matrix,
                    GpuRenderColorMatrixSource::InverseColorMatrix2,
                    metadata.illuminant2,
                )
            })
            .or_else(|| {
                metadata.color_matrix1.map(|matrix| {
                    (
                        matrix,
                        GpuRenderColorMatrixSource::InverseColorMatrix1,
                        metadata.illuminant1,
                    )
                })
            })
    })
}

fn select_d65_matrix(
    matrix1: Option<[f64; 9]>,
    source1: GpuRenderColorMatrixSource,
    illuminant1: Option<GpuRenderCalibrationIlluminant>,
    matrix2: Option<[f64; 9]>,
    source2: GpuRenderColorMatrixSource,
    illuminant2: Option<GpuRenderCalibrationIlluminant>,
) -> Option<(
    [f64; 9],
    GpuRenderColorMatrixSource,
    Option<GpuRenderCalibrationIlluminant>,
)> {
    if illuminant1 == Some(GpuRenderCalibrationIlluminant::D65) {
        if let Some(matrix) = matrix1 {
            return Some((matrix, source1, illuminant1));
        }
    }
    if illuminant2 == Some(GpuRenderCalibrationIlluminant::D65) {
        if let Some(matrix) = matrix2 {
            return Some((matrix, source2, illuminant2));
        }
    }
    None
}

fn multiply_3x3(a: [f64; 9], b: [f64; 9]) -> [f64; 9] {
    let mut out = [0.0_f64; 9];
    for row in 0..3 {
        for col in 0..3 {
            out[row * 3 + col] =
                a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col];
        }
    }
    out
}

fn invert_3x3(matrix: [f64; 9]) -> Option<[f64; 9]> {
    let m = matrix;
    let c00 = m[4] * m[8] - m[5] * m[7];
    let c01 = -(m[3] * m[8] - m[5] * m[6]);
    let c02 = m[3] * m[7] - m[4] * m[6];
    let c10 = -(m[1] * m[8] - m[2] * m[7]);
    let c11 = m[0] * m[8] - m[2] * m[6];
    let c12 = -(m[0] * m[7] - m[1] * m[6]);
    let c20 = m[1] * m[5] - m[2] * m[4];
    let c21 = -(m[0] * m[5] - m[2] * m[3]);
    let c22 = m[0] * m[4] - m[1] * m[3];
    let determinant = m[0] * c00 + m[1] * c01 + m[2] * c02;

    if !determinant.is_finite() || determinant.abs() < 1.0e-12 {
        return None;
    }

    let inv_det = 1.0 / determinant;
    Some([
        c00 * inv_det,
        c10 * inv_det,
        c20 * inv_det,
        c01 * inv_det,
        c11 * inv_det,
        c21 * inv_det,
        c02 * inv_det,
        c12 * inv_det,
        c22 * inv_det,
    ])
}

fn f64_matrix_to_f32(matrix: [f64; 9]) -> [f32; 9] {
    matrix.map(|value| value as f32)
}

fn p999_histogram_result_from_mapped(
    mapped: &[u8],
    config: GpuP999HistogramConfig,
    stats_offset: u64,
    extra_scale: f32,
) -> Result<GpuP999HistogramResult, DisplayRenderError> {
    let bin_count = usize::try_from(config.bin_count).map_err(|_| {
        DisplayRenderError::InvalidParams("p999 histogram bin count does not fit usize".to_string())
    })?;
    let bin_bytes = bin_count.checked_mul(4).ok_or_else(|| {
        DisplayRenderError::InvalidParams("p999 histogram bin byte count overflow".to_string())
    })?;
    let stats_offset = usize::try_from(stats_offset).map_err(|_| {
        DisplayRenderError::InvalidParams(
            "p999 histogram stats offset does not fit usize".to_string(),
        )
    })?;
    let stats_end = stats_offset.checked_add(4).ok_or_else(|| {
        DisplayRenderError::InvalidParams("p999 histogram stats offset overflow".to_string())
    })?;
    if mapped.len() < bin_bytes || mapped.len() < stats_end {
        return Err(DisplayRenderError::GpuBufferSizeMismatch {
            buffer: "p999 histogram readback",
            required_bytes: u64::try_from(stats_end.max(bin_bytes)).unwrap_or(u64::MAX),
            actual_bytes: u64::try_from(mapped.len()).unwrap_or(u64::MAX),
        });
    }

    let mut total_samples = 0_u64;
    let mut cumulative = 0_u64;
    let mut p999_bin = 0_usize;
    let mut max_nonzero_bin = None;
    let mut bins = mapped[..bin_bytes].chunks_exact(4).enumerate();
    for (index, bytes) in bins.by_ref() {
        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if count != 0 {
            max_nonzero_bin = Some(index);
        }
        total_samples = total_samples.saturating_add(u64::from(count));
    }
    if total_samples == 0 {
        return Ok(GpuP999HistogramResult {
            luma_p999: 0.0,
            scale_p999: config.max_scale,
            final_scale: config.max_scale * extra_scale,
            total_samples: 0,
            overflow_count: 0,
            max_luma_observed: 0.0,
            histogram_bins: config.bin_count,
            readback_bytes: mapped.len(),
        });
    }

    let target_rank = ((total_samples as f64) * f64::from(config.target_percentile))
        .ceil()
        .max(1.0) as u64;
    for (index, bytes) in mapped[..bin_bytes].chunks_exact(4).enumerate() {
        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        cumulative = cumulative.saturating_add(u64::from(count));
        if cumulative >= target_rank {
            p999_bin = index;
            break;
        }
    }

    let bin_width = (config.luma_max - config.luma_min) / config.bin_count as f32;
    let luma_p999 = (config.luma_min + ((p999_bin as f32) + 1.0) * bin_width)
        .clamp(config.luma_min, config.luma_max);
    let max_luma_observed = max_nonzero_bin
        .map(|index| (config.luma_min + ((index as f32) + 1.0) * bin_width).min(config.luma_max))
        .unwrap_or(0.0);
    let overflow_count = u32::from_le_bytes([
        mapped[stats_offset],
        mapped[stats_offset + 1],
        mapped[stats_offset + 2],
        mapped[stats_offset + 3],
    ]);
    let scale_p999 = p999_scale_for_luma(luma_p999, config);
    Ok(GpuP999HistogramResult {
        luma_p999,
        scale_p999,
        final_scale: scale_p999 * extra_scale,
        total_samples: u32::try_from(total_samples).unwrap_or(u32::MAX),
        overflow_count,
        max_luma_observed,
        histogram_bins: config.bin_count,
        readback_bytes: mapped.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preview_test_config() -> PreviewRenderConfig {
        PreviewRenderConfig {
            dimensions: FrameDimensions {
                width: 640,
                height: 360,
            },
            bayer_pattern: BayerPattern::Gbrg,
            source_bits: 10,
            black_level: [0.0; 4],
            white_level: 1023.0,
            sample_domain: RenderSampleDomain::raw(1023.0),
            color: GpuRenderColorParams::from_metadata(
                GpuRenderColorMode::MetadataSrgb,
                None,
                GpuRenderColorMetadata::default(),
            )
            .expect("preview color params"),
            texture_format: PreviewTextureFormat::Rgba8Unorm,
            transfer_mode: PreviewTransferMode::ShaderSrgb,
            scale_mode: PreviewScaleMode::FullResolution,
        }
    }

    #[test]
    fn preview_texture_format_labels_parse() {
        assert_eq!(
            PreviewTextureFormat::parse_label("rgba8unorm"),
            Some(PreviewTextureFormat::Rgba8Unorm)
        );
        assert_eq!(
            PreviewTextureFormat::parse_label("rgba8unorm-srgb"),
            Some(PreviewTextureFormat::Rgba8UnormSrgb)
        );
        assert_eq!(
            PreviewTextureFormat::parse_label("bgra8unorm"),
            Some(PreviewTextureFormat::Bgra8Unorm)
        );
        assert_eq!(
            PreviewTextureFormat::parse_label("bgra8unorm-srgb"),
            Some(PreviewTextureFormat::Bgra8UnormSrgb)
        );
        assert_eq!(PreviewTextureFormat::Rgba8Unorm.bytes_per_pixel(), 4);
    }

    #[test]
    fn preview_scale_modes_resolve_dimensions() {
        let input = FrameDimensions {
            width: 4032,
            height: 3024,
        };
        assert_eq!(
            PreviewScaleMode::FullResolution.resolve(input).unwrap(),
            input
        );
        assert_eq!(
            PreviewScaleMode::Half.resolve(input).unwrap(),
            FrameDimensions {
                width: 2016,
                height: 1512
            }
        );
        assert_eq!(
            PreviewScaleMode::Quarter.resolve(input).unwrap(),
            FrameDimensions {
                width: 1008,
                height: 756
            }
        );
        assert_eq!(
            PreviewScaleMode::Explicit {
                width: 1920,
                height: 1080
            }
            .resolve(input)
            .unwrap(),
            FrameDimensions {
                width: 1920,
                height: 1080
            }
        );
    }

    #[test]
    fn preview_config_reports_texture_bytes() {
        let config = PreviewRenderConfig {
            scale_mode: PreviewScaleMode::Explicit {
                width: 320,
                height: 180,
            },
            ..preview_test_config()
        };
        let info = config.validate().expect("preview config validates");
        assert_eq!(
            info.output_dimensions,
            FrameDimensions {
                width: 320,
                height: 180
            }
        );
        assert_eq!(info.texture_bytes, 320 * 180 * 4);
        assert!(info.no_full_frame_readback);
    }

    #[test]
    fn preview_shader_srgb_sets_uniform_transfer_flag() {
        let config = PreviewRenderConfig {
            transfer_mode: PreviewTransferMode::ShaderSrgb,
            ..preview_test_config()
        };
        let info = config.validate().expect("preview config validates");
        let bytes = config.to_uniform_bytes(info).expect("uniform bytes");
        let apply = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        assert_eq!(apply, 1);
    }

    #[test]
    fn preview_hardware_srgb_disables_uniform_transfer_flag() {
        let config = PreviewRenderConfig {
            transfer_mode: PreviewTransferMode::HardwareSrgb,
            ..preview_test_config()
        };
        let info = config.validate().expect("preview config validates");
        let bytes = config.to_uniform_bytes(info).expect("uniform bytes");
        let apply = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        assert_eq!(apply, 0);
    }

    #[test]
    fn preview_raw_domain_serializes_no_highlight_rolloff() {
        let config = preview_test_config();
        let info = config.validate().expect("preview config validates");
        let bytes = config.to_uniform_bytes(info).expect("uniform bytes");
        let highlight_mode = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        let sample_limit = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        assert_eq!(highlight_mode, 0);
        assert_eq!(sample_limit, 1023);
    }

    #[test]
    fn preview_motioncam_domain_serializes_highlight_rolloff_and_sample_limit() {
        let config = PreviewRenderConfig {
            white_level: 4095.0,
            sample_domain: RenderSampleDomain::motioncam_compatible_pixel_v1(4095.0),
            ..preview_test_config()
        };
        let info = config.validate().expect("preview config validates");
        let bytes = config.to_uniform_bytes(info).expect("uniform bytes");
        let highlight_mode = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        let sample_limit = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        assert_eq!(highlight_mode, 1);
        assert_eq!(sample_limit, 16_383);
    }

    #[test]
    fn preview_rgb_sink_policy_serializes_tone_guard_and_desat() {
        let config = preview_test_config();
        let policy = PreviewRgbSinkPolicy {
            rgb_sink_tone_scale: 0.5,
            rgb_sink_guard: GpuRgbSinkGuard {
                mode: GpuRgbSinkGuardMode::RbSumLimit,
                highlight_luma: 0.75,
                r_ratio: 0.0,
                b_ratio: 0.0,
            },
            rgb_sink_desat: GpuRgbSinkHighlightDesat {
                enabled: true,
                start_luma: 0.25,
                end_luma: 0.75,
                strength: 0.6,
            },
        };
        let info = config
            .validate_with_rgb_sink_policy(policy)
            .expect("preview config validates");
        let bytes = config
            .to_uniform_bytes_with_rgb_sink_policy(info, policy)
            .expect("uniform bytes");
        let read_f32 = |word_index: usize| {
            let offset = word_index * 4;
            f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        };

        assert_eq!(read_f32(32), 0.5);
        assert_eq!(read_f32(36), 1.0);
        assert_eq!(read_f32(37), 0.75);
        assert_eq!(read_f32(40), 1.0);
        assert_eq!(read_f32(41), 0.25);
        assert_eq!(read_f32(42), 0.75);
        assert_eq!(read_f32(43), 0.6);
    }

    #[test]
    fn preview_default_rgb_sink_policy_serializes_as_noop() {
        let config = preview_test_config();
        let info = config.validate().expect("preview config validates");
        let bytes = config.to_uniform_bytes(info).expect("uniform bytes");
        let read_f32 = |word_index: usize| {
            let offset = word_index * 4;
            f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        };

        assert_eq!(read_f32(32), 1.0);
        assert_eq!(read_f32(36), 0.0);
        assert_eq!(read_f32(40), 0.0);
    }

    #[test]
    fn preview_sample_domain_rejects_mismatched_nominal_white() {
        let config = PreviewRenderConfig {
            white_level: 4095.0,
            sample_domain: RenderSampleDomain::raw(1023.0),
            ..preview_test_config()
        };
        let error = config
            .validate()
            .expect_err("mismatched sample domain rejects");
        assert!(error.to_string().contains("nominal white"));
    }

    #[test]
    fn motioncam_pixel_sample_limit_tracks_overwhite_domain() {
        assert_eq!(motioncam_pixel_v1_display_sample_limit(4095.0), 16_383.0);
        assert_eq!(motioncam_pixel_v1_display_sample_limit(1023.0), 4095.0);
        assert_eq!(motioncam_pixel_v1_display_sample_limit(20_000.0), 65_535.0);
    }

    #[test]
    fn hue_preserving_rolloff_is_identity_below_white() {
        let rgb = [0.25, 0.5, 0.75];
        assert_eq!(hue_preserving_highlight_rolloff(rgb), rgb);
    }

    #[test]
    fn hue_preserving_rolloff_scales_overwhite_without_hue_shift() {
        let rolled = hue_preserving_highlight_rolloff([2.0, 1.0, 0.5]);
        assert_eq!(rolled, [1.0, 0.5, 0.25]);
    }

    #[test]
    fn hue_preserving_rolloff_clamps_negative_and_stays_finite() {
        let rolled = hue_preserving_highlight_rolloff([1000.0, -1.0, 250.0]);
        assert!(rolled.into_iter().all(f32::is_finite));
        assert_eq!(rolled, [1.0, 0.0, 0.25]);
    }

    #[test]
    fn preview_compute_path_rejects_non_rgba8_storage_formats() {
        let config = PreviewRenderConfig {
            texture_format: PreviewTextureFormat::Rgba8UnormSrgb,
            ..preview_test_config()
        };
        let error = config
            .validate()
            .expect_err("sRGB storage texture is rejected");
        assert!(error.to_string().contains("rgba8unorm"));
    }

    #[test]
    fn display_rgb_tone_policy_preserves_preview_values() {
        let policy = DisplayRgbTonePolicy::canonical();
        assert_eq!(policy.extra_scale(false), 1.0);
        assert!((policy.extra_scale(true) - 0.707_106_77).abs() < 1.0e-7);
        assert_eq!(
            policy.render_guard(0.5).mode,
            GpuRgbSinkGuardMode::RbSumLimit
        );
    }

    #[test]
    fn p999_scale_clamps_to_canonical_range() {
        let config = GpuP999HistogramConfig::canonical();
        assert!((p999_scale_for_luma(2.0, config) - 0.48).abs() < 1.0e-6);
        assert_eq!(p999_scale_for_luma(0.0, config), 8.0);
        assert_eq!(p999_scale_for_luma(1000.0, config), 1.0 / 64.0);
    }

    #[test]
    fn rb_sum_guard_scales_red_and_blue_only() {
        assert_eq!(apply_rb_sum_guard_rgb([0.4, 0.5, 0.6]), [0.4, 0.5, 0.6]);
        let guarded = apply_rb_sum_guard_rgb([2.0, 0.5, 2.0]);
        assert!((guarded[0] - 0.5).abs() < 1.0e-6);
        assert_eq!(guarded[1], 0.5);
        assert!((guarded[2] - 0.5).abs() < 1.0e-6);
        assert!(((guarded[0] + guarded[2]) - (2.0 * guarded[1])).abs() < 1.0e-6);
    }

    #[test]
    fn synthetic_histogram_returns_expected_upper_edge_p999() {
        let config = GpuP999HistogramConfig {
            bin_count: 4,
            luma_min: 0.0,
            luma_max: 8.0,
            target_percentile: 0.999,
            target_output_luma: 0.96,
            min_scale: 1.0 / 64.0,
            max_scale: 8.0,
        };
        let mut mapped = Vec::new();
        for count in [1_u32, 2, 997, 0] {
            mapped.extend_from_slice(&count.to_le_bytes());
        }
        mapped.extend_from_slice(&7_u32.to_le_bytes());
        let result = p999_histogram_result_from_mapped(&mapped, config, 16, 0.707_106_77).unwrap();
        assert_eq!(result.total_samples, 1000);
        assert_eq!(result.overflow_count, 7);
        assert_eq!(result.histogram_bins, 4);
        assert_eq!(result.readback_bytes, 20);
        assert!((result.luma_p999 - 6.0).abs() < 1.0e-6);
        assert!((result.scale_p999 - 0.16).abs() < 1.0e-6);
        assert!((result.final_scale - (0.16 * 0.707_106_77)).abs() < 1.0e-6);
    }

    #[test]
    fn cfa_mapping_matches_bayer_names() {
        assert_eq!(cfa_color_at(BayerPattern::Rggb, 0, 0), BayerColor::Red);
        assert_eq!(cfa_color_at(BayerPattern::Rggb, 1, 1), BayerColor::Blue);
        assert_eq!(cfa_color_at(BayerPattern::Bggr, 0, 0), BayerColor::Blue);
        assert_eq!(cfa_color_at(BayerPattern::Bggr, 1, 1), BayerColor::Red);
        assert_eq!(cfa_color_at(BayerPattern::Grbg, 1, 0), BayerColor::Red);
        assert_eq!(cfa_color_at(BayerPattern::Gbrg, 1, 0), BayerColor::Blue);
    }

    #[test]
    fn canonical_cfa_tiles_match_dng_pattern_semantics() {
        assert_cfa_tile(
            BayerPattern::Rggb,
            [
                BayerColor::Red,
                BayerColor::Green,
                BayerColor::Green,
                BayerColor::Blue,
            ],
        );
        assert_cfa_tile(
            BayerPattern::Gbrg,
            [
                BayerColor::Green,
                BayerColor::Blue,
                BayerColor::Red,
                BayerColor::Green,
            ],
        );
        assert_cfa_tile(
            BayerPattern::Grbg,
            [
                BayerColor::Green,
                BayerColor::Red,
                BayerColor::Blue,
                BayerColor::Green,
            ],
        );
        assert_cfa_tile(
            BayerPattern::Bggr,
            [
                BayerColor::Blue,
                BayerColor::Green,
                BayerColor::Green,
                BayerColor::Red,
            ],
        );
    }

    #[test]
    fn shader_numeric_cfa_enum_matches_rust_mapping() {
        assert_eq!(bayer_pattern_shader_value(BayerPattern::Rggb), 0);
        assert_eq!(bayer_pattern_shader_value(BayerPattern::Bggr), 1);
        assert_eq!(bayer_pattern_shader_value(BayerPattern::Grbg), 2);
        assert_eq!(bayer_pattern_shader_value(BayerPattern::Gbrg), 3);
    }

    #[test]
    fn bigfile_parsed_gbrg_semantics_are_top_left_green_top_right_blue() {
        assert_eq!(cfa_color_at(BayerPattern::Gbrg, 0, 0), BayerColor::Green);
        assert_eq!(cfa_color_at(BayerPattern::Gbrg, 1, 0), BayerColor::Blue);
        assert_eq!(cfa_color_at(BayerPattern::Gbrg, 0, 1), BayerColor::Red);
        assert_eq!(cfa_color_at(BayerPattern::Gbrg, 1, 1), BayerColor::Green);
    }

    #[test]
    fn white_balance_uses_green_normalized_as_shot_neutral() {
        assert_eq!(
            white_balance_from_as_shot_neutral(Some([0.5, 1.0, 0.25])),
            [2.0, 1.0, 4.0]
        );
        assert_eq!(white_balance_from_as_shot_neutral(None), [1.0; 3]);
        assert_eq!(
            white_balance_from_as_shot_neutral(Some([0.0, 1.0, 0.25])),
            [1.0; 3]
        );
    }

    #[test]
    fn srgb_oetf_matches_reference_values() {
        assert!((srgb_oetf(0.0) - 0.0).abs() < 0.000_001);
        assert!((srgb_oetf(0.003_130_8) - 0.040_449_9).abs() < 0.000_01);
        assert!((srgb_oetf(1.0) - 1.0).abs() < 0.000_001);
    }

    #[test]
    fn adaptation_and_srgb_matrices_are_finite() {
        assert!(bradford_d50_to_d65_matrix().into_iter().all(f64::is_finite));
        assert!(
            xyz_d65_to_linear_srgb_matrix()
                .into_iter()
                .all(f64::is_finite)
        );
    }

    #[test]
    fn matrix_inverse_uses_color_matrix_fallback_direction() {
        let inverse = invert_3x3([
            2.0, 0.0, 0.0, //
            0.0, 4.0, 0.0, //
            0.0, 0.0, 8.0,
        ])
        .expect("matrix inverts");

        assert!((inverse[0] - 0.5).abs() < 0.000_001);
        assert!((inverse[4] - 0.25).abs() < 0.000_001);
        assert!((inverse[8] - 0.125).abs() < 0.000_001);
    }

    #[test]
    fn matrix_multiplication_uses_row_major_column_vector_order() {
        let scale = [
            2.0, 0.0, 0.0, //
            0.0, 3.0, 0.0, //
            0.0, 0.0, 5.0,
        ];
        let mix = [
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ];
        assert_eq!(
            multiply_3x3(scale, mix),
            [
                2.0, 4.0, 6.0, //
                12.0, 15.0, 18.0, //
                35.0, 40.0, 45.0,
            ]
        );
    }

    #[test]
    fn bradford_d50_to_d65_maps_d50_white_near_d65_white() {
        let d50_white = [0.96422, 1.0, 0.82521];
        let adapted = multiply_matrix_vec3(BRADFORD_D50_TO_D65, d50_white);
        let d65_white = [0.95047, 1.0, 1.08883];
        for index in 0..3 {
            assert!((adapted[index] - d65_white[index]).abs() < 0.001);
        }
    }

    #[test]
    fn final_matrix_is_finite_and_deterministic_for_synthetic_metadata() {
        let metadata = GpuRenderColorMetadata {
            forward_matrix2: Some([
                0.9, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.1,
            ]),
            illuminant2: Some(GpuRenderCalibrationIlluminant::D65),
            ..GpuRenderColorMetadata::default()
        };
        let first = GpuRenderColorParams::from_metadata(
            GpuRenderColorMode::MetadataSrgb,
            Some([0.5, 1.0, 0.25]),
            metadata,
        )
        .expect("first color params build");
        let second = GpuRenderColorParams::from_metadata(
            GpuRenderColorMode::MetadataSrgb,
            Some([0.5, 1.0, 0.25]),
            metadata,
        )
        .expect("second color params build");

        assert_eq!(first.camera_to_srgb, second.camera_to_srgb);
        assert!(first.camera_to_srgb.into_iter().all(f32::is_finite));
        assert!(first.apply_srgb_transfer);
    }

    #[test]
    fn row_major_uniform_serialization_matches_shader_dot_rows() {
        let color = GpuRenderColorParams::from_metadata(
            GpuRenderColorMode::MetadataSrgb,
            None,
            GpuRenderColorMetadata::default(),
        )
        .expect("default params build");
        let params = GpuP999HistogramParams {
            dimensions: FrameDimensions {
                width: 2,
                height: 2,
            },
            bayer_pattern: BayerPattern::Rggb,
            source_bits: 10,
            black_level: [0.0; 4],
            white_level: 1023.0,
            sample_limit: 4095.0,
            color: GpuRenderColorParams {
                camera_to_srgb: [
                    1.0, 2.0, 3.0, //
                    4.0, 5.0, 6.0, //
                    7.0, 8.0, 9.0,
                ],
                ..color
            },
        };
        params.validate().expect("params validate");
        let bytes = params.to_uniform_bytes(4, 1).expect("uniform bytes build");
        let read_f32 = |index: usize| {
            let offset = index * 4;
            f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        };
        let read_u32 = |index: usize| {
            let offset = index * 4;
            u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        };

        assert_eq!([read_f32(20), read_f32(21), read_f32(22)], [1.0, 2.0, 3.0]);
        assert_eq!(read_f32(23), 4095.0);
        assert_eq!([read_f32(24), read_f32(25), read_f32(26)], [4.0, 5.0, 6.0]);
        assert_eq!([read_f32(28), read_f32(29), read_f32(30)], [7.0, 8.0, 9.0]);
        assert_eq!(read_u32(11), 1 | (2 << 8));
    }

    #[test]
    fn forward_matrix_is_preferred_over_color_matrix() {
        let color = GpuRenderColorParams::from_metadata(
            GpuRenderColorMode::MetadataSrgb,
            Some([0.5, 1.0, 0.25]),
            GpuRenderColorMetadata {
                color_matrix2: Some([
                    2.0, 0.0, 0.0, //
                    0.0, 2.0, 0.0, //
                    0.0, 0.0, 2.0,
                ]),
                forward_matrix2: Some(IDENTITY_3X3),
                illuminant2: Some(GpuRenderCalibrationIlluminant::D65),
                ..GpuRenderColorMetadata::default()
            },
        )
        .expect("color params build");

        assert_eq!(
            color.matrix_source,
            GpuRenderColorMatrixSource::ForwardMatrix2
        );
        assert_eq!(
            color.selected_illuminant,
            Some(GpuRenderCalibrationIlluminant::D65)
        );
        assert_eq!(color.white_balance_rgb, [2.0, 1.0, 4.0]);
        assert!(color.camera_to_srgb.into_iter().all(f32::is_finite));
        assert!(color.apply_srgb_transfer);
    }

    #[test]
    fn d65_associated_matrix_is_selected_first() {
        let color = GpuRenderColorParams::from_metadata(
            GpuRenderColorMode::MetadataSrgb,
            None,
            GpuRenderColorMetadata {
                forward_matrix1: Some([
                    2.0, 0.0, 0.0, //
                    0.0, 2.0, 0.0, //
                    0.0, 0.0, 2.0,
                ]),
                forward_matrix2: Some(IDENTITY_3X3),
                illuminant1: Some(GpuRenderCalibrationIlluminant::StandardA),
                illuminant2: Some(GpuRenderCalibrationIlluminant::D65),
                ..GpuRenderColorMetadata::default()
            },
        )
        .expect("color params build");

        assert_eq!(
            color.matrix_source,
            GpuRenderColorMatrixSource::ForwardMatrix2
        );
        assert_eq!(
            color.selected_illuminant,
            Some(GpuRenderCalibrationIlluminant::D65)
        );
    }

    #[test]
    fn color_matrix_fallback_uses_inverse_not_direct_matrix() {
        let selection = select_camera_to_xyz_d50(GpuRenderColorMetadata {
            color_matrix2: Some([
                2.0, 0.0, 0.0, //
                0.0, 4.0, 0.0, //
                0.0, 0.0, 8.0,
            ]),
            ..GpuRenderColorMetadata::default()
        })
        .expect("selection builds");

        assert_eq!(
            selection.source,
            GpuRenderColorMatrixSource::InverseColorMatrix2
        );
        assert!((selection.camera_to_xyz_d50[0] - 0.5).abs() < 0.000_001);
        assert!((selection.camera_to_xyz_d50[4] - 0.25).abs() < 0.000_001);
        assert!((selection.camera_to_xyz_d50[8] - 0.125).abs() < 0.000_001);
    }

    #[test]
    fn metadata_srgb_is_default_color_mode() {
        assert_eq!(
            GpuRenderColorMode::default(),
            GpuRenderColorMode::MetadataSrgb
        );
        let color = GpuRenderColorParams::default();
        assert_eq!(color.mode, GpuRenderColorMode::MetadataSrgb);
        assert!(color.apply_srgb_transfer);
        assert_eq!(
            color.matrix_source,
            GpuRenderColorMatrixSource::IdentityFallback
        );
    }

    fn assert_cfa_tile(pattern: BayerPattern, expected: [BayerColor; 4]) {
        assert_eq!(cfa_color_at(pattern, 0, 0), expected[0]);
        assert_eq!(cfa_color_at(pattern, 1, 0), expected[1]);
        assert_eq!(cfa_color_at(pattern, 0, 1), expected[2]);
        assert_eq!(cfa_color_at(pattern, 1, 1), expected[3]);
    }

    fn multiply_matrix_vec3(matrix: [f64; 9], vector: [f64; 3]) -> [f64; 3] {
        [
            matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
            matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
            matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
        ]
    }
}
