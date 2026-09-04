use std::error::Error;
use std::fmt;

use mcraw4vulkan_core::FrameDimensions;
use mcraw4vulkan_mcrawcontainer::LensShadingMapValidationError;

#[derive(Debug, Clone, PartialEq)]
pub enum VignetteCorrectionError {
    MissingSourceBlackLevel,
    MissingSourceWhiteLevel,
    NonuniformSourceWhiteLevel,
    InvalidSourceWhiteLevel {
        value: f64,
    },
    NonIntegerSourceWhiteLevel {
        value: f64,
    },
    MotionCamSpatialMissingLensShadingMap,
    EmptyLensShadingMapDimensions,
    LensShadingMapDimensionsOverflow {
        width: u32,
        height: u32,
    },
    LensShadingPlaneLengthMismatch {
        plane_index: usize,
        expected_len: usize,
        actual_len: usize,
    },
    NonFiniteLensShadingGain {
        plane_index: usize,
        sample_index: usize,
    },
    NegativeLensShadingGain {
        plane_index: usize,
        sample_index: usize,
        value: f32,
    },
    MissingLensShadingMap,
    UnsupportedLensShadingPlaneCount {
        expected: usize,
        actual: usize,
    },
    LensShadingPlaneOutOfRange {
        plane_index: usize,
        plane_count: usize,
    },
    InvalidBlackLevel {
        index: usize,
        value: f32,
    },
    FixedPointGainOverflow {
        plane_index: usize,
        sample_index: usize,
        value: f32,
    },
    FixedPointBlackLevelOverflow {
        index: usize,
        value: f32,
    },
    FixedPointCoordinateOverflow {
        position: usize,
        frame_dimension: u32,
        map_dimension: usize,
    },
    FixedPointCorrectionOverflow,
    FullResolutionGainMapGainOverflow {
        x: usize,
        y: usize,
        value: i64,
    },
    InvalidFrameDimensions {
        dimensions: FrameDimensions,
    },
    FrameDimensionsOverflow {
        dimensions: FrameDimensions,
    },
    FrameDimensionsMismatch {
        frame_dimensions: FrameDimensions,
        facts_dimensions: FrameDimensions,
    },
    GpuBufferSizeMismatch {
        buffer: &'static str,
        required_bytes: u64,
        actual_bytes: u64,
    },
    GpuFrameDimensionsMismatch {
        frame_dimensions: FrameDimensions,
        gain_map_dimensions: FrameDimensions,
    },
    GpuGainMapPixelCountMismatch {
        expected_pixel_count: usize,
        actual_pixel_count: usize,
    },
    GpuGainMapBindingTooLarge {
        required_bytes: u64,
        max_binding_bytes: u64,
        frame_dimensions: FrameDimensions,
        pixel_count: usize,
    },
    GpuInvalidTilePlan {
        reason: &'static str,
    },
    GpuComputeWorkgroupLimitTooSmall {
        max_workgroups: u64,
        workgroup_size: u32,
    },
    GpuTileDispatchTooLarge {
        tile_packed_word_count: usize,
        workgroup_size: u32,
        required_workgroups: u64,
        max_workgroups: u64,
    },
    GpuTileOffsetAlignment {
        offset: u64,
        alignment: u64,
    },
    GpuTileSizeZero,
    GpuTilePixelCountOverflow,
    GpuFixedPointBlackLevelOverflow {
        index: usize,
        value: i64,
    },
    DecodedFrameValidationFailed(String),
}

impl fmt::Display for VignetteCorrectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSourceBlackLevel => formatter.write_str("missing black level"),
            Self::MissingSourceWhiteLevel => formatter.write_str("missing white level"),
            Self::NonuniformSourceWhiteLevel => formatter.write_str("nonuniform source white"),
            Self::InvalidSourceWhiteLevel { value } => write!(
                formatter,
                "source white level must be finite and within 1..={}, got {value:.17}",
                u16::MAX
            ),
            Self::NonIntegerSourceWhiteLevel { value } => write!(
                formatter,
                "present white level must be integer-like, got {value:.17}"
            ),
            Self::MotionCamSpatialMissingLensShadingMap => {
                formatter.write_str("MotionCamSpatial requires a present lens map")
            }
            Self::EmptyLensShadingMapDimensions => {
                formatter.write_str("lensShadingMap dimensions must be non-zero")
            }
            Self::LensShadingMapDimensionsOverflow { width, height } => write!(
                formatter,
                "lensShadingMap dimensions {width}x{height} overflow"
            ),
            Self::LensShadingPlaneLengthMismatch {
                plane_index,
                expected_len,
                actual_len,
            } => write!(
                formatter,
                "lensShadingMap[{plane_index}] len {actual_len} does not match width * height {expected_len}"
            ),
            Self::NonFiniteLensShadingGain {
                plane_index,
                sample_index,
            } => write!(
                formatter,
                "lensShadingMap[{plane_index}][{sample_index}] must be finite"
            ),
            Self::NegativeLensShadingGain {
                plane_index,
                sample_index,
                value,
            } => write!(
                formatter,
                "lensShadingMap[{plane_index}][{sample_index}] is negative: {value}"
            ),
            Self::MissingLensShadingMap => {
                formatter.write_str("vignette correction is enabled but lensShadingMap is missing")
            }
            Self::UnsupportedLensShadingPlaneCount { expected, actual } => write!(
                formatter,
                "vignette correction requires {expected} lensShadingMap planes, got {actual}"
            ),
            Self::LensShadingPlaneOutOfRange {
                plane_index,
                plane_count,
            } => write!(
                formatter,
                "lensShadingMap plane index {plane_index} is out of range for {plane_count} planes"
            ),
            Self::InvalidBlackLevel { index, value } => write!(
                formatter,
                "black level {index} must be finite and non-negative, got {value}"
            ),
            Self::FixedPointGainOverflow {
                plane_index,
                sample_index,
                value,
            } => write!(
                formatter,
                "lensShadingMap[{plane_index}][{sample_index}] cannot be represented in fixed-point gain format: {value}"
            ),
            Self::FixedPointBlackLevelOverflow { index, value } => write!(
                formatter,
                "black level {index} cannot be represented in fixed-point format: {value}"
            ),
            Self::FixedPointCoordinateOverflow {
                position,
                frame_dimension,
                map_dimension,
            } => write!(
                formatter,
                "fixed-point vignette coordinate overflow for position {position}, frame dimension {frame_dimension}, map dimension {map_dimension}"
            ),
            Self::FixedPointCorrectionOverflow => {
                formatter.write_str("fixed-point vignette correction overflow")
            }
            Self::FullResolutionGainMapGainOverflow { x, y, value } => write!(
                formatter,
                "full-resolution fixed vignette gain at {x},{y} cannot be represented as u32: {value}"
            ),
            Self::InvalidFrameDimensions { dimensions } => write!(
                formatter,
                "frame dimensions {}x{} must be non-zero",
                dimensions.width, dimensions.height
            ),
            Self::FrameDimensionsOverflow { dimensions } => write!(
                formatter,
                "frame dimensions {}x{} overflow",
                dimensions.width, dimensions.height
            ),
            Self::FrameDimensionsMismatch {
                frame_dimensions,
                facts_dimensions,
            } => write!(
                formatter,
                "frame dimensions {}x{} do not match vignette facts dimensions {}x{}",
                frame_dimensions.width,
                frame_dimensions.height,
                facts_dimensions.width,
                facts_dimensions.height
            ),
            Self::GpuBufferSizeMismatch {
                buffer,
                required_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "GPU {buffer} buffer is too small: required {required_bytes} bytes, got {actual_bytes}"
            ),
            Self::GpuFrameDimensionsMismatch {
                frame_dimensions,
                gain_map_dimensions,
            } => write!(
                formatter,
                "GPU frame dimensions {}x{} do not match uploaded gain map dimensions {}x{}",
                frame_dimensions.width,
                frame_dimensions.height,
                gain_map_dimensions.width,
                gain_map_dimensions.height
            ),
            Self::GpuGainMapPixelCountMismatch {
                expected_pixel_count,
                actual_pixel_count,
            } => write!(
                formatter,
                "GPU gain map pixel count mismatch: expected {expected_pixel_count}, got {actual_pixel_count}"
            ),
            Self::GpuGainMapBindingTooLarge {
                required_bytes,
                max_binding_bytes,
                frame_dimensions,
                pixel_count,
            } => write!(
                formatter,
                "GPU gain map binding is too large for the device: required {required_bytes} bytes for {}x{} ({pixel_count} pixels), max binding size is {max_binding_bytes} bytes",
                frame_dimensions.width, frame_dimensions.height
            ),
            Self::GpuInvalidTilePlan { reason } => {
                write!(formatter, "GPU vignette tile plan is invalid: {reason}")
            }
            Self::GpuComputeWorkgroupLimitTooSmall {
                max_workgroups,
                workgroup_size,
            } => write!(
                formatter,
                "GPU vignette compute workgroup limit is too small: max {max_workgroups} groups with workgroup size {workgroup_size}"
            ),
            Self::GpuTileDispatchTooLarge {
                tile_packed_word_count,
                workgroup_size,
                required_workgroups,
                max_workgroups,
            } => write!(
                formatter,
                "GPU vignette tile dispatch is too large: {tile_packed_word_count} packed words require {required_workgroups} workgroups of size {workgroup_size}, max is {max_workgroups}"
            ),
            Self::GpuTileOffsetAlignment { offset, alignment } => write!(
                formatter,
                "GPU vignette tile gain offset {offset} is not aligned to {alignment} bytes"
            ),
            Self::GpuTileSizeZero => {
                formatter.write_str("GPU vignette tile gain binding size is zero")
            }
            Self::GpuTilePixelCountOverflow => {
                formatter.write_str("GPU vignette tile pixel count overflow")
            }
            Self::GpuFixedPointBlackLevelOverflow { index, value } => write!(
                formatter,
                "GPU black level {index} cannot be represented as u32 fixed-point: {value}"
            ),
            Self::DecodedFrameValidationFailed(error) => {
                write!(formatter, "corrected frame validation failed: {error}")
            }
        }
    }
}

impl Error for VignetteCorrectionError {}

impl From<LensShadingMapValidationError> for VignetteCorrectionError {
    fn from(error: LensShadingMapValidationError) -> Self {
        match error {
            LensShadingMapValidationError::EmptyDimensions => Self::EmptyLensShadingMapDimensions,
            LensShadingMapValidationError::DimensionsOverflow { width, height } => {
                Self::LensShadingMapDimensionsOverflow { width, height }
            }
            LensShadingMapValidationError::PlaneLengthMismatch {
                plane_index,
                expected_len,
                actual_len,
            } => Self::LensShadingPlaneLengthMismatch {
                plane_index,
                expected_len,
                actual_len,
            },
            LensShadingMapValidationError::NonFiniteGain {
                plane_index,
                sample_index,
            } => Self::NonFiniteLensShadingGain {
                plane_index,
                sample_index,
            },
            LensShadingMapValidationError::NegativeGain {
                plane_index,
                sample_index,
                value,
            } => Self::NegativeLensShadingGain {
                plane_index,
                sample_index,
                value,
            },
        }
    }
}
