use mcraw4vulkan_core::FrameDimensions;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DisplayRenderError {
    #[error("render dimensions must be non-zero, got {dimensions:?}")]
    InvalidDimensions { dimensions: FrameDimensions },

    #[error("display render requires an even pixel count, got {pixel_count}")]
    OddPixelCount { pixel_count: usize },

    #[error("display render byte count overflow for dimensions {dimensions:?}")]
    OutputByteCountOverflow { dimensions: FrameDimensions },

    #[error("render uniform parameter value is invalid: {0}")]
    InvalidParams(String),

    #[error(
        "render GPU buffer is too small: {buffer} requires {required_bytes} bytes, got {actual_bytes}"
    )]
    GpuBufferSizeMismatch {
        buffer: &'static str,
        required_bytes: u64,
        actual_bytes: u64,
    },

    #[error("failed to map render readback buffer: {0}")]
    MapFailed(String),

    #[error("render GPU operation failed: {0}")]
    Gpu(String),
}
