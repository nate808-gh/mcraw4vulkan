use thiserror::Error;

// Error type for DNG writer operations.
//
// This crate reports errors that happen while validating DNG inputs, planning
// output size, converting metadata, or writing DNG bytes/files.
#[derive(Debug, Error)]
pub enum DngWriterError {
    #[error("pixel buffer length does not match DNG dimensions: expected {expected}, got {actual}")]
    PixelCountMismatch { expected: usize, actual: usize },

    #[error("DNG pixel byte count overflow")]
    PixelByteCountOverflow,

    #[error("TIFF/DNG output is too large for standard TIFF")]
    TiffSizeOverflow,

    #[error("invalid DNG metadata: {0}")]
    InvalidMetadata(String),

    #[error("unsupported DNG layout: {0}")]
    UnsupportedLayout(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
