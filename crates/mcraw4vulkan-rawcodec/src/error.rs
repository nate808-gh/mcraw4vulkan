use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RawCodecError {
    #[error("invalid frozen geometry, metadata stream or residual span in recovered frame")]
    InvalidResolvedGeometry,

    #[error("unsupported block encoding value: {0}")]
    UnsupportedBlockEncoding(u16),

    #[error("metadata block count is out of bounds")]
    MetadataBlockCountOutOfBounds,

    #[error("metadata header is out of bounds")]
    MetadataHeaderOutOfBounds,

    #[error("invalid metadata block header")]
    InvalidMetadataBlockHeader,

    #[error("metadata block decode overflow")]
    MetadataBlockDecodeOverflow,
}
