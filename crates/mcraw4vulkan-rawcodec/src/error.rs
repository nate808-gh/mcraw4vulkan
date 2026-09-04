use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RawCodecError {
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
