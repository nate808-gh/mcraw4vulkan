use std::error::Error;
use std::fmt;

use mcraw4vulkan_rawcodec::RawCodecError;

use crate::raw_decoder::RawDecodeError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuDecodeError {
    MissingRawPayloadMetadataHeader,
    EncodedDimensionsZero,
    EncodedWidthNotMultipleOfBlock,
    VisibleDimensionsExceedEncoded,
    MetadataOffsetsOutsidePayload,
    EncodedDimensionsUnreasonablyLarge,
    PixelCountUnreasonablyLarge,
    MetadataStreamsTooShort,
    FrameDimensionsOverflow,
    RawCodec(RawCodecError),
    RawDecode(RawDecodeError),
}

impl fmt::Display for CpuDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRawPayloadMetadataHeader => {
                formatter.write_str("missing raw payload metadata header")
            }
            Self::EncodedDimensionsZero => {
                formatter.write_str("encoded dimensions must be non-zero")
            }
            Self::EncodedWidthNotMultipleOfBlock => {
                formatter.write_str("encoded width must be a multiple of 64")
            }
            Self::VisibleDimensionsExceedEncoded => {
                formatter.write_str("visible dimensions exceed encoded dimensions")
            }
            Self::MetadataOffsetsOutsidePayload => {
                formatter.write_str("metadata offsets are outside the raw payload")
            }
            Self::EncodedDimensionsUnreasonablyLarge => {
                formatter.write_str("encoded dimensions are unreasonably large")
            }
            Self::PixelCountUnreasonablyLarge => {
                formatter.write_str("pixel count is unreasonably large")
            }
            Self::MetadataStreamsTooShort => {
                formatter.write_str("metadata streams are shorter than expected")
            }
            Self::FrameDimensionsOverflow => formatter.write_str("frame dimensions overflow"),
            Self::RawCodec(error) => fmt::Display::fmt(error, formatter),
            Self::RawDecode(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl Error for CpuDecodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RawCodec(error) => Some(error),
            Self::RawDecode(error) => Some(error),
            _ => None,
        }
    }
}

impl From<RawCodecError> for CpuDecodeError {
    fn from(error: RawCodecError) -> Self {
        Self::RawCodec(error)
    }
}

impl From<RawDecodeError> for CpuDecodeError {
    fn from(error: RawDecodeError) -> Self {
        Self::RawDecode(error)
    }
}
