use std::borrow::Cow;
use std::error::Error;
use std::fmt;

use crate::FrameDimensions;

const BYTES_PER_BAYER_U16_SAMPLE: u64 = 2;

// Shared decoded video-frame contract for backend-neutral Bayer U16 output.
//
// The payload is exact tightly-packed little-endian Bayer samples: one u16
// sample per pixel, with no row padding or container metadata.
// Cow permits a zero-copy view while producer storage remains alive;
// into_owned copies only when a borrowed frame crosses that lifetime boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedBayerU16Frame<'a> {
    dimensions: FrameDimensions,
    pixel_bytes_le: Cow<'a, [u8]>,
}

impl<'a> DecodedBayerU16Frame<'a> {
    pub fn from_owned_le_bytes(
        dimensions: FrameDimensions,
        pixel_bytes_le: Vec<u8>,
    ) -> Result<Self, DecodedBayerU16FrameValidationError> {
        validate_pixel_byte_len(dimensions, pixel_bytes_le.len())?;

        Ok(Self {
            dimensions,
            pixel_bytes_le: Cow::Owned(pixel_bytes_le),
        })
    }

    pub fn from_borrowed_le_bytes(
        dimensions: FrameDimensions,
        pixel_bytes_le: &'a [u8],
    ) -> Result<Self, DecodedBayerU16FrameValidationError> {
        validate_pixel_byte_len(dimensions, pixel_bytes_le.len())?;

        Ok(Self {
            dimensions,
            pixel_bytes_le: Cow::Borrowed(pixel_bytes_le),
        })
    }

    pub fn dimensions(&self) -> FrameDimensions {
        self.dimensions
    }

    pub fn pixel_bytes_le(&self) -> &[u8] {
        self.pixel_bytes_le.as_ref()
    }

    pub fn into_owned_le_bytes(self) -> Vec<u8> {
        self.pixel_bytes_le.into_owned()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodedBayerU16FrameValidationError {
    ByteLengthOverflow {
        dimensions: FrameDimensions,
    },
    ByteLengthMismatch {
        dimensions: FrameDimensions,
        expected_len: usize,
        actual_len: usize,
    },
}

impl fmt::Display for DecodedBayerU16FrameValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ByteLengthOverflow { dimensions } => write!(
                formatter,
                "decoded Bayer U16 frame dimensions {}x{} overflow byte length",
                dimensions.width, dimensions.height
            ),
            Self::ByteLengthMismatch {
                dimensions,
                expected_len,
                actual_len,
            } => write!(
                formatter,
                "decoded Bayer U16 frame dimensions {}x{} require {expected_len} bytes, got {actual_len}",
                dimensions.width, dimensions.height
            ),
        }
    }
}

impl Error for DecodedBayerU16FrameValidationError {}

fn validate_pixel_byte_len(
    dimensions: FrameDimensions,
    actual_len: usize,
) -> Result<(), DecodedBayerU16FrameValidationError> {
    let expected_len = expected_pixel_byte_len(dimensions)?;

    if actual_len != expected_len {
        return Err(DecodedBayerU16FrameValidationError::ByteLengthMismatch {
            dimensions,
            expected_len,
            actual_len,
        });
    }

    Ok(())
}

fn expected_pixel_byte_len(
    dimensions: FrameDimensions,
) -> Result<usize, DecodedBayerU16FrameValidationError> {
    let pixel_count = u64::from(dimensions.width)
        .checked_mul(u64::from(dimensions.height))
        .ok_or(DecodedBayerU16FrameValidationError::ByteLengthOverflow { dimensions })?;
    let byte_len = pixel_count
        .checked_mul(BYTES_PER_BAYER_U16_SAMPLE)
        .ok_or(DecodedBayerU16FrameValidationError::ByteLengthOverflow { dimensions })?;

    usize::try_from(byte_len)
        .map_err(|_| DecodedBayerU16FrameValidationError::ByteLengthOverflow { dimensions })
}

#[cfg(test)]
mod tests {
    use super::{DecodedBayerU16Frame, DecodedBayerU16FrameValidationError};
    use crate::FrameDimensions;

    #[test]
    fn valid_borrowed_byte_length_is_accepted() {
        let dimensions = FrameDimensions {
            width: 2,
            height: 2,
        };
        let pixel_bytes = [0, 1, 2, 3, 4, 5, 6, 7];

        let frame = DecodedBayerU16Frame::from_borrowed_le_bytes(dimensions, &pixel_bytes)
            .expect("valid tightly-packed Bayer U16 bytes");

        assert_eq!(frame.dimensions(), dimensions);
        assert_eq!(frame.pixel_bytes_le(), pixel_bytes);
    }

    #[test]
    fn valid_owned_byte_length_is_accepted() {
        let dimensions = FrameDimensions {
            width: 2,
            height: 1,
        };
        let pixel_bytes = vec![0, 1, 2, 3];

        let frame = DecodedBayerU16Frame::from_owned_le_bytes(dimensions, pixel_bytes.clone())
            .expect("valid tightly-packed Bayer U16 bytes");

        assert_eq!(frame.dimensions(), dimensions);
        assert_eq!(frame.into_owned_le_bytes(), pixel_bytes);
    }

    #[test]
    fn too_few_bytes_are_rejected() {
        let dimensions = FrameDimensions {
            width: 2,
            height: 2,
        };

        let error = DecodedBayerU16Frame::from_borrowed_le_bytes(dimensions, &[0; 7])
            .expect_err("short Bayer U16 frame must be rejected");

        assert_eq!(
            error,
            DecodedBayerU16FrameValidationError::ByteLengthMismatch {
                dimensions,
                expected_len: 8,
                actual_len: 7,
            }
        );
    }

    #[test]
    fn too_many_bytes_are_rejected() {
        let dimensions = FrameDimensions {
            width: 2,
            height: 2,
        };

        let error = DecodedBayerU16Frame::from_borrowed_le_bytes(dimensions, &[0; 9])
            .expect_err("long Bayer U16 frame must be rejected");

        assert_eq!(
            error,
            DecodedBayerU16FrameValidationError::ByteLengthMismatch {
                dimensions,
                expected_len: 8,
                actual_len: 9,
            }
        );
    }

    #[test]
    fn byte_length_overflow_is_rejected() {
        let dimensions = FrameDimensions {
            width: u32::MAX,
            height: u32::MAX,
        };

        let error = DecodedBayerU16Frame::from_borrowed_le_bytes(dimensions, &[])
            .expect_err("overflowing dimensions must be rejected");

        assert_eq!(
            error,
            DecodedBayerU16FrameValidationError::ByteLengthOverflow { dimensions }
        );
    }
}
