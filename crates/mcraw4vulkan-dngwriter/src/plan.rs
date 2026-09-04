use crate::DngFrameDescription;
use crate::error::DngWriterError;

// A validated write plan for one uncompressed DNG frame.
//
// This does not write TIFF/DNG bytes yet. It records the sizes and layout that
// the writer uses, and it gives us a safe place to validate the
// caller-provided DNG description before building a DNG file.
#[derive(Debug, Clone)]
pub struct DngWritePlan {
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: u16,
    pub samples_per_pixel: u16,
    pub pixel_count: usize,
    pub pixel_byte_count: usize,
}

impl DngWritePlan {
    // Build a write plan from a neutral DNG frame description only.
    //
    // This is used by the FUSE metadata path to compute a stable DNG byte length
    // without decoding pixels or allocating the final DNG byte buffer.
    pub fn from_description(description: &DngFrameDescription) -> Result<Self, DngWriterError> {
        if description.bits_per_sample != 16 {
            return Err(DngWriterError::UnsupportedLayout(format!(
                "bits_per_sample must be 16, got {}",
                description.bits_per_sample
            )));
        }

        if description.samples_per_pixel != 1 {
            return Err(DngWriterError::UnsupportedLayout(format!(
                "samples_per_pixel must be 1, got {}",
                description.samples_per_pixel
            )));
        }

        let width = usize::try_from(description.dimensions.width).map_err(|_| {
            DngWriterError::UnsupportedLayout("width does not fit usize".to_string())
        })?;

        let height = usize::try_from(description.dimensions.height).map_err(|_| {
            DngWriterError::UnsupportedLayout("height does not fit usize".to_string())
        })?;

        let pixel_count = width
            .checked_mul(height)
            .ok_or(DngWriterError::PixelByteCountOverflow)?;

        let pixel_byte_count = pixel_count
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or(DngWriterError::PixelByteCountOverflow)?;

        Ok(Self {
            width: description.dimensions.width,
            height: description.dimensions.height,
            bits_per_sample: description.bits_per_sample,
            samples_per_pixel: description.samples_per_pixel,
            pixel_count,
            pixel_byte_count,
        })
    }

    // Build a write plan from a neutral DNG frame description and decoded pixels.
    //
    // The first DNG writer target is uncompressed 16-bit single-sample Bayer
    // data, so this validates that the pixel buffer matches the visible frame
    // dimensions exactly.
    pub fn from_description_and_pixels(
        description: &DngFrameDescription,
        pixels: &[u16],
    ) -> Result<Self, DngWriterError> {
        let plan = Self::from_description(description)?;

        if pixels.len() != plan.pixel_count {
            return Err(DngWriterError::PixelCountMismatch {
                expected: plan.pixel_count,
                actual: pixels.len(),
            });
        }

        Ok(plan)
    }
}
