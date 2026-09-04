use mcraw4vulkan_core::{DecodedBayerU16Frame, FrameDimensions, FrameNumber};
use mcraw4vulkan_mcrawcontainer::{ContainerMetadata, FrameMetadata};

use crate::{
    DngDescriptionError, DngFrameDescription, DngFrameDescriptionOverrides, DngWriterError,
};

// DNG sink correction state for decoded Bayer bytes.
//
// The writer stays decode-agnostic: this only describes whether the supplied
// pixels are still in the raw domain or have already gone through the canonical
// full-resolution LumaPlane0 correction stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DngOutputCorrection {
    None,
    LumaPlane0,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DngSinkVignetteMode {
    None,
    #[default]
    LumaPlane0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DngSinkDecodeSource {
    GpuCanonical,
    CpuFallbackGpuVignette,
    CpuFallbackNoVig,
}

// Complete DNG-ready frame payload produced by a decode/correction sink.
//
// The pixel payload is exact tightly-packed little-endian Bayer U16 bytes. This
// avoids reintroducing Vec<u16> into production DNG/FUSE paths.
#[derive(Debug, Clone)]
pub struct DngSinkFrame {
    frame_number: FrameNumber,
    dimensions: FrameDimensions,
    pixels: DecodedBayerU16Frame<'static>,
    vignette_mode: DngSinkVignetteMode,
    decode_source: DngSinkDecodeSource,
    description: DngFrameDescription,
}

impl DngSinkFrame {
    pub fn new(
        frame_number: FrameNumber,
        pixels: DecodedBayerU16Frame<'static>,
        vignette_mode: DngSinkVignetteMode,
        decode_source: DngSinkDecodeSource,
        description: DngFrameDescription,
    ) -> Result<Self, DngWriterError> {
        let dimensions = pixels.dimensions();

        if dimensions != description.dimensions {
            return Err(DngWriterError::InvalidMetadata(format!(
                "DNG sink frame dimensions {}x{} do not match description dimensions {}x{}",
                dimensions.width,
                dimensions.height,
                description.dimensions.width,
                description.dimensions.height
            )));
        }

        Ok(Self {
            frame_number,
            dimensions,
            pixels,
            vignette_mode,
            decode_source,
            description,
        })
    }

    pub fn frame_number(&self) -> FrameNumber {
        self.frame_number
    }

    pub fn dimensions(&self) -> FrameDimensions {
        self.dimensions
    }

    pub fn pixels(&self) -> &DecodedBayerU16Frame<'_> {
        &self.pixels
    }

    pub fn pixel_bytes_le(&self) -> &[u8] {
        self.pixels.pixel_bytes_le()
    }

    pub fn vignette_mode(&self) -> DngSinkVignetteMode {
        self.vignette_mode
    }

    pub fn decode_source(&self) -> DngSinkDecodeSource {
        self.decode_source
    }

    pub fn description(&self) -> &DngFrameDescription {
        &self.description
    }
}

impl From<DngSinkVignetteMode> for DngOutputCorrection {
    fn from(value: DngSinkVignetteMode) -> Self {
        match value {
            DngSinkVignetteMode::None => Self::None,
            DngSinkVignetteMode::LumaPlane0 => Self::LumaPlane0,
        }
    }
}

pub fn build_dng_frame_description_for_sink(
    container_metadata: &ContainerMetadata,
    frame_metadata: &FrameMetadata,
    frame_number: FrameNumber,
    timestamp_us: u64,
    correction: DngOutputCorrection,
) -> Result<DngFrameDescription, DngDescriptionError> {
    let description = DngFrameDescription::from_metadata(
        container_metadata,
        frame_metadata,
        frame_number,
        timestamp_us,
    )?;

    Ok(apply_dng_output_correction_policy(&description, correction))
}

pub fn apply_dng_output_correction_policy(
    description: &DngFrameDescription,
    correction: DngOutputCorrection,
) -> DngFrameDescription {
    match correction {
        DngOutputCorrection::None => description.clone(),
        DngOutputCorrection::LumaPlane0 => description
            .with_output_overrides(DngFrameDescriptionOverrides::corrected_pixel_domain()),
    }
}
