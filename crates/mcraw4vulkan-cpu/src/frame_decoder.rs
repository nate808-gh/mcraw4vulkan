use mcraw4vulkan_core::{DecodedBayerU16Frame, FrameDimensions, FramePayloadLayout};

use crate::error::CpuDecodeError;
use crate::raw_decoder::{RawDecodeInfo, decode_frame_payload_to_decoded_bayer_u16_frame};
use crate::scratch::FrameScratch;

// Reusable CPU frame decoder for raw MotionCam frame payloads.
//
// The caller owns container access and fills compressed_mut() with exactly one
// raw video payload before calling a decode_loaded_payload_* method.
#[derive(Debug, Default)]
pub struct CpuFrameDecoder {
    scratch: FrameScratch,
}

impl CpuFrameDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn prepare_compressed(&mut self, byte_count: usize) {
        self.scratch.prepare_compressed(byte_count);
    }

    pub fn compressed_mut(&mut self) -> &mut Vec<u8> {
        self.scratch.compressed_mut()
    }

    pub fn compressed(&self) -> &[u8] {
        self.scratch.compressed()
    }

    pub fn decode_loaded_payload_to_decoded_bayer_u16_frame(
        &mut self,
        dimensions: FrameDimensions,
    ) -> Result<(DecodedBayerU16Frame<'static>, RawDecodeInfo), CpuDecodeError> {
        self.decode_loaded_payload_to_decoded_bayer_u16_frame_with_layout(
            dimensions,
            FramePayloadLayout::CompressedRawcodecType7,
        )
    }

    #[rustfmt::skip]
    pub fn decode_loaded_payload_to_decoded_bayer_u16_frame_with_layout(
        &mut self,
        dimensions: FrameDimensions,
        payload_layout: FramePayloadLayout,
    ) -> Result<(DecodedBayerU16Frame<'static>, RawDecodeInfo), CpuDecodeError> {
        decode_frame_payload_to_decoded_bayer_u16_frame(
            self.scratch.compressed(),
            dimensions,
            payload_layout,
        )
    }
}
