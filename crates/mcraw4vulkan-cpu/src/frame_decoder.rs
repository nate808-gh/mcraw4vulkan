use std::time::{Duration, Instant};

use mcraw4vulkan_core::{DecodedBayerU16Frame, FrameDimensions, FrameNumber, FramePayloadLayout};

use crate::error::CpuDecodeError;
use crate::raw_decoder::{
    RawDecodeInfo, decode_frame_payload_into, decode_frame_payload_to_decoded_bayer_u16_frame,
};
use crate::scratch::FrameScratch;

// Metadata describing one legacy u16 decoded frame.
//
// The pixel buffer itself is borrowed separately through DecodedFrameRef so the
// CPU frame decoder can reuse its scratch allocation across frame requests.
#[derive(Debug, Clone, Copy)]
pub struct DecodedFrameInfo {
    pub frame_number: FrameNumber,
    pub dimensions: FrameDimensions,
    pub timestamp_us: u64,
}

// Borrowed view of a legacy u16 decoded frame.
//
// The pixels borrow from CpuFrameDecoder-owned scratch memory. Callers must
// finish using this view before asking the same decoder to decode another frame.
#[derive(Debug)]
pub struct DecodedFrameRef<'a> {
    pub info: DecodedFrameInfo,
    pub pixels: &'a [u16],
}

// Fine-grained timing for one CPU frame decode.
//
// These timings are meant for benchmarking and design decisions. They split the
// current decode operation into source payload read, buffer preparation, and raw
// CPU decode work. Payload read time is filled by the caller that owns payload
// access.
#[derive(Debug, Default, Clone, Copy)]
pub struct DecodeFrameTimings {
    pub payload_read_time: Duration,
    pub buffer_prepare_time: Duration,
    pub raw_decode_time: Duration,
}

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

    pub fn decode_loaded_payload_to_frame_ref(
        &mut self,
        frame_number: FrameNumber,
        dimensions: FrameDimensions,
        timestamp_us: u64,
        timings: &mut DecodeFrameTimings,
    ) -> Result<DecodedFrameRef<'_>, CpuDecodeError> {
        self.decode_loaded_payload_to_frame_ref_with_layout(
            frame_number,
            dimensions,
            timestamp_us,
            FramePayloadLayout::CompressedRawcodecType7,
            timings,
        )
    }

    pub fn decode_loaded_payload_to_frame_ref_with_layout(
        &mut self,
        frame_number: FrameNumber,
        dimensions: FrameDimensions,
        timestamp_us: u64,
        payload_layout: FramePayloadLayout,
        timings: &mut DecodeFrameTimings,
    ) -> Result<DecodedFrameRef<'_>, CpuDecodeError> {
        let prepare_pixels_start = Instant::now();
        self.scratch.prepare_for_frame(dimensions)?;
        timings.buffer_prepare_time += prepare_pixels_start.elapsed();

        let raw_decode_start = Instant::now();
        {
            let (raw_payload, pixels) = self.scratch.compressed_and_pixels_mut();
            decode_frame_payload_into(raw_payload, dimensions, payload_layout, pixels)?;
        }
        timings.raw_decode_time += raw_decode_start.elapsed();

        Ok(DecodedFrameRef {
            info: DecodedFrameInfo {
                frame_number,
                dimensions,
                timestamp_us,
            },
            pixels: self.scratch.pixels(),
        })
    }

    pub fn decode_loaded_payload_to_decoded_bayer_u16_frame(
        &mut self,
        dimensions: FrameDimensions,
        timings: &mut DecodeFrameTimings,
    ) -> Result<(DecodedBayerU16Frame<'static>, RawDecodeInfo), CpuDecodeError> {
        self.decode_loaded_payload_to_decoded_bayer_u16_frame_with_layout(
            dimensions,
            FramePayloadLayout::CompressedRawcodecType7,
            timings,
        )
    }

    pub fn decode_loaded_payload_to_decoded_bayer_u16_frame_with_layout(
        &mut self,
        dimensions: FrameDimensions,
        payload_layout: FramePayloadLayout,
        timings: &mut DecodeFrameTimings,
    ) -> Result<(DecodedBayerU16Frame<'static>, RawDecodeInfo), CpuDecodeError> {
        let raw_decode_start = Instant::now();
        let decode_result = decode_frame_payload_to_decoded_bayer_u16_frame(
            self.scratch.compressed(),
            dimensions,
            payload_layout,
        );
        timings.raw_decode_time += raw_decode_start.elapsed();

        decode_result
    }
}
