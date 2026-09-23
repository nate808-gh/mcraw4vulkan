use std::error::Error;
use std::fmt;

use mcraw4vulkan_core::{BlockEncoding, DecodedBayerU16Frame, FrameDimensions, FramePayloadLayout};
use mcraw4vulkan_rawcodec::{
    ENCODING_BLOCK, METADATA_OFFSET, MetadataHeader, block_encoding_from_raw, decode_block,
    decode_metadata, read_metadata_header,
};

use crate::error::CpuDecodeError;

const MCRAW_DECODED_BLOCK_SAMPLES: usize = 64;
const BYTES_PER_BAYER_U16_SAMPLE: usize = 2;
const LEGACY_RAW16_BLOCK_SAMPLES: usize = 16;
const LEGACY_RAW16_ENCODING_BLOCK: usize = LEGACY_RAW16_BLOCK_SAMPLES * 2;
const LEGACY_RAW16_HEADER_LENGTH: usize = 2;
const LEGACY_RAW16_BLOCK_LENGTHS: [usize; 17] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 32, 32, 32, 32, 32, 32,
];

// Prepared raw metadata needed by the CPU pixel decode loop.
//
// Raw payload header parsing and metadata expansion live in rawcodec. This type
// is the narrow CPU boundary: encoded dimensions, decoded block
// encodings/references, and the byte offset where block payloads begin.
#[derive(Debug, Clone, Copy)]
pub struct PreparedRawDecodeMetadata<'a> {
    pub encoded_dimensions: FrameDimensions,
    pub payload_base_offset: usize,
    pub blocks: &'a [PreparedRawDecodeBlock],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedRawDecodeBlock {
    pub encoding: BlockEncoding,
    pub reference: u16,
}

// Summary of a successful prepared raw payload decode.
#[derive(Debug, Clone, Copy)]
pub struct RawDecodeInfo {
    pub encoded_dimensions: FrameDimensions,
    pub visible_dimensions: FrameDimensions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawDecodeError {
    FrameDimensionsOverflow,
    OutputBufferLengthMismatch { expected: usize, actual: usize },
    MetadataStreamsTooShort,
    FrameBlockOffsetOutsidePayload,
    FrameBlockDecodeOverflow,
    FrameBlockPayloadOffsetOverflow,
    Raw16RowStrideTooShort { row_stride: u32, minimum: u64 },
    Raw16PayloadTooShort { required: u64, actual: u64 },
    Raw16PayloadLengthOverflow,
    DecodedFrameValidationFailed(String),
}

impl fmt::Display for RawDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameDimensionsOverflow => formatter.write_str("frame dimensions overflow"),
            Self::OutputBufferLengthMismatch { expected, actual } => write!(
                formatter,
                "output buffer length mismatch: expected {expected}, got {actual}"
            ),
            Self::MetadataStreamsTooShort => {
                formatter.write_str("metadata streams are shorter than expected")
            }
            Self::FrameBlockOffsetOutsidePayload => {
                formatter.write_str("frame block offset is outside payload")
            }
            Self::FrameBlockDecodeOverflow => formatter.write_str("frame block decode overflow"),
            Self::FrameBlockPayloadOffsetOverflow => {
                formatter.write_str("frame block payload offset overflow")
            }
            Self::Raw16RowStrideTooShort {
                row_stride,
                minimum,
            } => write!(
                formatter,
                "raw16 rowStride is too short: rowStride={row_stride}, minimum={minimum}"
            ),
            Self::Raw16PayloadTooShort { required, actual } => write!(
                formatter,
                "legacy raw16 payload is too short: required at least {required} bytes, got {actual}"
            ),
            Self::Raw16PayloadLengthOverflow => {
                formatter.write_str("raw16 payload byte length overflow")
            }
            Self::DecodedFrameValidationFailed(error) => {
                write!(
                    formatter,
                    "decoded Bayer U16 frame validation failed: {error}"
                )
            }
        }
    }
}

impl Error for RawDecodeError {}

// Validate a raw frame payload and return its parsed metadata header.
//
// This function does not decode pixels. It only checks that the payload header is
// sane for the visible frame dimensions supplied by parsed frame metadata.
pub fn validate_raw_payload(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
) -> Result<MetadataHeader, CpuDecodeError> {
    let header =
        read_metadata_header(raw_payload).ok_or(CpuDecodeError::MissingRawPayloadMetadataHeader)?;

    validate_raw_payload_header(raw_payload, &header, visible_dimensions)?;

    Ok(header)
}

pub fn validate_frame_payload(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    payload_layout: FramePayloadLayout,
) -> Result<RawDecodeInfo, CpuDecodeError> {
    match payload_layout {
        FramePayloadLayout::CompressedRawcodecType7 => {
            let header = validate_raw_payload(raw_payload, visible_dimensions)?;
            Ok(RawDecodeInfo {
                encoded_dimensions: FrameDimensions {
                    width: header.encoded_width,
                    height: header.encoded_height,
                },
                visible_dimensions,
            })
        }
        FramePayloadLayout::BinnedRaw16Type6 { row_stride } => {
            validate_binned_raw16_payload(raw_payload, visible_dimensions, row_stride)
                .map_err(Into::into)
        }
    }
}

// Validate one raw frame payload, prepare CPU decode metadata, and decode into a
// reusable legacy u16 output buffer.
//
// The caller must provide an output buffer whose length already matches the
// visible image dimensions. The decode loop still walks the encoded dimensions
// from the payload header because encoded rows may be padded to the MotionCam
// block size.
pub fn decode_raw_payload_into(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    output: &mut Vec<u16>,
) -> Result<RawDecodeInfo, CpuDecodeError> {
    decode_frame_payload_into(
        raw_payload,
        visible_dimensions,
        FramePayloadLayout::CompressedRawcodecType7,
        output,
    )
}

pub fn decode_frame_payload_into(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    payload_layout: FramePayloadLayout,
    output: &mut Vec<u16>,
) -> Result<RawDecodeInfo, CpuDecodeError> {
    match payload_layout {
        FramePayloadLayout::CompressedRawcodecType7 => {
            let prepared = prepare_raw_payload_decode(raw_payload, visible_dimensions)?;

            decode_prepared_raw_payload_into(
                raw_payload,
                visible_dimensions,
                prepared.as_metadata(),
                output,
            )
            .map_err(Into::into)
        }
        FramePayloadLayout::BinnedRaw16Type6 { row_stride } => {
            decode_binned_raw16_payload_into(raw_payload, visible_dimensions, row_stride, output)
                .map_err(Into::into)
        }
    }
}

// Validate one raw frame payload, prepare CPU decode metadata, and decode
// directly into little-endian u16 pixel bytes.
//
// The output bytes are tightly packed Bayer samples: two bytes per visible
// pixel, little-endian, with no row padding. This shares the same raw traversal
// as the legacy u16 path but writes samples directly as bytes.
pub fn decode_raw_payload_into_le_bytes(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    output_bytes_le: &mut Vec<u8>,
) -> Result<RawDecodeInfo, CpuDecodeError> {
    decode_frame_payload_into_le_bytes(
        raw_payload,
        visible_dimensions,
        FramePayloadLayout::CompressedRawcodecType7,
        output_bytes_le,
    )
}

pub fn decode_frame_payload_into_le_bytes(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    payload_layout: FramePayloadLayout,
    output_bytes_le: &mut Vec<u8>,
) -> Result<RawDecodeInfo, CpuDecodeError> {
    match payload_layout {
        FramePayloadLayout::CompressedRawcodecType7 => {
            let prepared = prepare_raw_payload_decode(raw_payload, visible_dimensions)?;

            decode_prepared_raw_payload_into_le_bytes(
                raw_payload,
                visible_dimensions,
                prepared.as_metadata(),
                output_bytes_le,
            )
            .map_err(Into::into)
        }
        FramePayloadLayout::BinnedRaw16Type6 { row_stride } => {
            decode_binned_raw16_payload_into_le_bytes(
                raw_payload,
                visible_dimensions,
                row_stride,
                output_bytes_le,
            )
            .map_err(Into::into)
        }
    }
}

// Decode one raw frame payload into the shared owned Bayer U16 frame contract.
//
// This allocates the final little-endian byte buffer once, fills it through the
// direct byte-output path, and then moves it into DecodedBayerU16Frame.
pub fn decode_raw_payload_to_decoded_bayer_u16_frame(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
) -> Result<(DecodedBayerU16Frame<'static>, RawDecodeInfo), CpuDecodeError> {
    decode_frame_payload_to_decoded_bayer_u16_frame(
        raw_payload,
        visible_dimensions,
        FramePayloadLayout::CompressedRawcodecType7,
    )
}

pub fn decode_frame_payload_to_decoded_bayer_u16_frame(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    payload_layout: FramePayloadLayout,
) -> Result<(DecodedBayerU16Frame<'static>, RawDecodeInfo), CpuDecodeError> {
    match payload_layout {
        FramePayloadLayout::CompressedRawcodecType7 => {
            let prepared = prepare_raw_payload_decode(raw_payload, visible_dimensions)?;

            decode_prepared_raw_payload_to_decoded_bayer_u16_frame(
                raw_payload,
                visible_dimensions,
                prepared.as_metadata(),
            )
            .map_err(Into::into)
        }
        FramePayloadLayout::BinnedRaw16Type6 { row_stride } => {
            let mut pixel_bytes_le = Vec::new();
            let info = decode_binned_raw16_payload_into_le_bytes(
                raw_payload,
                visible_dimensions,
                row_stride,
                &mut pixel_bytes_le,
            )?;
            let frame =
                DecodedBayerU16Frame::from_owned_le_bytes(info.visible_dimensions, pixel_bytes_le)
                    .map_err(|error| {
                        RawDecodeError::DecodedFrameValidationFailed(error.to_string())
                    })?;

            Ok((frame, info))
        }
    }
}

struct PreparedRawPayloadDecode {
    encoded_dimensions: FrameDimensions,
    blocks: Vec<PreparedRawDecodeBlock>,
}

impl PreparedRawPayloadDecode {
    fn as_metadata(&self) -> PreparedRawDecodeMetadata<'_> {
        PreparedRawDecodeMetadata {
            encoded_dimensions: self.encoded_dimensions,
            payload_base_offset: METADATA_OFFSET,
            blocks: &self.blocks,
        }
    }
}

fn prepare_raw_payload_decode(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
) -> Result<PreparedRawPayloadDecode, CpuDecodeError> {
    let header = validate_raw_payload(raw_payload, visible_dimensions)?;

    let encoded_dimensions = FrameDimensions {
        width: header.encoded_width,
        height: header.encoded_height,
    };

    let bits = decode_metadata(raw_payload, header.bits_offset as usize)?;
    let refs = decode_metadata(raw_payload, header.refs_offset as usize)?;

    let encoded_width = header.encoded_width as usize;
    let encoded_height = header.encoded_height as usize;

    let expected_metadata_blocks = (encoded_height / 4) * (encoded_width / ENCODING_BLOCK) * 4;
    if bits.len() < expected_metadata_blocks || refs.len() < expected_metadata_blocks {
        return Err(CpuDecodeError::MetadataStreamsTooShort);
    }

    let mut blocks = Vec::with_capacity(expected_metadata_blocks);
    for index in 0..expected_metadata_blocks {
        blocks.push(PreparedRawDecodeBlock {
            encoding: block_encoding_from_raw(bits[index])?,
            reference: refs[index],
        });
    }

    Ok(PreparedRawPayloadDecode {
        encoded_dimensions,
        blocks,
    })
}

pub fn validate_binned_raw16_payload(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    row_stride: u32,
) -> Result<RawDecodeInfo, RawDecodeError> {
    let visible_row_bytes = visible_raw16_row_byte_len(visible_dimensions)?;
    let minimum_row_stride =
        u64::try_from(visible_row_bytes).map_err(|_| RawDecodeError::Raw16PayloadLengthOverflow)?;
    if u64::from(row_stride) < minimum_row_stride {
        return Err(RawDecodeError::Raw16RowStrideTooShort {
            row_stride,
            minimum: minimum_row_stride,
        });
    }

    let padded_width = legacy_raw16_padded_width(visible_dimensions.width)?;
    let height = usize::try_from(visible_dimensions.height)
        .map_err(|_| RawDecodeError::FrameDimensionsOverflow)?;
    let mut offset = 0usize;

    for _ in 0..height {
        let mut x = 0usize;
        while x < padded_width {
            offset = advance_legacy_raw16_block(raw_payload, offset)?;
            offset = advance_legacy_raw16_block(raw_payload, offset)?;
            x += LEGACY_RAW16_ENCODING_BLOCK;
        }
    }

    Ok(RawDecodeInfo {
        encoded_dimensions: FrameDimensions {
            width: u32::try_from(padded_width)
                .map_err(|_| RawDecodeError::FrameDimensionsOverflow)?,
            height: visible_dimensions.height,
        },
        visible_dimensions,
    })
}

pub fn decode_binned_raw16_payload_into(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    row_stride: u32,
    output: &mut [u16],
) -> Result<RawDecodeInfo, RawDecodeError> {
    let info = validate_binned_raw16_payload(raw_payload, visible_dimensions, row_stride)?;
    let pixel_count = visible_pixel_count(visible_dimensions)?;
    if output.len() != pixel_count {
        return Err(RawDecodeError::OutputBufferLengthMismatch {
            expected: pixel_count,
            actual: output.len(),
        });
    }

    let width = usize::try_from(visible_dimensions.width)
        .map_err(|_| RawDecodeError::FrameDimensionsOverflow)?;
    let height = usize::try_from(visible_dimensions.height)
        .map_err(|_| RawDecodeError::FrameDimensionsOverflow)?;
    let padded_width = legacy_raw16_padded_width(visible_dimensions.width)?;
    let mut offset = 0usize;

    for row in 0..height {
        let dst_row_start = row
            .checked_mul(width)
            .ok_or(RawDecodeError::FrameDimensionsOverflow)?;
        let mut x = 0usize;
        while x < padded_width {
            let mut p0 = [0u16; LEGACY_RAW16_BLOCK_SAMPLES];
            let mut p1 = [0u16; LEGACY_RAW16_BLOCK_SAMPLES];
            let (reference0, consumed0) = decode_legacy_raw16_block(&mut p0, raw_payload, offset)?;
            offset = offset
                .checked_add(consumed0)
                .ok_or(RawDecodeError::Raw16PayloadLengthOverflow)?;
            let (reference1, consumed1) = decode_legacy_raw16_block(&mut p1, raw_payload, offset)?;
            offset = offset
                .checked_add(consumed1)
                .ok_or(RawDecodeError::Raw16PayloadLengthOverflow)?;

            let mut i = 0usize;
            while i < LEGACY_RAW16_ENCODING_BLOCK {
                let col0 = x + i;
                let col1 = col0 + 1;
                let src = i / 2;
                if col0 < width {
                    output[dst_row_start + col0] = p0[src].wrapping_add(reference0);
                }
                if col1 < width {
                    output[dst_row_start + col1] = p1[src].wrapping_add(reference1);
                }
                i += 2;
            }

            x += LEGACY_RAW16_ENCODING_BLOCK;
        }
    }

    Ok(info)
}

pub fn decode_binned_raw16_payload_into_le_bytes(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    row_stride: u32,
    output_bytes_le: &mut Vec<u8>,
) -> Result<RawDecodeInfo, RawDecodeError> {
    let pixel_count = visible_pixel_count(visible_dimensions)?;
    let mut pixels = vec![0u16; pixel_count];
    let info =
        decode_binned_raw16_payload_into(raw_payload, visible_dimensions, row_stride, &mut pixels)?;

    output_bytes_le.clear();
    output_bytes_le.reserve(pixel_byte_count(pixel_count)?);
    for pixel in pixels {
        output_bytes_le.extend_from_slice(&pixel.to_le_bytes());
    }

    Ok(info)
}

fn legacy_raw16_padded_width(width: u32) -> Result<usize, RawDecodeError> {
    let width = usize::try_from(width).map_err(|_| RawDecodeError::FrameDimensionsOverflow)?;
    width
        .checked_add(LEGACY_RAW16_ENCODING_BLOCK - 1)
        .map(|value| value / LEGACY_RAW16_ENCODING_BLOCK * LEGACY_RAW16_ENCODING_BLOCK)
        .ok_or(RawDecodeError::FrameDimensionsOverflow)
}

fn advance_legacy_raw16_block(raw_payload: &[u8], offset: usize) -> Result<usize, RawDecodeError> {
    let (_, consumed) =
        decode_legacy_raw16_block(&mut [0; LEGACY_RAW16_BLOCK_SAMPLES], raw_payload, offset)?;
    offset
        .checked_add(consumed)
        .ok_or(RawDecodeError::Raw16PayloadLengthOverflow)
}

fn decode_legacy_raw16_block(
    output: &mut [u16; LEGACY_RAW16_BLOCK_SAMPLES],
    raw_payload: &[u8],
    offset: usize,
) -> Result<(u16, usize), RawDecodeError> {
    let header_end = offset
        .checked_add(LEGACY_RAW16_HEADER_LENGTH)
        .ok_or(RawDecodeError::Raw16PayloadLengthOverflow)?;
    if header_end > raw_payload.len() {
        return Err(raw16_payload_too_short(header_end, raw_payload.len()));
    }

    let header0 = raw_payload[offset];
    let bits = ((header0 >> 4) & 0x0f) as usize;
    let reference = (u16::from(header0 & 0x0f) << 8) | u16::from(raw_payload[offset + 1]);
    let payload_len = LEGACY_RAW16_BLOCK_LENGTHS[bits];
    let payload_start = header_end;
    let payload_end = payload_start
        .checked_add(payload_len)
        .ok_or(RawDecodeError::Raw16PayloadLengthOverflow)?;
    if payload_end > raw_payload.len() {
        return Err(raw16_payload_too_short(payload_end, raw_payload.len()));
    }

    let payload = &raw_payload[payload_start..payload_end];
    if bits == 0 {
        output.fill(0);
    } else if bits >= 11 {
        for (sample, chunk) in output.iter_mut().zip(payload.chunks_exact(2)) {
            *sample = u16::from_be_bytes([chunk[0], chunk[1]]);
        }
    } else {
        decode_legacy_msb_packed_bits(output, payload, bits);
    }

    Ok((reference, LEGACY_RAW16_HEADER_LENGTH + payload_len))
}

fn decode_legacy_msb_packed_bits(
    output: &mut [u16; LEGACY_RAW16_BLOCK_SAMPLES],
    payload: &[u8],
    bits: usize,
) {
    let mut bit_offset = 0usize;
    for sample in output {
        let mut value = 0u16;
        for _ in 0..bits {
            let byte = payload[bit_offset / 8];
            let shift = 7 - (bit_offset % 8);
            value = (value << 1) | u16::from((byte >> shift) & 1);
            bit_offset += 1;
        }
        *sample = value;
    }
}

fn raw16_payload_too_short(required: usize, actual: usize) -> RawDecodeError {
    RawDecodeError::Raw16PayloadTooShort {
        required: u64::try_from(required).unwrap_or(u64::MAX),
        actual: u64::try_from(actual).unwrap_or(u64::MAX),
    }
}

pub fn decode_prepared_raw_payload_into(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    prepared_metadata: PreparedRawDecodeMetadata<'_>,
    output: &mut Vec<u16>,
) -> Result<RawDecodeInfo, RawDecodeError> {
    let mut output = U16PixelOutput { pixels: output };

    decode_prepared_raw_payload_with_output(
        raw_payload,
        visible_dimensions,
        prepared_metadata,
        &mut output,
    )
}

pub fn decode_prepared_raw_payload_into_le_bytes(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    prepared_metadata: PreparedRawDecodeMetadata<'_>,
    output_bytes_le: &mut Vec<u8>,
) -> Result<RawDecodeInfo, RawDecodeError> {
    let mut output = LePixelByteOutput {
        pixel_bytes_le: output_bytes_le,
    };

    decode_prepared_raw_payload_with_output(
        raw_payload,
        visible_dimensions,
        prepared_metadata,
        &mut output,
    )
}

pub fn decode_prepared_raw_payload_to_decoded_bayer_u16_frame(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    prepared_metadata: PreparedRawDecodeMetadata<'_>,
) -> Result<(DecodedBayerU16Frame<'static>, RawDecodeInfo), RawDecodeError> {
    let mut pixel_bytes_le = Vec::new();
    let info = decode_prepared_raw_payload_into_le_bytes(
        raw_payload,
        visible_dimensions,
        prepared_metadata,
        &mut pixel_bytes_le,
    )?;

    let frame = DecodedBayerU16Frame::from_owned_le_bytes(info.visible_dimensions, pixel_bytes_le)
        .map_err(|error| RawDecodeError::DecodedFrameValidationFailed(error.to_string()))?;

    Ok((frame, info))
}

fn decode_prepared_raw_payload_with_output<O: RawDecodeOutput>(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    prepared_metadata: PreparedRawDecodeMetadata<'_>,
    output: &mut O,
) -> Result<RawDecodeInfo, RawDecodeError> {
    let visible_width = visible_dimensions.width as usize;
    let visible_height = visible_dimensions.height as usize;
    let encoded_width = prepared_metadata.encoded_dimensions.width as usize;
    let encoded_height = prepared_metadata.encoded_dimensions.height as usize;

    let pixel_count = visible_pixel_count(visible_dimensions)?;

    output.prepare_for_decode(pixel_count)?;

    let expected_metadata_blocks =
        (encoded_height / 4) * (encoded_width / MCRAW_DECODED_BLOCK_SAMPLES) * 4;
    if prepared_metadata.blocks.len() < expected_metadata_blocks {
        return Err(RawDecodeError::MetadataStreamsTooShort);
    }

    let mut offset = prepared_metadata.payload_base_offset;
    let mut metadata_idx = 0usize;

    // Keep cursor updates explicit: payload and metadata advances are validated
    // before the next block is decoded.
    let mut y = 0usize;
    while y < encoded_height {
        let mut x = 0usize;

        while x < encoded_width {
            let block0 = prepared_metadata.blocks[metadata_idx];
            let block1 = prepared_metadata.blocks[metadata_idx + 1];
            let block2 = prepared_metadata.blocks[metadata_idx + 2];
            let block3 = prepared_metadata.blocks[metadata_idx + 3];

            metadata_idx += 4;

            let mut p0 = [0u16; MCRAW_DECODED_BLOCK_SAMPLES];
            let mut p1 = [0u16; MCRAW_DECODED_BLOCK_SAMPLES];
            let mut p2 = [0u16; MCRAW_DECODED_BLOCK_SAMPLES];
            let mut p3 = [0u16; MCRAW_DECODED_BLOCK_SAMPLES];

            decode_next_block(raw_payload, &mut offset, block0, &mut p0)?;
            decode_next_block(raw_payload, &mut offset, block1, &mut p1)?;
            decode_next_block(raw_payload, &mut offset, block2, &mut p2)?;
            decode_next_block(raw_payload, &mut offset, block3, &mut p3)?;

            // Each group is a 4-row by 64-column tile: p0/p1 alternate columns
            // in rows 0 and 2, while p2/p3 alternate columns in rows 1 and 3.
            // Both output sinks preserve identical raster sample order.
            let block = DecodedBlock {
                p0: &p0,
                p1: &p1,
                p2: &p2,
                p3: &p3,
                reference0: block0.reference,
                reference1: block1.reference,
                reference2: block2.reference,
                reference3: block3.reference,
            };

            // Most encoded blocks are fully inside the visible image. Use safe
            // row/block slices for those blocks so the hot output loop avoids
            // repeated full-buffer row indexing.
            //
            // Only padded right-edge or bottom-edge blocks use the output
            // sink's checked pixel write, preserving the existing crop behavior.
            if x + MCRAW_DECODED_BLOCK_SAMPLES <= visible_width && y + 3 < visible_height {
                output.write_visible_block(visible_width, y, x, block);
            } else {
                write_edge_pixels(output, visible_width, visible_height, y, x, block);
            }

            x += MCRAW_DECODED_BLOCK_SAMPLES;
        }

        y += 4;
    }

    Ok(RawDecodeInfo {
        encoded_dimensions: prepared_metadata.encoded_dimensions,
        visible_dimensions,
    })
}

#[derive(Clone, Copy)]
struct DecodedBlock<'a> {
    p0: &'a [u16; MCRAW_DECODED_BLOCK_SAMPLES],
    p1: &'a [u16; MCRAW_DECODED_BLOCK_SAMPLES],
    p2: &'a [u16; MCRAW_DECODED_BLOCK_SAMPLES],
    p3: &'a [u16; MCRAW_DECODED_BLOCK_SAMPLES],
    reference0: u16,
    reference1: u16,
    reference2: u16,
    reference3: u16,
}

trait RawDecodeOutput {
    fn prepare_for_decode(&mut self, pixel_count: usize) -> Result<(), RawDecodeError>;

    fn write_visible_block(
        &mut self,
        visible_width: usize,
        row: usize,
        col: usize,
        block: DecodedBlock<'_>,
    );

    fn write_pixel_if_visible(
        &mut self,
        width: usize,
        height: usize,
        row: usize,
        col: usize,
        value: u16,
    );
}

struct U16PixelOutput<'a> {
    pixels: &'a mut Vec<u16>,
}

impl RawDecodeOutput for U16PixelOutput<'_> {
    fn prepare_for_decode(&mut self, pixel_count: usize) -> Result<(), RawDecodeError> {
        if self.pixels.len() != pixel_count {
            return Err(RawDecodeError::OutputBufferLengthMismatch {
                expected: pixel_count,
                actual: self.pixels.len(),
            });
        }

        Ok(())
    }

    fn write_visible_block(
        &mut self,
        visible_width: usize,
        row: usize,
        col: usize,
        block: DecodedBlock<'_>,
    ) {
        let row0_start = row * visible_width;
        let row_block_len = visible_width * 4;

        let rows = &mut self.pixels[row0_start..row0_start + row_block_len];
        let (row0, rows) = rows.split_at_mut(visible_width);
        let (row1, rows) = rows.split_at_mut(visible_width);
        let (row2, row3) = rows.split_at_mut(visible_width);

        let row0_block = &mut row0[col..col + MCRAW_DECODED_BLOCK_SAMPLES];
        let row1_block = &mut row1[col..col + MCRAW_DECODED_BLOCK_SAMPLES];
        let row2_block = &mut row2[col..col + MCRAW_DECODED_BLOCK_SAMPLES];
        let row3_block = &mut row3[col..col + MCRAW_DECODED_BLOCK_SAMPLES];

        let mut i = 0usize;
        while i < MCRAW_DECODED_BLOCK_SAMPLES {
            let src = i / 2;
            let src_hi = MCRAW_DECODED_BLOCK_SAMPLES / 2 + src;

            row0_block[i] = block.p0[src].wrapping_add(block.reference0);
            row0_block[i + 1] = block.p1[src].wrapping_add(block.reference1);

            row1_block[i] = block.p2[src].wrapping_add(block.reference2);
            row1_block[i + 1] = block.p3[src].wrapping_add(block.reference3);

            row2_block[i] = block.p0[src_hi].wrapping_add(block.reference0);
            row2_block[i + 1] = block.p1[src_hi].wrapping_add(block.reference1);

            row3_block[i] = block.p2[src_hi].wrapping_add(block.reference2);
            row3_block[i + 1] = block.p3[src_hi].wrapping_add(block.reference3);

            i += 2;
        }
    }

    fn write_pixel_if_visible(
        &mut self,
        width: usize,
        height: usize,
        row: usize,
        col: usize,
        value: u16,
    ) {
        if row < height && col < width {
            self.pixels[row * width + col] = value;
        }
    }
}

struct LePixelByteOutput<'a> {
    pixel_bytes_le: &'a mut Vec<u8>,
}

impl RawDecodeOutput for LePixelByteOutput<'_> {
    fn prepare_for_decode(&mut self, pixel_count: usize) -> Result<(), RawDecodeError> {
        let pixel_byte_count = pixel_byte_count(pixel_count)?;

        self.pixel_bytes_le.resize(pixel_byte_count, 0);

        Ok(())
    }

    fn write_visible_block(
        &mut self,
        visible_width: usize,
        row: usize,
        col: usize,
        block: DecodedBlock<'_>,
    ) {
        let row_byte_len = visible_width * BYTES_PER_BAYER_U16_SAMPLE;
        let row0_start = row * row_byte_len;
        let row_block_len = row_byte_len * 4;
        let col_byte_start = col * BYTES_PER_BAYER_U16_SAMPLE;
        let block_byte_end =
            col_byte_start + MCRAW_DECODED_BLOCK_SAMPLES * BYTES_PER_BAYER_U16_SAMPLE;

        let rows = &mut self.pixel_bytes_le[row0_start..row0_start + row_block_len];
        let (row0, rows) = rows.split_at_mut(row_byte_len);
        let (row1, rows) = rows.split_at_mut(row_byte_len);
        let (row2, row3) = rows.split_at_mut(row_byte_len);

        let row0_block = &mut row0[col_byte_start..block_byte_end];
        let row1_block = &mut row1[col_byte_start..block_byte_end];
        let row2_block = &mut row2[col_byte_start..block_byte_end];
        let row3_block = &mut row3[col_byte_start..block_byte_end];

        let mut i = 0usize;
        while i < MCRAW_DECODED_BLOCK_SAMPLES {
            let src = i / 2;
            let src_hi = MCRAW_DECODED_BLOCK_SAMPLES / 2 + src;

            write_sample_le(row0_block, i, block.p0[src].wrapping_add(block.reference0));
            write_sample_le(
                row0_block,
                i + 1,
                block.p1[src].wrapping_add(block.reference1),
            );

            write_sample_le(row1_block, i, block.p2[src].wrapping_add(block.reference2));
            write_sample_le(
                row1_block,
                i + 1,
                block.p3[src].wrapping_add(block.reference3),
            );

            write_sample_le(
                row2_block,
                i,
                block.p0[src_hi].wrapping_add(block.reference0),
            );
            write_sample_le(
                row2_block,
                i + 1,
                block.p1[src_hi].wrapping_add(block.reference1),
            );

            write_sample_le(
                row3_block,
                i,
                block.p2[src_hi].wrapping_add(block.reference2),
            );
            write_sample_le(
                row3_block,
                i + 1,
                block.p3[src_hi].wrapping_add(block.reference3),
            );

            i += 2;
        }
    }

    fn write_pixel_if_visible(
        &mut self,
        width: usize,
        height: usize,
        row: usize,
        col: usize,
        value: u16,
    ) {
        if row < height && col < width {
            write_sample_le(self.pixel_bytes_le.as_mut_slice(), row * width + col, value);
        }
    }
}

fn write_edge_pixels<O: RawDecodeOutput>(
    output: &mut O,
    visible_width: usize,
    visible_height: usize,
    y: usize,
    x: usize,
    block: DecodedBlock<'_>,
) {
    let mut i = 0usize;
    while i < MCRAW_DECODED_BLOCK_SAMPLES {
        let src = i / 2;
        let src_hi = MCRAW_DECODED_BLOCK_SAMPLES / 2 + src;

        output.write_pixel_if_visible(
            visible_width,
            visible_height,
            y,
            x + i,
            block.p0[src].wrapping_add(block.reference0),
        );
        output.write_pixel_if_visible(
            visible_width,
            visible_height,
            y,
            x + i + 1,
            block.p1[src].wrapping_add(block.reference1),
        );

        output.write_pixel_if_visible(
            visible_width,
            visible_height,
            y + 1,
            x + i,
            block.p2[src].wrapping_add(block.reference2),
        );
        output.write_pixel_if_visible(
            visible_width,
            visible_height,
            y + 1,
            x + i + 1,
            block.p3[src].wrapping_add(block.reference3),
        );

        output.write_pixel_if_visible(
            visible_width,
            visible_height,
            y + 2,
            x + i,
            block.p0[src_hi].wrapping_add(block.reference0),
        );
        output.write_pixel_if_visible(
            visible_width,
            visible_height,
            y + 2,
            x + i + 1,
            block.p1[src_hi].wrapping_add(block.reference1),
        );

        output.write_pixel_if_visible(
            visible_width,
            visible_height,
            y + 3,
            x + i,
            block.p2[src_hi].wrapping_add(block.reference2),
        );
        output.write_pixel_if_visible(
            visible_width,
            visible_height,
            y + 3,
            x + i + 1,
            block.p3[src_hi].wrapping_add(block.reference3),
        );

        i += 2;
    }
}

fn visible_pixel_count(dimensions: FrameDimensions) -> Result<usize, RawDecodeError> {
    let pixel_count = u64::from(dimensions.width)
        .checked_mul(u64::from(dimensions.height))
        .ok_or(RawDecodeError::FrameDimensionsOverflow)?;

    usize::try_from(pixel_count).map_err(|_| RawDecodeError::FrameDimensionsOverflow)
}

fn pixel_byte_count(pixel_count: usize) -> Result<usize, RawDecodeError> {
    pixel_count
        .checked_mul(BYTES_PER_BAYER_U16_SAMPLE)
        .ok_or(RawDecodeError::FrameDimensionsOverflow)
}

fn visible_raw16_row_byte_len(dimensions: FrameDimensions) -> Result<usize, RawDecodeError> {
    let width =
        usize::try_from(dimensions.width).map_err(|_| RawDecodeError::FrameDimensionsOverflow)?;
    width
        .checked_mul(BYTES_PER_BAYER_U16_SAMPLE)
        .ok_or(RawDecodeError::FrameDimensionsOverflow)
}

fn write_sample_le(output: &mut [u8], pixel_index: usize, value: u16) {
    let byte_index = pixel_index * BYTES_PER_BAYER_U16_SAMPLE;
    let [lo, hi] = value.to_le_bytes();

    output[byte_index] = lo;
    output[byte_index + 1] = hi;
}

// Decode the next 64-sample block from the payload and advance the payload
// cursor. Raw metadata has already been expanded into a BlockEncoding.
fn decode_next_block(
    raw_payload: &[u8],
    offset: &mut usize,
    block: PreparedRawDecodeBlock,
    output: &mut [u16; MCRAW_DECODED_BLOCK_SAMPLES],
) -> Result<(), RawDecodeError> {
    let input = raw_payload
        .get(*offset..)
        .ok_or(RawDecodeError::FrameBlockOffsetOutsidePayload)?;

    let consumed = decode_block(output, block.encoding, input)
        .ok_or(RawDecodeError::FrameBlockDecodeOverflow)?;

    *offset = offset
        .checked_add(consumed)
        .ok_or(RawDecodeError::FrameBlockPayloadOffsetOverflow)?;

    Ok(())
}

// Check raw payload dimensions, padding, metadata offsets, and allocation size
// before decoding. This keeps malformed files or parser bugs from triggering
// unreasonable memory allocation requests.
fn validate_raw_payload_header(
    raw_payload: &[u8],
    header: &MetadataHeader,
    visible_dimensions: FrameDimensions,
) -> Result<(), CpuDecodeError> {
    if header.encoded_width == 0 || header.encoded_height == 0 {
        return Err(CpuDecodeError::EncodedDimensionsZero);
    }

    if !(header.encoded_width as usize).is_multiple_of(ENCODING_BLOCK) {
        return Err(CpuDecodeError::EncodedWidthNotMultipleOfBlock);
    }

    if visible_dimensions.width > header.encoded_width
        || visible_dimensions.height > header.encoded_height
    {
        return Err(CpuDecodeError::VisibleDimensionsExceedEncoded);
    }

    if header.bits_offset as usize > raw_payload.len()
        || header.refs_offset as usize > raw_payload.len()
    {
        return Err(CpuDecodeError::MetadataOffsetsOutsidePayload);
    }

    const MAX_REASONABLE_WIDTH: u32 = 16_384;
    const MAX_REASONABLE_HEIGHT: u32 = 16_384;
    const MAX_REASONABLE_PIXELS: u64 = 100_000_000;

    if header.encoded_width > MAX_REASONABLE_WIDTH || header.encoded_height > MAX_REASONABLE_HEIGHT
    {
        return Err(CpuDecodeError::EncodedDimensionsUnreasonablyLarge);
    }

    let pixel_count = u64::from(visible_dimensions.width) * u64::from(visible_dimensions.height);
    if pixel_count > MAX_REASONABLE_PIXELS {
        return Err(CpuDecodeError::PixelCountUnreasonablyLarge);
    }

    Ok(())
}
