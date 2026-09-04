// MotionCam raw-frame payload metadata parsing.
//
// This is raw codec metadata, not clip/container metadata. It reads the raw
// frame payload header plus packed per-block bit-depth/reference streams used by
// CPU decode preparation and GPU work-plan construction.

use mcraw4vulkan_core::BlockEncoding;

use crate::block::decode_block;
use crate::constants::{ENCODING_BLOCK, HEADER_LENGTH, METADATA_OFFSET};
use crate::error::RawCodecError;

#[derive(Debug, Clone, Copy)]
pub struct MetadataHeader {
    pub encoded_width: u32,
    pub encoded_height: u32,
    pub bits_offset: u32,
    pub refs_offset: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct BlockHeader {
    pub encoding: BlockEncoding,
    pub reference: u16,
}

// Parse the fixed raw payload metadata header at the start of one compressed raw
// video frame payload.
pub fn read_metadata_header(input: &[u8]) -> Option<MetadataHeader> {
    if input.len() < METADATA_OFFSET {
        return None;
    }

    Some(MetadataHeader {
        encoded_width: u32::from_le_bytes([input[0], input[1], input[2], input[3]]),
        encoded_height: u32::from_le_bytes([input[4], input[5], input[6], input[7]]),
        bits_offset: u32::from_le_bytes([input[8], input[9], input[10], input[11]]),
        refs_offset: u32::from_le_bytes([input[12], input[13], input[14], input[15]]),
    })
}

// Convert raw metadata stream values into the shared BlockEncoding contract.
pub fn block_encoding_from_raw(value: u16) -> Result<BlockEncoding, RawCodecError> {
    match value {
        0 => Ok(BlockEncoding::Zero),
        1 => Ok(BlockEncoding::Bits1),
        2 => Ok(BlockEncoding::Bits2),
        3 => Ok(BlockEncoding::Bits3),
        4 => Ok(BlockEncoding::Bits4),
        5 => Ok(BlockEncoding::Bits5),
        6 => Ok(BlockEncoding::Bits6),
        7 => Ok(BlockEncoding::Bits7),
        8 => Ok(BlockEncoding::Bits8),
        9 => Ok(BlockEncoding::Bits9),
        10 => Ok(BlockEncoding::Bits10),
        11..=15 => Ok(BlockEncoding::Bits16),
        16 => Ok(BlockEncoding::Bits16),
        other => Err(RawCodecError::UnsupportedBlockEncoding(other)),
    }
}

// Parse one packed metadata-stream block header. The high nibble selects the
// residual encoding; the low nibble plus the next byte form a 12-bit reference.
pub fn decode_header(input: &[u8]) -> Option<BlockHeader> {
    if input.len() < HEADER_LENGTH {
        return None;
    }

    let bits = (input[0] >> 4) & 0x0f;
    let reference = (u16::from(input[0] & 0x0f) << 8) | u16::from(input[1]);

    let encoding = block_encoding_from_raw(u16::from(bits)).ok()?;

    Some(BlockHeader {
        encoding,
        reference,
    })
}

// Expand one packed raw metadata stream into one value per raw decode block.
pub fn decode_metadata(input: &[u8], offset: usize) -> Result<Vec<u16>, RawCodecError> {
    let num_blocks_bytes = input
        .get(offset..offset + 4)
        .ok_or(RawCodecError::MetadataBlockCountOutOfBounds)?;

    let num_blocks = u32::from_le_bytes([
        num_blocks_bytes[0],
        num_blocks_bytes[1],
        num_blocks_bytes[2],
        num_blocks_bytes[3],
    ]) as usize;

    let mut out = vec![0u16; num_blocks];
    let mut cursor = offset + 4;

    // Metadata streams use the same 64-value packed blocks as pixel residuals.
    // A short final group publishes only its declared remaining value count.
    for i in (0..num_blocks).step_by(ENCODING_BLOCK) {
        let header_bytes = input
            .get(cursor..cursor + HEADER_LENGTH)
            .ok_or(RawCodecError::MetadataHeaderOutOfBounds)?;

        let block_header =
            decode_header(header_bytes).ok_or(RawCodecError::InvalidMetadataBlockHeader)?;

        cursor += HEADER_LENGTH;

        let mut decoded = [0u16; ENCODING_BLOCK];
        let consumed = decode_block(&mut decoded, block_header.encoding, &input[cursor..])
            .ok_or(RawCodecError::MetadataBlockDecodeOverflow)?;

        cursor += consumed;

        let count = (num_blocks - i).min(ENCODING_BLOCK);
        for x in 0..count {
            out[i + x] = decoded[x].wrapping_add(block_header.reference);
        }
    }

    Ok(out)
}
