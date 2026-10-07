// MotionCam raw-frame payload metadata parsing.
//
// This is raw codec metadata, not clip/container metadata. It reads the raw
// frame payload header plus packed per-block bit-depth/reference streams used by
// CPU decode preparation and GPU work-plan construction.

use mcraw4vulkan_core::BlockEncoding;

use crate::block::decode_block;
use crate::constants::{ENCODING_BLOCK, HEADER_LENGTH};
use crate::error::RawCodecError;

pub use mcraw4vulkan_core::{MetadataHeader, read_metadata_header};

#[derive(Debug, Clone, Copy)]
pub struct BlockHeader {
    pub encoding: BlockEncoding,
    pub reference: u16,
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

/// Validate the bounded streams and residual walk for a frozen recovered frame.
/// No Bayer samples are decoded and no image-sized coverage map is allocated.
pub fn validate_resolved_payload(
    input: &[u8],
    geometry: mcraw4vulkan_core::ResolvedFrameGeometry,
) -> Result<(), RawCodecError> {
    let invalid = || RawCodecError::InvalidResolvedGeometry;
    if geometry.output_sample_count().is_none() || !geometry.matches_payload(input) {
        return Err(invalid());
    }
    if geometry.reason.is_none() {
        return Ok(());
    }
    let evidence = geometry.type7.ok_or_else(invalid)?;
    let h = evidence.header;
    let lanes = (h.encoded_width / 64)
        .checked_mul(h.encoded_height)
        .ok_or_else(invalid)?;
    let maximum = lanes
        .checked_add(63)
        .map(|n| n / 64 * 64)
        .ok_or_else(invalid)?;
    if h.encoded_width == 0
        || h.encoded_height == 0
        || h.encoded_width > 16_384
        || h.encoded_height > 16_384
        || geometry.effective.width == 0
        || geometry.effective.height == 0
        || geometry.effective.width > h.encoded_width
        || geometry.effective.height > h.encoded_height
        || u64::from(geometry.effective.width) * u64::from(geometry.effective.height) > 100_000_000
        || h.encoded_width & 63 != 0
        || h.encoded_height & 3 != 0
        || h.bits_offset < 16
        || h.bits_offset
            .checked_add(4)
            .is_none_or(|n| n > h.refs_offset)
        || h.refs_offset
            .checked_add(4)
            .is_none_or(|n| n > evidence.payload_len)
        || evidence
            .counts
            .is_none_or(|counts| counts.into_iter().any(|n| n < lanes || n > maximum))
    {
        return Err(invalid());
    }
    let bits_start = usize::try_from(h.bits_offset).map_err(|_| invalid())?;
    let refs_start = usize::try_from(h.refs_offset).map_err(|_| invalid())?;
    let bits = decode_metadata(input.get(bits_start..refs_start).ok_or_else(invalid)?, 0)?;
    let refs = decode_metadata(input.get(refs_start..).ok_or_else(invalid)?, 0)?;
    if bits.len() < lanes as usize || refs.len() < lanes as usize {
        return Err(invalid());
    }
    let mut cursor = 16usize;
    for &encoding in &bits[..lanes as usize] {
        cursor = cursor
            .checked_add(crate::block_len(block_encoding_from_raw(encoding)?))
            .ok_or_else(invalid)?;
        if cursor > bits_start {
            return Err(invalid());
        }
    }
    Ok(())
}
