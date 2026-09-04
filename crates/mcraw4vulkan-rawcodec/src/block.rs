// Decodes MotionCam 64-value packed blocks for each supported bit depth using
// bounds-checked scalar Rust operations.
//
// The hot 5-bit and 6-bit paths use fixed 8-sample lanes. This keeps the code
// safe while giving LLVM clearer slice-size information than repeated dynamic
// indexing into the full input/output block.
use mcraw4vulkan_core::BlockEncoding;

use crate::constants::{ENCODING_BLOCK, ENCODING_BLOCK_LENGTH};

type OutputLaneMut<'a> = &'a mut [u16; 8];
type OutputLanesMut<'a> = (
    OutputLaneMut<'a>,
    OutputLaneMut<'a>,
    OutputLaneMut<'a>,
    OutputLaneMut<'a>,
    OutputLaneMut<'a>,
    OutputLaneMut<'a>,
    OutputLaneMut<'a>,
    OutputLaneMut<'a>,
);

#[inline(always)]
pub fn block_len(encoding: BlockEncoding) -> usize {
    ENCODING_BLOCK_LENGTH[encoding as usize]
}

#[inline(always)]
pub fn decode_block(
    output: &mut [u16; 64],
    encoding: BlockEncoding,
    input: &[u8],
) -> Option<usize> {
    let needed = block_len(encoding);
    if input.len() < needed {
        return None;
    }

    let consumed = match encoding {
        BlockEncoding::Zero => {
            output.fill(0);
            0
        }
        BlockEncoding::Bits1 => decode_1(output, input),
        BlockEncoding::Bits2 => decode_2(output, input),
        BlockEncoding::Bits3 => decode_3(output, input),
        BlockEncoding::Bits4 => decode_4(output, input),
        BlockEncoding::Bits5 => decode_5(output, input),
        BlockEncoding::Bits6 => decode_6(output, input),
        BlockEncoding::Bits7 | BlockEncoding::Bits8 => decode_8(output, input),
        BlockEncoding::Bits9 | BlockEncoding::Bits10 => decode_10(output, input),
        BlockEncoding::Bits16 => decode_16(output, input),
    };

    Some(consumed)
}

#[inline(always)]
fn decode_1(output: &mut [u16; ENCODING_BLOCK], input: &[u8]) -> usize {
    for j in 0..8 {
        let b = input[j];
        output[j] = u16::from(b & 0x01);
        output[8 + j] = u16::from((b >> 1) & 0x01);
        output[16 + j] = u16::from((b >> 2) & 0x01);
        output[24 + j] = u16::from((b >> 3) & 0x01);
        output[32 + j] = u16::from((b >> 4) & 0x01);
        output[40 + j] = u16::from((b >> 5) & 0x01);
        output[48 + j] = u16::from((b >> 6) & 0x01);
        output[56 + j] = u16::from((b >> 7) & 0x01);
    }

    ENCODING_BLOCK_LENGTH[1]
}

#[inline(always)]
fn decode_2(output: &mut [u16; ENCODING_BLOCK], input: &[u8]) -> usize {
    for j in 0..8 {
        let b0 = input[j];
        output[j] = u16::from(b0 & 0x03);
        output[8 + j] = u16::from((b0 >> 2) & 0x03);
        output[16 + j] = u16::from((b0 >> 4) & 0x03);
        output[24 + j] = u16::from((b0 >> 6) & 0x03);

        let b1 = input[8 + j];
        output[32 + j] = u16::from(b1 & 0x03);
        output[40 + j] = u16::from((b1 >> 2) & 0x03);
        output[48 + j] = u16::from((b1 >> 4) & 0x03);
        output[56 + j] = u16::from((b1 >> 6) & 0x03);
    }

    ENCODING_BLOCK_LENGTH[2]
}

#[inline(always)]
fn decode_3(output: &mut [u16; ENCODING_BLOCK], input: &[u8]) -> usize {
    for j in 0..8 {
        let p0 = input[j];
        let p1 = input[8 + j];
        let p2 = input[16 + j];

        output[j] = u16::from(p0 & 0x07);
        output[8 + j] = u16::from((p0 >> 3) & 0x07);
        output[16 + j] = u16::from(((p0 >> 6) & 0x03) | (((p2 >> 6) & 0x01) << 2));

        output[24 + j] = u16::from(p1 & 0x07);
        output[32 + j] = u16::from((p1 >> 3) & 0x07);
        output[40 + j] = u16::from(((p1 >> 6) & 0x03) | (((p2 >> 7) & 0x01) << 2));

        output[48 + j] = u16::from(p2 & 0x07);
        output[56 + j] = u16::from((p2 >> 3) & 0x07);
    }

    ENCODING_BLOCK_LENGTH[3]
}

#[inline(always)]
fn decode_4(output: &mut [u16; ENCODING_BLOCK], input: &[u8]) -> usize {
    for group in 0..4 {
        let base_in = group * 8;
        let base_out = group * 16;

        for j in 0..8 {
            let b = input[base_in + j];
            output[base_out + j] = u16::from(b & 0x0f);
            output[base_out + 8 + j] = u16::from((b >> 4) & 0x0f);
        }
    }

    ENCODING_BLOCK_LENGTH[4]
}

#[inline(always)]
fn decode_5(output: &mut [u16; ENCODING_BLOCK], input: &[u8]) -> usize {
    let i0 = input_lane(input, 0);
    let i1 = input_lane(input, 8);
    let i2 = input_lane(input, 16);
    let i3 = input_lane(input, 24);
    let i4 = input_lane(input, 32);

    let (o0, o1, o2, o3, o4, o5, o6, o7) = output_lanes_mut(output);

    for j in 0..8 {
        let p0 = i0[j];
        let p1 = i1[j];
        let p2 = i2[j];
        let p3 = i3[j];
        let p4 = i4[j];

        o0[j] = u16::from(p0 & 0x1f);
        o1[j] = u16::from(p1 & 0x1f);
        o2[j] = u16::from(p2 & 0x1f);
        o3[j] = u16::from(p3 & 0x1f);
        o4[j] = u16::from(p4 & 0x1f);

        o5[j] = u16::from(((p0 >> 5) & 0x07) | (((p3 >> 5) & 0x03) << 3));
        o6[j] = u16::from(((p1 >> 5) & 0x07) | (((p4 >> 5) & 0x03) << 3));

        let tmp = ((p2 >> 5) & 0x07) | (((p3 >> 7) & 0x01) << 3);
        o7[j] = u16::from(tmp | (((p4 >> 7) & 0x01) << 4));
    }

    ENCODING_BLOCK_LENGTH[5]
}

#[inline(always)]
fn decode_6(output: &mut [u16; ENCODING_BLOCK], input: &[u8]) -> usize {
    let i0 = input_lane(input, 0);
    let i1 = input_lane(input, 8);
    let i2 = input_lane(input, 16);
    let i3 = input_lane(input, 24);
    let i4 = input_lane(input, 32);
    let i5 = input_lane(input, 40);

    let (o0, o1, o2, o3, o4, o5, o6, o7) = output_lanes_mut(output);

    for j in 0..8 {
        let p0 = i0[j];
        let p1 = i1[j];
        let p2 = i2[j];
        let p3 = i3[j];
        let p4 = i4[j];
        let p5 = i5[j];

        o0[j] = u16::from(p0 & 0x3f);
        o1[j] = u16::from(p1 & 0x3f);
        o2[j] = u16::from(p2 & 0x3f);
        o3[j] = u16::from(p3 & 0x3f);
        o4[j] = u16::from(p4 & 0x3f);
        o5[j] = u16::from(p5 & 0x3f);

        o6[j] =
            u16::from(((p0 >> 6) & 0x03) | (((p1 >> 6) & 0x03) << 2) | (((p2 >> 6) & 0x03) << 4));

        o7[j] =
            u16::from(((p3 >> 6) & 0x03) | (((p4 >> 6) & 0x03) << 2) | (((p5 >> 6) & 0x03) << 4));
    }

    ENCODING_BLOCK_LENGTH[6]
}

#[inline(always)]
fn decode_8(output: &mut [u16; ENCODING_BLOCK], input: &[u8]) -> usize {
    for (dst, src) in output.iter_mut().zip(input.iter().take(ENCODING_BLOCK)) {
        *dst = u16::from(*src);
    }

    ENCODING_BLOCK_LENGTH[8]
}

#[inline(always)]
fn decode_10(output: &mut [u16; ENCODING_BLOCK], input: &[u8]) -> usize {
    for j in 0..8 {
        let p0 = input[j];
        let p1 = input[8 + j];
        let p2 = input[16 + j];
        let p3 = input[24 + j];
        let p4 = input[32 + j];

        output[j] = u16::from(p0) | (u16::from(p4 & 0x03) << 8);
        output[8 + j] = u16::from(p1) | (u16::from((p4 >> 2) & 0x03) << 8);
        output[16 + j] = u16::from(p2) | (u16::from((p4 >> 4) & 0x03) << 8);
        output[24 + j] = u16::from(p3) | (u16::from((p4 >> 6) & 0x03) << 8);

        let p5 = input[40 + j];
        let p6 = input[48 + j];
        let p7 = input[56 + j];
        let p8 = input[64 + j];
        let p9 = input[72 + j];

        output[32 + j] = u16::from(p5) | (u16::from(p9 & 0x03) << 8);
        output[40 + j] = u16::from(p6) | (u16::from((p9 >> 2) & 0x03) << 8);
        output[48 + j] = u16::from(p7) | (u16::from((p9 >> 4) & 0x03) << 8);
        output[56 + j] = u16::from(p8) | (u16::from((p9 >> 6) & 0x03) << 8);
    }

    ENCODING_BLOCK_LENGTH[10]
}

#[inline(always)]
fn decode_16(output: &mut [u16; ENCODING_BLOCK], input: &[u8]) -> usize {
    for i in 0..ENCODING_BLOCK {
        let lo = input[i * 2];
        let hi = input[i * 2 + 1];
        output[i] = u16::from_le_bytes([lo, hi]);
    }

    ENCODING_BLOCK_LENGTH[16]
}

#[inline(always)]
fn input_lane(input: &[u8], start: usize) -> &[u8; 8] {
    input[start..start + 8]
        .try_into()
        .expect("block decoder input lane must be 8 bytes")
}

#[inline(always)]
fn output_lanes_mut(output: &mut [u16; ENCODING_BLOCK]) -> OutputLanesMut<'_> {
    let (o0, rest) = output.split_at_mut(8);
    let (o1, rest) = rest.split_at_mut(8);
    let (o2, rest) = rest.split_at_mut(8);
    let (o3, rest) = rest.split_at_mut(8);
    let (o4, rest) = rest.split_at_mut(8);
    let (o5, rest) = rest.split_at_mut(8);
    let (o6, o7) = rest.split_at_mut(8);

    (
        o0.try_into().expect("output lane 0 must be 8 samples"),
        o1.try_into().expect("output lane 1 must be 8 samples"),
        o2.try_into().expect("output lane 2 must be 8 samples"),
        o3.try_into().expect("output lane 3 must be 8 samples"),
        o4.try_into().expect("output lane 4 must be 8 samples"),
        o5.try_into().expect("output lane 5 must be 8 samples"),
        o6.try_into().expect("output lane 6 must be 8 samples"),
        o7.try_into().expect("output lane 7 must be 8 samples"),
    )
}
