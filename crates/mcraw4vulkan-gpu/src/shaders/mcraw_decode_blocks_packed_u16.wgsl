// CPU serialization packs each descriptor into these four little-endian words;
// keep unpack_work_item bit ranges identical to pack_gpu_work_item.
struct WorkItem {
    word0: u32,
    word1: u32,
    word2: u32,
    word3: u32,
};

struct DecodedWorkItem {
    payload_offset: u32,
    payload_len: u32,
    raw_encoding: u32,
    reference_value: u32,
    macroblock_x: u32,
    macroblock_y: u32,
    lane_index: u32,
};

struct Params {
    visible_width: u32,
    visible_height: u32,
    work_item_count: u32,
    macroblock_count: u32,
    dispatch_width: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

@group(0) @binding(0)
var<storage, read> raw_payload_words: array<u32>;

@group(0) @binding(1)
var<storage, read> work_items: array<WorkItem>;

@group(0) @binding(2)
var<storage, read_write> output_pixels: array<atomic<u32>>;

@group(0) @binding(3)
var<uniform> params: Params;

fn read_payload_byte(byte_offset: u32) -> u32 {
    let word = raw_payload_words[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

fn unpack_work_item(item: WorkItem) -> DecodedWorkItem {
    return DecodedWorkItem(
        item.word0,
        (item.word1 >> 24u) & 0xffu,
        (item.word1 >> 16u) & 0xffu,
        item.word1 & 0xffffu,
        item.word2 & 0xffffu,
        (item.word2 >> 16u) & 0xffffu,
        item.word3 & 0xffu,
    );
}

fn read_block_byte_or_zero(item: DecodedWorkItem, local_byte_offset: u32) -> u32 {
    if (local_byte_offset >= item.payload_len) {
        return 0u;
    }

    return read_payload_byte(item.payload_offset + local_byte_offset);
}

fn decode_1(item: DecodedWorkItem, j: u32, row: u32) -> u32 {
    let b = read_block_byte_or_zero(item, j);
    return (b >> row) & 0x01u;
}

fn decode_2(item: DecodedWorkItem, j: u32, row: u32) -> u32 {
    if (row < 4u) {
        let b0 = read_block_byte_or_zero(item, j);
        return (b0 >> (row * 2u)) & 0x03u;
    }

    let b1 = read_block_byte_or_zero(item, 8u + j);
    return (b1 >> ((row - 4u) * 2u)) & 0x03u;
}

fn decode_3(item: DecodedWorkItem, j: u32, row: u32) -> u32 {
    let p0 = read_block_byte_or_zero(item, j);
    let p1 = read_block_byte_or_zero(item, 8u + j);
    let p2 = read_block_byte_or_zero(item, 16u + j);

    if (row == 0u) {
        return p0 & 0x07u;
    }
    if (row == 1u) {
        return (p0 >> 3u) & 0x07u;
    }
    if (row == 2u) {
        return ((p0 >> 6u) & 0x03u) | (((p2 >> 6u) & 0x01u) << 2u);
    }
    if (row == 3u) {
        return p1 & 0x07u;
    }
    if (row == 4u) {
        return (p1 >> 3u) & 0x07u;
    }
    if (row == 5u) {
        return ((p1 >> 6u) & 0x03u) | (((p2 >> 7u) & 0x01u) << 2u);
    }
    if (row == 6u) {
        return p2 & 0x07u;
    }

    return (p2 >> 3u) & 0x07u;
}

fn decode_4(item: DecodedWorkItem, j: u32, row: u32) -> u32 {
    let group = row >> 1u;
    let b = read_block_byte_or_zero(item, group * 8u + j);

    if ((row & 1u) == 0u) {
        return b & 0x0fu;
    }

    return (b >> 4u) & 0x0fu;
}

fn decode_5(item: DecodedWorkItem, j: u32, row: u32) -> u32 {
    let p0 = read_block_byte_or_zero(item, j);
    let p1 = read_block_byte_or_zero(item, 8u + j);
    let p2 = read_block_byte_or_zero(item, 16u + j);
    let p3 = read_block_byte_or_zero(item, 24u + j);
    let p4 = read_block_byte_or_zero(item, 32u + j);

    if (row == 0u) {
        return p0 & 0x1fu;
    }
    if (row == 1u) {
        return p1 & 0x1fu;
    }
    if (row == 2u) {
        return p2 & 0x1fu;
    }
    if (row == 3u) {
        return p3 & 0x1fu;
    }
    if (row == 4u) {
        return p4 & 0x1fu;
    }
    if (row == 5u) {
        return ((p0 >> 5u) & 0x07u) | (((p3 >> 5u) & 0x03u) << 3u);
    }
    if (row == 6u) {
        return ((p1 >> 5u) & 0x07u) | (((p4 >> 5u) & 0x03u) << 3u);
    }

    let tmp = ((p2 >> 5u) & 0x07u) | (((p3 >> 7u) & 0x01u) << 3u);
    return tmp | (((p4 >> 7u) & 0x01u) << 4u);
}

fn decode_6(item: DecodedWorkItem, j: u32, row: u32) -> u32 {
    let p0 = read_block_byte_or_zero(item, j);
    let p1 = read_block_byte_or_zero(item, 8u + j);
    let p2 = read_block_byte_or_zero(item, 16u + j);
    let p3 = read_block_byte_or_zero(item, 24u + j);
    let p4 = read_block_byte_or_zero(item, 32u + j);
    let p5 = read_block_byte_or_zero(item, 40u + j);

    if (row == 0u) {
        return p0 & 0x3fu;
    }
    if (row == 1u) {
        return p1 & 0x3fu;
    }
    if (row == 2u) {
        return p2 & 0x3fu;
    }
    if (row == 3u) {
        return p3 & 0x3fu;
    }
    if (row == 4u) {
        return p4 & 0x3fu;
    }
    if (row == 5u) {
        return p5 & 0x3fu;
    }
    if (row == 6u) {
        return ((p0 >> 6u) & 0x03u)
            | (((p1 >> 6u) & 0x03u) << 2u)
            | (((p2 >> 6u) & 0x03u) << 4u);
    }

    return ((p3 >> 6u) & 0x03u)
        | (((p4 >> 6u) & 0x03u) << 2u)
        | (((p5 >> 6u) & 0x03u) << 4u);
}

fn decode_8(item: DecodedWorkItem, sample_index: u32) -> u32 {
    return read_block_byte_or_zero(item, sample_index) & 0xffu;
}

fn decode_10(item: DecodedWorkItem, j: u32, row: u32) -> u32 {
    let p0 = read_block_byte_or_zero(item, j);
    let p1 = read_block_byte_or_zero(item, 8u + j);
    let p2 = read_block_byte_or_zero(item, 16u + j);
    let p3 = read_block_byte_or_zero(item, 24u + j);
    let p4 = read_block_byte_or_zero(item, 32u + j);

    if (row == 0u) {
        return p0 | ((p4 & 0x03u) << 8u);
    }
    if (row == 1u) {
        return p1 | (((p4 >> 2u) & 0x03u) << 8u);
    }
    if (row == 2u) {
        return p2 | (((p4 >> 4u) & 0x03u) << 8u);
    }
    if (row == 3u) {
        return p3 | (((p4 >> 6u) & 0x03u) << 8u);
    }

    let p5 = read_block_byte_or_zero(item, 40u + j);
    let p6 = read_block_byte_or_zero(item, 48u + j);
    let p7 = read_block_byte_or_zero(item, 56u + j);
    let p8 = read_block_byte_or_zero(item, 64u + j);
    let p9 = read_block_byte_or_zero(item, 72u + j);

    if (row == 4u) {
        return p5 | ((p9 & 0x03u) << 8u);
    }
    if (row == 5u) {
        return p6 | (((p9 >> 2u) & 0x03u) << 8u);
    }
    if (row == 6u) {
        return p7 | (((p9 >> 4u) & 0x03u) << 8u);
    }

    return p8 | (((p9 >> 6u) & 0x03u) << 8u);
}

fn decode_16(item: DecodedWorkItem, sample_index: u32) -> u32 {
    let byte_offset = sample_index * 2u;
    let lo = read_block_byte_or_zero(item, byte_offset);
    let hi = read_block_byte_or_zero(item, byte_offset + 1u);

    return lo | (hi << 8u);
}

fn decode_mcraw_sample(item: DecodedWorkItem, sample_index: u32) -> u32 {
    let j = sample_index & 7u;
    let row = sample_index >> 3u;
    let encoding = item.raw_encoding;

    if (encoding == 1u) {
        return decode_1(item, j, row);
    }
    if (encoding == 2u) {
        return decode_2(item, j, row);
    }
    if (encoding == 3u) {
        return decode_3(item, j, row);
    }
    if (encoding == 4u) {
        return decode_4(item, j, row);
    }
    if (encoding == 5u) {
        return decode_5(item, j, row);
    }
    if (encoding == 6u) {
        return decode_6(item, j, row);
    }
    if (encoding == 7u || encoding == 8u) {
        return decode_8(item, sample_index);
    }
    if (encoding == 9u || encoding == 10u) {
        return decode_10(item, j, row);
    }
    // CPU metadata maps raw encodings 11..=16 to BlockEncoding::Bits16.
    // Those blocks store 64 little-endian u16 samples in the payload.
    if (encoding >= 11u && encoding <= 16u) {
        return decode_16(item, sample_index);
    }

    return 0u;
}

fn write_visible_pixel(row: u32, col: u32, value: u32) {
    if (row < params.visible_height && col < params.visible_width) {
        let pixel_index = row * params.visible_width + col;
        let packed_index = pixel_index >> 1u;
        let sample = value & 0xffffu;

        // Raster neighbors share a word: the even pixel owns the low half and
        // the odd pixel owns the high half. The host clears words before atomicOr.
        if ((pixel_index & 1u) == 0u) {
            atomicOr(&output_pixels[packed_index], sample);
        } else {
            atomicOr(&output_pixels[packed_index], sample << 16u);
        }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
) {
    let macroblock_index = workgroup_id.x + workgroup_id.y * params.dispatch_width;
    let local_id = local_invocation_id.x;

    let lane_offset = local_id >> 6u;
    let sample_index = local_id & 63u;
    let descriptor_index = macroblock_index * 4u + lane_offset;

    if (macroblock_index >= params.macroblock_count) {
        return;
    }

    if (descriptor_index >= params.work_item_count) {
        return;
    }

    let item = unpack_work_item(work_items[descriptor_index]);

    var value = decode_mcraw_sample(item, sample_index);
    value = (value + item.reference_value) & 0xffffu;

    var column_parity = 0u;
    var top_row = 0u;
    var bottom_row = 0u;

    if (item.lane_index == 0u) {
        top_row = item.macroblock_y;
        bottom_row = item.macroblock_y + 2u;
        column_parity = 0u;
    } else if (item.lane_index == 1u) {
        top_row = item.macroblock_y;
        bottom_row = item.macroblock_y + 2u;
        column_parity = 1u;
    } else if (item.lane_index == 2u) {
        top_row = item.macroblock_y + 1u;
        bottom_row = item.macroblock_y + 3u;
        column_parity = 0u;
    } else if (item.lane_index == 3u) {
        top_row = item.macroblock_y + 1u;
        bottom_row = item.macroblock_y + 3u;
        column_parity = 1u;
    } else {
        return;
    }

    if (sample_index < 32u) {
        let col = item.macroblock_x + sample_index * 2u + column_parity;
        write_visible_pixel(top_row, col, value);
    } else {
        let col = item.macroblock_x + (sample_index - 32u) * 2u + column_parity;
        write_visible_pixel(bottom_row, col, value);
    }
}
