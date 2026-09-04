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
    chunk_count: u32,
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
var<storage, read_write> output_words: array<u32>;

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

fn decode_msb_packed_bits(item: DecodedWorkItem, sample_index: u32) -> u32 {
    var value = 0u;
    let start_bit = sample_index * item.raw_encoding;
    var bit_index = 0u;
    loop {
        if (bit_index >= item.raw_encoding) {
            break;
        }

        let payload_bit = start_bit + bit_index;
        let byte = read_payload_byte(item.payload_offset + (payload_bit >> 3u));
        let shift = 7u - (payload_bit & 7u);
        value = (value << 1u) | ((byte >> shift) & 1u);
        bit_index = bit_index + 1u;
    }
    return value;
}

fn decode_legacy_raw16_delta(item: DecodedWorkItem, sample_index: u32) -> u32 {
    if (item.raw_encoding == 0u) {
        return 0u;
    }

    if (item.raw_encoding >= 11u) {
        let offset = item.payload_offset + sample_index * 2u;
        return (read_payload_byte(offset) << 8u) | read_payload_byte(offset + 1u);
    }

    return decode_msb_packed_bits(item, sample_index);
}

fn decode_legacy_raw16_sample(item: DecodedWorkItem, sample_index: u32) -> u32 {
    return (decode_legacy_raw16_delta(item, sample_index) + item.reference_value) & 0xffffu;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear_group = global_id.y * params.dispatch_width + global_id.x / 128u;
    let local_index = global_id.x % 128u;
    let packed_word_index = linear_group * 128u + local_index;
    let chunk_index = packed_word_index >> 4u;

    if (chunk_index >= params.chunk_count) {
        return;
    }

    let sample_index = packed_word_index & 15u;
    let item_index = chunk_index * 2u;
    let item0 = unpack_work_item(work_items[item_index]);
    let item1 = unpack_work_item(work_items[item_index + 1u]);
    let row = item0.macroblock_y;
    let x0 = item0.macroblock_x + sample_index * 2u;
    let x1 = x0 + 1u;

    if (row >= params.visible_height || x0 >= params.visible_width) {
        return;
    }

    let low = decode_legacy_raw16_sample(item0, sample_index);
    var high = 0u;
    if (x1 < params.visible_width) {
        high = decode_legacy_raw16_sample(item1, sample_index);
    }

    // Adjacent raster samples share a word, with x0 in the low half. An absent
    // odd-edge sample remains zero in the high half.
    let output_word_index = ((row * params.visible_width) + x0) >> 1u;
    output_words[output_word_index] = low | (high << 16u);
}
