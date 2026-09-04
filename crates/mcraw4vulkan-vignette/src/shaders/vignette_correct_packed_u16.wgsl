// Rust writes this exact 128-byte sequence as 32 little-endian words; field
// order must remain synchronized with GpuVignetteCorrectionParams::to_le_bytes.
struct Params {
    global_pixel_count: u32,
    global_packed_word_count: u32,
    frame_width: u32,
    frame_height: u32,
    black_q0: u32,
    black_q1: u32,
    black_q2: u32,
    black_q3: u32,
    output_white_level: u32,
    base_pixel_offset: u32,
    base_packed_word_offset: u32,
    tile_pixel_count: u32,
    tile_packed_word_count: u32,
    bayer_pattern_tag: u32,
    source_map_width: u32,
    source_map_height: u32,
    source_plane_count: u32,
    conversion_policy_tag: u32,
    scale_num: u32,
    source_map_word_count: u32,
    scale_shift: u32,
    strength_num: u32,
    strength_shift: u32,
    _padding3: u32,
    _padding4: u32,
    _padding5: u32,
    _padding6: u32,
    _padding7: u32,
    _padding8: u32,
    _padding9: u32,
    _padding10: u32,
    _padding11: u32,
};

// Split-word integers preserve Q16.16 interpolation and correction without f32
// rounding differences from the CPU path.
struct U64Parts {
    lo: u32,
    hi: u32,
};

struct U96Parts {
    lo: u32,
    mid: u32,
    hi: u32,
};

struct AxisPosition {
    i0: u32,
    i1: u32,
    t_q: u32,
};

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;

// Compact gains are plane-major Q16.16 values, each stored low word then high.
@group(0) @binding(1)
var<storage, read> compact_gains_q: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output_words: array<u32>;

@group(0) @binding(3)
var<uniform> params: Params;

fn mul_u32_u32_to_u64(a: u32, b: u32) -> U64Parts {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;

    let p0 = a0 * b0;
    let p1 = a0 * b1;
    let p2 = a1 * b0;
    let p3 = a1 * b1;

    let middle = (p0 >> 16u) + (p1 & 0xffffu) + (p2 & 0xffffu);
    let lo = (p0 & 0xffffu) | (middle << 16u);
    let hi = p3 + (p1 >> 16u) + (p2 >> 16u) + (middle >> 16u);

    var result: U64Parts;
    result.lo = lo;
    result.hi = hi;
    return result;
}

fn add_u64(a: U64Parts, b: U64Parts) -> U64Parts {
    var result: U64Parts;
    result.lo = a.lo + b.lo;
    result.hi = a.hi + b.hi;
    if (result.lo < a.lo) {
        result.hi = result.hi + 1u;
    }
    return result;
}

fn sub_u32_from_u64(a: U64Parts, b: u32) -> U64Parts {
    var result: U64Parts;
    result.lo = a.lo - b;
    result.hi = a.hi;
    if (a.lo < b) {
        result.hi = result.hi - 1u;
    }
    return result;
}

fn add_u32_to_u64(a: U64Parts, b: u32) -> U64Parts {
    var result: U64Parts;
    result.lo = a.lo + b;
    result.hi = a.hi;
    if (result.lo < a.lo) {
        result.hi = result.hi + 1u;
    }
    return result;
}

fn add_u96(a: U96Parts, b: U96Parts) -> U96Parts {
    var result: U96Parts;
    result.lo = a.lo + b.lo;
    var carry0 = 0u;
    if (result.lo < a.lo) {
        carry0 = 1u;
    }
    result.mid = a.mid + b.mid;
    var carry1 = 0u;
    if (result.mid < a.mid) {
        carry1 = 1u;
    }
    result.mid = result.mid + carry0;
    if (result.mid < carry0) {
        carry1 = carry1 + 1u;
    }
    result.hi = a.hi + b.hi + carry1;
    return result;
}

fn u96_from_u64(a: U64Parts) -> U96Parts {
    var result: U96Parts;
    result.lo = a.lo;
    result.mid = a.hi;
    result.hi = 0u;
    return result;
}

fn shl_u32_to_u64(value: u32, shift: u32) -> U64Parts {
    var result: U64Parts;
    if (shift == 0u) {
        result.lo = value;
        result.hi = 0u;
        return result;
    }
    if (shift < 32u) {
        result.lo = value << shift;
        result.hi = value >> (32u - shift);
        return result;
    }
    result.lo = 0u;
    result.hi = value << (shift - 32u);
    return result;
}

fn mul_u64_u32_to_u96(a: U64Parts, b: u32) -> U96Parts {
    let lo_product = mul_u32_u32_to_u64(a.lo, b);
    let hi_product = mul_u32_u32_to_u64(a.hi, b);

    var result: U96Parts;
    result.lo = lo_product.lo;
    result.mid = lo_product.hi + hi_product.lo;
    var carry = 0u;
    if (result.mid < lo_product.hi) {
        carry = 1u;
    }
    result.hi = hi_product.hi + carry;
    return result;
}

fn add_power_of_two_to_u96(a: U96Parts, bit_index: u32) -> U96Parts {
    var result = a;
    if (bit_index < 32u) {
        let addend = 1u << bit_index;
        result.lo = result.lo + addend;
        if (result.lo < addend) {
            result.mid = result.mid + 1u;
            if (result.mid == 0u) {
                result.hi = result.hi + 1u;
            }
        }
        return result;
    }
    if (bit_index < 64u) {
        let addend = 1u << (bit_index - 32u);
        result.mid = result.mid + addend;
        if (result.mid < addend) {
            result.hi = result.hi + 1u;
        }
        return result;
    }
    result.hi = result.hi + (1u << (bit_index - 64u));
    return result;
}

fn shr_u96_to_u32(a: U96Parts, shift: u32) -> u32 {
    if (shift == 0u) {
        return a.lo;
    }
    if (shift < 32u) {
        return (a.lo >> shift) | (a.mid << (32u - shift));
    }
    if (shift == 32u) {
        return a.mid;
    }
    if (shift < 64u) {
        return (a.mid >> (shift - 32u)) | (a.hi << (64u - shift));
    }
    if (shift == 64u) {
        return a.hi;
    }
    return a.hi >> (shift - 64u);
}

fn shr_u96_to_u64(a: U96Parts, shift: u32) -> U64Parts {
    var result: U64Parts;
    if (shift == 0u) {
        result.lo = a.lo;
        result.hi = a.mid;
        return result;
    }
    if (shift < 32u) {
        result.lo = (a.lo >> shift) | (a.mid << (32u - shift));
        result.hi = (a.mid >> shift) | (a.hi << (32u - shift));
        return result;
    }
    if (shift == 32u) {
        result.lo = a.mid;
        result.hi = a.hi;
        return result;
    }
    if (shift < 64u) {
        result.lo = (a.mid >> (shift - 32u)) | (a.hi << (64u - shift));
        result.hi = a.hi >> (shift - 32u);
        return result;
    }
    if (shift == 64u) {
        result.lo = a.hi;
        result.hi = 0u;
        return result;
    }
    result.lo = a.hi >> (shift - 64u);
    result.hi = 0u;
    return result;
}

fn round_shift_u96_to_u32(a: U96Parts, shift: u32) -> u32 {
    let rounded = add_power_of_two_to_u96(a, shift - 1u);
    return shr_u96_to_u32(rounded, shift);
}

fn round_shift_u96_to_u64(a: U96Parts, shift: u32) -> U64Parts {
    let rounded = add_power_of_two_to_u96(a, shift - 1u);
    return shr_u96_to_u64(rounded, shift);
}

fn div_round_half_up_u32(numerator: u32, denominator: u32) -> u32 {
    return (numerator + (denominator / 2u)) / denominator;
}

fn fixed_axis_position(position: u32, frame_dimension: u32, map_dimension: u32) -> AxisPosition {
    var result: AxisPosition;
    if (map_dimension <= 1u || frame_dimension <= 1u) {
        result.i0 = 0u;
        result.i1 = 0u;
        result.t_q = 0u;
        return result;
    }

    let frame_last = frame_dimension - 1u;
    let map_last = map_dimension - 1u;
    let numerator = position * map_last;
    let i0 = numerator / frame_last;
    let remainder = numerator - (i0 * frame_last);
    result.i0 = min(i0, map_last);
    result.i1 = min(i0 + 1u, map_last);
    result.t_q = div_round_half_up_u32(remainder * 65536u, frame_last);
    return result;
}

fn lerp_fixed_q(a: U64Parts, b: U64Parts, t_q: u32) -> U64Parts {
    let inverse_t_q = 65536u - t_q;
    let weighted_a = mul_u64_u32_to_u96(a, inverse_t_q);
    let weighted_b = mul_u64_u32_to_u96(b, t_q);
    return round_shift_u96_to_u64(add_u96(weighted_a, weighted_b), 16u);
}

fn compact_plane_index(pixel_index: u32) -> u32 {
    let x = pixel_index % params.frame_width;
    let y = pixel_index / params.frame_width;
    let xe = x & 1u;
    let ye = y & 1u;

    if (params.source_plane_count <= 1u) {
        return 0u;
    }

    if (params.bayer_pattern_tag == 0u) {
        if (xe == 0u && ye == 0u) { return 0u; }
        if (xe == 1u && ye == 1u) { return 3u; }
        if (ye == 0u) { return 1u; }
        return 2u;
    }
    if (params.bayer_pattern_tag == 1u) {
        if (xe == 1u && ye == 1u) { return 0u; }
        if (xe == 0u && ye == 0u) { return 3u; }
        if (ye == 0u) { return 1u; }
        return 2u;
    }
    if (params.bayer_pattern_tag == 2u) {
        if (xe == 1u && ye == 0u) { return 0u; }
        if (xe == 0u && ye == 1u) { return 3u; }
        if (ye == 0u) { return 1u; }
        return 2u;
    }
    if (params.bayer_pattern_tag == 3u) {
        if (xe == 0u && ye == 1u) { return 0u; }
        if (xe == 1u && ye == 0u) { return 3u; }
        if (ye == 0u) { return 1u; }
        return 2u;
    }

    return ((ye * 2u) + xe) % max(params.source_plane_count, 1u);
}

fn compact_gain_q_at(gain_index: u32) -> U64Parts {
    let word_index = gain_index * 2u;
    var result: U64Parts;
    result.lo = compact_gains_q[word_index];
    result.hi = compact_gains_q[word_index + 1u];
    return result;
}

fn compact_gain_q_for_pixel(pixel_index: u32) -> U64Parts {
    let x = pixel_index % params.frame_width;
    let y = pixel_index / params.frame_width;
    let xp = fixed_axis_position(x, params.frame_width, params.source_map_width);
    let yp = fixed_axis_position(y, params.frame_height, params.source_map_height);
    let plane = compact_plane_index(pixel_index);
    let plane_len = params.source_map_width * params.source_map_height;
    let plane_base = plane * plane_len;

    let top_left_index = plane_base + (yp.i0 * params.source_map_width) + xp.i0;
    let top_right_index = plane_base + (yp.i0 * params.source_map_width) + xp.i1;
    let bottom_left_index = plane_base + (yp.i1 * params.source_map_width) + xp.i0;
    let bottom_right_index = plane_base + (yp.i1 * params.source_map_width) + xp.i1;

    let top = lerp_fixed_q(compact_gain_q_at(top_left_index), compact_gain_q_at(top_right_index), xp.t_q);
    let bottom = lerp_fixed_q(compact_gain_q_at(bottom_left_index), compact_gain_q_at(bottom_right_index), xp.t_q);
    return lerp_fixed_q(top, bottom, yp.t_q);
}

fn converted_gain_q16(raw_gain_q: U64Parts) -> u32 {
    if (params.conversion_policy_tag == 1u) {
        return raw_gain_q.lo;
    }

    var delta: U64Parts;
    delta.lo = 0u;
    delta.hi = 0u;
    if (raw_gain_q.hi > 0u || raw_gain_q.lo > 65536u) {
        delta = sub_u32_from_u64(raw_gain_q, 65536u);
    }

    let base = u96_from_u64(shl_u32_to_u64(65536u, params.strength_shift));
    let variable = mul_u64_u32_to_u96(delta, params.strength_num);
    let unscaled = add_u96(base, variable);
    let scaled = mul_u64_u32_to_u96(shr_u96_to_u64(unscaled, 0u), params.scale_num);
    return round_shift_u96_to_u32(scaled, params.scale_shift + params.strength_shift);
}

fn final_gain_q16_for_pixel(pixel_index: u32) -> u32 {
    return converted_gain_q16(compact_gain_q_for_pixel(pixel_index));
}

fn black_q_for_plane(plane_index: u32) -> u32 {
    if (plane_index == 0u) {
        return params.black_q0;
    }
    if (plane_index == 1u) {
        return params.black_q1;
    }
    if (plane_index == 2u) {
        return params.black_q2;
    }

    return params.black_q3;
}

fn corrected_sample(pixel_index: u32, raw_sample: u32) -> u32 {
    let x = pixel_index % params.frame_width;
    let y = pixel_index / params.frame_width;
    let plane_index = ((y & 1u) * 2u) + (x & 1u);
    let black_q = black_q_for_plane(plane_index);
    let raw_q = raw_sample << 16u;

    var signal_q = 0u;
    if (raw_q > black_q) {
        signal_q = raw_q - black_q;
    }

    let product = mul_u32_u32_to_u64(signal_q, final_gain_q16_for_pixel(pixel_index));
    var corrected = product.hi;
    if (product.lo >= 0x80000000u && corrected < 0xffffffffu) {
        corrected = corrected + 1u;
    }

    let white = min(params.output_white_level, 0xffffu);
    return min(corrected, white) & 0xffffu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let tile_packed_word_index = global_invocation_id.x;
    if (tile_packed_word_index >= params.tile_packed_word_count) {
        return;
    }

    let packed_word_index = params.base_packed_word_offset + tile_packed_word_index;
    if (packed_word_index >= params.global_packed_word_count) {
        return;
    }

    let word = input_words[packed_word_index];
    let first_pixel_index = packed_word_index * 2u;
    let first_raw_sample = word & 0xffffu;

    var output_word = 0u;
    if (first_pixel_index < params.global_pixel_count) {
        let first_tile_pixel_index = first_pixel_index - params.base_pixel_offset;
        if (first_tile_pixel_index < params.tile_pixel_count) {
            output_word = corrected_sample(first_pixel_index, first_raw_sample);
        }
    }

    let second_pixel_index = first_pixel_index + 1u;
    if (second_pixel_index < params.global_pixel_count) {
        let second_tile_pixel_index = second_pixel_index - params.base_pixel_offset;
        let second_raw_sample = (word >> 16u) & 0xffffu;
        if (second_tile_pixel_index < params.tile_pixel_count) {
            let second_corrected = corrected_sample(second_pixel_index, second_raw_sample);
            output_word = output_word | (second_corrected << 16u);
        }
    }

    // Raster neighbors keep their input word ownership: lower-indexed pixel in
    // the low half and an absent odd-edge pixel left as zero in the high half.
    output_words[packed_word_index] = output_word;
}
