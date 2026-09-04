// Bindings and terminal for direct normalized BT.2020-NCL to tightly packed
// planar TV-range 12-bit codes. The host source-composes the shared signed
// Bayer demosaic at the marked insertion point below.

struct DirectYuv12Params {
    // width, height, Bayer-pattern tag, plane words
    dimensions_pattern_plane: vec4<u32>,
    camera_to_ncl_row0: vec4<f32>,
    camera_to_ncl_row1: vec4<f32>,
    camera_to_ncl_row2: vec4<f32>,
};

struct DirectYuv12Status {
    flags: atomic<u32>,
    bad_pixel_count: atomic<u32>,
    first_bad_pixel: atomic<u32>,
    reserved: atomic<u32>,
};

@group(0) @binding(0)
var<storage, read> relative_linear_bayer: array<f32>;

@group(0) @binding(1)
var<storage, read_write> direct_yuv12_words: array<u32>;

@group(0) @binding(2)
var<storage, read_write> direct_yuv12_status: DirectYuv12Status;

@group(0) @binding(3)
var<uniform> direct_yuv12_params: DirectYuv12Params;

fn pipe_f32_bayer_width() -> u32 {
    return direct_yuv12_params.dimensions_pattern_plane.x;
}

fn pipe_f32_bayer_height() -> u32 {
    return direct_yuv12_params.dimensions_pattern_plane.y;
}

fn pipe_f32_bayer_pattern_tag() -> u32 {
    return direct_yuv12_params.dimensions_pattern_plane.z;
}

fn direct_yuv12_plane_words() -> u32 {
    return direct_yuv12_params.dimensions_pattern_plane.w;
}

fn load_pipe_f32_bayer_index(index: u32) -> f32 {
    return relative_linear_bayer[index];
}

// __SIGNED_BAYER_DEMOSAIC_WGSL__

// Direct normalized BT.2020-NCL to tightly packed planar TV-range 12-bit codes.
// This terminal deliberately performs no metadata resolution, white balance,
// OETF, tone map, gamut map, pre-NCL clamp, or intermediate RGB storage.

const DIRECT_YUV12_NONFINITE_CAMERA: u32 = 1u;
const DIRECT_YUV12_NONFINITE_NCL: u32 = 2u;
const DIRECT_YUV12_NONFINITE_MAPPED: u32 = 4u;

struct DirectYuv12PixelCodes {
    codes: vec3<u32>,
    failure_bits: u32,
};

struct DirectYuv12CheckedValue {
    value: f32,
    failed: bool,
};

fn direct_yuv12_finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn direct_yuv12_vec3_finite(value: vec3<f32>) -> bool {
    return direct_yuv12_finite(value.x) &&
        direct_yuv12_finite(value.y) &&
        direct_yuv12_finite(value.z);
}

fn direct_yuv12_f32_max() -> f32 {
    return bitcast<f32>(0x7f7fffffu);
}

// Some GPU implementations saturate a finite overflow before isFinite can
// observe it. Check the exact f32 operation's finite range before executing it.
fn direct_yuv12_mul_would_overflow(left: f32, right: f32) -> bool {
    if (!direct_yuv12_finite(left) || !direct_yuv12_finite(right)) {
        return true;
    }
    let abs_left = abs(left);
    let abs_right = abs(right);
    return abs_right != 0.0 && abs_left > direct_yuv12_f32_max() / abs_right;
}

fn direct_yuv12_add_would_overflow(left: f32, right: f32) -> bool {
    if (!direct_yuv12_finite(left) || !direct_yuv12_finite(right)) {
        return true;
    }
    let same_nonzero_sign = (left > 0.0 && right > 0.0) ||
        (left < 0.0 && right < 0.0);
    return same_nonzero_sign &&
        abs(left) > direct_yuv12_f32_max() - abs(right);
}

fn direct_yuv12_checked_row_dot(row: vec3<f32>, camera: vec3<f32>) -> DirectYuv12CheckedValue {
    if (direct_yuv12_mul_would_overflow(row.x, camera.x) ||
        direct_yuv12_mul_would_overflow(row.y, camera.y) ||
        direct_yuv12_mul_would_overflow(row.z, camera.z)) {
        return DirectYuv12CheckedValue(0.0, true);
    }
    let product0 = row.x * camera.x;
    let product1 = row.y * camera.y;
    let product2 = row.z * camera.z;
    if (direct_yuv12_add_would_overflow(product0, product1)) {
        return DirectYuv12CheckedValue(0.0, true);
    }
    let sum01 = product0 + product1;
    if (direct_yuv12_add_would_overflow(sum01, product2)) {
        return DirectYuv12CheckedValue(0.0, true);
    }
    // The products and ordered sums above are overflow preflight only. Keep
    // the adopted safe-path matrix operation as one WGSL row dot.
    let result = dot(row, camera);
    return DirectYuv12CheckedValue(result, !direct_yuv12_finite(result));
}

fn direct_yuv12_checked_affine(offset: f32, scale: f32, value: f32) -> DirectYuv12CheckedValue {
    if (direct_yuv12_mul_would_overflow(scale, value)) {
        return DirectYuv12CheckedValue(0.0, true);
    }
    let product = scale * value;
    if (direct_yuv12_add_would_overflow(offset, product)) {
        return DirectYuv12CheckedValue(0.0, true);
    }
    // This is the adopted WGSL affine expression. A backend may contract the
    // multiply and add; conformance oracles model that contracted result.
    let result = offset + scale * value;
    return DirectYuv12CheckedValue(result, !direct_yuv12_finite(result));
}

fn direct_yuv12_code(mapped: f32) -> u32 {
    return u32(floor(clamp(mapped, 16.0, 4079.0) + 0.5));
}

fn direct_yuv12_codes_valid(codes: vec3<u32>) -> bool {
    return all(codes >= vec3<u32>(16u)) &&
        all(codes <= vec3<u32>(4079u)) &&
        all(codes <= vec3<u32>(4095u));
}

fn direct_yuv12_pixel(x: u32, y: u32) -> DirectYuv12PixelCodes {
    var failure_bits = 0u;
    var camera = demosaic_signed_bayer(x, y);
    if (!direct_yuv12_vec3_finite(camera)) {
        failure_bits |= DIRECT_YUV12_NONFINITE_CAMERA;
        camera = vec3<f32>(0.0);
    }

    let ncl0 = direct_yuv12_checked_row_dot(
        direct_yuv12_params.camera_to_ncl_row0.xyz,
        camera,
    );
    let ncl1 = direct_yuv12_checked_row_dot(
        direct_yuv12_params.camera_to_ncl_row1.xyz,
        camera,
    );
    let ncl2 = direct_yuv12_checked_row_dot(
        direct_yuv12_params.camera_to_ncl_row2.xyz,
        camera,
    );
    var ncl = vec3<f32>(ncl0.value, ncl1.value, ncl2.value);
    if (ncl0.failed || ncl1.failed || ncl2.failed || !direct_yuv12_vec3_finite(ncl)) {
        failure_bits |= DIRECT_YUV12_NONFINITE_NCL;
        ncl = vec3<f32>(0.0);
    }

    let mapped0 = direct_yuv12_checked_affine(256.0, 3504.0, ncl.x);
    let mapped1 = direct_yuv12_checked_affine(2048.0, 3584.0, ncl.y);
    let mapped2 = direct_yuv12_checked_affine(2048.0, 3584.0, ncl.z);
    var mapped = vec3<f32>(mapped0.value, mapped1.value, mapped2.value);
    if (mapped0.failed || mapped1.failed || mapped2.failed ||
        !direct_yuv12_vec3_finite(mapped)) {
        failure_bits |= DIRECT_YUV12_NONFINITE_MAPPED;
        mapped = vec3<f32>(256.0, 2048.0, 2048.0);
    }
    var codes = vec3<u32>(
        direct_yuv12_code(mapped.x),
        direct_yuv12_code(mapped.y),
        direct_yuv12_code(mapped.z),
    );
    if (!direct_yuv12_codes_valid(codes)) {
        failure_bits |= DIRECT_YUV12_NONFINITE_MAPPED;
        codes = vec3<u32>(256u, 2048u, 2048u);
    }
    return DirectYuv12PixelCodes(codes, failure_bits);
}

fn direct_yuv12_record_failure(pixel_index: u32, failure_bits: u32) {
    if (failure_bits == 0u) {
        return;
    }
    atomicOr(&direct_yuv12_status.flags, failure_bits);
    atomicAdd(&direct_yuv12_status.bad_pixel_count, 1u);
    atomicMin(&direct_yuv12_status.first_bad_pixel, pixel_index);
}

fn direct_yuv12_pack_pair(low: u32, high: u32) -> u32 {
    return low | (high << 16u);
}

@compute @workgroup_size(16, 16, 1)
fn direct_yuv12_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let pair_width = pipe_f32_bayer_width() / 2u;
    let lane_has_pair = gid.x < pair_width && gid.y < pipe_f32_bayer_height();
    if (lane_has_pair) {
        let x0 = gid.x * 2u;
        let x1 = x0 + 1u;
        let pixel0 = direct_yuv12_pixel(x0, gid.y);
        let pixel1 = direct_yuv12_pixel(x1, gid.y);
        let linear0 = gid.y * pipe_f32_bayer_width() + x0;
        direct_yuv12_record_failure(linear0, pixel0.failure_bits);
        direct_yuv12_record_failure(linear0 + 1u, pixel1.failure_bits);

        let word = gid.y * pair_width + gid.x;
        let plane_words = direct_yuv12_plane_words();
        direct_yuv12_words[word] = direct_yuv12_pack_pair(pixel0.codes.x, pixel1.codes.x);
        direct_yuv12_words[plane_words + word] =
            direct_yuv12_pack_pair(pixel0.codes.y, pixel1.codes.y);
        direct_yuv12_words[2u * plane_words + word] =
            direct_yuv12_pack_pair(pixel0.codes.z, pixel1.codes.z);
    }
}
