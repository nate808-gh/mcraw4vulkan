// Rust serializes this exact 176-byte layout; the vec4 fields keep each matrix
// row and sink-policy block aligned to 16 bytes.
struct Params {
    frame_width: u32,
    frame_height: u32,
    pixel_count: u32,
    output_word_count: u32,
    format: u32,
    bayer_pattern: u32,
    source_bits: u32,
    tile_first_pixel: u32,
    tile_pixel_count: u32,
    color_mode: u32,
    apply_srgb_transfer: u32,
    _reserved2: u32,
    black0: f32,
    black1: f32,
    black2: f32,
    black3: f32,
    white_level: f32,
    wb_r: f32,
    wb_g: f32,
    wb_b: f32,
    color_row0: vec4<f32>,
    color_row1: vec4<f32>,
    color_row2: vec4<f32>,
    rgb_tone: vec4<f32>,
    rgb_guard: vec4<f32>,
    rgb_desat: vec4<f32>,
};

// Bayer input is raster-order U16 packed two samples per word, low half first.
@group(0) @binding(0)
var<storage, read> input_words: array<u32>;

@group(0) @binding(4)
var<uniform> params: Params;

@group(0) @binding(5)
var preview_output: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(6)
var<storage, read_write> p999_histogram_bins: array<atomic<u32>>;

@group(0) @binding(7)
var<storage, read_write> p999_histogram_stats: array<atomic<u32>>;

const P999_HISTOGRAM_LUMA_MIN: f32 = 0.0;
const P999_HISTOGRAM_LUMA_MAX: f32 = 8.0;
const P999_HISTOGRAM_WORKGROUP_THREADS: u32 = 256u;

fn clamp_coord(value: i32, upper: u32) -> u32 {
    if (value < 0) {
        return 0u;
    }
    let as_u = u32(value);
    if (as_u >= upper) {
        return upper - 1u;
    }
    return as_u;
}

fn cfa_position_index(x: u32, y: u32) -> u32 {
    return ((y & 1u) * 2u) + (x & 1u);
}

fn preview_width() -> u32 {
    return params.output_word_count;
}

fn preview_height() -> u32 {
    return params._reserved2;
}

fn preview_highlight_rolloff_enabled() -> bool {
    return params.tile_first_pixel == 1u;
}

fn preview_sample_limit() -> f32 {
    return max(f32(params.tile_pixel_count), params.white_level);
}

fn black_for_position(position: u32) -> f32 {
    if (position == 0u) {
        return params.black0;
    }
    if (position == 1u) {
        return params.black1;
    }
    if (position == 2u) {
        return params.black2;
    }
    return params.black3;
}

// Returns 0=R, 1=G, 2=B for the requested Bayer position.
fn color_at(x: u32, y: u32) -> u32 {
    let pos = cfa_position_index(x, y);
    if (params.bayer_pattern == 0u) {
        // RGGB
        if (pos == 0u) { return 0u; }
        if (pos == 3u) { return 2u; }
        return 1u;
    }
    if (params.bayer_pattern == 1u) {
        // BGGR
        if (pos == 0u) { return 2u; }
        if (pos == 3u) { return 0u; }
        return 1u;
    }
    if (params.bayer_pattern == 2u) {
        // GRBG
        if (pos == 1u) { return 0u; }
        if (pos == 2u) { return 2u; }
        return 1u;
    }
    // GBRG
    if (pos == 1u) { return 2u; }
    if (pos == 2u) { return 0u; }
    return 1u;
}

fn raw_sample_with_domain(x: i32, y: i32, allow_overwhite: bool, sample_limit: f32) -> f32 {
    let cx = clamp_coord(x, params.frame_width);
    let cy = clamp_coord(y, params.frame_height);
    let pixel_index = cy * params.frame_width + cx;
    let word = input_words[pixel_index / 2u];
    var sample = word & 0xffffu;
    if ((pixel_index & 1u) == 1u) {
        sample = (word >> 16u) & 0xffffu;
    }
    let black = black_for_position(cfa_position_index(cx, cy));
    let signal = max(f32(sample) - black, 0.0);
    let denom = max(params.white_level - black, 1.0);
    let normalized = max(signal / denom, 0.0);
    if (!allow_overwhite) {
        return min(normalized, 1.0);
    }
    let max_signal = max(sample_limit - black, denom);
    return min(normalized, max_signal / denom);
}

fn avg2(a: f32, b: f32) -> f32 {
    return (a + b) * 0.5;
}

fn avg4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    return (a + b + c + d) * 0.25;
}

fn demosaic_rgb_with_domain(x: u32, y: u32, allow_overwhite: bool, sample_limit: f32) -> vec3<f32> {
    let xi = i32(x);
    let yi = i32(y);
    let center = raw_sample_with_domain(xi, yi, allow_overwhite, sample_limit);
    let color = color_at(x, y);

    if (color == 0u) {
        let g = avg4(
            raw_sample_with_domain(xi - 1, yi, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi + 1, yi, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi, yi - 1, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi, yi + 1, allow_overwhite, sample_limit),
        );
        let b = avg4(
            raw_sample_with_domain(xi - 1, yi - 1, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi + 1, yi - 1, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi - 1, yi + 1, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi + 1, yi + 1, allow_overwhite, sample_limit),
        );
        return vec3<f32>(center, g, b);
    }

    if (color == 2u) {
        let g = avg4(
            raw_sample_with_domain(xi - 1, yi, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi + 1, yi, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi, yi - 1, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi, yi + 1, allow_overwhite, sample_limit),
        );
        let r = avg4(
            raw_sample_with_domain(xi - 1, yi - 1, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi + 1, yi - 1, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi - 1, yi + 1, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi + 1, yi + 1, allow_overwhite, sample_limit),
        );
        return vec3<f32>(r, g, center);
    }

    let red_horizontal = color_at(clamp_coord(xi - 1, params.frame_width), y) == 0u ||
        color_at(clamp_coord(xi + 1, params.frame_width), y) == 0u;
    var r = 0.0;
    var b = 0.0;
    if (red_horizontal) {
        r = avg2(
            raw_sample_with_domain(xi - 1, yi, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi + 1, yi, allow_overwhite, sample_limit),
        );
        b = avg2(
            raw_sample_with_domain(xi, yi - 1, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi, yi + 1, allow_overwhite, sample_limit),
        );
    } else {
        r = avg2(
            raw_sample_with_domain(xi, yi - 1, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi, yi + 1, allow_overwhite, sample_limit),
        );
        b = avg2(
            raw_sample_with_domain(xi - 1, yi, allow_overwhite, sample_limit),
            raw_sample_with_domain(xi + 1, yi, allow_overwhite, sample_limit),
        );
    }
    return vec3<f32>(r, center, b);
}

fn demosaic_rgb(x: u32, y: u32) -> vec3<f32> {
    return demosaic_rgb_with_domain(x, y, false, params.white_level);
}

fn preview_uses_downscale(out_width: u32, out_height: u32) -> bool {
    return out_width < params.frame_width || out_height < params.frame_height;
}

fn bayer_cell_origin(coord: u32, upper: u32) -> u32 {
    if (upper <= 1u) {
        return 0u;
    }
    var base = coord;
    if (base + 1u >= upper) {
        base = upper - 2u;
    }
    return base & 0xfffffffeu;
}

fn bayer_sample_rgb_contribution(x: u32, y: u32, sample: f32) -> vec3<f32> {
    let color = color_at(x, y);
    if (color == 0u) {
        return vec3<f32>(sample, 0.0, 0.0);
    }
    if (color == 2u) {
        return vec3<f32>(0.0, 0.0, sample);
    }
    return vec3<f32>(0.0, sample, 0.0);
}

fn bayer_sample_rgb_count(x: u32, y: u32) -> vec3<f32> {
    let color = color_at(x, y);
    if (color == 0u) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    if (color == 2u) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

fn bayer_cell_rgb_with_domain(x: u32, y: u32, allow_overwhite: bool, sample_limit: f32) -> vec3<f32> {
    let x0 = bayer_cell_origin(x, params.frame_width);
    let y0 = bayer_cell_origin(y, params.frame_height);
    let x1 = min(x0 + 1u, params.frame_width - 1u);
    let y1 = min(y0 + 1u, params.frame_height - 1u);

    let s00 = raw_sample_with_domain(i32(x0), i32(y0), allow_overwhite, sample_limit);
    let s10 = raw_sample_with_domain(i32(x1), i32(y0), allow_overwhite, sample_limit);
    let s01 = raw_sample_with_domain(i32(x0), i32(y1), allow_overwhite, sample_limit);
    let s11 = raw_sample_with_domain(i32(x1), i32(y1), allow_overwhite, sample_limit);

    let sum =
        bayer_sample_rgb_contribution(x0, y0, s00) +
        bayer_sample_rgb_contribution(x1, y0, s10) +
        bayer_sample_rgb_contribution(x0, y1, s01) +
        bayer_sample_rgb_contribution(x1, y1, s11);
    let counts =
        bayer_sample_rgb_count(x0, y0) +
        bayer_sample_rgb_count(x1, y0) +
        bayer_sample_rgb_count(x0, y1) +
        bayer_sample_rgb_count(x1, y1);
    return sum / max(counts, vec3<f32>(1.0));
}

fn white_balanced_rgb(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(rgb.x * params.wb_r, rgb.y * params.wb_g, rgb.z * params.wb_b);
}

fn camera_to_linear_srgb(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(params.color_row0.xyz, rgb),
        dot(params.color_row1.xyz, rgb),
        dot(params.color_row2.xyz, rgb),
    );
}

fn srgb_oetf_one(linear: f32) -> f32 {
    let clamped = clamp(linear, 0.0, 1.0);
    if (clamped <= 0.0031308) {
        return 12.92 * clamped;
    }
    return 1.055 * pow(clamped, 1.0 / 2.4) - 0.055;
}

fn srgb_oetf(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_oetf_one(rgb.x),
        srgb_oetf_one(rgb.y),
        srgb_oetf_one(rgb.z),
    );
}

fn hue_preserving_highlight_rolloff(rgb: vec3<f32>) -> vec3<f32> {
    let non_negative = max(rgb, vec3<f32>(0.0));
    let max_channel = max(max(non_negative.x, non_negative.y), non_negative.z);
    if (max_channel <= 1.0) {
        return non_negative;
    }
    return non_negative / max_channel;
}

fn render_rgb_with_highlight_policy(rgb: vec3<f32>, highlight_rolloff: bool) -> vec3<f32> {
    let balanced = white_balanced_rgb(rgb);

    let transformed = camera_to_linear_srgb(balanced);
    var linear_srgb = clamp(transformed, vec3<f32>(0.0), vec3<f32>(1.0));
    if (highlight_rolloff) {
        linear_srgb = hue_preserving_highlight_rolloff(transformed);
    }
    if (params.apply_srgb_transfer == 0u) {
        return linear_srgb;
    }
    return srgb_oetf(linear_srgb);
}

fn preview_rgb_sink_policy_enabled() -> bool {
    return abs(rgb_sink_tone_scale() - 1.0) > 0.000001 ||
        rgb_sink_guard_mode() != 0u ||
        rgb_sink_highlight_desat_enabled();
}

fn render_preview_rgb(rgb: vec3<f32>, highlight_rolloff: bool) -> vec3<f32> {
    if (!preview_rgb_sink_policy_enabled()) {
        return render_rgb_with_highlight_policy(rgb, highlight_rolloff);
    }

    if (params.color_mode == 2u) {
        return clamp(rgb * rgb_sink_tone_scale(), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    let balanced = white_balanced_rgb(rgb);

    if (params.color_mode == 0u) {
        return clamp(balanced * rgb_sink_tone_scale(), vec3<f32>(0.0), vec3<f32>(1.0));
    }

    let transformed = camera_to_linear_srgb(balanced);
    let guarded = apply_rgb_sink_guard(transformed * rgb_sink_tone_scale());
    let desatted = apply_rgb_sink_highlight_desat(guarded);
    let linear_srgb = clamp(desatted, vec3<f32>(0.0), vec3<f32>(1.0));
    if (params.apply_srgb_transfer == 0u) {
        return linear_srgb;
    }
    return srgb_oetf(linear_srgb);
}

fn display_sample_limit() -> f32 {
    return max(params.color_row0.w, params.white_level);
}

fn rgb_sink_tone_scale() -> f32 {
    return max(params.rgb_tone.x, 0.0);
}

fn rgb_sink_guard_mode() -> u32 {
    return u32(max(params.rgb_guard.x, 0.0) + 0.5);
}

fn rgb_sink_guard_luma() -> f32 {
    return max(params.rgb_guard.y, 0.0);
}

fn rgb_sink_guard_luma_for_rgb(rgb: vec3<f32>) -> f32 {
    let non_negative = max(rgb, vec3<f32>(0.0));
    return 0.2126 * non_negative.x + 0.7152 * non_negative.y + 0.0722 * non_negative.z;
}

fn apply_rgb_sink_guard(rgb: vec3<f32>) -> vec3<f32> {
    let mode = rgb_sink_guard_mode();
    if (mode == 0u) {
        return rgb;
    }

    let luma = rgb_sink_guard_luma_for_rgb(rgb);
    if (luma < rgb_sink_guard_luma()) {
        return rgb;
    }

    var guarded = max(rgb, vec3<f32>(0.0));
    if (mode == 1u) {
        let rb_sum = guarded.x + guarded.z;
        let limit = 2.0 * guarded.y;
        if (rb_sum > limit && rb_sum > 0.000001) {
            let scale = limit / rb_sum;
            guarded.x = guarded.x * scale;
            guarded.z = guarded.z * scale;
        }
        return guarded;
    }

    return rgb;
}

fn rgb_sink_highlight_desat_enabled() -> bool {
    return params.rgb_desat.x >= 0.5;
}

fn rgb_sink_highlight_desat_start() -> f32 {
    return max(params.rgb_desat.y, 0.0);
}

fn rgb_sink_highlight_desat_end() -> f32 {
    return max(params.rgb_desat.z, rgb_sink_highlight_desat_start());
}

fn rgb_sink_highlight_desat_strength() -> f32 {
    return clamp(params.rgb_desat.w, 0.0, 1.0);
}

fn apply_rgb_sink_highlight_desat(rgb: vec3<f32>) -> vec3<f32> {
    if (!rgb_sink_highlight_desat_enabled()) {
        return rgb;
    }
    let non_negative = max(rgb, vec3<f32>(0.0));
    let luma = rgb_sink_guard_luma_for_rgb(non_negative);
    let start = rgb_sink_highlight_desat_start();
    let end = max(rgb_sink_highlight_desat_end(), start + 0.000001);
    let t = smoothstep(start, end, luma) * rgb_sink_highlight_desat_strength();
    let gray = vec3<f32>(luma);
    return mix(non_negative, gray, t);
}

fn display_p999_linear_rgb(rgb: vec3<f32>) -> vec3<f32> {
    let balanced = white_balanced_rgb(rgb);

    return camera_to_linear_srgb(balanced);
}

@compute @workgroup_size(256, 1, 1)
fn p999_histogram_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let dispatch_groups_x = max(params.tile_first_pixel, 1u);
    let pixel = global_invocation_id.x +
        (global_invocation_id.y * dispatch_groups_x * P999_HISTOGRAM_WORKGROUP_THREADS);
    if (pixel >= params.pixel_count) {
        return;
    }

    let bin_count = arrayLength(&p999_histogram_bins);
    if (bin_count == 0u) {
        return;
    }

    let y = pixel / params.frame_width;
    let x = pixel - (y * params.frame_width);
    let sample_limit = display_sample_limit();
    let camera_rgb = demosaic_rgb_with_domain(x, y, true, sample_limit);
    let linear_rgb = display_p999_linear_rgb(camera_rgb);
    let luma = rgb_sink_guard_luma_for_rgb(linear_rgb);
    let luma_range = max(P999_HISTOGRAM_LUMA_MAX - P999_HISTOGRAM_LUMA_MIN, 0.000001);

    var bin_index = 0u;
    if (luma >= P999_HISTOGRAM_LUMA_MAX) {
        bin_index = bin_count - 1u;
        atomicAdd(&p999_histogram_stats[0], 1u);
    } else {
        let normalized = clamp((luma - P999_HISTOGRAM_LUMA_MIN) / luma_range, 0.0, 0.99999994);
        bin_index = min(u32(floor(normalized * f32(bin_count))), bin_count - 1u);
    }

    atomicAdd(&p999_histogram_bins[bin_index], 1u);
}

fn scaled_source_coord(out_coord: u32, out_size: u32, source_size: u32) -> u32 {
    let source = floor((f32(out_coord) + 0.5) * f32(source_size) / f32(out_size));
    let source_u = u32(source);
    if (source_u >= source_size) {
        return source_size - 1u;
    }
    return source_u;
}

@compute @workgroup_size(16, 16, 1)
fn preview_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let out_width = preview_width();
    let out_height = preview_height();
    let out_x = global_invocation_id.x;
    let out_y = global_invocation_id.y;
    if (out_x >= out_width || out_y >= out_height) {
        return;
    }

    let src_x = scaled_source_coord(out_x, out_width, params.frame_width);
    let src_y = scaled_source_coord(out_y, out_height, params.frame_height);
    let highlight_rolloff = preview_highlight_rolloff_enabled();
    let sample_limit = preview_sample_limit();
    var camera_rgb = demosaic_rgb_with_domain(src_x, src_y, highlight_rolloff, sample_limit);
    if (highlight_rolloff && preview_uses_downscale(out_width, out_height)) {
        camera_rgb = bayer_cell_rgb_with_domain(src_x, src_y, true, sample_limit);
    }
    let rgb = render_preview_rgb(camera_rgb, highlight_rolloff);

    textureStore(
        preview_output,
        vec2<i32>(i32(out_x), i32(out_y)),
        vec4<f32>(rgb, 1.0),
    );
}
