// Future-facing signed Bayer input fragment for the PIPE renderer.
//
// The including shader supplies:
//   pipe_f32_bayer_width() -> u32
//   pipe_f32_bayer_height() -> u32
//   pipe_f32_bayer_pattern_tag() -> u32 (0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG)
//   load_pipe_f32_bayer_index(index: u32) -> f32
//
// This body deliberately performs no black subtraction, correction, clamp,
// white normalization, color transform, or output packing.

fn signed_bayer_clamp_coord(value: i32, upper: u32) -> u32 {
    if (value < 0) {
        return 0u;
    }
    let as_u = u32(value);
    if (as_u >= upper) {
        return upper - 1u;
    }
    return as_u;
}

fn signed_bayer_cfa_position(x: u32, y: u32) -> u32 {
    return ((y & 1u) * 2u) + (x & 1u);
}

// Returns 0=R, 1=G, 2=B for the requested Bayer position.
fn signed_bayer_color_at(x: u32, y: u32) -> u32 {
    let pos = signed_bayer_cfa_position(x, y);
    let pattern = pipe_f32_bayer_pattern_tag();
    if (pattern == 0u) {
        if (pos == 0u) { return 0u; }
        if (pos == 3u) { return 2u; }
        return 1u;
    }
    if (pattern == 1u) {
        if (pos == 0u) { return 2u; }
        if (pos == 3u) { return 0u; }
        return 1u;
    }
    if (pattern == 2u) {
        if (pos == 1u) { return 0u; }
        if (pos == 2u) { return 2u; }
        return 1u;
    }
    if (pos == 1u) { return 2u; }
    if (pos == 2u) { return 0u; }
    return 1u;
}

fn load_pipe_f32_bayer(x: i32, y: i32) -> f32 {
    let cx = signed_bayer_clamp_coord(x, pipe_f32_bayer_width());
    let cy = signed_bayer_clamp_coord(y, pipe_f32_bayer_height());
    return load_pipe_f32_bayer_index(cy * pipe_f32_bayer_width() + cx);
}

fn signed_bayer_avg2(a: f32, b: f32) -> f32 {
    return (a + b) * 0.5;
}

fn signed_bayer_avg4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    return (a + b + c + d) * 0.25;
}

fn demosaic_signed_bayer(x: u32, y: u32) -> vec3<f32> {
    let xi = i32(x);
    let yi = i32(y);
    let center = load_pipe_f32_bayer(xi, yi);
    let color = signed_bayer_color_at(x, y);

    if (color == 0u) {
        let g = signed_bayer_avg4(
            load_pipe_f32_bayer(xi - 1, yi),
            load_pipe_f32_bayer(xi + 1, yi),
            load_pipe_f32_bayer(xi, yi - 1),
            load_pipe_f32_bayer(xi, yi + 1),
        );
        let b = signed_bayer_avg4(
            load_pipe_f32_bayer(xi - 1, yi - 1),
            load_pipe_f32_bayer(xi + 1, yi - 1),
            load_pipe_f32_bayer(xi - 1, yi + 1),
            load_pipe_f32_bayer(xi + 1, yi + 1),
        );
        return vec3<f32>(center, g, b);
    }

    if (color == 2u) {
        let g = signed_bayer_avg4(
            load_pipe_f32_bayer(xi - 1, yi),
            load_pipe_f32_bayer(xi + 1, yi),
            load_pipe_f32_bayer(xi, yi - 1),
            load_pipe_f32_bayer(xi, yi + 1),
        );
        let r = signed_bayer_avg4(
            load_pipe_f32_bayer(xi - 1, yi - 1),
            load_pipe_f32_bayer(xi + 1, yi - 1),
            load_pipe_f32_bayer(xi - 1, yi + 1),
            load_pipe_f32_bayer(xi + 1, yi + 1),
        );
        return vec3<f32>(r, g, center);
    }

    let red_horizontal =
        signed_bayer_color_at(signed_bayer_clamp_coord(xi - 1, pipe_f32_bayer_width()), y) == 0u ||
        signed_bayer_color_at(signed_bayer_clamp_coord(xi + 1, pipe_f32_bayer_width()), y) == 0u;
    if (red_horizontal) {
        return vec3<f32>(
            signed_bayer_avg2(
                load_pipe_f32_bayer(xi - 1, yi),
                load_pipe_f32_bayer(xi + 1, yi),
            ),
            center,
            signed_bayer_avg2(
                load_pipe_f32_bayer(xi, yi - 1),
                load_pipe_f32_bayer(xi, yi + 1),
            ),
        );
    }
    return vec3<f32>(
        signed_bayer_avg2(
            load_pipe_f32_bayer(xi, yi - 1),
            load_pipe_f32_bayer(xi, yi + 1),
        ),
        center,
        signed_bayer_avg2(
            load_pipe_f32_bayer(xi - 1, yi),
            load_pipe_f32_bayer(xi + 1, yi),
        ),
    );
}
