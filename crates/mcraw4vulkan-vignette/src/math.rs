use mcraw4vulkan_core::{BayerPattern, FrameDimensions};

use crate::{
    BAYER_CFA_PLANE_COUNT, LensShadingMap, PreparedFixedLensShadingMap, PreparedLensShadingMap,
    VIGNETTE_GAIN_SCALE, VignetteCoordinateMapping, VignetteCorrectionError,
};

pub fn cfa_position_plane_index(bayer_pattern: BayerPattern, x: usize, y: usize) -> usize {
    match bayer_pattern {
        // Lens-map planes are keyed by 2x2 CFA position rather than RGB label.
        // Keeping BayerPattern explicit prevents an accidental color remapping.
        BayerPattern::Rggb | BayerPattern::Bggr | BayerPattern::Grbg | BayerPattern::Gbrg => {
            ((y & 1) * 2) + (x & 1)
        }
    }
}

pub fn android_rggb_source_plane_index(bayer_pattern: BayerPattern, x: usize, y: usize) -> usize {
    match bayer_site_label(bayer_pattern, x, y) {
        "R" => 0,
        "B" => 3,
        _ if y & 1 == 0 => 1,
        _ => 2,
    }
}

pub fn bayer_site_label(pattern: BayerPattern, x: usize, y: usize) -> &'static str {
    match (pattern, x & 1, y & 1) {
        (BayerPattern::Rggb, 0, 0) => "R",
        (BayerPattern::Rggb, 1, 0) => "G_r",
        (BayerPattern::Rggb, 0, 1) => "G_b",
        (BayerPattern::Rggb, 1, 1) => "B",
        (BayerPattern::Grbg, 0, 0) => "G_r",
        (BayerPattern::Grbg, 1, 0) => "R",
        (BayerPattern::Grbg, 0, 1) => "B",
        (BayerPattern::Grbg, 1, 1) => "G_b",
        (BayerPattern::Gbrg, 0, 0) => "G_b",
        (BayerPattern::Gbrg, 1, 0) => "B",
        (BayerPattern::Gbrg, 0, 1) => "R",
        (BayerPattern::Gbrg, 1, 1) => "G_r",
        (BayerPattern::Bggr, 0, 0) => "B",
        (BayerPattern::Bggr, 1, 0) => "G_b",
        (BayerPattern::Bggr, 0, 1) => "G_r",
        (BayerPattern::Bggr, 1, 1) => "R",
        _ => "unknown",
    }
}

pub fn interpolated_gain(
    lens_shading_map: &LensShadingMap,
    plane_index: usize,
    x: usize,
    y: usize,
    dimensions: FrameDimensions,
    coordinate_mapping: VignetteCoordinateMapping,
) -> Result<f32, VignetteCorrectionError> {
    let prepared_map = PreparedLensShadingMap::from_typed_map(lens_shading_map)?;
    interpolated_prepared_gain(
        &prepared_map,
        plane_index,
        x,
        y,
        dimensions,
        coordinate_mapping,
    )
}

pub fn interpolated_prepared_gain(
    lens_shading_map: &PreparedLensShadingMap<'_>,
    plane_index: usize,
    x: usize,
    y: usize,
    dimensions: FrameDimensions,
    coordinate_mapping: VignetteCoordinateMapping,
) -> Result<f32, VignetteCorrectionError> {
    let plane = lens_shading_map.plane(plane_index)?;
    let (u, v) = normalized_coordinates(x, y, dimensions, coordinate_mapping);
    let map_width = lens_shading_map.width();
    let map_height = lens_shading_map.height();

    Ok(bilinear_sample(plane, map_width, map_height, u, v))
}

pub fn interpolated_fixed_gain(
    lens_shading_map: &PreparedFixedLensShadingMap<'_>,
    plane_index: usize,
    x: usize,
    y: usize,
    dimensions: FrameDimensions,
    coordinate_mapping: VignetteCoordinateMapping,
) -> Result<i64, VignetteCorrectionError> {
    let plane = lens_shading_map.plane_q(plane_index)?;
    let (x0, x1, tx_q) = fixed_axis_position(
        x,
        dimensions.width,
        lens_shading_map.width(),
        coordinate_mapping,
    )?;
    let (y0, y1, ty_q) = fixed_axis_position(
        y,
        dimensions.height,
        lens_shading_map.height(),
        coordinate_mapping,
    )?;
    let width = lens_shading_map.width();

    let top_left = plane[y0 * width + x0];
    let top_right = plane[y0 * width + x1];
    let bottom_left = plane[y1 * width + x0];
    let bottom_right = plane[y1 * width + x1];
    let top = lerp_fixed_q(top_left, top_right, tx_q)?;
    let bottom = lerp_fixed_q(bottom_left, bottom_right, tx_q)?;

    lerp_fixed_q(top, bottom, ty_q)
}

pub(crate) fn corrected_sample(
    raw_sample: u16,
    black_level: f32,
    gain: f32,
    output_white_level: u16,
) -> u16 {
    // Subtract the selected position's black level before gain, clamp at zero,
    // then round once and clamp to output white; reordering changes U16 results.
    let corrected_signal = (f32::from(raw_sample) - black_level).max(0.0) * gain;

    clamp_and_round_sample(corrected_signal, output_white_level)
}

pub(crate) fn corrected_fixed_sample(
    raw_sample: u16,
    black_level_q: i64,
    gain_q: i64,
    output_white_level: u16,
) -> Result<u16, VignetteCorrectionError> {
    // Signal and gain are Q16.16. Their Q32.32 product is rounded half-up once,
    // matching the integer sequence implemented by the WGSL correction path.
    let raw_q = i64::from(raw_sample)
        .checked_mul(VIGNETTE_GAIN_SCALE)
        .ok_or(VignetteCorrectionError::FixedPointCorrectionOverflow)?;
    let signal_q = raw_q.saturating_sub(black_level_q).max(0);
    let corrected_q2 = signal_q
        .checked_mul(gain_q)
        .ok_or(VignetteCorrectionError::FixedPointCorrectionOverflow)?;
    let scale_squared = VIGNETTE_GAIN_SCALE
        .checked_mul(VIGNETTE_GAIN_SCALE)
        .ok_or(VignetteCorrectionError::FixedPointCorrectionOverflow)?;
    let corrected = div_round_half_up(corrected_q2, scale_squared)?;

    Ok(corrected.clamp(0, i64::from(output_white_level)) as u16)
}

pub(crate) fn validate_black_level(
    black_level: [f32; BAYER_CFA_PLANE_COUNT],
) -> Result<(), VignetteCorrectionError> {
    for (index, value) in black_level.into_iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(VignetteCorrectionError::InvalidBlackLevel { index, value });
        }
    }

    Ok(())
}

fn fixed_axis_position(
    position: usize,
    frame_dimension: u32,
    map_dimension: usize,
    coordinate_mapping: VignetteCoordinateMapping,
) -> Result<(usize, usize, i64), VignetteCorrectionError> {
    match coordinate_mapping {
        VignetteCoordinateMapping::VisibleFrame => {}
    }

    if map_dimension <= 1 || frame_dimension <= 1 {
        return Ok((0, 0, 0));
    }

    let frame_last = usize::try_from(frame_dimension - 1).map_err(|_| {
        VignetteCorrectionError::FixedPointCoordinateOverflow {
            position,
            frame_dimension,
            map_dimension,
        }
    })?;
    let map_last = map_dimension - 1;
    let numerator = position.checked_mul(map_last).ok_or(
        VignetteCorrectionError::FixedPointCoordinateOverflow {
            position,
            frame_dimension,
            map_dimension,
        },
    )?;
    let x0 = numerator / frame_last;
    let remainder = numerator % frame_last;
    let x1 = (x0 + 1).min(map_last);
    let remainder_scaled = i64::try_from(remainder)
        .ok()
        .and_then(|value| value.checked_mul(VIGNETTE_GAIN_SCALE))
        .ok_or(VignetteCorrectionError::FixedPointCoordinateOverflow {
            position,
            frame_dimension,
            map_dimension,
        })?;
    let fraction_q = div_round_half_up(
        remainder_scaled,
        i64::try_from(frame_last).map_err(|_| {
            VignetteCorrectionError::FixedPointCoordinateOverflow {
                position,
                frame_dimension,
                map_dimension,
            }
        })?,
    )?;

    Ok((x0.min(map_last), x1, fraction_q))
}

fn lerp_fixed_q(a: i64, b: i64, t_q: i64) -> Result<i64, VignetteCorrectionError> {
    let inverse_t_q = VIGNETTE_GAIN_SCALE
        .checked_sub(t_q)
        .ok_or(VignetteCorrectionError::FixedPointCorrectionOverflow)?;
    let weighted_a = a
        .checked_mul(inverse_t_q)
        .ok_or(VignetteCorrectionError::FixedPointCorrectionOverflow)?;
    let weighted_b = b
        .checked_mul(t_q)
        .ok_or(VignetteCorrectionError::FixedPointCorrectionOverflow)?;
    let weighted = weighted_a
        .checked_add(weighted_b)
        .ok_or(VignetteCorrectionError::FixedPointCorrectionOverflow)?;

    div_round_half_up(weighted, VIGNETTE_GAIN_SCALE)
}

fn div_round_half_up(numerator: i64, denominator: i64) -> Result<i64, VignetteCorrectionError> {
    if denominator <= 0 || numerator < 0 {
        return Err(VignetteCorrectionError::FixedPointCorrectionOverflow);
    }

    numerator
        .checked_add(denominator / 2)
        .map(|rounded| rounded / denominator)
        .ok_or(VignetteCorrectionError::FixedPointCorrectionOverflow)
}

fn normalized_coordinates(
    x: usize,
    y: usize,
    dimensions: FrameDimensions,
    coordinate_mapping: VignetteCoordinateMapping,
) -> (f32, f32) {
    match coordinate_mapping {
        VignetteCoordinateMapping::VisibleFrame => {
            let u = normalized_axis_position(x, dimensions.width);
            let v = normalized_axis_position(y, dimensions.height);
            (u, v)
        }
    }
}

fn normalized_axis_position(position: usize, dimension: u32) -> f32 {
    if dimension <= 1 {
        return 0.0;
    }

    let last = (dimension - 1) as f32;
    (position as f32 / last).clamp(0.0, 1.0)
}

fn bilinear_sample(plane: &[f32], width: usize, height: usize, u: f32, v: f32) -> f32 {
    if width == 1 && height == 1 {
        return plane[0];
    }

    let map_x = u * (width.saturating_sub(1) as f32);
    let map_y = v * (height.saturating_sub(1) as f32);
    let x0 = map_x.floor() as usize;
    let y0 = map_y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = map_x - x0 as f32;
    let ty = map_y - y0 as f32;

    let top_left = plane[y0 * width + x0];
    let top_right = plane[y0 * width + x1];
    let bottom_left = plane[y1 * width + x0];
    let bottom_right = plane[y1 * width + x1];
    let top = lerp(top_left, top_right, tx);
    let bottom = lerp(bottom_left, bottom_right, tx);

    lerp(top, bottom, ty)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn clamp_and_round_sample(sample: f32, output_white_level: u16) -> u16 {
    sample
        .round()
        .clamp(0.0, f32::from(output_white_level).min(f32::from(u16::MAX))) as u16
}
