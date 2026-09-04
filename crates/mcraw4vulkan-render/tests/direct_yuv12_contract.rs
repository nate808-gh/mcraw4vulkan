//! Independent numeric, byte-layout, geometry, and GPU contract tests for the
//! production direct-YUV12 stage. The CPU oracle intentionally calls no render
//! production helper.

#[allow(clippy::excessive_precision)]
const BT2020_NCL: [f64; 9] = [
    0.2627,
    0.6780,
    0.0593,
    -0.13963006271925163,
    -0.36036993728074834,
    0.5,
    0.5,
    -0.45978570459785706,
    -0.040214295402142962,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceNonfinite {
    Camera,
    Matrix,
    Ncl,
    Mapped,
}

#[test]
fn independent_f64_nominal_and_excursion_vectors_are_exact() {
    let cases = [
        ("black", [0.0, 0.0, 0.0], [256, 2048, 2048]),
        ("white", [1.0, 1.0, 1.0], [3760, 2048, 2048]),
        ("red", [1.0, 0.0, 0.0], [1177, 1548, 3840]),
        ("green", [0.0, 1.0, 0.0], [2632, 756, 400]),
        ("blue", [0.0, 0.0, 1.0], [464, 3840, 1904]),
    ];
    for (name, rgb, expected) in cases {
        assert_eq!(reference_codes(rgb, BT2020_NCL), Ok(expected), "{name}");
    }
    assert_eq!(
        reference_codes([-0.125; 3], BT2020_NCL),
        Ok([16, 2048, 2048])
    );
    assert_eq!(
        reference_codes([1.25; 3], BT2020_NCL),
        Ok([4079, 2048, 2048])
    );
}

#[test]
fn independent_f64_proves_matrix_first_without_camera_component_clamp() {
    let transform = [0.20, 0.30, 0.10, 0.10, -0.20, 0.05, -0.10, 0.20, 0.10];
    for camera in [[-0.25, 0.50, 0.50], [1.25, 0.25, 0.50]] {
        let matrix_first = reference_codes(camera, transform).unwrap();
        let preclamped =
            reference_codes(camera.map(|value| value.clamp(0.0, 1.0)), transform).unwrap();
        assert_ne!(matrix_first, preclamped, "camera={camera:?}");
        assert!(
            matrix_first
                .into_iter()
                .all(|code| (16..=4079).contains(&code))
        );
    }
}

#[test]
fn independent_f64_exhausts_listed_boundaries_and_half_up_cases_per_plane() {
    const LISTED: [f64; 16] = [
        15.0, 16.0, 17.0, 255.0, 256.0, 257.0, 3759.0, 3760.0, 3761.0, 3839.0, 3840.0, 3841.0,
        4078.0, 4079.0, 4080.0, 4095.0,
    ];
    for plane in 0..3 {
        for mapped in LISTED {
            for value in [next_down(mapped), mapped, next_up(mapped)] {
                assert_eq!(
                    reference_code_from_mapped(value),
                    expected_code_from_mapped(value),
                    "plane={plane}, mapped={mapped:.17}, value={value:.17}"
                );
            }
            let normalized = normalized_for_mapped(plane, mapped);
            let remapped = map_normalized(plane, normalized);
            assert_eq!(
                reference_code_from_mapped(remapped),
                expected_code_from_mapped(remapped),
                "plane={plane}, normalized boundary={mapped}"
            );
        }
    }

    for (mapped, expected) in [(256.5, 257), (257.5, 258), (2048.5, 2049), (4078.5, 4079)] {
        assert_eq!(reference_code_from_mapped(mapped), Ok(expected));
    }
    assert_eq!(reference_code_from_mapped(15.0), Ok(16));
    assert_eq!(reference_code_from_mapped(4080.0), Ok(4079));
}

#[test]
#[allow(clippy::needless_range_loop)]
fn independent_layout_is_exact_planar_low12_little_endian_and_row_paired() {
    let width = 4_usize;
    let height = 3_usize;
    let pixels = width * height;
    let mut codes = Vec::with_capacity(pixels);
    for index in 0..pixels {
        codes.push([16 + index as u16, 1000 + index as u16, 3000 + index as u16]);
    }
    let bytes = pack_reference_planar(&codes);
    assert_eq!(bytes.len(), 6 * pixels);
    let plane_bytes = 2 * pixels;
    for plane in 0..3 {
        for index in 0..pixels {
            let offset = plane * plane_bytes + index * 2;
            assert_eq!(
                &bytes[offset..offset + 2],
                &codes[index][plane].to_le_bytes(),
                "plane={plane}, pixel={index}"
            );
            let word = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
            assert_eq!(word & 0xf000, 0);
        }
    }
    for plane in 0..3 {
        for row in 0..height {
            for pair in 0..width / 2 {
                let sample = row * width + pair * 2;
                let offset = plane * plane_bytes + sample * 2;
                let word = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                assert_eq!(word & 0xffff, u32::from(codes[sample][plane]));
                assert_eq!(word >> 16, u32::from(codes[sample + 1][plane]));
                assert_eq!(sample / width, (sample + 1) / width, "pair crossed row");
            }
        }
    }
}

#[test]
fn independent_geometry_covers_required_sizes_and_failure_classes() {
    for (width, height) in [
        (2, 1),
        (2, 2),
        (4, 2),
        (4, 3),
        (256, 3),
        (258, 3),
        (1920, 1080),
        (3840, 2160),
        (4080, 3072),
    ] {
        let layout = checked_reference_geometry(width, height, u32::MAX).unwrap();
        assert_eq!(layout.pixel_count, u64::from(width) * u64::from(height));
        assert_eq!(layout.plane_bytes, 2 * layout.pixel_count);
        assert_eq!(layout.visible_bytes, 6 * layout.pixel_count);
        assert_eq!(layout.pair_words, layout.pixel_count / 2);
    }
    assert_eq!(
        checked_reference_geometry(0, 1, u32::MAX),
        Err(ReferenceGeometryError::ZeroDimension)
    );
    assert_eq!(
        checked_reference_geometry(2, 0, u32::MAX),
        Err(ReferenceGeometryError::ZeroDimension)
    );
    assert_eq!(
        checked_reference_geometry(1, 1, u32::MAX),
        Err(ReferenceGeometryError::OddWidth)
    );
    assert_eq!(
        checked_reference_geometry(3, 2, u32::MAX),
        Err(ReferenceGeometryError::OddWidth)
    );
    assert_eq!(
        checked_reference_geometry(u32::MAX - 1, 2, u32::MAX),
        Err(ReferenceGeometryError::PixelIndexOverflow)
    );
    assert_eq!(
        checked_reference_geometry(2, u32::MAX, u32::MAX),
        Err(ReferenceGeometryError::PixelIndexOverflow)
    );
    assert_eq!(
        checked_visible_bytes(u64::MAX),
        Err(ReferenceGeometryError::ByteCountOverflow)
    );
    assert_eq!(
        checked_reference_geometry(258, 3, 8),
        Err(ReferenceGeometryError::DispatchLimit)
    );
}

#[test]
fn independent_nonfinite_reference_rejects_before_integer_conversion() {
    assert_eq!(
        reference_codes([f64::NAN, 0.0, 0.0], BT2020_NCL),
        Err(ReferenceNonfinite::Camera)
    );
    assert_eq!(
        reference_codes([f64::INFINITY, 0.0, 0.0], BT2020_NCL),
        Err(ReferenceNonfinite::Camera)
    );
    let mut matrix = BT2020_NCL;
    matrix[4] = f64::NEG_INFINITY;
    assert_eq!(
        reference_codes([1.0; 3], matrix),
        Err(ReferenceNonfinite::Matrix)
    );
    assert_eq!(
        reference_codes([f64::MAX, f64::MAX, f64::MAX], [1.0; 9]),
        Err(ReferenceNonfinite::Ncl)
    );
    assert_eq!(
        reference_code_from_mapped(f64::NAN),
        Err(ReferenceNonfinite::Mapped)
    );
    assert_eq!(
        reference_code_from_mapped(f64::INFINITY),
        Err(ReferenceNonfinite::Mapped)
    );
    assert_eq!(
        reference_code_from_mapped(f64::NEG_INFINITY),
        Err(ReferenceNonfinite::Mapped)
    );
}

fn reference_codes(camera: [f64; 3], matrix: [f64; 9]) -> Result<[u16; 3], ReferenceNonfinite> {
    if camera.iter().any(|value| !value.is_finite()) {
        return Err(ReferenceNonfinite::Camera);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(ReferenceNonfinite::Matrix);
    }
    let mut ncl = [0.0_f64; 3];
    for row in 0..3 {
        ncl[row] = matrix[row * 3] * camera[0]
            + matrix[row * 3 + 1] * camera[1]
            + matrix[row * 3 + 2] * camera[2];
    }
    if ncl.iter().any(|value| !value.is_finite()) {
        return Err(ReferenceNonfinite::Ncl);
    }
    let mapped = [
        256.0 + 3504.0 * ncl[0],
        2048.0 + 3584.0 * ncl[1],
        2048.0 + 3584.0 * ncl[2],
    ];
    Ok([
        reference_code_from_mapped(mapped[0])?,
        reference_code_from_mapped(mapped[1])?,
        reference_code_from_mapped(mapped[2])?,
    ])
}

fn reference_code_from_mapped(mapped: f64) -> Result<u16, ReferenceNonfinite> {
    if !mapped.is_finite() {
        return Err(ReferenceNonfinite::Mapped);
    }
    let code = (mapped.clamp(16.0, 4079.0) + 0.5).floor();
    let code = code as u32;
    assert!((16..=4079).contains(&code));
    assert!(code <= 4095);
    assert_eq!(code & 0xffff_f000, 0);
    Ok(code as u16)
}

#[allow(clippy::manual_clamp)]
fn expected_code_from_mapped(mapped: f64) -> Result<u16, ReferenceNonfinite> {
    if !mapped.is_finite() {
        return Err(ReferenceNonfinite::Mapped);
    }
    Ok(((mapped.max(16.0).min(4079.0) + 0.5).floor() as u32) as u16)
}

fn normalized_for_mapped(plane: usize, mapped: f64) -> f64 {
    match plane {
        0 => (mapped - 256.0) / 3504.0,
        1 | 2 => (mapped - 2048.0) / 3584.0,
        _ => unreachable!(),
    }
}

fn map_normalized(plane: usize, normalized: f64) -> f64 {
    match plane {
        0 => 256.0 + 3504.0 * normalized,
        1 | 2 => 2048.0 + 3584.0 * normalized,
        _ => unreachable!(),
    }
}

fn next_up(value: f64) -> f64 {
    f64::from_bits(value.to_bits() + 1)
}

fn next_down(value: f64) -> f64 {
    f64::from_bits(value.to_bits() - 1)
}

fn pack_reference_planar(codes: &[[u16; 3]]) -> Vec<u8> {
    let mut output = Vec::with_capacity(codes.len() * 6);
    for plane in 0..3 {
        for code in codes {
            output.extend_from_slice(&code[plane].to_le_bytes());
        }
    }
    output
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReferenceGeometry {
    pixel_count: u64,
    pair_words: u64,
    plane_bytes: u64,
    visible_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceGeometryError {
    ZeroDimension,
    OddWidth,
    PixelIndexOverflow,
    ByteCountOverflow,
    DispatchLimit,
}

fn checked_reference_geometry(
    width: u32,
    height: u32,
    max_workgroups: u32,
) -> Result<ReferenceGeometry, ReferenceGeometryError> {
    if width == 0 || height == 0 {
        return Err(ReferenceGeometryError::ZeroDimension);
    }
    if width & 1 != 0 {
        return Err(ReferenceGeometryError::OddWidth);
    }
    let pixel_count = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(ReferenceGeometryError::PixelIndexOverflow)?;
    let _ = u32::try_from(pixel_count).map_err(|_| ReferenceGeometryError::PixelIndexOverflow)?;
    let plane_bytes = pixel_count
        .checked_mul(2)
        .ok_or(ReferenceGeometryError::ByteCountOverflow)?;
    let visible_bytes = checked_visible_bytes(pixel_count)?;
    let dispatch_x = (width / 2).div_ceil(16);
    let dispatch_y = height.div_ceil(16);
    if dispatch_x > max_workgroups || dispatch_y > max_workgroups {
        return Err(ReferenceGeometryError::DispatchLimit);
    }
    Ok(ReferenceGeometry {
        pixel_count,
        pair_words: pixel_count / 2,
        plane_bytes,
        visible_bytes,
    })
}

fn checked_visible_bytes(pixel_count: u64) -> Result<u64, ReferenceGeometryError> {
    pixel_count
        .checked_mul(6)
        .ok_or(ReferenceGeometryError::ByteCountOverflow)
}
