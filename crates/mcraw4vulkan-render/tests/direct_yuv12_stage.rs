use std::sync::mpsc;
use std::time::Duration;

use mcraw4vulkan_core::{BayerPattern, FrameDimensions};
use mcraw4vulkan_render::{
    DIRECT_YUV12_STATUS_BYTE_LEN, DirectYuv12ColorTransform, DirectYuv12Error,
    GpuDirectYuv12EncodeInput, GpuDirectYuv12Stage, Yuv444p12lePackPolicy,
};
use mcraw4vulkan_vignette::{
    FixedPointVignetteInputFacts, GpuPipeF32BayerPrepareInput, GpuPipeF32BayerStage,
    PipeF32BayerCorrectionMode, VIGNETTE_GAIN_SCALE, VignetteCoordinateMapping,
    VignetteCorrectionInputFacts, VignetteCorrectionMode,
};

#[test]
fn corrected_bayer_direct_yuv12_matches_cpu_reference_for_all_cfa_and_odd_height() {
    pollster::block_on(async {
        let Some((device, queue)) = request_test_device().await else {
            return;
        };
        let dimensions = FrameDimensions {
            width: 18,
            height: 17,
        };
        for pattern in [
            BayerPattern::Rggb,
            BayerPattern::Bggr,
            BayerPattern::Grbg,
            BayerPattern::Gbrg,
        ] {
            let raw = mixed_raw_fixture(dimensions.pixel_count().expect("test dimensions fit"));
            let facts = identity_facts(dimensions, pattern);
            let expected_bayer = cpu_corrected_bayer(&facts, &raw);
            let expected_rgb = cpu_demosaic(&expected_bayer, dimensions, pattern);
            let expected_codes = cpu_planar_codes(&expected_rgb);
            let actual = run_direct_stage(&device, &queue, &facts, &raw)
                .expect("corrected-Bayer direct YUV12 stage runs");
            assert_eq!(actual.status, [0, 0, u32::MAX, 0], "{pattern:?}");
            assert_eq!(actual.codes, expected_codes, "{pattern:?}");
            assert_eq!(actual.visible_bytes, raw.len() * 6);
            assert_eq!(actual.plane_bytes, raw.len() * 2);
        }
    });
}

#[test]
fn direct_yuv12_rejects_odd_width_before_dispatch() {
    pollster::block_on(async {
        let Some((device, queue)) = request_test_device().await else {
            return;
        };
        let dimensions = FrameDimensions {
            width: 3,
            height: 3,
        };
        let raw = mixed_raw_fixture(9);
        let facts = identity_facts(dimensions, BayerPattern::Rggb);
        let error = run_direct_stage(&device, &queue, &facts, &raw)
            .expect_err("odd visible width must fail before submission");
        assert_eq!(error, DirectYuv12Error::OddVisibleWidth { width: 3 });
    });
}

#[derive(Debug)]
struct DirectReadback {
    codes: Vec<u16>,
    status: [u32; 4],
    visible_bytes: usize,
    plane_bytes: usize,
}

fn run_direct_stage(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    facts: &FixedPointVignetteInputFacts<'_>,
    raw: &[u16],
) -> Result<DirectReadback, DirectYuv12Error> {
    let packed = pack_u16_samples(raw);
    let input = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("direct YUV12 test packed-U16 input"),
        size: packed.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&input, 0, &packed);
    let output_bytes = u64::try_from(raw.len()).expect("test sample count fits") * 6;
    let output_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("direct YUV12 test output readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let status_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("direct YUV12 test status readback"),
        size: DIRECT_YUV12_STATUS_BYTE_LEN,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut bayer_stage =
        GpuPipeF32BayerStage::new(device, queue).expect("signed-f32 Bayer test stage creates");
    let mut direct_stage = GpuDirectYuv12Stage::new(device, queue)?;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("direct YUV12 test encoder"),
    });
    let (visible_bytes, plane_bytes) = {
        let bayer = bayer_stage
            .prepare_pipe_f32_bayer(GpuPipeF32BayerPrepareInput {
                device,
                queue,
                encoder: &mut encoder,
                input_buffer: &input,
                input_buffer_bytes: packed.len() as u64,
                facts,
                correction_mode: PipeF32BayerCorrectionMode::IdentitySpatialGain,
            })
            .expect("signed-f32 Bayer dispatch encodes");
        let direct = direct_stage.encode_direct_yuv12(GpuDirectYuv12EncodeInput {
            device,
            queue,
            encoder: &mut encoder,
            bayer: &bayer.view,
            color_transform: DirectYuv12ColorTransform {
                camera_to_normalized_ncl: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            pack_policy: Yuv444p12lePackPolicy::adopted(),
        })?;
        encoder.copy_buffer_to_buffer(
            direct.view.output_buffer(),
            0,
            &output_readback,
            0,
            direct.view.visible_byte_len(),
        );
        encoder.copy_buffer_to_buffer(
            direct.view.status_buffer(),
            0,
            &status_readback,
            0,
            direct.view.status_byte_len(),
        );
        (
            usize::try_from(direct.view.visible_byte_len()).expect("visible bytes fit"),
            usize::try_from(direct.view.plane_byte_len()).expect("plane bytes fit"),
        )
    };
    let submission = queue.submit(Some(encoder.finish()));
    let output = map_bytes(device, submission.clone(), &output_readback, output_bytes);
    let status_bytes = map_bytes(
        device,
        submission,
        &status_readback,
        DIRECT_YUV12_STATUS_BYTE_LEN,
    );
    let codes = output
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect();
    let mut status = [0_u32; 4];
    for (index, bytes) in status_bytes.chunks_exact(4).enumerate() {
        status[index] = u32::from_le_bytes(bytes.try_into().expect("status word bytes"));
    }
    Ok(DirectReadback {
        codes,
        status,
        visible_bytes,
        plane_bytes,
    })
}

fn map_bytes(
    device: &wgpu::Device,
    submission: wgpu::SubmissionIndex,
    buffer: &wgpu::Buffer,
    byte_len: u64,
) -> Vec<u8> {
    let slice = buffer.slice(..byte_len);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::wait_for(submission));
    receiver
        .recv_timeout(Duration::from_secs(30))
        .expect("readback callback completes")
        .expect("readback maps");
    let mapped = slice.get_mapped_range();
    let bytes = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    bytes
}

fn identity_facts(
    dimensions: FrameDimensions,
    pattern: BayerPattern,
) -> FixedPointVignetteInputFacts<'static> {
    let input = VignetteCorrectionInputFacts::new(
        VignetteCorrectionMode::Enabled,
        VignetteCoordinateMapping::VisibleFrame,
        dimensions,
        pattern,
        None,
        [64.0; 4],
        4095,
    )
    .expect("identity facts validate");
    FixedPointVignetteInputFacts::from_input_facts(&input).expect("facts quantize")
}

fn cpu_corrected_bayer(facts: &FixedPointVignetteInputFacts<'_>, raw: &[u16]) -> Vec<f32> {
    let gain_q = (f64::from(facts.pixel_domain.source_to_corrected_scale)
        * VIGNETTE_GAIN_SCALE as f64)
        .round() as u32;
    let width = facts.frame_dimensions.width as usize;
    raw.iter()
        .enumerate()
        .map(|(index, raw)| {
            let x = index % width;
            let y = index / width;
            let position = (y & 1) * 2 + (x & 1);
            let black_q = u32::try_from(facts.input_black_level_q[position])
                .expect("test black level is nonnegative");
            let black_f = black_q as f32 / VIGNETTE_GAIN_SCALE as f32;
            let gain_f = gain_q as f32 / VIGNETTE_GAIN_SCALE as f32;
            ((f32::from(*raw) - black_f) * gain_f)
                / f32::from(facts.pixel_domain.corrected_white_tag)
        })
        .collect()
}

fn cpu_planar_codes(rgb: &[[f32; 3]]) -> Vec<u16> {
    let mut output = Vec::with_capacity(rgb.len() * 3);
    for component in 0..3 {
        for value in rgb {
            let mapped = match component {
                0 => 256.0 + 3504.0 * value[0],
                1 => 2048.0 + 3584.0 * value[1],
                _ => 2048.0 + 3584.0 * value[2],
            };
            output.push((mapped.clamp(16.0, 4079.0) + 0.5).floor() as u16);
        }
    }
    output
}

fn cpu_demosaic(
    bayer: &[f32],
    dimensions: FrameDimensions,
    pattern: BayerPattern,
) -> Vec<[f32; 3]> {
    let mut output = Vec::with_capacity(bayer.len());
    for y in 0..dimensions.height {
        for x in 0..dimensions.width {
            output.push(cpu_demosaic_pixel(bayer, dimensions, pattern, x, y));
        }
    }
    output
}

fn cpu_demosaic_pixel(
    bayer: &[f32],
    dimensions: FrameDimensions,
    pattern: BayerPattern,
    x: u32,
    y: u32,
) -> [f32; 3] {
    let xi = x as i32;
    let yi = y as i32;
    let load = |x, y| cpu_load_clamped(bayer, dimensions, x, y);
    let center = load(xi, yi);
    match cpu_color_at(pattern, x, y) {
        0 => [
            center,
            avg4(
                load(xi - 1, yi),
                load(xi + 1, yi),
                load(xi, yi - 1),
                load(xi, yi + 1),
            ),
            avg4(
                load(xi - 1, yi - 1),
                load(xi + 1, yi - 1),
                load(xi - 1, yi + 1),
                load(xi + 1, yi + 1),
            ),
        ],
        2 => [
            avg4(
                load(xi - 1, yi - 1),
                load(xi + 1, yi - 1),
                load(xi - 1, yi + 1),
                load(xi + 1, yi + 1),
            ),
            avg4(
                load(xi - 1, yi),
                load(xi + 1, yi),
                load(xi, yi - 1),
                load(xi, yi + 1),
            ),
            center,
        ],
        _ => {
            let red_horizontal = cpu_color_at(pattern, clamp_coord(xi - 1, dimensions.width), y)
                == 0
                || cpu_color_at(pattern, clamp_coord(xi + 1, dimensions.width), y) == 0;
            if red_horizontal {
                [
                    avg2(load(xi - 1, yi), load(xi + 1, yi)),
                    center,
                    avg2(load(xi, yi - 1), load(xi, yi + 1)),
                ]
            } else {
                [
                    avg2(load(xi, yi - 1), load(xi, yi + 1)),
                    center,
                    avg2(load(xi - 1, yi), load(xi + 1, yi)),
                ]
            }
        }
    }
}

fn cpu_load_clamped(bayer: &[f32], dimensions: FrameDimensions, x: i32, y: i32) -> f32 {
    let x = clamp_coord(x, dimensions.width);
    let y = clamp_coord(y, dimensions.height);
    bayer[y as usize * dimensions.width as usize + x as usize]
}

fn clamp_coord(value: i32, upper: u32) -> u32 {
    if value < 0 {
        0
    } else {
        (value as u32).min(upper - 1)
    }
}

fn cpu_color_at(pattern: BayerPattern, x: u32, y: u32) -> u32 {
    let position = ((y & 1) * 2) + (x & 1);
    match pattern {
        BayerPattern::Rggb => match position {
            0 => 0,
            3 => 2,
            _ => 1,
        },
        BayerPattern::Bggr => match position {
            0 => 2,
            3 => 0,
            _ => 1,
        },
        BayerPattern::Grbg => match position {
            1 => 0,
            2 => 2,
            _ => 1,
        },
        BayerPattern::Gbrg => match position {
            1 => 2,
            2 => 0,
            _ => 1,
        },
    }
}

fn avg2(a: f32, b: f32) -> f32 {
    (a + b) * 0.5
}

fn avg4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    (a + b + c + d) * 0.25
}

fn mixed_raw_fixture(sample_count: usize) -> Vec<u16> {
    const VALUES: [u16; 16] = [
        60,
        64,
        65,
        512,
        1023,
        1024,
        4095,
        4096,
        8192,
        16_383,
        20_000,
        32_768,
        u16::MAX,
        63,
        96,
        2048,
    ];
    (0..sample_count)
        .map(|index| VALUES[(index * 7 + index / 3) % VALUES.len()])
        .collect()
}

fn pack_u16_samples(samples: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len().div_ceil(2) * 4);
    for pair in samples.chunks(2) {
        let low = u32::from(pair[0]);
        let high = pair.get(1).copied().map(u32::from).unwrap_or(0);
        bytes.extend_from_slice(&(low | (high << 16)).to_le_bytes());
    }
    bytes
}

async fn request_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    std::env::var_os("MCRAW4VULKAN_RUN_WGPU_TESTS")?;
    let instance = wgpu::Instance::default();
    for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
        let limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("direct YUV12 integration test device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
        };
        if let Ok(device) = adapter.request_device(&descriptor, None).await {
            return Some(device);
        }
    }
    panic!("MCRAW4VULKAN_RUN_WGPU_TESTS was set but no adapter created a device")
}
