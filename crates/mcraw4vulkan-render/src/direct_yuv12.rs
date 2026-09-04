//! Production direct linear-TV YUV12 terminal for the public PIPE renderer.
//!
//! This module accepts only the typed relative-linear Candidate-4 dispatch and
//! a pre-resolved camera-to-normalized-NCL matrix. Metadata interpretation and
//! frame scheduling deliberately remain outside the render crate.

use mcraw4vulkan_core::{BayerPattern, FrameDimensions};
use mcraw4vulkan_vignette::{GpuCorrectedF32BayerView, PipeF32BayerNumericDomain};
use thiserror::Error;

use crate::{DIRECT_YUV12_TERMINAL_WGSL, SIGNED_BAYER_DEMOSAIC_WGSL};

const DIRECT_YUV12_PARAMS_BYTE_LEN: u64 = 64;
const DIRECT_YUV12_STORAGE_BINDINGS: u32 = 3;
const DIRECT_YUV12_UNIFORM_BINDINGS: u32 = 1;
const DIRECT_YUV12_WORKGROUP_X: u32 = 16;
const DIRECT_YUV12_WORKGROUP_Y: u32 = 16;
const DIRECT_YUV12_SAMPLE_BYTES: u64 = 4;
const SIGNED_BAYER_DEMOSAIC_INSERTION_POINT: &str = "// __SIGNED_BAYER_DEMOSAIC_WGSL__";

fn direct_yuv12_shader_source() -> String {
    debug_assert_eq!(
        DIRECT_YUV12_TERMINAL_WGSL
            .matches(SIGNED_BAYER_DEMOSAIC_INSERTION_POINT)
            .count(),
        1
    );
    DIRECT_YUV12_TERMINAL_WGSL.replacen(
        SIGNED_BAYER_DEMOSAIC_INSERTION_POINT,
        SIGNED_BAYER_DEMOSAIC_WGSL,
        1,
    )
}

pub const DIRECT_YUV12_STATUS_BYTE_LEN: u64 = 16;
pub const DIRECT_YUV12_NONFINITE_CAMERA: u32 = 1;
pub const DIRECT_YUV12_NONFINITE_NCL: u32 = 2;
pub const DIRECT_YUV12_NONFINITE_MAPPED: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Yuv444p12leNumericDomain {
    LinearBt2020NclTvRangeV1,
}

/// The one adopted linear 12-bit TV-range planar packing policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Yuv444p12lePackPolicy {
    _adopted_only: (),
}

impl Yuv444p12lePackPolicy {
    pub const ADOPTED: Self = Self { _adopted_only: () };
    pub const DOMAIN: Yuv444p12leNumericDomain = Yuv444p12leNumericDomain::LinearBt2020NclTvRangeV1;
    pub const PLANE_COUNT: u8 = 3;
    pub const PLANE_ORDER: [&str; 3] = ["Y", "Cb", "Cr"];
    pub const BYTES_PER_SAMPLE: u8 = 2;
    pub const STORAGE_BYTES_PER_PIXEL: u8 = 6;
    pub const STORAGE_BITS_PER_SAMPLE: u8 = 16;
    pub const MEANINGFUL_BITS_PER_SAMPLE: u8 = 12;
    pub const LINEAR_SIGNAL_SCALE_NUMERATOR: u32 = 1;
    pub const LINEAR_SIGNAL_SCALE_DENOMINATOR: u32 = 2;
    pub const LINEAR_SIGNAL_SCALE_STOPS: i32 = -1;
    pub const LINEAR_SIGNAL_SCALE_FACTOR_F32: f32 = 0.5;
    pub const LUMA_OFFSET: u16 = 256;
    pub const LUMA_SCALE: u16 = 3504;
    pub const CHROMA_OFFSET: u16 = 2048;
    pub const CHROMA_SCALE: u16 = 3584;
    pub const NOMINAL_LUMA_MIN_CODE: u16 = Self::LUMA_OFFSET;
    pub const NOMINAL_LUMA_MAX_CODE: u16 = Self::LUMA_OFFSET + Self::LUMA_SCALE;
    pub const NOMINAL_CHROMA_MIN_CODE: u16 = Self::CHROMA_OFFSET - Self::CHROMA_SCALE / 2;
    pub const NOMINAL_CHROMA_MAX_CODE: u16 = Self::CHROMA_OFFSET + Self::CHROMA_SCALE / 2;
    pub const MIN_CODE: u16 = 16;
    pub const MAX_CODE: u16 = 4079;

    pub const fn adopted() -> Self {
        Self::ADOPTED
    }
}

/// Per-frame transform resolved by the strict host color authority.
///
/// Rows map signed, relative-linear camera RGB to normalized BT.2020-NCL
/// Y/Cb/Cr. No metadata or context fingerprint crosses this render boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectYuv12ColorTransform {
    pub camera_to_normalized_ncl: [f32; 9],
}

pub struct GpuDirectYuv12EncodeInput<'input, 'view, 'encoder> {
    pub device: &'encoder wgpu::Device,
    pub queue: &'encoder wgpu::Queue,
    pub encoder: &'encoder mut wgpu::CommandEncoder,
    pub bayer: &'input GpuCorrectedF32BayerView<'view>,
    pub color_transform: DirectYuv12ColorTransform,
    pub pack_policy: Yuv444p12lePackPolicy,
}

/// Borrowed tightly packed planar Y/Cb/Cr output plus its device status.
#[derive(Debug)]
pub struct GpuDirectYuv12View<'a> {
    output_buffer: &'a wgpu::Buffer,
    status_buffer: &'a wgpu::Buffer,
    dimensions: FrameDimensions,
    pixel_count: usize,
    plane_byte_len: u64,
    visible_byte_len: u64,
    numeric_domain: Yuv444p12leNumericDomain,
    pack_policy: Yuv444p12lePackPolicy,
}

impl<'a> GpuDirectYuv12View<'a> {
    pub fn output_buffer(&self) -> &'a wgpu::Buffer {
        self.output_buffer
    }

    pub fn status_buffer(&self) -> &'a wgpu::Buffer {
        self.status_buffer
    }

    pub fn dimensions(&self) -> FrameDimensions {
        self.dimensions
    }

    pub fn width(&self) -> u32 {
        self.dimensions.width
    }

    pub fn height(&self) -> u32 {
        self.dimensions.height
    }

    pub fn pixel_count(&self) -> usize {
        self.pixel_count
    }

    pub fn plane_byte_len(&self) -> u64 {
        self.plane_byte_len
    }

    pub fn visible_byte_len(&self) -> u64 {
        self.visible_byte_len
    }

    pub fn status_byte_len(&self) -> u64 {
        DIRECT_YUV12_STATUS_BYTE_LEN
    }

    pub fn numeric_domain(&self) -> Yuv444p12leNumericDomain {
        self.numeric_domain
    }

    pub fn pack_policy(&self) -> Yuv444p12lePackPolicy {
        self.pack_policy
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuDirectYuv12DispatchStats {
    pub numeric_domain: Yuv444p12leNumericDomain,
    pub pack_policy: Yuv444p12lePackPolicy,
    pub pixel_count: u64,
    pub input_binding_bytes: u64,
    pub visible_output_bytes: u64,
    pub allocated_output_bytes: u64,
    pub status_bytes: u64,
    pub dispatch_workgroups_x: u32,
    pub dispatch_workgroups_y: u32,
    pub workgroup_size_x: u32,
    pub workgroup_size_y: u32,
    pub max_abs_matrix_row_sum: f64,
    pub conservative_max_abs_camera_component: f64,
    pub conservative_max_abs_ncl_component: f64,
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
    pub output_buffer_allocated: bool,
    pub output_allocation_count: u64,
    pub output_reuse_count: u64,
    pub bind_group_allocation_count: u64,
    pub bind_group_reuse_count: u64,
}

#[derive(Debug)]
pub struct GpuDirectYuv12Dispatch<'a> {
    pub view: GpuDirectYuv12View<'a>,
    pub stats: GpuDirectYuv12DispatchStats,
}

struct GpuSizedBuffer {
    buffer: wgpu::Buffer,
    size: u64,
}

struct GpuDirectYuv12BindGroupCache {
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    input_binding_bytes: u64,
    output_binding_bytes: u64,
    bind_group: wgpu::BindGroup,
}

/// Reusable metadata-free direct-YUV stage.
///
/// The caller must encode output/status copies and submit before reusing this
/// instance for another frame. Queue ordering then protects its shared params,
/// output, and status resources without a host wait between frames.
pub struct GpuDirectYuv12Stage {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: GpuSizedBuffer,
    status_reset_buffer: GpuSizedBuffer,
    status_buffer: GpuSizedBuffer,
    output_buffer: Option<GpuSizedBuffer>,
    output_allocation_count: u64,
    output_reuse_count: u64,
    bind_group_cache: Option<GpuDirectYuv12BindGroupCache>,
    bind_group_allocation_count: u64,
    bind_group_reuse_count: u64,
}

impl GpuDirectYuv12Stage {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, DirectYuv12Error> {
        let shader_source = direct_yuv12_shader_source();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mcraw4vulkan direct linear-TV YUV12 shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mcraw4vulkan direct linear-TV YUV12 pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("direct_yuv12_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let params_buffer = GpuSizedBuffer {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mcraw4vulkan direct YUV12 params"),
                size: DIRECT_YUV12_PARAMS_BYTE_LEN,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            size: DIRECT_YUV12_PARAMS_BYTE_LEN,
        };
        let status_reset_buffer = GpuSizedBuffer {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mcraw4vulkan direct YUV12 status reset"),
                size: DIRECT_YUV12_STATUS_BYTE_LEN,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            size: DIRECT_YUV12_STATUS_BYTE_LEN,
        };
        queue.write_buffer(
            &status_reset_buffer.buffer,
            0,
            &u32_words_to_le_bytes(&[0, 0, u32::MAX, 0]),
        );
        let status_buffer = GpuSizedBuffer {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mcraw4vulkan direct YUV12 nonfinite status"),
                size: DIRECT_YUV12_STATUS_BYTE_LEN,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            size: DIRECT_YUV12_STATUS_BYTE_LEN,
        };
        Ok(Self {
            pipeline,
            bind_group_layout,
            params_buffer,
            status_reset_buffer,
            status_buffer,
            output_buffer: None,
            output_allocation_count: 0,
            output_reuse_count: 0,
            bind_group_cache: None,
            bind_group_allocation_count: 0,
            bind_group_reuse_count: 0,
        })
    }

    pub fn allocated_output_bytes(&self) -> u64 {
        self.output_buffer.as_ref().map_or(0, |buffer| buffer.size)
    }

    pub fn output_allocation_count(&self) -> u64 {
        self.output_allocation_count
    }

    pub fn output_reuse_count(&self) -> u64 {
        self.output_reuse_count
    }

    pub fn encode_direct_yuv12<'a>(
        &'a mut self,
        input: GpuDirectYuv12EncodeInput<'_, '_, '_>,
    ) -> Result<GpuDirectYuv12Dispatch<'a>, DirectYuv12Error> {
        let contract = DirectYuv12Contract::from_input(
            input.device,
            input.bayer,
            input.color_transform,
            input.pack_policy,
        )?;
        let output_buffer_allocated = ensure_output_buffer(
            input.device,
            &mut self.output_buffer,
            contract.visible_output_bytes,
        );
        if output_buffer_allocated {
            self.output_allocation_count = self.output_allocation_count.saturating_add(1);
        } else {
            self.output_reuse_count = self.output_reuse_count.saturating_add(1);
        }

        input.queue.write_buffer(
            &self.params_buffer.buffer,
            0,
            &contract.params.to_le_bytes(),
        );
        input.encoder.copy_buffer_to_buffer(
            &self.status_reset_buffer.buffer,
            0,
            &self.status_buffer.buffer,
            0,
            DIRECT_YUV12_STATUS_BYTE_LEN,
        );
        let output = self
            .output_buffer
            .as_ref()
            .expect("direct YUV12 output was ensured before binding");
        let cache_matches = self.bind_group_cache.as_ref().is_some_and(|cache| {
            cache.input_buffer == *input.bayer.buffer()
                && cache.output_buffer == output.buffer
                && cache.input_binding_bytes == contract.input_binding_bytes
                && cache.output_binding_bytes == contract.visible_output_bytes
        });
        if cache_matches {
            self.bind_group_reuse_count = self.bind_group_reuse_count.saturating_add(1);
        } else {
            let bind_group = input.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mcraw4vulkan direct linear-TV YUV12 bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: input.bayer.buffer(),
                            offset: 0,
                            size: wgpu::BufferSize::new(contract.input_binding_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &output.buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(contract.visible_output_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.status_buffer.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.params_buffer.buffer.as_entire_binding(),
                    },
                ],
            });
            self.bind_group_cache = Some(GpuDirectYuv12BindGroupCache {
                input_buffer: input.bayer.buffer().clone(),
                output_buffer: output.buffer.clone(),
                input_binding_bytes: contract.input_binding_bytes,
                output_binding_bytes: contract.visible_output_bytes,
                bind_group,
            });
            self.bind_group_allocation_count = self.bind_group_allocation_count.saturating_add(1);
        }
        let bind_group = &self
            .bind_group_cache
            .as_ref()
            .expect("direct YUV12 bind group cache was populated")
            .bind_group;

        {
            let mut pass = input
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mcraw4vulkan direct linear-TV YUV12 pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(
                contract.dispatch_workgroups_x,
                contract.dispatch_workgroups_y,
                1,
            );
        }

        let stats = GpuDirectYuv12DispatchStats {
            numeric_domain: Yuv444p12lePackPolicy::DOMAIN,
            pack_policy: input.pack_policy,
            pixel_count: u64::try_from(contract.pixel_count).unwrap_or(u64::MAX),
            input_binding_bytes: contract.input_binding_bytes,
            visible_output_bytes: contract.visible_output_bytes,
            allocated_output_bytes: output.size,
            status_bytes: DIRECT_YUV12_STATUS_BYTE_LEN,
            dispatch_workgroups_x: contract.dispatch_workgroups_x,
            dispatch_workgroups_y: contract.dispatch_workgroups_y,
            workgroup_size_x: DIRECT_YUV12_WORKGROUP_X,
            workgroup_size_y: DIRECT_YUV12_WORKGROUP_Y,
            max_abs_matrix_row_sum: contract.max_abs_matrix_row_sum,
            conservative_max_abs_camera_component: contract.conservative_max_abs_camera_component,
            conservative_max_abs_ncl_component: contract.conservative_max_abs_ncl_component,
            max_storage_buffer_binding_size: contract.limits.max_storage_buffer_binding_size,
            max_buffer_size: contract.limits.max_buffer_size,
            max_compute_workgroups_per_dimension: contract
                .limits
                .max_compute_workgroups_per_dimension,
            output_buffer_allocated,
            output_allocation_count: self.output_allocation_count,
            output_reuse_count: self.output_reuse_count,
            bind_group_allocation_count: self.bind_group_allocation_count,
            bind_group_reuse_count: self.bind_group_reuse_count,
        };
        let view = GpuDirectYuv12View {
            output_buffer: &output.buffer,
            status_buffer: &self.status_buffer.buffer,
            dimensions: contract.dimensions,
            pixel_count: contract.pixel_count,
            plane_byte_len: contract.plane_byte_len,
            visible_byte_len: contract.visible_output_bytes,
            numeric_domain: Yuv444p12lePackPolicy::DOMAIN,
            pack_policy: input.pack_policy,
        };

        Ok(GpuDirectYuv12Dispatch { view, stats })
    }
}

#[derive(Debug, Clone, Copy)]
struct DirectYuv12Limits {
    max_storage_buffer_binding_size: u64,
    max_buffer_size: u64,
    max_compute_workgroups_per_dimension: u32,
}

#[derive(Debug, Clone, Copy)]
struct DirectYuv12Geometry {
    pixel_count: usize,
    pixel_count_u32: u32,
    input_binding_bytes: u64,
    plane_byte_len: u64,
    visible_output_bytes: u64,
    dispatch_workgroups_x: u32,
    dispatch_workgroups_y: u32,
}

impl DirectYuv12Geometry {
    fn validate(
        dimensions: FrameDimensions,
        limits: DirectYuv12Limits,
    ) -> Result<Self, DirectYuv12Error> {
        if dimensions.width == 0 || dimensions.height == 0 {
            return Err(DirectYuv12Error::InvalidDimensions { dimensions });
        }
        if dimensions.width & 1 != 0 {
            return Err(DirectYuv12Error::OddVisibleWidth {
                width: dimensions.width,
            });
        }
        let pixel_count = dimensions
            .pixel_count()
            .ok_or(DirectYuv12Error::DimensionOverflow { dimensions })?;
        let pixel_count_u32 = u32::try_from(pixel_count)
            .map_err(|_| DirectYuv12Error::DimensionOverflow { dimensions })?;
        let pixel_count_u64 = u64::try_from(pixel_count)
            .map_err(|_| DirectYuv12Error::DimensionOverflow { dimensions })?;
        let input_binding_bytes = pixel_count_u64
            .checked_mul(DIRECT_YUV12_SAMPLE_BYTES)
            .ok_or(DirectYuv12Error::DimensionOverflow { dimensions })?;
        let plane_byte_len = pixel_count_u64
            .checked_mul(u64::from(Yuv444p12lePackPolicy::BYTES_PER_SAMPLE))
            .ok_or(DirectYuv12Error::DimensionOverflow { dimensions })?;
        let visible_output_bytes = pixel_count_u64
            .checked_mul(u64::from(Yuv444p12lePackPolicy::STORAGE_BYTES_PER_PIXEL))
            .ok_or(DirectYuv12Error::DimensionOverflow { dimensions })?;
        for (buffer, required_bytes) in [
            ("relative-linear f32 Bayer input", input_binding_bytes),
            ("direct YUV12 output", visible_output_bytes),
            ("direct YUV12 status", DIRECT_YUV12_STATUS_BYTE_LEN),
        ] {
            if required_bytes > limits.max_storage_buffer_binding_size {
                return Err(DirectYuv12Error::StorageBindingTooLarge {
                    buffer,
                    required_bytes,
                    max_binding_bytes: limits.max_storage_buffer_binding_size,
                });
            }
        }
        if visible_output_bytes > limits.max_buffer_size {
            return Err(DirectYuv12Error::OutputBufferTooLarge {
                required_bytes: visible_output_bytes,
                max_buffer_bytes: limits.max_buffer_size,
            });
        }
        let pair_width = dimensions.width / 2;
        let dispatch_workgroups_x = pair_width.div_ceil(DIRECT_YUV12_WORKGROUP_X);
        let dispatch_workgroups_y = dimensions.height.div_ceil(DIRECT_YUV12_WORKGROUP_Y);
        for (axis, required_workgroups) in
            [("x", dispatch_workgroups_x), ("y", dispatch_workgroups_y)]
        {
            if required_workgroups > limits.max_compute_workgroups_per_dimension {
                return Err(DirectYuv12Error::DispatchTooLarge {
                    axis,
                    required_workgroups,
                    max_workgroups: limits.max_compute_workgroups_per_dimension,
                });
            }
        }
        Ok(Self {
            pixel_count,
            pixel_count_u32,
            input_binding_bytes,
            plane_byte_len,
            visible_output_bytes,
            dispatch_workgroups_x,
            dispatch_workgroups_y,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct DirectYuv12Params {
    words: [u32; 16],
}

impl DirectYuv12Params {
    fn to_le_bytes(self) -> [u8; DIRECT_YUV12_PARAMS_BYTE_LEN as usize] {
        let mut bytes = [0_u8; DIRECT_YUV12_PARAMS_BYTE_LEN as usize];
        for (index, word) in self.words.into_iter().enumerate() {
            let start = index * std::mem::size_of::<u32>();
            bytes[start..start + std::mem::size_of::<u32>()].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

struct DirectYuv12Contract {
    dimensions: FrameDimensions,
    pixel_count: usize,
    input_binding_bytes: u64,
    plane_byte_len: u64,
    visible_output_bytes: u64,
    dispatch_workgroups_x: u32,
    dispatch_workgroups_y: u32,
    params: DirectYuv12Params,
    max_abs_matrix_row_sum: f64,
    conservative_max_abs_camera_component: f64,
    conservative_max_abs_ncl_component: f64,
    limits: DirectYuv12Limits,
}

impl DirectYuv12Contract {
    fn from_input(
        device: &wgpu::Device,
        bayer: &GpuCorrectedF32BayerView<'_>,
        color_transform: DirectYuv12ColorTransform,
        pack_policy: Yuv444p12lePackPolicy,
    ) -> Result<Self, DirectYuv12Error> {
        debug_assert_eq!(pack_policy, Yuv444p12lePackPolicy::ADOPTED);
        let conservative_max_abs_camera_component = bayer.conservative_max_abs_four_tap_sum();
        let dimensions = bayer.dimensions();
        if bayer.numeric_domain() != PipeF32BayerNumericDomain::RelativeLinearCorrectedCodeV1 {
            return Err(DirectYuv12Error::UnsupportedNumericDomain {
                domain: bayer.numeric_domain(),
            });
        }

        let matrix = color_transform.camera_to_normalized_ncl;
        for (index, value) in matrix.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(DirectYuv12Error::NonfiniteColorTransform { index, value });
            }
        }
        let max_abs_matrix_row_sum = matrix
            .chunks_exact(3)
            .map(|row| row.iter().map(|value| f64::from(value.abs())).sum::<f64>())
            .fold(0.0_f64, f64::max);
        if !conservative_max_abs_camera_component.is_finite()
            || conservative_max_abs_camera_component < 0.0
        {
            return Err(DirectYuv12Error::InvalidCameraMagnitudeBound {
                max_abs_camera_component: conservative_max_abs_camera_component,
            });
        }
        let conservative_max_abs_ncl_component =
            conservative_max_abs_camera_component * max_abs_matrix_row_sum;
        let allowed = f64::from(f32::MAX) / 2.0;
        if !conservative_max_abs_ncl_component.is_finite()
            || conservative_max_abs_ncl_component > allowed
        {
            return Err(DirectYuv12Error::UnsafeColorMagnitude {
                max_abs_camera_component: conservative_max_abs_camera_component,
                max_abs_matrix_row_sum,
                max_abs_ncl_component: conservative_max_abs_ncl_component,
                allowed_max_abs_ncl_component: allowed,
            });
        }

        let limits_raw = device.limits();
        if limits_raw.max_bind_groups < 1
            || limits_raw.max_storage_buffers_per_shader_stage < DIRECT_YUV12_STORAGE_BINDINGS
            || limits_raw.max_uniform_buffers_per_shader_stage < DIRECT_YUV12_UNIFORM_BINDINGS
            || limits_raw.max_compute_workgroup_size_x < DIRECT_YUV12_WORKGROUP_X
            || limits_raw.max_compute_workgroup_size_y < DIRECT_YUV12_WORKGROUP_Y
            || limits_raw.max_compute_invocations_per_workgroup
                < DIRECT_YUV12_WORKGROUP_X * DIRECT_YUV12_WORKGROUP_Y
        {
            return Err(DirectYuv12Error::InsufficientAdapterLimits {
                max_bind_groups: limits_raw.max_bind_groups,
                max_storage_buffers_per_shader_stage: limits_raw
                    .max_storage_buffers_per_shader_stage,
                max_uniform_buffers_per_shader_stage: limits_raw
                    .max_uniform_buffers_per_shader_stage,
                max_compute_workgroup_size_x: limits_raw.max_compute_workgroup_size_x,
                max_compute_workgroup_size_y: limits_raw.max_compute_workgroup_size_y,
                max_compute_invocations_per_workgroup: limits_raw
                    .max_compute_invocations_per_workgroup,
            });
        }
        let limits = DirectYuv12Limits {
            max_storage_buffer_binding_size: u64::from(limits_raw.max_storage_buffer_binding_size),
            max_buffer_size: limits_raw.max_buffer_size,
            max_compute_workgroups_per_dimension: limits_raw.max_compute_workgroups_per_dimension,
        };
        let geometry = DirectYuv12Geometry::validate(dimensions, limits)?;
        if bayer.sample_count() != geometry.pixel_count
            || bayer.visible_byte_len() != geometry.input_binding_bytes
        {
            return Err(DirectYuv12Error::InconsistentBayerDispatch {
                expected_samples: geometry.pixel_count,
                view_samples: bayer.sample_count(),
                expected_bytes: geometry.input_binding_bytes,
                view_bytes: bayer.visible_byte_len(),
            });
        }
        if bayer.buffer().size() < geometry.input_binding_bytes {
            return Err(DirectYuv12Error::InputBufferTooSmall {
                required_bytes: geometry.input_binding_bytes,
                allocated_bytes: bayer.buffer().size(),
            });
        }
        let plane_words = geometry.pixel_count_u32 / 2;

        let mut words = [0_u32; 16];
        words[0] = dimensions.width;
        words[1] = dimensions.height;
        words[2] = bayer_pattern_tag(bayer.bayer_pattern());
        words[3] = plane_words;
        for row in 0..3 {
            for column in 0..3 {
                words[4 + row * 4 + column] = matrix[row * 3 + column].to_bits();
            }
        }

        Ok(Self {
            dimensions,
            pixel_count: geometry.pixel_count,
            input_binding_bytes: geometry.input_binding_bytes,
            plane_byte_len: geometry.plane_byte_len,
            visible_output_bytes: geometry.visible_output_bytes,
            dispatch_workgroups_x: geometry.dispatch_workgroups_x,
            dispatch_workgroups_y: geometry.dispatch_workgroups_y,
            params: DirectYuv12Params { words },
            max_abs_matrix_row_sum,
            conservative_max_abs_camera_component,
            conservative_max_abs_ncl_component,
            limits,
        })
    }
}

fn ensure_output_buffer(
    device: &wgpu::Device,
    slot: &mut Option<GpuSizedBuffer>,
    required_bytes: u64,
) -> bool {
    if slot
        .as_ref()
        .is_some_and(|buffer| buffer.size >= required_bytes)
    {
        return false;
    }
    *slot = Some(GpuSizedBuffer {
        buffer: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcraw4vulkan direct linear-TV YUV12 output"),
            size: required_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        size: required_bytes,
    });
    true
}

fn bayer_pattern_tag(pattern: BayerPattern) -> u32 {
    match pattern {
        BayerPattern::Rggb => 0,
        BayerPattern::Bggr => 1,
        BayerPattern::Grbg => 2,
        BayerPattern::Gbrg => 3,
    }
}

fn u32_words_to_le_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum DirectYuv12Error {
    #[error("direct YUV12 dimensions {dimensions:?} must be positive")]
    InvalidDimensions { dimensions: FrameDimensions },
    #[error("direct YUV12 requires an even visible width, got {width}")]
    OddVisibleWidth { width: u32 },
    #[error("direct YUV12 dimensions {dimensions:?} overflow checked buffer/index arithmetic")]
    DimensionOverflow { dimensions: FrameDimensions },
    #[error("direct YUV12 requires RelativeLinearCorrectedCodeV1, got {domain:?}")]
    UnsupportedNumericDomain { domain: PipeF32BayerNumericDomain },
    #[error(
        "Candidate-4 view mismatch: expected {expected_samples} samples/{expected_bytes} bytes, view has {view_samples}/{view_bytes}"
    )]
    InconsistentBayerDispatch {
        expected_samples: usize,
        view_samples: usize,
        expected_bytes: u64,
        view_bytes: u64,
    },
    #[error(
        "Candidate-4 f32 input requires {required_bytes} allocated bytes, got {allocated_bytes}"
    )]
    InputBufferTooSmall {
        required_bytes: u64,
        allocated_bytes: u64,
    },
    #[error("direct YUV12 matrix element {index} is nonfinite: {value}")]
    NonfiniteColorTransform { index: usize, value: f32 },
    #[error(
        "Candidate-4 conservative camera magnitude bound is invalid: {max_abs_camera_component}"
    )]
    InvalidCameraMagnitudeBound { max_abs_camera_component: f64 },
    #[error(
        "direct YUV12 matrix bound is unsafe: camera={max_abs_camera_component}, row_sum={max_abs_matrix_row_sum}, ncl={max_abs_ncl_component}, allowed={allowed_max_abs_ncl_component}"
    )]
    UnsafeColorMagnitude {
        max_abs_camera_component: f64,
        max_abs_matrix_row_sum: f64,
        max_abs_ncl_component: f64,
        allowed_max_abs_ncl_component: f64,
    },
    #[error(
        "adapter cannot bind/dispatch direct YUV12 stage: bind_groups={max_bind_groups}, storage={max_storage_buffers_per_shader_stage}, uniforms={max_uniform_buffers_per_shader_stage}, workgroup=({max_compute_workgroup_size_x},{max_compute_workgroup_size_y}), invocations={max_compute_invocations_per_workgroup}"
    )]
    InsufficientAdapterLimits {
        max_bind_groups: u32,
        max_storage_buffers_per_shader_stage: u32,
        max_uniform_buffers_per_shader_stage: u32,
        max_compute_workgroup_size_x: u32,
        max_compute_workgroup_size_y: u32,
        max_compute_invocations_per_workgroup: u32,
    },
    #[error(
        "{buffer} requires a {required_bytes}-byte storage binding, adapter limit is {max_binding_bytes}"
    )]
    StorageBindingTooLarge {
        buffer: &'static str,
        required_bytes: u64,
        max_binding_bytes: u64,
    },
    #[error(
        "direct YUV12 output requires {required_bytes} bytes, adapter max buffer size is {max_buffer_bytes}"
    )]
    OutputBufferTooLarge {
        required_bytes: u64,
        max_buffer_bytes: u64,
    },
    #[error(
        "direct YUV12 {axis}-axis dispatch requires {required_workgroups} workgroups, adapter limit is {max_workgroups}"
    )]
    DispatchTooLarge {
        axis: &'static str,
        required_workgroups: u32,
        max_workgroups: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unrestricted_geometry_limits() -> DirectYuv12Limits {
        DirectYuv12Limits {
            max_storage_buffer_binding_size: u64::MAX,
            max_buffer_size: u64::MAX,
            max_compute_workgroups_per_dimension: u32::MAX,
        }
    }

    #[test]
    fn params_layout_is_four_vec4_words() {
        assert_eq!(std::mem::size_of::<DirectYuv12Params>(), 64);
        assert_eq!(DIRECT_YUV12_PARAMS_BYTE_LEN, 64);
    }

    #[test]
    fn adopted_pack_policy_is_the_single_storage_mapping_and_scale_authority() {
        assert_eq!(Yuv444p12lePackPolicy::PLANE_COUNT, 3);
        assert_eq!(Yuv444p12lePackPolicy::PLANE_ORDER, ["Y", "Cb", "Cr"]);
        assert_eq!(Yuv444p12lePackPolicy::BYTES_PER_SAMPLE, 2);
        assert_eq!(Yuv444p12lePackPolicy::STORAGE_BYTES_PER_PIXEL, 6);
        assert_eq!(Yuv444p12lePackPolicy::STORAGE_BITS_PER_SAMPLE, 16);
        assert_eq!(Yuv444p12lePackPolicy::MEANINGFUL_BITS_PER_SAMPLE, 12);
        assert_eq!(Yuv444p12lePackPolicy::LINEAR_SIGNAL_SCALE_NUMERATOR, 1);
        assert_eq!(Yuv444p12lePackPolicy::LINEAR_SIGNAL_SCALE_DENOMINATOR, 2);
        assert_eq!(Yuv444p12lePackPolicy::LINEAR_SIGNAL_SCALE_STOPS, -1);
        assert_eq!(
            Yuv444p12lePackPolicy::LINEAR_SIGNAL_SCALE_FACTOR_F32.to_bits(),
            0.5_f32.to_bits()
        );
        assert_eq!(Yuv444p12lePackPolicy::NOMINAL_LUMA_MIN_CODE, 256);
        assert_eq!(Yuv444p12lePackPolicy::NOMINAL_LUMA_MAX_CODE, 3760);
        assert_eq!(Yuv444p12lePackPolicy::NOMINAL_CHROMA_MIN_CODE, 256);
        assert_eq!(Yuv444p12lePackPolicy::NOMINAL_CHROMA_MAX_CODE, 3840);
        assert_eq!(Yuv444p12lePackPolicy::MIN_CODE, 16);
        assert_eq!(Yuv444p12lePackPolicy::MAX_CODE, 4079);
    }

    #[test]
    fn status_contract_is_four_u32_words() {
        assert_eq!(DIRECT_YUV12_STATUS_BYTE_LEN, 16);
        assert_eq!(
            DIRECT_YUV12_NONFINITE_CAMERA
                | DIRECT_YUV12_NONFINITE_NCL
                | DIRECT_YUV12_NONFINITE_MAPPED,
            7
        );
    }

    #[test]
    fn terminal_contains_exact_mapping_and_no_metadata_or_oetf() {
        for required in [
            "direct_yuv12_checked_affine(256.0, 3504.0, ncl.x)",
            "direct_yuv12_checked_affine(2048.0, 3584.0, ncl.y)",
            "direct_yuv12_checked_affine(2048.0, 3584.0, ncl.z)",
            "let result = dot(row, camera)",
            "floor(clamp(mapped, 16.0, 4079.0) + 0.5)",
            "var camera = demosaic_signed_bayer(x, y)",
            "let pixel0 = direct_yuv12_pixel(x0, gid.y)",
            "let pixel1 = direct_yuv12_pixel(x1, gid.y)",
            "return low | (high << 16u)",
            "all(codes <= vec3<u32>(4095u))",
        ] {
            assert!(DIRECT_YUV12_TERMINAL_WGSL.contains(required));
        }
        for forbidden in [
            "AsShotNeutral",
            "ForwardMatrix",
            "CameraCalibration",
            "oetf",
            "tone_map",
            "gamut_map",
            "& 0xffffu",
        ] {
            assert!(!DIRECT_YUV12_TERMINAL_WGSL.contains(forbidden));
        }
    }

    #[test]
    fn shader_composes_one_signed_demosaic_source() {
        let source = direct_yuv12_shader_source();
        assert!(!source.contains(SIGNED_BAYER_DEMOSAIC_INSERTION_POINT));
        assert_eq!(source.matches("fn demosaic_signed_bayer(").count(), 1);
        assert_eq!(
            source
                .matches("var camera = demosaic_signed_bayer(x, y)")
                .count(),
            1
        );
        assert_eq!(
            source
                .matches("let pixel0 = direct_yuv12_pixel(x0, gid.y)")
                .count(),
            1
        );
        assert_eq!(
            source
                .matches("let pixel1 = direct_yuv12_pixel(x1, gid.y)")
                .count(),
            1
        );
    }

    #[test]
    fn geometry_accepts_required_even_width_and_odd_height_shapes() {
        for dimensions in [
            FrameDimensions {
                width: 2,
                height: 1,
            },
            FrameDimensions {
                width: 2,
                height: 2,
            },
            FrameDimensions {
                width: 4,
                height: 3,
            },
            FrameDimensions {
                width: 30,
                height: 15,
            },
            FrameDimensions {
                width: 32,
                height: 16,
            },
            FrameDimensions {
                width: 34,
                height: 17,
            },
            FrameDimensions {
                width: 1920,
                height: 1080,
            },
            FrameDimensions {
                width: 3840,
                height: 2160,
            },
            FrameDimensions {
                width: 4080,
                height: 3072,
            },
        ] {
            let geometry =
                DirectYuv12Geometry::validate(dimensions, unrestricted_geometry_limits())
                    .expect("required geometry validates");
            let pixels = u64::from(dimensions.width) * u64::from(dimensions.height);
            assert_eq!(geometry.input_binding_bytes, pixels * 4);
            assert_eq!(geometry.plane_byte_len, pixels * 2);
            assert_eq!(geometry.visible_output_bytes, pixels * 6);
        }
    }

    #[test]
    fn geometry_rejects_zero_odd_and_index_overflow_before_dispatch() {
        for dimensions in [
            FrameDimensions {
                width: 0,
                height: 1,
            },
            FrameDimensions {
                width: 2,
                height: 0,
            },
        ] {
            assert_eq!(
                DirectYuv12Geometry::validate(dimensions, unrestricted_geometry_limits())
                    .unwrap_err(),
                DirectYuv12Error::InvalidDimensions { dimensions }
            );
        }
        for dimensions in [
            FrameDimensions {
                width: 1,
                height: 1,
            },
            FrameDimensions {
                width: 3,
                height: 2,
            },
        ] {
            assert_eq!(
                DirectYuv12Geometry::validate(dimensions, unrestricted_geometry_limits())
                    .unwrap_err(),
                DirectYuv12Error::OddVisibleWidth {
                    width: dimensions.width
                }
            );
        }
        let dimensions = FrameDimensions {
            width: u32::MAX - 1,
            height: 2,
        };
        assert_eq!(
            DirectYuv12Geometry::validate(dimensions, unrestricted_geometry_limits()).unwrap_err(),
            DirectYuv12Error::DimensionOverflow { dimensions }
        );
    }

    #[test]
    fn geometry_rejects_binding_buffer_and_dispatch_limits_precisely() {
        let dimensions = FrameDimensions {
            width: 2,
            height: 2,
        };
        assert!(matches!(
            DirectYuv12Geometry::validate(
                dimensions,
                DirectYuv12Limits {
                    max_storage_buffer_binding_size: 15,
                    ..unrestricted_geometry_limits()
                }
            ),
            Err(DirectYuv12Error::StorageBindingTooLarge {
                buffer: "relative-linear f32 Bayer input",
                required_bytes: 16,
                max_binding_bytes: 15,
            })
        ));
        assert_eq!(
            DirectYuv12Geometry::validate(
                dimensions,
                DirectYuv12Limits {
                    max_buffer_size: 23,
                    ..unrestricted_geometry_limits()
                }
            )
            .unwrap_err(),
            DirectYuv12Error::OutputBufferTooLarge {
                required_bytes: 24,
                max_buffer_bytes: 23,
            }
        );
        let dimensions = FrameDimensions {
            width: 34,
            height: 1,
        };
        assert_eq!(
            DirectYuv12Geometry::validate(
                dimensions,
                DirectYuv12Limits {
                    max_compute_workgroups_per_dimension: 1,
                    ..unrestricted_geometry_limits()
                }
            )
            .unwrap_err(),
            DirectYuv12Error::DispatchTooLarge {
                axis: "x",
                required_workgroups: 2,
                max_workgroups: 1,
            }
        );
    }
}
