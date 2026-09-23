//! PIPE-only preparation of relative-linear corrected Bayer samples.
//!
//! This module deliberately ends before the immutable DNG packed-U16 terminal.
//! Its output is one signed f32 camera-domain Bayer scalar per visible pixel.

use std::error::Error;
use std::fmt;
use std::mem::size_of;

use mcraw4vulkan_core::{BayerPattern, FrameDimensions};

use crate::gain_map::motioncam_compatible_gain_q16_from_raw_gain_q;
use crate::{
    CompactSpatialMapFingerprint, FixedPointVignetteInputFacts, GainConversionFingerprint,
    MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH, PIPE_CORRECT_BAYER_F32_WGSL,
    PIPE_CORRECT_BAYER_F32_WORKGROUP_SIZE, VIGNETTE_GAIN_SCALE, VignetteCorrectionError,
    VignetteCorrectionMode, VignetteCorrectionPolicy, VignettePixelDomainFacts,
};

const PIPE_F32_PARAMS_BYTE_LEN: u64 = 128;
const PIPE_F32_STORAGE_BINDINGS: u32 = 3;
const PIPE_F32_UNIFORM_BINDINGS: u32 = 1;
const PIPE_F32_SAMPLE_BYTES: u64 = 4;
const PACKED_U16_WORD_BYTES: u64 = 4;
const COMPACT_GAIN_BYTES: u64 = 8;

/// Spatial policy for the PIPE f32 preparation stage.
///
/// Identity mode disables only the spatial lens gain. It still applies the
/// positional black reference, MotionCam source-to-corrected scale, final-gain
/// quantization, and corrected-white normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipeF32BayerCorrectionMode {
    MotionCamSpatial,
    IdentitySpatialGain,
}

impl PipeF32BayerCorrectionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::MotionCamSpatial => "motioncam-spatial",
            Self::IdentitySpatialGain => "identity-spatial-gain",
        }
    }

    fn shader_tag(self) -> u32 {
        match self {
            Self::IdentitySpatialGain => 0,
            Self::MotionCamSpatial => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipeF32BayerNumericDomain {
    RelativeLinearCorrectedCodeV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipeF32BayerCorrectionFingerprint {
    pub spatial: Option<CompactSpatialMapFingerprint>,
    pub conversion: GainConversionFingerprint,
    pub black_q: [u32; 4],
    pub corrected_white: u16,
    pub mode: PipeF32BayerCorrectionMode,
}

impl PipeF32BayerCorrectionFingerprint {
    /// Stable, versioned bytes for correction-context identity and digests.
    ///
    /// The record is deliberately owned by the typed fingerprint so public
    /// orchestration and private diagnostics cannot drift into different
    /// serializations of the same correction facts.
    pub fn record_bytes(self) -> [u8; 88] {
        const PREFIX: &[u8; 36] = b"PipeF32BayerCorrectionFingerprintV1\0";

        let mut record = [0_u8; 88];
        record[..PREFIX.len()].copy_from_slice(PREFIX);
        record[36] = match self.mode {
            PipeF32BayerCorrectionMode::IdentitySpatialGain => 0,
            PipeF32BayerCorrectionMode::MotionCamSpatial => 1,
        };

        let spatial_words = match self.spatial {
            Some(spatial) => {
                record[37] = 1;
                spatial.words()
            }
            None => [0, 0],
        };
        for (index, word) in spatial_words.into_iter().enumerate() {
            let start = 38 + index * size_of::<u64>();
            record[start..start + size_of::<u64>()].copy_from_slice(&word.to_le_bytes());
        }
        for (index, word) in self.conversion.words().into_iter().enumerate() {
            let start = 54 + index * size_of::<u64>();
            record[start..start + size_of::<u64>()].copy_from_slice(&word.to_le_bytes());
        }
        for (index, black) in self.black_q.into_iter().enumerate() {
            let start = 70 + index * size_of::<u32>();
            record[start..start + size_of::<u32>()].copy_from_slice(&black.to_le_bytes());
        }
        record[86..].copy_from_slice(&self.corrected_white.to_le_bytes());
        record
    }

    /// Resolves the immutable correction identity directly from the supplied
    /// fixed-point facts. This deliberately does not depend on a dispatched
    /// GPU view, so schedulers can retain an expected context before submit
    /// and compare it with the context returned by the encoded stage.
    pub fn from_fixed_facts(
        facts: &FixedPointVignetteInputFacts<'_>,
        mode: PipeF32BayerCorrectionMode,
    ) -> Result<Self, PipeF32BayerError> {
        validate_correction_facts(facts)?;
        let corrected_white = facts.pixel_domain.corrected_white_tag;
        if facts.pixel_domain.source_white_storage == 0 || corrected_white == 0 {
            return Err(PipeF32BayerError::InvalidSourceWhite {
                source_white: facts.pixel_domain.source_white_storage,
                corrected_white,
            });
        }
        let spatial = match mode {
            PipeF32BayerCorrectionMode::MotionCamSpatial => {
                let map = facts
                    .lens_shading_map
                    .as_ref()
                    .ok_or(PipeF32BayerError::MissingSpatialMap)?;
                if map.plane_count() != 4 {
                    return Err(PipeF32BayerError::InvalidSpatialMap {
                        reason: "MotionCam Bayer correction requires four compact map planes",
                    });
                }
                Some(CompactSpatialMapFingerprint::from_fixed_facts(facts)?)
            }
            PipeF32BayerCorrectionMode::IdentitySpatialGain => None,
        };
        Ok(Self {
            spatial,
            conversion: GainConversionFingerprint::from_fixed_facts(facts)?,
            black_q: quantized_motioncam_black(facts)?,
            corrected_white,
            mode,
        })
    }
}

/// Borrowed, tightly packed GPU view produced by [`GpuPipeF32BayerStage`].
#[derive(Debug)]
pub struct GpuCorrectedF32BayerView<'a> {
    buffer: &'a wgpu::Buffer,
    dimensions: FrameDimensions,
    sample_count: usize,
    visible_byte_len: u64,
    bayer_pattern: BayerPattern,
    numeric_domain: PipeF32BayerNumericDomain,
    correction_mode: PipeF32BayerCorrectionMode,
    correction_fingerprint: PipeF32BayerCorrectionFingerprint,
    conservative_max_abs_four_tap_sum: f64,
}

impl<'a> GpuCorrectedF32BayerView<'a> {
    pub fn buffer(&self) -> &'a wgpu::Buffer {
        self.buffer
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

    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub fn visible_byte_len(&self) -> u64 {
        self.visible_byte_len
    }

    pub fn bayer_pattern(&self) -> BayerPattern {
        self.bayer_pattern
    }

    pub fn numeric_domain(&self) -> PipeF32BayerNumericDomain {
        self.numeric_domain
    }

    pub fn correction_mode(&self) -> PipeF32BayerCorrectionMode {
        self.correction_mode
    }

    pub fn correction_fingerprint(&self) -> PipeF32BayerCorrectionFingerprint {
        self.correction_fingerprint
    }

    /// Conservative bound produced by the exact Candidate-4 contract for the
    /// largest signed four-tap demosaic sum. Downstream stages consume this
    /// from the typed view so a caller cannot substitute a weaker bound.
    pub fn conservative_max_abs_four_tap_sum(&self) -> f64 {
        self.conservative_max_abs_four_tap_sum
    }
}

pub struct GpuPipeF32BayerPrepareInput<'a, 'encoder> {
    pub device: &'encoder wgpu::Device,
    pub queue: &'encoder wgpu::Queue,
    pub encoder: &'encoder mut wgpu::CommandEncoder,
    pub input_buffer: &'a wgpu::Buffer,
    /// Physical packed-word bytes available to the shader. An odd final sample
    /// occupies the low half of a zero-padded final u32 word.
    pub input_buffer_bytes: u64,
    pub facts: &'a FixedPointVignetteInputFacts<'a>,
    pub correction_mode: PipeF32BayerCorrectionMode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuPipeF32BayerDispatchStats {
    pub pixel_count: u64,
    pub input_binding_bytes: u64,
    pub visible_output_bytes: u64,
    pub allocated_output_bytes: u64,
    pub compact_gain_binding_bytes: u64,
    pub dispatch_workgroups: u32,
    pub workgroup_size: u32,
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
    pub output_buffer_allocated: bool,
    pub output_allocation_count: u64,
    pub output_reuse_count: u64,
    pub bind_group_allocation_count: u64,
    pub bind_group_reuse_count: u64,
    pub spatial_map_uploaded: bool,
    pub spatial_map_reused: bool,
    pub conservative_max_abs_sample: f64,
    pub conservative_max_abs_four_tap_sum: f64,
}

#[derive(Debug)]
pub struct GpuPipeF32BayerDispatch<'a> {
    pub view: GpuCorrectedF32BayerView<'a>,
    pub stats: GpuPipeF32BayerDispatchStats,
}

struct GpuSizedBuffer {
    buffer: wgpu::Buffer,
    size: u64,
}

struct GpuCompactSpatialMap {
    fingerprint: CompactSpatialMapFingerprint,
    buffer: wgpu::Buffer,
    allocated_bytes: u64,
}

struct GpuPipeF32BindGroupCache {
    input_buffer: wgpu::Buffer,
    gain_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    input_binding_bytes: u64,
    gain_binding_bytes: u64,
    output_binding_bytes: u64,
    bind_group: wgpu::BindGroup,
}

/// Reusable PIPE-only correction stage.
///
/// Callers must encode the signed demosaic consumer before releasing the
/// returned view. The application currently submits each frame before reusing
/// a stage instance; that ordering also protects the reusable uniform/output
/// resources from being paired with later frame facts.
pub struct GpuPipeF32BayerStage {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: GpuSizedBuffer,
    identity_gain_buffer: GpuSizedBuffer,
    spatial_map: Option<GpuCompactSpatialMap>,
    output_buffer: Option<GpuSizedBuffer>,
    output_allocation_count: u64,
    output_reuse_count: u64,
    bind_group_cache: Option<GpuPipeF32BindGroupCache>,
    bind_group_allocation_count: u64,
    bind_group_reuse_count: u64,
}

impl GpuPipeF32BayerStage {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, PipeF32BayerError> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mcraw4vulkan PIPE relative-linear f32 Bayer shader"),
            source: wgpu::ShaderSource::Wgsl(PIPE_CORRECT_BAYER_F32_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mcraw4vulkan PIPE relative-linear f32 Bayer pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let params_buffer = GpuSizedBuffer {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mcraw4vulkan PIPE f32 Bayer params"),
                size: PIPE_F32_PARAMS_BYTE_LEN,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            size: PIPE_F32_PARAMS_BYTE_LEN,
        };
        let identity_gain_buffer = GpuSizedBuffer {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mcraw4vulkan PIPE identity spatial gain"),
                size: COMPACT_GAIN_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            size: COMPACT_GAIN_BYTES,
        };
        queue.write_buffer(
            &identity_gain_buffer.buffer,
            0,
            &u32_words_to_le_bytes(&[u32::try_from(VIGNETTE_GAIN_SCALE).unwrap_or(65_536), 0]),
        );

        Ok(Self {
            pipeline,
            bind_group_layout,
            params_buffer,
            identity_gain_buffer,
            spatial_map: None,
            output_buffer: None,
            output_allocation_count: 0,
            output_reuse_count: 0,
            bind_group_cache: None,
            bind_group_allocation_count: 0,
            bind_group_reuse_count: 0,
        })
    }

    pub fn output_allocation_count(&self) -> u64 {
        self.output_allocation_count
    }

    pub fn output_reuse_count(&self) -> u64 {
        self.output_reuse_count
    }

    pub fn allocated_output_bytes(&self) -> u64 {
        self.output_buffer.as_ref().map_or(0, |buffer| buffer.size)
    }

    pub fn prepare_pipe_f32_bayer<'a>(
        &'a mut self,
        input: GpuPipeF32BayerPrepareInput<'_, '_>,
    ) -> Result<GpuPipeF32BayerDispatch<'a>, PipeF32BayerError> {
        let contract =
            PipeF32DispatchContract::from_input(input.device, input.facts, input.correction_mode)?;
        let allocated_input_bytes = input.input_buffer.size();
        if input.input_buffer_bytes > allocated_input_bytes {
            return Err(PipeF32BayerError::InputBufferLengthExceedsAllocation {
                declared_bytes: input.input_buffer_bytes,
                allocated_bytes: allocated_input_bytes,
            });
        }
        if input.input_buffer_bytes < contract.input_binding_bytes {
            return Err(PipeF32BayerError::InputBufferTooSmall {
                required_bytes: contract.input_binding_bytes,
                actual_bytes: input.input_buffer_bytes,
            });
        }

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

        let (spatial_map_uploaded, spatial_map_reused) = match input.correction_mode {
            PipeF32BayerCorrectionMode::MotionCamSpatial => {
                let fingerprint = contract
                    .fingerprint
                    .spatial
                    .expect("spatial mode contract contains a map fingerprint");
                ensure_spatial_map(
                    input.device,
                    input.queue,
                    &mut self.spatial_map,
                    input.facts,
                    fingerprint,
                    contract.compact_gain_binding_bytes,
                )?
            }
            PipeF32BayerCorrectionMode::IdentitySpatialGain => (false, true),
        };

        let params = contract.params.to_le_bytes();
        input
            .queue
            .write_buffer(&self.params_buffer.buffer, 0, &params);

        let output = self
            .output_buffer
            .as_ref()
            .expect("output buffer was ensured before binding");
        let gain_buffer = match input.correction_mode {
            PipeF32BayerCorrectionMode::MotionCamSpatial => {
                &self
                    .spatial_map
                    .as_ref()
                    .expect("spatial map was ensured before binding")
                    .buffer
            }
            PipeF32BayerCorrectionMode::IdentitySpatialGain => &self.identity_gain_buffer.buffer,
        };
        let cache_matches = self.bind_group_cache.as_ref().is_some_and(|cache| {
            cache.input_buffer == *input.input_buffer
                && cache.gain_buffer == *gain_buffer
                && cache.output_buffer == output.buffer
                && cache.input_binding_bytes == contract.input_binding_bytes
                && cache.gain_binding_bytes == contract.compact_gain_binding_bytes
                && cache.output_binding_bytes == contract.visible_output_bytes
        });
        if cache_matches {
            self.bind_group_reuse_count = self.bind_group_reuse_count.saturating_add(1);
        } else {
            let bind_group = input.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mcraw4vulkan PIPE f32 Bayer bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: input.input_buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(contract.input_binding_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: gain_buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(contract.compact_gain_binding_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &output.buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(contract.visible_output_bytes),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.params_buffer.buffer.as_entire_binding(),
                    },
                ],
            });
            self.bind_group_cache = Some(GpuPipeF32BindGroupCache {
                input_buffer: input.input_buffer.clone(),
                gain_buffer: gain_buffer.clone(),
                output_buffer: output.buffer.clone(),
                input_binding_bytes: contract.input_binding_bytes,
                gain_binding_bytes: contract.compact_gain_binding_bytes,
                output_binding_bytes: contract.visible_output_bytes,
                bind_group,
            });
            self.bind_group_allocation_count = self.bind_group_allocation_count.saturating_add(1);
        }
        let bind_group = &self
            .bind_group_cache
            .as_ref()
            .expect("PIPE f32 Bayer bind group cache was populated")
            .bind_group;

        {
            let mut pass = input
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mcraw4vulkan PIPE f32 Bayer preparation pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(contract.dispatch_grid.x, contract.dispatch_grid.y, 1);
        }

        let stats = GpuPipeF32BayerDispatchStats {
            pixel_count: u64::try_from(contract.pixel_count).unwrap_or(u64::MAX),
            input_binding_bytes: contract.input_binding_bytes,
            visible_output_bytes: contract.visible_output_bytes,
            allocated_output_bytes: output.size,
            compact_gain_binding_bytes: contract.compact_gain_binding_bytes,
            dispatch_workgroups: contract.dispatch_grid.total,
            workgroup_size: PIPE_CORRECT_BAYER_F32_WORKGROUP_SIZE,
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
            spatial_map_uploaded,
            spatial_map_reused,
            conservative_max_abs_sample: contract.conservative_max_abs_sample,
            conservative_max_abs_four_tap_sum: contract.conservative_max_abs_four_tap_sum,
        };
        let view = GpuCorrectedF32BayerView {
            buffer: &output.buffer,
            dimensions: input.facts.frame_dimensions,
            sample_count: contract.pixel_count,
            visible_byte_len: contract.visible_output_bytes,
            bayer_pattern: input.facts.bayer_pattern,
            numeric_domain: PipeF32BayerNumericDomain::RelativeLinearCorrectedCodeV1,
            correction_mode: input.correction_mode,
            correction_fingerprint: contract.fingerprint,
            conservative_max_abs_four_tap_sum: contract.conservative_max_abs_four_tap_sum,
        };

        Ok(GpuPipeF32BayerDispatch { view, stats })
    }
}

#[derive(Debug, Clone, Copy)]
struct PipeF32Limits {
    max_storage_buffer_binding_size: u64,
    max_buffer_size: u64,
    max_compute_workgroups_per_dimension: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PipeF32DispatchGrid {
    x: u32,
    y: u32,
    total: u32,
}

#[derive(Debug, Clone, Copy)]
struct PipeF32Params {
    words: [u32; 32],
}

impl PipeF32Params {
    fn to_le_bytes(self) -> [u8; PIPE_F32_PARAMS_BYTE_LEN as usize] {
        let mut bytes = [0_u8; PIPE_F32_PARAMS_BYTE_LEN as usize];
        for (index, word) in self.words.into_iter().enumerate() {
            let start = index * std::mem::size_of::<u32>();
            bytes[start..start + std::mem::size_of::<u32>()].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

struct PipeF32DispatchContract {
    pixel_count: usize,
    input_binding_bytes: u64,
    visible_output_bytes: u64,
    compact_gain_binding_bytes: u64,
    dispatch_grid: PipeF32DispatchGrid,
    params: PipeF32Params,
    fingerprint: PipeF32BayerCorrectionFingerprint,
    limits: PipeF32Limits,
    conservative_max_abs_sample: f64,
    conservative_max_abs_four_tap_sum: f64,
}

impl PipeF32DispatchContract {
    fn from_input(
        device: &wgpu::Device,
        facts: &FixedPointVignetteInputFacts<'_>,
        mode: PipeF32BayerCorrectionMode,
    ) -> Result<Self, PipeF32BayerError> {
        validate_correction_facts(facts)?;
        let dimensions = facts.frame_dimensions;
        if dimensions.width == 0 || dimensions.height == 0 {
            return Err(PipeF32BayerError::InvalidDimensions { dimensions });
        }
        let pixel_count = dimensions
            .pixel_count()
            .ok_or(PipeF32BayerError::DimensionOverflow { dimensions })?;
        let pixel_count_u32 = u32::try_from(pixel_count)
            .map_err(|_| PipeF32BayerError::DimensionOverflow { dimensions })?;
        let packed_word_count = pixel_count
            .checked_add(1)
            .and_then(|value| value.checked_div(2))
            .ok_or(PipeF32BayerError::DimensionOverflow { dimensions })?;
        let packed_word_count_u32 = u32::try_from(packed_word_count)
            .map_err(|_| PipeF32BayerError::DimensionOverflow { dimensions })?;
        let input_binding_bytes = u64::try_from(packed_word_count)
            .ok()
            .and_then(|count| count.checked_mul(PACKED_U16_WORD_BYTES))
            .ok_or(PipeF32BayerError::DimensionOverflow { dimensions })?;
        let visible_output_bytes = u64::try_from(pixel_count)
            .ok()
            .and_then(|count| count.checked_mul(PIPE_F32_SAMPLE_BYTES))
            .ok_or(PipeF32BayerError::DimensionOverflow { dimensions })?;
        let corrected_white = facts.pixel_domain.corrected_white_tag;
        if facts.pixel_domain.source_white_storage == 0 || corrected_white == 0 {
            return Err(PipeF32BayerError::InvalidSourceWhite {
                source_white: facts.pixel_domain.source_white_storage,
                corrected_white,
            });
        }
        let source_black_average = facts
            .pixel_domain
            .source_black_storage
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>()
            * 0.25;
        if f64::from(facts.pixel_domain.source_white_storage) - source_black_average < 1.0 {
            return Err(PipeF32BayerError::InvalidSourceRange {
                source_white: facts.pixel_domain.source_white_storage,
                source_black_average,
            });
        }

        validate_fixed_axis_domain(dimensions, facts, mode)?;
        let black_q = quantized_motioncam_black(facts)?;
        let (scale_num, scale_shift) = f32_power2_rational(
            facts.pixel_domain.source_to_corrected_scale,
        )
        .ok_or(PipeF32BayerError::InvalidGainConversion {
            reason: "source-to-corrected scale is not a positive finite f32 rational",
        })?;
        let (strength_num, strength_shift) = f32_power2_rational(
            MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH,
        )
        .ok_or(PipeF32BayerError::InvalidGainConversion {
            reason: "MotionCam gain strength is not a positive finite f32 rational",
        })?;

        let (source_map_width, source_map_height, source_plane_count, map_word_count, map_bytes) =
            match mode {
                PipeF32BayerCorrectionMode::MotionCamSpatial => {
                    let map = facts
                        .lens_shading_map
                        .as_ref()
                        .ok_or(PipeF32BayerError::MissingSpatialMap)?;
                    if map.plane_count() != 4 {
                        return Err(PipeF32BayerError::InvalidSpatialMap {
                            reason: "MotionCam Bayer correction requires four compact map planes",
                        });
                    }
                    let width = u32::try_from(map.width()).map_err(|_| {
                        PipeF32BayerError::InvalidSpatialMap {
                            reason: "compact map width does not fit u32",
                        }
                    })?;
                    let height = u32::try_from(map.height()).map_err(|_| {
                        PipeF32BayerError::InvalidSpatialMap {
                            reason: "compact map height does not fit u32",
                        }
                    })?;
                    let planes = u32::try_from(map.plane_count()).map_err(|_| {
                        PipeF32BayerError::InvalidSpatialMap {
                            reason: "compact map plane count does not fit u32",
                        }
                    })?;
                    let words = map
                        .width()
                        .checked_mul(map.height())
                        .and_then(|value| value.checked_mul(map.plane_count()))
                        .ok_or(PipeF32BayerError::InvalidSpatialMap {
                            reason: "compact map sample count overflow",
                        })?;
                    let bytes = u64::try_from(words)
                        .ok()
                        .and_then(|count| count.checked_mul(COMPACT_GAIN_BYTES))
                        .ok_or(PipeF32BayerError::InvalidSpatialMap {
                            reason: "compact map byte count overflow",
                        })?;
                    (
                        width,
                        height,
                        planes,
                        u32::try_from(words).map_err(|_| PipeF32BayerError::InvalidSpatialMap {
                            reason: "compact map sample count does not fit u32",
                        })?,
                        bytes,
                    )
                }
                PipeF32BayerCorrectionMode::IdentitySpatialGain => (1, 1, 1, 1, COMPACT_GAIN_BYTES),
            };
        let fingerprint = PipeF32BayerCorrectionFingerprint::from_fixed_facts(facts, mode)?;

        let raw_gain_q = maximum_raw_gain_q(facts, mode)?;
        let final_gain_q = motioncam_compatible_gain_q16_from_raw_gain_q(raw_gain_q, facts, 0, 0)?;
        let final_gain_q_u32 =
            u32::try_from(final_gain_q).map_err(|_| PipeF32BayerError::InvalidGainConversion {
                reason: "maximum final gain does not fit u32",
            })?;
        let (min_black_f, max_black_f) = black_q.into_iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(minimum, maximum), value| {
                let black = f64::from(value) / 65_536.0;
                (minimum.min(black), maximum.max(black))
            },
        );
        let max_code_delta = max_black_f.max((65_535.0 - min_black_f).abs());
        let conservative_max_abs_sample =
            max_code_delta * (f64::from(final_gain_q_u32) / 65_536.0) / f64::from(corrected_white);
        let conservative_max_abs_four_tap_sum = conservative_max_abs_sample * 4.0;
        if !conservative_max_abs_four_tap_sum.is_finite()
            || conservative_max_abs_four_tap_sum > f64::from(f32::MAX) * 0.5
        {
            return Err(PipeF32BayerError::UnsafeDemosaicMagnitude {
                max_abs_sample: conservative_max_abs_sample,
                max_abs_four_tap_sum: conservative_max_abs_four_tap_sum,
                allowed_four_tap_sum: f64::from(f32::MAX) * 0.5,
            });
        }

        let limits_raw = device.limits();
        if limits_raw.max_bind_groups < 1
            || limits_raw.max_storage_buffers_per_shader_stage < PIPE_F32_STORAGE_BINDINGS
            || limits_raw.max_uniform_buffers_per_shader_stage < PIPE_F32_UNIFORM_BINDINGS
        {
            return Err(PipeF32BayerError::InsufficientBindingLimits {
                max_bind_groups: limits_raw.max_bind_groups,
                max_storage_buffers_per_shader_stage: limits_raw
                    .max_storage_buffers_per_shader_stage,
                max_uniform_buffers_per_shader_stage: limits_raw
                    .max_uniform_buffers_per_shader_stage,
            });
        }
        let limits = PipeF32Limits {
            max_storage_buffer_binding_size: u64::from(limits_raw.max_storage_buffer_binding_size),
            max_buffer_size: limits_raw.max_buffer_size,
            max_compute_workgroups_per_dimension: limits_raw.max_compute_workgroups_per_dimension,
        };
        let dispatch_grid = validate_resource_contract(
            pixel_count,
            input_binding_bytes,
            visible_output_bytes,
            map_bytes,
            limits,
        )?;

        let mut words = [0_u32; 32];
        words[0] = pixel_count_u32;
        words[1] = packed_word_count_u32;
        words[2] = dimensions.width;
        words[3] = dimensions.height;
        words[4..8].copy_from_slice(&black_q);
        words[8] = u32::from(corrected_white);
        words[9] = mode.shader_tag();
        words[13] = bayer_pattern_tag(facts.bayer_pattern);
        words[14] = source_map_width;
        words[15] = source_map_height;
        words[16] = source_plane_count;
        words[17] = 0;
        words[18] = scale_num;
        words[19] = map_word_count;
        words[20] = scale_shift;
        words[21] = strength_num;
        words[22] = strength_shift;

        Ok(Self {
            pixel_count,
            input_binding_bytes,
            visible_output_bytes,
            compact_gain_binding_bytes: map_bytes,
            dispatch_grid,
            params: PipeF32Params { words },
            fingerprint,
            limits,
            conservative_max_abs_sample,
            conservative_max_abs_four_tap_sum,
        })
    }
}

fn validate_correction_facts(
    facts: &FixedPointVignetteInputFacts<'_>,
) -> Result<(), PipeF32BayerError> {
    if facts.correction_policy != VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1 {
        return Err(PipeF32BayerError::UnsupportedCorrectionPolicy {
            policy: facts.correction_policy,
        });
    }
    if facts.mode != VignetteCorrectionMode::Enabled {
        return Err(PipeF32BayerError::UnsupportedInputMode { mode: facts.mode });
    }
    match facts.coordinate_mapping {
        crate::VignetteCoordinateMapping::VisibleFrame => {}
    }

    let expected_pixel_domain = VignettePixelDomainFacts::from_source_levels(
        facts.input_black_level,
        facts.output_white_level,
    );
    if facts.pixel_domain != expected_pixel_domain {
        return Err(PipeF32BayerError::InconsistentPixelDomain {
            expected: expected_pixel_domain,
            actual: facts.pixel_domain,
        });
    }
    for (index, (&actual, &black)) in facts
        .input_black_level_q
        .iter()
        .zip(facts.input_black_level.iter())
        .enumerate()
    {
        let expected = (f64::from(black) * VIGNETTE_GAIN_SCALE as f64).round() as i64;
        if actual != expected {
            return Err(PipeF32BayerError::InconsistentFixedBlack {
                index,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn validate_resource_contract(
    pixel_count: usize,
    input_binding_bytes: u64,
    visible_output_bytes: u64,
    compact_gain_binding_bytes: u64,
    limits: PipeF32Limits,
) -> Result<PipeF32DispatchGrid, PipeF32BayerError> {
    for (buffer, required_bytes) in [
        ("packed-U16 input", input_binding_bytes),
        ("relative-linear f32 output", visible_output_bytes),
        ("compact gain map", compact_gain_binding_bytes),
    ] {
        if required_bytes > limits.max_storage_buffer_binding_size {
            return Err(PipeF32BayerError::StorageBindingTooLarge {
                buffer,
                required_bytes,
                max_binding_bytes: limits.max_storage_buffer_binding_size,
            });
        }
    }
    if visible_output_bytes > limits.max_buffer_size {
        return Err(PipeF32BayerError::OutputBufferTooLarge {
            required_bytes: visible_output_bytes,
            max_buffer_bytes: limits.max_buffer_size,
        });
    }
    let total_workgroups = u64::try_from(pixel_count)
        .unwrap_or(u64::MAX)
        .div_ceil(u64::from(PIPE_CORRECT_BAYER_F32_WORKGROUP_SIZE));
    let max_per_dimension = u64::from(limits.max_compute_workgroups_per_dimension);
    let maximum_grid_workgroups = max_per_dimension.saturating_mul(max_per_dimension);
    if total_workgroups == 0 || max_per_dimension == 0 || total_workgroups > maximum_grid_workgroups
    {
        return Err(PipeF32BayerError::DispatchTooLarge {
            required_workgroups: total_workgroups,
            max_workgroups: limits.max_compute_workgroups_per_dimension,
        });
    }
    let y = if total_workgroups <= max_per_dimension {
        1
    } else {
        total_workgroups.div_ceil(max_per_dimension)
    };
    let x = total_workgroups.div_ceil(y);
    if x > max_per_dimension || y > max_per_dimension {
        return Err(PipeF32BayerError::DispatchTooLarge {
            required_workgroups: total_workgroups,
            max_workgroups: limits.max_compute_workgroups_per_dimension,
        });
    }
    let convert = |value| {
        u32::try_from(value).map_err(|_| PipeF32BayerError::DispatchTooLarge {
            required_workgroups: total_workgroups,
            max_workgroups: limits.max_compute_workgroups_per_dimension,
        })
    };
    Ok(PipeF32DispatchGrid {
        x: convert(x)?,
        y: convert(y)?,
        total: convert(total_workgroups)?,
    })
}

fn ensure_output_buffer(
    device: &wgpu::Device,
    slot: &mut Option<GpuSizedBuffer>,
    required_bytes: u64,
) -> bool {
    if slot
        .as_ref()
        .is_some_and(|existing| existing.size >= required_bytes)
    {
        return false;
    }
    *slot = Some(GpuSizedBuffer {
        buffer: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcraw4vulkan PIPE relative-linear f32 Bayer output"),
            size: required_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        size: required_bytes,
    });
    true
}

fn ensure_spatial_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    slot: &mut Option<GpuCompactSpatialMap>,
    facts: &FixedPointVignetteInputFacts<'_>,
    fingerprint: CompactSpatialMapFingerprint,
    required_bytes: u64,
) -> Result<(bool, bool), PipeF32BayerError> {
    if slot
        .as_ref()
        .is_some_and(|existing| existing.fingerprint == fingerprint)
    {
        return Ok((false, true));
    }
    let bytes = compact_gain_bytes(facts)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != required_bytes {
        return Err(PipeF32BayerError::InvalidSpatialMap {
            reason: "serialized compact map size does not match validated binding size",
        });
    }
    let existing = slot.take();
    let (buffer, allocated_bytes) = match existing {
        Some(existing) if existing.allocated_bytes >= required_bytes => {
            (existing.buffer, existing.allocated_bytes)
        }
        _ => (
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mcraw4vulkan PIPE compact fixed lens map"),
                size: required_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            required_bytes,
        ),
    };
    queue.write_buffer(&buffer, 0, &bytes);
    *slot = Some(GpuCompactSpatialMap {
        fingerprint,
        buffer,
        allocated_bytes,
    });
    Ok((true, false))
}

fn compact_gain_bytes(
    facts: &FixedPointVignetteInputFacts<'_>,
) -> Result<Vec<u8>, PipeF32BayerError> {
    let map = facts
        .lens_shading_map
        .as_ref()
        .ok_or(PipeF32BayerError::MissingSpatialMap)?;
    let mut words = Vec::new();
    for plane_index in 0..map.plane_count() {
        let plane = map.plane_q(plane_index)?;
        words.reserve(plane.len().saturating_mul(2));
        for value in plane {
            let value =
                u64::try_from(*value).map_err(|_| PipeF32BayerError::InvalidSpatialMap {
                    reason: "compact fixed gain is negative",
                })?;
            words.push(value as u32);
            words.push((value >> 32) as u32);
        }
    }
    Ok(u32_words_to_le_bytes(&words))
}

fn maximum_raw_gain_q(
    facts: &FixedPointVignetteInputFacts<'_>,
    mode: PipeF32BayerCorrectionMode,
) -> Result<i64, PipeF32BayerError> {
    if mode == PipeF32BayerCorrectionMode::IdentitySpatialGain {
        return Ok(VIGNETTE_GAIN_SCALE);
    }
    let map = facts
        .lens_shading_map
        .as_ref()
        .ok_or(PipeF32BayerError::MissingSpatialMap)?;
    let mut maximum = 0_i64;
    for plane_index in 0..map.plane_count() {
        for value in map.plane_q(plane_index)? {
            maximum = maximum.max(*value);
        }
    }
    Ok(maximum)
}

fn quantized_motioncam_black(
    facts: &FixedPointVignetteInputFacts<'_>,
) -> Result<[u32; 4], PipeF32BayerError> {
    let mut result = [0_u32; 4];
    for (index, value) in facts
        .pixel_domain
        .source_black_storage
        .iter()
        .copied()
        .enumerate()
    {
        let quantized = f64::from(value) * VIGNETTE_GAIN_SCALE as f64;
        if !quantized.is_finite() || quantized < 0.0 || quantized > f64::from(u32::MAX) {
            return Err(PipeF32BayerError::BlackLevelOverflow { index, value });
        }
        result[index] = quantized.round() as u32;
    }
    Ok(result)
}

fn validate_fixed_axis_domain(
    dimensions: FrameDimensions,
    facts: &FixedPointVignetteInputFacts<'_>,
    mode: PipeF32BayerCorrectionMode,
) -> Result<(), PipeF32BayerError> {
    if mode == PipeF32BayerCorrectionMode::IdentitySpatialGain {
        return Ok(());
    }
    let map = facts
        .lens_shading_map
        .as_ref()
        .ok_or(PipeF32BayerError::MissingSpatialMap)?;
    for (frame_dimension, map_dimension, axis) in [
        (dimensions.width, map.width(), "x"),
        (dimensions.height, map.height(), "y"),
    ] {
        let map_dimension_u64 = u64::try_from(map_dimension).map_err(|_| {
            PipeF32BayerError::FixedAxisDomainTooLarge {
                axis,
                frame_dimension,
                map_dimension,
            }
        })?;
        let frame_last = u64::from(frame_dimension.saturating_sub(1));
        let map_last = map_dimension_u64.saturating_sub(1);
        if frame_last
            .checked_mul(map_last)
            .is_none_or(|value| value > u64::from(u32::MAX))
            || frame_last
                .checked_mul(65_536)
                .is_none_or(|value| value > u64::from(u32::MAX))
        {
            return Err(PipeF32BayerError::FixedAxisDomainTooLarge {
                axis,
                frame_dimension,
                map_dimension,
            });
        }
    }
    Ok(())
}

fn bayer_pattern_tag(pattern: BayerPattern) -> u32 {
    match pattern {
        BayerPattern::Rggb => 0,
        BayerPattern::Bggr => 1,
        BayerPattern::Grbg => 2,
        BayerPattern::Gbrg => 3,
    }
}

fn f32_power2_rational(value: f32) -> Option<(u32, u32)> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let bits = value.to_bits();
    let exponent_bits = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;
    if exponent_bits == 0 {
        return (fraction != 0).then_some((fraction, 149));
    }
    let mantissa = (1_u32 << 23) | fraction;
    let exponent = i32::try_from(exponent_bits).ok()? - 127;
    let shift = 23_i32.checked_sub(exponent)?;
    (shift >= 0).then_some((mantissa, u32::try_from(shift).ok()?))
}

fn u32_words_to_le_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipeF32BayerError {
    Vignette(VignetteCorrectionError),
    UnsupportedCorrectionPolicy {
        policy: VignetteCorrectionPolicy,
    },
    UnsupportedInputMode {
        mode: VignetteCorrectionMode,
    },
    InconsistentPixelDomain {
        expected: VignettePixelDomainFacts,
        actual: VignettePixelDomainFacts,
    },
    InconsistentFixedBlack {
        index: usize,
        expected: i64,
        actual: i64,
    },
    InvalidDimensions {
        dimensions: FrameDimensions,
    },
    DimensionOverflow {
        dimensions: FrameDimensions,
    },
    InputBufferTooSmall {
        required_bytes: u64,
        actual_bytes: u64,
    },
    InputBufferLengthExceedsAllocation {
        declared_bytes: u64,
        allocated_bytes: u64,
    },
    InvalidSourceWhite {
        source_white: u16,
        corrected_white: u16,
    },
    InvalidSourceRange {
        source_white: u16,
        source_black_average: f64,
    },
    MissingSpatialMap,
    InvalidSpatialMap {
        reason: &'static str,
    },
    BlackLevelOverflow {
        index: usize,
        value: f32,
    },
    InvalidGainConversion {
        reason: &'static str,
    },
    FixedAxisDomainTooLarge {
        axis: &'static str,
        frame_dimension: u32,
        map_dimension: usize,
    },
    UnsafeDemosaicMagnitude {
        max_abs_sample: f64,
        max_abs_four_tap_sum: f64,
        allowed_four_tap_sum: f64,
    },
    InsufficientBindingLimits {
        max_bind_groups: u32,
        max_storage_buffers_per_shader_stage: u32,
        max_uniform_buffers_per_shader_stage: u32,
    },
    StorageBindingTooLarge {
        buffer: &'static str,
        required_bytes: u64,
        max_binding_bytes: u64,
    },
    OutputBufferTooLarge {
        required_bytes: u64,
        max_buffer_bytes: u64,
    },
    DispatchTooLarge {
        required_workgroups: u64,
        max_workgroups: u32,
    },
}

impl From<VignetteCorrectionError> for PipeF32BayerError {
    fn from(error: VignetteCorrectionError) -> Self {
        Self::Vignette(error)
    }
}

impl fmt::Display for PipeF32BayerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vignette(error) => write!(formatter, "{error}"),
            Self::UnsupportedCorrectionPolicy { policy } => write!(
                formatter,
                "PIPE f32 Bayer preparation requires MotionCamCompatiblePixelDomainV1, got {policy:?}"
            ),
            Self::UnsupportedInputMode { mode } => write!(
                formatter,
                "PIPE f32 Bayer preparation requires enabled correction facts; {mode:?} is the immutable DNG/pass-through mode, not identity spatial gain"
            ),
            Self::InconsistentPixelDomain { expected, actual } => write!(
                formatter,
                "PIPE f32 Bayer correction facts have an inconsistent source pixel domain: expected {expected:?}, got {actual:?}"
            ),
            Self::InconsistentFixedBlack {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "PIPE f32 Bayer fixed black fact {index} is inconsistent: expected Q16.16 {expected}, got {actual}"
            ),
            Self::InvalidDimensions { dimensions } => write!(
                formatter,
                "PIPE f32 Bayer dimensions {}x{} must be positive",
                dimensions.width, dimensions.height
            ),
            Self::DimensionOverflow { dimensions } => write!(
                formatter,
                "PIPE f32 Bayer dimensions {}x{} overflow checked buffer/index arithmetic",
                dimensions.width, dimensions.height
            ),
            Self::InputBufferTooSmall {
                required_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "packed-U16 input requires {required_bytes} bytes including odd-lane padding, got {actual_bytes}"
            ),
            Self::InputBufferLengthExceedsAllocation {
                declared_bytes,
                allocated_bytes,
            } => write!(
                formatter,
                "packed-U16 input declares {declared_bytes} bytes, exceeding its {allocated_bytes}-byte GPU allocation"
            ),
            Self::InvalidSourceWhite {
                source_white,
                corrected_white,
            } => write!(
                formatter,
                "source/corrected white must be nonzero, got source={source_white} CW={corrected_white}"
            ),
            Self::InvalidSourceRange {
                source_white,
                source_black_average,
            } => write!(
                formatter,
                "source white {source_white} must exceed average black {source_black_average} by at least one code"
            ),
            Self::MissingSpatialMap => formatter.write_str(
                "MotionCam spatial PIPE correction requires a prepared compact lens map",
            ),
            Self::InvalidSpatialMap { reason } => {
                write!(formatter, "invalid PIPE compact spatial map: {reason}")
            }
            Self::BlackLevelOverflow { index, value } => write!(
                formatter,
                "PIPE black level {index} cannot be represented as unsigned Q16.16: {value}"
            ),
            Self::InvalidGainConversion { reason } => {
                write!(
                    formatter,
                    "invalid PIPE MotionCam gain conversion: {reason}"
                )
            }
            Self::FixedAxisDomainTooLarge {
                axis,
                frame_dimension,
                map_dimension,
            } => write!(
                formatter,
                "PIPE fixed-point {axis}-axis interpolation exceeds the WGSL u32 domain: frame={frame_dimension}, map={map_dimension}"
            ),
            Self::UnsafeDemosaicMagnitude {
                max_abs_sample,
                max_abs_four_tap_sum,
                allowed_four_tap_sum,
            } => write!(
                formatter,
                "PIPE correction bound is unsafe for signed bilinear sums: sample={max_abs_sample}, four-tap={max_abs_four_tap_sum}, allowed={allowed_four_tap_sum}"
            ),
            Self::InsufficientBindingLimits {
                max_bind_groups,
                max_storage_buffers_per_shader_stage,
                max_uniform_buffers_per_shader_stage,
            } => write!(
                formatter,
                "adapter cannot bind PIPE f32 Bayer stage: bind_groups={max_bind_groups}, storage_buffers={max_storage_buffers_per_shader_stage}, uniform_buffers={max_uniform_buffers_per_shader_stage}"
            ),
            Self::StorageBindingTooLarge {
                buffer,
                required_bytes,
                max_binding_bytes,
            } => write!(
                formatter,
                "{buffer} requires {required_bytes} bytes, exceeding max_storage_buffer_binding_size={max_binding_bytes}"
            ),
            Self::OutputBufferTooLarge {
                required_bytes,
                max_buffer_bytes,
            } => write!(
                formatter,
                "PIPE f32 Bayer output requires {required_bytes} bytes, exceeding max_buffer_size={max_buffer_bytes}"
            ),
            Self::DispatchTooLarge {
                required_workgroups,
                max_workgroups,
            } => write!(
                formatter,
                "PIPE f32 Bayer dispatch requires {required_workgroups} workgroups, exceeding a 2D grid with per-dimension limit {max_workgroups}"
            ),
        }
    }
}

impl Error for PipeF32BayerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Vignette(error) => Some(error),
            _ => None,
        }
    }
}
