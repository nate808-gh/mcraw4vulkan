use std::time::{Duration, Instant};

use mcraw4vulkan_core::{BayerPattern, FrameDimensions};

use crate::{
    BAYER_CFA_PLANE_COUNT, CompactSpatialMapFingerprint, FixedPointVignetteInputFacts,
    GainConversionFingerprint, LensShadingMap, MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH,
    PreparedFixedLensShadingMap, PreparedFullResolutionFixedGainMap, PreparedLensShadingMap,
    VIGNETTE_CORRECT_PACKED_U16_WGSL, VIGNETTE_CORRECT_WORKGROUP_SIZE, VIGNETTE_GAIN_SCALE,
    VIGNETTE_PACKED_U16_BYTES_PER_WORD, VIGNETTE_PACKED_U16_SAMPLES_PER_WORD,
    VignetteCorrectionError, VignetteCorrectionMode, VignetteCorrectionOptions,
    VignetteCorrectionPolicy, gain_map::motioncam_compatible_gain_q16_from_raw_gain_q,
};

// A generic binding pairs a borrowed GPU resource with its lens-map dimensions
// without assuming a concrete native handle type.
#[derive(Debug)]
pub struct GpuLensShadingMapBinding<'a, LensShadingResource> {
    pub resource: &'a LensShadingResource,
    pub width: u32,
    pub height: u32,
    pub plane_count: usize,
}

impl<'a, LensShadingResource> GpuLensShadingMapBinding<'a, LensShadingResource> {
    pub fn from_prepared_map_resource(
        resource: &'a LensShadingResource,
        map: &PreparedLensShadingMap<'_>,
    ) -> Self {
        Self {
            resource,
            width: map.typed_map().width(),
            height: map.typed_map().height(),
            plane_count: map.plane_count(),
        }
    }

    pub fn from_validated_map_resource(
        resource: &'a LensShadingResource,
        map: &LensShadingMap,
    ) -> Self {
        Self {
            resource,
            width: map.width(),
            height: map.height(),
            plane_count: map.plane_count(),
        }
    }
}

// Metadata binding for one raster-order Q16.16 gain per visible pixel.
#[derive(Debug)]
pub struct GpuFullResolutionGainMapBinding<'a, GainMapResource> {
    pub resource: &'a GainMapResource,
    pub width: u32,
    pub height: u32,
    pub pixel_count: usize,
    pub memory_bytes: usize,
    pub fractional_bits: u32,
}

impl<'a, GainMapResource> GpuFullResolutionGainMapBinding<'a, GainMapResource> {
    pub fn from_prepared_full_resolution_gain_map_resource(
        resource: &'a GainMapResource,
        gain_map: &PreparedFullResolutionFixedGainMap,
    ) -> Self {
        Self {
            resource,
            width: gain_map.width(),
            height: gain_map.height(),
            pixel_count: gain_map.pixel_count(),
            memory_bytes: gain_map.memory_bytes(),
            fractional_bits: gain_map.fixed_gain_fractional_bits(),
        }
    }
}

#[derive(Debug)]
pub struct GpuVignetteCorrectionInput<'a, FrameResource, LensShadingResource> {
    pub input_frame: &'a FrameResource,
    pub output_frame: &'a mut FrameResource,
    pub lens_shading_map: GpuLensShadingMapBinding<'a, LensShadingResource>,
    pub frame_dimensions: FrameDimensions,
    pub options: VignetteCorrectionOptions,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuVignetteCorrectionResult {
    pub output_black_level: [f32; BAYER_CFA_PLANE_COUNT],
    pub output_white_level: u16,
    pub applied: bool,
}

// Backend-specific frame and map resources remain opaque to this interface.
// The shared result reports only output levels and whether correction ran.
pub trait GpuVignetteCorrectionBackend {
    type FrameResource;
    type LensShadingResource;
    type Error;

    fn apply_vignette_correction_gpu(
        &mut self,
        input: GpuVignetteCorrectionInput<'_, Self::FrameResource, Self::LensShadingResource>,
    ) -> Result<GpuVignetteCorrectionResult, Self::Error>;
}

pub fn gpu_vignette_correction_result_for_mode(
    options: VignetteCorrectionOptions,
) -> GpuVignetteCorrectionResult {
    GpuVignetteCorrectionResult {
        output_black_level: options.output_black_level(),
        output_white_level: options.output_white_level,
        applied: options.mode == VignetteCorrectionMode::Enabled,
    }
}

pub struct GpuVignetteCorrector {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    output_buffer: Option<GpuSizedBuffer>,
    params_buffers: Vec<GpuSizedBuffer>,
    compact_spatial_map: Option<GpuCompactSpatialMapResource>,
    gain_conversion: Option<GainConversionFingerprint>,
    resource_stats: GpuCompactVignetteResourceStats,
}

#[derive(Clone)]
pub struct GpuUploadedFullResolutionGainMap {
    buffer: wgpu::Buffer,
    frame_dimensions: FrameDimensions,
    pixel_count: usize,
    memory_bytes: usize,
    fractional_bits: u32,
    source_map_width: usize,
    source_map_height: usize,
    source_plane_count: usize,
    compact_interpolated: bool,
}

#[derive(Clone)]
struct GpuCompactSpatialMapResource {
    fingerprint: CompactSpatialMapFingerprint,
    buffer: wgpu::Buffer,
    memory_bytes: usize,
    source_map_width: usize,
    source_map_height: usize,
    source_plane_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuCompactVignetteResourceStats {
    pub spatial_uploads: u64,
    pub conversion_state_updates: u64,
    pub spatial_reuses: u64,
    pub conversion_state_reuses: u64,
    pub spatial_buffer_creates: u64,
    pub conversion_buffer_creates: u64,
    pub conversion_coverage_expansions: u64,
    pub spatial_allocated_bytes: u64,
    pub spatial_queue_writes: u64,
    pub spatial_queue_written_bytes: u64,
    pub conversion_allocated_bytes: u64,
    pub conversion_queue_writes: u64,
    pub conversion_queue_written_bytes: u64,
}

#[derive(Clone)]
pub struct GpuVignetteGainMapUpload {
    pub uploaded: GpuUploadedFullResolutionGainMap,
    pub timings: GpuVignetteGainMapUploadTimings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuVignetteGainMapUploadTimings {
    pub buffer_create: Duration,
    pub byte_conversion: Duration,
    pub queue_write: Duration,
    pub total: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuVignetteCorrectionParams {
    pub mode: VignetteCorrectionMode,
    pub frame_dimensions: FrameDimensions,
    pub bayer_pattern_tag: u32,
    pub source_map_width: u32,
    pub source_map_height: u32,
    pub source_plane_count: u32,
    pub input_black_level: [f32; BAYER_CFA_PLANE_COUNT],
    pub input_black_level_q: [u32; BAYER_CFA_PLANE_COUNT],
    pub output_white_level: u16,
    conversion_policy_tag: u32,
    scale_num: u32,
    scale_shift: u32,
    strength_num: u32,
    strength_shift: u32,
}

pub struct GpuVignettePackedU16DispatchInput<'a, 'encoder> {
    pub device: &'encoder wgpu::Device,
    pub queue: &'encoder wgpu::Queue,
    pub encoder: &'encoder mut wgpu::CommandEncoder,
    pub input_buffer: &'a wgpu::Buffer,
    pub input_buffer_bytes: u64,
    pub uploaded_gain_map: &'a GpuUploadedFullResolutionGainMap,
    pub params: GpuVignetteCorrectionParams,
}

#[derive(Debug)]
pub struct GpuVignetteCorrectionDispatch<'a> {
    pub output: GpuVignetteCorrectionOutput<'a>,
    pub output_black_level: [f32; BAYER_CFA_PLANE_COUNT],
    pub output_white_level: u16,
    pub applied: bool,
    pub stats: GpuVignetteCorrectionStats,
}

#[derive(Debug)]
pub enum GpuVignetteCorrectionOutput<'a> {
    InputPassthrough(&'a wgpu::Buffer),
    Corrected(&'a wgpu::Buffer),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuVignetteCorrectionStats {
    pub corrected_pixel_count: u64,
    pub packed_word_count: u64,
    pub input_buffer_bytes: u64,
    pub output_buffer_bytes: u64,
    pub gain_buffer_bytes: u64,
    pub max_storage_buffer_binding_size: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub max_compute_workgroups_per_dimension: u64,
    pub tiling_used: bool,
    pub tile_count: u64,
    pub tile_dispatch_count: u64,
    pub max_tile_gain_bytes: u64,
    pub max_tile_workgroups: u64,
    pub dispatch_submitted: bool,
    pub gain_buffer_reused: bool,
    pub copied_input: bool,
    pub output_buffer_allocated: bool,
    pub params_buffer_count: u64,
    pub params_upload_count: u64,
    pub bind_group_creates: u64,
    pub bind_group_reuses: u64,
}

struct GpuSizedBuffer {
    buffer: wgpu::Buffer,
    size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GpuVignetteBindingLimits {
    max_storage_buffer_binding_size: u64,
    min_storage_buffer_offset_alignment: u64,
    max_compute_workgroups_per_dimension: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GpuGainMapTile {
    base_pixel_offset: usize,
    base_packed_word_offset: usize,
    tile_pixel_count: usize,
    tile_packed_word_count: usize,
    gain_buffer_byte_offset: u64,
    gain_buffer_byte_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GpuGainMapTilePlan {
    tiles: Vec<GpuGainMapTile>,
    max_tile_gain_bytes: u64,
    max_tile_workgroups: u64,
    max_storage_buffer_binding_size: u64,
    min_storage_buffer_offset_alignment: u64,
    max_compute_workgroups_per_dimension: u64,
}

impl GpuVignetteCorrector {
    pub fn new(device: &wgpu::Device) -> Result<Self, VignetteCorrectionError> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mcraw4vulkan vignette packed-u16 correction shader"),
            source: wgpu::ShaderSource::Wgsl(VIGNETTE_CORRECT_PACKED_U16_WGSL.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mcraw4vulkan vignette packed-u16 correction pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        Ok(Self {
            pipeline,
            bind_group_layout,
            output_buffer: None,
            params_buffers: Vec::new(),
            compact_spatial_map: None,
            gain_conversion: None,
            resource_stats: GpuCompactVignetteResourceStats::default(),
        })
    }

    pub fn upload_full_resolution_gain_map(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gain_map: &PreparedFullResolutionFixedGainMap,
    ) -> Result<GpuUploadedFullResolutionGainMap, VignetteCorrectionError> {
        Ok(self
            .upload_full_resolution_gain_map_with_timings(device, queue, gain_map)?
            .uploaded)
    }

    pub fn upload_full_resolution_gain_map_with_timings(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gain_map: &PreparedFullResolutionFixedGainMap,
    ) -> Result<GpuVignetteGainMapUpload, VignetteCorrectionError> {
        let total_start = Instant::now();
        let byte_len = u64::try_from(gain_map.memory_bytes()).map_err(|_| {
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: gain_map.frame_dimensions(),
            }
        })?;

        let buffer_create_start = Instant::now();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcraw4vulkan vignette full-resolution Q16.16 gain map"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer_create = buffer_create_start.elapsed();

        let byte_conversion_start = Instant::now();
        let bytes = u32_slice_to_le_bytes(gain_map.gains_q16());
        let byte_conversion = byte_conversion_start.elapsed();

        let queue_write_start = Instant::now();
        queue.write_buffer(&buffer, 0, &bytes);
        let queue_write = queue_write_start.elapsed();

        let uploaded = GpuUploadedFullResolutionGainMap {
            buffer,
            frame_dimensions: gain_map.frame_dimensions(),
            pixel_count: gain_map.pixel_count(),
            memory_bytes: gain_map.memory_bytes(),
            fractional_bits: gain_map.fixed_gain_fractional_bits(),
            source_map_width: gain_map.source_map_width(),
            source_map_height: gain_map.source_map_height(),
            source_plane_count: gain_map.source_plane_count(),
            compact_interpolated: false,
        };

        Ok(GpuVignetteGainMapUpload {
            uploaded,
            timings: GpuVignetteGainMapUploadTimings {
                buffer_create,
                byte_conversion,
                queue_write,
                total: total_start.elapsed(),
            },
        })
    }

    pub fn upload_compact_gain_map_from_fixed_facts(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<GpuUploadedFullResolutionGainMap, VignetteCorrectionError> {
        self.ensure_compact_gain_map_from_fixed_facts(device, queue, facts)
    }

    pub fn ensure_compact_gain_map_from_fixed_facts(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<GpuUploadedFullResolutionGainMap, VignetteCorrectionError> {
        Ok(self
            .ensure_compact_gain_map_from_fixed_facts_with_timings(device, queue, facts)?
            .uploaded)
    }

    pub fn ensure_compact_gain_map_from_fixed_facts_with_timings(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<GpuVignetteGainMapUpload, VignetteCorrectionError> {
        let total_start = Instant::now();
        let prepared_map = facts
            .lens_shading_map
            .as_ref()
            .ok_or(VignetteCorrectionError::MissingLensShadingMap)?;
        validate_compact_direct_conversion_domain(prepared_map, facts)?;
        let spatial_fingerprint = CompactSpatialMapFingerprint::from_fixed_facts(facts)?;
        let conversion_fingerprint = GainConversionFingerprint::from_fixed_facts(facts)?;
        let mut buffer_create = Duration::ZERO;
        let mut byte_conversion = Duration::ZERO;
        let mut queue_write = Duration::ZERO;

        if self
            .compact_spatial_map
            .as_ref()
            .is_some_and(|resource| resource.fingerprint == spatial_fingerprint)
        {
            self.resource_stats.spatial_reuses =
                self.resource_stats.spatial_reuses.saturating_add(1);
        } else {
            let byte_conversion_start = Instant::now();
            let compact_words = compact_gain_words(prepared_map)?;
            let compact_bytes = u32_slice_to_le_bytes(&compact_words);
            byte_conversion += byte_conversion_start.elapsed();
            let compact_byte_len = u64::try_from(compact_bytes.len()).map_err(|_| {
                VignetteCorrectionError::FrameDimensionsOverflow {
                    dimensions: facts.frame_dimensions,
                }
            })?;

            let existing = self.compact_spatial_map.take();
            let (buffer, buffer_memory_bytes, created) = match existing {
                Some(resource) if resource.memory_bytes >= compact_bytes.len() => {
                    (resource.buffer, resource.memory_bytes, false)
                }
                _ => {
                    let buffer_create_start = Instant::now();
                    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("mcraw4vulkan compact fixed lens map"),
                        size: compact_byte_len,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    buffer_create += buffer_create_start.elapsed();
                    (buffer, compact_bytes.len(), true)
                }
            };

            let queue_write_start = Instant::now();
            queue.write_buffer(&buffer, 0, &compact_bytes);
            queue_write += queue_write_start.elapsed();
            self.resource_stats.spatial_uploads =
                self.resource_stats.spatial_uploads.saturating_add(1);
            self.resource_stats.spatial_queue_writes =
                self.resource_stats.spatial_queue_writes.saturating_add(1);
            self.resource_stats.spatial_queue_written_bytes = self
                .resource_stats
                .spatial_queue_written_bytes
                .saturating_add(u64::try_from(compact_bytes.len()).unwrap_or(u64::MAX));
            if created {
                self.resource_stats.spatial_buffer_creates =
                    self.resource_stats.spatial_buffer_creates.saturating_add(1);
                self.resource_stats.spatial_allocated_bytes = self
                    .resource_stats
                    .spatial_allocated_bytes
                    .saturating_add(u64::try_from(compact_bytes.len()).unwrap_or(u64::MAX));
            }
            self.compact_spatial_map = Some(GpuCompactSpatialMapResource {
                fingerprint: spatial_fingerprint,
                buffer,
                memory_bytes: buffer_memory_bytes,
                source_map_width: prepared_map.width(),
                source_map_height: prepared_map.height(),
                source_plane_count: prepared_map.plane_count(),
            });
        }

        if self.gain_conversion == Some(conversion_fingerprint) {
            self.resource_stats.conversion_state_reuses = self
                .resource_stats
                .conversion_state_reuses
                .saturating_add(1);
        } else {
            self.resource_stats.conversion_state_updates = self
                .resource_stats
                .conversion_state_updates
                .saturating_add(1);
            self.gain_conversion = Some(conversion_fingerprint);
        }

        let spatial = self
            .compact_spatial_map
            .as_ref()
            .expect("compact spatial map was ensured");
        let pixel_count = facts.frame_dimensions.pixel_count().ok_or(
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: facts.frame_dimensions,
            },
        )?;

        Ok(GpuVignetteGainMapUpload {
            uploaded: GpuUploadedFullResolutionGainMap {
                buffer: spatial.buffer.clone(),
                frame_dimensions: facts.frame_dimensions,
                pixel_count,
                memory_bytes: spatial.memory_bytes,
                fractional_bits: crate::VIGNETTE_GAIN_FRACTIONAL_BITS,
                source_map_width: spatial.source_map_width,
                source_map_height: spatial.source_map_height,
                source_plane_count: spatial.source_plane_count,
                compact_interpolated: true,
            },
            timings: GpuVignetteGainMapUploadTimings {
                buffer_create,
                byte_conversion,
                queue_write,
                total: total_start.elapsed(),
            },
        })
    }

    pub fn compact_resource_stats(&self) -> GpuCompactVignetteResourceStats {
        self.resource_stats
    }

    pub fn compact_resource_counts(&self) -> (usize, usize) {
        (
            usize::from(self.compact_spatial_map.is_some()),
            usize::from(self.gain_conversion.is_some()),
        )
    }

    pub fn upload_compact_gain_map_from_fixed_facts_unbounded_e14(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<GpuUploadedFullResolutionGainMap, VignetteCorrectionError> {
        let prepared_map = facts
            .lens_shading_map
            .as_ref()
            .ok_or(VignetteCorrectionError::MissingLensShadingMap)?;
        validate_compact_direct_conversion_domain(prepared_map, facts)?;
        let compact_words = compact_gain_words(prepared_map)?;
        let compact_bytes = u32_slice_to_le_bytes(&compact_words);
        let compact_byte_len = u64::try_from(compact_bytes.len()).map_err(|_| {
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: facts.frame_dimensions,
            }
        })?;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcraw4vulkan E14 compact fixed lens map"),
            size: compact_byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &compact_bytes);

        let pixel_count = facts.frame_dimensions.pixel_count().ok_or(
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: facts.frame_dimensions,
            },
        )?;

        Ok(GpuUploadedFullResolutionGainMap {
            buffer,
            frame_dimensions: facts.frame_dimensions,
            pixel_count,
            memory_bytes: compact_bytes.len(),
            fractional_bits: crate::VIGNETTE_GAIN_FRACTIONAL_BITS,
            source_map_width: prepared_map.width(),
            source_map_height: prepared_map.height(),
            source_plane_count: prepared_map.plane_count(),
            compact_interpolated: true,
        })
    }

    pub fn dispatch_packed_u16<'a>(
        &'a mut self,
        input: GpuVignettePackedU16DispatchInput<'a, '_>,
    ) -> Result<GpuVignetteCorrectionDispatch<'a>, VignetteCorrectionError> {
        let binding_limits = GpuVignetteBindingLimits::from_device(input.device);
        self.dispatch_packed_u16_with_limits(input, binding_limits)
    }

    pub fn dispatch_packed_u16_with_output_clear<'a>(
        &'a mut self,
        input: GpuVignettePackedU16DispatchInput<'a, '_>,
    ) -> Result<GpuVignetteCorrectionDispatch<'a>, VignetteCorrectionError> {
        let binding_limits = GpuVignetteBindingLimits::from_device(input.device);
        self.dispatch_packed_u16_with_limits_and_options(input, binding_limits, true)
    }

    fn dispatch_packed_u16_with_limits<'a>(
        &'a mut self,
        input: GpuVignettePackedU16DispatchInput<'a, '_>,
        binding_limits: GpuVignetteBindingLimits,
    ) -> Result<GpuVignetteCorrectionDispatch<'a>, VignetteCorrectionError> {
        self.dispatch_packed_u16_with_limits_and_options(input, binding_limits, false)
    }

    fn dispatch_packed_u16_with_limits_and_options<'a>(
        &'a mut self,
        input: GpuVignettePackedU16DispatchInput<'a, '_>,
        binding_limits: GpuVignetteBindingLimits,
        clear_output_before_dispatch: bool,
    ) -> Result<GpuVignetteCorrectionDispatch<'a>, VignetteCorrectionError> {
        let GpuVignettePackedU16DispatchInput {
            device,
            queue,
            encoder,
            input_buffer,
            input_buffer_bytes,
            uploaded_gain_map,
            params,
        } = input;

        let pixel_count = params.pixel_count()?;
        let output_buffer_bytes = packed_u16_byte_len(pixel_count)?;
        if input_buffer_bytes < output_buffer_bytes {
            return Err(VignetteCorrectionError::GpuBufferSizeMismatch {
                buffer: "input packed-u16",
                required_bytes: output_buffer_bytes,
                actual_bytes: input_buffer_bytes,
            });
        }

        let packed_word_count = packed_word_count(pixel_count)?;
        let gain_buffer_bytes = u64::try_from(uploaded_gain_map.memory_bytes()).map_err(|_| {
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: uploaded_gain_map.frame_dimensions(),
            }
        })?;

        if params.mode == VignetteCorrectionMode::Disabled {
            return Ok(GpuVignetteCorrectionDispatch {
                output: GpuVignetteCorrectionOutput::InputPassthrough(input_buffer),
                output_black_level: params.output_black_level(),
                output_white_level: params.output_white_level,
                applied: false,
                stats: GpuVignetteCorrectionStats {
                    corrected_pixel_count: 0,
                    packed_word_count: u64::try_from(packed_word_count).unwrap_or(u64::MAX),
                    input_buffer_bytes,
                    output_buffer_bytes: input_buffer_bytes,
                    gain_buffer_bytes,
                    max_storage_buffer_binding_size: binding_limits.max_storage_buffer_binding_size,
                    min_storage_buffer_offset_alignment: binding_limits
                        .min_storage_buffer_offset_alignment,
                    max_compute_workgroups_per_dimension: binding_limits
                        .max_compute_workgroups_per_dimension,
                    tiling_used: false,
                    tile_count: 0,
                    tile_dispatch_count: 0,
                    max_tile_gain_bytes: 0,
                    max_tile_workgroups: 0,
                    dispatch_submitted: false,
                    gain_buffer_reused: true,
                    copied_input: false,
                    output_buffer_allocated: false,
                    params_buffer_count: 0,
                    params_upload_count: 0,
                    bind_group_creates: 0,
                    bind_group_reuses: 0,
                },
            });
        }

        validate_uploaded_gain_map(uploaded_gain_map, params.frame_dimensions, pixel_count)?;
        let tile_plan = plan_gain_map_tiles(params.frame_dimensions, pixel_count, binding_limits)?;

        let output_buffer_allocated = ensure_buffer_slot(
            device,
            &mut self.output_buffer,
            "mcraw4vulkan vignette corrected packed-u16 output",
            output_buffer_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        ensure_buffer_slots(
            device,
            &mut self.params_buffers,
            tile_plan.tiles.len(),
            "mcraw4vulkan vignette correction params",
            VIGNETTE_GPU_PARAMS_BYTE_LEN,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let output_buffer = &self
            .output_buffer
            .as_ref()
            .expect("output buffer slot was just ensured")
            .buffer;
        if clear_output_before_dispatch {
            encoder.clear_buffer(output_buffer, 0, None);
        }
        let input_binding_size = wgpu::BufferSize::new(output_buffer_bytes)
            .ok_or(VignetteCorrectionError::GpuTileSizeZero)?;
        let output_binding_size = wgpu::BufferSize::new(output_buffer_bytes)
            .ok_or(VignetteCorrectionError::GpuTileSizeZero)?;
        let mut bind_groups = Vec::with_capacity(tile_plan.tiles.len());

        for (tile_index, tile) in tile_plan.tiles.iter().enumerate() {
            let params_buffer = &self
                .params_buffers
                .get(tile_index)
                .expect("params buffer slot was just ensured")
                .buffer;
            let params_bytes = params.to_le_bytes(
                pixel_count,
                packed_word_count,
                *tile,
                uploaded_gain_map.source_map_word_count(),
            )?;
            queue.write_buffer(params_buffer, 0, &params_bytes);

            let gain_binding_size =
                wgpu::BufferSize::new(u64::try_from(uploaded_gain_map.memory_bytes()).map_err(
                    |_| VignetteCorrectionError::FrameDimensionsOverflow {
                        dimensions: uploaded_gain_map.frame_dimensions(),
                    },
                )?)
                .ok_or(VignetteCorrectionError::GpuTileSizeZero)?;
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mcraw4vulkan vignette correction tile bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: input_buffer,
                            offset: 0,
                            size: Some(input_binding_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: uploaded_gain_map.buffer(),
                            offset: 0,
                            size: Some(gain_binding_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: output_buffer,
                            offset: 0,
                            size: Some(output_binding_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            }));
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mcraw4vulkan vignette correction compute pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            for (tile, bind_group) in tile_plan.tiles.iter().zip(bind_groups.iter()) {
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_width(tile.tile_packed_word_count), 1, 1);
            }
        }

        let tile_count = u64::try_from(tile_plan.tiles.len()).unwrap_or(u64::MAX);

        Ok(GpuVignetteCorrectionDispatch {
            output: GpuVignetteCorrectionOutput::Corrected(output_buffer),
            output_black_level: params.output_black_level(),
            output_white_level: params.output_white_level,
            applied: true,
            stats: GpuVignetteCorrectionStats {
                corrected_pixel_count: u64::try_from(pixel_count).unwrap_or(u64::MAX),
                packed_word_count: u64::try_from(packed_word_count).unwrap_or(u64::MAX),
                input_buffer_bytes,
                output_buffer_bytes,
                gain_buffer_bytes,
                max_storage_buffer_binding_size: tile_plan.max_storage_buffer_binding_size,
                min_storage_buffer_offset_alignment: tile_plan.min_storage_buffer_offset_alignment,
                max_compute_workgroups_per_dimension: tile_plan
                    .max_compute_workgroups_per_dimension,
                tiling_used: tile_plan.tiles.len() > 1,
                tile_count,
                tile_dispatch_count: tile_count,
                max_tile_gain_bytes: tile_plan.max_tile_gain_bytes,
                max_tile_workgroups: tile_plan.max_tile_workgroups,
                dispatch_submitted: true,
                gain_buffer_reused: true,
                copied_input: false,
                output_buffer_allocated,
                params_buffer_count: tile_count,
                params_upload_count: tile_count,
                bind_group_creates: tile_count,
                bind_group_reuses: 0,
            },
        })
    }
}

impl GpuUploadedFullResolutionGainMap {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn frame_dimensions(&self) -> FrameDimensions {
        self.frame_dimensions
    }

    pub fn width(&self) -> u32 {
        self.frame_dimensions.width
    }

    pub fn height(&self) -> u32 {
        self.frame_dimensions.height
    }

    pub fn pixel_count(&self) -> usize {
        self.pixel_count
    }

    pub fn memory_bytes(&self) -> usize {
        self.memory_bytes
    }

    pub fn fixed_gain_fractional_bits(&self) -> u32 {
        self.fractional_bits
    }

    pub fn source_map_width(&self) -> usize {
        self.source_map_width
    }

    pub fn source_map_height(&self) -> usize {
        self.source_map_height
    }

    pub fn source_plane_count(&self) -> usize {
        self.source_plane_count
    }

    pub fn source_map_word_count(&self) -> usize {
        self.source_map_width
            .saturating_mul(self.source_map_height)
            .saturating_mul(self.source_plane_count)
    }

    pub fn is_compact_interpolated(&self) -> bool {
        self.compact_interpolated
    }
}

fn compact_gain_words(
    prepared_map: &PreparedFixedLensShadingMap<'_>,
) -> Result<Vec<u32>, VignetteCorrectionError> {
    let mut words = Vec::new();
    for plane_index in 0..prepared_map.plane_count() {
        let plane = prepared_map.plane_q(plane_index)?;
        words.reserve(plane.len().saturating_mul(2));
        for (sample_index, value) in plane.iter().copied().enumerate() {
            let value = u64::try_from(value).map_err(|_| {
                VignetteCorrectionError::FixedPointGainOverflow {
                    plane_index,
                    sample_index,
                    value: f32::INFINITY,
                }
            })?;
            words.push(value as u32);
            words.push((value >> 32) as u32);
        }
    }
    Ok(words)
}

fn validate_compact_direct_conversion_domain(
    prepared_map: &PreparedFixedLensShadingMap<'_>,
    facts: &FixedPointVignetteInputFacts<'_>,
) -> Result<(), VignetteCorrectionError> {
    let mut max_gain_q = 0_i64;
    for plane_index in 0..prepared_map.plane_count() {
        for (sample_index, value) in prepared_map
            .plane_q(plane_index)?
            .iter()
            .copied()
            .enumerate()
        {
            if value < 0 {
                return Err(VignetteCorrectionError::FixedPointGainOverflow {
                    plane_index,
                    sample_index,
                    value: f32::INFINITY,
                });
            }
            max_gain_q = max_gain_q.max(value);
        }
    }

    let max_converted_gain_q = match facts.correction_policy {
        VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1 => {
            motioncam_compatible_gain_q16_from_raw_gain_q(max_gain_q, facts, 0, 0)?
        }
        VignetteCorrectionPolicy::LumaPlane0 => max_gain_q,
    };
    if max_converted_gain_q < 0 || max_converted_gain_q > i64::from(u32::MAX) {
        return Err(VignetteCorrectionError::FullResolutionGainMapGainOverflow {
            x: 0,
            y: 0,
            value: max_converted_gain_q,
        });
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

fn conversion_policy_tag(policy: VignetteCorrectionPolicy) -> u32 {
    match policy {
        VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1 => 0,
        VignetteCorrectionPolicy::LumaPlane0 => 1,
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
        if fraction == 0 {
            return None;
        }
        return Some((fraction, 149));
    }

    let mantissa = (1_u32 << 23) | fraction;
    let exponent = i32::try_from(exponent_bits).ok()? - 127;
    let shift = 23_i32.checked_sub(exponent)?;
    if shift < 0 {
        return None;
    }

    Some((mantissa, u32::try_from(shift).ok()?))
}

impl GpuVignetteBindingLimits {
    fn from_device(device: &wgpu::Device) -> Self {
        let limits = device.limits();

        Self {
            max_storage_buffer_binding_size: u64::from(limits.max_storage_buffer_binding_size),
            min_storage_buffer_offset_alignment: u64::from(
                limits.min_storage_buffer_offset_alignment,
            ),
            max_compute_workgroups_per_dimension: u64::from(
                limits.max_compute_workgroups_per_dimension,
            ),
        }
    }
}

impl GpuVignetteCorrectionParams {
    pub fn from_fixed_facts(
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<Self, VignetteCorrectionError> {
        let prepared_map = facts
            .lens_shading_map
            .as_ref()
            .ok_or(VignetteCorrectionError::MissingLensShadingMap)?;
        let source_map_width = u32::try_from(prepared_map.width()).map_err(|_| {
            VignetteCorrectionError::LensShadingMapDimensionsOverflow {
                width: u32::MAX,
                height: u32::MAX,
            }
        })?;
        let source_map_height = u32::try_from(prepared_map.height()).map_err(|_| {
            VignetteCorrectionError::LensShadingMapDimensionsOverflow {
                width: u32::MAX,
                height: u32::MAX,
            }
        })?;
        let source_plane_count = u32::try_from(prepared_map.plane_count()).map_err(|_| {
            VignetteCorrectionError::UnsupportedLensShadingPlaneCount {
                expected: BAYER_CFA_PLANE_COUNT,
                actual: prepared_map.plane_count(),
            }
        })?;
        let (input_black_level, input_black_level_q, output_white_level) = if facts
            .correction_policy
            == VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1
        {
            let mut input_black_level_q = [0_u32; BAYER_CFA_PLANE_COUNT];
            for (index, value) in facts
                .pixel_domain
                .source_black_storage
                .iter()
                .copied()
                .enumerate()
            {
                let q = f64::from(value) * VIGNETTE_GAIN_SCALE as f64;
                if !q.is_finite() || q < 0.0 || q > f64::from(u32::MAX) {
                    return Err(VignetteCorrectionError::GpuFixedPointBlackLevelOverflow {
                        index,
                        value: i64::MAX,
                    });
                }
                input_black_level_q[index] = q.round() as u32;
            }
            (
                facts.pixel_domain.source_black_storage,
                input_black_level_q,
                facts.pixel_domain.sample_limit,
            )
        } else {
            let mut input_black_level_q = [0_u32; BAYER_CFA_PLANE_COUNT];
            for (index, value) in facts.input_black_level_q.into_iter().enumerate() {
                input_black_level_q[index] = u32::try_from(value).map_err(|_| {
                    VignetteCorrectionError::GpuFixedPointBlackLevelOverflow { index, value }
                })?;
            }
            (
                facts.input_black_level,
                input_black_level_q,
                facts.output_white_level,
            )
        };
        let (scale_num, scale_shift) = f32_power2_rational(
            facts.pixel_domain.source_to_corrected_scale,
        )
        .ok_or(VignetteCorrectionError::GpuInvalidTilePlan {
            reason: "source-to-corrected scale is not a positive finite f32 rational",
        })?;
        let (strength_num, strength_shift) = f32_power2_rational(
            MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH,
        )
        .ok_or(VignetteCorrectionError::GpuInvalidTilePlan {
            reason: "MotionCam gain strength is not a positive finite f32 rational",
        })?;

        Ok(Self {
            mode: facts.mode,
            frame_dimensions: facts.frame_dimensions,
            bayer_pattern_tag: bayer_pattern_tag(facts.bayer_pattern),
            source_map_width,
            source_map_height,
            source_plane_count,
            input_black_level,
            input_black_level_q,
            output_white_level,
            conversion_policy_tag: conversion_policy_tag(facts.correction_policy),
            scale_num,
            scale_shift,
            strength_num,
            strength_shift,
        })
    }

    pub fn output_black_level(&self) -> [f32; BAYER_CFA_PLANE_COUNT] {
        match self.mode {
            VignetteCorrectionMode::Enabled => [0.0; BAYER_CFA_PLANE_COUNT],
            VignetteCorrectionMode::Disabled => self.input_black_level,
        }
    }

    pub fn pixel_count(&self) -> Result<usize, VignetteCorrectionError> {
        if self.frame_dimensions.width == 0 || self.frame_dimensions.height == 0 {
            return Err(VignetteCorrectionError::InvalidFrameDimensions {
                dimensions: self.frame_dimensions,
            });
        }

        self.frame_dimensions.pixel_count().ok_or(
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: self.frame_dimensions,
            },
        )
    }

    fn to_le_bytes(
        self,
        pixel_count: usize,
        packed_word_count: usize,
        tile: GpuGainMapTile,
        source_map_word_count: usize,
    ) -> Result<[u8; VIGNETTE_GPU_PARAMS_BYTE_LEN as usize], VignetteCorrectionError> {
        // These 32 little-endian words are the host half of WGSL Params. Field
        // order and the 128-byte size must change together with the shader.
        let pixel_count = u32::try_from(pixel_count).map_err(|_| {
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: self.frame_dimensions,
            }
        })?;
        let packed_word_count = u32::try_from(packed_word_count).map_err(|_| {
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: self.frame_dimensions,
            }
        })?;
        let base_pixel_offset = u32::try_from(tile.base_pixel_offset).map_err(|_| {
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: self.frame_dimensions,
            }
        })?;
        let base_packed_word_offset =
            u32::try_from(tile.base_packed_word_offset).map_err(|_| {
                VignetteCorrectionError::FrameDimensionsOverflow {
                    dimensions: self.frame_dimensions,
                }
            })?;
        let tile_pixel_count = u32::try_from(tile.tile_pixel_count).map_err(|_| {
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: self.frame_dimensions,
            }
        })?;
        let tile_packed_word_count = u32::try_from(tile.tile_packed_word_count).map_err(|_| {
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: self.frame_dimensions,
            }
        })?;

        let words = [
            pixel_count,
            packed_word_count,
            self.frame_dimensions.width,
            self.frame_dimensions.height,
            self.input_black_level_q[0],
            self.input_black_level_q[1],
            self.input_black_level_q[2],
            self.input_black_level_q[3],
            u32::from(self.output_white_level),
            base_pixel_offset,
            base_packed_word_offset,
            tile_pixel_count,
            tile_packed_word_count,
            self.bayer_pattern_tag,
            self.source_map_width,
            self.source_map_height,
            self.source_plane_count,
            self.conversion_policy_tag,
            self.scale_num,
            u32::try_from(source_map_word_count).map_err(|_| {
                VignetteCorrectionError::GpuInvalidTilePlan {
                    reason: "compact source map word count does not fit u32",
                }
            })?,
            self.scale_shift,
            self.strength_num,
            self.strength_shift,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let mut bytes = [0_u8; VIGNETTE_GPU_PARAMS_BYTE_LEN as usize];
        for (index, word) in words.into_iter().enumerate() {
            let start = index * std::mem::size_of::<u32>();
            bytes[start..start + std::mem::size_of::<u32>()].copy_from_slice(&word.to_le_bytes());
        }

        Ok(bytes)
    }
}

impl<'a> GpuVignetteCorrectionOutput<'a> {
    pub fn buffer(&self) -> &'a wgpu::Buffer {
        match self {
            Self::InputPassthrough(buffer) | Self::Corrected(buffer) => buffer,
        }
    }

    pub fn is_passthrough(&self) -> bool {
        matches!(self, Self::InputPassthrough(_))
    }
}

const VIGNETTE_GPU_PARAMS_BYTE_LEN: u64 = 128;
const VIGNETTE_GAIN_BYTES_PER_PIXEL: u64 = std::mem::size_of::<u32>() as u64;

fn validate_uploaded_gain_map(
    gain_map: &GpuUploadedFullResolutionGainMap,
    frame_dimensions: FrameDimensions,
    pixel_count: usize,
) -> Result<(), VignetteCorrectionError> {
    if !gain_map.is_compact_interpolated() {
        return Err(VignetteCorrectionError::GpuInvalidTilePlan {
            reason: "E14 compact shader requires compact uploaded gain data",
        });
    }
    if gain_map.frame_dimensions() != frame_dimensions {
        return Err(VignetteCorrectionError::GpuFrameDimensionsMismatch {
            frame_dimensions,
            gain_map_dimensions: gain_map.frame_dimensions(),
        });
    }
    if gain_map.pixel_count() != pixel_count {
        return Err(VignetteCorrectionError::GpuGainMapPixelCountMismatch {
            expected_pixel_count: pixel_count,
            actual_pixel_count: gain_map.pixel_count(),
        });
    }

    Ok(())
}

fn plan_gain_map_tiles(
    frame_dimensions: FrameDimensions,
    pixel_count: usize,
    limits: GpuVignetteBindingLimits,
) -> Result<GpuGainMapTilePlan, VignetteCorrectionError> {
    // Non-final tiles start on an even pixel so no packed U16 word is split
    // across dispatches; tile size also respects storage and workgroup limits.
    let total_gain_bytes = gain_byte_len(pixel_count)?;
    if pixel_count == 0 {
        return Err(VignetteCorrectionError::InvalidFrameDimensions {
            dimensions: frame_dimensions,
        });
    }

    let max_pixels_per_binding =
        usize::try_from(limits.max_storage_buffer_binding_size / VIGNETTE_GAIN_BYTES_PER_PIXEL)
            .map_err(|_| VignetteCorrectionError::GpuGainMapBindingTooLarge {
                required_bytes: total_gain_bytes,
                max_binding_bytes: limits.max_storage_buffer_binding_size,
                frame_dimensions,
                pixel_count,
            })?;
    if max_pixels_per_binding == 0 {
        return Err(VignetteCorrectionError::GpuGainMapBindingTooLarge {
            required_bytes: total_gain_bytes,
            max_binding_bytes: limits.max_storage_buffer_binding_size,
            frame_dimensions,
            pixel_count,
        });
    }
    let max_dispatch_words = limits
        .max_compute_workgroups_per_dimension
        .checked_mul(u64::from(VIGNETTE_CORRECT_WORKGROUP_SIZE))
        .ok_or(VignetteCorrectionError::GpuTilePixelCountOverflow)?;
    let max_dispatch_pixels = max_dispatch_words
        .checked_mul(u64::from(VIGNETTE_PACKED_U16_SAMPLES_PER_WORD))
        .ok_or(VignetteCorrectionError::GpuTilePixelCountOverflow)?;
    let max_pixels_per_dispatch = usize::try_from(max_dispatch_pixels)
        .map_err(|_| VignetteCorrectionError::GpuTilePixelCountOverflow)?;
    if max_pixels_per_dispatch == 0 {
        return Err(VignetteCorrectionError::GpuComputeWorkgroupLimitTooSmall {
            max_workgroups: limits.max_compute_workgroups_per_dimension,
            workgroup_size: VIGNETTE_CORRECT_WORKGROUP_SIZE,
        });
    }

    let alignment_pixels = gain_pixel_offset_alignment(limits.min_storage_buffer_offset_alignment)?;
    let max_pixels_per_tile = max_pixels_per_binding.min(max_pixels_per_dispatch);
    if max_pixels_per_tile == 0 {
        return Err(VignetteCorrectionError::GpuTileSizeZero);
    }
    let non_final_tile_pixels = (max_pixels_per_tile / alignment_pixels) * alignment_pixels;
    if pixel_count > max_pixels_per_tile && non_final_tile_pixels == 0 {
        return Err(VignetteCorrectionError::GpuGainMapBindingTooLarge {
            required_bytes: total_gain_bytes,
            max_binding_bytes: limits.max_storage_buffer_binding_size,
            frame_dimensions,
            pixel_count,
        });
    }

    let mut tiles = Vec::new();
    let mut base_pixel_offset = 0usize;
    let mut max_tile_gain_bytes = 0u64;
    let mut max_tile_workgroups = 0u64;

    while base_pixel_offset < pixel_count {
        let remaining = pixel_count - base_pixel_offset;
        let tile_pixel_count = if remaining <= max_pixels_per_tile {
            remaining
        } else {
            non_final_tile_pixels
        };
        if tile_pixel_count == 0 {
            return Err(VignetteCorrectionError::GpuTileSizeZero);
        }

        let gain_buffer_byte_offset = gain_byte_len(base_pixel_offset)?;
        let gain_buffer_byte_size = gain_byte_len(tile_pixel_count)?;
        if gain_buffer_byte_size > limits.max_storage_buffer_binding_size {
            return Err(VignetteCorrectionError::GpuGainMapBindingTooLarge {
                required_bytes: total_gain_bytes,
                max_binding_bytes: limits.max_storage_buffer_binding_size,
                frame_dimensions,
                pixel_count,
            });
        }
        if limits.min_storage_buffer_offset_alignment > 0
            && gain_buffer_byte_offset % limits.min_storage_buffer_offset_alignment != 0
        {
            return Err(VignetteCorrectionError::GpuTileOffsetAlignment {
                offset: gain_buffer_byte_offset,
                alignment: limits.min_storage_buffer_offset_alignment,
            });
        }
        if !base_pixel_offset.is_multiple_of(2) {
            return Err(VignetteCorrectionError::GpuInvalidTilePlan {
                reason: "tile base pixel offset is not even",
            });
        }
        let next_base = base_pixel_offset
            .checked_add(tile_pixel_count)
            .ok_or(VignetteCorrectionError::GpuTilePixelCountOverflow)?;
        if next_base < pixel_count && tile_pixel_count % 2 != 0 {
            return Err(VignetteCorrectionError::GpuInvalidTilePlan {
                reason: "non-final tile pixel count is not even",
            });
        }
        let tile_packed_word_count = tile_pixel_count.div_ceil(2);
        let tile_workgroups = tile_workgroup_count(tile_packed_word_count)?;
        if tile_workgroups > limits.max_compute_workgroups_per_dimension {
            return Err(VignetteCorrectionError::GpuTileDispatchTooLarge {
                tile_packed_word_count,
                workgroup_size: VIGNETTE_CORRECT_WORKGROUP_SIZE,
                required_workgroups: tile_workgroups,
                max_workgroups: limits.max_compute_workgroups_per_dimension,
            });
        }

        max_tile_gain_bytes = max_tile_gain_bytes.max(gain_buffer_byte_size);
        max_tile_workgroups = max_tile_workgroups.max(tile_workgroups);
        tiles.push(GpuGainMapTile {
            base_pixel_offset,
            base_packed_word_offset: base_pixel_offset / 2,
            tile_pixel_count,
            tile_packed_word_count,
            gain_buffer_byte_offset,
            gain_buffer_byte_size,
        });
        base_pixel_offset = next_base;
    }

    if tiles.is_empty() {
        return Err(VignetteCorrectionError::GpuTileSizeZero);
    }

    Ok(GpuGainMapTilePlan {
        tiles,
        max_tile_gain_bytes,
        max_tile_workgroups,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
    })
}

fn tile_workgroup_count(packed_word_count: usize) -> Result<u64, VignetteCorrectionError> {
    let workgroup_size = VIGNETTE_CORRECT_WORKGROUP_SIZE as usize;
    let workgroups = packed_word_count.div_ceil(workgroup_size).max(1);

    u64::try_from(workgroups).map_err(|_| VignetteCorrectionError::GpuTilePixelCountOverflow)
}

fn gain_byte_len(pixel_count: usize) -> Result<u64, VignetteCorrectionError> {
    let pixel_count = u64::try_from(pixel_count)
        .map_err(|_| VignetteCorrectionError::GpuTilePixelCountOverflow)?;

    pixel_count
        .checked_mul(VIGNETTE_GAIN_BYTES_PER_PIXEL)
        .ok_or(VignetteCorrectionError::GpuTilePixelCountOverflow)
}

fn gain_pixel_offset_alignment(byte_alignment: u64) -> Result<usize, VignetteCorrectionError> {
    let normalized_byte_alignment = byte_alignment.max(1);
    let gain_byte_count = VIGNETTE_GAIN_BYTES_PER_PIXEL;
    let gcd = gcd_u64(normalized_byte_alignment, gain_byte_count);
    let byte_aligned_pixels = normalized_byte_alignment / gcd;
    let combined_alignment = lcm_u64(byte_aligned_pixels, 2)?;

    usize::try_from(combined_alignment).map_err(|_| VignetteCorrectionError::GpuInvalidTilePlan {
        reason: "tile pixel alignment does not fit in usize",
    })
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }

    left
}

fn lcm_u64(left: u64, right: u64) -> Result<u64, VignetteCorrectionError> {
    let gcd = gcd_u64(left, right);
    left.checked_div(gcd)
        .and_then(|value| value.checked_mul(right))
        .ok_or(VignetteCorrectionError::GpuInvalidTilePlan {
            reason: "tile pixel alignment overflow",
        })
}

fn ensure_buffer_slot(
    device: &wgpu::Device,
    slot: &mut Option<GpuSizedBuffer>,
    label: &'static str,
    required_size: u64,
    usage: wgpu::BufferUsages,
) -> bool {
    let needs_allocation = slot
        .as_ref()
        .map(|existing| existing.size < required_size)
        .unwrap_or(true);

    if needs_allocation {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: required_size,
            usage,
            mapped_at_creation: false,
        });

        *slot = Some(GpuSizedBuffer {
            buffer,
            size: required_size,
        });
    }
    needs_allocation
}

fn ensure_buffer_slots(
    device: &wgpu::Device,
    slots: &mut Vec<GpuSizedBuffer>,
    slot_count: usize,
    label: &'static str,
    required_size: u64,
    usage: wgpu::BufferUsages,
) {
    while slots.len() < slot_count {
        slots.push(GpuSizedBuffer {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: required_size,
                usage,
                mapped_at_creation: false,
            }),
            size: required_size,
        });
    }

    for slot in slots.iter_mut().take(slot_count) {
        if slot.size < required_size {
            *slot = GpuSizedBuffer {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: required_size,
                    usage,
                    mapped_at_creation: false,
                }),
                size: required_size,
            };
        }
    }
}

fn packed_word_count(pixel_count: usize) -> Result<usize, VignetteCorrectionError> {
    Ok(pixel_count.div_ceil(2))
}

fn packed_u16_byte_len(pixel_count: usize) -> Result<u64, VignetteCorrectionError> {
    let word_count = packed_word_count(pixel_count)?;
    let bytes = word_count
        .checked_mul(VIGNETTE_PACKED_U16_BYTES_PER_WORD as usize)
        .ok_or(VignetteCorrectionError::FixedPointCorrectionOverflow)?;

    u64::try_from(bytes).map_err(|_| VignetteCorrectionError::FixedPointCorrectionOverflow)
}

fn dispatch_width(packed_word_count: usize) -> u32 {
    tile_workgroup_count(packed_word_count)
        .expect("tile planner checked dispatch width")
        .try_into()
        .expect("tile planner checked dispatch width fits u32")
}

fn u32_slice_to_le_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use mcraw4vulkan_core::{BayerPattern, FrameDimensions};

    use super::*;
    use crate::{
        FixedPointVignetteInputFacts, LensShadingMap, PreparedFixedLensShadingMap,
        VignetteCoordinateMapping, VignetteCorrectionInputFacts,
    };

    #[test]
    fn gpu_params_pack_expected_uniform_words() {
        let params = GpuVignetteCorrectionParams {
            mode: VignetteCorrectionMode::Enabled,
            frame_dimensions: FrameDimensions {
                width: 3,
                height: 2,
            },
            bayer_pattern_tag: 2,
            source_map_width: 5,
            source_map_height: 7,
            source_plane_count: 4,
            input_black_level: [1.0, 2.0, 3.0, 4.0],
            input_black_level_q: [10, 20, 30, 40],
            output_white_level: 4095,
            conversion_policy_tag: 0,
            scale_num: 100,
            scale_shift: 5,
            strength_num: 200,
            strength_shift: 7,
        };

        let tile = GpuGainMapTile {
            base_pixel_offset: 2,
            base_packed_word_offset: 1,
            tile_pixel_count: 4,
            tile_packed_word_count: 2,
            gain_buffer_byte_offset: 8,
            gain_buffer_byte_size: 16,
        };
        let bytes = params.to_le_bytes(6, 3, tile, 140).expect("params pack");
        let words: Vec<u32> = bytes
            .chunks_exact(std::mem::size_of::<u32>())
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        assert_eq!(
            words,
            vec![
                6, 3, 3, 2, 10, 20, 30, 40, 4095, 2, 1, 4, 2, 2, 5, 7, 4, 0, 100, 140, 5, 200, 7,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]
        );
    }

    #[test]
    fn tile_plan_uses_one_tile_when_gain_binding_fits() {
        let dimensions = FrameDimensions {
            width: 3840,
            height: 2160,
        };
        let pixel_count = dimensions.pixel_count().expect("pixel count fits");
        let plan = plan_gain_map_tiles(
            dimensions,
            pixel_count,
            GpuVignetteBindingLimits {
                max_storage_buffer_binding_size: 134_217_728,
                min_storage_buffer_offset_alignment: 256,
                max_compute_workgroups_per_dimension: 65_535,
            },
        )
        .expect("tile plan succeeds");

        assert_eq!(plan.tiles.len(), 1);
        assert_eq!(plan.tiles[0].base_pixel_offset, 0);
        assert_eq!(plan.tiles[0].tile_pixel_count, pixel_count);
        assert_eq!(
            plan.tiles[0].gain_buffer_byte_size,
            (pixel_count as u64) * VIGNETTE_GAIN_BYTES_PER_PIXEL
        );
    }

    #[test]
    fn tile_plan_splits_observed_8k_gain_map_at_binding_limit() {
        let dimensions = FrameDimensions {
            width: 8192,
            height: 4608,
        };
        let pixel_count = dimensions.pixel_count().expect("pixel count fits");
        let plan = plan_gain_map_tiles(
            dimensions,
            pixel_count,
            GpuVignetteBindingLimits {
                max_storage_buffer_binding_size: 134_217_728,
                min_storage_buffer_offset_alignment: 256,
                max_compute_workgroups_per_dimension: 65_535,
            },
        )
        .expect("tile plan succeeds");

        assert_eq!(pixel_count, 37_748_736);
        assert_eq!(plan.tiles.len(), 2);
        assert_eq!(plan.tiles[0].base_pixel_offset, 0);
        assert_eq!(plan.tiles[0].tile_pixel_count, 33_553_920);
        assert_eq!(plan.tiles[0].gain_buffer_byte_size, 134_215_680);
        assert_eq!(plan.tiles[0].tile_packed_word_count, 16_776_960);
        assert_eq!(
            tile_workgroup_count(plan.tiles[0].tile_packed_word_count).unwrap(),
            65_535
        );
        assert_eq!(plan.tiles[1].base_pixel_offset, 33_553_920);
        assert_eq!(plan.tiles[1].base_packed_word_offset, 16_776_960);
        assert_eq!(plan.tiles[1].tile_pixel_count, 4_194_816);
        assert_eq!(plan.tiles[1].gain_buffer_byte_offset, 134_215_680);
        assert_eq!(plan.tiles[1].gain_buffer_byte_size, 16_779_264);
        assert_eq!(plan.tiles[1].tile_packed_word_count, 2_097_408);
        assert_eq!(
            tile_workgroup_count(plan.tiles[1].tile_packed_word_count).unwrap(),
            8_193
        );
        assert_eq!(plan.max_tile_workgroups, 65_535);
        for tile in &plan.tiles {
            assert!(
                tile_workgroup_count(tile.tile_packed_word_count).unwrap()
                    <= plan.max_compute_workgroups_per_dimension
            );
        }
    }

    #[test]
    fn tile_plan_forced_small_limit_creates_aligned_tiles() {
        let dimensions = FrameDimensions {
            width: 10,
            height: 1,
        };
        let plan = plan_gain_map_tiles(
            dimensions,
            10,
            GpuVignetteBindingLimits {
                max_storage_buffer_binding_size: 16,
                min_storage_buffer_offset_alignment: 16,
                max_compute_workgroups_per_dimension: 65_535,
            },
        )
        .expect("tile plan succeeds");

        assert_eq!(
            plan.tiles
                .iter()
                .map(|tile| tile.tile_pixel_count)
                .collect::<Vec<_>>(),
            vec![4, 4, 2]
        );
        for tile in &plan.tiles {
            assert_eq!(tile.base_pixel_offset % 2, 0);
            assert_eq!(tile.gain_buffer_byte_offset % 16, 0);
            assert!(tile.gain_buffer_byte_size <= 16);
        }
    }

    #[test]
    fn tile_plan_forced_small_dispatch_limit_creates_dispatch_safe_tiles() {
        let max_dispatch_pixels = (VIGNETTE_CORRECT_WORKGROUP_SIZE as usize) * 2;
        let pixel_count = max_dispatch_pixels
            .checked_mul(2)
            .and_then(|value| value.checked_add(2))
            .expect("test pixel count fits");
        let dimensions = FrameDimensions {
            width: u32::try_from(pixel_count).expect("test width fits"),
            height: 1,
        };
        let plan = plan_gain_map_tiles(
            dimensions,
            pixel_count,
            GpuVignetteBindingLimits {
                max_storage_buffer_binding_size: 1_000_000,
                min_storage_buffer_offset_alignment: 8,
                max_compute_workgroups_per_dimension: 1,
            },
        )
        .expect("tile plan succeeds");

        assert_eq!(
            plan.tiles
                .iter()
                .map(|tile| tile.tile_pixel_count)
                .collect::<Vec<_>>(),
            vec![max_dispatch_pixels, max_dispatch_pixels, 2]
        );
        assert_eq!(plan.max_tile_workgroups, 1);
        for tile in &plan.tiles {
            assert_eq!(
                tile_workgroup_count(tile.tile_packed_word_count).unwrap(),
                1
            );
        }
    }

    #[test]
    fn tile_plan_allows_odd_final_pixel() {
        let dimensions = FrameDimensions {
            width: 5,
            height: 1,
        };
        let plan = plan_gain_map_tiles(
            dimensions,
            5,
            GpuVignetteBindingLimits {
                max_storage_buffer_binding_size: 16,
                min_storage_buffer_offset_alignment: 8,
                max_compute_workgroups_per_dimension: 65_535,
            },
        )
        .expect("tile plan succeeds");

        assert_eq!(
            plan.tiles
                .iter()
                .map(|tile| tile.tile_pixel_count)
                .collect::<Vec<_>>(),
            vec![4, 1]
        );
        assert_eq!(plan.tiles[1].tile_packed_word_count, 1);
    }

    #[test]
    fn tile_plan_rejects_impossible_binding_limit() {
        let dimensions = FrameDimensions {
            width: 2,
            height: 1,
        };
        let error = plan_gain_map_tiles(
            dimensions,
            2,
            GpuVignetteBindingLimits {
                max_storage_buffer_binding_size: 3,
                min_storage_buffer_offset_alignment: 256,
                max_compute_workgroups_per_dimension: 65_535,
            },
        )
        .expect_err("impossible binding limit is rejected");

        assert_eq!(
            error,
            VignetteCorrectionError::GpuGainMapBindingTooLarge {
                required_bytes: 8,
                max_binding_bytes: 3,
                frame_dimensions: dimensions,
                pixel_count: 2,
            }
        );
    }

    #[test]
    fn tile_plan_rejects_impossible_workgroup_limit() {
        let dimensions = FrameDimensions {
            width: 2,
            height: 1,
        };
        let error = plan_gain_map_tiles(
            dimensions,
            2,
            GpuVignetteBindingLimits {
                max_storage_buffer_binding_size: 8,
                min_storage_buffer_offset_alignment: 1,
                max_compute_workgroups_per_dimension: 0,
            },
        )
        .expect_err("impossible workgroup limit is rejected");

        assert_eq!(
            error,
            VignetteCorrectionError::GpuComputeWorkgroupLimitTooSmall {
                max_workgroups: 0,
                workgroup_size: VIGNETTE_CORRECT_WORKGROUP_SIZE,
            }
        );
    }

    #[test]
    fn direct_conversion_accepts_e17_out_of_table_regression_case() {
        let map = constant_map(1, 2, &[1.0, 20.0, 1.0, 20.0]);
        let facts = fixed_facts_for_map(
            FrameDimensions {
                width: 2,
                height: 2,
            },
            &map,
            VignetteCorrectionMode::Enabled,
            [0.0; 4],
            u16::MAX,
        );
        let _full_reference = PreparedFullResolutionFixedGainMap::from_fixed_facts(&facts)
            .expect("current full-resolution reference accepts this map");
        let fixed_map = facts.lens_shading_map.as_ref().expect("fixed map");
        validate_compact_direct_conversion_domain(fixed_map, &facts)
            .expect("direct conversion has no dense-table range cap");
    }

    #[test]
    fn direct_conversion_accepts_reference_domain_above_u32_raw_gain() {
        let map = constant_map(1, 1, &[65_536.0; 4]);
        let facts = fixed_facts_for_map(
            FrameDimensions {
                width: 1,
                height: 1,
            },
            &map,
            VignetteCorrectionMode::Enabled,
            [0.0; 4],
            u16::MAX,
        );
        let _full_reference = PreparedFullResolutionFixedGainMap::from_fixed_facts(&facts)
            .expect("current full-resolution reference accepts this raw-gain value");
        let fixed_map = facts.lens_shading_map.as_ref().expect("fixed map");
        validate_compact_direct_conversion_domain(fixed_map, &facts)
            .expect("direct conversion accepts the CPU-reference raw-gain interval");

        let words = compact_gain_words(fixed_map).expect("compact words");
        assert_eq!(&words[..2], &[0, 1]);
    }

    #[test]
    fn direct_integer_conversion_matches_cpu_reference_dense_domain() {
        let map = constant_map(1, 1, &[1.0; 4]);
        let conversion_states = [
            fixed_facts_for_map(
                FrameDimensions {
                    width: 1,
                    height: 1,
                },
                &map,
                VignetteCorrectionMode::Enabled,
                [0.0; 4],
                u16::MAX,
            ),
            fixed_facts_for_map(
                FrameDimensions {
                    width: 1,
                    height: 1,
                },
                &map,
                VignetteCorrectionMode::Enabled,
                [64.0; 4],
                1023,
            ),
            fixed_facts_for_map(
                FrameDimensions {
                    width: 1,
                    height: 1,
                },
                &map,
                VignetteCorrectionMode::Enabled,
                [63.0, 64.0, 65.0, 66.0],
                1023,
            ),
        ];

        let mut compared = 0_u64;
        let mut changed = 0_u64;
        let mut max_difference = 0_i64;
        let mut first_mismatch = None;

        for facts in &conversion_states {
            let max_raw = max_reference_u32_raw_gain_q(facts);
            let mut candidates = vec![
                0,
                1,
                65_535,
                65_536,
                65_537,
                u64::from(u32::MAX) - 1,
                u64::from(u32::MAX),
                u64::from(u32::MAX) + 1,
                65_536_u64 * 20,
                max_raw.saturating_sub(1),
                max_raw,
            ];
            for index in 0..350_000_u64 {
                candidates.push((u128::from(index) * u128::from(max_raw) / 349_999) as u64);
            }
            candidates.sort_unstable();
            candidates.dedup();

            for raw_gain_q in candidates {
                let Ok(reference) = motioncam_compatible_gain_q16_from_raw_gain_q(
                    i64::try_from(raw_gain_q).expect("test raw gain fits i64"),
                    facts,
                    0,
                    0,
                ) else {
                    continue;
                };
                if reference < 0 || reference > i64::from(u32::MAX) {
                    continue;
                }
                let actual = direct_integer_motioncam_gain_q16(raw_gain_q, facts)
                    .expect("direct integer conversion");
                let difference = i64::from(actual) - reference;
                compared += 1;
                if difference != 0 {
                    changed += 1;
                    max_difference = max_difference.max(difference.abs());
                    first_mismatch.get_or_insert((raw_gain_q, actual, reference));
                }
            }
        }

        assert!(compared >= 1_000_000, "compared only {compared} values");
        assert_eq!(
            (changed, max_difference, first_mismatch),
            (0, 0, None),
            "direct integer conversion diverged from the f64 reference"
        );
    }

    fn max_reference_u32_raw_gain_q(facts: &FixedPointVignetteInputFacts<'_>) -> u64 {
        let mut high = 65_536_u64;
        loop {
            let reference = motioncam_compatible_gain_q16_from_raw_gain_q(
                i64::try_from(high).expect("test high fits i64"),
                facts,
                0,
                0,
            )
            .expect("reference conversion");
            if reference > i64::from(u32::MAX) {
                break;
            }
            high = high.checked_mul(2).expect("test high remains bounded");
        }

        let mut low = high / 2;
        while low + 1 < high {
            let mid = low + ((high - low) / 2);
            let reference = motioncam_compatible_gain_q16_from_raw_gain_q(
                i64::try_from(mid).expect("test mid fits i64"),
                facts,
                0,
                0,
            )
            .expect("reference conversion");
            if reference <= i64::from(u32::MAX) {
                low = mid;
            } else {
                high = mid;
            }
        }

        low
    }

    fn direct_integer_motioncam_gain_q16(
        raw_gain_q: u64,
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Option<u32> {
        let (scale_num, scale_shift) =
            f32_power2_rational(facts.pixel_domain.source_to_corrected_scale)?;
        let (strength_num, strength_shift) =
            f32_power2_rational(MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH)?;
        let delta = raw_gain_q.saturating_sub(65_536);
        let unscaled =
            (65_536_u128 << strength_shift) + (u128::from(delta) * u128::from(strength_num));
        let scaled = unscaled.checked_mul(u128::from(scale_num))?;
        let shift = scale_shift.checked_add(strength_shift)?;
        let rounded = scaled.checked_add(1_u128 << (shift - 1))? >> shift;
        u32::try_from(rounded).ok()
    }

    #[test]
    fn shader_contract_declares_packed_u16_bindings() {
        assert!(VIGNETTE_CORRECT_PACKED_U16_WGSL.contains("@binding(0)"));
        assert!(VIGNETTE_CORRECT_PACKED_U16_WGSL.contains("input_words: array<u32>"));
        assert!(VIGNETTE_CORRECT_PACKED_U16_WGSL.contains("compact_gains_q: array<u32>"));
        assert!(VIGNETTE_CORRECT_PACKED_U16_WGSL.contains("output_words: array<u32>"));
        assert!(VIGNETTE_CORRECT_PACKED_U16_WGSL.contains("base_pixel_offset"));
        assert!(VIGNETTE_CORRECT_PACKED_U16_WGSL.contains("tile_pixel_count"));
        assert!(VIGNETTE_CORRECT_PACKED_U16_WGSL.contains("mul_u32_u32_to_u64"));
        assert!(VIGNETTE_CORRECT_PACKED_U16_WGSL.contains("scale_num"));
        assert!(VIGNETTE_CORRECT_PACKED_U16_WGSL.contains("strength_num"));
        assert!(!VIGNETTE_CORRECT_PACKED_U16_WGSL.contains("gain_conversion_q16"));
    }

    fn fixed_facts_for_map<'a>(
        dimensions: FrameDimensions,
        map: &'a LensShadingMap,
        mode: VignetteCorrectionMode,
        black_level: [f32; BAYER_CFA_PLANE_COUNT],
        output_white_level: u16,
    ) -> FixedPointVignetteInputFacts<'a> {
        let fixed_map =
            PreparedFixedLensShadingMap::from_typed_map(map).expect("fixed map prepares");
        let facts = VignetteCorrectionInputFacts::new(
            mode,
            VignetteCoordinateMapping::VisibleFrame,
            dimensions,
            BayerPattern::Rggb,
            None,
            black_level,
            output_white_level,
        )
        .expect("facts validate");

        FixedPointVignetteInputFacts::from_input_facts_with_fixed_map(&facts, Some(fixed_map))
            .expect("fixed facts validate")
    }

    fn constant_map(width: u32, height: u32, gains: &[f32; 4]) -> LensShadingMap {
        let pixel_count = (width as usize) * (height as usize);
        LensShadingMap::new(
            width,
            height,
            gains.iter().map(|gain| vec![*gain; pixel_count]).collect(),
        )
        .expect("valid lens shading map")
    }
}
