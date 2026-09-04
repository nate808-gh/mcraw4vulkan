use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use mcraw4vulkan_core::{BayerPattern, FrameDimensions, FrameNumber, FramePayloadLayout};
use mcraw4vulkan_cpu::{CpuFrameDecoder, DecodeFrameTimings};
use mcraw4vulkan_gpu::{
    GpuDecodeBackend, GpuDecodeConfig, GpuDecodeScratch, OptionalGpuVignetteCorrection,
};
use mcraw4vulkan_mcrawcontainer::{
    ColorIlluminant, ContainerMetadata, FrameMetadata, LensShadingMap, McrawContainer,
    SensorArrangement,
    payload_reader::{PayloadFeeder, PayloadFrame, PayloadReadPlan},
};
use mcraw4vulkan_render::{
    GpuPreviewRenderEncodeOutput, GpuPreviewTextureRenderer, GpuRenderCalibrationIlluminant,
    GpuRenderColorMetadata, GpuRenderColorMode, GpuRenderColorParams, PreviewRenderConfig,
    RenderSampleDomain,
};
use mcraw4vulkan_vignette::{
    FixedPointVignetteInputFacts, GpuUploadedFullResolutionGainMap, GpuVignetteCorrectionParams,
    GpuVignetteCorrector, PreparedFixedLensShadingMap, VignetteCoordinateMapping,
    VignetteCorrectionInputFacts, VignetteCorrectionMode, VignetteGainMapFingerprint,
};

use super::types::{
    DisplayGpuExecutionProfile, DisplayMeasurementRequest, DisplayMetrics, DisplayWarmupMetrics,
    DisplayWarmupStatus, FrameCountMetrics, GpuStageMetrics, InputClipMetrics, MeasurementResult,
    MeasurementStatus, MeasurementTiming, PayloadMetrics,
};

// Display measurement stops at preview texture encoding: it performs no
// full-frame readback and produces no output byte stream, unlike PIPE.
pub struct DisplayMeasurementRunner;

impl DisplayMeasurementRunner {
    pub fn new() -> Self {
        Self
    }

    pub fn measure(&self, request: DisplayMeasurementRequest) -> Result<MeasurementResult> {
        request.display.validate()?;
        if request.display.mode.uses_cpu_decode() {
            self.measure_cpu_display(request)
        } else {
            self.measure_gpu_display(request)
        }
    }

    fn measure_cpu_display(&self, request: DisplayMeasurementRequest) -> Result<MeasurementResult> {
        let setup_start = Instant::now();
        let container = McrawContainer::open(&request.run.input_path)?;
        let input_dimensions = first_selected_dimensions(&container, &request)?;
        let (mut payload_feeder, resolved_payload_note) =
            display_payload_feeder_for_frames(&container, &request, &request.run.selected_frames)?;
        let mut cpu_decoder = CpuFrameDecoder::new();
        let mut backend = GpuDecodeBackend::new_blocking(GpuDecodeConfig {
            backend_preference: request.gpu.backend_preference,
            enable_gpu_timestamps: false,
        })?;
        let mut preview = GpuPreviewTextureRenderer::new(backend.device())?;
        let mut uploader = CpuPreviewUploadBuffer::default();
        let setup = setup_start.elapsed();

        let warmup = run_cpu_display_warmup(
            &container,
            &request,
            &mut cpu_decoder,
            &mut backend,
            &mut preview,
            &mut uploader,
        )?;

        let run_start = Instant::now();
        let loop_output = run_cpu_display_loop(
            &container,
            &mut payload_feeder,
            &mut cpu_decoder,
            &mut backend,
            &mut preview,
            &mut uploader,
            &request,
        )?;
        let run = run_start.elapsed();
        let flush_start = Instant::now();
        let payload_metrics = PayloadMetrics::from_feeder_stats(payload_feeder.finish()?);
        let flush = flush_start.elapsed();

        Ok(display_result(DisplayResultParts {
            request,
            setup,
            run,
            flush,
            frames_processed: loop_output.frames_processed,
            payload: payload_metrics,
            gpu: loop_output.gpu,
            preview_width: loop_output.preview_width,
            preview_height: loop_output.preview_height,
            preview_target_bytes_estimate: loop_output.preview_target_bytes_estimate,
            input_dimensions,
            display_warmup: warmup,
            notes: vec![
                resolved_payload_note,
                "window-sized aspect-fit no-window preview texture proxy".to_string(),
                "CPU decode is context only".to_string(),
            ],
        }))
    }

    fn measure_gpu_display(&self, request: DisplayMeasurementRequest) -> Result<MeasurementResult> {
        let setup_start = Instant::now();
        let container = McrawContainer::open(&request.run.input_path)?;
        let input_dimensions = first_selected_dimensions(&container, &request)?;
        let (mut payload_feeder, resolved_payload_note) =
            display_payload_feeder_for_frames(&container, &request, &request.run.selected_frames)?;
        let mut backend = GpuDecodeBackend::new_blocking(GpuDecodeConfig {
            backend_preference: request.gpu.backend_preference,
            enable_gpu_timestamps: false,
        })?;
        let mut preview = GpuPreviewTextureRenderer::new(backend.device())?;
        let mut scratch = GpuDecodeScratch::new();
        let use_vignette = request.display.mode.uses_gpu_vignette();
        let mut vignette_corrector = if use_vignette {
            Some(backend.create_vignette_corrector()?)
        } else {
            None
        };
        let mut gain_cache = None;
        let setup = setup_start.elapsed();

        let warmup = run_gpu_display_warmup(
            &container,
            &request,
            &mut backend,
            &mut preview,
            &mut scratch,
            GpuDisplayVignetteState {
                enabled: use_vignette,
                vignette_corrector: &mut vignette_corrector,
                gain_cache: &mut gain_cache,
            },
        )?;

        let run_start = Instant::now();
        let loop_output = run_gpu_display_loop(
            &container,
            &mut payload_feeder,
            &mut backend,
            &mut preview,
            &mut scratch,
            &request,
            GpuDisplayVignetteState {
                enabled: use_vignette,
                vignette_corrector: &mut vignette_corrector,
                gain_cache: &mut gain_cache,
            },
        )?;
        let run = run_start.elapsed();
        let flush_start = Instant::now();
        let payload_metrics = PayloadMetrics::from_feeder_stats(payload_feeder.finish()?);
        let flush = flush_start.elapsed();

        Ok(display_result(DisplayResultParts {
            request,
            setup,
            run,
            flush,
            frames_processed: loop_output.frames_processed,
            payload: payload_metrics,
            gpu: loop_output.gpu,
            preview_width: loop_output.preview_width,
            preview_height: loop_output.preview_height,
            preview_target_bytes_estimate: loop_output.preview_target_bytes_estimate,
            input_dimensions,
            display_warmup: warmup,
            notes: vec![
                resolved_payload_note,
                "window-sized aspect-fit no-window preview texture proxy".to_string(),
                "no full-frame raw output readback".to_string(),
            ],
        }))
    }
}

impl Default for DisplayMeasurementRunner {
    fn default() -> Self {
        Self::new()
    }
}

struct DisplayResultParts {
    request: DisplayMeasurementRequest,
    setup: Duration,
    run: Duration,
    flush: Duration,
    frames_processed: usize,
    payload: PayloadMetrics,
    gpu: GpuStageMetrics,
    preview_width: u32,
    preview_height: u32,
    preview_target_bytes_estimate: u64,
    input_dimensions: FrameDimensions,
    display_warmup: DisplayWarmupMetrics,
    notes: Vec<String>,
}

fn display_result(parts: DisplayResultParts) -> MeasurementResult {
    let DisplayResultParts {
        request,
        setup,
        run,
        flush,
        frames_processed,
        payload,
        gpu,
        preview_width,
        preview_height,
        preview_target_bytes_estimate,
        input_dimensions,
        display_warmup,
        notes,
    } = parts;

    MeasurementResult {
        sink: request.display.mode.sink(),
        status: MeasurementStatus::Ok,
        failure_stage: None,
        timing: MeasurementTiming { setup, run, flush },
        frames: FrameCountMetrics {
            frames_requested: request.run.frames_requested,
            frames_processed,
        },
        input: InputClipMetrics {
            input_basename: input_basename(&request.run.input_path),
            frame_width: input_dimensions.width,
            frame_height: input_dimensions.height,
        },
        payload,
        gpu,
        display: DisplayMetrics {
            preview_width,
            preview_height,
            preview_target_bytes_estimate,
            full_frame_gpu_readback_performed: false,
            readback_kind: super::types::ReadbackKind::None,
            output_bytes: 0,
            output_bytes_kind: super::types::OutputBytesKind::None,
        },
        display_warmup,
        notes,
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct DisplayLoopOutput {
    frames_processed: usize,
    preview_target_bytes_estimate: u64,
    preview_width: u32,
    preview_height: u32,
    gpu: GpuStageMetrics,
}

fn display_payload_feeder_for_frames(
    container: &McrawContainer,
    request: &DisplayMeasurementRequest,
    frames: &[FrameNumber],
) -> Result<(PayloadFeeder, String)> {
    let payload_plan = PayloadReadPlan::from_core_frame_numbers(container, frames)?;
    let resolved_payload = request.payload.resolve(&payload_plan)?;
    let payload_feeder = PayloadFeeder::spawn(
        &request.run.input_path,
        payload_plan,
        resolved_payload.feeder_options,
    )?;
    Ok((payload_feeder, resolved_payload.note))
}

fn display_warmup_frame_list(selected_frames: &[FrameNumber], count: usize) -> Vec<FrameNumber> {
    if count == 0 || selected_frames.is_empty() {
        return Vec::new();
    }
    selected_frames
        .iter()
        .copied()
        .cycle()
        .take(count)
        .collect()
}

fn run_cpu_display_warmup(
    container: &McrawContainer,
    request: &DisplayMeasurementRequest,
    cpu_decoder: &mut CpuFrameDecoder,
    backend: &mut GpuDecodeBackend,
    preview: &mut GpuPreviewTextureRenderer,
    uploader: &mut CpuPreviewUploadBuffer,
) -> Result<DisplayWarmupMetrics> {
    let warmup_frames = display_warmup_frame_list(
        &request.run.selected_frames,
        request.run.display_warmup_frames,
    );
    if warmup_frames.is_empty() {
        return Ok(DisplayWarmupMetrics::not_requested());
    }
    let warmup_start = Instant::now();
    let (mut warmup_feeder, _) =
        display_payload_feeder_for_frames(container, request, &warmup_frames)?;
    let output = run_cpu_display_loop(
        container,
        &mut warmup_feeder,
        cpu_decoder,
        backend,
        preview,
        uploader,
        request,
    )?;
    let _ = warmup_feeder.finish()?;
    Ok(DisplayWarmupMetrics {
        frames_requested: request.run.display_warmup_frames,
        frames_processed: output.frames_processed,
        duration: warmup_start.elapsed(),
        status: DisplayWarmupStatus::Ok,
    })
}

fn run_cpu_display_loop(
    container: &McrawContainer,
    payload_feeder: &mut PayloadFeeder,
    cpu_decoder: &mut CpuFrameDecoder,
    backend: &mut GpuDecodeBackend,
    preview: &mut GpuPreviewTextureRenderer,
    uploader: &mut CpuPreviewUploadBuffer,
    request: &DisplayMeasurementRequest,
) -> Result<DisplayLoopOutput> {
    let mut output = DisplayLoopOutput::default();
    while let Some(payload) = payload_feeder.next_frame()? {
        let preview_info = process_cpu_display_payload(
            container,
            payload,
            cpu_decoder,
            backend,
            preview,
            uploader,
            request,
        )?;
        output.preview_width = preview_info.width;
        output.preview_height = preview_info.height;
        output.preview_target_bytes_estimate = output
            .preview_target_bytes_estimate
            .saturating_add(preview_info.preview_target_bytes_estimate);
        output.frames_processed += 1;
    }
    Ok(output)
}

fn process_cpu_display_payload(
    container: &McrawContainer,
    payload: PayloadFrame,
    cpu_decoder: &mut CpuFrameDecoder,
    backend: &mut GpuDecodeBackend,
    preview: &mut GpuPreviewTextureRenderer,
    uploader: &mut CpuPreviewUploadBuffer,
    request: &DisplayMeasurementRequest,
) -> Result<CpuPreviewInfo> {
    let frame_number = payload.frame_number_core()?;
    let metadata = container.frame_metadata(frame_number)?;
    let facts =
        DisplayFrameFacts::from_typed_metadata(container.container_metadata(), metadata, false)?;
    let mut timings = DecodeFrameTimings::default();
    cpu_decoder.prepare_compressed(payload.data.len());
    payload.data.copy_into(cpu_decoder.compressed_mut());
    let (decoded, _) = cpu_decoder
        .decode_loaded_payload_to_decoded_bayer_u16_frame(facts.dimensions, &mut timings)?;
    let pixel_bytes = decoded.into_owned_le_bytes();
    render_cpu_bytes_to_preview(
        backend,
        preview,
        uploader,
        &facts,
        request,
        false,
        &pixel_bytes,
    )
}

fn run_gpu_display_warmup(
    container: &McrawContainer,
    request: &DisplayMeasurementRequest,
    backend: &mut GpuDecodeBackend,
    preview: &mut GpuPreviewTextureRenderer,
    scratch: &mut GpuDecodeScratch,
    vignette: GpuDisplayVignetteState<'_>,
) -> Result<DisplayWarmupMetrics> {
    let warmup_frames = display_warmup_frame_list(
        &request.run.selected_frames,
        request.run.display_warmup_frames,
    );
    if warmup_frames.is_empty() {
        return Ok(DisplayWarmupMetrics::not_requested());
    }
    let warmup_start = Instant::now();
    let (mut warmup_feeder, _) =
        display_payload_feeder_for_frames(container, request, &warmup_frames)?;
    let output = run_gpu_display_loop(
        container,
        &mut warmup_feeder,
        backend,
        preview,
        scratch,
        request,
        vignette,
    )?;
    let _ = warmup_feeder.finish()?;
    Ok(DisplayWarmupMetrics {
        frames_requested: request.run.display_warmup_frames,
        frames_processed: output.frames_processed,
        duration: warmup_start.elapsed(),
        status: DisplayWarmupStatus::Ok,
    })
}

fn run_gpu_display_loop(
    container: &McrawContainer,
    payload_feeder: &mut PayloadFeeder,
    backend: &mut GpuDecodeBackend,
    preview: &mut GpuPreviewTextureRenderer,
    scratch: &mut GpuDecodeScratch,
    request: &DisplayMeasurementRequest,
    mut vignette: GpuDisplayVignetteState<'_>,
) -> Result<DisplayLoopOutput> {
    let mut output = DisplayLoopOutput::default();
    while let Some(payload) = payload_feeder.next_frame()? {
        let frame_output = process_gpu_display_payload(
            container,
            payload,
            backend,
            preview,
            scratch,
            request,
            &mut vignette,
        )?;
        output.preview_width = frame_output.preview_width;
        output.preview_height = frame_output.preview_height;
        output.preview_target_bytes_estimate = output
            .preview_target_bytes_estimate
            .saturating_add(frame_output.preview_target_bytes_estimate);
        output.gpu.dispatch_s += frame_output.gpu.dispatch_s;
        output.gpu.wait_s += frame_output.gpu.wait_s;
        output.gpu.full_frame_readback_s = 0.0;
        output.frames_processed += 1;
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy)]
struct GpuDisplayFrameOutput {
    preview_width: u32,
    preview_height: u32,
    preview_target_bytes_estimate: u64,
    gpu: GpuStageMetrics,
}

struct GpuDisplayVignetteState<'a> {
    enabled: bool,
    vignette_corrector: &'a mut Option<GpuVignetteCorrector>,
    gain_cache: &'a mut Option<UploadedGainMapCache>,
}

fn process_gpu_display_payload(
    container: &McrawContainer,
    payload: PayloadFrame,
    backend: &mut GpuDecodeBackend,
    preview: &mut GpuPreviewTextureRenderer,
    scratch: &mut GpuDecodeScratch,
    request: &DisplayMeasurementRequest,
    vignette: &mut GpuDisplayVignetteState<'_>,
) -> Result<GpuDisplayFrameOutput> {
    let frame_number = payload.frame_number_core()?;
    let metadata = container.frame_metadata(frame_number)?;
    let facts = DisplayFrameFacts::from_typed_metadata(
        container.container_metadata(),
        metadata,
        vignette.enabled,
    )?;
    let correction = if vignette.enabled {
        let fixed_facts = facts.fixed_facts_enabled()?;
        let uploaded = ensure_uploaded_gain_map(
            vignette.gain_cache,
            backend,
            vignette
                .vignette_corrector
                .as_mut()
                .context("vignette corrector missing")?,
            &fixed_facts,
        )?;
        let params = GpuVignetteCorrectionParams::from_fixed_facts(&fixed_facts)?;
        Some((uploaded, params))
    } else {
        None
    };
    let preview_config = facts.preview_config(request, vignette.enabled)?;
    let output = if let Some((uploaded, params)) = correction {
        let corrector = vignette
            .vignette_corrector
            .as_mut()
            .context("vignette corrector missing")?;
        let correction = Some(OptionalGpuVignetteCorrection {
            corrector,
            uploaded_gain_map: &uploaded,
            params,
        });
        decode_gpu_display_payload(
            backend,
            preview,
            scratch,
            GpuDisplayDecodeRequest {
                payload: payload.data.as_slice(),
                facts: &facts,
                correction,
                preview_config,
                execution_profile: request.gpu.execution_profile,
            },
        )?
    } else {
        decode_gpu_display_payload(
            backend,
            preview,
            scratch,
            GpuDisplayDecodeRequest {
                payload: payload.data.as_slice(),
                facts: &facts,
                correction: None,
                preview_config,
                execution_profile: request.gpu.execution_profile,
            },
        )?
    };
    let info = output.stage.encoded.info;
    Ok(GpuDisplayFrameOutput {
        preview_width: info.output_dimensions.width,
        preview_height: info.output_dimensions.height,
        preview_target_bytes_estimate: info.texture_bytes,
        gpu: GpuStageMetrics {
            dispatch_s: output.timings.total.as_secs_f64(),
            wait_s: output.timings.wait_map.as_secs_f64(),
            full_frame_readback_s: 0.0,
        },
    })
}

struct GpuDisplayDecodeRequest<'a> {
    payload: &'a [u8],
    facts: &'a DisplayFrameFacts,
    correction: Option<OptionalGpuVignetteCorrection<'a>>,
    preview_config: PreviewRenderConfig,
    execution_profile: DisplayGpuExecutionProfile,
}

fn decode_gpu_display_payload(
    backend: &mut GpuDecodeBackend,
    preview: &mut GpuPreviewTextureRenderer,
    scratch: &mut GpuDecodeScratch,
    request: GpuDisplayDecodeRequest<'_>,
) -> Result<mcraw4vulkan_gpu::GpuNoReadbackGpuStageOutput<PreviewStageOutput>> {
    match request.facts.payload_layout {
        FramePayloadLayout::CompressedRawcodecType7 => {
            if request.execution_profile == DisplayGpuExecutionProfile::WorkPlanScratchReuse {
                backend
                    .decode_raw_payload_packed_u16_gpu_stage_no_readback_with_vignette_with_scratch(
                        request.payload,
                        request.facts.dimensions,
                        request.correction,
                        scratch,
                        |device, queue, encoder, decoded| {
                            encode_preview_stage(
                                preview,
                                device,
                                queue,
                                encoder,
                                decoded.buffer,
                                decoded.byte_len,
                                request.preview_config,
                            )
                        },
                    )
            } else {
                backend.decode_raw_payload_packed_u16_gpu_stage_no_readback_with_vignette(
                    request.payload,
                    request.facts.dimensions,
                    request.correction,
                    |device, queue, encoder, decoded| {
                        encode_preview_stage(
                            preview,
                            device,
                            queue,
                            encoder,
                            decoded.buffer,
                            decoded.byte_len,
                            request.preview_config,
                        )
                    },
                )
            }
        }
        FramePayloadLayout::BinnedRaw16Type6 { row_stride } => {
            if request.execution_profile == DisplayGpuExecutionProfile::WorkPlanScratchReuse {
                backend.decode_legacy_raw16_payload_packed_u16_gpu_stage_no_readback_with_vignette_with_scratch(
                    request.payload,
                    request.facts.dimensions,
                    row_stride,
                    request.correction,
                    scratch,
                    |device, queue, encoder, decoded| {
                        encode_preview_stage(
                            preview,
                            device,
                            queue,
                            encoder,
                            decoded.buffer,
                            decoded.byte_len,
                            request.preview_config,
                        )
                    },
                )
            } else {
                backend.decode_legacy_raw16_payload_packed_u16_gpu_stage_no_readback_with_vignette(
                    request.payload,
                    request.facts.dimensions,
                    row_stride,
                    request.correction,
                    |device, queue, encoder, decoded| {
                        encode_preview_stage(
                            preview,
                            device,
                            queue,
                            encoder,
                            decoded.buffer,
                            decoded.byte_len,
                            request.preview_config,
                        )
                    },
                )
            }
        }
    }
}

fn first_selected_dimensions(
    container: &McrawContainer,
    request: &DisplayMeasurementRequest,
) -> Result<FrameDimensions> {
    let first_frame = request
        .run
        .selected_frames
        .first()
        .copied()
        .context("display measurement selected frame range is empty")?;
    Ok(container.frame_metadata(first_frame)?.dimensions)
}

fn input_basename(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

#[derive(Debug, Clone)]
struct PreviewStageOutput {
    encoded: GpuPreviewRenderEncodeOutput,
}

fn encode_preview_stage(
    preview: &mut GpuPreviewTextureRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    input_buffer: &wgpu::Buffer,
    input_buffer_bytes: u64,
    preview_config: PreviewRenderConfig,
) -> Result<PreviewStageOutput> {
    let encoded = preview
        .encode_render_to_texture(
            device,
            queue,
            encoder,
            input_buffer,
            input_buffer_bytes,
            preview_config,
        )
        .map_err(|error| anyhow!("failed to encode GPU preview texture: {error}"))?;
    Ok(PreviewStageOutput { encoded })
}

#[derive(Default)]
struct CpuPreviewUploadBuffer {
    buffer: Option<wgpu::Buffer>,
    byte_len: u64,
}

#[derive(Debug, Clone, Copy)]
struct CpuPreviewInfo {
    width: u32,
    height: u32,
    preview_target_bytes_estimate: u64,
}

fn render_cpu_bytes_to_preview(
    backend: &mut GpuDecodeBackend,
    preview: &mut GpuPreviewTextureRenderer,
    uploader: &mut CpuPreviewUploadBuffer,
    facts: &DisplayFrameFacts,
    request: &DisplayMeasurementRequest,
    vignette_applied: bool,
    pixel_bytes: &[u8],
) -> Result<CpuPreviewInfo> {
    let byte_len =
        u64::try_from(pixel_bytes.len()).context("CPU preview byte len overflows u64")?;
    if uploader.buffer.is_none() || uploader.byte_len < byte_len {
        uploader.buffer = Some(backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("mcraw4vulkan display measurement CPU preview upload buffer"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        uploader.byte_len = byte_len;
    }
    let buffer = uploader
        .buffer
        .as_ref()
        .context("CPU preview upload buffer missing")?;
    backend.queue().write_buffer(buffer, 0, pixel_bytes);
    let preview_config = facts.preview_config(request, vignette_applied)?;
    let mut encoder = backend
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mcraw4vulkan display measurement CPU preview render encoder"),
        });
    let encoded = preview.encode_render_to_texture(
        backend.device(),
        backend.queue(),
        &mut encoder,
        buffer,
        byte_len,
        preview_config,
    )?;
    let submission_index = backend.queue().submit(Some(encoder.finish()));
    backend
        .device()
        .poll(wgpu::Maintain::wait_for(submission_index));
    Ok(CpuPreviewInfo {
        width: encoded.info.output_dimensions.width,
        height: encoded.info.output_dimensions.height,
        preview_target_bytes_estimate: encoded.info.texture_bytes,
    })
}

#[derive(Debug)]
struct DisplayFrameFacts {
    dimensions: FrameDimensions,
    payload_layout: FramePayloadLayout,
    lens_shading_map: Option<LensShadingMap>,
    input_black_level: [f32; 4],
    output_white_level: u16,
    source_bits: u16,
    bayer_pattern: BayerPattern,
    as_shot_neutral: Option<[f64; 3]>,
    color_metadata: GpuRenderColorMetadata,
}

impl DisplayFrameFacts {
    fn from_typed_metadata(
        container_metadata: &ContainerMetadata,
        frame_metadata: &FrameMetadata,
        require_lens_map: bool,
    ) -> Result<Self> {
        let input_black_level = extract_input_black_level(container_metadata, frame_metadata)?;
        let output_white_level = extract_output_white_level(container_metadata, frame_metadata)?;
        if require_lens_map && frame_metadata.lens_shading_map.is_none() {
            bail!("frame metadata is missing lensShadingMap");
        }
        let payload_layout = frame_metadata.payload_layout()?;
        Ok(Self {
            dimensions: frame_metadata.dimensions,
            payload_layout,
            lens_shading_map: frame_metadata.lens_shading_map.clone(),
            input_black_level,
            output_white_level,
            source_bits: source_bits_from_white_level(output_white_level),
            bayer_pattern: bayer_pattern_from_sensor_arrangement(
                &container_metadata.sensor_arrangement,
            )?,
            as_shot_neutral: frame_metadata.as_shot_neutral,
            color_metadata: color_metadata_from_container(container_metadata),
        })
    }

    fn fixed_facts_enabled(&self) -> Result<FixedPointVignetteInputFacts<'_>> {
        let lens_shading_map = self
            .lens_shading_map
            .as_ref()
            .context("frame metadata is missing lensShadingMap")?;
        let fixed_map = PreparedFixedLensShadingMap::from_typed_map(lens_shading_map)?;
        let input_facts = VignetteCorrectionInputFacts::new(
            VignetteCorrectionMode::Enabled,
            VignetteCoordinateMapping::VisibleFrame,
            self.dimensions,
            self.bayer_pattern,
            None,
            self.input_black_level,
            self.output_white_level,
        )?;
        FixedPointVignetteInputFacts::from_input_facts_with_fixed_map(&input_facts, Some(fixed_map))
            .context("failed to build fixed-point vignette input facts")
    }

    fn preview_config(
        &self,
        request: &DisplayMeasurementRequest,
        vignette_applied: bool,
    ) -> Result<PreviewRenderConfig> {
        let color = GpuRenderColorParams::from_metadata(
            GpuRenderColorMode::MetadataSrgb,
            self.as_shot_neutral,
            self.color_metadata,
        )
        .map_err(|error| anyhow!("failed to build preview color params: {error}"))?;
        Ok(PreviewRenderConfig {
            dimensions: self.dimensions,
            bayer_pattern: self.bayer_pattern,
            source_bits: self.source_bits,
            black_level: if vignette_applied {
                [0.0; 4]
            } else {
                self.input_black_level
            },
            white_level: f32::from(self.output_white_level),
            sample_domain: if vignette_applied {
                RenderSampleDomain::motioncam_compatible_pixel_v1(f32::from(
                    self.output_white_level,
                ))
            } else {
                RenderSampleDomain::raw(f32::from(self.output_white_level))
            },
            color,
            texture_format: request.display.texture_format,
            transfer_mode: request.display.transfer_mode,
            scale_mode: request.display.proxy.preview_scale_mode(self.dimensions),
        })
    }
}

struct UploadedGainMapCache {
    fingerprint: VignetteGainMapFingerprint,
    uploaded: GpuUploadedFullResolutionGainMap,
}

fn ensure_uploaded_gain_map(
    cache: &mut Option<UploadedGainMapCache>,
    backend: &GpuDecodeBackend,
    corrector: &mut GpuVignetteCorrector,
    facts: &FixedPointVignetteInputFacts<'_>,
) -> Result<GpuUploadedFullResolutionGainMap> {
    let fingerprint = VignetteGainMapFingerprint::from_fixed_facts(facts)?;
    if let Some(entry) = cache
        .as_ref()
        .filter(|entry| entry.fingerprint == fingerprint)
    {
        return Ok(entry.uploaded.clone());
    }
    let uploaded = backend.upload_compact_vignette_gain_map(corrector, facts)?;
    *cache = Some(UploadedGainMapCache {
        fingerprint,
        uploaded: uploaded.clone(),
    });
    Ok(uploaded)
}

fn extract_input_black_level(
    container_metadata: &ContainerMetadata,
    frame_metadata: &FrameMetadata,
) -> Result<[f32; 4]> {
    let values = frame_metadata
        .dynamic_black_level
        .or_else(|| container_metadata.black_level.map(|level| level.values))
        .context("missing dynamicBlackLevel and container blackLevel")?;
    let mut output = [0.0_f32; 4];
    for (index, value) in values.into_iter().enumerate() {
        if !value.is_finite() || value < 0.0 || value > f64::from(f32::MAX) {
            bail!("black level {index} must be finite, non-negative, and fit f32");
        }
        output[index] = value as f32;
    }
    Ok(output)
}

fn extract_output_white_level(
    container_metadata: &ContainerMetadata,
    frame_metadata: &FrameMetadata,
) -> Result<u16> {
    let values = frame_metadata
        .dynamic_white_level
        .map(|level| [level; 4])
        .or_else(|| container_metadata.white_level.map(|level| level.values))
        .context("missing dynamicWhiteLevel and container whiteLevel")?;
    let first = values[0];
    if !first.is_finite() || first < 0.0 || first > f64::from(u16::MAX) {
        bail!("invalid white level value {first}");
    }
    for value in values {
        if (value - first).abs() > 0.000001 {
            bail!("per-plane white levels are not uniform: {values:?}");
        }
    }
    let rounded = first.round();
    if (rounded - first).abs() > 0.000001 {
        bail!("white level value is not an integer: {first}");
    }
    Ok(rounded as u16)
}

fn source_bits_from_white_level(white_level: u16) -> u16 {
    let levels = u32::from(white_level).saturating_add(1).max(2);
    (u32::BITS - (levels - 1).leading_zeros()) as u16
}

fn bayer_pattern_from_sensor_arrangement(arrangement: &SensorArrangement) -> Result<BayerPattern> {
    match arrangement {
        SensorArrangement::Rggb => Ok(BayerPattern::Rggb),
        SensorArrangement::Grbg => Ok(BayerPattern::Grbg),
        SensorArrangement::Gbrg => Ok(BayerPattern::Gbrg),
        SensorArrangement::Bggr => Ok(BayerPattern::Bggr),
        other => bail!("unsupported sensor arrangement for display measurement: {other:?}"),
    }
}

fn color_metadata_from_container(container_metadata: &ContainerMetadata) -> GpuRenderColorMetadata {
    GpuRenderColorMetadata {
        color_matrix1: container_metadata.color_matrix1.map(|matrix| matrix.values),
        color_matrix2: container_metadata.color_matrix2.map(|matrix| matrix.values),
        forward_matrix1: container_metadata
            .forward_matrix1
            .map(|matrix| matrix.values),
        forward_matrix2: container_metadata
            .forward_matrix2
            .map(|matrix| matrix.values),
        illuminant1: container_metadata
            .color_illuminant1
            .as_ref()
            .map(render_illuminant_from_container),
        illuminant2: container_metadata
            .color_illuminant2
            .as_ref()
            .map(render_illuminant_from_container),
    }
}

fn render_illuminant_from_container(
    illuminant: &ColorIlluminant,
) -> GpuRenderCalibrationIlluminant {
    match illuminant {
        ColorIlluminant::StandardA => GpuRenderCalibrationIlluminant::StandardA,
        ColorIlluminant::D65 => GpuRenderCalibrationIlluminant::D65,
        ColorIlluminant::Other(_) => GpuRenderCalibrationIlluminant::Other,
    }
}
