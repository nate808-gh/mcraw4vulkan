use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use mcraw4vulkan_core::{
    BayerPattern, DecodedBayerU16Frame, FrameDimensions, FrameNumber, FramePayloadLayout, FrameRate,
};
use mcraw4vulkan_cpu::raw_decoder::validate_frame_payload;
use mcraw4vulkan_cpu::{CpuFrameDecoder, DecodeFrameTimings};
use mcraw4vulkan_dngwriter::{
    DngFrameDescription, DngOutputCorrection, DngSinkDecodeSource, DngSinkFrame,
    DngSinkVignetteMode, DngWriter, DngWriterConfig, build_dng_frame_description_for_sink,
};
use mcraw4vulkan_gpu::{
    GpuBackendPreference, GpuDecodeBackend, GpuDecodeConfig, GpuMappedRingFrame,
    GpuMappedRingVignetteCorrection, OptionalGpuVignetteCorrection,
};
use mcraw4vulkan_mcrawcontainer::{
    ContainerMetadata, FrameMetadata, McrawContainer, SensorArrangement,
};
use mcraw4vulkan_vignette::{
    FixedPointVignetteInputFacts, GpuUploadedFullResolutionGainMap, GpuVignetteCorrectionParams,
    GpuVignetteCorrector, PreparedFixedLensShadingMap, VignetteCoordinateMapping,
    VignetteCorrectionInputFacts, VignetteCorrectionMode, VignetteGainMapFingerprint,
};

// Backend choice for generating complete DNG frame bytes.
//
// Platform adapters use CPU or GPU generation behind the same cache interface.
// GPU is the configured default; CPU retains the same decoded Bayer U16 output
// contract as the fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DngGenerationBackend {
    Cpu,
    Gpu {
        backend_preference: GpuBackendPreference,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DngGenerationExecutionPolicy {
    Inflight2Default,
}

impl DngGenerationExecutionPolicy {
    pub const FINAL_APP_FACING_VALUES: [Self; 1] = [Self::Inflight2Default];

    pub fn label(self) -> &'static str {
        match self {
            Self::Inflight2Default => "inflight_2_default",
        }
    }

    pub fn pipeline_depth(self) -> usize {
        match self {
            Self::Inflight2Default => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DngGenerationConfig {
    pub backend: DngGenerationBackend,
    pub vignette_mode: DngSinkVignetteMode,
    pub execution_policy: DngGenerationExecutionPolicy,
}

impl Default for DngGenerationConfig {
    fn default() -> Self {
        Self {
            backend: DngGenerationBackend::Gpu {
                backend_preference: GpuBackendPreference::VulkanOnly,
            },
            vignette_mode: DngSinkVignetteMode::default(),
            execution_policy: DngGenerationExecutionPolicy::Inflight2Default,
        }
    }
}

// Complete DNG bytes for one frame plus timing information.
//
// This is the output that the virtual read path caches and exposes as a normal
// file. Platform adapters should slice these bytes for random reads instead of
// trying to stream decode output directly from read callbacks.
#[derive(Debug)]
pub struct GeneratedDngFrame {
    pub frame_index: usize,
    pub bytes: Vec<u8>,
    pub timings: DngGenerationTimings,
}

// Timing and byte-count breakdown for one DNG frame generation.
//
// These counters are instrumentation only. They do not change output bytes,
// cache behavior, or Resolve-facing file semantics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DngGenerationTimings {
    pub raw_payload_bytes: u64,
    pub decoded_pixel_bytes: u64,
    pub final_dng_bytes: u64,

    pub dng_description_time: Duration,
    pub payload_read_time: Duration,
    pub decode_time: Duration,
    pub gpu_work_plan_time: Duration,
    pub gpu_cpu_prepare_time: Duration,
    pub gpu_upload_time: Duration,
    pub gpu_encode_submit_time: Duration,
    pub gpu_wait_map_time: Duration,
    pub gpu_readback_convert_time: Duration,
    pub gpu_dispatch_readback_time: Duration,
    pub dng_build_time: Duration,
    pub total_time: Duration,
}

// Reusable frame generator for complete DNG byte buffers.
//
// This type owns one McrawContainer, one CPU decoder scratch, one DngWriter, and
// optionally one reusable GPU backend. The cache layer wraps this generator
// behind a Mutex so platform callbacks can request stable frame bytes without
// sharing mutable decode scratch buffers directly.
//
// raw_payload_scratch is used by the GPU DNG path so repeated cache-miss
// generation reuses the same compressed-payload Vec allocation instead of
// allocating a fresh ~multi-MiB Vec for every frame.
pub struct DngFrameGenerator {
    container: McrawContainer,
    cpu_frame_decoder: CpuFrameDecoder,
    dng_writer: DngWriter,
    vignette_mode: DngSinkVignetteMode,
    backend: ActiveDngGenerationBackend,
    raw_payload_scratch: Vec<u8>,
}

// Lightweight DNG size calculator for virtual metadata/getattr.
//
// This owns only container metadata and DNG writer state. It deliberately does
// not create CPU/GPU generation backends, so mounted Cold clips can answer file
// size queries without retaining frame-sized generation resources.
pub struct DngFrameByteLenCalculator {
    container: McrawContainer,
    dng_writer: DngWriter,
    vignette_mode: DngSinkVignetteMode,
}

enum ActiveDngGenerationBackend {
    Cpu {
        gpu_vignette: Option<ActiveGpuVignetteBackend>,
    },
    Gpu(ActiveGpuDngBackend),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuCanonicalDngBatchRoute {
    InFlightRawcodecRing,
    SerialCanonical,
}

struct ActiveGpuDngBackend {
    backend: GpuDecodeBackend,
    vignette: Option<ActiveGpuVignetteState>,
}

struct ActiveGpuVignetteBackend {
    backend: GpuDecodeBackend,
    state: ActiveGpuVignetteState,
}

struct ActiveGpuVignetteState {
    corrector: GpuVignetteCorrector,
    uploaded_gain_map_cache: Option<UploadedGainMapCache>,
}

struct UploadedGainMapCache {
    fingerprint: VignetteGainMapFingerprint,
    uploaded: GpuUploadedFullResolutionGainMap,
}

impl UploadedGainMapCache {
    fn matches(&self, fingerprint: VignetteGainMapFingerprint) -> bool {
        self.fingerprint == fingerprint
    }
}

impl DngFrameGenerator {
    // Open one clip and create a DNG generator using the requested backend.
    pub fn open(path: &Path, backend: DngGenerationBackend) -> Result<Self> {
        Self::open_with_config(
            path,
            DngGenerationConfig {
                backend,
                vignette_mode: DngSinkVignetteMode::default(),
                execution_policy: DngGenerationExecutionPolicy::Inflight2Default,
            },
        )
    }

    pub fn open_with_config(path: &Path, config: DngGenerationConfig) -> Result<Self> {
        let container = McrawContainer::open(path).with_context(|| {
            format!("failed to open clip for DNG generation: {}", path.display())
        })?;

        validate_first_raw_payload(&container)
            .context("failed to validate first raw payload for DNG generation")?;

        let backend = match config.backend {
            DngGenerationBackend::Cpu => {
                let gpu_vignette = match config.vignette_mode {
                    DngSinkVignetteMode::None => None,
                    DngSinkVignetteMode::LumaPlane0 => Some(create_gpu_vignette_backend(
                        GpuBackendPreference::VulkanOnly,
                        "CPU fallback DNG LumaPlane0 correction",
                    )?),
                };

                ActiveDngGenerationBackend::Cpu { gpu_vignette }
            }
            DngGenerationBackend::Gpu { backend_preference } => {
                let gpu_backend_result = GpuDecodeBackend::new_blocking(GpuDecodeConfig {
                    backend_preference,
                    ..GpuDecodeConfig::default()
                })
                .context("failed to create GPU backend for DNG generation");
                let gpu_backend = gpu_backend_result?;
                let vignette = match config.vignette_mode {
                    DngSinkVignetteMode::None => None,
                    DngSinkVignetteMode::LumaPlane0 => {
                        Some(create_gpu_vignette_state(&gpu_backend)?)
                    }
                };

                ActiveDngGenerationBackend::Gpu(ActiveGpuDngBackend {
                    backend: gpu_backend,
                    vignette,
                })
            }
        };

        let frame_rate = source_video_frame_rate(&container);

        let generator = Self {
            container,
            cpu_frame_decoder: CpuFrameDecoder::new(),
            dng_writer: DngWriter::new(DngWriterConfig {
                frame_rate,
                ..DngWriterConfig::default()
            }),
            vignette_mode: config.vignette_mode,
            backend,
            raw_payload_scratch: Vec::new(),
        };
        Ok(generator)
    }

    // Return the number of frames available in this clip.
    pub fn frame_count(&self) -> usize {
        self.container.frame_count()
    }

    // Return the final DNG byte length for one frame without decoding pixels.
    //
    // This is the cheap metadata/getattr path. It reads only cached frame
    // metadata and asks the DNG writer to compute the final uncompressed DNG
    // layout size.
    pub fn frame_byte_len(&mut self, frame_index: usize) -> Result<u64> {
        if frame_index >= self.container.frame_count() {
            anyhow::bail!(
                "frame index {} is out of range for clip with {} frames",
                frame_index,
                self.container.frame_count()
            );
        }

        let frame_number = frame_number_from_index(frame_index)?;

        let dng_description = dng_description_for_container_frame(
            &self.container,
            frame_number,
            self.vignette_mode.into(),
        )
        .with_context(|| format!("failed to build DNG description for frame {frame_index}"))?;

        self.dng_writer
            .uncompressed_frame_byte_len_from_description(&dng_description)
            .with_context(|| format!("failed to compute DNG byte length for frame {frame_index}"))
    }

    // Generate complete DNG bytes for one frame.
    //
    // The returned bytes are complete and stable. The virtual filesystem should
    // cache them and serve read requests by slicing. Metadata/getattr should use
    // frame_byte_len() instead of this method.
    //
    // CPU generation decodes into the shared DecodedBayerU16Frame contract.
    // GPU generation uses the validated direct mapped-readback DNG path, which
    // borrows the exact little-endian Bayer U16 byte prefix as a
    // DecodedBayerU16Frame without first materializing a Vec<u16>.
    pub fn generate_frame(&mut self, frame_index: usize) -> Result<GeneratedDngFrame> {
        if frame_index >= self.container.frame_count() {
            anyhow::bail!(
                "frame index {} is out of range for clip with {} frames",
                frame_index,
                self.container.frame_count()
            );
        }

        let dng_writer = self.dng_writer;

        match &mut self.backend {
            ActiveDngGenerationBackend::Cpu { gpu_vignette } => generate_frame_cpu(
                &self.container,
                &mut self.cpu_frame_decoder,
                &dng_writer,
                self.vignette_mode,
                gpu_vignette.as_mut(),
                frame_index,
            ),
            ActiveDngGenerationBackend::Gpu(gpu_backend) => generate_frame_gpu_canonical_dng(
                &self.container,
                &dng_writer,
                gpu_backend,
                self.vignette_mode,
                &mut self.raw_payload_scratch,
                frame_index,
            ),
        }
    }

    pub fn generate_frames_with_execution_policy(
        &mut self,
        frame_indices: &[usize],
        execution_policy: DngGenerationExecutionPolicy,
    ) -> Result<Vec<GeneratedDngFrame>> {
        if frame_indices.is_empty() {
            return Ok(Vec::new());
        }

        match execution_policy {
            DngGenerationExecutionPolicy::Inflight2Default => {
                if matches!(&self.backend, ActiveDngGenerationBackend::Cpu { .. }) {
                    return self.generate_frames_serially(frame_indices);
                }

                if select_gpu_canonical_dng_batch_route(&self.container, frame_indices)?
                    == GpuCanonicalDngBatchRoute::SerialCanonical
                {
                    return self.generate_frames_serially(frame_indices);
                }

                let dng_writer = self.dng_writer;
                match &mut self.backend {
                    ActiveDngGenerationBackend::Cpu { .. } => unreachable!(),
                    ActiveDngGenerationBackend::Gpu(gpu_backend) => {
                        generate_frames_gpu_canonical_dng_in_flight(
                            &self.container,
                            &dng_writer,
                            gpu_backend,
                            self.vignette_mode,
                            frame_indices,
                            execution_policy.pipeline_depth(),
                        )
                    }
                }
            }
        }
    }

    fn generate_frames_serially(
        &mut self,
        frame_indices: &[usize],
    ) -> Result<Vec<GeneratedDngFrame>> {
        let mut frames = Vec::with_capacity(frame_indices.len());
        for &frame_index in frame_indices {
            frames.push(self.generate_frame(frame_index)?);
        }
        Ok(frames)
    }

    // Generate a prefetch/cache-warming batch through the same canonical
    // per-frame sink producer as foreground reads.
    //
    // This keeps DNG correction mode, metadata policy, and CPU/GPU fallback
    // behavior identical between requested reads and speculative read-ahead.
    pub fn generate_frames_for_prefetch(
        &mut self,
        frame_indices: &[usize],
        execution_policy: DngGenerationExecutionPolicy,
    ) -> Result<Vec<GeneratedDngFrame>> {
        self.generate_frames_with_execution_policy(frame_indices, execution_policy)
    }
}

impl DngFrameByteLenCalculator {
    pub fn open_with_config(path: &Path, config: DngGenerationConfig) -> Result<Self> {
        let container = McrawContainer::open(path).with_context(|| {
            format!(
                "failed to open clip for DNG metadata sizing: {}",
                path.display()
            )
        })?;

        validate_first_raw_payload(&container)
            .context("failed to validate first raw payload for DNG metadata sizing")?;

        let frame_rate = source_video_frame_rate(&container);

        Ok(Self {
            container,
            dng_writer: DngWriter::new(DngWriterConfig {
                frame_rate,
                ..DngWriterConfig::default()
            }),
            vignette_mode: config.vignette_mode,
        })
    }

    pub fn frame_count(&self) -> usize {
        self.container.frame_count()
    }

    pub fn frame_byte_len(&mut self, frame_index: usize) -> Result<u64> {
        if frame_index >= self.container.frame_count() {
            anyhow::bail!(
                "frame index {} is out of range for clip with {} frames",
                frame_index,
                self.container.frame_count()
            );
        }

        let frame_number = frame_number_from_index(frame_index)?;
        let dng_description = dng_description_for_container_frame(
            &self.container,
            frame_number,
            self.vignette_mode.into(),
        )
        .with_context(|| format!("failed to build DNG description for frame {frame_index}"))?;

        self.dng_writer
            .uncompressed_frame_byte_len_from_description(&dng_description)
            .with_context(|| format!("failed to compute DNG byte length for frame {frame_index}"))
    }
}

struct PreparedGpuCanonicalDngFrame {
    frame_index: usize,
    dng_description: DngFrameDescription,
    dng_description_time: Duration,
    payload_read_time: Duration,
    raw_payload_bytes: u64,
    exact_pixel_byte_len: usize,
    decoded_pixel_bytes: u64,
}

struct ConsumedGpuCanonicalDngFrame {
    frame_index: usize,
    bytes: Vec<u8>,
    dng_description_time: Duration,
    payload_read_time: Duration,
    raw_payload_bytes: u64,
    decoded_pixel_bytes: u64,
    dng_build_time: Duration,
}

fn select_gpu_canonical_dng_batch_route(
    container: &McrawContainer,
    frame_indices: &[usize],
) -> Result<GpuCanonicalDngBatchRoute> {
    let mut route = GpuCanonicalDngBatchRoute::InFlightRawcodecRing;

    for &frame_index in frame_indices {
        if frame_index >= container.frame_count() {
            anyhow::bail!(
                "frame index {} is out of range for clip with {} frames",
                frame_index,
                container.frame_count()
            );
        }

        let frame_number = frame_number_from_index(frame_index)?;
        let payload_layout = container.frame_metadata(frame_number)?.payload_layout()?;
        route = route_for_gpu_canonical_dng_layout(route, payload_layout);
    }

    Ok(route)
}

fn route_for_gpu_canonical_dng_layout(
    current: GpuCanonicalDngBatchRoute,
    payload_layout: FramePayloadLayout,
) -> GpuCanonicalDngBatchRoute {
    if current == GpuCanonicalDngBatchRoute::SerialCanonical
        || !payload_layout.uses_compressed_rawcodec_work_plan()
    {
        GpuCanonicalDngBatchRoute::SerialCanonical
    } else {
        GpuCanonicalDngBatchRoute::InFlightRawcodecRing
    }
}

fn generate_frames_gpu_canonical_dng_in_flight(
    container: &McrawContainer,
    dng_writer: &DngWriter,
    gpu_backend: &mut ActiveGpuDngBackend,
    vignette_mode: DngSinkVignetteMode,
    frame_indices: &[usize],
    pipeline_depth: usize,
) -> Result<Vec<GeneratedDngFrame>> {
    let prepared = RefCell::new(HashMap::<usize, PreparedGpuCanonicalDngFrame>::new());
    let depth = pipeline_depth.max(1);

    let output = match vignette_mode {
        DngSinkVignetteMode::None => gpu_backend
            .backend
            .decode_raw_payloads_packed_u16_ring_mapped(
                frame_indices.len(),
                depth,
                |input_index| {
                    let frame_index = frame_indices[input_index];
                    let (prepared_frame, raw_payload) =
                        prepare_gpu_canonical_dng_frame(container, frame_index, vignette_mode)?;
                    let dimensions = prepared_frame.dng_description.dimensions;
                    prepared
                        .borrow_mut()
                        .insert(prepared_frame.frame_index, prepared_frame);
                    Ok(GpuMappedRingFrame {
                        frame_index,
                        raw_payload,
                        visible_dimensions: dimensions,
                    })
                },
                |frame_index, mapped_pixel_bytes| {
                    consume_gpu_canonical_dng_frame(
                        &prepared,
                        dng_writer,
                        frame_index,
                        mapped_pixel_bytes,
                    )
                },
            )?,
        DngSinkVignetteMode::LumaPlane0 => {
            let vignette = gpu_backend
                .vignette
                .as_mut()
                .context("GPU LumaPlane0 DNG generation requires a GPU vignette state")?;
            gpu_backend.backend.decode_raw_payloads_packed_u16_ring_mapped_with_vignette(
                frame_indices.len(),
                depth,
                |input_index, backend| {
                    let frame_index = frame_indices[input_index];
                    let (prepared_frame, raw_payload) =
                        prepare_gpu_canonical_dng_frame(container, frame_index, vignette_mode)?;
                    let frame_number = frame_number_from_index(frame_index)?;
                    let frame_metadata = container.frame_metadata(frame_number)?;
                    let facts = fixed_vignette_facts_for_frame(
                        container.container_metadata(),
                        frame_metadata,
                        VignetteCorrectionMode::Enabled,
                    )
                    .with_context(|| {
                        format!(
                            "failed to build in-flight LumaPlane0 vignette facts for frame {frame_index}"
                        )
                    })?;
                    let uploaded_gain_map = ensure_uploaded_gain_map(
                        &mut vignette.uploaded_gain_map_cache,
                        backend,
                        &mut vignette.corrector,
                        &facts,
                    )
                    .with_context(|| {
                        format!(
                            "failed to prepare in-flight GPU LumaPlane0 gain map for frame {frame_index}"
                        )
                    })?
                    .clone();
                    let params = GpuVignetteCorrectionParams::from_fixed_facts(&facts)
                        .context("failed to build in-flight GPU LumaPlane0 parameters")?;
                    let dimensions = prepared_frame.dng_description.dimensions;
                    prepared
                        .borrow_mut()
                        .insert(prepared_frame.frame_index, prepared_frame);
                    Ok((
                        GpuMappedRingFrame {
                            frame_index,
                            raw_payload,
                            visible_dimensions: dimensions,
                        },
                        GpuMappedRingVignetteCorrection {
                            uploaded_gain_map,
                            params,
                        },
                    ))
                },
                |frame_index, mapped_pixel_bytes| {
                    consume_gpu_canonical_dng_frame(
                        &prepared,
                        dng_writer,
                        frame_index,
                        mapped_pixel_bytes,
                    )
                },
            )?
        }
    };

    anyhow::ensure!(
        output.values.len() == output.timings.len(),
        "in-flight DNG generation returned {} values for {} timing rows",
        output.values.len(),
        output.timings.len()
    );

    output
        .values
        .into_iter()
        .zip(output.timings)
        .map(|(consumed, gpu_timings)| {
            let decode_time = gpu_timings.total;
            let final_dng_bytes = usize_to_u64(consumed.bytes.len(), "final DNG byte length")?;
            Ok(GeneratedDngFrame {
                frame_index: consumed.frame_index,
                bytes: consumed.bytes,
                timings: DngGenerationTimings {
                    raw_payload_bytes: consumed.raw_payload_bytes,
                    decoded_pixel_bytes: consumed.decoded_pixel_bytes,
                    final_dng_bytes,
                    dng_description_time: consumed.dng_description_time,
                    payload_read_time: consumed.payload_read_time,
                    decode_time,
                    gpu_work_plan_time: gpu_timings.work_plan,
                    gpu_cpu_prepare_time: gpu_timings.cpu_prepare,
                    gpu_upload_time: gpu_timings.upload,
                    gpu_encode_submit_time: gpu_timings.encode_submit,
                    gpu_wait_map_time: gpu_timings.wait_map,
                    gpu_readback_convert_time: gpu_timings.readback_convert,
                    gpu_dispatch_readback_time: gpu_timings.dispatch_readback,
                    dng_build_time: consumed.dng_build_time,
                    total_time: consumed
                        .dng_description_time
                        .saturating_add(consumed.payload_read_time)
                        .saturating_add(decode_time)
                        .saturating_add(consumed.dng_build_time),
                },
            })
        })
        .collect::<Result<Vec<_>>>()
}

fn prepare_gpu_canonical_dng_frame(
    container: &McrawContainer,
    frame_index: usize,
    vignette_mode: DngSinkVignetteMode,
) -> Result<(PreparedGpuCanonicalDngFrame, Vec<u8>)> {
    if frame_index >= container.frame_count() {
        anyhow::bail!(
            "frame index {} is out of range for clip with {} frames",
            frame_index,
            container.frame_count()
        );
    }

    let frame_number = frame_number_from_index(frame_index)?;
    let frame_metadata = container.frame_metadata(frame_number)?;
    let payload_layout = frame_metadata.payload_layout()?;
    if !payload_layout.uses_compressed_rawcodec_work_plan() {
        bail_unsupported_gpu_decode::<()>(payload_layout)?;
    }

    let description_start = Instant::now();
    let dng_description =
        dng_description_for_container_frame(container, frame_number, vignette_mode.into())
            .with_context(|| format!("failed to build DNG description for frame {frame_index}"))?;
    let dng_description_time = description_start.elapsed();

    let payload_start = Instant::now();
    let raw_payload = raw_payload_vec_for_frame(container, frame_number)
        .with_context(|| format!("failed to read raw payload for frame {frame_index}"))?;
    let payload_read_time = payload_start.elapsed();
    let raw_payload_bytes = usize_to_u64(raw_payload.len(), "raw payload byte length")?;

    let exact_pixel_byte_len = exact_pixel_byte_len(dng_description.dimensions)?;
    let decoded_pixel_bytes = usize_to_u64(exact_pixel_byte_len, "frame u16 pixel byte length")?;

    Ok((
        PreparedGpuCanonicalDngFrame {
            frame_index,
            dng_description,
            dng_description_time,
            payload_read_time,
            raw_payload_bytes,
            exact_pixel_byte_len,
            decoded_pixel_bytes,
        },
        raw_payload,
    ))
}

fn consume_gpu_canonical_dng_frame(
    prepared: &RefCell<HashMap<usize, PreparedGpuCanonicalDngFrame>>,
    dng_writer: &DngWriter,
    frame_index: usize,
    mapped_pixel_bytes: &[u8],
) -> Result<ConsumedGpuCanonicalDngFrame> {
    let prepared_frame = prepared
        .borrow_mut()
        .remove(&frame_index)
        .with_context(|| format!("missing prepared in-flight DNG frame {frame_index}"))?;
    let exact_pixel_bytes = mapped_pixel_bytes
        .get(..prepared_frame.exact_pixel_byte_len)
        .with_context(|| {
            format!(
                "mapped in-flight GPU output for frame {frame_index} is smaller than exact Bayer U16 payload: expected {} bytes, got {}",
                prepared_frame.exact_pixel_byte_len,
                mapped_pixel_bytes.len()
            )
        })?;
    let decoded_bayer_frame = DecodedBayerU16Frame::from_borrowed_le_bytes(
        prepared_frame.dng_description.dimensions,
        exact_pixel_bytes,
    )
    .with_context(|| {
        format!("mapped in-flight GPU output/frame contract mismatch for frame {frame_index}")
    })?;
    let dng_build_start = Instant::now();
    let bytes = dng_writer
        .write_uncompressed_frame_from_decoded_bayer_u16_frame_to_vec(
            &prepared_frame.dng_description,
            &decoded_bayer_frame,
        )
        .with_context(|| {
            format!("failed to build in-flight canonical GPU DNG bytes for frame {frame_index}")
        })?;
    let dng_build_time = dng_build_start.elapsed();

    Ok(ConsumedGpuCanonicalDngFrame {
        frame_index,
        bytes,
        dng_description_time: prepared_frame.dng_description_time,
        payload_read_time: prepared_frame.payload_read_time,
        raw_payload_bytes: prepared_frame.raw_payload_bytes,
        decoded_pixel_bytes: prepared_frame.decoded_pixel_bytes,
        dng_build_time,
    })
}

fn bail_unsupported_gpu_decode<T>(payload_layout: FramePayloadLayout) -> Result<T> {
    if let Some(message) = payload_layout.unsupported_gpu_decode_message() {
        anyhow::bail!("{message}");
    }
    anyhow::bail!(
        "payload layout {} does not support this GPU decode route",
        payload_layout.label()
    )
}

// Generate one DNG using the CPU reference decoder.
//
// This is a free helper rather than an &mut self method so generate_frame() can
// borrow DngFrameGenerator fields independently without double-borrowing self.
fn generate_frame_cpu(
    container: &McrawContainer,
    cpu_frame_decoder: &mut CpuFrameDecoder,
    dng_writer: &DngWriter,
    vignette_mode: DngSinkVignetteMode,
    gpu_vignette: Option<&mut ActiveGpuVignetteBackend>,
    frame_index: usize,
) -> Result<GeneratedDngFrame> {
    let total_start = Instant::now();
    let frame_number = frame_number_from_index(frame_index)?;
    let (entry_byte_len, entry_dimensions) = {
        let entry = container.frame_entry(frame_number)?;
        (entry.byte_len, entry.dimensions)
    };

    let description_start = Instant::now();
    let dng_description =
        dng_description_for_container_frame(container, frame_number, vignette_mode.into())
            .with_context(|| format!("failed to build DNG description for frame {frame_index}"))?;
    let dng_description_time = description_start.elapsed();

    let payload_start = Instant::now();
    cpu_frame_decoder.prepare_compressed(entry_byte_len as usize);
    container
        .read_video_payload_into(frame_number, cpu_frame_decoder.compressed_mut())
        .with_context(|| format!("failed to read raw payload for frame {frame_index}"))?;
    let payload_read_time = payload_start.elapsed();

    let decode_start = Instant::now();
    let frame_metadata = container.frame_metadata(frame_number)?;
    let payload_layout = frame_metadata.payload_layout()?;
    let mut decode_timings = DecodeFrameTimings::default();
    let (decoded_bayer_frame, _raw_decode_info) = cpu_frame_decoder
        .decode_loaded_payload_to_decoded_bayer_u16_frame_with_layout(
            entry_dimensions,
            payload_layout,
            &mut decode_timings,
        )
        .with_context(|| format!("failed to CPU-decode frame {frame_index}"))?;

    let (sink_frame, gpu_timings) = match vignette_mode {
        DngSinkVignetteMode::None => {
            let owned_frame = DecodedBayerU16Frame::from_owned_le_bytes(
                decoded_bayer_frame.dimensions(),
                decoded_bayer_frame.into_owned_le_bytes(),
            )
            .context("CPU decoded Bayer U16 frame failed sink validation")?;
            (
                DngSinkFrame::new(
                    frame_number,
                    owned_frame,
                    vignette_mode,
                    DngSinkDecodeSource::CpuFallbackNoVig,
                    dng_description,
                )?,
                None,
            )
        }
        DngSinkVignetteMode::LumaPlane0 => {
            let gpu_vignette = gpu_vignette.context(
                "CPU fallback LumaPlane0 DNG generation requires a GPU vignette backend",
            )?;
            let facts = fixed_vignette_facts_for_frame(
                container.container_metadata(),
                frame_metadata,
                VignetteCorrectionMode::Enabled,
            )
            .with_context(|| {
                format!("failed to build LumaPlane0 vignette facts for frame {frame_index}")
            })?;
            let uploaded_gain_map = ensure_uploaded_gain_map(
                &mut gpu_vignette.state.uploaded_gain_map_cache,
                &gpu_vignette.backend,
                &mut gpu_vignette.state.corrector,
                &facts,
            )
            .with_context(|| {
                format!("failed to prepare GPU LumaPlane0 gain map for frame {frame_index}")
            })?;
            let params = GpuVignetteCorrectionParams::from_fixed_facts(&facts)
                .context("failed to build GPU LumaPlane0 parameters")?;
            let exact_pixel_byte_len = decoded_bayer_frame.pixel_bytes_le().len();
            let corrected = gpu_vignette
                .backend
                .correct_decoded_bayer_u16_mapped_with_vignette(
                    decoded_bayer_frame.pixel_bytes_le(),
                    decoded_bayer_frame.dimensions(),
                    OptionalGpuVignetteCorrection {
                        corrector: &mut gpu_vignette.state.corrector,
                        uploaded_gain_map,
                        params,
                    },
                    |mapped_pixel_bytes| {
                        mapped_prefix_to_owned_frame(
                            mapped_pixel_bytes,
                            decoded_bayer_frame.dimensions(),
                            exact_pixel_byte_len,
                        )
                    },
                )
                .with_context(|| {
                    format!("failed to GPU-correct CPU-decoded frame {frame_index}")
                })?;
            (
                DngSinkFrame::new(
                    frame_number,
                    corrected.value,
                    vignette_mode,
                    DngSinkDecodeSource::CpuFallbackGpuVignette,
                    dng_description,
                )?,
                Some(corrected.timings),
            )
        }
    };
    let decode_time = decode_start.elapsed();

    let decoded_pixel_bytes = usize_to_u64(
        sink_frame.pixel_bytes_le().len(),
        "decoded Bayer U16 byte length",
    )?;

    let dng_build_start = Instant::now();
    let bytes = dng_writer
        .write_uncompressed_dng_sink_frame_to_vec(&sink_frame)
        .with_context(|| format!("failed to build CPU DNG bytes for frame {frame_index}"))?;
    let dng_build_time = dng_build_start.elapsed();
    let final_dng_bytes = usize_to_u64(bytes.len(), "final DNG byte length")?;
    let gpu_timings = gpu_timings.unwrap_or_default();

    Ok(GeneratedDngFrame {
        frame_index,
        bytes,
        timings: DngGenerationTimings {
            raw_payload_bytes: u64::from(entry_byte_len),
            decoded_pixel_bytes,
            final_dng_bytes,
            dng_description_time,
            payload_read_time,
            decode_time,
            gpu_work_plan_time: gpu_timings.work_plan,
            gpu_cpu_prepare_time: gpu_timings.cpu_prepare,
            gpu_upload_time: gpu_timings.upload,
            gpu_encode_submit_time: gpu_timings.encode_submit,
            gpu_wait_map_time: gpu_timings.wait_map,
            gpu_readback_convert_time: gpu_timings.readback_convert,
            gpu_dispatch_readback_time: gpu_timings.dispatch_readback,
            dng_build_time,
            total_time: total_start.elapsed(),
        },
    })
}

// Generate one DNG through the mapped packed-u16 GPU readback path.
//
// The GPU backend maps the readback buffer and exposes the mapped little-endian
// 16-bit Bayer bytes only inside a closure. The DNG writer builds the final DNG
// Vec inside that closure, before the GPU readback buffer is unmapped. This
// path is the DNG/FUSE sink boundary that validators compare against the CPU
// fallback DNG bytes.
fn generate_frame_gpu_canonical_dng(
    container: &McrawContainer,
    dng_writer: &DngWriter,
    gpu_backend: &mut ActiveGpuDngBackend,
    vignette_mode: DngSinkVignetteMode,
    raw_payload_scratch: &mut Vec<u8>,
    frame_index: usize,
) -> Result<GeneratedDngFrame> {
    let total_start = Instant::now();
    let frame_number = frame_number_from_index(frame_index)?;
    let frame_metadata = container.frame_metadata(frame_number)?;
    let payload_layout = frame_metadata.payload_layout()?;
    if !payload_layout.supports_native_gpu_decode() {
        bail_unsupported_gpu_decode::<()>(payload_layout)?;
    }

    let description_start = Instant::now();
    let dng_description =
        dng_description_for_container_frame(container, frame_number, vignette_mode.into())
            .with_context(|| format!("failed to build DNG description for frame {frame_index}"))?;
    let dng_description_time = description_start.elapsed();

    let payload_start = Instant::now();
    container
        .read_video_payload_into(frame_number, raw_payload_scratch)
        .with_context(|| format!("failed to read raw payload for frame {frame_index}"))?;
    let payload_read_time = payload_start.elapsed();
    let raw_payload_bytes = usize_to_u64(raw_payload_scratch.len(), "raw payload byte length")?;

    let exact_pixel_byte_len = exact_pixel_byte_len(dng_description.dimensions)?;
    let decoded_pixel_bytes = usize_to_u64(exact_pixel_byte_len, "frame u16 pixel byte length")?;

    let decode_start = Instant::now();
    let (sink_frame, gpu_timings) = match vignette_mode {
        DngSinkVignetteMode::None => match payload_layout {
            FramePayloadLayout::CompressedRawcodecType7 => {
                let output = gpu_backend
                    .backend
                    .decode_raw_payload_to_canonical_bayer_u16_mapped(
                        raw_payload_scratch,
                        dng_description.dimensions,
                        |mapped_pixel_bytes| {
                            let frame = mapped_prefix_to_owned_frame(
                                mapped_pixel_bytes,
                                dng_description.dimensions,
                                exact_pixel_byte_len,
                            )?;
                            DngSinkFrame::new(
                                frame_number,
                                frame,
                                vignette_mode,
                                DngSinkDecodeSource::GpuCanonical,
                                dng_description.clone(),
                            )
                            .map_err(anyhow::Error::from)
                        },
                    )
                    .with_context(|| format!("failed to GPU-decode mapped frame {frame_index}"))?;

                (output.value, output.timings)
            }
            FramePayloadLayout::BinnedRaw16Type6 { row_stride } => {
                let output = gpu_backend
                    .backend
                    .decode_legacy_raw16_payload_to_canonical_bayer_u16_mapped(
                        raw_payload_scratch,
                        dng_description.dimensions,
                        row_stride,
                        |mapped_pixel_bytes| {
                            let frame = mapped_prefix_to_owned_frame(
                                mapped_pixel_bytes,
                                dng_description.dimensions,
                                exact_pixel_byte_len,
                            )?;
                            DngSinkFrame::new(
                                frame_number,
                                frame,
                                vignette_mode,
                                DngSinkDecodeSource::GpuCanonical,
                                dng_description.clone(),
                            )
                            .map_err(anyhow::Error::from)
                        },
                    )
                    .with_context(|| {
                        format!("failed to GPU-decode legacy raw16 mapped frame {frame_index}")
                    })?;

                (output.value, output.timings)
            }
        },
        DngSinkVignetteMode::LumaPlane0 => {
            let vignette = gpu_backend
                .vignette
                .as_mut()
                .context("GPU LumaPlane0 DNG generation requires a GPU vignette state")?;
            let facts = fixed_vignette_facts_for_frame(
                container.container_metadata(),
                frame_metadata,
                VignetteCorrectionMode::Enabled,
            )
            .with_context(|| {
                format!("failed to build LumaPlane0 vignette facts for frame {frame_index}")
            })?;
            let uploaded_gain_map = ensure_uploaded_gain_map(
                &mut vignette.uploaded_gain_map_cache,
                &gpu_backend.backend,
                &mut vignette.corrector,
                &facts,
            )
            .with_context(|| {
                format!("failed to prepare GPU LumaPlane0 gain map for frame {frame_index}")
            })?;
            let params = GpuVignetteCorrectionParams::from_fixed_facts(&facts)
                .context("failed to build GPU LumaPlane0 parameters")?;

            match payload_layout {
                FramePayloadLayout::CompressedRawcodecType7 => {
                    let output = gpu_backend
                        .backend
                        .decode_raw_payload_to_canonical_bayer_u16_mapped_with_vignette(
                            raw_payload_scratch,
                            dng_description.dimensions,
                            Some(OptionalGpuVignetteCorrection {
                                corrector: &mut vignette.corrector,
                                uploaded_gain_map,
                                params,
                            }),
                            |mapped_pixel_bytes| {
                                let frame = mapped_prefix_to_owned_frame(
                                    mapped_pixel_bytes,
                                    dng_description.dimensions,
                                    exact_pixel_byte_len,
                                )?;
                                DngSinkFrame::new(
                                    frame_number,
                                    frame,
                                    vignette_mode,
                                    DngSinkDecodeSource::GpuCanonical,
                                    dng_description.clone(),
                                )
                                .map_err(anyhow::Error::from)
                            },
                        )
                        .with_context(|| {
                            format!(
                                "failed to GPU-decode and LumaPlane0-correct frame {frame_index}"
                            )
                        })?;

                    (output.value, output.timings)
                }
                FramePayloadLayout::BinnedRaw16Type6 { row_stride } => {
                    let output = gpu_backend
                        .backend
                        .decode_legacy_raw16_payload_to_canonical_bayer_u16_mapped_with_vignette(
                            raw_payload_scratch,
                            dng_description.dimensions,
                            row_stride,
                            Some(OptionalGpuVignetteCorrection {
                                corrector: &mut vignette.corrector,
                                uploaded_gain_map,
                                params,
                            }),
                            |mapped_pixel_bytes| {
                                let frame = mapped_prefix_to_owned_frame(
                                    mapped_pixel_bytes,
                                    dng_description.dimensions,
                                    exact_pixel_byte_len,
                                )?;
                                DngSinkFrame::new(
                                    frame_number,
                                    frame,
                                    vignette_mode,
                                    DngSinkDecodeSource::GpuCanonical,
                                    dng_description.clone(),
                                )
                                .map_err(anyhow::Error::from)
                            },
                        )
                        .with_context(|| {
                            format!(
                                "failed to GPU-decode legacy raw16 and LumaPlane0-correct frame {frame_index}"
                            )
                        })?;

                    (output.value, output.timings)
                }
            }
        }
    };
    let decode_time = decode_start.elapsed();

    let dng_build_start = Instant::now();
    let bytes = dng_writer
        .write_uncompressed_dng_sink_frame_to_vec(&sink_frame)
        .with_context(|| {
            format!("failed to build canonical GPU DNG bytes for frame {frame_index}")
        })?;
    let dng_build_time = dng_build_start.elapsed();
    let final_dng_bytes = usize_to_u64(bytes.len(), "final DNG byte length")?;

    Ok(GeneratedDngFrame {
        frame_index,
        bytes,
        timings: DngGenerationTimings {
            raw_payload_bytes,
            decoded_pixel_bytes,
            final_dng_bytes,
            dng_description_time,
            payload_read_time,
            decode_time,
            gpu_work_plan_time: gpu_timings.work_plan,
            gpu_cpu_prepare_time: gpu_timings.cpu_prepare,
            gpu_upload_time: gpu_timings.upload,
            gpu_encode_submit_time: gpu_timings.encode_submit,
            gpu_wait_map_time: gpu_timings.wait_map,
            gpu_readback_convert_time: gpu_timings.readback_convert,
            gpu_dispatch_readback_time: gpu_timings.dispatch_readback,
            dng_build_time,
            total_time: total_start.elapsed(),
        },
    })
}

fn exact_pixel_byte_len(dimensions: FrameDimensions) -> Result<usize> {
    let pixel_count = dimensions
        .pixel_count()
        .context("DNG frame dimensions do not fit in usize")?;

    pixel_count
        .checked_mul(std::mem::size_of::<u16>())
        .context("frame u16 pixel byte count overflow")
}

fn mapped_prefix_to_owned_frame(
    mapped_pixel_bytes: &[u8],
    dimensions: FrameDimensions,
    exact_pixel_byte_len: usize,
) -> Result<DecodedBayerU16Frame<'static>> {
    let exact_pixel_bytes = mapped_pixel_bytes
        .get(..exact_pixel_byte_len)
        .with_context(|| {
            format!(
                "mapped GPU output is smaller than exact Bayer U16 payload: expected {exact_pixel_byte_len} bytes, got {}",
                mapped_pixel_bytes.len()
            )
        })?;

    DecodedBayerU16Frame::from_owned_le_bytes(dimensions, exact_pixel_bytes.to_vec())
        .context("mapped GPU output/frame contract mismatch")
}

fn create_gpu_vignette_backend(
    backend_preference: GpuBackendPreference,
    label: &str,
) -> Result<ActiveGpuVignetteBackend> {
    let backend = GpuDecodeBackend::new_blocking(GpuDecodeConfig {
        backend_preference,
        ..GpuDecodeConfig::default()
    })
    .with_context(|| format!("failed to create GPU backend for {label}"))?;
    let state = create_gpu_vignette_state(&backend)?;

    Ok(ActiveGpuVignetteBackend { backend, state })
}

fn create_gpu_vignette_state(backend: &GpuDecodeBackend) -> Result<ActiveGpuVignetteState> {
    let corrector = backend
        .create_vignette_corrector()
        .context("failed to create GPU LumaPlane0 vignette corrector")?;

    Ok(ActiveGpuVignetteState {
        corrector,
        uploaded_gain_map_cache: None,
    })
}

fn ensure_uploaded_gain_map<'a>(
    cache: &'a mut Option<UploadedGainMapCache>,
    gpu_backend: &GpuDecodeBackend,
    gpu_corrector: &mut GpuVignetteCorrector,
    facts: &FixedPointVignetteInputFacts<'_>,
) -> Result<&'a GpuUploadedFullResolutionGainMap> {
    let fingerprint = VignetteGainMapFingerprint::from_fixed_facts(facts)
        .context("failed to fingerprint full-resolution fixed vignette gain map")?;
    let reuse_existing = cache
        .as_ref()
        .map(|cached| cached.matches(fingerprint))
        .unwrap_or(false);

    if !reuse_existing {
        let uploaded = gpu_backend
            .upload_compact_vignette_gain_map(gpu_corrector, facts)
            .context("failed to upload GPU vignette gain map")?;

        *cache = Some(UploadedGainMapCache {
            fingerprint,
            uploaded,
        });
    }

    Ok(&cache
        .as_ref()
        .context("uploaded gain map cache was not populated")?
        .uploaded)
}

fn fixed_vignette_facts_for_frame<'a>(
    container_metadata: &ContainerMetadata,
    frame_metadata: &'a FrameMetadata,
    mode: VignetteCorrectionMode,
) -> Result<FixedPointVignetteInputFacts<'a>> {
    let lens_shading_map = frame_metadata
        .lens_shading_map
        .as_ref()
        .context("frame metadata missing lensShadingMap")?;
    let fixed_map = PreparedFixedLensShadingMap::from_typed_map(lens_shading_map)
        .context("failed to prepare fixed lens shading map")?;
    let input_facts = VignetteCorrectionInputFacts::new(
        mode,
        VignetteCoordinateMapping::VisibleFrame,
        frame_metadata.dimensions,
        bayer_pattern_from_sensor_arrangement(&container_metadata.sensor_arrangement)?,
        None,
        input_black_level(container_metadata, frame_metadata)?,
        output_white_level(container_metadata, frame_metadata)?,
    )
    .context("failed to build typed vignette input facts")?;

    FixedPointVignetteInputFacts::from_input_facts_with_fixed_map(&input_facts, Some(fixed_map))
        .context("failed to build fixed-point vignette input facts")
}

fn input_black_level(
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
            anyhow::bail!("black level {index} must be finite, non-negative, and fit f32");
        }
        output[index] = value as f32;
    }

    Ok(output)
}

fn output_white_level(
    container_metadata: &ContainerMetadata,
    frame_metadata: &FrameMetadata,
) -> Result<u16> {
    if let Some(value) = frame_metadata.dynamic_white_level {
        return f64_level_to_u16(value, "dynamicWhiteLevel");
    }

    let white_level = container_metadata
        .white_level
        .context("missing dynamicWhiteLevel and container whiteLevel")?;
    let value = uniform_level(white_level.values)
        .context("container whiteLevel must be uniform for vignette output white level")?;

    f64_level_to_u16(value, "container whiteLevel")
}

fn uniform_level(values: [f64; 4]) -> Option<f64> {
    let first = values[0];
    values
        .into_iter()
        .all(|value| (value - first).abs() <= 0.000_001)
        .then_some(first)
}

fn f64_level_to_u16(value: f64, label: &str) -> Result<u16> {
    if !value.is_finite() || value < 0.0 || value > f64::from(u16::MAX) {
        anyhow::bail!("{label} must be finite, non-negative, and fit u16");
    }

    let rounded = value.round();
    if (value - rounded).abs() > 0.000_001 {
        anyhow::bail!("{label} must be integer-like for fixed-point vignette output");
    }

    Ok(rounded as u16)
}

fn bayer_pattern_from_sensor_arrangement(
    sensor_arrangement: &SensorArrangement,
) -> Result<BayerPattern> {
    match sensor_arrangement {
        SensorArrangement::Rggb => Ok(BayerPattern::Rggb),
        SensorArrangement::Bggr => Ok(BayerPattern::Bggr),
        SensorArrangement::Grbg => Ok(BayerPattern::Grbg),
        SensorArrangement::Gbrg => Ok(BayerPattern::Gbrg),
        SensorArrangement::Unknown(value) => {
            anyhow::bail!("unsupported sensor arrangement: {value}")
        }
        SensorArrangement::Missing => anyhow::bail!("missing sensor arrangement"),
    }
}

fn validate_first_raw_payload(container: &McrawContainer) -> Result<()> {
    let clip_info = container.clip_info();
    let clip_dimensions = FrameDimensions {
        width: clip_info.width,
        height: clip_info.height,
    };
    let first_metadata = container
        .frame_metadata(FrameNumber(0))
        .context("failed to read first frame metadata")?;
    let payload_layout = first_metadata.payload_layout()?;

    let mut first_payload = Vec::new();
    container
        .read_video_payload_into(FrameNumber(0), &mut first_payload)
        .context("failed to read first raw payload")?;
    validate_frame_payload(&first_payload, clip_dimensions, payload_layout)
        .context("first raw payload validation failed")?;

    Ok(())
}

fn raw_payload_vec_for_frame(
    container: &McrawContainer,
    frame_number: FrameNumber,
) -> Result<Vec<u8>> {
    let entry = container.frame_entry(frame_number)?;
    let capacity =
        usize::try_from(entry.byte_len).context("frame payload length overflows usize")?;
    let mut raw_payload = Vec::with_capacity(capacity);

    container.read_video_payload_into(frame_number, &mut raw_payload)?;

    Ok(raw_payload)
}

fn dng_description_for_container_frame(
    container: &McrawContainer,
    frame_number: FrameNumber,
    correction: DngOutputCorrection,
) -> Result<DngFrameDescription> {
    let entry = container.frame_entry(frame_number)?;
    let frame_metadata = container.frame_metadata(frame_number)?;

    Ok(build_dng_frame_description_for_sink(
        container.container_metadata(),
        frame_metadata,
        entry.frame_number,
        entry.timestamp_us,
        correction,
    )?)
}

fn source_video_frame_rate(container: &McrawContainer) -> Option<FrameRate> {
    container
        .clip_info()
        .timing
        .video_reported_frame_rate
        .map(FrameRate::reduced)
}

// Convert usize byte counts into u64 counters for instrumentation.
fn usize_to_u64(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("{label} does not fit in u64"))
}

// Convert a usize frame index to the shared typed FrameNumber.
fn frame_number_from_index(frame_index: usize) -> Result<FrameNumber> {
    Ok(FrameNumber(
        u32::try_from(frame_index).context("frame index does not fit in u32")?,
    ))
}
