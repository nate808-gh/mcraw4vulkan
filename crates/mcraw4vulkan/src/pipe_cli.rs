use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

use anyhow::{Context, Result, anyhow, bail};
use mcraw4vulkan_core::{
    BayerPattern, FrameDimensions, FrameNumber, FramePayloadLayout, McrawClipInfo,
};
use mcraw4vulkan_cpu::{CpuFrameDecoder, DecodeFrameTimings};
use mcraw4vulkan_fuse::{AudioWavMetadata, LazyAudioWav, LazyAudioWavConfig};
use mcraw4vulkan_gpu::{
    GpuBackendPreference, GpuDecodeBackend, GpuDecodeConfig, GpuDecodeRequiredLimits,
};
use mcraw4vulkan_mcrawcontainer::{
    FrameMetadata, McrawContainer, StrictColorProfileProvenance,
    payload_reader::{PayloadFeeder, PayloadFeederOptions, PayloadFeederStats, PayloadReadPlan},
};
use mcraw4vulkan_render::DIRECT_YUV12_STATUS_BYTE_LEN;
use mcraw4vulkan_vignette::{
    FixedPointVignetteInputFacts, PipeF32BayerCorrectionFingerprint, PipeF32BayerCorrectionMode,
    PipeF32BayerNumericDomain, motioncam_pipe_f32_bayer_facts,
};
use serde_json::{Value, json};

use crate::direct_yuv12_pipeline::{
    DirectYuv12FrameFeeder, DirectYuv12FrameInput, DirectYuv12FrameSink, DirectYuv12PipelineStats,
    DirectYuv12SourceFrameRange, DirectYuv12WriteSink,
    OneSharedComputeTwoReadbackDirectYuv12Scheduler,
};
use crate::pipe_contract::{
    PipeAspectRatio, PipeAudioContractV3, PipeExampleFacts, PipeMovCadence, PipeSidecarV3,
    checked_pipe_bytes_per_frame, checked_pipe_total_bytes, validate_pipe_sidecar_v3,
};
use crate::strict_motioncam_color::{
    ClipSourceSha256, ColorContextFingerprintFacts, DeferredColorContextFingerprintV2,
    StrictMotionCamColorProfile, StrictMotionCamForwardMatrixColorV2,
    VerifiedStrictPipeColorContextV2,
};

const PIPE_GPU_MEMORY_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;
static PIPE_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn pipe_gpu_required_limits(dimensions: FrameDimensions) -> Result<GpuDecodeRequiredLimits> {
    if dimensions.width == 0 || dimensions.height == 0 {
        bail!(
            "PIPE GPU resource requirements need positive dimensions, got {}x{}",
            dimensions.width,
            dimensions.height,
        );
    }
    let visible_output_bytes = checked_pipe_bytes_per_frame(dimensions.width, dimensions.height)
        .context("failed to calculate PIPE direct-YUV storage-binding requirement")?;
    let composite_end = visible_output_bytes
        .checked_add(DIRECT_YUV12_STATUS_BYTE_LEN)
        .context("PIPE direct-YUV output/status readback byte count overflow")?;
    let alignment = wgpu::COPY_BUFFER_ALIGNMENT;
    let composite_readback_bytes = composite_end
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("PIPE direct-YUV output/status readback alignment overflow")?;
    Ok(GpuDecodeRequiredLimits::new(
        dimensions,
        "direct YUV12 output",
        visible_output_bytes,
        "direct YUV12 output/status readback",
        composite_readback_bytes,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipeCliBackend {
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipeCliVignette {
    NoCorrection,
    WithCorrection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PipeCliOutput {
    Stdout,
    File(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PipeCliRunConfig {
    pub(crate) input_path: PathBuf,
    pub(crate) backend: PipeCliBackend,
    pub(crate) vignette: PipeCliVignette,
    pub(crate) output: PipeCliOutput,
    pub(crate) payload_feeder_options: PayloadFeederOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PipeOutputLayout {
    mode: PipeOutputMode,
    output_stem: String,
    raw_output_path: Option<PathBuf>,
    audio_sidecar_path: PathBuf,
    metadata_sidecar_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PipeVisibleOutputs {
    output_stem: String,
    raw_output_path: Option<PathBuf>,
    audio_sidecar_path: PathBuf,
    metadata_sidecar_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipeOutputMode {
    File,
    Stdout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PipeAudioSidecarSummary {
    metadata: AudioWavMetadata,
    pcm_data_bytes: u64,
    wav_file_bytes: Option<u64>,
    source_audio_bytes: Option<u64>,
    virtual_wav_bytes: u64,
}

struct PreparedPipeAudio {
    lazy: LazyAudioWav,
    summary: PipeAudioSidecarSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PipeVideoRunSummary {
    frames_processed: usize,
    bytes_written: u64,
    bytes_per_frame: u64,
    expected_total_video_bytes: u64,
    metadata_preflight: Duration,
    source_sha256: Duration,
    source_sha256_bytes_read: u64,
    source_sha256_completed_before_stream_end: bool,
    stream: Duration,
    first_output_latency: Option<Duration>,
    maximum_pending_frames: u32,
    decode_work_plan_reused_frames: u64,
}

struct PipeRenderedVideo {
    summary: PipeVideoRunSummary,
    contexts: PipeStreamContextCollector,
    stream_finished_at: Instant,
}

#[derive(Debug)]
pub(crate) struct PipeFramePreflight {
    pub(crate) clip_info: McrawClipInfo,
    pub(crate) dimensions: FrameDimensions,
    pub(crate) bayer: BayerPattern,
    pub(crate) runtime_source_identity: ClipSourceSha256,
    pub(crate) source_file_bytes: u64,
    pub(crate) payload_plan: PayloadReadPlan,
    pub(crate) color_profile: StrictMotionCamColorProfile,
    pub(crate) cadence: PipeMovCadence,
    pub(crate) metadata_preflight: Duration,
    pub(crate) bytes_per_frame: u64,
    pub(crate) expected_total_video_bytes: u64,
    pub(crate) selected_frame_count: usize,
}

pub(crate) struct PipeFrameStreamSummary {
    pub(crate) setup: Duration,
    pub(crate) frames_processed: usize,
    pub(crate) stream: Duration,
    pub(crate) payload: PayloadFeederStats,
    pub(crate) scheduler: DirectYuv12PipelineStats,
    contexts: PipeStreamContextCollector,
    stream_finished_at: Instant,
}

struct PipeResolvedFrameContext<'a> {
    layout: FramePayloadLayout,
    correction: FixedPointVignetteInputFacts<'a>,
    correction_fingerprint: PipeF32BayerCorrectionFingerprint,
    verified_color: VerifiedStrictPipeColorContextV2,
    deferred_color_fingerprint: DeferredColorContextFingerprintV2,
}

struct PipeStreamContextCollector {
    frame_count: usize,
    deferred_color_fingerprints: Vec<DeferredColorContextFingerprintV2>,
    correction_contexts: BTreeSet<String>,
    ordered_correction: blake3::Hasher,
    payload_layouts: BTreeSet<String>,
    original_dimensions: BTreeSet<String>,
    row_strides: BTreeSet<String>,
    binned_values: BTreeSet<String>,
    remosaic_values: BTreeSet<String>,
    compressed_values: BTreeSet<String>,
}

struct PipeFinalizedContextIdentities {
    source_payload_geometry_identity: Value,
    strict_color_context_identity: Value,
    correction_context_identity: Value,
}

struct PipeSourceHashWorker {
    handle: Option<JoinHandle<std::result::Result<PipeSourceHashCompletion, String>>>,
    cancelled: Arc<AtomicBool>,
}

struct PipeSourceHashCompletion {
    source_sha256: ClipSourceSha256,
    elapsed: Duration,
    finished_at: Instant,
    bytes_read: u64,
}

impl PipeSourceHashWorker {
    fn spawn(input_path: &Path) -> Result<Self> {
        let input_path = input_path.to_path_buf();
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = Arc::clone(&cancelled);
        let handle = thread::Builder::new()
            .name("mcraw4vulkan-pipe-source-sha256".to_string())
            .spawn(move || {
                let started = Instant::now();
                let Some((source_sha256, bytes_read)) =
                    ClipSourceSha256::read_once_until_cancelled(&input_path, &worker_cancelled)
                        .map_err(|error| format!("{error}"))?
                else {
                    return Err("source SHA-256 cancelled after PIPE failure".to_string());
                };
                Ok(PipeSourceHashCompletion {
                    source_sha256,
                    elapsed: started.elapsed(),
                    finished_at: Instant::now(),
                    bytes_read,
                })
            })
            .context("failed to start PIPE source SHA-256 worker")?;
        Ok(Self {
            handle: Some(handle),
            cancelled,
        })
    }

    fn finish(mut self) -> Result<PipeSourceHashCompletion> {
        let handle = self
            .handle
            .take()
            .context("PIPE source SHA-256 worker was already consumed")?;
        handle
            .join()
            .map_err(|_| anyhow!("PIPE source SHA-256 worker panicked"))?
            .map_err(|detail| anyhow!("failed to compute strict PIPE source identity: {detail}"))
    }
}

impl Drop for PipeSourceHashWorker {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.cancelled.store(true, Ordering::Relaxed);
            let _ = handle.join();
        }
    }
}

impl PipeStreamContextCollector {
    fn new(selected_frame_count: usize) -> Self {
        let mut ordered_correction = blake3::Hasher::new();
        ordered_correction.update(b"mcraw4vulkan:public-pipe-correction-context-sequence:v1\0");
        Self {
            frame_count: 0,
            deferred_color_fingerprints: Vec::with_capacity(selected_frame_count),
            correction_contexts: BTreeSet::new(),
            ordered_correction,
            payload_layouts: BTreeSet::new(),
            original_dimensions: BTreeSet::new(),
            row_strides: BTreeSet::new(),
            binned_values: BTreeSet::new(),
            remosaic_values: BTreeSet::new(),
            compressed_values: BTreeSet::new(),
        }
    }

    fn collect(
        &mut self,
        index: usize,
        metadata: &FrameMetadata,
        layout: FramePayloadLayout,
        correction_fingerprint: PipeF32BayerCorrectionFingerprint,
        deferred_color_fingerprint: DeferredColorContextFingerprintV2,
    ) -> Result<()> {
        if index != self.frame_count {
            bail!(
                "PIPE context collection order changed: expected logical frame {}, got {index}",
                self.frame_count,
            );
        }
        let correction_record = correction_fingerprint.record_bytes();
        self.ordered_correction
            .update(&u64::try_from(index)?.to_le_bytes());
        self.ordered_correction.update(&correction_record);
        self.correction_contexts
            .insert(blake3::hash(&correction_record).to_hex().to_string());
        self.deferred_color_fingerprints
            .push(deferred_color_fingerprint);
        self.payload_layouts.insert(payload_layout_label(layout));
        self.original_dimensions.insert(format!(
            "{}x{}",
            metadata.original_width.unwrap_or(metadata.dimensions.width),
            metadata
                .original_height
                .unwrap_or(metadata.dimensions.height),
        ));
        self.row_strides.insert(
            metadata
                .row_stride
                .map_or_else(|| "absent".to_string(), |value| value.to_string()),
        );
        self.binned_values
            .insert(option_bool_label(metadata.is_binned));
        self.remosaic_values
            .insert(option_bool_label(metadata.need_remosaic));
        self.compressed_values
            .insert(option_bool_label(metadata.is_compressed));
        self.frame_count = self
            .frame_count
            .checked_add(1)
            .context("PIPE context frame counter overflow")?;
        Ok(())
    }

    fn finalize(
        self,
        preflight: &PipeFramePreflight,
        correction_mode: PipeF32BayerCorrectionMode,
        source_sha256: ClipSourceSha256,
    ) -> Result<PipeFinalizedContextIdentities> {
        if self.frame_count != preflight.selected_frame_count
            || self.deferred_color_fingerprints.len() != preflight.selected_frame_count
        {
            bail!(
                "PIPE context completion count mismatch: collected={} colors={} expected={}",
                self.frame_count,
                self.deferred_color_fingerprints.len(),
                preflight.selected_frame_count,
            );
        }

        let mut ordered_color = blake3::Hasher::new();
        ordered_color.update(b"mcraw4vulkan:public-pipe-color-context-sequence:v1\0");
        let mut color_contexts = BTreeSet::<String>::new();
        let mut color_policies = std::collections::BTreeMap::new();
        let mut first_color = None;
        let mut last_color = None;
        for (index, deferred) in self.deferred_color_fingerprints.into_iter().enumerate() {
            let (name, digest) = deferred.policy_identity();
            color_policies.insert(name, hex_bytes(&digest));
            let fingerprint = deferred.finalize(source_sha256).with_context(|| {
                format!("PIPE strict color identity finalization failed at logical frame {index}")
            })?;
            let bytes = fingerprint.bytes();
            let fingerprint_hex = hex_bytes(&bytes);
            ordered_color.update(&u64::try_from(index)?.to_le_bytes());
            ordered_color.update(&bytes);
            color_contexts.insert(fingerprint_hex.clone());
            if first_color.is_none() {
                first_color = Some(fingerprint_hex.clone());
            }
            last_color = Some(fingerprint_hex);
        }

        let source_sha_hex = hex_bytes(&source_sha256.bytes());
        let source_payload_geometry_identity = json!({
            "source_sha256": source_sha_hex,
            "source_file_bytes": preflight.source_file_bytes,
            "frame_count": preflight.selected_frame_count,
            "visible_width": preflight.dimensions.width,
            "visible_height": preflight.dimensions.height,
            "cfa": bayer_pattern_label(preflight.bayer),
            "payload_layouts": self.payload_layouts.into_iter().collect::<Vec<_>>(),
            "original_dimensions": self.original_dimensions.into_iter().collect::<Vec<_>>(),
            "row_strides": self.row_strides.into_iter().collect::<Vec<_>>(),
            "binned_values": self.binned_values.into_iter().collect::<Vec<_>>(),
            "remosaic_values": self.remosaic_values.into_iter().collect::<Vec<_>>(),
            "compressed_values": self.compressed_values.into_iter().collect::<Vec<_>>(),
            "payload_total_bytes": preflight.payload_plan.total_payload_bytes,
            "payload_max_bytes": preflight.payload_plan.max_payload_bytes,
            "payload_spans_validated_without_decode": true,
        });
        let (policy_name, policy_digest) = if color_policies.len() == 1 {
            let (&name, digest) = color_policies.first_key_value().expect("one color policy");
            (name, Value::String(digest.clone()))
        } else {
            (
                "MixedMotionCamColorProfilesV1",
                serde_json::to_value(&color_policies)?,
            )
        };
        let strict_color_context_identity = json!({
            "policy_id": policy_name,
            "policy_digest_sha256": policy_digest,
            "source_sha256": source_sha_hex,
            "context_count": self.frame_count,
            "unique_context_count": color_contexts.len(),
            "ordered_contexts_blake3": ordered_color.finalize().to_hex().to_string(),
            "first_context_fingerprint": first_color,
            "last_context_fingerprint": last_color,
        });
        let correction_context_identity = json!({
            "policy_id": crate::pipe_contract::PIPE_CORRECTION_POLICY_ID,
            "terminal_id": crate::pipe_contract::PIPE_CORRECTION_TERMINAL_ID,
            "context_count": self.frame_count,
            "unique_context_count": self.correction_contexts.len(),
            "ordered_contexts_blake3": self.ordered_correction.finalize().to_hex().to_string(),
            "spatial_mode": correction_mode.label(),
            "non_spatial_processing_retained": true,
        });

        Ok(PipeFinalizedContextIdentities {
            source_payload_geometry_identity,
            strict_color_context_identity,
            correction_context_identity,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PipeTempPaths {
    owned_dir: PathBuf,
    raw_part_path: Option<PathBuf>,
    audio_part_path: Option<PathBuf>,
    metadata_part_path: PathBuf,
}

#[derive(Debug)]
struct PublishedPipeOutput {
    path: PathBuf,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

#[derive(Debug)]
struct PipeCliCompletion {
    video: PipeVideoRunSummary,
    audio_present: bool,
    published: Vec<PublishedPipeOutput>,
}

impl PipeCliRunConfig {
    pub fn correction_mode(&self) -> PipeF32BayerCorrectionMode {
        match self.vignette {
            PipeCliVignette::NoCorrection => PipeF32BayerCorrectionMode::IdentitySpatialGain,
            PipeCliVignette::WithCorrection => PipeF32BayerCorrectionMode::MotionCamSpatial,
        }
    }

    pub fn validate_before_open(&self) -> Result<()> {
        if matches!(self.output, PipeCliOutput::Stdout) && std::io::stdout().is_terminal() {
            bail!(
                "refusing to write raw yuv444p12le bytes to a terminal; pass --output FILE or pipe stdout to a file"
            );
        }
        Ok(())
    }

    fn output_layout(&self) -> Result<PipeOutputLayout> {
        match &self.output {
            PipeCliOutput::File(path) => pipe_file_output_layout(&self.input_path, path),
            PipeCliOutput::Stdout => pipe_stdout_output_layout(&self.input_path),
        }
    }
}

/// Read only the indexed source facts needed to format a GUI Pipe Example.
///
/// The display-fast container path parses frame zero plus the compact frame
/// index/timestamps. It does not read payload bytes, hash the complete source,
/// initialize a GPU, or run the production all-frame context preflight.
pub fn pipe_example_facts_for_input(input_path: &Path) -> Result<PipeExampleFacts> {
    let container = McrawContainer::open_for_display(input_path)
        .with_context(|| format!("failed to inspect input {}", input_path.display()))?;
    pipe_example_facts_from_container(&container)
}

fn pipe_example_facts_from_container(container: &McrawContainer) -> Result<PipeExampleFacts> {
    let clip = container.clip_info();
    let source_rate = clip.timing.playback_frame_rate.reduced();
    let cadence = PipeMovCadence::from_source_rate(source_rate.numerator, source_rate.denominator)
        .context("failed to derive established bounded PIPE cadence")?;
    let sample_aspect_ratio = PipeAspectRatio::square_pixels();
    let display_aspect_ratio =
        PipeAspectRatio::display_for_frame(clip.width, clip.height, sample_aspect_ratio)
            .context("failed to derive PIPE display aspect ratio")?;

    Ok(PipeExampleFacts {
        width: clip.width,
        height: clip.height,
        cadence,
        sample_aspect_ratio,
        display_aspect_ratio,
    })
}

// Fixed clip/output facts are bounded before streaming. Frame-specific
// correction and strict-color validation happens immediately before each
// frame's one selected decode; the completed source hash is joined only for
// success-sidecar publication.
pub(crate) fn run_pipe_cli(config: PipeCliRunConfig) -> Result<()> {
    config.validate_before_open()?;
    let layout = config.output_layout()?;
    ensure_output_is_not_input(&config.input_path, &layout)?;

    // Reuse the indexed, audio-aware lazy-frame container: it validates the
    // compact container/index and frame zero now, then parses later typed
    // frame metadata on demand in stream_pipe_frames.
    let container = McrawContainer::open_for_display_with_audio(&config.input_path)
        .with_context(|| format!("failed to open input {}", config.input_path.display()))?;
    let audio_present = container.audio_info().is_some();
    prepare_output_paths(&layout, audio_present)?;
    let selected_frame_count = usize::try_from(container.clip_info().frame_count)
        .context("PIPE frame count does not fit usize")?;
    let preflight = preflight_pipe_frames(
        &config.input_path,
        config.backend,
        &container,
        selected_frame_count,
    )?;
    let temps = create_temp_paths_for_layout(&layout, audio_present)?;
    let source_hash_worker = PipeSourceHashWorker::spawn(&config.input_path)?;

    let run_result = run_pipe_cli_inner(
        &config,
        &container,
        &preflight,
        &layout,
        &temps,
        source_hash_worker,
    );
    let cleanup_result = cleanup_temp_paths(&temps);
    match (run_result, cleanup_result) {
        (Ok(completion), Ok(())) => {
            print_pipe_completion(&completion, &layout);
            Ok(())
        }
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(cleanup_error)) => Err(anyhow!(
            "{error:#}; additionally failed to clean owned PIPE temporary directory {}: {cleanup_error:#}",
            temps.owned_dir.display(),
        )),
        (Ok(completion), Err(cleanup_error)) => {
            let rollback_result = rollback_published_outputs(&completion.published);
            match rollback_result {
                Ok(()) => Err(cleanup_error).with_context(|| {
                    format!(
                        "failed to clean owned PIPE temporary directory {}; published outputs rolled back",
                        temps.owned_dir.display(),
                    )
                }),
                Err(rollback_error) => Err(anyhow!(
                    "failed to clean owned PIPE temporary directory {}: {cleanup_error:#}; additionally failed to roll back published outputs: {rollback_error:#}",
                    temps.owned_dir.display(),
                )),
            }
        }
    }
}

fn run_pipe_cli_inner(
    config: &PipeCliRunConfig,
    container: &McrawContainer,
    preflight: &PipeFramePreflight,
    layout: &PipeOutputLayout,
    temps: &PipeTempPaths,
    source_hash_worker: PipeSourceHashWorker,
) -> Result<PipeCliCompletion> {
    let rendered = match layout.mode {
        PipeOutputMode::File => {
            let raw_part_path = temps
                .raw_part_path
                .as_ref()
                .context("raw part path missing")?;
            let mut writer = BufWriter::new(create_new_file(raw_part_path)?);
            let mut sink = DirectYuv12WriteSink::new(&mut writer);
            let stream = stream_pipe_frames(
                &config.input_path,
                config.backend,
                config.correction_mode(),
                GpuBackendPreference::VulkanOnly,
                config.payload_feeder_options,
                preflight.payload_plan.clone(),
                container,
                preflight,
                &mut sink,
            )
            .with_context(|| {
                format!(
                    "failed while writing PIPE raw video to {}",
                    raw_part_path.display()
                )
            })?;
            writer.flush()?;
            public_video_summary(preflight, stream)
        }
        PipeOutputMode::Stdout => {
            let stdout = std::io::stdout();
            let mut writer = stdout.lock();
            let mut sink = DirectYuv12WriteSink::new(&mut writer);
            let stream = stream_pipe_frames(
                &config.input_path,
                config.backend,
                config.correction_mode(),
                GpuBackendPreference::VulkanOnly,
                config.payload_feeder_options,
                preflight.payload_plan.clone(),
                container,
                preflight,
                &mut sink,
            )
            .context("failed while writing PIPE raw video to stdout")?;
            writer.flush()?;
            public_video_summary(preflight, stream)
        }
    };
    let mut video = rendered.summary;

    // Audio layout construction and PCM materialization happen only after the
    // rawvideo stream completes. The initial container open already validated
    // the indexed audio metadata needed to decide whether the sidecar exists;
    // an audio failure here still prevents every success-sidecar publication.
    let mut prepared_audio = prepare_pipe_audio(&config.input_path, container)?;
    let audio_summary = if let Some(prepared_audio) = prepared_audio.as_mut() {
        let audio_part_path = temps
            .audio_part_path
            .as_ref()
            .context("audio part path missing")?;
        Some(write_audio_sidecar_part(prepared_audio, audio_part_path)?)
    } else {
        None
    };

    if video.bytes_per_frame != preflight.bytes_per_frame
        || video.bytes_written != preflight.expected_total_video_bytes
        || video.frames_processed != preflight.clip_info.frame_count as usize
    {
        bail!(
            "PIPE completion count mismatch: frames={}/{} bytes_per_frame={}/{} bytes={}/{}",
            video.frames_processed,
            preflight.clip_info.frame_count,
            video.bytes_per_frame,
            preflight.bytes_per_frame,
            video.bytes_written,
            preflight.expected_total_video_bytes,
        );
    }

    let source_hash = source_hash_worker.finish()?;
    video.source_sha256 = source_hash.elapsed;
    video.source_sha256_bytes_read = source_hash.bytes_read;
    video.source_sha256_completed_before_stream_end =
        source_hash.finished_at <= rendered.stream_finished_at;
    let identities = rendered.contexts.finalize(
        preflight,
        config.correction_mode(),
        source_hash.source_sha256,
    )?;

    let sidecar =
        pipe_metadata_sidecar_json(config, preflight, &identities, layout, audio_summary, video)?;
    validate_pipe_sidecar_v3(&sidecar).context("generated PIPE sidecar v3 failed validation")?;
    write_metadata_sidecar_part(&temps.metadata_part_path, &sidecar)?;

    let mut published = Vec::<PublishedPipeOutput>::new();
    let publish_result = (|| -> Result<()> {
        if let Some(audio_part_path) = &temps.audio_part_path {
            published.push(
                publish_temp_no_replace(audio_part_path, &layout.audio_sidecar_path)
                    .context("failed to finalize PIPE WAV sidecar")?,
            );
        }
        published.push(
            publish_temp_no_replace(&temps.metadata_part_path, &layout.metadata_sidecar_path)
                .context("failed to finalize PIPE metadata sidecar")?,
        );
        if let (Some(raw_part_path), Some(raw_output_path)) =
            (&temps.raw_part_path, &layout.raw_output_path)
        {
            published.push(
                publish_temp_no_replace(raw_part_path, raw_output_path)
                    .context("failed to finalize raw PIPE output")?,
            );
        }
        Ok(())
    })();
    if let Err(error) = publish_result {
        return match rollback_published_outputs(&published) {
            Ok(()) => Err(error),
            Err(rollback_error) => Err(anyhow!(
                "{error:#}; additionally failed to roll back partially published PIPE outputs: {rollback_error:#}"
            )),
        };
    }

    Ok(PipeCliCompletion {
        video,
        audio_present: audio_summary.is_some(),
        published,
    })
}

fn print_pipe_completion(completion: &PipeCliCompletion, layout: &PipeOutputLayout) {
    let video = completion.video;
    eprintln!(
        "PIPE wrote {} yuv444p12le frames, {} bytes/frame, {} video bytes; metadata_preflight={:.3}s source_sha256={:.3}s source_sha256_bytes={} source_sha256_completed_before_stream_end={} stream={:.3}s first_output={:.3}s max_pending={}",
        video.frames_processed,
        video.bytes_per_frame,
        video.bytes_written,
        video.metadata_preflight.as_secs_f64(),
        video.source_sha256.as_secs_f64(),
        video.source_sha256_bytes_read,
        video.source_sha256_completed_before_stream_end,
        video.stream.as_secs_f64(),
        video.first_output_latency.unwrap_or_default().as_secs_f64(),
        video.maximum_pending_frames,
    );
    match layout.raw_output_path.as_ref() {
        Some(path) => eprintln!("PIPE raw output: {}", path.display()),
        None => eprintln!("PIPE raw output: stdout"),
    }
    if completion.audio_present {
        eprintln!("PIPE WAV sidecar: {}", layout.audio_sidecar_path.display());
    } else {
        eprintln!("PIPE WAV sidecar: none (input has no audio)");
    }
    eprintln!(
        "PIPE metadata sidecar: {}",
        layout.metadata_sidecar_path.display()
    );
}

fn rollback_published_outputs(published: &[PublishedPipeOutput]) -> Result<()> {
    let mut failures = Vec::new();
    for output in published.iter().rev() {
        if let Err(error) = output.rollback() {
            failures.push(format!("{}: {error:#}", output.path.display()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        bail!("{}", failures.join("; "))
    }
}

impl PublishedPipeOutput {
    fn capture(path: &Path) -> Result<Self> {
        let metadata = fs::symlink_metadata(path).with_context(|| {
            format!("failed to inspect published PIPE output {}", path.display())
        })?;
        if !metadata.file_type().is_file() {
            bail!(
                "published PIPE output is not a regular file: {}",
                path.display()
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
        })
    }

    fn rollback(&self) -> Result<()> {
        let metadata = match fs::symlink_metadata(&self.path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to inspect published PIPE output {}",
                        self.path.display()
                    )
                });
            }
        };
        #[cfg(unix)]
        if metadata.dev() != self.device || metadata.ino() != self.inode {
            bail!(
                "refusing to remove replaced PIPE output {} during rollback",
                self.path.display(),
            );
        }
        fs::remove_file(&self.path)
            .with_context(|| format!("failed to roll back PIPE output {}", self.path.display()))
    }
}

pub(crate) fn preflight_pipe_frames(
    input_path: &Path,
    backend: PipeCliBackend,
    container: &McrawContainer,
    selected_frame_count: usize,
) -> Result<PipeFramePreflight> {
    let started = Instant::now();
    let clip_info = container.clip_info().clone();
    if clip_info.frame_count == 0 {
        bail!("PIPE output requires at least one video frame");
    }
    let full_frame_count =
        usize::try_from(clip_info.frame_count).context("PIPE frame count does not fit usize")?;
    if container.frame_count() != full_frame_count {
        bail!(
            "PIPE indexed frame count {} contradicts clip count {}",
            container.frame_count(),
            clip_info.frame_count
        );
    }
    if selected_frame_count == 0 || selected_frame_count > full_frame_count {
        bail!("PIPE selected frame count {selected_frame_count} is outside 1..={full_frame_count}");
    }

    let source_file_bytes = fs::metadata(input_path)
        .with_context(|| format!("failed to stat {}", input_path.display()))?
        .len();
    // The complete file digest is deliberately deferred to the background
    // worker. This internal token binds the profile, per-frame resolve, and
    // scheduler identity while streaming; it is never published. Deferred
    // fingerprint records are rebound to the exact source digest before the
    // success sidecar is constructed.
    let runtime_source_identity = ClipSourceSha256::deferred_stream_identity();
    let provenance =
        StrictColorProfileProvenance::from_source_sha256(runtime_source_identity.bytes());
    let resolver = StrictMotionCamForwardMatrixColorV2::new()?;
    let color_profile = resolver
        .parse_supported_profile(container.container_metadata_json(), provenance)
        .context("strict PIPE color-profile preflight failed")?;
    let sensor = &container.container_metadata().sensor_arrangement;
    let bayer = sensor
        .bayer_pattern()
        .with_context(|| format!("unsupported CFA {sensor:?}"))?;
    let first_metadata = container
        .frame_metadata(FrameNumber(0))
        .context("failed to read first frame metadata")?;
    let dimensions = first_metadata.dimensions;
    let first_layout = first_metadata
        .payload_layout()
        .context("PIPE first-frame payload layout failed")?;
    if backend == PipeCliBackend::Gpu && !first_layout.supports_native_gpu_decode() {
        bail!("PIPE GPU route does not support first-frame payload layout");
    }
    let bytes_per_frame = checked_pipe_bytes_per_frame(dimensions.width, dimensions.height)
        .context("PIPE bytes-per-frame overflow")?;
    let expected_total_video_bytes = checked_pipe_total_bytes(
        dimensions.width,
        dimensions.height,
        u64::try_from(selected_frame_count)?,
    )
    .context("PIPE total-byte count overflow")?;

    let frames = (0..selected_frame_count)
        .map(|index| {
            u32::try_from(index)
                .map(FrameNumber)
                .with_context(|| format!("PIPE logical frame {index} does not fit u32"))
        })
        .collect::<Result<Vec<_>>>()?;
    let payload_plan = PayloadReadPlan::from_core_frame_numbers(container, &frames)
        .context("PIPE payload-span preflight failed")?;
    if payload_plan.selected_frames() != selected_frame_count {
        bail!("PIPE payload plan omitted indexed frames");
    }
    for entry in &payload_plan.entries {
        let end = entry.end_offset()?;
        if entry.payload_len == 0 || end > source_file_bytes {
            bail!(
                "PIPE payload span is invalid at logical frame {}: offset={} bytes={} source_bytes={}",
                entry.frame_number,
                entry.payload_offset,
                entry.payload_len,
                source_file_bytes,
            );
        }
    }

    let source_rate = clip_info.timing.playback_frame_rate.reduced();
    let cadence = PipeMovCadence::from_source_rate(source_rate.numerator, source_rate.denominator)
        .context("failed to derive established bounded PIPE cadence")?;

    Ok(PipeFramePreflight {
        clip_info,
        dimensions,
        bayer,
        runtime_source_identity,
        source_file_bytes,
        payload_plan,
        color_profile,
        cadence,
        metadata_preflight: started.elapsed(),
        bytes_per_frame,
        expected_total_video_bytes,
        selected_frame_count,
    })
}

fn resolve_pipe_frame_context<'a>(
    resolver: &StrictMotionCamForwardMatrixColorV2,
    container: &McrawContainer,
    preflight: &PipeFramePreflight,
    metadata: &'a FrameMetadata,
    number: FrameNumber,
    selected_backend: PipeCliBackend,
    correction_mode: PipeF32BayerCorrectionMode,
) -> Result<PipeResolvedFrameContext<'a>> {
    let index = number.0 as usize;
    if metadata.dimensions != preflight.dimensions {
        bail!(
            "PIPE frame dimensions changed at logical frame {index}: expected {}x{}, got {}x{}",
            preflight.dimensions.width,
            preflight.dimensions.height,
            metadata.dimensions.width,
            metadata.dimensions.height,
        );
    }
    let layout = metadata
        .payload_layout()
        .with_context(|| format!("PIPE payload layout failed at logical frame {index}"))?;
    if selected_backend == PipeCliBackend::Gpu && !layout.supports_native_gpu_decode() {
        bail!("PIPE GPU route does not support payload layout at logical frame {index}");
    }
    let correction = motioncam_pipe_f32_bayer_facts(
        container.container_metadata(),
        metadata,
        preflight.bayer,
        correction_mode,
    )
    .with_context(|| format!("PIPE correction validation failed at logical frame {index}"))?;
    let correction_fingerprint =
        PipeF32BayerCorrectionFingerprint::from_fixed_facts(&correction, correction_mode)
            .with_context(|| format!("PIPE correction identity failed at logical frame {index}"))?;
    let frame_json = container
        .frame_metadata_json(number)
        .with_context(|| format!("PIPE color metadata read failed at logical frame {index}"))?;
    let provenance =
        StrictColorProfileProvenance::from_source_sha256(preflight.runtime_source_identity.bytes());
    let frame_input = resolver
        .parse_frame_input(&frame_json, u64::try_from(index)?, provenance)
        .with_context(|| format!("PIPE strict color input failed at logical frame {index}"))?;
    let effective_profile =
        resolver.effective_profile(&preflight.color_profile, &metadata.color_overrides)?;
    let (verified_color, deferred_color_fingerprint) = resolver
        .resolve_stream_context(
            &effective_profile,
            &frame_input,
            ColorContextFingerprintFacts {
                numeric_domain: PipeF32BayerNumericDomain::RelativeLinearCorrectedCodeV1,
                dimensions: preflight.dimensions,
                bayer_pattern: preflight.bayer,
                correction_mode,
                source_sha256: preflight.runtime_source_identity,
                source_frame_index: u64::try_from(index)?,
            },
        )
        .with_context(|| format!("PIPE strict color solve failed at logical frame {index}"))?;

    Ok(PipeResolvedFrameContext {
        layout,
        correction,
        correction_fingerprint,
        verified_color,
        deferred_color_fingerprint,
    })
}

fn collect_pipe_frame_contexts_without_decode(
    container: &McrawContainer,
    preflight: &PipeFramePreflight,
    selected_backend: PipeCliBackend,
    correction_mode: PipeF32BayerCorrectionMode,
) -> Result<PipeStreamContextCollector> {
    let resolver = StrictMotionCamForwardMatrixColorV2::new()?;
    let mut contexts = PipeStreamContextCollector::new(preflight.selected_frame_count);
    for index in 0..preflight.selected_frame_count {
        let number = FrameNumber(
            u32::try_from(index)
                .with_context(|| format!("PIPE logical frame {index} does not fit u32"))?,
        );
        let metadata = container
            .frame_metadata(number)
            .with_context(|| format!("PIPE metadata validation failed at logical frame {index}"))?;
        let context = resolve_pipe_frame_context(
            &resolver,
            container,
            preflight,
            metadata,
            number,
            selected_backend,
            correction_mode,
        )?;
        contexts.collect(
            index,
            metadata,
            context.layout,
            context.correction_fingerprint,
            context.deferred_color_fingerprint,
        )?;
    }
    Ok(contexts)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn stream_pipe_frames<S: DirectYuv12FrameSink>(
    input_path: &Path,
    selected_backend: PipeCliBackend,
    correction_mode: PipeF32BayerCorrectionMode,
    backend_preference: GpuBackendPreference,
    payload_feeder_options: PayloadFeederOptions,
    payload_plan: PayloadReadPlan,
    container: &McrawContainer,
    preflight: &PipeFramePreflight,
    sink: &mut S,
) -> Result<PipeFrameStreamSummary> {
    let setup_started = Instant::now();
    let selected_frame_count = payload_plan.selected_frames();
    if selected_frame_count == 0 || selected_frame_count > preflight.selected_frame_count {
        bail!(
            "PIPE stream frame count {selected_frame_count} is outside preflighted range 1..={}",
            preflight.selected_frame_count,
        );
    }
    for (sequence, entry) in payload_plan.entries.iter().enumerate() {
        if entry.frame_number != sequence {
            bail!(
                "PIPE stream plan must be first-N contiguous: expected logical frame {sequence}, got {}",
                entry.frame_number,
            );
        }
    }
    let required_limits = pipe_gpu_required_limits(preflight.dimensions)?;
    let mut payload_feeder =
        PayloadFeeder::spawn(input_path, payload_plan, payload_feeder_options)?;
    let backend = GpuDecodeBackend::new_blocking_with_required_limits(
        GpuDecodeConfig {
            backend_preference,
            ..GpuDecodeConfig::default()
        },
        required_limits,
    )?;
    let mut scheduler = OneSharedComputeTwoReadbackDirectYuv12Scheduler::new(
        backend,
        DirectYuv12SourceFrameRange {
            first_frame_index: 0,
            frame_count: u64::try_from(selected_frame_count)?,
        },
        PIPE_GPU_MEMORY_BUDGET_BYTES,
    )?;
    let mut cpu_decoder = (selected_backend == PipeCliBackend::Cpu).then(CpuFrameDecoder::new);
    let resolver = StrictMotionCamForwardMatrixColorV2::new()?;
    let mut contexts = PipeStreamContextCollector::new(selected_frame_count);
    let setup = setup_started.elapsed();
    let started = Instant::now();
    let mut frames_processed = 0usize;

    while let Some(payload) = payload_feeder.next_frame()? {
        let index = frames_processed;
        if payload.frame_number != index || payload.playback_index != index {
            bail!(
                "PIPE payload order changed: expected logical frame {index}, got source={} playback={}",
                payload.frame_number,
                payload.playback_index,
            );
        }
        let number = payload.frame_number_core()?;
        let metadata = container
            .frame_metadata(number)
            .with_context(|| format!("PIPE metadata validation failed at logical frame {index}"))?;
        let context = resolve_pipe_frame_context(
            &resolver,
            container,
            preflight,
            metadata,
            number,
            selected_backend,
            correction_mode,
        )?;

        match selected_backend {
            PipeCliBackend::Gpu => scheduler.submit_frame(
                DirectYuv12FrameInput {
                    feeder: DirectYuv12FrameFeeder::NativePayload {
                        raw_payload: payload.data.as_slice(),
                        payload_layout: context.layout,
                    },
                    dimensions: preflight.dimensions,
                    correction_facts: &context.correction,
                    correction_mode,
                    verified_color: &context.verified_color,
                    source_sha256: preflight.runtime_source_identity,
                    source_frame_index: u64::try_from(index)?,
                },
                sink,
            ),
            PipeCliBackend::Cpu => {
                let decoder = cpu_decoder.as_mut().context("PIPE CPU decoder missing")?;
                decoder.prepare_compressed(payload.data.len());
                decoder
                    .compressed_mut()
                    .extend_from_slice(payload.data.as_slice());
                let mut timings = DecodeFrameTimings::default();
                let (decoded, _) = decoder
                    .decode_loaded_payload_to_decoded_bayer_u16_frame_with_layout(
                        preflight.dimensions,
                        context.layout,
                        &mut timings,
                    )
                    .with_context(|| format!("CPU decoding PIPE logical frame {index}"))?;
                let decoded_bytes = decoded.into_owned_le_bytes();
                scheduler.submit_frame(
                    DirectYuv12FrameInput {
                        feeder: DirectYuv12FrameFeeder::CpuDecodedPackedU16 {
                            decoded_pixel_bytes_le: &decoded_bytes,
                        },
                        dimensions: preflight.dimensions,
                        correction_facts: &context.correction,
                        correction_mode,
                        verified_color: &context.verified_color,
                        source_sha256: preflight.runtime_source_identity,
                        source_frame_index: u64::try_from(index)?,
                    },
                    sink,
                )
            }
        }
        .with_context(|| format!("PIPE direct-YUV processing failed at logical frame {index}"))?;
        contexts.collect(
            index,
            metadata,
            context.layout,
            context.correction_fingerprint,
            context.deferred_color_fingerprint,
        )?;
        frames_processed = frames_processed
            .checked_add(1)
            .context("PIPE frame counter overflow")?;
    }
    let payload = payload_feeder.finish()?;
    let scheduler = scheduler.finish(sink)?;
    let stream = started.elapsed();
    if scheduler.published_frames != u64::try_from(frames_processed)? {
        bail!(
            "PIPE scheduler published {} of {} submitted frames",
            scheduler.published_frames,
            frames_processed,
        );
    }

    Ok(PipeFrameStreamSummary {
        setup,
        frames_processed,
        stream,
        payload,
        scheduler,
        contexts,
        stream_finished_at: Instant::now(),
    })
}

fn public_video_summary(
    preflight: &PipeFramePreflight,
    stream: PipeFrameStreamSummary,
) -> PipeRenderedVideo {
    let bytes_written = preflight
        .bytes_per_frame
        .checked_mul(stream.scheduler.published_frames)
        .expect("preflighted PIPE byte count remains representable");
    PipeRenderedVideo {
        summary: PipeVideoRunSummary {
            frames_processed: stream.frames_processed,
            bytes_written,
            bytes_per_frame: preflight.bytes_per_frame,
            expected_total_video_bytes: preflight.expected_total_video_bytes,
            metadata_preflight: preflight.metadata_preflight,
            source_sha256: Duration::ZERO,
            source_sha256_bytes_read: 0,
            source_sha256_completed_before_stream_end: false,
            stream: stream.stream,
            first_output_latency: stream.scheduler.first_output_latency,
            maximum_pending_frames: stream.scheduler.maximum_pending_frames,
            decode_work_plan_reused_frames: stream.scheduler.decode_work_plan_reused_frames,
        },
        contexts: stream.contexts,
        stream_finished_at: stream.stream_finished_at,
    }
}

fn pipe_file_output_layout(input: &Path, raw_output_path: &Path) -> Result<PipeOutputLayout> {
    let current_dir =
        env::current_dir().context("failed to resolve current directory for PIPE outputs")?;
    pipe_file_output_layout_in_current_dir(input, raw_output_path, &current_dir)
}

fn pipe_file_output_layout_in_current_dir(
    input: &Path,
    raw_output_path: &Path,
    current_dir: &Path,
) -> Result<PipeOutputLayout> {
    let outputs = resolve_pipe_visible_outputs(input, Some(raw_output_path), current_dir)?;
    Ok(PipeOutputLayout {
        mode: PipeOutputMode::File,
        output_stem: outputs.output_stem,
        raw_output_path: outputs.raw_output_path,
        audio_sidecar_path: outputs.audio_sidecar_path,
        metadata_sidecar_path: outputs.metadata_sidecar_path,
    })
}

fn pipe_stdout_output_layout(input: &Path) -> Result<PipeOutputLayout> {
    let current_dir =
        env::current_dir().context("failed to resolve current directory for PIPE outputs")?;
    pipe_stdout_output_layout_in_current_dir(input, &current_dir)
}

fn pipe_stdout_output_layout_in_current_dir(
    input: &Path,
    current_dir: &Path,
) -> Result<PipeOutputLayout> {
    let outputs = resolve_pipe_visible_outputs(input, None, current_dir)?;
    Ok(PipeOutputLayout {
        mode: PipeOutputMode::Stdout,
        output_stem: outputs.output_stem,
        raw_output_path: None,
        audio_sidecar_path: outputs.audio_sidecar_path,
        metadata_sidecar_path: outputs.metadata_sidecar_path,
    })
}

fn resolve_pipe_visible_outputs(
    input: &Path,
    output_arg: Option<&Path>,
    current_dir: &Path,
) -> Result<PipeVisibleOutputs> {
    let (parent, base_stem, raw_output_path) = if let Some(output_arg) = output_arg {
        let parent = output_parent_for_path(output_arg, current_dir);
        let base_stem = visible_base_stem_for_output_arg(output_arg)?;
        let output_stem = visible_direct_yuv_stem(&base_stem);
        let raw_output_path = Some(parent.join(format!("{output_stem}.yuv444p12le")));
        (parent, base_stem, raw_output_path)
    } else {
        (current_dir.to_path_buf(), path_stem(input)?, None)
    };

    let output_stem = visible_direct_yuv_stem(&base_stem);
    Ok(PipeVisibleOutputs {
        audio_sidecar_path: parent.join(format!("{base_stem}-audio.wav")),
        metadata_sidecar_path: parent.join(format!("{output_stem}.json")),
        raw_output_path,
        output_stem,
    })
}

fn output_parent_for_path(output_arg: &Path, current_dir: &Path) -> PathBuf {
    let parent = output_arg.parent().unwrap_or_else(|| Path::new(""));
    if parent.as_os_str().is_empty() || parent == Path::new(".") {
        current_dir.to_path_buf()
    } else if parent.is_absolute() {
        parent.to_path_buf()
    } else {
        current_dir.join(parent)
    }
}

fn visible_base_stem_for_output_arg(output_arg: &Path) -> Result<String> {
    let file_name = output_arg
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            anyhow!(
                "path {:?} does not have a valid UTF-8 file name",
                output_arg
            )
        })?;
    let stem = if let Some(stem) = file_name.strip_suffix(".yuv444p12le") {
        stem
    } else {
        output_arg
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| {
                anyhow!(
                    "path {:?} does not have a valid UTF-8 file stem",
                    output_arg
                )
            })?
    };
    let stem = stem.strip_suffix("-BT2020-linear-tv").unwrap_or(stem);
    if stem.is_empty() {
        bail!(
            "path {:?} does not have a non-empty output stem",
            output_arg
        );
    }
    Ok(stem.to_string())
}

fn visible_direct_yuv_stem(base_stem: &str) -> String {
    let base_stem = base_stem
        .strip_suffix("-BT2020-linear-tv")
        .unwrap_or(base_stem);
    format!("{base_stem}-BT2020-linear-tv")
}

fn ensure_output_is_not_input(input: &Path, layout: &PipeOutputLayout) -> Result<()> {
    if let Some(raw_path) = &layout.raw_output_path {
        if raw_path == input {
            bail!("pipe --output path must not equal the input path");
        }
    }
    Ok(())
}

fn prepare_output_paths(layout: &PipeOutputLayout, audio_present: bool) -> Result<()> {
    ensure_distinct_final_paths(layout, audio_present)?;

    if let Some(raw_output_path) = &layout.raw_output_path {
        ensure_output_parent(raw_output_path)?;
        ensure_missing(raw_output_path)?;
    }

    if audio_present {
        ensure_output_parent(&layout.audio_sidecar_path)?;
        ensure_missing(&layout.audio_sidecar_path)?;
    }
    ensure_output_parent(&layout.metadata_sidecar_path)?;
    ensure_missing(&layout.metadata_sidecar_path)?;
    Ok(())
}

fn ensure_output_parent(path: &Path) -> Result<()> {
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    match fs::symlink_metadata(parent) {
        Ok(metadata) => {
            if !metadata.is_dir() {
                bail!(
                    "PIPE output parent exists but is not a directory: {}",
                    parent.display()
                );
            }
            return Ok(());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            bail!(
                "failed to inspect PIPE output parent {}: {error}",
                parent.display(),
            );
        }
    }
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create PIPE output parent {}", parent.display()))
}

fn ensure_distinct_final_paths(layout: &PipeOutputLayout, audio_present: bool) -> Result<()> {
    let mut paths: Vec<(&str, &Path)> = Vec::new();
    if let Some(raw_output_path) = layout.raw_output_path.as_deref() {
        paths.push(("raw output", raw_output_path));
    }
    if audio_present {
        paths.push(("WAV sidecar", &layout.audio_sidecar_path));
    }
    paths.push(("metadata sidecar", &layout.metadata_sidecar_path));

    for (left_index, (left_label, left_path)) in paths.iter().enumerate() {
        for (right_label, right_path) in paths.iter().skip(left_index + 1) {
            if *left_path == *right_path {
                bail!(
                    "PIPE {left_label} path must differ from {right_label} path: {}",
                    left_path.display()
                );
            }
        }
    }

    Ok(())
}

fn create_temp_paths_for_layout(
    layout: &PipeOutputLayout,
    audio_present: bool,
) -> Result<PipeTempPaths> {
    let parent = layout
        .metadata_sidecar_path
        .parent()
        .context("PIPE metadata sidecar has no output parent")?;
    for path in [
        layout.raw_output_path.as_deref(),
        audio_present.then_some(layout.audio_sidecar_path.as_path()),
    ]
    .into_iter()
    .flatten()
    {
        if path.parent() != Some(parent) {
            bail!("PIPE outputs must share one same-filesystem parent");
        }
    }
    let process = std::process::id();
    let owned_dir = (0_u32..1024)
        .find_map(|_| {
            let sequence = PIPE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let candidate = parent.join(format!(".mcraw4vulkan-pipe-{process}-{sequence:016x}"));
            match fs::create_dir(&candidate) {
                Ok(()) => Some(Ok(candidate)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => Some(Err(error)),
            }
        })
        .transpose()
        .with_context(|| {
            format!(
                "failed to create owned PIPE temporary directory in {}",
                parent.display()
            )
        })?
        .context("failed to allocate a unique PIPE temporary directory")?;
    let temp_member = |final_path: &Path| -> Result<PathBuf> {
        let name = final_path
            .file_name()
            .context("PIPE output path has no file name")?;
        Ok(owned_dir.join(name))
    };
    Ok(PipeTempPaths {
        raw_part_path: layout
            .raw_output_path
            .as_deref()
            .map(&temp_member)
            .transpose()?,
        audio_part_path: audio_present
            .then(|| temp_member(&layout.audio_sidecar_path))
            .transpose()?,
        metadata_part_path: temp_member(&layout.metadata_sidecar_path)?,
        owned_dir,
    })
}

fn ensure_missing(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => bail!(
            "refusing to overwrite existing PIPE output path {}",
            path.display(),
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("failed to inspect PIPE output path {}", path.display())),
    }
}

fn cleanup_temp_paths(temps: &PipeTempPaths) -> Result<()> {
    match fs::remove_dir_all(&temps.owned_dir) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| {
            format!(
                "failed to remove owned PIPE temporary directory {}",
                temps.owned_dir.display(),
            )
        }),
    }
}

fn publish_temp_no_replace(temp_path: &Path, final_path: &Path) -> Result<PublishedPipeOutput> {
    fs::hard_link(temp_path, final_path).with_context(|| {
        format!(
            "refusing to replace PIPE output {} while publishing {}",
            final_path.display(),
            temp_path.display(),
        )
    })?;
    let published = match PublishedPipeOutput::capture(final_path) {
        Ok(published) => published,
        Err(error) => {
            return match fs::remove_file(final_path) {
                Ok(()) => {
                    Err(error).context("published PIPE link rolled back after inspection failure")
                }
                Err(rollback_error) => Err(anyhow!(
                    "{error:#}; additionally failed to roll back unverified published PIPE output {}: {rollback_error}",
                    final_path.display(),
                )),
            };
        }
    };
    if let Err(error) = fs::remove_file(temp_path) {
        let rollback = published.rollback();
        return match rollback {
            Ok(()) => Err(error).with_context(|| {
                format!(
                    "published {} but failed to remove temporary link; final link rolled back",
                    final_path.display(),
                )
            }),
            Err(rollback_error) => bail!(
                "published {} but failed to remove temp {} ({error}) and rollback final ({rollback_error})",
                final_path.display(),
                temp_path.display(),
            ),
        };
    }
    Ok(published)
}

fn create_new_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("failed to create {}", path.display()))
}

fn prepare_pipe_audio(
    input_path: &Path,
    container: &McrawContainer,
) -> Result<Option<PreparedPipeAudio>> {
    if container.audio_info().is_none() {
        return Ok(None);
    }

    let source_audio_bytes = Some(container_source_audio_bytes(container)?);
    let audio_container =
        McrawContainer::open_for_display_with_audio(input_path).with_context(|| {
            format!(
                "failed to open lazy PIPE audio from {}",
                input_path.display()
            )
        })?;
    let lazy = LazyAudioWav::from_container(audio_container, LazyAudioWavConfig::default())?;
    let layout = lazy.summary();
    let metadata = layout.metadata;
    let pcm_data_bytes = audio_pcm_data_bytes(
        metadata.sample_frames,
        metadata.channels,
        metadata.bits_per_sample,
    )?;
    if pcm_data_bytes != layout.data_byte_len {
        bail!(
            "PCM data byte count mismatch: formula={} layout={}",
            pcm_data_bytes,
            layout.data_byte_len
        );
    }

    Ok(Some(PreparedPipeAudio {
        lazy,
        summary: PipeAudioSidecarSummary {
            metadata,
            pcm_data_bytes,
            wav_file_bytes: None,
            source_audio_bytes,
            virtual_wav_bytes: metadata.byte_len,
        },
    }))
}

fn write_audio_sidecar_part(
    prepared: &mut PreparedPipeAudio,
    part_path: &Path,
) -> Result<PipeAudioSidecarSummary> {
    let mut writer = BufWriter::new(create_new_file(part_path)?);
    let mut buffer = vec![0u8; 64 * 1024];
    let mut offset = 0u64;
    let byte_len = prepared.lazy.byte_len();
    while offset < byte_len {
        let remaining = usize::try_from((byte_len - offset).min(buffer.len() as u64))
            .context("WAV sidecar read length overflows usize")?;
        let read = prepared
            .lazy
            .read_at(offset, &mut buffer[..remaining])
            .with_context(|| format!("failed to read WAV sidecar bytes at offset {offset}"))?;
        if read == 0 {
            bail!("WAV sidecar read returned EOF before expected byte length at offset {offset}");
        }
        writer.write_all(&buffer[..read])?;
        offset = offset
            .checked_add(u64::try_from(read).context("WAV read length overflows u64")?)
            .context("WAV sidecar offset overflow")?;
    }
    writer.flush()?;
    let mut summary = prepared.summary;
    summary.wav_file_bytes = Some(byte_len);
    Ok(summary)
}

fn container_source_audio_bytes(container: &McrawContainer) -> Result<u64> {
    let mut total = 0_u64;
    for chunk in container.audio_chunks() {
        total = total
            .checked_add(u64::from(chunk.byte_len))
            .context("source audio byte count overflowed")?;
    }
    Ok(total)
}

// A sample frame contains one sample for every interleaved channel, so byte size
// multiplies time-domain frames by channel count and bytes per sample.
fn audio_pcm_data_bytes(sample_frames: u64, channels: u16, bits_per_sample: u16) -> Result<u64> {
    if !bits_per_sample.is_multiple_of(8) {
        bail!("audio bits_per_sample must be byte-aligned");
    }
    sample_frames
        .checked_mul(u64::from(channels))
        .and_then(|value| value.checked_mul(u64::from(bits_per_sample / 8)))
        .context("PCM data byte count overflowed")
}

fn write_metadata_sidecar_part(path: &Path, value: &Value) -> Result<()> {
    let mut writer = BufWriter::new(create_new_file(path)?);
    write_pipe_metadata_json(&mut writer, value)
        .with_context(|| format!("failed to write PIPE metadata sidecar {}", path.display()))
}

fn write_pipe_metadata_json<W: Write>(writer: &mut W, value: &Value) -> Result<()> {
    serde_json::to_writer_pretty(&mut *writer, value)
        .context("failed to serialize PIPE metadata JSON")?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

pub(crate) fn write_pipe_metadata_json_for_config<W: Write>(
    config: &PipeCliRunConfig,
    writer: &mut W,
) -> Result<()> {
    let value = pipe_metadata_json_for_config(config)?;
    write_pipe_metadata_json(writer, &value)
}

fn pipe_metadata_json_for_config(config: &PipeCliRunConfig) -> Result<Value> {
    if !matches!(config.output, PipeCliOutput::Stdout) {
        bail!("PIPE metadata-only config must use stdout output mode");
    }
    let layout = config.output_layout()?;

    let container = McrawContainer::open_for_display_with_audio(&config.input_path)
        .with_context(|| format!("failed to open input {}", config.input_path.display()))?;
    let selected_frame_count = usize::try_from(container.clip_info().frame_count)
        .context("PIPE frame count does not fit usize")?;
    let preflight = preflight_pipe_frames(
        &config.input_path,
        config.backend,
        &container,
        selected_frame_count,
    )?;
    let source_hash_worker = PipeSourceHashWorker::spawn(&config.input_path)?;
    let contexts = collect_pipe_frame_contexts_without_decode(
        &container,
        &preflight,
        config.backend,
        config.correction_mode(),
    )?;
    let source_hash = source_hash_worker.finish()?;
    let identities = contexts.finalize(
        &preflight,
        config.correction_mode(),
        source_hash.source_sha256,
    )?;
    let video = PipeVideoRunSummary {
        frames_processed: usize::try_from(preflight.clip_info.frame_count)?,
        bytes_written: preflight.expected_total_video_bytes,
        bytes_per_frame: preflight.bytes_per_frame,
        expected_total_video_bytes: preflight.expected_total_video_bytes,
        metadata_preflight: preflight.metadata_preflight,
        source_sha256: source_hash.elapsed,
        source_sha256_bytes_read: source_hash.bytes_read,
        source_sha256_completed_before_stream_end: false,
        stream: Duration::ZERO,
        first_output_latency: None,
        maximum_pending_frames: 0,
        decode_work_plan_reused_frames: 0,
    };
    let audio_summary = metadata_only_audio_summary(container)?;
    pipe_metadata_sidecar_json(
        config,
        &preflight,
        &identities,
        &layout,
        audio_summary,
        video,
    )
}

fn metadata_only_audio_summary(
    container: McrawContainer,
) -> Result<Option<PipeAudioSidecarSummary>> {
    if container.audio_info().is_none() {
        return Ok(None);
    }

    let source_audio_bytes = Some(container_source_audio_bytes(&container)?);
    let lazy = LazyAudioWav::from_container(container, LazyAudioWavConfig::default())?;
    let summary = lazy.summary();
    let metadata = summary.metadata;
    let pcm_data_bytes = audio_pcm_data_bytes(
        metadata.sample_frames,
        metadata.channels,
        metadata.bits_per_sample,
    )?;
    if pcm_data_bytes != summary.data_byte_len {
        bail!(
            "PCM data byte count mismatch: formula={} layout={}",
            pcm_data_bytes,
            summary.data_byte_len
        );
    }

    Ok(Some(PipeAudioSidecarSummary {
        metadata,
        pcm_data_bytes,
        wav_file_bytes: Some(metadata.byte_len),
        source_audio_bytes,
        virtual_wav_bytes: metadata.byte_len,
    }))
}

fn pipe_metadata_sidecar_json(
    config: &PipeCliRunConfig,
    preflight: &PipeFramePreflight,
    identities: &PipeFinalizedContextIdentities,
    layout: &PipeOutputLayout,
    audio_summary: Option<PipeAudioSidecarSummary>,
    video: PipeVideoRunSummary,
) -> Result<Value> {
    if video.bytes_per_frame != preflight.bytes_per_frame
        || video.expected_total_video_bytes != preflight.expected_total_video_bytes
    {
        bail!("PIPE sidecar video counts contradict metadata preflight");
    }
    let sample_aspect_ratio = PipeAspectRatio::square_pixels();
    let display_aspect_ratio = PipeAspectRatio::display_for_frame(
        preflight.dimensions.width,
        preflight.dimensions.height,
        sample_aspect_ratio,
    )?;
    let audio = match audio_summary {
        Some(summary) => PipeAudioContractV3::pcm_s16le(
            summary.metadata.sample_rate_hz,
            summary.metadata.channels,
            summary.metadata.sample_frames,
            summary.metadata.byte_len,
        )?,
        None => PipeAudioContractV3::absent(),
    };
    let contract = PipeSidecarV3::new(
        preflight.dimensions.width,
        preflight.dimensions.height,
        u64::from(preflight.clip_info.frame_count),
        preflight.cadence,
        sample_aspect_ratio,
        display_aspect_ratio,
        config.correction_mode().label().to_string(),
        identities.source_payload_geometry_identity.clone(),
        identities.strict_color_context_identity.clone(),
        identities.correction_context_identity.clone(),
        audio,
    )?;
    let mut value = contract.to_value();
    let root = value
        .as_object_mut()
        .context("PIPE sidecar builder did not return an object")?;
    let source_file_basename = config
        .input_path
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| config.input_path.to_string_lossy().to_string());

    root.insert(
        "source_file_basename".to_string(),
        json!(source_file_basename),
    );
    root.insert(
        "source_file_stem".to_string(),
        json!(path_stem(&config.input_path)?),
    );
    root.insert("output_stem".to_string(), json!(layout.output_stem));
    root.insert(
        "raw_output".to_string(),
        json!({
            "mode": match layout.mode {
                PipeOutputMode::File => "file",
                PipeOutputMode::Stdout => "stdout",
            },
            "path": layout.raw_output_path.as_ref().map(|path| path_to_string(path)),
            "stdout": matches!(layout.mode, PipeOutputMode::Stdout),
        }),
    );
    root.insert(
        "byte_clean_stdout".to_string(),
        json!(matches!(layout.mode, PipeOutputMode::Stdout)),
    );
    root.insert("diagnostics_stream".to_string(), json!("stderr"));
    root.insert(
        "scene_linear_display_note".to_string(),
        json!("This scene-linear editing derivative uses a fixed 1/2 signal scale. It is intentionally not display-ready and may appear dark until exposure and a display transform are applied in the editor."),
    );
    root.insert("prores_is_lossy".to_string(), json!(true));
    root.insert(
        "dng_is_camera_domain_preservation_output".to_string(),
        json!(true),
    );
    root.insert(
        "metadata_preflight_decoded_bayer_frames".to_string(),
        json!(0),
    );
    root.insert(
        "source_file_bytes".to_string(),
        json!(preflight.source_file_bytes),
    );
    root.insert("created_by".to_string(), json!("mcraw4vulkan"));
    root.insert(
        "mcraw4vulkan_version".to_string(),
        json!(env!("CARGO_PKG_VERSION")),
    );
    root.insert("mcraw4vulkan_commit".to_string(), Value::Null);

    validate_pipe_sidecar_v3(&value).context("PIPE sidecar v3 final validation failed")?;
    Ok(value)
}
fn payload_layout_label(layout: FramePayloadLayout) -> String {
    match layout {
        FramePayloadLayout::CompressedRawcodecType7 => "compressed_rawcodec_type7".to_string(),
        FramePayloadLayout::BinnedRaw16Type6 { row_stride } => {
            format!("binned_raw16_type6:row_stride={row_stride}")
        }
    }
}

fn option_bool_label(value: Option<bool>) -> String {
    value.map_or_else(|| "absent".to_string(), |present| present.to_string())
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn bayer_pattern_label(pattern: BayerPattern) -> &'static str {
    match pattern {
        BayerPattern::Rggb => "rggb",
        BayerPattern::Bggr => "bggr",
        BayerPattern::Grbg => "grbg",
        BayerPattern::Gbrg => "gbrg",
    }
}
fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn path_stem(path: &Path) -> Result<String> {
    let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        bail!("path {:?} does not have a valid UTF-8 file stem", path);
    };
    if stem.is_empty() {
        bail!("path {:?} does not have a non-empty file stem", path);
    }
    Ok(stem.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;

    fn fake_config(output: PipeCliOutput) -> PipeCliRunConfig {
        PipeCliRunConfig {
            input_path: PathBuf::from("clip.mcraw"),
            backend: PipeCliBackend::Gpu,
            vignette: PipeCliVignette::WithCorrection,
            output,
            payload_feeder_options: PayloadFeederOptions::production_default(),
        }
    }

    fn test_abs_path(parts: &[&str]) -> PathBuf {
        let mut path = std::env::temp_dir();
        for part in parts {
            path.push(part);
        }
        path
    }

    fn test_owned_directory(label: &str) -> PathBuf {
        let sequence = PIPE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mcraw4vulkan-pipe-test-{label}-{}-{sequence:016x}",
            std::process::id(),
        ));
        fs::create_dir(&path).expect("create unique test directory");
        path
    }

    #[test]
    fn production_modes_map_only_the_spatial_gain_choice() {
        let mut config = fake_config(PipeCliOutput::Stdout);
        assert_eq!(
            config.correction_mode(),
            PipeF32BayerCorrectionMode::MotionCamSpatial
        );
        config.vignette = PipeCliVignette::NoCorrection;
        assert_eq!(
            config.correction_mode(),
            PipeF32BayerCorrectionMode::IdentitySpatialGain
        );
    }

    #[test]
    fn pipe_example_facts_path_is_index_only_and_independent_of_production_preflight() {
        let source = include_str!("pipe_cli.rs");
        let facts_path = source
            .split("pub fn pipe_example_facts_for_input")
            .nth(1)
            .and_then(|tail| tail.split("pub(crate) fn run_pipe_cli").next())
            .expect("Pipe Example facts source");

        assert!(facts_path.contains("McrawContainer::open_for_display(input_path)"));
        assert!(facts_path.contains("PipeMovCadence::from_source_rate"));
        assert!(facts_path.contains("PipeAspectRatio::display_for_frame"));
        for forbidden in [
            "McrawContainer::open(input_path)",
            "open_for_display_with_audio",
            "preflight_pipe_frames",
            "ClipSourceSha256",
            "read_video_payload",
            "stream_pipe_frames",
            "GpuDecodeBackend",
        ] {
            assert!(
                !facts_path.contains(forbidden),
                "Pipe Example facts path contains forbidden work: {forbidden}"
            );
        }
    }

    #[test]
    fn file_layout_uses_neutral_direct_yuv_names() {
        let current = test_abs_path(&["pipe-layout"]);
        let layout = pipe_file_output_layout_in_current_dir(
            Path::new("source.mcraw"),
            Path::new("clip.yuv444p12le"),
            &current,
        )
        .expect("file layout");
        assert_eq!(layout.output_stem, "clip-BT2020-linear-tv");
        assert_eq!(
            layout.raw_output_path,
            Some(current.join("clip-BT2020-linear-tv.yuv444p12le"))
        );
        assert_eq!(layout.audio_sidecar_path, current.join("clip-audio.wav"));
        assert_eq!(
            layout.metadata_sidecar_path,
            current.join("clip-BT2020-linear-tv.json")
        );
        let names = [
            layout.output_stem,
            path_to_string(layout.raw_output_path.as_ref().unwrap()),
            path_to_string(&layout.audio_sidecar_path),
            path_to_string(&layout.metadata_sidecar_path),
        ]
        .join("\n");
        assert!(!names.contains("sRGB"));
        assert!(!names.contains("gbrp16le"));
    }

    #[test]
    fn stdout_layout_uses_input_basename_and_neutral_sidecars() {
        let current = test_abs_path(&["pipe-stdout"]);
        let layout =
            pipe_stdout_output_layout_in_current_dir(Path::new("input/ocean.mcraw"), &current)
                .expect("stdout layout");
        assert_eq!(layout.output_stem, "ocean-BT2020-linear-tv");
        assert_eq!(layout.raw_output_path, None);
        assert_eq!(layout.audio_sidecar_path, current.join("ocean-audio.wav"));
        assert_eq!(
            layout.metadata_sidecar_path,
            current.join("ocean-BT2020-linear-tv.json")
        );
    }

    #[test]
    fn already_neutral_output_suffix_is_not_duplicated() {
        let current = test_abs_path(&["pipe-neutral"]);
        let layout = pipe_file_output_layout_in_current_dir(
            Path::new("source.mcraw"),
            Path::new("clip-BT2020-linear-tv.yuv444p12le"),
            &current,
        )
        .expect("file layout");
        assert_eq!(layout.output_stem, "clip-BT2020-linear-tv");
        assert_eq!(
            layout.raw_output_path,
            Some(current.join("clip-BT2020-linear-tv.yuv444p12le"))
        );
        assert_eq!(layout.audio_sidecar_path, current.join("clip-audio.wav"));
    }

    #[test]
    fn temp_paths_are_confined_to_their_final_paths() {
        let suffix = format!("pipe-temp-test-{}", std::process::id());
        let parent = std::env::temp_dir();
        let layout = PipeOutputLayout {
            mode: PipeOutputMode::File,
            output_stem: "clip-BT2020-linear-tv".to_string(),
            raw_output_path: Some(parent.join(format!("{suffix}.yuv444p12le"))),
            audio_sidecar_path: parent.join(format!("{suffix}-audio.wav")),
            metadata_sidecar_path: parent.join(format!("{suffix}.json")),
        };
        let temps = create_temp_paths_for_layout(&layout, true).expect("owned temp directory");
        assert_eq!(temps.owned_dir.parent(), Some(parent.as_path()));
        assert_eq!(
            temps.raw_part_path.as_deref().and_then(Path::parent),
            Some(temps.owned_dir.as_path())
        );
        assert_eq!(
            temps.audio_part_path.as_deref().and_then(Path::parent),
            Some(temps.owned_dir.as_path())
        );
        assert_eq!(
            temps.metadata_part_path.parent(),
            Some(temps.owned_dir.as_path())
        );
        cleanup_temp_paths(&temps).expect("owned temp cleanup");
        assert!(!temps.owned_dir.exists());
    }

    #[test]
    fn publication_is_no_replace_and_rollback_removes_only_the_published_link() {
        let directory = test_owned_directory("publication");
        let temp = directory.join("temporary");
        let final_path = directory.join("final");
        fs::write(&temp, b"owned bytes").expect("write temporary file");

        let published = publish_temp_no_replace(&temp, &final_path).expect("publish link");
        assert!(!temp.exists());
        assert_eq!(fs::read(&final_path).expect("read final"), b"owned bytes");

        let replacement_temp = directory.join("replacement-temporary");
        fs::write(&replacement_temp, b"replacement").expect("write replacement temporary");
        assert!(publish_temp_no_replace(&replacement_temp, &final_path).is_err());
        assert_eq!(
            fs::read(&final_path).expect("read preserved final"),
            b"owned bytes"
        );
        assert_eq!(
            fs::read(&replacement_temp).expect("read preserved replacement temporary"),
            b"replacement"
        );

        published.rollback().expect("roll back published link");
        assert!(!final_path.exists());
        fs::remove_dir_all(&directory).expect("remove test directory");
    }

    #[cfg(unix)]
    #[test]
    fn rollback_refuses_to_remove_an_inode_replacement() {
        let directory = test_owned_directory("inode-guard");
        let temp = directory.join("temporary");
        let final_path = directory.join("final");
        fs::write(&temp, b"owned bytes").expect("write temporary file");
        let published = publish_temp_no_replace(&temp, &final_path).expect("publish link");

        fs::remove_file(&final_path).expect("remove published link");
        fs::write(&final_path, b"competitor").expect("write competitor replacement");
        assert!(published.rollback().is_err());
        assert_eq!(
            fs::read(&final_path).expect("read competitor"),
            b"competitor"
        );
        fs::remove_dir_all(&directory).expect("remove test directory");
    }

    #[cfg(unix)]
    #[test]
    fn dangling_symlink_counts_as_an_existing_output() {
        use std::os::unix::fs::symlink;

        let directory = test_owned_directory("dangling-link");
        let path = directory.join("final");
        symlink(directory.join("missing-target"), &path).expect("create dangling symlink");
        assert!(ensure_missing(&path).is_err());
        fs::remove_file(&path).expect("remove dangling symlink");
        fs::remove_dir(&directory).expect("remove test directory");
    }

    #[test]
    fn temporary_cleanup_failure_is_reported() {
        let directory = test_owned_directory("cleanup-error");
        let not_a_directory = directory.join("temporary-file");
        fs::write(&not_a_directory, b"not a directory").expect("write cleanup fixture");
        let temps = PipeTempPaths {
            owned_dir: not_a_directory.clone(),
            raw_part_path: None,
            audio_part_path: None,
            metadata_part_path: not_a_directory.clone(),
        };
        assert!(cleanup_temp_paths(&temps).is_err());
        assert!(not_a_directory.exists());
        fs::remove_file(&not_a_directory).expect("remove cleanup fixture");
        fs::remove_dir(&directory).expect("remove test directory");
    }

    #[test]
    fn output_paths_must_be_distinct() {
        let shared = test_abs_path(&["pipe-test", "shared"]);
        let layout = PipeOutputLayout {
            mode: PipeOutputMode::File,
            output_stem: "clip".to_string(),
            raw_output_path: Some(shared.clone()),
            audio_sidecar_path: shared.clone(),
            metadata_sidecar_path: test_abs_path(&["pipe-test", "metadata.json"]),
        };
        assert!(ensure_distinct_final_paths(&layout, true).is_err());
    }

    #[test]
    fn bytes_per_frame_is_exactly_six_bytes_per_pixel() {
        let width = 3840_u64;
        let height = 2160_u64;
        assert_eq!(width * height * 6, 49_766_400);
        assert_eq!(49_766_400_u64 * 1_941, 96_596_582_400);
    }

    #[test]
    fn pipe_gpu_requirements_cover_the_exact_8k_storage_and_readback_shapes() {
        let dimensions = FrameDimensions {
            width: 8_192,
            height: 4_608,
        };
        let required = pipe_gpu_required_limits(dimensions).expect("8K requirements fit");

        assert_eq!(required.dimensions(), dimensions);
        assert_eq!(required.storage_buffer_resource(), "direct YUV12 output");
        assert_eq!(required.storage_buffer_binding_bytes(), 226_492_416);
        assert_eq!(
            required.buffer_resource(),
            "direct YUV12 output/status readback"
        );
        assert_eq!(required.buffer_bytes(), 226_492_432);
    }

    #[test]
    fn pipe_gpu_requirements_reject_zero_and_overflowing_dimensions() {
        for dimensions in [
            FrameDimensions {
                width: 0,
                height: 4_608,
            },
            FrameDimensions {
                width: 8_192,
                height: 0,
            },
            FrameDimensions {
                width: u32::MAX,
                height: u32::MAX,
            },
        ] {
            assert!(pipe_gpu_required_limits(dimensions).is_err());
        }
    }

    #[test]
    fn audio_pcm_formula_counts_interleaved_channels() {
        assert_eq!(audio_pcm_data_bytes(48_000, 2, 16).unwrap(), 192_000);
        assert!(audio_pcm_data_bytes(1, 2, 12).is_err());
    }

    #[test]
    fn audio_layout_and_payload_work_happen_after_video() {
        let source = include_str!("pipe_cli.rs");
        let run_inner = source
            .split("fn run_pipe_cli_inner")
            .nth(1)
            .and_then(|tail| tail.split("fn print_pipe_completion").next())
            .expect("production PIPE inner function source");
        let render = run_inner.find("stream_pipe_frames").expect("video render");
        let prepare = run_inner
            .find("prepare_pipe_audio")
            .expect("lazy audio metadata preparation");
        let materialize = run_inner
            .find("write_audio_sidecar_part")
            .expect("audio materialization");
        assert!(render < prepare);
        assert!(prepare < materialize);
    }

    #[test]
    fn payload_labels_are_format_specific_without_color_policy_forks() {
        assert_eq!(
            payload_layout_label(FramePayloadLayout::CompressedRawcodecType7),
            "compressed_rawcodec_type7"
        );
        assert_eq!(
            payload_layout_label(FramePayloadLayout::BinnedRaw16Type6 { row_stride: 7680 }),
            "binned_raw16_type6:row_stride=7680"
        );
    }

    #[test]
    fn public_source_has_no_validation_only_hot_path() {
        let source = include_str!("pipe_cli.rs");
        let preflight = source
            .split("fn preflight_pipe_frames")
            .nth(1)
            .and_then(|tail| tail.split("fn resolve_pipe_frame_context").next())
            .expect("production preflight source");
        for forbidden in [
            "ClipSourceSha256::read_once",
            "read_once_until_cancelled",
            "read_video_payload_into",
            "decode_loaded_payload",
            "resolve_pipe_frame_context",
            "motioncam_pipe_f32_bayer_facts",
            "parse_frame_input",
            "resolve_stream_context",
            "enable_validation_clamp",
            "set_validation_linear_signal_scale",
            "sha256_bytes",
            "validation_json",
            "source_saturation",
        ] {
            assert!(
                !preflight.contains(forbidden),
                "production preflight contains forbidden hot-path token {forbidden}"
            );
        }
        let legacy_cpu_preflight = ["preflight_frame_payloads", "cpu"].join("_");
        assert!(!preflight.contains(&legacy_cpu_preflight));
        assert_eq!(preflight.matches(".frame_metadata(").count(), 1);
        assert!(preflight.contains("FrameNumber(0)"));
        assert!(preflight.contains("PayloadReadPlan::from_core_frame_numbers"));
        let renderer = source
            .split("fn stream_pipe_frames")
            .nth(1)
            .and_then(|tail| tail.split("fn pipe_file_output_layout").next())
            .expect("production renderer source");
        assert!(renderer.contains("DirectYuv12FrameFeeder::NativePayload"));
        assert!(renderer.contains("OneSharedComputeTwoReadbackDirectYuv12Scheduler::new"));
        assert!(!renderer.contains("enable_validation_clamp"));
        assert!(!renderer.contains("set_validation_linear_signal_scale"));
        assert!(!renderer.contains("GpuRenderPacker"));
        assert!(!renderer.contains("Gbrp16"));
    }

    #[test]
    fn public_pipe_opens_one_lazy_frame_container_before_streaming() {
        let source = include_str!("pipe_cli.rs");
        let run = source
            .split("pub(crate) fn run_pipe_cli")
            .nth(1)
            .and_then(|tail| tail.split("fn run_pipe_cli_inner").next())
            .expect("public PIPE orchestration source");
        assert_eq!(
            run.matches("McrawContainer::open_for_display_with_audio")
                .count(),
            1
        );
        assert!(!run.contains("McrawContainer::open(&config.input_path)"));
        let preflight = run
            .find("preflight_pipe_frames")
            .expect("bounded preflight");
        let hash = run
            .find("PipeSourceHashWorker::spawn")
            .expect("hash worker");
        let stream = run
            .find("run_pipe_cli_inner")
            .expect("stream orchestration");
        assert!(preflight < hash && hash < stream);
    }

    #[test]
    fn frame_context_is_resolved_just_in_time_before_the_selected_decode() {
        let source = include_str!("pipe_cli.rs");
        let renderer = source
            .split("pub(crate) fn stream_pipe_frames")
            .nth(1)
            .and_then(|tail| tail.split("fn public_video_summary").next())
            .expect("production stream source");
        let metadata = renderer
            .find(".frame_metadata(number)")
            .expect("frame metadata");
        let resolve = renderer
            .find("resolve_pipe_frame_context")
            .expect("frame context resolve");
        let submit = renderer
            .find("scheduler.submit_frame")
            .expect("selected decode/submit");
        let collect = renderer
            .find("contexts.collect")
            .expect("context collection");
        assert!(metadata < resolve && resolve < submit && submit < collect);
    }

    #[test]
    fn source_hash_is_joined_only_after_video_and_before_success_sidecar() {
        let source = include_str!("pipe_cli.rs");
        let run_inner = source
            .split("fn run_pipe_cli_inner")
            .nth(1)
            .and_then(|tail| tail.split("fn print_pipe_completion").next())
            .expect("production PIPE inner function source");
        let stream = run_inner.find("stream_pipe_frames").expect("video stream");
        let audio = run_inner.find("prepare_pipe_audio").expect("lazy audio");
        let hash = run_inner
            .find("source_hash_worker.finish")
            .expect("source hash join");
        let sidecar = run_inner
            .find("pipe_metadata_sidecar_json")
            .expect("sidecar construction");
        assert!(stream < audio && audio < hash && hash < sidecar);
    }

    #[test]
    fn source_hash_worker_reports_exact_digest_and_missing_source_failure() {
        let directory = test_owned_directory("source-hash-worker");
        let source = directory.join("source.mcraw");
        fs::write(&source, b"bounded source identity").expect("write hash fixture");
        let expected = ClipSourceSha256::read_once(&source).expect("synchronous test digest");
        let completion = PipeSourceHashWorker::spawn(&source)
            .expect("spawn source hash")
            .finish()
            .expect("finish source hash");
        assert_eq!(completion.source_sha256, expected);
        assert_eq!(completion.bytes_read, 23);

        let missing = directory.join("missing.mcraw");
        assert!(
            PipeSourceHashWorker::spawn(&missing)
                .expect("spawn missing-source worker")
                .finish()
                .is_err()
        );
        fs::remove_file(source).expect("remove hash fixture");
        fs::remove_dir(directory).expect("remove hash test directory");
    }

    #[test]
    fn gpu_path_does_not_construct_a_cpu_decoder() {
        let source = include_str!("pipe_cli.rs");
        assert!(
            source.contains("(selected_backend == PipeCliBackend::Cpu).then(CpuFrameDecoder::new)")
        );
        let forbidden = ["let mut cpu_", "decoder = CpuFrameDecoder::new();"].concat();
        assert!(!source.contains(&forbidden));
    }

    #[test]
    fn sidecar_writer_is_byte_clean_and_newline_terminated() {
        let mut bytes = Vec::new();
        write_pipe_metadata_json(&mut bytes, &json!({"metadata_version": 3})).expect("write JSON");
        assert_eq!(bytes.last(), Some(&b'\n'));
        let parsed: Value = serde_json::from_slice(&bytes).expect("valid JSON");
        assert_eq!(parsed["metadata_version"], 3);
    }

    #[test]
    fn public_algorithm_has_no_display_compensation() {
        let source = include_str!("pipe_cli.rs");
        let renderer = source
            .split("fn stream_pipe_frames")
            .nth(1)
            .and_then(|tail| tail.split("fn pipe_file_output_layout").next())
            .expect("production renderer source");
        for forbidden in [
            "P999",
            "Srgb",
            "srgb",
            "tone",
            "exposure",
            "display_transform",
            "RgbSink",
        ] {
            assert!(
                !renderer.contains(forbidden),
                "public direct-YUV renderer contains {forbidden}"
            );
        }
    }

    #[test]
    fn final_publication_orders_raw_file_last() {
        let source = include_str!("pipe_cli.rs");
        let publication = source
            .split("let publish_result")
            .nth(1)
            .and_then(|tail| tail.split("eprintln!(").next())
            .expect("publication source");
        let metadata = publication
            .find("metadata_part_path")
            .expect("metadata publication");
        let raw = publication.find("raw_part_path").expect("raw publication");
        assert!(metadata < raw, "raw output must be the last success marker");
    }

    #[test]
    fn no_vignette_semantics_remain_non_spatial_only() {
        let source = include_str!("pipe_cli.rs");
        assert!(source.contains("PipeF32BayerCorrectionMode::IdentitySpatialGain"));
        assert!(source.contains("motioncam_pipe_f32_bayer_facts("));
        assert!(source.contains("PipeF32BayerCorrectionFingerprint::from_fixed_facts"));
        let correction_metadata = include_str!("../../mcraw4vulkan-vignette/src/metadata.rs");
        assert!(correction_metadata.contains("VignetteCorrectionMode::Enabled"));
        assert!(correction_metadata.contains("IdentitySpatialGain => None"));
        assert!(correction_metadata.contains("from_input_facts_with_fixed_map"));
    }
}
