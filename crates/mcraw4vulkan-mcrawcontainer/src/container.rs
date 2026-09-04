use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use mcraw4vulkan_core::frame_index::{AudioChunkEntry, FrameEntry};
use mcraw4vulkan_core::{
    AudioTrackInfo, ClipId, ClipTimingInfo, ContainerFlavor, FrameDimensions, FrameNumber,
    FrameRate, McrawClipInfo,
};

use crate::audio_metadata::AudioDataMetadata;
use crate::container_metadata::ContainerMetadata;
use crate::error::McrawContainerError;
use crate::frame_metadata::FrameMetadata;
use crate::index::ClipIndex;
use crate::parser::{ParsedAudioChunk, ParsedClip, parse_clip, parse_clip_for_display};
use crate::payload::PayloadSpan;
use crate::timing::{AudioChunkTimingInfo, AudioSyncInfo, VideoFrameRateInfo};

// Opened .mcraw container with typed metadata, an index, and random-access
// payload helpers. Decode crates build CPU/GPU work on top of this boundary.
#[derive(Debug)]
pub struct McrawContainer {
    parsed_clip: ParsedClip,
    container_metadata: ContainerMetadata,
    frame_metadata: Vec<OnceLock<FrameMetadata>>,
    index: ClipIndex,
    video_frame_rate_info: VideoFrameRateInfo,
    audio_sync_info: Option<AudioSyncInfo>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct McrawContainerOpenPhaseTiming {
    pub start_offset: Duration,
    pub duration: Duration,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct McrawContainerOpenTimings {
    pub total: Duration,
    pub parse_clip: McrawContainerOpenPhaseTiming,
    pub container_metadata_parse: McrawContainerOpenPhaseTiming,
    pub frame_metadata_json_parse_loop: McrawContainerOpenPhaseTiming,
    pub audio_metadata_index_sync_setup: McrawContainerOpenPhaseTiming,
    pub parsed_all_frame_metadata: bool,
    pub parsed_frame_metadata_count: usize,
    pub parsed_audio_metadata: bool,
}

impl McrawContainer {
    pub fn open(path: &Path) -> Result<Self, McrawContainerError> {
        Self::open_inner(path, McrawContainerOpenMode::Full, None)
    }

    pub fn open_with_timing(
        path: &Path,
        timings: &mut McrawContainerOpenTimings,
    ) -> Result<Self, McrawContainerError> {
        *timings = McrawContainerOpenTimings::default();
        Self::open_inner(path, McrawContainerOpenMode::Full, Some(timings))
    }

    pub fn open_for_display(path: &Path) -> Result<Self, McrawContainerError> {
        Self::open_inner(path, McrawContainerOpenMode::DisplayFast, None)
    }

    pub fn open_for_display_with_timing(
        path: &Path,
        timings: &mut McrawContainerOpenTimings,
    ) -> Result<Self, McrawContainerError> {
        *timings = McrawContainerOpenTimings::default();
        Self::open_inner(path, McrawContainerOpenMode::DisplayFast, Some(timings))
    }

    pub fn open_for_display_with_audio(path: &Path) -> Result<Self, McrawContainerError> {
        Self::open_inner(path, McrawContainerOpenMode::DisplayWithAudio, None)
    }

    pub fn open_for_display_with_audio_with_timing(
        path: &Path,
        timings: &mut McrawContainerOpenTimings,
    ) -> Result<Self, McrawContainerError> {
        *timings = McrawContainerOpenTimings::default();
        Self::open_inner(
            path,
            McrawContainerOpenMode::DisplayWithAudio,
            Some(timings),
        )
    }

    fn open_inner(
        path: &Path,
        mode: McrawContainerOpenMode,
        mut timings: Option<&mut McrawContainerOpenTimings>,
    ) -> Result<Self, McrawContainerError> {
        let total_start = timings.as_ref().map(|_| Instant::now());

        let parse_start = total_start.map(|_| Instant::now());
        let parsed = match mode {
            McrawContainerOpenMode::Full | McrawContainerOpenMode::DisplayWithAudio => {
                parse_clip(path)?
            }
            McrawContainerOpenMode::DisplayFast => parse_clip_for_display(path)?,
        };
        record_open_phase(
            total_start,
            parse_start,
            timings.as_deref_mut(),
            |timings| &mut timings.parse_clip,
        );

        let metadata_start = total_start.map(|_| Instant::now());
        let container_metadata = ContainerMetadata::parse(&parsed.container_metadata_json)?;
        record_open_phase(
            total_start,
            metadata_start,
            timings.as_deref_mut(),
            |timings| &mut timings.container_metadata_parse,
        );

        let frame_count = parsed.frame_count() as u32;
        if frame_count == 0 {
            return Err(McrawContainerError::UnsupportedFormat(
                "clip contains no frames".to_string(),
            ));
        }

        let frame_metadata_start = total_start.map(|_| Instant::now());
        let prepared_frames = match mode {
            McrawContainerOpenMode::Full => prepare_full_frame_index_and_metadata(&parsed)?,
            McrawContainerOpenMode::DisplayFast | McrawContainerOpenMode::DisplayWithAudio => {
                prepare_display_frame_index_and_metadata(&parsed)?
            }
        };
        if let Some(timings) = timings.as_deref_mut() {
            timings.parsed_all_frame_metadata = mode == McrawContainerOpenMode::Full;
            timings.parsed_frame_metadata_count = prepared_frames.parsed_metadata_count;
        }

        record_open_phase(
            total_start,
            frame_metadata_start,
            timings.as_deref_mut(),
            |timings| &mut timings.frame_metadata_json_parse_loop,
        );

        let audio_parse_index_start = total_start.map(|_| Instant::now());
        let (audio_chunks, audio_info, audio_parse_index_duration) = match mode {
            McrawContainerOpenMode::Full | McrawContainerOpenMode::DisplayWithAudio => {
                let audio_metadata = AudioDataMetadata::parse_sources(
                    &parsed.container_metadata_json,
                    parsed.audio_metadata_jsons(),
                )?;
                let (audio_chunks, audio_info) =
                    build_audio_index(parsed.audio_chunks(), audio_metadata)?;
                (
                    audio_chunks,
                    audio_info,
                    audio_parse_index_start.map(|start| start.elapsed()),
                )
            }
            McrawContainerOpenMode::DisplayFast => (Vec::new(), None, None),
        };
        if let Some(timings) = timings.as_deref_mut() {
            timings.parsed_audio_metadata = mode.includes_audio_metadata();
        }

        let video_frame_rate_info =
            derive_video_frame_rate_info(&prepared_frames.video_timestamps_ns);
        let video_reported_frame_rate = video_frame_rate_info
            .median_frame_rate
            .or(video_frame_rate_info.average_frame_rate);

        // Do not use raw total AUDIO_DATA sample count as playback FPS. Real
        // clips can contain audio lead-in/tail outside the exact video frame
        // span. Video FPS comes from frame timestamps, while audio is aligned
        // separately by AUDIO_DATA_METADATA timestamps.
        let timing = ClipTimingInfo::from_sources(
            u64::from(frame_count),
            video_reported_frame_rate,
            None,
            None,
            None,
            video_reported_frame_rate.unwrap_or_else(|| FrameRate::integer(24)),
        );

        let audio_sync_start = total_start.map(|_| Instant::now());
        let audio_sync_info = build_audio_sync_info(
            video_frame_rate_info.first_timestamp_ns,
            audio_chunks.first(),
            audio_info,
        );
        if let (Some(total_start), Some(parse_index_start), Some(parse_index_duration)) = (
            total_start,
            audio_parse_index_start,
            audio_parse_index_duration,
        ) {
            if let Some(timings) = timings.as_deref_mut() {
                timings.audio_metadata_index_sync_setup = McrawContainerOpenPhaseTiming {
                    start_offset: parse_index_start.duration_since(total_start),
                    duration: parse_index_duration
                        + audio_sync_start.map_or(Duration::ZERO, |start| start.elapsed()),
                };
            }
        }

        let display_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown.mcraw")
            .to_string();

        let clip_info = McrawClipInfo {
            clip_id: ClipId(display_name.clone()),
            display_name,
            width: prepared_frames.clip_dimensions.width,
            height: prepared_frames.clip_dimensions.height,
            frame_count,
            frame_rate: timing.playback_frame_rate,
            duration_us: timing.playback_duration_us,
            timing,
            audio: audio_info,
            container_flavor: ContainerFlavor::Current,
        };

        let index = ClipIndex {
            clip_info,
            frames: prepared_frames.frames,
            audio_chunks,
        };

        let result = Ok(Self {
            parsed_clip: parsed,
            container_metadata,
            frame_metadata: prepared_frames.frame_metadata,
            index,
            video_frame_rate_info,
            audio_sync_info,
        });
        if let (Some(total_start), Some(timings)) = (total_start, timings) {
            timings.total = total_start.elapsed();
        }
        result
    }

    pub fn clip_info(&self) -> &McrawClipInfo {
        &self.index.clip_info
    }

    pub fn frame_count(&self) -> usize {
        self.index.frame_count()
    }

    pub fn audio_info(&self) -> Option<AudioTrackInfo> {
        self.index.audio_info()
    }

    pub fn index(&self) -> &ClipIndex {
        &self.index
    }

    pub fn video_frame_rate_info(&self) -> VideoFrameRateInfo {
        self.video_frame_rate_info
    }

    pub fn audio_sync_info(&self) -> Option<AudioSyncInfo> {
        self.audio_sync_info
    }

    pub fn audio_chunk_timing_info(&self) -> AudioChunkTimingInfo {
        build_audio_chunk_timing_info(&self.index.audio_chunks)
    }

    pub fn container_metadata(&self) -> &ContainerMetadata {
        &self.container_metadata
    }

    pub fn container_metadata_json(&self) -> &str {
        &self.parsed_clip.container_metadata_json
    }

    pub fn frame_metadata(
        &self,
        frame_number: FrameNumber,
    ) -> Result<&FrameMetadata, McrawContainerError> {
        let frame_index = frame_number.0 as usize;
        self.frame_entry(frame_number)?;
        self.frame_metadata
            .get(frame_index)
            .ok_or(McrawContainerError::FrameOutOfRange(frame_number.0))?
            .get()
            .map_or_else(|| self.parse_and_cache_frame_metadata(frame_index), Ok)
    }

    pub fn frame_metadata_json(
        &self,
        frame_number: FrameNumber,
    ) -> Result<String, McrawContainerError> {
        let frame_index = frame_number.0 as usize;
        self.frame_entry(frame_number)?;
        self.parsed_clip.read_frame_metadata_json(frame_index)
    }

    pub fn frame_entry(
        &self,
        frame_number: FrameNumber,
    ) -> Result<&FrameEntry, McrawContainerError> {
        self.index
            .frames
            .get(frame_number.0 as usize)
            .ok_or(McrawContainerError::FrameOutOfRange(frame_number.0))
    }

    pub fn video_payload_span(
        &self,
        frame_number: FrameNumber,
    ) -> Result<PayloadSpan, McrawContainerError> {
        let entry = self.frame_entry(frame_number)?;

        Ok(PayloadSpan {
            offset: entry.byte_offset,
            len: u64::from(entry.byte_len),
        })
    }

    pub fn read_video_payload_into(
        &self,
        frame_number: FrameNumber,
        output: &mut Vec<u8>,
    ) -> Result<(), McrawContainerError> {
        self.frame_entry(frame_number)?;
        self.parsed_clip
            .read_frame_payload_into(frame_number.0 as usize, output)
    }

    pub fn read_video_payload(
        &self,
        frame_number: FrameNumber,
    ) -> Result<Vec<u8>, McrawContainerError> {
        let entry = self.frame_entry(frame_number)?;
        let capacity = usize::try_from(entry.byte_len).map_err(|_| {
            McrawContainerError::InvalidPayloadSpan(
                "frame payload length overflows usize".to_string(),
            )
        })?;

        let mut payload = Vec::with_capacity(capacity);
        self.read_video_payload_into(frame_number, &mut payload)?;
        Ok(payload)
    }

    pub fn audio_chunks(&self) -> &[AudioChunkEntry] {
        &self.index.audio_chunks
    }

    pub fn audio_payload_span(
        &self,
        chunk_index: usize,
    ) -> Result<PayloadSpan, McrawContainerError> {
        let chunk = self
            .index
            .audio_chunks
            .get(chunk_index)
            .ok_or(McrawContainerError::AudioChunkOutOfRange(chunk_index))?;

        Ok(PayloadSpan {
            offset: chunk.byte_offset,
            len: u64::from(chunk.byte_len),
        })
    }

    pub fn read_audio_payload_into(
        &self,
        chunk_index: usize,
        output: &mut Vec<u8>,
    ) -> Result<(), McrawContainerError> {
        self.audio_payload_span(chunk_index)?;
        self.parsed_clip
            .read_audio_chunk_payload_into(chunk_index, output)
    }

    pub fn read_audio_payload(&self, chunk_index: usize) -> Result<Vec<u8>, McrawContainerError> {
        let span = self.audio_payload_span(chunk_index)?;
        let capacity = usize::try_from(span.len).map_err(|_| {
            McrawContainerError::InvalidPayloadSpan(
                "audio payload length overflows usize".to_string(),
            )
        })?;

        let mut payload = Vec::with_capacity(capacity);
        self.read_audio_payload_into(chunk_index, &mut payload)?;
        Ok(payload)
    }

    fn parse_frame_metadata_at_index(
        &self,
        frame_index: usize,
    ) -> Result<FrameMetadata, McrawContainerError> {
        let metadata_json = self.parsed_clip.read_frame_metadata_json(frame_index)?;
        FrameMetadata::parse(&metadata_json)
    }

    fn parse_and_cache_frame_metadata(
        &self,
        frame_index: usize,
    ) -> Result<&FrameMetadata, McrawContainerError> {
        let metadata = self.parse_frame_metadata_at_index(frame_index)?;
        let slot = self
            .frame_metadata
            .get(frame_index)
            .ok_or(McrawContainerError::FrameOutOfRange(frame_index as u32))?;
        // Concurrent callers may parse the same frame, but OnceLock retains one
        // value and ties every returned reference to the container-owned cache.
        let _ = slot.set(metadata);
        slot.get().ok_or_else(|| {
            McrawContainerError::UnsupportedFormat(
                "frame metadata cache was not initialized".to_string(),
            )
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum McrawContainerOpenMode {
    Full,
    DisplayFast,
    DisplayWithAudio,
}

impl McrawContainerOpenMode {
    fn includes_audio_metadata(self) -> bool {
        matches!(self, Self::Full | Self::DisplayWithAudio)
    }
}

struct PreparedFrameIndexAndMetadata {
    frames: Vec<FrameEntry>,
    frame_metadata: Vec<OnceLock<FrameMetadata>>,
    video_timestamps_ns: Vec<u64>,
    clip_dimensions: FrameDimensions,
    parsed_metadata_count: usize,
}

fn prepare_full_frame_index_and_metadata(
    parsed: &ParsedClip,
) -> Result<PreparedFrameIndexAndMetadata, McrawContainerError> {
    let frame_count = parsed.frame_count();
    let mut frames = Vec::with_capacity(frame_count);
    let frame_metadata = empty_frame_metadata_cache(frame_count);
    let mut video_timestamps_ns = Vec::with_capacity(frame_count);
    let mut clip_dimensions: Option<FrameDimensions> = None;

    // Full open parses every frame's typed metadata and rejects dimension
    // mismatches before returning, rather than deferring failure to frame access.
    for (index, parsed_frame) in parsed.frames().iter().enumerate() {
        let metadata_json = parsed.read_frame_metadata_json(index)?;
        let metadata = FrameMetadata::parse(&metadata_json)?;

        match clip_dimensions {
            Some(dimensions) if metadata.dimensions != dimensions => {
                return Err(McrawContainerError::UnsupportedFormat(format!(
                    "frame {index} dimensions do not match clip dimensions"
                )));
            }
            Some(_) => {}
            None => {
                clip_dimensions = Some(metadata.dimensions);
            }
        }

        let timestamp_ns = u64::try_from(parsed_frame.timestamp_ns).unwrap_or(0);
        video_timestamps_ns.push(timestamp_ns);
        frames.push(frame_entry_from_parsed(
            index,
            parsed_frame,
            timestamp_ns,
            metadata.dimensions,
        ));
        initialize_frame_metadata(&frame_metadata[index], metadata)?;
    }

    let clip_dimensions = clip_dimensions.ok_or_else(|| {
        McrawContainerError::UnsupportedFormat(
            "clip dimensions could not be determined".to_string(),
        )
    })?;

    Ok(PreparedFrameIndexAndMetadata {
        frames,
        frame_metadata,
        video_timestamps_ns,
        clip_dimensions,
        parsed_metadata_count: frame_count,
    })
}

fn prepare_display_frame_index_and_metadata(
    parsed: &ParsedClip,
) -> Result<PreparedFrameIndexAndMetadata, McrawContainerError> {
    let frame_count = parsed.frame_count();
    let first_metadata_json = parsed.read_frame_metadata_json(0)?;
    let first_metadata = FrameMetadata::parse(&first_metadata_json)?;
    let clip_dimensions = first_metadata.dimensions;
    let mut frames = Vec::with_capacity(frame_count);
    let frame_metadata = empty_frame_metadata_cache(frame_count);
    let mut video_timestamps_ns = Vec::with_capacity(frame_count);

    // Display needs payload offsets, timestamps, frame count, frame rate, and the
    // first frame's typed metadata before first paint. Later typed metadata is
    // parsed and cached on demand by frame_metadata().
    for (index, parsed_frame) in parsed.frames().iter().enumerate() {
        let timestamp_ns = u64::try_from(parsed_frame.timestamp_ns).unwrap_or(0);
        video_timestamps_ns.push(timestamp_ns);
        frames.push(frame_entry_from_parsed(
            index,
            parsed_frame,
            timestamp_ns,
            clip_dimensions,
        ));
    }
    initialize_frame_metadata(&frame_metadata[0], first_metadata)?;

    Ok(PreparedFrameIndexAndMetadata {
        frames,
        frame_metadata,
        video_timestamps_ns,
        clip_dimensions,
        parsed_metadata_count: 1,
    })
}

fn empty_frame_metadata_cache(frame_count: usize) -> Vec<OnceLock<FrameMetadata>> {
    (0..frame_count).map(|_| OnceLock::new()).collect()
}

fn initialize_frame_metadata(
    slot: &OnceLock<FrameMetadata>,
    metadata: FrameMetadata,
) -> Result<(), McrawContainerError> {
    slot.set(metadata).map_err(|_| {
        McrawContainerError::UnsupportedFormat(
            "frame metadata cache was initialized twice".to_string(),
        )
    })
}

fn frame_entry_from_parsed(
    index: usize,
    parsed_frame: &crate::parser::ParsedFrame,
    timestamp_ns: u64,
    dimensions: FrameDimensions,
) -> FrameEntry {
    FrameEntry {
        frame_number: FrameNumber(index as u32),
        byte_offset: parsed_frame.payload_offset,
        byte_len: parsed_frame.payload_len(),
        timestamp_us: timestamp_ns / 1_000,
        dimensions,
        is_key_frame: true,
    }
}

fn record_open_phase(
    total_start: Option<Instant>,
    phase_start: Option<Instant>,
    timings: Option<&mut McrawContainerOpenTimings>,
    field: impl FnOnce(&mut McrawContainerOpenTimings) -> &mut McrawContainerOpenPhaseTiming,
) {
    let (Some(total_start), Some(phase_start), Some(timings)) = (total_start, phase_start, timings)
    else {
        return;
    };
    *field(timings) = McrawContainerOpenPhaseTiming {
        start_offset: phase_start.duration_since(total_start),
        duration: phase_start.elapsed(),
    };
}

fn build_audio_index(
    parsed_chunks: &[ParsedAudioChunk],
    metadata: AudioDataMetadata,
) -> Result<(Vec<AudioChunkEntry>, Option<AudioTrackInfo>), McrawContainerError> {
    if parsed_chunks.is_empty() {
        return Ok((Vec::new(), None));
    }

    if metadata.bits_per_sample != 16 {
        return Err(McrawContainerError::UnsupportedFormat(format!(
            "only 16-bit PCM audio is currently supported, found {} bits",
            metadata.bits_per_sample
        )));
    }

    let bytes_per_sample_frame = usize::from(metadata.channels)
        .checked_mul(usize::from(metadata.bits_per_sample / 8))
        .ok_or(McrawContainerError::InvalidAudioBufferSize)?;

    if bytes_per_sample_frame == 0 {
        return Err(McrawContainerError::InvalidAudioBufferSize);
    }

    let mut entries = Vec::with_capacity(parsed_chunks.len());
    let mut start_sample = 0u64;

    for (index, chunk) in parsed_chunks.iter().enumerate() {
        let payload_len = chunk.payload_len_usize()?;

        if payload_len % bytes_per_sample_frame != 0 {
            return Err(McrawContainerError::UnsupportedFormat(format!(
                "audio chunk {index} byte length is not aligned to sample frame size"
            )));
        }

        let sample_count = payload_len / bytes_per_sample_frame;
        let sample_count_u32 =
            u32::try_from(sample_count).map_err(|_| McrawContainerError::InvalidAudioBufferSize)?;

        entries.push(AudioChunkEntry {
            chunk_index: u32::try_from(index)
                .map_err(|_| McrawContainerError::InvalidAudioBufferSize)?,
            byte_offset: chunk.payload_offset,
            byte_len: chunk.payload_len(),
            start_sample,
            sample_count: sample_count_u32,
            timestamp_ns: chunk.timestamp_ns,
        });

        start_sample = start_sample
            .checked_add(
                u64::try_from(sample_count)
                    .map_err(|_| McrawContainerError::InvalidAudioBufferSize)?,
            )
            .ok_or(McrawContainerError::InvalidAudioBufferSize)?;
    }

    let track = AudioTrackInfo {
        sample_rate_hz: metadata.sample_rate_hz,
        channels: metadata.channels,
        bits_per_sample: metadata.bits_per_sample,
        total_samples: start_sample,
    };

    Ok((entries, Some(track)))
}

fn derive_video_frame_rate_info(timestamps_ns: &[u64]) -> VideoFrameRateInfo {
    let first_timestamp_ns = timestamps_ns.first().copied();
    let last_timestamp_ns = timestamps_ns.last().copied();

    let mut durations_ns = Vec::new();

    for window in timestamps_ns.windows(2) {
        let first = window[0];
        let second = window[1];

        if second > first {
            durations_ns.push(second - first);
        }
    }

    let valid_duration_count = u32::try_from(durations_ns.len()).unwrap_or(u32::MAX);

    let total_span_ns = match (first_timestamp_ns, last_timestamp_ns) {
        (Some(first), Some(last)) if last > first => Some(last - first),
        _ => None,
    };

    let average_frame_rate = total_span_ns.and_then(|span| {
        if span == 0 || durations_ns.is_empty() {
            return None;
        }

        let interval_count = u64::try_from(durations_ns.len()).ok()?;
        Some(FrameRate::new(interval_count.checked_mul(1_000_000_000)?, span).reduced())
    });

    let median_frame_rate = median_frame_rate_from_durations_ns(&mut durations_ns);

    VideoFrameRateInfo {
        first_timestamp_ns,
        last_timestamp_ns,
        total_span_ns,
        valid_duration_count,
        median_frame_rate,
        average_frame_rate,
    }
}

fn median_frame_rate_from_durations_ns(durations_ns: &mut [u64]) -> Option<FrameRate> {
    if durations_ns.is_empty() {
        return None;
    }

    durations_ns.sort_unstable();

    let mid = durations_ns.len() / 2;

    if durations_ns.len() & 1 == 0 {
        let left = durations_ns[mid - 1];
        let right = durations_ns[mid];
        let denominator = left.checked_add(right)?;

        if denominator == 0 {
            return None;
        }

        Some(FrameRate::new(2_000_000_000, denominator).reduced())
    } else {
        let denominator = durations_ns[mid];

        if denominator == 0 {
            return None;
        }

        Some(FrameRate::new(1_000_000_000, denominator).reduced())
    }
}

fn build_audio_sync_info(
    first_video_timestamp_ns: Option<u64>,
    first_audio_chunk: Option<&AudioChunkEntry>,
    track: Option<AudioTrackInfo>,
) -> Option<AudioSyncInfo> {
    let first_video_timestamp_ns = first_video_timestamp_ns?;
    let first_audio_timestamp_ns = first_audio_chunk?.timestamp_ns?;
    let track = track?;

    let drift_ns = i128::from(first_audio_timestamp_ns) - i128::from(first_video_timestamp_ns);
    let abs_drift_ns = drift_ns.unsigned_abs();
    let sync_is_within_one_second = abs_drift_ns <= 1_000_000_000;

    let sample_frames = round_ns_to_sample_frames(abs_drift_ns, track.sample_rate_hz)?;

    // drift_ns = first_audio_timestamp_ns - first_video_timestamp_ns.
    //
    // If drift is negative, audio starts before video, so trim the beginning.
    // If drift is positive, audio starts after video, so insert silence first.
    let (sample_frames_to_trim, sample_frames_to_insert_silence) = if drift_ns < 0 {
        (sample_frames, 0)
    } else if drift_ns > 0 {
        (0, sample_frames)
    } else {
        (0, 0)
    };

    Some(AudioSyncInfo {
        first_video_timestamp_ns,
        first_audio_timestamp_ns,
        drift_ns,
        sample_frames_to_trim,
        sample_frames_to_insert_silence,
        sync_is_within_one_second,
    })
}

fn build_audio_chunk_timing_info(audio_chunks: &[AudioChunkEntry]) -> AudioChunkTimingInfo {
    let chunk_count = audio_chunks.len();
    let first_start_sample = audio_chunks.first().map(|chunk| chunk.start_sample);
    let last_end_sample = audio_chunks.last().and_then(|chunk| {
        chunk
            .start_sample
            .checked_add(u64::from(chunk.sample_count))
    });

    let mut timestamped_chunk_count = 0usize;
    let mut first_timestamp_ns = None;
    let mut last_timestamp_ns = None;
    let mut previous_timestamp_ns = None;
    let mut min_timestamp_delta_ns: Option<i128> = None;
    let mut max_timestamp_delta_ns: Option<i128> = None;

    for chunk in audio_chunks {
        let Some(timestamp_ns) = chunk.timestamp_ns else {
            continue;
        };

        timestamped_chunk_count += 1;
        first_timestamp_ns.get_or_insert(timestamp_ns);

        if let Some(previous) = previous_timestamp_ns {
            let delta = i128::from(timestamp_ns) - i128::from(previous);

            min_timestamp_delta_ns = Some(match min_timestamp_delta_ns {
                Some(current) => current.min(delta),
                None => delta,
            });
            max_timestamp_delta_ns = Some(match max_timestamp_delta_ns {
                Some(current) => current.max(delta),
                None => delta,
            });
        }

        previous_timestamp_ns = Some(timestamp_ns);
        last_timestamp_ns = Some(timestamp_ns);
    }

    let timestamp_span_ns = match (first_timestamp_ns, last_timestamp_ns) {
        (Some(first), Some(last)) => Some(i128::from(last) - i128::from(first)),
        _ => None,
    };

    AudioChunkTimingInfo {
        chunk_count,
        timestamped_chunk_count,
        first_timestamp_ns,
        last_timestamp_ns,
        timestamp_span_ns,
        min_timestamp_delta_ns,
        max_timestamp_delta_ns,
        first_start_sample,
        last_end_sample,
    }
}

fn round_ns_to_sample_frames(abs_ns: u128, sample_rate_hz: u32) -> Option<u64> {
    let numerator = abs_ns.checked_mul(u128::from(sample_rate_hz))?;
    let rounded = numerator.checked_add(500_000_000)? / 1_000_000_000;

    u64::try_from(rounded).ok()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{build_audio_index, build_audio_sync_info};
    use crate::{AudioDataMetadata, McrawContainer, McrawContainerError, ParsedAudioChunk};
    use mcraw4vulkan_core::{FrameNumber, frame_index::AudioChunkEntry};

    #[test]
    fn build_audio_index_preserves_chunk_offsets_and_sample_ranges() {
        let chunks = [
            ParsedAudioChunk {
                payload_offset: 100,
                payload_len: 8,
                timestamp_ns: Some(1_000),
            },
            ParsedAudioChunk {
                payload_offset: 200,
                payload_len: 12,
                timestamp_ns: Some(2_000),
            },
        ];

        let metadata = AudioDataMetadata {
            sample_rate_hz: 48_000,
            channels: 2,
            bits_per_sample: 16,
            reported_frame_rate: None,
        };

        let (entries, track) =
            build_audio_index(&chunks, metadata).expect("valid PCM chunks index");

        assert_eq!(track.expect("track").total_samples, 5);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].byte_offset, 100);
        assert_eq!(entries[0].start_sample, 0);
        assert_eq!(entries[0].sample_count, 2);
        assert_eq!(entries[1].byte_offset, 200);
        assert_eq!(entries[1].start_sample, 2);
        assert_eq!(entries[1].sample_count, 3);
    }

    #[test]
    fn build_audio_index_rejects_unaligned_audio_payloads() {
        let chunks = [ParsedAudioChunk {
            payload_offset: 100,
            payload_len: 7,
            timestamp_ns: None,
        }];

        let metadata = AudioDataMetadata {
            sample_rate_hz: 48_000,
            channels: 2,
            bits_per_sample: 16,
            reported_frame_rate: None,
        };

        let error = build_audio_index(&chunks, metadata)
            .expect_err("unaligned chunk is not a valid PCM sample range");

        assert!(matches!(error, McrawContainerError::UnsupportedFormat(_)));
    }

    #[test]
    fn audio_sync_info_reports_trim_when_audio_starts_before_video() {
        let first_audio_chunk = AudioChunkEntry {
            chunk_index: 0,
            byte_offset: 0,
            byte_len: 0,
            start_sample: 0,
            sample_count: 0,
            timestamp_ns: Some(0),
        };
        let track = mcraw4vulkan_core::AudioTrackInfo {
            sample_rate_hz: 48_000,
            channels: 2,
            bits_per_sample: 16,
            total_samples: 48_000,
        };

        let sync =
            build_audio_sync_info(Some(1_000_000_000), Some(&first_audio_chunk), Some(track))
                .expect("sync info");

        assert_eq!(sync.sample_frames_to_trim, 48_000);
        assert_eq!(sync.sample_frames_to_insert_silence, 0);
        assert!(sync.sync_is_within_one_second);
    }

    #[test]
    fn display_open_defers_later_frame_metadata_parse() {
        let path = write_test_clip(
            "display-lazy-metadata",
            &[
                br#"{"width":4,"height":4,"compressionType":7}"#.as_slice(),
                br#"{"height":4,"compressionType":7}"#.as_slice(),
            ],
        );

        let display_container =
            McrawContainer::open_for_display(&path).expect("display open only needs frame zero");
        assert_eq!(display_container.frame_count(), 2);
        assert_eq!(
            display_container
                .frame_metadata(FrameNumber(0))
                .expect("first metadata is cached")
                .dimensions
                .width,
            4
        );
        assert!(
            display_container.frame_metadata(FrameNumber(1)).is_err(),
            "invalid frame-one metadata is parsed only when that frame is requested"
        );

        let full_error = McrawContainer::open(&path).expect_err("full open parses every frame");
        assert!(full_error.to_string().contains("missing width"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn display_with_audio_open_keeps_later_frame_metadata_lazy() {
        let path = write_test_clip(
            "display-audio-lazy-metadata",
            &[
                br#"{"width":4,"height":4,"compressionType":7}"#.as_slice(),
                br#"{"height":4,"compressionType":7}"#.as_slice(),
            ],
        );

        let display_container = McrawContainer::open_for_display_with_audio(&path)
            .expect("display with audio open only needs frame zero metadata");
        assert_eq!(display_container.frame_count(), 2);
        assert!(display_container.audio_info().is_none());
        assert!(
            display_container.frame_metadata(FrameNumber(1)).is_err(),
            "display-with-audio keeps later frame metadata lazy"
        );

        let _ = fs::remove_file(path);
    }

    fn write_test_clip(name: &str, frame_metadata: &[&[u8]]) -> PathBuf {
        let path = test_output_path(name);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MOTION ");
        bytes.push(3);

        push_item(
            &mut bytes,
            3,
            br#"{"sensorArrangment":"rggb","blackLevel":64,"whiteLevel":1023}"#,
        );

        let mut frame_offsets = Vec::new();
        for (index, metadata) in frame_metadata.iter().enumerate() {
            let buffer_offset = push_item(&mut bytes, 2, &[index as u8; 4]);
            push_item(&mut bytes, 3, metadata);
            frame_offsets.push((buffer_offset, index as i64 * 41_666_667));
        }

        let mut index_payload = Vec::with_capacity(frame_offsets.len() * 16);
        for (offset, timestamp) in frame_offsets {
            index_payload.extend_from_slice(&(offset as i64).to_le_bytes());
            index_payload.extend_from_slice(&timestamp.to_le_bytes());
        }
        let index_data_offset = push_item(&mut bytes, 1, &index_payload) + 8;

        let mut trailer_payload = Vec::with_capacity(16);
        trailer_payload.extend_from_slice(&0x8A90_5612_u32.to_le_bytes());
        trailer_payload.extend_from_slice(&(frame_metadata.len() as u32).to_le_bytes());
        trailer_payload.extend_from_slice(&index_data_offset.to_le_bytes());
        push_item(&mut bytes, 0, &trailer_payload);

        let mut file = fs::File::create(&path).expect("create test clip");
        file.write_all(&bytes).expect("write test clip");
        path
    }

    fn push_item(bytes: &mut Vec<u8>, item_type: u32, payload: &[u8]) -> u64 {
        let item_offset = bytes.len() as u64;
        bytes.extend_from_slice(&item_type.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(payload);
        item_offset
    }

    fn test_output_path(name: &str) -> PathBuf {
        let root = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        fs::create_dir_all(&root).expect("create test output root");
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        root.join(format!(
            "mcraw4vulkan-{name}-{}-{unique}.mcraw",
            std::process::id()
        ))
    }
}
