//! Raw MotionCam audio range decoding.
//!
//! This crate intentionally stops below WAV/BW64 serialization and FUSE cache
//! policy. It reads caller-requested byte ranges from the raw AUDIO_DATA stream
//! exposed by `mcraw4vulkan-mcrawcontainer`.

use std::path::Path;

use mcraw4vulkan_core::{AudioSampleRange, AudioTrackInfo, ClipTimingInfo, FrameRate};
use mcraw4vulkan_mcrawcontainer::{AudioSyncInfo, McrawContainer, McrawContainerError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioDecodeError {
    #[error("clip does not contain audio")]
    AudioUnavailable,

    #[error("audio writer currently supports only 16-bit PCM, found {0} bits")]
    UnsupportedBitsPerSample(u16),

    #[error("audio channel count must be at least 1")]
    InvalidChannelCount,

    #[error("invalid audio byte range")]
    InvalidByteRange,

    #[error("invalid audio buffer size")]
    InvalidBufferSize,

    #[error("audio chunk {chunk_index} payload is too short for requested range")]
    AudioChunkPayloadTooShort { chunk_index: usize },

    #[error("container error: {0}")]
    Container(#[from] McrawContainerError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioByteRange {
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSampleFrameRange {
    pub start_sample_frame: u64,
    pub sample_frame_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedAudioInfo {
    pub track: AudioTrackInfo,
    pub start_sample: u64,
    pub sample_count: u64,
}

#[derive(Debug)]
pub struct DecodedAudioRef<'a> {
    pub info: DecodedAudioInfo,
    pub samples: &'a [i16],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineAudioAlignmentInfo {
    pub raw_sample_frames: u64,
    pub start_synced_sample_frames: u64,
    pub target_timeline_sample_frames: u64,
    pub output_sample_frames: u64,
    pub leading_trim_sample_frames: u64,
    pub leading_silence_sample_frames: u64,
    pub tail_trim_sample_frames: u64,
    pub tail_silence_sample_frames: u64,
    pub target_duration_us: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct AudioVideoTimelineRequest {
    pub timing: ClipTimingInfo,
    pub video_frame_count: u64,
    pub audio_track: Option<AudioTrackInfo>,
    pub audio_sync_info: Option<AudioSyncInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioVideoTimelinePlan {
    pub playback_frame_rate: FrameRate,
    pub playback_duration_us: u64,
    pub video_frame_count: u64,
    pub audio_track: Option<AudioTrackInfo>,
    pub audio_alignment: Option<TimelineAudioAlignmentInfo>,
}

impl AudioVideoTimelinePlan {
    pub fn audio_is_present(&self) -> bool {
        self.audio_alignment.is_some()
    }
}

#[derive(Debug)]
pub struct TimelineAlignedAudioRef<'a> {
    pub info: DecodedAudioInfo,
    pub alignment: TimelineAudioAlignmentInfo,
    pub samples: &'a [i16],
}

// Owns reusable compressed and interleaved PCM storage. Decoded views borrow
// pcm_samples, so their logical contents last only until this scratch is reused.
#[derive(Debug, Default)]
pub struct AudioScratch {
    compressed: Vec<u8>,
    pcm_samples: Vec<i16>,
}

impl AudioScratch {
    pub fn new() -> Self {
        Self::default()
    }

    fn prepare_compressed(&mut self, byte_len: usize) {
        self.compressed.clear();
        self.compressed.reserve(byte_len);
    }

    fn prepare_pcm_samples(&mut self, sample_count: usize) {
        self.pcm_samples.clear();
        self.pcm_samples.reserve(sample_count);
    }

    fn compressed_and_pcm_samples_mut(&mut self) -> (&[u8], &mut [i16]) {
        (&self.compressed, self.pcm_samples.as_mut_slice())
    }
}

pub struct RawAudioRangeDecoder {
    container: McrawContainer,
    track: AudioTrackInfo,
    bytes_per_sample_frame: u64,
    chunk_scratch: Vec<u8>,
}

impl RawAudioRangeDecoder {
    pub fn open(path: &Path) -> Result<Self, AudioDecodeError> {
        Self::from_container(McrawContainer::open(path)?)
    }

    pub fn from_container(container: McrawContainer) -> Result<Self, AudioDecodeError> {
        let track = container
            .audio_info()
            .ok_or(AudioDecodeError::AudioUnavailable)?;
        let bytes_per_sample_frame = bytes_per_sample_frame(track)?;

        Ok(Self {
            container,
            track,
            bytes_per_sample_frame,
            chunk_scratch: Vec::new(),
        })
    }

    pub fn track(&self) -> AudioTrackInfo {
        self.track
    }

    pub fn bytes_per_sample_frame(&self) -> u64 {
        self.bytes_per_sample_frame
    }

    pub fn pcm_data_byte_len(&self) -> Result<u64, AudioDecodeError> {
        self.track
            .total_samples
            .checked_mul(self.bytes_per_sample_frame)
            .ok_or(AudioDecodeError::InvalidBufferSize)
    }

    pub fn sample_frame_range_for_pcm_byte_range(
        &self,
        range: AudioByteRange,
    ) -> Result<AudioSampleFrameRange, AudioDecodeError> {
        sample_frame_range_for_pcm_byte_range(self.track, range)
    }

    pub fn read_pcm_s16le_at(
        &mut self,
        offset: u64,
        output: &mut [u8],
    ) -> Result<usize, AudioDecodeError> {
        if output.is_empty() {
            return Ok(0);
        }

        let data_len = self.pcm_data_byte_len()?;
        if offset >= data_len {
            return Ok(0);
        }

        let readable = u64::try_from(output.len())
            .map_err(|_| AudioDecodeError::InvalidBufferSize)?
            .min(data_len - offset);
        let readable_usize =
            usize::try_from(readable).map_err(|_| AudioDecodeError::InvalidBufferSize)?;
        output[..readable_usize].fill(0);

        let request_end = offset
            .checked_add(readable)
            .ok_or(AudioDecodeError::InvalidByteRange)?;
        let chunk_count = self.container.audio_chunks().len();

        for chunk_position in 0..chunk_count {
            let chunk = self.container.audio_chunks()[chunk_position].clone();
            let chunk_start = chunk
                .start_sample
                .checked_mul(self.bytes_per_sample_frame)
                .ok_or(AudioDecodeError::InvalidBufferSize)?;
            let chunk_len = u64::from(chunk.sample_count)
                .checked_mul(self.bytes_per_sample_frame)
                .ok_or(AudioDecodeError::InvalidBufferSize)?;
            let chunk_end = chunk_start
                .checked_add(chunk_len)
                .ok_or(AudioDecodeError::InvalidBufferSize)?;

            let overlap_start = offset.max(chunk_start);
            let overlap_end = request_end.min(chunk_end);

            if overlap_start >= overlap_end {
                continue;
            }

            let chunk_index = usize::try_from(chunk.chunk_index)
                .map_err(|_| AudioDecodeError::InvalidBufferSize)?;
            self.container
                .read_audio_payload_into(chunk_index, &mut self.chunk_scratch)?;

            let source_start = usize::try_from(overlap_start - chunk_start)
                .map_err(|_| AudioDecodeError::InvalidBufferSize)?;
            let source_len = usize::try_from(overlap_end - overlap_start)
                .map_err(|_| AudioDecodeError::InvalidBufferSize)?;
            let source_end = source_start
                .checked_add(source_len)
                .ok_or(AudioDecodeError::InvalidBufferSize)?;

            if source_end > self.chunk_scratch.len() {
                return Err(AudioDecodeError::AudioChunkPayloadTooShort { chunk_index });
            }

            let destination_start = usize::try_from(overlap_start - offset)
                .map_err(|_| AudioDecodeError::InvalidBufferSize)?;
            let destination_end = destination_start
                .checked_add(source_len)
                .ok_or(AudioDecodeError::InvalidBufferSize)?;

            output[destination_start..destination_end]
                .copy_from_slice(&self.chunk_scratch[source_start..source_end]);
        }

        Ok(readable_usize)
    }
}

pub fn read_audio_i16<'a>(
    container: &McrawContainer,
    range: AudioSampleRange,
    scratch: &'a mut AudioScratch,
) -> Result<DecodedAudioRef<'a>, AudioDecodeError> {
    let track = container
        .audio_info()
        .ok_or(AudioDecodeError::AudioUnavailable)?;

    read_audio_range_into_scratch(container, range, track, scratch)?;

    Ok(DecodedAudioRef {
        info: DecodedAudioInfo {
            track,
            start_sample: range.start_sample,
            sample_count: range.sample_count,
        },
        samples: scratch.pcm_samples.as_slice(),
    })
}

pub fn read_timeline_aligned_audio<'a>(
    container: &McrawContainer,
    scratch: &'a mut AudioScratch,
) -> Result<TimelineAlignedAudioRef<'a>, AudioDecodeError> {
    let track = container
        .audio_info()
        .ok_or(AudioDecodeError::AudioUnavailable)?;
    let raw_range = AudioSampleRange {
        start_sample: 0,
        sample_count: track.total_samples,
    };

    read_audio_range_into_scratch(container, raw_range, track, scratch)?;

    let alignment = plan_audio_alignment_for_playback(
        track,
        track.total_samples,
        container.audio_sync_info(),
        container.clip_info().timing.playback_duration_us,
    )?;
    apply_audio_alignment_plan(scratch.pcm_samples.as_mut(), track, alignment)?;

    Ok(TimelineAlignedAudioRef {
        info: DecodedAudioInfo {
            track,
            start_sample: 0,
            sample_count: alignment.output_sample_frames,
        },
        alignment,
        samples: scratch.pcm_samples.as_slice(),
    })
}

pub fn plan_audio_video_timeline(
    request: AudioVideoTimelineRequest,
) -> Result<AudioVideoTimelinePlan, AudioDecodeError> {
    let audio_alignment = match request.audio_track {
        Some(track) => Some(plan_audio_alignment_for_playback(
            track,
            track.total_samples,
            request.audio_sync_info,
            request.timing.playback_duration_us,
        )?),
        None => None,
    };

    Ok(AudioVideoTimelinePlan {
        playback_frame_rate: request.timing.playback_frame_rate,
        playback_duration_us: request.timing.playback_duration_us,
        video_frame_count: request.video_frame_count,
        audio_track: request.audio_track,
        audio_alignment,
    })
}

pub fn plan_audio_alignment_for_playback(
    track: AudioTrackInfo,
    source_audio_sample_frames: u64,
    sync_info: Option<AudioSyncInfo>,
    playback_duration_us: u64,
) -> Result<TimelineAudioAlignmentInfo, AudioDecodeError> {
    plan_audio_alignment_for_playback_with_raw_sample_frames(
        track,
        source_audio_sample_frames,
        source_audio_sample_frames,
        sync_info,
        playback_duration_us,
    )
}

pub fn align_audio_samples_to_playback_timeline(
    samples: &mut Vec<i16>,
    track: AudioTrackInfo,
    raw_sample_frames: u64,
    sync_info: Option<AudioSyncInfo>,
    playback_duration_us: u64,
) -> Result<TimelineAudioAlignmentInfo, AudioDecodeError> {
    let channels = usize::from(track.channels);
    if channels == 0 {
        return Err(AudioDecodeError::InvalidChannelCount);
    }

    let source_audio_sample_frames = sample_frames_in_interleaved_samples(samples.len(), channels)?;
    let alignment = plan_audio_alignment_for_playback_with_raw_sample_frames(
        track,
        raw_sample_frames,
        source_audio_sample_frames,
        sync_info,
        playback_duration_us,
    )?;

    apply_audio_alignment_plan(samples, track, alignment)?;

    Ok(alignment)
}

fn plan_audio_alignment_for_playback_with_raw_sample_frames(
    track: AudioTrackInfo,
    raw_sample_frames: u64,
    source_audio_sample_frames: u64,
    sync_info: Option<AudioSyncInfo>,
    playback_duration_us: u64,
) -> Result<TimelineAudioAlignmentInfo, AudioDecodeError> {
    if track.channels == 0 {
        return Err(AudioDecodeError::InvalidChannelCount);
    }

    // The one-second flag gates only leading trim or silence. Tail alignment remains
    // driven independently by the requested playback duration below.
    let (leading_trim_sample_frames, leading_silence_sample_frames) = sync_info
        .filter(|sync| sync.sync_is_within_one_second)
        .map(|sync| {
            (
                sync.sample_frames_to_trim.min(source_audio_sample_frames),
                sync.sample_frames_to_insert_silence,
            )
        })
        .unwrap_or((0, 0));

    let start_synced_sample_frames = source_audio_sample_frames
        .checked_sub(leading_trim_sample_frames)
        .and_then(|frames| frames.checked_add(leading_silence_sample_frames))
        .ok_or(AudioDecodeError::InvalidBufferSize)?;

    let target_timeline_sample_frames =
        timeline_sample_frames_for_duration_us(playback_duration_us, track.sample_rate_hz)
            .ok_or(AudioDecodeError::InvalidBufferSize)?;

    let tail_trim_sample_frames =
        start_synced_sample_frames.saturating_sub(target_timeline_sample_frames);
    let tail_silence_sample_frames =
        target_timeline_sample_frames.saturating_sub(start_synced_sample_frames);

    Ok(TimelineAudioAlignmentInfo {
        raw_sample_frames,
        start_synced_sample_frames,
        target_timeline_sample_frames,
        output_sample_frames: target_timeline_sample_frames,
        leading_trim_sample_frames,
        leading_silence_sample_frames,
        tail_trim_sample_frames,
        tail_silence_sample_frames,
        target_duration_us: playback_duration_us,
    })
}

fn apply_audio_alignment_plan(
    samples: &mut Vec<i16>,
    track: AudioTrackInfo,
    alignment: TimelineAudioAlignmentInfo,
) -> Result<(), AudioDecodeError> {
    let channels = usize::from(track.channels);
    if channels == 0 {
        return Err(AudioDecodeError::InvalidChannelCount);
    }

    if alignment.leading_trim_sample_frames > 0 {
        let trim_interleaved_samples = usize::try_from(alignment.leading_trim_sample_frames)
            .map_err(|_| AudioDecodeError::InvalidBufferSize)?
            .checked_mul(channels)
            .ok_or(AudioDecodeError::InvalidBufferSize)?;

        samples.drain(0..trim_interleaved_samples);
    }

    if alignment.leading_silence_sample_frames > 0 {
        let silence_interleaved_samples = usize::try_from(alignment.leading_silence_sample_frames)
            .map_err(|_| AudioDecodeError::InvalidBufferSize)?
            .checked_mul(channels)
            .ok_or(AudioDecodeError::InvalidBufferSize)?;

        let old_len = samples.len();
        samples.resize(
            old_len
                .checked_add(silence_interleaved_samples)
                .ok_or(AudioDecodeError::InvalidBufferSize)?,
            0,
        );
        samples.copy_within(0..old_len, silence_interleaved_samples);
        samples[..silence_interleaved_samples].fill(0);
    }

    let start_synced_sample_frames = sample_frames_in_interleaved_samples(samples.len(), channels)?;
    if start_synced_sample_frames != alignment.start_synced_sample_frames {
        return Err(AudioDecodeError::InvalidBufferSize);
    }

    if alignment.tail_trim_sample_frames > 0 || alignment.tail_silence_sample_frames > 0 {
        let target_interleaved_samples = usize::try_from(alignment.target_timeline_sample_frames)
            .map_err(|_| AudioDecodeError::InvalidBufferSize)?
            .checked_mul(channels)
            .ok_or(AudioDecodeError::InvalidBufferSize)?;

        if alignment.tail_trim_sample_frames > 0 {
            samples.truncate(target_interleaved_samples);
        } else {
            samples.resize(target_interleaved_samples, 0);
        }
    }

    let output_sample_frames = sample_frames_in_interleaved_samples(samples.len(), channels)?;
    if output_sample_frames != alignment.output_sample_frames {
        return Err(AudioDecodeError::InvalidBufferSize);
    }

    Ok(())
}

pub fn bytes_per_sample_frame(track: AudioTrackInfo) -> Result<u64, AudioDecodeError> {
    if track.channels == 0 {
        return Err(AudioDecodeError::InvalidChannelCount);
    }

    if track.bits_per_sample != 16 {
        return Err(AudioDecodeError::UnsupportedBitsPerSample(
            track.bits_per_sample,
        ));
    }

    u64::from(track.channels)
        .checked_mul(u64::from(track.bits_per_sample / 8))
        .ok_or(AudioDecodeError::InvalidBufferSize)
}

pub fn sample_frame_range_for_pcm_byte_range(
    track: AudioTrackInfo,
    range: AudioByteRange,
) -> Result<AudioSampleFrameRange, AudioDecodeError> {
    // Round byte requests outward to complete interleaved sample frames so both
    // boundary frames are available to callers serving arbitrary byte ranges.
    let bytes_per_frame = bytes_per_sample_frame(track)?;
    let end = range
        .offset
        .checked_add(range.len)
        .ok_or(AudioDecodeError::InvalidByteRange)?;
    let start_sample_frame = range.offset / bytes_per_frame;
    let end_sample_frame = end
        .checked_add(bytes_per_frame - 1)
        .ok_or(AudioDecodeError::InvalidByteRange)?
        / bytes_per_frame;

    Ok(AudioSampleFrameRange {
        start_sample_frame,
        sample_frame_count: end_sample_frame.saturating_sub(start_sample_frame),
    })
}

fn read_audio_range_into_scratch(
    container: &McrawContainer,
    range: AudioSampleRange,
    track: AudioTrackInfo,
    scratch: &mut AudioScratch,
) -> Result<(), AudioDecodeError> {
    let end_sample = range
        .start_sample
        .checked_add(range.sample_count)
        .ok_or(AudioDecodeError::InvalidBufferSize)?;

    if end_sample > track.total_samples {
        return Err(AudioDecodeError::InvalidBufferSize);
    }

    let channels = usize::from(track.channels);
    let requested_sample_frames =
        usize::try_from(range.sample_count).map_err(|_| AudioDecodeError::InvalidBufferSize)?;
    let requested_interleaved_samples = requested_sample_frames
        .checked_mul(channels)
        .ok_or(AudioDecodeError::InvalidBufferSize)?;
    let bytes_per_sample_frame = usize::try_from(bytes_per_sample_frame(track)?)
        .map_err(|_| AudioDecodeError::InvalidBufferSize)?;

    // Ranges count sample frames, while pcm_samples stores channel-interleaved
    // i16 values, so each requested frame reserves one value per channel.
    scratch.prepare_pcm_samples(requested_interleaved_samples);
    scratch.pcm_samples.resize(requested_interleaved_samples, 0);

    let audio_chunks = container.audio_chunks().to_vec();

    for chunk in audio_chunks {
        let chunk_start = chunk.start_sample;
        let chunk_end = chunk_start
            .checked_add(u64::from(chunk.sample_count))
            .ok_or(AudioDecodeError::InvalidBufferSize)?;

        let overlap_start = range.start_sample.max(chunk_start);
        let overlap_end = end_sample.min(chunk_end);

        if overlap_start >= overlap_end {
            continue;
        }

        let chunk_index =
            usize::try_from(chunk.chunk_index).map_err(|_| AudioDecodeError::InvalidBufferSize)?;
        let chunk_byte_len =
            usize::try_from(chunk.byte_len).map_err(|_| AudioDecodeError::InvalidBufferSize)?;

        scratch.prepare_compressed(chunk_byte_len);
        container.read_audio_payload_into(chunk_index, &mut scratch.compressed)?;

        let source_sample_offset = usize::try_from(overlap_start - chunk_start)
            .map_err(|_| AudioDecodeError::InvalidBufferSize)?;
        let destination_sample_offset = usize::try_from(overlap_start - range.start_sample)
            .map_err(|_| AudioDecodeError::InvalidBufferSize)?;
        let overlap_sample_count = usize::try_from(overlap_end - overlap_start)
            .map_err(|_| AudioDecodeError::InvalidBufferSize)?;

        let source_byte_offset = source_sample_offset
            .checked_mul(bytes_per_sample_frame)
            .ok_or(AudioDecodeError::InvalidBufferSize)?;
        let source_byte_count = overlap_sample_count
            .checked_mul(bytes_per_sample_frame)
            .ok_or(AudioDecodeError::InvalidBufferSize)?;
        let source_byte_end = source_byte_offset
            .checked_add(source_byte_count)
            .ok_or(AudioDecodeError::InvalidBufferSize)?;

        if source_byte_end > scratch.compressed.len() {
            return Err(AudioDecodeError::AudioChunkPayloadTooShort { chunk_index });
        }

        let destination_sample_index = destination_sample_offset
            .checked_mul(channels)
            .ok_or(AudioDecodeError::InvalidBufferSize)?;

        let (compressed, samples) = scratch.compressed_and_pcm_samples_mut();

        copy_s16le_interleaved(
            &compressed[source_byte_offset..source_byte_end],
            samples,
            destination_sample_index,
        )?;
    }

    Ok(())
}

fn timeline_sample_frames_for_duration_us(duration_us: u64, sample_rate_hz: u32) -> Option<u64> {
    if sample_rate_hz == 0 {
        return None;
    }

    let numerator = u128::from(duration_us).checked_mul(u128::from(sample_rate_hz))?;
    let rounded = numerator.checked_add(500_000)? / 1_000_000;

    u64::try_from(rounded).ok()
}

fn sample_frames_in_interleaved_samples(
    interleaved_sample_count: usize,
    channels: usize,
) -> Result<u64, AudioDecodeError> {
    if channels == 0 || (interleaved_sample_count / channels) * channels != interleaved_sample_count
    {
        return Err(AudioDecodeError::InvalidBufferSize);
    }

    u64::try_from(interleaved_sample_count / channels)
        .map_err(|_| AudioDecodeError::InvalidBufferSize)
}

fn copy_s16le_interleaved(
    source: &[u8],
    destination: &mut [i16],
    destination_sample_index: usize,
) -> Result<(), AudioDecodeError> {
    if source.len() & 1 != 0 {
        return Err(AudioDecodeError::InvalidBufferSize);
    }

    let source_sample_count = source.len() / 2;
    let destination_end = destination_sample_index
        .checked_add(source_sample_count)
        .ok_or(AudioDecodeError::InvalidBufferSize)?;

    if destination_end > destination.len() {
        return Err(AudioDecodeError::InvalidBufferSize);
    }

    for (sample_offset, bytes) in source.chunks_exact(2).enumerate() {
        destination[destination_sample_index + sample_offset] =
            i16::from_le_bytes([bytes[0], bytes[1]]);
    }

    Ok(())
}
