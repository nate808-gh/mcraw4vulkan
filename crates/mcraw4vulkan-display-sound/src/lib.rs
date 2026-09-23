use std::mem;

use mcraw4vulkan_audio::{AudioDecodeError, AudioScratch, read_timeline_aligned_audio};
use mcraw4vulkan_core::AudioTrackInfo;
use mcraw4vulkan_mcrawcontainer::McrawContainer;
use sdl2::audio::{AudioQueue, AudioSpecDesired};
use thiserror::Error;

const SDL_AUDIO_BUFFER_SAMPLES: u16 = 1024;
const TARGET_QUEUE_MILLIS: u64 = 500;
const MAX_QUEUE_CHUNK_SAMPLE_FRAMES: usize = 4096;

#[derive(Debug, Error)]
pub enum DisplaySoundError {
    #[error("no audio track found for --with-sound")]
    NoAudioTrack,

    #[error("audio decode failed: {0}")]
    AudioDecode(#[from] AudioDecodeError),

    #[error("audio sample rate is not supported by SDL2: {0}")]
    UnsupportedSampleRate(u32),

    #[error("audio channel count is not supported by SDL2: {0}")]
    UnsupportedChannelCount(u16),

    #[error("invalid audio sample count")]
    InvalidSampleCount,

    #[error("SDL2 audio init failed: {0}")]
    SdlInit(String),

    #[error("SDL2 audio device open failed: {0}")]
    SdlOpen(String),

    #[error("SDL2 audio queue failed: {0}")]
    SdlQueue(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplaySoundSummary {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub sample_frames: u64,
    pub duration_ms: u64,
    pub leading_trim_sample_frames: u64,
    pub leading_silence_sample_frames: u64,
    pub tail_trim_sample_frames: u64,
    pub tail_silence_sample_frames: u64,
}

pub struct DisplaySoundPlan {
    samples: Vec<i16>,
    summary: DisplaySoundSummary,
    channels: usize,
    bytes_per_sample_frame: usize,
    target_queue_sample_frames: u64,
}

impl DisplaySoundPlan {
    pub fn from_container(container: &McrawContainer) -> Result<Self, DisplaySoundError> {
        let mut scratch = AudioScratch::new();
        let aligned = read_timeline_aligned_audio(container, &mut scratch).map_err(|error| {
            if matches!(error, AudioDecodeError::AudioUnavailable) {
                DisplaySoundError::NoAudioTrack
            } else {
                DisplaySoundError::AudioDecode(error)
            }
        })?;

        let track = aligned.info.track;
        let spec = sdl_audio_spec(track)?;
        let channels = usize::from(spec.channels);
        let bytes_per_sample_frame = bytes_per_sample_frame(channels)?;
        let target_queue_sample_frames =
            sample_frames_for_millis(spec.sample_rate_hz, TARGET_QUEUE_MILLIS);
        let summary = DisplaySoundSummary {
            sample_rate_hz: track.sample_rate_hz,
            channels: track.channels,
            bits_per_sample: track.bits_per_sample,
            sample_frames: aligned.info.sample_count,
            duration_ms: duration_millis(aligned.info.sample_count, track.sample_rate_hz),
            leading_trim_sample_frames: aligned.alignment.leading_trim_sample_frames,
            leading_silence_sample_frames: aligned.alignment.leading_silence_sample_frames,
            tail_trim_sample_frames: aligned.alignment.tail_trim_sample_frames,
            tail_silence_sample_frames: aligned.alignment.tail_silence_sample_frames,
        };

        Ok(Self {
            samples: aligned.samples.to_vec(),
            summary,
            channels,
            bytes_per_sample_frame,
            target_queue_sample_frames,
        })
    }

    pub fn summary(&self) -> DisplaySoundSummary {
        self.summary
    }

    pub fn sample_frame_offset_for_frame(&self, frame_index: usize, frame_count: usize) -> u64 {
        sample_frame_offset_for_frame(frame_index, frame_count, self.summary.sample_frames)
    }
}

pub struct DisplaySoundSession {
    _sdl: sdl2::Sdl,
    _audio_subsystem: sdl2::AudioSubsystem,
    queue: AudioQueue<i16>,
    samples: Vec<i16>,
    cursor: usize,
    channels: usize,
    bytes_per_sample_frame: usize,
    target_queue_sample_frames: u64,
    started: bool,
    stopped: bool,
    summary: DisplaySoundSummary,
}

impl DisplaySoundSession {
    pub fn from_container(container: &McrawContainer) -> Result<Self, DisplaySoundError> {
        Self::from_plan(DisplaySoundPlan::from_container(container)?)
    }

    pub fn from_plan(plan: DisplaySoundPlan) -> Result<Self, DisplaySoundError> {
        let summary = plan.summary;
        let track = AudioTrackInfo {
            sample_rate_hz: summary.sample_rate_hz,
            channels: summary.channels,
            bits_per_sample: summary.bits_per_sample,
            total_samples: summary.sample_frames,
        };
        let spec = sdl_audio_spec(track)?;

        let sdl = sdl2::init().map_err(DisplaySoundError::SdlInit)?;
        let audio_subsystem = sdl.audio().map_err(DisplaySoundError::SdlInit)?;
        let queue = audio_subsystem
            .open_queue::<i16, _>(None, &spec.desired)
            .map_err(DisplaySoundError::SdlOpen)?;

        eprintln!(
            "display audio enabled: sample_rate_hz={} channels={} bits_per_sample={} aligned_sample_frames={} duration_ms={} leading_trim_sample_frames={} leading_silence_sample_frames={} tail_trim_sample_frames={} tail_silence_sample_frames={}",
            summary.sample_rate_hz,
            summary.channels,
            summary.bits_per_sample,
            summary.sample_frames,
            summary.duration_ms,
            summary.leading_trim_sample_frames,
            summary.leading_silence_sample_frames,
            summary.tail_trim_sample_frames,
            summary.tail_silence_sample_frames,
        );
        eprintln!(
            "display audio device: sample_rate_hz={} channels={} samples={} format={:?}",
            queue.spec().freq,
            queue.spec().channels,
            queue.spec().samples,
            queue.spec().format,
        );

        Ok(Self {
            _sdl: sdl,
            _audio_subsystem: audio_subsystem,
            queue,
            samples: plan.samples,
            cursor: 0,
            channels: plan.channels,
            bytes_per_sample_frame: plan.bytes_per_sample_frame,
            target_queue_sample_frames: plan.target_queue_sample_frames,
            started: false,
            stopped: false,
            summary,
        })
    }

    pub fn summary(&self) -> DisplaySoundSummary {
        self.summary
    }

    pub fn is_started(&self) -> bool {
        self.started && !self.stopped
    }

    pub fn sample_frame_offset_for_frame(&self, frame_index: usize, frame_count: usize) -> u64 {
        sample_frame_offset_for_frame(frame_index, frame_count, self.summary.sample_frames)
    }

    pub fn start_after_first_video_submit(&mut self) -> Result<(), DisplaySoundError> {
        // Audio remains paused until video establishes the playback origin. A
        // later pause or seek clears SDL's queue so pre-seek samples cannot play.
        if self.started || self.stopped {
            return Ok(());
        }
        self.fill_queue()?;
        self.queue.resume();
        self.started = true;
        eprintln!(
            "display audio started after first video submit: queued_sample_frames={}",
            queued_sample_frames(self.queue.size(), self.bytes_per_sample_frame)
        );
        Ok(())
    }

    pub fn start_from_sample_frame_after_video_submit(
        &mut self,
        sample_frame: u64,
    ) -> Result<(), DisplaySoundError> {
        self.seek_to_sample_frame_paused(sample_frame)?;
        self.start_after_first_video_submit()
    }

    pub fn pause(&mut self) {
        self.queue.pause();
        self.queue.clear();
        self.started = false;
    }

    pub fn seek_to_sample_frame_paused(
        &mut self,
        sample_frame: u64,
    ) -> Result<(), DisplaySoundError> {
        let sample_frame = sample_frame.min(self.summary.sample_frames);
        let cursor = usize::try_from(sample_frame)
            .map_err(|_| DisplaySoundError::InvalidSampleCount)?
            .checked_mul(self.channels)
            .ok_or(DisplaySoundError::InvalidSampleCount)?
            .min(self.samples.len());
        self.queue.pause();
        self.queue.clear();
        self.cursor = cursor;
        self.started = false;
        self.stopped = false;
        Ok(())
    }

    pub fn pump(&mut self) -> Result<(), DisplaySoundError> {
        if self.started && !self.stopped {
            self.fill_queue()?;
        }
        Ok(())
    }

    pub fn stop(&mut self) {
        if self.stopped {
            return;
        }
        self.queue.pause();
        self.queue.clear();
        self.stopped = true;
    }

    fn fill_queue(&mut self) -> Result<(), DisplaySoundError> {
        // SDL reports occupancy in bytes, while scheduling uses sample frames.
        // Enqueue only complete interleaved channel groups so refills stay aligned.
        while self.cursor < self.samples.len()
            && queued_sample_frames(self.queue.size(), self.bytes_per_sample_frame)
                < self.target_queue_sample_frames
        {
            let queued = queued_sample_frames(self.queue.size(), self.bytes_per_sample_frame);
            let target_frames = self.target_queue_sample_frames.saturating_sub(queued);
            let max_interleaved = usize::try_from(target_frames)
                .map_err(|_| DisplaySoundError::InvalidSampleCount)?
                .checked_mul(self.channels)
                .ok_or(DisplaySoundError::InvalidSampleCount)?;
            let chunk_interleaved = MAX_QUEUE_CHUNK_SAMPLE_FRAMES
                .checked_mul(self.channels)
                .ok_or(DisplaySoundError::InvalidSampleCount)?
                .min(max_interleaved)
                .max(self.channels);
            let end = self
                .cursor
                .saturating_add(chunk_interleaved)
                .min(self.samples.len());
            let frame_aligned_end =
                self.cursor + ((end - self.cursor) / self.channels) * self.channels;
            if frame_aligned_end == self.cursor {
                break;
            }
            self.queue
                .queue_audio(&self.samples[self.cursor..frame_aligned_end])
                .map_err(DisplaySoundError::SdlQueue)?;
            self.cursor = frame_aligned_end;
        }
        Ok(())
    }
}

impl Drop for DisplaySoundSession {
    fn drop(&mut self) {
        self.stop();
    }
}

struct SdlAudioSpec {
    sample_rate_hz: u32,
    channels: u8,
    desired: AudioSpecDesired,
}

fn sdl_audio_spec(track: AudioTrackInfo) -> Result<SdlAudioSpec, DisplaySoundError> {
    let sample_rate_hz = track.sample_rate_hz;
    let sample_rate_i32 = i32::try_from(sample_rate_hz)
        .map_err(|_| DisplaySoundError::UnsupportedSampleRate(sample_rate_hz))?;
    let channels = u8::try_from(track.channels)
        .map_err(|_| DisplaySoundError::UnsupportedChannelCount(track.channels))?;
    if channels == 0 {
        return Err(DisplaySoundError::UnsupportedChannelCount(track.channels));
    }
    if track.bits_per_sample != 16 {
        return Err(DisplaySoundError::AudioDecode(
            AudioDecodeError::UnsupportedBitsPerSample(track.bits_per_sample),
        ));
    }

    Ok(SdlAudioSpec {
        sample_rate_hz,
        channels,
        desired: AudioSpecDesired {
            freq: Some(sample_rate_i32),
            channels: Some(channels),
            samples: Some(SDL_AUDIO_BUFFER_SAMPLES),
        },
    })
}

fn bytes_per_sample_frame(channels: usize) -> Result<usize, DisplaySoundError> {
    if channels == 0 {
        return Err(DisplaySoundError::UnsupportedChannelCount(0));
    }
    channels
        .checked_mul(mem::size_of::<i16>())
        .ok_or(DisplaySoundError::InvalidSampleCount)
}

fn queued_sample_frames(queued_bytes: u32, bytes_per_sample_frame: usize) -> u64 {
    if bytes_per_sample_frame == 0 {
        return 0;
    }
    u64::from(queued_bytes) / u64::try_from(bytes_per_sample_frame).unwrap_or(u64::MAX)
}

fn sample_frames_for_millis(sample_rate_hz: u32, millis: u64) -> u64 {
    u64::from(sample_rate_hz).saturating_mul(millis) / 1_000
}

fn duration_millis(sample_frames: u64, sample_rate_hz: u32) -> u64 {
    if sample_rate_hz == 0 {
        return 0;
    }
    sample_frames.saturating_mul(1_000) / u64::from(sample_rate_hz)
}

pub fn sample_frame_offset_for_frame(
    frame_index: usize,
    frame_count: usize,
    sample_frames: u64,
) -> u64 {
    if frame_count == 0 || sample_frames == 0 {
        return 0;
    }
    let clamped_frame = frame_index.min(frame_count.saturating_sub(1)) as u128;
    let frame_count = frame_count as u128;
    let sample_frames = u128::from(sample_frames);
    let offset = clamped_frame.saturating_mul(sample_frames) / frame_count;
    u64::try_from(offset).unwrap_or(u64::MAX)
}
