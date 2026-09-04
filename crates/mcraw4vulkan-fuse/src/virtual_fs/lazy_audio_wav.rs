use std::collections::VecDeque;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use mcraw4vulkan_audio::RawAudioRangeDecoder;
use mcraw4vulkan_audiowriter::{
    Bw64ChnaMetadata, Bw64IxmlMetadata, Bw64IxmlSpeed, Bw64LayoutInfo, Bw64PcmS16LeDescription,
    bw64_layout_info_for_sample_frames_with_description,
    bw64_pcm_s16le_header_bytes_for_sample_frames_with_description,
};
use mcraw4vulkan_core::AudioTrackInfo;
use mcraw4vulkan_mcrawcontainer::McrawContainer;

use super::audio_wav::AudioWavMetadata;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LazyAudioWavConfig {
    pub page_size: usize,
    pub cache_pages: usize,
}

impl Default for LazyAudioWavConfig {
    fn default() -> Self {
        Self {
            page_size: 64 * 1024,
            cache_pages: 16,
        }
    }
}

impl LazyAudioWavConfig {
    fn normalized(self) -> Self {
        Self {
            page_size: self.page_size.max(1),
            cache_pages: self.cache_pages.max(1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LazyAudioWavSummary {
    pub metadata: AudioWavMetadata,
    pub layout: Bw64LayoutInfo,
    pub header_byte_len: u64,
    pub data_byte_len: u64,
    pub page_size: usize,
    pub cache_pages: usize,
}

// The header is built once, while PCM data is decoded into a bounded page cache.
// All read offsets are byte offsets in the final BW64 file, so a range may cross
// the header/data boundary without materializing the complete recording.
pub struct LazyAudioWav {
    decoder: RawAudioRangeDecoder,
    header_bytes: Vec<u8>,
    metadata: AudioWavMetadata,
    layout: Bw64LayoutInfo,
    cache: AudioPageCache,
}

impl LazyAudioWav {
    pub fn open(path: &Path, config: LazyAudioWavConfig) -> Result<Self> {
        let container = McrawContainer::open(path)
            .with_context(|| format!("failed to open container: {}", path.display()))?;
        Self::from_container(container, config)
    }

    pub fn from_container(container: McrawContainer, config: LazyAudioWavConfig) -> Result<Self> {
        let track = container
            .audio_info()
            .context("clip does not contain decoded audio")?;
        ensure!(
            track.bits_per_sample == 16,
            "lazy BW64 audio currently supports only 16-bit PCM, found {} bits",
            track.bits_per_sample
        );

        let description = motioncam_bw64_description(&container, track);
        let layout =
            bw64_layout_info_for_sample_frames_with_description(&description, track.total_samples)
                .context("failed to compute lazy BW64 layout")?;
        let header_bytes = bw64_pcm_s16le_header_bytes_for_sample_frames_with_description(
            &description,
            track.total_samples,
        )
        .context("failed to build lazy BW64 header bytes")?;
        let header_byte_len =
            u64::try_from(header_bytes.len()).context("BW64 header length overflows u64")?;
        ensure!(
            header_byte_len + layout.data_size_64 == layout.file_size,
            "lazy BW64 header/data size does not match layout"
        );

        let byte_len = layout.file_size;
        let decoder =
            RawAudioRangeDecoder::from_container(container).context("failed to open raw audio")?;
        let config = config.normalized();

        Ok(Self {
            decoder,
            header_bytes,
            metadata: AudioWavMetadata {
                sample_rate_hz: track.sample_rate_hz,
                channels: track.channels,
                bits_per_sample: track.bits_per_sample,
                sample_frames: layout.sample_frames,
                byte_len,
            },
            layout,
            cache: AudioPageCache::new(config.page_size, config.cache_pages),
        })
    }

    pub fn byte_len(&self) -> u64 {
        self.metadata.byte_len
    }

    pub fn metadata(&self) -> AudioWavMetadata {
        self.metadata
    }

    pub fn summary(&self) -> LazyAudioWavSummary {
        LazyAudioWavSummary {
            metadata: self.metadata,
            layout: self.layout,
            header_byte_len: self.header_byte_len(),
            data_byte_len: self.layout.data_size_64,
            page_size: self.cache.page_size,
            cache_pages: self.cache.max_pages,
        }
    }

    pub fn read_at(&mut self, offset: u64, output: &mut [u8]) -> Result<usize> {
        if output.is_empty() || offset >= self.byte_len() {
            return Ok(0);
        }

        let readable = u64::try_from(output.len())
            .context("lazy BW64 output buffer length overflows u64")?
            .min(self.byte_len() - offset);
        let mut written = 0usize;

        while u64::try_from(written).unwrap_or(u64::MAX) < readable {
            let current_offset = offset + u64::try_from(written).unwrap_or(u64::MAX);
            let remaining = usize::try_from(readable - u64::try_from(written).unwrap_or(0))
                .context("lazy BW64 read length overflows usize")?;

            if current_offset < self.header_byte_len() {
                let copied = self.copy_header_at(current_offset, &mut output[written..], remaining);
                written += copied;
            } else {
                let data_offset = current_offset - self.header_byte_len();
                let copied = self.read_data_at(data_offset, &mut output[written..])?;
                if copied == 0 {
                    break;
                }
                written += copied;
            }
        }

        Ok(written)
    }

    fn header_byte_len(&self) -> u64 {
        self.header_bytes.len() as u64
    }

    fn copy_header_at(&self, offset: u64, output: &mut [u8], max_len: usize) -> usize {
        let start = usize::try_from(offset).unwrap_or(usize::MAX);
        if start >= self.header_bytes.len() || output.is_empty() || max_len == 0 {
            return 0;
        }

        let len = max_len
            .min(output.len())
            .min(self.header_bytes.len().saturating_sub(start));
        output[..len].copy_from_slice(&self.header_bytes[start..start + len]);
        len
    }

    fn read_data_at(&mut self, offset: u64, output: &mut [u8]) -> Result<usize> {
        if output.is_empty() || offset >= self.layout.data_size_64 {
            return Ok(0);
        }

        let readable = u64::try_from(output.len())
            .context("lazy BW64 data output buffer length overflows u64")?
            .min(self.layout.data_size_64 - offset);
        let readable = usize::try_from(readable).context("lazy BW64 data read overflows usize")?;
        let mut written = 0usize;

        while written < readable {
            let current_offset =
                offset + u64::try_from(written).context("lazy BW64 data offset overflow")?;
            let page_index = current_offset / self.cache.page_size_u64();
            let page_offset = usize::try_from(current_offset % self.cache.page_size_u64())
                .context("lazy BW64 page offset overflows usize")?;
            let page_position = self.ensure_cached_data_page(page_index)?;
            let page = &self.cache.pages[page_position];

            if page_offset >= page.bytes.len() {
                break;
            }

            let copy_len = readable
                .saturating_sub(written)
                .min(page.bytes.len() - page_offset);
            output[written..written + copy_len]
                .copy_from_slice(&page.bytes[page_offset..page_offset + copy_len]);
            written += copy_len;
        }

        Ok(written)
    }

    fn ensure_cached_data_page(&mut self, page_index: u64) -> Result<usize> {
        if let Some(position) = self.cache.position(page_index) {
            return Ok(position);
        }

        let page_start = page_index
            .checked_mul(self.cache.page_size_u64())
            .context("lazy BW64 page start overflow")?;
        let page_len = self
            .cache
            .page_size_u64()
            .min(self.layout.data_size_64.saturating_sub(page_start));
        let mut bytes =
            vec![0u8; usize::try_from(page_len).context("lazy BW64 page length overflow")?];
        let returned = self
            .decoder
            .read_pcm_s16le_at(page_start, &mut bytes)
            .context("failed to read lazy raw audio page")?;
        bytes.truncate(returned);

        self.cache.insert(AudioPage { page_index, bytes });
        Ok(self.cache.pages.len() - 1)
    }
}

struct AudioPageCache {
    page_size: usize,
    max_pages: usize,
    pages: VecDeque<AudioPage>,
}

impl AudioPageCache {
    fn new(page_size: usize, max_pages: usize) -> Self {
        Self {
            page_size,
            max_pages,
            pages: VecDeque::new(),
        }
    }

    fn page_size_u64(&self) -> u64 {
        self.page_size as u64
    }

    fn position(&self, page_index: u64) -> Option<usize> {
        self.pages
            .iter()
            .position(|page| page.page_index == page_index)
    }

    fn insert(&mut self, page: AudioPage) {
        if self.pages.len() >= self.max_pages {
            self.pages.pop_front();
        }
        self.pages.push_back(page);
    }
}

struct AudioPage {
    page_index: u64,
    bytes: Vec<u8>,
}

fn motioncam_bw64_description(
    container: &McrawContainer,
    track: AudioTrackInfo,
) -> Bw64PcmS16LeDescription {
    Bw64PcmS16LeDescription {
        track,
        ixml: Some(motioncam_ixml_metadata(container)),
        chna: motioncam_chna_metadata(track),
    }
}

fn motioncam_ixml_metadata(container: &McrawContainer) -> Bw64IxmlMetadata {
    let speed = source_fps_string(container).map(|fps| Bw64IxmlSpeed {
        current_speed: fps.clone(),
        master_speed: fps.clone(),
        timecode_rate: fps,
        timecode_flag: "NDF".to_string(),
    });

    Bw64IxmlMetadata {
        project: Some("MotionCam".to_string()),
        note: Some("Recorded with MotionCam".to_string()),
        circled: Some("FALSE".to_string()),
        blackmagic_keywords: Some(String::new()),
        scene: Some("1".to_string()),
        blackmagic_shot: Some("1".to_string()),
        take: Some("1".to_string()),
        blackmagic_angle: Some("ms".to_string()),
        tape: Some("1".to_string()),
        speed,
    }
}

fn motioncam_chna_metadata(track: AudioTrackInfo) -> Option<Bw64ChnaMetadata> {
    if track.sample_rate_hz == 48_000 && track.channels == 2 && track.bits_per_sample == 16 {
        Some(Bw64ChnaMetadata::empty_reserved_audio_ids(1024))
    } else {
        None
    }
}

fn source_fps_string(container: &McrawContainer) -> Option<String> {
    let frame_rate = container.clip_info().timing.playback_frame_rate.reduced();
    if frame_rate.numerator == 0 || frame_rate.denominator == 0 {
        return None;
    }

    Some(format!(
        "{}/{}",
        frame_rate.numerator, frame_rate.denominator
    ))
}
