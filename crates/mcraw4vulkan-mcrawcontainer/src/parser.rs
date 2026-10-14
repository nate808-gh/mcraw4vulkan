use std::cmp::Ordering;
use std::fs::File;
use std::path::Path;

use crate::error::McrawContainerError as DecodeError;

const INDEX_MAGIC_NUMBER: u32 = 0x8A90_5612;
const CONTAINER_VERSION: u8 = 3;
const CONTAINER_ID: [u8; 7] = *b"MOTION ";
const ITEM_HEADER_SIZE: u64 = 8;
const BUFFER_INDEX_SIZE: usize = 16;

// MotionCam container item types.
//
// The .mcraw file is a sequence of typed items. The frame index near the end of
// the file points to BUFFER items, and each BUFFER is followed by frame metadata.
// Audio is stored in separate AUDIO_* items that are indexed here and decoded by
// the audio crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ContainerItemType {
    BufferIndex = 0,
    BufferIndexData = 1,
    Buffer = 2,
    Metadata = 3,
    AudioIndex = 4,
    AudioData = 5,
    AudioDataMetadata = 6,
    AudioDataF32 = 7,
}

impl ContainerItemType {
    fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::BufferIndex),
            1 => Some(Self::BufferIndexData),
            2 => Some(Self::Buffer),
            3 => Some(Self::Metadata),
            4 => Some(Self::AudioIndex),
            5 => Some(Self::AudioData),
            6 => Some(Self::AudioDataMetadata),
            7 => Some(Self::AudioDataF32),
            _ => None,
        }
    }
}

// Fixed 8-byte .mcraw file header.
#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub ident: [u8; 7],
    pub version: u8,
}

// Decoded 8-byte item header plus its checked payload endpoint.
#[derive(Debug, Clone, Copy)]
pub struct Item {
    // Container item IDs are independent of frame compressionType. Unknown IDs
    // remain available to context-specific consumers and diagnostics.
    pub item_type: u32,
    pub size: u32,
    pub payload_end: u64,
}

// Entry from the buffer offset table.
//
// The offset points to a BUFFER item. The timestamp is used to sort frames into
// playback order. MotionCam timestamps are nanosecond-scale values.
#[derive(Debug, Clone, Copy)]
pub struct BufferOffset {
    pub offset: i64,
    pub timestamp: i64,
}

// Trailer index structure found near the end of the .mcraw file.
#[derive(Debug, Clone, Copy)]
pub struct BufferIndex {
    pub num_offsets: u32,
    pub index_data_offset: u64,
}

// Cached file locations for one indexed frame.
//
// This intentionally stores only compact offsets, lengths, and timestamp data.
// The compressed payload bytes and frame metadata JSON are read from disk only
// when needed, avoiding duplicated per-frame JSON storage for large clips.
#[derive(Debug, Clone, Copy)]
pub struct ParsedFrame {
    pub payload_offset: u64,
    pub payload_len: u32,
    pub metadata_offset: u64,
    pub metadata_len: u32,
    pub timestamp_ns: i64,
}

impl ParsedFrame {
    pub fn payload_len(&self) -> u32 {
        self.payload_len
    }

    pub fn payload_len_usize(&self) -> Result<usize, DecodeError> {
        usize::try_from(self.payload_len).map_err(|_| {
            DecodeError::UnsupportedFormat("frame payload length overflows usize".to_string())
        })
    }

    pub fn metadata_len_usize(&self) -> Result<usize, DecodeError> {
        usize::try_from(self.metadata_len).map_err(|_| {
            DecodeError::UnsupportedFormat("frame metadata length overflows usize".to_string())
        })
    }
}

// Cached file location for one AUDIO_DATA payload.
#[derive(Debug, Clone, Copy)]
pub struct ParsedAudioChunk {
    pub payload_offset: u64,
    pub payload_len: u32,

    // Parsed from the following AUDIO_DATA_METADATA item when present.
    //
    // The original C++ struct stores this as int64 timestampNs.
    pub timestamp_ns: Option<i64>,
}

impl ParsedAudioChunk {
    pub fn payload_len(&self) -> u32 {
        self.payload_len
    }

    pub fn payload_len_usize(&self) -> Result<usize, DecodeError> {
        usize::try_from(self.payload_len).map_err(|_| {
            DecodeError::UnsupportedFormat("audio payload length overflows usize".to_string())
        })
    }
}

// Cached file location for one AUDIO_INDEX payload.
//
// PCM range reads use AUDIO_DATA records. AUDIO_INDEX counts and targets are
// checked against walked audio boundaries; its timing fields remain unused.
#[derive(Debug, Clone, Copy)]
pub struct ParsedAudioIndex {
    pub payload_offset: u64,
    pub payload_len: u32,
}

// Parsed .mcraw container with random-access file ownership.
//
// This object does not store the whole .mcraw file in RAM. It owns the open file
// and small parsed records so large clips can be decoded one frame or audio range
// at a time.
#[derive(Debug)]
pub struct ParsedClip {
    file: File,
    file_len: u64,
    pub container_metadata_json: String,
    frames: Vec<ParsedFrame>,
    audio_chunks: Vec<ParsedAudioChunk>,
    audio_index_items: Vec<ParsedAudioIndex>,
    audio_metadata_jsons: Vec<String>,
}

impl ParsedClip {
    pub(crate) fn geometry_source_stamp(&self) -> Result<String, DecodeError> {
        let metadata = self
            .file
            .metadata()
            .map_err(|e| DecodeError::Io(e.to_string()))?;
        if metadata.len() != self.file_len {
            return Err(DecodeError::InvalidPayloadSpan(
                "source size changed during preparation".into(),
            ));
        }
        Ok(format!(
            "{}:{:?}",
            metadata.len(),
            metadata
                .modified()
                .map_err(|e| DecodeError::Io(e.to_string()))?
        ))
    }

    /// Fixed header and, only for a mismatch, two bounded count words. No expansion.
    pub(crate) fn read_type7_geometry(
        &self,
        index: usize,
        declared: mcraw4vulkan_core::FrameDimensions,
    ) -> Result<mcraw4vulkan_core::Type7GeometryEvidence, DecodeError> {
        let frame = self.frame_at_index(index)?;
        let read = |relative: u32, length: u32| -> Result<u64, DecodeError> {
            if relative
                .checked_add(length)
                .is_none_or(|end| end > frame.payload_len)
            {
                return Err(DecodeError::InvalidPayloadSpan(
                    "type-7 header/count outside payload".into(),
                ));
            }
            frame
                .payload_offset
                .checked_add(u64::from(relative))
                .ok_or_else(|| DecodeError::InvalidPayloadSpan("type-7 offset overflow".into()))
        };
        let bytes = read_array_at::<16>(&self.file, read(0, 16)?)?;
        let header = mcraw4vulkan_core::read_metadata_header(&bytes)
            .ok_or_else(|| DecodeError::InvalidMetadata("missing type-7 header".into()))?;
        let counts =
            if declared.width > header.encoded_width || declared.height > header.encoded_height {
                Some([
                    u32::from_le_bytes(read_array_at::<4>(
                        &self.file,
                        read(header.bits_offset, 4)?,
                    )?),
                    u32::from_le_bytes(read_array_at::<4>(
                        &self.file,
                        read(header.refs_offset, 4)?,
                    )?),
                ])
            } else {
                None
            };
        Ok(mcraw4vulkan_core::Type7GeometryEvidence {
            header,
            payload_len: frame.payload_len,
            counts,
        })
    }

    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn frames(&self) -> &[ParsedFrame] {
        &self.frames
    }

    pub fn audio_chunks(&self) -> &[ParsedAudioChunk] {
        &self.audio_chunks
    }

    pub fn audio_index_items(&self) -> &[ParsedAudioIndex] {
        &self.audio_index_items
    }

    pub fn audio_metadata_jsons(&self) -> &[String] {
        &self.audio_metadata_jsons
    }

    pub fn first_frame(&self) -> Result<&ParsedFrame, DecodeError> {
        self.frame_at_index(0)
    }

    pub fn frame_at_index(&self, index: usize) -> Result<&ParsedFrame, DecodeError> {
        self.frames
            .get(index)
            .ok_or_else(|| DecodeError::UnsupportedFormat("frame index out of range".to_string()))
    }

    // Read one frame's metadata JSON on demand.
    //
    // Higher-level frame readers cache the typed FrameMetadata; ParsedClip keeps
    // compact offsets rather than duplicating each frame's JSON text.
    pub fn read_frame_metadata_json(&self, index: usize) -> Result<String, DecodeError> {
        let frame = self.frame_at_index(index)?;
        let metadata_len = frame.metadata_len_usize()?;

        let metadata_bytes = read_vec_at(&self.file, frame.metadata_offset, metadata_len)?;
        String::from_utf8(metadata_bytes).map_err(|_| {
            DecodeError::UnsupportedFormat("frame metadata is not valid UTF-8".to_string())
        })
    }

    // Read one compressed frame payload into a reusable caller-owned buffer.
    //
    // This is the key production-size behavior: decoding frame N reads only that
    // frame's compressed payload, not the full .mcraw file.
    pub fn read_frame_payload_into(
        &self,
        index: usize,
        output: &mut Vec<u8>,
    ) -> Result<(), DecodeError> {
        let frame = self.frame_at_index(index)?;
        let payload_len = frame.payload_len_usize()?;

        read_vec_at_into(&self.file, frame.payload_offset, payload_len, output)
    }

    // Read one audio payload into a reusable caller-owned buffer.
    //
    // Audio chunks are raw PCM payloads in the currently supported path, so the
    // audio reader can convert little-endian sample bytes directly into i16 samples.
    pub fn read_audio_chunk_payload_into(
        &self,
        index: usize,
        output: &mut Vec<u8>,
    ) -> Result<(), DecodeError> {
        let chunk = self.audio_chunks.get(index).ok_or_else(|| {
            DecodeError::UnsupportedFormat("audio chunk index out of range".to_string())
        })?;
        let payload_len = chunk.payload_len_usize()?;

        read_vec_at_into(&self.file, chunk.payload_offset, payload_len, output)
    }
}

// Parse the .mcraw container without reading the entire file into memory.
//
// The parser reads the file header, the first container metadata item, the frame
// offset table, compact per-frame file locations, and compact audio payload
// locations. Frame payload bytes and audio payload bytes remain on disk until
// requested by caller-owned readers.
pub fn parse_clip(path: &Path) -> Result<ParsedClip, DecodeError> {
    parse_clip_inner(path, true)
}

pub(crate) fn parse_clip_for_display(path: &Path) -> Result<ParsedClip, DecodeError> {
    parse_clip_inner(path, false)
}

fn parse_clip_inner(path: &Path, include_audio_records: bool) -> Result<ParsedClip, DecodeError> {
    let file = File::open(path).map_err(|err| DecodeError::Io(err.to_string()))?;
    let file_len = file
        .metadata()
        .map_err(|err| DecodeError::Io(err.to_string()))?
        .len();

    ensure_range(file_len, 0, 8)?;
    let header = read_header(&file)?;
    if header.ident != CONTAINER_ID {
        return Err(DecodeError::UnsupportedFormat(
            "invalid container id".into(),
        ));
    }
    if header.version != CONTAINER_VERSION {
        return Err(DecodeError::UnsupportedFormat(format!(
            "invalid container version: {}",
            header.version
        )));
    }

    let buffer_index = read_buffer_index_from_trailer(&file, file_len)?;
    let mut frame_offsets = read_buffer_offsets(&file, file_len, buffer_index)?;
    let item_stream_end = item_stream_end_before_buffer_index_data(&file, file_len, buffer_index)?;
    let (metadata_item, metadata_cursor) = read_item_at(&file, file_len, 8, 8..item_stream_end)?;
    require_item_type(
        metadata_item,
        ContainerItemType::Metadata,
        8,
        "camera metadata",
    )?;
    let metadata_bytes = read_vec_at(&file, metadata_cursor, metadata_item.size as usize)?;
    let container_metadata_json = String::from_utf8(metadata_bytes).map_err(|_| {
        DecodeError::UnsupportedFormat("container metadata is not valid UTF-8".into())
    })?;
    let item_stream_start = metadata_item.payload_end;

    // Validate indexed groups in physical order. Timestamp order is only for
    // playback and cannot protect the item walk from overlapping/fabricated spans.
    frame_offsets.sort_by_key(|entry| entry.offset);
    let mut frames = read_frame_records(
        &file,
        file_len,
        item_stream_start..item_stream_end,
        &frame_offsets,
    )?;
    let parsed_audio = if include_audio_records {
        read_audio_records(&file, file_len, item_stream_start, item_stream_end, &frames)?
    } else {
        ParsedAudioRecords::default()
    };
    // Diagnostics share the application's stderr channel, never PIPE/JSON stdout.
    let _ = parsed_audio
        .unknown
        .write_summary(&mut std::io::stderr().lock());
    frames.sort_by(|a, b| match a.timestamp_ns.cmp(&b.timestamp_ns) {
        Ordering::Equal => a.payload_offset.cmp(&b.payload_offset),
        other => other,
    });

    Ok(ParsedClip {
        file,
        file_len,
        container_metadata_json,
        frames,
        audio_chunks: parsed_audio.chunks,
        audio_index_items: parsed_audio.index_items,
        audio_metadata_jsons: parsed_audio.metadata_jsons,
    })
}

fn read_header(file: &File) -> Result<Header, DecodeError> {
    let raw = read_array_at::<8>(file, 0)?;

    let mut ident = [0u8; 7];
    ident.copy_from_slice(&raw[..7]);

    Ok(Header {
        ident,
        version: raw[7],
    })
}

// Read only the common envelope. Interpretation belongs to the caller; an
// unfamiliar ID is not itself a malformed record in an optional scan position.
fn read_item_at(
    file: &File,
    file_len: u64,
    offset: u64,
    region: std::ops::Range<u64>,
) -> Result<(Item, u64), DecodeError> {
    let header_end = checked_item_end(file_len, &region, offset, ITEM_HEADER_SIZE, None)?;
    let raw = read_array_at::<8>(file, offset)?;
    let item_type = u32::from_le_bytes(raw[..4].try_into().unwrap());
    let size = u32::from_le_bytes(raw[4..].try_into().unwrap());
    let payload_end = checked_item_end(
        file_len,
        &region,
        offset,
        ITEM_HEADER_SIZE + u64::from(size),
        Some(item_type),
    )?;
    Ok((
        Item {
            item_type,
            size,
            payload_end,
        },
        header_end,
    ))
}

fn checked_item_end(
    file_len: u64,
    region: &std::ops::Range<u64>,
    offset: u64,
    length: u64,
    item_type: Option<u32>,
) -> Result<u64, DecodeError> {
    let error = |reason| {
        DecodeError::InvalidPayloadSpan(format!(
            "item type {item_type:?} at offset {offset}: {reason}; region {}..{}, file length {file_len}",
            region.start, region.end,
        ))
    };
    let end = offset
        .checked_add(length)
        .ok_or_else(|| error("offset overflow"))?;
    if region.start > region.end
        || region.end > file_len
        || offset < region.start
        || end > file_len
        || end > region.end
    {
        return Err(error(if item_type.is_none() {
            "truncated or out-of-range item header"
        } else {
            "payload exceeds permitted region or file"
        }));
    }
    Ok(end)
}

fn require_item_type(
    item: Item,
    expected: ContainerItemType,
    offset: u64,
    context: &str,
) -> Result<(), DecodeError> {
    if item.item_type != expected as u32 {
        return Err(DecodeError::UnsupportedFormat(format!(
            "{context} at offset {offset}: expected {expected:?}, found item type {}",
            item.item_type,
        )));
    }
    Ok(())
}

fn read_buffer_index_from_trailer(file: &File, file_len: u64) -> Result<BufferIndex, DecodeError> {
    let trailer_size = ITEM_HEADER_SIZE + BUFFER_INDEX_SIZE as u64;

    if file_len < trailer_size {
        return Err(DecodeError::UnsupportedFormat(
            "file is too small to contain trailer index".to_string(),
        ));
    }

    let trailer_offset = file_len - trailer_size;
    let (item, cursor) = read_item_at(file, file_len, trailer_offset, trailer_offset..file_len)?;

    require_item_type(
        item,
        ContainerItemType::BufferIndex,
        trailer_offset,
        "video index trailer",
    )?;

    if item.size as usize != BUFFER_INDEX_SIZE {
        return Err(DecodeError::UnsupportedFormat(
            "BUFFER_INDEX item has unexpected size".to_string(),
        ));
    }

    ensure_range(file_len, cursor, BUFFER_INDEX_SIZE)?;
    let raw = read_array_at::<BUFFER_INDEX_SIZE>(file, cursor)?;

    let magic_number = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let num_offsets = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let index_data_offset = u64::from_le_bytes([
        raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15],
    ]);

    if magic_number != INDEX_MAGIC_NUMBER {
        return Err(DecodeError::UnsupportedFormat(
            "corrupted or invalid buffer index".to_string(),
        ));
    }

    Ok(BufferIndex {
        num_offsets,
        index_data_offset,
    })
}

fn read_buffer_offsets(
    file: &File,
    file_len: u64,
    index: BufferIndex,
) -> Result<Vec<BufferOffset>, DecodeError> {
    let count = usize::try_from(index.num_offsets)
        .map_err(|_| DecodeError::UnsupportedFormat("invalid offset count".to_string()))?;

    let total_len = count
        .checked_mul(16)
        .ok_or_else(|| DecodeError::UnsupportedFormat("offset table size overflow".to_string()))?;

    let trailer_offset = buffer_index_trailer_offset(file_len)?;
    ensure_range(trailer_offset, index.index_data_offset, total_len)
        .map_err(|e| DecodeError::InvalidIndex(format!("video offset table: {e}")))?;
    if index.index_data_offset < 8 {
        return Err(DecodeError::InvalidIndex(
            "video offset table precedes item stream".into(),
        ));
    }
    let raw = read_vec_at(file, index.index_data_offset, total_len)?;

    let mut offsets = Vec::with_capacity(count);

    for chunk in raw.chunks_exact(16) {
        let offset = i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);

        let timestamp = i64::from_le_bytes([
            chunk[8], chunk[9], chunk[10], chunk[11], chunk[12], chunk[13], chunk[14], chunk[15],
        ]);

        offsets.push(BufferOffset { offset, timestamp });
    }

    Ok(offsets)
}

fn read_frame_records(
    file: &File,
    file_len: u64,
    region: std::ops::Range<u64>,
    frame_offsets: &[BufferOffset],
) -> Result<Vec<ParsedFrame>, DecodeError> {
    let mut frames = Vec::with_capacity(frame_offsets.len());
    let mut previous_end = region.start;
    for frame_offset in frame_offsets {
        let item_offset = u64::try_from(frame_offset.offset)
            .map_err(|_| DecodeError::InvalidIndex("negative frame buffer offset".into()))?;
        if item_offset < previous_end {
            return Err(DecodeError::InvalidIndex(format!(
                "indexed BUFFER at {item_offset} overlaps preceding required media ending at {previous_end}"
            )));
        }
        let (buffer, payload_offset) = read_item_at(file, file_len, item_offset, region.clone())?;
        require_item_type(
            buffer,
            ContainerItemType::Buffer,
            item_offset,
            "indexed video buffer",
        )?;
        let (metadata, metadata_offset) =
            read_item_at(file, file_len, buffer.payload_end, region.clone())?;
        require_item_type(
            metadata,
            ContainerItemType::Metadata,
            buffer.payload_end,
            "BUFFER/METADATA association",
        )?;
        previous_end = metadata.payload_end;
        frames.push(ParsedFrame {
            payload_offset,
            payload_len: buffer.size,
            metadata_offset,
            metadata_len: metadata.size,
            timestamp_ns: frame_offset.timestamp,
        });
    }
    Ok(frames)
}

// Fixed storage and a single diagnostic per completed ancillary walk. IDs 8..13
// are known gyro/OIS/accelerometer records, intentionally unused and quiet.
#[derive(Debug, Default)]
struct UnknownRecordSummary {
    types: [Option<(u32, u64)>; 8],
    total: u64,
    unlisted: u64,
}

impl UnknownRecordSummary {
    fn record(&mut self, id: u32) {
        self.total += 1;
        for slot in &mut self.types {
            match slot {
                Some((known, count)) if *known == id => {
                    *count += 1;
                    return;
                }
                None => {
                    *slot = Some((id, 1));
                    return;
                }
                _ => {}
            }
        }
        self.unlisted += 1;
    }

    fn write_summary(&self, output: &mut impl std::io::Write) -> std::io::Result<()> {
        if self.total != 0 {
            write!(
                output,
                "Unrecognized ancillary container records were not interpreted: {} records; types/counts",
                self.total
            )?;
            for (id, count) in self.types.iter().flatten() {
                write!(output, " {id}:{count}")?;
            }
            writeln!(output, "; other records={}", self.unlisted)?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct ParsedAudioRecords {
    chunks: Vec<ParsedAudioChunk>,
    index_items: Vec<ParsedAudioIndex>,
    metadata_jsons: Vec<String>,
    unknown: UnknownRecordSummary,
}

fn read_audio_records(
    file: &File,
    file_len: u64,
    start: u64,
    end: u64,
    physical_frames: &[ParsedFrame],
) -> Result<ParsedAudioRecords, DecodeError> {
    if end < start || end > file_len {
        return Err(DecodeError::InvalidPayloadSpan(
            "invalid ordinary item region".into(),
        ));
    }
    let mut cursor = start;
    let mut frames = physical_frames.iter().peekable();
    let mut records = ParsedAudioRecords::default();
    while cursor < end {
        // Reuse both validated headers of each indexed group. All other spans
        // must end before the next group, not merely somewhere inside the file.
        if let Some(frame) = frames.peek() {
            let frame_start = frame.payload_offset - ITEM_HEADER_SIZE;
            if cursor == frame_start {
                cursor = frame.metadata_offset + u64::from(frame.metadata_len);
                frames.next();
                continue;
            }
            if cursor > frame_start {
                return Err(DecodeError::InvalidIndex(format!(
                    "walk passed indexed BUFFER at {frame_start}"
                )));
            }
        }
        let (item, payload_offset) = read_item_at(file, file_len, cursor, start..end)?;
        if let Some(frame) = frames.peek() {
            let frame_start = frame.payload_offset - ITEM_HEADER_SIZE;
            if item.payload_end > frame_start {
                return Err(DecodeError::InvalidIndex(format!(
                    "item type {} at {cursor} crosses indexed BUFFER boundary {frame_start}",
                    item.item_type,
                )));
            }
        }
        match ContainerItemType::from_u32(item.item_type) {
            Some(ContainerItemType::AudioIndex) => records.index_items.push(ParsedAudioIndex {
                payload_offset,
                payload_len: item.size,
            }),
            Some(ContainerItemType::AudioData) => records.chunks.push(ParsedAudioChunk {
                payload_offset,
                payload_len: item.size,
                timestamp_ns: None,
            }),
            Some(ContainerItemType::AudioDataF32) => {
                return Err(DecodeError::UnsupportedFormat(format!(
                    "unsupported floating-point audio (item type 7) at offset {cursor}; only 16-bit PCM audio is supported"
                )));
            }
            Some(ContainerItemType::AudioDataMetadata) => {
                let bytes = read_vec_at(file, payload_offset, item.size as usize)?;
                if let Some(timestamp_ns) = parse_audio_timestamp_ns(&bytes) {
                    if let Some(chunk) = records
                        .chunks
                        .iter_mut()
                        .rev()
                        .find(|chunk| chunk.timestamp_ns.is_none())
                    {
                        chunk.timestamp_ns = Some(timestamp_ns);
                    }
                }
                // Preserve the existing optional JSON/binary audio-metadata
                // policy. AudioDataMetadata's consumer validates audio facts.
                if let Ok(text) = std::str::from_utf8(&bytes) {
                    let trimmed = text.trim_start();
                    if trimmed.starts_with('{') || trimmed.starts_with('[') {
                        records.metadata_jsons.push(text.to_string());
                    }
                }
            }
            Some(
                ContainerItemType::Buffer
                | ContainerItemType::BufferIndex
                | ContainerItemType::BufferIndexData
                | ContainerItemType::Metadata,
            ) => {
                // The final trailer/index alone selects video frames. Bounded
                // unreferenced media and stale index records are unused here;
                // do not expose them or follow their pointers. The span guard
                // above still prevents hiding an authoritative indexed group.
            }
            None => {
                // No payload reads, allocation, telemetry version/count checks,
                // or unused-index dereferences. Forward acceptance is best effort
                // within the supported outer format, not semantic validation.
                if !(8..=13).contains(&item.item_type) {
                    records.unknown.record(item.item_type);
                }
            }
        }
        cursor = item.payload_end;
    }
    if frames.next().is_some() {
        return Err(DecodeError::InvalidIndex(
            "walk did not reach all indexed video groups".into(),
        ));
    }
    validate_audio_indexes(file, &records)?;
    Ok(records)
}

// Audio is consumed, unlike telemetry. Index targets must be actual walked PCM
// records, not plausible headers found by seeking into a skipped payload.
fn validate_audio_indexes(file: &File, records: &ParsedAudioRecords) -> Result<(), DecodeError> {
    for index in &records.index_items {
        let invalid = |reason: &str| {
            DecodeError::InvalidIndex(format!(
                "AUDIO_INDEX at {}: {reason}",
                index.payload_offset - ITEM_HEADER_SIZE,
            ))
        };
        if index.payload_len < 16 {
            return Err(invalid("truncated index header"));
        }
        let raw = read_array_at::<16>(file, index.payload_offset)?;
        let count = u64::try_from(i64::from_le_bytes(raw[..8].try_into().unwrap()))
            .map_err(|_| invalid("negative offset count"))?;
        let expected = count
            .checked_mul(16)
            .and_then(|n| n.checked_add(16))
            .ok_or_else(|| invalid("offset count overflow"))?;
        if expected != u64::from(index.payload_len) {
            return Err(invalid("inconsistent offset count/length"));
        }
        for i in 0..count {
            // Count-derived length and the entire outer span were checked above.
            let raw = read_array_at::<16>(file, index.payload_offset + 16 + i * 16)?;
            let offset = u64::try_from(i64::from_le_bytes(raw[..8].try_into().unwrap()))
                .map_err(|_| invalid("negative audio offset"))?;
            if records
                .chunks
                .binary_search_by_key(&offset, |chunk| chunk.payload_offset - ITEM_HEADER_SIZE)
                .is_err()
            {
                return Err(invalid(
                    "offset does not target a walked AUDIO_DATA boundary",
                ));
            }
        }
    }
    Ok(())
}

fn parse_audio_timestamp_ns(bytes: &[u8]) -> Option<i64> {
    if bytes.len() != 8 {
        return None;
    }

    Some(i64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn item_stream_end_before_buffer_index_data(
    file: &File,
    file_len: u64,
    index: BufferIndex,
) -> Result<u64, DecodeError> {
    let trailer_offset = buffer_index_trailer_offset(file_len)?;
    let default_end = index.index_data_offset.min(trailer_offset);

    if index.index_data_offset < ITEM_HEADER_SIZE {
        return Ok(default_end);
    }

    let index_payload_len = usize::try_from(index.num_offsets)
        .ok()
        .and_then(|count| count.checked_mul(16));

    let Some(index_payload_len) = index_payload_len else {
        return Ok(default_end);
    };

    let possible_header_offset = index.index_data_offset - ITEM_HEADER_SIZE;
    let Ok((item, payload_offset)) =
        read_item_at(file, file_len, possible_header_offset, 8..trailer_offset)
    else {
        return Ok(default_end);
    };

    if item.item_type == ContainerItemType::BufferIndexData as u32
        && payload_offset == index.index_data_offset
        && item.size as usize == index_payload_len
    {
        return Ok(possible_header_offset.min(trailer_offset));
    }

    Ok(default_end)
}

fn buffer_index_trailer_offset(file_len: u64) -> Result<u64, DecodeError> {
    let trailer_size = ITEM_HEADER_SIZE + BUFFER_INDEX_SIZE as u64;

    file_len.checked_sub(trailer_size).ok_or_else(|| {
        DecodeError::UnsupportedFormat("file is too small to contain trailer index".to_string())
    })
}

fn read_array_at<const N: usize>(file: &File, offset: u64) -> Result<[u8; N], DecodeError> {
    let mut raw = [0u8; N];
    read_exact_at(file, offset, &mut raw)?;
    Ok(raw)
}

fn read_vec_at(file: &File, offset: u64, len: usize) -> Result<Vec<u8>, DecodeError> {
    let mut raw = vec![0u8; len];
    read_exact_at(file, offset, raw.as_mut_slice())?;
    Ok(raw)
}

fn read_vec_at_into(
    file: &File,
    offset: u64,
    len: usize,
    output: &mut Vec<u8>,
) -> Result<(), DecodeError> {
    output.clear();

    if output.capacity() < len {
        output.reserve(len - output.capacity());
    }

    output.resize(len, 0);
    read_exact_at(file, offset, output.as_mut_slice())
}

// Unix and Windows use positional reads so shared access does not mutate a file
// cursor. Other targets preserve the immutable API by seeking a cloned handle.
#[cfg(unix)]
fn read_exact_at(file: &File, offset: u64, output: &mut [u8]) -> Result<(), DecodeError> {
    use std::os::unix::fs::FileExt;

    let mut filled = 0usize;
    while filled < output.len() {
        let read_offset = offset
            .checked_add(filled as u64)
            .ok_or_else(|| DecodeError::UnsupportedFormat("read offset overflow".to_string()))?;
        let read = file
            .read_at(&mut output[filled..], read_offset)
            .map_err(|err| DecodeError::Io(err.to_string()))?;
        if read == 0 {
            return Err(DecodeError::Io(
                "unexpected EOF while reading mcraw file".to_string(),
            ));
        }
        filled += read;
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at(file: &File, offset: u64, output: &mut [u8]) -> Result<(), DecodeError> {
    use std::os::windows::fs::FileExt;

    let mut filled = 0usize;
    while filled < output.len() {
        let read_offset = offset
            .checked_add(filled as u64)
            .ok_or_else(|| DecodeError::UnsupportedFormat("read offset overflow".to_string()))?;
        let read = file
            .seek_read(&mut output[filled..], read_offset)
            .map_err(|err| DecodeError::Io(err.to_string()))?;
        if read == 0 {
            return Err(DecodeError::Io(
                "unexpected EOF while reading mcraw file".to_string(),
            ));
        }
        filled += read;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_exact_at(file: &File, offset: u64, output: &mut [u8]) -> Result<(), DecodeError> {
    use std::io::{Read, Seek, SeekFrom};

    let mut reader = file
        .try_clone()
        .map_err(|err| DecodeError::Io(err.to_string()))?;

    reader
        .seek(SeekFrom::Start(offset))
        .map_err(|err| DecodeError::Io(err.to_string()))?;

    reader
        .read_exact(output)
        .map_err(|err| DecodeError::Io(err.to_string()))
}

// Validate file-derived spans with checked arithmetic before allocation or I/O,
// so malformed offsets cannot wrap into an apparently in-bounds range.
fn ensure_range(file_len: u64, start: u64, len: usize) -> Result<(), DecodeError> {
    let end = start
        .checked_add(len as u64)
        .ok_or_else(|| DecodeError::UnsupportedFormat("range overflow".to_string()))?;

    if end > file_len {
        return Err(DecodeError::UnsupportedFormat(
            "range is outside file bounds".to_string(),
        ));
    }

    Ok(())
}
