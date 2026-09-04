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

// Fixed 8-byte item header used throughout the .mcraw container.
#[derive(Debug, Clone, Copy)]
pub struct Item {
    pub item_type: ContainerItemType,
    pub size: u32,
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
// PCM range reads use AUDIO_DATA records. AUDIO_INDEX spans remain available as
// container facts without interpreting their payload in this layer.
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

    let header = read_header(&file)?;
    if header.ident != CONTAINER_ID {
        return Err(DecodeError::UnsupportedFormat(
            "invalid container id".to_string(),
        ));
    }

    if header.version != CONTAINER_VERSION {
        return Err(DecodeError::UnsupportedFormat(format!(
            "invalid container version: {}",
            header.version
        )));
    }

    let (metadata_item, metadata_cursor) = read_item_at(&file, 8)?;
    if metadata_item.item_type != ContainerItemType::Metadata {
        return Err(DecodeError::UnsupportedFormat(
            "first item is not METADATA".to_string(),
        ));
    }

    let metadata_len = usize::try_from(metadata_item.size)
        .map_err(|_| DecodeError::UnsupportedFormat("invalid metadata size".to_string()))?;

    ensure_range(file_len, metadata_cursor, metadata_len)?;
    let metadata_bytes = read_vec_at(&file, metadata_cursor, metadata_len)?;
    let container_metadata_json = String::from_utf8(metadata_bytes).map_err(|_| {
        DecodeError::UnsupportedFormat("container metadata is not valid UTF-8".to_string())
    })?;

    let buffer_index = read_buffer_index_from_trailer(&file, file_len)?;
    let mut frame_offsets = read_buffer_offsets(&file, file_len, buffer_index)?;

    frame_offsets.sort_by(|a, b| match a.timestamp.cmp(&b.timestamp) {
        Ordering::Equal => a.offset.cmp(&b.offset),
        other => other,
    });

    let frames = read_frame_records(&file, file_len, &frame_offsets)?;

    let parsed_audio = if include_audio_records {
        let item_stream_start = metadata_cursor
            .checked_add(u64::from(metadata_item.size))
            .ok_or_else(|| DecodeError::UnsupportedFormat("metadata end overflow".to_string()))?;

        let item_stream_end =
            item_stream_end_before_buffer_index_data(&file, file_len, buffer_index)?;
        read_audio_records(&file, file_len, item_stream_start, item_stream_end)?
    } else {
        ParsedAudioRecords::default()
    };

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

fn read_item_at(file: &File, offset: u64) -> Result<(Item, u64), DecodeError> {
    let raw = read_array_at::<8>(file, offset)?;
    let item_type_raw = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let size = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);

    let item_type = ContainerItemType::from_u32(item_type_raw).ok_or_else(|| {
        DecodeError::UnsupportedFormat(format!("unknown item type: {item_type_raw}"))
    })?;

    let cursor = offset
        .checked_add(ITEM_HEADER_SIZE)
        .ok_or_else(|| DecodeError::UnsupportedFormat("item cursor overflow".to_string()))?;

    Ok((Item { item_type, size }, cursor))
}

fn read_buffer_index_from_trailer(file: &File, file_len: u64) -> Result<BufferIndex, DecodeError> {
    let trailer_size = ITEM_HEADER_SIZE + BUFFER_INDEX_SIZE as u64;

    if file_len < trailer_size {
        return Err(DecodeError::UnsupportedFormat(
            "file is too small to contain trailer index".to_string(),
        ));
    }

    let trailer_offset = file_len - trailer_size;
    let (item, cursor) = read_item_at(file, trailer_offset)?;

    if item.item_type != ContainerItemType::BufferIndex {
        return Err(DecodeError::UnsupportedFormat(
            "trailer item is not BUFFER_INDEX".to_string(),
        ));
    }

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

    ensure_range(file_len, index.index_data_offset, total_len)?;
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
    frame_offsets: &[BufferOffset],
) -> Result<Vec<ParsedFrame>, DecodeError> {
    let mut frames = Vec::with_capacity(frame_offsets.len());

    for frame_offset in frame_offsets {
        let item_offset = u64::try_from(frame_offset.offset).map_err(|_| {
            DecodeError::UnsupportedFormat("negative frame buffer offset".to_string())
        })?;

        let (buffer_item, payload_offset) = read_item_at(file, item_offset)?;
        if buffer_item.item_type != ContainerItemType::Buffer {
            return Err(DecodeError::UnsupportedFormat(
                "frame offset does not point to a BUFFER item".to_string(),
            ));
        }

        let payload_len = buffer_item.size;
        let payload_len_usize = usize::try_from(payload_len).map_err(|_| {
            DecodeError::UnsupportedFormat("frame payload length overflows usize".to_string())
        })?;

        ensure_range(file_len, payload_offset, payload_len_usize)?;

        let metadata_item_offset = payload_offset
            .checked_add(u64::from(payload_len))
            .ok_or_else(|| {
                DecodeError::UnsupportedFormat("metadata offset overflow".to_string())
            })?;

        let (metadata_item, metadata_offset) = read_item_at(file, metadata_item_offset)?;
        if metadata_item.item_type != ContainerItemType::Metadata {
            return Err(DecodeError::UnsupportedFormat(
                "BUFFER item is not followed by METADATA".to_string(),
            ));
        }

        let metadata_len = metadata_item.size;
        let metadata_len_usize = usize::try_from(metadata_len).map_err(|_| {
            DecodeError::UnsupportedFormat("frame metadata length overflows usize".to_string())
        })?;

        ensure_range(file_len, metadata_offset, metadata_len_usize)?;

        frames.push(ParsedFrame {
            payload_offset,
            payload_len,
            metadata_offset,
            metadata_len,
            timestamp_ns: frame_offset.timestamp,
        });
    }

    Ok(frames)
}

#[derive(Debug, Default)]
struct ParsedAudioRecords {
    chunks: Vec<ParsedAudioChunk>,
    index_items: Vec<ParsedAudioIndex>,
    metadata_jsons: Vec<String>,
}

fn read_audio_records(
    file: &File,
    file_len: u64,
    start: u64,
    end: u64,
) -> Result<ParsedAudioRecords, DecodeError> {
    if end < start {
        return Ok(ParsedAudioRecords::default());
    }

    let mut cursor = start;
    let mut records = ParsedAudioRecords::default();

    while cursor
        .checked_add(ITEM_HEADER_SIZE)
        .is_some_and(|next| next <= end)
    {
        let item_offset = cursor;
        let (item, payload_offset) = read_item_at(file, item_offset)?;
        let payload_len = usize::try_from(item.size).map_err(|_| {
            DecodeError::UnsupportedFormat("container item length overflows usize".to_string())
        })?;

        ensure_range(file_len, payload_offset, payload_len)?;

        let next_cursor = payload_offset
            .checked_add(u64::from(item.size))
            .ok_or_else(|| DecodeError::UnsupportedFormat("item end overflow".to_string()))?;

        if next_cursor > end {
            return Err(DecodeError::UnsupportedFormat(
                "container item extends beyond indexed item stream".to_string(),
            ));
        }

        match item.item_type {
            ContainerItemType::AudioIndex => {
                records.index_items.push(ParsedAudioIndex {
                    payload_offset,
                    payload_len: item.size,
                });
            }
            ContainerItemType::AudioData => {
                records.chunks.push(ParsedAudioChunk {
                    payload_offset,
                    payload_len: item.size,
                    timestamp_ns: None,
                });
            }
            ContainerItemType::AudioDataMetadata => {
                let bytes = read_vec_at(file, payload_offset, payload_len)?;

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

                // Keep JSON metadata only when the payload actually looks like
                // JSON. Real AUDIO_DATA_METADATA payloads are commonly binary
                // timestamp structs, so non-JSON must remain non-fatal.
                if let Ok(text) = std::str::from_utf8(&bytes) {
                    let trimmed = text.trim_start();
                    if trimmed.starts_with('{') || trimmed.starts_with('[') {
                        records.metadata_jsons.push(text.to_string());
                    }
                }
            }
            ContainerItemType::BufferIndex
            | ContainerItemType::BufferIndexData
            | ContainerItemType::Buffer
            | ContainerItemType::Metadata => {}
        }

        cursor = next_cursor;
    }

    Ok(records)
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
    let Ok((item, payload_offset)) = read_item_at(file, possible_header_offset) else {
        return Ok(default_end);
    };

    if item.item_type == ContainerItemType::BufferIndexData
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
