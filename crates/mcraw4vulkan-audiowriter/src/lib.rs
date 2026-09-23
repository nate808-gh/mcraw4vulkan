// Writes mcraw4vulkan PCM audio as BW64 `.wav` files.
//
// This crate is intentionally small and format-focused. Source parsing,
// metadata derivation, PCM decode, timestamp sync, and timeline policy all stay
// outside the writer. Callers hand a writer-facing description plus interleaved
// PCM samples to this crate and receive stable BW64 bytes or a BW64 file on disk
// without any sample timing changes here.
//
// Project policy: always write BW64, even for small audio files. This keeps one
// cross-platform output path and avoids switching between classic RIFF/WAV and
// BW64 when recordings cross the 4 GiB boundary.
//
// Compatibility note:
// BW64 still has 32-bit size fields in the outer chunk and data chunk. When a
// size fits in u32, this writer stores the actual value there. When a size does
// not fit, it stores 0xffff_ffff and puts the real 64-bit size in ds64. The file
// is always BW64 because the top-level chunk id is always "BW64" and the ds64
// chunk is always present.

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use mcraw4vulkan_core::AudioTrackInfo;
use thiserror::Error;

const SIZE_SENTINEL: u32 = 0xFFFF_FFFF;
const WAVE_FORMAT_PCM: u16 = 1;
const DS64_CHUNK_SIZE: u32 = 28;
const FMT_CHUNK_SIZE_PCM: u32 = 16;
const DS64_TABLE_LENGTH: u32 = 0;
const CHNA_AUDIO_ID_RECORD_SIZE: usize = 40;
const CHNA_UID_BYTES: usize = 12;
const CHNA_TRACK_FORMAT_ID_BYTES: usize = 14;
const CHNA_PACK_FORMAT_ID_BYTES: usize = 11;

// Writer-facing description for BW64 signed 16-bit PCM output.
//
// This contains the PCM track shape plus optional WAV/BW64 metadata supplied by
// callers. Source-specific metadata parsing must stay in the caller, not in this
// serializer crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bw64PcmS16LeDescription {
    pub track: AudioTrackInfo,
    pub ixml: Option<Bw64IxmlMetadata>,
    pub chna: Option<Bw64ChnaMetadata>,
}

impl Bw64PcmS16LeDescription {
    pub fn new(track: AudioTrackInfo) -> Self {
        Self {
            track,
            ixml: None,
            chna: None,
        }
    }
}

// Optional iXML metadata to serialize into a BW64 `iXML` chunk.
//
// This is deliberately writer-facing and source-agnostic. Container/session
// layers should parse or derive values such as project names, scene/take labels,
// and frame-rate strings before constructing this metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bw64IxmlMetadata {
    pub project: Option<String>,
    pub note: Option<String>,
    pub circled: Option<String>,
    pub blackmagic_keywords: Option<String>,
    pub scene: Option<String>,
    pub blackmagic_shot: Option<String>,
    pub take: Option<String>,
    pub blackmagic_angle: Option<String>,
    pub tape: Option<String>,
    pub speed: Option<Bw64IxmlSpeed>,
}

// iXML SPEED block values.
//
// The strings are serialized as supplied so callers can choose decimal or
// rational formatting. The writer does not derive FPS from audio/video
// timestamps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bw64IxmlSpeed {
    pub current_speed: String,
    pub master_speed: String,
    pub timecode_rate: String,
    pub timecode_flag: String,
}

// Optional CHNA metadata to serialize into a BW64 `chna` chunk.
//
// MotionCam files may carry an empty reserved CHNA table: zero track and UID
// counts followed by zero-filled 40-byte audioID records. Populated and reserved
// entries share the same fixed-width record layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bw64ChnaMetadata {
    pub num_tracks: u16,
    pub num_uids: u16,
    pub audio_ids: Vec<Bw64ChnaAudioId>,
    pub reserved_audio_id_count: usize,
}

impl Bw64ChnaMetadata {
    pub fn empty_reserved_audio_ids(reserved_audio_id_count: usize) -> Self {
        Self {
            num_tracks: 0,
            num_uids: 0,
            audio_ids: Vec::new(),
            reserved_audio_id_count,
        }
    }
}

// One fixed-width CHNA audioID record.
//
// Records serialize to 40 bytes: u16 trackIndex, 12-byte UID, 14-byte
// trackFormatID, 11-byte packFormatID, and one trailing zero byte. String fields
// are ASCII identifiers and are zero-filled to their fixed widths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bw64ChnaAudioId {
    pub track_index: u16,
    pub uid: String,
    pub track_format_id: String,
    pub pack_format_id: String,
}

// Errors returned by the BW64 writer.
//
// These are intentionally validation-oriented so callers get clear failures for
// unsupported audio formats or impossible byte/sample counts.
#[derive(Debug, Error)]
pub enum AudioWriteError {
    #[error("audio writer currently supports only 16-bit PCM, found {0} bits")]
    UnsupportedBitsPerSample(u16),

    #[error("audio writer requires at least one channel")]
    InvalidChannelCount,

    #[error("interleaved sample count is not divisible by channel count")]
    InvalidInterleavedSampleCount,

    #[error("audio byte count overflow")]
    AudioByteCountOverflow,

    #[error("BW64 file size overflow")]
    FileSizeOverflow,

    #[error("iXML chunk size overflow")]
    IxmlChunkSizeOverflow,

    #[error("chna chunk size overflow")]
    ChnaChunkSizeOverflow,

    #[error("chna numUIDs does not match audioID record count")]
    InvalidChnaAudioIdCount,

    #[error("chna field {field} must be ASCII")]
    ChnaFieldNotAscii { field: &'static str },

    #[error("chna field {field} is {actual_bytes} bytes, maximum is {max_bytes}")]
    ChnaFieldTooLong {
        field: &'static str,
        max_bytes: usize,
        actual_bytes: usize,
    },

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

// Summary of the BW64 file layout produced by this crate.
//
// FUSE can use this to expose a stable file size before any read request, and
// validation tools can print it when debugging output format behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bw64LayoutInfo {
    pub riff_size_64: u64,
    pub data_size_64: u64,
    pub sample_frames: u64,
    pub file_size: u64,
    pub byte_rate: u32,
    pub block_align: u16,
}

// Build a complete BW64 byte buffer for interleaved signed 16-bit PCM.
//
// The returned bytes use a `.wav`-compatible BW64 container:
//
//   BW64 + WAVE
//   ds64 chunk with 64-bit riff/data/sample counts
//   fmt  chunk for PCM
//   data chunk
//   interleaved PCM payload
//
// For compatibility, the 32-bit size fields contain the actual size when the
// value fits in u32, and the BW64 sentinel 0xffff_ffff only when the 64-bit ds64
// value is required.
pub fn bw64_pcm_s16le_bytes(
    track: AudioTrackInfo,
    interleaved_samples: &[i16],
) -> Result<Vec<u8>, AudioWriteError> {
    let description = Bw64PcmS16LeDescription::new(track);

    bw64_pcm_s16le_bytes_with_description(&description, interleaved_samples)
}

// Build a complete BW64 byte buffer from a writer-facing PCM description.
//
// This is the preferred entry point for callers that already assembled output
// metadata from a container/session layer. Borrowing the description avoids
// cloning its owned metadata for ordinary writes.
pub fn bw64_pcm_s16le_bytes_with_description(
    description: &Bw64PcmS16LeDescription,
    interleaved_samples: &[i16],
) -> Result<Vec<u8>, AudioWriteError> {
    let layout = bw64_layout_info_with_description(description, interleaved_samples.len())?;
    let capacity = usize::try_from(layout.file_size).unwrap_or(0);

    let mut output = Vec::with_capacity(capacity);
    write_bw64_pcm_s16le_with_description(&mut output, description, interleaved_samples)?;

    Ok(output)
}

// Build only the BW64 header and metadata bytes for signed 16-bit PCM.
//
// The returned bytes end immediately after the `data` chunk header. Callers can
// append or serve the PCM data bytes lazily while preserving the exact BW64
// layout produced by the eager writer.
pub fn bw64_pcm_s16le_header_bytes_for_sample_frames_with_description(
    description: &Bw64PcmS16LeDescription,
    sample_frames: u64,
) -> Result<Vec<u8>, AudioWriteError> {
    let layout = bw64_layout_info_for_sample_frames_with_description(description, sample_frames)?;
    let header_size = layout
        .file_size
        .checked_sub(layout.data_size_64)
        .ok_or(AudioWriteError::FileSizeOverflow)?;
    let capacity = usize::try_from(header_size).unwrap_or(0);
    let mut output = Vec::with_capacity(capacity);

    write_bw64_pcm_s16le_header_for_sample_frames_with_description(
        &mut output,
        description,
        sample_frames,
    )?;

    Ok(output)
}

// Write a complete BW64 file to disk.
//
// This materializes directly to disk. Byte-buffer and header-only APIs cover
// callers that serve complete or lazy virtual files.
pub fn write_bw64_pcm_s16le_file(
    path: impl AsRef<Path>,
    track: AudioTrackInfo,
    interleaved_samples: &[i16],
) -> Result<Bw64LayoutInfo, AudioWriteError> {
    let description = Bw64PcmS16LeDescription::new(track);

    write_bw64_pcm_s16le_file_with_description(path, &description, interleaved_samples)
}

// Write a complete BW64 file to disk from a writer-facing PCM description.
pub fn write_bw64_pcm_s16le_file_with_description(
    path: impl AsRef<Path>,
    description: &Bw64PcmS16LeDescription,
    interleaved_samples: &[i16],
) -> Result<Bw64LayoutInfo, AudioWriteError> {
    let layout = bw64_layout_info_with_description(description, interleaved_samples.len())?;
    let mut file = File::create(path)?;

    write_bw64_pcm_s16le_with_description(&mut file, description, interleaved_samples)?;

    Ok(layout)
}

// Write BW64 bytes to any std::io::Write sink.
//
// This is the shared implementation used by both byte-buffer and file output.
pub fn write_bw64_pcm_s16le<W: Write>(
    writer: &mut W,
    track: AudioTrackInfo,
    interleaved_samples: &[i16],
) -> Result<Bw64LayoutInfo, AudioWriteError> {
    let description = Bw64PcmS16LeDescription::new(track);

    write_bw64_pcm_s16le_with_description(writer, &description, interleaved_samples)
}

// Write BW64 bytes from a writer-facing PCM description to any std::io::Write sink.
pub fn write_bw64_pcm_s16le_with_description<W: Write>(
    writer: &mut W,
    description: &Bw64PcmS16LeDescription,
    interleaved_samples: &[i16],
) -> Result<Bw64LayoutInfo, AudioWriteError> {
    let track = description.track;
    let layout = bw64_layout_info_with_description(description, interleaved_samples.len())?;

    writer.write_all(b"BW64")?;
    writer.write_all(&riff_size_field(layout).to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    write_ds64_chunk(writer, layout)?;
    write_fmt_chunk(writer, track, layout)?;
    write_ixml_chunk_if_present(writer, description)?;
    write_chna_chunk_if_present(writer, description)?;
    write_data_chunk_header(writer, layout)?;

    for sample in interleaved_samples {
        writer.write_all(&sample.to_le_bytes())?;
    }

    Ok(layout)
}

// Write only BW64 header and metadata chunks through the `data` chunk header.
//
// This is the layout-only entry point used by lazy virtual files. It does not
// write PCM samples and does not change eager writer behavior.
pub fn write_bw64_pcm_s16le_header_for_sample_frames_with_description<W: Write>(
    writer: &mut W,
    description: &Bw64PcmS16LeDescription,
    sample_frames: u64,
) -> Result<Bw64LayoutInfo, AudioWriteError> {
    let track = description.track;
    let layout = bw64_layout_info_for_sample_frames_with_description(description, sample_frames)?;

    writer.write_all(b"BW64")?;
    writer.write_all(&riff_size_field(layout).to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    write_ds64_chunk(writer, layout)?;
    write_fmt_chunk(writer, track, layout)?;
    write_ixml_chunk_if_present(writer, description)?;
    write_chna_chunk_if_present(writer, description)?;
    write_data_chunk_header(writer, layout)?;

    Ok(layout)
}

// Compute the exact BW64 layout without writing bytes.
//
// This lets FUSE compute stable file size before generating or serving a WAV.
pub fn bw64_layout_info(
    track: AudioTrackInfo,
    interleaved_sample_count: usize,
) -> Result<Bw64LayoutInfo, AudioWriteError> {
    let description = Bw64PcmS16LeDescription::new(track);

    bw64_layout_info_with_description(&description, interleaved_sample_count)
}

// Compute the exact BW64 layout from a writer-facing PCM description.
pub fn bw64_layout_info_with_description(
    description: &Bw64PcmS16LeDescription,
    interleaved_sample_count: usize,
) -> Result<Bw64LayoutInfo, AudioWriteError> {
    let track = description.track;
    validate_track(track)?;

    let channels_usize = usize::from(track.channels);

    if (interleaved_sample_count / channels_usize) * channels_usize != interleaved_sample_count {
        return Err(AudioWriteError::InvalidInterleavedSampleCount);
    }

    let sample_frames = u64::try_from(interleaved_sample_count / channels_usize)
        .map_err(|_| AudioWriteError::AudioByteCountOverflow)?;
    bw64_layout_info_for_sample_frames_with_description(description, sample_frames)
}

// Compute the exact BW64 layout from a sample-frame count without requiring the
// caller to materialize or count an interleaved PCM slice.
pub fn bw64_layout_info_for_sample_frames_with_description(
    description: &Bw64PcmS16LeDescription,
    sample_frames: u64,
) -> Result<Bw64LayoutInfo, AudioWriteError> {
    let track = description.track;
    validate_track(track)?;

    let block_align_u32 = u32::from(track.channels)
        .checked_mul(u32::from(track.bits_per_sample / 8))
        .ok_or(AudioWriteError::AudioByteCountOverflow)?;

    let data_size_64 = sample_frames
        .checked_mul(u64::from(block_align_u32))
        .ok_or(AudioWriteError::AudioByteCountOverflow)?;

    let block_align =
        u16::try_from(block_align_u32).map_err(|_| AudioWriteError::AudioByteCountOverflow)?;

    let byte_rate = track
        .sample_rate_hz
        .checked_mul(u32::from(block_align))
        .ok_or(AudioWriteError::AudioByteCountOverflow)?;

    let metadata_chunk_bytes = ixml_chunk_total_bytes(description)?
        .checked_add(chna_chunk_total_bytes(description)?)
        .ok_or(AudioWriteError::FileSizeOverflow)?;
    let header_bytes = fixed_header_bytes()
        .checked_add(metadata_chunk_bytes)
        .ok_or(AudioWriteError::FileSizeOverflow)?;

    let file_size = header_bytes
        .checked_add(data_size_64)
        .ok_or(AudioWriteError::FileSizeOverflow)?;

    // RIFF/BW64-family size is the file length minus the 8 bytes occupied by
    // the chunk id and size field.
    let riff_size_64 = file_size
        .checked_sub(8)
        .ok_or(AudioWriteError::FileSizeOverflow)?;

    Ok(Bw64LayoutInfo {
        riff_size_64,
        data_size_64,
        sample_frames,
        file_size,
        byte_rate,
        block_align,
    })
}

fn validate_track(track: AudioTrackInfo) -> Result<(), AudioWriteError> {
    if track.channels == 0 {
        return Err(AudioWriteError::InvalidChannelCount);
    }

    if track.bits_per_sample != 16 {
        return Err(AudioWriteError::UnsupportedBitsPerSample(
            track.bits_per_sample,
        ));
    }

    Ok(())
}

fn riff_size_field(layout: Bw64LayoutInfo) -> u32 {
    u32::try_from(layout.riff_size_64).unwrap_or(SIZE_SENTINEL)
}

fn data_size_field(layout: Bw64LayoutInfo) -> u32 {
    u32::try_from(layout.data_size_64).unwrap_or(SIZE_SENTINEL)
}

fn write_ds64_chunk<W: Write>(
    writer: &mut W,
    layout: Bw64LayoutInfo,
) -> Result<(), AudioWriteError> {
    writer.write_all(b"ds64")?;
    writer.write_all(&DS64_CHUNK_SIZE.to_le_bytes())?;
    writer.write_all(&layout.riff_size_64.to_le_bytes())?;
    writer.write_all(&layout.data_size_64.to_le_bytes())?;
    writer.write_all(&layout.sample_frames.to_le_bytes())?;
    writer.write_all(&DS64_TABLE_LENGTH.to_le_bytes())?;

    Ok(())
}

fn write_fmt_chunk<W: Write>(
    writer: &mut W,
    track: AudioTrackInfo,
    layout: Bw64LayoutInfo,
) -> Result<(), AudioWriteError> {
    writer.write_all(b"fmt ")?;
    writer.write_all(&FMT_CHUNK_SIZE_PCM.to_le_bytes())?;
    writer.write_all(&WAVE_FORMAT_PCM.to_le_bytes())?;
    writer.write_all(&track.channels.to_le_bytes())?;
    writer.write_all(&track.sample_rate_hz.to_le_bytes())?;
    writer.write_all(&layout.byte_rate.to_le_bytes())?;
    writer.write_all(&layout.block_align.to_le_bytes())?;
    writer.write_all(&track.bits_per_sample.to_le_bytes())?;

    Ok(())
}

fn write_ixml_chunk_if_present<W: Write>(
    writer: &mut W,
    description: &Bw64PcmS16LeDescription,
) -> Result<(), AudioWriteError> {
    let Some(ixml) = description.ixml.as_ref() else {
        return Ok(());
    };

    let xml = build_ixml_document(ixml);
    let xml_bytes = xml.as_bytes();
    let chunk_size =
        u32::try_from(xml_bytes.len()).map_err(|_| AudioWriteError::IxmlChunkSizeOverflow)?;

    writer.write_all(b"iXML")?;
    writer.write_all(&chunk_size.to_le_bytes())?;
    writer.write_all(xml_bytes)?;

    if xml_bytes.len() & 1 != 0 {
        writer.write_all(&[0])?;
    }

    Ok(())
}

fn write_chna_chunk_if_present<W: Write>(
    writer: &mut W,
    description: &Bw64PcmS16LeDescription,
) -> Result<(), AudioWriteError> {
    let Some(chna) = description.chna.as_ref() else {
        return Ok(());
    };

    validate_chna(chna)?;
    let chunk_size = chna_payload_bytes(chna)?;
    let chunk_size =
        u32::try_from(chunk_size).map_err(|_| AudioWriteError::ChnaChunkSizeOverflow)?;

    writer.write_all(b"chna")?;
    writer.write_all(&chunk_size.to_le_bytes())?;
    writer.write_all(&chna.num_tracks.to_le_bytes())?;
    writer.write_all(&chna.num_uids.to_le_bytes())?;

    for audio_id in &chna.audio_ids {
        write_chna_audio_id_record(writer, audio_id)?;
    }

    let reserved_record = [0u8; CHNA_AUDIO_ID_RECORD_SIZE];
    for _ in 0..chna.reserved_audio_id_count {
        writer.write_all(&reserved_record)?;
    }

    Ok(())
}

fn write_data_chunk_header<W: Write>(
    writer: &mut W,
    layout: Bw64LayoutInfo,
) -> Result<(), AudioWriteError> {
    writer.write_all(b"data")?;
    writer.write_all(&data_size_field(layout).to_le_bytes())?;

    Ok(())
}

fn ixml_chunk_total_bytes(description: &Bw64PcmS16LeDescription) -> Result<u64, AudioWriteError> {
    let Some(ixml) = description.ixml.as_ref() else {
        return Ok(0);
    };

    let xml = build_ixml_document(ixml);
    let xml_len = u64::try_from(xml.len()).map_err(|_| AudioWriteError::IxmlChunkSizeOverflow)?;

    if xml_len > u64::from(u32::MAX) {
        return Err(AudioWriteError::IxmlChunkSizeOverflow);
    }

    let padded_xml_len = xml_len
        .checked_add(xml_len % 2)
        .ok_or(AudioWriteError::IxmlChunkSizeOverflow)?;

    8u64.checked_add(padded_xml_len)
        .ok_or(AudioWriteError::IxmlChunkSizeOverflow)
}

fn chna_chunk_total_bytes(description: &Bw64PcmS16LeDescription) -> Result<u64, AudioWriteError> {
    let Some(chna) = description.chna.as_ref() else {
        return Ok(0);
    };

    validate_chna(chna)?;
    let payload_bytes = chna_payload_bytes(chna)?;

    8u64.checked_add(payload_bytes)
        .ok_or(AudioWriteError::ChnaChunkSizeOverflow)
}

fn chna_payload_bytes(chna: &Bw64ChnaMetadata) -> Result<u64, AudioWriteError> {
    let record_count = chna
        .audio_ids
        .len()
        .checked_add(chna.reserved_audio_id_count)
        .ok_or(AudioWriteError::ChnaChunkSizeOverflow)?;

    let records_bytes = record_count
        .checked_mul(CHNA_AUDIO_ID_RECORD_SIZE)
        .ok_or(AudioWriteError::ChnaChunkSizeOverflow)?;

    let payload_bytes = 4usize
        .checked_add(records_bytes)
        .ok_or(AudioWriteError::ChnaChunkSizeOverflow)?;

    u64::try_from(payload_bytes).map_err(|_| AudioWriteError::ChnaChunkSizeOverflow)
}

fn validate_chna(chna: &Bw64ChnaMetadata) -> Result<(), AudioWriteError> {
    if usize::from(chna.num_uids) != chna.audio_ids.len() {
        return Err(AudioWriteError::InvalidChnaAudioIdCount);
    }

    for audio_id in &chna.audio_ids {
        validate_chna_string("uid", &audio_id.uid, CHNA_UID_BYTES)?;
        validate_chna_string(
            "track_format_id",
            &audio_id.track_format_id,
            CHNA_TRACK_FORMAT_ID_BYTES,
        )?;
        validate_chna_string(
            "pack_format_id",
            &audio_id.pack_format_id,
            CHNA_PACK_FORMAT_ID_BYTES,
        )?;
    }

    Ok(())
}

fn validate_chna_string(
    field: &'static str,
    value: &str,
    max_bytes: usize,
) -> Result<(), AudioWriteError> {
    if !value.is_ascii() {
        return Err(AudioWriteError::ChnaFieldNotAscii { field });
    }

    let actual_bytes = value.len();
    if actual_bytes > max_bytes {
        return Err(AudioWriteError::ChnaFieldTooLong {
            field,
            max_bytes,
            actual_bytes,
        });
    }

    Ok(())
}

fn write_chna_audio_id_record<W: Write>(
    writer: &mut W,
    audio_id: &Bw64ChnaAudioId,
) -> Result<(), AudioWriteError> {
    writer.write_all(&audio_id.track_index.to_le_bytes())?;
    write_zero_filled_ascii_field(writer, "uid", &audio_id.uid, CHNA_UID_BYTES)?;
    write_zero_filled_ascii_field(
        writer,
        "track_format_id",
        &audio_id.track_format_id,
        CHNA_TRACK_FORMAT_ID_BYTES,
    )?;
    write_zero_filled_ascii_field(
        writer,
        "pack_format_id",
        &audio_id.pack_format_id,
        CHNA_PACK_FORMAT_ID_BYTES,
    )?;
    writer.write_all(&[0])?;

    Ok(())
}

fn write_zero_filled_ascii_field<W: Write>(
    writer: &mut W,
    field: &'static str,
    value: &str,
    width: usize,
) -> Result<(), AudioWriteError> {
    validate_chna_string(field, value, width)?;

    writer.write_all(value.as_bytes())?;

    for _ in value.len()..width {
        writer.write_all(&[0])?;
    }

    Ok(())
}

fn build_ixml_document(ixml: &Bw64IxmlMetadata) -> String {
    let mut output = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?><BWFXML>");

    append_xml_element(&mut output, "IXML_VERSION", "1.5");
    append_optional_xml_element(&mut output, "PROJECT", ixml.project.as_deref());
    append_optional_xml_element(&mut output, "NOTE", ixml.note.as_deref());
    append_optional_xml_element(&mut output, "CIRCLED", ixml.circled.as_deref());
    append_optional_xml_element(
        &mut output,
        "BLACKMAGIC-KEYWORDS",
        ixml.blackmagic_keywords.as_deref(),
    );
    append_optional_xml_element(&mut output, "TAPE", ixml.tape.as_deref());
    append_optional_xml_element(&mut output, "SCENE", ixml.scene.as_deref());
    append_optional_xml_element(
        &mut output,
        "BLACKMAGIC-SHOT",
        ixml.blackmagic_shot.as_deref(),
    );
    append_optional_xml_element(&mut output, "TAKE", ixml.take.as_deref());
    append_optional_xml_element(
        &mut output,
        "BLACKMAGIC-ANGLE",
        ixml.blackmagic_angle.as_deref(),
    );

    if let Some(speed) = ixml.speed.as_ref() {
        output.push_str("<SPEED>");
        append_xml_element(&mut output, "MASTER_SPEED", &speed.master_speed);
        append_xml_element(&mut output, "CURRENT_SPEED", &speed.current_speed);
        append_xml_element(&mut output, "TIMECODE_RATE", &speed.timecode_rate);
        append_xml_element(&mut output, "TIMECODE_FLAG", &speed.timecode_flag);
        output.push_str("</SPEED>");
    }

    output.push_str("</BWFXML>");
    output
}

fn append_optional_xml_element(output: &mut String, tag: &str, value: Option<&str>) {
    if let Some(value) = value {
        append_xml_element(output, tag, value);
    }
}

fn append_xml_element(output: &mut String, tag: &str, value: &str) {
    output.push('<');
    output.push_str(tag);
    output.push('>');
    append_escaped_xml_text(output, value);
    output.push_str("</");
    output.push_str(tag);
    output.push('>');
}

fn append_escaped_xml_text(output: &mut String, value: &str) {
    for character in value.chars() {
        match character {
            '&' => output.push_str("&amp;"),
            '<' => output.push_str("&lt;"),
            '>' => output.push_str("&gt;"),
            '"' => output.push_str("&quot;"),
            '\'' => output.push_str("&apos;"),
            _ => output.push(character),
        }
    }
}

const fn fixed_header_bytes() -> u64 {
    // BW64 header:
    //   "BW64" + u32 size + "WAVE" = 12
    // ds64 chunk:
    //   id + size + 28 byte payload = 36
    // fmt chunk:
    //   id + size + 16 byte PCM payload = 24
    // data header:
    //   id + u32 size = 8
    //
    // Total before PCM payload = 80 bytes.
    12 + 36 + 24 + 8
}
