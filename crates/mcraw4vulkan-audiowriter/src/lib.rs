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

#[cfg(test)]
mod tests {
    use mcraw4vulkan_core::AudioTrackInfo;

    use super::{
        Bw64ChnaAudioId, Bw64ChnaMetadata, Bw64IxmlMetadata, Bw64IxmlSpeed,
        Bw64PcmS16LeDescription, CHNA_AUDIO_ID_RECORD_SIZE, SIZE_SENTINEL, bw64_layout_info,
        bw64_layout_info_for_sample_frames_with_description, bw64_layout_info_with_description,
        bw64_pcm_s16le_bytes, bw64_pcm_s16le_bytes_with_description,
        bw64_pcm_s16le_header_bytes_for_sample_frames_with_description,
    };

    #[derive(Debug, Clone, Copy)]
    struct ChunkInfo {
        offset: usize,
        payload_offset: usize,
        size: usize,
        end_offset: usize,
    }

    #[test]
    fn writes_bw64_header_and_ds64_chunk_for_small_file() {
        let track = AudioTrackInfo {
            sample_rate_hz: 48_000,
            channels: 2,
            bits_per_sample: 16,
            total_samples: 4,
        };

        let samples = [1i16, -1, 2, -2, 3, -3, 4, -4];
        let bytes = bw64_pcm_s16le_bytes(track, &samples).expect("BW64 bytes");

        assert_eq!(&bytes[0..4], b"BW64");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 88);
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"ds64");

        let data_header_offset = 72;
        assert_eq!(&bytes[data_header_offset..data_header_offset + 4], b"data");
        assert_eq!(
            u32::from_le_bytes(
                bytes[data_header_offset + 4..data_header_offset + 8]
                    .try_into()
                    .unwrap()
            ),
            16
        );
    }

    #[test]
    fn computes_stable_file_size() {
        let track = AudioTrackInfo {
            sample_rate_hz: 48_000,
            channels: 2,
            bits_per_sample: 16,
            total_samples: 4,
        };

        let layout = bw64_layout_info(track, 8).expect("layout");

        assert_eq!(layout.sample_frames, 4);
        assert_eq!(layout.data_size_64, 16);
        assert_eq!(layout.file_size, 96);
        assert_eq!(layout.riff_size_64, 88);
        assert_eq!(layout.byte_rate, 192_000);
        assert_eq!(layout.block_align, 4);
    }

    #[test]
    fn description_api_matches_compatibility_api() {
        let track = AudioTrackInfo {
            sample_rate_hz: 48_000,
            channels: 2,
            bits_per_sample: 16,
            total_samples: 4,
        };
        let description = Bw64PcmS16LeDescription::new(track);
        let samples = [1i16, -1, 2, -2, 3, -3, 4, -4];

        let legacy_bytes = bw64_pcm_s16le_bytes(track, &samples).expect("legacy BW64 bytes");
        let description_bytes = bw64_pcm_s16le_bytes_with_description(&description, &samples)
            .expect("description BW64 bytes");
        let legacy_layout = bw64_layout_info(track, samples.len()).expect("legacy layout");
        let description_layout = bw64_layout_info_with_description(&description, samples.len())
            .expect("description layout");

        assert_eq!(description.track, track);
        assert_eq!(description_bytes, legacy_bytes);
        assert_eq!(description_layout, legacy_layout);
    }

    #[test]
    fn header_bytes_match_eager_prefix() {
        let description = description_with_ixml("Recorded with MotionCam");
        let samples = [1i16, -1, 2, -2];
        let eager_bytes = bw64_pcm_s16le_bytes_with_description(&description, &samples)
            .expect("eager BW64 bytes");
        let layout = bw64_layout_info_for_sample_frames_with_description(&description, 2)
            .expect("sample-frame layout");
        let header_bytes =
            bw64_pcm_s16le_header_bytes_for_sample_frames_with_description(&description, 2)
                .expect("header bytes");
        let header_len = usize::try_from(layout.file_size - layout.data_size_64).unwrap();

        assert_eq!(header_bytes.len(), header_len);
        assert_eq!(header_bytes, eager_bytes[..header_len]);
    }

    #[test]
    fn writes_ixml_chunk_before_data_when_provided() {
        let description = description_with_ixml("Recorded with MotionCam");
        let samples = [1i16, -1, 2, -2];
        let bytes = bw64_pcm_s16le_bytes_with_description(&description, &samples)
            .expect("BW64 bytes with iXML");
        let ixml = find_chunk(&bytes, b"iXML").expect("iXML chunk");
        let data = find_chunk(&bytes, b"data").expect("data chunk");

        assert!(ixml.offset < data.offset);
        assert_eq!(data.offset, ixml.end_offset);
    }

    #[test]
    fn pads_odd_sized_ixml_chunk() {
        let samples = [1i16, -1, 2, -2];

        let (bytes, ixml, data) = ["x", "xy"]
            .into_iter()
            .find_map(|note| {
                let description = description_with_ixml(note);
                let bytes = bw64_pcm_s16le_bytes_with_description(&description, &samples).ok()?;
                let ixml = find_chunk(&bytes, b"iXML")?;
                let data = find_chunk(&bytes, b"data")?;

                if ixml.size % 2 == 1 {
                    Some((bytes, ixml, data))
                } else {
                    None
                }
            })
            .expect("test metadata can produce an odd iXML payload");

        assert_eq!(bytes[ixml.payload_offset + ixml.size], 0);
        assert_eq!(data.offset, ixml.payload_offset + ixml.size + 1);
    }

    #[test]
    fn layout_matches_data_offset_and_total_byte_len_with_ixml() {
        let description = description_with_ixml("Recorded with MotionCam");
        let samples = [1i16, -1, 2, -2];
        let bytes = bw64_pcm_s16le_bytes_with_description(&description, &samples)
            .expect("BW64 bytes with iXML");
        let layout =
            bw64_layout_info_with_description(&description, samples.len()).expect("layout");
        let ixml = find_chunk(&bytes, b"iXML").expect("iXML chunk");
        let data = find_chunk(&bytes, b"data").expect("data chunk");

        assert_eq!(ixml.offset, 72);
        assert_eq!(data.offset, ixml.end_offset);
        assert_eq!(data.size, samples.len() * std::mem::size_of::<i16>());
        assert_eq!(bytes.len(), data.payload_offset + data.size);
        assert_eq!(layout.file_size, bytes.len() as u64);
        assert_eq!(layout.riff_size_64, bytes.len() as u64 - 8);
        assert_eq!(layout.data_size_64, data.size as u64);
        assert_eq!(layout.sample_frames, 2);
    }

    #[test]
    fn generated_ixml_is_utf8_and_contains_motioncam_fields() {
        let description = description_with_ixml("Recorded with MotionCam");
        let samples = [1i16, -1, 2, -2];
        let bytes = bw64_pcm_s16le_bytes_with_description(&description, &samples)
            .expect("BW64 bytes with iXML");
        let ixml = find_chunk(&bytes, b"iXML").expect("iXML chunk");
        let text =
            std::str::from_utf8(&bytes[ixml.payload_offset..ixml.payload_offset + ixml.size])
                .expect("valid UTF-8 iXML");

        assert!(text.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?><BWFXML>"));
        assert!(text.contains("<IXML_VERSION>1.5</IXML_VERSION>"));
        assert!(text.contains("<PROJECT>MotionCam</PROJECT>"));
        assert!(text.contains("<NOTE>Recorded with MotionCam</NOTE>"));
        assert!(text.contains("<CIRCLED>FALSE</CIRCLED>"));
        assert!(text.contains("<BLACKMAGIC-KEYWORDS></BLACKMAGIC-KEYWORDS>"));
        assert!(text.contains("<SCENE>1</SCENE>"));
        assert!(text.contains("<BLACKMAGIC-SHOT>1</BLACKMAGIC-SHOT>"));
        assert!(text.contains("<TAKE>1</TAKE>"));
        assert!(text.contains("<BLACKMAGIC-ANGLE>ms</BLACKMAGIC-ANGLE>"));
        assert!(text.contains("<TAPE>1</TAPE>"));
        assert!(text.contains("<MASTER_SPEED>29.997377294867</MASTER_SPEED>"));
        assert!(text.contains("<CURRENT_SPEED>29.997377294867</CURRENT_SPEED>"));
        assert!(text.contains("<TIMECODE_RATE>29.997377294867</TIMECODE_RATE>"));
        assert!(text.contains("<TIMECODE_FLAG>NDF</TIMECODE_FLAG>"));
        assert!(text.ends_with("</BWFXML>"));
    }

    #[test]
    fn writes_chna_chunk_before_data_when_provided() {
        let description = description_with_chna(Bw64ChnaMetadata::empty_reserved_audio_ids(1024));
        let samples = [1i16, -1, 2, -2];
        let bytes = bw64_pcm_s16le_bytes_with_description(&description, &samples)
            .expect("BW64 bytes with chna");
        let chna = find_chunk(&bytes, b"chna").expect("chna chunk");
        let data = find_chunk(&bytes, b"data").expect("data chunk");

        assert!(chna.offset < data.offset);
        assert_eq!(data.offset, chna.end_offset);
    }

    #[test]
    fn writes_ixml_chna_data_order_when_both_are_provided() {
        let description = description_with_ixml_and_chna();
        let samples = [1i16, -1, 2, -2];
        let bytes = bw64_pcm_s16le_bytes_with_description(&description, &samples)
            .expect("BW64 bytes with iXML and chna");
        let fmt = find_chunk(&bytes, b"fmt ").expect("fmt chunk");
        let ixml = find_chunk(&bytes, b"iXML").expect("iXML chunk");
        let chna = find_chunk(&bytes, b"chna").expect("chna chunk");
        let data = find_chunk(&bytes, b"data").expect("data chunk");

        assert_eq!(ixml.offset, fmt.end_offset);
        assert_eq!(chna.offset, ixml.end_offset);
        assert_eq!(data.offset, chna.end_offset);
    }

    #[test]
    fn writes_motioncam_like_reserved_chna_table() {
        let description = description_with_chna(Bw64ChnaMetadata::empty_reserved_audio_ids(1024));
        let samples = [1i16, -1, 2, -2];
        let bytes = bw64_pcm_s16le_bytes_with_description(&description, &samples)
            .expect("BW64 bytes with chna");
        let layout =
            bw64_layout_info_with_description(&description, samples.len()).expect("layout");
        let chna = find_chunk(&bytes, b"chna").expect("chna chunk");
        let data = find_chunk(&bytes, b"data").expect("data chunk");
        let chna_payload = &bytes[chna.payload_offset..chna.payload_offset + chna.size];

        assert_eq!(chna.size, 4 + 1024 * CHNA_AUDIO_ID_RECORD_SIZE);
        assert_eq!(chna_payload[0..4], [0, 0, 0, 0]);
        assert!(chna_payload[4..].iter().all(|value| *value == 0));
        assert_eq!(data.offset, chna.end_offset);
        assert_eq!(bytes.len(), data.payload_offset + data.size);
        assert_eq!(layout.file_size, bytes.len() as u64);
        assert_eq!(layout.riff_size_64, bytes.len() as u64 - 8);
        assert_eq!(layout.data_size_64, data.size as u64);
        assert_eq!(layout.sample_frames, 2);
    }

    #[test]
    fn serializes_populated_chna_audio_id_records() {
        let description = description_with_chna(Bw64ChnaMetadata {
            num_tracks: 1,
            num_uids: 1,
            audio_ids: vec![Bw64ChnaAudioId {
                track_index: 1,
                uid: "ATU_00000001".to_string(),
                track_format_id: "AT_00010001_01".to_string(),
                pack_format_id: "AP_00010001".to_string(),
            }],
            reserved_audio_id_count: 1,
        });
        let samples = [1i16, -1, 2, -2];
        let bytes = bw64_pcm_s16le_bytes_with_description(&description, &samples)
            .expect("BW64 bytes with populated chna");
        let chna = find_chunk(&bytes, b"chna").expect("chna chunk");
        let payload = &bytes[chna.payload_offset..chna.payload_offset + chna.size];
        let first_record = &payload[4..4 + CHNA_AUDIO_ID_RECORD_SIZE];
        let reserved_record =
            &payload[4 + CHNA_AUDIO_ID_RECORD_SIZE..4 + 2 * CHNA_AUDIO_ID_RECORD_SIZE];

        assert_eq!(payload[0..2], [1, 0]);
        assert_eq!(payload[2..4], [1, 0]);
        assert_eq!(&first_record[0..2], &[1, 0]);
        assert_eq!(&first_record[2..14], b"ATU_00000001");
        assert_eq!(&first_record[14..28], b"AT_00010001_01");
        assert_eq!(&first_record[28..39], b"AP_00010001");
        assert_eq!(first_record[39], 0);
        assert!(reserved_record.iter().all(|value| *value == 0));
    }

    #[test]
    fn rejects_interleaved_samples_not_divisible_by_channels() {
        let track = AudioTrackInfo {
            sample_rate_hz: 48_000,
            channels: 2,
            bits_per_sample: 16,
            total_samples: 0,
        };

        let err = bw64_layout_info(track, 3).expect_err("invalid interleaved sample count");

        assert!(format!("{err}").contains("interleaved sample count"));
    }

    #[test]
    fn keeps_size_sentinel_for_oversized_values_only() {
        let layout = super::Bw64LayoutInfo {
            riff_size_64: u64::from(u32::MAX) + 1,
            data_size_64: u64::from(u32::MAX) + 1,
            sample_frames: 0,
            file_size: u64::from(u32::MAX) + 9,
            byte_rate: 192_000,
            block_align: 4,
        };

        assert_eq!(super::riff_size_field(layout), SIZE_SENTINEL);
        assert_eq!(super::data_size_field(layout), SIZE_SENTINEL);
    }

    fn description_with_ixml(note: &str) -> Bw64PcmS16LeDescription {
        Bw64PcmS16LeDescription {
            track: AudioTrackInfo {
                sample_rate_hz: 48_000,
                channels: 2,
                bits_per_sample: 16,
                total_samples: 2,
            },
            ixml: Some(Bw64IxmlMetadata {
                project: Some("MotionCam".to_string()),
                note: Some(note.to_string()),
                circled: Some("FALSE".to_string()),
                blackmagic_keywords: Some(String::new()),
                scene: Some("1".to_string()),
                blackmagic_shot: Some("1".to_string()),
                take: Some("1".to_string()),
                blackmagic_angle: Some("ms".to_string()),
                tape: Some("1".to_string()),
                speed: Some(Bw64IxmlSpeed {
                    current_speed: "29.997377294867".to_string(),
                    master_speed: "29.997377294867".to_string(),
                    timecode_rate: "29.997377294867".to_string(),
                    timecode_flag: "NDF".to_string(),
                }),
            }),
            chna: None,
        }
    }

    fn description_with_chna(chna: Bw64ChnaMetadata) -> Bw64PcmS16LeDescription {
        Bw64PcmS16LeDescription {
            track: AudioTrackInfo {
                sample_rate_hz: 48_000,
                channels: 2,
                bits_per_sample: 16,
                total_samples: 2,
            },
            ixml: None,
            chna: Some(chna),
        }
    }

    fn description_with_ixml_and_chna() -> Bw64PcmS16LeDescription {
        let mut description = description_with_ixml("Recorded with MotionCam");
        description.chna = Some(Bw64ChnaMetadata::empty_reserved_audio_ids(1024));
        description
    }

    fn find_chunk(bytes: &[u8], id: &[u8; 4]) -> Option<ChunkInfo> {
        if bytes.len() < 12 || &bytes[0..4] != b"BW64" || &bytes[8..12] != b"WAVE" {
            return None;
        }

        let mut offset = 12usize;
        while offset.checked_add(8)? <= bytes.len() {
            let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().ok()?);
            let size = usize::try_from(size).ok()?;
            let payload_offset = offset + 8;
            let payload_end = payload_offset.checked_add(size)?;
            let end_offset = payload_end.checked_add(size % 2)?;

            if end_offset > bytes.len() {
                return None;
            }

            if &bytes[offset..offset + 4] == id {
                return Some(ChunkInfo {
                    offset,
                    payload_offset,
                    size,
                    end_offset,
                });
            }

            offset = end_offset;
        }

        None
    }
}
