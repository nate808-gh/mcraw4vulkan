use mcraw4vulkan_core::{DecodedBayerU16Frame, FrameRate};
use mcraw4vulkan_mcrawcontainer::ColorMatrix;

use crate::error::DngWriterError;
use crate::plan::DngWritePlan;
use crate::{DngCompression, DngFrameDescription, DngPhotometricInterpretation, DngSinkFrame};

// TIFF baseline tag numbers used by the first uncompressed DNG writer.
//
// These are ordinary TIFF tags used to describe the image size, storage layout,
// compression mode, strip data, and sample layout.
const TAG_NEW_SUBFILE_TYPE: u16 = 254;
const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_PHOTOMETRIC_INTERPRETATION: u16 = 262;
const TAG_MAKE: u16 = 271;
const TAG_MODEL: u16 = 272;
const TAG_STRIP_OFFSETS: u16 = 273;
const TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TAG_ROWS_PER_STRIP: u16 = 278;
const TAG_STRIP_BYTE_COUNTS: u16 = 279;
const TAG_PLANAR_CONFIGURATION: u16 = 284;

// TIFF/EP CFA tags used by DNG for Bayer pattern description.
const TAG_CFA_REPEAT_PATTERN_DIM: u16 = 33421;
const TAG_CFA_PATTERN: u16 = 33422;

// DNG-specific tag numbers needed for a minimal Bayer RAW DNG.
const TAG_DNG_VERSION: u16 = 50706;
const TAG_DNG_BACKWARD_VERSION: u16 = 50707;
const TAG_UNIQUE_CAMERA_MODEL: u16 = 50708;
const TAG_CFA_PLANE_COLOR: u16 = 50710;
const TAG_CFA_LAYOUT: u16 = 50711;
const TAG_BLACK_LEVEL_REPEAT_DIM: u16 = 50713;
const TAG_BLACK_LEVEL: u16 = 50714;
const TAG_WHITE_LEVEL: u16 = 50717;
const TAG_DEFAULT_CROP_ORIGIN: u16 = 50719;
const TAG_DEFAULT_CROP_SIZE: u16 = 50720;
const TAG_COLOR_MATRIX1: u16 = 50721;
const TAG_COLOR_MATRIX2: u16 = 50722;
const TAG_AS_SHOT_NEUTRAL: u16 = 50728;
const TAG_CALIBRATION_ILLUMINANT1: u16 = 50778;
const TAG_CALIBRATION_ILLUMINANT2: u16 = 50779;
const TAG_ACTIVE_AREA: u16 = 50829;
const TAG_FORWARD_MATRIX1: u16 = 50964;
const TAG_FORWARD_MATRIX2: u16 = 50965;
const TAG_FRAME_RATE: u16 = 51044;

// TIFF field type identifiers.
//
// Standard TIFF IFD entries store a tag, type, count, and either an inline value
// or an offset to a larger value block.
const TIFF_TYPE_BYTE: u16 = 1;
const TIFF_TYPE_ASCII: u16 = 2;
const TIFF_TYPE_SHORT: u16 = 3;
const TIFF_TYPE_LONG: u16 = 4;
const TIFF_TYPE_RATIONAL: u16 = 5;
const TIFF_TYPE_SRATIONAL: u16 = 10;

// TIFF/DNG constant values for the first writer implementation.
const TIFF_MAGIC: u16 = 42;
const TIFF_FIRST_IFD_OFFSET: u32 = 8;
const COMPRESSION_NONE: u16 = 1;
const PHOTOMETRIC_CFA: u16 = 32803;
const PLANAR_CONFIG_CONTIGUOUS: u16 = 1;
const CFA_LAYOUT_RECTANGULAR: u16 = 1;

// Rational conversion scale for floating-point metadata.
//
// DNG stores many calibration values as RATIONAL or SRATIONAL pairs. A denominator
// of one million preserves enough precision for the current metadata while still
// fitting comfortably in 32-bit TIFF rational fields.
const RATIONAL_SCALE: i64 = 1_000_000;

// Configuration for decisions that affect deterministic DNG bytes.
//
// The optional frame rate is serialized as an exact CinemaDNG signed rational;
// planning and writing otherwise share the same fixed uncompressed layout.
#[derive(Debug, Clone, Copy)]
pub struct DngWriterConfig {
    pub validate_pixel_count: bool,
    pub frame_rate: Option<FrameRate>,
}

impl Default for DngWriterConfig {
    fn default() -> Self {
        Self {
            validate_pixel_count: true,
            frame_rate: None,
        }
    }
}

// Stateless DNG writer facade.
//
// Frame-writing methods compute the complete uncompressed layout before building
// bytes. Callers own the returned buffer and any caching or file lifecycle.
#[derive(Debug, Clone, Copy)]
pub struct DngWriter {
    config: DngWriterConfig,
}

impl DngWriter {
    pub fn new(config: DngWriterConfig) -> Self {
        Self { config }
    }

    // Validate one frame description and return a write plan without pixels.
    //
    // This is used by the FUSE getattr/stat path to report stable DNG file sizes
    // without decoding raw payloads or allocating full DNG byte buffers.
    pub fn plan_uncompressed_frame_from_description(
        &self,
        description: &DngFrameDescription,
    ) -> Result<DngWritePlan, DngWriterError> {
        let _validate_pixel_count = self.config.validate_pixel_count;
        validate_supported_description(description)?;
        DngWritePlan::from_description(description)
    }

    // Return the final byte length of the uncompressed DNG that would be written
    // for this frame description.
    //
    // This follows the same TIFF/DNG layout calculation as full frame writing,
    // but does not need decoded pixels.
    pub fn uncompressed_frame_byte_len_from_description(
        &self,
        description: &DngFrameDescription,
    ) -> Result<u64, DngWriterError> {
        validate_supported_description(description)?;

        let plan = self.plan_uncompressed_frame_from_description(description)?;
        let mut entries = build_ifd_entries(description, &plan, self.config.frame_rate)?;
        let byte_len = calculate_tiff_byte_len(&mut entries, plan.pixel_byte_count)?;

        Ok(u64::from(byte_len))
    }

    // Build a complete uncompressed single-frame DNG from low-level
    // already-little-endian 16-bit Bayer pixel bytes.
    //
    // This is the byte-oriented implementation used by the shared decoded-frame
    // writer and by compatibility callers that already have exact or
    // prefix-padded little-endian pixel bytes.
    //
    // The input may be larger than the exact frame payload because packed GPU
    // output is u32-word aligned and odd pixel counts can include one padding
    // sample. Only the exact DNG strip byte count is written.
    pub fn write_uncompressed_frame_from_le_pixel_bytes_to_vec(
        &self,
        description: &DngFrameDescription,
        pixel_bytes_le: &[u8],
    ) -> Result<Vec<u8>, DngWriterError> {
        validate_supported_description(description)?;

        let plan = self.plan_uncompressed_frame_from_description(description)?;

        if pixel_bytes_le.len() < plan.pixel_byte_count {
            return Err(DngWriterError::InvalidMetadata(format!(
                "little-endian pixel byte buffer is too small: expected at least {}, got {}",
                plan.pixel_byte_count,
                pixel_bytes_le.len()
            )));
        }

        let exact_pixel_bytes = &pixel_bytes_le[..plan.pixel_byte_count];
        let mut entries = build_ifd_entries(description, &plan, self.config.frame_rate)?;

        build_tiff_bytes_from_le_pixel_bytes(
            &mut entries,
            exact_pixel_bytes,
            plan.pixel_byte_count,
            description.dng_sample_left_shift,
        )
    }

    // Build a complete uncompressed single-frame DNG from the shared decoded
    // Bayer U16 frame contract.
    //
    // This is the preferred API for new CPU, export, and FUSE DNG generation
    // paths. The frame owns or borrows exact tightly-packed little-endian Bayer
    // U16 bytes and carries its dimensions with the payload.
    pub fn write_uncompressed_frame_from_decoded_bayer_u16_frame_to_vec(
        &self,
        description: &DngFrameDescription,
        frame: &DecodedBayerU16Frame<'_>,
    ) -> Result<Vec<u8>, DngWriterError> {
        let frame_dimensions = frame.dimensions();

        if frame_dimensions != description.dimensions {
            return Err(DngWriterError::InvalidMetadata(format!(
                "decoded Bayer U16 frame dimensions {}x{} do not match DNG description dimensions {}x{}",
                frame_dimensions.width,
                frame_dimensions.height,
                description.dimensions.width,
                description.dimensions.height
            )));
        }

        self.write_uncompressed_frame_from_le_pixel_bytes_to_vec(
            description,
            frame.pixel_bytes_le(),
        )
    }

    // Build a complete uncompressed DNG from the canonical sink-frame contract.
    pub fn write_uncompressed_dng_sink_frame_to_vec(
        &self,
        frame: &DngSinkFrame,
    ) -> Result<Vec<u8>, DngWriterError> {
        self.write_uncompressed_frame_from_decoded_bayer_u16_frame_to_vec(
            frame.description(),
            frame.pixels(),
        )
    }
}

// One TIFF IFD entry before final offset assignment.
//
// Values longer than 4 bytes are written into a separate value area after the IFD.
// Values of 4 bytes or fewer are stored inline in the entry's value/offset field.
#[derive(Debug, Clone)]
struct TiffEntry {
    tag: u16,
    field_type: u16,
    count: u32,
    value_bytes: Vec<u8>,
    assigned_offset: Option<u32>,
}

impl TiffEntry {
    fn new(
        tag: u16,
        field_type: u16,
        count: u32,
        value_bytes: Vec<u8>,
    ) -> Result<Self, DngWriterError> {
        if value_bytes.is_empty() {
            return Err(DngWriterError::InvalidMetadata(format!(
                "tag {tag} has empty value"
            )));
        }

        Ok(Self {
            tag,
            field_type,
            count,
            value_bytes,
            assigned_offset: None,
        })
    }

    fn needs_external_value_block(&self) -> bool {
        self.value_bytes.len() > 4
    }
}

// Validate the DNG description fields supported by the first writer.
//
// This keeps unsupported layouts from silently producing files that look valid
// but are not actually readable as the intended Bayer RAW image.
fn validate_supported_description(description: &DngFrameDescription) -> Result<(), DngWriterError> {
    if description.compression != DngCompression::Uncompressed {
        return Err(DngWriterError::UnsupportedLayout(
            "only uncompressed DNG output is currently supported".to_string(),
        ));
    }

    if description.photometric_interpretation != DngPhotometricInterpretation::Cfa {
        return Err(DngWriterError::UnsupportedLayout(
            "only CFA/Bayer DNG output is currently supported".to_string(),
        ));
    }

    if description.bits_per_sample != 16 {
        return Err(DngWriterError::UnsupportedLayout(format!(
            "bits_per_sample must be 16, got {}",
            description.bits_per_sample
        )));
    }

    if description.samples_per_pixel != 1 {
        return Err(DngWriterError::UnsupportedLayout(format!(
            "samples_per_pixel must be 1, got {}",
            description.samples_per_pixel
        )));
    }

    if description.dng_sample_left_shift >= u16::BITS as u8 {
        return Err(DngWriterError::UnsupportedLayout(format!(
            "dng_sample_left_shift must be less than {}, got {}",
            u16::BITS,
            description.dng_sample_left_shift
        )));
    }

    Ok(())
}

// Build the TIFF/DNG IFD entries for the first uncompressed DNG implementation.
fn build_ifd_entries(
    description: &DngFrameDescription,
    plan: &DngWritePlan,
    frame_rate: Option<FrameRate>,
) -> Result<Vec<TiffEntry>, DngWriterError> {
    let width = plan.width;
    let height = plan.height;
    let strip_byte_count = u32_from_usize(plan.pixel_byte_count, "strip byte count")?;
    let white_level = uniform_level_to_u32(description.white_level.values, "whiteLevel")?;

    let mut entries = vec![
        entry_long(TAG_NEW_SUBFILE_TYPE, &[0])?,
        entry_long(TAG_IMAGE_WIDTH, &[width])?,
        entry_long(TAG_IMAGE_LENGTH, &[height])?,
        entry_short(TAG_BITS_PER_SAMPLE, &[description.bits_per_sample])?,
        entry_short(TAG_COMPRESSION, &[COMPRESSION_NONE])?,
        entry_short(TAG_PHOTOMETRIC_INTERPRETATION, &[PHOTOMETRIC_CFA])?,
        entry_ascii(TAG_MAKE, "MotionCam")?,
        entry_ascii(TAG_MODEL, &description.unique_camera_model)?,
        entry_long(TAG_STRIP_OFFSETS, &[0])?,
        entry_short(TAG_SAMPLES_PER_PIXEL, &[description.samples_per_pixel])?,
        entry_long(TAG_ROWS_PER_STRIP, &[height])?,
        entry_long(TAG_STRIP_BYTE_COUNTS, &[strip_byte_count])?,
        entry_short(TAG_PLANAR_CONFIGURATION, &[PLANAR_CONFIG_CONTIGUOUS])?,
        entry_short(
            TAG_CFA_REPEAT_PATTERN_DIM,
            &description.cfa_pattern.repeat_pattern_dim,
        )?,
        entry_byte(TAG_CFA_PATTERN, &description.cfa_pattern.pattern)?,
        entry_byte(TAG_DNG_VERSION, &description.dng_version)?,
        entry_byte(TAG_DNG_BACKWARD_VERSION, &description.dng_backward_version)?,
        entry_ascii(TAG_UNIQUE_CAMERA_MODEL, &description.unique_camera_model)?,
        entry_byte(
            TAG_CFA_PLANE_COLOR,
            &description.cfa_pattern.cfa_plane_color,
        )?,
        entry_short(TAG_CFA_LAYOUT, &[CFA_LAYOUT_RECTANGULAR])?,
        entry_short(TAG_BLACK_LEVEL_REPEAT_DIM, &[2, 2])?,
        entry_rational_f64(TAG_BLACK_LEVEL, &description.black_level.values)?,
        entry_long(TAG_WHITE_LEVEL, &[white_level])?,
        entry_long(TAG_DEFAULT_CROP_ORIGIN, &description.default_crop_origin)?,
        entry_long(TAG_DEFAULT_CROP_SIZE, &description.default_crop_size)?,
        entry_short(
            TAG_CALIBRATION_ILLUMINANT1,
            &[description.calibration_illuminant1],
        )?,
        entry_short(
            TAG_CALIBRATION_ILLUMINANT2,
            &[description.calibration_illuminant2],
        )?,
        entry_long(TAG_ACTIVE_AREA, &[0, 0, height, width])?,
    ];

    if let Some(matrix) = description.camera_calibration1 {
        entries.push(entry_srational_f64(50723, &matrix.values)?);
    }
    if let Some(matrix) = description.camera_calibration2 {
        entries.push(entry_srational_f64(50724, &matrix.values)?);
    }
    if let Some(balance) = description.analog_balance {
        entries.push(entry_rational_f64(50727, &balance)?);
    }

    if let Some(as_shot_neutral) = description.as_shot_neutral {
        entries.push(entry_rational_f64(TAG_AS_SHOT_NEUTRAL, &as_shot_neutral)?);
    }

    if let Some(frame_rate) = frame_rate {
        entries.push(entry_frame_rate(frame_rate)?);
    }

    push_optional_matrix(&mut entries, TAG_COLOR_MATRIX1, description.color_matrix1)?;
    push_optional_matrix(&mut entries, TAG_COLOR_MATRIX2, description.color_matrix2)?;
    push_optional_matrix(
        &mut entries,
        TAG_FORWARD_MATRIX1,
        description.forward_matrix1,
    )?;
    push_optional_matrix(
        &mut entries,
        TAG_FORWARD_MATRIX2,
        description.forward_matrix2,
    )?;

    entries.sort_by_key(|entry| entry.tag);

    Ok(entries)
}

// Add the CinemaDNG/DNG video frame-rate tag as an exact signed rational.
fn entry_frame_rate(frame_rate: FrameRate) -> Result<TiffEntry, DngWriterError> {
    let (numerator, denominator) = frame_rate_to_srational(frame_rate)?;
    entry_srational_i32(TAG_FRAME_RATE, &[(numerator, denominator)])
}

// Add an optional signed rational 3x3 matrix to the IFD entry list.
fn push_optional_matrix(
    entries: &mut Vec<TiffEntry>,
    tag: u16,
    matrix: Option<ColorMatrix>,
) -> Result<(), DngWriterError> {
    if let Some(matrix) = matrix {
        entries.push(entry_srational_f64(tag, &matrix.values)?);
    }

    Ok(())
}

// Calculate the final little-endian TIFF/DNG byte length without writing pixels.
//
// This mirrors the layout math used by build_tiff_bytes(). It assigns external
// value-block offsets and the StripOffsets value so the computed size matches
// the real DNG writer exactly.
fn calculate_tiff_byte_len(
    entries: &mut [TiffEntry],
    pixel_byte_count: usize,
) -> Result<u32, DngWriterError> {
    let _entry_count = u16::try_from(entries.len())
        .map_err(|_| DngWriterError::UnsupportedLayout("too many TIFF IFD entries".to_string()))?;

    let ifd_bytes = 2usize
        .checked_add(
            entries
                .len()
                .checked_mul(12)
                .ok_or(DngWriterError::TiffSizeOverflow)?,
        )
        .and_then(|value| value.checked_add(4))
        .ok_or(DngWriterError::TiffSizeOverflow)?;

    let mut data_cursor = TIFF_FIRST_IFD_OFFSET
        .checked_add(u32_from_usize(ifd_bytes, "IFD byte count")?)
        .ok_or(DngWriterError::TiffSizeOverflow)?;

    for entry in entries.iter_mut() {
        if entry.needs_external_value_block() {
            data_cursor = align_to_even(data_cursor);
            entry.assigned_offset = Some(data_cursor);

            let value_len = u32_from_usize(entry.value_bytes.len(), "IFD value block length")?;
            data_cursor = data_cursor
                .checked_add(value_len)
                .ok_or(DngWriterError::TiffSizeOverflow)?;
        }
    }

    let pixel_offset = align_to_even(data_cursor);
    let pixel_len = u32_from_usize(pixel_byte_count, "pixel byte count")?;
    let final_size = pixel_offset
        .checked_add(pixel_len)
        .ok_or(DngWriterError::TiffSizeOverflow)?;

    set_strip_offset(entries, pixel_offset)?;

    Ok(final_size)
}

// Build a full little-endian standard TIFF/DNG byte buffer from pixel bytes that
// are already in the final little-endian 16-bit sample representation.
fn build_tiff_bytes_from_le_pixel_bytes(
    entries: &mut [TiffEntry],
    pixel_bytes_le: &[u8],
    pixel_byte_count: usize,
    dng_sample_left_shift: u8,
) -> Result<Vec<u8>, DngWriterError> {
    if pixel_bytes_le.len() != pixel_byte_count {
        return Err(DngWriterError::InvalidMetadata(format!(
            "little-endian pixel byte count mismatch: expected {}, got {}",
            pixel_byte_count,
            pixel_bytes_le.len()
        )));
    }

    let entry_count = u16::try_from(entries.len())
        .map_err(|_| DngWriterError::UnsupportedLayout("too many TIFF IFD entries".to_string()))?;

    let ifd_bytes = 2usize
        .checked_add(
            entries
                .len()
                .checked_mul(12)
                .ok_or(DngWriterError::TiffSizeOverflow)?,
        )
        .and_then(|value| value.checked_add(4))
        .ok_or(DngWriterError::TiffSizeOverflow)?;

    let mut data_cursor = TIFF_FIRST_IFD_OFFSET
        .checked_add(u32_from_usize(ifd_bytes, "IFD byte count")?)
        .ok_or(DngWriterError::TiffSizeOverflow)?;

    for entry in entries.iter_mut() {
        if entry.needs_external_value_block() {
            data_cursor = align_to_even(data_cursor);
            entry.assigned_offset = Some(data_cursor);

            let value_len = u32_from_usize(entry.value_bytes.len(), "IFD value block length")?;
            data_cursor = data_cursor
                .checked_add(value_len)
                .ok_or(DngWriterError::TiffSizeOverflow)?;
        }
    }

    let pixel_offset = align_to_even(data_cursor);
    let pixel_len = u32_from_usize(pixel_byte_count, "pixel byte count")?;
    let final_size = pixel_offset
        .checked_add(pixel_len)
        .ok_or(DngWriterError::TiffSizeOverflow)?;

    set_strip_offset(entries, pixel_offset)?;

    let final_capacity =
        usize::try_from(final_size).map_err(|_| DngWriterError::TiffSizeOverflow)?;

    let mut out = Vec::with_capacity(final_capacity);

    // TIFF header: little-endian marker, magic number, first IFD offset.
    out.extend_from_slice(b"II");
    out.extend_from_slice(&TIFF_MAGIC.to_le_bytes());
    out.extend_from_slice(&TIFF_FIRST_IFD_OFFSET.to_le_bytes());

    out.extend_from_slice(&entry_count.to_le_bytes());

    for entry in entries.iter() {
        write_ifd_entry(&mut out, entry)?;
    }

    // Zero means there is no next IFD.
    out.extend_from_slice(&0u32.to_le_bytes());

    for entry in entries.iter() {
        if let Some(offset) = entry.assigned_offset {
            pad_to_offset(&mut out, offset)?;
            out.extend_from_slice(&entry.value_bytes);

            if out.len() % 2 != 0 {
                out.push(0);
            }
        }
    }

    pad_to_offset(&mut out, pixel_offset)?;
    append_le_pixel_bytes(&mut out, pixel_bytes_le, dng_sample_left_shift);

    Ok(out)
}

// Write one 12-byte TIFF IFD entry.
fn write_ifd_entry(out: &mut Vec<u8>, entry: &TiffEntry) -> Result<(), DngWriterError> {
    out.extend_from_slice(&entry.tag.to_le_bytes());
    out.extend_from_slice(&entry.field_type.to_le_bytes());
    out.extend_from_slice(&entry.count.to_le_bytes());

    if let Some(offset) = entry.assigned_offset {
        out.extend_from_slice(&offset.to_le_bytes());
    } else {
        let mut inline = [0u8; 4];

        for (index, value) in entry.value_bytes.iter().enumerate() {
            inline[index] = *value;
        }

        out.extend_from_slice(&inline);
    }

    Ok(())
}

// Update the StripOffsets tag once the pixel data offset is known.
fn set_strip_offset(entries: &mut [TiffEntry], pixel_offset: u32) -> Result<(), DngWriterError> {
    let Some(entry) = entries
        .iter_mut()
        .find(|entry| entry.tag == TAG_STRIP_OFFSETS)
    else {
        return Err(DngWriterError::InvalidMetadata(
            "missing StripOffsets tag".to_string(),
        ));
    };

    entry.value_bytes = pixel_offset.to_le_bytes().to_vec();

    Ok(())
}

// Append already-little-endian u16 samples, optionally expanding their DNG output
// code range without changing the decoded-frame contract or allocating a second
// full-frame buffer.
fn append_le_pixel_bytes(out: &mut Vec<u8>, pixel_bytes_le: &[u8], dng_sample_left_shift: u8) {
    if dng_sample_left_shift == 0 {
        out.extend_from_slice(pixel_bytes_le);
        return;
    }

    const CHUNK_PIXELS: usize = 8192;
    const BYTES_PER_PIXEL: usize = std::mem::size_of::<u16>();
    const CHUNK_BYTES: usize = CHUNK_PIXELS * BYTES_PER_PIXEL;

    debug_assert_eq!(pixel_bytes_le.len() % BYTES_PER_PIXEL, 0);

    let mut scratch = [0u8; CHUNK_BYTES];
    let mut byte_index = 0usize;

    while byte_index < pixel_bytes_le.len() {
        let remaining_bytes = pixel_bytes_le.len() - byte_index;
        let pixels_this_chunk = (remaining_bytes / BYTES_PER_PIXEL).min(CHUNK_PIXELS);
        let bytes_this_chunk = pixels_this_chunk * BYTES_PER_PIXEL;
        let input_chunk = &pixel_bytes_le[byte_index..byte_index + bytes_this_chunk];

        for (chunk_index, sample_bytes) in input_chunk.chunks_exact(BYTES_PER_PIXEL).enumerate() {
            let sample = u16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
            let sample = transform_sample_for_dng(sample, dng_sample_left_shift);
            let [lo, hi] = sample.to_le_bytes();
            let output_byte_index = chunk_index * BYTES_PER_PIXEL;

            scratch[output_byte_index] = lo;
            scratch[output_byte_index + 1] = hi;
        }

        out.extend_from_slice(&scratch[..bytes_this_chunk]);
        byte_index += bytes_this_chunk;
    }
}

fn transform_sample_for_dng(sample: u16, dng_sample_left_shift: u8) -> u16 {
    if dng_sample_left_shift == 0 {
        return sample;
    }

    let shifted = u32::from(sample)
        .checked_shl(u32::from(dng_sample_left_shift))
        .unwrap_or(u32::MAX);

    shifted.min(u32::from(u16::MAX)) as u16
}

// Create a BYTE-valued TIFF entry.
fn entry_byte(tag: u16, values: &[u8]) -> Result<TiffEntry, DngWriterError> {
    TiffEntry::new(
        tag,
        TIFF_TYPE_BYTE,
        u32_from_usize(values.len(), "BYTE count")?,
        values.to_vec(),
    )
}

// Create a null-terminated ASCII TIFF entry.
fn entry_ascii(tag: u16, value: &str) -> Result<TiffEntry, DngWriterError> {
    let mut bytes = value.as_bytes().to_vec();
    bytes.push(0);

    TiffEntry::new(
        tag,
        TIFF_TYPE_ASCII,
        u32_from_usize(bytes.len(), "ASCII count")?,
        bytes,
    )
}

// Create a SHORT-valued TIFF entry.
fn entry_short(tag: u16, values: &[u16]) -> Result<TiffEntry, DngWriterError> {
    let mut bytes = Vec::with_capacity(values.len() * 2);

    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    TiffEntry::new(
        tag,
        TIFF_TYPE_SHORT,
        u32_from_usize(values.len(), "SHORT count")?,
        bytes,
    )
}

// Create a LONG-valued TIFF entry.
fn entry_long(tag: u16, values: &[u32]) -> Result<TiffEntry, DngWriterError> {
    let mut bytes = Vec::with_capacity(values.len() * 4);

    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    TiffEntry::new(
        tag,
        TIFF_TYPE_LONG,
        u32_from_usize(values.len(), "LONG count")?,
        bytes,
    )
}

// Create an unsigned RATIONAL-valued TIFF entry from f64 values.
fn entry_rational_f64(tag: u16, values: &[f64]) -> Result<TiffEntry, DngWriterError> {
    let mut bytes = Vec::with_capacity(values.len() * 8);

    for value in values {
        let (numerator, denominator) = f64_to_rational_u32(*value)?;
        bytes.extend_from_slice(&numerator.to_le_bytes());
        bytes.extend_from_slice(&denominator.to_le_bytes());
    }

    TiffEntry::new(
        tag,
        TIFF_TYPE_RATIONAL,
        u32_from_usize(values.len(), "RATIONAL count")?,
        bytes,
    )
}

// Create a signed SRATIONAL-valued TIFF entry from f64 values.
fn entry_srational_f64(tag: u16, values: &[f64]) -> Result<TiffEntry, DngWriterError> {
    let mut bytes = Vec::with_capacity(values.len() * 8);

    for value in values {
        let (numerator, denominator) = f64_to_srational_i32(*value)?;
        bytes.extend_from_slice(&numerator.to_le_bytes());
        bytes.extend_from_slice(&denominator.to_le_bytes());
    }

    TiffEntry::new(
        tag,
        TIFF_TYPE_SRATIONAL,
        u32_from_usize(values.len(), "SRATIONAL count")?,
        bytes,
    )
}

// Create a signed SRATIONAL-valued TIFF entry from exact i32 numerator/denominator pairs.
fn entry_srational_i32(tag: u16, values: &[(i32, i32)]) -> Result<TiffEntry, DngWriterError> {
    let mut bytes = Vec::with_capacity(values.len() * 8);

    for &(numerator, denominator) in values {
        if denominator <= 0 {
            return Err(DngWriterError::InvalidMetadata(
                "SRATIONAL denominator must be positive".to_string(),
            ));
        }

        bytes.extend_from_slice(&numerator.to_le_bytes());
        bytes.extend_from_slice(&denominator.to_le_bytes());
    }

    TiffEntry::new(
        tag,
        TIFF_TYPE_SRATIONAL,
        u32_from_usize(values.len(), "SRATIONAL count")?,
        bytes,
    )
}

// Convert the source clip frame-rate rational to the SRATIONAL representation.
fn frame_rate_to_srational(frame_rate: FrameRate) -> Result<(i32, i32), DngWriterError> {
    if !frame_rate.is_valid() {
        return Err(DngWriterError::InvalidMetadata(
            "FrameRate denominator must be positive".to_string(),
        ));
    }

    let frame_rate = frame_rate.reduced();
    let numerator = i32::try_from(frame_rate.numerator).map_err(|_| {
        DngWriterError::InvalidMetadata("FrameRate numerator overflows i32".to_string())
    })?;
    let denominator = i32::try_from(frame_rate.denominator).map_err(|_| {
        DngWriterError::InvalidMetadata("FrameRate denominator overflows i32".to_string())
    })?;

    if denominator <= 0 {
        return Err(DngWriterError::InvalidMetadata(
            "FrameRate denominator must be positive".to_string(),
        ));
    }

    Ok((numerator, denominator))
}

// Convert a uniform level array into the single integer value used by WhiteLevel.
//
// The metadata model allows four values because it mirrors blackLevel. For DNG
// WhiteLevel in this first writer, all values must match.
fn uniform_level_to_u32(values: [f64; 4], label: &str) -> Result<u32, DngWriterError> {
    let first = values[0];

    for value in values {
        if (value - first).abs() > f64::EPSILON {
            return Err(DngWriterError::UnsupportedLayout(format!(
                "{label} must be uniform for the first DNG writer"
            )));
        }
    }

    f64_to_exact_u32(first, label)
}

// Convert a non-negative whole-number f64 into u32.
fn f64_to_exact_u32(value: f64, label: &str) -> Result<u32, DngWriterError> {
    if !value.is_finite() {
        return Err(DngWriterError::InvalidMetadata(format!(
            "{label} must be finite"
        )));
    }

    if value < 0.0 {
        return Err(DngWriterError::InvalidMetadata(format!(
            "{label} must not be negative"
        )));
    }

    if value.fract() != 0.0 {
        return Err(DngWriterError::InvalidMetadata(format!(
            "{label} must be an integer-like value"
        )));
    }

    if value > f64::from(u32::MAX) {
        return Err(DngWriterError::InvalidMetadata(format!(
            "{label} overflows u32"
        )));
    }

    Ok(value as u32)
}

// Convert a non-negative f64 to TIFF RATIONAL.
fn f64_to_rational_u32(value: f64) -> Result<(u32, u32), DngWriterError> {
    if !value.is_finite() {
        return Err(DngWriterError::InvalidMetadata(
            "RATIONAL value must be finite".to_string(),
        ));
    }

    if value < 0.0 {
        return Err(DngWriterError::InvalidMetadata(
            "RATIONAL value must not be negative".to_string(),
        ));
    }

    if value.fract() == 0.0 && value <= f64::from(u32::MAX) {
        return Ok((value as u32, 1));
    }

    let numerator = (value * RATIONAL_SCALE as f64).round();

    if numerator < 0.0 || numerator > u32::MAX as f64 {
        return Err(DngWriterError::InvalidMetadata(
            "RATIONAL numerator overflows u32".to_string(),
        ));
    }

    let numerator = numerator as u32;
    let denominator = RATIONAL_SCALE as u32;
    let divisor = gcd_u32(numerator, denominator);

    Ok((numerator / divisor, denominator / divisor))
}

// Convert an f64 to TIFF SRATIONAL.
fn f64_to_srational_i32(value: f64) -> Result<(i32, i32), DngWriterError> {
    if !value.is_finite() {
        return Err(DngWriterError::InvalidMetadata(
            "SRATIONAL value must be finite".to_string(),
        ));
    }

    if value.fract() == 0.0 && value >= f64::from(i32::MIN) && value <= f64::from(i32::MAX) {
        return Ok((value as i32, 1));
    }

    let numerator = (value * RATIONAL_SCALE as f64).round();

    if numerator < f64::from(i32::MIN) || numerator > f64::from(i32::MAX) {
        return Err(DngWriterError::InvalidMetadata(
            "SRATIONAL numerator overflows i32".to_string(),
        ));
    }

    let numerator = numerator as i32;
    let denominator = RATIONAL_SCALE as i32;
    let divisor = gcd_u32(numerator.unsigned_abs(), denominator as u32) as i32;

    Ok((numerator / divisor, denominator / divisor))
}

// Greatest common divisor for rational simplification.
fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }

    a.max(1)
}

// Convert usize to u32 for standard TIFF offsets/counts.
fn u32_from_usize(value: usize, label: &str) -> Result<u32, DngWriterError> {
    u32::try_from(value).map_err(|_| {
        DngWriterError::InvalidMetadata(format!("{label} does not fit in standard TIFF u32"))
    })
}

// Align a TIFF value offset to an even byte boundary.
fn align_to_even(value: u32) -> u32 {
    if value.is_multiple_of(2) {
        value
    } else {
        value + 1
    }
}

// Pad the output buffer until it reaches the requested absolute byte offset.
fn pad_to_offset(out: &mut Vec<u8>, offset: u32) -> Result<(), DngWriterError> {
    let offset = usize::try_from(offset).map_err(|_| DngWriterError::TiffSizeOverflow)?;

    if out.len() > offset {
        return Err(DngWriterError::TiffSizeOverflow);
    }

    while out.len() < offset {
        out.push(0);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use mcraw4vulkan_core::{
        BayerPattern, DecodedBayerU16Frame, FrameDimensions, FrameNumber, FrameRate,
    };
    use mcraw4vulkan_mcrawcontainer::{BlackLevel, WhiteLevel};

    use super::{
        DngWriter, DngWriterConfig, TAG_BLACK_LEVEL, TAG_DEFAULT_CROP_ORIGIN,
        TAG_DEFAULT_CROP_SIZE, TAG_FRAME_RATE, TAG_WHITE_LEVEL, TIFF_TYPE_LONG, TIFF_TYPE_RATIONAL,
        TIFF_TYPE_SRATIONAL, frame_rate_to_srational,
    };
    use crate::{
        CfaPattern, DngCompression, DngFrameDescription, DngFrameDescriptionOverrides,
        DngPhotometricInterpretation,
    };

    #[test]
    fn source_calibration_tags_preserve_frame_slots_and_optional_presence() {
        use mcraw4vulkan_mcrawcontainer::{ContainerMetadata, FrameMetadata};
        let container = ContainerMetadata::parse(
            r#"{
            "sensorArrangment":"rggb", "blackLevel":16, "whiteLevel":4095,
            "colorIlluminant1":"standarda", "colorIlluminant2":"d50",
            "colorMatrix1":[1,0,0,0,1,0,0,0,1],
            "colorMatrix2":[2,0,0,0,2,0,0,0,2],
            "forwardMatrix1":[], "forwardMatrix2":[],
            "calibrationMatrix1":[], "calibrationMatrix2":[]
        }"#,
        )
        .unwrap();
        let frame = FrameMetadata::parse(
            r#"{
            "width":2, "height":2, "asShotNeutral":[0.5,1,0.75],
            "colorIlluminant1":"d50", "colorIlluminant2":"standarda",
            "colorMatrix1":[3,0,0,0,3,0,0,0,3],
            "calibrationMatrix1":[1.1,-0.1,0,0,1,0,0,0,0.9],
            "analogBalance":[1.25,1,0.8]
        }"#,
        )
        .unwrap();
        let description =
            DngFrameDescription::from_metadata(&container, &frame, FrameNumber(0), 0).unwrap();
        assert_eq!(description.calibration_illuminant1, 23);
        assert_eq!(description.calibration_illuminant2, 17);
        assert_eq!(description.color_matrix1.unwrap().values[0], 3.0);
        assert_eq!(description.color_matrix2.unwrap().values[0], 2.0);
        assert!(description.forward_matrix1.is_none());
        assert!(description.forward_matrix2.is_none());
        assert!(description.camera_calibration2.is_none());
        let writer = DngWriter::new(DngWriterConfig::default());
        let bytes = write_samples_to_vec(&writer, &description, &[0, 16, 4095, 4096]).unwrap();
        assert_eq!(
            srational_tag_values(&bytes, 50723),
            vec![
                (11, 10),
                (-1, 10),
                (0, 1),
                (0, 1),
                (1, 1),
                (0, 1),
                (0, 1),
                (0, 1),
                (9, 10)
            ]
        );
        assert_eq!(
            rational_tag_values(&bytes, 50727),
            vec![(5, 4), (1, 1), (4, 5)]
        );
        let count = usize::from(read_u16(&bytes, 8));
        let tags: Vec<_> = (0..count).map(|i| read_u16(&bytes, 10 + i * 12)).collect();
        assert!(!tags.contains(&50724));
        assert!(!tags.contains(&50964));
        assert!(!tags.contains(&50965));
    }

    #[test]
    fn le_byte_path_without_sample_shift_remains_byte_for_byte_identical() {
        let samples = [0_u16, 1, 1023, 4095];
        let pixel_bytes = le_bytes(&samples);
        let description = description_for_samples(samples.len(), 0);
        let writer = DngWriter::new(DngWriterConfig::default());

        let from_bytes = writer
            .write_uncompressed_frame_from_le_pixel_bytes_to_vec(&description, &pixel_bytes)
            .expect("LE byte DNG write should succeed");
        let from_decoded_frame = write_samples_to_vec(&writer, &description, &samples)
            .expect("decoded-frame DNG write should succeed");

        assert_eq!(from_bytes, from_decoded_frame);
        assert!(from_bytes.ends_with(&pixel_bytes));
    }

    #[test]
    fn dng_sample_shift_expands_all_uncompressed_writer_paths() {
        let samples = [0_u16, 1, 1023, 4095];
        let pixel_bytes = le_bytes(&samples);
        let expected_pixel_bytes = le_bytes(&[0_u16, 4, 4092, 16380]);
        let description = description_for_samples(samples.len(), 2);
        let writer = DngWriter::new(DngWriterConfig::default());

        let from_bytes = writer
            .write_uncompressed_frame_from_le_pixel_bytes_to_vec(&description, &pixel_bytes)
            .expect("LE byte DNG write should succeed");
        let decoded_frame =
            DecodedBayerU16Frame::from_borrowed_le_bytes(description.dimensions, &pixel_bytes)
                .expect("test frame bytes should match dimensions");
        let from_decoded_frame = writer
            .write_uncompressed_frame_from_decoded_bayer_u16_frame_to_vec(
                &description,
                &decoded_frame,
            )
            .expect("decoded-frame DNG write should succeed");

        assert_eq!(from_decoded_frame, from_bytes);
        assert!(from_bytes.ends_with(&expected_pixel_bytes));
    }

    #[test]
    fn writer_emits_original_black_level_without_override() {
        let samples = [0_u16, 1, 1023, 4095];
        let description = description_for_samples(samples.len(), 0);
        let writer = DngWriter::new(DngWriterConfig::default());

        let dng_bytes = write_samples_to_vec(&writer, &description, &samples)
            .expect("DNG write should succeed");

        assert_eq!(
            rational_tag_values(&dng_bytes, TAG_BLACK_LEVEL),
            vec![(256, 1); 4]
        );
    }

    #[test]
    fn writer_emits_overridden_black_level() {
        let samples = [0_u16, 1, 1023, 4095];
        let description = description_for_samples(samples.len(), 2);
        let overridden =
            description.with_output_overrides(DngFrameDescriptionOverrides::black_level_zero());
        let writer = DngWriter::new(DngWriterConfig::default());

        let dng_bytes = write_samples_to_vec(&writer, &overridden, &samples)
            .expect("DNG write with overridden black level should succeed");

        assert_eq!(
            rational_tag_values(&dng_bytes, TAG_BLACK_LEVEL),
            vec![(0, 1); 4]
        );
        assert_eq!(
            long_tag_values(&dng_bytes, TAG_DEFAULT_CROP_ORIGIN),
            vec![0, 0]
        );
        assert_eq!(
            long_tag_values(&dng_bytes, TAG_DEFAULT_CROP_SIZE),
            vec![4, 1]
        );
    }

    #[test]
    fn writer_emits_default_crop_tags() {
        let samples = [0_u16, 1, 1023, 4095];
        let description = description_for_samples(samples.len(), 0);
        let writer = DngWriter::new(DngWriterConfig::default());

        let dng_bytes = write_samples_to_vec(&writer, &description, &samples)
            .expect("DNG write should succeed");

        assert_eq!(
            long_tag_values(&dng_bytes, TAG_DEFAULT_CROP_ORIGIN),
            vec![0, 0]
        );
        assert_eq!(
            long_tag_values(&dng_bytes, TAG_DEFAULT_CROP_SIZE),
            vec![4, 1]
        );
    }

    #[test]
    fn writer_emits_source_frame_rate_as_srational() {
        let samples = [0_u16, 1, 1023, 4095];
        let description = description_for_samples(samples.len(), 0);
        let writer = DngWriter::new(DngWriterConfig {
            frame_rate: Some(FrameRate::new(62_500_000, 2_606_379)),
            ..DngWriterConfig::default()
        });

        let dng_bytes = write_samples_to_vec(&writer, &description, &samples)
            .expect("DNG write should succeed");

        assert_eq!(
            srational_tag_values(&dng_bytes, TAG_FRAME_RATE),
            vec![(62_500_000, 2_606_379)]
        );
    }

    #[test]
    fn frame_rate_srational_encoding_reduces_before_i32_validation() {
        assert_eq!(
            frame_rate_to_srational(FrameRate::new(120_000, 5_000))
                .expect("reducible frame rate should fit SRATIONAL"),
            (24, 1)
        );

        let overflowing_numerator = u64::try_from(i32::MAX).expect("i32::MAX should fit u64") + 1;
        assert!(frame_rate_to_srational(FrameRate::new(overflowing_numerator, 1)).is_err());
    }

    #[test]
    fn writer_black_override_preserves_white_level_and_sample_shift() {
        let samples = [0_u16, 1, 1023, 4095];
        let expected_shifted_pixel_bytes = le_bytes(&[0_u16, 4, 4092, 16380]);
        let description = description_for_samples(samples.len(), 2);
        let overridden =
            description.with_output_overrides(DngFrameDescriptionOverrides::black_level_zero());
        let writer = DngWriter::new(DngWriterConfig::default());

        let original_bytes = write_samples_to_vec(&writer, &description, &samples)
            .expect("DNG write without override should succeed");
        let overridden_bytes = write_samples_to_vec(&writer, &overridden, &samples)
            .expect("DNG write with black override should succeed");

        assert_eq!(
            long_tag_values(&overridden_bytes, TAG_WHITE_LEVEL),
            long_tag_values(&original_bytes, TAG_WHITE_LEVEL)
        );
        assert_eq!(
            long_tag_values(&overridden_bytes, TAG_DEFAULT_CROP_ORIGIN),
            long_tag_values(&original_bytes, TAG_DEFAULT_CROP_ORIGIN)
        );
        assert_eq!(
            long_tag_values(&overridden_bytes, TAG_DEFAULT_CROP_SIZE),
            long_tag_values(&original_bytes, TAG_DEFAULT_CROP_SIZE)
        );
        assert!(original_bytes.ends_with(&expected_shifted_pixel_bytes));
        assert!(overridden_bytes.ends_with(&expected_shifted_pixel_bytes));
    }

    #[test]
    fn crop_tags_preserve_range_metadata_and_pixel_bytes() {
        let samples = [0_u16, 1, 1023, 4095];
        let expected_shifted_pixel_bytes = le_bytes(&[0_u16, 4, 4092, 16380]);
        let description = description_for_samples(samples.len(), 2);
        let writer = DngWriter::new(DngWriterConfig::default());

        let dng_bytes = write_samples_to_vec(&writer, &description, &samples)
            .expect("DNG write should succeed");

        assert_eq!(
            long_tag_values(&dng_bytes, TAG_DEFAULT_CROP_ORIGIN),
            description.default_crop_origin.to_vec()
        );
        assert_eq!(
            long_tag_values(&dng_bytes, TAG_DEFAULT_CROP_SIZE),
            description.default_crop_size.to_vec()
        );
        assert_eq!(long_tag_values(&dng_bytes, TAG_WHITE_LEVEL), vec![4095]);
        assert_eq!(
            rational_tag_values(&dng_bytes, TAG_BLACK_LEVEL),
            vec![(256, 1); 4]
        );
        assert!(dng_bytes.ends_with(&expected_shifted_pixel_bytes));
    }

    fn le_bytes(samples: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(samples));

        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        bytes
    }

    fn write_samples_to_vec(
        writer: &DngWriter,
        description: &DngFrameDescription,
        samples: &[u16],
    ) -> Result<Vec<u8>, crate::DngWriterError> {
        let pixel_bytes = le_bytes(samples);
        let decoded_frame =
            DecodedBayerU16Frame::from_borrowed_le_bytes(description.dimensions, &pixel_bytes)
                .expect("test frame bytes should match dimensions");

        writer.write_uncompressed_frame_from_decoded_bayer_u16_frame_to_vec(
            description,
            &decoded_frame,
        )
    }

    fn description_for_samples(
        sample_count: usize,
        dng_sample_left_shift: u8,
    ) -> DngFrameDescription {
        DngFrameDescription {
            camera_calibration1: None,
            camera_calibration2: None,
            analog_balance: None,
            frame_number: FrameNumber(0),
            timestamp_us: 0,
            dimensions: FrameDimensions {
                width: u32::try_from(sample_count).expect("test sample count should fit u32"),
                height: 1,
            },
            default_crop_origin: [0, 0],
            default_crop_size: [
                u32::try_from(sample_count).expect("test sample count should fit u32"),
                1,
            ],
            bits_per_sample: 16,
            samples_per_pixel: 1,
            compression: DngCompression::Uncompressed,
            photometric_interpretation: DngPhotometricInterpretation::Cfa,
            cfa_pattern: CfaPattern {
                bayer_pattern: BayerPattern::Gbrg,
                repeat_pattern_dim: [2, 2],
                cfa_plane_color: [0, 1, 2],
                pattern: [1, 2, 0, 1],
            },
            black_level: BlackLevel { values: [256.0; 4] },
            white_level: WhiteLevel {
                values: [4095.0; 4],
            },
            dng_sample_left_shift,
            as_shot_neutral: None,
            color_matrix1: None,
            color_matrix2: None,
            forward_matrix1: None,
            forward_matrix2: None,
            calibration_illuminant1: 21,
            calibration_illuminant2: 17,
            dng_version: [1, 4, 0, 0],
            dng_backward_version: [1, 1, 0, 0],
            unique_camera_model: "MotionCam".to_string(),
        }
    }

    fn rational_tag_values(bytes: &[u8], tag: u16) -> Vec<(u32, u32)> {
        let (field_type, count, value_or_offset) = find_ifd_entry(bytes, tag);
        assert_eq!(field_type, TIFF_TYPE_RATIONAL);

        let value_offset = usize::try_from(value_or_offset).expect("tag offset should fit usize");
        let count = usize::try_from(count).expect("tag count should fit usize");
        let value_byte_count = count.checked_mul(8).expect("value byte count should fit");
        assert!(value_offset + value_byte_count <= bytes.len());

        (0..count)
            .map(|index| {
                let offset = value_offset + index * 8;
                (read_u32(bytes, offset), read_u32(bytes, offset + 4))
            })
            .collect()
    }

    fn srational_tag_values(bytes: &[u8], tag: u16) -> Vec<(i32, i32)> {
        let (field_type, count, value_or_offset) = find_ifd_entry(bytes, tag);
        assert_eq!(field_type, TIFF_TYPE_SRATIONAL);

        let value_offset = usize::try_from(value_or_offset).expect("tag offset should fit usize");
        let count = usize::try_from(count).expect("tag count should fit usize");
        let value_byte_count = count.checked_mul(8).expect("value byte count should fit");
        assert!(value_offset + value_byte_count <= bytes.len());

        (0..count)
            .map(|index| {
                let offset = value_offset + index * 8;
                (read_i32(bytes, offset), read_i32(bytes, offset + 4))
            })
            .collect()
    }

    fn long_tag_values(bytes: &[u8], tag: u16) -> Vec<u32> {
        let (field_type, count, value_or_offset) = find_ifd_entry(bytes, tag);
        assert_eq!(field_type, TIFF_TYPE_LONG);

        let count = usize::try_from(count).expect("tag count should fit usize");
        let value_byte_count = count.checked_mul(4).expect("value byte count should fit");

        if value_byte_count <= 4 {
            return vec![value_or_offset];
        }

        let value_offset = usize::try_from(value_or_offset).expect("tag offset should fit usize");
        assert!(value_offset + value_byte_count <= bytes.len());

        (0..count)
            .map(|index| read_u32(bytes, value_offset + index * 4))
            .collect()
    }

    fn find_ifd_entry(bytes: &[u8], tag: u16) -> (u16, u32, u32) {
        assert!(bytes.len() >= 8);
        assert_eq!(&bytes[0..2], b"II");

        let ifd_offset = usize::try_from(read_u32(bytes, 4)).expect("IFD offset should fit usize");
        let entry_count = usize::from(read_u16(bytes, ifd_offset));
        let ifd_entries_start = ifd_offset + 2;

        for entry_index in 0..entry_count {
            let entry_offset = ifd_entries_start + entry_index * 12;
            let entry_tag = read_u16(bytes, entry_offset);

            if entry_tag == tag {
                return (
                    read_u16(bytes, entry_offset + 2),
                    read_u32(bytes, entry_offset + 4),
                    read_u32(bytes, entry_offset + 8),
                );
            }
        }

        panic!("DNG tag {tag} was not found");
    }

    fn read_u16(bytes: &[u8], offset: usize) -> u16 {
        u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
    }

    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    fn read_i32(bytes: &[u8], offset: usize) -> i32 {
        i32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }
}
