use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use jiff::civil::DateTime;
use jiff::tz::TimeZone;

// Stable timestamp for virtual filesystem nodes.
//
// MotionCam .mcraw filenames normally begin with local capture time:
//
//   YYMMDD_HHMMSS_...
//
// For example:
//
//   260509_213010_VIDEO_25mm.mcraw
//   -> 2026-05-09 21:30:10 local capture time
//
// The project treats that local filename timestamp as authoritative. We do not
// try to infer where the recording happened. For filesystem APIs, we convert the
// local civil timestamp using the system timezone so each OS receives a normal
// seconds/nanoseconds timestamp.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VirtualTimestamp {
    pub seconds: i64,
    pub nanos: u32,
}

// Choose the virtual timestamp for one input .mcraw file.
//
// A valid YYMMDD_HHMMSS prefix supplies capture time. Other names use a
// representable source-file mtime, or epoch zero when that fallback is unavailable.
pub fn virtual_timestamp_for_input_path(input_path: &Path) -> VirtualTimestamp {
    if let Some(stem) = input_path.file_stem().and_then(|value| value.to_str()) {
        if let Some(timestamp) = parse_motioncam_capture_timestamp_from_stem(stem) {
            return timestamp;
        }
    }

    virtual_timestamp_from_file_mtime(input_path).unwrap_or_default()
}

// Parse a MotionCam filename stem prefix as local capture time.
//
// The parser only requires the initial YYMMDD_HHMMSS prefix. Extra filename
// details after the prefix are ignored:
//
//   260509_213010_VIDEO_25mm
//   260509_213010_anything_else
//
// The year is interpreted as 2000 + YY because MotionCam capture filenames are
// modern camera/video filenames, not historical archive dates.
pub fn parse_motioncam_capture_timestamp_from_stem(stem: &str) -> Option<VirtualTimestamp> {
    let bytes = stem.as_bytes();

    if bytes.len() < 13 {
        return None;
    }

    if bytes[6] != b'_' {
        return None;
    }

    if !bytes[0..6].iter().all(u8::is_ascii_digit) || !bytes[7..13].iter().all(u8::is_ascii_digit) {
        return None;
    }

    let year = 2000 + parse_u32_ascii(&bytes[0..2])? as i16;
    let month = parse_i8_ascii(&bytes[2..4])?;
    let day = parse_i8_ascii(&bytes[4..6])?;
    let hour = parse_i8_ascii(&bytes[7..9])?;
    let minute = parse_i8_ascii(&bytes[9..11])?;
    let second = parse_i8_ascii(&bytes[11..13])?;

    let civil_datetime = DateTime::new(year, month, day, hour, minute, second, 0).ok()?;
    let zoned = civil_datetime.to_zoned(TimeZone::system()).ok()?;
    let timestamp = zoned.timestamp();

    let nanos = timestamp.subsec_nanosecond();

    if nanos < 0 {
        return None;
    }

    Some(VirtualTimestamp {
        seconds: timestamp.as_second(),
        nanos: nanos as u32,
    })
}

// Convert a source file mtime into the shared virtual timestamp type.
fn virtual_timestamp_from_file_mtime(input_path: &Path) -> Option<VirtualTimestamp> {
    let modified = std::fs::metadata(input_path).ok()?.modified().ok()?;
    virtual_timestamp_from_system_time(modified)
}

// Convert SystemTime into VirtualTimestamp without panicking.
fn virtual_timestamp_from_system_time(time: SystemTime) -> Option<VirtualTimestamp> {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            let seconds = i64::try_from(duration.as_secs()).ok()?;

            Some(VirtualTimestamp {
                seconds,
                nanos: duration.subsec_nanos(),
            })
        }
        Err(error) => {
            let duration = error.duration();
            let seconds = i64::try_from(duration.as_secs()).ok()?;

            Some(VirtualTimestamp {
                seconds: -seconds,
                nanos: duration.subsec_nanos(),
            })
        }
    }
}

// Parse a small ASCII decimal digit slice into u32.
fn parse_u32_ascii(bytes: &[u8]) -> Option<u32> {
    let mut value = 0u32;

    for &byte in bytes {
        if !byte.is_ascii_digit() {
            return None;
        }

        value = value.checked_mul(10)?;
        value = value.checked_add(u32::from(byte - b'0'))?;
    }

    Some(value)
}

// Parse a small ASCII decimal digit slice into i8 for Jiff civil date/time APIs.
fn parse_i8_ascii(bytes: &[u8]) -> Option<i8> {
    i8::try_from(parse_u32_ascii(bytes)?).ok()
}

#[cfg(test)]
mod tests {
    use super::parse_motioncam_capture_timestamp_from_stem;

    #[test]
    fn parses_motioncam_capture_timestamp_prefix() {
        let timestamp = parse_motioncam_capture_timestamp_from_stem("260509_213010_VIDEO_25mm");

        assert!(timestamp.is_some());
    }

    #[test]
    fn ignores_non_motioncam_test_names() {
        assert!(parse_motioncam_capture_timestamp_from_stem("bigfile").is_none());
        assert!(parse_motioncam_capture_timestamp_from_stem("260509-213010").is_none());
        assert!(parse_motioncam_capture_timestamp_from_stem("260509_21301").is_none());
    }

    #[test]
    fn ignores_invalid_dates() {
        assert!(parse_motioncam_capture_timestamp_from_stem("261399_213010_VIDEO").is_none());
        assert!(parse_motioncam_capture_timestamp_from_stem("260099_213010_VIDEO").is_none());
        assert!(parse_motioncam_capture_timestamp_from_stem("260500_213010_VIDEO").is_none());
        assert!(parse_motioncam_capture_timestamp_from_stem("260230_213010_VIDEO").is_none());
        assert!(parse_motioncam_capture_timestamp_from_stem("250229_213010_VIDEO").is_none());
        assert!(parse_motioncam_capture_timestamp_from_stem("260509_253010_VIDEO").is_none());
        assert!(parse_motioncam_capture_timestamp_from_stem("260509_246010_VIDEO").is_none());
        assert!(parse_motioncam_capture_timestamp_from_stem("260509_213060_VIDEO").is_none());
    }
}
