use std::fs;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use xxhash_rust::xxh3::xxh3_128;

const HASH_DOMAIN_SEPARATOR: &str = "mcraw4vulkan-mount-folder-v1";
const FULL_HASH_HEX_LEN: usize = 32;
const FOLDER_NAME_SEPARATOR: &str = "__";
const MAX_SANITIZED_STEM_CHARS: usize = 96;
const MAX_VISIBLE_FOLDER_NAME_CHARS: usize = 120;

/// Default visible suffix length for `sanitized_stem__suffix` mount folders.
pub const DEFAULT_CLIP_MOUNT_SUFFIX_HEX_LEN: usize = 10;
/// Future registries should try these xxh3-128 prefix lengths before adding a
/// deterministic numeric disambiguator.
pub const CLIP_MOUNT_SUFFIX_EXPANSION_HEX_LENGTHS: [usize; 5] = [10, 12, 16, 24, 32];

/// Cheap, stable identity inputs for one source clip.
///
/// This intentionally carries file metadata and path identity only. Normal mount
/// naming must not hash the full `.mcraw` file contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MountClipIdentityInput {
    pub display_stem: String,
    pub canonical_path: Option<String>,
    pub original_path: String,
    pub file_len: Option<u64>,
    pub modified_time: Option<MountClipModifiedTime>,
    pub intrinsic_id: Option<String>,
}

impl MountClipIdentityInput {
    /// Build identity input from a path by reading filesystem metadata.
    ///
    /// Canonicalization is best-effort; if it fails, the original path remains
    /// the path identity input.
    pub fn from_path(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let metadata = fs::metadata(path)?;
        let display_stem = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "clip".to_string());

        Ok(Self {
            display_stem,
            canonical_path: fs::canonicalize(path)
                .ok()
                .map(|canonical_path| path_to_stable_string(&canonical_path)),
            original_path: path_to_stable_string(path),
            file_len: Some(metadata.len()),
            modified_time: metadata.modified().ok().map(system_time_to_unix_parts),
            intrinsic_id: None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MountClipModifiedTime {
    pub unix_seconds: i64,
    pub nanos: u32,
}

// Retains the full xxh3-128 identity so callers can extend visible hash suffixes
// when shorter names collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MountClipIdentity {
    full_xxh3_128: u128,
}

impl MountClipIdentity {
    pub const fn from_xxh3_128(full_xxh3_128: u128) -> Self {
        Self { full_xxh3_128 }
    }

    pub const fn as_u128(self) -> u128 {
        self.full_xxh3_128
    }

    pub fn full_hex(self) -> String {
        format!("{:032x}", self.full_xxh3_128)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MountClipFolderName {
    pub visible: String,
    pub sanitized_stem: String,
    pub suffix: String,
    pub identity: MountClipIdentity,
}

pub fn folder_name_for_path(path: impl AsRef<Path>) -> io::Result<MountClipFolderName> {
    let input = MountClipIdentityInput::from_path(path)?;
    Ok(clip_mount_folder_name(&input))
}

pub fn clip_mount_folder_name(input: &MountClipIdentityInput) -> MountClipFolderName {
    clip_mount_folder_name_with_suffix_len(input, DEFAULT_CLIP_MOUNT_SUFFIX_HEX_LEN)
}

pub fn clip_mount_folder_name_with_suffix_len(
    input: &MountClipIdentityInput,
    suffix_hex_len: usize,
) -> MountClipFolderName {
    let identity = stable_clip_mount_hash(input);
    let suffix = format_clip_mount_hash_suffix(identity, suffix_hex_len);
    let sanitized_stem = truncate_stem_for_suffix(
        &sanitize_mount_folder_stem(&input.display_stem),
        suffix.len(),
    );
    let visible = format!("{sanitized_stem}{FOLDER_NAME_SEPARATOR}{suffix}");

    MountClipFolderName {
        visible,
        sanitized_stem,
        suffix,
        identity,
    }
}

pub fn stable_clip_mount_hash(input: &MountClipIdentityInput) -> MountClipIdentity {
    MountClipIdentity::from_xxh3_128(xxh3_128(&stable_hash_input(input)))
}

pub fn format_clip_mount_hash_suffix(identity: MountClipIdentity, suffix_hex_len: usize) -> String {
    let suffix_hex_len = suffix_hex_len.clamp(1, FULL_HASH_HEX_LEN);
    identity.full_hex()[..suffix_hex_len].to_string()
}

pub fn sanitize_mount_folder_stem(stem: &str) -> String {
    // The visible stem is shared by platform adapters, so it rejects Windows
    // device names and path characters regardless of the generating host.
    let mut sanitized = String::with_capacity(stem.len());
    let mut previous_was_underscore = false;
    let mut saw_substantive_preserved_char = false;

    for value in stem.chars() {
        let should_replace = should_replace_in_mount_folder_stem(value);
        if !should_replace && !value.is_whitespace() && value != '.' {
            saw_substantive_preserved_char = true;
        }

        let value = if should_replace { '_' } else { value };
        if value == '_' {
            if !previous_was_underscore {
                sanitized.push('_');
                previous_was_underscore = true;
            }
        } else {
            sanitized.push(value);
            previous_was_underscore = false;
        }
    }

    let mut sanitized = trim_mount_folder_stem_edges(&sanitized);

    if sanitized.is_empty() || !saw_substantive_preserved_char {
        sanitized = "clip".to_string();
    }

    if sanitized.starts_with('.') {
        sanitized = format!("clip_{sanitized}");
    }

    if is_windows_reserved_folder_stem(&sanitized) {
        sanitized = format!("clip_{sanitized}");
    }

    let mut sanitized = truncate_to_chars(&sanitized, MAX_SANITIZED_STEM_CHARS);
    sanitized = trim_mount_folder_stem_edges(&sanitized);

    if sanitized.is_empty() {
        "clip".to_string()
    } else {
        sanitized
    }
}

fn stable_hash_input(input: &MountClipIdentityInput) -> Vec<u8> {
    // Length-prefix named fields to prevent concatenation ambiguity. Explicit
    // presence tags also keep a missing value distinct from an empty value.
    let mut bytes = Vec::new();
    append_hash_field(&mut bytes, "domain", HASH_DOMAIN_SEPARATOR.as_bytes());
    append_hash_field(&mut bytes, "display_stem", input.display_stem.as_bytes());

    match input.canonical_path.as_deref() {
        Some(canonical_path) => {
            append_hash_field(&mut bytes, "path_kind", b"canonical");
            append_hash_field(&mut bytes, "identity_path", canonical_path.as_bytes());
        }
        None => {
            append_hash_field(&mut bytes, "path_kind", b"original");
            append_hash_field(&mut bytes, "identity_path", input.original_path.as_bytes());
        }
    }

    append_optional_u64(&mut bytes, "file_len", input.file_len);
    match input.modified_time {
        Some(modified_time) => {
            append_hash_field(&mut bytes, "modified_time.present", b"1");
            append_hash_field(
                &mut bytes,
                "modified_time.unix_seconds",
                modified_time.unix_seconds.to_string().as_bytes(),
            );
            append_hash_field(
                &mut bytes,
                "modified_time.nanos",
                modified_time.nanos.to_string().as_bytes(),
            );
        }
        None => {
            append_hash_field(&mut bytes, "modified_time.present", b"0");
        }
    }
    append_optional_str(&mut bytes, "intrinsic_id", input.intrinsic_id.as_deref());

    bytes
}

fn append_optional_u64(bytes: &mut Vec<u8>, name: &str, value: Option<u64>) {
    match value {
        Some(value) => {
            append_hash_field(bytes, &format!("{name}.present"), b"1");
            append_hash_field(
                bytes,
                &format!("{name}.value"),
                value.to_string().as_bytes(),
            );
        }
        None => append_hash_field(bytes, &format!("{name}.present"), b"0"),
    }
}

fn append_optional_str(bytes: &mut Vec<u8>, name: &str, value: Option<&str>) {
    match value {
        Some(value) => {
            append_hash_field(bytes, &format!("{name}.present"), b"1");
            append_hash_field(bytes, &format!("{name}.value"), value.as_bytes());
        }
        None => append_hash_field(bytes, &format!("{name}.present"), b"0"),
    }
}

fn append_hash_field(bytes: &mut Vec<u8>, name: &str, value: &[u8]) {
    bytes.extend_from_slice(name.as_bytes());
    bytes.push(0x1f);
    bytes.extend_from_slice(value.len().to_string().as_bytes());
    bytes.push(0x1f);
    bytes.extend_from_slice(value);
    bytes.push(0x1e);
}

fn truncate_stem_for_suffix(stem: &str, suffix_hex_len: usize) -> String {
    let separator_len = FOLDER_NAME_SEPARATOR.chars().count();
    let max_for_visible = MAX_VISIBLE_FOLDER_NAME_CHARS
        .saturating_sub(separator_len)
        .saturating_sub(suffix_hex_len)
        .max(1);
    let max_stem_chars = MAX_SANITIZED_STEM_CHARS.min(max_for_visible);
    let mut truncated = truncate_to_chars(stem, max_stem_chars);
    truncated = trim_mount_folder_stem_edges(&truncated);

    if truncated.is_empty() {
        "clip".to_string()
    } else {
        truncated
    }
}

fn truncate_to_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn trim_mount_folder_stem_edges(value: &str) -> String {
    let mut trimmed = value.trim_matches(char::is_whitespace).to_string();

    loop {
        let before = trimmed.clone();
        while trimmed.ends_with('.') {
            trimmed.pop();
        }
        trimmed = trimmed.trim_matches(char::is_whitespace).to_string();

        if trimmed == before {
            return trimmed;
        }
    }
}

fn should_replace_in_mount_folder_stem(value: char) -> bool {
    value == '\0'
        || value.is_control()
        || matches!(value, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*')
}

fn is_windows_reserved_folder_stem(value: &str) -> bool {
    let device_name = value
        .trim_end_matches([' ', '.'])
        .split('.')
        .next()
        .unwrap_or(value)
        .to_ascii_uppercase();

    matches!(device_name.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || device_name
            .strip_prefix("COM")
            .and_then(|suffix| suffix.parse::<u8>().ok())
            .is_some_and(|number| (1..=9).contains(&number))
        || device_name
            .strip_prefix("LPT")
            .and_then(|suffix| suffix.parse::<u8>().ok())
            .is_some_and(|number| (1..=9).contains(&number))
}

fn path_to_stable_string(path: &Path) -> String {
    path.as_os_str().to_string_lossy().into_owned()
}

fn system_time_to_unix_parts(time: SystemTime) -> MountClipModifiedTime {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => MountClipModifiedTime {
            unix_seconds: i64::try_from(duration.as_secs()).unwrap_or(i64::MAX),
            nanos: duration.subsec_nanos(),
        },
        Err(error) => {
            let duration = error.duration();
            let seconds = i64::try_from(duration.as_secs()).unwrap_or(i64::MAX);
            let nanos = duration.subsec_nanos();

            if nanos == 0 {
                MountClipModifiedTime {
                    unix_seconds: seconds.saturating_neg(),
                    nanos,
                }
            } else {
                MountClipModifiedTime {
                    unix_seconds: seconds.saturating_add(1).saturating_neg(),
                    nanos: 1_000_000_000 - nanos,
                }
            }
        }
    }
}
