use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

const REGISTRY_SCHEMA_VERSION: &str = "2";
const REGISTRY_SCHEMA_VERSION_V1: &str = "1";
const REGISTRY_BACKEND: &str = "dng_fuse";
const REGISTRY_OWNER: &str = "mcraw4vulkan";
const REGISTRY_HEADER: &str = "schema_version\tsource_path_canonical\tmountpoint_path\tprocess_id\tcreated_unix_s\tplatform\tbackend\towned_by\tmount_id\tinstance_id";
const REGISTRY_HEADER_V1: &str = "schema_version\tsource_path_canonical\tmountpoint_path\tprocess_id\tcreated_unix_s\tplatform\tbackend\towned_by\tmount_id";

// A row records cleanup authority, not proof that a mount is still live.
// Platform teardown distinguishes an active mount from an already-absent one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DngMountRegistryRecord {
    pub schema_version: String,
    pub source_path_canonical: PathBuf,
    pub mountpoint_path: PathBuf,
    pub process_id: u32,
    pub created_unix_s: u64,
    pub platform: String,
    pub backend: String,
    pub owned_by: String,
    pub mount_id: String,
    pub instance_id: Option<String>,
}

impl DngMountRegistryRecord {
    pub fn new(
        source_path_canonical: PathBuf,
        mountpoint_path: PathBuf,
        mount_index: usize,
    ) -> Self {
        Self::new_with_instance_id(source_path_canonical, mountpoint_path, mount_index, None)
    }

    pub fn new_with_instance_id(
        source_path_canonical: PathBuf,
        mountpoint_path: PathBuf,
        mount_index: usize,
        instance_id: Option<String>,
    ) -> Self {
        let created_unix_s = current_unix_s();
        let process_id = std::process::id();
        Self {
            schema_version: REGISTRY_SCHEMA_VERSION.to_string(),
            source_path_canonical,
            mountpoint_path,
            process_id,
            created_unix_s,
            platform: platform_label().to_string(),
            backend: REGISTRY_BACKEND.to_string(),
            owned_by: REGISTRY_OWNER.to_string(),
            mount_id: format!("{process_id}-{created_unix_s}-{mount_index}"),
            instance_id,
        }
    }

    pub fn is_mcraw4vulkan_owned_dng_mount(&self) -> bool {
        matches!(
            self.schema_version.as_str(),
            REGISTRY_SCHEMA_VERSION | REGISTRY_SCHEMA_VERSION_V1
        ) && self.backend == REGISTRY_BACKEND
            && self.owned_by == REGISTRY_OWNER
    }

    fn to_tsv_row(&self) -> Result<String> {
        let fields = [
            self.schema_version.clone(),
            path_to_str(&self.source_path_canonical)?.to_string(),
            path_to_str(&self.mountpoint_path)?.to_string(),
            self.process_id.to_string(),
            self.created_unix_s.to_string(),
            self.platform.clone(),
            self.backend.clone(),
            self.owned_by.clone(),
            self.mount_id.clone(),
            self.instance_id.clone().unwrap_or_default(),
        ];

        Ok(fields
            .iter()
            .map(|field| escape_tsv_field(field))
            .collect::<Vec<_>>()
            .join("\t"))
    }

    fn from_tsv_row(row: &str) -> Result<Self> {
        let fields: Vec<String> = row
            .split('\t')
            .map(unescape_tsv_field)
            .collect::<Result<_>>()?;
        if fields.len() != 9 && fields.len() != 10 {
            bail!("registry row has {} fields, expected 9 or 10", fields.len());
        }
        let instance_id = fields.get(9).filter(|value| !value.is_empty()).cloned();

        Ok(Self {
            schema_version: fields[0].clone(),
            source_path_canonical: PathBuf::from(&fields[1]),
            mountpoint_path: PathBuf::from(&fields[2]),
            process_id: fields[3]
                .parse()
                .with_context(|| format!("invalid registry process id {:?}", fields[3]))?,
            created_unix_s: fields[4]
                .parse()
                .with_context(|| format!("invalid registry timestamp {:?}", fields[4]))?,
            platform: fields[5].clone(),
            backend: fields[6].clone(),
            owned_by: fields[7].clone(),
            mount_id: fields[8].clone(),
            instance_id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DngMountRegistry {
    path: PathBuf,
}

impl DngMountRegistry {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn default_path() -> Result<PathBuf> {
        runtime_registry_dir().map(|dir| dir.join("dng-mounts.tsv"))
    }

    pub fn from_default_path() -> Result<Self> {
        Ok(Self::new(Self::default_path()?))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load_records(&self) -> Result<Vec<DngMountRegistryRecord>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let text = fs::read_to_string(&self.path).with_context(|| {
            format!("failed to read DNG mount registry {}", self.path.display())
        })?;
        let mut records = Vec::new();

        for (line_index, line) in text.lines().enumerate() {
            if line_index == 0 && (line == REGISTRY_HEADER || line == REGISTRY_HEADER_V1) {
                continue;
            }
            if line.trim().is_empty() {
                continue;
            }
            records.push(DngMountRegistryRecord::from_tsv_row(line).with_context(|| {
                format!(
                    "failed to parse DNG mount registry line {} in {}",
                    line_index + 1,
                    self.path.display()
                )
            })?);
        }

        Ok(records)
    }

    pub fn write_records(&self, records: &[DngMountRegistryRecord]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create DNG mount registry dir {}",
                    parent.display()
                )
            })?;
        }

        let mut text = String::from(REGISTRY_HEADER);
        text.push('\n');
        for record in records {
            text.push_str(&record.to_tsv_row()?);
            text.push('\n');
        }

        fs::write(&self.path, text)
            .with_context(|| format!("failed to write DNG mount registry {}", self.path.display()))
    }

    pub fn add_record(&self, record: DngMountRegistryRecord) -> Result<()> {
        let mut records = self.load_records()?;
        records.retain(|existing| {
            existing.mount_id != record.mount_id
                && !(existing.source_path_canonical == record.source_path_canonical
                    && existing.mountpoint_path == record.mountpoint_path)
        });
        records.push(record);
        self.write_records(&records)
    }

    pub fn records_for_source(&self, source_path: &Path) -> Result<Vec<DngMountRegistryRecord>> {
        Ok(self
            .load_records()?
            .into_iter()
            .filter(|record| {
                record.is_mcraw4vulkan_owned_dng_mount()
                    && record.source_path_canonical == source_path
            })
            .collect())
    }

    // Cleanup may act only on rows matching this schema, backend, and owner;
    // unrelated registry data is never interpreted as an application-owned mount.
    pub fn all_owned_records(&self) -> Result<Vec<DngMountRegistryRecord>> {
        Ok(self
            .load_records()?
            .into_iter()
            .filter(DngMountRegistryRecord::is_mcraw4vulkan_owned_dng_mount)
            .collect())
    }

    pub fn remove_mount_ids(&self, mount_ids: &[String]) -> Result<()> {
        if mount_ids.is_empty() {
            return Ok(());
        }

        let mut records = self.load_records()?;
        records.retain(|record| {
            !mount_ids
                .iter()
                .any(|mount_id| mount_id == &record.mount_id)
        });
        self.write_records(&records)
    }
}

pub fn canonical_source_path(path: &Path) -> Result<PathBuf> {
    fs::canonicalize(path)
        .with_context(|| format!("failed to resolve source file {}", path.display()))
}

pub fn platform_label() -> &'static str {
    if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "unknown"
    }
}

fn runtime_registry_dir() -> Result<PathBuf> {
    if cfg!(target_os = "linux") {
        if let Some(path) = env_path("XDG_RUNTIME_DIR") {
            return Ok(path.join("mcraw4vulkan"));
        }
        if let Some(path) = env_path("TMPDIR") {
            return Ok(path.join("mcraw4vulkan"));
        }
        bail!("no safe runtime directory found for DNG mount registry; set XDG_RUNTIME_DIR")
    } else if cfg!(target_os = "windows") {
        if let Some(path) = env_path("LOCALAPPDATA") {
            return Ok(path.join("mcraw4vulkan"));
        }
        if let Some(path) = env_path("TEMP") {
            return Ok(path.join("mcraw4vulkan"));
        }
        bail!("no safe runtime directory found for DNG mount registry; set LOCALAPPDATA or TEMP")
    } else if cfg!(target_os = "macos") {
        if let Some(path) = env_path("TMPDIR") {
            return Ok(path.join("mcraw4vulkan"));
        }
        bail!("no safe runtime directory found for DNG mount registry; set TMPDIR")
    } else if let Some(path) = env_path("TMPDIR") {
        Ok(path.join("mcraw4vulkan"))
    } else {
        bail!("no safe runtime directory found for DNG mount registry")
    }
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn path_to_str(path: &Path) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("registry path is not valid UTF-8: {}", path.display()))
}

fn escape_tsv_field(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch as u32 {
            92 => {
                escaped.push(char::from_u32(92).expect("backslash code point is valid"));
                escaped.push(char::from_u32(92).expect("backslash code point is valid"));
            }
            9 => {
                escaped.push(char::from_u32(92).expect("backslash code point is valid"));
                escaped.push('t');
            }
            10 => {
                escaped.push(char::from_u32(92).expect("backslash code point is valid"));
                escaped.push('n');
            }
            13 => {
                escaped.push(char::from_u32(92).expect("backslash code point is valid"));
                escaped.push('r');
            }
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn unescape_tsv_field(value: &str) -> Result<String> {
    let mut unescaped = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            unescaped.push(ch);
            continue;
        }

        let Some(escaped) = chars.next() else {
            bail!("registry field ends with an incomplete escape");
        };
        match escaped {
            '\\' => unescaped.push('\\'),
            't' => unescaped.push('\t'),
            'n' => unescaped.push('\n'),
            'r' => unescaped.push('\r'),
            other => bail!("registry field contains unknown escape code {other}"),
        }
    }
    Ok(unescaped)
}

fn current_unix_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
