use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "windows")]
pub use windows::{
    Projection, process_id_is_live, start_projection, start_projection_with_instance_id,
};

#[cfg(not(target_os = "windows"))]
mod unsupported {
    use std::path::{Path, PathBuf};

    use crate::{ProjectionInstanceId, ProjectionProvider, StartProjectionError};

    #[derive(Debug)]
    pub struct Projection {
        root: PathBuf,
    }

    impl Projection {
        pub fn root(&self) -> &Path {
            &self.root
        }

        pub fn instance_id(&self) -> ProjectionInstanceId {
            ProjectionInstanceId::from_u128(0)
        }

        pub fn stop(&mut self) {}
    }

    pub fn start_projection<P>(
        root: impl AsRef<Path>,
        _provider: P,
    ) -> Result<Projection, StartProjectionError>
    where
        P: ProjectionProvider + 'static,
    {
        let root = root.as_ref();
        Err(StartProjectionError::UnsupportedPlatform {
            path: root.to_path_buf(),
        })
    }

    pub fn start_projection_with_instance_id<P>(
        root: impl AsRef<Path>,
        _provider: P,
        _instance_id: ProjectionInstanceId,
    ) -> Result<Projection, StartProjectionError>
    where
        P: ProjectionProvider + 'static,
    {
        let root = root.as_ref();
        Err(StartProjectionError::UnsupportedPlatform {
            path: root.to_path_buf(),
        })
    }

    pub fn process_id_is_live(process_id: u32) -> bool {
        process_id != 0 && process_id == std::process::id()
    }
}

#[cfg(not(target_os = "windows"))]
pub use unsupported::{
    Projection, process_id_is_live, start_projection, start_projection_with_instance_id,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProjectionInstanceId(u128);

impl ProjectionInstanceId {
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    pub const fn as_u128(self) -> u128 {
        self.0
    }

    pub fn to_hyphenated_string(self) -> String {
        let value = self.0;
        format!(
            "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
            (value >> 96) as u32,
            ((value >> 80) & 0xffff) as u16,
            ((value >> 64) & 0xffff) as u16,
            ((value >> 48) & 0xffff) as u16,
            value & 0x0000_ffff_ffff_ffff
        )
    }
}

impl fmt::Display for ProjectionInstanceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hyphenated_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    Directory,
    RegularFile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaceholderInfo {
    // byte_len is bytes; time fields are Windows FILETIME ticks (100 ns since
    // 1601) and are passed unchanged to the native placeholder metadata.
    pub kind: FileKind,
    pub byte_len: u64,
    pub creation_time: i64,
    pub last_access_time: i64,
    pub last_write_time: i64,
    pub change_time: i64,
}

impl PlaceholderInfo {
    pub fn directory() -> Self {
        Self {
            kind: FileKind::Directory,
            byte_len: 0,
            creation_time: 0,
            last_access_time: 0,
            last_write_time: 0,
            change_time: 0,
        }
    }

    pub fn regular_file(byte_len: u64) -> Self {
        Self {
            kind: FileKind::RegularFile,
            byte_len,
            creation_time: 0,
            last_access_time: 0,
            last_write_time: 0,
            change_time: 0,
        }
    }

    pub fn with_file_times(
        mut self,
        creation_time: i64,
        last_access_time: i64,
        last_write_time: i64,
        change_time: i64,
    ) -> Self {
        self.creation_time = creation_time;
        self.last_access_time = last_access_time;
        self.last_write_time = last_write_time;
        self.change_time = change_time;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryEntry {
    pub name: OsString,
    pub info: PlaceholderInfo,
}

impl DirectoryEntry {
    pub fn new(name: impl Into<OsString>, info: PlaceholderInfo) -> Self {
        Self {
            name: name.into(),
            info,
        }
    }
}

pub type ProviderResult<T> = Result<T, ProviderError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    NotFound,
    NotAFile,
    Unsupported,
    InvalidPath(String),
    Internal(String),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => formatter.write_str("provider path was not found"),
            Self::NotAFile => formatter.write_str("provider path is not a regular file"),
            Self::Unsupported => formatter.write_str("provider operation is unsupported"),
            Self::InvalidPath(message) => write!(formatter, "invalid provider path: {message}"),
            Self::Internal(message) => write!(formatter, "provider internal error: {message}"),
        }
    }
}

impl Error for ProviderError {}

// ProjFS may invoke provider methods concurrently on native worker threads. The
// Projection owns the provider until native stop has completed all callbacks.
pub trait ProjectionProvider: Send + Sync {
    fn list_directory(&self, relative_path: &Path) -> ProviderResult<Vec<DirectoryEntry>>;

    fn placeholder_info(&self, relative_path: &Path) -> ProviderResult<PlaceholderInfo>;

    fn read_file(
        &self,
        _relative_path: &Path,
        _byte_offset: u64,
        _length: u32,
    ) -> ProviderResult<Vec<u8>> {
        Err(ProviderError::Unsupported)
    }
}

#[derive(Debug)]
pub enum StartProjectionError {
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    InvalidRoot {
        path: PathBuf,
        message: String,
    },
    Hresult {
        operation: &'static str,
        hresult: i32,
    },
    UnsupportedPlatform {
        path: PathBuf,
    },
}

impl fmt::Display for StartProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                source,
            } => write!(
                formatter,
                "{} failed for {}: {}",
                operation,
                path.display(),
                source
            ),
            Self::InvalidRoot { path, message } => {
                write!(
                    formatter,
                    "invalid ProjFS root {}: {}",
                    path.display(),
                    message
                )
            }
            Self::Hresult { operation, hresult } => {
                write!(
                    formatter,
                    "{} failed with HRESULT 0x{:08X}",
                    operation, *hresult as u32
                )
            }
            Self::UnsupportedPlatform { path } => write!(
                formatter,
                "Windows ProjFS projection is not available for {} on this platform",
                path.display()
            ),
        }
    }
}

impl Error for StartProjectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::InvalidRoot { .. } | Self::Hresult { .. } | Self::UnsupportedPlatform { .. } => {
                None
            }
        }
    }
}

pub fn start_empty_projection(root: impl AsRef<Path>) -> Result<Projection, StartProjectionError> {
    start_projection(root, EmptyProjectionProvider)
}

struct EmptyProjectionProvider;

impl ProjectionProvider for EmptyProjectionProvider {
    fn list_directory(&self, _relative_path: &Path) -> ProviderResult<Vec<DirectoryEntry>> {
        Ok(Vec::new())
    }

    fn placeholder_info(&self, _relative_path: &Path) -> ProviderResult<PlaceholderInfo> {
        Err(ProviderError::NotFound)
    }
}
