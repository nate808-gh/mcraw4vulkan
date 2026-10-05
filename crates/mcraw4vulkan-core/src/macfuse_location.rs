//! Side-effect-free macFUSE FUSE3 location policy shared by preflight and mounting.
//! This module uses only std; it never loads a library or starts a mount helper.

use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub const LIBRARY_OVERRIDE: &str = "MCRAW4VULKAN_MACFUSE_LIBRARY";
const STANDARD_LIBRARY: &str = "/usr/local/lib/libfuse3.4.dylib";
const MACPORTS_LIBRARY: &str = "/opt/local/lib/libfuse3.4.dylib";
const REGISTERED_BUNDLE: &str = "/Library/Filesystems/macfuse.fs";
const MACPORTS_BUNDLE: &str = "/opt/local/Library/Filesystems/macfuse.fs";
const MOUNT_HELPER: &str = "Contents/Resources/mount_macfuse";

/// Resolve once for a preflight report or a fresh native API acquisition.
/// Child launchers inherit this process setting through their existing Command
/// environment. Changing installation after successful API loading requires a
/// new application process; the native owner retains its successful library.
pub fn library_from_environment() -> io::Result<PathBuf> {
    resolve_library(std::env::var_os(LIBRARY_OVERRIDE))
}

/// An explicit value selects one absolute file without trimming, expansion or
/// fallback. Otherwise prefer the upstream installer (also used by Homebrew's
/// cask), then MacPorts' normal prefix. File presence is not native acceptance.
pub fn resolve_library(explicit: Option<OsString>) -> io::Result<PathBuf> {
    let registered = Path::new(REGISTERED_BUNDLE);
    resolve_in(
        explicit,
        &[
            (Path::new(STANDARD_LIBRARY), registered),
            (Path::new(MACPORTS_LIBRARY), Path::new(MACPORTS_BUNDLE)),
        ],
        registered,
    )
}

fn resolve_in(
    explicit: Option<OsString>,
    installations: &[(&Path, &Path)],
    registered: &Path,
) -> io::Result<PathBuf> {
    if let Some(value) = explicit {
        let library = PathBuf::from(value);
        if !library.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{LIBRARY_OVERRIDE} must be a nonempty absolute path to a macFUSE FUSE3 library"
                ),
            ));
        }
        // Known layouts keep their library/bundle pairing. For a custom file,
        // the installed system-registered bundle remains authoritative: do not
        // guess a sibling prefix or mix in another automatic library. The user
        // must supply compatible helpers/framework dependencies for that file.
        let bundle = installations
            .iter()
            .find_map(|(candidate, bundle)| (*candidate == library.as_path()).then_some(*bundle))
            .unwrap_or(registered);
        if complete_installation(&library, bundle, registered) {
            return Ok(library);
        }
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "the selected macFUSE FUSE3 library or its registered mount helper is unavailable: {}",
                library.display()
            ),
        ));
    }

    installations
        .iter()
        .find(|(library, bundle)| complete_installation(library, bundle, registered))
        .map(|(library, _)| library.to_path_buf())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "macFUSE FUSE3 library, filesystem bundle or registered mount helper was not found",
            )
        })
}

fn complete_installation(library: &Path, bundle: &Path, registered: &Path) -> bool {
    // These follow ordinary symlinks, rejecting missing/broken targets and
    // directories where a library or helper file is required. MacPorts requires
    // its relocated bundle to be registered at the system bundle path; observe
    // that relationship without creating a symlink or changing OS registration.
    library.is_file()
        && bundle.is_dir()
        && bundle.join(MOUNT_HELPER).is_file()
        && (bundle == registered
            || fs::canonicalize(bundle)
                .ok()
                .zip(fs::canonicalize(registered).ok())
                .is_some_and(|(bundle, registered)| bundle == registered))
}
