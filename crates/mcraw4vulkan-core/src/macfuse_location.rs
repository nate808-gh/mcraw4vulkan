//! Noninteractive macFUSE client/component discovery. No native code is loaded here.
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub const LIBRARY_OVERRIDE: &str = "MCRAW4VULKAN_MACFUSE_LIBRARY";
pub const SETUP_MESSAGE: &str = "DNG mounting requires macFUSE 5.4.0 with its File System Extension enabled in macOS Settings. Install the official macFUSE package if needed. Other application features remain available.";
const STANDARD_LIBRARY: &str = "/usr/local/lib/libfuse3.4.dylib";
const MACPORTS_LIBRARY: &str = "/opt/local/lib/libfuse3.4.dylib";
const REGISTERED_BUNDLE: &str = "/Library/Filesystems/macfuse.fs";
const MACPORTS_BUNDLE: &str = "/opt/local/Library/Filesystems/macfuse.fs";
const FRAMEWORK: &str = "Contents/Frameworks/MFMount.framework/Versions/A/MFMount";
const COMPONENTS: &[&str] = &[
    FRAMEWORK,
    "Contents/Resources/mount_macfuse",
    "Contents/Resources/macfuse.app/Contents/MacOS/macfuse",
    "Contents/Resources/macfuse.app/Contents/Extensions/io.macfuse.app.fsmodule.macfuse.appex/Contents/MacOS/io.macfuse.app.fsmodule.macfuse",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacFuseRuntime {
    pub library: PathBuf,
    pub framework: PathBuf,
}

/// Prefer the signed app-private client, then the two supported installed layouts.
/// The filename is never sufficient identity. No activation or test mount occurs.
/// The explicit developer override remains strict and is not a user setup requirement.
pub fn discover() -> io::Result<MacFuseRuntime> {
    let registered = Path::new(REGISTERED_BUNDLE);
    let mut clients = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(contents) = exe
            .parent()
            .and_then(Path::parent)
            .filter(|p| p.file_name().is_some_and(|n| n == "Contents"))
        {
            clients.push((
                contents.join("Frameworks/macFUSE/libfuse3.4.dylib"),
                registered.to_path_buf(),
            ));
        }
    }
    clients.push((PathBuf::from(STANDARD_LIBRARY), registered.to_path_buf()));
    clients.push((
        PathBuf::from(MACPORTS_LIBRARY),
        PathBuf::from(MACPORTS_BUNDLE),
    ));
    resolve_in(
        std::env::var_os(LIBRARY_OVERRIDE),
        &clients,
        registered,
        verified_components,
    )
}

fn resolve_in(
    explicit: Option<OsString>,
    clients: &[(PathBuf, PathBuf)],
    registered: &Path,
    verify: impl Fn(&Path, &Path) -> bool,
) -> io::Result<MacFuseRuntime> {
    let strict = explicit.is_some();
    let choices = if let Some(value) = explicit {
        let library = PathBuf::from(value);
        if !library.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{LIBRARY_OVERRIDE} must be an absolute macFUSE FUSE3 client path"),
            ));
        }
        let bundle = clients
            .iter()
            .find(|(p, _)| *p == library)
            .map(|(_, b)| b.clone())
            .unwrap_or_else(|| registered.to_path_buf());
        vec![(library, bundle)]
    } else {
        clients.to_vec()
    };
    let mut seen = std::collections::HashSet::new();
    for (library, bundle) in choices {
        let Ok(library) = fs::canonicalize(library) else {
            continue;
        };
        if !seen.insert(library.clone()) {
            continue;
        }
        let registered_bundle = fs::canonicalize(&bundle)
            .ok()
            .zip(fs::canonicalize(registered).ok())
            .is_some_and(|(b, r)| b == r);
        if registered_bundle && library.is_file() && verify(&library, &bundle) {
            return Ok(MacFuseRuntime {
                library,
                framework: fs::canonicalize(bundle.join(FRAMEWORK))?,
            });
        }
    }
    Err(io::Error::new(
        if strict {
            io::ErrorKind::InvalidInput
        } else {
            io::ErrorKind::NotFound
        },
        if strict {
            "The explicit macFUSE client is missing, foreign, or incompatible with the installed macFUSE components."
        } else {
            SETUP_MESSAGE
        },
    ))
}

fn verified_components(library: &Path, bundle: &Path) -> bool {
    // The supported baseline is a vendor-signed 5.4.0 installation and FUSE3 ABI318
    // (checked at deferred load). Reinstall/re-signing does not require matching
    // incidental component hashes. Future releases need explicit validation.
    signed_macfuse(library, true)
        && bundle_value(bundle, "CFBundleIdentifier").as_deref()
            == Some("io.macfuse.filesystems.fs.macfuse")
        && supported_release(bundle_value(bundle, "CFBundleShortVersionString").as_deref())
        && COMPONENTS
            .iter()
            .all(|p| signed_macfuse(&bundle.join(p), false))
}
fn supported_release(release: Option<&str>) -> bool {
    release == Some("5.4.0")
}
fn bundle_value(bundle: &Path, key: &str) -> Option<String> {
    let path = bundle.join("Contents/Info.plist");
    // plutil and codesign are macOS runtime utilities; no developer-tool lookup.
    let output = bounded_command(
        "/usr/bin/plutil",
        &[
            "-extract".as_ref(),
            key.as_ref(),
            "raw".as_ref(),
            "-o".as_ref(),
            "-".as_ref(),
            path.as_os_str(),
        ],
    )
    .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
fn signed_macfuse(path: &Path, client: bool) -> bool {
    if !path.is_file() {
        return false;
    }
    let mut requirement =
        String::from(r#"=anchor apple generic and certificate leaf[subject.OU] = "3T5GSNBU6W""#);
    if client {
        requirement.push_str(" and identifier \"libfuse3\"");
    }
    bounded_command(
        "/usr/bin/codesign",
        &[
            "--verify".as_ref(),
            "--strict".as_ref(),
            "-R".as_ref(),
            requirement.as_ref(),
            path.as_os_str(),
        ],
    )
    .is_ok_and(|o| o.status.success())
}

/// Small runtime queries have an owned child and deadline, including failure.
#[cfg(target_os = "macos")]
pub fn bounded_command(
    program: impl AsRef<std::ffi::OsStr>,
    args: &[&std::ffi::OsStr],
) -> io::Result<std::process::Output> {
    use std::{
        io::Read,
        os::unix::fs::DirBuilderExt,
        process::{Child, Command, Stdio},
        sync::atomic::{AtomicU64, Ordering},
        thread,
        time::{Duration, Instant},
    };
    // Regular owned spool files avoid pipe deadlock, including inherited writers.
    // The child guard reaps on every error; captured bytes and elapsed time are bounded.
    const LIMIT: u64 = 1024 * 1024;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let directory =
        std::env::temp_dir().join(format!("mcraw-runtime-query-{}-{id}", std::process::id()));
    fs::DirBuilder::new().mode(0o700).create(&directory)?;
    struct Spool(PathBuf);
    impl Drop for Spool {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    struct QueryChild(Child);
    impl Drop for QueryChild {
        fn drop(&mut self) {
            if !matches!(self.0.try_wait(), Ok(Some(_))) {
                let _ = self.0.kill();
            }
            let _ = self.0.wait();
        }
    }
    let _spool = Spool(directory.clone());
    let stdout = directory.join("stdout");
    let stderr = directory.join("stderr");
    let mut child = QueryChild(
        Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(fs::File::create_new(&stdout)?)
            .stderr(fs::File::create_new(&stderr)?)
            .spawn()?,
    );
    let end = Instant::now() + Duration::from_secs(5);
    loop {
        let over_limit = fs::metadata(&stdout)?
            .len()
            .saturating_add(fs::metadata(&stderr)?.len())
            > LIMIT;
        if over_limit || Instant::now() >= end {
            return Err(io::Error::new(
                if over_limit {
                    io::ErrorKind::InvalidData
                } else {
                    io::ErrorKind::TimedOut
                },
                "runtime query exceeded its output/time bound",
            ));
        }
        match child.0.try_wait()? {
            Some(status) => {
                let mut output = std::process::Output {
                    status,
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                };
                fs::File::open(stdout)?
                    .take(LIMIT + 1)
                    .read_to_end(&mut output.stdout)?;
                fs::File::open(stderr)?
                    .take(LIMIT + 1)
                    .read_to_end(&mut output.stderr)?;
                if output.stdout.len().saturating_add(output.stderr.len()) > LIMIT as usize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "runtime query exceeded its output bound",
                    ));
                }
                return Ok(output);
            }
            None => thread::sleep(Duration::from_millis(10)),
        }
    }
}
