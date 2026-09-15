use std::error::Error;
use std::ffi::{CString, c_char, c_int, c_void};
use std::fmt;
use std::fs;
use std::io;
use std::marker::PhantomData;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::rc::Rc;
use std::thread;
use std::time::{Duration, Instant};

const CLEANUP_OBSERVE_TIMEOUT: Duration = Duration::from_secs(5);
const CLEANUP_OBSERVE_POLL: Duration = Duration::from_millis(25);

#[repr(C)]
struct FuseArgs {
    argc: c_int,
    argv: *mut *mut c_char,
    allocated: c_int,
}

enum FuseSession {}

#[link(name = "fuse3")]
unsafe extern "C" {
    fn fuse_opt_add_arg(args: *mut FuseArgs, arg: *const c_char) -> c_int;
    fn fuse_opt_free_args(args: *mut FuseArgs);

    // The installed header maps source callers to a versioned constructor, but
    // macFUSE also exports this four-argument compatibility ABI for external
    // request dispatchers such as fuser.
    #[link_name = "fuse_session_new"]
    fn fuse_session_new_abi(
        args: *mut FuseArgs,
        operations: *const c_void,
        operations_size: usize,
        userdata: *mut c_void,
    ) -> *mut FuseSession;
    fn fuse_session_mount(session: *mut FuseSession, mountpoint: *const c_char) -> c_int;
    fn fuse_session_fd(session: *mut FuseSession) -> c_int;
    fn fuse_session_exit(session: *mut FuseSession);
    fn fuse_session_unmount(session: *mut FuseSession);
    fn fuse_session_destroy(session: *mut FuseSession);
}

unsafe extern "C" {
    // Public, thread-safe macOS mount-table API. The returned allocation is
    // caller-owned and must be released with free(3).
    fn getmntinfo_r_np(entries: *mut *mut libc::statfs, flags: c_int) -> c_int;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacFuseSessionErrorKind {
    NativeSetup,
    ReaderSetup,
    MountObservation,
    Busy,
    Unmount,
    Worker,
    Cleanup,
}

#[derive(Debug)]
pub struct MacFuseSessionError {
    kind: MacFuseSessionErrorKind,
    operation: &'static str,
    mountpoint: PathBuf,
    source: Option<io::Error>,
    cleanup_failures: Vec<String>,
}

impl MacFuseSessionError {
    fn new(
        kind: MacFuseSessionErrorKind,
        operation: &'static str,
        mountpoint: &Path,
        source: Option<io::Error>,
    ) -> Self {
        Self {
            kind,
            operation,
            mountpoint: mountpoint.to_path_buf(),
            source,
            cleanup_failures: Vec::new(),
        }
    }

    fn with_cleanup_failures(mut self, failures: Vec<String>) -> Self {
        self.cleanup_failures = failures;
        self
    }

    pub fn kind(&self) -> MacFuseSessionErrorKind {
        self.kind
    }

    pub fn mountpoint(&self) -> &Path {
        &self.mountpoint
    }

    pub fn cleanup_failures(&self) -> &[String] {
        &self.cleanup_failures
    }

    pub fn raw_os_error(&self) -> Option<i32> {
        self.source.as_ref().and_then(io::Error::raw_os_error)
    }
}

impl fmt::Display for MacFuseSessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} for {}",
            self.operation,
            self.mountpoint.display()
        )?;
        if let Some(source) = &self.source {
            write!(formatter, ": {source}")?;
        }
        for failure in &self.cleanup_failures {
            write!(formatter, "; terminal cleanup also failed: {failure}")?;
        }
        Ok(())
    }
}

impl Error for MacFuseSessionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source as &(dyn Error + 'static))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnmountOutcome {
    /// The ordinary unmount call disconnected the observed mount.
    Unmounted,
    /// The observed mount was already absent, or this guard had already been
    /// finalized; any retained worker/native ownership has been released.
    AlreadyUnmounted,
}

/// Owns the native macFUSE session and fuser's sole request reader together.
///
/// The `Rc` marker makes this guard neither `Send` nor `Sync`: all native
/// lifecycle calls and Drop therefore remain on the thread that constructed
/// the session. The fuser request loop itself runs on its own managed worker.
pub struct MacFuseSession {
    worker: Option<fuser::BackgroundSession>,
    native: Option<NativeSession>,
    mountpoint: PathBuf,
    _thread_bound: PhantomData<Rc<()>>,
}

impl MacFuseSession {
    pub fn mount<FS>(
        filesystem: FS,
        mountpoint: impl AsRef<Path>,
        native_options: &[String],
        config: fuser::Config,
    ) -> Result<Self, MacFuseSessionError>
    where
        FS: fuser::Filesystem + Send + 'static,
    {
        let requested_mountpoint = mountpoint.as_ref();
        let mountpoint = fs::canonicalize(requested_mountpoint).map_err(|source| {
            MacFuseSessionError::new(
                MacFuseSessionErrorKind::NativeSetup,
                "could not resolve the macFUSE mount target before native setup",
                requested_mountpoint,
                Some(source),
            )
        })?;
        let (mut native, channel) = NativeSession::open_channel(&mountpoint, native_options)?;
        let acl = config.acl;

        let session = match fuser::Session::from_fd(filesystem, channel, acl, config) {
            Ok(session) => session,
            Err(source) => {
                let cleanup = native.cancel_and_destroy();
                return Err(MacFuseSessionError::new(
                    MacFuseSessionErrorKind::ReaderSetup,
                    "registry fuser failed its Session::from_fd handshake",
                    &mountpoint,
                    Some(source),
                )
                .with_cleanup_failures(cleanup));
            }
        };

        if let Err(source) = native.capture_mount_identity(CLEANUP_OBSERVE_TIMEOUT) {
            drop(session);
            let cleanup = native.cancel_and_destroy();
            return Err(MacFuseSessionError::new(
                MacFuseSessionErrorKind::MountObservation,
                "could not identify the mounted macFUSE session",
                &mountpoint,
                Some(source),
            )
            .with_cleanup_failures(cleanup));
        }

        let worker = match session.spawn() {
            Ok(worker) => worker,
            Err(source) => {
                let cleanup = native.cancel_and_destroy();
                return Err(MacFuseSessionError::new(
                    MacFuseSessionErrorKind::ReaderSetup,
                    "registry fuser failed to start its request worker",
                    &mountpoint,
                    Some(source),
                )
                .with_cleanup_failures(cleanup));
            }
        };

        Ok(Self {
            worker: Some(worker),
            native: Some(native),
            mountpoint,
            _thread_bound: PhantomData,
        })
    }

    /// Returns the one resolved absolute target used for native mount,
    /// observation, and unmount operations.
    pub fn mountpoint(&self) -> &Path {
        &self.mountpoint
    }

    /// Returns the mount-table status of this session's resolved target.
    ///
    /// Retaining native ownership after an external unmount does not make this
    /// return `true`. An observation failure is reported distinctly from a
    /// confirmed absence.
    pub fn is_mounted(&self) -> Result<bool, MacFuseSessionError> {
        let Some(native) = self.native.as_ref() else {
            return Ok(false);
        };
        native.owned_mount_is_present().map_err(|source| {
            MacFuseSessionError::new(
                MacFuseSessionErrorKind::MountObservation,
                "could not inspect the macOS mount table for this session",
                &self.mountpoint,
                Some(source),
            )
        })
    }

    /// Attempts one ordinary, non-forced unmount.
    ///
    /// A refusal or mount-observation error before disconnection leaves this
    /// guard, its native owner, its reader, and its provider retained. In
    /// particular, `Busy` is the actual `EBUSY` from macOS `unmount(2)` and can
    /// be retried on this same value after the busy resource is released.
    ///
    /// Once unmount succeeds, or absence is observed after an external
    /// unmount, teardown is terminal. A `Worker` error returned after that
    /// point reports failed worker finalization; it does not mean the mount is
    /// still live or that this guard remains retryable.
    pub fn try_unmount(&mut self) -> Result<UnmountOutcome, MacFuseSessionError> {
        let Some(native) = self.native.as_ref() else {
            return Ok(UnmountOutcome::AlreadyUnmounted);
        };

        match native.owned_mount_is_present() {
            Ok(false) => return self.finish_disconnected(UnmountOutcome::AlreadyUnmounted),
            Ok(true) => {}
            Err(source) => {
                return Err(MacFuseSessionError::new(
                    MacFuseSessionErrorKind::MountObservation,
                    "could not inspect the macOS mount table before unmount",
                    &self.mountpoint,
                    Some(source),
                ));
            }
        }

        // SAFETY: `native.mountpoint_c` is a live, NUL-terminated path for the
        // duration of the call. `unmount(2)` does not retain the pointer. Flags
        // are zero, so this is an ordinary (never forced) unmount. Its documented
        // synchronous result reports success only after VFS disassociates the
        // filesystem from this mountpoint.
        let result = unsafe { libc::unmount(native.mountpoint_c.as_ptr(), 0) };
        if result != 0 {
            // Capture errno before mount-table inspection or any cleanup call.
            let source = io::Error::last_os_error();
            if source.raw_os_error() == Some(libc::EBUSY) {
                return Err(MacFuseSessionError::new(
                    MacFuseSessionErrorKind::Busy,
                    "macOS refused the orderly unmount because the mount is busy",
                    &self.mountpoint,
                    Some(source),
                ));
            }

            // An external unmount may have won the race after the initial
            // observation. Finalize only if our recorded mount identity is gone.
            if matches!(native.owned_mount_is_present(), Ok(false)) {
                return self.finish_disconnected(UnmountOutcome::AlreadyUnmounted);
            }
            return Err(MacFuseSessionError::new(
                MacFuseSessionErrorKind::Unmount,
                "macOS rejected the orderly unmount",
                &self.mountpoint,
                Some(source),
            ));
        }

        self.finish_disconnected(UnmountOutcome::Unmounted)
    }

    /// Blocks until an external unmount ends fuser's reader, then verifies the
    /// recorded mount identity has disappeared before releasing native state.
    pub fn wait_until_unmounted(mut self) -> Result<(), MacFuseSessionError> {
        let worker_result = self.join_worker();
        let presence_result = self
            .native
            .as_ref()
            .map_or(Ok(false), NativeSession::owned_mount_is_present);

        match (worker_result, presence_result) {
            (Ok(()), Ok(false)) => {
                self.release_native_orderly();
                Ok(())
            }
            (worker_result, presence_result) => {
                let mut primary = match worker_result {
                    Err(source) => MacFuseSessionError::new(
                        MacFuseSessionErrorKind::Worker,
                        "registry fuser request worker ended with an error",
                        &self.mountpoint,
                        Some(source),
                    ),
                    Ok(()) => match presence_result {
                        Ok(true) => MacFuseSessionError::new(
                            MacFuseSessionErrorKind::MountObservation,
                            "request worker ended while its macFUSE mount remained present",
                            &self.mountpoint,
                            None,
                        ),
                        Err(source) => MacFuseSessionError::new(
                            MacFuseSessionErrorKind::MountObservation,
                            "could not confirm disappearance after the request worker ended",
                            &self.mountpoint,
                            Some(source),
                        ),
                        Ok(false) => unreachable!(),
                    },
                };
                let cleanup = self.terminal_cleanup();
                primary.cleanup_failures.extend(cleanup);
                Err(primary)
            }
        }
    }

    /// Terminates a session that cannot be kept as a usable mounted provider.
    /// This is intentionally separate from retryable orderly unmount.
    pub fn abort(mut self) -> Result<(), MacFuseSessionError> {
        let failures = self.terminal_cleanup();
        if failures.is_empty() {
            Ok(())
        } else {
            Err(MacFuseSessionError::new(
                MacFuseSessionErrorKind::Cleanup,
                "terminal macFUSE session cleanup failed",
                &self.mountpoint,
                None,
            )
            .with_cleanup_failures(failures))
        }
    }

    fn finish_disconnected(
        &mut self,
        outcome: UnmountOutcome,
    ) -> Result<UnmountOutcome, MacFuseSessionError> {
        let worker_result = self.join_worker();
        self.release_native_orderly();
        worker_result.map(|()| outcome).map_err(|source| {
            MacFuseSessionError::new(
                MacFuseSessionErrorKind::Worker,
                "registry fuser request worker failed during orderly teardown",
                &self.mountpoint,
                Some(source),
            )
        })
    }

    fn join_worker(&mut self) -> io::Result<()> {
        match self.worker.take() {
            Some(worker) => worker.join(),
            None => Ok(()),
        }
    }

    fn release_native_orderly(&mut self) {
        if let Some(mut native) = self.native.take() {
            // The OS mount is already gone and the duplicate fuser descriptor
            // has been joined/dropped. This call reconciles macFUSE's internal
            // mount state; its void return is not used as evidence of success.
            native.request_native_unmount();
            native.destroy();
        }
    }

    fn terminal_cleanup(&mut self) -> Vec<String> {
        let mut failures = Vec::new();
        if let Some(native) = self.native.as_mut() {
            let _ = native.force_os_unmount_if_present();
            native.force_unmount();
            if let Err(error) = native.force_os_unmount_until_absent(CLEANUP_OBSERVE_TIMEOUT) {
                failures.push(format!("forced OS unmount: {error}"));
            }
        }

        if let Err(error) = self.join_worker() {
            failures.push(format!("request worker join: {error}"));
        }
        if let Some(mut native) = self.native.take() {
            if let Err(error) = native.wait_until_owned_mount_absent(CLEANUP_OBSERVE_TIMEOUT) {
                failures.push(format!("mount disappearance: {error}"));
            }
            native.destroy();
        }
        failures
    }
}

impl Drop for MacFuseSession {
    fn drop(&mut self) {
        // Dropping the final owner is terminal abandonment, not an ordinary
        // retryable unmount. A public forced VFS unmount closes the request
        // channel, and macFUSE's documented exit/unmount path then reconciles
        // its native state. Joining before destroying the caller reference
        // keeps the provider and descriptor ordering intact.
        // Every operation here returns normally; errors are deliberately
        // ignored because Drop cannot report them and must not panic while
        // unwinding.
        let _ = self.terminal_cleanup();
    }
}

struct OwnedFuseArgs {
    raw: FuseArgs,
}

impl OwnedFuseArgs {
    fn new(options: &[String]) -> io::Result<Self> {
        let mut args = Self {
            raw: FuseArgs {
                argc: 0,
                argv: ptr::null_mut(),
                allocated: 0,
            },
        };
        args.add("mcraw4vulkan")?;
        for option in options {
            args.add("-o")?;
            args.add(option)?;
        }
        Ok(args)
    }

    fn add(&mut self, value: &str) -> io::Result<()> {
        let value = CString::new(value).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "a native libfuse3 argument contains an interior NUL byte",
            )
        })?;
        // SAFETY: `self.raw` is the exclusively owned public `fuse_args`
        // structure and `value` remains valid for the call. libfuse3 copies the
        // argument into its own mutable vector. The API documents only an
        // allocation-failure outcome, so no potentially stale errno is added.
        if unsafe { fuse_opt_add_arg(&mut self.raw, value.as_ptr()) } == -1 {
            return Err(io::Error::other(
                "libfuse3 could not allocate its argument vector",
            ));
        }
        Ok(())
    }
}

impl Drop for OwnedFuseArgs {
    fn drop(&mut self) {
        // SAFETY: this value owns the mutable argv allocated by libfuse3.
        // `fuse_opt_free_args` accepts the zero/null initial state and clears
        // partial allocation after any failed add or constructor call.
        unsafe { fuse_opt_free_args(&mut self.raw) };
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
// Practical mount-table checks for the application's already uniquely named
// target. These fields reject an unexpected entry during ordinary operation;
// they are not a race-proof kernel mount-instance identifier.
struct MountIdentity {
    mountpoint: Vec<u8>,
    source: Vec<u8>,
    filesystem_type: Vec<u8>,
}

struct NativeSession {
    raw: Option<NonNull<FuseSession>>,
    mountpoint_c: CString,
    mountpoint_key: Vec<u8>,
    identity: Option<MountIdentity>,
    mount_requested: bool,
    _thread_bound: PhantomData<Rc<()>>,
}

impl NativeSession {
    fn open_channel(
        mountpoint: &Path,
        options: &[String],
    ) -> Result<(Self, OwnedFd), MacFuseSessionError> {
        let mountpoint_c = CString::new(mountpoint.as_os_str().as_bytes()).map_err(|_| {
            MacFuseSessionError::new(
                MacFuseSessionErrorKind::NativeSetup,
                "the macFUSE mountpoint contains an interior NUL byte",
                mountpoint,
                None,
            )
        })?;
        let mut args = OwnedFuseArgs::new(options).map_err(|source| {
            MacFuseSessionError::new(
                MacFuseSessionErrorKind::NativeSetup,
                "could not build the native libfuse3 argument vector",
                mountpoint,
                Some(source),
            )
        })?;

        // SAFETY: `args` is live and valid for the call. The exported macFUSE
        // compatibility ABI accepts null operations for an external dispatcher;
        // no Rust callback or userdata pointer is retained. libfuse3 parses or
        // copies the argument data before returning.
        let raw = unsafe { fuse_session_new_abi(&mut args.raw, ptr::null(), 0, ptr::null_mut()) };
        let Some(raw) = NonNull::new(raw) else {
            // This constructor documents null as rejection but does not promise
            // errno, so report the outcome without attaching stale errno.
            return Err(MacFuseSessionError::new(
                MacFuseSessionErrorKind::NativeSetup,
                "native libfuse3 rejected the session or mount options",
                mountpoint,
                None,
            ));
        };

        let mut owner = Self {
            raw: Some(raw),
            mountpoint_c,
            mountpoint_key: mountpoint.as_os_str().as_bytes().to_vec(),
            identity: None,
            mount_requested: false,
            _thread_bound: PhantomData,
        };

        // SAFETY: `owner` holds the live caller reference and `mountpoint_c` is
        // a valid NUL-terminated path for the call. macFUSE copies the path when
        // accepting its deferred mount request.
        let mount_result = unsafe { fuse_session_mount(raw.as_ptr(), owner.mountpoint_c.as_ptr()) };
        if mount_result != 0 {
            let cleanup = owner.cancel_and_destroy();
            // `fuse_session_mount` documents only its zero/-1 outcome, not
            // errno. Do not attach whatever errno cleanup or an earlier call
            // happened to leave behind.
            return Err(MacFuseSessionError::new(
                MacFuseSessionErrorKind::NativeSetup,
                "native libfuse3 failed to create the mount session",
                mountpoint,
                None,
            )
            .with_cleanup_failures(cleanup));
        }
        owner.mount_requested = true;

        // SAFETY: `__error` returns this thread's live errno slot. Clearing it
        // prevents a failure path that does not set errno from being mislabeled
        // with an earlier call's value. `raw` remains protected by the caller
        // reference. The returned descriptor is borrowed from macFUSE and is
        // not adopted by Rust.
        let borrowed_fd = unsafe {
            *libc::__error() = 0;
            fuse_session_fd(raw.as_ptr())
        };
        if borrowed_fd < 0 {
            // `fuse_session_fd`'s macOS implementation reports the channel
            // creation failure through errno. Capture it immediately, before
            // forced startup cleanup can overwrite it.
            let source =
                last_native_error("native libfuse3 failed to provide its mount channel descriptor");
            let cleanup = owner.cancel_and_destroy();
            return Err(MacFuseSessionError::new(
                MacFuseSessionErrorKind::NativeSetup,
                "native libfuse3 failed to provide its mount channel descriptor",
                mountpoint,
                Some(source),
            )
            .with_cleanup_failures(cleanup));
        }

        // SAFETY: macFUSE guarantees its descriptor is valid until native
        // unmount. `owner` is live for this operation, and the temporary borrow
        // is used only to duplicate the descriptor. fuser receives the duplicate
        // as `OwnedFd`; ownership of macFUSE's original is never transferred.
        let borrowed_fd = unsafe { BorrowedFd::borrow_raw(borrowed_fd) };
        let channel = match borrowed_fd.try_clone_to_owned() {
            Ok(channel) => channel,
            Err(source) => {
                let cleanup = owner.cancel_and_destroy();
                return Err(MacFuseSessionError::new(
                    MacFuseSessionErrorKind::NativeSetup,
                    "could not duplicate macFUSE's borrowed channel descriptor",
                    mountpoint,
                    Some(source),
                )
                .with_cleanup_failures(cleanup));
            }
        };

        Ok((owner, channel))
    }

    fn capture_mount_identity(&mut self, timeout: Duration) -> io::Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            self.identity = self.find_mount_identity()?;
            if self.identity.is_some() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "the new macFUSE mount was absent from the macOS mount table after {:.3}s",
                        timeout.as_secs_f64()
                    ),
                ));
            }
            thread::sleep(CLEANUP_OBSERVE_POLL);
        }
    }

    fn find_mount_identity(&self) -> io::Result<Option<MountIdentity>> {
        let table = MountTable::load()?;
        Ok(table.entries().iter().find_map(|entry| {
            let mountpoint = c_field_bytes(&entry.f_mntonname);
            (self.mountpoint_key == mountpoint).then(|| MountIdentity {
                mountpoint,
                source: c_field_bytes(&entry.f_mntfromname),
                filesystem_type: c_field_bytes(&entry.f_fstypename),
            })
        }))
    }

    fn owned_mount_is_present(&self) -> io::Result<bool> {
        let Some(identity) = &self.identity else {
            // Native cancellation is valid before the handshake records the
            // application's mount-table key, but lack of that key cannot prove
            // that the asynchronous helper or mount is absent.
            return Err(io::Error::other(
                "the session has no recorded mount-table identity, so mount presence is unknown",
            ));
        };
        let table = MountTable::load()?;
        Ok(table.entries().iter().any(|entry| {
            MountIdentity {
                mountpoint: c_field_bytes(&entry.f_mntonname),
                source: c_field_bytes(&entry.f_mntfromname),
                filesystem_type: c_field_bytes(&entry.f_fstypename),
            } == *identity
        }))
    }

    fn request_native_unmount(&mut self) {
        if let Some(raw) = self.raw {
            // SAFETY: this object owns the live caller reference. The function
            // is void and asynchronous on macOS; callers never interpret it as
            // success. macFUSE serializes repeated requests internally.
            unsafe { fuse_session_unmount(raw.as_ptr()) };
        }
    }

    fn force_unmount(&mut self) {
        if self.mount_requested {
            if let Some(raw) = self.raw {
                // SAFETY: the caller reference is live and all lifecycle access is
                // confined to the constructing thread. `exit` marks an incomplete
                // or abandoned request service terminal; macFUSE's documented
                // unmount path then closes the channel and requests a forced backend
                // unmount. No Rust pointer is shared with its native worker.
                unsafe {
                    fuse_session_exit(raw.as_ptr());
                    fuse_session_unmount(raw.as_ptr());
                }
            }
        }
    }

    fn force_os_unmount_if_present(&self) -> io::Result<()> {
        let present = self.owned_mount_is_present()?;
        if !present {
            return Ok(());
        }

        // SAFETY: `mountpoint_c` remains a valid NUL-terminated path for the
        // call and is not retained. This forced public VFS operation is used
        // only after the final owner has entered terminal abandonment with a
        // recorded mount-table identity, never in response to an ordinary
        // retryable Busy result. It closes the kernel channel even if
        // macFUSE's Disk Arbitration cleanup cannot create a DADisk reference.
        let result = unsafe { libc::unmount(self.mountpoint_c.as_ptr(), libc::MNT_FORCE) };
        if result == 0 {
            return Ok(());
        }

        // Preserve the syscall failure before checking whether macFUSE's own
        // concurrent forced-unmount request won the race.
        let source = io::Error::last_os_error();
        if matches!(self.owned_mount_is_present(), Ok(false)) {
            Ok(())
        } else {
            Err(source)
        }
    }

    fn force_os_unmount_until_absent(&self, timeout: Duration) -> io::Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.force_os_unmount_if_present() {
                Ok(()) => return Ok(()),
                Err(error) if Instant::now() >= deadline => {
                    return Err(io::Error::new(
                        error.kind(),
                        format!(
                            "forced unmount remained refused after {:.3}s: {error}",
                            timeout.as_secs_f64()
                        ),
                    ));
                }
                Err(_) => thread::sleep(CLEANUP_OBSERVE_POLL),
            }
        }
    }

    fn wait_until_owned_mount_absent(&self, timeout: Duration) -> io::Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            if !self.owned_mount_is_present()? {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "the recorded macFUSE mount remained present after {:.3}s",
                        timeout.as_secs_f64()
                    ),
                ));
            }
            thread::sleep(CLEANUP_OBSERVE_POLL);
        }
    }

    fn cancel_and_destroy(&mut self) -> Vec<String> {
        let mut failures = Vec::new();
        // Before an identity is recorded, pathname-based forced unmount is not
        // authorized and absence cannot be observed through this owner. The
        // native exit/unmount path still cancels the delayed/connecting mount;
        // macFUSE's callback holds its own counted session reference.
        let can_observe_mount = self.identity.is_some();
        if can_observe_mount {
            let _ = self.force_os_unmount_if_present();
        }
        self.force_unmount();
        if can_observe_mount {
            if let Err(error) = self.force_os_unmount_until_absent(CLEANUP_OBSERVE_TIMEOUT) {
                failures.push(format!("forced OS unmount: {error}"));
            }
            if let Err(error) = self.wait_until_owned_mount_absent(CLEANUP_OBSERVE_TIMEOUT) {
                failures.push(format!("mount disappearance: {error}"));
            }
        }
        self.destroy();
        failures
    }

    fn destroy(&mut self) {
        if let Some(raw) = self.raw.take() {
            // SAFETY: this releases exactly the caller reference returned by
            // `fuse_session_new`. No fuser reader remains when called by the
            // composite guard. During construction failure or NativeSession
            // Drop, forced channel shutdown precedes release. macFUSE's own
            // asynchronous workers retain separate counted references, so
            // releasing this caller reference does not invalidate their state.
            unsafe { fuse_session_destroy(raw.as_ptr()) };
            self.mount_requested = false;
        }
    }
}

impl Drop for NativeSession {
    fn drop(&mut self) {
        // Handles partial construction, including a native channel created
        // before any fuser reader exists. In that pre-identity state this
        // requests source-supported native cancellation but does not claim
        // mount/helper disappearance was observed. This never panics; the
        // composite MacFuseSession performs the stronger reader-join ordering
        // once a worker has been established.
        let _ = self.cancel_and_destroy();
    }
}

struct MountTable {
    entries: NonNull<libc::statfs>,
    len: usize,
}

impl MountTable {
    fn load() -> io::Result<Self> {
        let mut entries = ptr::null_mut();
        // SAFETY: `entries` is a valid out-pointer. `getmntinfo_r_np` allocates
        // and initializes `count` statfs records, transfers that allocation to
        // this caller, and does not retain the out-pointer itself.
        let count = unsafe { getmntinfo_r_np(&mut entries, libc::MNT_NOWAIT) };
        if count <= 0 {
            // The API promises errno on failure; capture it before any call that
            // might modify thread-local errno.
            return Err(io::Error::last_os_error());
        }
        let Some(entries) = NonNull::new(entries) else {
            return Err(io::Error::other(
                "getmntinfo_r_np returned a positive count with a null allocation",
            ));
        };
        Ok(Self {
            entries,
            len: count as usize,
        })
    }

    fn entries(&self) -> &[libc::statfs] {
        // SAFETY: construction accepts only the non-null allocation and exact
        // initialized element count returned by `getmntinfo_r_np`. The slice is
        // bounded by `self`, which frees the allocation only in Drop.
        unsafe { std::slice::from_raw_parts(self.entries.as_ptr(), self.len) }
    }
}

impl Drop for MountTable {
    fn drop(&mut self) {
        // SAFETY: `getmntinfo_r_np` transferred this allocation to the caller
        // with the documented requirement that it be released using free(3).
        // This guard owns it exactly once and no derived slice outlives Drop.
        unsafe { libc::free(self.entries.as_ptr().cast()) };
    }
}

fn c_field_bytes<const N: usize>(field: &[c_char; N]) -> Vec<u8> {
    field
        .iter()
        .map(|value| *value as u8)
        .take_while(|value| *value != 0)
        .collect()
}

fn last_native_error(context: &str) -> io::Error {
    let error = io::Error::last_os_error();
    if error.raw_os_error().is_some_and(|code| code != 0) {
        error
    } else {
        io::Error::other(context.to_string())
    }
}

// Native lifecycle tests use test-owned process inspection and termination and
// are intentionally retained only in the private validation source.
