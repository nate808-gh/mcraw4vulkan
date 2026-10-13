use crate::abi::*;
use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};
use mcraw4vulkan_core::macfuse_location::{MacFuseRuntime, bounded_command};
use std::cell::Cell;
use std::ffi::{CStr, CString, OsStr, c_char, c_int, c_void};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub type FsResult<T> = Result<T, i32>;
#[derive(Clone, Debug)]
pub struct Metadata {
    pub inode: u64,
    pub len: u64,
    pub directory: bool,
    pub permissions: u16,
    pub links: u32,
    pub seconds: i64,
    pub nanos: u32,
}
#[derive(Clone, Debug)]
pub struct DirectoryEntry {
    pub name: std::ffi::OsString,
    pub metadata: Metadata,
    pub next_cookie: u64,
}
/// Safe filesystem operation surface. The buffer is borrowed only for this call;
/// the implementation owns inode, handle, cache and generated-frame lifetimes.
pub trait CallbackFileSystem: Send + Sync + 'static {
    fn getattr(&self, path: &Path) -> FsResult<Metadata>;
    fn open(&self, path: &Path) -> FsResult<u64>;
    fn release(&self, handle: u64) -> FsResult<()>;
    fn read(&self, path: &Path, handle: u64, offset: u64, buffer: &mut [u8]) -> FsResult<usize>;
    fn readdir(&self, path: &Path, cookie: u64) -> FsResult<Vec<DirectoryEntry>>;
}

struct Api {
    new: unsafe extern "C" fn(
        *mut Args,
        *const c_void,
        usize,
        *const Version,
        *mut c_void,
    ) -> *mut c_void,
    mount: unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int,
    run: unsafe extern "C" fn(*mut c_void) -> c_int,
    destroy: unsafe extern "C" fn(*mut c_void),
    add_arg: unsafe extern "C" fn(*mut Args, *const c_char) -> c_int,
    free_args: unsafe extern "C" fn(*mut Args),
    context: unsafe extern "C" fn() -> *mut Context,
    _library: Library,
    _framework: Library,
}
// Vendor asynchronous workers may outlive native destroy. Successful tables
// remain loaded, keyed by canonical library identity; failed loads never cache.
type ApiKey = (PathBuf, u64, u64, u64, i64, i64);
static APIS: OnceLock<Mutex<Vec<(ApiKey, &'static Api)>>> = OnceLock::new();
static ACTIVE: AtomicBool = AtomicBool::new(false);
thread_local! { static CALLBACK_API:Cell<Option<&'static Api>> = const {Cell::new(None)}; }
impl Api {
    fn load(runtime: &MacFuseRuntime) -> io::Result<&'static Self> {
        let metadata = std::fs::metadata(&runtime.library)?;
        let key = (
            runtime.library.clone(),
            metadata.dev(),
            metadata.ino(),
            metadata.len(),
            metadata.mtime(),
            metadata.mtime_nsec(),
        );
        let mut cache = APIS
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .map_err(|_| io::Error::other("native API cache poisoned"))?;
        if let Some((_, api)) = cache.iter().find(|(p, _)| p == &key) {
            return Ok(api);
        }
        if cache
            .iter()
            .any(|(prior, _)| prior.0 == key.0 && prior != &key)
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "provider client changed after loading; a fresh mount host is required",
            ));
        }
        // SAFETY: discovery verifies vendor code identity. Symbols resolve only
        // from this local handle; the table retains that handle for all workers.
        let library = unsafe {
            Library::open(
                Some(&runtime.library),
                RTLD_NOW | RTLD_LOCAL | libc::RTLD_FIRST,
            )
        }
        .map_err(|e| io::Error::new(io::ErrorKind::Unsupported, e))?;
        unsafe fn symbol<T: Copy>(l: &Library, n: &[u8]) -> io::Result<T> {
            // SAFETY: callers supply the header-verified signature of each name.
            unsafe {
                l.get::<T>(n)
                    .map(|p| *p)
                    .map_err(|e| io::Error::new(io::ErrorKind::Unsupported, e))
            }
        }
        // SAFETY: macFUSE 5.4 uses a delay-initialized MFMount dependency.
        // Load and pin its verified installation explicitly before construction;
        // packaging checks the same dependency path. This is deferred until Mount,
        // and never loads or registers a filesystem extension by itself.
        let framework = unsafe {
            Library::open(
                Some(&runtime.framework),
                RTLD_NOW | RTLD_LOCAL | libc::RTLD_FIRST,
            )
        }
        .map_err(|e| io::Error::new(io::ErrorKind::Unsupported, e))?;
        for name in [b"MFMount\0".as_slice(), b"MFChannelCreate\0".as_slice()] {
            let address = unsafe { symbol::<*const c_void>(&framework, name)? };
            verify_image(address, &runtime.framework)?;
        }
        // SAFETY: the vendor-verified FUSE3 version query has no arguments and
        // does not create a session. Reject unverified ABI generations before
        // passing any provider-specific structure.
        let version =
            unsafe { symbol::<unsafe extern "C" fn() -> c_int>(&library, b"fuse_version\0")?() };
        if !supported_abi(version) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("unsupported provider FUSE3 ABI {version}"),
            ));
        }
        // SAFETY: ABI checked against the matching macFUSE FUSE3 headers, including
        // the five-argument _fuse_new_31 (not its four-argument compatibility name).
        let api = unsafe {
            Self {
                new: symbol(&library, b"_fuse_new_31\0")?,
                mount: symbol(&library, b"fuse_mount\0")?,
                run: symbol(&library, b"fuse_loop\0")?,
                destroy: symbol(&library, b"fuse_destroy\0")?,
                add_arg: symbol(&library, b"fuse_opt_add_arg\0")?,
                free_args: symbol(&library, b"fuse_opt_free_args\0")?,
                context: symbol(&library, b"fuse_get_context\0")?,
                _library: library,
                _framework: framework,
            }
        };
        // Two-level Mach-O binding plus RTLD_FIRST prevents another provider's
        // identically named API from satisfying this table. Check every entry's
        // actual image before any constructor owns callbacks.
        for address in [
            api.new as *const c_void,
            api.mount as *const c_void,
            api.run as *const c_void,
            api.destroy as *const c_void,
            api.add_arg as *const c_void,
            api.free_args as *const c_void,
            api.context as *const c_void,
        ] {
            verify_image(address, &runtime.library)?;
        }
        let api = Box::leak(Box::new(api));
        cache.push((key, api));
        Ok(api)
    }
}
fn verify_image(address: *const c_void, expected: &Path) -> io::Result<()> {
    // SAFETY: address comes from a live selected library handle. dladdr writes
    // one Dl_info; its image name remains valid while that handle is retained.
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    if unsafe { libc::dladdr(address, &mut info) } == 0
        || info.dli_fname.is_null()
        || std::fs::canonicalize(OsStr::from_bytes(
            unsafe { CStr::from_ptr(info.dli_fname) }.to_bytes(),
        ))? != expected
    {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "macFUSE API or framework binding is outside the verified installation",
        ));
    }
    Ok(())
}
fn supported_abi(version: c_int) -> bool {
    version == 318
}
struct State {
    fs: Arc<dyn CallbackFileSystem>,
    initialized: AtomicBool,
    challenge: CString,
    challenge_seen: AtomicBool,
    release_native: AtomicBool,
    loop_returned: AtomicBool,
    native_destroyed: AtomicBool,
    failed: AtomicBool,
}
fn request<T>(f: impl FnOnce(&State) -> FsResult<T>) -> FsResult<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        CALLBACK_API.with(|slot| {
            let api = slot.get().ok_or(libc::EIO)?;
            // SAFETY: called synchronously on the single fuse_loop thread. The
            // library's TLS context owns private_data independently of requester
            // credentials (which FSKit need not supply). The worker retains State
            // until the loop stops and native destroy completes.
            let context = unsafe { (api.context)().as_ref() }.ok_or(libc::EIO)?;
            let state =
                unsafe { context.private_data.cast::<State>().as_ref() }.ok_or(libc::EIO)?;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(state)))
                .unwrap_or(Err(libc::EIO));
            if matches!(result, Err(libc::EIO)) {
                state.failed.store(true, Ordering::Release);
            }
            result
        })
    }))
    .unwrap_or(Err(libc::EIO))
}
unsafe fn path<'a>(p: *const c_char) -> FsResult<&'a Path> {
    if p.is_null() {
        return Err(libc::EINVAL);
    }
    // SAFETY: FUSE supplies a NUL-terminated path valid for this callback only.
    Ok(Path::new(OsStr::from_bytes(
        unsafe { CStr::from_ptr(p) }.to_bytes(),
    )))
}
fn code(r: FsResult<()>) -> c_int {
    r.map_or_else(|e| -e, |()| 0)
}
fn attr(m: &Metadata) -> libc::stat {
    // SAFETY: all-zero is valid for the system C stat structure; populate all
    // visible metadata, including Darwin birthtime, before handing it to FUSE.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    st.st_ino = m.inode;
    st.st_mode = (if m.directory {
        libc::S_IFDIR
    } else {
        libc::S_IFREG
    }) | m.permissions;
    st.st_nlink = m.links.min(u16::MAX as u32) as u16;
    // SAFETY: getuid/getgid take no pointers and are process identity queries.
    st.st_uid = unsafe { libc::getuid() };
    st.st_gid = unsafe { libc::getgid() };
    st.st_size = m.len.min(i64::MAX as u64) as i64;
    st.st_blocks = m.len.div_ceil(512).min(i64::MAX as u64) as i64;
    st.st_blksize = 512;
    st.st_atime = m.seconds;
    st.st_atime_nsec = i64::from(m.nanos);
    st.st_mtime = m.seconds;
    st.st_mtime_nsec = i64::from(m.nanos);
    st.st_ctime = m.seconds;
    st.st_ctime_nsec = i64::from(m.nanos);
    st.st_birthtime = m.seconds;
    st.st_birthtime_nsec = i64::from(m.nanos);
    st
}
unsafe extern "C" fn getattr(p: *const c_char, out: *mut libc::stat, _: *mut FileInfo) -> c_int {
    code(request(|s| {
        if !p.is_null() && unsafe { CStr::from_ptr(p) } == s.challenge.as_c_str() {
            // An unpredictable metadata request from the pinned root descriptor
            // binds startup to this callback owner, not to pathname occupancy.
            s.challenge_seen.store(true, Ordering::Release);
            return Err(libc::ENOENT);
        }
        let m = s.fs.getattr(unsafe { path(p) }?)?;
        if m.len > i64::MAX as u64 {
            return Err(libc::EOVERFLOW);
        }
        if out.is_null() {
            return Err(libc::EINVAL);
        }
        unsafe { out.write(attr(&m)) };
        Ok(())
    }))
}
unsafe extern "C" fn open(p: *const c_char, fi: *mut FileInfo) -> c_int {
    code(request(|s| {
        let fi = unsafe { fi.as_mut() }.ok_or(libc::EINVAL)?;
        if fi.flags & libc::O_TRUNC != 0 || fi.flags & libc::O_ACCMODE == libc::O_WRONLY {
            return Err(libc::EROFS);
        }
        fi.fh = s.fs.open(unsafe { path(p) }?)?;
        Ok(())
    }))
}
unsafe extern "C" fn release(_: *const c_char, fi: *mut FileInfo) -> c_int {
    code(request(|s| {
        s.fs.release(unsafe { fi.as_ref() }.ok_or(libc::EBADF)?.fh)
    }))
}
unsafe extern "C" fn read(
    p: *const c_char,
    b: *mut c_char,
    n: usize,
    o: libc::off_t,
    fi: *mut FileInfo,
) -> c_int {
    request(|s| {
        if o < 0 || n > c_int::MAX as usize || (b.is_null() && n != 0) {
            return Err(libc::EINVAL);
        }
        let h = unsafe { fi.as_ref() }.map_or(0, |f| f.fh);
        // SAFETY: FUSE provides exactly n writable bytes for this synchronous
        // request. No pointer or slice escapes; zero length needs no pointer.
        let buffer = if n == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(b.cast::<u8>(), n) }
        };
        let count = s.fs.read(unsafe { path(p) }?, h, o as u64, buffer)?;
        if count > n {
            return Err(libc::EIO);
        }
        Ok(count as c_int)
    })
    .unwrap_or_else(|e| -e)
}
unsafe extern "C" fn readdir(
    p: *const c_char,
    b: *mut c_void,
    fill: Fill,
    o: libc::off_t,
    _: *mut FileInfo,
    _: c_int,
) -> c_int {
    code(request(|s| {
        if o < 0 {
            return Err(libc::EINVAL);
        }
        for entry in s.fs.readdir(unsafe { path(p) }?, o as u64)? {
            let name = CString::new(entry.name.as_bytes()).map_err(|_| libc::EIO)?;
            let st = attr(&entry.metadata);
            let cookie = i64::try_from(entry.next_cookie).map_err(|_| libc::EOVERFLOW)?;
            // SAFETY: fill belongs to this request/provider; neither local name
            // nor stat is retained. Nonzero means full; resume uses next_cookie.
            if unsafe { fill(b, name.as_ptr(), &st, cookie, 0) } != 0 {
                break;
            }
        }
        Ok(())
    }))
}
unsafe extern "C" fn opendir(p: *const c_char, _: *mut FileInfo) -> c_int {
    code(request(|s| {
        if s.fs.getattr(unsafe { path(p) }?)?.directory {
            Ok(())
        } else {
            Err(libc::ENOTDIR)
        }
    }))
}
unsafe extern "C" fn access(p: *const c_char, mask: c_int) -> c_int {
    code(request(|s| {
        let m = s.fs.getattr(unsafe { path(p) }?)?;
        if mask & libc::W_OK != 0 {
            return Err(libc::EROFS);
        }
        if mask & libc::X_OK != 0 && m.permissions & 0o111 == 0 {
            return Err(libc::EACCES);
        }
        Ok(())
    }))
}
unsafe extern "C" fn getxattr(
    p: *const c_char,
    _: *const c_char,
    _: *mut c_char,
    _: usize,
) -> c_int {
    code(request(|s| {
        s.fs.getattr(unsafe { path(p) }?)?;
        Err(libc::ENOATTR)
    }))
}
unsafe extern "C" fn listxattr(p: *const c_char, _: *mut c_char, _: usize) -> c_int {
    code(request(|s| {
        s.fs.getattr(unsafe { path(p) }?)?;
        Ok(())
    }))
}
unsafe extern "C" fn statfs(_: *const c_char, p: *mut libc::statvfs) -> c_int {
    if p.is_null() {
        return -libc::EINVAL;
    }
    // SAFETY: callback receives one writable statvfs; zero initializes optional fields.
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    s.f_bsize = 512;
    s.f_frsize = 512;
    s.f_blocks = 1_048_576;
    s.f_bfree = s.f_blocks;
    s.f_bavail = s.f_blocks;
    s.f_files = 1_000_000;
    s.f_ffree = s.f_files;
    s.f_favail = s.f_files;
    s.f_namemax = 255;
    unsafe { p.write(s) };
    0
}
unsafe extern "C" fn init(_: *mut c_void, cfg: *mut ConfigPrefix) -> *mut c_void {
    request(|s| {
        if let Some(c) = unsafe { cfg.as_mut() } {
            c.use_ino = 1;
        }
        s.initialized.store(true, Ordering::Release);
        Ok((s as *const State).cast_mut().cast::<c_void>())
    })
    .unwrap_or(ptr::null_mut())
}
unsafe extern "C" fn ok(_: *const c_char, _: *mut FileInfo) -> c_int {
    0
}
unsafe extern "C" fn sync(_: *const c_char, _: c_int, _: *mut FileInfo) -> c_int {
    0
}
// Mutation signatures remain distinct; no unlike function-pointer casts.
unsafe extern "C" fn ro1(_: *const c_char) -> c_int {
    -libc::EROFS
}
unsafe extern "C" fn ro2(_: *const c_char, _: *const c_char) -> c_int {
    -libc::EROFS
}
unsafe extern "C" fn rename(_: *const c_char, _: *const c_char, _: u32) -> c_int {
    -libc::EROFS
}
unsafe extern "C" fn mkdir(_: *const c_char, _: libc::mode_t) -> c_int {
    -libc::EROFS
}
unsafe extern "C" fn mknod(_: *const c_char, _: libc::mode_t, _: libc::dev_t) -> c_int {
    -libc::EROFS
}
unsafe extern "C" fn chmod(_: *const c_char, _: libc::mode_t, _: *mut FileInfo) -> c_int {
    -libc::EROFS
}
unsafe extern "C" fn chown(_: *const c_char, _: u32, _: u32, _: *mut FileInfo) -> c_int {
    -libc::EROFS
}
unsafe extern "C" fn truncate(_: *const c_char, _: libc::off_t, _: *mut FileInfo) -> c_int {
    -libc::EROFS
}
unsafe extern "C" fn write(
    _: *const c_char,
    _: *const c_char,
    _: usize,
    _: libc::off_t,
    _: *mut FileInfo,
) -> c_int {
    -libc::EROFS
}
unsafe extern "C" fn setxattr(
    _: *const c_char,
    _: *const c_char,
    _: *const c_char,
    _: usize,
    _: c_int,
) -> c_int {
    -libc::EROFS
}
unsafe extern "C" fn utimens(
    _: *const c_char,
    _: *const libc::timespec,
    _: *mut FileInfo,
) -> c_int {
    -libc::EROFS
}
unsafe extern "C" fn setattr(
    _: *const c_char,
    _: *mut c_void,
    _: c_int,
    _: *mut FileInfo,
) -> c_int {
    -libc::EROFS
}
unsafe extern "C" fn chflags(_: *const c_char, _: *mut FileInfo, _: u32) -> c_int {
    -libc::EROFS
}
fn operations() -> OperationsTail {
    OperationsTail {
        mknod: Some(mknod),
        mkdir: Some(mkdir),
        unlink: Some(ro1),
        rmdir: Some(ro1),
        symlink: Some(ro2),
        rename: Some(rename),
        link: Some(ro2),
        chmod: Some(chmod),
        chown: Some(chown),
        truncate: Some(truncate),
        open: Some(open),
        read: Some(read),
        write: Some(write),
        statfs: Some(statfs),
        flush: Some(ok),
        release: Some(release),
        fsync: Some(sync),
        setxattr: Some(setxattr),
        getxattr: Some(getxattr),
        listxattr: Some(listxattr),
        removexattr: Some(ro2),
        opendir: Some(opendir),
        readdir: Some(readdir),
        releasedir: Some(ok),
        fsyncdir: Some(sync),
        init: Some(init),
        access: Some(access),
        create: Some(chmod),
        utimens: Some(utimens),
        ..Default::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnmountOutcome {
    Unmounted,
    AlreadyUnmounted,
}
pub struct MacFuseSession {
    directory: MountDirectory,
    worker: Option<JoinHandle<io::Result<i32>>>,
    mountpoint: PathBuf,
    binding: Option<MountBinding>,
    probe: Option<JoinHandle<io::Result<MountBinding>>>,
    state: Arc<State>,
    cancellation: Arc<AtomicBool>,
}
impl MacFuseSession {
    pub fn mount(
        fs: Arc<dyn CallbackFileSystem>,
        directory: MountDirectory,
        runtime: MacFuseRuntime,
        volume: &str,
        cancellation: Arc<AtomicBool>,
    ) -> io::Result<Self> {
        if cancellation.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "DNG mount canceled",
            ));
        }
        if ACTIVE.swap(true, Ordering::AcqRel) {
            return Err(io::Error::other(
                "one native session per mount process is required; launch each clip through its owned CLI process",
            ));
        }
        let result = Self::start(fs, directory, runtime, volume, cancellation);
        if result.is_err() {
            ACTIVE.store(false, Ordering::Release);
        }
        result
    }
    fn start(
        fs: Arc<dyn CallbackFileSystem>,
        mut directory: MountDirectory,
        runtime: MacFuseRuntime,
        volume: &str,
        cancellation: Arc<AtomicBool>,
    ) -> io::Result<Self> {
        // Refuse mounting before creating native work on systems without the
        // instance-bound cleanup API. Preview/export can still run there.
        let _ = instance_unmount_api()?;
        let api = Api::load(&runtime)?;
        let target = directory.path.clone();
        if mounted(&target)? {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "mountpoint is already mounted",
            ));
        }
        let cpath = CString::new(target.as_os_str().as_bytes())?;
        let options = mount_options(volume);
        let mut random = [0u8; 24];
        // SAFETY: arc4random_buf initializes precisely this owned byte array.
        unsafe { libc::arc4random_buf(random.as_mut_ptr().cast(), random.len()) };
        let challenge = CString::new(format!(
            "{}.mcraw4vulkan-owner-{}",
            std::path::MAIN_SEPARATOR,
            random
                .iter()
                .map(|v| format!("{v:02x}"))
                .collect::<String>()
        ))?;
        let state = Arc::new(State {
            fs,
            initialized: AtomicBool::new(false),
            challenge,
            challenge_seen: AtomicBool::new(false),
            release_native: AtomicBool::new(false),
            loop_returned: AtomicBool::new(false),
            native_destroyed: AtomicBool::new(true),
            failed: AtomicBool::new(false),
        });
        let worker_state = state.clone();
        let worker_cancellation = cancellation.clone();
        let worker_target = target.clone();
        directory.cleanup_on_drop = false;
        let worker = thread::Builder::new()
            .name("dng-fuse3".into())
            .spawn(move || {
                CALLBACK_API.with(|s| s.set(Some(api)));
                let mut args = Args {
                    argc: 0,
                    argv: ptr::null_mut(),
                    allocated: 0,
                };
                for arg in ["mcraw4vulkan", "-o", &options] {
                    let arg = CString::new(arg)?;
                    // SAFETY: same provider allocates/frees argv; it copies arg.
                    if unsafe { (api.add_arg)(&mut args, arg.as_ptr()) } != 0 {
                        unsafe { (api.free_args)(&mut args) };
                        return Err(io::Error::other("FUSE argument allocation failed"));
                    }
                }
                let mac = MacOperations {
                    getattr: Some(getattr),
                    setattr: Some(setattr),
                    tail: operations(),
                    chflags: Some(chflags),
                    setvolname: Some(ro1),
                    monitor: None,
                    statx: None,
                };
                let op = (&mac as *const MacOperations).cast();
                let size = std::mem::size_of_val(&mac);
                let version = Version {
                    major: 3,
                    minor: 18,
                    hotfix: 2,
                    flags: 0,
                };
                // SAFETY: typed provider table, version flags explicitly disable
                // Darwin extensions. State lives through loop and destroy. No
                // per-session signal handlers overwrite the application lifecycle.
                if worker_cancellation.load(Ordering::Acquire) {
                    unsafe { (api.free_args)(&mut args) };
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "DNG mount canceled before native creation",
                    ));
                }
                worker_state
                    .native_destroyed
                    .store(false, Ordering::Release);
                let raw = unsafe {
                    (api.new)(
                        &mut args,
                        op,
                        size,
                        &version,
                        Arc::as_ptr(&worker_state).cast_mut().cast(),
                    )
                };
                unsafe { (api.free_args)(&mut args) };
                if raw.is_null() {
                    worker_state.native_destroyed.store(true, Ordering::Release);
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "FUSE3 constructor rejected initialization",
                    ));
                }
                let mount = unsafe { (api.mount)(raw, cpath.as_ptr()) };
                if mount != 0 {
                    let error = io::Error::last_os_error();
                    unsafe { (api.destroy)(raw) };
                    worker_state.native_destroyed.store(true, Ordering::Release);
                    return Err(io::Error::new(
                        error.kind(),
                        format!("FUSE3 mount rejected: {error}"),
                    ));
                }
                let result = unsafe { (api.run)(raw) };
                worker_state.loop_returned.store(true, Ordering::Release);
                // Native unmount is deliberately not used: macFUSE can capture a
                // DADisk by pathname during late startup. Only the instance-bound
                // owner may detach.
                // Destroy closes this session's transport after reconciliation.
                while !worker_state.release_native.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_millis(20));
                }
                let detached = !mounted(&worker_target)?;
                unsafe { (api.destroy)(raw) };
                worker_state.native_destroyed.store(true, Ordering::Release);
                CALLBACK_API.with(|s| s.set(None));
                if detached && !worker_state.failed.load(Ordering::Acquire) && result == 0 {
                    Ok(result)
                } else {
                    Err(io::Error::other(format!(
                        "macFUSE FSKit loop return={result}, detached={detached}"
                    )))
                }
            });
        let worker = match worker {
            Ok(worker) => worker,
            Err(error) => {
                directory.cleanup_on_drop = true; // No worker was started.
                return Err(error);
            }
        };
        let mut session = Self {
            directory,
            worker: Some(worker),
            mountpoint: target,
            binding: None,
            probe: None,
            state: state.clone(),
            cancellation,
        };
        let end = Instant::now() + Duration::from_secs(20);
        loop {
            if session.cancellation.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "DNG mount canceled",
                ));
            }
            if session.worker.as_ref().is_some_and(JoinHandle::is_finished) {
                session.finish()?;
                return Err(io::Error::new(
                    if state.initialized.load(Ordering::Acquire) {
                        io::ErrorKind::Other
                    } else {
                        io::ErrorKind::Unsupported
                    },
                    "FUSE3 startup ended before ready",
                ));
            }
            if state.loop_returned.load(Ordering::Acquire) && !mounted(&session.mountpoint)? {
                state.release_native.store(true, Ordering::Release);
            }
            session.poll_binding()?;
            if state.initialized.load(Ordering::Acquire)
                && session.binding.is_some()
                && session.is_mounted()?
            {
                verify_fskit_mount(&session.mountpoint)?;
                // This final acquire is the publication boundary. Cancellation is
                // latched forever; a subsequent cancellation is owned shutdown.
                if session.cancellation.load(Ordering::Acquire) {
                    continue;
                }
                return Ok(session);
            }
            if Instant::now() >= end {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "FUSE3 startup did not become ready within 20 seconds",
                ));
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
    fn poll_binding(&mut self) -> io::Result<()> {
        if self.binding.is_some() {
            return Ok(());
        }
        if self.probe.as_ref().is_some_and(JoinHandle::is_finished) {
            let result = self
                .probe
                .take()
                .unwrap()
                .join()
                .map_err(|_| io::Error::other("ownership probe panicked"))?;
            let binding = result?;
            self.binding = Some(binding);
        } else if self.probe.is_none()
            && !self.state.loop_returned.load(Ordering::Acquire)
            && mounted(&self.mountpoint)?
        {
            let target = self.mountpoint.clone();
            let state = self.state.clone();
            self.probe = Some(
                thread::Builder::new()
                    .name("dng-mount-identity".into())
                    .spawn(move || bind_mount(&target, &state))?,
            );
        }
        Ok(())
    }
    fn reconcile_pending(&mut self) {
        let mut announced = self.worker.as_ref().is_some_and(|w| !w.is_finished())
            || self.probe.as_ref().is_some_and(|p| !p.is_finished());
        if announced {
            eprintln!(
                "DNG mount cleanup pending: macFUSE FSKit; target={}; owner={}; waiting for owned native work",
                self.mountpoint.display(),
                std::process::id()
            );
        }
        // Runs only in the existing isolated CLI mount owner, never the GUI.
        // A timed-out native call remains here, visibly owned. No fallback or
        // success is possible until both workers and the volume are reconciled.
        loop {
            let _ = self.poll_binding();
            let result = self.try_unmount();
            if result.is_ok() && self.probe.is_none() {
                break;
            }
            if !announced {
                eprintln!(
                    "DNG mount cleanup pending: macFUSE FSKit; target={}; owner={}; {}",
                    self.mountpoint.display(),
                    std::process::id(),
                    result
                        .err()
                        .map(|e| e.to_string())
                        .unwrap_or_else(|| "identity probe still running".into())
                );
                announced = true;
            }
            thread::sleep(Duration::from_millis(100));
        }
    }
    pub fn instance_id(&self) -> io::Result<String> {
        self.is_mounted()?;
        self.binding
            .as_ref()
            .map(|b| format_identity(&b.identity))
            .ok_or_else(|| io::Error::other("mount ownership is not established"))
    }
    pub fn mountpoint(&self) -> &Path {
        &self.mountpoint
    }
    pub fn is_mounted(&self) -> io::Result<bool> {
        let current = mount_identity(&self.mountpoint)?;
        owned_observation(self.binding.as_ref().map(|b| &b.identity), current.as_ref())
    }
    pub fn try_unmount(&mut self) -> io::Result<UnmountOutcome> {
        let present = self.is_mounted()?;
        if present {
            let binding = self.binding.as_ref().ok_or_else(|| {
                io::Error::other("cleanup pending: mount ownership is not established")
            })?;
            unmount_binding(binding)?;
        }
        if self.is_mounted()? {
            return Err(io::Error::other("unmount returned before detachment"));
        }
        self.state.release_native.store(true, Ordering::Release);
        self.finish()?;
        Ok(if present {
            UnmountOutcome::Unmounted
        } else {
            UnmountOutcome::AlreadyUnmounted
        })
    }
    fn finish(&mut self) -> io::Result<()> {
        let end = Instant::now() + Duration::from_secs(10);
        while self.worker.as_ref().is_some_and(|w| !w.is_finished()) {
            if Instant::now() >= end {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "native worker has not exited; retaining its state and library in this mount process",
                ));
            }
            thread::sleep(Duration::from_millis(20));
        }
        if let Some(w) = self.worker.take() {
            let result = w
                .join()
                .map_err(|_| io::Error::other("FUSE3 worker panicked"))?;
            result?;
        }
        if self.probe.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "ownership probe cleanup is still pending",
            ));
        }
        if self.state.native_destroyed.load(Ordering::Acquire) && !mounted(&self.mountpoint)? {
            self.directory.cleanup();
        } else if self.directory.cleanup_pending {
            self.directory.cleanup_pending = false;
            eprintln!(
                "DNG mount folder retained: {}: native teardown or detachment is not confirmed",
                self.mountpoint.display()
            );
        }
        ACTIVE.store(false, Ordering::Release);
        Ok(())
    }

    pub fn wait_until_unmounted(mut self) -> io::Result<()> {
        while self.worker.as_ref().is_some_and(|w| !w.is_finished()) {
            let present = self.is_mounted()?;
            if !present {
                self.state.release_native.store(true, Ordering::Release);
            }
            if self.state.failed.load(Ordering::Acquire) {
                return Err(io::Error::other(
                    "DNG callback failed; reconciling this mount before exit",
                ));
            }
            if self.cancellation.load(Ordering::Acquire) {
                match self.try_unmount() {
                    Ok(_) => return Ok(()),
                    Err(e) if e.kind() == io::ErrorKind::ResourceBusy => {
                        thread::sleep(Duration::from_secs(1));
                        eprintln!("DNG volume is busy; close its files and retry unmount: {e}");
                    }
                    Err(e) => return Err(e),
                }
            }
            thread::sleep(Duration::from_millis(50));
        }
        self.finish()
    }
}
impl Drop for MacFuseSession {
    fn drop(&mut self) {
        if self.worker.is_some() || self.probe.is_some() || ACTIVE.load(Ordering::Acquire) {
            // Dropping a rejected attempt is not user cancellation. The request
            // token belongs to the CLI/GUI owner and is never reset or synthesized here.
            self.reconcile_pending();
        }
    }
}

// The source marker identifies a reusable backing leaf, not a mounted instance.
// Removal also requires its original pinned identity and a reconciled session.
const LEAF_OWNER: &CStr = c"org.mcraw4vulkan.mount-source";
fn leaf_key(source: &Path) -> Vec<u8> {
    let mut key = b"v1\0".to_vec();
    key.extend_from_slice(source.as_os_str().as_bytes());
    key
}
fn ordinary_leaf(path: &Path) -> io::Result<std::fs::File> {
    if mounted(path)? {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "mount leaf is occupied",
        ));
    }
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(path)?;
    let metadata = file.metadata()?;
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("mount leaf has no parent"))?;
    // SAFETY: geteuid has no pointer arguments or process mutation.
    if metadata.uid() != unsafe { libc::geteuid() }
        || metadata.dev() != std::fs::metadata(parent)?.dev()
    {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "mount leaf is not a same-user directory on its parent's filesystem",
        ));
    }
    Ok(file)
}
pub fn reusable_mount_directory(path: &Path, source: &Path) -> io::Result<bool> {
    let file = match ordinary_leaf(path) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };
    let expected = leaf_key(source);
    if expected.len() > 8192 {
        return Ok(false);
    }
    let mut value = vec![0u8; expected.len() + 1];
    // SAFETY: the pinned ordinary-directory descriptor, terminated attribute
    // name and writable buffer remain valid throughout this bounded xattr call.
    let n = unsafe {
        libc::fgetxattr(
            file.as_raw_fd(),
            LEAF_OWNER.as_ptr(),
            value.as_mut_ptr().cast(),
            value.len(),
            0,
            0,
        )
    };
    let matches = n == expected.len() as isize && value[..expected.len()] == expected;
    if matches && std::fs::read_dir(path)?.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "the existing DNG mount folder contains files; preserve them and empty the folder before mounting",
        ));
    }
    Ok(matches)
}
/// Pins the ordinary backing directory before native work starts. This is not
/// mounted-instance authority. The session disarms Drop until native teardown.
pub struct MountDirectory {
    path: PathBuf,
    parent: std::fs::File,
    leaf: std::fs::File,
    name: CString,
    key: Vec<u8>,
    owned: bool,
    marker_required: bool,
    cleanup_on_drop: bool,
    cleanup_pending: bool,
}
impl MountDirectory {
    pub fn prepare(path: &Path, source: &Path) -> io::Result<Self> {
        let key = leaf_key(source);
        if key.len() > 8192 || !path.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid mount leaf or source identity",
            ));
        }
        let name = path
            .file_name()
            .ok_or_else(|| io::Error::other("mount leaf has no name"))?;
        let parent_path = std::fs::canonicalize(
            path.parent()
                .ok_or_else(|| io::Error::other("mount leaf has no parent"))?,
        )?;
        let path = parent_path.join(name);
        let name = CString::new(name.as_bytes())?;
        let parent = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(&parent_path)?;
        if mounted(&path)? {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "mount leaf is occupied",
            ));
        }
        // Exclusive creation never adopts or marks an existing user directory.
        let created = match std::fs::create_dir(&path) {
            Ok(()) => true,
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => false,
            Err(e) => return Err(e),
        };
        let leaf = match ordinary_leaf(&path) {
            Ok(leaf) => leaf,
            Err(error) => {
                // Creation and open are not atomic. Without the original pinned
                // identity, reopening this name cannot authorize rollback.
                eprintln!(
                    "DNG mount folder retained: {}: backing identity was not established during preparation: {error}",
                    path.display()
                );
                return Err(error);
            }
        };
        // Serialize cooperating mount owners even while native startup has not
        // published a mount. Holding the backing vnode does not pin the volume.
        if unsafe { libc::flock(leaf.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let error = io::Error::last_os_error();
            // No rollback may delete beneath a cooperating owner holding this
            // leaf, even if this attempt originally created the directory.
            eprintln!(
                "DNG mount folder retained: {}: backing directory lock was not acquired during preparation: {error}",
                path.display()
            );
            return Err(error);
        }
        let owned = created || leaf_owner_matches(&leaf, &key);
        let mut directory = Self {
            path,
            parent,
            leaf,
            name,
            key,
            owned,
            marker_required: !created,
            cleanup_on_drop: true,
            cleanup_pending: true,
        };
        if std::fs::read_dir(&directory.path)?.next().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::DirectoryNotEmpty,
                "mount folder contains files",
            ));
        }
        if created {
            // SAFETY: descriptor, terminated name and bounded value are live.
            // If marking fails, only this exclusively created, pinned empty
            // directory can be removed, before any native startup is possible.
            if unsafe {
                libc::fsetxattr(
                    directory.leaf.as_raw_fd(),
                    LEAF_OWNER.as_ptr(),
                    directory.key.as_ptr().cast(),
                    directory.key.len(),
                    0,
                    libc::XATTR_CREATE,
                )
            } != 0
            {
                return Err(io::Error::last_os_error());
            }
            directory.marker_required = true;
        }
        Ok(directory)
    }

    fn remove_empty(&self) -> io::Result<bool> {
        if mounted(&self.path)? {
            return Err(io::Error::other(
                "a filesystem still occupies the mount path",
            ));
        }
        let parent_now = std::fs::symlink_metadata(self.path.parent().unwrap())?;
        let parent = self.parent.metadata()?;
        if !parent_now.is_dir() || !same_directory(&parent_now, &parent) {
            return Err(io::Error::other("backing parent identity changed"));
        }
        // SAFETY: the pinned parent and one normal, terminated leaf name remain
        // live. O_NOFOLLOW refuses replacement symlinks, including broken ones.
        let fd = unsafe {
            libc::openat(
                self.parent.as_raw_fd(),
                self.name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            let error = io::Error::last_os_error();
            return if error.kind() == io::ErrorKind::NotFound {
                Ok(false)
            } else {
                Err(error)
            };
        }
        let current = unsafe { std::fs::File::from_raw_fd(fd) };
        let expected = self.leaf.metadata()?;
        if !same_directory(&current.metadata()?, &expected)
            || (self.marker_required && !leaf_owner_matches(&current, &self.key))
        {
            return Err(io::Error::other(
                "backing leaf identity or source marker changed",
            ));
        }
        let mut final_stat: libc::stat = unsafe { std::mem::zeroed() };
        // Recheck the name relative to the same parent immediately before rmdir.
        // POSIX has no inode-conditional rmdir: an uncooperative writer can still
        // race the final check/unlinkat. This is not a directory security boundary.
        if unsafe {
            libc::fstatat(
                self.parent.as_raw_fd(),
                self.name.as_ptr(),
                &mut final_stat,
                libc::AT_SYMLINK_NOFOLLOW,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        if final_stat.st_mode & libc::S_IFMT != libc::S_IFDIR
            || final_stat.st_dev as u64 != expected.dev()
            || final_stat.st_ino != expected.ino()
        {
            return Err(io::Error::other("backing leaf changed before removal"));
        }
        // Nonrecursive removal is the emptiness check, including hidden entries.
        // No child, marker, or parent is ever removed to make this succeed.
        if unsafe {
            libc::unlinkat(
                self.parent.as_raw_fd(),
                self.name.as_ptr(),
                libc::AT_REMOVEDIR,
            )
        } != 0
        {
            let error = io::Error::last_os_error();
            return if error.kind() == io::ErrorKind::NotFound {
                Ok(false)
            } else {
                Err(error)
            };
        }
        Ok(true)
    }

    fn cleanup(&mut self) {
        if !self.cleanup_pending {
            return;
        }
        self.cleanup_pending = false;
        let outcome = if self.owned {
            self.remove_empty()
        } else {
            Err(io::Error::other(
                "directory was supplied by the user, without matching application ownership",
            ))
        };
        match outcome {
            Ok(true) => eprintln!("DNG mount folder removed: {}", self.path.display()),
            Ok(false) => eprintln!("DNG mount folder already absent: {}", self.path.display()),
            Err(error) => eprintln!(
                "DNG mount folder retained: {}: {error}",
                self.path.display()
            ),
        }
    }
}
impl Drop for MountDirectory {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            self.cleanup();
        }
    }
}
fn same_directory(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    left.dev() == right.dev() && left.ino() == right.ino()
}
fn leaf_owner_matches(file: &std::fs::File, expected: &[u8]) -> bool {
    let mut value = vec![0u8; expected.len() + 1];
    // SAFETY: the descriptor, attribute name and bounded output buffer are live.
    let n = unsafe {
        libc::fgetxattr(
            file.as_raw_fd(),
            LEAF_OWNER.as_ptr(),
            value.as_mut_ptr().cast(),
            value.len(),
            0,
            0,
        )
    };
    n == expected.len() as isize && value[..expected.len()] == *expected
}

struct MountBinding {
    identity: MountIdentity,
    // O_EVTONLY pins the actual root vnode without making normal eject busy.
    root: std::fs::File,
}
fn bind_mount(target: &Path, state: &State) -> io::Result<MountBinding> {
    let root = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_EVTONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(target)?;
    let relative = CString::new(&state.challenge.as_bytes()[1..])?;
    // SAFETY: both stat structures are initialized writable C objects; the
    // descriptor and challenge string remain live through these synchronous calls.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    state.challenge_seen.store(false, Ordering::Release);
    let result = unsafe {
        libc::fstatat(
            root.as_raw_fd(),
            relative.as_ptr(),
            &mut st,
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if result != -1
        || io::Error::last_os_error().raw_os_error() != Some(libc::ENOENT)
        || !state.challenge_seen.load(Ordering::Acquire)
    {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "target did not answer this mount owner's challenge; refusing path cleanup",
        ));
    }
    let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstatfs(root.as_raw_fd(), &mut stat) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let identity = identity_from_statfs(&stat);
    owned_observation(Some(&identity), mount_identity(target)?.as_ref())?;
    Ok(MountBinding { identity, root })
}
type Funmount = unsafe extern "C" fn(c_int, c_int) -> c_int;
fn instance_unmount_api() -> io::Result<(Library, Funmount)> {
    // Deferred OS API: no new eager symbol on older supported systems. The
    // descriptor pins this filesystem, so a replacement path cannot be targeted.
    // No path-based fallback is allowed if the instance operation is unavailable.
    let library =
        unsafe { Library::open(Some("/usr/lib/libSystem.B.dylib"), RTLD_NOW | RTLD_LOCAL) }
            .map_err(io::Error::other)?;
    let function = unsafe { library.get::<Funmount>(b"funmount\0") }.map_err(|_| {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "DNG mounting requires macOS support for instance-bound unmount; other application features remain available",
        )
    })?;
    let function = *function;
    Ok((library, function))
}
fn unmount_binding(binding: &MountBinding) -> io::Result<()> {
    let (_library, function) = instance_unmount_api()?;
    // SAFETY: the pinned root descriptor and verified OS function remain live
    // through this call. MNT_NOBLOCK requests normal, non-forced instance unmount.
    if unsafe { function(binding.root.as_raw_fd(), libc::MNT_NOBLOCK) } == 0 {
        Ok(())
    } else {
        let e = io::Error::last_os_error();
        if e.raw_os_error() == Some(libc::EBUSY) {
            Err(io::Error::new(io::ErrorKind::ResourceBusy, e))
        } else {
            Err(e)
        }
    }
}
unsafe extern "C" {
    fn getmntinfo_r_np(entries: *mut *mut libc::statfs, flags: c_int) -> c_int;
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct MountIdentity {
    fsid: [i32; 2],
    source: Vec<u8>,
    kind: Vec<u8>,
}
fn owned_observation(
    owned: Option<&MountIdentity>,
    current: Option<&MountIdentity>,
) -> io::Result<bool> {
    if current.is_some() && (owned.is_none() || current != owned) {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "mount identity changed",
        ));
    }
    Ok(current.is_some())
}

fn mount_identity(path: &Path) -> io::Result<Option<MountIdentity>> {
    let mut entries = ptr::null_mut();
    // SAFETY: public NOWAIT mount-table snapshot never resolves through a
    // possibly blocked mount. Its caller-owned allocation is freed exactly once.
    let count = unsafe { getmntinfo_r_np(&mut entries, libc::MNT_NOWAIT) };
    let Some(entries) = NonNull::new(entries) else {
        return Err(io::Error::last_os_error());
    };
    if count < 0 {
        unsafe { libc::free(entries.as_ptr().cast()) };
        return Err(io::Error::last_os_error());
    }
    let found = unsafe { std::slice::from_raw_parts(entries.as_ptr(), count as usize) }
        .iter()
        .find_map(|s| {
            let bytes: Vec<u8> = s
                .f_mntonname
                .iter()
                .map(|c| *c as u8)
                .take_while(|c| *c != 0)
                .collect();
            (bytes == path.as_os_str().as_bytes()).then(|| identity_from_statfs(s))
        });
    unsafe { libc::free(entries.as_ptr().cast()) };
    Ok(found)
}

fn identity_from_statfs(s: &libc::statfs) -> MountIdentity {
    fn bytes(a: &[c_char]) -> Vec<u8> {
        a.iter().map(|c| *c as u8).take_while(|c| *c != 0).collect()
    }
    MountIdentity {
        // SAFETY: Darwin's fsid is exactly two 32-bit words, size checked by transmute.
        fsid: unsafe { std::mem::transmute::<libc::fsid_t, [i32; 2]>(s.f_fsid) },
        source: bytes(&s.f_mntfromname),
        kind: bytes(&s.f_fstypename),
    }
}

pub fn mounted(path: &Path) -> io::Result<bool> {
    Ok(mount_identity(path)?.is_some())
}

fn mount_options(volume: &str) -> String {
    format!(
        "ro,quiet,volname={},backend=fskit",
        volume.replace([',', '\0'], "_")
    )
}
fn verify_fskit_mount(path: &Path) -> io::Result<()> {
    let table = bounded_command("/sbin/mount", &[])?;
    if !table.status.success() {
        return Err(io::Error::other("could not observe FSKit mount table"));
    }
    let text = String::from_utf8_lossy(&table.stdout);
    let suffix = format!(" on {} (", path.display());
    let line = text
        .lines()
        .find(|l| l.contains(&suffix))
        .ok_or_else(|| io::Error::other("ready FSKit mount absent from mount table"))?;
    if !line.starts_with("macfuse://") || !line.contains("(macfuse,") || !line.contains(", fskit,")
    {
        return Err(io::Error::other(format!(
            "unexpected FSKit mount identity: {line}"
        )));
    }
    Ok(())
}

/// Mount-table identity only; never touches files through the mounted volume.
pub fn mount_instance_id(path: &Path) -> io::Result<Option<String>> {
    Ok(mount_identity(path)?.as_ref().map(format_identity))
}
fn format_identity(m: &MountIdentity) -> String {
    let bytes = m
        .kind
        .iter()
        .copied()
        .chain(std::iter::once(0))
        .chain(m.source.iter().copied());
    format!(
        "owned-v2:{:08x}{:08x}-{}",
        m.fsid[0],
        m.fsid[1],
        bytes.map(|b| format!("{b:02x}")).collect::<String>()
    )
}
/// A registry request must supply the identity published by the owning session.
/// Opening a replacement path cannot authorize it: the pinned descriptor must
/// match that saved identity before the instance-bound unmount operation.
pub fn unmount_instance(path: &Path, expected: &str) -> io::Result<UnmountOutcome> {
    if !expected.starts_with("owned-v2:") {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "legacy or missing mount ownership; use its owning process or Finder",
        ));
    }
    if !mounted(path)? {
        return Ok(UnmountOutcome::AlreadyUnmounted);
    }
    let root = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_EVTONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(path)?;
    // SAFETY: valid live descriptor and one writable system statfs structure.
    let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstatfs(root.as_raw_fd(), &mut stat) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let binding = MountBinding {
        root,
        identity: identity_from_statfs(&stat),
    };
    if format_identity(&binding.identity) != expected {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "registered mount was replaced; refusing unmount",
        ));
    }
    unmount_binding(&binding)?;
    Ok(UnmountOutcome::Unmounted)
}
