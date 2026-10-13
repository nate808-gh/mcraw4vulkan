//! macFUSE FUSE_USE_VERSION=31, Darwin extensions disabled; checked ABI318.
use libc::{c_char, c_int, c_void, off_t, stat, statvfs};

#[repr(C)]
pub(super) struct Args {
    pub argc: c_int,
    pub argv: *mut *mut c_char,
    pub allocated: c_int,
}
#[repr(C)]
pub(super) struct Version {
    pub major: u32,
    pub minor: u32,
    pub hotfix: u32,
    pub flags: u32,
}
#[repr(C)]
pub(super) struct FileInfo {
    pub flags: i32,
    pub bits: u32,
    pub padding: [u32; 2],
    pub fh: u64,
    pub lock_owner: u64,
    pub poll_events: u32,
    pub backing_id: i32,
    pub compat_flags: u64,
    pub reserved: [u64; 2],
}
#[repr(C)]
pub(super) struct Context {
    pub fuse: *mut c_void,
    pub uid: u32,
    pub gid: u32,
    pub pid: i32,
    pub private_data: *mut c_void,
    pub umask: u16,
}
// Only the stable prefix is accessed. Remaining provider config/connection
// fields are opaque; no negotiation fields are guessed from a version number.
#[repr(C)]
pub(super) struct ConfigPrefix {
    pub ownership: [u32; 6],
    pub entry_timeout: f64,
    pub negative_timeout: f64,
    pub attr_timeout: f64,
    pub intr: i32,
    pub intr_signal: i32,
    pub remember: i32,
    pub hard_remove: i32,
    pub use_ino: i32,
}
pub(super) type Fill =
    unsafe extern "C" fn(*mut c_void, *const c_char, *const stat, off_t, c_int) -> c_int;
pub(super) type Getattr = unsafe extern "C" fn(*const c_char, *mut stat, *mut FileInfo) -> c_int;
type Unused = Option<unsafe extern "C" fn()>;

#[repr(C)]
#[derive(Default)]
pub(super) struct OperationsTail {
    pub readlink: Unused,
    pub mknod: Option<unsafe extern "C" fn(*const c_char, libc::mode_t, libc::dev_t) -> c_int>,
    pub mkdir: Option<unsafe extern "C" fn(*const c_char, libc::mode_t) -> c_int>,
    pub unlink: Option<unsafe extern "C" fn(*const c_char) -> c_int>,
    pub rmdir: Option<unsafe extern "C" fn(*const c_char) -> c_int>,
    pub symlink: Option<unsafe extern "C" fn(*const c_char, *const c_char) -> c_int>,
    pub rename: Option<unsafe extern "C" fn(*const c_char, *const c_char, u32) -> c_int>,
    pub link: Option<unsafe extern "C" fn(*const c_char, *const c_char) -> c_int>,
    pub chmod: Option<unsafe extern "C" fn(*const c_char, libc::mode_t, *mut FileInfo) -> c_int>,
    pub chown: Option<unsafe extern "C" fn(*const c_char, u32, u32, *mut FileInfo) -> c_int>,
    pub truncate: Option<unsafe extern "C" fn(*const c_char, off_t, *mut FileInfo) -> c_int>,
    pub open: Option<unsafe extern "C" fn(*const c_char, *mut FileInfo) -> c_int>,
    pub read: Option<
        unsafe extern "C" fn(*const c_char, *mut c_char, usize, off_t, *mut FileInfo) -> c_int,
    >,
    pub write: Option<
        unsafe extern "C" fn(*const c_char, *const c_char, usize, off_t, *mut FileInfo) -> c_int,
    >,
    pub statfs: Option<unsafe extern "C" fn(*const c_char, *mut statvfs) -> c_int>,
    pub flush: Option<unsafe extern "C" fn(*const c_char, *mut FileInfo) -> c_int>,
    pub release: Option<unsafe extern "C" fn(*const c_char, *mut FileInfo) -> c_int>,
    pub fsync: Option<unsafe extern "C" fn(*const c_char, c_int, *mut FileInfo) -> c_int>,
    pub setxattr: Option<
        unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, usize, c_int) -> c_int,
    >,
    pub getxattr:
        Option<unsafe extern "C" fn(*const c_char, *const c_char, *mut c_char, usize) -> c_int>,
    pub listxattr: Option<unsafe extern "C" fn(*const c_char, *mut c_char, usize) -> c_int>,
    pub removexattr: Option<unsafe extern "C" fn(*const c_char, *const c_char) -> c_int>,
    pub opendir: Option<unsafe extern "C" fn(*const c_char, *mut FileInfo) -> c_int>,
    pub readdir: Option<
        unsafe extern "C" fn(
            *const c_char,
            *mut c_void,
            Fill,
            off_t,
            *mut FileInfo,
            c_int,
        ) -> c_int,
    >,
    pub releasedir: Option<unsafe extern "C" fn(*const c_char, *mut FileInfo) -> c_int>,
    pub fsyncdir: Option<unsafe extern "C" fn(*const c_char, c_int, *mut FileInfo) -> c_int>,
    pub init: Option<unsafe extern "C" fn(*mut c_void, *mut ConfigPrefix) -> *mut c_void>,
    pub destroy: Unused,
    pub access: Option<unsafe extern "C" fn(*const c_char, c_int) -> c_int>,
    pub create: Option<unsafe extern "C" fn(*const c_char, libc::mode_t, *mut FileInfo) -> c_int>,
    pub lock: Unused,
    pub utimens:
        Option<unsafe extern "C" fn(*const c_char, *const libc::timespec, *mut FileInfo) -> c_int>,
    pub bmap: Unused,
    pub ioctl: Unused,
    pub poll: Unused,
    pub write_buf: Unused,
    pub read_buf: Unused,
    pub flock: Unused,
    pub fallocate: Unused,
    pub copy_file_range: Unused,
    pub lseek: Unused,
}
#[repr(C)]
pub(super) struct MacOperations {
    pub getattr: Option<Getattr>,
    pub setattr:
        Option<unsafe extern "C" fn(*const c_char, *mut c_void, c_int, *mut FileInfo) -> c_int>,
    pub tail: OperationsTail,
    pub chflags: Option<unsafe extern "C" fn(*const c_char, *mut FileInfo, u32) -> c_int>,
    pub setvolname: Option<unsafe extern "C" fn(*const c_char) -> c_int>,
    pub monitor: Unused,
    pub statx: Unused,
}
// Compile-time representation checks apply to every supported Darwin target.
const _: () = {
    assert!(std::mem::size_of::<MacOperations>() == 376);
    assert!(std::mem::size_of::<FileInfo>() == 64);
    assert!(std::mem::offset_of!(FileInfo, fh) == 16);
    assert!(std::mem::size_of::<Context>() == 40);
    assert!(std::mem::offset_of!(Context, private_data) == 24);
    assert!(std::mem::offset_of!(ConfigPrefix, use_ino) == 64);
    assert!(std::mem::size_of::<libc::stat>() == 144);
};
