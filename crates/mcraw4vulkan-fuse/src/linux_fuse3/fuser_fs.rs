use std::ffi::OsStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
#[cfg(target_os = "macos")]
use fuser::ReplyXTimes;
use fuser::{
    AccessFlags, Errno, FileAttr, FileHandle, FileType, Filesystem, FopenFlags, Generation,
    INodeNo, KernelConfig, LockOwner, OpenFlags, ReplyAttr, ReplyData, ReplyDirectory, ReplyEmpty,
    ReplyEntry, ReplyOpen, ReplyStatfs, ReplyXattr, Request,
};
use tracing::info;

use crate::shared_root::{SharedRootVirtualFileSystem, SingleClipRootVirtualFileSystem};
use crate::virtual_fs::{
    VirtualDirEntry, VirtualFileKind, VirtualFileMetadata, VirtualFileSystem, VirtualNode,
    VirtualReadData,
};

// Attribute and entry cache TTL returned to the kernel.
//
// These virtual files are deterministic during one mount, so a short positive TTL
// is safe and avoids immediate repeated getattr/lookup churn.
const TTL: Duration = Duration::from_secs(1);
const BLOCK_SIZE: u32 = 512;
const READ_STATS_LOG_INTERVAL: u64 = 2048;
const STATFS_BLOCKS: u64 = 1_048_576;
const STATFS_FILES: u64 = 1_000_000;
const MAX_NAME_LENGTH: u32 = 255;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FuserFileAttrOwner {
    pub(crate) uid: u32,
    pub(crate) gid: u32,
}

impl FuserFileAttrOwner {
    pub(crate) const fn new(uid: u32, gid: u32) -> Self {
        Self { uid, gid }
    }
}

impl Default for FuserFileAttrOwner {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

// Linux fuser adapter for the platform-neutral virtual filesystem facade.
//
// This type is intentionally thin. It translates fuser request/reply types into
// the platform-neutral filesystem-shaped methods:
//
// - lookup  -> filesystem lookup()
// - getattr -> filesystem getattr()
// - readdir -> filesystem readdir()
// - read    -> filesystem read_data_for_handle()
//
// Decode, DNG byte generation, caching, inode identity, timestamp selection, EOF
// behavior, instrumentation counters, and prefetch triggering remain in the
// validated support layers.
pub struct LinuxFuse3FileSystem<F = VirtualFileSystem> {
    fs: Arc<F>,
    attr_owner: FuserFileAttrOwner,
    next_file_handle: AtomicU64,
}

impl LinuxFuse3FileSystem {
    #[cfg(target_os = "linux")]
    pub fn new(fs: Arc<VirtualFileSystem>) -> Self {
        Self {
            fs,
            attr_owner: FuserFileAttrOwner::default(),
            next_file_handle: AtomicU64::new(1),
        }
    }

    #[allow(dead_code)]
    pub fn inner(&self) -> &VirtualFileSystem {
        self.fs.as_ref()
    }
}

#[allow(dead_code)]
pub(crate) type LinuxSharedRootFuse3FileSystem = LinuxFuse3FileSystem<SharedRootVirtualFileSystem>;
#[allow(dead_code)]
pub(crate) type LinuxSingleClipRootFuse3FileSystem =
    LinuxFuse3FileSystem<SingleClipRootVirtualFileSystem>;

impl LinuxSharedRootFuse3FileSystem {
    #[allow(dead_code)]
    pub(crate) fn new_shared_root(fs: Arc<SharedRootVirtualFileSystem>) -> Self {
        Self {
            fs,
            attr_owner: FuserFileAttrOwner::default(),
            next_file_handle: AtomicU64::new(1),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn new_shared_root_with_owner(
        fs: Arc<SharedRootVirtualFileSystem>,
        attr_owner: FuserFileAttrOwner,
    ) -> Self {
        Self {
            fs,
            attr_owner,
            next_file_handle: AtomicU64::new(1),
        }
    }
}

impl LinuxSingleClipRootFuse3FileSystem {
    #[allow(dead_code)]
    pub(crate) fn new_single_clip_root(fs: Arc<SingleClipRootVirtualFileSystem>) -> Self {
        Self {
            fs,
            attr_owner: FuserFileAttrOwner::default(),
            next_file_handle: AtomicU64::new(1),
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn new_single_clip_root_with_owner(
        fs: Arc<SingleClipRootVirtualFileSystem>,
        attr_owner: FuserFileAttrOwner,
    ) -> Self {
        Self {
            fs,
            attr_owner,
            next_file_handle: AtomicU64::new(1),
        }
    }
}

impl<F> LinuxFuse3FileSystem<F>
where
    F: LinuxFuse3FileSystemView,
{
    fn reply_entry_from_metadata(&self, reply: ReplyEntry, metadata: VirtualFileMetadata) {
        let attr = file_attr_from_metadata(metadata, self.attr_owner);
        reply.entry(&TTL, &attr, Generation(0));
    }

    fn reply_attr_from_metadata(&self, reply: ReplyAttr, metadata: VirtualFileMetadata) {
        let attr = file_attr_from_metadata(metadata, self.attr_owner);
        reply.attr(&TTL, &attr);
    }

    fn lookup_metadata(
        &self,
        parent_inode: u64,
        name: &OsStr,
    ) -> Result<Option<VirtualFileMetadata>> {
        self.fs.lookup(parent_inode, name)
    }

    fn getattr_metadata(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
        self.fs.getattr(inode)
    }

    fn readdir_entries(&self, inode: u64, offset: u64) -> Result<Option<Vec<FuserDirEntry>>> {
        let Some(entries) = self.fs.readdir(inode)? else {
            return Ok(None);
        };
        let parent_inode = self.fs.parent_inode_for_directory(inode).unwrap_or(inode);
        let all_entries =
            directory_entries_with_dot_entries(INodeNo(inode), INodeNo(parent_inode), entries);
        let start_index = usize::try_from(offset).unwrap_or(usize::MAX);

        Ok(Some(all_entries.into_iter().skip(start_index).collect()))
    }

    fn read_data_for_handle(
        &self,
        inode: u64,
        handle: u64,
        offset: u64,
        size: u32,
    ) -> Result<Option<F::ReadData<'_>>> {
        self.fs.read_data_for_handle(inode, handle, offset, size)
    }

    fn next_regular_file_handle(&self) -> u64 {
        self.next_file_handle.fetch_add(1, Ordering::Relaxed)
    }

    fn metadata_exists(&self, inode: u64) -> Result<bool> {
        Ok(self.getattr_metadata(inode)?.is_some())
    }
}

impl<F> Filesystem for LinuxFuse3FileSystem<F>
where
    F: LinuxFuse3FileSystemView,
{
    fn init(&mut self, _req: &Request, _config: &mut KernelConfig) -> std::io::Result<()> {
        Ok(())
    }

    fn lookup(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEntry) {
        match self.lookup_metadata(parent.0, name) {
            Ok(Some(metadata)) => self.reply_entry_from_metadata(reply, metadata),
            Ok(None) => reply.error(Errno::ENOENT),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn getattr(&self, _req: &Request, ino: INodeNo, _fh: Option<FileHandle>, reply: ReplyAttr) {
        match self.getattr_metadata(ino.0) {
            Ok(Some(metadata)) => self.reply_attr_from_metadata(reply, metadata),
            Ok(None) => reply.error(Errno::ENOENT),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    #[cfg(target_os = "macos")]
    fn getxtimes(&self, _req: &Request, ino: INodeNo, reply: ReplyXTimes) {
        match self.getattr_metadata(ino.0) {
            Ok(Some(metadata)) => {
                let time = system_time_from_virtual_timestamp(
                    metadata.timestamp.seconds,
                    metadata.timestamp.nanos,
                );
                reply.xtimes(time, time);
            }
            Ok(None) => reply.error(Errno::ENOENT),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn open(&self, _req: &Request, ino: INodeNo, _flags: OpenFlags, reply: ReplyOpen) {
        self.fs.record_open();

        match self.getattr_metadata(ino.0) {
            Ok(Some(metadata)) if metadata.kind == VirtualFileKind::RegularFile => {
                let handle = self.next_regular_file_handle();
                if self.fs.open_file_handle(ino.0, handle).is_err() {
                    reply.error(Errno::EIO);
                    return;
                }
                reply.opened(FileHandle(handle), FopenFlags::empty());
            }
            Ok(Some(_)) => reply.error(Errno::EISDIR),
            Ok(None) => reply.error(Errno::ENOENT),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn flush(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _lock_owner: LockOwner,
        reply: ReplyEmpty,
    ) {
        // Read-only virtual files have no dirty state to flush. Returning OK
        // keeps close()/flush-heavy applications quiet and avoids fuser's default
        // "Not Implemented" warning.
        reply.ok();
    }

    fn release(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        // Releasing the logical file handle drops any DNG allocation pinned by
        // reads performed through that open handle.
        self.fs.release_file_handle(_ino.0, _fh.0);
        reply.ok();
    }

    fn fsync(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        // Read-only virtual files have no writable state to synchronize.
        reply.ok();
    }

    fn opendir(&self, _req: &Request, ino: INodeNo, _flags: OpenFlags, reply: ReplyOpen) {
        self.fs.record_opendir();

        match self.getattr_metadata(ino.0) {
            Ok(Some(metadata)) if metadata.kind == VirtualFileKind::Directory => {
                reply.opened(FileHandle(ino.0), FopenFlags::empty());
            }
            Ok(Some(_)) => reply.error(Errno::ENOTDIR),
            Ok(None) => reply.error(Errno::ENOENT),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn readdir(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        mut reply: ReplyDirectory,
    ) {
        let entries = match self.readdir_entries(ino.0, offset) {
            Ok(Some(entries)) => entries,
            Ok(None) => {
                reply.error(Errno::ENOTDIR);
                return;
            }
            Err(_) => {
                reply.error(Errno::EIO);
                return;
            }
        };

        for (index, entry) in entries.into_iter().enumerate() {
            let next_offset = offset
                .saturating_add(u64::try_from(index).unwrap_or(u64::MAX))
                .saturating_add(1);
            let full = reply.add(
                INodeNo(entry.inode),
                next_offset,
                file_type_for_node(entry.node),
                entry.name,
            );

            if full {
                break;
            }
        }

        reply.ok();
    }

    fn releasedir(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _flags: OpenFlags,
        reply: ReplyEmpty,
    ) {
        // Directory handles carry no per-open state, so release has nothing to
        // clean up.
        reply.ok();
    }

    fn fsyncdir(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        // Finder and LaunchServices may synchronize directory handles even
        // though this read-only virtual filesystem has no dirty directory state.
        reply.ok();
    }

    fn statfs(&self, _req: &Request, _ino: INodeNo, reply: ReplyStatfs) {
        reply.statfs(
            STATFS_BLOCKS,
            STATFS_BLOCKS,
            STATFS_BLOCKS,
            STATFS_FILES,
            STATFS_FILES,
            BLOCK_SIZE,
            MAX_NAME_LENGTH,
            BLOCK_SIZE,
        );
    }

    fn setxattr(
        &self,
        _req: &Request,
        ino: INodeNo,
        _name: &OsStr,
        _value: &[u8],
        _flags: i32,
        _position: u32,
        reply: ReplyEmpty,
    ) {
        match self.metadata_exists(ino.0) {
            Ok(true) => reply.error(Errno::EROFS),
            Ok(false) => reply.error(Errno::ENOENT),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn getxattr(&self, _req: &Request, ino: INodeNo, _name: &OsStr, _size: u32, reply: ReplyXattr) {
        match self.metadata_exists(ino.0) {
            Ok(true) => reply.error(Errno::NO_XATTR),
            Ok(false) => reply.error(Errno::ENOENT),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn listxattr(&self, _req: &Request, ino: INodeNo, size: u32, reply: ReplyXattr) {
        match self.metadata_exists(ino.0) {
            Ok(true) if size == 0 => reply.size(0),
            Ok(true) => reply.data(&[]),
            Ok(false) => reply.error(Errno::ENOENT),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn removexattr(&self, _req: &Request, ino: INodeNo, _name: &OsStr, reply: ReplyEmpty) {
        match self.metadata_exists(ino.0) {
            Ok(true) => reply.error(Errno::NO_XATTR),
            Ok(false) => reply.error(Errno::ENOENT),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn access(&self, _req: &Request, ino: INodeNo, mask: AccessFlags, reply: ReplyEmpty) {
        match self.getattr_metadata(ino.0) {
            Ok(Some(metadata)) if virtual_access_allowed(metadata.permissions, mask) => reply.ok(),
            Ok(Some(_)) => reply.error(Errno::EACCES),
            Ok(None) => reply.error(Errno::ENOENT),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn read(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        size: u32,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        reply: ReplyData,
    ) {
        match self.read_data_for_handle(ino.0, _fh.0, offset, size) {
            Ok(Some(read_data)) => {
                reply.data(read_data.as_slice());
                self.fs.maybe_log_read_stats();
            }
            Ok(None) => {
                reply.error(Errno::EISDIR);
                self.fs.maybe_log_read_stats();
            }
            Err(_) => {
                reply.error(Errno::EIO);
                self.fs.maybe_log_read_stats();
            }
        }
    }
}

pub trait LinuxFuse3ReadData {
    fn as_slice(&self) -> &[u8];
}

impl LinuxFuse3ReadData for VirtualReadData {
    fn as_slice(&self) -> &[u8] {
        VirtualReadData::as_slice(self)
    }
}

impl LinuxFuse3ReadData for Vec<u8> {
    fn as_slice(&self) -> &[u8] {
        self.as_slice()
    }
}

pub trait LinuxFuse3FileSystemView: Send + Sync + 'static {
    type ReadData<'a>: LinuxFuse3ReadData
    where
        Self: 'a;

    fn lookup(&self, parent_inode: u64, name: &OsStr) -> Result<Option<VirtualFileMetadata>>;
    fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>>;
    fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>>;
    fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Self::ReadData<'_>>>;
    fn read_data_for_handle(
        &self,
        inode: u64,
        handle: u64,
        offset: u64,
        size: u32,
    ) -> Result<Option<Self::ReadData<'_>>> {
        let _ = handle;
        self.read_data(inode, offset, size)
    }
    fn parent_inode_for_directory(&self, inode: u64) -> Option<u64>;

    fn open_file_handle(&self, _inode: u64, _handle: u64) -> Result<()> {
        Ok(())
    }

    fn release_file_handle(&self, _inode: u64, _handle: u64) {}

    fn record_open(&self) {}

    fn record_opendir(&self) {}

    fn maybe_log_read_stats(&self) {}
}

impl LinuxFuse3FileSystemView for VirtualFileSystem {
    type ReadData<'a> = VirtualReadData;

    fn lookup(&self, parent_inode: u64, name: &OsStr) -> Result<Option<VirtualFileMetadata>> {
        VirtualFileSystem::lookup(self, parent_inode, name)
    }

    fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
        VirtualFileSystem::getattr(self, inode)
    }

    fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>> {
        VirtualFileSystem::readdir(self, inode)
    }

    fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Self::ReadData<'_>>> {
        VirtualFileSystem::read_data(self, inode, offset, size)
    }

    fn read_data_for_handle(
        &self,
        inode: u64,
        handle: u64,
        offset: u64,
        size: u32,
    ) -> Result<Option<Self::ReadData<'_>>> {
        VirtualFileSystem::read_data_for_handle(self, inode, handle, offset, size)
    }

    fn parent_inode_for_directory(&self, inode: u64) -> Option<u64> {
        VirtualFileSystem::parent_inode_for_directory(self, inode)
    }

    fn open_file_handle(&self, inode: u64, handle: u64) -> Result<()> {
        VirtualFileSystem::open_file_handle(self, inode, handle)
    }

    fn release_file_handle(&self, _inode: u64, handle: u64) {
        VirtualFileSystem::release_file_handle(self, handle);
    }

    fn record_open(&self) {
        VirtualFileSystem::record_open(self);
    }

    fn record_opendir(&self) {
        VirtualFileSystem::record_opendir(self);
    }

    fn maybe_log_read_stats(&self) {
        let Ok(snapshot) = self.stats_snapshot() else {
            return;
        };

        let read_count = snapshot.runtime.read.read_count;

        if read_count > 0 && read_count % READ_STATS_LOG_INTERVAL == 0 {
            if let Ok(summary) = self.stats_summary_line() {
                info!("mcraw4vulkan FUSE stats: {summary}");
            }
        }
    }
}

impl LinuxFuse3FileSystemView for SharedRootVirtualFileSystem {
    type ReadData<'a> = Vec<u8>;

    fn lookup(&self, parent_inode: u64, name: &OsStr) -> Result<Option<VirtualFileMetadata>> {
        SharedRootVirtualFileSystem::lookup(self, parent_inode, name)
    }

    fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
        SharedRootVirtualFileSystem::getattr(self, inode)
    }

    fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>> {
        SharedRootVirtualFileSystem::readdir(self, inode)
    }

    fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Self::ReadData<'_>>> {
        SharedRootVirtualFileSystem::read_data(self, inode, offset, size)
    }

    fn read_data_for_handle(
        &self,
        inode: u64,
        handle: u64,
        offset: u64,
        size: u32,
    ) -> Result<Option<Self::ReadData<'_>>> {
        SharedRootVirtualFileSystem::read_data_for_handle(self, inode, handle, offset, size)
    }

    fn parent_inode_for_directory(&self, inode: u64) -> Option<u64> {
        SharedRootVirtualFileSystem::parent_inode_for_directory(self, inode)
    }

    fn open_file_handle(&self, inode: u64, handle: u64) -> Result<()> {
        SharedRootVirtualFileSystem::open_file_handle(self, inode, handle)
    }

    fn release_file_handle(&self, inode: u64, handle: u64) {
        SharedRootVirtualFileSystem::release_file_handle(self, inode, handle);
    }
}

impl LinuxFuse3FileSystemView for SingleClipRootVirtualFileSystem {
    type ReadData<'a> = Vec<u8>;

    fn lookup(&self, parent_inode: u64, name: &OsStr) -> Result<Option<VirtualFileMetadata>> {
        SingleClipRootVirtualFileSystem::lookup(self, parent_inode, name)
    }

    fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
        SingleClipRootVirtualFileSystem::getattr(self, inode)
    }

    fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>> {
        SingleClipRootVirtualFileSystem::readdir(self, inode)
    }

    fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Self::ReadData<'_>>> {
        SingleClipRootVirtualFileSystem::read_data(self, inode, offset, size)
    }

    fn read_data_for_handle(
        &self,
        inode: u64,
        handle: u64,
        offset: u64,
        size: u32,
    ) -> Result<Option<Self::ReadData<'_>>> {
        SingleClipRootVirtualFileSystem::read_data_for_handle(self, inode, handle, offset, size)
    }

    fn parent_inode_for_directory(&self, inode: u64) -> Option<u64> {
        SingleClipRootVirtualFileSystem::parent_inode_for_directory(self, inode)
    }

    fn open_file_handle(&self, inode: u64, handle: u64) -> Result<()> {
        SingleClipRootVirtualFileSystem::open_file_handle(self, inode, handle)
    }

    fn release_file_handle(&self, inode: u64, handle: u64) {
        SingleClipRootVirtualFileSystem::release_file_handle(self, inode, handle);
    }
}

// Directory entry format used internally by the fuser adapter.
//
// This includes "." and ".." entries in addition to the real virtual entries
// supplied by InodeMap.
struct FuserDirEntry {
    inode: u64,
    name: std::ffi::OsString,
    node: VirtualNode,
}

fn directory_entries_with_dot_entries(
    directory_inode: INodeNo,
    parent_inode: INodeNo,
    entries: Vec<VirtualDirEntry>,
) -> Vec<FuserDirEntry> {
    let mut out = Vec::with_capacity(entries.len() + 2);

    out.push(FuserDirEntry {
        inode: directory_inode.0,
        name: ".".into(),
        node: VirtualNode::Root,
    });

    out.push(FuserDirEntry {
        inode: parent_inode.0,
        name: "..".into(),
        node: VirtualNode::Root,
    });

    for entry in entries {
        out.push(FuserDirEntry {
            inode: entry.inode,
            name: entry.name,
            node: entry.node,
        });
    }

    out
}

fn file_attr_from_metadata(
    metadata: VirtualFileMetadata,
    attr_owner: FuserFileAttrOwner,
) -> FileAttr {
    FileAttr {
        ino: INodeNo(metadata.inode),
        size: metadata.byte_len,
        blocks: metadata.byte_len.div_ceil(u64::from(BLOCK_SIZE)),
        atime: system_time_from_virtual_timestamp(
            metadata.timestamp.seconds,
            metadata.timestamp.nanos,
        ),
        mtime: system_time_from_virtual_timestamp(
            metadata.timestamp.seconds,
            metadata.timestamp.nanos,
        ),
        ctime: system_time_from_virtual_timestamp(
            metadata.timestamp.seconds,
            metadata.timestamp.nanos,
        ),
        crtime: system_time_from_virtual_timestamp(
            metadata.timestamp.seconds,
            metadata.timestamp.nanos,
        ),
        kind: file_type_for_kind(metadata.kind),
        perm: metadata.permissions,
        nlink: metadata.hard_links,
        uid: attr_owner.uid,
        gid: attr_owner.gid,
        rdev: 0,
        blksize: BLOCK_SIZE,
        flags: 0,
    }
}

fn virtual_access_allowed(permissions: u16, mask: AccessFlags) -> bool {
    let requested = mask.bits();
    let read_requested = requested & AccessFlags::R_OK.bits() != 0;
    let write_requested = requested & AccessFlags::W_OK.bits() != 0;
    let execute_requested = requested & AccessFlags::X_OK.bits() != 0;

    !write_requested
        && (!read_requested || permissions & 0o444 != 0)
        && (!execute_requested || permissions & 0o111 != 0)
}

fn file_type_for_kind(kind: VirtualFileKind) -> FileType {
    match kind {
        VirtualFileKind::Directory => FileType::Directory,
        VirtualFileKind::RegularFile => FileType::RegularFile,
    }
}

fn file_type_for_node(node: VirtualNode) -> FileType {
    match node {
        VirtualNode::Root | VirtualNode::ClipDirectory => FileType::Directory,
        VirtualNode::DngFrame { .. } | VirtualNode::AudioWav => FileType::RegularFile,
    }
}

fn system_time_from_virtual_timestamp(seconds: i64, nanos: u32) -> SystemTime {
    if seconds >= 0 {
        UNIX_EPOCH + Duration::new(seconds as u64, nanos)
    } else {
        UNIX_EPOCH - Duration::new(seconds.unsigned_abs(), nanos)
    }
}
