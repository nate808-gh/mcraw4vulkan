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

    #[cfg(test)]
    fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<F::ReadData<'_>>> {
        self.fs.read_data(inode, offset, size)
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ffi::{OsStr, OsString};

    use mcraw4vulkan_core::MountClipIdentityInput;

    use super::*;
    use crate::shared_root::{
        AddClipResult, SHARED_ROOT_INODE, SharedRootClipFileSystem, SharedRootVirtualFileSystem,
    };
    use crate::virtual_fs::{AUDIO_WAV_INODE, CLIP_DIR_INODE, ROOT_INODE, VirtualTimestamp};

    const FAKE_FILE_INODE: u64 = 10;

    #[test]
    fn shared_root_root_readdir_exposes_two_clip_folders() {
        let adapter = shared_root_adapter_with_two_clips();
        let entries = adapter
            .readdir_entries(SHARED_ROOT_INODE, 0)
            .expect("readdir")
            .expect("root entries");
        let real_entries = entries_without_dot_entries(entries);

        assert_eq!(real_entries.len(), 2);
        assert!(
            real_entries
                .iter()
                .all(|entry| entry.node == VirtualNode::ClipDirectory)
        );
        assert_ne!(real_entries[0].name, real_entries[1].name);
    }

    #[test]
    fn shared_root_lookup_finds_top_level_clip_folders() {
        let adapter = shared_root_adapter_with_two_clips();
        let root_entries = real_root_entries(&adapter);

        for entry in root_entries {
            let metadata = adapter
                .lookup_metadata(SHARED_ROOT_INODE, entry.name.as_os_str())
                .expect("lookup")
                .expect("clip folder metadata");
            assert_eq!(metadata.inode, entry.inode);
            assert_eq!(metadata.kind, VirtualFileKind::Directory);
            assert_eq!(metadata.node, VirtualNode::ClipDirectory);
        }
    }

    #[test]
    fn shared_root_clip_folder_readdir_delegates_without_double_nesting() {
        let adapter = shared_root_adapter_with_two_clips();
        let first_clip = real_root_entries(&adapter).remove(0);
        let entries = adapter
            .readdir_entries(first_clip.inode, 0)
            .expect("clip readdir")
            .expect("clip entries");
        let names = entries
            .into_iter()
            .map(|entry| entry.name)
            .collect::<Vec<_>>();

        assert!(names.contains(&OsString::from("clip_a_000000.dng")));
        assert!(names.contains(&OsString::from("clip_a.wav")));
        assert!(!names.contains(&first_clip.name));
    }

    #[test]
    fn shared_root_lookup_under_clip_routes_to_correct_clip() {
        let adapter = shared_root_adapter_with_two_clips();
        let root_entries = real_root_entries(&adapter);
        let first_clip = &root_entries[0];
        let second_clip = &root_entries[1];

        let first_file = adapter
            .lookup_metadata(first_clip.inode, OsStr::new("clip_a_000000.dng"))
            .expect("first lookup")
            .expect("first clip file");
        let second_file = adapter
            .lookup_metadata(second_clip.inode, OsStr::new("clip_b_000000.dng"))
            .expect("second lookup")
            .expect("second clip file");

        assert_ne!(first_file.inode, second_file.inode);
        assert_eq!(
            adapter
                .lookup_metadata(first_clip.inode, OsStr::new("clip_b_000000.dng"))
                .expect("cross lookup"),
            None
        );
    }

    #[test]
    fn shared_root_getattr_reports_root_clip_folder_and_virtual_file_metadata() {
        let adapter = shared_root_adapter_with_two_clips();
        let root = adapter
            .getattr_metadata(SHARED_ROOT_INODE)
            .expect("root getattr")
            .expect("root metadata");
        assert_eq!(root.kind, VirtualFileKind::Directory);
        assert_eq!(root.node, VirtualNode::Root);

        let first_clip = real_root_entries(&adapter).remove(0);
        let clip = adapter
            .getattr_metadata(first_clip.inode)
            .expect("clip getattr")
            .expect("clip metadata");
        assert_eq!(clip.kind, VirtualFileKind::Directory);
        assert_eq!(clip.node, VirtualNode::ClipDirectory);

        let file = adapter
            .lookup_metadata(first_clip.inode, OsStr::new("clip_a_000000.dng"))
            .unwrap()
            .unwrap();
        let file_metadata = adapter
            .getattr_metadata(file.inode)
            .expect("file getattr")
            .expect("file metadata");
        assert_eq!(file_metadata.kind, VirtualFileKind::RegularFile);
        assert_eq!(file_metadata.node, VirtualNode::DngFrame { frame_index: 0 });
    }

    #[test]
    fn shared_root_read_returns_bytes_from_correct_clip() {
        let adapter = shared_root_adapter_with_two_clips();
        let root_entries = real_root_entries(&adapter);
        let first_file = adapter
            .lookup_metadata(root_entries[0].inode, OsStr::new("clip_a_000000.dng"))
            .unwrap()
            .unwrap();
        let second_file = adapter
            .lookup_metadata(root_entries[1].inode, OsStr::new("clip_b_000000.dng"))
            .unwrap()
            .unwrap();

        assert_eq!(
            adapter
                .read_data(first_file.inode, 0, 16)
                .unwrap()
                .unwrap()
                .as_slice(),
            b"alpha"
        );
        assert_eq!(
            adapter
                .read_data(second_file.inode, 1, 3)
                .unwrap()
                .unwrap()
                .as_slice(),
            b"eta"
        );
    }

    #[test]
    fn shared_root_identical_local_inodes_remain_distinct_in_fuse_adapter() {
        let adapter = shared_root_adapter_with_two_clips();
        let root_entries = real_root_entries(&adapter);
        let first_file = adapter
            .lookup_metadata(root_entries[0].inode, OsStr::new("clip_a_000000.dng"))
            .unwrap()
            .unwrap();
        let second_file = adapter
            .lookup_metadata(root_entries[1].inode, OsStr::new("clip_b_000000.dng"))
            .unwrap()
            .unwrap();

        assert_ne!(first_file.inode, second_file.inode);
        assert_ne!(first_file.inode, FAKE_FILE_INODE);
        assert_ne!(second_file.inode, FAKE_FILE_INODE);
    }

    #[test]
    fn shared_root_unknown_top_level_folder_returns_not_found() {
        let adapter = shared_root_adapter_with_two_clips();

        assert_eq!(
            adapter
                .lookup_metadata(SHARED_ROOT_INODE, OsStr::new("missing"))
                .expect("lookup"),
            None
        );
    }

    #[test]
    fn single_clip_root_readdir_exposes_files_directly() {
        let adapter = single_clip_root_adapter("clip", b"alpha");
        let entries = adapter
            .readdir_entries(ROOT_INODE, 0)
            .expect("root readdir")
            .expect("root entries");
        let names = entries
            .into_iter()
            .map(|entry| entry.name)
            .collect::<Vec<_>>();

        assert!(names.contains(&OsString::from("clip_000000.dng")));
        assert!(names.contains(&OsString::from("clip.wav")));
        assert!(!names.contains(&OsString::from("clip")));
    }

    #[test]
    fn single_clip_root_lookup_does_not_double_nest_clip_folder() {
        let adapter = single_clip_root_adapter("clip", b"alpha");

        let dng = adapter
            .lookup_metadata(ROOT_INODE, OsStr::new("clip_000000.dng"))
            .expect("dng lookup")
            .expect("dng metadata");
        assert_eq!(dng.kind, VirtualFileKind::RegularFile);
        assert_eq!(dng.node, VirtualNode::DngFrame { frame_index: 0 });

        assert_eq!(
            adapter
                .lookup_metadata(ROOT_INODE, OsStr::new("clip"))
                .expect("clip folder lookup"),
            None
        );
    }

    #[test]
    fn single_clip_root_reads_file_bytes_at_mount_root() {
        let adapter = single_clip_root_adapter("clip", b"alpha");
        let dng = adapter
            .lookup_metadata(ROOT_INODE, OsStr::new("clip_000000.dng"))
            .unwrap()
            .unwrap();

        assert_eq!(
            adapter
                .read_data(dng.inode, 1, 3)
                .unwrap()
                .unwrap()
                .as_slice(),
            b"lph"
        );
    }

    #[test]
    fn file_attrs_use_configured_owner() {
        let metadata = VirtualFileMetadata {
            inode: FAKE_FILE_INODE,
            node: VirtualNode::DngFrame { frame_index: 0 },
            kind: VirtualFileKind::RegularFile,
            byte_len: 128,
            permissions: 0o444,
            hard_links: 1,
            timestamp: VirtualTimestamp {
                seconds: 1_700_000_000,
                nanos: 123,
            },
        };
        let owner = FuserFileAttrOwner::new(501, 20);
        let attr = file_attr_from_metadata(metadata, owner);

        assert_eq!(attr.uid, 501);
        assert_eq!(attr.gid, 20);
        assert_eq!(attr.perm, 0o444);
        assert_eq!(attr.size, 128);
    }

    #[test]
    fn virtual_access_allows_read_execute_and_rejects_writes() {
        assert!(virtual_access_allowed(0o755, AccessFlags::F_OK));
        assert!(virtual_access_allowed(0o444, AccessFlags::R_OK));
        assert!(virtual_access_allowed(0o755, AccessFlags::X_OK));
        assert!(virtual_access_allowed(
            0o755,
            AccessFlags::R_OK | AccessFlags::X_OK
        ));

        assert!(!virtual_access_allowed(0o444, AccessFlags::W_OK));
        assert!(!virtual_access_allowed(0o444, AccessFlags::X_OK));
    }

    #[test]
    fn shared_root_duplicate_add_remains_already_mounted() {
        let mut fs = SharedRootVirtualFileSystem::new();
        let input = identity_input("clip", "source/clip.mcraw", 100);
        let first = fs
            .insert_clip(
                "clip.mcraw".into(),
                &input,
                Arc::new(FakeClipFileSystem::new("clip", b"one")),
            )
            .expect("first insert");
        let second = fs
            .insert_clip(
                "clip-again.mcraw".into(),
                &input,
                Arc::new(FakeClipFileSystem::new("clip", b"two")),
            )
            .expect("second insert");

        assert!(matches!(first, AddClipResult::Added(_)));
        assert!(matches!(second, AddClipResult::AlreadyMounted(_)));
        assert_eq!(first.folder_name(), second.folder_name());
        assert_eq!(fs.registry().len(), 1);
    }

    fn shared_root_adapter_with_two_clips() -> LinuxSharedRootFuse3FileSystem {
        let mut fs = SharedRootVirtualFileSystem::new();
        fs.insert_clip(
            "clip_a.mcraw".into(),
            &identity_input("clip_a", "source/a.mcraw", 5),
            Arc::new(FakeClipFileSystem::new("clip_a", b"alpha")),
        )
        .expect("insert first clip");
        fs.insert_clip(
            "clip_b.mcraw".into(),
            &identity_input("clip_b", "source/b.mcraw", 4),
            Arc::new(FakeClipFileSystem::new("clip_b", b"beta")),
        )
        .expect("insert second clip");

        LinuxSharedRootFuse3FileSystem::new_shared_root(Arc::new(fs))
    }

    fn single_clip_root_adapter(
        clip_stem: &str,
        dng_bytes: &[u8],
    ) -> LinuxSingleClipRootFuse3FileSystem {
        let fs = SingleClipRootVirtualFileSystem::from_clip_file_system(Arc::new(
            FakeClipFileSystem::new(clip_stem, dng_bytes),
        ));
        LinuxSingleClipRootFuse3FileSystem::new_single_clip_root(Arc::new(fs))
    }

    fn real_root_entries(adapter: &LinuxSharedRootFuse3FileSystem) -> Vec<FuserDirEntry> {
        entries_without_dot_entries(
            adapter
                .readdir_entries(SHARED_ROOT_INODE, 0)
                .expect("root readdir")
                .expect("root entries"),
        )
    }

    fn entries_without_dot_entries(entries: Vec<FuserDirEntry>) -> Vec<FuserDirEntry> {
        entries
            .into_iter()
            .filter(|entry| entry.name != "." && entry.name != "..")
            .collect()
    }

    fn identity_input(stem: &str, path: &str, file_len: u64) -> MountClipIdentityInput {
        let mut input = MountClipIdentityInput::new(stem, path);
        input.canonical_path = Some(format!("canonical/{path}"));
        input.file_len = Some(file_len);
        input
    }

    struct FakeClipFileSystem {
        files: BTreeMap<OsString, FakeFile>,
        metadata: BTreeMap<u64, VirtualFileMetadata>,
    }

    impl FakeClipFileSystem {
        fn new(clip_stem: &str, dng_bytes: &[u8]) -> Self {
            let timestamp = VirtualTimestamp {
                seconds: 1_700_000_000,
                nanos: 123,
            };
            let mut metadata = BTreeMap::new();
            metadata.insert(
                CLIP_DIR_INODE,
                VirtualFileMetadata {
                    inode: CLIP_DIR_INODE,
                    node: VirtualNode::ClipDirectory,
                    kind: VirtualFileKind::Directory,
                    byte_len: 0,
                    permissions: 0o755,
                    hard_links: 2,
                    timestamp,
                },
            );
            metadata.insert(
                FAKE_FILE_INODE,
                VirtualFileMetadata {
                    inode: FAKE_FILE_INODE,
                    node: VirtualNode::DngFrame { frame_index: 0 },
                    kind: VirtualFileKind::RegularFile,
                    byte_len: dng_bytes.len() as u64,
                    permissions: 0o444,
                    hard_links: 1,
                    timestamp,
                },
            );
            metadata.insert(
                AUDIO_WAV_INODE,
                VirtualFileMetadata {
                    inode: AUDIO_WAV_INODE,
                    node: VirtualNode::AudioWav,
                    kind: VirtualFileKind::RegularFile,
                    byte_len: 3,
                    permissions: 0o444,
                    hard_links: 1,
                    timestamp,
                },
            );

            let mut files = BTreeMap::new();
            files.insert(
                OsString::from(format!("{clip_stem}_000000.dng")),
                FakeFile {
                    inode: FAKE_FILE_INODE,
                    node: VirtualNode::DngFrame { frame_index: 0 },
                    bytes: dng_bytes.to_vec(),
                },
            );
            files.insert(
                OsString::from(format!("{clip_stem}.wav")),
                FakeFile {
                    inode: AUDIO_WAV_INODE,
                    node: VirtualNode::AudioWav,
                    bytes: b"wav".to_vec(),
                },
            );

            Self { files, metadata }
        }
    }

    impl SharedRootClipFileSystem for FakeClipFileSystem {
        fn lookup(&self, parent_inode: u64, name: &OsStr) -> Result<Option<VirtualFileMetadata>> {
            if parent_inode != CLIP_DIR_INODE {
                return Ok(None);
            }
            let Some(file) = self.files.get(name) else {
                return Ok(None);
            };
            self.getattr(file.inode)
        }

        fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
            Ok(self.metadata.get(&inode).cloned())
        }

        fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>> {
            if inode != CLIP_DIR_INODE {
                return Ok(None);
            }
            Ok(Some(
                self.files
                    .iter()
                    .map(|(name, file)| VirtualDirEntry {
                        inode: file.inode,
                        name: name.clone(),
                        node: file.node,
                    })
                    .collect(),
            ))
        }

        fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Vec<u8>>> {
            let Some(file) = self.files.values().find(|file| file.inode == inode) else {
                return Ok(None);
            };
            let start = usize::try_from(offset).unwrap_or(usize::MAX);
            let requested = usize::try_from(size).unwrap_or(usize::MAX);
            if start >= file.bytes.len() {
                return Ok(Some(Vec::new()));
            }
            let end = start.saturating_add(requested).min(file.bytes.len());
            Ok(Some(file.bytes[start..end].to_vec()))
        }

        fn parent_inode_for_directory(&self, inode: u64) -> Option<u64> {
            match inode {
                CLIP_DIR_INODE => Some(ROOT_INODE),
                _ => None,
            }
        }
    }

    struct FakeFile {
        inode: u64,
        node: VirtualNode,
        bytes: Vec<u8>,
    }
}
