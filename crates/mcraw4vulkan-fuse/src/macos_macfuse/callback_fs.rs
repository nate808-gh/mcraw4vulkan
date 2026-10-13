use crate::{
    filesystem_view::{FileSystemReadData, FileSystemView},
    virtual_fs::{VirtualFileKind, VirtualFileMetadata},
};
use mcraw4vulkan_macfuse_ffi::{CallbackFileSystem, DirectoryEntry, FsResult, Metadata};
use std::{
    collections::HashMap,
    path::{Component, Path},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

pub(super) struct CallbackView<F> {
    fs: Arc<F>,
    handles: Mutex<HashMap<u64, u64>>,
    next: AtomicU64,
}
impl<F: FileSystemView> CallbackView<F> {
    pub fn new(fs: Arc<F>) -> Self {
        Self {
            fs,
            handles: Mutex::new(HashMap::new()),
            next: AtomicU64::new(1),
        }
    }
    fn resolve(&self, path: &Path) -> FsResult<VirtualFileMetadata> {
        let mut m = self
            .fs
            .getattr(1)
            .map_err(|_| libc::EIO)?
            .ok_or(libc::ENOENT)?;
        for part in path.components() {
            match part {
                Component::RootDir => {}
                Component::Normal(name) => {
                    if m.kind != VirtualFileKind::Directory {
                        return Err(libc::ENOTDIR);
                    }
                    m = self
                        .fs
                        .lookup(m.inode, name)
                        .map_err(|_| libc::EIO)?
                        .ok_or(libc::ENOENT)?
                }
                _ => return Err(libc::EINVAL),
            }
        }
        Ok(m)
    }
}
fn metadata(m: VirtualFileMetadata) -> Metadata {
    Metadata {
        inode: m.inode,
        len: m.byte_len,
        directory: m.kind == VirtualFileKind::Directory,
        permissions: m.permissions,
        links: m.hard_links,
        seconds: m.timestamp.seconds,
        nanos: m.timestamp.nanos,
    }
}
impl<F: FileSystemView> CallbackFileSystem for CallbackView<F> {
    fn getattr(&self, path: &Path) -> FsResult<Metadata> {
        self.resolve(path).map(metadata)
    }
    fn open(&self, path: &Path) -> FsResult<u64> {
        let m = self.resolve(path)?;
        if m.kind != VirtualFileKind::RegularFile {
            return Err(libc::EISDIR);
        }
        let handle = self.next.fetch_add(1, Ordering::Relaxed);
        if handle == 0 {
            return Err(libc::EMFILE);
        }
        self.fs
            .open_file_handle(m.inode, handle)
            .map_err(|_| libc::EIO)?;
        self.handles
            .lock()
            .map_err(|_| libc::EIO)?
            .insert(handle, m.inode);
        Ok(handle)
    }
    fn release(&self, handle: u64) -> FsResult<()> {
        let inode = self
            .handles
            .lock()
            .map_err(|_| libc::EIO)?
            .remove(&handle)
            .ok_or(libc::EBADF)?;
        self.fs.release_file_handle(inode, handle);
        Ok(())
    }
    fn read(&self, path: &Path, handle: u64, offset: u64, buffer: &mut [u8]) -> FsResult<usize> {
        let size = u32::try_from(buffer.len()).map_err(|_| libc::EINVAL)?;
        let inode = if handle == 0 {
            self.resolve(path)?.inode
        } else {
            *self
                .handles
                .lock()
                .map_err(|_| libc::EIO)?
                .get(&handle)
                .ok_or(libc::EBADF)?
        };
        let data = self
            .fs
            .read_data_for_handle(inode, handle, offset, size)
            .map_err(|_| libc::EIO)?
            .ok_or(libc::ENOENT)?;
        let bytes = data.as_slice();
        if bytes.len() > buffer.len() {
            return Err(libc::EIO);
        }
        buffer[..bytes.len()].copy_from_slice(bytes);
        Ok(bytes.len())
    }
    fn readdir(&self, path: &Path, cookie: u64) -> FsResult<Vec<DirectoryEntry>> {
        let directory = self.resolve(path)?;
        if directory.kind != VirtualFileKind::Directory {
            return Err(libc::ENOTDIR);
        }
        let parent = self
            .fs
            .parent_inode_for_directory(directory.inode)
            .unwrap_or(directory.inode);
        let entries = self
            .fs
            .readdir(directory.inode)
            .map_err(|_| libc::EIO)?
            .ok_or(libc::ENOTDIR)?;
        // No READDIR_PLUS reply flag is set: FUSE uses only type and inode.
        // Do not query every remaining DNG length again on each directory page.
        let mut names = vec![
            (".".into(), directory.inode, true),
            ("..".into(), parent, true),
        ];
        names.extend(entries.into_iter().map(|e| {
            (
                e.name,
                e.inode,
                matches!(
                    e.node,
                    crate::virtual_fs::VirtualNode::Root
                        | crate::virtual_fs::VirtualNode::ClipDirectory
                ),
            )
        }));
        Ok(names
            .into_iter()
            .enumerate()
            .skip(usize::try_from(cookie).unwrap_or(usize::MAX))
            .map(|(i, (name, inode, is_dir))| DirectoryEntry {
                name,
                metadata: Metadata {
                    inode,
                    len: 0,
                    directory: is_dir,
                    permissions: if is_dir { 0o555 } else { 0o444 },
                    links: 1,
                    seconds: directory.timestamp.seconds,
                    nanos: directory.timestamp.nanos,
                },
                next_cookie: i as u64 + 1,
            })
            .collect())
    }
}
