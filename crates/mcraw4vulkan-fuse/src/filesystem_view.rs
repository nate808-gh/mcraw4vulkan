//! Shared operation facade used by the native filesystem adapters.
use crate::shared_root::SingleClipRootVirtualFileSystem;
use crate::virtual_fs::{VirtualDirEntry, VirtualFileMetadata, VirtualFileSystem, VirtualReadData};
use anyhow::Result;
use std::ffi::OsStr;

pub trait FileSystemReadData {
    fn as_slice(&self) -> &[u8];
}

impl FileSystemReadData for VirtualReadData {
    fn as_slice(&self) -> &[u8] {
        VirtualReadData::as_slice(self)
    }
}

impl FileSystemReadData for Vec<u8> {
    fn as_slice(&self) -> &[u8] {
        self.as_slice()
    }
}

pub trait FileSystemView: Send + Sync + 'static {
    type ReadData<'a>: FileSystemReadData
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
}

impl FileSystemView for VirtualFileSystem {
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
}

impl FileSystemView for SingleClipRootVirtualFileSystem {
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
