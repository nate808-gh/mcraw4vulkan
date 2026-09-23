#![allow(dead_code)]

use std::ffi::{OsStr, OsString};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, bail};
use mcraw4vulkan_core::{
    CLIP_MOUNT_SUFFIX_EXPANSION_HEX_LENGTHS, MountClipFolderName, MountClipIdentity,
    MountClipIdentityInput, clip_mount_folder_name_with_suffix_len, format_clip_mount_hash_suffix,
};

use crate::virtual_fs::{
    CLIP_DIR_INODE, ROOT_INODE, VirtualDirEntry, VirtualFileKind, VirtualFileMetadata,
    VirtualFileSystem, VirtualNode, VirtualTimestamp,
};

pub(crate) const SHARED_ROOT_INODE: u64 = ROOT_INODE;

// An aggregate inode uses the high bit as a namespace tag, the next 15 bits as
// the clip slot, and the low 48 bits as the clip-local inode. This keeps equal
// local inode values distinct across clips without changing their local maps.
const AGGREGATE_INODE_TAG: u64 = 1 << 63;
const LOCAL_INODE_BITS: u32 = 48;
const LOCAL_INODE_MASK: u64 = (1u64 << LOCAL_INODE_BITS) - 1;
const MAX_REGISTRY_SLOTS: usize = 1 << (63 - LOCAL_INODE_BITS);

pub(crate) trait SharedRootClipFileSystem: Send + Sync {
    fn lookup(&self, parent_inode: u64, name: &OsStr) -> Result<Option<VirtualFileMetadata>>;
    fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>>;
    fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>>;
    fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Vec<u8>>>;
    fn read_data_for_handle(
        &self,
        inode: u64,
        handle: u64,
        offset: u64,
        size: u32,
    ) -> Result<Option<Vec<u8>>> {
        let _ = handle;
        self.read_data(inode, offset, size)
    }
    fn parent_inode_for_directory(&self, inode: u64) -> Option<u64>;

    fn open_file_handle(&self, _inode: u64, _handle: u64) -> Result<()> {
        Ok(())
    }

    fn release_file_handle(&self, _inode: u64, _handle: u64) {}
}

impl SharedRootClipFileSystem for VirtualFileSystem {
    fn lookup(&self, parent_inode: u64, name: &OsStr) -> Result<Option<VirtualFileMetadata>> {
        VirtualFileSystem::lookup(self, parent_inode, name)
    }

    fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
        VirtualFileSystem::getattr(self, inode)
    }

    fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>> {
        VirtualFileSystem::readdir(self, inode)
    }

    fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Vec<u8>>> {
        Ok(VirtualFileSystem::read_data(self, inode, offset, size)?
            .map(|data| data.as_slice().to_vec()))
    }

    fn read_data_for_handle(
        &self,
        inode: u64,
        handle: u64,
        offset: u64,
        size: u32,
    ) -> Result<Option<Vec<u8>>> {
        Ok(
            VirtualFileSystem::read_data_for_handle(self, inode, handle, offset, size)?
                .map(|data| data.as_slice().to_vec()),
        )
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MountedClipSlot(u16);

impl MountedClipSlot {
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MountedClipReference {
    pub(crate) slot: MountedClipSlot,
    pub(crate) folder_name: String,
    pub(crate) identity: MountClipIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AddClipResult {
    Added(MountedClipReference),
    AlreadyMounted(MountedClipReference),
}

impl AddClipResult {
    pub(crate) fn reference(&self) -> &MountedClipReference {
        match self {
            Self::Added(reference) | Self::AlreadyMounted(reference) => reference,
        }
    }

    pub(crate) fn folder_name(&self) -> &str {
        &self.reference().folder_name
    }
}

pub(crate) struct MountedClip {
    source_path: PathBuf,
    folder_name: String,
    identity: MountClipIdentity,
    slot: MountedClipSlot,
    fs: Arc<dyn SharedRootClipFileSystem>,
}

impl MountedClip {
    pub(crate) fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub(crate) fn folder_name(&self) -> &str {
        &self.folder_name
    }

    pub(crate) fn identity(&self) -> MountClipIdentity {
        self.identity
    }

    pub(crate) fn slot(&self) -> MountedClipSlot {
        self.slot
    }

    fn reference(&self) -> MountedClipReference {
        MountedClipReference {
            slot: self.slot,
            folder_name: self.folder_name.clone(),
            identity: self.identity,
        }
    }

    fn clip_dir_inode(&self) -> Result<u64> {
        aggregate_inode_for_local(self.slot, CLIP_DIR_INODE)
    }
}

#[derive(Default)]
pub(crate) struct MountedClipRegistry {
    clips: Vec<MountedClip>,
}

impl MountedClipRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn len(&self) -> usize {
        self.clips.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    pub(crate) fn add_virtual_file_system_from_path(
        &mut self,
        source_path: PathBuf,
        fs: Arc<VirtualFileSystem>,
    ) -> Result<AddClipResult> {
        let identity_input = MountClipIdentityInput::from_path(&source_path)?;
        let fs: Arc<dyn SharedRootClipFileSystem> = fs;
        self.insert_clip(source_path, &identity_input, fs)
    }

    pub(crate) fn insert_clip(
        &mut self,
        source_path: PathBuf,
        identity_input: &MountClipIdentityInput,
        fs: Arc<dyn SharedRootClipFileSystem>,
    ) -> Result<AddClipResult> {
        let candidates = CLIP_MOUNT_SUFFIX_EXPANSION_HEX_LENGTHS
            .into_iter()
            .map(|suffix_len| clip_mount_folder_name_with_suffix_len(identity_input, suffix_len))
            .collect::<Vec<_>>();

        self.insert_clip_with_folder_name(source_path, candidates, fs)
    }

    pub(crate) fn lookup_by_identity(&self, identity: MountClipIdentity) -> Option<&MountedClip> {
        self.clips.iter().find(|clip| clip.identity == identity)
    }

    pub(crate) fn lookup_by_folder_name(&self, folder_name: &OsStr) -> Option<&MountedClip> {
        self.clips
            .iter()
            .find(|clip| clip.folder_name == folder_name.to_string_lossy())
    }

    pub(crate) fn root_entries(&self) -> Result<Vec<VirtualDirEntry>> {
        self.clips
            .iter()
            .map(|clip| {
                Ok(VirtualDirEntry {
                    inode: clip.clip_dir_inode()?,
                    name: OsString::from(&clip.folder_name),
                    node: VirtualNode::ClipDirectory,
                })
            })
            .collect()
    }

    fn insert_clip_with_folder_name(
        &mut self,
        source_path: PathBuf,
        candidates: Vec<MountClipFolderName>,
        fs: Arc<dyn SharedRootClipFileSystem>,
    ) -> Result<AddClipResult> {
        let Some(first_candidate) = candidates.first() else {
            bail!("mounted clip insert requires at least one folder-name candidate");
        };

        if let Some(existing) = self.lookup_by_identity(first_candidate.identity) {
            return Ok(AddClipResult::AlreadyMounted(existing.reference()));
        }

        if self.clips.len() >= MAX_REGISTRY_SLOTS {
            bail!("shared-root clip registry slot capacity exceeded");
        }

        let chosen = self.choose_non_conflicting_folder_name(&candidates)?;
        let slot_index = u16::try_from(self.clips.len()).expect("MAX_REGISTRY_SLOTS fits in u16");
        let slot = MountedClipSlot(slot_index);
        let clip = MountedClip {
            source_path,
            folder_name: chosen.visible,
            identity: chosen.identity,
            slot,
            fs,
        };
        let reference = clip.reference();
        self.clips.push(clip);

        Ok(AddClipResult::Added(reference))
    }

    fn choose_non_conflicting_folder_name(
        &self,
        candidates: &[MountClipFolderName],
    ) -> Result<ChosenFolderName> {
        let mut last_candidate = None;

        for candidate in candidates {
            last_candidate = Some(candidate);
            if !self.folder_name_conflicts(&candidate.visible) {
                return Ok(ChosenFolderName {
                    visible: candidate.visible.clone(),
                    identity: candidate.identity,
                });
            }
        }

        let Some(candidate) = last_candidate else {
            bail!("mounted clip insert requires at least one folder-name candidate");
        };

        let full_suffix = format_clip_mount_hash_suffix(candidate.identity, 32);
        for disambiguator in 2usize.. {
            let visible = format!(
                "{}__{}__{}",
                candidate.sanitized_stem, full_suffix, disambiguator
            );
            if !self.folder_name_conflicts(&visible) {
                return Ok(ChosenFolderName {
                    visible,
                    identity: candidate.identity,
                });
            }
        }

        bail!("shared-root clip folder disambiguator exhausted")
    }

    fn folder_name_conflicts(&self, candidate: &str) -> bool {
        // Rejecting ASCII case variants prevents a shared-root collision on adapters
        // that fold ASCII names during lookup.
        self.clips.iter().any(|clip| {
            clip.folder_name == candidate || clip.folder_name.eq_ignore_ascii_case(candidate)
        })
    }

    fn clip_for_slot(&self, slot: MountedClipSlot) -> Option<&MountedClip> {
        self.clips.get(slot.index())
    }
}

struct ChosenFolderName {
    visible: String,
    identity: MountClipIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AggregateInode {
    slot: MountedClipSlot,
    local_inode: u64,
}

pub(crate) struct SharedRootVirtualFileSystem {
    registry: MountedClipRegistry,
}

pub(crate) struct SingleClipRootVirtualFileSystem {
    fs: Arc<dyn SharedRootClipFileSystem>,
}

impl SingleClipRootVirtualFileSystem {
    pub(crate) fn new(fs: Arc<VirtualFileSystem>) -> Self {
        let fs: Arc<dyn SharedRootClipFileSystem> = fs;
        Self { fs }
    }

    pub(crate) fn lookup(
        &self,
        parent_inode: u64,
        name: &OsStr,
    ) -> Result<Option<VirtualFileMetadata>> {
        if parent_inode == ROOT_INODE {
            return self.fs.lookup(CLIP_DIR_INODE, name);
        }

        Ok(None)
    }

    pub(crate) fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
        if inode == CLIP_DIR_INODE {
            return Ok(None);
        }

        self.fs.getattr(inode)
    }

    pub(crate) fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>> {
        if inode == ROOT_INODE {
            return self.fs.readdir(CLIP_DIR_INODE);
        }

        Ok(None)
    }

    pub(crate) fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Vec<u8>>> {
        if matches!(inode, ROOT_INODE | CLIP_DIR_INODE) {
            return Ok(None);
        }

        self.fs.read_data(inode, offset, size)
    }

    pub(crate) fn read_data_for_handle(
        &self,
        inode: u64,
        handle: u64,
        offset: u64,
        size: u32,
    ) -> Result<Option<Vec<u8>>> {
        if matches!(inode, ROOT_INODE | CLIP_DIR_INODE) {
            return Ok(None);
        }

        self.fs.read_data_for_handle(inode, handle, offset, size)
    }

    pub(crate) fn open_file_handle(&self, inode: u64, handle: u64) -> Result<()> {
        if matches!(inode, ROOT_INODE | CLIP_DIR_INODE) {
            return Ok(());
        }

        self.fs.open_file_handle(inode, handle)
    }

    pub(crate) fn release_file_handle(&self, inode: u64, handle: u64) {
        if matches!(inode, ROOT_INODE | CLIP_DIR_INODE) {
            return;
        }

        self.fs.release_file_handle(inode, handle);
    }

    pub(crate) fn parent_inode_for_directory(&self, inode: u64) -> Option<u64> {
        if inode == ROOT_INODE {
            Some(ROOT_INODE)
        } else {
            None
        }
    }

    pub(crate) fn lookup_path(&self, relative_path: &Path) -> Result<Option<VirtualFileMetadata>> {
        let components = normalized_relative_components(relative_path)?;

        match components.as_slice() {
            [] => self.getattr(ROOT_INODE),
            [child_name] => self.lookup(ROOT_INODE, child_name.as_os_str()),
            _ => Ok(None),
        }
    }
}

impl SharedRootVirtualFileSystem {
    pub(crate) fn new() -> Self {
        Self {
            registry: MountedClipRegistry::new(),
        }
    }

    pub(crate) fn from_registry(registry: MountedClipRegistry) -> Self {
        Self { registry }
    }

    pub(crate) fn registry(&self) -> &MountedClipRegistry {
        &self.registry
    }

    pub(crate) fn registry_mut(&mut self) -> &mut MountedClipRegistry {
        &mut self.registry
    }

    pub(crate) fn insert_clip(
        &mut self,
        source_path: PathBuf,
        identity_input: &MountClipIdentityInput,
        fs: Arc<dyn SharedRootClipFileSystem>,
    ) -> Result<AddClipResult> {
        self.registry.insert_clip(source_path, identity_input, fs)
    }

    pub(crate) fn lookup(
        &self,
        parent_inode: u64,
        name: &OsStr,
    ) -> Result<Option<VirtualFileMetadata>> {
        if parent_inode == SHARED_ROOT_INODE {
            let Some(clip) = self.registry.lookup_by_folder_name(name) else {
                return Ok(None);
            };
            return self.metadata_for_clip_local_inode(clip, CLIP_DIR_INODE);
        }

        let Some((clip, local_parent_inode)) = self.resolve_global_inode(parent_inode)? else {
            return Ok(None);
        };
        let Some(metadata) = clip.fs.lookup(local_parent_inode, name)? else {
            return Ok(None);
        };

        self.remap_metadata(clip, metadata)
    }

    pub(crate) fn getattr(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
        if inode == SHARED_ROOT_INODE {
            return Ok(Some(self.root_metadata()));
        }

        let Some((clip, local_inode)) = self.resolve_global_inode(inode)? else {
            return Ok(None);
        };

        self.metadata_for_clip_local_inode(clip, local_inode)
    }

    pub(crate) fn readdir(&self, inode: u64) -> Result<Option<Vec<VirtualDirEntry>>> {
        if inode == SHARED_ROOT_INODE {
            return self.registry.root_entries().map(Some);
        }

        let Some((clip, local_inode)) = self.resolve_global_inode(inode)? else {
            return Ok(None);
        };
        let Some(entries) = clip.fs.readdir(local_inode)? else {
            return Ok(None);
        };

        entries
            .into_iter()
            .map(|entry| self.remap_dir_entry(clip, entry))
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    pub(crate) fn read_data(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Vec<u8>>> {
        if inode == SHARED_ROOT_INODE {
            return Ok(None);
        }

        let Some((clip, local_inode)) = self.resolve_global_inode(inode)? else {
            return Ok(None);
        };

        clip.fs.read_data(local_inode, offset, size)
    }

    pub(crate) fn read_data_for_handle(
        &self,
        inode: u64,
        handle: u64,
        offset: u64,
        size: u32,
    ) -> Result<Option<Vec<u8>>> {
        if inode == SHARED_ROOT_INODE {
            return Ok(None);
        }

        let Some((clip, local_inode)) = self.resolve_global_inode(inode)? else {
            return Ok(None);
        };

        clip.fs
            .read_data_for_handle(local_inode, handle, offset, size)
    }

    pub(crate) fn open_file_handle(&self, inode: u64, handle: u64) -> Result<()> {
        if inode == SHARED_ROOT_INODE {
            return Ok(());
        }

        let Some((clip, local_inode)) = self.resolve_global_inode(inode)? else {
            return Ok(());
        };

        clip.fs.open_file_handle(local_inode, handle)
    }

    pub(crate) fn release_file_handle(&self, inode: u64, handle: u64) {
        if inode == SHARED_ROOT_INODE {
            return;
        }

        if let Ok(Some((clip, local_inode))) = self.resolve_global_inode(inode) {
            clip.fs.release_file_handle(local_inode, handle);
        }
    }

    pub(crate) fn parent_inode_for_directory(&self, inode: u64) -> Option<u64> {
        if inode == SHARED_ROOT_INODE {
            return Some(SHARED_ROOT_INODE);
        }

        let aggregate = decode_aggregate_inode(inode).ok()??;
        let clip = self.registry.clip_for_slot(aggregate.slot)?;
        let local_parent = clip.fs.parent_inode_for_directory(aggregate.local_inode)?;

        if local_parent == ROOT_INODE {
            Some(SHARED_ROOT_INODE)
        } else {
            aggregate_inode_for_local(aggregate.slot, local_parent).ok()
        }
    }

    pub(crate) fn lookup_path(&self, relative_path: &Path) -> Result<Option<VirtualFileMetadata>> {
        let components = normalized_relative_components(relative_path)?;

        match components.as_slice() {
            [] => self.getattr(SHARED_ROOT_INODE),
            [folder_name] => self.lookup(SHARED_ROOT_INODE, folder_name.as_os_str()),
            [folder_name, child_name] => {
                let Some(folder_metadata) =
                    self.lookup(SHARED_ROOT_INODE, folder_name.as_os_str())?
                else {
                    return Ok(None);
                };
                self.lookup(folder_metadata.inode, child_name.as_os_str())
            }
            _ => Ok(None),
        }
    }

    fn metadata_for_clip_local_inode(
        &self,
        clip: &MountedClip,
        local_inode: u64,
    ) -> Result<Option<VirtualFileMetadata>> {
        let Some(metadata) = clip.fs.getattr(local_inode)? else {
            return Ok(None);
        };

        self.remap_metadata(clip, metadata)
    }

    fn remap_metadata(
        &self,
        clip: &MountedClip,
        mut metadata: VirtualFileMetadata,
    ) -> Result<Option<VirtualFileMetadata>> {
        metadata.inode = aggregate_inode_for_local(clip.slot, metadata.inode)?;
        Ok(Some(metadata))
    }

    fn remap_dir_entry(
        &self,
        clip: &MountedClip,
        mut entry: VirtualDirEntry,
    ) -> Result<VirtualDirEntry> {
        entry.inode = aggregate_inode_for_local(clip.slot, entry.inode)?;
        Ok(entry)
    }

    fn resolve_global_inode(&self, inode: u64) -> Result<Option<(&MountedClip, u64)>> {
        let Some(aggregate) = decode_aggregate_inode(inode)? else {
            return Ok(None);
        };

        let Some(clip) = self.registry.clip_for_slot(aggregate.slot) else {
            return Ok(None);
        };

        Ok(Some((clip, aggregate.local_inode)))
    }

    fn root_metadata(&self) -> VirtualFileMetadata {
        VirtualFileMetadata {
            inode: SHARED_ROOT_INODE,
            node: VirtualNode::Root,
            kind: VirtualFileKind::Directory,
            byte_len: 0,
            permissions: 0o755,
            hard_links: 2,
            timestamp: VirtualTimestamp::default(),
        }
    }
}

fn aggregate_inode_for_local(slot: MountedClipSlot, local_inode: u64) -> Result<u64> {
    if local_inode == ROOT_INODE {
        bail!("local root inode is not exposed inside shared-root clip folders");
    }
    if local_inode > LOCAL_INODE_MASK {
        bail!("local inode cannot be represented in shared-root aggregate inode namespace");
    }

    let slot_bits = u64::from(slot.0) << LOCAL_INODE_BITS;
    Ok(AGGREGATE_INODE_TAG | slot_bits | local_inode)
}

fn decode_aggregate_inode(inode: u64) -> Result<Option<AggregateInode>> {
    if inode & AGGREGATE_INODE_TAG == 0 {
        return Ok(None);
    }

    let slot_bits = (inode & !AGGREGATE_INODE_TAG) >> LOCAL_INODE_BITS;
    let slot = u16::try_from(slot_bits)
        .map(MountedClipSlot)
        .expect("decoded aggregate slot always fits in u16");
    let local_inode = inode & LOCAL_INODE_MASK;

    if local_inode == ROOT_INODE {
        bail!("shared-root aggregate inode maps to an invalid local root inode");
    }

    Ok(Some(AggregateInode { slot, local_inode }))
}

fn normalized_relative_components(path: &Path) -> Result<Vec<OsString>> {
    let mut components = Vec::new();

    if path.as_os_str().is_empty() {
        return Ok(components);
    }

    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(name) => components.push(name.to_os_string()),
            Component::ParentDir => bail!("parent traversal is not supported"),
            Component::Prefix(_) | Component::RootDir => bail!("absolute paths are not supported"),
        }
    }

    Ok(components)
}
