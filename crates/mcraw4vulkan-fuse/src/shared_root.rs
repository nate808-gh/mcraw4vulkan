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

    #[cfg(test)]
    fn insert_clip_with_forced_candidates(
        &mut self,
        source_path: PathBuf,
        candidates: Vec<MountClipFolderName>,
        fs: Arc<dyn SharedRootClipFileSystem>,
    ) -> Result<AddClipResult> {
        self.insert_clip_with_folder_name(source_path, candidates, fs)
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

    #[cfg(test)]
    pub(crate) fn from_clip_file_system(fs: Arc<dyn SharedRootClipFileSystem>) -> Self {
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use mcraw4vulkan_core::{
        DEFAULT_CLIP_MOUNT_SUFFIX_HEX_LEN, MountClipIdentity, clip_mount_folder_name,
        sanitize_mount_folder_stem,
    };

    use super::*;
    use crate::virtual_fs::AUDIO_WAV_INODE;

    const FAKE_FILE_INODE: u64 = 10;

    #[test]
    fn adding_first_clip_creates_one_root_folder() {
        let mut registry = MountedClipRegistry::new();
        let result = registry
            .insert_clip(
                PathBuf::from("clip.mcraw"),
                &identity_input("clip", "source/clip.mcraw", 100),
                fake_fs("clip", b"clip-bytes"),
            )
            .expect("insert clip");

        assert!(matches!(result, AddClipResult::Added(_)));
        assert_eq!(registry.len(), 1);

        let entries = registry.root_entries().expect("root entries");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, OsStr::new(result.folder_name()));
        assert_eq!(entries[0].node, VirtualNode::ClipDirectory);
    }

    #[test]
    fn adding_same_clip_identity_twice_returns_existing_folder() {
        let input = identity_input("clip", "source/clip.mcraw", 100);
        let mut registry = MountedClipRegistry::new();
        let first = registry
            .insert_clip(PathBuf::from("clip.mcraw"), &input, fake_fs("clip", b"one"))
            .expect("first insert");
        let second = registry
            .insert_clip(PathBuf::from("clip.mcraw"), &input, fake_fs("clip", b"two"))
            .expect("second insert");

        assert!(matches!(first, AddClipResult::Added(_)));
        assert!(matches!(second, AddClipResult::AlreadyMounted(_)));
        assert_eq!(registry.len(), 1);
        assert_eq!(first.folder_name(), second.folder_name());
    }

    #[test]
    fn adding_two_different_clips_with_same_sanitized_stem_uses_distinct_suffixes() {
        let mut registry = MountedClipRegistry::new();
        let first = registry
            .insert_clip(
                PathBuf::from("clip-a.mcraw"),
                &identity_input("clip", "source/a/clip.mcraw", 100),
                fake_fs("clip", b"one"),
            )
            .expect("first insert");
        let second = registry
            .insert_clip(
                PathBuf::from("clip-b.mcraw"),
                &identity_input("clip", "source/b/clip.mcraw", 100),
                fake_fs("clip", b"two"),
            )
            .expect("second insert");

        assert_ne!(first.folder_name(), second.folder_name());
        assert!(first.folder_name().starts_with("clip__"));
        assert!(second.folder_name().starts_with("clip__"));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn suffix_expansion_uses_policy_lengths_when_collisions_are_forced() {
        let mut registry = MountedClipRegistry::new();
        let stem = "clip";
        let identity = MountClipIdentity::from_xxh3_128(0xaaaa_aaaa_aaaa_bbbb_cccc_dddd_eeee_ffff);

        for suffix_len in [10, 12, 16, 24] {
            let suffix = format_clip_mount_hash_suffix(identity, suffix_len);
            let conflict =
                forced_folder_name(stem, &suffix, identity.as_u128() + suffix_len as u128);
            registry
                .insert_clip_with_forced_candidates(
                    PathBuf::from(format!("conflict-{suffix_len}.mcraw")),
                    vec![conflict],
                    fake_fs("clip", b"conflict"),
                )
                .expect("insert conflict");
        }

        let result = registry
            .insert_clip_with_forced_candidates(
                PathBuf::from("expanded.mcraw"),
                forced_candidates_for_identity(stem, identity),
                fake_fs("clip", b"expanded"),
            )
            .expect("insert expanded");

        let suffix = result.folder_name().rsplit_once("__").unwrap().1;
        assert_eq!(suffix.len(), 32);
        assert_eq!(suffix, format_clip_mount_hash_suffix(identity, 32).as_str());
    }

    #[test]
    fn deterministic_numeric_fallback_is_used_after_full_suffix_collision() {
        let mut registry = MountedClipRegistry::new();
        let stem = "clip";
        let identity = MountClipIdentity::from_xxh3_128(0x1111_1111_1111_2222_3333_4444_5555_6666);

        for suffix_len in [10, 12, 16, 24, 32] {
            let suffix = format_clip_mount_hash_suffix(identity, suffix_len);
            let conflict =
                forced_folder_name(stem, &suffix, identity.as_u128() + suffix_len as u128);
            registry
                .insert_clip_with_forced_candidates(
                    PathBuf::from(format!("conflict-{suffix_len}.mcraw")),
                    vec![conflict],
                    fake_fs("clip", b"conflict"),
                )
                .expect("insert conflict");
        }

        let result = registry
            .insert_clip_with_forced_candidates(
                PathBuf::from("numeric.mcraw"),
                forced_candidates_for_identity(stem, identity),
                fake_fs("clip", b"numeric"),
            )
            .expect("insert numeric fallback");

        assert_eq!(
            result.folder_name(),
            format!(
                "{}__{}__2",
                stem,
                format_clip_mount_hash_suffix(identity, 32)
            )
        );
    }

    #[test]
    fn root_readdir_lists_all_clip_folders() {
        let fs = shared_root_with_two_clips();
        let entries = fs
            .readdir(SHARED_ROOT_INODE)
            .expect("root readdir")
            .expect("root entries");

        assert_eq!(entries.len(), 2);
        assert!(
            entries
                .iter()
                .all(|entry| entry.node == VirtualNode::ClipDirectory)
        );
        assert_ne!(entries[0].name, entries[1].name);
    }

    #[test]
    fn top_level_clip_folder_getattr_is_directory_metadata() {
        let fs = shared_root_with_two_clips();
        let root_entries = fs.readdir(SHARED_ROOT_INODE).unwrap().unwrap();
        let metadata = fs
            .getattr(root_entries[0].inode)
            .expect("getattr")
            .expect("metadata");

        assert_eq!(metadata.kind, VirtualFileKind::Directory);
        assert_eq!(metadata.node, VirtualNode::ClipDirectory);
        assert_eq!(metadata.inode, root_entries[0].inode);
    }

    #[test]
    fn clip_folder_readdir_delegates_without_double_nesting() {
        let fs = shared_root_with_one_clip("clip", b"bytes");
        let clip_entry = fs.readdir(SHARED_ROOT_INODE).unwrap().unwrap().remove(0);
        let entries = fs
            .readdir(clip_entry.inode)
            .expect("clip readdir")
            .expect("clip entries");
        let names = entries
            .iter()
            .map(|entry| entry.name.clone())
            .collect::<Vec<_>>();

        assert!(names.contains(&OsString::from("clip_000000.dng")));
        assert!(names.contains(&OsString::from("clip.wav")));
        assert!(!names.contains(&clip_entry.name));
    }

    #[test]
    fn lookup_under_clip_folder_maps_to_correct_clip() {
        let fs = shared_root_with_two_clips();
        let root_entries = fs.readdir(SHARED_ROOT_INODE).unwrap().unwrap();
        let first_clip = &root_entries[0];
        let second_clip = &root_entries[1];

        let first_file = fs
            .lookup(first_clip.inode, OsStr::new("clip_a_000000.dng"))
            .expect("first lookup")
            .expect("first file");
        let second_file = fs
            .lookup(second_clip.inode, OsStr::new("clip_b_000000.dng"))
            .expect("second lookup")
            .expect("second file");

        assert_ne!(first_file.inode, second_file.inode);
        assert_eq!(
            fs.lookup(first_clip.inode, OsStr::new("clip_b_000000.dng"))
                .unwrap(),
            None
        );
    }

    #[test]
    fn read_data_under_clip_folder_maps_to_correct_clip_and_local_inode() {
        let fs = shared_root_with_two_clips();
        let root_entries = fs.readdir(SHARED_ROOT_INODE).unwrap().unwrap();
        let first_file = fs
            .lookup(root_entries[0].inode, OsStr::new("clip_a_000000.dng"))
            .unwrap()
            .unwrap();
        let second_file = fs
            .lookup(root_entries[1].inode, OsStr::new("clip_b_000000.dng"))
            .unwrap()
            .unwrap();

        assert_eq!(
            fs.read_data(first_file.inode, 0, 16).unwrap().unwrap(),
            b"alpha".to_vec()
        );
        assert_eq!(
            fs.read_data(second_file.inode, 1, 3).unwrap().unwrap(),
            b"eta".to_vec()
        );
    }

    #[test]
    fn identical_local_inode_values_do_not_collide_globally() {
        let fs = shared_root_with_two_clips();
        let root_entries = fs.readdir(SHARED_ROOT_INODE).unwrap().unwrap();
        let first_file = fs
            .lookup(root_entries[0].inode, OsStr::new("clip_a_000000.dng"))
            .unwrap()
            .unwrap();
        let second_file = fs
            .lookup(root_entries[1].inode, OsStr::new("clip_b_000000.dng"))
            .unwrap()
            .unwrap();

        assert_ne!(first_file.inode, second_file.inode);
        assert_eq!(
            decode_aggregate_inode(first_file.inode)
                .unwrap()
                .unwrap()
                .local_inode,
            FAKE_FILE_INODE
        );
        assert_eq!(
            decode_aggregate_inode(second_file.inode)
                .unwrap()
                .unwrap()
                .local_inode,
            FAKE_FILE_INODE
        );
    }

    #[test]
    fn unknown_top_level_folder_returns_not_found() {
        let fs = shared_root_with_one_clip("clip", b"bytes");

        assert_eq!(
            fs.lookup(SHARED_ROOT_INODE, OsStr::new("missing"))
                .expect("lookup"),
            None
        );
    }

    #[test]
    fn traversal_like_paths_are_rejected() {
        let fs = shared_root_with_one_clip("clip", b"bytes");
        let absolute_path = PathBuf::from(format!("{}absolute", std::path::MAIN_SEPARATOR));

        assert!(fs.lookup_path(Path::new("../clip")).is_err());
        assert!(fs.lookup_path(&absolute_path).is_err());
    }

    #[test]
    fn ascii_case_insensitive_folder_collision_is_detected() {
        let mut registry = MountedClipRegistry::new();
        registry
            .insert_clip_with_forced_candidates(
                PathBuf::from("upper.mcraw"),
                vec![forced_visible_folder_name(
                    "Clip__abcdef1234",
                    0x1000,
                    "Clip",
                    "abcdef1234",
                )],
                fake_fs("clip", b"upper"),
            )
            .expect("insert upper");

        let result = registry
            .insert_clip_with_forced_candidates(
                PathBuf::from("lower.mcraw"),
                vec![
                    forced_visible_folder_name("clip__abcdef1234", 0x2000, "clip", "abcdef1234"),
                    forced_visible_folder_name(
                        "clip__abcdef123456",
                        0x2000,
                        "clip",
                        "abcdef123456",
                    ),
                ],
                fake_fs("clip", b"lower"),
            )
            .expect("insert lower");

        assert_eq!(result.folder_name(), "clip__abcdef123456");
    }

    #[test]
    fn empty_registry_is_safe() {
        let fs = SharedRootVirtualFileSystem::new();

        assert_eq!(fs.registry().len(), 0);
        assert!(fs.registry().is_empty());
        assert!(fs.readdir(SHARED_ROOT_INODE).unwrap().unwrap().is_empty());
        assert!(fs.getattr(SHARED_ROOT_INODE).unwrap().is_some());
        assert_eq!(fs.read_data(SHARED_ROOT_INODE, 0, 8).unwrap(), None);
    }

    #[test]
    fn folder_names_use_core_naming_api() {
        let input = identity_input("clip:bad*name?", "source/clip.mcraw", 100);
        let expected = clip_mount_folder_name(&input);
        let mut registry = MountedClipRegistry::new();
        let result = registry
            .insert_clip(
                PathBuf::from("clip.mcraw"),
                &input,
                fake_fs("clip", b"bytes"),
            )
            .expect("insert clip");

        assert_eq!(result.folder_name(), expected.visible);
        assert_eq!(
            expected.sanitized_stem,
            sanitize_mount_folder_stem("clip:bad*name?")
        );
        assert_eq!(
            result.folder_name().rsplit_once("__").unwrap().1.len(),
            DEFAULT_CLIP_MOUNT_SUFFIX_HEX_LEN
        );
    }

    #[test]
    fn local_root_inode_is_rejected_in_aggregate_mapping() {
        assert!(aggregate_inode_for_local(MountedClipSlot(0), ROOT_INODE).is_err());
    }

    #[test]
    fn local_inode_overflow_is_rejected_in_aggregate_mapping() {
        assert!(aggregate_inode_for_local(MountedClipSlot(0), LOCAL_INODE_MASK + 1).is_err());
    }

    fn shared_root_with_one_clip(stem: &str, bytes: &[u8]) -> SharedRootVirtualFileSystem {
        let mut fs = SharedRootVirtualFileSystem::new();
        fs.insert_clip(
            PathBuf::from(format!("{stem}.mcraw")),
            &identity_input(stem, &format!("source/{stem}.mcraw"), bytes.len() as u64),
            fake_fs(stem, bytes),
        )
        .expect("insert clip");
        fs
    }

    fn shared_root_with_two_clips() -> SharedRootVirtualFileSystem {
        let mut fs = SharedRootVirtualFileSystem::new();
        fs.insert_clip(
            PathBuf::from("clip_a.mcraw"),
            &identity_input("clip_a", "source/a.mcraw", 5),
            fake_fs("clip_a", b"alpha"),
        )
        .expect("insert first");
        fs.insert_clip(
            PathBuf::from("clip_b.mcraw"),
            &identity_input("clip_b", "source/b.mcraw", 4),
            fake_fs("clip_b", b"beta"),
        )
        .expect("insert second");
        fs
    }

    fn identity_input(stem: &str, path: &str, file_len: u64) -> MountClipIdentityInput {
        let mut input = MountClipIdentityInput::new(stem, path);
        input.canonical_path = Some(format!("canonical/{path}"));
        input.file_len = Some(file_len);
        input
    }

    fn fake_fs(stem: &str, dng_bytes: &[u8]) -> Arc<dyn SharedRootClipFileSystem> {
        Arc::new(FakeClipFileSystem::new(stem, dng_bytes))
    }

    fn forced_candidates_for_identity(
        stem: &str,
        identity: MountClipIdentity,
    ) -> Vec<MountClipFolderName> {
        CLIP_MOUNT_SUFFIX_EXPANSION_HEX_LENGTHS
            .into_iter()
            .map(|suffix_len| {
                forced_folder_name(
                    stem,
                    &format_clip_mount_hash_suffix(identity, suffix_len),
                    identity.as_u128(),
                )
            })
            .collect()
    }

    fn forced_folder_name(stem: &str, suffix: &str, identity: u128) -> MountClipFolderName {
        forced_visible_folder_name(&format!("{stem}__{suffix}"), identity, stem, suffix)
    }

    fn forced_visible_folder_name(
        visible: &str,
        identity: u128,
        sanitized_stem: &str,
        suffix: &str,
    ) -> MountClipFolderName {
        MountClipFolderName {
            visible: visible.to_string(),
            sanitized_stem: sanitized_stem.to_string(),
            suffix: suffix.to_string(),
            identity: MountClipIdentity::from_xxh3_128(identity),
        }
    }

    struct FakeClipFileSystem {
        files: BTreeMap<OsString, FakeFile>,
        metadata: BTreeMap<u64, VirtualFileMetadata>,
    }

    impl FakeClipFileSystem {
        fn new(clip_stem: &str, dng_bytes: &[u8]) -> Self {
            let mut files = BTreeMap::new();
            let mut metadata = BTreeMap::new();
            let timestamp = VirtualTimestamp {
                seconds: 1_700_000_000,
                nanos: 123,
            };

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
