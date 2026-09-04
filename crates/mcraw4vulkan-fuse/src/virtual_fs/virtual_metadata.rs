use std::ffi::OsStr;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};

use crate::virtual_fs::{
    AudioWavMetadata, DngFrameByteCache, InodeMap, LazyAudioWav, VirtualDirEntry, VirtualNode,
};
use crate::virtual_timestamp::VirtualTimestamp;

// File kind for the virtual filesystem metadata layer.
//
// This is intentionally independent of any platform callback crate type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualFileKind {
    Directory,
    RegularFile,
}

// Stable metadata for one virtual node.
//
// Platform getattr callbacks translate this into file attributes. For DNG
// frames, byte_len is computed from DNG metadata/layout only and does not force
// raw decode or full DNG byte generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualFileMetadata {
    pub inode: u64,
    pub node: VirtualNode,
    pub kind: VirtualFileKind,
    pub byte_len: u64,
    pub permissions: u16,
    pub hard_links: u32,
    pub timestamp: VirtualTimestamp,
}

// Complete bytes for a regular virtual file.
//
// This is primarily for DNG validation/cache checks. Platform read callbacks
// should use range reads so audio stays lazy and does not materialize the full
// BW64 file.
#[derive(Debug, Clone)]
pub struct VirtualFileBytes {
    pub inode: u64,
    pub node: VirtualNode,
    pub bytes: Arc<[u8]>,
}

// Metadata and byte provider for one mounted .mcraw clip.
//
// This combines:
// - deterministic inode/name identity
// - cheap DNG metadata sizes
// - complete cached DNG bytes for reads
// - lazy BW64 WAV range reads
// - stable capture-time or fallback timestamps
//
// Platform-specific filesystem adapters should be thin wrappers around this
// shared virtual metadata behavior.
pub struct VirtualMetadataProvider {
    inode_map: InodeMap,
    dng_cache: Arc<DngFrameByteCache>,
    audio_wav: Option<Mutex<LazyAudioWav>>,
    audio_metadata: Option<AudioWavMetadata>,
    timestamp: VirtualTimestamp,
}

impl VirtualMetadataProvider {
    pub fn new(
        inode_map: InodeMap,
        dng_cache: Arc<DngFrameByteCache>,
        audio_wav: LazyAudioWav,
    ) -> Self {
        Self::new_with_timestamp(inode_map, dng_cache, audio_wav, VirtualTimestamp::default())
    }

    pub fn new_with_timestamp(
        inode_map: InodeMap,
        dng_cache: Arc<DngFrameByteCache>,
        audio_wav: LazyAudioWav,
        timestamp: VirtualTimestamp,
    ) -> Self {
        Self::new_optional_audio_with_timestamp(inode_map, dng_cache, Some(audio_wav), timestamp)
    }

    pub fn new_optional_audio_with_timestamp(
        inode_map: InodeMap,
        dng_cache: Arc<DngFrameByteCache>,
        audio_wav: Option<LazyAudioWav>,
        timestamp: VirtualTimestamp,
    ) -> Self {
        debug_assert_eq!(inode_map.has_audio(), audio_wav.is_some());
        let audio_metadata = audio_wav.as_ref().map(LazyAudioWav::metadata);
        Self {
            inode_map,
            dng_cache,
            audio_wav: audio_wav.map(Mutex::new),
            audio_metadata,
            timestamp,
        }
    }

    pub fn inode_map(&self) -> &InodeMap {
        &self.inode_map
    }

    pub fn audio_metadata(&self) -> Option<AudioWavMetadata> {
        self.audio_metadata
    }

    pub fn timestamp(&self) -> VirtualTimestamp {
        self.timestamp
    }

    // Return stable metadata for one inode.
    //
    // For DNG frames, this does not generate full DNG bytes. It only computes
    // the final DNG byte length from metadata/layout.
    pub fn metadata_for_inode(&self, inode: u64) -> Result<Option<VirtualFileMetadata>> {
        let Some(node) = self.inode_map.node_for_inode(inode) else {
            return Ok(None);
        };

        self.metadata_for_node(inode, node)
    }

    // Return stable metadata for one child name under a directory inode.
    pub fn metadata_for_child_name(
        &self,
        parent_inode: u64,
        name: &OsStr,
    ) -> Result<Option<VirtualFileMetadata>> {
        let Some(entry) = self.inode_map.lookup_child(parent_inode, name) else {
            return Ok(None);
        };

        self.metadata_for_node(entry.inode, entry.node)
    }

    // Return root-directory entries from the identity layer.
    pub fn root_directory_entries(&self) -> Vec<VirtualDirEntry> {
        self.inode_map.root_directory_entries()
    }

    // Return clip-directory entries from the identity layer.
    pub fn clip_directory_entries(&self) -> Vec<VirtualDirEntry> {
        self.inode_map.clip_directory_entries()
    }

    // Return directory entries for a directory inode.
    pub fn directory_entries(&self, inode: u64) -> Option<Vec<VirtualDirEntry>> {
        self.inode_map.directory_entries(inode)
    }

    // Return the parent inode for a directory inode.
    pub fn parent_inode_for_directory(&self, inode: u64) -> Option<u64> {
        self.inode_map.parent_inode_for_directory(inode)
    }

    // Return complete bytes for a regular file inode.
    //
    // Directories return None. Unknown inodes also return None. This path can
    // force DNG byte generation or full audio materialization, so platform
    // filesystem reads should use read_range/read_data instead.
    pub fn bytes_for_inode(&self, inode: u64) -> Result<Option<VirtualFileBytes>> {
        self.bytes_for_inode_with_demand(inode, true)
    }

    pub fn bytes_for_inode_with_demand(
        &self,
        inode: u64,
        count_as_demand: bool,
    ) -> Result<Option<VirtualFileBytes>> {
        let Some(node) = self.inode_map.node_for_inode(inode) else {
            return Ok(None);
        };

        match node {
            VirtualNode::Root | VirtualNode::ClipDirectory => Ok(None),
            VirtualNode::AudioWav => {
                let Some(bytes) = self.materialize_audio_wav()? else {
                    return Ok(None);
                };
                Ok(Some(VirtualFileBytes { inode, node, bytes }))
            }
            VirtualNode::DngFrame { frame_index } => {
                let cached = self
                    .dng_cache
                    .get_or_generate_for_read(frame_index, count_as_demand)
                    .with_context(|| format!("failed to get cached DNG frame {frame_index}"))?;

                Ok(Some(VirtualFileBytes {
                    inode,
                    node,
                    bytes: cached.bytes.clone(),
                }))
            }
        }
    }

    // Return a byte slice range for a regular virtual file.
    //
    // This helper mirrors the behavior platform read callbacks need: offsets
    // past EOF return an empty buffer, and reads that extend past EOF are
    // truncated.
    pub fn read_range(&self, inode: u64, offset: u64, size: u32) -> Result<Option<Vec<u8>>> {
        if matches!(
            self.inode_map.node_for_inode(inode),
            Some(VirtualNode::AudioWav)
        ) {
            return self.read_audio_range(offset, size);
        }

        let Some(file_bytes) = self.bytes_for_inode(inode)? else {
            return Ok(None);
        };

        let start = usize::try_from(offset).context("read offset does not fit usize")?;
        let requested = usize::try_from(size).context("read size does not fit usize")?;

        if start >= file_bytes.bytes.len() {
            return Ok(Some(Vec::new()));
        }

        let end = start.saturating_add(requested).min(file_bytes.bytes.len());

        Ok(Some(file_bytes.bytes[start..end].to_vec()))
    }

    // Build metadata for a known inode/node pair.
    fn metadata_for_node(
        &self,
        inode: u64,
        node: VirtualNode,
    ) -> Result<Option<VirtualFileMetadata>> {
        let metadata = match node {
            VirtualNode::Root => VirtualFileMetadata {
                inode,
                node,
                kind: VirtualFileKind::Directory,
                byte_len: 0,
                permissions: 0o755,
                hard_links: 2,
                timestamp: self.timestamp,
            },
            VirtualNode::ClipDirectory => VirtualFileMetadata {
                inode,
                node,
                kind: VirtualFileKind::Directory,
                byte_len: 0,
                permissions: 0o755,
                hard_links: 2,
                timestamp: self.timestamp,
            },
            VirtualNode::AudioWav => VirtualFileMetadata {
                inode,
                node,
                kind: VirtualFileKind::RegularFile,
                byte_len: self
                    .audio_metadata
                    .context("audio WAV metadata requested for no-audio clip")?
                    .byte_len,
                permissions: 0o444,
                hard_links: 1,
                timestamp: self.timestamp,
            },
            VirtualNode::DngFrame { frame_index } => {
                let byte_len = self.dng_cache.dng_byte_len(frame_index).with_context(|| {
                    format!("failed to compute DNG byte length for frame {frame_index}")
                })?;

                VirtualFileMetadata {
                    inode,
                    node,
                    kind: VirtualFileKind::RegularFile,
                    byte_len,
                    permissions: 0o444,
                    hard_links: 1,
                    timestamp: self.timestamp,
                }
            }
        };

        Ok(Some(metadata))
    }

    fn read_audio_range(&self, offset: u64, size: u32) -> Result<Option<Vec<u8>>> {
        let Some(audio_metadata) = self.audio_metadata else {
            return Ok(None);
        };
        let Some(audio_wav) = &self.audio_wav else {
            return Ok(None);
        };
        let requested = usize::try_from(size).context("read size does not fit usize")?;
        if requested == 0 || offset >= audio_metadata.byte_len {
            return Ok(Some(Vec::new()));
        }

        let readable = u64::from(size).min(audio_metadata.byte_len.saturating_sub(offset));
        let mut output =
            vec![0u8; usize::try_from(readable).context("audio read length does not fit usize")?];
        let returned_len = audio_wav
            .lock()
            .map_err(|_| anyhow!("lazy audio WAV mutex poisoned"))?
            .read_at(offset, &mut output)
            .context("failed to read lazy audio WAV range")?;
        output.truncate(returned_len);

        Ok(Some(output))
    }

    fn materialize_audio_wav(&self) -> Result<Option<Arc<[u8]>>> {
        let Some(audio_metadata) = self.audio_metadata else {
            return Ok(None);
        };
        let Some(audio_wav) = &self.audio_wav else {
            return Ok(None);
        };
        let byte_len = usize::try_from(audio_metadata.byte_len)
            .context("audio WAV byte length does not fit usize")?;
        let mut bytes = vec![0u8; byte_len];
        let returned_len = audio_wav
            .lock()
            .map_err(|_| anyhow!("lazy audio WAV mutex poisoned"))?
            .read_at(0, &mut bytes)
            .context("failed to materialize lazy audio WAV bytes")?;
        bytes.truncate(returned_len);

        Ok(Some(Arc::from(bytes.into_boxed_slice())))
    }
}
