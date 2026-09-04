use std::ffi::{OsStr, OsString};

// Stable inode constants for the virtual filesystem.
//
// These values should not depend on generation order, cache state, or whether a
// frame has already been decoded. DaVinci Resolve and desktop file browsers may
// ask for metadata repeatedly and expect stable answers.
pub const ROOT_INODE: u64 = 1;
pub const CLIP_DIR_INODE: u64 = 2;
pub const AUDIO_WAV_INODE: u64 = 3;
pub const FIRST_FRAME_INODE: u64 = 10;

// Virtual node identity inside one mounted .mcraw clip.
//
// This is intentionally filesystem-API-neutral. Platform callbacks can convert
// these identities into attributes, directory entries, and read behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualNode {
    Root,
    ClipDirectory,
    DngFrame { frame_index: usize },
    AudioWav,
}

// One stable directory entry.
//
// Platform readdir callbacks can use this as the source of truth for inode/name
// pairs. The DNG byte cache can use frame_index from VirtualNode::DngFrame when
// a read request needs complete frame bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualDirEntry {
    pub inode: u64,
    pub name: OsString,
    pub node: VirtualNode,
}

// Deterministic inode/name mapping for one mounted .mcraw clip.
//
// The mounted virtual filesystem exposes:
//
//   <mount_root>/
//     <clip_stem>/
//       <clip_stem>.wav
//       <clip_stem>_000000.dng
//       <clip_stem>_000001.dng
//       ...
//
// This layer deliberately does not know how to decode frames or read bytes. It
// only answers identity questions that platform lookup/getattr/readdir/read
// callbacks need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InodeMap {
    clip_stem: String,
    frame_count: usize,
    frame_number_width: usize,
    has_audio: bool,
}

impl InodeMap {
    // Create a deterministic map for one clip.
    //
    // The frame-number width is at least six digits to match common image
    // sequence naming. Larger clips automatically grow the width so names remain
    // sortable.
    pub fn new(clip_stem: impl Into<String>, frame_count: usize) -> Self {
        Self::new_with_audio(clip_stem, frame_count, true)
    }

    pub fn new_with_audio(
        clip_stem: impl Into<String>,
        frame_count: usize,
        has_audio: bool,
    ) -> Self {
        let frame_number_width = decimal_digits(frame_count.saturating_sub(1)).max(6);

        Self {
            clip_stem: clip_stem.into(),
            frame_count,
            frame_number_width,
            has_audio,
        }
    }

    pub fn clip_stem(&self) -> &str {
        &self.clip_stem
    }

    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub fn frame_number_width(&self) -> usize {
        self.frame_number_width
    }

    pub fn has_audio(&self) -> bool {
        self.has_audio
    }

    pub fn root_inode(&self) -> u64 {
        ROOT_INODE
    }

    pub fn clip_dir_inode(&self) -> u64 {
        CLIP_DIR_INODE
    }

    pub fn audio_inode(&self) -> u64 {
        AUDIO_WAV_INODE
    }

    pub fn first_frame_inode(&self) -> u64 {
        FIRST_FRAME_INODE
    }

    // Return the stable inode for one DNG frame index.
    pub fn frame_inode(&self, frame_index: usize) -> Option<u64> {
        if frame_index >= self.frame_count {
            return None;
        }

        FIRST_FRAME_INODE.checked_add(u64::try_from(frame_index).ok()?)
    }

    // Return the frame index represented by a frame inode.
    pub fn frame_index_for_inode(&self, inode: u64) -> Option<usize> {
        let offset = inode.checked_sub(FIRST_FRAME_INODE)?;
        let frame_index = usize::try_from(offset).ok()?;

        if frame_index < self.frame_count {
            Some(frame_index)
        } else {
            None
        }
    }

    // Return the virtual node represented by an inode.
    pub fn node_for_inode(&self, inode: u64) -> Option<VirtualNode> {
        match inode {
            ROOT_INODE => Some(VirtualNode::Root),
            CLIP_DIR_INODE => Some(VirtualNode::ClipDirectory),
            AUDIO_WAV_INODE if self.has_audio => Some(VirtualNode::AudioWav),
            _ => self
                .frame_index_for_inode(inode)
                .map(|frame_index| VirtualNode::DngFrame { frame_index }),
        }
    }

    // Return the deterministic filename for one DNG frame.
    //
    // Frame numbers are zero-based:
    //
    //   <clip_stem>_000000.dng
    //   <clip_stem>_000001.dng
    pub fn frame_filename(&self, frame_index: usize) -> Option<String> {
        if frame_index >= self.frame_count {
            return None;
        }

        Some(format!(
            "{}_{:0width$}.dng",
            self.clip_stem,
            frame_index,
            width = self.frame_number_width
        ))
    }

    // Return the deterministic BW64 WAV filename.
    pub fn audio_filename(&self) -> String {
        format!("{}.wav", self.clip_stem)
    }

    // Return the stable inode for a child file inside the clip directory.
    pub fn inode_for_name(&self, name: &OsStr) -> Option<u64> {
        if self.has_audio && name == OsStr::new(&self.audio_filename()) {
            return Some(AUDIO_WAV_INODE);
        }

        let frame_index = self.frame_index_for_name(name)?;
        self.frame_inode(frame_index)
    }

    // Return the virtual node for a child file inside the clip directory.
    pub fn node_for_name(&self, name: &OsStr) -> Option<VirtualNode> {
        if self.has_audio && name == OsStr::new(&self.audio_filename()) {
            return Some(VirtualNode::AudioWav);
        }

        self.frame_index_for_name(name)
            .map(|frame_index| VirtualNode::DngFrame { frame_index })
    }

    // Return the child name for an inode.
    //
    // Root itself has no child name, so ROOT_INODE returns None.
    pub fn child_name_for_inode(&self, inode: u64) -> Option<OsString> {
        match inode {
            ROOT_INODE => None,
            CLIP_DIR_INODE => Some(OsString::from(self.clip_stem())),
            AUDIO_WAV_INODE if self.has_audio => Some(OsString::from(self.audio_filename())),
            _ => {
                let frame_index = self.frame_index_for_inode(inode)?;
                self.frame_filename(frame_index).map(OsString::from)
            }
        }
    }

    // Return one directory child entry for a name.
    pub fn lookup_child(&self, parent_inode: u64, name: &OsStr) -> Option<VirtualDirEntry> {
        match parent_inode {
            ROOT_INODE => self.lookup_root_child(name),
            CLIP_DIR_INODE => self.lookup_clip_dir_child(name),
            _ => None,
        }
    }

    // Return stable root-directory entries.
    //
    // The root contains exactly one clip directory named after the source stem.
    pub fn root_directory_entries(&self) -> Vec<VirtualDirEntry> {
        vec![VirtualDirEntry {
            inode: CLIP_DIR_INODE,
            name: OsString::from(self.clip_stem()),
            node: VirtualNode::ClipDirectory,
        }]
    }

    // Return stable clip-directory entries.
    //
    // DNG frames are listed in frame order, followed by the BW64 WAV file. This
    // ordering is deterministic and independent of cache state.
    pub fn clip_directory_entries(&self) -> Vec<VirtualDirEntry> {
        let mut entries =
            Vec::with_capacity(self.frame_count.saturating_add(usize::from(self.has_audio)));

        for frame_index in 0..self.frame_count {
            let Some(inode) = self.frame_inode(frame_index) else {
                continue;
            };
            let Some(name) = self.frame_filename(frame_index) else {
                continue;
            };

            entries.push(VirtualDirEntry {
                inode,
                name: OsString::from(name),
                node: VirtualNode::DngFrame { frame_index },
            });
        }

        if self.has_audio {
            entries.push(VirtualDirEntry {
                inode: AUDIO_WAV_INODE,
                name: OsString::from(self.audio_filename()),
                node: VirtualNode::AudioWav,
            });
        }

        entries
    }

    // Return directory entries for a directory inode.
    pub fn directory_entries(&self, inode: u64) -> Option<Vec<VirtualDirEntry>> {
        match inode {
            ROOT_INODE => Some(self.root_directory_entries()),
            CLIP_DIR_INODE => Some(self.clip_directory_entries()),
            _ => None,
        }
    }

    // Return the parent inode for a directory inode.
    pub fn parent_inode_for_directory(&self, inode: u64) -> Option<u64> {
        match inode {
            ROOT_INODE => Some(ROOT_INODE),
            CLIP_DIR_INODE => Some(ROOT_INODE),
            _ => None,
        }
    }

    fn lookup_root_child(&self, name: &OsStr) -> Option<VirtualDirEntry> {
        if name != OsStr::new(self.clip_stem()) {
            return None;
        }

        Some(VirtualDirEntry {
            inode: CLIP_DIR_INODE,
            name: name.to_os_string(),
            node: VirtualNode::ClipDirectory,
        })
    }

    fn lookup_clip_dir_child(&self, name: &OsStr) -> Option<VirtualDirEntry> {
        let inode = self.inode_for_name(name)?;
        let node = self.node_for_inode(inode)?;

        Some(VirtualDirEntry {
            inode,
            name: name.to_os_string(),
            node,
        })
    }

    // Parse a deterministic DNG filename back into a frame index.
    fn frame_index_for_name(&self, name: &OsStr) -> Option<usize> {
        let name = name.to_str()?;
        let prefix = format!("{}_", self.clip_stem);
        let digits = name.strip_prefix(&prefix)?.strip_suffix(".dng")?;

        if digits.len() != self.frame_number_width {
            return None;
        }

        if !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }

        let frame_index = digits.parse::<usize>().ok()?;

        if frame_index >= self.frame_count {
            return None;
        }

        let expected_name = self.frame_filename(frame_index)?;

        if name == expected_name {
            Some(frame_index)
        } else {
            None
        }
    }
}

// Count base-10 digits in a non-negative usize.
fn decimal_digits(mut value: usize) -> usize {
    let mut digits = 1;

    while value >= 10 {
        value /= 10;
        digits += 1;
    }

    digits
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::{
        AUDIO_WAV_INODE, CLIP_DIR_INODE, FIRST_FRAME_INODE, InodeMap, ROOT_INODE, VirtualNode,
    };

    #[test]
    fn maps_clip_directory_frames_and_audio_to_stable_inodes() {
        let map = InodeMap::new("bigfile", 5574);

        assert_eq!(map.root_inode(), ROOT_INODE);
        assert_eq!(map.clip_dir_inode(), CLIP_DIR_INODE);
        assert_eq!(map.audio_inode(), AUDIO_WAV_INODE);
        assert_eq!(map.frame_inode(0), Some(FIRST_FRAME_INODE));
        assert_eq!(map.frame_inode(5573), Some(FIRST_FRAME_INODE + 5573));
        assert_eq!(map.frame_inode(5574), None);

        assert_eq!(map.node_for_inode(ROOT_INODE), Some(VirtualNode::Root));
        assert_eq!(
            map.node_for_inode(CLIP_DIR_INODE),
            Some(VirtualNode::ClipDirectory)
        );
        assert_eq!(
            map.node_for_inode(AUDIO_WAV_INODE),
            Some(VirtualNode::AudioWav)
        );
        assert_eq!(map.frame_index_for_inode(FIRST_FRAME_INODE), Some(0));
        assert_eq!(
            map.frame_index_for_inode(FIRST_FRAME_INODE + 5573),
            Some(5573)
        );
        assert_eq!(map.frame_index_for_inode(FIRST_FRAME_INODE + 5574), None);
    }

    #[test]
    fn round_trips_deterministic_filenames_inside_clip_directory() {
        let map = InodeMap::new("bigfile", 5574);

        assert_eq!(
            map.frame_filename(87).as_deref(),
            Some("bigfile_000087.dng")
        );
        assert_eq!(map.audio_filename(), "bigfile.wav");

        assert_eq!(
            map.inode_for_name(OsStr::new("bigfile_000087.dng")),
            Some(FIRST_FRAME_INODE + 87)
        );
        assert_eq!(
            map.node_for_name(OsStr::new("bigfile_000087.dng")),
            Some(VirtualNode::DngFrame { frame_index: 87 })
        );
        assert_eq!(
            map.inode_for_name(OsStr::new("bigfile.wav")),
            Some(AUDIO_WAV_INODE)
        );
    }

    #[test]
    fn root_contains_only_clip_directory() {
        let map = InodeMap::new("bigfile", 2);
        let entries = map.root_directory_entries();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].inode, CLIP_DIR_INODE);
        assert_eq!(entries[0].name, OsStr::new("bigfile"));
        assert_eq!(entries[0].node, VirtualNode::ClipDirectory);
    }

    #[test]
    fn clip_directory_contains_frames_and_audio() {
        let map = InodeMap::new("bigfile", 2);
        let entries = map.clip_directory_entries();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, OsStr::new("bigfile_000000.dng"));
        assert_eq!(entries[1].name, OsStr::new("bigfile_000001.dng"));
        assert_eq!(entries[2].name, OsStr::new("bigfile.wav"));
    }

    #[test]
    fn no_audio_clip_directory_omits_audio_inode_and_filename() {
        let map = InodeMap::new_with_audio("bigfile", 2, false);
        let entries = map.clip_directory_entries();

        assert!(!map.has_audio());
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, OsStr::new("bigfile_000000.dng"));
        assert_eq!(entries[1].name, OsStr::new("bigfile_000001.dng"));
        assert_eq!(map.node_for_inode(AUDIO_WAV_INODE), None);
        assert_eq!(map.inode_for_name(OsStr::new("bigfile.wav")), None);
        assert_eq!(map.node_for_name(OsStr::new("bigfile.wav")), None);
        assert_eq!(map.child_name_for_inode(AUDIO_WAV_INODE), None);
    }
}
