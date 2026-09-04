// Shared core domain types for mcraw4vulkan.
//
// This crate should stay lightweight and OS-agnostic. It is the right place for
// simple shared types used by the app, GUI, CLI, decoder, GPU backend, player,
// and platform filesystem adapters.
//
// Keep GPU, GUI, mount, and other implementation-specific dependencies out of
// this crate.

pub mod frame_index;
pub mod mount_naming;

mod clip_id;
mod decode_backend_choice;
mod decoded_bayer_u16_frame;
mod types;

pub use clip_id::RegisteredClipId;
pub use decode_backend_choice::{DecodeBackendChoice, ParseDecodeBackendChoiceError};
pub use decoded_bayer_u16_frame::{DecodedBayerU16Frame, DecodedBayerU16FrameValidationError};
pub use mount_naming::{
    CLIP_MOUNT_SUFFIX_EXPANSION_HEX_LENGTHS, DEFAULT_CLIP_MOUNT_SUFFIX_HEX_LEN,
    MountClipFolderName, MountClipIdentity, MountClipIdentityInput, MountClipModifiedTime,
    clip_mount_folder_name, clip_mount_folder_name_with_suffix_len, folder_name_for_path,
    format_clip_mount_hash_suffix, sanitize_mount_folder_stem, stable_clip_mount_hash,
};

pub use types::{
    AudioSampleRange, AudioTrackInfo, BayerPattern, BlockEncoding, ClipId, ClipTimingInfo,
    ContainerFlavor, DecodeBackend, FrameDimensions, FrameNumber, FramePayloadLayout, FrameRate,
    McrawClipInfo, TimelineFrameRateSource,
};
