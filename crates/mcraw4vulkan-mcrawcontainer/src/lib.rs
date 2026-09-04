// Typed .mcraw container metadata, indexing, and payload access.
//
// This crate owns the file/container semantics. Decoder-facing crates can build
// CPU/GPU decode orchestration on top without duplicating parser or index state.

mod audio_metadata;
mod container;
mod container_metadata;
mod error;
mod frame_metadata;
mod index;
mod lens_shading_map;
mod parser;
mod payload;
pub mod payload_reader;
mod strict_color;
mod timing;

pub use audio_metadata::AudioDataMetadata;
pub use container::{McrawContainer, McrawContainerOpenPhaseTiming, McrawContainerOpenTimings};
pub use container_metadata::{
    BlackLevel, ColorIlluminant, ColorMatrix, ContainerMetadata, DeviceSpecificProfile,
    SensorArrangement, WhiteLevel,
};
pub use error::McrawContainerError;
pub use frame_metadata::{CompressionType, FrameMetadata, UnsupportedFramePayloadLayout};
pub use index::ClipIndex;
pub use lens_shading_map::{LensShadingMap, LensShadingMapValidationError};
pub use parser::{ParsedAudioChunk, ParsedAudioIndex, ParsedClip, ParsedFrame, parse_clip};
pub use payload::PayloadSpan;
pub use strict_color::{
    RawCamera2ColorProfile, RawCamera2ColorSourceError, RawCamera2FrameColor, RawCamera2Matrix,
    RawCamera2MatrixKind, RawColorCalibrationSlot, RawColorCalibrationSlotIndex,
    RawIlluminantToken, StrictColorProfileProvenance,
};
pub use timing::{AudioChunkTimingInfo, AudioSyncInfo, VideoFrameRateInfo};
