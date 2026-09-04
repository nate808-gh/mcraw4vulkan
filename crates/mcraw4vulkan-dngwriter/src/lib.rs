// Public entry point for the mcraw4vulkan DNG writer crate.
//
// This crate is responsible for turning a DngFrameDescription plus decoded or
// corrected u16 Bayer pixels into complete DNG bytes or files. It intentionally
// does not parse .mcraw files, decode frames, or apply post-decode correction.

mod description;
mod error;
mod plan;
mod sink;
mod writer;

pub use description::{
    CfaPattern, DngCompression, DngDescriptionError, DngFrameDescription,
    DngFrameDescriptionOverrides, DngPhotometricInterpretation,
};
pub use error::DngWriterError;
pub use plan::DngWritePlan;
pub use sink::{
    DngOutputCorrection, DngSinkDecodeSource, DngSinkFrame, DngSinkVignetteMode,
    apply_dng_output_correction_policy, build_dng_frame_description_for_sink,
};
pub use writer::{DngWriter, DngWriterConfig};
