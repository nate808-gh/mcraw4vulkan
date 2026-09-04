#![forbid(unsafe_code)]

//! CPU video pixel decode backend for mcraw4vulkan.
//!
//! This crate will own only CPU video pixel decode: prepared video payloads and
//! decode parameters in, `DecodedBayerU16Frame` out.
//!
//! It must not grow into a general "anything CPU-side" crate. Audio decode,
//! metadata parsing, DNG writing, FUSE cache or prefetch policy, and mounted
//! filesystem layout belong in their dedicated crates.

pub mod error;
pub mod frame_decoder;
pub mod raw_decoder;
pub mod scratch;

pub use error::CpuDecodeError;
pub use frame_decoder::{CpuFrameDecoder, DecodeFrameTimings, DecodedFrameInfo, DecodedFrameRef};
