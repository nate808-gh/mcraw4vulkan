//! Backend-neutral MotionCam raw payload codec primitives.
//!
//! This crate owns only raw video payload layout, block unpacking, and metadata
//! stream expansion. It does not parse `.mcraw` containers, orchestrate decode
//! sessions, own GPU runtime code, or serialize outputs.

mod block;
mod constants;
mod error;
mod metadata;

pub use block::{block_len, decode_block};
pub use constants::{ENCODING_BLOCK, ENCODING_BLOCK_LENGTH, HEADER_LENGTH, METADATA_OFFSET};
pub use error::RawCodecError;
pub use metadata::{
    BlockHeader, MetadataHeader, block_encoding_from_raw, decode_header, decode_metadata,
    read_metadata_header,
};
