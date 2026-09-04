// MotionCam raw video payload layout constants.
//
// These describe the compressed raw codec payload, not the broader .mcraw clip
// container.

// One encoded raw block represents 64 Bayer samples.
pub const ENCODING_BLOCK: usize = 64;

// Packed metadata streams use a two-byte block header.
pub const HEADER_LENGTH: usize = 2;

// The fixed raw payload header is 16 bytes; block payload bytes begin after it.
pub const METADATA_OFFSET: usize = 16;

// Compressed payload byte length for each BlockEncoding discriminant.
pub const ENCODING_BLOCK_LENGTH: [usize; 17] = [
    0, 8, 16, 24, 32, 40, 48, 64, 64, 80, 80, 128, 128, 128, 128, 128, 128,
];
