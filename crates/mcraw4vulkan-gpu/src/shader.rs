// Shared GPU shader source and shader-contract constants for mcraw4vulkan.
//
// Keeping these sources and byte-layout constants independent of device state
// lets CPU serialization and WGSL bindings share one reviewable contract.

// WGSL compute shader that decodes descriptor-described MCRAW blocks into packed
// 16-bit Bayer samples, storing two decoded RAW samples per u32 output word.
//
// The lower-indexed raster sample occupies bits 0..15 and the next sample
// occupies bits 16..31, matching the common Bayer U16 byte contract.
pub const MCRAW_DECODE_BLOCKS_PACKED_U16_WGSL: &str =
    include_str!("shaders/mcraw_decode_blocks_packed_u16.wgsl");

// WGSL compute shader that decodes legacy compressionType 6 raw16 binned
// payload blocks directly into packed 16-bit Bayer samples.
pub const MCRAW_DECODE_LEGACY_RAW16_PACKED_U16_WGSL: &str =
    include_str!("shaders/mcraw_decode_legacy_raw16_packed_u16.wgsl");

// One workgroup decodes one 64x4 macroblock.
pub const MCRAW_DECODE_WORKGROUP_SIZE: u32 = 256;

// Each macroblock has four 64-sample lane descriptors.
pub const MCRAW_DESCRIPTORS_PER_MACROBLOCK: u32 = 4;

// Each descriptor decodes 64 samples.
pub const MCRAW_SAMPLES_PER_DESCRIPTOR: u32 = 64;

// The unpacked layout reserves one u32 word per output pixel.
pub const MCRAW_GPU_OUTPUT_BYTES_PER_PIXEL: u32 = 4;

// The packed layout stores two U16 Bayer samples per u32 word. An even
// pixel count therefore uses two output bytes per pixel.
pub const MCRAW_GPU_PACKED_OUTPUT_BYTES_PER_PIXEL: u32 = 2;
