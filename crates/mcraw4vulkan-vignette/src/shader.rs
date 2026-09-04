pub const VIGNETTE_CORRECT_PACKED_U16_WGSL: &str =
    include_str!("shaders/vignette_correct_packed_u16.wgsl");
pub const PIPE_CORRECT_BAYER_F32_WGSL: &str = include_str!("shaders/pipe_correct_bayer_f32.wgsl");

pub const VIGNETTE_CORRECT_WORKGROUP_SIZE: u32 = 256;
pub const PIPE_CORRECT_BAYER_F32_WORKGROUP_SIZE: u32 = 256;
pub const VIGNETTE_PACKED_U16_SAMPLES_PER_WORD: u32 = 2;
pub const VIGNETTE_PACKED_U16_BYTES_PER_WORD: u32 = 4;
