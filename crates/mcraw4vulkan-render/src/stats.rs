use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PreviewRenderStats {
    pub input_bytes: u64,
    pub texture_bytes: u64,
    pub tile_count: usize,
    pub max_tile_pixels: u64,
    pub texture_create: Duration,
    pub params_upload: Duration,
    pub bind_group: Duration,
    pub dispatch_encode: Duration,
}
