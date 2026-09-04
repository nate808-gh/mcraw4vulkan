use std::time::Duration;

use crate::{BAYER_CFA_PLANE_COUNT, VignetteCorrectionMode};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VignetteCorrectedFrameInfo {
    pub applied: bool,
    pub input_black_level: [f32; BAYER_CFA_PLANE_COUNT],
    pub output_black_level: [f32; BAYER_CFA_PLANE_COUNT],
    pub output_white_level: u16,
    pub mode: VignetteCorrectionMode,
}

impl VignetteCorrectedFrameInfo {
    pub fn from_correction_mode(
        mode: VignetteCorrectionMode,
        input_black_level: [f32; BAYER_CFA_PLANE_COUNT],
        output_white_level: u16,
    ) -> Self {
        let applied = mode == VignetteCorrectionMode::Enabled;
        let output_black_level = if applied {
            [0.0; BAYER_CFA_PLANE_COUNT]
        } else {
            input_black_level
        };

        Self {
            applied,
            input_black_level,
            output_black_level,
            output_white_level,
            mode,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VignetteCorrectionStats {
    pub applied: bool,
    pub corrected_pixel_count: u64,
    pub cpu_passthrough_without_copy: bool,
    pub cpu_output_buffer_bytes: Option<u64>,
    pub lens_shading_map_dimensions: Option<(u32, u32)>,
    pub lens_shading_plane_count: Option<usize>,
    pub full_resolution_gain_map_bytes: Option<u64>,
    pub gpu_input_buffer_bytes: Option<u64>,
    pub gpu_output_buffer_bytes: Option<u64>,
    pub gpu_packed_word_count: Option<u64>,
    pub gpu_max_storage_buffer_binding_size: Option<u64>,
    pub gpu_min_storage_buffer_offset_alignment: Option<u64>,
    pub gpu_max_compute_workgroups_per_dimension: Option<u64>,
    pub gpu_tiling_used: bool,
    pub gpu_tile_count: Option<u64>,
    pub gpu_tile_dispatch_count: Option<u64>,
    pub gpu_max_tile_gain_bytes: Option<u64>,
    pub gpu_max_tile_workgroups: Option<u64>,
    pub gpu_dispatch_submitted: bool,
    pub gpu_gain_buffer_reused: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VignetteCorrectionTimings {
    pub total: Option<Duration>,
    pub cpu_loop: Option<Duration>,
    pub gpu_upload: Option<Duration>,
    pub gpu_dispatch: Option<Duration>,
    pub gpu_readback: Option<Duration>,
}
