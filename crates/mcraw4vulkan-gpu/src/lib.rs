// Reusable wgpu GPU decode runtime.
//
// CPU work plans validate payload ranges and serialize fixed GPU descriptors;
// the backend owns pipelines, reusable buffers, and mapped-buffer teardown.

mod backend;
mod shader;
mod work_plan;

pub use backend::GpuBackendPreference;
pub use backend::GpuDecodeBackend;
pub use backend::GpuDecodeConfig;
pub use backend::GpuDecodeRequiredLimits;
pub use backend::GpuDecodeScratch;
pub use backend::GpuDecodeTimings;
pub use backend::GpuDecodedPackedU16BufferView;
pub use backend::GpuExternalReadbackBuffer;
pub use backend::GpuMappedBatchInput;
pub use backend::GpuMappedGpuStageOutput;
pub use backend::GpuMappedPackedU16Output;
pub use backend::GpuMappedPackedU16WithVignetteOutput;
pub use backend::GpuMappedRingFrame;
pub use backend::GpuMappedRingOutput;
pub use backend::GpuMappedRingStats;
pub use backend::GpuMappedRingVignetteCorrection;
pub use backend::GpuNoReadbackGpuStageOutput;
pub use backend::GpuNoReadbackSlotAllocation;
pub use backend::GpuPreparedType6WorkPlan;
pub use backend::GpuPreparedType7WorkPlan;
pub use backend::GpuSubmittedNoReadbackGpuStageOutput;
pub use backend::GpuSubmittedTimestampProfile;
pub use backend::GpuTimestampProfile;
pub use backend::GpuTimestampSupport;
pub use backend::OptionalGpuVignetteCorrection;
pub use shader::{
    MCRAW_DECODE_BLOCKS_PACKED_U16_WGSL, MCRAW_DECODE_LEGACY_RAW16_PACKED_U16_WGSL,
    MCRAW_DECODE_WORKGROUP_SIZE, MCRAW_DESCRIPTORS_PER_MACROBLOCK,
    MCRAW_GPU_OUTPUT_BYTES_PER_PIXEL, MCRAW_GPU_PACKED_OUTPUT_BYTES_PER_PIXEL,
    MCRAW_SAMPLES_PER_DESCRIPTOR,
};
pub use work_plan::{
    build_vulkan_work_plan, build_vulkan_work_plan_into, McrawVulkanBlockWorkItem,
    McrawVulkanWorkPlan, McrawVulkanWorkPlanBuildStats, McrawVulkanWorkPlanRef,
    McrawVulkanWorkPlanScratch,
};
