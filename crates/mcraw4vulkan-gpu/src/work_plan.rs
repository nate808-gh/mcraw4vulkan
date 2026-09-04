use anyhow::{anyhow, Result};
use mcraw4vulkan_core::FrameDimensions;
use mcraw4vulkan_rawcodec::{
    block_encoding_from_raw, block_len, decode_metadata, read_metadata_header, MetadataHeader,
    ENCODING_BLOCK, METADATA_OFFSET,
};

// CPU-built descriptor plan for one MCRAW GPU decode.
//
// It turns the serial payload walk into checked block byte ranges, reference
// values, lane identities, and output locations. The backend can then decode
// blocks in parallel without rediscovering payload offsets.
//
// Metadata expansion remains on the CPU so scalar and GPU paths share one raw
// payload interpretation.
#[derive(Debug, Clone)]
pub struct McrawVulkanWorkPlan {
    pub encoded_dimensions: FrameDimensions,
    pub visible_dimensions: FrameDimensions,
    pub payload_base_offset: u32,
    pub blocks: Vec<McrawVulkanBlockWorkItem>,
}

impl McrawVulkanWorkPlan {
    pub fn as_ref(&self) -> McrawVulkanWorkPlanRef<'_> {
        McrawVulkanWorkPlanRef {
            encoded_dimensions: self.encoded_dimensions,
            visible_dimensions: self.visible_dimensions,
            payload_base_offset: self.payload_base_offset,
            blocks: &self.blocks,
            blocks_capacity: self.blocks.capacity(),
            build_stats: McrawVulkanWorkPlanBuildStats {
                reused_scratch: false,
                grow_count: 0,
            },
        }
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct McrawVulkanWorkPlanRef<'a> {
    pub encoded_dimensions: FrameDimensions,
    pub visible_dimensions: FrameDimensions,
    pub payload_base_offset: u32,
    pub blocks: &'a [McrawVulkanBlockWorkItem],
    pub blocks_capacity: usize,
    pub build_stats: McrawVulkanWorkPlanBuildStats,
}

impl McrawVulkanWorkPlanRef<'_> {
    pub fn block_count(self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(self) -> bool {
        self.blocks.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct McrawVulkanWorkPlanBuildStats {
    pub reused_scratch: bool,
    pub grow_count: u64,
}

#[derive(Debug, Default)]
pub struct McrawVulkanWorkPlanScratch {
    blocks: Vec<McrawVulkanBlockWorkItem>,
    last_build_stats: McrawVulkanWorkPlanBuildStats,
}

impl McrawVulkanWorkPlanScratch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            blocks: Vec::with_capacity(capacity),
            last_build_stats: McrawVulkanWorkPlanBuildStats::default(),
        }
    }

    pub fn clear_for_next_frame(&mut self) {
        self.blocks.clear();
        self.last_build_stats = McrawVulkanWorkPlanBuildStats::default();
    }

    pub fn blocks_capacity(&self) -> usize {
        self.blocks.capacity()
    }

    pub fn blocks_len(&self) -> usize {
        self.blocks.len()
    }

    pub fn last_build_stats(&self) -> McrawVulkanWorkPlanBuildStats {
        self.last_build_stats
    }
}

// One independently decodable 64-sample MCRAW block.
//
// Four of these descriptors are generated for each encoded 64x4 macroblock:
// lane 0: rows y/y+2, even columns
// lane 1: rows y/y+2, odd columns
// lane 2: rows y+1/y+3, even columns
// lane 3: rows y+1/y+3, odd columns
//
// lane_index and the macroblock coordinates determine the shader's output
// positions, so descriptor order must remain identical to metadata lane order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct McrawVulkanBlockWorkItem {
    pub payload_offset: u32,
    pub payload_len: u32,
    pub raw_encoding: u32,
    pub reference_value: u32,
    pub macroblock_x: u32,
    pub macroblock_y: u32,
    pub lane_index: u32,
    pub reserved: u32,
}

// Build a work plan from one raw frame payload.
//
// This mirrors the scalar decoder's serial offset walk but records each checked
// byte range instead of decoding it. Keeping the walk CPU-side avoids a device
// prefix sum and rejects invalid ranges before descriptor upload.
pub fn build_vulkan_work_plan(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
) -> Result<McrawVulkanWorkPlan> {
    let header = read_metadata_header(raw_payload)
        .ok_or_else(|| anyhow!("missing raw payload metadata header"))?;

    let mut blocks = Vec::new();
    build_vulkan_work_plan_blocks(raw_payload, visible_dimensions, &header, &mut blocks)?;

    Ok(McrawVulkanWorkPlan {
        encoded_dimensions: FrameDimensions {
            width: header.encoded_width,
            height: header.encoded_height,
        },
        visible_dimensions,
        payload_base_offset: METADATA_OFFSET as u32,
        blocks,
    })
}

pub fn build_vulkan_work_plan_into<'a>(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    scratch: &'a mut McrawVulkanWorkPlanScratch,
) -> Result<McrawVulkanWorkPlanRef<'a>> {
    let header = read_metadata_header(raw_payload)
        .ok_or_else(|| anyhow!("missing raw payload metadata header"))?;

    let capacity_before = scratch.blocks.capacity();
    scratch.clear_for_next_frame();
    build_vulkan_work_plan_blocks(
        raw_payload,
        visible_dimensions,
        &header,
        &mut scratch.blocks,
    )?;
    let capacity_after = scratch.blocks.capacity();
    scratch.last_build_stats = McrawVulkanWorkPlanBuildStats {
        reused_scratch: capacity_before >= scratch.blocks.len() && capacity_before > 0,
        grow_count: u64::from(capacity_after > capacity_before),
    };

    Ok(McrawVulkanWorkPlanRef {
        encoded_dimensions: FrameDimensions {
            width: header.encoded_width,
            height: header.encoded_height,
        },
        visible_dimensions,
        payload_base_offset: METADATA_OFFSET as u32,
        blocks: &scratch.blocks,
        blocks_capacity: scratch.blocks.capacity(),
        build_stats: scratch.last_build_stats,
    })
}

fn build_vulkan_work_plan_blocks(
    raw_payload: &[u8],
    visible_dimensions: FrameDimensions,
    header: &MetadataHeader,
    blocks: &mut Vec<McrawVulkanBlockWorkItem>,
) -> Result<()> {
    validate_work_plan_header(raw_payload, header, visible_dimensions)?;

    let bits = decode_metadata(raw_payload, header.bits_offset as usize)?;
    let refs = decode_metadata(raw_payload, header.refs_offset as usize)?;

    let encoded_width = header.encoded_width as usize;
    let encoded_height = header.encoded_height as usize;
    let expected_metadata_blocks = (encoded_height / 4) * (encoded_width / ENCODING_BLOCK) * 4;

    if bits.len() < expected_metadata_blocks || refs.len() < expected_metadata_blocks {
        return Err(anyhow!("metadata streams are shorter than expected"));
    }

    blocks.clear();
    blocks.reserve(expected_metadata_blocks);
    let mut payload_offset = METADATA_OFFSET;
    let mut metadata_idx = 0usize;

    let mut y = 0usize;
    while y < encoded_height {
        let mut x = 0usize;

        while x < encoded_width {
            let mut lane_index = 0usize;

            while lane_index < 4 {
                let raw_encoding = bits[metadata_idx + lane_index];
                let reference_value = refs[metadata_idx + lane_index];
                let encoding = block_encoding_from_raw(raw_encoding)?;
                let payload_len = block_len(encoding);
                let payload_end = payload_offset
                    .checked_add(payload_len)
                    .ok_or_else(|| anyhow!("frame block payload offset overflow"))?;

                if payload_end > raw_payload.len() {
                    return Err(anyhow!("frame block payload range exceeds raw payload"));
                }

                blocks.push(McrawVulkanBlockWorkItem {
                    payload_offset: usize_to_u32(payload_offset, "payload offset")?,
                    payload_len: usize_to_u32(payload_len, "payload length")?,
                    raw_encoding: u32::from(raw_encoding),
                    reference_value: u32::from(reference_value),
                    macroblock_x: usize_to_u32(x, "macroblock x")?,
                    macroblock_y: usize_to_u32(y, "macroblock y")?,
                    lane_index: usize_to_u32(lane_index, "lane index")?,
                    reserved: 0,
                });

                payload_offset = payload_end;
                lane_index += 1;
            }

            metadata_idx += 4;
            x += ENCODING_BLOCK;
        }

        y += 4;
    }

    Ok(())
}

// Validate only the constraints needed to create a safe GPU work plan.
//
// This intentionally mirrors the raw decoder's sanity checks so malformed input
// cannot create absurd descriptor counts or invalid payload ranges.
fn validate_work_plan_header(
    raw_payload: &[u8],
    header: &MetadataHeader,
    visible_dimensions: FrameDimensions,
) -> Result<()> {
    if header.encoded_width == 0 || header.encoded_height == 0 {
        return Err(anyhow!("encoded dimensions must be non-zero"));
    }

    if header.encoded_width as usize & (ENCODING_BLOCK - 1) != 0 {
        return Err(anyhow!("encoded width must be a multiple of 64"));
    }

    if header.encoded_height & 3 != 0 {
        return Err(anyhow!("encoded height must be a multiple of 4"));
    }

    if visible_dimensions.width > header.encoded_width
        || visible_dimensions.height > header.encoded_height
    {
        return Err(anyhow!("visible dimensions exceed encoded dimensions"));
    }

    if header.bits_offset as usize > raw_payload.len()
        || header.refs_offset as usize > raw_payload.len()
    {
        return Err(anyhow!("metadata offsets are outside the raw payload"));
    }

    const MAX_REASONABLE_WIDTH: u32 = 16_384;
    const MAX_REASONABLE_HEIGHT: u32 = 16_384;
    const MAX_REASONABLE_PIXELS: u64 = 100_000_000;

    if header.encoded_width > MAX_REASONABLE_WIDTH || header.encoded_height > MAX_REASONABLE_HEIGHT
    {
        return Err(anyhow!("encoded dimensions are unreasonably large"));
    }

    let pixel_count = u64::from(visible_dimensions.width) * u64::from(visible_dimensions.height);
    if pixel_count > MAX_REASONABLE_PIXELS {
        return Err(anyhow!("pixel count is unreasonably large"));
    }

    Ok(())
}

// Convert descriptor fields to u32 because GPU storage buffers should use fixed
// width values. Supported frame and payload bounds fit below u32::MAX, so any
// larger value is rejected before descriptor serialization.
fn usize_to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| anyhow!("{label} does not fit in u32"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_zero_payload(width: u32, height: u32) -> Vec<u8> {
        let expected_metadata_blocks =
            (height as usize / 4) * (width as usize / ENCODING_BLOCK) * 4;
        let bits_stream = metadata_zero_stream(expected_metadata_blocks);
        let refs_stream = metadata_zero_stream(expected_metadata_blocks);
        let bits_offset = METADATA_OFFSET;
        let refs_offset = bits_offset + bits_stream.len();
        let mut payload = vec![0u8; refs_offset + refs_stream.len()];
        payload[0..4].copy_from_slice(&width.to_le_bytes());
        payload[4..8].copy_from_slice(&height.to_le_bytes());
        payload[8..12].copy_from_slice(&(bits_offset as u32).to_le_bytes());
        payload[12..16].copy_from_slice(&(refs_offset as u32).to_le_bytes());
        payload[bits_offset..bits_offset + bits_stream.len()].copy_from_slice(&bits_stream);
        payload[refs_offset..refs_offset + refs_stream.len()].copy_from_slice(&refs_stream);
        payload
    }

    fn metadata_zero_stream(num_blocks: usize) -> Vec<u8> {
        let mut stream = Vec::new();
        stream.extend_from_slice(&(num_blocks as u32).to_le_bytes());
        for _ in (0..num_blocks).step_by(ENCODING_BLOCK) {
            stream.extend_from_slice(&[0, 0]);
        }
        stream
    }

    fn descriptor_bytes(blocks: &[McrawVulkanBlockWorkItem]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(blocks.len() * 32);
        for block in blocks {
            bytes.extend_from_slice(&block.payload_offset.to_le_bytes());
            bytes.extend_from_slice(&block.payload_len.to_le_bytes());
            bytes.extend_from_slice(&block.raw_encoding.to_le_bytes());
            bytes.extend_from_slice(&block.reference_value.to_le_bytes());
            bytes.extend_from_slice(&block.macroblock_x.to_le_bytes());
            bytes.extend_from_slice(&block.macroblock_y.to_le_bytes());
            bytes.extend_from_slice(&block.lane_index.to_le_bytes());
            bytes.extend_from_slice(&block.reserved.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn work_plan_scratch_matches_owned_builder() {
        let payload = synthetic_zero_payload(128, 8);
        let dimensions = FrameDimensions {
            width: 128,
            height: 8,
        };
        let owned = build_vulkan_work_plan(&payload, dimensions).unwrap();
        let mut scratch = McrawVulkanWorkPlanScratch::new();
        let scratch_plan = build_vulkan_work_plan_into(&payload, dimensions, &mut scratch).unwrap();

        assert_eq!(owned.block_count(), scratch_plan.block_count());
        assert_eq!(owned.blocks, scratch_plan.blocks);
        assert_eq!(
            descriptor_bytes(&owned.blocks),
            descriptor_bytes(scratch_plan.blocks)
        );
        assert_eq!(owned.encoded_dimensions, scratch_plan.encoded_dimensions);
        assert_eq!(owned.visible_dimensions, scratch_plan.visible_dimensions);
        assert_eq!(owned.payload_base_offset, scratch_plan.payload_base_offset);
    }

    #[test]
    fn work_plan_scratch_reuses_capacity() {
        let payload = synthetic_zero_payload(128, 8);
        let dimensions = FrameDimensions {
            width: 128,
            height: 8,
        };
        let mut scratch = McrawVulkanWorkPlanScratch::new();
        let first = build_vulkan_work_plan_into(&payload, dimensions, &mut scratch).unwrap();
        let first_len = first.block_count();
        let first_capacity = scratch.blocks_capacity();

        let second = build_vulkan_work_plan_into(&payload, dimensions, &mut scratch).unwrap();
        let second_count = second.block_count();
        let second_stats = second.build_stats;

        assert_eq!(second_count, first_len);
        assert!(second_stats.reused_scratch);
        assert_eq!(second_stats.grow_count, 0);
        assert_eq!(scratch.blocks_capacity(), first_capacity);
    }

    #[test]
    fn work_plan_scratch_grows_when_needed() {
        let small_payload = synthetic_zero_payload(64, 4);
        let large_payload = synthetic_zero_payload(256, 16);
        let small_dimensions = FrameDimensions {
            width: 64,
            height: 4,
        };
        let large_dimensions = FrameDimensions {
            width: 256,
            height: 16,
        };
        let mut scratch = McrawVulkanWorkPlanScratch::new();
        build_vulkan_work_plan_into(&small_payload, small_dimensions, &mut scratch).unwrap();
        let small_capacity = scratch.blocks_capacity();
        let large_owned = build_vulkan_work_plan(&large_payload, large_dimensions).unwrap();
        let large_scratch =
            build_vulkan_work_plan_into(&large_payload, large_dimensions, &mut scratch).unwrap();
        let large_scratch_blocks = large_scratch.blocks.to_vec();
        let large_scratch_grow_count = large_scratch.build_stats.grow_count;

        assert!(scratch.blocks_capacity() > small_capacity);
        assert_eq!(large_owned.blocks, large_scratch_blocks);
        assert_eq!(large_scratch_grow_count, 1);
    }

    #[test]
    fn work_plan_scratch_clear_no_stale_blocks() {
        let large_payload = synthetic_zero_payload(256, 16);
        let small_payload = synthetic_zero_payload(64, 4);
        let large_dimensions = FrameDimensions {
            width: 256,
            height: 16,
        };
        let small_dimensions = FrameDimensions {
            width: 64,
            height: 4,
        };
        let mut scratch = McrawVulkanWorkPlanScratch::new();
        let large_plan =
            build_vulkan_work_plan_into(&large_payload, large_dimensions, &mut scratch).unwrap();
        assert!(large_plan.block_count() > 4);

        let small_owned = build_vulkan_work_plan(&small_payload, small_dimensions).unwrap();
        let small_plan =
            build_vulkan_work_plan_into(&small_payload, small_dimensions, &mut scratch).unwrap();

        assert_eq!(small_plan.block_count(), small_owned.block_count());
        assert_eq!(small_plan.blocks, small_owned.blocks);
        assert_eq!(
            descriptor_bytes(small_plan.blocks).len(),
            small_owned.blocks.len() * 32
        );
    }
}
