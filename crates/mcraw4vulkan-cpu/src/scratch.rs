use mcraw4vulkan_core::FrameDimensions;

use crate::error::CpuDecodeError;

// Reusable CPU frame decode scratch memory.
//
// This avoids allocating fresh compressed/pixel/temp buffers for every frame.
#[derive(Debug, Default)]
pub struct FrameScratch {
    compressed: Vec<u8>,
    pixels: Vec<u16>,
    temp: Vec<u16>,
}

impl FrameScratch {
    pub fn new() -> Self {
        Self::default()
    }

    // Prepare pixel/temp buffers for a visible output frame.
    //
    // This validates that the pixel count can fit in usize before any reserve
    // call, preventing malformed metadata from causing absurd allocations.
    pub fn prepare_for_frame(&mut self, dimensions: FrameDimensions) -> Result<(), CpuDecodeError> {
        let pixel_count = dimensions
            .pixel_count()
            .ok_or(CpuDecodeError::FrameDimensionsOverflow)?;

        // Keep the decoded pixel buffer at the visible frame length.
        //
        // The first frame allocates and initializes the buffer. Later frames with
        // the same dimensions reuse the existing initialized memory without
        // clearing or zero-filling it. The raw decoder is responsible for
        // overwriting every visible output pixel.
        if self.pixels.len() != pixel_count {
            self.pixels.resize(pixel_count, 0);
        }

        // Reserve temp independently and leave its length at zero, reusing the
        // allocation without clearing the decoded pixel output.
        if self.temp.capacity() < pixel_count {
            self.temp.reserve(pixel_count - self.temp.capacity());
        }
        self.temp.clear();

        Ok(())
    }

    // Reserve storage for the compressed frame payload.
    //
    // The actual file read resizes and fills this buffer. This method keeps
    // capacity reuse explicit at the caller/session layer.
    pub fn prepare_compressed(&mut self, byte_count: usize) {
        if self.compressed.capacity() < byte_count {
            self.compressed
                .reserve(byte_count - self.compressed.capacity());
        }
        self.compressed.clear();
    }

    pub fn pixels(&self) -> &[u16] {
        &self.pixels
    }

    pub fn pixels_mut(&mut self) -> &mut Vec<u16> {
        &mut self.pixels
    }

    pub fn compressed(&self) -> &[u8] {
        &self.compressed
    }

    pub fn compressed_mut(&mut self) -> &mut Vec<u8> {
        &mut self.compressed
    }

    pub fn temp_mut(&mut self) -> &mut Vec<u16> {
        &mut self.temp
    }

    // Borrow compressed input and mutable pixel output at the same time.
    //
    // The raw decoder needs both slices simultaneously. This method exposes
    // disjoint fields safely without forcing extra payload copies.
    pub fn compressed_and_pixels_mut(&mut self) -> (&[u8], &mut Vec<u16>) {
        (&self.compressed, &mut self.pixels)
    }
}
