// Reusable CPU frame decode scratch memory.
//
// This reuses compressed payload storage across frames.
#[derive(Debug, Default)]
pub struct FrameScratch {
    compressed: Vec<u8>,
}

impl FrameScratch {
    pub fn new() -> Self {
        Self::default()
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

    pub fn compressed(&self) -> &[u8] {
        &self.compressed
    }

    pub fn compressed_mut(&mut self) -> &mut Vec<u8> {
        &mut self.compressed
    }
}
