// Lightweight file span for raw .mcraw payload access.
//
// The container stays file-backed; callers can inspect spans for planning and
// use read_into APIs to reuse allocations without introducing borrowed slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadSpan {
    pub offset: u64,
    pub len: u64,
}
