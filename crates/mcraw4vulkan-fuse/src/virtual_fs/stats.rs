use std::sync::atomic::{AtomicU64, Ordering};

// Runtime counters for the platform-neutral virtual filesystem facade.
//
// These counters are intentionally simple atomics so platform worker threads can
// update them without taking a global stats mutex. They are diagnostic counters,
// not correctness-critical state.
#[derive(Debug, Default)]
pub struct VirtualRuntimeStats {
    lookup_count: AtomicU64,
    getattr_count: AtomicU64,
    open_count: AtomicU64,
    opendir_count: AtomicU64,
    readdir_count: AtomicU64,
    read_count: AtomicU64,
    read_bytes_total: AtomicU64,
    read_size_requested_total: AtomicU64,
    read_size_min: AtomicU64,
    read_size_max: AtomicU64,
    read_offset_zero_count: AtomicU64,
    read_empty_count: AtomicU64,
    read_none_count: AtomicU64,
}

impl VirtualRuntimeStats {
    pub fn new() -> Self {
        Self {
            lookup_count: AtomicU64::new(0),
            getattr_count: AtomicU64::new(0),
            open_count: AtomicU64::new(0),
            opendir_count: AtomicU64::new(0),
            readdir_count: AtomicU64::new(0),
            read_count: AtomicU64::new(0),
            read_bytes_total: AtomicU64::new(0),
            read_size_requested_total: AtomicU64::new(0),
            read_size_min: AtomicU64::new(u64::MAX),
            read_size_max: AtomicU64::new(0),
            read_offset_zero_count: AtomicU64::new(0),
            read_empty_count: AtomicU64::new(0),
            read_none_count: AtomicU64::new(0),
        }
    }

    pub fn record_lookup(&self) {
        self.lookup_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_getattr(&self) {
        self.getattr_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_open(&self) {
        self.open_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_opendir(&self) {
        self.opendir_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_readdir(&self) {
        self.readdir_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_read_result(&self, offset: u64, requested_size: u32, returned_bytes: usize) {
        let requested_size = u64::from(requested_size);
        let returned_bytes = u64::try_from(returned_bytes).unwrap_or(u64::MAX);

        self.read_count.fetch_add(1, Ordering::Relaxed);
        self.read_bytes_total
            .fetch_add(returned_bytes, Ordering::Relaxed);
        self.read_size_requested_total
            .fetch_add(requested_size, Ordering::Relaxed);

        update_min(&self.read_size_min, requested_size);
        update_max(&self.read_size_max, requested_size);

        if offset == 0 {
            self.read_offset_zero_count.fetch_add(1, Ordering::Relaxed);
        }

        if returned_bytes == 0 && requested_size > 0 {
            self.read_empty_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_read_none(&self) {
        self.read_count.fetch_add(1, Ordering::Relaxed);
        self.read_none_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> VirtualRuntimeStatsSnapshot {
        let read_count = self.read_count.load(Ordering::Relaxed);
        let read_size_min = self.read_size_min.load(Ordering::Relaxed);

        VirtualRuntimeStatsSnapshot {
            lookup_count: self.lookup_count.load(Ordering::Relaxed),
            getattr_count: self.getattr_count.load(Ordering::Relaxed),
            open_count: self.open_count.load(Ordering::Relaxed),
            opendir_count: self.opendir_count.load(Ordering::Relaxed),
            readdir_count: self.readdir_count.load(Ordering::Relaxed),
            read: ReadStatsSnapshot {
                read_count,
                read_bytes_total: self.read_bytes_total.load(Ordering::Relaxed),
                read_size_requested_total: self.read_size_requested_total.load(Ordering::Relaxed),
                read_size_min: if read_count == 0 || read_size_min == u64::MAX {
                    0
                } else {
                    read_size_min
                },
                read_size_max: self.read_size_max.load(Ordering::Relaxed),
                read_offset_zero_count: self.read_offset_zero_count.load(Ordering::Relaxed),
                read_empty_count: self.read_empty_count.load(Ordering::Relaxed),
                read_none_count: self.read_none_count.load(Ordering::Relaxed),
            },
        }
    }
}

// Snapshot of virtual filesystem operation counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualRuntimeStatsSnapshot {
    pub lookup_count: u64,
    pub getattr_count: u64,
    pub open_count: u64,
    pub opendir_count: u64,
    pub readdir_count: u64,
    pub read: ReadStatsSnapshot,
}

// Snapshot of read-specific counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadStatsSnapshot {
    pub read_count: u64,
    pub read_bytes_total: u64,
    pub read_size_requested_total: u64,
    pub read_size_min: u64,
    pub read_size_max: u64,
    pub read_offset_zero_count: u64,
    pub read_empty_count: u64,
    pub read_none_count: u64,
}

impl ReadStatsSnapshot {
    pub fn average_requested_size(&self) -> u64 {
        self.read_size_requested_total
            .checked_div(self.read_count)
            .unwrap_or(0)
    }
}

// Atomically lower a counter to a new minimum value.
fn update_min(target: &AtomicU64, candidate: u64) {
    let mut current = target.load(Ordering::Relaxed);

    while candidate < current {
        match target.compare_exchange_weak(current, candidate, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => return,
            Err(next_current) => current = next_current,
        }
    }
}

// Atomically raise a counter to a new maximum value.
fn update_max(target: &AtomicU64, candidate: u64) {
    let mut current = target.load(Ordering::Relaxed);

    while candidate > current {
        match target.compare_exchange_weak(current, candidate, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => return,
            Err(next_current) => current = next_current,
        }
    }
}
