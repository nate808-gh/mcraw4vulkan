use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitMode {
    RenderNow,
    WaitIndefinitely,
    WaitTimeout(Duration),
}

#[derive(Debug, Clone)]
pub struct LazyRepaintState {
    dirty: bool,
    repaint_deadline: Option<Instant>,
    fixed_cadence: Option<FixedCadenceState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FixedCadenceState {
    frame_duration: Duration,
    next_video_deadline: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LazyFrameDecision {
    pub active_vsync_fixed_cadence: bool,
    pub source_frame_duration: Option<Duration>,
    pub video_tick_due: bool,
    pub video_advance_allowed: bool,
    pub ui_only_repaint: bool,
}

impl LazyRepaintState {
    pub fn new(now: Instant) -> Self {
        Self {
            dirty: true,
            repaint_deadline: Some(now),
            fixed_cadence: None,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
        self.repaint_deadline = None;
    }

    pub fn mark_sdl_event(&mut self) {
        self.mark_dirty();
    }

    pub fn mark_resize(&mut self) {
        self.mark_dirty();
    }

    pub fn after_frame(&mut self, now: Instant, repaint_after: Duration) {
        self.after_frame_base(now, repaint_after);
        self.fixed_cadence = None;
    }

    // Cadence advances from the previous deadline to avoid drift; missed periods
    // are skipped so a late frame cannot trigger a burst of catch-up frames.
    pub fn after_frame_with_video_cadence(
        &mut self,
        now: Instant,
        repaint_after: Duration,
        active_vsync_frame_duration: Option<Duration>,
        video_frame_advanced: bool,
    ) {
        self.after_frame_base(now, repaint_after);
        let Some(frame_duration) =
            active_vsync_frame_duration.filter(|duration| !duration.is_zero())
        else {
            self.fixed_cadence = None;
            return;
        };

        let previous_deadline = self
            .fixed_cadence
            .filter(|cadence| cadence.frame_duration == frame_duration)
            .map(|cadence| cadence.next_video_deadline);
        let next_video_deadline = if video_frame_advanced {
            next_deadline_after_video_advance(previous_deadline, now, frame_duration)
        } else {
            previous_deadline
                .unwrap_or_else(|| initial_video_deadline(now, repaint_after, frame_duration))
        };
        self.fixed_cadence = Some(FixedCadenceState {
            frame_duration,
            next_video_deadline,
        });
        if repaint_after.is_zero() && next_video_deadline > now {
            self.dirty = false;
            self.repaint_deadline = None;
        }
    }

    fn after_frame_base(&mut self, now: Instant, repaint_after: Duration) {
        if repaint_after.is_zero() {
            self.dirty = true;
            self.repaint_deadline = Some(now);
        } else if repaint_after == Duration::MAX {
            self.dirty = false;
            self.repaint_deadline = None;
        } else {
            self.dirty = false;
            self.repaint_deadline = Some(now + repaint_after);
        }
    }

    pub fn wait_mode(&self, now: Instant) -> WaitMode {
        if self.dirty {
            return WaitMode::RenderNow;
        }

        match self.next_repaint_deadline() {
            Some(deadline) if deadline <= now => WaitMode::RenderNow,
            Some(deadline) => WaitMode::WaitTimeout(deadline.duration_since(now)),
            None => WaitMode::WaitIndefinitely,
        }
    }

    pub fn mark_elapsed_repaint_deadline(&mut self, now: Instant) {
        if self
            .next_repaint_deadline()
            .is_some_and(|deadline| deadline <= now)
        {
            self.dirty = true;
            if self
                .repaint_deadline
                .is_some_and(|deadline| deadline <= now)
            {
                self.repaint_deadline = None;
            }
        }
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    // UI dirtiness may request a repaint before the next video deadline, but that
    // repaint must reuse the current video frame rather than advance playback.
    pub fn frame_decision(&self, now: Instant) -> LazyFrameDecision {
        let Some(cadence) = self.fixed_cadence else {
            return LazyFrameDecision {
                active_vsync_fixed_cadence: false,
                source_frame_duration: None,
                video_tick_due: true,
                video_advance_allowed: true,
                ui_only_repaint: false,
            };
        };
        let video_tick_due = cadence.next_video_deadline <= now;
        LazyFrameDecision {
            active_vsync_fixed_cadence: true,
            source_frame_duration: Some(cadence.frame_duration),
            video_tick_due,
            video_advance_allowed: video_tick_due,
            ui_only_repaint: self.dirty && !video_tick_due,
        }
    }

    fn next_repaint_deadline(&self) -> Option<Instant> {
        match (
            self.repaint_deadline,
            self.fixed_cadence
                .map(|cadence| cadence.next_video_deadline),
        ) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (Some(deadline), None) | (None, Some(deadline)) => Some(deadline),
            (None, None) => None,
        }
    }
}

pub fn timeout_millis(timeout: Duration) -> u32 {
    let millis = timeout.as_millis();
    if millis == 0 {
        0
    } else {
        millis.min(u128::from(u32::MAX)) as u32
    }
}

fn is_waitable_repaint_after(repaint_after: Duration) -> bool {
    !repaint_after.is_zero() && repaint_after != Duration::MAX
}

fn initial_video_deadline(
    now: Instant,
    repaint_after: Duration,
    frame_duration: Duration,
) -> Instant {
    let delay = if is_waitable_repaint_after(repaint_after) {
        repaint_after.min(frame_duration)
    } else {
        frame_duration
    };
    now + delay
}

fn next_deadline_after_video_advance(
    previous_deadline: Option<Instant>,
    now: Instant,
    frame_duration: Duration,
) -> Instant {
    let Some(previous_deadline) = previous_deadline else {
        return now + frame_duration;
    };
    let next_deadline = previous_deadline + frame_duration;
    first_deadline_after(now, next_deadline, frame_duration)
}

fn first_deadline_after(
    now: Instant,
    candidate_deadline: Instant,
    frame_duration: Duration,
) -> Instant {
    if candidate_deadline > now {
        return candidate_deadline;
    }

    let frame_nanos = frame_duration.as_nanos().max(1);
    let late_nanos = now.duration_since(candidate_deadline).as_nanos();
    let skipped_periods = late_nanos / frame_nanos + 1;
    candidate_deadline + duration_mul_u128(frame_duration, skipped_periods)
}

fn duration_mul_u128(duration: Duration, factor: u128) -> Duration {
    let nanos = duration
        .as_nanos()
        .saturating_mul(factor)
        .min(u128::from(u64::MAX));
    Duration::from_nanos(nanos as u64)
}
