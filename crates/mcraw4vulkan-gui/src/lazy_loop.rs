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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_frame_is_dirty() {
        let now = Instant::now();
        let scheduler = LazyRepaintState::new(now);

        assert_eq!(scheduler.wait_mode(now), WaitMode::RenderNow);
        assert!(scheduler.is_dirty());
    }

    #[test]
    fn idle_after_frame_without_repaint_request_blocks_indefinitely() {
        let now = Instant::now();
        let mut scheduler = LazyRepaintState::new(now);
        scheduler.after_frame(now, Duration::MAX);

        assert_eq!(scheduler.wait_mode(now), WaitMode::WaitIndefinitely);
        assert!(!scheduler.is_dirty());
    }

    #[test]
    fn sdl_event_marks_dirty() {
        let now = Instant::now();
        let mut scheduler = LazyRepaintState::new(now);
        scheduler.after_frame(now, Duration::MAX);
        scheduler.mark_sdl_event();

        assert_eq!(scheduler.wait_mode(now), WaitMode::RenderNow);
    }

    #[test]
    fn resize_marks_dirty() {
        let now = Instant::now();
        let mut scheduler = LazyRepaintState::new(now);
        scheduler.after_frame(now, Duration::MAX);
        scheduler.mark_resize();

        assert!(scheduler.is_dirty());
    }

    #[test]
    fn finite_repaint_after_creates_timeout() {
        let now = Instant::now();
        let mut scheduler = LazyRepaintState::new(now);
        scheduler.after_frame(now, Duration::from_millis(250));

        assert_eq!(
            scheduler.wait_mode(now),
            WaitMode::WaitTimeout(Duration::from_millis(250))
        );
    }

    #[test]
    fn elapsed_repaint_after_marks_dirty() {
        let now = Instant::now();
        let mut scheduler = LazyRepaintState::new(now);
        scheduler.after_frame(now, Duration::from_millis(5));
        scheduler.mark_elapsed_repaint_deadline(now + Duration::from_millis(6));

        assert!(scheduler.is_dirty());
    }

    #[test]
    fn no_spin_when_no_dirty_flag_and_no_deadline() {
        let now = Instant::now();
        let mut scheduler = LazyRepaintState::new(now);
        scheduler.after_frame(now, Duration::MAX);

        assert_eq!(scheduler.wait_mode(now), WaitMode::WaitIndefinitely);
    }

    #[test]
    fn positive_submillisecond_timeout_uses_nonblocking_poll() {
        assert_eq!(timeout_millis(Duration::from_nanos(1)), 0);
        assert_eq!(timeout_millis(Duration::from_micros(999)), 0);
        assert_eq!(timeout_millis(Duration::from_millis(1)), 1);
    }

    #[test]
    fn active_vsync_cadence_creates_video_deadline_even_when_lazy_idle() {
        let now = Instant::now();
        let mut scheduler = LazyRepaintState::new(now);
        scheduler.after_frame_with_video_cadence(
            now,
            Duration::MAX,
            Some(Duration::from_millis(33)),
            true,
        );

        assert_eq!(
            scheduler.wait_mode(now),
            WaitMode::WaitTimeout(Duration::from_millis(33))
        );
    }

    #[test]
    fn active_vsync_event_before_video_tick_is_ui_only() {
        let now = Instant::now();
        let mut scheduler = LazyRepaintState::new(now);
        scheduler.after_frame_with_video_cadence(
            now,
            Duration::MAX,
            Some(Duration::from_millis(33)),
            true,
        );
        scheduler.mark_sdl_event();

        let decision = scheduler.frame_decision(now + Duration::from_millis(10));

        assert!(decision.active_vsync_fixed_cadence);
        assert!(!decision.video_advance_allowed);
        assert!(decision.ui_only_repaint);
    }

    #[test]
    fn active_vsync_due_video_tick_allows_video_advance() {
        let now = Instant::now();
        let mut scheduler = LazyRepaintState::new(now);
        scheduler.after_frame_with_video_cadence(
            now,
            Duration::MAX,
            Some(Duration::from_millis(33)),
            true,
        );
        let due = now + Duration::from_millis(33);
        scheduler.mark_elapsed_repaint_deadline(due);

        let decision = scheduler.frame_decision(due);

        assert!(decision.video_tick_due);
        assert!(decision.video_advance_allowed);
    }

    #[test]
    fn active_vsync_zero_repaint_before_video_tick_waits_for_video_tick() {
        let now = Instant::now();
        let mut scheduler = LazyRepaintState::new(now);
        scheduler.after_frame_with_video_cadence(
            now,
            Duration::MAX,
            Some(Duration::from_millis(33)),
            true,
        );

        let ui_repaint = now + Duration::from_millis(10);
        scheduler.after_frame_with_video_cadence(
            ui_repaint,
            Duration::ZERO,
            Some(Duration::from_millis(33)),
            false,
        );

        assert_eq!(
            scheduler.wait_mode(ui_repaint),
            WaitMode::WaitTimeout(Duration::from_millis(23))
        );
    }

    #[test]
    fn max_speed_without_fixed_cadence_keeps_video_advance_allowed() {
        let now = Instant::now();
        let mut scheduler = LazyRepaintState::new(now);
        scheduler.after_frame_with_video_cadence(now, Duration::ZERO, None, true);

        let decision = scheduler.frame_decision(now);

        assert!(!decision.active_vsync_fixed_cadence);
        assert!(decision.video_advance_allowed);
    }

    #[test]
    fn active_vsync_uses_source_frame_duration_without_30fps_coercion() {
        let now = Instant::now();
        let mut scheduler = LazyRepaintState::new(now);
        let source_24fps = Duration::from_secs_f64(1.0 / 24.0);
        scheduler.after_frame_with_video_cadence(now, Duration::MAX, Some(source_24fps), true);

        let decision = scheduler.frame_decision(now + Duration::from_millis(1));

        assert_eq!(decision.source_frame_duration, Some(source_24fps));
    }

    #[test]
    fn active_vsync_30fps_deadline_does_not_accumulate_small_lateness() {
        let start = Instant::now();
        let frame_duration = Duration::from_secs_f64(1.0 / 30.0);
        let lateness = Duration::from_micros(600);
        let mut scheduler = LazyRepaintState::new(start);
        scheduler.after_frame_with_video_cadence(start, Duration::MAX, Some(frame_duration), true);
        let mut expected_deadline = start + frame_duration;

        for _ in 0..300 {
            let wake = expected_deadline + lateness;
            let decision = scheduler.frame_decision(wake);
            assert!(decision.video_advance_allowed);

            scheduler.after_frame_with_video_cadence(
                wake,
                Duration::MAX,
                Some(frame_duration),
                true,
            );
            expected_deadline += frame_duration;

            let cadence = scheduler.fixed_cadence.expect("active cadence");
            assert_eq!(cadence.next_video_deadline, expected_deadline);
            assert_eq!(
                scheduler.wait_mode(wake),
                WaitMode::WaitTimeout(frame_duration - lateness)
            );
        }
    }

    #[test]
    fn active_vsync_24fps_deadline_does_not_accumulate_small_lateness() {
        let start = Instant::now();
        let frame_duration = Duration::from_secs_f64(1.0 / 24.0);
        let lateness = Duration::from_micros(600);
        let mut scheduler = LazyRepaintState::new(start);
        scheduler.after_frame_with_video_cadence(start, Duration::MAX, Some(frame_duration), true);
        let mut expected_deadline = start + frame_duration;

        for _ in 0..240 {
            let wake = expected_deadline + lateness;
            scheduler.after_frame_with_video_cadence(
                wake,
                Duration::MAX,
                Some(frame_duration),
                true,
            );
            expected_deadline += frame_duration;

            assert_eq!(
                scheduler
                    .fixed_cadence
                    .expect("active cadence")
                    .next_video_deadline,
                expected_deadline
            );
        }
    }

    #[test]
    fn active_vsync_preserves_2997_fractional_deadline_cadence() {
        let start = Instant::now();
        let frame_duration = Duration::from_secs_f64(1001.0 / 30000.0);
        let lateness = Duration::from_micros(600);
        let mut scheduler = LazyRepaintState::new(start);
        scheduler.after_frame_with_video_cadence(start, Duration::MAX, Some(frame_duration), true);
        let mut expected_deadline = start + frame_duration;

        for _ in 0..300 {
            let wake = expected_deadline + lateness;
            scheduler.after_frame_with_video_cadence(
                wake,
                Duration::MAX,
                Some(frame_duration),
                true,
            );
            expected_deadline += frame_duration;
        }

        assert_eq!(
            scheduler
                .fixed_cadence
                .expect("active cadence")
                .next_video_deadline,
            expected_deadline
        );
    }

    #[test]
    fn active_vsync_preserves_23976_fractional_deadline_cadence() {
        let start = Instant::now();
        let frame_duration = Duration::from_secs_f64(1001.0 / 24000.0);
        let lateness = Duration::from_micros(600);
        let mut scheduler = LazyRepaintState::new(start);
        scheduler.after_frame_with_video_cadence(start, Duration::MAX, Some(frame_duration), true);
        let mut expected_deadline = start + frame_duration;

        for _ in 0..240 {
            let wake = expected_deadline + lateness;
            scheduler.after_frame_with_video_cadence(
                wake,
                Duration::MAX,
                Some(frame_duration),
                true,
            );
            expected_deadline += frame_duration;
        }

        assert_eq!(
            scheduler
                .fixed_cadence
                .expect("active cadence")
                .next_video_deadline,
            expected_deadline
        );
    }

    #[test]
    fn active_vsync_small_lateness_shortens_next_wait_instead_of_rebasing() {
        let start = Instant::now();
        let frame_duration = Duration::from_millis(33);
        let lateness = Duration::from_micros(600);
        let mut scheduler = LazyRepaintState::new(start);
        scheduler.after_frame_with_video_cadence(start, Duration::MAX, Some(frame_duration), true);
        let first_deadline = start + frame_duration;
        let wake = first_deadline + lateness;

        scheduler.after_frame_with_video_cadence(wake, Duration::MAX, Some(frame_duration), true);

        assert_eq!(
            scheduler.wait_mode(wake),
            WaitMode::WaitTimeout(frame_duration - lateness)
        );
        assert_eq!(
            scheduler
                .fixed_cadence
                .expect("active cadence")
                .next_video_deadline,
            first_deadline + frame_duration
        );
    }

    #[test]
    fn active_vsync_large_lateness_skips_deadlines_to_avoid_burst() {
        let start = Instant::now();
        let frame_duration = Duration::from_millis(33);
        let mut scheduler = LazyRepaintState::new(start);
        scheduler.after_frame_with_video_cadence(start, Duration::MAX, Some(frame_duration), true);
        let first_deadline = start + frame_duration;
        let wake = first_deadline + frame_duration * 3 + Duration::from_millis(1);

        scheduler.after_frame_with_video_cadence(wake, Duration::MAX, Some(frame_duration), true);

        let next_deadline = scheduler
            .fixed_cadence
            .expect("active cadence")
            .next_video_deadline;
        assert!(next_deadline > wake);
        assert!(next_deadline <= wake + frame_duration);
    }

    #[test]
    fn active_vsync_early_ui_repaint_does_not_move_deadline() {
        let start = Instant::now();
        let frame_duration = Duration::from_millis(33);
        let mut scheduler = LazyRepaintState::new(start);
        scheduler.after_frame_with_video_cadence(start, Duration::MAX, Some(frame_duration), true);
        let expected_deadline = start + frame_duration;
        let ui_repaint = start + Duration::from_millis(10);
        scheduler.mark_sdl_event();

        scheduler.after_frame_with_video_cadence(
            ui_repaint,
            Duration::ZERO,
            Some(frame_duration),
            false,
        );

        let cadence = scheduler.fixed_cadence.expect("active cadence");
        assert_eq!(cadence.next_video_deadline, expected_deadline);
        assert_eq!(
            scheduler.wait_mode(ui_repaint),
            WaitMode::WaitTimeout(expected_deadline.duration_since(ui_repaint))
        );
    }

    #[test]
    fn inactive_or_no_vsync_frame_clears_active_deadline_state() {
        let start = Instant::now();
        let mut scheduler = LazyRepaintState::new(start);
        scheduler.after_frame_with_video_cadence(
            start,
            Duration::MAX,
            Some(Duration::from_millis(33)),
            true,
        );

        scheduler.after_frame_with_video_cadence(start, Duration::ZERO, None, true);

        assert!(scheduler.fixed_cadence.is_none());
        assert!(scheduler.frame_decision(start).video_advance_allowed);
    }
}
