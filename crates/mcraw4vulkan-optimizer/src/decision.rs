use crate::optimized_state::PayloadProfile;

pub const MIN_COMBINED_THROUGHPUT_RATIO: f64 = 1.10;
pub const MIN_PER_SINK_THROUGHPUT_RATIO: f64 = 0.95;
pub const LOW_RAM_THRESHOLD_BYTES: u64 = 17 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThroughputMedians {
    pub default_display_fps: f64,
    pub default_pipe_fps: f64,
    pub offset_display_fps: f64,
    pub offset_pipe_fps: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoredSink {
    Display,
    Pipe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowRamGate {
    Pass,
    Fail,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationReason {
    OffsetThroughputGain,
    OffsetLowRamNoRegression,
    DefaultSinkRegression(ScoredSink),
    DefaultRamNotLow,
    DefaultRamUnavailable,
    DefaultInvalidMeasurements,
    DefaultMeasurementFailure,
}

impl RecommendationReason {
    pub fn token(self) -> &'static str {
        // These tokens are the machine-readable policy result; user-facing
        // presentation may change without changing their meanings.
        match self {
            Self::OffsetThroughputGain => "offset_throughput_gain",
            Self::OffsetLowRamNoRegression => "offset_low_ram_no_regression",
            Self::DefaultSinkRegression(ScoredSink::Display) => "default_display_sink_regression",
            Self::DefaultSinkRegression(ScoredSink::Pipe) => "default_pipe_sink_regression",
            Self::DefaultRamNotLow => "default_ram_not_low",
            Self::DefaultRamUnavailable => "default_ram_unavailable",
            Self::DefaultInvalidMeasurements => "default_invalid_measurements",
            Self::DefaultMeasurementFailure => "default_measurement_failure",
        }
    }

    pub fn measurements_are_valid(self) -> bool {
        !matches!(
            self,
            Self::DefaultInvalidMeasurements | Self::DefaultMeasurementFailure
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PayloadProfileDecision {
    pub selected: PayloadProfile,
    pub reason: RecommendationReason,
    pub display_ratio: Option<f64>,
    pub pipe_ratio: Option<f64>,
    pub combined_ratio: Option<f64>,
    pub no_loss_over_five_percent: Option<bool>,
    pub throughput_gain_over_ten_percent: Option<bool>,
    pub low_ram_gate: LowRamGate,
}

pub fn recommend_payload_profile(
    measurements: Option<ThroughputMedians>,
    total_ram_bytes: Option<u64>,
) -> PayloadProfileDecision {
    // The low-RAM gate is strict: exactly 17 GiB belongs to the non-low-RAM
    // branch, while an unavailable fact remains distinct from either result.
    let low_ram_gate = match total_ram_bytes {
        Some(bytes) if bytes < LOW_RAM_THRESHOLD_BYTES => LowRamGate::Pass,
        Some(_) => LowRamGate::Fail,
        None => LowRamGate::Unavailable,
    };
    let Some(measurements) = measurements else {
        return default_decision(
            RecommendationReason::DefaultMeasurementFailure,
            low_ram_gate,
        );
    };
    if ![
        measurements.default_display_fps,
        measurements.default_pipe_fps,
        measurements.offset_display_fps,
        measurements.offset_pipe_fps,
    ]
    .into_iter()
    .all(|value| value.is_finite() && value > 0.0)
    {
        return default_decision(
            RecommendationReason::DefaultInvalidMeasurements,
            low_ram_gate,
        );
    }

    let display_ratio = measurements.offset_display_fps / measurements.default_display_fps;
    let pipe_ratio = measurements.offset_pipe_fps / measurements.default_pipe_fps;
    // The geometric mean gives Display and PIPE equal multiplicative weight;
    // neither sink's absolute FPS scale can dominate the combined score.
    let combined_ratio = (display_ratio * pipe_ratio).sqrt();
    if !display_ratio.is_finite() || !pipe_ratio.is_finite() || !combined_ratio.is_finite() {
        return default_decision(
            RecommendationReason::DefaultInvalidMeasurements,
            low_ram_gate,
        );
    }

    let no_loss_over_five_percent = display_ratio >= MIN_PER_SINK_THROUGHPUT_RATIO
        && pipe_ratio >= MIN_PER_SINK_THROUGHPUT_RATIO;
    let throughput_gain_over_ten_percent = combined_ratio > MIN_COMBINED_THROUGHPUT_RATIO;
    // Each sink must meet the inclusive 0.95 guard before either selection rule.
    // The strict >1.10 combined-gain rule takes priority over the low-RAM branch.
    let (selected, reason) = if display_ratio < MIN_PER_SINK_THROUGHPUT_RATIO {
        (
            PayloadProfile::DefaultChunked64,
            RecommendationReason::DefaultSinkRegression(ScoredSink::Display),
        )
    } else if pipe_ratio < MIN_PER_SINK_THROUGHPUT_RATIO {
        (
            PayloadProfile::DefaultChunked64,
            RecommendationReason::DefaultSinkRegression(ScoredSink::Pipe),
        )
    } else if throughput_gain_over_ten_percent {
        (
            PayloadProfile::OffsetPrefetch,
            RecommendationReason::OffsetThroughputGain,
        )
    } else {
        match low_ram_gate {
            LowRamGate::Pass => (
                PayloadProfile::OffsetPrefetch,
                RecommendationReason::OffsetLowRamNoRegression,
            ),
            LowRamGate::Fail => (
                PayloadProfile::DefaultChunked64,
                RecommendationReason::DefaultRamNotLow,
            ),
            LowRamGate::Unavailable => (
                PayloadProfile::DefaultChunked64,
                RecommendationReason::DefaultRamUnavailable,
            ),
        }
    };

    PayloadProfileDecision {
        selected,
        reason,
        display_ratio: Some(display_ratio),
        pipe_ratio: Some(pipe_ratio),
        combined_ratio: Some(combined_ratio),
        no_loss_over_five_percent: Some(no_loss_over_five_percent),
        throughput_gain_over_ten_percent: Some(throughput_gain_over_ten_percent),
        low_ram_gate,
    }
}

fn default_decision(
    reason: RecommendationReason,
    low_ram_gate: LowRamGate,
) -> PayloadProfileDecision {
    PayloadProfileDecision {
        selected: PayloadProfile::DefaultChunked64,
        reason,
        display_ratio: None,
        pipe_ratio: None,
        combined_ratio: None,
        no_loss_over_five_percent: None,
        throughput_gain_over_ten_percent: None,
        low_ram_gate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;
    const HIGH_RAM: u64 = 64 * GIB;
    const NOMINAL_16_GIB: u64 = 17_179_869_184;

    fn metrics(display_ratio: f64, pipe_ratio: f64) -> ThroughputMedians {
        ThroughputMedians {
            default_display_fps: 100.0,
            default_pipe_fps: 100.0,
            offset_display_fps: 100.0 * display_ratio,
            offset_pipe_fps: 100.0 * pipe_ratio,
        }
    }

    fn decide(display_ratio: f64, pipe_ratio: f64, ram: Option<u64>) -> PayloadProfileDecision {
        recommend_payload_profile(Some(metrics(display_ratio, pipe_ratio)), ram)
    }

    #[test]
    fn throughput_win_on_high_ram_uses_rule_one() {
        let decision = decide(1.21, 1.01, Some(HIGH_RAM));
        assert_eq!(decision.selected, PayloadProfile::OffsetPrefetch);
        assert_eq!(decision.reason, RecommendationReason::OffsetThroughputGain);
    }

    #[test]
    fn exact_combined_ratio_does_not_pass_strict_throughput_gate_on_high_ram() {
        let decision = decide(1.21, 1.0, Some(HIGH_RAM));
        assert_eq!(decision.combined_ratio, Some(1.10));
        assert_eq!(decision.throughput_gain_over_ten_percent, Some(false));
        assert_eq!(decision.selected, PayloadProfile::DefaultChunked64);
        assert_eq!(decision.reason, RecommendationReason::DefaultRamNotLow);
    }

    #[test]
    fn exact_combined_ratio_uses_low_ram_rule_on_16_gib() {
        let decision = decide(1.21, 1.0, Some(NOMINAL_16_GIB));
        assert_eq!(decision.combined_ratio, Some(1.10));
        assert_eq!(
            decision.reason,
            RecommendationReason::OffsetLowRamNoRegression
        );
    }

    #[test]
    fn low_ram_near_tie_uses_rule_two() {
        assert_eq!(
            decide(1.001, 0.999, Some(NOMINAL_16_GIB)).reason,
            RecommendationReason::OffsetLowRamNoRegression
        );
    }

    #[test]
    fn low_ram_accepts_slowdowns_up_to_five_percent() {
        let decision = decide(0.96, 0.951, Some(NOMINAL_16_GIB));
        assert_eq!(
            decision.reason,
            RecommendationReason::OffsetLowRamNoRegression
        );
    }

    #[test]
    fn exact_five_percent_loss_passes_no_loss_guard() {
        let decision = decide(0.95, 1.01, Some(NOMINAL_16_GIB));
        assert_eq!(decision.no_loss_over_five_percent, Some(true));
        assert_eq!(decision.selected, PayloadProfile::OffsetPrefetch);
    }

    #[test]
    fn immediately_greater_than_five_percent_loss_rejects_even_with_combined_win() {
        let decision = decide(f64::from_bits(0.95f64.to_bits() - 1), 1.50, Some(8 * GIB));
        assert!(decision.combined_ratio.is_some_and(|ratio| ratio > 1.10));
        assert_eq!(
            decision.reason,
            RecommendationReason::DefaultSinkRegression(ScoredSink::Display)
        );
    }

    #[test]
    fn high_ram_near_tie_keeps_default() {
        assert_eq!(
            decide(1.01, 1.0, Some(HIGH_RAM)).reason,
            RecommendationReason::DefaultRamNotLow
        );
    }

    #[test]
    fn exactly_17_gib_is_not_low_ram() {
        let decision = decide(1.0, 1.0, Some(LOW_RAM_THRESHOLD_BYTES));
        assert_eq!(decision.low_ram_gate, LowRamGate::Fail);
        assert_eq!(decision.selected, PayloadProfile::DefaultChunked64);
    }

    #[test]
    fn one_byte_below_17_gib_is_low_ram() {
        let decision = decide(1.0, 1.0, Some(LOW_RAM_THRESHOLD_BYTES - 1));
        assert_eq!(decision.low_ram_gate, LowRamGate::Pass);
        assert_eq!(decision.selected, PayloadProfile::OffsetPrefetch);
    }

    #[test]
    fn nominal_16_gib_host_qualifies_as_low_ram() {
        assert_eq!(NOMINAL_16_GIB, 16 * GIB);
        assert_eq!(
            decide(1.0, 1.0, Some(NOMINAL_16_GIB)).reason,
            RecommendationReason::OffsetLowRamNoRegression
        );
    }

    #[test]
    fn unavailable_ram_does_not_block_throughput_win() {
        assert_eq!(
            decide(1.20, 1.02, None).reason,
            RecommendationReason::OffsetThroughputGain
        );
    }

    #[test]
    fn unavailable_ram_without_throughput_win_keeps_default() {
        let decision = decide(1.0, 1.0, None);
        assert_eq!(decision.low_ram_gate, LowRamGate::Unavailable);
        assert_eq!(decision.reason, RecommendationReason::DefaultRamUnavailable);
    }

    #[test]
    fn invalid_default_metrics_never_recommend_offset() {
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut value = metrics(1.2, 1.2);
            value.default_display_fps = invalid;
            assert_eq!(
                recommend_payload_profile(Some(value), Some(8 * GIB)).reason,
                RecommendationReason::DefaultInvalidMeasurements
            );
        }
    }

    #[test]
    fn invalid_offset_metrics_never_recommend_offset() {
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut value = metrics(1.2, 1.2);
            value.offset_pipe_fps = invalid;
            assert_eq!(
                recommend_payload_profile(Some(value), Some(8 * GIB)).reason,
                RecommendationReason::DefaultInvalidMeasurements
            );
        }
    }

    #[test]
    fn missing_required_result_is_measurement_failure() {
        let decision = recommend_payload_profile(None, Some(8 * GIB));
        assert_eq!(decision.selected, PayloadProfile::DefaultChunked64);
        assert_eq!(
            decision.reason,
            RecommendationReason::DefaultMeasurementFailure
        );
    }

    #[test]
    fn throughput_reason_has_priority_when_both_rules_pass() {
        assert_eq!(
            decide(1.20, 1.02, Some(8 * GIB)).reason,
            RecommendationReason::OffsetThroughputGain
        );
    }

    #[test]
    fn one_large_gain_cannot_conceal_pipe_regression() {
        let decision = decide(1.80, 0.94, Some(8 * GIB));
        assert!(decision.combined_ratio.is_some_and(|ratio| ratio > 1.10));
        assert_eq!(
            decision.reason,
            RecommendationReason::DefaultSinkRegression(ScoredSink::Pipe)
        );
    }

    #[test]
    fn thresholds_have_one_exact_source_of_truth() {
        assert_eq!(MIN_COMBINED_THROUGHPUT_RATIO, 1.10);
        assert_eq!(MIN_PER_SINK_THROUGHPUT_RATIO, 0.95);
        assert_eq!(LOW_RAM_THRESHOLD_BYTES, 18_253_611_008);
    }

    #[test]
    fn large_memory_host_is_not_low_ram() {
        assert_eq!(
            decide(1.0, 1.0, Some(1024 * GIB)).low_ram_gate,
            LowRamGate::Fail
        );
    }
}
