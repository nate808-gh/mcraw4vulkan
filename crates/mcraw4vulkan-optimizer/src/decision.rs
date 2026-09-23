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
