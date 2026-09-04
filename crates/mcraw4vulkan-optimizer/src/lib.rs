#![recursion_limit = "256"]

//! Production optimizer shell-out runner and optimized settings state.
//!
//! The crate owns the narrow user-facing optimizer path: hidden measurement
//! reports are parsed by `shellout`, and accepted recommendations are persisted
//! through the optional per-user JSON state in `optimized_state`.

mod decision;
pub mod optimized_state;
pub mod shellout;

pub use decision::{
    LOW_RAM_THRESHOLD_BYTES, LowRamGate, MIN_COMBINED_THROUGHPUT_RATIO,
    MIN_PER_SINK_THROUGHPUT_RATIO, PayloadProfileDecision, RecommendationReason, ScoredSink,
    ThroughputMedians, recommend_payload_profile,
};

pub use optimized_state::{
    EffectiveOptimizerSettings, EffectiveSettingsSource, OPTIMIZED_STATE_FILE_NAME,
    OPTIMIZED_STATE_SCHEMA_VERSION, OptimizedState, OptimizedStateError, OptimizedStateLoadOutcome,
    OptimizedStatePathEnv, PayloadProfile, RestoreOptimizedStateOutcome, SettingsSourceSelection,
    StatePathPlatform, built_in_default_settings, format_optimized_state_json,
    load_optimized_state, load_optimized_state_from_path, optimized_state_path,
    optimized_state_path_from_env, parse_optimized_state_json, resolve_effective_settings,
    resolve_effective_settings_from_load, restore_optimized_state, restore_optimized_state_at_path,
    save_optimized_state, save_optimized_state_to_path,
};
pub use shellout::{
    MeasurementReport, OptimizerRecommendation as ShelloutOptimizerRecommendation,
    OptimizerRunConfig, OptimizerShelloutRunner, RecommendedPayloadProfile, run_optimizer,
};
