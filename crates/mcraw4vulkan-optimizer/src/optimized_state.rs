use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use mcraw4vulkan_mcrawcontainer::payload_reader::{PayloadFeederOptions, PayloadReadMode};
use serde_json::Value;

pub const OPTIMIZED_STATE_SCHEMA_VERSION: u32 = 2;
pub const OPTIMIZED_STATE_FILE_NAME: &str = "optimized-state.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadProfile {
    DefaultChunked64,
    OffsetPrefetch,
}

impl PayloadProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::DefaultChunked64 => "default_chunked64",
            Self::OffsetPrefetch => "offset_prefetch",
        }
    }

    pub fn parse(value: &str) -> Result<Self, OptimizedStateError> {
        match value {
            "default_chunked64" => Ok(Self::DefaultChunked64),
            "offset_prefetch" => Ok(Self::OffsetPrefetch),
            _ => Err(OptimizedStateError::Invalid(format!(
                "unsupported payload_profile {value:?}"
            ))),
        }
    }

    pub fn payload_feeder_options(self) -> PayloadFeederOptions {
        let mut options = PayloadFeederOptions::production_default();
        if self == Self::OffsetPrefetch {
            options.mode = PayloadReadMode::OffsetPrefetch;
        }
        options
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizedState {
    // Apart from the schema version, persisted state contains only the selected
    // payload override. Measurements, input identity, and host facts remain run-local.
    pub version: u32,
    pub payload_profile: PayloadProfile,
}

impl OptimizedState {
    pub fn new(payload_profile: PayloadProfile) -> Self {
        Self {
            version: OPTIMIZED_STATE_SCHEMA_VERSION,
            payload_profile,
        }
    }
}

#[derive(Debug)]
pub enum OptimizedStateError {
    Io(io::Error),
    Invalid(String),
    PathUnavailable(String),
}

impl fmt::Display for OptimizedStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Invalid(message) => write!(formatter, "{message}"),
            Self::PathUnavailable(message) => write!(formatter, "{message}"),
        }
    }
}

impl Error for OptimizedStateError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Invalid(_) | Self::PathUnavailable(_) => None,
        }
    }
}

impl From<io::Error> for OptimizedStateError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatePathPlatform {
    Linux,
    Macos,
    Windows,
}

impl StatePathPlatform {
    pub fn current() -> Self {
        if cfg!(target_os = "windows") {
            Self::Windows
        } else if cfg!(target_os = "macos") {
            Self::Macos
        } else {
            Self::Linux
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OptimizedStatePathEnv {
    pub xdg_config_home: Option<PathBuf>,
    pub home: Option<PathBuf>,
    pub appdata: Option<PathBuf>,
    pub localappdata: Option<PathBuf>,
    pub userprofile: Option<PathBuf>,
}

impl OptimizedStatePathEnv {
    pub fn from_process_env() -> Self {
        Self {
            xdg_config_home: nonempty_env_path("XDG_CONFIG_HOME"),
            home: nonempty_env_path("HOME"),
            appdata: nonempty_env_path("APPDATA"),
            localappdata: nonempty_env_path("LOCALAPPDATA"),
            userprofile: nonempty_env_path("USERPROFILE"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizedStateLoadOutcome {
    Missing {
        path: PathBuf,
    },
    Loaded {
        path: PathBuf,
        state: OptimizedState,
    },
    Invalid {
        path: PathBuf,
        error: String,
    },
    Unreadable {
        path: PathBuf,
        error: String,
    },
    PathUnavailable {
        error: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectiveSettingsSource {
    BuiltInDefault,
    OptimizedState { path: PathBuf },
    OptimizedStateUnavailableFallback { warning: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveOptimizerSettings {
    pub payload_profile: PayloadProfile,
    pub source: EffectiveSettingsSource,
}

impl EffectiveOptimizerSettings {
    pub fn warning(&self) -> Option<&str> {
        match &self.source {
            EffectiveSettingsSource::OptimizedStateUnavailableFallback { warning } => Some(warning),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsSourceSelection {
    Default,
    Optimized,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestoreOptimizedStateOutcome {
    Deleted { path: PathBuf },
    Missing { path: PathBuf },
    PathUnavailable { error: String },
}

pub fn built_in_default_settings() -> EffectiveOptimizerSettings {
    EffectiveOptimizerSettings {
        payload_profile: PayloadProfile::DefaultChunked64,
        source: EffectiveSettingsSource::BuiltInDefault,
    }
}

pub fn resolve_effective_settings(
    selection: SettingsSourceSelection,
) -> EffectiveOptimizerSettings {
    match selection {
        SettingsSourceSelection::Default => built_in_default_settings(),
        SettingsSourceSelection::Optimized => {
            resolve_effective_settings_from_load(load_optimized_state())
        }
    }
}

pub fn resolve_effective_settings_from_load(
    load: OptimizedStateLoadOutcome,
) -> EffectiveOptimizerSettings {
    // Missing, unreadable, or invalid state selects built-in settings without
    // rewriting or removing the state path.
    match load {
        OptimizedStateLoadOutcome::Loaded { path, state } => EffectiveOptimizerSettings {
            payload_profile: state.payload_profile,
            source: EffectiveSettingsSource::OptimizedState { path },
        },
        OptimizedStateLoadOutcome::Missing { path } => fallback_settings(format!(
            "optimized settings file not found at {}; no current optimized payload profile exists; using built-in defaults; run optimizer to create one",
            path.display()
        )),
        OptimizedStateLoadOutcome::Invalid { path, error } => fallback_settings(format!(
            "optimized settings file at {} is invalid or obsolete: {}; using built-in defaults; run optimizer to create current settings",
            path.display(),
            error
        )),
        OptimizedStateLoadOutcome::Unreadable { path, error } => fallback_settings(format!(
            "optimized settings file at {} is unreadable: {}; using built-in defaults",
            path.display(),
            error
        )),
        OptimizedStateLoadOutcome::PathUnavailable { error } => fallback_settings(format!(
            "optimized settings file path is unavailable: {}; using built-in defaults",
            error
        )),
    }
}

pub fn optimized_state_path() -> Result<PathBuf, OptimizedStateError> {
    optimized_state_path_from_env(
        &OptimizedStatePathEnv::from_process_env(),
        StatePathPlatform::current(),
    )
}

pub fn optimized_state_path_from_env(
    env: &OptimizedStatePathEnv,
    platform: StatePathPlatform,
) -> Result<PathBuf, OptimizedStateError> {
    let config_dir = match platform {
        StatePathPlatform::Linux => env
            .xdg_config_home
            .clone()
            .or_else(|| env.home.as_ref().map(|home| home.join(".config")))
            .ok_or_else(|| {
                OptimizedStateError::PathUnavailable(
                    "neither XDG_CONFIG_HOME nor HOME is available".to_string(),
                )
            })?,
        StatePathPlatform::Macos => env
            .home
            .as_ref()
            .map(|home| home.join("Library").join("Application Support"))
            .ok_or_else(|| {
                OptimizedStateError::PathUnavailable("HOME is not available".to_string())
            })?,
        StatePathPlatform::Windows => env
            .appdata
            .clone()
            .or_else(|| env.localappdata.clone())
            .or_else(|| {
                env.userprofile
                    .as_ref()
                    .map(|value| value.join("AppData").join("Roaming"))
            })
            .ok_or_else(|| {
                OptimizedStateError::PathUnavailable(
                    "APPDATA, LOCALAPPDATA, and USERPROFILE are unavailable".to_string(),
                )
            })?,
    };

    Ok(config_dir
        .join("mcraw4vulkan")
        .join(OPTIMIZED_STATE_FILE_NAME))
}

pub fn load_optimized_state() -> OptimizedStateLoadOutcome {
    match optimized_state_path() {
        Ok(path) => load_optimized_state_from_path(&path),
        Err(OptimizedStateError::PathUnavailable(error)) => {
            OptimizedStateLoadOutcome::PathUnavailable { error }
        }
        Err(error) => OptimizedStateLoadOutcome::PathUnavailable {
            error: error.to_string(),
        },
    }
}

pub fn load_optimized_state_from_path(path: &Path) -> OptimizedStateLoadOutcome {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return OptimizedStateLoadOutcome::Missing {
                path: path.to_path_buf(),
            };
        }
        Err(error) => {
            return OptimizedStateLoadOutcome::Unreadable {
                path: path.to_path_buf(),
                error: error.to_string(),
            };
        }
    };

    match parse_optimized_state_json(&text) {
        Ok(state) => OptimizedStateLoadOutcome::Loaded {
            path: path.to_path_buf(),
            state,
        },
        Err(error) => OptimizedStateLoadOutcome::Invalid {
            path: path.to_path_buf(),
            error: error.to_string(),
        },
    }
}

pub fn save_optimized_state(state: &OptimizedState) -> Result<PathBuf, OptimizedStateError> {
    let path = optimized_state_path()?;
    save_optimized_state_to_path(&path, state)?;
    Ok(path)
}

pub fn save_optimized_state_to_path(
    path: &Path,
    state: &OptimizedState,
) -> Result<(), OptimizedStateError> {
    validate_optimized_state(state)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let temp_path = temp_path_for_state_path(path);
    // Writing and syncing a sibling staging file prevents readers from seeing
    // partial JSON; the final replacement step follows platform rename rules.
    let write_result = (|| -> Result<(), OptimizedStateError> {
        let mut file = fs::File::create(&temp_path)?;
        file.write_all(format_optimized_state_json(state).as_bytes())?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }

    if cfg!(windows) && path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(&temp_path, path)?;
    Ok(())
}

pub fn restore_optimized_state() -> Result<RestoreOptimizedStateOutcome, OptimizedStateError> {
    match optimized_state_path() {
        Ok(path) => restore_optimized_state_at_path(&path),
        Err(OptimizedStateError::PathUnavailable(error)) => {
            Ok(RestoreOptimizedStateOutcome::PathUnavailable { error })
        }
        Err(error) => Err(error),
    }
}

pub fn restore_optimized_state_at_path(
    path: &Path,
) -> Result<RestoreOptimizedStateOutcome, OptimizedStateError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(RestoreOptimizedStateOutcome::Deleted {
            path: path.to_path_buf(),
        }),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            Ok(RestoreOptimizedStateOutcome::Missing {
                path: path.to_path_buf(),
            })
        }
        Err(error) => Err(OptimizedStateError::Io(error)),
    }
}

pub fn format_optimized_state_json(state: &OptimizedState) -> String {
    format!(
        "{{\n  \"version\": {},\n  \"payload_profile\": \"{}\"\n}}\n",
        state.version,
        state.payload_profile.label()
    )
}

pub fn parse_optimized_state_json(text: &str) -> Result<OptimizedState, OptimizedStateError> {
    let value: Value = serde_json::from_str(text)
        .map_err(|error| OptimizedStateError::Invalid(error.to_string()))?;
    let object = value
        .as_object()
        .ok_or_else(|| OptimizedStateError::Invalid("state root must be an object".to_string()))?;

    reject_unknown_keys(object, &["version", "payload_profile"])?;
    let version = required_u64(object, "version")?;
    if version != u64::from(OPTIMIZED_STATE_SCHEMA_VERSION) {
        return Err(OptimizedStateError::Invalid(format!(
            "unsupported version {version}"
        )));
    }
    let version = u32::try_from(version)
        .map_err(|_| OptimizedStateError::Invalid("version is too large".to_string()))?;
    let payload_profile = PayloadProfile::parse(&required_string(object, "payload_profile")?)?;

    let state = OptimizedState {
        version,
        payload_profile,
    };
    validate_optimized_state(&state)?;
    Ok(state)
}

pub fn temp_path_for_state_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(OPTIMIZED_STATE_FILE_NAME);
    path.with_file_name(format!("{file_name}.tmp.{}", std::process::id()))
}

fn fallback_settings(warning: String) -> EffectiveOptimizerSettings {
    EffectiveOptimizerSettings {
        payload_profile: PayloadProfile::DefaultChunked64,
        source: EffectiveSettingsSource::OptimizedStateUnavailableFallback { warning },
    }
}

fn validate_optimized_state(state: &OptimizedState) -> Result<(), OptimizedStateError> {
    if state.version != OPTIMIZED_STATE_SCHEMA_VERSION {
        return Err(OptimizedStateError::Invalid(format!(
            "unsupported version {}",
            state.version
        )));
    }
    // Absence of an override represents the built-in default, so persisted
    // state can encode only the selected non-default payload profile.
    if state.payload_profile != PayloadProfile::OffsetPrefetch {
        return Err(OptimizedStateError::Invalid(format!(
            "optimized settings may save only payload_profile {:?}; {:?} is the built-in default",
            PayloadProfile::OffsetPrefetch.label(),
            PayloadProfile::DefaultChunked64.label()
        )));
    }
    Ok(())
}

fn nonempty_env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn reject_unknown_keys(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
) -> Result<(), OptimizedStateError> {
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    let unknown = object
        .keys()
        .find(|key| !allowed.contains(key.as_str()))
        .cloned();
    if let Some(key) = unknown {
        return Err(OptimizedStateError::Invalid(format!(
            "unknown field {key:?} in optimizer settings"
        )));
    }
    Ok(())
}

fn required_value<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a Value, OptimizedStateError> {
    object
        .get(key)
        .ok_or_else(|| OptimizedStateError::Invalid(format!("missing required field {key}")))
}

fn required_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<String, OptimizedStateError> {
    required_value(object, key)?
        .as_str()
        .map(ToString::to_string)
        .ok_or_else(|| OptimizedStateError::Invalid(format!("{key} must be a string")))
}

fn required_u64(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<u64, OptimizedStateError> {
    required_value(object, key)?
        .as_u64()
        .ok_or_else(|| OptimizedStateError::Invalid(format!("{key} must be an unsigned integer")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mcraw4vulkan-optimized-state-test-{}-{name}",
            std::process::id()
        ))
    }

    fn state() -> OptimizedState {
        OptimizedState::new(PayloadProfile::OffsetPrefetch)
    }

    fn unsupported_schema_state_json() -> &'static str {
        "{\n  \"version\": 1,\n  \"payload_profile\": \"offset_prefetch\"\n}\n"
    }

    #[test]
    fn linux_xdg_path_resolver_uses_xdg_config_home() {
        let env = OptimizedStatePathEnv {
            xdg_config_home: Some(PathBuf::from("xdg")),
            home: Some(PathBuf::from("home/user")),
            ..OptimizedStatePathEnv::default()
        };
        assert_eq!(
            optimized_state_path_from_env(&env, StatePathPlatform::Linux).unwrap(),
            PathBuf::from("xdg/mcraw4vulkan/optimized-state.json")
        );
    }

    #[test]
    fn linux_home_fallback_path_resolver_uses_home_config() {
        let env = OptimizedStatePathEnv {
            home: Some(PathBuf::from("home/user")),
            ..OptimizedStatePathEnv::default()
        };
        assert_eq!(
            optimized_state_path_from_env(&env, StatePathPlatform::Linux).unwrap(),
            PathBuf::from("home/user/.config/mcraw4vulkan/optimized-state.json")
        );
    }

    #[test]
    fn macos_home_path_resolver_uses_application_support() {
        let env = OptimizedStatePathEnv {
            home: Some(PathBuf::from("Users/example")),
            ..OptimizedStatePathEnv::default()
        };
        assert_eq!(
            optimized_state_path_from_env(&env, StatePathPlatform::Macos).unwrap(),
            PathBuf::from(
                "Users/example/Library/Application Support/mcraw4vulkan/optimized-state.json"
            )
        );
    }

    #[test]
    fn windows_path_resolver_uses_appdata_then_fallbacks() {
        let env = OptimizedStatePathEnv {
            appdata: Some(PathBuf::from(windows_path(&[
                "Users", "example", "AppData", "Roaming",
            ]))),
            localappdata: Some(PathBuf::from(windows_path(&[
                "Users", "example", "AppData", "Local",
            ]))),
            userprofile: Some(PathBuf::from(windows_path(&["Users", "example"]))),
            ..OptimizedStatePathEnv::default()
        };
        assert_eq!(
            optimized_state_path_from_env(&env, StatePathPlatform::Windows).unwrap(),
            PathBuf::from(windows_path(&["Users", "example", "AppData", "Roaming"]))
                .join("mcraw4vulkan")
                .join(OPTIMIZED_STATE_FILE_NAME)
        );

        let env = OptimizedStatePathEnv {
            localappdata: Some(PathBuf::from(windows_path(&[
                "Users", "example", "AppData", "Local",
            ]))),
            userprofile: Some(PathBuf::from(windows_path(&["Users", "example"]))),
            ..OptimizedStatePathEnv::default()
        };
        assert_eq!(
            optimized_state_path_from_env(&env, StatePathPlatform::Windows).unwrap(),
            PathBuf::from(windows_path(&["Users", "example", "AppData", "Local"]))
                .join("mcraw4vulkan")
                .join(OPTIMIZED_STATE_FILE_NAME)
        );

        let env = OptimizedStatePathEnv {
            userprofile: Some(PathBuf::from(windows_path(&["Users", "example"]))),
            ..OptimizedStatePathEnv::default()
        };
        assert_eq!(
            optimized_state_path_from_env(&env, StatePathPlatform::Windows).unwrap(),
            PathBuf::from(windows_path(&["Users", "example"]))
                .join("AppData")
                .join("Roaming")
                .join("mcraw4vulkan")
                .join(OPTIMIZED_STATE_FILE_NAME)
        );
    }

    fn windows_path(parts: &[&str]) -> String {
        parts.join(&windows_separator().to_string())
    }

    fn windows_separator() -> char {
        char::from_u32(92).expect("backslash code point is valid")
    }

    #[test]
    fn path_unavailable_when_no_suitable_env_exists() {
        let error = optimized_state_path_from_env(
            &OptimizedStatePathEnv::default(),
            StatePathPlatform::Linux,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("XDG_CONFIG_HOME"));
        assert!(error.contains("HOME"));
    }

    #[test]
    fn schema_v2_offset_prefetch_parses() {
        let parsed =
            parse_optimized_state_json("{\"version\":2,\"payload_profile\":\"offset_prefetch\"}")
                .unwrap();
        assert_eq!(parsed, state());
    }

    #[test]
    fn schema_v2_default_profile_is_not_a_valid_saved_override() {
        let error =
            parse_optimized_state_json("{\"version\":2,\"payload_profile\":\"default_chunked64\"}")
                .expect_err("built-in Default must not load as an optimized override")
                .to_string();

        assert!(error.contains("offset_prefetch"));
        assert!(error.contains("built-in default"));
    }

    #[test]
    fn schema_v2_serialization_is_exact_payload_only_json() {
        assert_eq!(
            format_optimized_state_json(&state()),
            "{\n  \"version\": 2,\n  \"payload_profile\": \"offset_prefetch\"\n}\n"
        );
    }

    #[test]
    fn schema_v2_default_profile_cannot_be_saved() {
        let dir = temp_root("reject-default-save");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(OPTIMIZED_STATE_FILE_NAME);
        let default = OptimizedState::new(PayloadProfile::DefaultChunked64);

        assert!(save_optimized_state_to_path(&path, &default).is_err());
        assert!(!path.exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn schema_v2_unknown_fields_are_rejected() {
        let error = parse_optimized_state_json(
            "{\"version\":2,\"payload_profile\":\"offset_prefetch\",\"extra\":true}",
        )
        .expect_err("unknown field rejected")
        .to_string();

        assert!(error.contains("unknown field"));
    }

    #[test]
    fn unsupported_schema_resolves_to_default_without_rewrite() {
        let dir = temp_root("unsupported-schema");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(OPTIMIZED_STATE_FILE_NAME);
        let text = unsupported_schema_state_json();
        fs::write(&path, text).unwrap();

        let load = load_optimized_state_from_path(&path);
        assert!(matches!(load, OptimizedStateLoadOutcome::Invalid { .. }));
        let effective = resolve_effective_settings_from_load(load);
        assert_eq!(effective.payload_profile, PayloadProfile::DefaultChunked64);
        assert!(effective.warning().unwrap().contains("obsolete"));
        assert_eq!(fs::read_to_string(&path).unwrap(), text);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_state_resolves_to_default_with_rerun_warning() {
        let path = temp_root("missing").join(OPTIMIZED_STATE_FILE_NAME);
        let load = load_optimized_state_from_path(&path);
        assert!(matches!(load, OptimizedStateLoadOutcome::Missing { .. }));

        let effective = resolve_effective_settings_from_load(load);
        assert_eq!(effective.payload_profile, PayloadProfile::DefaultChunked64);
        assert!(effective.warning().unwrap().contains("run optimizer"));
    }

    #[test]
    fn malformed_state_resolves_to_default_without_rewrite() {
        let dir = temp_root("corrupt");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(OPTIMIZED_STATE_FILE_NAME);
        let text = "{ bad json\n";
        fs::write(&path, text).unwrap();
        let load = load_optimized_state_from_path(&path);
        assert!(matches!(load, OptimizedStateLoadOutcome::Invalid { .. }));

        let effective = resolve_effective_settings_from_load(load);
        assert_eq!(effective.payload_profile, PayloadProfile::DefaultChunked64);
        assert!(effective.warning().unwrap().contains("invalid"));
        assert_eq!(fs::read_to_string(&path).unwrap(), text);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn unsupported_version_and_unknown_payload_are_invalid() {
        assert!(
            parse_optimized_state_json("{\"version\":3,\"payload_profile\":\"offset_prefetch\"}")
                .is_err()
        );
        assert!(parse_optimized_state_json("{\"version\":2,\"payload_profile\":\"bad\"}").is_err());
    }

    #[test]
    fn state_json_contains_no_private_or_benchmark_data() {
        let text = format_optimized_state_json(&state());
        assert!(text.contains("\"version\": 2"));
        for forbidden in [
            "input_path",
            "input_basename",
            "benchmark",
            "history",
            "profile_list",
            "machine_label",
            "run_label",
            "matrix",
            "host",
            "user",
            "schema_version",
        ] {
            assert!(!text.contains(forbidden), "state JSON contains {forbidden}");
        }
    }

    #[test]
    fn save_creates_parent_writes_valid_json_and_overwrites() {
        let dir = temp_root("save");
        let path = dir.join("nested").join(OPTIMIZED_STATE_FILE_NAME);
        save_optimized_state_to_path(&path, &state()).unwrap();
        assert!(path.exists());
        assert_eq!(
            load_optimized_state_from_path(&path),
            OptimizedStateLoadOutcome::Loaded {
                path: path.clone(),
                state: state()
            }
        );

        fs::write(&path, b"stale").unwrap();
        save_optimized_state_to_path(&path, &state()).unwrap();
        assert_eq!(
            load_optimized_state_from_path(&path),
            OptimizedStateLoadOutcome::Loaded {
                path: path.clone(),
                state: state()
            }
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn save_uses_temp_file_and_failure_does_not_create_state() {
        let dir = temp_root("save-failure");
        fs::create_dir_all(&dir).unwrap();
        let parent_file = dir.join("not-a-dir");
        fs::write(&parent_file, b"block").unwrap();
        let path = parent_file.join(OPTIMIZED_STATE_FILE_NAME);
        assert!(save_optimized_state_to_path(&path, &state()).is_err());
        assert!(!path.exists());
        assert!(
            temp_path_for_state_path(Path::new("optimized-state.json"))
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains(".tmp.")
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_deletes_only_the_state_file_and_missing_succeeds() {
        let dir = temp_root("restore");
        fs::create_dir_all(&dir).unwrap();
        let unrelated = dir.join("unrelated.txt");
        fs::write(&unrelated, b"keep").unwrap();
        let path = dir.join(OPTIMIZED_STATE_FILE_NAME);

        save_optimized_state_to_path(&path, &state()).unwrap();
        assert_eq!(
            restore_optimized_state_at_path(&path).unwrap(),
            RestoreOptimizedStateOutcome::Deleted { path: path.clone() }
        );
        assert!(!path.exists());

        fs::write(&path, unsupported_schema_state_json()).unwrap();
        assert_eq!(
            restore_optimized_state_at_path(&path).unwrap(),
            RestoreOptimizedStateOutcome::Deleted { path: path.clone() }
        );
        assert!(!path.exists());

        assert_eq!(
            restore_optimized_state_at_path(&path).unwrap(),
            RestoreOptimizedStateOutcome::Missing { path: path.clone() }
        );
        assert!(unrelated.exists());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn effective_settings_apply_defaults_valid_state_and_fallbacks() {
        let default = built_in_default_settings();
        assert_eq!(default.payload_profile, PayloadProfile::DefaultChunked64);

        let loaded = resolve_effective_settings_from_load(OptimizedStateLoadOutcome::Loaded {
            path: PathBuf::from("state.json"),
            state: state(),
        });
        assert_eq!(loaded.payload_profile, PayloadProfile::OffsetPrefetch);
        assert!(matches!(
            loaded.source,
            EffectiveSettingsSource::OptimizedState { .. }
        ));

        let missing = resolve_effective_settings_from_load(OptimizedStateLoadOutcome::Missing {
            path: PathBuf::from("missing.json"),
        });
        assert_eq!(missing.payload_profile, PayloadProfile::DefaultChunked64);
        assert!(missing.warning().unwrap().contains("run optimizer"));

        let invalid = resolve_effective_settings_from_load(OptimizedStateLoadOutcome::Invalid {
            path: PathBuf::from("bad.json"),
            error: "bad".to_string(),
        });
        assert_eq!(invalid.payload_profile, PayloadProfile::DefaultChunked64);
        assert!(invalid.warning().unwrap().contains("invalid"));
    }

    #[test]
    fn profile_mappings_are_final_app_values_only() {
        let default_options = PayloadProfile::DefaultChunked64.payload_feeder_options();
        let offset_options = PayloadProfile::OffsetPrefetch.payload_feeder_options();
        assert_eq!(default_options, PayloadFeederOptions::production_default());
        assert_eq!(default_options.reorder_window, 128);
        assert_eq!(offset_options.mode, PayloadReadMode::OffsetPrefetch);
        assert_eq!(offset_options.reorder_window, 128);
    }
}
