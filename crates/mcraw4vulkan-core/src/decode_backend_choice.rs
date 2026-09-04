use std::error::Error;
use std::fmt;
use std::str::FromStr;

// Shared user-facing choice used by CLI, GUI, playlists, and playback. GPU is
// the default; CPU remains the explicit parity and fallback path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeBackendChoice {
    #[default]
    Gpu,
    Cpu,
}

impl DecodeBackendChoice {
    pub const DEFAULT: Self = Self::Gpu;
    pub const CLI_VALUES: &'static str = "gpu|cpu";

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Gpu => "GPU",
            Self::Cpu => "CPU",
        }
    }

    pub fn is_gpu(self) -> bool {
        matches!(self, Self::Gpu)
    }

    pub fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }
}

impl fmt::Display for DecodeBackendChoice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for DecodeBackendChoice {
    type Err = ParseDecodeBackendChoiceError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "gpu" => Ok(Self::Gpu),
            "cpu" => Ok(Self::Cpu),
            _ => Err(ParseDecodeBackendChoiceError),
        }
    }
}

// Error returned when parsing a user-provided decode backend choice fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseDecodeBackendChoiceError;

impl fmt::Display for ParseDecodeBackendChoiceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid decode backend; expected one of: {}",
            DecodeBackendChoice::CLI_VALUES
        )
    }
}

impl Error for ParseDecodeBackendChoiceError {}
