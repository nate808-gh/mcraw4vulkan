use std::fmt;

// Stable application-level identity for one registered .mcraw clip.
//
// This is intentionally separate from the existing ClipId used by clip metadata.
// RegisteredClipId is for the future GUI/resource-manager layer:
//
//   RegisteredClipId -> path, metadata, backend choice, mount state, preview state
//
// Keeping this numeric ID separate avoids breaking existing decoder/session code
// that currently uses ClipId for clip/display metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegisteredClipId(u64);

impl RegisteredClipId {
    // Create a registered clip ID from an already-assigned numeric value.
    //
    // The future AppResourceManager should assign these monotonically. This type
    // deliberately does not perform allocation or global registration by itself.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    // Return the numeric value for logging, diagnostics, serialization glue, or
    // stable lookup tables.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for RegisteredClipId {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl From<RegisteredClipId> for u64 {
    fn from(value: RegisteredClipId) -> Self {
        value.as_u64()
    }
}

impl fmt::Display for RegisteredClipId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "clip-{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::RegisteredClipId;

    #[test]
    fn registered_clip_id_round_trips_to_u64() {
        let clip_id = RegisteredClipId::new(42);

        assert_eq!(clip_id.as_u64(), 42);
        assert_eq!(u64::from(clip_id), 42);
        assert_eq!(RegisteredClipId::from(42), clip_id);
    }

    #[test]
    fn registered_clip_id_display_is_stable_for_logs() {
        assert_eq!(RegisteredClipId::new(7).to_string(), "clip-7");
    }
}
