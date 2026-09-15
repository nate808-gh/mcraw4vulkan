//! Application-owned safe boundary around the small part of macFUSE used by
//! mcraw4vulkan.
//!
//! The implementation and its native linkage exist only on macOS. Other
//! workspace targets neither probe for nor link macFUSE.

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "macos")]
pub use macos::{MacFuseSession, MacFuseSessionError, MacFuseSessionErrorKind, UnmountOutcome};
