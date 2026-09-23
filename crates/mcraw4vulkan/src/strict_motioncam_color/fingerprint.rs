use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use mcraw4vulkan_core::{BayerPattern, FrameDimensions};
use mcraw4vulkan_mcrawcontainer::RawIlluminantToken;
use mcraw4vulkan_vignette::{PipeF32BayerCorrectionMode, PipeF32BayerNumericDomain};

use super::policy::{CONTEXT_MAGIC, CONTEXT_SCHEMA, push_f64, push_i64, push_u32, push_u64};
use super::{
    ResolvedStrictPipeColor, Sha256, StrictMotionCamColorError, StrictMotionCamColorProfile,
    StrictMotionCamForwardMatrixColorV2, StrictMotionCamFrameColorInput,
};

/// Complete-source SHA-256 computed once and reused for all frame identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClipSourceSha256([u8; 32]);

impl ClipSourceSha256 {
    pub(crate) fn read_once_until_cancelled(
        path: impl AsRef<Path>,
        cancelled: &AtomicBool,
    ) -> Result<Option<(Self, u64)>, StrictMotionCamColorError> {
        let file =
            File::open(path.as_ref()).map_err(|error| StrictMotionCamColorError::SourceShaIo {
                detail: error.to_string(),
            })?;
        let mut reader = BufReader::new(file);
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        let mut bytes_read = 0_u64;
        loop {
            if cancelled.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let count = reader.read(&mut buffer).map_err(|error| {
                StrictMotionCamColorError::SourceShaIo {
                    detail: error.to_string(),
                }
            })?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
            bytes_read = bytes_read.checked_add(count as u64).ok_or_else(|| {
                StrictMotionCamColorError::SourceShaIo {
                    detail: "source SHA-256 byte count overflowed".to_string(),
                }
            })?;
        }
        Ok(Some((Self(hasher.finalize()), bytes_read)))
    }

    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }

    /// Internal identity used only while production streams ahead of the
    /// complete-source digest. It is never serialized or published as a
    /// source hash; the deferred context records are rebound to the completed
    /// digest before sidecar construction.
    pub(crate) const fn deferred_stream_identity() -> Self {
        Self([0xa5; 32])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorContextFingerprintFacts {
    pub numeric_domain: PipeF32BayerNumericDomain,
    pub dimensions: FrameDimensions,
    pub bayer_pattern: BayerPattern,
    pub correction_mode: PipeF32BayerCorrectionMode,
    pub source_sha256: ClipSourceSha256,
    pub source_frame_index: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorContextFingerprintV2([u8; 32]);

impl ColorContextFingerprintV2 {
    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Canonical strict-color context record whose complete-source digest is
/// finalized after streaming. All numeric color work has already been
/// validated; finalization changes only the source-identity bytes in the
/// fingerprint record and does not rerun the color solve.
#[derive(Debug, Clone)]
pub(crate) struct DeferredColorContextFingerprintV2 {
    policy_name: &'static str,
    policy_digest: [u8; 32],
    record: Vec<u8>,
    source_sha256_offset: usize,
}

impl DeferredColorContextFingerprintV2 {
    pub(crate) fn policy_identity(&self) -> (&'static str, [u8; 32]) {
        (self.policy_name, self.policy_digest)
    }
    pub(crate) fn finalize(
        mut self,
        source_sha256: ClipSourceSha256,
    ) -> Result<ColorContextFingerprintV2, StrictMotionCamColorError> {
        let end = self
            .source_sha256_offset
            .checked_add(source_sha256.bytes().len())
            .ok_or(StrictMotionCamColorError::ContextIdentityMismatch)?;
        let destination = self
            .record
            .get_mut(self.source_sha256_offset..end)
            .ok_or(StrictMotionCamColorError::ContextIdentityMismatch)?;
        destination.copy_from_slice(&source_sha256.bytes());
        Ok(ColorContextFingerprintV2(Sha256::digest(&self.record)))
    }
}

/// Atomic scheduler-facing color context. Its private fields prevent callers
/// from pairing a valid fingerprint with another frame's resolved matrix.
#[derive(Debug, Clone)]
pub struct VerifiedStrictPipeColorContextV2 {
    resolved: ResolvedStrictPipeColor,
    fingerprint: ColorContextFingerprintV2,
    facts: ColorContextFingerprintFacts,
}

impl VerifiedStrictPipeColorContextV2 {
    pub fn resolved(&self) -> &ResolvedStrictPipeColor {
        &self.resolved
    }

    pub const fn fingerprint(&self) -> ColorContextFingerprintV2 {
        self.fingerprint
    }

    pub const fn fingerprint_facts(&self) -> ColorContextFingerprintFacts {
        self.facts
    }
}

impl StrictMotionCamForwardMatrixColorV2 {
    pub(crate) fn resolve_stream_context(
        &self,
        profile: &StrictMotionCamColorProfile,
        frame: &StrictMotionCamFrameColorInput,
        facts: ColorContextFingerprintFacts,
    ) -> Result<
        (
            VerifiedStrictPipeColorContextV2,
            DeferredColorContextFingerprintV2,
        ),
        StrictMotionCamColorError,
    > {
        let resolved = self.resolve(profile, frame)?;
        let (record, source_sha256_offset) =
            self.context_record_with_source_offset(profile, frame, &resolved, facts)?;
        let fingerprint = ColorContextFingerprintV2(Sha256::digest(&record));
        Ok((
            VerifiedStrictPipeColorContextV2 {
                resolved,
                fingerprint,
                facts,
            },
            DeferredColorContextFingerprintV2 {
                policy_name: profile.policy_name(),
                policy_digest: self.policy_digest_for(profile).bytes(),
                record,
                source_sha256_offset,
            },
        ))
    }

    fn context_record_with_source_offset(
        &self,
        profile: &StrictMotionCamColorProfile,
        frame: &StrictMotionCamFrameColorInput,
        resolved: &ResolvedStrictPipeColor,
        facts: ColorContextFingerprintFacts,
    ) -> Result<(Vec<u8>, usize), StrictMotionCamColorError> {
        let provenance = profile.provenance();
        if frame.provenance() != provenance
            || resolved.provenance() != provenance
            || facts.source_sha256.bytes() != provenance.source_sha256()
            || frame.source_frame_index() != resolved.source_frame_index()
            || frame.source_frame_index() != facts.source_frame_index
            || facts.dimensions.width == 0
            || facts.dimensions.height == 0
        {
            return Err(StrictMotionCamColorError::ContextIdentityMismatch);
        }

        let mut output = Vec::with_capacity(801);
        output.extend_from_slice(if profile.is_color_matrix_only() {
            b"mcraw4vulkan:StrictMotionCamColorMatrixColorV1:context\0".as_slice()
        } else {
            CONTEXT_MAGIC
        });
        push_u32(&mut output, CONTEXT_SCHEMA);
        output.extend_from_slice(&self.policy_digest_for(profile).bytes());
        output.push(match facts.numeric_domain {
            PipeF32BayerNumericDomain::RelativeLinearCorrectedCodeV1 => 1,
        });
        push_u32(&mut output, facts.dimensions.width);
        push_u32(&mut output, facts.dimensions.height);
        output.push(match facts.bayer_pattern {
            BayerPattern::Rggb => 0,
            BayerPattern::Bggr => 1,
            BayerPattern::Grbg => 2,
            BayerPattern::Gbrg => 3,
        });
        output.push(match facts.correction_mode {
            PipeF32BayerCorrectionMode::IdentitySpatialGain => 0,
            PipeF32BayerCorrectionMode::MotionCamSpatial => 1,
        });
        let source_sha256_offset = output.len();
        output.extend_from_slice(&facts.source_sha256.bytes());
        push_u64(&mut output, facts.source_frame_index);
        output.push(1);
        output.push(
            u8::try_from(profile.raw.slots.len())
                .map_err(|_| StrictMotionCamColorError::ContextIdentityMismatch)?,
        );
        for slot in &profile.raw.slots {
            output.push(slot.source_slot.number());
            match slot
                .illuminant
                .as_ref()
                .ok_or(StrictMotionCamColorError::ContextIdentityMismatch)?
            {
                RawIlluminantToken::String(value) => {
                    output.push(1);
                    push_u32(
                        &mut output,
                        u32::try_from(value.len())
                            .map_err(|_| StrictMotionCamColorError::ContextIdentityMismatch)?,
                    );
                    output.extend_from_slice(value.as_bytes());
                }
                RawIlluminantToken::Integer(value) => {
                    output.push(2);
                    push_i64(&mut output, *value);
                }
            }
            let presence = 0b001
                | if slot.forward_matrix.is_some() {
                    0b100
                } else {
                    0
                }
                | if slot.camera_calibration.is_some() {
                    0b010
                } else {
                    0
                };
            output.push(presence);
            append_matrix(
                &mut output,
                slot.color_matrix
                    .ok_or(StrictMotionCamColorError::ContextIdentityMismatch)?
                    .values,
            );
            if let Some(camera_calibration) = slot.camera_calibration {
                append_matrix(&mut output, camera_calibration.values);
            }
            if let Some(forward_matrix) = slot.forward_matrix {
                append_matrix(&mut output, forward_matrix.values);
            }
        }
        match profile.raw.analog_balance {
            None => output.push(0),
            Some(values) => {
                output.push(1);
                append_values(&mut output, values);
            }
        }
        append_values(
            &mut output,
            frame
                .raw
                .as_shot_neutral
                .ok_or(StrictMotionCamColorError::ContextIdentityMismatch)?,
        );
        append_values(&mut output, resolved.camera_neutral());
        for value in [
            resolved.white_xy()[0],
            resolved.white_xy()[1],
            resolved.temperature_kelvin(),
            resolved.tint(),
            resolved.low_temperature_weight(),
        ] {
            push_f64(&mut output, value);
        }
        append_matrix(&mut output, resolved.t50());
        for value in resolved.camera_to_normalized_ncl_f32() {
            output.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        Ok((output, source_sha256_offset))
    }
}

fn append_values<const N: usize>(output: &mut Vec<u8>, values: [f64; N]) {
    for value in values {
        push_f64(output, value);
    }
}

fn append_matrix(output: &mut Vec<u8>, values: [f64; 9]) {
    append_values(output, values);
}
