use mcraw4vulkan_core::{BayerPattern, FrameDimensions};

use crate::math::{
    android_rggb_source_plane_index, cfa_position_plane_index, interpolated_fixed_gain,
};
use crate::{
    FixedPointVignetteInputFacts, MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH,
    PreparedFixedLensShadingMap, VIGNETTE_GAIN_FRACTIONAL_BITS, VIGNETTE_GAIN_SCALE,
    VignetteCoordinateMapping, VignetteCorrectionError, VignetteCorrectionMode,
    VignetteCorrectionPolicy,
};

// The full fingerprint covers every value that affects final per-pixel gains.
// Split spatial/conversion fingerprints allow the GPU to retain map bytes when
// only source-domain conversion changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VignetteGainMapFingerprint {
    high: u64,
    low: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompactSpatialMapFingerprint {
    high: u64,
    low: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GainConversionFingerprint {
    high: u64,
    low: u64,
}

impl CompactSpatialMapFingerprint {
    /// Stable typed words for higher-level context serialization.
    pub const fn words(self) -> [u64; 2] {
        [self.high, self.low]
    }
}

impl GainConversionFingerprint {
    /// Stable typed words for higher-level context serialization.
    pub const fn words(self) -> [u64; 2] {
        [self.high, self.low]
    }
}

impl VignetteGainMapFingerprint {
    pub fn from_fixed_facts(
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<Self, VignetteCorrectionError> {
        let prepared_map = facts
            .lens_shading_map
            .as_ref()
            .ok_or(VignetteCorrectionError::MissingLensShadingMap)?;
        let mut hash = StableVignetteHash::new();

        hash.write_bytes(b"mcraw4vulkan:vignette-full-res-gain:v1");
        hash.write_u32(facts.frame_dimensions.width);
        hash.write_u32(facts.frame_dimensions.height);
        hash.write_u8(vignette_mode_tag(facts.mode));
        hash.write_u8(vignette_policy_tag(facts.correction_policy));
        hash.write_u8(coordinate_mapping_tag(facts.coordinate_mapping));
        hash.write_u8(bayer_pattern_tag(facts.bayer_pattern));
        hash.write_usize(prepared_map.width());
        hash.write_usize(prepared_map.height());
        hash.write_usize(prepared_map.plane_count());
        hash.write_u32(prepared_map.policy().fractional_bits);
        hash.write_i64(prepared_map.policy().gain_scale);
        hash.write_u8(vignette_policy_tag(prepared_map.correction_policy()));
        hash.write_u32(VIGNETTE_GAIN_FRACTIONAL_BITS);
        hash.write_i64(VIGNETTE_GAIN_SCALE);
        hash.write_f32(MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH);

        for value in facts.input_black_level {
            hash.write_f32(value);
        }
        for value in facts.input_black_level_q {
            hash.write_i64(value);
        }
        hash.write_u16(facts.output_white_level);
        for value in facts.pixel_domain.source_black_storage {
            hash.write_f32(value);
        }
        hash.write_u16(facts.pixel_domain.source_white_storage);
        hash.write_u16(facts.pixel_domain.corrected_white_tag);
        hash.write_f32(facts.pixel_domain.source_to_corrected_scale);
        hash.write_u16(facts.pixel_domain.sample_limit);

        for plane_index in 0..prepared_map.plane_count() {
            let plane = prepared_map.plane_q(plane_index)?;
            hash.write_usize(plane.len());
            for value in plane {
                hash.write_i64(*value);
            }
        }

        Ok(hash.finish())
    }

    pub fn high(self) -> u64 {
        self.high
    }

    pub fn low(self) -> u64 {
        self.low
    }
}

impl CompactSpatialMapFingerprint {
    pub fn from_fixed_facts(
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<Self, VignetteCorrectionError> {
        let prepared_map = facts
            .lens_shading_map
            .as_ref()
            .ok_or(VignetteCorrectionError::MissingLensShadingMap)?;
        let mut hash = StableVignetteHash::new();

        hash.write_bytes(b"mcraw4vulkan:vignette-compact-spatial:v1");
        hash.write_u8(coordinate_mapping_tag(facts.coordinate_mapping));
        hash.write_usize(prepared_map.width());
        hash.write_usize(prepared_map.height());
        hash.write_usize(prepared_map.plane_count());
        hash.write_u32(prepared_map.policy().fractional_bits);
        hash.write_i64(prepared_map.policy().gain_scale);
        hash.write_u8(vignette_policy_tag(prepared_map.correction_policy()));

        for plane_index in 0..prepared_map.plane_count() {
            let plane = prepared_map.plane_q(plane_index)?;
            hash.write_usize(plane.len());
            for value in plane {
                hash.write_i64(*value);
            }
        }

        let (high, low) = hash.finish_parts();
        Ok(Self { high, low })
    }

    pub fn high(self) -> u64 {
        self.high
    }

    pub fn low(self) -> u64 {
        self.low
    }
}

impl GainConversionFingerprint {
    pub fn from_fixed_facts(
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<Self, VignetteCorrectionError> {
        let mut hash = StableVignetteHash::new();

        hash.write_bytes(b"mcraw4vulkan:vignette-gain-conversion:v1");
        hash.write_u8(vignette_policy_tag(facts.correction_policy));
        hash.write_u32(VIGNETTE_GAIN_FRACTIONAL_BITS);
        hash.write_i64(VIGNETTE_GAIN_SCALE);
        match facts.correction_policy {
            VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1 => {
                hash.write_f32(MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH);
                hash.write_f32(facts.pixel_domain.source_to_corrected_scale);
            }
            VignetteCorrectionPolicy::LumaPlane0 => {}
        }

        let (high, low) = hash.finish_parts();
        Ok(Self { high, low })
    }

    pub fn high(self) -> u64 {
        self.high
    }

    pub fn low(self) -> u64 {
        self.low
    }
}

// Cache keys are derived from output-affecting values rather than metadata
// object identity; a match reuses the cache's owned prepared map.
#[derive(Debug, Default)]
pub struct PreparedFullResolutionGainMapCache {
    entry: Option<PreparedFullResolutionGainMapCacheEntry>,
    stats: PreparedFullResolutionGainMapCacheStats,
}

#[derive(Debug)]
struct PreparedFullResolutionGainMapCacheEntry {
    fingerprint: VignetteGainMapFingerprint,
    map: PreparedFullResolutionFixedGainMap,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PreparedFullResolutionGainMapCacheStats {
    pub hits: u64,
    pub misses: u64,
}

impl PreparedFullResolutionGainMapCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_prepare(
        &mut self,
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<&PreparedFullResolutionFixedGainMap, VignetteCorrectionError> {
        let fingerprint = VignetteGainMapFingerprint::from_fixed_facts(facts)?;
        if self
            .entry
            .as_ref()
            .is_some_and(|entry| entry.fingerprint == fingerprint)
        {
            self.stats.hits = self.stats.hits.saturating_add(1);
            return Ok(&self.entry.as_ref().expect("entry matched").map);
        }

        let map = PreparedFullResolutionFixedGainMap::from_fixed_facts(facts)?;
        self.entry = Some(PreparedFullResolutionGainMapCacheEntry { fingerprint, map });
        self.stats.misses = self.stats.misses.saturating_add(1);

        Ok(&self.entry.as_ref().expect("entry was just populated").map)
    }

    pub fn clear(&mut self) {
        self.entry = None;
    }

    pub fn stats(&self) -> PreparedFullResolutionGainMapCacheStats {
        self.stats
    }

    pub fn cached_fingerprint(&self) -> Option<VignetteGainMapFingerprint> {
        self.entry.as_ref().map(|entry| entry.fingerprint)
    }
}

// One raster-order Q16.16 gain is owned for every visible pixel, giving cache
// and upload paths deterministic bytes independent of the source map's storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedFullResolutionFixedGainMap {
    frame_dimensions: FrameDimensions,
    source_map_width: usize,
    source_map_height: usize,
    source_plane_count: usize,
    fractional_bits: u32,
    gains_q16: Vec<u32>,
}

impl PreparedFullResolutionFixedGainMap {
    pub fn from_prepared_fixed_map(
        prepared_map: &PreparedFixedLensShadingMap<'_>,
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<Self, VignetteCorrectionError> {
        validate_frame_dimensions(facts.frame_dimensions)?;

        let pixel_count = facts.frame_dimensions.pixel_count().ok_or(
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: facts.frame_dimensions,
            },
        )?;
        let width = usize::try_from(facts.frame_dimensions.width).map_err(|_| {
            VignetteCorrectionError::FrameDimensionsOverflow {
                dimensions: facts.frame_dimensions,
            }
        })?;
        let mut gains_q16 = Vec::with_capacity(pixel_count);

        for sample_index in 0..pixel_count {
            let x = sample_index % width;
            let y = sample_index / width;
            let gain_q = match facts.correction_policy {
                VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1 => {
                    motioncam_compatible_pixel_domain_gain_q16(prepared_map, facts, x, y)?
                }
                VignetteCorrectionPolicy::LumaPlane0 => {
                    let plane_index = cfa_position_plane_index(facts.bayer_pattern, x, y);
                    interpolated_fixed_gain(
                        prepared_map,
                        plane_index,
                        x,
                        y,
                        facts.frame_dimensions,
                        facts.coordinate_mapping,
                    )?
                }
            };

            gains_q16.push(u32::try_from(gain_q).map_err(|_| {
                VignetteCorrectionError::FullResolutionGainMapGainOverflow {
                    x,
                    y,
                    value: gain_q,
                }
            })?);
        }

        Ok(Self {
            frame_dimensions: facts.frame_dimensions,
            source_map_width: prepared_map.width(),
            source_map_height: prepared_map.height(),
            source_plane_count: prepared_map.plane_count(),
            fractional_bits: VIGNETTE_GAIN_FRACTIONAL_BITS,
            gains_q16,
        })
    }

    pub fn from_fixed_facts(
        facts: &FixedPointVignetteInputFacts<'_>,
    ) -> Result<Self, VignetteCorrectionError> {
        let prepared_map = facts
            .lens_shading_map
            .as_ref()
            .ok_or(VignetteCorrectionError::MissingLensShadingMap)?;

        Self::from_prepared_fixed_map(prepared_map, facts)
    }

    pub fn frame_dimensions(&self) -> FrameDimensions {
        self.frame_dimensions
    }

    pub fn width(&self) -> u32 {
        self.frame_dimensions.width
    }

    pub fn height(&self) -> u32 {
        self.frame_dimensions.height
    }

    pub fn pixel_count(&self) -> usize {
        self.gains_q16.len()
    }

    pub fn len(&self) -> usize {
        self.gains_q16.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gains_q16.is_empty()
    }

    pub fn gains_q16(&self) -> &[u32] {
        &self.gains_q16
    }

    pub fn gain_at(&self, x: u32, y: u32) -> Option<u32> {
        if x >= self.frame_dimensions.width || y >= self.frame_dimensions.height {
            return None;
        }

        let width = usize::try_from(self.frame_dimensions.width).ok()?;
        let x = usize::try_from(x).ok()?;
        let y = usize::try_from(y).ok()?;
        let index = y.checked_mul(width)?.checked_add(x)?;

        self.gains_q16.get(index).copied()
    }

    pub fn memory_bytes(&self) -> usize {
        self.gains_q16
            .len()
            .saturating_mul(std::mem::size_of::<u32>())
    }

    pub fn fixed_gain_fractional_bits(&self) -> u32 {
        self.fractional_bits
    }

    pub fn source_map_width(&self) -> usize {
        self.source_map_width
    }

    pub fn source_map_height(&self) -> usize {
        self.source_map_height
    }

    pub fn source_plane_count(&self) -> usize {
        self.source_plane_count
    }
}

fn motioncam_compatible_pixel_domain_gain_q16(
    prepared_map: &PreparedFixedLensShadingMap<'_>,
    facts: &FixedPointVignetteInputFacts<'_>,
    x: usize,
    y: usize,
) -> Result<i64, VignetteCorrectionError> {
    let plane_index = android_rggb_source_plane_index(facts.bayer_pattern, x, y);
    let gain_q = interpolated_fixed_gain(
        prepared_map,
        plane_index,
        x,
        y,
        facts.frame_dimensions,
        facts.coordinate_mapping,
    )?;

    motioncam_compatible_gain_q16_from_raw_gain_q(gain_q, facts, x, y)
}

pub(crate) fn motioncam_compatible_gain_q16_from_raw_gain_q(
    gain_q: i64,
    facts: &FixedPointVignetteInputFacts<'_>,
    x: usize,
    y: usize,
) -> Result<i64, VignetteCorrectionError> {
    let gain = gain_q as f64 / VIGNETTE_GAIN_SCALE as f64;
    let compressed_gain =
        1.0 + f64::from(MOTIONCAM_PIXEL_GAIN_LINEAR_STRENGTH) * (gain - 1.0).max(0.0);
    let final_gain = f64::from(facts.pixel_domain.source_to_corrected_scale) * compressed_gain;
    let gain_q16 = (final_gain.max(0.0) * VIGNETTE_GAIN_SCALE as f64).round();
    if !gain_q16.is_finite() || gain_q16 < 0.0 || gain_q16 > i64::MAX as f64 {
        return Err(VignetteCorrectionError::FullResolutionGainMapGainOverflow {
            x,
            y,
            value: i64::MAX,
        });
    }

    Ok(gain_q16 as i64)
}

fn validate_frame_dimensions(dimensions: FrameDimensions) -> Result<(), VignetteCorrectionError> {
    if dimensions.width == 0 || dimensions.height == 0 {
        return Err(VignetteCorrectionError::InvalidFrameDimensions { dimensions });
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct StableVignetteHash {
    high: u64,
    low: u64,
}

impl StableVignetteHash {
    const HIGH_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const LOW_OFFSET: u64 = 0x8422_2325_cbf2_9ce4;
    const HIGH_PRIME: u64 = 0x0000_0100_0000_01b3;
    const LOW_PRIME: u64 = 0x0000_0100_0000_01b5;

    fn new() -> Self {
        Self {
            high: Self::HIGH_OFFSET,
            low: Self::LOW_OFFSET,
        }
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.high ^= u64::from(*byte);
            self.high = self.high.wrapping_mul(Self::HIGH_PRIME);
            self.low ^= u64::from(*byte).rotate_left(1);
            self.low = self.low.wrapping_mul(Self::LOW_PRIME);
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.write_bytes(&[value]);
    }

    fn write_u16(&mut self, value: u16) {
        self.write_bytes(&value.to_le_bytes());
    }

    fn write_u32(&mut self, value: u32) {
        self.write_bytes(&value.to_le_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    fn write_i64(&mut self, value: i64) {
        self.write_bytes(&value.to_le_bytes());
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(u64::try_from(value).unwrap_or(u64::MAX));
    }

    fn write_f32(&mut self, value: f32) {
        self.write_u32(value.to_bits());
    }

    fn finish(self) -> VignetteGainMapFingerprint {
        VignetteGainMapFingerprint {
            high: self.high,
            low: self.low,
        }
    }

    fn finish_parts(self) -> (u64, u64) {
        (self.high, self.low)
    }
}

fn vignette_mode_tag(mode: VignetteCorrectionMode) -> u8 {
    match mode {
        VignetteCorrectionMode::Disabled => 0,
        VignetteCorrectionMode::Enabled => 1,
    }
}

fn vignette_policy_tag(policy: VignetteCorrectionPolicy) -> u8 {
    match policy {
        VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1 => 0,
        VignetteCorrectionPolicy::LumaPlane0 => 1,
    }
}

fn coordinate_mapping_tag(mapping: VignetteCoordinateMapping) -> u8 {
    match mapping {
        VignetteCoordinateMapping::VisibleFrame => 0,
    }
}

fn bayer_pattern_tag(pattern: BayerPattern) -> u8 {
    match pattern {
        BayerPattern::Rggb => 0,
        BayerPattern::Bggr => 1,
        BayerPattern::Grbg => 2,
        BayerPattern::Gbrg => 3,
    }
}
