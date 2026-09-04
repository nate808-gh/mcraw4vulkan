use crate::{
    BAYER_CFA_PLANE_COUNT, FixedPointVignettePolicy, VIGNETTE_GAIN_SCALE, VignetteCorrectionError,
    VignetteCorrectionPolicy,
};

pub use mcraw4vulkan_mcrawcontainer::LensShadingMap;

// Float preparation borrows validated source planes. Fixed preparation owns
// their Q16.16 copies so correction loops share stable quantized inputs.
#[derive(Debug, Clone, Copy)]
pub struct PreparedLensShadingMap<'a> {
    typed_map: &'a LensShadingMap,
    width: usize,
    height: usize,
    plane_count: usize,
}

impl<'a> PreparedLensShadingMap<'a> {
    pub fn from_typed_map(
        lens_shading_map: &'a LensShadingMap,
    ) -> Result<Self, VignetteCorrectionError> {
        validate_bayer_lens_shading_map(lens_shading_map)?;

        let width = usize::try_from(lens_shading_map.width()).map_err(|_| {
            VignetteCorrectionError::LensShadingMapDimensionsOverflow {
                width: lens_shading_map.width(),
                height: lens_shading_map.height(),
            }
        })?;
        let height = usize::try_from(lens_shading_map.height()).map_err(|_| {
            VignetteCorrectionError::LensShadingMapDimensionsOverflow {
                width: lens_shading_map.width(),
                height: lens_shading_map.height(),
            }
        })?;

        Ok(Self {
            typed_map: lens_shading_map,
            width,
            height,
            plane_count: lens_shading_map.plane_count(),
        })
    }

    pub fn typed_map(&self) -> &'a LensShadingMap {
        self.typed_map
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn plane_count(&self) -> usize {
        self.plane_count
    }

    pub(crate) fn plane(&self, plane_index: usize) -> Result<&[f32], VignetteCorrectionError> {
        self.typed_map.plane(plane_index).ok_or(
            VignetteCorrectionError::LensShadingPlaneOutOfRange {
                plane_index,
                plane_count: self.plane_count,
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct PreparedFixedLensShadingMap<'a> {
    base: PreparedLensShadingMap<'a>,
    gains_q: Vec<Vec<i64>>,
    policy: FixedPointVignettePolicy,
    correction_policy: VignetteCorrectionPolicy,
}

impl<'a> PreparedFixedLensShadingMap<'a> {
    pub fn from_prepared_map(
        base: PreparedLensShadingMap<'a>,
    ) -> Result<Self, VignetteCorrectionError> {
        Self::from_prepared_map_with_policy(base, VignetteCorrectionPolicy::default())
    }

    pub fn from_prepared_map_with_policy(
        base: PreparedLensShadingMap<'a>,
        correction_policy: VignetteCorrectionPolicy,
    ) -> Result<Self, VignetteCorrectionError> {
        let policy = FixedPointVignettePolicy::default();
        let planes = policy_planes(base.typed_map(), correction_policy)?;
        let mut gains_q = Vec::with_capacity(planes.len());

        for (plane_index, plane) in planes.iter().enumerate() {
            let mut quantized_plane = Vec::with_capacity(plane.len());
            for (sample_index, gain) in plane.iter().copied().enumerate() {
                quantized_plane.push(quantize_gain_to_fixed(
                    gain,
                    plane_index,
                    sample_index,
                    policy,
                )?);
            }
            gains_q.push(quantized_plane);
        }

        Ok(Self {
            base,
            gains_q,
            policy,
            correction_policy,
        })
    }

    pub fn from_typed_map(
        lens_shading_map: &'a LensShadingMap,
    ) -> Result<Self, VignetteCorrectionError> {
        Self::from_typed_map_with_policy(lens_shading_map, VignetteCorrectionPolicy::default())
    }

    pub fn from_typed_map_with_policy(
        lens_shading_map: &'a LensShadingMap,
        correction_policy: VignetteCorrectionPolicy,
    ) -> Result<Self, VignetteCorrectionError> {
        Self::from_prepared_map_with_policy(
            PreparedLensShadingMap::from_typed_map(lens_shading_map)?,
            correction_policy,
        )
    }

    pub fn base(&self) -> &PreparedLensShadingMap<'a> {
        &self.base
    }

    pub fn width(&self) -> usize {
        self.base.width()
    }

    pub fn height(&self) -> usize {
        self.base.height()
    }

    pub fn plane_count(&self) -> usize {
        self.base.plane_count()
    }

    pub fn policy(&self) -> FixedPointVignettePolicy {
        self.policy
    }

    pub fn correction_policy(&self) -> VignetteCorrectionPolicy {
        self.correction_policy
    }

    pub(crate) fn plane_q(&self, plane_index: usize) -> Result<&[i64], VignetteCorrectionError> {
        self.gains_q.get(plane_index).map(Vec::as_slice).ok_or(
            VignetteCorrectionError::LensShadingPlaneOutOfRange {
                plane_index,
                plane_count: self.plane_count(),
            },
        )
    }
}

pub fn prepare_lens_shading_map(
    lens_shading_map: &LensShadingMap,
) -> Result<PreparedLensShadingMap<'_>, VignetteCorrectionError> {
    PreparedLensShadingMap::from_typed_map(lens_shading_map)
}

pub fn prepare_fixed_lens_shading_map(
    lens_shading_map: &LensShadingMap,
) -> Result<PreparedFixedLensShadingMap<'_>, VignetteCorrectionError> {
    PreparedFixedLensShadingMap::from_typed_map(lens_shading_map)
}

pub fn lens_shading_map_for_policy(
    lens_shading_map: &LensShadingMap,
    policy: VignetteCorrectionPolicy,
) -> Result<LensShadingMap, VignetteCorrectionError> {
    match policy {
        VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1 => {
            validate_bayer_lens_shading_map(lens_shading_map)?;
            Ok(lens_shading_map.clone())
        }
        VignetteCorrectionPolicy::LumaPlane0 => {
            // Plane 0 deliberately serves every CFA position; materializing four
            // planes preserves the downstream per-position indexing contract.
            let plane = lens_shading_map.plane(0).ok_or(
                VignetteCorrectionError::LensShadingPlaneOutOfRange {
                    plane_index: 0,
                    plane_count: lens_shading_map.plane_count(),
                },
            )?;
            LensShadingMap::new(
                lens_shading_map.width(),
                lens_shading_map.height(),
                vec![
                    plane.to_vec(),
                    plane.to_vec(),
                    plane.to_vec(),
                    plane.to_vec(),
                ],
            )
            .map_err(VignetteCorrectionError::from)
        }
    }
}

pub fn validate_bayer_lens_shading_map(
    lens_shading_map: &LensShadingMap,
) -> Result<(), VignetteCorrectionError> {
    if lens_shading_map.plane_count() != BAYER_CFA_PLANE_COUNT {
        return Err(VignetteCorrectionError::UnsupportedLensShadingPlaneCount {
            expected: BAYER_CFA_PLANE_COUNT,
            actual: lens_shading_map.plane_count(),
        });
    }

    Ok(())
}

fn policy_planes(
    lens_shading_map: &LensShadingMap,
    policy: VignetteCorrectionPolicy,
) -> Result<Vec<&[f32]>, VignetteCorrectionError> {
    match policy {
        VignetteCorrectionPolicy::MotionCamCompatiblePixelDomainV1 => {
            validate_bayer_lens_shading_map(lens_shading_map)?;
            Ok(lens_shading_map
                .planes()
                .iter()
                .map(Vec::as_slice)
                .collect())
        }
        VignetteCorrectionPolicy::LumaPlane0 => {
            let plane = lens_shading_map.plane(0).ok_or(
                VignetteCorrectionError::LensShadingPlaneOutOfRange {
                    plane_index: 0,
                    plane_count: lens_shading_map.plane_count(),
                },
            )?;
            Ok(vec![plane, plane, plane, plane])
        }
    }
}

fn quantize_gain_to_fixed(
    gain: f32,
    plane_index: usize,
    sample_index: usize,
    policy: FixedPointVignettePolicy,
) -> Result<i64, VignetteCorrectionError> {
    let scaled = f64::from(gain) * policy.gain_scale as f64;
    if !scaled.is_finite() || scaled < 0.0 || scaled > i64::MAX as f64 {
        return Err(VignetteCorrectionError::FixedPointGainOverflow {
            plane_index,
            sample_index,
            value: gain,
        });
    }

    Ok(scaled.round() as i64)
}

pub(crate) fn quantize_nonnegative_to_fixed(value: f32) -> Option<i64> {
    let scaled = f64::from(value) * VIGNETTE_GAIN_SCALE as f64;
    if !scaled.is_finite() || scaled < 0.0 || scaled > i64::MAX as f64 {
        return None;
    }

    Some(scaled.round() as i64)
}
