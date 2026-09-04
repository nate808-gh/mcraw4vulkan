use std::error::Error;
use std::fmt;

// Runtime-sized .mcraw lens shading map metadata.
//
// MotionCam stores map dimensions per frame. Real samples already include
// device-specific sizes such as 33x25 and 17x13, so validation must never rely
// on fixed Pixel dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct LensShadingMap {
    width: u32,
    height: u32,
    planes: Vec<Vec<f32>>,
}

impl LensShadingMap {
    pub fn new(
        width: u32,
        height: u32,
        planes: Vec<Vec<f32>>,
    ) -> Result<Self, LensShadingMapValidationError> {
        if width == 0 || height == 0 {
            return Err(LensShadingMapValidationError::EmptyDimensions);
        }

        let expected_plane_len = expected_plane_len(width, height)?;

        for (plane_index, plane) in planes.iter().enumerate() {
            if plane.len() != expected_plane_len {
                return Err(LensShadingMapValidationError::PlaneLengthMismatch {
                    plane_index,
                    expected_len: expected_plane_len,
                    actual_len: plane.len(),
                });
            }

            for (sample_index, value) in plane.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(LensShadingMapValidationError::NonFiniteGain {
                        plane_index,
                        sample_index,
                    });
                }

                if value < 0.0 {
                    return Err(LensShadingMapValidationError::NegativeGain {
                        plane_index,
                        sample_index,
                        value,
                    });
                }
            }
        }

        Ok(Self {
            width,
            height,
            planes,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn plane_count(&self) -> usize {
        self.planes.len()
    }

    pub fn expected_plane_len(&self) -> usize {
        // Safe because construction already validated nonzero dimensions.
        expected_plane_len(self.width, self.height).unwrap_or(0)
    }

    pub fn planes(&self) -> &[Vec<f32>] {
        &self.planes
    }

    pub fn plane(&self, plane_index: usize) -> Option<&[f32]> {
        self.planes.get(plane_index).map(Vec::as_slice)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LensShadingMapValidationError {
    EmptyDimensions,
    DimensionsOverflow {
        width: u32,
        height: u32,
    },
    PlaneLengthMismatch {
        plane_index: usize,
        expected_len: usize,
        actual_len: usize,
    },
    NonFiniteGain {
        plane_index: usize,
        sample_index: usize,
    },
    NegativeGain {
        plane_index: usize,
        sample_index: usize,
        value: f32,
    },
}

impl fmt::Display for LensShadingMapValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimensions => {
                formatter.write_str("lensShadingMap dimensions must be non-zero")
            }
            Self::DimensionsOverflow { width, height } => {
                write!(
                    formatter,
                    "lensShadingMap dimensions {width}x{height} overflow"
                )
            }
            Self::PlaneLengthMismatch {
                plane_index,
                expected_len,
                actual_len,
            } => write!(
                formatter,
                "lensShadingMap[{plane_index}] len {actual_len} does not match width * height {expected_len}"
            ),
            Self::NonFiniteGain {
                plane_index,
                sample_index,
            } => write!(
                formatter,
                "lensShadingMap[{plane_index}][{sample_index}] must be finite"
            ),
            Self::NegativeGain {
                plane_index,
                sample_index,
                value,
            } => write!(
                formatter,
                "lensShadingMap[{plane_index}][{sample_index}] is negative: {value}"
            ),
        }
    }
}

impl Error for LensShadingMapValidationError {}

fn expected_plane_len(width: u32, height: u32) -> Result<usize, LensShadingMapValidationError> {
    let pixel_count = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(LensShadingMapValidationError::DimensionsOverflow { width, height })?;

    usize::try_from(pixel_count)
        .map_err(|_| LensShadingMapValidationError::DimensionsOverflow { width, height })
}
