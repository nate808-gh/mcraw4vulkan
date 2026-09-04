use super::StrictMotionCamColorError;
use super::policy::{
    MAX_COMPOSITE_ELEMENT, MAX_CONDITION_1, MAX_INPUT_ELEMENT, MIN_RELATIVE_DETERMINANT,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Matrix3(pub(super) [f64; 9]);

pub(super) const IDENTITY: Matrix3 = Matrix3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

impl Matrix3 {
    pub(super) fn multiply(self, right: Self) -> Self {
        let mut output = [0.0; 9];
        for row in 0..3 {
            for column in 0..3 {
                output[row * 3 + column] = python_sum3([
                    self.0[row * 3] * right.0[column],
                    self.0[row * 3 + 1] * right.0[3 + column],
                    self.0[row * 3 + 2] * right.0[6 + column],
                ]);
            }
        }
        Self(output)
    }

    pub(super) fn apply(self, vector: [f64; 3]) -> [f64; 3] {
        let mut output = [0.0; 3];
        for (row, value) in output.iter_mut().enumerate() {
            *value = python_sum3([
                self.0[row * 3] * vector[0],
                self.0[row * 3 + 1] * vector[1],
                self.0[row * 3 + 2] * vector[2],
            ]);
        }
        output
    }

    pub(super) fn determinant(self) -> f64 {
        self.0[0] * (self.0[4] * self.0[8] - self.0[5] * self.0[7])
            - self.0[1] * (self.0[3] * self.0[8] - self.0[5] * self.0[6])
            + self.0[2] * (self.0[3] * self.0[7] - self.0[4] * self.0[6])
    }

    pub(super) fn norm1(self) -> f64 {
        let mut maximum = 0.0_f64;
        for column in 0..3 {
            let sum = python_sum3([
                self.0[column].abs(),
                self.0[3 + column].abs(),
                self.0[6 + column].abs(),
            ]);
            maximum = maximum.max(sum);
        }
        maximum
    }

    pub(super) fn max_abs(self) -> f64 {
        self.0.iter().copied().map(f64::abs).fold(0.0, f64::max)
    }

    pub(super) fn all_finite(self) -> bool {
        self.0.iter().all(|value| value.is_finite())
    }

    pub(super) fn inverse(self, stage: &'static str) -> Result<Self, StrictMotionCamColorError> {
        if !self.all_finite() {
            return Err(StrictMotionCamColorError::NonfiniteMatrix { stage });
        }
        let determinant = self.determinant();
        let norm = self.norm1();
        let relative_determinant = if norm == 0.0 {
            0.0
        } else {
            determinant.abs() / norm.powi(3)
        };
        if !relative_determinant.is_finite() || relative_determinant <= MIN_RELATIVE_DETERMINANT {
            return Err(StrictMotionCamColorError::SingularMatrix {
                stage,
                relative_determinant,
            });
        }
        let m = self.0;
        let inverse = Self([
            (m[4] * m[8] - m[5] * m[7]) / determinant,
            (m[2] * m[7] - m[1] * m[8]) / determinant,
            (m[1] * m[5] - m[2] * m[4]) / determinant,
            (m[5] * m[6] - m[3] * m[8]) / determinant,
            (m[0] * m[8] - m[2] * m[6]) / determinant,
            (m[2] * m[3] - m[0] * m[5]) / determinant,
            (m[3] * m[7] - m[4] * m[6]) / determinant,
            (m[1] * m[6] - m[0] * m[7]) / determinant,
            (m[0] * m[4] - m[1] * m[3]) / determinant,
        ]);
        if !inverse.all_finite() {
            return Err(StrictMotionCamColorError::NonfiniteMatrix { stage });
        }
        let condition = norm * inverse.norm1();
        if !condition.is_finite() || condition > MAX_CONDITION_1 {
            return Err(StrictMotionCamColorError::IllConditionedMatrix { stage, condition });
        }
        Ok(inverse)
    }

    pub(super) fn validate_input(
        self,
        stage: &'static str,
    ) -> Result<Self, StrictMotionCamColorError> {
        if !self.all_finite() {
            return Err(StrictMotionCamColorError::NonfiniteMatrix { stage });
        }
        let maximum = self.max_abs();
        if maximum > MAX_INPUT_ELEMENT {
            return Err(StrictMotionCamColorError::UnreasonableMatrix { stage, maximum });
        }
        let _ = self.inverse(stage)?;
        Ok(self)
    }

    pub(super) fn validate_composite(
        self,
        stage: &'static str,
    ) -> Result<Self, StrictMotionCamColorError> {
        if !self.all_finite() {
            return Err(StrictMotionCamColorError::NonfiniteComposite { stage });
        }
        let maximum = self.max_abs();
        if maximum > MAX_COMPOSITE_ELEMENT {
            return Err(StrictMotionCamColorError::UnreasonableComposite { stage, maximum });
        }
        let _ = self.inverse(stage)?;
        Ok(self)
    }
}

/// CPython 3.14 specializes built-in sum for floats with a Neumaier
/// compensation. The hash-pinned strict-color reference uses that operation for every
/// three-term row/column sum, so the production implementation preserves it.
pub(super) fn python_sum3(values: [f64; 3]) -> f64 {
    let mut high = 0.0;
    let mut low = 0.0;
    for value in values {
        let next = high + value;
        if high.abs() >= value.abs() {
            low += (high - next) + value;
        } else {
            low += (value - next) + high;
        }
        high = next;
    }
    high + low
}

pub(super) fn diagonal(values: [f64; 3]) -> Matrix3 {
    Matrix3([
        values[0], 0.0, 0.0, 0.0, values[1], 0.0, 0.0, 0.0, values[2],
    ])
}

pub(super) fn lerp(left: Matrix3, right: Matrix3, weight_left: f64) -> Matrix3 {
    let weight_right = 1.0 - weight_left;
    let mut output = [0.0; 9];
    for (index, value) in output.iter_mut().enumerate() {
        *value = weight_left * left.0[index] + weight_right * right.0[index];
    }
    Matrix3(output)
}
