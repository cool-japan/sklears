//! Isotonic regression for probability calibration
//!
//! Implements isotonic regression which finds a non-decreasing approximation
//! to a mapping that can be used for probability calibration.

use crate::CalibrationEstimator;
use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, types::Float};
use std::cmp::Ordering;

/// Isotonic calibrator using the Pool Adjacent Violators Algorithm (PAVA)
#[derive(Debug, Clone)]
pub struct IsotonicCalibrator {
    /// Increasing function values (y)
    y_: Vec<Float>,
    /// Increasing function inputs (x)
    x_: Vec<Float>,
}

impl IsotonicCalibrator {
    /// Create a new isotonic calibrator
    pub fn new() -> Self {
        Self {
            y_: Vec::new(),
            x_: Vec::new(),
        }
    }

    /// Fit the isotonic calibrator using Pool Adjacent Violators Algorithm
    pub fn fit(mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<Self> {
        if probabilities.len() != y_true.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input arrays must have the same length".to_string(),
            ));
        }

        if probabilities.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "No probabilities provided".to_string(),
            ));
        }

        // Convert to tuples and sort by probability
        let mut pairs: Vec<(Float, Float)> = probabilities
            .iter()
            .zip(y_true.iter())
            .map(|(&prob, &y)| (prob, y as Float))
            .collect();

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Apply Pool Adjacent Violators Algorithm
        let (x_isotonic, y_isotonic) = self.pool_adjacent_violators(&pairs);

        self.x_ = x_isotonic;
        self.y_ = y_isotonic;

        Ok(self)
    }

    /// Pool Adjacent Violators Algorithm implementation
    fn pool_adjacent_violators(&self, pairs: &[(Float, Float)]) -> (Vec<Float>, Vec<Float>) {
        if pairs.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let mut x_pooled = Vec::new();
        let mut y_pooled = Vec::new();
        let mut weights = Vec::new();

        // Initialize with first point
        x_pooled.push(pairs[0].0);
        y_pooled.push(pairs[0].1);
        weights.push(1.0);

        for &(x, y) in pairs.iter().skip(1) {
            x_pooled.push(x);
            y_pooled.push(y);
            weights.push(1.0);

            // Check for violations and pool adjacent violators
            let mut i = x_pooled.len() - 1;
            while i > 0 {
                if y_pooled[i] < y_pooled[i - 1] {
                    // Violation detected, pool the violators
                    let total_weight = weights[i - 1] + weights[i];
                    let pooled_y = (weights[i - 1] * y_pooled[i - 1] + weights[i] * y_pooled[i])
                        / total_weight;

                    // Update the previous point with pooled values
                    y_pooled[i - 1] = pooled_y;
                    weights[i - 1] = total_weight;

                    // Remove the current point
                    x_pooled.remove(i);
                    y_pooled.remove(i);
                    weights.remove(i);

                    i = i.saturating_sub(1);
                } else {
                    break;
                }
            }
        }

        (x_pooled, y_pooled)
    }

    /// Predict calibrated probabilities using linear interpolation
    pub fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if self.x_.is_empty() || self.y_.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Calibrator not fitted".to_string(),
            ));
        }

        let calibrated = probabilities.mapv(|prob| self.interpolate(prob));
        Ok(calibrated)
    }

    /// Linear interpolation for a single probability value
    fn interpolate(&self, x: Float) -> Float {
        if self.x_.is_empty() {
            return x; // fallback
        }

        // Handle edge cases
        if x <= self.x_[0] {
            return self.y_[0];
        }
        if x >= *self.x_.last().unwrap() {
            return *self.y_.last().unwrap();
        }

        // Find the two points to interpolate between
        for i in 0..self.x_.len() - 1 {
            if x >= self.x_[i] && x <= self.x_[i + 1] {
                let x0 = self.x_[i];
                let x1 = self.x_[i + 1];
                let y0 = self.y_[i];
                let y1 = self.y_[i + 1];

                if (x1 - x0).abs() < Float::EPSILON {
                    return y0;
                }

                return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
            }
        }

        x // fallback
    }
}

impl Default for IsotonicCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationEstimator for IsotonicCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        *self = Self::new().fit(probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        IsotonicCalibrator::predict_proba(self, probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_isotonic_calibrator_basic() {
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1];

        let calibrator = IsotonicCalibrator::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.len(), 4);

        // Calibrated probabilities should be valid
        for &prob in calibrated.iter() {
            assert!(
                (0.0..=1.0).contains(&prob),
                "Probability {} out of bounds",
                prob
            );
        }

        // Should be isotonic (non-decreasing)
        for i in 1..calibrated.len() {
            assert!(
                calibrated[i] >= calibrated[i - 1] - Float::EPSILON,
                "Isotonic property violated: {} < {}",
                calibrated[i],
                calibrated[i - 1]
            );
        }
    }

    #[test]
    fn test_isotonic_calibrator_monotonic_data() {
        // Test with perfectly calibrated data (should remain unchanged)
        let probabilities = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let y_true = array![0, 0, 0, 1, 1];

        let calibrator = IsotonicCalibrator::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        // Should preserve isotonic property
        for i in 1..calibrated.len() {
            assert!(calibrated[i] >= calibrated[i - 1] - Float::EPSILON);
        }
    }

    #[test]
    fn test_isotonic_calibrator_edge_cases() {
        // Test with single point
        let probabilities = array![0.5];
        let y_true = array![1];

        let calibrator = IsotonicCalibrator::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&array![0.5]).unwrap();
        assert_eq!(calibrated.len(), 1);
        assert!(calibrated[0] >= 0.0 && calibrated[0] <= 1.0);

        // Test with identical probabilities
        let probabilities = array![0.5, 0.5, 0.5];
        let y_true = array![0, 1, 1];

        let calibrator = IsotonicCalibrator::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();
        assert_eq!(calibrated.len(), 3);
    }

    #[test]
    fn test_pool_adjacent_violators() {
        let calibrator = IsotonicCalibrator::new();

        // Test case with violations
        let pairs = vec![(0.1, 0.8), (0.3, 0.2), (0.7, 0.9)];
        let (x, y) = calibrator.pool_adjacent_violators(&pairs);

        // Should have removed violations
        assert!(x.len() <= pairs.len());

        // Result should be isotonic
        for i in 1..y.len() {
            assert!(y[i] >= y[i - 1] - Float::EPSILON);
        }
    }
}
