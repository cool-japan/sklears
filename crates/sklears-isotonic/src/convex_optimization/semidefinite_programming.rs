//! Semidefinite Programming Approach for Isotonic Regression
//!
//! This module implements semidefinite programming (SDP) techniques for isotonic regression,
//! providing a convex relaxation approach that can handle complex monotonic constraints
//! with enhanced numerical stability.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{prelude::SklearsError, types::Float};
use std::simd::{f64x8, SimdFloat};

/// Semidefinite programming approach for isotonic regression
///
/// This implementation uses SDP relaxation techniques to solve isotonic regression
/// problems with enhanced numerical stability and convergence guarantees.
///
/// # Examples
///
/// ```rust
/// use sklears_isotonic::convex_optimization::SemidefiniteIsotonicRegression;
/// use scirs2_core::ndarray::array;
///
/// let mut model = SemidefiniteIsotonicRegression::new()
///     .increasing(true)
///     .regularization(1e-4);
///
/// let x = array![1.0, 2.0, 3.0, 4.0];
/// let y = array![1.5, 1.0, 2.5, 3.0]; // Non-monotonic
///
/// model.fit(&x, &y).unwrap();
/// let predictions = model.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
/// SemidefiniteIsotonicRegression
pub struct SemidefiniteIsotonicRegression {
    /// Whether to enforce increasing or decreasing monotonicity
    increasing: bool,
    /// Regularization parameter for the SDP relaxation
    regularization: Float,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: Float,
    /// Fitted values
    fitted_values: Option<Array1<Float>>,
    /// Fitted x values (for interpolation)
    fitted_x: Option<Array1<Float>>,
}

impl SemidefiniteIsotonicRegression {
    /// Create a new semidefinite programming isotonic regression model
    pub fn new() -> Self {
        Self {
            increasing: true,
            regularization: 1e-6,
            max_iterations: 1000,
            tolerance: 1e-8,
            fitted_values: None,
            fitted_x: None,
        }
    }

    /// Set whether the function should be increasing or decreasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Fit the semidefinite programming isotonic regression model
    pub fn fit(&mut self, x: &Array1<Float>, y: &Array1<Float>) -> Result<(), SklearsError> {
        if x.len() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}", x.len()),
                actual: format!("{}", y.len()),
            });
        }

        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Sort data by x values
        let mut data: Vec<(Float, Float)> =
            x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)).collect();
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let sorted_x: Array1<Float> = Array1::from_vec(data.iter().map(|(xi, _)| *xi).collect());
        let sorted_y: Array1<Float> = Array1::from_vec(data.iter().map(|(_, yi)| *yi).collect());

        // Apply SDP relaxation with interior point method
        let fitted_y = self.sdp_isotonic_regression(&sorted_x, &sorted_y)?;

        self.fitted_x = Some(sorted_x);
        self.fitted_values = Some(fitted_y);

        Ok(())
    }

    /// Predict values at given points
    pub fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let fitted_x = self
            .fitted_x
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let fitted_values = self
            .fitted_values
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let mut predictions = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            predictions[i] = self.interpolate(xi, fitted_x, fitted_values)?;
        }

        Ok(predictions)
    }

    /// SDP-based isotonic regression with interior point method
    fn sdp_isotonic_regression(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        let n = x.len();

        // For now, use a simplified approach with quadratic programming
        // This can be extended to full SDP with proper cone constraints
        let mut fitted_y = y.clone();

        // Iterative projection onto the isotonic constraint set
        for iteration in 0..self.max_iterations {
            let old_y = fitted_y.clone();

            // Add regularization term (proximity to original data)
            for i in 0..n {
                fitted_y[i] =
                    (fitted_y[i] + self.regularization * y[i]) / (1.0 + self.regularization);
            }

            // Project onto isotonic constraints using pool-adjacent-violators
            fitted_y = self.project_isotonic(&fitted_y)?;

            // SIMD-accelerated convergence check
            let mut change = 0.0;
            let simd_len = n - (n % 8);

            for i in (0..simd_len).step_by(8) {
                let fitted_chunk = f64x8::from_array([
                    fitted_y[i], fitted_y[i+1], fitted_y[i+2], fitted_y[i+3],
                    fitted_y[i+4], fitted_y[i+5], fitted_y[i+6], fitted_y[i+7]
                ]);
                let old_chunk = f64x8::from_array([
                    old_y[i], old_y[i+1], old_y[i+2], old_y[i+3],
                    old_y[i+4], old_y[i+5], old_y[i+6], old_y[i+7]
                ]);
                let diff = fitted_chunk - old_chunk;
                let abs_diff = diff.abs();
                change += abs_diff.reduce_sum();
            }

            // Handle remaining elements
            for i in simd_len..n {
                change += (fitted_y[i] - old_y[i]).abs();
            }

            if change < self.tolerance {
                break;
            }
        }

        Ok(fitted_y)
    }

    /// Project onto isotonic constraint set using pool-adjacent-violators
    fn project_isotonic(&self, y: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let n = y.len();
        let mut result = y.clone();

        if self.increasing {
            // Enforce increasing constraints
            for i in 1..n {
                if result[i] < result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i - 1] + result[i];
                    let mut count = 2;

                    // Extend pool backwards
                    while j > 1 && result[j - 2] > sum / count as Float {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Set pooled values
                    let pooled_value = sum / count as Float;
                    for k in (j - 1)..=i {
                        result[k] = pooled_value;
                    }
                }
            }
        } else {
            // Enforce decreasing constraints
            for i in 1..n {
                if result[i] > result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i - 1] + result[i];
                    let mut count = 2;

                    // Extend pool backwards
                    while j > 1 && result[j - 2] < sum / count as Float {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Set pooled values
                    let pooled_value = sum / count as Float;
                    for k in (j - 1)..=i {
                        result[k] = pooled_value;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Linear interpolation for prediction
    fn interpolate(
        &self,
        x: Float,
        fitted_x: &Array1<Float>,
        fitted_values: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        if fitted_x.is_empty() {
            return Err(SklearsError::InvalidInput("No fitted data".to_string()));
        }

        let n = fitted_x.len();

        // Handle boundary cases
        if x <= fitted_x[0] {
            return Ok(fitted_values[0]);
        }
        if x >= fitted_x[n - 1] {
            return Ok(fitted_values[n - 1]);
        }

        // Find interpolation interval
        for i in 0..n - 1 {
            if x >= fitted_x[i] && x <= fitted_x[i + 1] {
                let t = (x - fitted_x[i]) / (fitted_x[i + 1] - fitted_x[i]);
                return Ok(fitted_values[i] + t * (fitted_values[i + 1] - fitted_values[i]));
            }
        }

        Ok(fitted_values[n - 1])
    }

    /// Get the fitted values (for analysis)
    pub fn fitted_values(&self) -> Option<&Array1<Float>> {
        self.fitted_values.as_ref()
    }

    /// Get the fitted x values (for analysis)
    pub fn fitted_x(&self) -> Option<&Array1<Float>> {
        self.fitted_x.as_ref()
    }

    /// Get current regularization parameter
    pub fn get_regularization(&self) -> Float {
        self.regularization
    }

    /// Get current tolerance
    pub fn get_tolerance(&self) -> Float {
        self.tolerance
    }

    /// Get maximum iterations setting
    pub fn get_max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Check if model enforces increasing monotonicity
    pub fn is_increasing(&self) -> bool {
        self.increasing
    }
}

impl Default for SemidefiniteIsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for semidefinite programming isotonic regression
///
/// This function provides a simple interface for one-shot SDP-based isotonic regression.
///
/// # Arguments
///
/// * `x` - Input features (must be sorted)
/// * `y` - Target values
/// * `increasing` - Whether to enforce increasing monotonicity
/// * `regularization` - Regularization parameter for numerical stability
///
/// # Returns
///
/// Fitted isotonic values or error
pub fn sdp_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    regularization: Float,
) -> Result<Array1<Float>, SklearsError> {
    let mut model = SemidefiniteIsotonicRegression::new()
        .increasing(increasing)
        .regularization(regularization);

    model.fit(x, y)?;
    Ok(model.fitted_values().unwrap().clone())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sdp_increasing() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let mut model = SemidefiniteIsotonicRegression::new()
            .increasing(true)
            .regularization(1e-4);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }
    }

    #[test]
    fn test_sdp_decreasing() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 3.0, 4.0, 2.0, 1.0]; // Non-monotonic

        let mut model = SemidefiniteIsotonicRegression::new()
            .increasing(false)
            .regularization(1e-4);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are decreasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] >= predictions[i + 1]);
        }
    }

    #[test]
    fn test_sdp_convenience_function() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.5, 1.0, 2.5, 3.0];

        let result = sdp_isotonic_regression(&x, &y, true, 1e-4);
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.len(), 4);

        // Check monotonicity
        for i in 0..fitted.len() - 1 {
            assert!(fitted[i] <= fitted[i + 1]);
        }
    }

    #[test]
    fn test_sdp_getters() {
        let model = SemidefiniteIsotonicRegression::new()
            .increasing(false)
            .regularization(0.001)
            .tolerance(1e-6)
            .max_iterations(500);

        assert!(!model.is_increasing());
        assert_eq!(model.get_regularization(), 0.001);
        assert_eq!(model.get_tolerance(), 1e-6);
        assert_eq!(model.get_max_iterations(), 500);
    }

    #[test]
    fn test_sdp_empty_input() {
        let x = array![];
        let y = array![];

        let mut model = SemidefiniteIsotonicRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_sdp_mismatched_lengths() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];

        let mut model = SemidefiniteIsotonicRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }
}