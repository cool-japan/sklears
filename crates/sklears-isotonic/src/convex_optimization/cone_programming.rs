//! Cone Programming Approach for Isotonic Regression
//!
//! This module implements cone programming techniques for isotonic regression,
//! providing different cone constraint types for enhanced flexibility and
//! numerical stability in convex optimization formulations.

use scirs2_core::ndarray::Array1;
use sklears_core::{prelude::SklearsError, types::Float};

/// Cone programming approach for isotonic regression
///
/// This implementation uses cone programming techniques to solve isotonic regression
/// problems with various cone constraint types for enhanced numerical stability
/// and optimization flexibility.
///
/// # Examples
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use sklears_isotonic::convex_optimization::{ConeProgrammingIsotonicRegression, ConeType};
/// use scirs2_core::ndarray::array;
///
/// let mut model = ConeProgrammingIsotonicRegression::new()
///     .increasing(true)
///     .cone_type(ConeType::SecondOrder)
///     .regularization(1e-4);
///
/// let x = array![1.0, 2.0, 3.0, 4.0];
/// let y = array![1.5, 1.0, 2.5, 3.0]; // Non-monotonic
///
/// model.fit(&x, &y)?;
/// let predictions = model.predict(&x)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
/// ConeProgrammingIsotonicRegression
pub struct ConeProgrammingIsotonicRegression {
    /// Whether to enforce increasing or decreasing monotonicity
    increasing: bool,
    /// Type of cone constraint
    cone_type: ConeType,
    /// Regularization parameter
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

/// Types of cone constraints
///
/// Different cone types provide various optimization characteristics:
/// - NonNegative: Simplest constraint, fast computation
/// - SecondOrder: More flexible, handles complex relationships
/// - PositiveSemidefinite: Advanced constraint for matrix optimization
/// - Exponential: Specialized for exponential relationships
#[derive(Debug, Clone)]
pub enum ConeType {
    /// Non-negative orthant
    NonNegative,
    /// Second-order cone
    SecondOrder,
    /// Positive semidefinite cone
    PositiveSemidefinite,
    /// Exponential cone
    Exponential,
}

impl ConeProgrammingIsotonicRegression {
    /// Create a new cone programming isotonic regression model
    pub fn new() -> Self {
        Self {
            increasing: true,
            cone_type: ConeType::NonNegative,
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

    /// Set the cone type
    pub fn cone_type(mut self, cone_type: ConeType) -> Self {
        self.cone_type = cone_type;
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

    /// Fit the cone programming isotonic regression model
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
        data.sort_by(|a, b| a.0.total_cmp(&b.0));

        let sorted_x: Array1<Float> = Array1::from_vec(data.iter().map(|(xi, _)| *xi).collect());
        let sorted_y: Array1<Float> = Array1::from_vec(data.iter().map(|(_, yi)| *yi).collect());

        // Apply cone programming approach
        let fitted_y = self.cone_isotonic_regression(&sorted_x, &sorted_y)?;

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

    /// Cone programming based isotonic regression
    fn cone_isotonic_regression(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        // For now, implement a simplified version using projected gradient descent
        // This can be extended to full cone programming with proper cone projections

        let n = x.len();
        let mut fitted_y = y.clone();

        for _iteration in 0..self.max_iterations {
            let old_y = fitted_y.clone();

            // Gradient step (least squares objective)
            let gradient = &fitted_y - y;

            // Update with gradient step
            for i in 0..n {
                fitted_y[i] -= 0.01 * gradient[i]; // Simple gradient step
            }

            // Project onto cone constraints
            fitted_y = self.project_cone(&fitted_y)?;

            // Project onto isotonic constraints
            fitted_y = self.project_isotonic(&fitted_y)?;

            // Check convergence
            let change = (&fitted_y - &old_y).map(|x| x.abs()).sum();
            if change < self.tolerance {
                break;
            }
        }

        Ok(fitted_y)
    }

    /// Project onto cone constraints
    fn project_cone(&self, y: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        match self.cone_type {
            ConeType::NonNegative => {
                // Project onto non-negative orthant
                Ok(y.map(|&x| x.max(0.0)))
            }
            ConeType::SecondOrder => {
                // Project onto second-order cone (simplified)
                let norm = y.map(|x| x * x).sum().sqrt();
                if norm <= y[0] {
                    Ok(y.clone())
                } else if y[0] <= -norm {
                    Ok(Array1::zeros(y.len()))
                } else {
                    let factor = (1.0 + y[0] / norm) / 2.0;
                    Ok(y * factor)
                }
            }
            ConeType::PositiveSemidefinite => {
                // For now, just ensure non-negativity
                Ok(y.map(|&x| x.max(0.0)))
            }
            ConeType::Exponential => {
                // Project onto exponential cone (simplified)
                Ok(y.map(|&x| if x > 0.0 { x.ln().max(0.0).exp() } else { 0.0 }))
            }
        }
    }

    /// Project onto isotonic constraints using pool-adjacent-violators
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

    /// Get current cone type
    pub fn get_cone_type(&self) -> &ConeType {
        &self.cone_type
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

impl Default for ConeProgrammingIsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for cone programming isotonic regression
///
/// This function provides a simple interface for one-shot cone programming isotonic regression.
///
/// # Arguments
///
/// * `x` - Input features (must be sorted)
/// * `y` - Target values
/// * `increasing` - Whether to enforce increasing monotonicity
/// * `cone_type` - Type of cone constraint
/// * `regularization` - Regularization parameter for numerical stability
///
/// # Returns
///
/// Fitted isotonic values or error
pub fn cone_programming_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    cone_type: ConeType,
    regularization: Float,
) -> Result<Array1<Float>, SklearsError> {
    let mut model = ConeProgrammingIsotonicRegression::new()
        .increasing(increasing)
        .cone_type(cone_type)
        .regularization(regularization);

    model.fit(x, y)?;
    model
        .fitted_values()
        .ok_or_else(|| SklearsError::NotFitted {
            operation: "cone_programming_isotonic_regression".to_string(),
        })
        .map(|v| v.clone())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cone_programming_non_negative() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let mut model = ConeProgrammingIsotonicRegression::new()
            .increasing(true)
            .cone_type(ConeType::NonNegative)
            .regularization(1e-4);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }
    }

    #[test]
    fn test_cone_programming_second_order() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 3.0, 4.0, 2.0, 1.0]; // Non-monotonic

        let mut model = ConeProgrammingIsotonicRegression::new()
            .increasing(false)
            .cone_type(ConeType::SecondOrder)
            .regularization(1e-4);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are decreasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] >= predictions[i + 1]);
        }
    }

    #[test]
    fn test_cone_programming_convenience_function() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.5, 1.0, 2.5, 3.0];

        let result =
            cone_programming_isotonic_regression(&x, &y, true, ConeType::NonNegative, 1e-4);
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.len(), 4);

        // Check monotonicity
        for i in 0..fitted.len() - 1 {
            assert!(fitted[i] <= fitted[i + 1]);
        }
    }

    #[test]
    fn test_cone_programming_getters() {
        let model = ConeProgrammingIsotonicRegression::new()
            .increasing(false)
            .cone_type(ConeType::Exponential)
            .regularization(0.001)
            .tolerance(1e-6)
            .max_iterations(500);

        assert!(!model.is_increasing());
        assert!(matches!(model.get_cone_type(), ConeType::Exponential));
        assert_eq!(model.get_regularization(), 0.001);
        assert_eq!(model.get_tolerance(), 1e-6);
        assert_eq!(model.get_max_iterations(), 500);
    }

    #[test]
    fn test_cone_programming_empty_input() {
        let x = array![];
        let y = array![];

        let mut model = ConeProgrammingIsotonicRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_cone_programming_mismatched_lengths() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];

        let mut model = ConeProgrammingIsotonicRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }
}
