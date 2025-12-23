//! Proximal Gradient Method for Regularized Isotonic Regression
//!
//! This module implements proximal gradient methods for regularized isotonic regression,
//! providing efficient optimization for various regularization types including L1, L2,
//! Elastic Net, and Total Variation regularization.

use scirs2_core::ndarray::Array1;
use sklears_core::{prelude::SklearsError, types::Float};

/// Proximal gradient method for regularized isotonic regression
///
/// This implementation uses proximal gradient methods to solve regularized isotonic
/// regression problems efficiently. The algorithm alternates between gradient steps
/// on the smooth part of the objective and proximal steps for the non-smooth
/// regularization terms.
///
/// # Mathematical Framework
///
/// The proximal gradient algorithm solves:
/// ```text
/// minimize   f(x) + λ g(x)
/// subject to x ∈ isotonic constraint set
/// ```
///
/// where f(x) is the smooth loss function, g(x) is the regularization term,
/// and λ is the regularization parameter.
#[derive(Debug, Clone)]
pub struct ProximalGradientIsotonicRegression {
    increasing: bool,
    regularization: Float,
    step_size: Float,
    max_iterations: usize,
    tolerance: Float,
    regularization_type: RegularizationType,
    fitted_values: Option<Array1<Float>>,
    fitted_x: Option<Array1<Float>>,
}

/// Type of regularization for proximal gradient
///
/// Different regularization types provide various characteristics:
/// - L1: Promotes sparsity, non-differentiable
/// - L2: Promotes smoothness, differentiable
/// - ElasticNet: Combination of L1 and L2
/// - TotalVariation: Promotes piecewise constant solutions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegularizationType {
    /// L1 regularization (Lasso) - promotes sparsity
    L1,
    /// L2 regularization (Ridge) - promotes smoothness
    L2,
    /// Elastic Net (L1 + L2) - balanced sparsity and smoothness
    ElasticNet { l1_ratio: Float },
    /// Total variation regularization - promotes piecewise constant solutions
    TotalVariation,
}

impl ProximalGradientIsotonicRegression {
    /// Create a new proximal gradient isotonic regression model
    pub fn new() -> Self {
        Self {
            increasing: true,
            regularization: 0.01,
            step_size: 0.1,
            max_iterations: 1000,
            tolerance: 1e-6,
            regularization_type: RegularizationType::L1,
            fitted_values: None,
            fitted_x: None,
        }
    }

    /// Set whether the function should be increasing or decreasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set regularization parameter (λ)
    ///
    /// Controls the strength of regularization. Larger values
    /// increase regularization strength.
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set step size for gradient descent
    ///
    /// The step size should be chosen to ensure convergence.
    /// Too large values may cause divergence, too small values
    /// may lead to slow convergence.
    pub fn step_size(mut self, step_size: Float) -> Self {
        self.step_size = step_size;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set regularization type
    pub fn regularization_type(mut self, reg_type: RegularizationType) -> Self {
        self.regularization_type = reg_type;
        self
    }

    /// Fit the proximal gradient isotonic regression model
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

        let _n = x.len();

        // Sort data by x values
        let mut data: Vec<(Float, Float)> =
            x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi, yi)).collect();
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let sorted_x: Array1<Float> = Array1::from_vec(data.iter().map(|(xi, _)| *xi).collect());
        let sorted_y: Array1<Float> = Array1::from_vec(data.iter().map(|(_, yi)| *yi).collect());

        // Initialize with least squares solution
        let mut beta = sorted_y.clone();

        for _iter in 0..self.max_iterations {
            let prev_beta = beta.clone();

            // Gradient step: beta = beta - step_size * gradient
            let gradient = self.compute_gradient(&beta, &sorted_y);
            let mut temp_beta = &beta - &(self.step_size * &gradient);

            // Proximal step: apply regularization and constraints
            temp_beta = self.proximal_operator(&temp_beta);

            // Project onto isotonic constraint set
            beta = self.isotonic_projection(&temp_beta);

            // Check convergence
            let diff = (&beta - &prev_beta).mapv(|x| x.abs()).sum();
            if diff < self.tolerance {
                break;
            }
        }

        self.fitted_values = Some(beta);
        self.fitted_x = Some(sorted_x);
        Ok(())
    }

    /// Compute gradient of the loss function
    ///
    /// For squared loss, the gradient is: ∇f(β) = 2(β - y)
    fn compute_gradient(&self, beta: &Array1<Float>, y: &Array1<Float>) -> Array1<Float> {
        // Gradient of squared loss: 2 * (beta - y)
        2.0 * (beta - y)
    }

    /// Apply proximal operator for regularization
    ///
    /// The proximal operator depends on the regularization type:
    /// - L1: Soft thresholding
    /// - L2: Scaling
    /// - Elastic Net: Combination of both
    /// - Total Variation: Iterative denoising
    fn proximal_operator(&self, beta: &Array1<Float>) -> Array1<Float> {
        match self.regularization_type {
            RegularizationType::L1 => {
                self.soft_threshold(beta, self.step_size * self.regularization)
            }
            RegularizationType::L2 => beta / (1.0 + self.step_size * self.regularization),
            RegularizationType::ElasticNet { l1_ratio } => {
                let l1_penalty = self.regularization * l1_ratio;
                let l2_penalty = self.regularization * (1.0 - l1_ratio);
                let l2_result = beta / (1.0 + self.step_size * l2_penalty);
                self.soft_threshold(&l2_result, self.step_size * l1_penalty)
            }
            RegularizationType::TotalVariation => self.total_variation_proximal(beta),
        }
    }

    /// Soft thresholding operator for L1 regularization
    ///
    /// The soft thresholding operator is defined as:
    /// soft_threshold(x, λ) = sign(x) * max(|x| - λ, 0)
    fn soft_threshold(&self, beta: &Array1<Float>, threshold: Float) -> Array1<Float> {
        beta.mapv(|x| {
            if x > threshold {
                x - threshold
            } else if x < -threshold {
                x + threshold
            } else {
                0.0
            }
        })
    }

    /// Total variation proximal operator
    ///
    /// Applies total variation denoising to promote piecewise constant solutions.
    /// Uses iterative soft thresholding on finite differences.
    fn total_variation_proximal(&self, beta: &Array1<Float>) -> Array1<Float> {
        let n = beta.len();
        if n <= 1 {
            return beta.clone();
        }

        let mut result = beta.clone();
        let lambda = self.step_size * self.regularization;

        // Simple TV denoising using iterative soft thresholding
        for _ in 0..10 {
            let mut diff = Array1::zeros(n - 1);
            for i in 0..n - 1 {
                diff[i] = result[i + 1] - result[i];
            }

            let soft_diff = self.soft_threshold(&diff, lambda);

            // Reconstruct from differences
            result[0] = beta[0];
            for i in 1..n {
                result[i] = result[i - 1] + soft_diff[i - 1];
            }
        }

        result
    }

    /// Project onto isotonic constraint set using Pool Adjacent Violators
    ///
    /// Enforces monotonicity constraints while preserving the optimization structure.
    fn isotonic_projection(&self, beta: &Array1<Float>) -> Array1<Float> {
        let n = beta.len();
        let mut result = beta.clone();

        if self.increasing {
            // Increasing PAVA
            for i in 1..n {
                if result[i] < result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i] + result[i - 1];
                    let mut count = 2;

                    // Find the range to pool
                    while j > 1 && sum / (count as Float) < result[j - 2] {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Update the pooled values
                    let avg = sum / (count as Float);
                    for k in (j - 1)..=i {
                        result[k] = avg;
                    }
                }
            }
        } else {
            // Decreasing PAVA
            for i in 1..n {
                if result[i] > result[i - 1] {
                    // Pool adjacent violators
                    let mut j = i;
                    let mut sum = result[i] + result[i - 1];
                    let mut count = 2;

                    // Find the range to pool
                    while j > 1 && sum / (count as Float) > result[j - 2] {
                        j -= 1;
                        sum += result[j - 1];
                        count += 1;
                    }

                    // Update the pooled values
                    let avg = sum / (count as Float);
                    for k in (j - 1)..=i {
                        result[k] = avg;
                    }
                }
            }
        }

        result
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
            predictions[i] = self.interpolate(xi, fitted_x, fitted_values);
        }

        Ok(predictions)
    }

    /// Linear interpolation for prediction
    fn interpolate(
        &self,
        x: Float,
        fitted_x: &Array1<Float>,
        fitted_values: &Array1<Float>,
    ) -> Float {
        if fitted_x.is_empty() {
            return Float::NAN;
        }

        let n = fitted_x.len().min(fitted_values.len());

        if n == 1 {
            return fitted_values[0];
        }

        // Handle boundary cases
        if x <= fitted_x[0] {
            return fitted_values[0];
        }
        if x >= fitted_x[n - 1] {
            return fitted_values[n - 1];
        }

        // Find interpolation interval
        for i in 0..n - 1 {
            if x >= fitted_x[i] && x <= fitted_x[i + 1] {
                if fitted_x[i + 1] == fitted_x[i] {
                    return fitted_values[i];
                }
                let t = (x - fitted_x[i]) / (fitted_x[i + 1] - fitted_x[i]);
                return fitted_values[i] + t * (fitted_values[i + 1] - fitted_values[i]);
            }
        }

        fitted_values[n - 1]
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

    /// Get current step size
    pub fn get_step_size(&self) -> Float {
        self.step_size
    }

    /// Get current tolerance
    pub fn get_tolerance(&self) -> Float {
        self.tolerance
    }

    /// Get maximum iterations setting
    pub fn get_max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Get current regularization type
    pub fn get_regularization_type(&self) -> RegularizationType {
        self.regularization_type
    }

    /// Check if model enforces increasing monotonicity
    pub fn is_increasing(&self) -> bool {
        self.increasing
    }
}

impl Default for ProximalGradientIsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for proximal gradient isotonic regression
///
/// This function provides a simple interface for one-shot proximal gradient isotonic regression.
///
/// # Arguments
///
/// * `x` - Input features (must be sorted)
/// * `y` - Target values
/// * `increasing` - Whether to enforce increasing monotonicity
/// * `regularization` - Regularization parameter
/// * `reg_type` - Type of regularization
/// * `step_size` - Optional step size (uses default if None)
///
/// # Returns
///
/// Fitted isotonic values or error
pub fn proximal_gradient_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    regularization: Float,
    reg_type: RegularizationType,
    step_size: Option<Float>,
) -> Result<Array1<Float>, SklearsError> {
    let mut model = ProximalGradientIsotonicRegression::new()
        .increasing(increasing)
        .regularization(regularization)
        .regularization_type(reg_type);

    if let Some(step) = step_size {
        model = model.step_size(step);
    }

    model.fit(x, y)?;
    Ok(model.fitted_values().unwrap().clone())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_proximal_gradient_l1() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let mut model = ProximalGradientIsotonicRegression::new()
            .increasing(true)
            .regularization(0.1)
            .regularization_type(RegularizationType::L1);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }
    }

    #[test]
    fn test_proximal_gradient_l2() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 3.0, 4.0, 2.0, 1.0]; // Non-monotonic

        let mut model = ProximalGradientIsotonicRegression::new()
            .increasing(false)
            .regularization(0.05)
            .regularization_type(RegularizationType::L2);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are decreasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] >= predictions[i + 1]);
        }
    }

    #[test]
    fn test_proximal_gradient_elastic_net() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 3.0, 4.0, 2.0, 1.0]; // Non-monotonic

        let mut model = ProximalGradientIsotonicRegression::new()
            .increasing(false)
            .regularization(0.05)
            .regularization_type(RegularizationType::ElasticNet { l1_ratio: 0.5 });

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are decreasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] >= predictions[i + 1]);
        }
    }

    #[test]
    fn test_proximal_gradient_total_variation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0]; // Non-monotonic

        let mut model = ProximalGradientIsotonicRegression::new()
            .increasing(true)
            .regularization(0.02)
            .regularization_type(RegularizationType::TotalVariation);

        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x).unwrap();

        // Check that predictions are increasing
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i] <= predictions[i + 1]);
        }
    }

    #[test]
    fn test_proximal_gradient_convenience_function() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.5, 1.0, 2.5, 3.0];

        let result = proximal_gradient_isotonic_regression(
            &x,
            &y,
            true,
            0.1,
            RegularizationType::L1,
            Some(0.05),
        );
        assert!(result.is_ok());

        let fitted = result.unwrap();
        assert_eq!(fitted.len(), 4);

        // Check monotonicity
        for i in 0..fitted.len() - 1 {
            assert!(fitted[i] <= fitted[i + 1]);
        }
    }

    #[test]
    fn test_proximal_gradient_getters() {
        let model = ProximalGradientIsotonicRegression::new()
            .increasing(false)
            .regularization(0.05)
            .step_size(0.02)
            .tolerance(1e-7)
            .max_iterations(500)
            .regularization_type(RegularizationType::ElasticNet { l1_ratio: 0.3 });

        assert!(!model.is_increasing());
        assert_eq!(model.get_regularization(), 0.05);
        assert_eq!(model.get_step_size(), 0.02);
        assert_eq!(model.get_tolerance(), 1e-7);
        assert_eq!(model.get_max_iterations(), 500);
        assert!(matches!(
            model.get_regularization_type(),
            RegularizationType::ElasticNet { l1_ratio } if l1_ratio == 0.3
        ));
    }

    #[test]
    fn test_proximal_gradient_empty_input() {
        let x = array![];
        let y = array![];

        let mut model = ProximalGradientIsotonicRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_proximal_gradient_mismatched_lengths() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];

        let mut model = ProximalGradientIsotonicRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_soft_threshold() {
        let model = ProximalGradientIsotonicRegression::new();
        let x = array![2.0, -1.5, 0.5, -0.3, 3.0];
        let threshold = 1.0;

        let result = model.soft_threshold(&x, threshold);

        assert_eq!(result[0], 1.0); // 2.0 - 1.0
        assert_eq!(result[1], -0.5); // -1.5 + 1.0
        assert_eq!(result[2], 0.0); // |0.5| < 1.0
        assert_eq!(result[3], 0.0); // |-0.3| < 1.0
        assert_eq!(result[4], 2.0); // 3.0 - 1.0
    }

    #[test]
    fn test_regularization_types() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let reg_types = vec![
            RegularizationType::L1,
            RegularizationType::L2,
            RegularizationType::ElasticNet { l1_ratio: 0.5 },
            RegularizationType::TotalVariation,
        ];

        for reg_type in reg_types {
            let mut model = ProximalGradientIsotonicRegression::new().regularization_type(reg_type);
            assert!(model.fit(&x, &y).is_ok());
            assert!(model.predict(&x).is_ok());
        }
    }
}
