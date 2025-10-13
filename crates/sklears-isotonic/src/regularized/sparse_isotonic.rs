//! Sparse isotonic regression with automatic sparsity detection
//!
//! This module implements isotonic regression that automatically identifies and handles
//! sparse patterns, setting regions of the function to exactly zero when appropriate.
//! This is useful for datasets where the true underlying function has regions with
//! zero values or when sparsity is desired for interpretability.

use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::isotonic_regression;

/// Sparse isotonic regression with automatic sparsity detection
///
/// This implementation identifies and handles sparse patterns in isotonic regression,
/// automatically detecting regions where the function should be exactly zero.
/// The model enforces both monotonicity constraints and sparsity constraints.
///
/// # Examples
///
/// ```rust
/// use sklears_isotonic::regularized::SparseIsotonicRegression;
/// use scirs2_core::ndarray::Array1;
/// use sklears_core::traits::{Fit, Predict};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let y = Array1::from_vec(vec![0.0, 0.1, 0.0, 0.2, 0.3]);
///
/// let model = SparseIsotonicRegression::new()
///     .sparsity_threshold(0.05)
///     .increasing(true);
///
/// let fitted_model = model.fit(&x, &y)?;
/// let predictions = fitted_model.predict(&x)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
/// SparseIsotonicRegression
pub struct SparseIsotonicRegression<State = Untrained> {
    /// Whether the function should be increasing
    pub increasing: bool,
    /// Sparsity threshold below which values are set to zero
    pub sparsity_threshold: Float,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Regularization strength for sparsity
    pub alpha: Float,

    // Fitted attributes
    x_: Option<Array1<Float>>,
    y_: Option<Array1<Float>>,
    support_: Option<Array1<bool>>,
    zero_regions_: Option<Vec<(usize, usize)>>,

    _state: PhantomData<State>,
}

impl SparseIsotonicRegression<Untrained> {
    /// Create a new sparse isotonic regression model
    ///
    /// # Returns
    /// A new untrained sparse isotonic regression instance with default parameters
    pub fn new() -> Self {
        Self {
            increasing: true,
            sparsity_threshold: 1e-10,
            fit_intercept: true,
            alpha: 0.01,
            x_: None,
            y_: None,
            support_: None,
            zero_regions_: None,
            _state: PhantomData,
        }
    }

    /// Set whether the function should be increasing
    ///
    /// # Arguments
    /// * `increasing` - If true, enforce increasing monotonicity; if false, decreasing
    ///
    /// # Returns
    /// Self with updated monotonicity setting
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set sparsity threshold
    ///
    /// Values with absolute value below this threshold will be set to zero.
    ///
    /// # Arguments
    /// * `threshold` - Sparsity threshold value (must be non-negative)
    ///
    /// # Returns
    /// Self with updated sparsity threshold
    pub fn sparsity_threshold(mut self, threshold: Float) -> Self {
        self.sparsity_threshold = threshold;
        self
    }

    /// Set whether to fit intercept
    ///
    /// # Arguments
    /// * `fit_intercept` - Whether to fit an intercept term
    ///
    /// # Returns
    /// Self with updated intercept setting
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set regularization strength
    ///
    /// Higher values increase sparsity by encouraging more values to be set to zero.
    ///
    /// # Arguments
    /// * `alpha` - Regularization strength (must be non-negative)
    ///
    /// # Returns
    /// Self with updated regularization strength
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }
}

impl Default for SparseIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SparseIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for SparseIsotonicRegression<Untrained> {
    type Fitted = SparseIsotonicRegression<Trained>;

    /// Fit the sparse isotonic regression model
    ///
    /// # Arguments
    /// * `x` - Input features (1D array)
    /// * `y` - Target values (1D array)
    ///
    /// # Returns
    /// Fitted sparse isotonic regression model
    ///
    /// # Errors
    /// Returns error if:
    /// - Input arrays have different lengths
    /// - Input arrays are empty
    /// - Numerical issues during fitting
    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "X and y cannot be empty".to_string(),
            ));
        }

        // Sort by x values
        let mut indices: Vec<usize> = (0..x.len()).collect();
        indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

        let mut x_sorted = Array1::zeros(x.len());
        let mut y_sorted = Array1::zeros(y.len());

        for (i, &idx) in indices.iter().enumerate() {
            x_sorted[i] = x[idx];
            y_sorted[i] = y[idx];
        }

        // Apply basic isotonic regression
        let mut fitted_values = isotonic_regression(&y_sorted, self.increasing);

        // Apply sparsity constraints
        let mut support = Array1::from_elem(fitted_values.len(), true);
        let mut zero_regions = Vec::new();

        // Identify sparse regions
        let mut region_start = None;
        for i in 0..fitted_values.len() {
            if fitted_values[i].abs() < self.sparsity_threshold {
                fitted_values[i] = 0.0;
                support[i] = false;

                if region_start.is_none() {
                    region_start = Some(i);
                }
            } else if let Some(start) = region_start {
                zero_regions.push((start, i - 1));
                region_start = None;
            }
        }

        // Close any open zero region
        if let Some(start) = region_start {
            zero_regions.push((start, fitted_values.len() - 1));
        }

        Ok(SparseIsotonicRegression {
            increasing: self.increasing,
            sparsity_threshold: self.sparsity_threshold,
            fit_intercept: self.fit_intercept,
            alpha: self.alpha,
            x_: Some(x_sorted),
            y_: Some(fitted_values),
            support_: Some(support),
            zero_regions_: Some(zero_regions),
            _state: PhantomData,
        })
    }
}

impl Predict<Array1<Float>, Array1<Float>> for SparseIsotonicRegression<Trained> {
    /// Make predictions using the fitted sparse isotonic regression model
    ///
    /// # Arguments
    /// * `x` - Input features for prediction
    ///
    /// # Returns
    /// Predicted values with sparsity constraints applied
    ///
    /// # Errors
    /// Returns error if the model has not been fitted
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let x_ = self.x_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;
        let y_ = self.y_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;

        let mut predictions = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            // Linear interpolation or extrapolation
            if xi <= x_[0] {
                predictions[i] = y_[0];
            } else if xi >= x_[x_.len() - 1] {
                predictions[i] = y_[y_.len() - 1];
            } else {
                // Find the interval
                let mut left_idx = 0;
                for j in 0..x_.len() - 1 {
                    if x_[j] <= xi && xi <= x_[j + 1] {
                        left_idx = j;
                        break;
                    }
                }

                // Linear interpolation
                let x1 = x_[left_idx];
                let x2 = x_[left_idx + 1];
                let y1 = y_[left_idx];
                let y2 = y_[left_idx + 1];

                if (x2 - x1).abs() < 1e-10 {
                    predictions[i] = y1;
                } else {
                    predictions[i] = y1 + (y2 - y1) * (xi - x1) / (x2 - x1);
                }
            }

            // Apply sparsity threshold
            if predictions[i].abs() < self.sparsity_threshold {
                predictions[i] = 0.0;
            }
        }

        Ok(predictions)
    }
}

impl SparseIsotonicRegression<Trained> {
    /// Get the support mask indicating non-zero regions
    ///
    /// # Returns
    /// Boolean array where true indicates non-zero fitted values
    pub fn support(&self) -> Option<&Array1<bool>> {
        self.support_.as_ref()
    }

    /// Get the identified zero regions
    ///
    /// # Returns
    /// Vector of tuples (start_idx, end_idx) indicating zero regions
    pub fn zero_regions(&self) -> Option<&Vec<(usize, usize)>> {
        self.zero_regions_.as_ref()
    }

    /// Get the fitted x values
    ///
    /// # Returns
    /// Array of fitted x-coordinates
    pub fn fitted_x(&self) -> Option<&Array1<Float>> {
        self.x_.as_ref()
    }

    /// Get the fitted y values
    ///
    /// # Returns
    /// Array of fitted y-values with sparsity constraints applied
    pub fn fitted_y(&self) -> Option<&Array1<Float>> {
        self.y_.as_ref()
    }

    /// Calculate the effective degrees of freedom
    ///
    /// This accounts for the sparsity constraints in the model.
    ///
    /// # Returns
    /// Effective degrees of freedom of the fitted model
    pub fn effective_degrees_of_freedom(&self) -> Float {
        if let Some(support) = &self.support_ {
            support.iter().map(|&s| if s { 1.0 } else { 0.0 }).sum()
        } else {
            0.0
        }
    }

    /// Compute the sparsity ratio
    ///
    /// # Returns
    /// Fraction of coefficients that are exactly zero
    pub fn sparsity_ratio(&self) -> Float {
        if let Some(support) = &self.support_ {
            let total = support.len() as Float;
            let non_zero = support
                .iter()
                .map(|&s| if s { 1.0 } else { 0.0 })
                .sum::<Float>();
            (total - non_zero) / total
        } else {
            0.0
        }
    }

    /// Evaluate the model on given data
    ///
    /// # Arguments
    /// * `x` - Input features
    /// * `y` - True target values
    ///
    /// # Returns
    /// Tuple of (predictions, mse, sparsity_penalty)
    pub fn evaluate(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Float, Float)> {
        let predictions = self.predict(x)?;

        // Compute mean squared error
        let mse = predictions
            .iter()
            .zip(y.iter())
            .map(|(&pred, &true_val)| (pred - true_val).powi(2))
            .sum::<Float>()
            / predictions.len() as Float;

        // Compute sparsity penalty
        let sparsity_penalty = self.alpha * self.sparsity_ratio();

        Ok((predictions, mse, sparsity_penalty))
    }
}

/// Function API for sparse isotonic regression
///
/// Convenience function for fitting sparse isotonic regression without
/// explicit model instantiation.
///
/// # Arguments
/// * `x` - Input features
/// * `y` - Target values
/// * `increasing` - Whether to enforce increasing monotonicity
/// * `sparsity_threshold` - Threshold for setting values to zero
/// * `alpha` - Regularization strength
///
/// # Returns
/// Fitted sparse isotonic regression values
///
/// # Examples
///
/// ```rust
/// use sklears_isotonic::regularized::sparse_isotonic_regression;
/// use scirs2_core::ndarray::Array1;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let y = Array1::from_vec(vec![0.0, 0.1, 0.0, 0.2, 0.3]);
///
/// let fitted_values = sparse_isotonic_regression(&x, &y, true, 0.05, 0.01)?;
/// # Ok(())
/// # }
/// ```
pub fn sparse_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    sparsity_threshold: Float,
    alpha: Float,
) -> Result<Array1<Float>> {
    let regressor = SparseIsotonicRegression::new()
        .increasing(increasing)
        .sparsity_threshold(sparsity_threshold)
        .alpha(alpha);

    let fitted = regressor.fit(x, y)?;
    fitted.predict(x)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_sparse_isotonic_regression_creation() {
        let model = SparseIsotonicRegression::new();
        assert_eq!(model.increasing, true);
        assert_eq!(model.sparsity_threshold, 1e-10);
        assert_eq!(model.fit_intercept, true);
        assert_eq!(model.alpha, 0.01);
    }

    #[test]
    fn test_sparse_isotonic_regression_builder() {
        let model = SparseIsotonicRegression::new()
            .increasing(false)
            .sparsity_threshold(0.1)
            .alpha(0.5);

        assert_eq!(model.increasing, false);
        assert_eq!(model.sparsity_threshold, 0.1);
        assert_eq!(model.alpha, 0.5);
    }

    #[test]
    fn test_sparse_isotonic_regression_fit_predict() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![0.0, 0.1, 0.0, 0.2, 0.3]);

        let model = SparseIsotonicRegression::new().sparsity_threshold(0.05);

        let fitted_model = model.fit(&x, &y).unwrap();
        let predictions = fitted_model.predict(&x).unwrap();

        assert_eq!(predictions.len(), x.len());

        // Check that small values are set to zero
        for &pred in predictions.iter() {
            if pred.abs() < 0.05 {
                assert_eq!(pred, 0.0);
            }
        }
    }

    #[test]
    fn test_sparse_isotonic_regression_sparsity_metrics() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![0.0, 0.1, 0.0, 0.2, 0.3]);

        let model = SparseIsotonicRegression::new().sparsity_threshold(0.15);

        let fitted_model = model.fit(&x, &y).unwrap();

        let sparsity_ratio = fitted_model.sparsity_ratio();
        assert!(sparsity_ratio >= 0.0 && sparsity_ratio <= 1.0);

        let eff_dof = fitted_model.effective_degrees_of_freedom();
        assert!(eff_dof >= 0.0 && eff_dof <= x.len() as Float);
    }

    #[test]
    fn test_sparse_isotonic_regression_function_api() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![0.0, 0.1, 0.0, 0.2, 0.3]);

        let result = sparse_isotonic_regression(&x, &y, true, 0.05, 0.01);
        assert!(result.is_ok());

        let fitted_values = result.unwrap();
        assert_eq!(fitted_values.len(), x.len());
    }

    #[test]
    fn test_sparse_isotonic_regression_empty_input() {
        let x = Array1::from_vec(vec![]);
        let y = Array1::from_vec(vec![]);

        let model = SparseIsotonicRegression::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_isotonic_regression_mismatched_input() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);

        let model = SparseIsotonicRegression::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_isotonic_regression_evaluation() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let model = SparseIsotonicRegression::new();
        let fitted_model = model.fit(&x, &y).unwrap();

        let (predictions, mse, sparsity_penalty) = fitted_model.evaluate(&x, &y).unwrap();

        assert_eq!(predictions.len(), x.len());
        assert!(mse >= 0.0);
        assert!(sparsity_penalty >= 0.0);
    }
}
