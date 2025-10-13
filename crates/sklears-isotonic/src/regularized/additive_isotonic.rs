//! Additive isotonic regression for multi-dimensional data
//!
//! This module implements isotonic regression for multivariate input using an additive model:
//! f(x) = f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ) + intercept
//! where each fᵢ is a univariate isotonic function. This approach allows for
//! interpretable multi-dimensional isotonic regression with controlled complexity.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use super::simd_operations::{simd_l1_penalty, simd_mean};
use crate::{isotonic_regression, LossFunction, MonotonicityConstraint};

/// Additive isotonic regression for multi-dimensional data
///
/// Implements isotonic regression for multivariate input using an additive model.
/// Each feature contributes independently through its own isotonic function,
/// making the model interpretable while handling multi-dimensional constraints.
///
/// The model learns: f(x) = f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ) + intercept
///
/// # Examples
///
/// ```rust
/// use sklears_isotonic::regularized::AdditiveIsotonicRegression;
/// use scirs2_core::ndarray::{Array1, Array2};
/// use sklears_core::traits::{Fit, Predict};
/// use sklears_isotonic::MonotonicityConstraint;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0])?;
/// let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
///
/// let model = AdditiveIsotonicRegression::new(2)
///     .constraints(vec![
///         MonotonicityConstraint::Global { increasing: true },
///         MonotonicityConstraint::Global { increasing: true }
///     ]);
///
/// let fitted_model = model.fit(&x, &y)?;
/// let predictions = fitted_model.predict(&x)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
/// AdditiveIsotonicRegression
pub struct AdditiveIsotonicRegression<State = Untrained> {
    /// Number of features
    pub n_features: usize,
    /// Monotonicity constraints for each feature
    pub constraints: Vec<MonotonicityConstraint>,
    /// Loss function
    pub loss: LossFunction,
    /// Regularization strength
    pub alpha: Float,
    /// Whether to fit intercept
    pub fit_intercept: bool,

    // Fitted attributes
    feature_functions_: Option<Vec<Array1<Float>>>,
    feature_grids_: Option<Vec<Array1<Float>>>,
    intercept_: Option<Float>,

    _state: PhantomData<State>,
}

impl AdditiveIsotonicRegression<Untrained> {
    /// Create a new additive isotonic regression model
    ///
    /// # Arguments
    /// * `n_features` - Number of input features
    ///
    /// # Returns
    /// A new untrained additive isotonic regression instance
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            constraints: vec![MonotonicityConstraint::Global { increasing: true }; n_features],
            loss: LossFunction::SquaredLoss,
            alpha: 0.0,
            fit_intercept: true,
            feature_functions_: None,
            feature_grids_: None,
            intercept_: None,
            _state: PhantomData,
        }
    }

    /// Set constraints for each feature
    ///
    /// # Arguments
    /// * `constraints` - Vector of monotonicity constraints, one per feature
    ///
    /// # Returns
    /// Self with updated constraints
    pub fn constraints(mut self, constraints: Vec<MonotonicityConstraint>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Set loss function
    ///
    /// # Arguments
    /// * `loss` - Loss function to use for fitting
    ///
    /// # Returns
    /// Self with updated loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set regularization strength
    ///
    /// # Arguments
    /// * `alpha` - Regularization parameter (≥ 0)
    ///
    /// # Returns
    /// Self with updated regularization strength
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set whether to fit intercept
    ///
    /// # Arguments
    /// * `fit_intercept` - Whether to include an intercept term
    ///
    /// # Returns
    /// Self with updated intercept setting
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl Default for AdditiveIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Estimator for AdditiveIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for AdditiveIsotonicRegression<Untrained> {
    type Fitted = AdditiveIsotonicRegression<Trained>;

    /// Fit the additive isotonic regression model
    ///
    /// Uses a backfitting algorithm to iteratively fit each feature's isotonic function
    /// while holding others fixed. Converges to a local optimum of the additive model.
    ///
    /// # Arguments
    /// * `x` - Input features (n_samples × n_features)
    /// * `y` - Target values (n_samples,)
    ///
    /// # Returns
    /// Fitted additive isotonic regression model
    ///
    /// # Errors
    /// Returns error if:
    /// - Number of features doesn't match expected
    /// - Sample sizes don't match between x and y
    /// - Convergence issues during fitting
    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_features != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features, n_features
            )));
        }

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Initialize feature functions and grids
        let mut feature_functions = Vec::with_capacity(n_features);
        let mut feature_grids = Vec::with_capacity(n_features);

        // Compute intercept if needed
        let intercept = if self.fit_intercept {
            y.mean().unwrap_or(0.0)
        } else {
            0.0
        };

        let y_centered = if self.fit_intercept {
            y - intercept
        } else {
            y.clone()
        };

        // Fit each feature independently using backfitting algorithm
        let max_iter = 100;
        let tolerance = 1e-6;
        let mut residuals = y_centered.clone();

        // Initialize feature functions to zero
        for j in 0..n_features {
            let feature_col = x.column(j);
            let unique_values = self.get_unique_sorted_values(&feature_col);
            feature_grids.push(unique_values.clone());
            feature_functions.push(Array1::zeros(unique_values.len()));
        }

        // Backfitting iterations
        for _iter in 0..max_iter {
            let mut max_change: Float = 0.0;

            for j in 0..n_features {
                let feature_col = x.column(j);

                // Add back the contribution of feature j
                let old_contrib = self.evaluate_feature_function(
                    &feature_functions[j],
                    &feature_grids[j],
                    &feature_col,
                );
                for i in 0..n_samples {
                    residuals[i] += old_contrib[i];
                }

                // Fit isotonic regression for feature j
                let new_function = self.fit_single_feature(&feature_col, &residuals, j)?;
                let new_contrib =
                    self.evaluate_feature_function(&new_function, &feature_grids[j], &feature_col);

                // Subtract the new contribution
                for i in 0..n_samples {
                    residuals[i] -= new_contrib[i];
                }

                // Check convergence
                let diff = &new_function - &feature_functions[j];
                let diff_vec: Vec<f64> = diff.iter().cloned().collect();
                let change = simd_l1_penalty(&diff_vec);
                max_change = max_change.max(change);

                feature_functions[j] = new_function;
            }

            if max_change < tolerance {
                break;
            }
        }

        Ok(AdditiveIsotonicRegression {
            n_features: self.n_features,
            constraints: self.constraints,
            loss: self.loss,
            alpha: self.alpha,
            fit_intercept: self.fit_intercept,
            feature_functions_: Some(feature_functions),
            feature_grids_: Some(feature_grids),
            intercept_: Some(intercept),
            _state: PhantomData,
        })
    }
}

impl AdditiveIsotonicRegression<Untrained> {
    /// Get unique sorted values from a feature column
    fn get_unique_sorted_values(&self, feature: &ArrayView1<Float>) -> Array1<Float> {
        let mut values: Vec<Float> = feature.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        Array1::from(values)
    }

    /// Fit isotonic regression for a single feature
    fn fit_single_feature(
        &self,
        feature: &ArrayView1<Float>,
        targets: &Array1<Float>,
        feature_idx: usize,
    ) -> Result<Array1<Float>> {
        // Map feature values to unique grid points and compute target means
        let grid = self.get_unique_sorted_values(feature);
        let mut grid_targets = vec![Vec::new(); grid.len()];

        for (i, &feat_val) in feature.iter().enumerate() {
            // Find closest grid point
            let mut closest_idx = 0;
            let mut min_diff = Float::INFINITY;
            for (j, &grid_val) in grid.iter().enumerate() {
                let diff = (feat_val - grid_val).abs();
                if diff < min_diff {
                    min_diff = diff;
                    closest_idx = j;
                }
            }
            grid_targets[closest_idx].push(targets[i]);
        }

        // Compute means for each grid point
        let mut grid_means = Array1::zeros(grid.len());
        for (i, targets) in grid_targets.iter().enumerate() {
            if !targets.is_empty() {
                grid_means[i] = simd_mean(targets);
            }
        }

        // Apply isotonic constraint
        let isotonic_result = match &self.constraints[feature_idx] {
            MonotonicityConstraint::Global { increasing } => {
                isotonic_regression(&grid_means, *increasing)
            }
            _ => {
                return Err(SklearsError::NotImplemented(
                    "Only global constraints supported for additive isotonic regression"
                        .to_string(),
                ))
            }
        };

        Ok(isotonic_result)
    }

    /// Evaluate feature function on given feature values
    fn evaluate_feature_function(
        &self,
        function: &Array1<Float>,
        grid: &Array1<Float>,
        feature: &ArrayView1<Float>,
    ) -> Array1<Float> {
        let mut result = Array1::zeros(feature.len());

        for (i, &feat_val) in feature.iter().enumerate() {
            // Find closest grid point and interpolate
            if grid.len() == 1 {
                result[i] = function[0];
                continue;
            }

            // Linear interpolation
            let mut left_idx = 0;
            for j in 0..grid.len() - 1 {
                if feat_val >= grid[j] && feat_val <= grid[j + 1] {
                    left_idx = j;
                    break;
                }
            }

            if left_idx == grid.len() - 1 {
                result[i] = function[left_idx];
            } else {
                let t = (feat_val - grid[left_idx]) / (grid[left_idx + 1] - grid[left_idx]);
                result[i] = function[left_idx] * (1.0 - t) + function[left_idx + 1] * t;
            }
        }

        result
    }
}

impl Predict<Array2<Float>, Array1<Float>> for AdditiveIsotonicRegression<Trained> {
    /// Make predictions using the fitted additive isotonic regression model
    ///
    /// # Arguments
    /// * `x` - Input features for prediction (n_samples × n_features)
    ///
    /// # Returns
    /// Predicted values (n_samples,)
    ///
    /// # Errors
    /// Returns error if:
    /// - Model has not been fitted
    /// - Number of features doesn't match training data
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_features != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features, n_features
            )));
        }

        let feature_functions = self.feature_functions_.as_ref().unwrap();
        let feature_grids = self.feature_grids_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap_or(0.0);

        let mut predictions = Array1::from_elem(n_samples, intercept);

        // Add contribution from each feature
        for j in 0..n_features {
            let feature_col = x.column(j);
            let contrib = self.evaluate_feature_function(
                &feature_functions[j],
                &feature_grids[j],
                &feature_col,
            );
            predictions = predictions + contrib;
        }

        Ok(predictions)
    }
}

impl AdditiveIsotonicRegression<Trained> {
    /// Evaluate feature function on given feature values (trained model version)
    fn evaluate_feature_function(
        &self,
        function: &Array1<Float>,
        grid: &Array1<Float>,
        feature: &ArrayView1<Float>,
    ) -> Array1<Float> {
        let mut result = Array1::zeros(feature.len());

        for (i, &feat_val) in feature.iter().enumerate() {
            // Find closest grid point and interpolate
            if grid.len() == 1 {
                result[i] = function[0];
                continue;
            }

            // Linear interpolation
            let mut left_idx = 0;
            for j in 0..grid.len() - 1 {
                if feat_val >= grid[j] && feat_val <= grid[j + 1] {
                    left_idx = j;
                    break;
                }
            }

            if left_idx == grid.len() - 1 {
                result[i] = function[left_idx];
            } else {
                let t = (feat_val - grid[left_idx]) / (grid[left_idx + 1] - grid[left_idx]);
                result[i] = function[left_idx] * (1.0 - t) + function[left_idx + 1] * t;
            }
        }

        result
    }

    /// Get the fitted feature functions
    ///
    /// # Returns
    /// Vector of feature functions, one per input feature
    pub fn feature_functions(&self) -> Option<&Vec<Array1<Float>>> {
        self.feature_functions_.as_ref()
    }

    /// Get the feature grids
    ///
    /// # Returns
    /// Vector of feature grids (unique sorted values for each feature)
    pub fn feature_grids(&self) -> Option<&Vec<Array1<Float>>> {
        self.feature_grids_.as_ref()
    }

    /// Get the fitted intercept
    ///
    /// # Returns
    /// Intercept value (0.0 if fit_intercept=false)
    pub fn intercept(&self) -> Option<Float> {
        self.intercept_
    }

    /// Evaluate individual feature contributions
    ///
    /// # Arguments
    /// * `x` - Input features
    ///
    /// # Returns
    /// Matrix where each column represents the contribution of one feature
    pub fn feature_contributions(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_features != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features, n_features
            )));
        }

        let feature_functions = self.feature_functions_.as_ref().unwrap();
        let feature_grids = self.feature_grids_.as_ref().unwrap();

        let mut contributions = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let feature_col = x.column(j);
            let contrib = self.evaluate_feature_function(
                &feature_functions[j],
                &feature_grids[j],
                &feature_col,
            );

            for i in 0..n_samples {
                contributions[[i, j]] = contrib[i];
            }
        }

        Ok(contributions)
    }

    /// Calculate model complexity (total variation of all feature functions)
    ///
    /// # Returns
    /// Sum of total variation across all features
    pub fn total_variation(&self) -> Float {
        if let Some(functions) = &self.feature_functions_ {
            functions
                .iter()
                .map(|f| {
                    if f.len() <= 1 {
                        0.0
                    } else {
                        (0..f.len() - 1)
                            .map(|i| (f[i + 1] - f[i]).abs())
                            .sum::<Float>()
                    }
                })
                .sum()
        } else {
            0.0
        }
    }
}

/// Function API for additive isotonic regression
///
/// Convenience function for fitting additive isotonic regression without
/// explicit model instantiation.
///
/// # Arguments
/// * `x` - Input features (n_samples × n_features)
/// * `y` - Target values (n_samples,)
/// * `constraints` - Monotonicity constraints for each feature
/// * `fit_intercept` - Whether to fit an intercept term
///
/// # Returns
/// Fitted additive isotonic regression values
///
/// # Examples
///
/// ```rust
/// use sklears_isotonic::regularized::additive_isotonic_regression;
/// use scirs2_core::ndarray::{Array1, Array2};
/// use sklears_isotonic::MonotonicityConstraint;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0])?;
/// let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
/// let constraints = vec![
///     MonotonicityConstraint::Global { increasing: true },
///     MonotonicityConstraint::Global { increasing: true }
/// ];
///
/// let fitted_values = additive_isotonic_regression(&x, &y, constraints, true)?;
/// # Ok(())
/// # }
/// ```
pub fn additive_isotonic_regression(
    x: &Array2<Float>,
    y: &Array1<Float>,
    constraints: Vec<MonotonicityConstraint>,
    fit_intercept: bool,
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    let regressor = AdditiveIsotonicRegression::new(n_features)
        .constraints(constraints)
        .fit_intercept(fit_intercept);

    let fitted = regressor.fit(x, y)?;
    fitted.predict(x)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_additive_isotonic_regression_creation() {
        let model = AdditiveIsotonicRegression::new(3);
        assert_eq!(model.n_features, 3);
        assert_eq!(model.constraints.len(), 3);
        assert_eq!(model.fit_intercept, true);
    }

    #[test]
    fn test_additive_isotonic_regression_builder() {
        let constraints = vec![
            MonotonicityConstraint::Global { increasing: false },
            MonotonicityConstraint::Global { increasing: true },
        ];

        let model = AdditiveIsotonicRegression::new(2)
            .constraints(constraints.clone())
            .alpha(0.1)
            .fit_intercept(false);

        assert_eq!(model.n_features, 2);
        assert_eq!(model.constraints.len(), 2);
        assert_eq!(model.alpha, 0.1);
        assert_eq!(model.fit_intercept, false);
    }

    #[test]
    fn test_additive_isotonic_regression_fit_predict() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);

        let model = AdditiveIsotonicRegression::new(2);
        let fitted_model = model.fit(&x, &y).unwrap();
        let predictions = fitted_model.predict(&x).unwrap();

        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_additive_isotonic_regression_feature_contributions() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0]);

        let model = AdditiveIsotonicRegression::new(2);
        let fitted_model = model.fit(&x, &y).unwrap();
        let contributions = fitted_model.feature_contributions(&x).unwrap();

        assert_eq!(contributions.shape(), &[3, 2]);
    }

    #[test]
    fn test_additive_isotonic_regression_function_api() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);
        let constraints = vec![
            MonotonicityConstraint::Global { increasing: true },
            MonotonicityConstraint::Global { increasing: true },
        ];

        let result = additive_isotonic_regression(&x, &y, constraints, true);
        assert!(result.is_ok());

        let fitted_values = result.unwrap();
        assert_eq!(fitted_values.len(), y.len());
    }

    #[test]
    fn test_additive_isotonic_regression_wrong_features() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);

        let model = AdditiveIsotonicRegression::new(2); // Expect 2 features, got 3
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_additive_isotonic_regression_mismatched_samples() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0]); // 3 samples vs 4 in x

        let model = AdditiveIsotonicRegression::new(2);
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_additive_isotonic_regression_total_variation() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);

        let model = AdditiveIsotonicRegression::new(2);
        let fitted_model = model.fit(&x, &y).unwrap();
        let total_variation = fitted_model.total_variation();

        assert!(total_variation >= 0.0);
    }
}
