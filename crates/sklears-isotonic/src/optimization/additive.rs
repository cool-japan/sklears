//! Additive isotonic regression models
//!
//! This module implements additive isotonic regression where separate isotonic constraints
//! are applied to different features and combined additively: f(x) = f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ)

use crate::{isotonic_regression, LossFunction, MonotonicityConstraint};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Additive isotonic regression model
///
/// This model applies separate isotonic constraints to different features
/// and combines them additively: f(x) = f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ)
#[derive(Debug, Clone)]
/// AdditiveIsotonicRegression
pub struct AdditiveIsotonicRegression<State = Untrained> {
    /// Number of features
    pub n_features: usize,
    /// Monotonicity constraints for each feature
    pub constraints: Vec<MonotonicityConstraint>,
    /// Loss function for robust regression
    pub loss: LossFunction,
    /// Regularization strength for additive components
    pub alpha: Float,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Maximum number of iterations for coordinate descent
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,

    // Fitted attributes
    feature_functions_: Option<Vec<Array1<Float>>>,
    feature_grids_: Option<Vec<Array1<Float>>>,
    intercept_: Option<Float>,

    _state: PhantomData<State>,
}

impl AdditiveIsotonicRegression<Untrained> {
    /// Create a new additive isotonic regression model
    pub fn new(n_features: usize) -> Self {
        let constraints = vec![MonotonicityConstraint::Global { increasing: true }; n_features];

        Self {
            n_features,
            constraints,
            loss: LossFunction::SquaredLoss,
            alpha: 0.0,
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-6,
            feature_functions_: None,
            feature_grids_: None,
            intercept_: None,
            _state: PhantomData,
        }
    }

    /// Set monotonicity constraints for all features
    pub fn constraints(mut self, mut constraints: Vec<MonotonicityConstraint>) -> Self {
        // Handle constraint length mismatch gracefully
        if constraints.len() != self.n_features {
            if constraints.len() < self.n_features {
                // Extend with default increasing constraints
                let default_constraint = MonotonicityConstraint::Global { increasing: true };
                constraints.resize(self.n_features, default_constraint);
            } else {
                // Truncate to required length
                constraints.truncate(self.n_features);
            }
        }

        self.constraints = constraints;
        self
    }

    /// Set whether a specific feature should be increasing
    pub fn feature_increasing(mut self, feature_idx: usize, increasing: bool) -> Self {
        if feature_idx < self.n_features {
            self.constraints[feature_idx] = MonotonicityConstraint::Global { increasing };
        }
        self
    }

    /// Set the loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set optimization parameters
    pub fn optimization_params(mut self, max_iterations: usize, tolerance: Float) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self
    }
}

impl Default for AdditiveIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Estimator for AdditiveIsotonicRegression<Untrained> {
    type Config = Self;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<Array2<Float>, Array1<Float>> for AdditiveIsotonicRegression<Untrained> {
    type Fitted = AdditiveIsotonicRegression<Trained>;

    fn fit(
        self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<AdditiveIsotonicRegression<Trained>> {
        self.fit_weighted(x, y, None)
    }
}

impl AdditiveIsotonicRegression<Untrained> {
    /// Fit with sample weights using coordinate descent
    pub fn fit_weighted(
        self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        sample_weights: Option<&Array1<Float>>,
    ) -> Result<AdditiveIsotonicRegression<Trained>> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_features != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features in X ({}) does not match expected ({})",
                n_features, self.n_features
            )));
        }

        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if let Some(weights) = sample_weights {
            if weights.len() != n_samples {
                return Err(SklearsError::InvalidInput(
                    "Sample weights must have the same length as the number of samples".to_string(),
                ));
            }
        }

        let default_weights = Array1::ones(n_samples);
        let weights = sample_weights.unwrap_or(&default_weights);

        // Initialize feature functions and grids
        let mut feature_functions = Vec::new();
        let mut feature_grids = Vec::new();
        let mut residuals = y.clone();

        // Initialize intercept
        let mut intercept = if self.fit_intercept {
            let weighted_sum: Float = residuals
                .iter()
                .zip(weights.iter())
                .map(|(r, w)| r * w)
                .sum();
            let weight_sum: Float = weights.sum();
            if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                0.0
            }
        } else {
            0.0
        };

        if self.fit_intercept {
            for i in 0..n_samples {
                residuals[i] -= intercept;
            }
        }

        // Initialize feature functions to zero
        for _ in 0..n_features {
            feature_functions.push(Array1::zeros(n_samples));
            feature_grids.push(Array1::zeros(n_samples));
        }

        // Coordinate descent iterations
        for _iteration in 0..self.max_iterations {
            let mut max_change: Float = 0.0;

            // Update each feature function
            for j in 0..n_features {
                // Compute partial residuals (excluding current feature)
                let mut partial_residuals = residuals.clone();
                for i in 0..n_samples {
                    partial_residuals[i] += feature_functions[j][i];
                }

                // Extract feature values for current dimension
                let x_j = x.column(j).to_owned();

                // Fit isotonic regression to partial residuals
                let new_function = match &self.constraints[j] {
                    MonotonicityConstraint::Global { increasing } => self.fit_univariate_isotonic(
                        &x_j,
                        &partial_residuals,
                        Some(weights),
                        *increasing,
                    )?,
                    _ => {
                        return Err(SklearsError::NotImplemented(
                            "Complex constraints not yet implemented for additive isotonic regression".to_string()
                        ));
                    }
                };

                // Compute change in function
                let change = (&new_function - &feature_functions[j])
                    .mapv(|x| x.abs())
                    .sum();
                max_change = max_change.max(change);

                // Update residuals
                for i in 0..n_samples {
                    residuals[i] -= new_function[i] - feature_functions[j][i];
                }

                // Store new function
                feature_functions[j] = new_function;
                feature_grids[j] = x_j;
            }

            // Update intercept if enabled
            if self.fit_intercept {
                let old_intercept = intercept;
                let weighted_sum: Float = residuals
                    .iter()
                    .zip(weights.iter())
                    .map(|(r, w)| r * w)
                    .sum();
                let weight_sum: Float = weights.sum();
                intercept = if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    0.0
                };

                let intercept_change = (intercept - old_intercept).abs();
                max_change = max_change.max(intercept_change);

                // Update residuals
                let intercept_diff = intercept - old_intercept;
                for i in 0..n_samples {
                    residuals[i] -= intercept_diff;
                }
            }

            // Check convergence
            if max_change < self.tolerance {
                break;
            }
        }

        Ok(AdditiveIsotonicRegression {
            n_features: self.n_features,
            constraints: self.constraints,
            loss: self.loss,
            alpha: self.alpha,
            fit_intercept: self.fit_intercept,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            feature_functions_: Some(feature_functions),
            feature_grids_: Some(feature_grids),
            intercept_: Some(intercept),
            _state: PhantomData,
        })
    }

    /// Fit univariate isotonic regression for a single feature
    fn fit_univariate_isotonic(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        sample_weights: Option<&Array1<Float>>,
        increasing: bool,
    ) -> Result<Array1<Float>> {
        match self.loss {
            LossFunction::SquaredLoss => {
                // For squared loss, use the standard isotonic regression
                let sorted_indices = self.argsort(x);
                let mut sorted_y = Array1::zeros(y.len());

                for (i, &idx) in sorted_indices.iter().enumerate() {
                    sorted_y[i] = y[idx];
                }

                let isotonic_result = isotonic_regression(&sorted_y, increasing);

                // Map back to original order
                let mut result = Array1::zeros(y.len());
                for (i, &idx) in sorted_indices.iter().enumerate() {
                    result[idx] = isotonic_result[i];
                }

                Ok(result)
            }
            _ => Err(SklearsError::NotImplemented(
                "Only squared loss is currently supported for additive isotonic regression"
                    .to_string(),
            )),
        }
    }

    /// Compute argsort indices for sorting
    fn argsort(&self, x: &Array1<Float>) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..x.len()).collect();
        indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
        indices
    }
}

impl Predict<Array2<Float>, Array1<Float>> for AdditiveIsotonicRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_features != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Input has {} features but model was trained on {} features",
                n_features, self.n_features
            )));
        }

        let feature_functions = self.feature_functions_.as_ref().unwrap();
        let feature_grids = self.feature_grids_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap_or(0.0);

        let mut predictions = Array1::zeros(n_samples);

        // Add intercept
        if self.fit_intercept {
            for i in 0..n_samples {
                predictions[i] += intercept;
            }
        }

        // Add contribution from each feature
        for j in 0..n_features {
            let x_j = x.column(j);
            let grid_j = &feature_grids[j];
            let function_j = &feature_functions[j];

            for i in 0..n_samples {
                let x_val = x_j[i];

                // Interpolate the feature function value
                let feature_contribution =
                    self.interpolate_feature_function(x_val, grid_j, function_j);

                predictions[i] += feature_contribution;
            }
        }

        Ok(predictions)
    }
}

impl AdditiveIsotonicRegression<Trained> {
    /// Interpolate feature function value at a given point
    fn interpolate_feature_function(
        &self,
        x_val: Float,
        grid: &Array1<Float>,
        function: &Array1<Float>,
    ) -> Float {
        if grid.is_empty() {
            return 0.0;
        }

        // Find the closest grid point(s) for interpolation
        let mut closest_idx = 0;
        let mut min_distance = Float::INFINITY;

        for (i, &grid_val) in grid.iter().enumerate() {
            let distance = (x_val - grid_val).abs();
            if distance < min_distance {
                min_distance = distance;
                closest_idx = i;
            }
        }

        // For simplicity, use nearest neighbor interpolation
        // In a more advanced implementation, this could use linear interpolation
        function[closest_idx]
    }

    /// Get the fitted feature functions
    pub fn feature_functions(&self) -> &[Array1<Float>] {
        self.feature_functions_.as_ref().unwrap()
    }

    /// Get the fitted feature grids
    pub fn feature_grids(&self) -> &[Array1<Float>] {
        self.feature_grids_.as_ref().unwrap()
    }

    /// Get the fitted intercept
    pub fn intercept(&self) -> Float {
        self.intercept_.unwrap_or(0.0)
    }

    /// Compute feature importance based on function variance
    pub fn feature_importance(&self) -> Array1<Float> {
        let feature_functions = self.feature_functions_.as_ref().unwrap();
        let mut importance = Array1::zeros(self.n_features);

        for (j, function) in feature_functions.iter().enumerate() {
            if !function.is_empty() {
                let mean = function.sum() / function.len() as Float;
                let variance =
                    function.mapv(|x| (x - mean).powi(2)).sum() / function.len() as Float;
                importance[j] = variance.sqrt(); // Use standard deviation as importance measure
            }
        }

        importance
    }

    /// Evaluate the additive model's complexity
    pub fn model_complexity(&self) -> Float {
        let feature_functions = self.feature_functions_.as_ref().unwrap();
        let mut complexity = 0.0;

        for function in feature_functions {
            // Count effective degrees of freedom (simplified as function length)
            complexity += function.len() as Float;
        }

        complexity
    }
}

/// Functional API for additive isotonic regression
pub fn additive_isotonic_regression(
    x: &Array2<Float>,
    y: &Array1<Float>,
    constraints: &[bool],
    alpha: Option<Float>,
    fit_intercept: Option<bool>,
) -> Result<Array1<Float>> {
    let n_features = x.shape()[1];
    let mut regressor = AdditiveIsotonicRegression::new(n_features);

    // Set constraints
    let monotonic_constraints: Vec<MonotonicityConstraint> = constraints
        .iter()
        .map(|&increasing| MonotonicityConstraint::Global { increasing })
        .collect();
    regressor = regressor.constraints(monotonic_constraints);

    if let Some(alpha_val) = alpha {
        regressor = regressor.alpha(alpha_val);
    }

    if let Some(intercept) = fit_intercept {
        regressor = regressor.fit_intercept(intercept);
    }

    let fitted = regressor.fit(x, y)?;
    fitted.predict(x)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_additive_creation() {
        let regressor = AdditiveIsotonicRegression::new(3)
            .alpha(0.1)
            .fit_intercept(false)
            .optimization_params(50, 1e-8);

        assert_eq!(regressor.n_features, 3);
        assert!((regressor.alpha - 0.1).abs() < 1e-10);
        assert!(!regressor.fit_intercept);
        assert_eq!(regressor.max_iterations, 50);
        assert!((regressor.tolerance - 1e-8).abs() < 1e-15);
    }

    #[test]
    fn test_feature_constraints() {
        let mut regressor = AdditiveIsotonicRegression::new(2);
        regressor = regressor.feature_increasing(0, true);
        regressor = regressor.feature_increasing(1, false);

        match &regressor.constraints[0] {
            MonotonicityConstraint::Global { increasing } => assert!(*increasing),
            _ => panic!("Expected Global constraint"),
        }

        match &regressor.constraints[1] {
            MonotonicityConstraint::Global { increasing } => assert!(!*increasing),
            _ => panic!("Expected Global constraint"),
        }
    }

    #[test]
    fn test_simple_additive_fit() {
        let x = array![[1.0, 1.0], [2.0, 0.5], [3.0, 2.0]];
        let y = array![2.0, 2.5, 5.0];

        let regressor = AdditiveIsotonicRegression::new(2);
        let fitted = regressor.fit(&x, &y).unwrap();

        assert_eq!(fitted.feature_functions().len(), 2);
        assert_eq!(fitted.feature_grids().len(), 2);
    }

    #[test]
    fn test_additive_prediction() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let y = array![1.0, 4.0, 9.0]; // Roughly x1^2 + x2^2

        let regressor = AdditiveIsotonicRegression::new(2);
        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 3);
        // Predictions should be reasonable approximations
        for &pred in predictions.iter() {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_intercept_fitting() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![5.0, 6.0, 7.0]; // Linear with intercept

        let regressor = AdditiveIsotonicRegression::new(1).fit_intercept(true);
        let fitted = regressor.fit(&x, &y).unwrap();

        let intercept = fitted.intercept();
        assert!(intercept.is_finite()); // Should learn a finite intercept
    }

    #[test]
    fn test_feature_importance() {
        let x = array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let y = array![1.0, 4.0, 9.0]; // Only first feature matters

        let regressor = AdditiveIsotonicRegression::new(2);
        let fitted = regressor.fit(&x, &y).unwrap();
        let importance = fitted.feature_importance();

        // First feature should be more important than second
        assert!(importance[0] > importance[1]);
    }

    #[test]
    fn test_weighted_fitting() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 10.0, 3.0]; // Middle point is outlier
        let weights = array![1.0, 0.1, 1.0]; // Low weight on outlier

        let regressor = AdditiveIsotonicRegression::new(1);
        let fitted = regressor.fit_weighted(&x, &y, Some(&weights)).unwrap();

        // Should still produce reasonable results despite outlier
        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_functional_api() {
        let x = array![[1.0, 1.0], [2.0, 0.5], [3.0, 2.0]];
        let y = array![2.0, 2.5, 5.0];
        let constraints = vec![true, false];

        let result = additive_isotonic_regression(&x, &y, &constraints, Some(0.1), Some(true));
        assert!(result.is_ok());

        let predictions = result.unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_mismatched_features() {
        let x = array![[1.0, 2.0], [3.0, 4.0]]; // 2 features
        let y = array![1.0, 2.0];

        let regressor = AdditiveIsotonicRegression::new(3); // Expecting 3 features
        let result = regressor.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_samples() {
        let x = array![[1.0], [2.0]]; // 2 samples
        let y = array![1.0, 2.0, 3.0]; // 3 samples

        let regressor = AdditiveIsotonicRegression::new(1);
        let result = regressor.fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_model_complexity() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let y = array![1.0, 4.0, 9.0];

        let regressor = AdditiveIsotonicRegression::new(2);
        let fitted = regressor.fit(&x, &y).unwrap();

        let complexity = fitted.model_complexity();
        assert!(complexity > 0.0);
    }

    #[test]
    fn test_constraint_length_mismatch() {
        let constraints = vec![MonotonicityConstraint::Global { increasing: true }]; // Only 1 constraint
        let regressor = AdditiveIsotonicRegression::new(3) // But 3 features
            .constraints(constraints);

        // Should automatically extend with default constraints
        assert_eq!(regressor.constraints.len(), 3);
    }

    #[test]
    fn test_argsort() {
        let regressor = AdditiveIsotonicRegression::new(1);
        let x = array![3.0, 1.0, 4.0, 2.0];

        let sorted_indices = regressor.argsort(&x);
        assert_eq!(sorted_indices, vec![1, 3, 0, 2]); // Indices of sorted values [1.0, 2.0, 3.0, 4.0]
    }

    #[test]
    fn test_empty_input() {
        let x = Array2::<Float>::zeros((0, 1));
        let y = Array1::<Float>::zeros(0);

        let regressor = AdditiveIsotonicRegression::new(1);
        let fitted = regressor.fit(&x, &y).unwrap();

        let x_test = Array2::<Float>::zeros((0, 1));
        let predictions = fitted.predict(&x_test).unwrap();
        assert_eq!(predictions.len(), 0);
    }
}
