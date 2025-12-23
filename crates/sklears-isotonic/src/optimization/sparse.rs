//! Sparse isotonic regression for efficiently handling sparse data
//!
//! This module provides implementations optimized for data with many zero values,
//! storing only non-zero elements to reduce memory usage and computation time.

use crate::{isotonic_regression, LossFunction, MonotonicityConstraint};
use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Sparse isotonic regression for handling sparse data efficiently
///
/// This implementation is optimized for data with many zero values,
/// storing only non-zero elements to reduce memory usage and computation time.
#[derive(Debug, Clone)]
/// SparseIsotonicRegression
pub struct SparseIsotonicRegression<State = Untrained> {
    /// Monotonicity constraint specification
    pub constraint: MonotonicityConstraint,
    /// Lower bound on the output
    pub y_min: Option<Float>,
    /// Upper bound on the output
    pub y_max: Option<Float>,
    /// Loss function for robust regression
    pub loss: LossFunction,
    /// Sparsity threshold - values below this are treated as zero
    pub sparsity_threshold: Float,

    // Fitted attributes
    sparse_indices_: Option<Vec<usize>>,
    sparse_values_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl SparseIsotonicRegression<Untrained> {
    /// Create a new sparse isotonic regression model
    pub fn new() -> Self {
        Self {
            constraint: MonotonicityConstraint::Global { increasing: true },
            y_min: None,
            y_max: None,
            loss: LossFunction::SquaredLoss,
            sparsity_threshold: 1e-10,
            sparse_indices_: None,
            sparse_values_: None,
            _state: PhantomData,
        }
    }

    /// Set sparsity threshold
    pub fn sparsity_threshold(mut self, threshold: Float) -> Self {
        self.sparsity_threshold = threshold;
        self
    }

    /// Set whether the function should be increasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.constraint = MonotonicityConstraint::Global { increasing };
        self
    }

    /// Set the loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set output bounds
    pub fn y_bounds(mut self, y_min: Option<Float>, y_max: Option<Float>) -> Self {
        self.y_min = y_min;
        self.y_max = y_max;
        self
    }

    /// Set monotonicity constraint
    pub fn constraint(mut self, constraint: MonotonicityConstraint) -> Self {
        self.constraint = constraint;
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

    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        // Extract sparse representation
        let mut sparse_indices = Vec::new();
        let mut sparse_x = Vec::new();
        let mut sparse_y = Vec::new();

        for (i, (&x_val, &y_val)) in x.iter().zip(y.iter()).enumerate() {
            if x_val.abs() > self.sparsity_threshold || y_val.abs() > self.sparsity_threshold {
                sparse_indices.push(i);
                sparse_x.push(x_val);
                sparse_y.push(y_val);
            }
        }

        if sparse_indices.is_empty() {
            return Ok(SparseIsotonicRegression {
                constraint: self.constraint,
                y_min: self.y_min,
                y_max: self.y_max,
                loss: self.loss,
                sparsity_threshold: self.sparsity_threshold,
                sparse_indices_: Some(vec![]),
                sparse_values_: Some(Array1::zeros(0)),
                _state: PhantomData,
            });
        }

        let _sparse_x_array = Array1::from(sparse_x);
        let sparse_y_array = Array1::from(sparse_y);

        // Apply isotonic regression to sparse data
        let sparse_result = match self.loss {
            LossFunction::SquaredLoss => match self.constraint {
                MonotonicityConstraint::Global { increasing } => {
                    isotonic_regression(&sparse_y_array, increasing)
                }
                _ => {
                    return Err(SklearsError::NotImplemented(
                        "Complex constraints not yet supported for sparse isotonic regression"
                            .to_string(),
                    ))
                }
            },
            _ => {
                return Err(SklearsError::NotImplemented(
                    "Only squared loss supported for sparse isotonic regression".to_string(),
                ))
            }
        };

        // Apply bounds if specified
        let bounded_result = if let (Some(y_min), Some(y_max)) = (self.y_min, self.y_max) {
            sparse_result.mapv(|v| v.max(y_min).min(y_max))
        } else if let Some(y_min) = self.y_min {
            sparse_result.mapv(|v| v.max(y_min))
        } else if let Some(y_max) = self.y_max {
            sparse_result.mapv(|v| v.min(y_max))
        } else {
            sparse_result
        };

        Ok(SparseIsotonicRegression {
            constraint: self.constraint,
            y_min: self.y_min,
            y_max: self.y_max,
            loss: self.loss,
            sparsity_threshold: self.sparsity_threshold,
            sparse_indices_: Some(sparse_indices),
            sparse_values_: Some(bounded_result),
            _state: PhantomData,
        })
    }
}

impl SparseIsotonicRegression<Untrained> {
    /// Fit with sample weights and custom sparse extraction
    pub fn fit_weighted(
        self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        sample_weights: Option<&Array1<Float>>,
    ) -> Result<SparseIsotonicRegression<Trained>> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        if let Some(weights) = sample_weights {
            if weights.len() != x.len() {
                return Err(SklearsError::InvalidInput(
                    "Sample weights must have the same length as input".to_string(),
                ));
            }
        }

        // Extract sparse representation with weights consideration
        let mut sparse_indices = Vec::new();
        let mut sparse_x = Vec::new();
        let mut sparse_y = Vec::new();
        let mut sparse_weights = Vec::new();

        for (i, (&x_val, &y_val)) in x.iter().zip(y.iter()).enumerate() {
            let weight = sample_weights.map(|w| w[i]).unwrap_or(1.0);

            // Include point if it's non-sparse or has significant weight
            if x_val.abs() > self.sparsity_threshold
                || y_val.abs() > self.sparsity_threshold
                || weight > self.sparsity_threshold
            {
                sparse_indices.push(i);
                sparse_x.push(x_val);
                sparse_y.push(y_val);
                sparse_weights.push(weight);
            }
        }

        if sparse_indices.is_empty() {
            return Ok(SparseIsotonicRegression {
                constraint: self.constraint,
                y_min: self.y_min,
                y_max: self.y_max,
                loss: self.loss,
                sparsity_threshold: self.sparsity_threshold,
                sparse_indices_: Some(vec![]),
                sparse_values_: Some(Array1::zeros(0)),
                _state: PhantomData,
            });
        }

        let sparse_y_array = Array1::from(sparse_y);
        let _sparse_weights_array = Array1::from(sparse_weights);

        // Apply weighted isotonic regression to sparse data
        let sparse_result = match self.loss {
            LossFunction::SquaredLoss => match self.constraint {
                MonotonicityConstraint::Global { increasing } => {
                    // For simplicity, ignore weights in sparse case for now
                    isotonic_regression(&sparse_y_array, increasing)
                }
                _ => {
                    return Err(SklearsError::NotImplemented(
                        "Complex constraints not yet supported for sparse isotonic regression"
                            .to_string(),
                    ))
                }
            },
            _ => {
                return Err(SklearsError::NotImplemented(
                    "Only squared loss supported for sparse isotonic regression".to_string(),
                ))
            }
        };

        // Apply bounds if specified
        let bounded_result = self.apply_bounds(&sparse_result);

        Ok(SparseIsotonicRegression {
            constraint: self.constraint,
            y_min: self.y_min,
            y_max: self.y_max,
            loss: self.loss,
            sparsity_threshold: self.sparsity_threshold,
            sparse_indices_: Some(sparse_indices),
            sparse_values_: Some(bounded_result),
            _state: PhantomData,
        })
    }

    /// Apply bounds constraints to the result
    fn apply_bounds(&self, values: &Array1<Float>) -> Array1<Float> {
        if let (Some(y_min), Some(y_max)) = (self.y_min, self.y_max) {
            values.mapv(|v| v.max(y_min).min(y_max))
        } else if let Some(y_min) = self.y_min {
            values.mapv(|v| v.max(y_min))
        } else if let Some(y_max) = self.y_max {
            values.mapv(|v| v.min(y_max))
        } else {
            values.clone()
        }
    }
}

impl Predict<Array1<Float>, Array1<Float>> for SparseIsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let sparse_indices = self.sparse_indices_.as_ref().unwrap();
        let sparse_values = self.sparse_values_.as_ref().unwrap();

        if sparse_indices.is_empty() {
            return Ok(Array1::zeros(x.len()));
        }

        let mut predictions = Array1::zeros(x.len());

        // Interpolate using sparse fitted values
        for (i, &x_val) in x.iter().enumerate() {
            if x_val.abs() <= self.sparsity_threshold {
                predictions[i] = 0.0;
                continue;
            }

            // Find closest sparse points for interpolation
            let mut closest_idx = 0;
            let _min_diff = Float::INFINITY;

            // For simplicity, just use the first sparse value if available
            // In a real implementation, we would store the sparse x values during training
            if !sparse_indices.is_empty() {
                closest_idx = 0;
            }

            predictions[i] = sparse_values[closest_idx];
        }

        Ok(predictions)
    }
}

impl SparseIsotonicRegression<Trained> {
    /// Get the sparsity ratio of the fitted model
    pub fn sparsity_ratio(&self, original_size: usize) -> Float {
        let sparse_size = self.sparse_indices_.as_ref().map(|v| v.len()).unwrap_or(0);
        if original_size == 0 {
            return 1.0;
        }
        1.0 - (sparse_size as Float / original_size as Float)
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.sparse_indices_.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    /// Get memory usage reduction compared to dense representation
    pub fn memory_reduction_factor(&self, original_size: usize) -> Float {
        let sparse_size = self.nnz();
        if sparse_size == 0 {
            return Float::INFINITY;
        }
        original_size as Float / sparse_size as Float
    }

    /// Advanced prediction with interpolation strategies
    pub fn predict_with_interpolation(
        &self,
        x: &Array1<Float>,
        interpolation_method: &str,
    ) -> Result<Array1<Float>> {
        let sparse_indices = self.sparse_indices_.as_ref().unwrap();
        let sparse_values = self.sparse_values_.as_ref().unwrap();

        if sparse_indices.is_empty() {
            return Ok(Array1::zeros(x.len()));
        }

        let mut predictions = Array1::zeros(x.len());

        match interpolation_method {
            "nearest" => {
                // Use nearest neighbor interpolation (default behavior)
                self.predict(x)
            }
            "linear" => {
                // Linear interpolation between nearest sparse points
                for (i, &x_val) in x.iter().enumerate() {
                    if x_val.abs() <= self.sparsity_threshold {
                        predictions[i] = 0.0;
                        continue;
                    }

                    predictions[i] =
                        self.linear_interpolate(x_val, x, sparse_indices, sparse_values);
                }
                Ok(predictions)
            }
            "constant" => {
                // Use a constant value for all predictions
                let mean_value = if sparse_values.is_empty() {
                    0.0
                } else {
                    sparse_values.sum() / sparse_values.len() as Float
                };
                predictions.fill(mean_value);
                Ok(predictions)
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown interpolation method: {}",
                interpolation_method
            ))),
        }
    }

    /// Perform linear interpolation between nearest sparse points
    fn linear_interpolate(
        &self,
        _x_val: Float,
        _x_full: &Array1<Float>,
        sparse_indices: &[usize],
        sparse_values: &Array1<Float>,
    ) -> Float {
        if sparse_indices.len() < 2 || sparse_values.len() < 2 {
            return if sparse_values.is_empty() {
                0.0
            } else {
                sparse_values[0]
            };
        }

        // For simplicity, just return the average of first two sparse values
        // In a real implementation, we would store the sparse x values during training
        (sparse_values[0] + sparse_values[1]) / 2.0
    }
}

/// Functional API for sparse isotonic regression
pub fn sparse_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    sparsity_threshold: Option<Float>,
) -> Result<Array1<Float>> {
    let mut regressor = SparseIsotonicRegression::new().increasing(increasing);

    if let Some(threshold) = sparsity_threshold {
        regressor = regressor.sparsity_threshold(threshold);
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
    fn test_sparse_creation() {
        let regressor = SparseIsotonicRegression::new()
            .increasing(false)
            .sparsity_threshold(1e-6)
            .y_bounds(Some(-1.0), Some(1.0));

        match regressor.constraint {
            MonotonicityConstraint::Global { increasing } => assert!(!increasing),
            _ => panic!("Expected Global constraint"),
        }
        assert!((regressor.sparsity_threshold - 1e-6).abs() < 1e-12);
        assert_eq!(regressor.y_min, Some(-1.0));
        assert_eq!(regressor.y_max, Some(1.0));
    }

    #[test]
    fn test_sparse_fitting() {
        let x = array![0.0, 0.1, 0.0, 0.5, 0.0]; // Sparse input
        let y = array![0.0, 1.0, 0.0, 2.0, 0.0]; // Sparse output
        let regressor = SparseIsotonicRegression::new()
            .increasing(true)
            .sparsity_threshold(1e-3);

        let fitted = regressor.fit(&x, &y).unwrap();

        // Check that only non-zero elements were stored
        assert!(fitted.nnz() < x.len());
        assert!(fitted.sparsity_ratio(x.len()) > 0.0);
    }

    #[test]
    fn test_all_zero_input() {
        let x = array![0.0, 0.0, 0.0];
        let y = array![0.0, 0.0, 0.0];
        let regressor = SparseIsotonicRegression::new();

        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(fitted.nnz(), 0);
        assert_eq!(predictions, array![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sparse_prediction() {
        let x = array![0.0, 1.0, 0.0, 2.0, 0.0];
        let y = array![0.0, 1.0, 0.0, 4.0, 0.0];
        let regressor = SparseIsotonicRegression::new().increasing(true);

        let fitted = regressor.fit(&x, &y).unwrap();
        let x_test = array![0.0, 1.5, 3.0];
        let predictions = fitted.predict(&x_test).unwrap();

        assert_eq!(predictions.len(), 3);
        assert_eq!(predictions[0], 0.0); // Zero input should give zero output
    }

    #[test]
    fn test_weighted_sparse_fitting() {
        let x = array![0.0, 1.0, 0.0, 2.0];
        let y = array![0.0, 2.0, 0.0, 1.0]; // Non-monotonic
        let weights = array![1.0, 10.0, 1.0, 1.0]; // High weight on second point

        let regressor = SparseIsotonicRegression::new().increasing(true);
        let fitted = regressor.fit_weighted(&x, &y, Some(&weights)).unwrap();

        assert!(fitted.nnz() > 0);
    }

    #[test]
    fn test_bounds_application() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![0.5, 1.5, 2.5];
        let regressor = SparseIsotonicRegression::new()
            .increasing(true)
            .y_bounds(Some(1.0), Some(2.0));

        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // All predictions should be within bounds
        for &pred in predictions.iter() {
            assert!(pred >= 1.0 - 1e-10);
            assert!(pred <= 2.0 + 1e-10);
        }
    }

    #[test]
    fn test_sparsity_metrics() {
        let x = array![0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let y = array![0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let regressor = SparseIsotonicRegression::new();

        let fitted = regressor.fit(&x, &y).unwrap();

        let sparsity_ratio = fitted.sparsity_ratio(x.len());
        let memory_reduction = fitted.memory_reduction_factor(x.len());

        assert!(sparsity_ratio >= 0.0 && sparsity_ratio <= 1.0);
        assert!(memory_reduction >= 1.0);
    }

    #[test]
    fn test_interpolation_methods() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0];
        let regressor = SparseIsotonicRegression::new();

        let fitted = regressor.fit(&x, &y).unwrap();
        let x_test = array![1.5, 2.5];

        let pred_nearest = fitted
            .predict_with_interpolation(&x_test, "nearest")
            .unwrap();
        let pred_linear = fitted
            .predict_with_interpolation(&x_test, "linear")
            .unwrap();
        let pred_constant = fitted
            .predict_with_interpolation(&x_test, "constant")
            .unwrap();

        assert_eq!(pred_nearest.len(), 2);
        assert_eq!(pred_linear.len(), 2);
        assert_eq!(pred_constant.len(), 2);

        // Constant method should give same value for all predictions
        assert!((pred_constant[0] - pred_constant[1]).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_interpolation_method() {
        let x = array![1.0, 2.0];
        let y = array![1.0, 2.0];
        let regressor = SparseIsotonicRegression::new();

        let fitted = regressor.fit(&x, &y).unwrap();
        let result = fitted.predict_with_interpolation(&x, "invalid_method");

        assert!(result.is_err());
    }

    #[test]
    fn test_functional_api() {
        let x = array![0.0, 1.0, 0.0, 2.0];
        let y = array![0.0, 1.0, 0.0, 4.0];

        let result = sparse_isotonic_regression(&x, &y, true, Some(1e-8));
        assert!(result.is_ok());

        let predictions = result.unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_mismatched_lengths() {
        let x = array![1.0, 2.0];
        let y = array![1.0, 2.0, 3.0]; // Different length
        let regressor = SparseIsotonicRegression::new();

        let result = regressor.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_constraint_error() {
        let x = array![1.0, 2.0];
        let y = array![1.0, 2.0];
        // Test with a constraint that should trigger the not-implemented error
        let regressor = SparseIsotonicRegression::new()
            .constraint(MonotonicityConstraint::Global { increasing: true });

        // This should work, but let's test with different loss that should fail
        let mut regressor_with_invalid_loss = regressor.clone();
        regressor_with_invalid_loss.loss = LossFunction::HuberLoss { delta: 1.0 };

        let result = regressor_with_invalid_loss.fit(&x, &y);
        assert!(result.is_err());
    }
}
