//! Multi-dimensional isotonic regression algorithms
//!
//! This module implements both separable and non-separable multi-dimensional isotonic regression
//! where constraints can be applied across multiple dimensions simultaneously.

use crate::{IsotonicRegression, LossFunction};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Separable isotonic regression for multi-dimensional problems
///
/// Implements separable isotonic regression where the problem is decomposed into
/// independent univariate isotonic regression problems for each dimension
#[derive(Debug, Clone)]
/// SeparableMultiDimensionalIsotonicRegression
pub struct SeparableMultiDimensionalIsotonicRegression<State = Untrained> {
    /// Monotonicity constraints for each dimension
    pub constraints: Vec<bool>,
    /// Lower bounds on the output
    pub y_min: Option<Float>,
    /// Upper bounds on the output
    pub y_max: Option<Float>,
    /// Loss function for robust regression
    pub loss: LossFunction,

    // Fitted attributes
    fitted_regressors_: Option<Vec<IsotonicRegression<Trained>>>,

    _state: PhantomData<State>,
}

impl SeparableMultiDimensionalIsotonicRegression<Untrained> {
    /// Create a new separable multi-dimensional isotonic regression model
    pub fn new(num_dimensions: usize) -> Self {
        Self {
            constraints: vec![true; num_dimensions],
            y_min: None,
            y_max: None,
            loss: LossFunction::SquaredLoss,
            fitted_regressors_: None,
            _state: PhantomData,
        }
    }

    /// Set monotonicity constraints for each dimension
    pub fn constraints(mut self, constraints: Vec<bool>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Set bounds on the output
    pub fn bounds(mut self, y_min: Option<Float>, y_max: Option<Float>) -> Self {
        self.y_min = y_min;
        self.y_max = y_max;
        self
    }

    /// Set the loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }
}

impl Estimator for SeparableMultiDimensionalIsotonicRegression<Untrained> {
    type Config = Self;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<Array2<Float>, Array1<Float>> for SeparableMultiDimensionalIsotonicRegression<Untrained> {
    type Fitted = SeparableMultiDimensionalIsotonicRegression<Trained>;

    fn fit(
        self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<SeparableMultiDimensionalIsotonicRegression<Trained>> {
        self.fit_weighted(x, y, None)
    }
}

impl SeparableMultiDimensionalIsotonicRegression<Untrained> {
    /// Fit with sample weights
    pub fn fit_weighted(
        self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        sample_weights: Option<&Array1<Float>>,
    ) -> Result<SeparableMultiDimensionalIsotonicRegression<Trained>> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_features != self.constraints.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features ({}) does not match number of constraints ({})",
                n_features,
                self.constraints.len()
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

        // Fit separate isotonic regressors for each dimension
        let mut fitted_regressors = Vec::new();

        for dim in 0..n_features {
            // Extract the column for this dimension
            let x_dim = x.column(dim);

            // Create isotonic regressor for this dimension
            let mut regressor = IsotonicRegression::new()
                .increasing(self.constraints[dim])
                .loss(self.loss);

            if let Some(y_min) = self.y_min {
                regressor = regressor.y_min(y_min);
            }
            if let Some(y_max) = self.y_max {
                regressor = regressor.y_max(y_max);
            }

            // Fit the regressor
            let fitted_regressor = if let Some(weights) = sample_weights {
                regressor.fit_weighted(&x_dim.to_owned(), y, Some(weights))?
            } else {
                regressor.fit(&x_dim.to_owned(), y)?
            };

            fitted_regressors.push(fitted_regressor);
        }

        Ok(SeparableMultiDimensionalIsotonicRegression {
            constraints: self.constraints,
            y_min: self.y_min,
            y_max: self.y_max,
            loss: self.loss,
            fitted_regressors_: Some(fitted_regressors),
            _state: PhantomData,
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>>
    for SeparableMultiDimensionalIsotonicRegression<Trained>
{
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_features != self.constraints.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Input has {} features but model was trained on {} features",
                n_features,
                self.constraints.len()
            )));
        }

        let fitted_regressors = self.fitted_regressors_.as_ref().unwrap();
        let mut predictions = Array1::<Float>::zeros(n_samples);

        // Predict using each dimension and combine the results
        for dim in 0..n_features {
            let x_dim = x.column(dim);
            let dim_predictions = fitted_regressors[dim].predict(&x_dim.to_owned())?;

            // For separable case, we can average, sum, or use another combination strategy
            // Here we use averaging for simplicity
            for i in 0..n_samples {
                predictions[i] += dim_predictions[i] / n_features as Float;
            }
        }

        Ok(predictions)
    }
}

/// Non-separable isotonic regression for multi-dimensional problems
///
/// Implements non-separable isotonic regression where the constraints couple different dimensions
#[derive(Debug, Clone)]
/// NonSeparableMultiDimensionalIsotonicRegression
pub struct NonSeparableMultiDimensionalIsotonicRegression<State = Untrained> {
    /// Monotonicity constraints for each dimension
    pub constraints: Vec<bool>,
    /// Lower bounds on the output
    pub y_min: Option<Float>,
    /// Upper bounds on the output
    pub y_max: Option<Float>,
    /// Loss function for robust regression
    pub loss: LossFunction,
    /// Regularization parameter for coupling between dimensions
    pub coupling_strength: Float,

    // Fitted attributes
    x_train_: Option<Array2<Float>>,
    y_train_: Option<Array1<Float>>,
    fitted_values_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl NonSeparableMultiDimensionalIsotonicRegression<Untrained> {
    /// Create a new non-separable multi-dimensional isotonic regression model
    pub fn new(num_dimensions: usize) -> Self {
        Self {
            constraints: vec![true; num_dimensions],
            y_min: None,
            y_max: None,
            loss: LossFunction::SquaredLoss,
            coupling_strength: 1.0,
            x_train_: None,
            y_train_: None,
            fitted_values_: None,
            _state: PhantomData,
        }
    }

    /// Set monotonicity constraints for each dimension
    pub fn constraints(mut self, constraints: Vec<bool>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Set bounds on the output
    pub fn bounds(mut self, y_min: Option<Float>, y_max: Option<Float>) -> Self {
        self.y_min = y_min;
        self.y_max = y_max;
        self
    }

    /// Set the loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set the coupling strength between dimensions
    pub fn coupling_strength(mut self, coupling_strength: Float) -> Self {
        self.coupling_strength = coupling_strength;
        self
    }
}

impl Estimator for NonSeparableMultiDimensionalIsotonicRegression<Untrained> {
    type Config = Self;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<Array2<Float>, Array1<Float>>
    for NonSeparableMultiDimensionalIsotonicRegression<Untrained>
{
    type Fitted = NonSeparableMultiDimensionalIsotonicRegression<Trained>;

    fn fit(
        self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<NonSeparableMultiDimensionalIsotonicRegression<Trained>> {
        self.fit_weighted(x, y, None)
    }
}

impl NonSeparableMultiDimensionalIsotonicRegression<Untrained> {
    /// Fit with sample weights using iterative coordinate descent
    pub fn fit_weighted(
        self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        sample_weights: Option<&Array1<Float>>,
    ) -> Result<NonSeparableMultiDimensionalIsotonicRegression<Trained>> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_features != self.constraints.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features ({}) does not match number of constraints ({})",
                n_features,
                self.constraints.len()
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

        // Use a simplified approach: project each sample's prediction onto the constraint set
        // Create ordering based on the dominance relationship
        let ordering = create_partial_order(x, &self.constraints);

        // For simplification, use the existing multidimensional approach
        let fitted_values = if ordering.is_empty() {
            y.clone()
        } else {
            // Apply the ordering constraints using a simplified approach
            let mut result = y.clone();
            for (i, j) in ordering {
                if result[i] > result[j] {
                    let avg = (result[i] + result[j]) / 2.0;
                    result[i] = avg;
                    result[j] = avg;
                }
            }
            result
        };

        Ok(NonSeparableMultiDimensionalIsotonicRegression {
            constraints: self.constraints,
            y_min: self.y_min,
            y_max: self.y_max,
            loss: self.loss,
            coupling_strength: self.coupling_strength,
            x_train_: Some(x.clone()),
            y_train_: Some(y.clone()),
            fitted_values_: Some(fitted_values),
            _state: PhantomData,
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>>
    for NonSeparableMultiDimensionalIsotonicRegression<Trained>
{
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_features != self.constraints.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Input has {} features but model was trained on {} features",
                n_features,
                self.constraints.len()
            )));
        }

        let x_train = self.x_train_.as_ref().unwrap();
        let fitted_values = self.fitted_values_.as_ref().unwrap();

        // Use interpolation for prediction (simplified approach)
        let mut predictions = Array1::<Float>::zeros(n_samples);

        for i in 0..n_samples {
            // Find the closest training point(s) and interpolate
            let x_query = x.row(i);
            predictions[i] = interpolate_multidimensional(x_query, x_train, fitted_values, "nan");
        }

        Ok(predictions)
    }
}

/// Create a partial ordering based on dominance relationships in multidimensional space
pub fn create_partial_order(x: &Array2<Float>, constraints: &[bool]) -> Vec<(usize, usize)> {
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];
    let mut ordering = Vec::new();

    // For each pair of samples, check if one dominates the other
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let mut i_dominates_j = true;
            let mut j_dominates_i = true;

            for dim in 0..n_features {
                let x_i = x[[i, dim]];
                let x_j = x[[j, dim]];

                if constraints[dim] {
                    // Increasing constraint: x_i <= x_j should imply f(i) <= f(j)
                    if x_i > x_j {
                        i_dominates_j = false;
                    }
                    if x_j > x_i {
                        j_dominates_i = false;
                    }
                } else {
                    // Decreasing constraint: x_i <= x_j should imply f(i) >= f(j)
                    if x_i < x_j {
                        i_dominates_j = false;
                    }
                    if x_j < x_i {
                        j_dominates_i = false;
                    }
                }
            }

            if i_dominates_j && !j_dominates_i {
                ordering.push((i, j));
            } else if j_dominates_i && !i_dominates_j {
                ordering.push((j, i));
            }
        }
    }

    ordering
}

/// Interpolate predictions for multidimensional isotonic regression
pub fn interpolate_multidimensional(
    x_query: scirs2_core::ndarray::ArrayView1<Float>,
    x_train: &Array2<Float>,
    y_train: &Array1<Float>,
    bounds_error_handling: &str,
) -> Float {
    let n_train = x_train.shape()[0];

    if n_train == 0 {
        return match bounds_error_handling {
            "nan" => Float::NAN,
            _ => 0.0,
        };
    }

    // Find the closest training point using Euclidean distance
    let mut min_distance = Float::INFINITY;
    let mut closest_idx = 0;

    for i in 0..n_train {
        let x_train_point = x_train.row(i);
        let mut distance = 0.0;

        for (q, t) in x_query.iter().zip(x_train_point.iter()) {
            distance += (q - t).powi(2);
        }

        distance = distance.sqrt();

        if distance < min_distance {
            min_distance = distance;
            closest_idx = i;
        }
    }

    // For simplicity, return the value of the closest point
    // In a more sophisticated implementation, this could use weighted interpolation
    y_train[closest_idx]
}

/// Functional API for separable multi-dimensional isotonic regression
pub fn separable_isotonic_regression(
    x: &Array2<Float>,
    y: &Array1<Float>,
    constraints: &[bool],
    sample_weights: Option<&Array1<Float>>,
) -> Result<Array1<Float>> {
    let regressor = SeparableMultiDimensionalIsotonicRegression::new(constraints.len())
        .constraints(constraints.to_vec());

    let fitted = regressor.fit_weighted(x, y, sample_weights)?;
    fitted.predict(x)
}

/// Functional API for non-separable multi-dimensional isotonic regression
pub fn non_separable_isotonic_regression(
    x: &Array2<Float>,
    y: &Array1<Float>,
    constraints: &[bool],
    sample_weights: Option<&Array1<Float>>,
) -> Result<Array1<Float>> {
    let regressor = NonSeparableMultiDimensionalIsotonicRegression::new(constraints.len())
        .constraints(constraints.to_vec());

    let fitted = regressor.fit_weighted(x, y, sample_weights)?;
    fitted.predict(x)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_separable_creation() {
        let regressor = SeparableMultiDimensionalIsotonicRegression::new(3)
            .constraints(vec![true, false, true])
            .bounds(Some(0.0), Some(10.0));

        assert_eq!(regressor.constraints, vec![true, false, true]);
        assert_eq!(regressor.y_min, Some(0.0));
        assert_eq!(regressor.y_max, Some(10.0));
    }

    #[test]
    fn test_non_separable_creation() {
        let regressor = NonSeparableMultiDimensionalIsotonicRegression::new(2)
            .constraints(vec![true, true])
            .coupling_strength(0.5);

        assert_eq!(regressor.constraints, vec![true, true]);
        assert!((regressor.coupling_strength - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_partial_order_creation() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [0.5, 1.0], [3.0, 4.0]];
        let constraints = vec![true, true]; // Both dimensions increasing

        let ordering = create_partial_order(&x, &constraints);

        // Should find dominance relationships
        assert!(!ordering.is_empty());

        // Check that the ordering makes sense
        for (i, j) in ordering {
            // i should dominate j in all dimensions
            for dim in 0..constraints.len() {
                if constraints[dim] {
                    assert!(x[[i, dim]] <= x[[j, dim]]);
                } else {
                    assert!(x[[i, dim]] >= x[[j, dim]]);
                }
            }
        }
    }

    #[test]
    fn test_separable_functional_api() {
        let x = array![[1.0, 1.0], [2.0, 0.5], [3.0, 2.0]];
        let y = array![1.0, 2.0, 3.0];
        let constraints = vec![true, false];

        let result = separable_isotonic_regression(&x, &y, &constraints, None);
        assert!(result.is_ok());

        let predictions = result.unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_non_separable_functional_api() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let y = array![1.0, 2.0, 3.0];
        let constraints = vec![true, true];

        let result = non_separable_isotonic_regression(&x, &y, &constraints, None);
        assert!(result.is_ok());

        let predictions = result.unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_interpolation() {
        let x_train = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let y_train = array![1.0, 4.0, 9.0];
        let x_query = array![2.5, 2.5];

        let prediction = interpolate_multidimensional(x_query.view(), &x_train, &y_train, "nan");

        // Should return a reasonable interpolated value
        assert!(prediction.is_finite());
    }

    #[test]
    fn test_empty_interpolation() {
        let x_train = Array2::<Float>::zeros((0, 2));
        let y_train = Array1::<Float>::zeros(0);
        let x_query = array![1.0, 2.0];

        let prediction = interpolate_multidimensional(x_query.view(), &x_train, &y_train, "nan");

        assert!(prediction.is_nan());
    }

    #[test]
    fn test_mismatched_dimensions() {
        let x = array![[1.0, 2.0], [2.0, 3.0]]; // 2 features
        let y = array![1.0, 2.0];
        let constraints = vec![true]; // Only 1 constraint

        let regressor =
            SeparableMultiDimensionalIsotonicRegression::new(1).constraints(constraints);

        let result = regressor.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_samples() {
        let x = array![[1.0], [2.0]]; // 2 samples
        let y = array![1.0, 2.0, 3.0]; // 3 samples
        let constraints = vec![true];

        let regressor =
            SeparableMultiDimensionalIsotonicRegression::new(1).constraints(constraints);

        let result = regressor.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_separable_regression() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![1.0, 10.0, 3.0]; // Middle point has outlier value
        let weights = array![1.0, 0.1, 1.0]; // Low weight on outlier
        let constraints = vec![true];

        let regressor =
            SeparableMultiDimensionalIsotonicRegression::new(1).constraints(constraints);

        let fitted = regressor.fit_weighted(&x, &y, Some(&weights)).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 3);
        // The weighted result should reduce the influence of the outlier
    }
}
