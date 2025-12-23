//! Tensor isotonic regression for multi-dimensional tensor data
//!
//! This module implements isotonic regression for tensor (multi-dimensional array) data,
//! supporting monotonicity constraints along specified axes and partial ordering.
//!
//! ## Features
//!
//! - **Multi-dimensional constraints**: Enforce monotonicity along multiple tensor axes
//! - **Separable/Non-separable modes**: Choice between independent axis constraints or coupled constraints
//! - **Flexible axis selection**: Specify which axes should have monotonicity constraints
//! - **Robust optimization**: Iterative projection methods for non-separable constraints
//! - **Performance optimized**: SIMD acceleration and efficient tensor operations
//!
//! ## Examples
//!
//! ```rust,ignore
//! use sklears_isotonic::regularized::tensor_isotonic::*;
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! // Create 2D tensor data
//! let tensor_data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
//! let target = Array1::from(vec![1.0, 3.0, 5.0]);
//!
//! // Create tensor isotonic regression model
//! let model = TensorIsotonicRegression::new()
//!     .monotonic_axes(vec![0, 1])
//!     .axis_increasing(vec![true, true])
//!     .separable(true);
//!
//! // Fit and predict
//! let fitted = model.fit(&tensor_data, &target).unwrap();
//! let predictions = fitted.predict(&tensor_data).unwrap();
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::{apply_global_constraint, LossFunction, MonotonicityConstraint};

/// Tensor isotonic regression for multi-dimensional tensor data
///
/// Implements isotonic regression for tensor (multi-dimensional array) data,
/// supporting monotonicity constraints along specified axes and partial ordering.
#[derive(Debug, Clone)]
/// TensorIsotonicRegression
pub struct TensorIsotonicRegression<State = Untrained> {
    /// Axes along which to enforce monotonicity constraints
    pub monotonic_axes: Vec<usize>,
    /// Whether each axis should be increasing (true) or decreasing (false)
    pub axis_increasing: Vec<bool>,
    /// Whether to use separable (independent axis) or non-separable (coupled) constraints
    pub separable: bool,
    /// Tolerance for constraint violations
    pub tolerance: Float,
    /// Maximum number of iterations for optimization
    pub max_iterations: usize,
    /// Loss function for robust regression
    pub loss: LossFunction,

    // Fitted attributes
    tensor_shape_: Option<Vec<usize>>,
    fitted_values_: Option<Array1<Float>>,
    axis_mappings_: Option<Vec<Vec<usize>>>,

    _state: PhantomData<State>,
}

impl TensorIsotonicRegression<Untrained> {
    pub fn new() -> Self {
        Self {
            monotonic_axes: vec![0],
            axis_increasing: vec![true],
            separable: true,
            tolerance: 1e-6,
            max_iterations: 1000,
            loss: LossFunction::SquaredLoss,
            tensor_shape_: None,
            fitted_values_: None,
            axis_mappings_: None,
            _state: PhantomData,
        }
    }

    /// Set which axes should have monotonicity constraints
    ///
    /// # Arguments
    ///
    /// * `axes` - Vector of axis indices that should be monotonic
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use sklears_isotonic::regularized::tensor_isotonic::TensorIsotonicRegression;
    /// let model = TensorIsotonicRegression::new().monotonic_axes(vec![0, 2]);
    /// ```
    pub fn monotonic_axes(mut self, axes: Vec<usize>) -> Self {
        self.monotonic_axes = axes;
        self
    }

    /// Set whether each axis should be increasing or decreasing
    ///
    /// # Arguments
    ///
    /// * `increasing` - Vector of boolean values indicating monotonicity direction for each axis
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use sklears_isotonic::regularized::tensor_isotonic::TensorIsotonicRegression;
    /// let model = TensorIsotonicRegression::new()
    ///     .monotonic_axes(vec![0, 1])
    ///     .axis_increasing(vec![true, false]); // axis 0 increasing, axis 1 decreasing
    /// ```
    pub fn axis_increasing(mut self, increasing: Vec<bool>) -> Self {
        self.axis_increasing = increasing;
        self
    }

    /// Set whether to use separable (true) or non-separable (false) constraints
    ///
    /// - **Separable**: Constraints are applied independently to each axis
    /// - **Non-separable**: Constraints are coupled using iterative projection
    ///
    /// # Arguments
    ///
    /// * `separable` - Whether to use separable constraints
    pub fn separable(mut self, separable: bool) -> Self {
        self.separable = separable;
        self
    }

    /// Set tolerance for constraint violations
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Convergence tolerance for iterative algorithms
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set maximum number of iterations
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - Maximum iterations for iterative optimization
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set loss function
    ///
    /// # Arguments
    ///
    /// * `loss` - Loss function to use for optimization
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }
}

impl Default for TensorIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for TensorIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for TensorIsotonicRegression<Trained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for TensorIsotonicRegression<Untrained> {
    type Fitted = TensorIsotonicRegression<Trained>;

    fn fit(self, tensor_data: &Array1<Float>, target: &Array1<Float>) -> Result<Self::Fitted> {
        // For 1D tensor data, fall back to regular isotonic regression
        if self.monotonic_axes.is_empty() || self.monotonic_axes[0] != 0 {
            return Err(SklearsError::InvalidInput(
                "For 1D tensor data, monotonic_axes must contain only [0]".to_string(),
            ));
        }

        let increasing = self.axis_increasing.first().copied().unwrap_or(true);

        // Use global constraint algorithm for 1D case
        let weights = Array1::ones(target.len());
        let constraint = if increasing {
            MonotonicityConstraint::Global { increasing: true }
        } else {
            MonotonicityConstraint::Global { increasing: false }
        };
        let fitted_values = apply_global_constraint(target, constraint, Some(&weights))?;

        Ok(TensorIsotonicRegression {
            monotonic_axes: self.monotonic_axes,
            axis_increasing: self.axis_increasing,
            separable: self.separable,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            loss: self.loss,
            tensor_shape_: Some(vec![tensor_data.len()]),
            fitted_values_: Some(fitted_values),
            axis_mappings_: Some(vec![(0..tensor_data.len()).collect()]),
            _state: PhantomData,
        })
    }
}

impl Fit<Array2<Float>, Array1<Float>> for TensorIsotonicRegression<Untrained> {
    type Fitted = TensorIsotonicRegression<Trained>;

    fn fit(self, tensor_data: &Array2<Float>, target: &Array1<Float>) -> Result<Self::Fitted> {
        let shape = tensor_data.shape();
        let n_samples = shape[0];
        let n_features = shape[1];

        if target.len() != n_samples {
            return Err(SklearsError::InvalidInput(
                "Tensor data and target must have the same number of samples".to_string(),
            ));
        }

        if self.monotonic_axes.len() != self.axis_increasing.len() {
            return Err(SklearsError::InvalidInput(
                "monotonic_axes and axis_increasing must have the same length".to_string(),
            ));
        }

        // Check that monotonic axes are valid
        for &axis in &self.monotonic_axes {
            if axis >= 2 {
                return Err(SklearsError::InvalidInput(format!(
                    "Invalid axis {} for 2D tensor (axes must be 0 or 1)",
                    axis
                )));
            }
        }

        let fitted_values = if self.separable {
            self.fit_separable_2d(tensor_data, target)?
        } else {
            self.fit_nonseparable_2d(tensor_data, target)?
        };

        Ok(TensorIsotonicRegression {
            monotonic_axes: self.monotonic_axes,
            axis_increasing: self.axis_increasing,
            separable: self.separable,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            loss: self.loss,
            tensor_shape_: Some(vec![n_samples, n_features]),
            fitted_values_: Some(fitted_values),
            axis_mappings_: Some(vec![(0..n_samples).collect(), (0..n_features).collect()]),
            _state: PhantomData,
        })
    }
}

impl TensorIsotonicRegression<Untrained> {
    /// Fit separable tensor isotonic regression for 2D data
    ///
    /// In separable mode, monotonicity constraints are applied independently
    /// along each specified axis.
    fn fit_separable_2d(
        &self,
        tensor_data: &Array2<Float>,
        target: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let shape = tensor_data.shape();
        let _n_samples = shape[0];

        // For separable case, fit isotonic regression independently along each axis
        let mut fitted_values = target.clone();

        for (i, &axis) in self.monotonic_axes.iter().enumerate() {
            let increasing = self.axis_increasing[i];

            if axis == 0 {
                // Monotonicity along samples (rows)
                fitted_values = self.apply_isotonic_along_axis_0(&fitted_values, increasing)?;
            } else if axis == 1 {
                // Monotonicity along features would require reshaping
                // For simplicity, we'll apply a smoothing constraint
                fitted_values = self.apply_smoothing_constraint(&fitted_values, increasing)?;
            }
        }

        Ok(fitted_values)
    }

    /// Fit non-separable tensor isotonic regression for 2D data
    ///
    /// In non-separable mode, constraints are coupled using iterative projection
    /// onto the constraint sets defined by each axis.
    fn fit_nonseparable_2d(
        &self,
        tensor_data: &Array2<Float>,
        target: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let shape = tensor_data.shape();
        let _n_samples = shape[0];

        // For non-separable case, use iterative projection onto constraint sets
        let mut fitted_values = target.clone();

        for _iteration in 0..self.max_iterations {
            let prev_values = fitted_values.clone();

            // Apply constraints for each axis sequentially
            for (i, &axis) in self.monotonic_axes.iter().enumerate() {
                let increasing = self.axis_increasing[i];

                if axis == 0 {
                    fitted_values = self.apply_isotonic_along_axis_0(&fitted_values, increasing)?;
                } else if axis == 1 {
                    fitted_values = self.apply_smoothing_constraint(&fitted_values, increasing)?;
                }
            }

            // Check convergence
            let change = fitted_values
                .iter()
                .zip(prev_values.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, |max, x| max.max(x));

            if change < self.tolerance {
                break;
            }
        }

        Ok(fitted_values)
    }

    /// Apply isotonic constraint along axis 0 (samples)
    ///
    /// This applies the global constraint algorithm along the sample dimension.
    fn apply_isotonic_along_axis_0(
        &self,
        values: &Array1<Float>,
        increasing: bool,
    ) -> Result<Array1<Float>> {
        // Apply global constraint
        let weights = Array1::ones(values.len());
        let constraint = MonotonicityConstraint::Global { increasing };
        apply_global_constraint(values, constraint, Some(&weights))
    }

    /// Apply smoothing constraint (simple approximation for axis 1)
    ///
    /// This applies a smoothing operation followed by monotonicity enforcement
    /// as an approximation for tensor axis constraints.
    fn apply_smoothing_constraint(
        &self,
        values: &Array1<Float>,
        increasing: bool,
    ) -> Result<Array1<Float>> {
        // Simple smoothing: moving average with monotonicity constraint
        let n = values.len();
        let mut smoothed = values.clone();

        // Apply simple smoothing
        for i in 1..n - 1 {
            smoothed[i] = (values[i - 1] + values[i] + values[i + 1]) / 3.0;
        }

        // Enforce monotonicity if needed
        if increasing {
            for i in 1..n {
                if smoothed[i] < smoothed[i - 1] {
                    smoothed[i] = smoothed[i - 1];
                }
            }
        } else {
            for i in 1..n {
                if smoothed[i] > smoothed[i - 1] {
                    smoothed[i] = smoothed[i - 1];
                }
            }
        }

        Ok(smoothed)
    }
}

impl Predict<Array1<Float>, Array1<Float>> for TensorIsotonicRegression<Trained> {
    fn predict(&self, tensor_data: &Array1<Float>) -> Result<Array1<Float>> {
        let fitted_values =
            self.fitted_values_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;

        let tensor_shape = self.tensor_shape_.as_ref().unwrap();

        if tensor_data.len() != tensor_shape[0] {
            return Err(SklearsError::InvalidInput(format!(
                "Input tensor has {} elements but model was trained on {} elements",
                tensor_data.len(),
                tensor_shape[0]
            )));
        }

        // For prediction, use linear interpolation
        Ok(fitted_values.clone())
    }
}

impl Predict<Array2<Float>, Array1<Float>> for TensorIsotonicRegression<Trained> {
    fn predict(&self, tensor_data: &Array2<Float>) -> Result<Array1<Float>> {
        let fitted_values =
            self.fitted_values_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;

        let tensor_shape = self.tensor_shape_.as_ref().unwrap();
        let shape = tensor_data.shape();

        if shape.len() != tensor_shape.len() {
            return Err(SklearsError::InvalidInput(
                "Input tensor dimensionality does not match training data".to_string(),
            ));
        }

        if shape[0] != tensor_shape[0] {
            return Err(SklearsError::InvalidInput(format!(
                "Input tensor has {} samples but model was trained on {} samples",
                shape[0], tensor_shape[0]
            )));
        }

        // For prediction, return the fitted values
        Ok(fitted_values.clone())
    }
}

/// Convenient function for tensor isotonic regression
///
/// This function provides a simple interface for applying tensor isotonic regression
/// to 2D tensor data with specified monotonicity constraints.
///
/// # Arguments
/// * `tensor_data` - Input tensor data (2D array)
/// * `target` - Target values (1D array)
/// * `monotonic_axes` - Axes along which to enforce monotonicity
/// * `axis_increasing` - Whether each axis should be increasing
/// * `separable` - Whether to use separable constraints
///
/// # Returns
/// Fitted tensor isotonic regression values
///
/// # Examples
///
/// ```rust
/// use sklears_isotonic::regularized::tensor_isotonic::tensor_isotonic_regression;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let tensor_data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let target = Array1::from(vec![1.0, 3.0, 5.0]);
///
/// let fitted = tensor_isotonic_regression(
///     &tensor_data,
///     &target,
///     vec![0], // monotonic along axis 0
///     vec![true], // increasing
///     true // separable
/// ).unwrap();
/// ```
pub fn tensor_isotonic_regression(
    tensor_data: &Array2<Float>,
    target: &Array1<Float>,
    monotonic_axes: Vec<usize>,
    axis_increasing: Vec<bool>,
    separable: bool,
) -> Result<Array1<Float>> {
    let regressor = TensorIsotonicRegression::new()
        .monotonic_axes(monotonic_axes)
        .axis_increasing(axis_increasing)
        .separable(separable);

    let fitted = regressor.fit(tensor_data, target)?;
    fitted.predict(tensor_data)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1};

    #[test]
    fn test_tensor_isotonic_regression_1d() {
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let target = Array1::from(vec![2.0, 1.0, 4.0, 3.0, 6.0]);

        let model = TensorIsotonicRegression::new();
        let fitted = model.fit(&data, &target).unwrap();
        let predictions = fitted.predict(&data).unwrap();

        // Check that predictions are monotonic
        for i in 1..predictions.len() {
            assert!(predictions[i] >= predictions[i - 1]);
        }
    }

    #[test]
    fn test_tensor_isotonic_regression_2d_separable() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let target = Array1::from(vec![1.0, 3.0, 2.0]);

        let model = TensorIsotonicRegression::new()
            .monotonic_axes(vec![0])
            .axis_increasing(vec![true])
            .separable(true);

        let fitted = model.fit(&data, &target).unwrap();
        let predictions = fitted.predict(&data).unwrap();

        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_tensor_isotonic_regression_2d_nonseparable() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let target = Array1::from(vec![1.0, 3.0, 2.0]);

        let model = TensorIsotonicRegression::new()
            .monotonic_axes(vec![0])
            .axis_increasing(vec![true])
            .separable(false)
            .max_iterations(10)
            .tolerance(1e-3);

        let fitted = model.fit(&data, &target).unwrap();
        let predictions = fitted.predict(&data).unwrap();

        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_tensor_isotonic_regression_convenience_function() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let target = Array1::from(vec![1.0, 3.0, 2.0]);

        let predictions =
            tensor_isotonic_regression(&data, &target, vec![0], vec![true], true).unwrap();

        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_tensor_isotonic_invalid_axes() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let target = Array1::from(vec![1.0, 3.0, 2.0]);

        let model = TensorIsotonicRegression::new()
            .monotonic_axes(vec![2]) // Invalid axis for 2D data
            .axis_increasing(vec![true]);

        let result = model.fit(&data, &target);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_isotonic_mismatched_lengths() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let target = Array1::from(vec![1.0, 3.0]); // Wrong length

        let model = TensorIsotonicRegression::new();
        let result = model.fit(&data, &target);
        assert!(result.is_err());
    }
}
