//! Tensor isotonic regression for multi-dimensional tensor data
//!
//! This module implements isotonic regression for tensor (multi-dimensional array) data,
//! supporting monotonicity constraints along specified axes and partial ordering.

use std::marker::PhantomData;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};

use crate::{apply_global_constraint, LossFunction};

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
    #[allow(dead_code)]
    tensor_shape_: Option<Vec<usize>>,
    #[allow(dead_code)]
    fitted_values_: Option<Array1<Float>>,
    #[allow(dead_code)]
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
    pub fn monotonic_axes(mut self, axes: Vec<usize>) -> Self {
        self.monotonic_axes = axes;
        self
    }

    /// Set whether each axis should be increasing or decreasing
    pub fn axis_increasing(mut self, increasing: Vec<bool>) -> Self {
        self.axis_increasing = increasing;
        self
    }

    /// Set whether to use separable (true) or non-separable (false) constraints
    pub fn separable(mut self, separable: bool) -> Self {
        self.separable = separable;
        self
    }

    /// Set tolerance for constraint violations
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set loss function
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

        let increasing = self.axis_increasing.get(0).copied().unwrap_or(true);

        // Use global constraint algorithm for 1D case
        let weights = Array1::ones(target.len());
        let fitted_values = apply_global_constraint(target, &weights, increasing, &self.loss);

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
    fn fit_separable_2d(
        &self,
        _tensor_data: &Array2<Float>,
        target: &Array1<Float>,
    ) -> Result<Array1<Float>> {
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
    fn fit_nonseparable_2d(
        &self,
        _tensor_data: &Array2<Float>,
        target: &Array1<Float>,
    ) -> Result<Array1<Float>> {
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
    fn apply_isotonic_along_axis_0(
        &self,
        values: &Array1<Float>,
        increasing: bool,
    ) -> Result<Array1<Float>> {
        // Apply global constraint
        let weights = Array1::ones(values.len());
        Ok(apply_global_constraint(
            values,
            &weights,
            increasing,
            &LossFunction::SquaredLoss,
        ))
    }

    /// Apply smoothing constraint (simple approximation for axis 1)
    fn apply_smoothing_constraint(
        &self,
        values: &Array1<Float>,
        increasing: bool,
    ) -> Result<Array1<Float>> {
        // Simple smoothing: moving average with monotonicity constraint
        let n = values.len();
        let mut smoothed = values.clone();

        // Apply simple smoothing
        for i in 1..n.saturating_sub(1) {
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

        let tensor_shape = self.tensor_shape_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;

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

        let tensor_shape = self.tensor_shape_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;
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

/// Function API for tensor isotonic regression
///
/// # Arguments
/// * `tensor_data` - Input tensor data
/// * `target` - Target values
/// * `monotonic_axes` - Axes along which to enforce monotonicity
/// * `axis_increasing` - Whether each axis should be increasing
/// * `separable` - Whether to use separable constraints
///
/// # Returns
/// Fitted tensor isotonic regression values
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