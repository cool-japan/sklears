//! Regularized isotonic regression with L1/L2 regularization
//!
//! This module implements isotonic regression with regularization terms to control smoothness
//! and overfitting, supporting both L1 (Lasso) and L2 (Ridge) regularization.
//!
//! ## Features
//!
//! - **L1 Regularization (Lasso)**: Promotes sparsity in the fitted function
//! - **L2 Regularization (Ridge)**: Promotes smoothness in the fitted function
//! - **Elastic Net**: Combination of L1 and L2 regularization
//! - **Bounded constraints**: Optional upper and lower bounds on output values
//! - **Multiple loss functions**: Support for various robust loss functions
//! - **Proximal gradient descent**: Efficient optimization with regularization
//!
//! ## Mathematical Formulation
//!
//! The regularized isotonic regression solves:
//!
//! ```text
//! minimize: L(f, y) + λ₁||f||₁ + λ₂||f||₂²
//! subject to: f is monotonic and y_min ≤ f ≤ y_max
//! ```
//!
//! Where:
//! - `L(f, y)` is the loss function (squared, absolute, Huber, or quantile)
//! - `λ₁` is the L1 regularization strength
//! - `λ₂` is the L2 regularization strength
//!
//! ## Examples
//!
//! ```rust,ignore
//! use sklears_isotonic::regularized::regularized_isotonic::*;
//! use scirs2_core::ndarray::Array1;
//!
//! let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//! let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 6.0]);
//!
//! // Create regularized isotonic regression with both L1 and L2 penalties
//! let model = RegularizedIsotonicRegression::new()
//!     .increasing(true)
//!     .l1_alpha(0.1)    // L1 penalty for sparsity
//!     .l2_alpha(0.05)   // L2 penalty for smoothness
//!     .tolerance(1e-6)
//!     .max_iterations(1000);
//!
//! let fitted = model.fit(&x, &y).unwrap();
//! let predictions = fitted.predict(&x).unwrap();
//! ```

use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::{apply_global_constraint, LossFunction, MonotonicityConstraint};

/// Regularized isotonic regression with L1/L2 regularization
///
/// Implements isotonic regression with regularization terms to control smoothness
/// and overfitting, supporting both L1 (Lasso) and L2 (Ridge) regularization.
#[derive(Debug, Clone)]
/// RegularizedIsotonicRegression
pub struct RegularizedIsotonicRegression<State = Untrained> {
    /// Whether the function should be increasing
    pub increasing: bool,
    /// L1 regularization strength (Lasso penalty)
    pub l1_alpha: Float,
    /// L2 regularization strength (Ridge penalty)
    pub l2_alpha: Float,
    /// Tolerance for convergence
    pub tolerance: Float,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Lower bound on the output
    pub y_min: Option<Float>,
    /// Upper bound on the output
    pub y_max: Option<Float>,
    /// Loss function
    pub loss: LossFunction,

    // Fitted attributes
    x_: Option<Array1<Float>>,
    y_: Option<Array1<Float>>,
    regularization_path_: Option<Vec<Array1<Float>>>,

    _state: PhantomData<State>,
}

impl RegularizedIsotonicRegression<Untrained> {
    /// Create a new regularized isotonic regression model
    ///
    /// # Returns
    ///
    /// A new `RegularizedIsotonicRegression` instance with default parameters:
    /// - Increasing monotonicity
    /// - No L1 or L2 regularization
    /// - Tolerance of 1e-6
    /// - Maximum 1000 iterations
    /// - No bounds on output
    /// - Squared loss function
    pub fn new() -> Self {
        Self {
            increasing: true,
            l1_alpha: 0.0,
            l2_alpha: 0.0,
            tolerance: 1e-6,
            max_iterations: 1000,
            y_min: None,
            y_max: None,
            loss: LossFunction::SquaredLoss,
            x_: None,
            y_: None,
            regularization_path_: None,
            _state: PhantomData,
        }
    }

    /// Set whether the function should be increasing
    ///
    /// # Arguments
    ///
    /// * `increasing` - If true, enforces increasing monotonicity; if false, decreasing
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sklears_isotonic::regularized::regularized_isotonic::RegularizedIsotonicRegression;
    /// let model = RegularizedIsotonicRegression::new().increasing(false); // Decreasing
    /// ```
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set L1 regularization strength (Lasso penalty)
    ///
    /// L1 regularization promotes sparsity in the fitted function by penalizing
    /// the sum of absolute values of the fitted values.
    ///
    /// # Arguments
    ///
    /// * `l1_alpha` - L1 regularization strength (≥ 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sklears_isotonic::regularized::regularized_isotonic::RegularizedIsotonicRegression;
    /// let model = RegularizedIsotonicRegression::new().l1_alpha(0.1);
    /// ```
    pub fn l1_alpha(mut self, l1_alpha: Float) -> Self {
        self.l1_alpha = l1_alpha;
        self
    }

    /// Set L2 regularization strength (Ridge penalty)
    ///
    /// L2 regularization promotes smoothness in the fitted function by penalizing
    /// the sum of squared values of the fitted values.
    ///
    /// # Arguments
    ///
    /// * `l2_alpha` - L2 regularization strength (≥ 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sklears_isotonic::regularized::regularized_isotonic::RegularizedIsotonicRegression;
    /// let model = RegularizedIsotonicRegression::new().l2_alpha(0.05);
    /// ```
    pub fn l2_alpha(mut self, l2_alpha: Float) -> Self {
        self.l2_alpha = l2_alpha;
        self
    }

    /// Set tolerance for convergence
    ///
    /// The optimization algorithm stops when the maximum change in fitted values
    /// between iterations is below this tolerance.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Convergence tolerance (> 0)
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set maximum number of iterations
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - Maximum iterations for the optimization algorithm
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set lower bound on the output
    ///
    /// If specified, all fitted values will be at least `y_min`.
    ///
    /// # Arguments
    ///
    /// * `y_min` - Lower bound on fitted values
    pub fn y_min(mut self, y_min: Float) -> Self {
        self.y_min = Some(y_min);
        self
    }

    /// Set upper bound on the output
    ///
    /// If specified, all fitted values will be at most `y_max`.
    ///
    /// # Arguments
    ///
    /// * `y_max` - Upper bound on fitted values
    pub fn y_max(mut self, y_max: Float) -> Self {
        self.y_max = Some(y_max);
        self
    }

    /// Set loss function
    ///
    /// # Arguments
    ///
    /// * `loss` - Loss function to optimize
    ///
    /// # Supported Loss Functions
    ///
    /// - `SquaredLoss`: L2 loss, sensitive to outliers
    /// - `AbsoluteLoss`: L1 loss, robust to outliers
    /// - `HuberLoss`: Combination of L2 and L1 loss
    /// - `QuantileLoss`: Asymmetric loss for quantile regression
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }
}

impl Default for RegularizedIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RegularizedIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for RegularizedIsotonicRegression<Trained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for RegularizedIsotonicRegression<Untrained> {
    type Fitted = RegularizedIsotonicRegression<Trained>;

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

        // Fit regularized isotonic regression
        let fitted_values = self.fit_regularized_isotonic(&x_sorted, &y_sorted)?;

        Ok(RegularizedIsotonicRegression {
            increasing: self.increasing,
            l1_alpha: self.l1_alpha,
            l2_alpha: self.l2_alpha,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            y_min: self.y_min,
            y_max: self.y_max,
            loss: self.loss,
            x_: Some(x_sorted),
            y_: Some(fitted_values),
            regularization_path_: None,
            _state: PhantomData,
        })
    }
}

impl RegularizedIsotonicRegression<Untrained> {
    /// Fit regularized isotonic regression using proximal gradient descent
    ///
    /// This method uses proximal gradient descent to solve the regularized optimization problem.
    /// The algorithm alternates between gradient descent steps and proximal operators for
    /// L1 and L2 regularization, followed by projection onto the isotonic constraint.
    fn fit_regularized_isotonic(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n = y.len();
        let mut fitted_values = y.clone();

        // Use proximal gradient descent for optimization
        for iteration in 0..self.max_iterations {
            let prev_values = fitted_values.clone();

            // Compute gradient of the loss function
            let gradient = self.compute_loss_gradient(&fitted_values, y);

            // Proximal gradient step
            let step_size = 0.01; // Adaptive step size could be implemented

            // Gradient descent step
            for i in 0..n {
                fitted_values[i] -= step_size * gradient[i];
            }

            // Apply L2 regularization (shrinkage)
            if self.l2_alpha > 0.0 {
                let shrinkage_factor = 1.0 / (1.0 + step_size * self.l2_alpha);
                for i in 0..n {
                    fitted_values[i] *= shrinkage_factor;
                }
            }

            // Apply L1 regularization (soft thresholding)
            if self.l1_alpha > 0.0 {
                let threshold = step_size * self.l1_alpha;
                for i in 0..n {
                    fitted_values[i] = self.soft_threshold(fitted_values[i], threshold);
                }
            }

            // Apply isotonic constraint
            fitted_values = self.apply_isotonic_constraint(&fitted_values)?;

            // Apply bounds if specified
            if let Some(y_min) = self.y_min {
                for i in 0..n {
                    fitted_values[i] = fitted_values[i].max(y_min);
                }
            }
            if let Some(y_max) = self.y_max {
                for i in 0..n {
                    fitted_values[i] = fitted_values[i].min(y_max);
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

    /// Compute gradient of the loss function
    ///
    /// Returns the gradient of the loss function with respect to the fitted values.
    fn compute_loss_gradient(&self, fitted: &Array1<Float>, y: &Array1<Float>) -> Array1<Float> {
        let n = fitted.len();
        let mut gradient = Array1::zeros(n);

        match self.loss {
            LossFunction::SquaredLoss => {
                for i in 0..n {
                    gradient[i] = 2.0 * (fitted[i] - y[i]);
                }
            }
            LossFunction::AbsoluteLoss => {
                for i in 0..n {
                    gradient[i] = if fitted[i] > y[i] {
                        1.0
                    } else if fitted[i] < y[i] {
                        -1.0
                    } else {
                        0.0
                    };
                }
            }
            LossFunction::HuberLoss { delta } => {
                for i in 0..n {
                    let diff = fitted[i] - y[i];
                    if diff.abs() <= delta {
                        gradient[i] = diff;
                    } else {
                        gradient[i] = delta * diff.signum();
                    }
                }
            }
            LossFunction::QuantileLoss { quantile } => {
                for i in 0..n {
                    let diff = fitted[i] - y[i];
                    gradient[i] = if diff > 0.0 { quantile } else { quantile - 1.0 };
                }
            }
        }

        gradient
    }

    /// Soft thresholding operator for L1 regularization
    ///
    /// The soft thresholding operator is the proximal operator for the L1 norm.
    /// It applies the transformation:
    /// - If x > threshold: x - threshold
    /// - If x < -threshold: x + threshold
    /// - Otherwise: 0
    fn soft_threshold(&self, x: Float, threshold: Float) -> Float {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    }

    /// Apply isotonic constraint
    ///
    /// Projects the current fitted values onto the isotonic constraint set.
    fn apply_isotonic_constraint(&self, values: &Array1<Float>) -> Result<Array1<Float>> {
        let weights = Array1::ones(values.len());
        let constraint = MonotonicityConstraint::Global {
            increasing: self.increasing,
        };
        apply_global_constraint(values, constraint, Some(&weights))
    }
}

impl Predict<Array1<Float>, Array1<Float>> for RegularizedIsotonicRegression<Trained> {
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
        }

        Ok(predictions)
    }
}

impl RegularizedIsotonicRegression<Trained> {
    /// Get the regularization path if available
    ///
    /// Returns the sequence of fitted values at different stages of the optimization,
    /// which can be useful for analyzing the regularization effect.
    pub fn regularization_path(&self) -> Option<&Vec<Array1<Float>>> {
        self.regularization_path_.as_ref()
    }

    /// Get the effective regularization strength
    ///
    /// Returns a tuple of (L1 strength, L2 strength) that was used during fitting.
    pub fn regularization_strength(&self) -> (Float, Float) {
        (self.l1_alpha, self.l2_alpha)
    }

    /// Check if the model uses any regularization
    ///
    /// Returns true if either L1 or L2 regularization is applied.
    pub fn is_regularized(&self) -> bool {
        self.l1_alpha > 0.0 || self.l2_alpha > 0.0
    }
}

/// Convenient function for regularized isotonic regression
///
/// This function provides a simple interface for applying regularized isotonic regression
/// with specified L1 and L2 regularization strengths.
///
/// # Arguments
/// * `x` - Input features
/// * `y` - Target values
/// * `increasing` - Whether the function should be increasing
/// * `l1_alpha` - L1 regularization strength
/// * `l2_alpha` - L2 regularization strength
///
/// # Returns
/// Fitted regularized isotonic regression values
///
/// # Examples
///
/// ```rust
/// use sklears_isotonic::regularized::regularized_isotonic::regularized_isotonic_regression;
/// use scirs2_core::ndarray::Array1;
///
/// let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 6.0]);
///
/// let fitted = regularized_isotonic_regression(
///     &x, &y, true, 0.1, 0.05
/// ).unwrap();
/// ```
pub fn regularized_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    l1_alpha: Float,
    l2_alpha: Float,
) -> Result<Array1<Float>> {
    let regressor = RegularizedIsotonicRegression::new()
        .increasing(increasing)
        .l1_alpha(l1_alpha)
        .l2_alpha(l2_alpha);

    let fitted = regressor.fit(x, y)?;
    fitted.predict(x)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_regularized_isotonic_regression() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 6.0]);

        let model = RegularizedIsotonicRegression::new()
            .increasing(true)
            .l1_alpha(0.1)
            .l2_alpha(0.05);

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Check that predictions are monotonic
        for i in 1..predictions.len() {
            assert!(predictions[i] >= predictions[i - 1]);
        }
    }

    #[test]
    fn test_l1_regularization_only() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 6.0]);

        let model = RegularizedIsotonicRegression::new()
            .increasing(true)
            .l1_alpha(0.5)
            .l2_alpha(0.0);

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 5);
        assert!(fitted.is_regularized());
    }

    #[test]
    fn test_l2_regularization_only() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 6.0]);

        let model = RegularizedIsotonicRegression::new()
            .increasing(true)
            .l1_alpha(0.0)
            .l2_alpha(0.1);

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 5);
        assert!(fitted.is_regularized());
    }

    #[test]
    fn test_bounded_regularized_regression() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 6.0]);

        let model = RegularizedIsotonicRegression::new()
            .increasing(true)
            .l1_alpha(0.1)
            .l2_alpha(0.05)
            .y_min(0.5)
            .y_max(5.5);

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Check bounds
        for &pred in predictions.iter() {
            assert!(pred >= 0.5);
            assert!(pred <= 5.5);
        }
    }

    #[test]
    fn test_decreasing_regularized_regression() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![6.0, 4.0, 5.0, 2.0, 1.0]);

        let model = RegularizedIsotonicRegression::new()
            .increasing(false)
            .l1_alpha(0.1)
            .l2_alpha(0.05);

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Check that predictions are monotonic decreasing
        for i in 1..predictions.len() {
            assert!(predictions[i] <= predictions[i - 1]);
        }
    }

    #[test]
    fn test_regularized_isotonic_convenience_function() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 6.0]);

        let predictions = regularized_isotonic_regression(&x, &y, true, 0.1, 0.05).unwrap();

        assert_eq!(predictions.len(), 5);
        // Check that predictions are monotonic
        for i in 1..predictions.len() {
            assert!(predictions[i] >= predictions[i - 1]);
        }
    }

    #[test]
    fn test_soft_threshold() {
        let model = RegularizedIsotonicRegression::new();

        assert_eq!(model.soft_threshold(1.5, 0.5), 1.0);
        assert_eq!(model.soft_threshold(-1.5, 0.5), -1.0);
        assert_eq!(model.soft_threshold(0.3, 0.5), 0.0);
        assert_eq!(model.soft_threshold(-0.3, 0.5), 0.0);
    }

    #[test]
    fn test_invalid_input() {
        let x = Array1::from(vec![1.0, 2.0, 3.0]);
        let y = Array1::from(vec![1.0, 2.0]); // Different length

        let model = RegularizedIsotonicRegression::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input() {
        let x = Array1::from(vec![]);
        let y = Array1::from(vec![]);

        let model = RegularizedIsotonicRegression::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }
}
