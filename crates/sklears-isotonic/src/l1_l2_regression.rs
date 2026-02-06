//! Regularized isotonic regression with L1/L2 regularization
//!
//! This module implements isotonic regression with regularization terms to control smoothness
//! and overfitting, supporting both L1 (Lasso) and L2 (Ridge) regularization.

use std::marker::PhantomData;
use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};

use crate::{apply_global_constraint, LossFunction};

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
    #[allow(dead_code)]
    x_: Option<Array1<Float>>,
    #[allow(dead_code)]
    y_: Option<Array1<Float>>,
    #[allow(dead_code)]
    regularization_path_: Option<Vec<Array1<Float>>>,

    _state: PhantomData<State>,
}

impl RegularizedIsotonicRegression<Untrained> {
    /// Create a new regularized isotonic regression model
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
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set L1 regularization strength (Lasso penalty)
    pub fn l1_alpha(mut self, l1_alpha: Float) -> Self {
        self.l1_alpha = l1_alpha;
        self
    }

    /// Set L2 regularization strength (Ridge penalty)
    pub fn l2_alpha(mut self, l2_alpha: Float) -> Self {
        self.l2_alpha = l2_alpha;
        self
    }

    /// Set tolerance for convergence
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set lower bound on the output
    pub fn y_min(mut self, y_min: Float) -> Self {
        self.y_min = Some(y_min);
        self
    }

    /// Set upper bound on the output
    pub fn y_max(mut self, y_max: Float) -> Self {
        self.y_max = Some(y_max);
        self
    }

    /// Set loss function
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
        indices.sort_by(|&a, &b| x[a].total_cmp(&x[b]));

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
    fn fit_regularized_isotonic(
        &self,
        _x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n = y.len();
        let mut fitted_values = y.clone();

        // Use proximal gradient descent for optimization
        for _iteration in 0..self.max_iterations {
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
    fn apply_isotonic_constraint(&self, values: &Array1<Float>) -> Result<Array1<Float>> {
        let weights = Array1::ones(values.len());
        Ok(apply_global_constraint(
            values,
            &weights,
            self.increasing,
            &LossFunction::SquaredLoss,
        ))
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

/// Function API for regularized isotonic regression
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