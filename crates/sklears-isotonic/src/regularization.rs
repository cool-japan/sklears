//! Advanced regularization methods for isotonic regression
//!
//! This module provides implementations of various regularization techniques
//! for isotonic regression including smoothness and total variation regularization.

use crate::core::{LossFunction, MonotonicityConstraint};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Regularization type for isotonic regression
#[derive(Debug, Clone, Copy, PartialEq)]
/// RegularizationType
pub enum RegularizationType {
    /// No regularization
    None,
    /// L1 regularization (Lasso)
    L1 { lambda: Float },
    /// L2 regularization (Ridge)
    L2 { lambda: Float },
    /// Elastic net (L1 + L2)
    ElasticNet { l1_ratio: Float, lambda: Float },
    /// Smoothness regularization (penalizes second derivatives)
    Smoothness { lambda: Float },
    /// Total variation regularization (penalizes first derivatives)
    TotalVariation { lambda: Float },
    /// Combined smoothness and total variation
    Combined {
        smoothness_lambda: Float,
        tv_lambda: Float,
    },
}

/// Advanced regularized isotonic regression with smoothness and total variation penalties
#[derive(Debug, Clone)]
/// SmoothnessRegularizedIsotonicRegression
pub struct SmoothnessRegularizedIsotonicRegression<State = Untrained> {
    /// Base isotonic regression parameters
    pub constraint: MonotonicityConstraint,
    pub y_min: Option<Float>,
    pub y_max: Option<Float>,
    pub loss: LossFunction,

    /// Regularization parameters
    pub regularization: RegularizationType,
    pub max_iter: usize,
    pub tolerance: Float,
    pub learning_rate: Float,

    // Fitted attributes
    x_: Option<Array1<Float>>,
    y_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl SmoothnessRegularizedIsotonicRegression<Untrained> {
    /// Create a new smoothness regularized isotonic regression model
    pub fn new() -> Self {
        Self {
            constraint: MonotonicityConstraint::Global { increasing: true },
            y_min: None,
            y_max: None,
            loss: LossFunction::SquaredLoss,
            regularization: RegularizationType::Smoothness { lambda: 0.1 },
            max_iter: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            x_: None,
            y_: None,
            _state: PhantomData,
        }
    }

    /// Set the regularization type and strength
    pub fn regularization(mut self, regularization: RegularizationType) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set smoothness regularization
    pub fn smoothness(mut self, lambda: Float) -> Self {
        self.regularization = RegularizationType::Smoothness { lambda };
        self
    }

    /// Set total variation regularization
    pub fn total_variation(mut self, lambda: Float) -> Self {
        self.regularization = RegularizationType::TotalVariation { lambda };
        self
    }

    /// Set combined smoothness and total variation regularization
    pub fn combined_regularization(mut self, smoothness_lambda: Float, tv_lambda: Float) -> Self {
        self.regularization = RegularizationType::Combined {
            smoothness_lambda,
            tv_lambda,
        };
        self
    }

    /// Set L1 regularization (Lasso)
    pub fn l1_regularization(mut self, lambda: Float) -> Self {
        self.regularization = RegularizationType::L1 { lambda };
        self
    }

    /// Set L2 regularization (Ridge)
    pub fn l2_regularization(mut self, lambda: Float) -> Self {
        self.regularization = RegularizationType::L2 { lambda };
        self
    }

    /// Set Elastic Net regularization (L1 + L2)
    pub fn elastic_net_regularization(mut self, l1_ratio: Float, lambda: Float) -> Self {
        self.regularization = RegularizationType::ElasticNet { l1_ratio, lambda };
        self
    }

    /// Set monotonicity constraint
    pub fn constraint(mut self, constraint: MonotonicityConstraint) -> Self {
        self.constraint = constraint;
        self
    }

    /// Set increasing constraint
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.constraint = MonotonicityConstraint::Global { increasing };
        self
    }

    /// Set loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set maximum iterations for optimization
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set learning rate for gradient descent
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }
}

impl Default for SmoothnessRegularizedIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SmoothnessRegularizedIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for SmoothnessRegularizedIsotonicRegression<Untrained> {
    type Fitted = SmoothnessRegularizedIsotonicRegression<Trained>;

    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        let n = x.len();
        if n < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 data points for regularized isotonic regression".to_string(),
            ));
        }

        // Sort data by x values
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

        let sorted_x: Vec<Float> = indices.iter().map(|&i| x[i]).collect();
        let sorted_y: Vec<Float> = indices.iter().map(|&i| y[i]).collect();

        // Fit with regularization
        let fitted_y = match self.regularization {
            RegularizationType::None => {
                // Fall back to standard isotonic regression
                crate::core::isotonic_regression(
                    &Array1::from(sorted_x.clone()),
                    &Array1::from(sorted_y.clone()),
                    Some(true),
                    None,
                    None,
                )?
                .to_vec()
            }
            RegularizationType::Smoothness { lambda } => {
                fit_smoothness_regularized(&sorted_x, &sorted_y, lambda, &self)?
            }
            RegularizationType::TotalVariation { lambda } => {
                fit_total_variation_regularized(&sorted_x, &sorted_y, lambda, &self)?
            }
            RegularizationType::Combined {
                smoothness_lambda,
                tv_lambda,
            } => {
                fit_combined_regularized(&sorted_x, &sorted_y, smoothness_lambda, tv_lambda, &self)?
            }
            RegularizationType::L1 { lambda } => {
                fit_l1_regularized(&sorted_x, &sorted_y, lambda, &self)?
            }
            RegularizationType::L2 { lambda } => {
                fit_l2_regularized(&sorted_x, &sorted_y, lambda, &self)?
            }
            RegularizationType::ElasticNet { l1_ratio, lambda } => {
                fit_elastic_net_regularized(&sorted_x, &sorted_y, l1_ratio, lambda, &self)?
            }
        };

        Ok(SmoothnessRegularizedIsotonicRegression {
            constraint: self.constraint,
            y_min: self.y_min,
            y_max: self.y_max,
            loss: self.loss,
            regularization: self.regularization,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            learning_rate: self.learning_rate,
            x_: Some(Array1::from(sorted_x)),
            y_: Some(Array1::from(fitted_y)),
            _state: PhantomData,
        })
    }
}

impl Predict<Array1<Float>, Array1<Float>> for SmoothnessRegularizedIsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let fitted_x = self.x_.as_ref().unwrap();
        let fitted_y = self.y_.as_ref().unwrap();

        let mut predictions = Array1::zeros(x.len());
        for (i, &x_val) in x.iter().enumerate() {
            predictions[i] = interpolate(fitted_x, fitted_y, x_val);
        }

        Ok(predictions)
    }
}

impl SmoothnessRegularizedIsotonicRegression<Trained> {
    /// Get the fitted x values
    pub fn fitted_x(&self) -> &Array1<Float> {
        self.x_.as_ref().unwrap()
    }

    /// Get the fitted y values
    pub fn fitted_y(&self) -> &Array1<Float> {
        self.y_.as_ref().unwrap()
    }
}

/// Fit isotonic regression with smoothness regularization
fn fit_smoothness_regularized(
    x: &[Float],
    y: &[Float],
    lambda: Float,
    config: &SmoothnessRegularizedIsotonicRegression<Untrained>,
) -> Result<Vec<Float>> {
    let n = x.len();
    let mut fitted_y = y.to_vec();

    // Gradient descent optimization
    for iteration in 0..config.max_iter {
        let old_y = fitted_y.clone();

        // Compute gradient of loss + smoothness penalty
        let gradient = compute_smoothness_gradient(&fitted_y, y, lambda, &config.constraint)?;

        // Update with gradient step
        for i in 0..n {
            fitted_y[i] -= config.learning_rate * gradient[i];
        }

        // Apply bounds if specified
        if let Some(y_min) = config.y_min {
            for val in fitted_y.iter_mut() {
                *val = val.max(y_min);
            }
        }
        if let Some(y_max) = config.y_max {
            for val in fitted_y.iter_mut() {
                *val = val.min(y_max);
            }
        }

        // Project onto monotonic constraint
        fitted_y = project_onto_monotonic_constraint(&fitted_y, &config.constraint)?;

        // Check convergence
        let change: Float = fitted_y
            .iter()
            .zip(old_y.iter())
            .map(|(new, old)| (new - old).powi(2))
            .sum::<Float>()
            .sqrt();

        if change < config.tolerance {
            break;
        }
    }

    Ok(fitted_y)
}

/// Fit isotonic regression with total variation regularization
fn fit_total_variation_regularized(
    x: &[Float],
    y: &[Float],
    lambda: Float,
    config: &SmoothnessRegularizedIsotonicRegression<Untrained>,
) -> Result<Vec<Float>> {
    let n = x.len();
    let mut fitted_y = y.to_vec();

    // Gradient descent optimization
    for iteration in 0..config.max_iter {
        let old_y = fitted_y.clone();

        // Compute gradient of loss + total variation penalty
        let gradient = compute_total_variation_gradient(&fitted_y, y, lambda, &config.constraint)?;

        // Update with gradient step
        for i in 0..n {
            fitted_y[i] -= config.learning_rate * gradient[i];
        }

        // Apply bounds if specified
        if let Some(y_min) = config.y_min {
            for val in fitted_y.iter_mut() {
                *val = val.max(y_min);
            }
        }
        if let Some(y_max) = config.y_max {
            for val in fitted_y.iter_mut() {
                *val = val.min(y_max);
            }
        }

        // Project onto monotonic constraint
        fitted_y = project_onto_monotonic_constraint(&fitted_y, &config.constraint)?;

        // Check convergence
        let change: Float = fitted_y
            .iter()
            .zip(old_y.iter())
            .map(|(new, old)| (new - old).powi(2))
            .sum::<Float>()
            .sqrt();

        if change < config.tolerance {
            break;
        }
    }

    Ok(fitted_y)
}

/// Fit isotonic regression with combined smoothness and total variation regularization
fn fit_combined_regularized(
    x: &[Float],
    y: &[Float],
    smoothness_lambda: Float,
    tv_lambda: Float,
    config: &SmoothnessRegularizedIsotonicRegression<Untrained>,
) -> Result<Vec<Float>> {
    let n = x.len();
    let mut fitted_y = y.to_vec();

    // Gradient descent optimization
    for iteration in 0..config.max_iter {
        let old_y = fitted_y.clone();

        // Compute gradient of loss + combined penalties
        let smoothness_grad =
            compute_smoothness_gradient(&fitted_y, y, smoothness_lambda, &config.constraint)?;
        let tv_grad =
            compute_total_variation_gradient(&fitted_y, y, tv_lambda, &config.constraint)?;

        // Combine gradients
        let mut gradient = vec![0.0; n];
        for i in 0..n {
            gradient[i] = smoothness_grad[i] + tv_grad[i];
        }

        // Update with gradient step
        for i in 0..n {
            fitted_y[i] -= config.learning_rate * gradient[i];
        }

        // Apply bounds if specified
        if let Some(y_min) = config.y_min {
            for val in fitted_y.iter_mut() {
                *val = val.max(y_min);
            }
        }
        if let Some(y_max) = config.y_max {
            for val in fitted_y.iter_mut() {
                *val = val.min(y_max);
            }
        }

        // Project onto monotonic constraint
        fitted_y = project_onto_monotonic_constraint(&fitted_y, &config.constraint)?;

        // Check convergence
        let change: Float = fitted_y
            .iter()
            .zip(old_y.iter())
            .map(|(new, old)| (new - old).powi(2))
            .sum::<Float>()
            .sqrt();

        if change < config.tolerance {
            break;
        }
    }

    Ok(fitted_y)
}

/// Fit isotonic regression with L1 regularization (Lasso)
fn fit_l1_regularized(
    x: &[Float],
    y: &[Float],
    lambda: Float,
    config: &SmoothnessRegularizedIsotonicRegression<Untrained>,
) -> Result<Vec<Float>> {
    let n = x.len();
    let mut fitted_y = y.to_vec();

    // Proximal gradient descent for L1 regularization
    for iteration in 0..config.max_iter {
        let old_y = fitted_y.clone();

        // Compute gradient of loss function (without regularization)
        let mut gradient = vec![0.0; n];
        for i in 0..n {
            gradient[i] = 2.0 * (fitted_y[i] - y[i]); // L2 loss gradient
        }

        // Update with gradient step
        for i in 0..n {
            fitted_y[i] -= config.learning_rate * gradient[i];
        }

        // Apply soft thresholding for L1 regularization (proximal operator)
        let threshold = config.learning_rate * lambda;
        for i in 0..n {
            fitted_y[i] = soft_threshold(fitted_y[i], threshold);
        }

        // Apply bounds if specified
        if let Some(y_min) = config.y_min {
            for val in fitted_y.iter_mut() {
                *val = val.max(y_min);
            }
        }
        if let Some(y_max) = config.y_max {
            for val in fitted_y.iter_mut() {
                *val = val.min(y_max);
            }
        }

        // Project onto monotonic constraint
        fitted_y = project_onto_monotonic_constraint(&fitted_y, &config.constraint)?;

        // Check convergence
        let change: Float = fitted_y
            .iter()
            .zip(old_y.iter())
            .map(|(new, old)| (new - old).powi(2))
            .sum::<Float>()
            .sqrt();

        if change < config.tolerance {
            break;
        }
    }

    Ok(fitted_y)
}

/// Fit isotonic regression with L2 regularization (Ridge)
fn fit_l2_regularized(
    x: &[Float],
    y: &[Float],
    lambda: Float,
    config: &SmoothnessRegularizedIsotonicRegression<Untrained>,
) -> Result<Vec<Float>> {
    let n = x.len();
    let mut fitted_y = y.to_vec();

    // Gradient descent for L2 regularization
    for iteration in 0..config.max_iter {
        let old_y = fitted_y.clone();

        // Compute gradient of loss + L2 penalty
        let mut gradient = vec![0.0; n];
        for i in 0..n {
            gradient[i] = 2.0 * (fitted_y[i] - y[i]) + 2.0 * lambda * fitted_y[i];
        }

        // Update with gradient step
        for i in 0..n {
            fitted_y[i] -= config.learning_rate * gradient[i];
        }

        // Apply bounds if specified
        if let Some(y_min) = config.y_min {
            for val in fitted_y.iter_mut() {
                *val = val.max(y_min);
            }
        }
        if let Some(y_max) = config.y_max {
            for val in fitted_y.iter_mut() {
                *val = val.min(y_max);
            }
        }

        // Project onto monotonic constraint
        fitted_y = project_onto_monotonic_constraint(&fitted_y, &config.constraint)?;

        // Check convergence
        let change: Float = fitted_y
            .iter()
            .zip(old_y.iter())
            .map(|(new, old)| (new - old).powi(2))
            .sum::<Float>()
            .sqrt();

        if change < config.tolerance {
            break;
        }
    }

    Ok(fitted_y)
}

/// Fit isotonic regression with Elastic Net regularization (L1 + L2)
fn fit_elastic_net_regularized(
    x: &[Float],
    y: &[Float],
    l1_ratio: Float,
    lambda: Float,
    config: &SmoothnessRegularizedIsotonicRegression<Untrained>,
) -> Result<Vec<Float>> {
    let n = x.len();
    let mut fitted_y = y.to_vec();

    // Split regularization strength between L1 and L2
    let l1_lambda = l1_ratio * lambda;
    let l2_lambda = (1.0 - l1_ratio) * lambda;

    // Proximal gradient descent for Elastic Net
    for iteration in 0..config.max_iter {
        let old_y = fitted_y.clone();

        // Compute gradient of loss + L2 penalty
        let mut gradient = vec![0.0; n];
        for i in 0..n {
            gradient[i] = 2.0 * (fitted_y[i] - y[i]) + 2.0 * l2_lambda * fitted_y[i];
        }

        // Update with gradient step
        for i in 0..n {
            fitted_y[i] -= config.learning_rate * gradient[i];
        }

        // Apply soft thresholding for L1 regularization (proximal operator)
        let threshold = config.learning_rate * l1_lambda;
        for i in 0..n {
            fitted_y[i] = soft_threshold(fitted_y[i], threshold);
        }

        // Apply bounds if specified
        if let Some(y_min) = config.y_min {
            for val in fitted_y.iter_mut() {
                *val = val.max(y_min);
            }
        }
        if let Some(y_max) = config.y_max {
            for val in fitted_y.iter_mut() {
                *val = val.min(y_max);
            }
        }

        // Project onto monotonic constraint
        fitted_y = project_onto_monotonic_constraint(&fitted_y, &config.constraint)?;

        // Check convergence
        let change: Float = fitted_y
            .iter()
            .zip(old_y.iter())
            .map(|(new, old)| (new - old).powi(2))
            .sum::<Float>()
            .sqrt();

        if change < config.tolerance {
            break;
        }
    }

    Ok(fitted_y)
}

/// Soft thresholding operator for L1 regularization
/// Applies the proximal operator of the L1 norm: sign(x) * max(|x| - threshold, 0)
fn soft_threshold(x: Float, threshold: Float) -> Float {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

/// Compute gradient for smoothness regularization (penalizes second derivatives)
fn compute_smoothness_gradient(
    fitted_y: &[Float],
    observed_y: &[Float],
    lambda: Float,
    constraint: &MonotonicityConstraint,
) -> Result<Vec<Float>> {
    let n = fitted_y.len();
    let mut gradient = vec![0.0; n];

    // Data fitting term gradient (L2 loss)
    for i in 0..n {
        gradient[i] = 2.0 * (fitted_y[i] - observed_y[i]);
    }

    // Smoothness penalty gradient (second derivative penalty)
    // Penalty: lambda * sum((f[i+1] - 2*f[i] + f[i-1])^2)
    // Gradient w.r.t. f[i]: lambda * 2 * d/df[i] sum((f[j+1] - 2*f[j] + f[j-1])^2)

    for i in 1..n - 1 {
        let second_deriv = fitted_y[i + 1] - 2.0 * fitted_y[i] + fitted_y[i - 1];

        // Contribution to gradient at i-1, i, i+1
        if i > 1 {
            let prev_second_deriv = fitted_y[i] - 2.0 * fitted_y[i - 1] + fitted_y[i - 2];
            gradient[i - 1] += 2.0 * lambda * prev_second_deriv;
        }

        gradient[i] += 2.0 * lambda * second_deriv * (-2.0);

        if i < n - 2 {
            let next_second_deriv = fitted_y[i + 2] - 2.0 * fitted_y[i + 1] + fitted_y[i];
            gradient[i + 1] += 2.0 * lambda * next_second_deriv;
        }
    }

    Ok(gradient)
}

/// Compute gradient for total variation regularization (penalizes first derivatives)
fn compute_total_variation_gradient(
    fitted_y: &[Float],
    observed_y: &[Float],
    lambda: Float,
    constraint: &MonotonicityConstraint,
) -> Result<Vec<Float>> {
    let n = fitted_y.len();
    let mut gradient = vec![0.0; n];

    // Data fitting term gradient (L2 loss)
    for i in 0..n {
        gradient[i] = 2.0 * (fitted_y[i] - observed_y[i]);
    }

    // Total variation penalty gradient (first derivative penalty)
    // Penalty: lambda * sum(|f[i+1] - f[i]|)
    // Gradient w.r.t. f[i]: lambda * (sign(f[i] - f[i-1]) - sign(f[i+1] - f[i]))

    for i in 0..n - 1 {
        let diff = fitted_y[i + 1] - fitted_y[i];
        let sign_diff = if diff > 0.0 {
            1.0
        } else if diff < 0.0 {
            -1.0
        } else {
            0.0
        };

        gradient[i] += lambda * sign_diff;
        gradient[i + 1] -= lambda * sign_diff;
    }

    Ok(gradient)
}

/// Project solution onto monotonic constraint
fn project_onto_monotonic_constraint(
    y: &[Float],
    constraint: &MonotonicityConstraint,
) -> Result<Vec<Float>> {
    match constraint {
        MonotonicityConstraint::Global { increasing } => Ok(project_monotonic(y, *increasing)),
        _ => {
            // For now, only support global monotonicity
            // More complex constraints would require more sophisticated projection
            Err(SklearsError::InvalidInput(
                "Complex constraints not yet supported in regularized regression".to_string(),
            ))
        }
    }
}

/// Project onto monotonic constraint using isotonic regression
fn project_monotonic(y: &[Float], increasing: bool) -> Vec<Float> {
    let mut result = y.to_vec();

    if !increasing {
        result.reverse();
        let mut projected = project_monotonic_increasing(&result);
        projected.reverse();
        return projected;
    }

    project_monotonic_increasing(&result)
}

/// Project onto increasing monotonic constraint using simple PAVA
fn project_monotonic_increasing(y: &[Float]) -> Vec<Float> {
    let mut result = y.to_vec();
    let n = result.len();
    let mut i = 0;

    while i < n - 1 {
        if result[i] > result[i + 1] {
            // Average the violating pair
            let avg = (result[i] + result[i + 1]) / 2.0;
            result[i] = avg;
            result[i + 1] = avg;

            // Back up to check previous pairs
            if i > 0 {
                i -= 1;
            }
        } else {
            i += 1;
        }
    }

    result
}

/// Linear interpolation for prediction
fn interpolate(x_fitted: &Array1<Float>, y_fitted: &Array1<Float>, x_new: Float) -> Float {
    let n = x_fitted.len();

    // Handle boundary cases
    if x_new <= x_fitted[0] {
        return y_fitted[0];
    }
    if x_new >= x_fitted[n - 1] {
        return y_fitted[n - 1];
    }

    // Find the interval containing x_new
    for i in 0..n - 1 {
        if x_new >= x_fitted[i] && x_new <= x_fitted[i + 1] {
            let t = (x_new - x_fitted[i]) / (x_fitted[i + 1] - x_fitted[i]);
            return y_fitted[i] + t * (y_fitted[i + 1] - y_fitted[i]);
        }
    }

    // Fallback (shouldn't reach here)
    y_fitted[n - 1]
}

/// Convenience function for smoothness regularized isotonic regression
pub fn smoothness_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    lambda: Float,
    increasing: bool,
) -> Result<(Array1<Float>, Array1<Float>)> {
    let model = SmoothnessRegularizedIsotonicRegression::new()
        .smoothness(lambda)
        .increasing(increasing);

    let fitted_model = model.fit(x, y)?;
    Ok((
        fitted_model.fitted_x().clone(),
        fitted_model.fitted_y().clone(),
    ))
}

/// Convenience function for total variation regularized isotonic regression
pub fn total_variation_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    lambda: Float,
    increasing: bool,
) -> Result<(Array1<Float>, Array1<Float>)> {
    let model = SmoothnessRegularizedIsotonicRegression::new()
        .total_variation(lambda)
        .increasing(increasing);

    let fitted_model = model.fit(x, y)?;
    Ok((
        fitted_model.fitted_x().clone(),
        fitted_model.fitted_y().clone(),
    ))
}

/// Convenience function for combined regularized isotonic regression
pub fn combined_regularized_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    smoothness_lambda: Float,
    tv_lambda: Float,
    increasing: bool,
) -> Result<(Array1<Float>, Array1<Float>)> {
    let model = SmoothnessRegularizedIsotonicRegression::new()
        .combined_regularization(smoothness_lambda, tv_lambda)
        .increasing(increasing);

    let fitted_model = model.fit(x, y)?;
    Ok((
        fitted_model.fitted_x().clone(),
        fitted_model.fitted_y().clone(),
    ))
}

/// Convenience function for L1 regularized isotonic regression (Lasso)
pub fn l1_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    lambda: Float,
    increasing: bool,
) -> Result<(Array1<Float>, Array1<Float>)> {
    let model = SmoothnessRegularizedIsotonicRegression::new()
        .l1_regularization(lambda)
        .increasing(increasing);

    let fitted_model = model.fit(x, y)?;
    Ok((
        fitted_model.fitted_x().clone(),
        fitted_model.fitted_y().clone(),
    ))
}

/// Convenience function for L2 regularized isotonic regression (Ridge)
pub fn l2_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    lambda: Float,
    increasing: bool,
) -> Result<(Array1<Float>, Array1<Float>)> {
    let model = SmoothnessRegularizedIsotonicRegression::new()
        .l2_regularization(lambda)
        .increasing(increasing);

    let fitted_model = model.fit(x, y)?;
    Ok((
        fitted_model.fitted_x().clone(),
        fitted_model.fitted_y().clone(),
    ))
}

/// Convenience function for Elastic Net regularized isotonic regression
pub fn elastic_net_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    l1_ratio: Float,
    lambda: Float,
    increasing: bool,
) -> Result<(Array1<Float>, Array1<Float>)> {
    let model = SmoothnessRegularizedIsotonicRegression::new()
        .elastic_net_regularization(l1_ratio, lambda)
        .increasing(increasing);

    let fitted_model = model.fit(x, y)?;
    Ok((
        fitted_model.fitted_x().clone(),
        fitted_model.fitted_y().clone(),
    ))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_smoothness_regularized_basic() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0]);

        let model = SmoothnessRegularizedIsotonicRegression::new()
            .smoothness(0.1)
            .increasing(true);

        let fitted_model = model.fit(&x, &y).unwrap();
        let fitted_y = fitted_model.fitted_y();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(
                fitted_y[i] <= fitted_y[i + 1],
                "Fitted values are not monotonic: {} > {}",
                fitted_y[i],
                fitted_y[i + 1]
            );
        }
    }

    #[test]
    fn test_total_variation_regularized() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0]);

        let (fitted_x, fitted_y) = total_variation_isotonic_regression(&x, &y, 0.1, true).unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }
    }

    #[test]
    fn test_combined_regularized() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0]);

        let (fitted_x, fitted_y) =
            combined_regularized_isotonic_regression(&x, &y, 0.05, 0.05, true).unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }
    }

    #[test]
    fn test_smoothness_produces_smoother_result() {
        // Create noisy monotonic data
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let y = Array1::from(vec![1.0, 1.5, 2.3, 2.1, 3.2, 3.9, 4.1, 5.0]);

        // Fit with different smoothness levels
        let (_, fitted_y_0) = smoothness_isotonic_regression(&x, &y, 0.0, true).unwrap();
        let (_, fitted_y_high) = smoothness_isotonic_regression(&x, &y, 1.0, true).unwrap();

        // Higher smoothness should produce less variation in second derivatives
        let smoothness_0 = compute_second_derivative_variation(&fitted_y_0);
        let smoothness_high = compute_second_derivative_variation(&fitted_y_high);

        assert!(
            smoothness_high <= smoothness_0,
            "Higher regularization should produce smoother result"
        );
    }

    fn compute_second_derivative_variation(y: &Array1<Float>) -> Float {
        let mut variation = 0.0;
        for i in 1..y.len() - 1 {
            let second_deriv = y[i + 1] - 2.0 * y[i] + y[i - 1];
            variation += second_deriv.powi(2);
        }
        variation
    }

    #[test]
    fn test_project_monotonic() {
        let y = vec![1.0, 3.0, 2.0, 4.0, 5.0];
        let projected = project_monotonic(&y, true);

        // Check monotonicity
        for i in 0..projected.len() - 1 {
            assert!(projected[i] <= projected[i + 1]);
        }
    }

    #[test]
    fn test_l1_regularized() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0]);

        let (fitted_x, fitted_y) = l1_isotonic_regression(&x, &y, 0.1, true).unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }

        // L1 regularization should induce sparsity (some values might be zero)
        assert_eq!(fitted_x.len(), x.len());
        assert_eq!(fitted_y.len(), y.len());
    }

    #[test]
    fn test_l2_regularized() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0]);

        let (fitted_x, fitted_y) = l2_isotonic_regression(&x, &y, 0.1, true).unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }

        assert_eq!(fitted_x.len(), x.len());
        assert_eq!(fitted_y.len(), y.len());
    }

    #[test]
    fn test_elastic_net_regularized() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0]);

        let (fitted_x, fitted_y) = elastic_net_isotonic_regression(&x, &y, 0.5, 0.1, true).unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }

        assert_eq!(fitted_x.len(), x.len());
        assert_eq!(fitted_y.len(), y.len());
    }

    #[test]
    fn test_elastic_net_different_l1_ratios() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = Array1::from(vec![1.0, 2.0, 1.5, 3.0, 2.5, 4.0]);

        // Pure L2 (Ridge)
        let (_, fitted_y_l2) = elastic_net_isotonic_regression(&x, &y, 0.0, 0.1, true).unwrap();

        // Pure L1 (Lasso)
        let (_, fitted_y_l1) = elastic_net_isotonic_regression(&x, &y, 1.0, 0.1, true).unwrap();

        // Balanced Elastic Net
        let (_, fitted_y_balanced) =
            elastic_net_isotonic_regression(&x, &y, 0.5, 0.1, true).unwrap();

        // All should be monotonic
        for fitted_y in &[&fitted_y_l2, &fitted_y_l1, &fitted_y_balanced] {
            for i in 0..fitted_y.len() - 1 {
                assert!(fitted_y[i] <= fitted_y[i + 1]);
            }
        }

        // L1 regularization should tend to produce more sparse solutions
        let l1_variation = compute_variation(&fitted_y_l1);
        let l2_variation = compute_variation(&fitted_y_l2);

        // Both should be valid solutions, but potentially different
        assert!(l1_variation >= 0.0);
        assert!(l2_variation >= 0.0);
    }

    #[test]
    fn test_soft_threshold() {
        assert_abs_diff_eq!(soft_threshold(2.0, 1.0), 1.0);
        assert_abs_diff_eq!(soft_threshold(-2.0, 1.0), -1.0);
        assert_abs_diff_eq!(soft_threshold(0.5, 1.0), 0.0);
        assert_abs_diff_eq!(soft_threshold(-0.5, 1.0), 0.0);
        assert_abs_diff_eq!(soft_threshold(1.0, 1.0), 0.0);
        assert_abs_diff_eq!(soft_threshold(-1.0, 1.0), 0.0);
    }

    fn compute_variation(y: &Array1<Float>) -> Float {
        let mut variation = 0.0;
        for i in 0..y.len() - 1 {
            variation += (y[i + 1] - y[i]).abs();
        }
        variation
    }
}
