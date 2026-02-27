//! Core isotonic regression types and implementations
//!
//! This module provides the basic isotonic regression functionality including
//! the main `IsotonicRegression` estimator, loss functions, and core algorithms.

use crate::utils::safe_float_cmp;
use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Monotonicity constraint options
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// MonotonicityConstraint
pub enum MonotonicityConstraint {
    /// Function should be increasing
    #[default]
    Increasing,
    /// Function should be decreasing
    Decreasing,
    /// No monotonicity constraint
    None,
    /// Global monotonicity constraint (for backward compatibility)
    Global {
        /// Whether the constraint is increasing
        increasing: bool,
    },
    /// Piecewise monotonic constraint with breakpoints
    Piecewise {
        /// Breakpoints for piecewise constraints
        breakpoints: Vec<Float>,
        /// Segment increasing flags for each piece
        segments: Vec<bool>,
    },
    /// Convex constraint (increasing and convex)
    Convex,
    /// Concave constraint (increasing and concave)
    Concave,
    /// Convex decreasing constraint
    ConvexDecreasing,
    /// Concave decreasing constraint
    ConcaveDecreasing,
}

/// Loss function options for robust isotonic regression
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// LossFunction
pub enum LossFunction {
    /// Standard squared loss (L2)
    #[default]
    SquaredLoss,
    /// Absolute loss (L1) for robust regression
    AbsoluteLoss,
    /// Huber loss with given delta parameter
    HuberLoss {
        /// Delta parameter for Huber loss
        delta: Float,
    },
    /// Quantile loss for quantile regression
    QuantileLoss {
        /// Target quantile (between 0 and 1)
        quantile: Float,
    },
}

/// Main isotonic regression estimator
///
/// Fits isotonic regression using Pool Adjacent Violators Algorithm (PAVA)
/// with support for different loss functions and constraints.
#[derive(Debug, Clone)]
/// IsotonicRegression
pub struct IsotonicRegression<State = Untrained> {
    /// Monotonicity constraint
    pub constraint: MonotonicityConstraint,
    /// Loss function
    pub loss: LossFunction,
    /// Minimum value for output bounds
    pub y_min: Option<Float>,
    /// Maximum value for output bounds
    pub y_max: Option<Float>,
    /// Whether to fit intercept
    pub fit_intercept: bool,

    // Fitted attributes (only available when State = Trained)
    #[allow(dead_code)]
    x_: Option<Array1<Float>>,
    #[allow(dead_code)]
    y_: Option<Array1<Float>>,
    x_thresholds_: Option<Array1<Float>>,
    y_thresholds_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl IsotonicRegression<Untrained> {
    /// Create a new untrained isotonic regression model
    pub fn new() -> Self {
        Self {
            constraint: MonotonicityConstraint::Increasing,
            loss: LossFunction::SquaredLoss,
            y_min: None,
            y_max: None,
            fit_intercept: true,
            x_: None,
            y_: None,
            x_thresholds_: None,
            y_thresholds_: None,
            _state: PhantomData,
        }
    }

    /// Set whether the function should be increasing or decreasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.constraint = if increasing {
            MonotonicityConstraint::Increasing
        } else {
            MonotonicityConstraint::Decreasing
        };
        self
    }

    /// Set the monotonicity constraint
    pub fn constraint(mut self, constraint: MonotonicityConstraint) -> Self {
        self.constraint = constraint;
        self
    }

    /// Set the loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set the minimum bound for outputs
    pub fn y_min(mut self, y_min: Float) -> Self {
        self.y_min = Some(y_min);
        self
    }

    /// Set the maximum bound for outputs
    pub fn y_max(mut self, y_max: Float) -> Self {
        self.y_max = Some(y_max);
        self
    }

    /// Set both minimum and maximum bounds
    pub fn bounds(mut self, y_min: Float, y_max: Float) -> Self {
        self.y_min = Some(y_min);
        self.y_max = Some(y_max);
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set convex constraint (increasing and convex)
    pub fn convex(mut self) -> Self {
        self.constraint = MonotonicityConstraint::Convex;
        self
    }

    /// Set concave constraint (increasing and concave)
    pub fn concave(mut self) -> Self {
        self.constraint = MonotonicityConstraint::Concave;
        self
    }

    /// Set convex decreasing constraint
    pub fn convex_decreasing(mut self) -> Self {
        self.constraint = MonotonicityConstraint::ConvexDecreasing;
        self
    }

    /// Set concave decreasing constraint
    pub fn concave_decreasing(mut self) -> Self {
        self.constraint = MonotonicityConstraint::ConcaveDecreasing;
        self
    }

    /// Set piecewise monotonic constraints with breakpoints
    pub fn piecewise(mut self, breakpoints: Vec<Float>, segments: Vec<bool>) -> Self {
        self.constraint = MonotonicityConstraint::Piecewise {
            breakpoints,
            segments,
        };
        self
    }
}

impl Default for IsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for IsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl IsotonicRegression<Untrained> {
    /// Fit the isotonic regression model with optional sample weights
    pub fn fit_weighted(
        self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        sample_weight: Option<&Array1<Float>>,
    ) -> Result<IsotonicRegression<Trained>> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "X and y must have the same length, got {} and {}",
                x.len(),
                y.len()
            )));
        }

        if let Some(weights) = sample_weight {
            if weights.len() != x.len() {
                return Err(SklearsError::InvalidInput(format!(
                    "Sample weights must have the same length as X, got {} and {}",
                    weights.len(),
                    x.len()
                )));
            }

            // Check for negative weights
            if weights.iter().any(|&w| w < 0.0) {
                return Err(SklearsError::InvalidInput(
                    "Sample weights must be non-negative".to_string(),
                ));
            }
        }

        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        // Sort by x values
        let mut indices: Vec<usize> = (0..x.len()).collect();
        indices.sort_by(|&i, &j| safe_float_cmp(&x[i], &x[j]));

        let x_sorted: Array1<Float> = indices.iter().map(|&i| x[i]).collect();
        let y_sorted: Array1<Float> = indices.iter().map(|&i| y[i]).collect();
        let weights_sorted: Option<Array1<Float>> =
            sample_weight.map(|w| indices.iter().map(|&i| w[i]).collect());

        // Apply isotonic regression based on the loss function
        let isotonic_result = match self.loss {
            LossFunction::SquaredLoss => crate::pool_adjacent_violators_l2(
                &y_sorted,
                weights_sorted.as_ref(),
                self.constraint == MonotonicityConstraint::Increasing,
            ),
            LossFunction::AbsoluteLoss => crate::pool_adjacent_violators_l1(
                &y_sorted,
                weights_sorted.as_ref(),
                self.constraint == MonotonicityConstraint::Increasing,
            ),
            LossFunction::HuberLoss { delta } => crate::pool_adjacent_violators_huber(
                &y_sorted,
                weights_sorted.as_ref(),
                delta,
                self.constraint == MonotonicityConstraint::Increasing,
            ),
            LossFunction::QuantileLoss { quantile } => {
                if quantile <= 0.0 || quantile >= 1.0 {
                    return Err(SklearsError::InvalidInput(
                        "Quantile must be between 0 and 1".to_string(),
                    ));
                }
                crate::pool_adjacent_violators_quantile(
                    &y_sorted,
                    weights_sorted.as_ref(),
                    quantile,
                    self.constraint == MonotonicityConstraint::Increasing,
                )
            }
        };

        let mut y_isotonic = isotonic_result?;

        // Apply bounds if specified
        if let Some(min_val) = self.y_min {
            y_isotonic.mapv_inplace(|x| x.max(min_val));
        }
        if let Some(max_val) = self.y_max {
            y_isotonic.mapv_inplace(|x| x.min(max_val));
        }

        // Find unique thresholds for interpolation
        let (x_thresholds, y_thresholds) = compute_thresholds(&x_sorted, &y_isotonic)?;

        Ok(IsotonicRegression {
            constraint: self.constraint,
            loss: self.loss,
            y_min: self.y_min,
            y_max: self.y_max,
            fit_intercept: self.fit_intercept,
            x_: Some(x_sorted),
            y_: Some(y_isotonic),
            x_thresholds_: Some(x_thresholds),
            y_thresholds_: Some(y_thresholds),
            _state: PhantomData,
        })
    }
}

impl Fit<Array1<Float>, Array1<Float>> for IsotonicRegression<Untrained> {
    type Fitted = IsotonicRegression<Trained>;

    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        self.fit_weighted(x, y, None)
    }
}

impl IsotonicRegression<Trained> {
    /// Get the fitted x values
    pub fn x_thresholds(&self) -> Result<&Array1<Float>> {
        self.x_thresholds_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "x_thresholds".to_string(),
            })
    }

    /// Get the fitted y values
    pub fn y_thresholds(&self) -> Result<&Array1<Float>> {
        self.y_thresholds_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "y_thresholds".to_string(),
            })
    }
}

impl Predict<Array1<Float>, Array1<Float>> for IsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let x_thresh = self
            .x_thresholds_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;
        let y_thresh = self
            .y_thresholds_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let predictions =
            x.map(|&xi| linear_interpolate(xi, x_thresh, y_thresh, self.y_min, self.y_max));

        Ok(predictions)
    }
}

/// Compute unique thresholds for interpolation
fn compute_thresholds(
    x: &Array1<Float>,
    y: &Array1<Float>,
) -> Result<(Array1<Float>, Array1<Float>)> {
    if x.is_empty() || y.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Input arrays cannot be empty".to_string(),
        ));
    }

    let mut thresholds: Vec<(Float, Float)> = Vec::new();
    let mut prev_x = x[0];
    let mut prev_y = y[0];

    thresholds.push((prev_x, prev_y));

    for i in 1..x.len() {
        if (x[i] - prev_x).abs() > Float::EPSILON || (y[i] - prev_y).abs() > Float::EPSILON {
            thresholds.push((x[i], y[i]));
            prev_x = x[i];
            prev_y = y[i];
        }
    }

    let x_thresh: Array1<Float> = thresholds.iter().map(|(x, _)| *x).collect();
    let y_thresh: Array1<Float> = thresholds.iter().map(|(_, y)| *y).collect();

    Ok((x_thresh, y_thresh))
}

/// Linear interpolation with bounds handling
fn linear_interpolate(
    x: Float,
    x_thresh: &Array1<Float>,
    y_thresh: &Array1<Float>,
    y_min: Option<Float>,
    y_max: Option<Float>,
) -> Float {
    if x_thresh.is_empty() {
        return 0.0;
    }

    if x <= x_thresh[0] {
        let mut result = y_thresh[0];
        if let Some(min_val) = y_min {
            result = result.max(min_val);
        }
        if let Some(max_val) = y_max {
            result = result.min(max_val);
        }
        return result;
    }

    if x >= x_thresh[x_thresh.len() - 1] {
        let mut result = y_thresh[y_thresh.len() - 1];
        if let Some(min_val) = y_min {
            result = result.max(min_val);
        }
        if let Some(max_val) = y_max {
            result = result.min(max_val);
        }
        return result;
    }

    // Find the interval containing x
    for i in 0..(x_thresh.len() - 1) {
        if x >= x_thresh[i] && x <= x_thresh[i + 1] {
            let x0 = x_thresh[i];
            let x1 = x_thresh[i + 1];
            let y0 = y_thresh[i];
            let y1 = y_thresh[i + 1];

            let t = if (x1 - x0).abs() < Float::EPSILON {
                0.0
            } else {
                (x - x0) / (x1 - x0)
            };

            let mut result = y0 + t * (y1 - y0);

            if let Some(min_val) = y_min {
                result = result.max(min_val);
            }
            if let Some(max_val) = y_max {
                result = result.min(max_val);
            }
            return result;
        }
    }

    // Fallback (should not happen)
    let mut result = y_thresh[0];
    if let Some(min_val) = y_min {
        result = result.max(min_val);
    }
    if let Some(max_val) = y_max {
        result = result.min(max_val);
    }
    result
}

/// Basic isotonic regression function
pub fn isotonic_regression(y: &Array1<Float>, increasing: bool) -> Array1<Float> {
    pool_adjacent_violators_l2(y, None, increasing).unwrap_or_else(|_| y.clone())
}

/// Weighted isotonic regression function
pub fn isotonic_regression_weighted(
    y: &Array1<Float>,
    weights: &Array1<Float>,
    increasing: bool,
) -> Array1<Float> {
    pool_adjacent_violators_l2(y, Some(weights), increasing).unwrap_or_else(|_| y.clone())
}

// Forward declarations for PAV functions (will be implemented in pav.rs)
pub use crate::pav::{
    pool_adjacent_violators_huber, pool_adjacent_violators_l1, pool_adjacent_violators_l2,
    pool_adjacent_violators_quantile,
};
