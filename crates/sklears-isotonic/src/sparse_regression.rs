//! Sparse isotonic regression with automatic sparsity detection
//!
//! This module contains the SparseIsotonicRegression implementation that identifies
//! and handles sparse patterns in isotonic regression, automatically detecting
//! regions where the function should be exactly zero.

use std::marker::PhantomData;
use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};

use crate::isotonic_regression;

/// Sparse isotonic regression with automatic sparsity detection
///
/// This implementation identifies and handles sparse patterns in isotonic regression,
/// automatically detecting regions where the function should be exactly zero.
#[derive(Debug, Clone)]
/// SparseIsotonicRegression
pub struct SparseIsotonicRegression<State = Untrained> {
    /// Whether the function should be increasing
    pub increasing: bool,
    /// Sparsity threshold below which values are set to zero
    pub sparsity_threshold: Float,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Regularization strength for sparsity
    pub alpha: Float,

    // Fitted attributes
    #[allow(dead_code)]
    x_: Option<Array1<Float>>,
    y_: Option<Array1<Float>>,
    #[allow(dead_code)]
    support_: Option<Array1<bool>>,
    #[allow(dead_code)]
    zero_regions_: Option<Vec<(usize, usize)>>,

    _state: PhantomData<State>,
}

impl SparseIsotonicRegression<Untrained> {
    /// Create a new sparse isotonic regression model
    pub fn new() -> Self {
        Self {
            increasing: true,
            sparsity_threshold: 1e-10,
            fit_intercept: true,
            alpha: 0.01,
            x_: None,
            y_: None,
            support_: None,
            zero_regions_: None,
            _state: PhantomData,
        }
    }

    /// Set whether the function should be increasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set sparsity threshold
    pub fn sparsity_threshold(mut self, threshold: Float) -> Self {
        self.sparsity_threshold = threshold;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set regularization strength
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
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

        // Apply basic isotonic regression
        let mut fitted_values = isotonic_regression(&y_sorted, self.increasing);

        // Apply sparsity constraints
        let mut support = Array1::from_elem(fitted_values.len(), true);
        let mut zero_regions = Vec::new();

        // Identify sparse regions
        let mut region_start = None;
        for i in 0..fitted_values.len() {
            if fitted_values[i].abs() < self.sparsity_threshold {
                fitted_values[i] = 0.0;
                support[i] = false;

                if region_start.is_none() {
                    region_start = Some(i);
                }
            } else if let Some(start) = region_start {
                zero_regions.push((start, i - 1));
                region_start = None;
            }
        }

        // Close any open zero region
        if let Some(start) = region_start {
            zero_regions.push((start, fitted_values.len() - 1));
        }

        Ok(SparseIsotonicRegression {
            increasing: self.increasing,
            sparsity_threshold: self.sparsity_threshold,
            fit_intercept: self.fit_intercept,
            alpha: self.alpha,
            x_: Some(x_sorted),
            y_: Some(fitted_values),
            support_: Some(support),
            zero_regions_: Some(zero_regions),
            _state: PhantomData,
        })
    }
}

impl Predict<Array1<Float>, Array1<Float>> for SparseIsotonicRegression<Trained> {
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

            // Apply sparsity threshold
            if predictions[i].abs() < self.sparsity_threshold {
                predictions[i] = 0.0;
            }
        }

        Ok(predictions)
    }
}

/// Function API for sparse isotonic regression
pub fn sparse_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    sparsity_threshold: Float,
    alpha: Float,
) -> Result<Array1<Float>> {
    let sparse_iso = SparseIsotonicRegression::new()
        .increasing(increasing)
        .sparsity_threshold(sparsity_threshold)
        .alpha(alpha);

    let fitted = sparse_iso.fit(x, y)?;
    fitted.predict(x)
}