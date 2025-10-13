//! Common types and utilities for Bayesian methods

use sklears_core::error::SklearsError;
use scirs2_core::ndarray::{Array1, Array2};

/// Common result type for Bayesian analyses
pub type BayesianResult<T> = Result<T, SklearsError>;

/// Statistics for posterior distributions
#[derive(Debug, Clone)]
pub struct PosteriorStats {
    /// Posterior mean
    pub mean: Array1<f64>,
    /// Posterior standard deviation
    pub std: Array1<f64>,
    /// Credible intervals
    pub credible_intervals: Array2<f64>,
}

impl PosteriorStats {
    /// Create new posterior statistics
    pub fn new(mean: Array1<f64>, std: Array1<f64>, credible_intervals: Array2<f64>) -> Self {
        Self {
            mean,
            std,
            credible_intervals,
        }
    }
}