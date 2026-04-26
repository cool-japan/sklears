//! Bayesian statistical tests for feature selection

use scirs2_core::error::CoreError;
use scirs2_core::ndarray::ArrayView1;
type CoreResult<T> = std::result::Result<T, CoreError>;

/// bayes_factor_test
pub fn bayes_factor_test(_x: &ArrayView1<f64>, _y: &ArrayView1<f64>) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

/// bayesian_t_test
pub fn bayesian_t_test(_x: &ArrayView1<f64>, _y: &ArrayView1<f64>) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

/// bayesian_correlation
pub fn bayesian_correlation(_x: &ArrayView1<f64>, _y: &ArrayView1<f64>) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}
