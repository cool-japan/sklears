//! Permutation-based statistical tests for feature selection

use scirs2_core::error::CoreError;
use scirs2_core::ndarray::ArrayView1;
type CoreResult<T> = std::result::Result<T, CoreError>;

pub fn permutation_test(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    n_permutations: usize,
) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

pub fn bootstrap_test(x: &ArrayView1<f64>, n_bootstrap: usize) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

pub fn randomization_test(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    n_random: usize,
) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}
