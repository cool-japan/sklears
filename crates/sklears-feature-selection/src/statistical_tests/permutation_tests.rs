//! Permutation-based statistical tests for feature selection

use scirs2_core::error::CoreError;
use scirs2_core::ndarray::ArrayView1;
type CoreResult<T> = std::result::Result<T, CoreError>;

/// permutation_test
pub fn permutation_test(
    _x: &ArrayView1<f64>,
    _y: &ArrayView1<f64>,
    _n_permutations: usize,
) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

/// bootstrap_test
pub fn bootstrap_test(_x: &ArrayView1<f64>, _n_bootstrap: usize) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

/// randomization_test
pub fn randomization_test(
    _x: &ArrayView1<f64>,
    _y: &ArrayView1<f64>,
    _n_random: usize,
) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}
