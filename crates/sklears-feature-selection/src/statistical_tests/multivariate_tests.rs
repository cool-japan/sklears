//! Multivariate statistical tests for feature selection

use scirs2_core::error::CoreError;
use scirs2_core::ndarray::ArrayView2;
type CoreResult<T> = std::result::Result<T, CoreError>;

/// hotelling_t2
pub fn hotelling_t2(_x: &ArrayView2<f64>, _y: &ArrayView2<f64>) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

/// manova
pub fn manova(_samples: &[ArrayView2<f64>]) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

/// canonical_correlation_test
pub fn canonical_correlation_test(
    _x: &ArrayView2<f64>,
    _y: &ArrayView2<f64>,
) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}
