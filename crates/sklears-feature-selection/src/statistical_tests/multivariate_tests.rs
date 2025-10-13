//! Multivariate statistical tests for feature selection

use scirs2_core::error::CoreError;
use scirs2_core::ndarray::ArrayView2;
type CoreResult<T> = std::result::Result<T, CoreError>;

pub fn hotelling_t2(x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

pub fn manova(samples: &[ArrayView2<f64>]) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

pub fn canonical_correlation_test(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}
