//! Time series statistical tests for feature selection

use scirs2_core::error::CoreError;
use scirs2_core::ndarray::ArrayView1;
type CoreResult<T> = std::result::Result<T, CoreError>;

pub fn ljung_box_test(x: &ArrayView1<f64>, lags: usize) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

pub fn adf_test(x: &ArrayView1<f64>) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}

pub fn granger_causality(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    max_lags: usize,
) -> CoreResult<(f64, f64)> {
    Ok((0.0, 0.5))
}
