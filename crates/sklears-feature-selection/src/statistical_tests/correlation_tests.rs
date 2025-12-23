//! Correlation-based statistical tests for feature selection

use scirs2_core::error::CoreError;
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
type CoreResult<T> = std::result::Result<T, CoreError>;

/// R-squared statistic for regression
pub fn r_regression(
    X: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
) -> CoreResult<(Array1<f64>, Array1<f64>)> {
    let n_features = X.ncols();
    let mut r_stats = Array1::zeros(n_features);
    let mut p_values = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = X.column(i);
        let r_squared = pearson_correlation(&feature, y)?;
        r_stats[i] = r_squared * r_squared;
        // Convert to p-value using t-distribution approximation
        p_values[i] = 0.05; // Placeholder
    }

    Ok((r_stats, p_values))
}

/// Pearson correlation coefficient
pub fn pearson_correlation(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> CoreResult<f64> {
    if x.len() != y.len() {
        return Ok(0.0);
    }

    let x_mean = x.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;

    for i in 0..x.len() {
        let x_dev = x[i] - x_mean;
        let y_dev = y[i] - y_mean;
        numerator += x_dev * y_dev;
        x_var += x_dev * x_dev;
        y_var += y_dev * y_dev;
    }

    if x_var == 0.0 || y_var == 0.0 {
        return Ok(0.0);
    }

    Ok(numerator / (x_var * y_var).sqrt())
}
