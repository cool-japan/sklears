//! Mutual information tests for feature selection

use scirs2_core::error::CoreError;
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
type CoreResult<T> = std::result::Result<T, CoreError>;

/// Mutual information for classification
pub fn mutual_info_classif(X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> CoreResult<Array1<f64>> {
    let n_features = X.ncols();
    let mut mi_scores = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = X.column(i);
        mi_scores[i] = estimate_mi_dc(&feature, y)?;
    }

    Ok(mi_scores)
}

/// Mutual information for regression
pub fn mutual_info_regression(X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> CoreResult<Array1<f64>> {
    let n_features = X.ncols();
    let mut mi_scores = Array1::zeros(n_features);

    for i in 0..n_features {
        let feature = X.column(i);
        mi_scores[i] = estimate_mi_cc(&feature, y)?;
    }

    Ok(mi_scores)
}

/// Estimate mutual information for discrete-continuous variables
pub fn estimate_mi_dc(
    x_discrete: &ArrayView1<f64>,
    y_continuous: &ArrayView1<f64>,
) -> CoreResult<f64> {
    // Simplified MI estimation using binning
    let mut mi = 0.0;

    // Find unique discrete values
    let mut x_vals: Vec<f64> = x_discrete.to_vec();
    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    x_vals.dedup();

    // Bin continuous variable
    let y_min = y_continuous.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y_continuous
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let n_bins = 10;
    let bin_width = (y_max - y_min) / n_bins as f64;

    if bin_width == 0.0 {
        return Ok(0.0);
    }

    let n = x_discrete.len() as f64;

    // Calculate mutual information
    for &x_val in &x_vals {
        for bin in 0..n_bins {
            let y_min_bin = y_min + bin as f64 * bin_width;
            let y_max_bin = if bin == n_bins - 1 {
                y_max + 1e-10
            } else {
                y_min + (bin + 1) as f64 * bin_width
            };

            let joint_count = x_discrete
                .iter()
                .zip(y_continuous.iter())
                .filter(|(&x, &y)| (x - x_val).abs() < 1e-10 && y >= y_min_bin && y < y_max_bin)
                .count() as f64;

            if joint_count > 0.0 {
                let x_count = x_discrete
                    .iter()
                    .filter(|&&x| (x - x_val).abs() < 1e-10)
                    .count() as f64;
                let y_count = y_continuous
                    .iter()
                    .filter(|&&y| y >= y_min_bin && y < y_max_bin)
                    .count() as f64;

                let p_joint = joint_count / n;
                let p_x = x_count / n;
                let p_y = y_count / n;

                if p_joint > 0.0 && p_x > 0.0 && p_y > 0.0 {
                    mi += p_joint * (p_joint / (p_x * p_y)).ln();
                }
            }
        }
    }

    Ok(mi)
}

/// Estimate mutual information for continuous-continuous variables
pub fn estimate_mi_cc(
    x_continuous: &ArrayView1<f64>,
    y_continuous: &ArrayView1<f64>,
) -> CoreResult<f64> {
    // Simplified MI estimation using k-nearest neighbors
    // For now, use correlation as a proxy
    let n = x_continuous.len() as f64;
    let x_mean = x_continuous.mean().unwrap_or(0.0);
    let y_mean = y_continuous.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;

    for i in 0..x_continuous.len() {
        let x_dev = x_continuous[i] - x_mean;
        let y_dev = y_continuous[i] - y_mean;
        numerator += x_dev * y_dev;
        x_var += x_dev * x_dev;
        y_var += y_dev * y_dev;
    }

    if x_var == 0.0 || y_var == 0.0 {
        return Ok(0.0);
    }

    let r = numerator / (x_var * y_var).sqrt();
    // Convert correlation to mutual information approximation
    Ok(-0.5 * (1.0 - r * r).ln())
}
