//! Robust regression metrics for evaluation
//!
//! This module provides robust regression metrics that are less sensitive to outliers.
//! Implements SciRS2 Policy for array operations and numerical computations.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;

/// Calculate trimmed mean error (excluding outliers)
pub fn trimmed_mean_error(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    trim_fraction: f64,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let mut errors: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .collect();

    errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = errors.len();
    let trim_count = (n as f64 * trim_fraction) as usize;
    let trimmed_errors = &errors[trim_count..n - trim_count];

    if trimmed_errors.is_empty() {
        return Err(MetricsError::InvalidInput("Too much trimming".to_string()));
    }

    Ok(trimmed_errors.iter().sum::<f64>() / trimmed_errors.len() as f64)
}

/// Calculate winsorized mean error
pub fn winsorized_mean_error(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    limits: (f64, f64),
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let mut errors: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .collect();

    errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = errors.len();
    let lower_idx = (n as f64 * limits.0) as usize;
    let upper_idx = (n as f64 * (1.0 - limits.1)) as usize;

    // Winsorize: replace extreme values
    let lower_val = errors[lower_idx];
    let upper_val = errors[upper_idx.min(n - 1)];

    for error in &mut errors {
        if *error < lower_val {
            *error = lower_val;
        } else if *error > upper_val {
            *error = upper_val;
        }
    }

    Ok(errors.iter().sum::<f64>() / n as f64)
}

/// Calculate biweight midvariance
pub fn biweight_midvariance(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    c: f64,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let residuals: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| t - p)
        .collect();

    // Calculate median
    let mut sorted_residuals = residuals.clone();
    sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted_residuals.len();
    let median = if n % 2 == 0 {
        (sorted_residuals[n / 2 - 1] + sorted_residuals[n / 2]) / 2.0
    } else {
        sorted_residuals[n / 2]
    };

    // Calculate MAD
    let mad_values: Vec<f64> = residuals.iter().map(|r| (r - median).abs()).collect();
    let mut sorted_mad = mad_values;
    sorted_mad.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = if n % 2 == 0 {
        (sorted_mad[n / 2 - 1] + sorted_mad[n / 2]) / 2.0
    } else {
        sorted_mad[n / 2]
    };

    if mad == 0.0 {
        return Ok(0.0);
    }

    // Calculate biweight
    let sum: f64 = residuals
        .iter()
        .map(|r| {
            let u = (r - median) / (c * mad);
            if u.abs() < 1.0 {
                (r - median).powi(2) * (1.0_f64 - u.powi(2)).powi(4)
            } else {
                0.0
            }
        })
        .sum();

    let count = residuals
        .iter()
        .filter(|&r| ((r - median) / (c * mad)).abs() < 1.0)
        .count() as f64;

    Ok(sum / count.max(1.0))
}

/// Calculate Theil-Sen slope
pub fn theil_sen_slope(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "Need at least 2 points".to_string(),
        ));
    }

    let mut slopes = Vec::new();
    let n = y_true.len();

    for i in 0..n {
        for j in i + 1..n {
            let dy = y_pred[j] - y_pred[i];
            let dx = y_true[j] - y_true[i];
            if dx.abs() > f64::EPSILON {
                slopes.push(dy / dx);
            }
        }
    }

    if slopes.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No valid slopes found".to_string(),
        ));
    }

    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_slopes = slopes.len();
    let median_slope = if n_slopes % 2 == 0 {
        (slopes[n_slopes / 2 - 1] + slopes[n_slopes / 2]) / 2.0
    } else {
        slopes[n_slopes / 2]
    };

    Ok(median_slope)
}

/// Calculate Kendall tau distance
pub fn kendall_tau_distance(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "Need at least 2 points".to_string(),
        ));
    }

    let n = y_true.len();
    let mut concordant = 0;
    let mut discordant = 0;

    for i in 0..n {
        for j in i + 1..n {
            let true_diff = y_true[j] - y_true[i];
            let pred_diff = y_pred[j] - y_pred[i];

            if true_diff * pred_diff > 0.0 {
                concordant += 1;
            } else if true_diff * pred_diff < 0.0 {
                discordant += 1;
            }
        }
    }

    let total_pairs = n * (n - 1) / 2;
    let tau = (concordant - discordant) as f64 / total_pairs as f64;

    Ok(1.0 - tau.abs()) // Distance version (0 = perfect correlation, 1 = no correlation)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_trimmed_mean_error() {
        let y_true = array![1.0, 2.0, 3.0, 100.0]; // One outlier
        let y_pred = array![1.0, 2.0, 3.0, 3.0];
        let error = trimmed_mean_error(&y_true, &y_pred, 0.25).unwrap();
        assert!(error < 1.0); // Should be small after trimming outlier
    }

    #[test]
    fn test_theil_sen_slope() {
        let y_true = array![1.0, 2.0, 3.0, 4.0];
        let y_pred = array![1.0, 2.0, 3.0, 4.0]; // Perfect correlation
        let slope = theil_sen_slope(&y_true, &y_pred).unwrap();
        assert!((slope - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_kendall_tau_distance_perfect() {
        let y_true = array![1.0, 2.0, 3.0, 4.0];
        let y_pred = array![1.0, 2.0, 3.0, 4.0]; // Perfect correlation
        let distance = kendall_tau_distance(&y_true, &y_pred).unwrap();
        assert!(distance < 0.1); // Should be close to 0
    }
}
