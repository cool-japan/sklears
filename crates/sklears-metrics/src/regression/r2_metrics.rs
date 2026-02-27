//! Coefficient of determination (R²) metrics for regression evaluation
//!
//! This module provides R² score and robust variants including median-based,
//! trimmed, Huber, and MAD-based R² scores. Also includes explained variance score.
//! Implements SciRS2 Policy for array operations and numerical computations.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;

/// Calculate R² (coefficient of determination) score
pub fn r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let y_mean = y_true.mean().unwrap();

    let ss_res = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>();

    let ss_tot = y_true.iter().map(|t| (t - y_mean).powi(2)).sum::<f64>();

    if ss_tot.abs() < f64::EPSILON {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(1.0 - ss_res / ss_tot)
}

/// Calculate explained variance score
pub fn explained_variance_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let y_true_mean = y_true.mean().unwrap();
    let residuals: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| t - p)
        .collect();

    let residuals_mean = residuals.iter().sum::<f64>() / residuals.len() as f64;

    let var_y = y_true
        .iter()
        .map(|t| (t - y_true_mean).powi(2))
        .sum::<f64>()
        / (y_true.len() - 1) as f64;

    let var_residuals = residuals
        .iter()
        .map(|r| (r - residuals_mean).powi(2))
        .sum::<f64>()
        / (residuals.len() - 1) as f64;

    if var_y.abs() < f64::EPSILON {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(1.0 - var_residuals / var_y)
}

/// Calculate robust R² score using different robust estimators
pub fn robust_r2_score(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    method: &str,
) -> MetricsResult<f64> {
    match method {
        "median" => median_r2_score(y_true, y_pred),
        "trimmed" => trimmed_r2_score(y_true, y_pred, 0.1),
        "huber" => huber_r2_score(y_true, y_pred, 1.35),
        "mad" => mad_r2_score(y_true, y_pred),
        _ => Err(MetricsError::InvalidParameter(format!(
            "Unknown robust method: {}",
            method
        ))),
    }
}

/// Calculate median-based robust R²
pub fn median_r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let mut y_sorted = y_true.to_vec();
    y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = y_sorted.len();
    let y_median = if n % 2 == 0 {
        (y_sorted[n / 2 - 1] + y_sorted[n / 2]) / 2.0
    } else {
        y_sorted[n / 2]
    };

    let mad_y = {
        let deviations: Vec<f64> = y_true.iter().map(|y| (y - y_median).abs()).collect();
        let mut dev_sorted = deviations;
        dev_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if n % 2 == 0 {
            (dev_sorted[n / 2 - 1] + dev_sorted[n / 2]) / 2.0
        } else {
            dev_sorted[n / 2]
        }
    };

    let mad_res = {
        let residuals: Vec<f64> = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).abs())
            .collect();
        let mut res_sorted = residuals;
        res_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if n % 2 == 0 {
            (res_sorted[n / 2 - 1] + res_sorted[n / 2]) / 2.0
        } else {
            res_sorted[n / 2]
        }
    };

    if mad_y.abs() < f64::EPSILON {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(1.0 - mad_res / mad_y)
}

/// Calculate trimmed R² score (excluding outliers)
pub fn trimmed_r2_score(
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

    if !(0.0..=0.5).contains(&trim_fraction) {
        return Err(MetricsError::InvalidParameter(
            "Trim fraction must be between 0.0 and 0.5".to_string(),
        ));
    }

    // Calculate residuals and sort
    let residuals: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .collect();

    let mut indices: Vec<usize> = (0..residuals.len()).collect();
    indices.sort_by(|&a, &b| residuals[a].partial_cmp(&residuals[b]).unwrap());

    // Trim extreme values
    let n = residuals.len();
    let trim_count = (n as f64 * trim_fraction) as usize;
    let trimmed_indices = &indices[trim_count..n - trim_count];

    if trimmed_indices.is_empty() {
        return Err(MetricsError::InvalidInput("Too much trimming".to_string()));
    }

    // Calculate R² on trimmed data
    let trimmed_true: Vec<f64> = trimmed_indices.iter().map(|&i| y_true[i]).collect();
    let trimmed_pred: Vec<f64> = trimmed_indices.iter().map(|&i| y_pred[i]).collect();

    let trimmed_true_array = Array1::from(trimmed_true);
    let trimmed_pred_array = Array1::from(trimmed_pred);

    r2_score(&trimmed_true_array, &trimmed_pred_array)
}

/// Calculate Huber loss-based robust R²
pub fn huber_r2_score(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    delta: f64,
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

    let y_mean = y_true.mean().unwrap();

    let huber_loss = |r: f64| -> f64 {
        if r.abs() <= delta {
            0.5 * r.powi(2)
        } else {
            delta * (r.abs() - 0.5 * delta)
        }
    };

    let ss_res = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| huber_loss(t - p))
        .sum::<f64>();

    let ss_tot = y_true.iter().map(|t| huber_loss(t - y_mean)).sum::<f64>();

    if ss_tot.abs() < f64::EPSILON {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(1.0 - ss_res / ss_tot)
}

/// Calculate MAD-based robust R²
pub fn mad_r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Use median_r2_score as it's based on MAD
    median_r2_score(y_true, y_pred)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_r2_score_perfect() {
        let y_true = array![1.0, 2.0, 3.0, 4.0];
        let y_pred = array![1.0, 2.0, 3.0, 4.0];
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert!((r2 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_r2_score_mean_prediction() {
        let y_true = array![1.0, 2.0, 3.0, 4.0];
        let mean_val = y_true.mean().unwrap();
        let y_pred = array![mean_val, mean_val, mean_val, mean_val];
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert!((r2 - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_explained_variance_score() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];
        let evs = explained_variance_score(&y_true, &y_pred).unwrap();
        assert!((0.0..=1.0).contains(&evs));
    }

    #[test]
    fn test_robust_r2_methods() {
        let y_true = array![1.0, 2.0, 3.0, 4.0];
        let y_pred = array![1.1, 1.9, 3.1, 3.9];

        let r2_median = robust_r2_score(&y_true, &y_pred, "median").unwrap();
        let r2_trimmed = robust_r2_score(&y_true, &y_pred, "trimmed").unwrap();
        let r2_huber = robust_r2_score(&y_true, &y_pred, "huber").unwrap();
        let r2_mad = robust_r2_score(&y_true, &y_pred, "mad").unwrap();

        // All should give reasonable values
        assert!(r2_median > 0.5);
        assert!(r2_trimmed > 0.5);
        assert!(r2_huber > 0.5);
        assert!(r2_mad > 0.5);
    }

    #[test]
    fn test_invalid_robust_method() {
        let y_true = array![1.0, 2.0];
        let y_pred = array![1.0, 2.0];
        assert!(robust_r2_score(&y_true, &y_pred, "invalid").is_err());
    }
}
