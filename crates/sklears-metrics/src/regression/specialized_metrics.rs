//! Specialized regression metrics
//!
//! This module provides specialized regression metrics including concordance correlation
//! coefficient and bias-related metrics.
//! Implements SciRS2 Policy for array operations and numerical computations.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;

/// Calculate Concordance Correlation Coefficient
pub fn concordance_correlation_coefficient(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
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

    let mean_true = y_true.mean().unwrap();
    let mean_pred = y_pred.mean().unwrap();

    let var_true =
        y_true.iter().map(|x| (x - mean_true).powi(2)).sum::<f64>() / (y_true.len() - 1) as f64;
    let var_pred =
        y_pred.iter().map(|x| (x - mean_pred).powi(2)).sum::<f64>() / (y_pred.len() - 1) as f64;

    let covar = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - mean_true) * (p - mean_pred))
        .sum::<f64>()
        / (y_true.len() - 1) as f64;

    let denominator = var_true + var_pred + (mean_true - mean_pred).powi(2);

    if denominator < f64::EPSILON {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(2.0 * covar / denominator)
}

/// Calculate Lin's Concordance Correlation Coefficient
pub fn lin_concordance_coefficient(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> MetricsResult<f64> {
    // Same as concordance_correlation_coefficient for now
    concordance_correlation_coefficient(y_true, y_pred)
}

/// Calculate Mean Bias Error
pub fn mean_bias_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let bias = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| p - t)
        .sum::<f64>()
        / y_true.len() as f64;

    Ok(bias)
}

/// Calculate Mean Absolute Bias
pub fn mean_absolute_bias(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let abs_bias = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (p - t).abs())
        .sum::<f64>()
        / y_true.len() as f64;

    Ok(abs_bias)
}

/// Calculate relative error metrics
pub fn relative_error_metrics(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> MetricsResult<(f64, f64, f64)> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let mut relative_errors: Vec<f64> = Vec::new();

    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        if t.abs() > f64::EPSILON {
            relative_errors.push((p - t).abs() / t.abs());
        }
    }

    if relative_errors.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No valid relative errors".to_string(),
        ));
    }

    // Calculate mean, max, and median relative error
    let mean_rel_error = relative_errors.iter().sum::<f64>() / relative_errors.len() as f64;
    let max_rel_error = relative_errors.iter().fold(0.0f64, |a, &b| a.max(b));

    let mut sorted_errors = relative_errors;
    sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted_errors.len();
    let median_rel_error = if n % 2 == 0 {
        (sorted_errors[n / 2 - 1] + sorted_errors[n / 2]) / 2.0
    } else {
        sorted_errors[n / 2]
    };

    Ok((mean_rel_error, max_rel_error, median_rel_error))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_concordance_correlation_coefficient() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let ccc = concordance_correlation_coefficient(&y_true, &y_pred).unwrap();
        assert!((ccc - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mean_bias_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 3.1]; // Systematic overestimation
        let mbe = mean_bias_error(&y_true, &y_pred).unwrap();
        assert!((mbe - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_relative_error_metrics() {
        let y_true = array![1.0, 2.0, 4.0];
        let y_pred = array![1.1, 2.2, 4.4];
        let (mean_rel, max_rel, median_rel) = relative_error_metrics(&y_true, &y_pred).unwrap();

        assert!(mean_rel > 0.0);
        assert!(max_rel > 0.0);
        assert!(median_rel > 0.0);
    }
}
