//! Logarithmic error metrics for regression evaluation
//!
//! This module provides metrics based on logarithmic transformations including
//! mean squared logarithmic error, root mean squared logarithmic error,
//! log-cosh error, and logarithmic absolute error.
//! Implements SciRS2 Policy for array operations and numerical computations.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;

/// Calculate Mean Squared Logarithmic Error (MSLE)
pub fn mean_squared_log_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Check for non-negative values
    if y_true.iter().any(|&x| x < 0.0) || y_pred.iter().any(|&x| x < 0.0) {
        return Err(MetricsError::InvalidInput(
            "All values must be non-negative for logarithmic metrics".to_string(),
        ));
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let log_t = (1.0 + t).ln();
            let log_p = (1.0 + p).ln();
            (log_t - log_p).powi(2)
        })
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

/// Calculate Root Mean Squared Logarithmic Error (RMSLE)
pub fn root_mean_squared_log_error(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> MetricsResult<f64> {
    let msle = mean_squared_log_error(y_true, y_pred)?;
    Ok(msle.sqrt())
}

/// Calculate log-cosh error (log of hyperbolic cosine of prediction error)
pub fn log_cosh_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let diff = p - t;
            // Use stable computation for log(cosh(x))
            if diff.abs() < 1e-12 {
                0.5 * diff.powi(2)
            } else if diff.abs() > 700.0 {
                // For large values, log(cosh(x)) ≈ |x| - log(2)
                diff.abs() - (2.0_f64).ln()
            } else {
                diff.cosh().ln()
            }
        })
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

/// Calculate logarithmic absolute error
pub fn log_absolute_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Check for positive values (needed for log)
    if y_true.iter().any(|&x| x <= 0.0) || y_pred.iter().any(|&x| x <= 0.0) {
        return Err(MetricsError::InvalidInput(
            "All values must be positive for logarithmic absolute error".to_string(),
        ));
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t.ln() - p.ln()).abs())
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mean_squared_log_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let msle = mean_squared_log_error(&y_true, &y_pred).unwrap();
        assert!((msle - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_root_mean_squared_log_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let rmsle = root_mean_squared_log_error(&y_true, &y_pred).unwrap();
        assert!((rmsle - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_log_cosh_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9];
        let log_cosh = log_cosh_error(&y_true, &y_pred).unwrap();
        assert!(log_cosh >= 0.0);
    }

    #[test]
    fn test_log_absolute_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9];
        let log_abs = log_absolute_error(&y_true, &y_pred).unwrap();
        assert!(log_abs >= 0.0);
    }

    #[test]
    fn test_negative_values_error() {
        let y_true = array![-1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!(mean_squared_log_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_zero_values_error() {
        let y_true = array![0.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!(log_absolute_error(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_large_values_log_cosh() {
        let y_true = array![0.0];
        let y_pred = array![1000.0]; // Large value
        let log_cosh = log_cosh_error(&y_true, &y_pred).unwrap();
        assert!(log_cosh > 0.0);
        assert!(log_cosh.is_finite());
    }
}
