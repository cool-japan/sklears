//! Basic error metrics for regression evaluation
//!
//! This module provides fundamental error measures including MAE, MSE, RMSE,
//! maximum error, median absolute error, and percentage error metrics.
//! Implements SciRS2 Policy for array operations and random number generation.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;

/// Calculate Mean Absolute Error (MAE)
pub fn mean_absolute_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
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
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

/// Calculate Mean Squared Error (MSE)
pub fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
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
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

/// Calculate Root Mean Squared Error (RMSE)
pub fn root_mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    let mse = mean_squared_error(y_true, y_pred)?;
    Ok(mse.sqrt())
}

/// Calculate Maximum Error
pub fn max_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let max_error = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .fold(0.0, f64::max);

    Ok(max_error)
}

/// Calculate Median Absolute Error
pub fn median_absolute_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
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
    let median = if n % 2 == 0 {
        (errors[n / 2 - 1] + errors[n / 2]) / 2.0
    } else {
        errors[n / 2]
    };

    Ok(median)
}

/// Calculate Mean Absolute Percentage Error (MAPE)
pub fn mean_absolute_percentage_error(
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

    // Check for zero values in y_true
    if y_true.iter().any(|&x| x.abs() < f64::EPSILON) {
        return Err(MetricsError::DivisionByZero);
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| ((t - p) / t).abs())
        .sum::<f64>();

    Ok(100.0 * sum / y_true.len() as f64)
}

/// Calculate Median Absolute Percentage Error (MdAPE)
pub fn median_absolute_percentage_error(
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

    // Check for zero values in y_true
    if y_true.iter().any(|&x| x.abs() < f64::EPSILON) {
        return Err(MetricsError::DivisionByZero);
    }

    let mut errors: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| ((t - p) / t).abs())
        .collect();

    errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = errors.len();
    let median = if n % 2 == 0 {
        (errors[n / 2 - 1] + errors[n / 2]) / 2.0
    } else {
        errors[n / 2]
    };

    Ok(100.0 * median)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mean_absolute_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9];
        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert!((mae - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mean_squared_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert!((mse - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_root_mean_squared_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let rmse = root_mean_squared_error(&y_true, &y_pred).unwrap();
        assert!((rmse - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.5, 2.0, 2.5];
        let max_err = max_error(&y_true, &y_pred).unwrap();
        assert!((max_err - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_input() {
        let y_true = array![];
        let y_pred = array![];
        assert!(matches!(
            mean_absolute_error(&y_true, &y_pred),
            Err(MetricsError::EmptyInput)
        ));
    }

    #[test]
    fn test_shape_mismatch() {
        let y_true = array![1.0, 2.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!(matches!(
            mean_absolute_error(&y_true, &y_pred),
            Err(MetricsError::ShapeMismatch { .. })
        ));
    }
}
