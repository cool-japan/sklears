//! Time series specific regression metrics
//!
//! This module provides metrics specifically designed for time series forecasting evaluation.
//! Implements SciRS2 Policy for array operations and numerical computations.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;

/// Calculate Mean Absolute Scaled Error (MASE)
pub fn mean_absolute_scaled_error(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    y_train: &Array1<f64>,
    seasonality: usize,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() || y_train.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if seasonality == 0 || seasonality >= y_train.len() {
        return Err(MetricsError::InvalidParameter(
            "Invalid seasonality parameter".to_string(),
        ));
    }

    // Calculate MAE of forecast
    let mae_forecast = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>()
        / y_true.len() as f64;

    // Calculate MAE of seasonal naive forecast on training data
    let mae_naive = y_train
        .windows(seasonality + 1)
        .into_iter()
        .map(|window| (window[seasonality] - window[0]).abs())
        .sum::<f64>()
        / (y_train.len() - seasonality) as f64;

    if mae_naive < f64::EPSILON {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(mae_forecast / mae_naive)
}

/// Calculate Mean Directional Accuracy
pub fn mean_directional_accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
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

    let mut correct_directions = 0;
    let mut total_predictions = 0;

    for i in 1..y_true.len() {
        let true_direction = y_true[i] - y_true[i - 1];
        let pred_direction = y_pred[i] - y_pred[i - 1];

        if true_direction * pred_direction >= 0.0 {
            correct_directions += 1;
        }
        total_predictions += 1;
    }

    Ok(correct_directions as f64 / total_predictions as f64)
}

/// Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
pub fn symmetric_mean_absolute_percentage_error(
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

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let numerator = (t - p).abs();
            let denominator = (t.abs() + p.abs()) / 2.0;
            if denominator < f64::EPSILON {
                0.0 // Handle case where both are zero
            } else {
                numerator / denominator
            }
        })
        .sum::<f64>();

    Ok(100.0 * sum / y_true.len() as f64)
}

/// Calculate Weighted Absolute Percentage Error (WAPE)
pub fn weighted_absolute_percentage_error(
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

    let total_absolute_error: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum();

    let total_actual: f64 = y_true.iter().map(|t| t.abs()).sum();

    if total_actual < f64::EPSILON {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(100.0 * total_absolute_error / total_actual)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mean_absolute_scaled_error() {
        let y_train = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_true = array![7.0, 8.0];
        let y_pred = array![6.9, 8.1];

        let mase = mean_absolute_scaled_error(&y_true, &y_pred, &y_train, 1).unwrap();

        assert!(mase > 0.0);
    }

    #[test]
    fn test_mean_directional_accuracy() {
        let y_true = array![1.0, 2.0, 3.0, 2.0]; // Up, up, down
        let y_pred = array![1.0, 2.1, 2.9, 2.1]; // Up, down, up

        let mda = mean_directional_accuracy(&y_true, &y_pred).unwrap();
        assert!((0.0..=1.0).contains(&mda));
    }

    #[test]
    fn test_symmetric_mean_absolute_percentage_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9];

        let smape = symmetric_mean_absolute_percentage_error(&y_true, &y_pred).unwrap();
        assert!((0.0..=200.0).contains(&smape));
    }

    #[test]
    fn test_weighted_absolute_percentage_error() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9];

        let wape = weighted_absolute_percentage_error(&y_true, &y_pred).unwrap();
        assert!(wape >= 0.0);
    }
}
