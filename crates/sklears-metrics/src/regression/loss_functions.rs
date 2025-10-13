//! Loss functions for regression evaluation
//!
//! This module provides various loss functions including Huber loss, quantile loss (pinball loss),
//! epsilon-insensitive loss, hinge loss, and squared hinge loss.
//! Implements SciRS2 Policy for array operations and numerical computations.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;

/// Calculate Huber loss (robust loss function)
pub fn huber_loss(y_true: &Array1<f64>, y_pred: &Array1<f64>, delta: f64) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if delta <= 0.0 {
        return Err(MetricsError::InvalidParameter(
            "Delta must be positive".to_string(),
        ));
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let residual = (t - p).abs();
            if residual <= delta {
                0.5 * residual.powi(2)
            } else {
                delta * (residual - 0.5 * delta)
            }
        })
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

/// Calculate quantile loss (pinball loss)
pub fn quantile_loss(y_true: &Array1<f64>, y_pred: &Array1<f64>, alpha: f64) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if !(0.0..=1.0).contains(&alpha) {
        return Err(MetricsError::InvalidParameter(
            "Alpha must be between 0 and 1".to_string(),
        ));
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let error = t - p;
            if error >= 0.0 {
                alpha * error
            } else {
                (alpha - 1.0) * error
            }
        })
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

/// Calculate mean pinball loss (alias for quantile loss)
pub fn mean_pinball_loss(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    alpha: f64,
) -> MetricsResult<f64> {
    quantile_loss(y_true, y_pred, alpha)
}

/// Calculate epsilon-insensitive loss (SVR loss)
pub fn epsilon_insensitive_loss(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    epsilon: f64,
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

    if epsilon < 0.0 {
        return Err(MetricsError::InvalidParameter(
            "Epsilon must be non-negative".to_string(),
        ));
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let error = (t - p).abs();
            if error <= epsilon {
                0.0
            } else {
                error - epsilon
            }
        })
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

/// Calculate hinge loss
pub fn hinge_loss(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Check that y_true contains only -1 and 1
    if !y_true.iter().all(|&x| x == -1.0 || x == 1.0) {
        return Err(MetricsError::InvalidInput(
            "y_true must contain only -1 and 1 values for hinge loss".to_string(),
        ));
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let margin = t * p;
            if margin >= 1.0 {
                0.0
            } else {
                1.0 - margin
            }
        })
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

/// Calculate squared hinge loss
pub fn squared_hinge_loss(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Check that y_true contains only -1 and 1
    if !y_true.iter().all(|&x| x == -1.0 || x == 1.0) {
        return Err(MetricsError::InvalidInput(
            "y_true must contain only -1 and 1 values for squared hinge loss".to_string(),
        ));
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let margin = t * p;
            if margin >= 1.0 {
                0.0
            } else {
                (1.0 - margin).powi(2)
            }
        })
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_huber_loss() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let loss = huber_loss(&y_true, &y_pred, 1.0).unwrap();
        assert!((loss - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_huber_loss_large_error() {
        let y_true = array![0.0];
        let y_pred = array![10.0]; // Large error
        let delta = 1.0;
        let loss = huber_loss(&y_true, &y_pred, delta).unwrap();
        let expected = delta * (10.0 - 0.5 * delta); // Linear region
        assert!((loss - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quantile_loss_median() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let loss = quantile_loss(&y_true, &y_pred, 0.5).unwrap();
        assert!((loss - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quantile_loss_upper_quantile() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![0.5, 1.5, 2.5]; // Under-predictions
        let loss = quantile_loss(&y_true, &y_pred, 0.9).unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn test_mean_pinball_loss() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let loss = mean_pinball_loss(&y_true, &y_pred, 0.5).unwrap();
        assert!((loss - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_epsilon_insensitive_loss() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9]; // Small errors
        let epsilon = 0.2;
        let loss = epsilon_insensitive_loss(&y_true, &y_pred, epsilon).unwrap();
        assert!((loss - 0.0).abs() < f64::EPSILON); // All errors within epsilon
    }

    #[test]
    fn test_epsilon_insensitive_loss_large_error() {
        let y_true = array![1.0];
        let y_pred = array![2.0]; // Error = 1.0
        let epsilon = 0.5;
        let loss = epsilon_insensitive_loss(&y_true, &y_pred, epsilon).unwrap();
        assert!((loss - 0.5).abs() < f64::EPSILON); // 1.0 - 0.5 = 0.5
    }

    #[test]
    fn test_hinge_loss() {
        let y_true = array![1.0, -1.0, 1.0];
        let y_pred = array![2.0, -2.0, 0.5]; // Margins: 2.0, 2.0, 0.5
        let loss = hinge_loss(&y_true, &y_pred).unwrap();
        let expected = (0.0 + 0.0 + 0.5) / 3.0; // Only last has loss
        assert!((loss - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_squared_hinge_loss() {
        let y_true = array![1.0, -1.0];
        let y_pred = array![0.5, -0.5]; // Margins: 0.5, 0.5
        let loss = squared_hinge_loss(&y_true, &y_pred).unwrap();
        let expected = ((1.0f64 - 0.5).powi(2) + (1.0f64 - 0.5).powi(2)) / 2.0;
        assert!((loss - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_invalid_hinge_loss_labels() {
        let y_true = array![0.0, 1.0]; // Invalid labels
        let y_pred = array![0.5, 1.5];
        assert!(hinge_loss(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_invalid_parameters() {
        let y_true = array![1.0, 2.0];
        let y_pred = array![1.0, 2.0];

        // Invalid delta for Huber loss
        assert!(huber_loss(&y_true, &y_pred, -1.0).is_err());

        // Invalid alpha for quantile loss
        assert!(quantile_loss(&y_true, &y_pred, 1.5).is_err());

        // Invalid epsilon
        assert!(epsilon_insensitive_loss(&y_true, &y_pred, -0.1).is_err());
    }
}
