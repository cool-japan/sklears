//! D² score metrics for regression evaluation
//!
//! This module provides D² (deviance explained) scores which are coefficient of determination
//! variants based on specific deviance functions. Includes D² scores based on absolute error,
//! pinball loss, Tweedie deviance, Gamma deviance, and Poisson deviance.
//! Implements SciRS2 Policy for array operations and numerical computations.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;

/// Calculate D² score based on absolute error
pub fn d2_absolute_error_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let y_median = {
        let mut y_sorted = y_true.to_vec();
        y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = y_sorted.len();
        if n % 2 == 0 {
            (y_sorted[n / 2 - 1] + y_sorted[n / 2]) / 2.0
        } else {
            y_sorted[n / 2]
        }
    };

    let dev_residuals = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>();

    let dev_null = y_true.iter().map(|t| (t - y_median).abs()).sum::<f64>();

    if dev_null.abs() < f64::EPSILON {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(1.0 - dev_residuals / dev_null)
}

/// Calculate D² score based on pinball loss
pub fn d2_pinball_score(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    alpha: f64,
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

    if !(0.0..=1.0).contains(&alpha) {
        return Err(MetricsError::InvalidParameter(
            "Alpha must be between 0 and 1".to_string(),
        ));
    }

    // Calculate quantile of y_true
    let y_quantile = {
        let mut y_sorted = y_true.to_vec();
        y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = y_sorted.len();
        let index = (alpha * (n - 1) as f64) as usize;
        let fraction = alpha * (n - 1) as f64 - index as f64;

        if index == n - 1 {
            y_sorted[index]
        } else {
            y_sorted[index] * (1.0 - fraction) + y_sorted[index + 1] * fraction
        }
    };

    let pinball_loss = |error: f64, alpha: f64| -> f64 {
        if error >= 0.0 {
            alpha * error
        } else {
            (alpha - 1.0) * error
        }
    };

    let dev_residuals = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| pinball_loss(t - p, alpha))
        .sum::<f64>();

    let dev_null = y_true
        .iter()
        .map(|t| pinball_loss(t - y_quantile, alpha))
        .sum::<f64>();

    if dev_null.abs() < f64::EPSILON {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(1.0 - dev_residuals / dev_null)
}

/// Calculate D² score based on Tweedie deviance
pub fn d2_tweedie_score(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    power: f64,
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

    if power < 0.0 {
        return Err(MetricsError::InvalidParameter(
            "Power must be non-negative".to_string(),
        ));
    }

    let y_mean = y_true.mean().unwrap();

    let tweedie_deviance = |y: f64, y_hat: f64, power: f64| -> f64 {
        if power == 0.0 {
            // Normal distribution
            (y - y_hat).powi(2)
        } else if power == 1.0 {
            // Poisson distribution
            if y == 0.0 {
                2.0 * y_hat
            } else {
                2.0 * (y_hat - y + y * (y / y_hat).ln())
            }
        } else if power == 2.0 {
            // Gamma distribution
            if y <= 0.0 || y_hat <= 0.0 {
                return f64::INFINITY;
            }
            2.0 * ((y - y_hat) / y_hat - (y / y_hat).ln())
        } else {
            // General Tweedie case
            if y_hat <= 0.0 {
                return f64::INFINITY;
            }

            if y == 0.0 {
                2.0 * y_hat.powf(2.0 - power) / (2.0 - power)
            } else {
                let term1 = if (1.0 - power).abs() < f64::EPSILON {
                    y * y.ln()
                } else {
                    y.powf(2.0 - power) / (2.0 - power)
                };

                let term2 = if (1.0 - power).abs() < f64::EPSILON {
                    y * y_hat.ln()
                } else {
                    y * y_hat.powf(1.0 - power) / (1.0 - power)
                };

                let term3 = if (2.0 - power).abs() < f64::EPSILON {
                    y_hat.ln()
                } else {
                    y_hat.powf(2.0 - power) / (2.0 - power)
                };

                2.0 * (term1 - term2 + term3)
            }
        }
    };

    let dev_residuals = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| tweedie_deviance(*t, *p, power))
        .sum::<f64>();

    let dev_null = y_true
        .iter()
        .map(|t| tweedie_deviance(*t, y_mean, power))
        .sum::<f64>();

    if dev_null.abs() < f64::EPSILON || !dev_residuals.is_finite() || !dev_null.is_finite() {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(1.0 - dev_residuals / dev_null)
}

/// Calculate D² score based on Gamma deviance
pub fn d2_gamma_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    d2_tweedie_score(y_true, y_pred, 2.0)
}

/// Calculate D² score based on Poisson deviance
pub fn d2_poisson_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    d2_tweedie_score(y_true, y_pred, 1.0)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_d2_absolute_error_score_perfect() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let d2 = d2_absolute_error_score(&y_true, &y_pred).unwrap();
        assert!((d2 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_d2_pinball_score_median() {
        let y_true = array![1.0, 2.0, 3.0, 4.0];
        let y_pred = array![1.0, 2.0, 3.0, 4.0];
        let d2 = d2_pinball_score(&y_true, &y_pred, 0.5).unwrap();
        assert!((d2 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_d2_tweedie_score_power_0() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let d2 = d2_tweedie_score(&y_true, &y_pred, 0.0).unwrap();
        assert!((d2 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_d2_tweedie_score_power_1() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let d2 = d2_tweedie_score(&y_true, &y_pred, 1.0).unwrap();
        assert!((d2 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_d2_tweedie_score_power_2() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let d2 = d2_tweedie_score(&y_true, &y_pred, 2.0).unwrap();
        assert!((d2 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_d2_gamma_score() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let d2 = d2_gamma_score(&y_true, &y_pred).unwrap();
        assert!((d2 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_d2_poisson_score() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let d2 = d2_poisson_score(&y_true, &y_pred).unwrap();
        assert!((d2 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_d2_scores_imperfect_prediction() {
        let y_true = array![1.0, 2.0, 3.0, 4.0];
        let y_pred = array![1.1, 1.9, 3.1, 3.9];

        let d2_abs = d2_absolute_error_score(&y_true, &y_pred).unwrap();
        let d2_pinball = d2_pinball_score(&y_true, &y_pred, 0.5).unwrap();
        let d2_gamma = d2_gamma_score(&y_true, &y_pred).unwrap();
        let d2_poisson = d2_poisson_score(&y_true, &y_pred).unwrap();

        // All should be less than 1 but greater than 0 for good predictions
        assert!(0.0 < d2_abs && d2_abs < 1.0);
        assert!(0.0 < d2_pinball && d2_pinball < 1.0);
        assert!(0.0 < d2_gamma && d2_gamma < 1.0);
        assert!(0.0 < d2_poisson && d2_poisson < 1.0);
    }

    #[test]
    fn test_invalid_parameters() {
        let y_true = array![1.0, 2.0];
        let y_pred = array![1.0, 2.0];

        // Invalid alpha for pinball score
        assert!(d2_pinball_score(&y_true, &y_pred, 1.5).is_err());

        // Invalid power for Tweedie score
        assert!(d2_tweedie_score(&y_true, &y_pred, -0.5).is_err());
    }
}
