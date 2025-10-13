//! Deviance metrics for regression evaluation
//!
//! This module provides deviance functions including Gamma, Poisson, and Tweedie deviance,
//! as well as negative log-likelihood and Kullback-Leibler divergence.
//! Implements SciRS2 Policy for array operations and numerical computations.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;

/// Calculate mean Gamma deviance
pub fn mean_gamma_deviance(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Check for positive values
    if y_true.iter().any(|&x| x <= 0.0) || y_pred.iter().any(|&x| x <= 0.0) {
        return Err(MetricsError::InvalidInput(
            "All values must be positive for Gamma deviance".to_string(),
        ));
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            // Gamma deviance: 2 * ((y_true - y_pred) / y_pred - ln(y_true / y_pred))
            2.0 * ((t - p) / p - (t / p).ln())
        })
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

/// Calculate mean Poisson deviance
pub fn mean_poisson_deviance(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
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
    if y_true.iter().any(|&x| x < 0.0) || y_pred.iter().any(|&x| x <= 0.0) {
        return Err(MetricsError::InvalidInput(
            "y_true must be non-negative and y_pred must be positive for Poisson deviance"
                .to_string(),
        ));
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            // Poisson deviance: 2 * (y_pred - y_true + y_true * ln(y_true / y_pred))
            if *t == 0.0 {
                2.0 * p
            } else {
                2.0 * (p - t + t * (t / p).ln())
            }
        })
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

/// Calculate mean Tweedie deviance
pub fn mean_tweedie_deviance(
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

    // Check power parameter
    if power < 0.0 {
        return Err(MetricsError::InvalidParameter(
            "Power must be non-negative".to_string(),
        ));
    }

    // Check for valid values based on power
    if power == 0.0 {
        // Normal distribution (Gaussian)
        return Ok(y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>()
            / y_true.len() as f64);
    } else if power == 1.0 {
        // Poisson distribution
        return mean_poisson_deviance(y_true, y_pred);
    } else if power == 2.0 {
        // Gamma distribution
        return mean_gamma_deviance(y_true, y_pred);
    }

    // General Tweedie case
    if y_pred.iter().any(|&x| x <= 0.0) {
        return Err(MetricsError::InvalidInput(
            "All predicted values must be positive for Tweedie deviance".to_string(),
        ));
    }

    let sum = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            if *t == 0.0 {
                // Special case for y_true = 0
                2.0 * p.powf(2.0 - power) / (2.0 - power)
            } else if power == 1.0 {
                2.0 * (p - t + t * (t / p).ln())
            } else {
                let term1 = if (1.0 - power).abs() < f64::EPSILON {
                    t * t.ln()
                } else {
                    t.powf(2.0 - power) / (2.0 - power)
                };

                let term2 = if (1.0 - power).abs() < f64::EPSILON {
                    t * p.ln()
                } else {
                    t * p.powf(1.0 - power) / (1.0 - power)
                };

                let term3 = if (2.0 - power).abs() < f64::EPSILON {
                    p.ln()
                } else {
                    p.powf(2.0 - power) / (2.0 - power)
                };

                2.0 * (term1 - term2 + term3)
            }
        })
        .sum::<f64>();

    Ok(sum / y_true.len() as f64)
}

/// Calculate negative log-likelihood
pub fn negative_log_likelihood(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    distribution: &str,
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

    match distribution {
        "normal" | "gaussian" => {
            // Assume unit variance for simplicity
            let sum = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(t, p)| 0.5 * (t - p).powi(2) + 0.5 * (2.0 * std::f64::consts::PI).ln())
                .sum::<f64>();
            Ok(sum / y_true.len() as f64)
        }
        "poisson" => {
            if y_pred.iter().any(|&x| x <= 0.0) {
                return Err(MetricsError::InvalidInput(
                    "Predicted values must be positive for Poisson distribution".to_string(),
                ));
            }
            let sum = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(t, p)| p - t * p.ln() + log_factorial(*t))
                .sum::<f64>();
            Ok(sum / y_true.len() as f64)
        }
        "gamma" => {
            if y_true.iter().any(|&x| x <= 0.0) || y_pred.iter().any(|&x| x <= 0.0) {
                return Err(MetricsError::InvalidInput(
                    "All values must be positive for Gamma distribution".to_string(),
                ));
            }
            // Assuming shape parameter = 1 for simplicity
            let sum = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(t, p)| t / p + p.ln())
                .sum::<f64>();
            Ok(sum / y_true.len() as f64)
        }
        _ => Err(MetricsError::InvalidParameter(format!(
            "Unknown distribution: {}",
            distribution
        ))),
    }
}

/// Calculate Kullback-Leibler divergence
pub fn kullback_leibler_divergence(p: &Array1<f64>, q: &Array1<f64>) -> MetricsResult<f64> {
    if p.len() != q.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![p.len()],
            actual: vec![q.len()],
        });
    }

    if p.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Check if they are valid probability distributions
    let p_sum: f64 = p.iter().sum();
    let q_sum: f64 = q.iter().sum();

    if (p_sum - 1.0).abs() > 1e-6 || (q_sum - 1.0).abs() > 1e-6 {
        return Err(MetricsError::InvalidInput(
            "Input arrays must sum to 1.0 (probability distributions)".to_string(),
        ));
    }

    if p.iter().any(|&x| x < 0.0) || q.iter().any(|&x| x <= 0.0) {
        return Err(MetricsError::InvalidInput(
            "P must be non-negative and Q must be positive".to_string(),
        ));
    }

    let kl_div = p
        .iter()
        .zip(q.iter())
        .map(|(p_i, q_i)| {
            if *p_i == 0.0 {
                0.0
            } else {
                p_i * (p_i / q_i).ln()
            }
        })
        .sum::<f64>();

    Ok(kl_div)
}

/// Helper function to compute log factorial (Stirling's approximation for large n)
fn log_factorial(n: f64) -> f64 {
    if n <= 1.0 {
        0.0
    } else if n < 12.0 {
        // Exact computation for small n
        (2..=n as i32).map(|i| (i as f64).ln()).sum()
    } else {
        // Stirling's approximation: ln(n!) ≈ n*ln(n) - n + 0.5*ln(2πn)
        n * n.ln() - n + 0.5 * (2.0 * std::f64::consts::PI * n).ln()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mean_gamma_deviance() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let deviance = mean_gamma_deviance(&y_true, &y_pred).unwrap();
        assert!((deviance - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mean_poisson_deviance() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let deviance = mean_poisson_deviance(&y_true, &y_pred).unwrap();
        assert!((deviance - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mean_tweedie_deviance_power_0() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9];
        let deviance = mean_tweedie_deviance(&y_true, &y_pred, 0.0).unwrap();
        assert!(deviance > 0.0);
    }

    #[test]
    fn test_mean_tweedie_deviance_power_1() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let deviance = mean_tweedie_deviance(&y_true, &y_pred, 1.0).unwrap();
        assert!((deviance - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mean_tweedie_deviance_power_2() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        let deviance = mean_tweedie_deviance(&y_true, &y_pred, 2.0).unwrap();
        assert!((deviance - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_negative_log_likelihood_normal() {
        let y_true = array![0.0, 1.0, -1.0];
        let y_pred = array![0.1, 0.9, -0.9];
        let nll = negative_log_likelihood(&y_true, &y_pred, "normal").unwrap();
        assert!(nll > 0.0);
    }

    #[test]
    fn test_kullback_leibler_divergence() {
        let p = array![0.5, 0.3, 0.2];
        let q = array![0.4, 0.4, 0.2];
        let kl = kullback_leibler_divergence(&p, &q).unwrap();
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_invalid_gamma_deviance() {
        let y_true = array![-1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0, 3.0];
        assert!(mean_gamma_deviance(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_invalid_probability_distribution() {
        let p = array![0.5, 0.3, 0.3]; // Sums to 1.1
        let q = array![0.4, 0.4, 0.2];
        assert!(kullback_leibler_divergence(&p, &q).is_err());
    }
}
