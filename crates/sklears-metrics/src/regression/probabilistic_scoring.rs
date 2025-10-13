//! Probabilistic scoring rules for regression evaluation
//!
//! This module provides proper scoring rules including CRPS, energy score,
//! Dawid-Sebastiani score, and logarithmic score for evaluating probabilistic forecasts.
//! Implements SciRS2 Policy for array operations and numerical computations.

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, ArrayView1};

/// Calculate Continuous Ranked Probability Score (CRPS)
pub fn continuous_ranked_probability_score(
    y_true: &Array1<f64>,
    y_pred_samples: &[ArrayView1<f64>],
) -> MetricsResult<f64> {
    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if y_pred_samples.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No prediction samples provided".to_string(),
        ));
    }

    let mut total_crps = 0.0;
    let n = y_true.len();

    for i in 0..n {
        let obs = y_true[i];
        let mut samples: Vec<f64> = y_pred_samples.iter().map(|arr| arr[i]).collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let m = samples.len() as f64;
        let mut crps_i = 0.0;

        // Calculate CRPS for this observation
        for (j, &sample) in samples.iter().enumerate() {
            let p = (j + 1) as f64 / m;
            let indicator = if obs <= sample { 1.0 } else { 0.0 };
            crps_i += (p - indicator).powi(2);
        }

        // Normalize by number of samples
        crps_i /= m;
        total_crps += crps_i;
    }

    Ok(total_crps / n as f64)
}

/// Calculate CRPS for ensemble forecasts (convenience function)
pub fn crps_ensemble(
    y_true: &Array1<f64>,
    y_pred_ensemble: &[ArrayView1<f64>],
) -> MetricsResult<f64> {
    continuous_ranked_probability_score(y_true, y_pred_ensemble)
}

/// Calculate CRPS for Gaussian predictions (analytical formula)
pub fn crps_gaussian(
    y_true: &Array1<f64>,
    y_pred_mean: &Array1<f64>,
    y_pred_std: &Array1<f64>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred_mean.len() || y_true.len() != y_pred_std.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred_mean.len(), y_pred_std.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if y_pred_std.iter().any(|&x| x <= 0.0) {
        return Err(MetricsError::InvalidInput(
            "Standard deviations must be positive".to_string(),
        ));
    }

    let sqrt_pi = std::f64::consts::PI.sqrt();

    let crps_sum = y_true
        .iter()
        .zip(y_pred_mean.iter())
        .zip(y_pred_std.iter())
        .map(|((obs, mean), std)| {
            let z = (obs - mean) / std;
            let phi_z = (-0.5_f64 * z.powi(2)).exp() / (2.0 * std::f64::consts::PI).sqrt();
            let phi_z_cdf = 0.5 * (1.0 + erf(z / (2.0_f64).sqrt()));

            std * (z * (2.0 * phi_z_cdf - 1.0) + 2.0 * phi_z - 1.0 / sqrt_pi)
        })
        .sum::<f64>();

    Ok(crps_sum / y_true.len() as f64)
}

/// Calculate Energy Score for multivariate distributional forecasts
pub fn energy_score(
    y_true: &Array1<f64>,
    y_pred_samples: &[ArrayView1<f64>],
    beta: f64,
) -> MetricsResult<f64> {
    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if y_pred_samples.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No prediction samples provided".to_string(),
        ));
    }

    if beta <= 0.0 || beta > 2.0 {
        return Err(MetricsError::InvalidParameter(
            "Beta must be in (0, 2]".to_string(),
        ));
    }

    let n = y_true.len();
    let m = y_pred_samples.len();
    let mut total_score = 0.0;

    for i in 0..n {
        let obs = y_true[i];

        // First term: E[||X - y||^β]
        let term1: f64 = y_pred_samples
            .iter()
            .map(|sample| (sample[i] - obs).abs().powf(beta))
            .sum::<f64>()
            / m as f64;

        // Second term: 0.5 * E[||X - X'||^β]
        let mut term2 = 0.0;
        for j in 0..m {
            for k in j + 1..m {
                term2 += (y_pred_samples[j][i] - y_pred_samples[k][i])
                    .abs()
                    .powf(beta);
            }
        }
        term2 = term2 * 2.0 / (m * (m - 1)) as f64; // Multiply by 2 since we only sum over j < k

        total_score += term1 - 0.5 * term2;
    }

    Ok(total_score / n as f64)
}

/// Calculate Dawid-Sebastiani Score
pub fn dawid_sebastiani_score(
    y_true: &Array1<f64>,
    y_pred_mean: &Array1<f64>,
    y_pred_var: &Array1<f64>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred_mean.len() || y_true.len() != y_pred_var.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred_mean.len(), y_pred_var.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if y_pred_var.iter().any(|&x| x <= 0.0) {
        return Err(MetricsError::InvalidInput(
            "Variances must be positive".to_string(),
        ));
    }

    let ds_sum = y_true
        .iter()
        .zip(y_pred_mean.iter())
        .zip(y_pred_var.iter())
        .map(|((obs, mean), var)| {
            let squared_error = (obs - mean).powi(2);
            squared_error / var + var.ln()
        })
        .sum::<f64>();

    Ok(ds_sum / y_true.len() as f64)
}

/// Calculate Logarithmic Score
pub fn logarithmic_score(
    y_true: &Array1<f64>,
    y_pred_mean: &Array1<f64>,
    y_pred_std: &Array1<f64>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred_mean.len() || y_true.len() != y_pred_std.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred_mean.len(), y_pred_std.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    if y_pred_std.iter().any(|&x| x <= 0.0) {
        return Err(MetricsError::InvalidInput(
            "Standard deviations must be positive".to_string(),
        ));
    }

    let log_score_sum = y_true
        .iter()
        .zip(y_pred_mean.iter())
        .zip(y_pred_std.iter())
        .map(|((obs, mean), std)| {
            let z = (obs - mean) / std;
            0.5 * (z.powi(2) + (2.0 * std::f64::consts::PI * std.powi(2)).ln())
        })
        .sum::<f64>();

    Ok(log_score_sum / y_true.len() as f64)
}

/// Helper function to compute error function (erf)
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_crps_gaussian_perfect() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred_mean = array![1.0, 2.0, 3.0];
        let y_pred_std = array![0.1, 0.1, 0.1];

        let crps = crps_gaussian(&y_true, &y_pred_mean, &y_pred_std).unwrap();
        assert!(crps >= 0.0);
        assert!(crps < 0.1); // Should be small for perfect predictions with small std
    }

    #[test]
    fn test_energy_score() {
        let y_true = array![1.0, 2.0];
        let sample1 = array![1.1, 2.1];
        let sample2 = array![0.9, 1.9];
        let samples = vec![sample1.view(), sample2.view()];

        let score = energy_score(&y_true, &samples, 1.0).unwrap();
        assert!(score >= 0.0);
    }

    #[test]
    fn test_dawid_sebastiani_score() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred_mean = array![1.1, 2.1, 2.9];
        let y_pred_var = array![0.1, 0.1, 0.1];

        let ds = dawid_sebastiani_score(&y_true, &y_pred_mean, &y_pred_var).unwrap();

        // Dawid-Sebastiani score can be negative (lower is better)
        // With small variances (0.1) and small errors, ln(var) dominates and makes score negative
        assert!(ds.is_finite());

        // Test with perfect predictions (should give ln(var) as the score)
        let perfect_ds = dawid_sebastiani_score(&y_true, &y_true, &y_pred_var).unwrap();
        assert!((perfect_ds - 0.1_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_logarithmic_score() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred_mean = array![1.0, 2.0, 3.0];
        let y_pred_std = array![1.0, 1.0, 1.0];

        let log_score = logarithmic_score(&y_true, &y_pred_mean, &y_pred_std).unwrap();
        assert!(log_score >= 0.0);
    }

    #[test]
    fn test_invalid_std_values() {
        let y_true = array![1.0, 2.0];
        let y_pred_mean = array![1.0, 2.0];
        let y_pred_std = array![0.0, 1.0]; // Invalid: contains zero

        assert!(crps_gaussian(&y_true, &y_pred_mean, &y_pred_std).is_err());
    }

    #[test]
    fn test_invalid_energy_score_beta() {
        let y_true = array![1.0];
        let sample = array![1.0];
        let samples = vec![sample.view()];

        assert!(energy_score(&y_true, &samples, 0.0).is_err()); // Beta must be > 0
        assert!(energy_score(&y_true, &samples, 3.0).is_err()); // Beta must be <= 2
    }
}
