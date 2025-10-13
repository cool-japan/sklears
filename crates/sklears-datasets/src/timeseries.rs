//! Time series dataset generators
//!
//! This module provides specialized generators for time series data with various
//! types of non-stationarity and temporal patterns.

use scirs2_core::ndarray::Array1;
use scirs2_core::random::{Random, rng};
use scirs2_core::random::distributions::{Normal, StandardNormal};
use sklears_core::error::{Result, SklearsError};

/// Generate non-stationary time series with various types of non-stationarity
///
/// Creates time series data exhibiting different forms of non-stationarity, which is
/// useful for testing time series analysis methods and studying the effects of
/// various temporal patterns.
///
/// # Parameters
/// - `n_samples`: Number of time points to generate
/// - `non_stationary_type`: Type of non-stationarity to introduce
/// - `initial_value`: Starting value for the series
/// - `innovation_std`: Standard deviation of innovations
/// - `parameters`: Additional parameters specific to non-stationarity type
/// - `random_state`: Random seed for reproducibility
///
/// # Non-stationary types
/// - `"random_walk"`: Pure random walk (unit root), parameters ignored
/// - `"changing_variance"`: GARCH-like changing variance, parameters[0] = persistence
/// - `"structural_break"`: Break in mean at parameters[0] * n_samples, parameters[1] = break magnitude
/// - `"time_varying_ar"`: AR coefficient changes over time, parameters[0] = initial AR coeff, parameters[1] = final AR coeff
///
/// # Returns
/// Time series array with specified non-stationary properties
pub fn make_nonstationary_timeseries(
    n_samples: usize,
    non_stationary_type: &str,
    initial_value: f64,
    innovation_std: f64,
    parameters: &[f64],
    random_state: Option<u64>,
) -> Result<Array1<f64>> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut y = Array1::zeros(n_samples);
    y[0] = initial_value;

    match non_stationary_type {
        "random_walk" => {
            // Simple random walk: y_t = y_{t-1} + ε_t
            let noise_dist = Normal::new(0.0, innovation_std).unwrap();

            for t in 1..n_samples {
                let innovation: f64 = rng.sample(noise_dist);
                y[t] = y[t - 1] + innovation;
            }
        }

        "changing_variance" => {
            // GARCH(1,1)-like model: σ²_t = α₀ + α₁ε²_{t-1} + β₁σ²_{t-1}
            if parameters.is_empty() {
                return Err(SklearsError::InvalidInput(
                    "changing_variance requires persistence parameter".to_string(),
                ));
            }

            let persistence = parameters[0].clamp(0.0, 0.99);
            let alpha0 = 0.1;
            let alpha1 = 0.1;
            let beta1 = persistence;

            let mut variance = innovation_std * innovation_std;
            let standard_normal = StandardNormal;

            for t in 1..n_samples {
                let z: f64 = rng.sample(standard_normal);
                let innovation = variance.sqrt() * z;
                y[t] = y[t - 1] + innovation;

                // Update variance for next period
                variance = alpha0 + alpha1 * innovation * innovation + beta1 * variance;
            }
        }

        "structural_break" => {
            // Structural break in mean/trend
            if parameters.len() < 2 {
                return Err(SklearsError::InvalidInput(
                    "structural_break requires break_point and break_magnitude parameters"
                        .to_string(),
                ));
            }

            let break_point = ((parameters[0] * n_samples as f64) as usize).min(n_samples - 1);
            let break_magnitude = parameters[1];
            let noise_dist = Normal::new(0.0, innovation_std).unwrap();

            for t in 1..n_samples {
                let innovation: f64 = rng.sample(noise_dist);
                let mean_shift = if t >= break_point {
                    break_magnitude
                } else {
                    0.0
                };
                y[t] = y[t - 1] + mean_shift + innovation;
            }
        }

        "time_varying_ar" => {
            // AR coefficient changes linearly over time
            if parameters.len() < 2 {
                return Err(SklearsError::InvalidInput(
                    "time_varying_ar requires initial_ar and final_ar parameters".to_string(),
                ));
            }

            let initial_ar = parameters[0].clamp(-0.99, 0.99);
            let final_ar = parameters[1].clamp(-0.99, 0.99);
            let noise_dist = Normal::new(0.0, innovation_std).unwrap();

            for t in 1..n_samples {
                let progress = t as f64 / (n_samples - 1) as f64;
                let ar_coeff = initial_ar + progress * (final_ar - initial_ar);

                let innovation: f64 = rng.sample(noise_dist);
                y[t] = ar_coeff * y[t - 1] + innovation;
            }
        }

        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown non_stationary_type: {}",
                non_stationary_type
            )));
        }
    }

    Ok(y)
}

/// Generate stationary time series with specified ARIMA properties
///
/// Creates stationary time series following ARMA(p,q) models useful for
/// testing time series analysis algorithms.
///
/// # Parameters
/// - `n_samples`: Number of time points to generate
/// - `ar_coeffs`: Autoregressive coefficients (empty for pure MA)
/// - `ma_coeffs`: Moving average coefficients (empty for pure AR)
/// - `innovation_std`: Standard deviation of white noise innovations
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Stationary time series following the specified ARMA model
pub fn make_stationary_arma(
    n_samples: usize,
    ar_coeffs: &[f64],
    ma_coeffs: &[f64],
    innovation_std: f64,
    random_state: Option<u64>,
) -> Result<Array1<f64>> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    // Check stationarity constraints for AR coefficients
    if !ar_coeffs.is_empty() {
        let sum_ar: f64 = ar_coeffs.iter().sum();
        if sum_ar.abs() >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "AR coefficients must satisfy stationarity conditions".to_string(),
            ));
        }
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));
    let noise_dist = Normal::new(0.0, innovation_std).unwrap();

    let p = ar_coeffs.len();
    let q = ma_coeffs.len();
    let burn_in = (p + q + 100).max(100); // Burn-in period
    let total_samples = n_samples + burn_in;

    let mut y = Array1::zeros(total_samples);
    let mut innovations = Array1::zeros(total_samples);

    // Generate innovations
    for t in 0..total_samples {
        innovations[t] = rng.sample(noise_dist);
    }

    // Generate ARMA series
    for t in 1..total_samples {
        let mut value = 0.0;

        // AR component
        for j in 0..p {
            if t > j {
                value += ar_coeffs[j] * y[t - 1 - j];
            }
        }

        // MA component
        for j in 0..=q {
            if t > j {
                let coeff = if j == 0 { 1.0 } else { ma_coeffs[j - 1] };
                value += coeff * innovations[t - j];
            }
        }

        y[t] = value;
    }

    // Return the series after burn-in
    Ok(y.slice(s![burn_in..]).to_owned())
}

use scirs2_core::ndarray::s;