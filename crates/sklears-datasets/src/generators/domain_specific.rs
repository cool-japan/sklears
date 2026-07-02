//! Domain-specific dataset generators (finance, sensors, survival analysis)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Normal, RngExt, StandardNormal};
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

/// Simulate volatility-clustered financial returns using a lightweight GARCH(1,1) process.
///
/// The conditional variance follows the standard GARCH(1,1) recursion
/// `sigma2_t = omega + vol_persistence * sigma2_{t-1} + vol_innovation * eps_{t-1}^2`, and
/// each return is drawn as `r_t = sigma_t * z_t` with `z_t ~ N(0, 1)`.
///
/// `omega` is derived automatically so that the unconditional (long-run) variance of the
/// process matches `initial_vol^2`:
/// `omega = initial_vol^2 * (1 - vol_persistence - vol_innovation)`.
///
/// This requires `vol_persistence + vol_innovation < 1.0`, the standard GARCH(1,1)
/// covariance-stationarity condition. Without it, the implied unconditional variance
/// `omega / (1 - vol_persistence - vol_innovation)` is negative or undefined, so the
/// volatility process would have no finite long-run level and would not be a valid
/// stationary GARCH process. This is a genuine mathematical requirement of the model, not
/// an arbitrary restriction.
///
/// Returns the simulated return series `r_t` (not the underlying volatility path).
pub fn make_financial_returns(
    n_samples: usize,
    initial_vol: f64,
    vol_persistence: f64,
    vol_innovation: f64,
    random_state: Option<u64>,
) -> Result<Array1<f64>> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }
    if initial_vol <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "initial_vol must be positive".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&vol_persistence) {
        return Err(SklearsError::InvalidInput(
            "vol_persistence must lie in [0.0, 1.0)".to_string(),
        ));
    }
    if vol_innovation < 0.0 {
        return Err(SklearsError::InvalidInput(
            "vol_innovation must be non-negative".to_string(),
        ));
    }
    // Covariance-stationarity condition for GARCH(1,1): the unconditional variance
    // omega / (1 - vol_persistence - vol_innovation) is only finite and positive when
    // vol_persistence + vol_innovation < 1. This is a real mathematical requirement of the
    // model (not an arbitrary restriction) -- without it the recursion below has no stable
    // long-run variance level.
    if vol_persistence + vol_innovation >= 1.0 {
        return Err(SklearsError::InvalidInput(
            "vol_persistence + vol_innovation must be < 1.0 (GARCH(1,1) covariance-stationarity condition)"
                .to_string(),
        ));
    }

    // Derive omega so the unconditional (long-run) variance matches initial_vol^2.
    let omega = initial_vol.powi(2) * (1.0 - vol_persistence - vol_innovation);

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let mut sigma2 = vec![0.0_f64; n_samples];
    let mut eps = vec![0.0_f64; n_samples];

    sigma2[0] = initial_vol.powi(2);
    eps[0] = sigma2[0].sqrt() * rng.sample::<f64, _>(StandardNormal);

    for t in 1..n_samples {
        sigma2[t] = omega + vol_persistence * sigma2[t - 1] + vol_innovation * eps[t - 1].powi(2);
        eps[t] = sigma2[t].sqrt() * rng.sample::<f64, _>(StandardNormal);
    }

    // `eps` is the simulated return series r_t, not the volatility path sigma_t.
    Ok(Array1::from_vec(eps))
}

/// Simulate a multi-channel sensor stream sharing one underlying oscillation.
///
/// Every sensor channel observes the same oscillation frequency but with a distinct phase
/// offset `phase_j = 2*pi*j / n_sensors`, which makes channels correlated with each other
/// through `cos(phase_i - phase_j)`. Each channel also carries a shared linear drift term
/// and independent additive Gaussian noise.
pub fn make_sensor_stream(
    n_samples: usize,
    n_sensors: usize,
    drift_rate: f64,
    noise_std: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }
    if n_sensors == 0 {
        return Err(SklearsError::InvalidInput(
            "n_sensors must be at least 1".to_string(),
        ));
    }
    if noise_std < 0.0 {
        return Err(SklearsError::InvalidInput(
            "noise_std must be non-negative".to_string(),
        ));
    }

    // Fixed oscillation period, in samples, shared by all sensor channels.
    const SENSOR_PERIOD: f64 = 50.0;

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let normal = Normal::new(0.0, noise_std).expect("operation should succeed");
    let mut x = Array2::zeros((n_samples, n_sensors));

    for j in 0..n_sensors {
        let phase = 2.0 * PI * j as f64 / n_sensors as f64;
        for t in 0..n_samples {
            let common_signal = (2.0 * PI * t as f64 / SENSOR_PERIOD + phase).sin();
            let drift = drift_rate * t as f64;
            x[[t, j]] = common_signal + drift + rng.sample::<f64, _>(normal);
        }
    }

    Ok(x)
}

/// Simulate right-censored survival data from a Cox-proportional-hazards-style model with a
/// constant (exponential) baseline hazard.
///
/// Each sample's hazard is `hazard_i = baseline_hazard * exp(X_i . beta)` for a fixed,
/// deterministic, alternating-sign, decaying-magnitude linear-predictor coefficient vector
/// `beta` (`beta_j = (1 if j even else -1) * 0.5^(j / 2)`), and the true survival time is
/// drawn as `T_i ~ Exponential(rate = hazard_i)`.
///
/// Censoring is independent random censoring: each sample is censored (independent of `X`
/// and `T_i`) with probability `censoring_rate`. If censored, the recorded time is a value
/// strictly less than `T_i` (a censoring event that cut the observation short) and
/// `event_observed = false`; otherwise the recorded time is the true `T_i` and
/// `event_observed = true`.
///
/// Returns `(X, time, event_observed)`.
pub fn make_survival_data(
    n_samples: usize,
    n_features: usize,
    baseline_hazard: f64,
    censoring_rate: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>, Array1<bool>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }
    if n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "n_features must be at least 1".to_string(),
        ));
    }
    if baseline_hazard <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "baseline_hazard must be positive".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&censoring_rate) {
        return Err(SklearsError::InvalidInput(
            "censoring_rate must lie in [0.0, 1.0]".to_string(),
        ));
    }

    // Fixed, deterministic linear-predictor coefficients (alternating decaying weights) --
    // not returned to the caller, but documented here so the Cox-PH structure is
    // inspectable/reproducible.
    let beta: Vec<f64> = (0..n_features)
        .map(|j| {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            sign * 0.5_f64.powi(j as i32 / 2)
        })
        .collect();

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let mut x = Array2::zeros((n_samples, n_features));
    let mut time = Array1::zeros(n_samples);
    let mut event_observed = Array1::from_elem(n_samples, true);

    for i in 0..n_samples {
        let mut linear_predictor = 0.0;
        for j in 0..n_features {
            let val: f64 = rng.sample(StandardNormal);
            x[[i, j]] = val;
            linear_predictor += val * beta[j];
        }
        let hazard = baseline_hazard * linear_predictor.exp();

        // Inverse-CDF sampling of Exponential(rate = hazard): T = -ln(U) / hazard, U in (0,1].
        // Using `1.0 - rng.random::<f64>()` maps rand's [0,1) draw to (0,1], avoiding ln(0).
        let u_t = 1.0 - rng.random::<f64>();
        let true_time = -u_t.ln() / hazard;

        if rng.random::<f64>() < censoring_rate {
            // Independent censoring: cut the observation short at a random point strictly
            // before true_time.
            let u_c = 1.0 - rng.random::<f64>();
            let censor_time = true_time * u_c; // in (0, true_time)
            time[i] = censor_time;
            event_observed[i] = false;
        } else {
            time[i] = true_time;
            event_observed[i] = true;
        }
    }

    Ok((x, time, event_observed))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pearson correlation coefficient between two equal-length slices, computed directly
    /// from mean/covariance sums (no external stats crate dependency).
    fn manual_correlation(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "slices must have equal length");
        let n = a.len() as f64;
        let mean_a = a.iter().sum::<f64>() / n;
        let mean_b = b.iter().sum::<f64>() / n;
        let cov: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - mean_a) * (y - mean_b))
            .sum();
        let var_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum();
        let var_b: f64 = b.iter().map(|y| (y - mean_b).powi(2)).sum();
        cov / (var_a.sqrt() * var_b.sqrt())
    }

    // ---- make_financial_returns ----

    #[test]
    fn test_make_financial_returns_shape() {
        let returns = make_financial_returns(500, 0.02, 0.85, 0.10, Some(1))
            .expect("operation should succeed");
        assert_eq!(returns.len(), 500);
    }

    #[test]
    fn test_make_financial_returns_seed_determinism() {
        let a = make_financial_returns(200, 0.02, 0.85, 0.10, Some(7))
            .expect("operation should succeed");
        let b = make_financial_returns(200, 0.02, 0.85, 0.10, Some(7))
            .expect("operation should succeed");
        assert_eq!(a, b);
    }

    #[test]
    fn test_make_financial_returns_invalid_vol_persistence_at_one() {
        let result = make_financial_returns(100, 0.02, 1.0, 0.05, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_make_financial_returns_invalid_non_stationary_sum() {
        let result = make_financial_returns(100, 0.02, 0.7, 0.5, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_make_financial_returns_invalid_initial_vol_zero() {
        let result = make_financial_returns(100, 0.0, 0.5, 0.2, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_make_financial_returns_volatility_clustering() {
        let returns =
            make_financial_returns(5000, 1.0, 0.7, 0.2, Some(3)).expect("operation should succeed");

        assert!(returns.iter().all(|v| v.is_finite()));

        let sq: Vec<f64> = returns.iter().map(|v| v * v).collect();
        let corr = manual_correlation(&sq[..sq.len() - 1], &sq[1..]);

        assert!(
            corr > 0.05,
            "expected positive lag-1 autocorrelation in squared returns (volatility clustering), got {corr}"
        );
    }

    // ---- make_sensor_stream ----

    #[test]
    fn test_make_sensor_stream_shape() {
        let x = make_sensor_stream(300, 4, 0.001, 0.1, Some(2)).expect("operation should succeed");
        assert_eq!(x.shape(), &[300, 4]);
    }

    #[test]
    fn test_make_sensor_stream_seed_determinism() {
        let a = make_sensor_stream(200, 3, 0.0, 0.1, Some(9)).expect("operation should succeed");
        let b = make_sensor_stream(200, 3, 0.0, 0.1, Some(9)).expect("operation should succeed");
        assert_eq!(a, b);
    }

    #[test]
    fn test_make_sensor_stream_invalid_n_sensors_zero() {
        let result = make_sensor_stream(100, 0, 0.0, 0.1, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_make_sensor_stream_channel_correlation() {
        let x = make_sensor_stream(2000, 3, 0.0, 0.05, Some(4)).expect("operation should succeed");
        let col0: Vec<f64> = (0..x.nrows()).map(|t| x[[t, 0]]).collect();
        let col1: Vec<f64> = (0..x.nrows()).map(|t| x[[t, 1]]).collect();
        let corr = manual_correlation(&col0, &col1);

        assert!(
            corr.abs() > 0.3,
            "expected substantial correlation between sensor channels sharing a common oscillation, got {corr}"
        );
    }

    // ---- make_survival_data ----

    #[test]
    fn test_make_survival_data_shapes() {
        let (x, time, event_observed) =
            make_survival_data(300, 5, 1.0, 0.3, Some(1)).expect("operation should succeed");
        assert_eq!(x.shape(), &[300, 5]);
        assert_eq!(time.len(), 300);
        assert_eq!(event_observed.len(), 300);
    }

    #[test]
    fn test_make_survival_data_seed_determinism() {
        let (x1, t1, e1) =
            make_survival_data(200, 4, 1.0, 0.2, Some(6)).expect("operation should succeed");
        let (x2, t2, e2) =
            make_survival_data(200, 4, 1.0, 0.2, Some(6)).expect("operation should succeed");
        assert_eq!(x1, x2);
        assert_eq!(t1, t2);
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_make_survival_data_invalid_n_features_zero() {
        let result = make_survival_data(100, 0, 1.0, 0.2, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_make_survival_data_invalid_censoring_rate() {
        let result = make_survival_data(100, 3, 1.0, 1.5, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_make_survival_data_time_finite_positive() {
        let (_, time, _) =
            make_survival_data(500, 4, 1.0, 0.3, Some(2)).expect("operation should succeed");
        assert!(time.iter().all(|&t| t.is_finite() && t > 0.0));
    }

    #[test]
    fn test_make_survival_data_censoring_rate_property() {
        let (_, _, event_observed) =
            make_survival_data(5000, 4, 1.0, 0.3, Some(5)).expect("operation should succeed");
        let censored = event_observed.iter().filter(|&&e| !e).count();
        let fraction_censored = censored as f64 / event_observed.len() as f64;

        assert!(
            (fraction_censored - 0.3).abs() < 0.05,
            "expected censoring fraction near 0.3, got {fraction_censored}"
        );
    }

    #[test]
    fn test_make_survival_data_hazard_monotonicity() {
        let (_, time_low_hazard, _) =
            make_survival_data(5000, 4, 0.5, 0.0, Some(11)).expect("operation should succeed");
        let (_, time_high_hazard, _) =
            make_survival_data(5000, 4, 5.0, 0.0, Some(11)).expect("operation should succeed");

        let mean_low = time_low_hazard.iter().sum::<f64>() / time_low_hazard.len() as f64;
        let mean_high = time_high_hazard.iter().sum::<f64>() / time_high_hazard.len() as f64;

        assert!(
            mean_high < mean_low,
            "higher baseline_hazard should yield shorter expected survival time: mean_high={mean_high}, mean_low={mean_low}"
        );
    }
}
