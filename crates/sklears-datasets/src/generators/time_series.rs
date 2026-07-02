//! Time series and temporal data generators.
//!
//! This module provides synthetic generators for common stochastic and
//! deterministic processes used in time-series forecasting research and
//! testing: autoregressive (AR), moving-average (MA), and combined ARMA
//! processes, a deterministic trend-plus-seasonality generator, and a
//! (drifted) random walk.
//!
//! # Design: raw series, not windowed features
//!
//! Every function here returns the **raw 1-D time series** as `Array1<f64>`
//! of length `n_samples` — NOT windowed lag-feature matrices. These
//! generators return the raw simulated series. Callers who need supervised
//! windowed features (e.g. `lag_1..lag_k -> next value`) can build them from
//! the returned `Array1<f64>` via a simple sliding-window pass; that
//! windowing step is intentionally left to the caller since window size is a
//! modeling choice, not a generation-time concern.

use scirs2_core::ndarray::Array1;
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Normal, RngExt};
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

/// Simulate an autoregressive AR(p) process.
///
/// `y_t = sum_{i=0}^{order-1} coeffs[i] * y_{t-1-i} + eps_t`, with
/// `eps_t ~ N(0, noise_std)`.
///
/// The recursion is initialized with an arbitrary all-zero pre-history for
/// the first `order` values. Without correcting for this, the returned
/// series would carry a transient near the start that biases the variance
/// of the early samples away from the process's true stationary variance.
/// To avoid that, this function simulates a **burn-in** period of
/// `(order * 20).max(50)` extra steps before the requested `n_samples` and
/// discards them, so the samples that are actually returned come from
/// (approximately) the stationary distribution of the process.
///
/// # Errors
/// Returns `SklearsError::InvalidInput` if `coeffs.len() != order`, if
/// `n_samples <= order`, or if `noise_std < 0.0`.
pub fn make_ar_process(
    n_samples: usize,
    order: usize,
    coeffs: &[f64],
    noise_std: f64,
    random_state: Option<u64>,
) -> Result<Array1<f64>> {
    if coeffs.len() != order {
        return Err(SklearsError::InvalidInput(
            "coeffs.len() must equal order".to_string(),
        ));
    }
    if n_samples <= order {
        return Err(SklearsError::InvalidInput(
            "n_samples must exceed the AR order".to_string(),
        ));
    }
    if noise_std < 0.0 {
        return Err(SklearsError::InvalidInput(
            "noise_std must be non-negative".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let normal = Normal::new(0.0, noise_std).expect("operation should succeed");

    // Burn-in: simulate this many extra steps up front and discard them so
    // the returned series starts from (approximately) the stationary
    // distribution instead of the arbitrary zero pre-history.
    let burn_in = (order * 20).max(50);
    let total = burn_in + n_samples;

    // `y` is laid out with `order` zero-valued pre-history entries at the
    // front, followed by `total` simulated values.
    let mut y = vec![0.0_f64; order + total];
    for t in 0..total {
        // Safe: coeffs.len() == order, so for i in 0..order the index
        // `order + t - 1 - i` ranges within [0, order + total - 1] since
        // i <= order - 1; it never underflows. When order == 0, coeffs is
        // empty and the closure below is never invoked, so the (otherwise
        // underflowing) index expression is never evaluated either.
        let ar_term: f64 = coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| c * y[order + t - 1 - i])
            .sum();
        let eps: f64 = rng.sample(normal);
        y[order + t] = ar_term + eps;
    }

    let series: Vec<f64> = y[(order + burn_in)..(order + total)].to_vec();
    Ok(Array1::from_vec(series))
}

/// Simulate a moving-average MA(q) process.
///
/// `y_t = eps_t + sum_{i=0}^{order-1} coeffs[i] * eps_{t-1-i}`, with
/// `eps_t ~ N(0, noise_std)`.
///
/// Unlike [`make_ar_process`], an MA process has no autoregressive feedback:
/// each `y_t` is a fixed linear combination of a bounded window of i.i.d.
/// noise terms, so the process is stationary from the very first sample and
/// needs **no burn-in**. We do still draw `order` extra "pre-history" noise
/// samples (real random draws, not zeros) so that the earliest returned
/// samples are not biased by an artificial zero history.
///
/// # Errors
/// Returns `SklearsError::InvalidInput` if `coeffs.len() != order`, if
/// `n_samples <= order`, or if `noise_std < 0.0`.
pub fn make_ma_process(
    n_samples: usize,
    order: usize,
    coeffs: &[f64],
    noise_std: f64,
    random_state: Option<u64>,
) -> Result<Array1<f64>> {
    if coeffs.len() != order {
        return Err(SklearsError::InvalidInput(
            "coeffs.len() must equal order".to_string(),
        ));
    }
    if n_samples <= order {
        return Err(SklearsError::InvalidInput(
            "n_samples must exceed the MA order".to_string(),
        ));
    }
    if noise_std < 0.0 {
        return Err(SklearsError::InvalidInput(
            "noise_std must be non-negative".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let normal = Normal::new(0.0, noise_std).expect("operation should succeed");

    let mut eps = vec![0.0_f64; order + n_samples];
    for e in eps.iter_mut() {
        *e = rng.sample(normal);
    }

    let mut y = Array1::zeros(n_samples);
    for t in 0..n_samples {
        let idx = order + t;
        let ma_term: f64 = coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| c * eps[idx - 1 - i])
            .sum();
        y[t] = eps[idx] + ma_term;
    }
    Ok(y)
}

/// Simulate a combined ARMA(p, q) process.
///
/// `y_t = sum_i ar_coeffs[i] * y_{t-1-i} + eps_t + sum_j ma_coeffs[j] * eps_{t-1-j}`,
/// with `eps_t ~ N(0, noise_std)`.
///
/// The AR and MA orders are simply `ar_coeffs.len()` (`p`) and
/// `ma_coeffs.len()` (`q`) — there is no separate `order` parameter to
/// cross-validate against, since the slice lengths already are the orders.
/// As with [`make_ar_process`], the process has autoregressive feedback
/// whenever `ar_coeffs` is non-empty, so a burn-in period of
/// `(max(p, q) * 20).max(50)` steps is simulated and discarded before the
/// requested samples. Passing an empty `ma_coeffs` degrades this to a pure
/// AR(p) process, an empty `ar_coeffs` degrades it to a pure MA(q) process,
/// and passing both empty produces (burnt-in, though burn-in is irrelevant
/// here) white noise.
///
/// # Errors
/// Returns `SklearsError::InvalidInput` if `n_samples` does not exceed
/// `max(ar_coeffs.len(), ma_coeffs.len())`, or if `noise_std < 0.0`.
pub fn make_arma_process(
    n_samples: usize,
    ar_coeffs: &[f64],
    ma_coeffs: &[f64],
    noise_std: f64,
    random_state: Option<u64>,
) -> Result<Array1<f64>> {
    let p = ar_coeffs.len();
    let q = ma_coeffs.len();
    let max_order = p.max(q);

    if n_samples <= max_order {
        return Err(SklearsError::InvalidInput(
            "n_samples must exceed the larger of the AR and MA orders".to_string(),
        ));
    }
    if noise_std < 0.0 {
        return Err(SklearsError::InvalidInput(
            "noise_std must be non-negative".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let normal = Normal::new(0.0, noise_std).expect("operation should succeed");

    let burn_in = (max_order * 20).max(50);
    let total = burn_in + n_samples;

    let mut eps = vec![0.0_f64; q + total];
    for e in eps.iter_mut() {
        *e = rng.sample(normal);
    }

    let mut y = vec![0.0_f64; p + total];
    for t in 0..total {
        let ar_term: f64 = ar_coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| c * y[p + t - 1 - i])
            .sum();
        let ma_term: f64 = ma_coeffs
            .iter()
            .enumerate()
            .map(|(j, &c)| c * eps[q + t - 1 - j])
            .sum();
        y[p + t] = ar_term + eps[q + t] + ma_term;
    }

    let series: Vec<f64> = y[(p + burn_in)..(p + total)].to_vec();
    Ok(Array1::from_vec(series))
}

/// Simulate a series with linear trend, sinusoidal seasonality, and Gaussian
/// noise.
///
/// `y_t = trend_slope * t + amplitude * sin(2*PI*t/period) + noise_t`, with
/// `noise_t ~ N(0, noise_std)`.
///
/// # Errors
/// Returns `SklearsError::InvalidInput` if `period < 1`, if
/// `n_samples <= period`, or if `noise_std < 0.0`.
pub fn make_seasonal_trend(
    n_samples: usize,
    period: usize,
    trend_slope: f64,
    amplitude: f64,
    noise_std: f64,
    random_state: Option<u64>,
) -> Result<Array1<f64>> {
    if period < 1 {
        return Err(SklearsError::InvalidInput(
            "period must be at least 1".to_string(),
        ));
    }
    if n_samples <= period {
        return Err(SklearsError::InvalidInput(
            "n_samples must exceed one full period".to_string(),
        ));
    }
    if noise_std < 0.0 {
        return Err(SklearsError::InvalidInput(
            "noise_std must be non-negative".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let normal = Normal::new(0.0, noise_std).expect("operation should succeed");

    let mut y = Array1::zeros(n_samples);
    for t in 0..n_samples {
        let trend = trend_slope * t as f64;
        let seasonal = amplitude * (2.0 * PI * t as f64 / period as f64).sin();
        y[t] = trend + seasonal + rng.sample::<f64, _>(normal);
    }
    Ok(y)
}

/// Simulate a random walk with optional drift.
///
/// `y_0 = 0`, `y_t = y_{t-1} + drift + step_t`, with `step_t ~ N(0, step_std)`.
///
/// # Errors
/// Returns `SklearsError::InvalidInput` if `n_samples < 1` or if
/// `step_std < 0.0`.
pub fn make_random_walk(
    n_samples: usize,
    step_std: f64,
    drift: f64,
    random_state: Option<u64>,
) -> Result<Array1<f64>> {
    if n_samples < 1 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be at least 1".to_string(),
        ));
    }
    if step_std < 0.0 {
        return Err(SklearsError::InvalidInput(
            "step_std must be non-negative".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let normal = Normal::new(0.0, step_std).expect("operation should succeed");

    let mut y = Array1::zeros(n_samples);
    for t in 1..n_samples {
        y[t] = y[t - 1] + drift + rng.sample::<f64, _>(normal);
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // make_ar_process
    // ---------------------------------------------------------------

    #[test]
    fn test_make_ar_process_length() {
        let series =
            make_ar_process(200, 2, &[0.5, -0.2], 1.0, Some(42)).expect("operation should succeed");
        assert_eq!(series.len(), 200);
    }

    #[test]
    fn test_make_ar_process_deterministic_with_seed() {
        let a =
            make_ar_process(150, 2, &[0.5, -0.2], 1.0, Some(7)).expect("operation should succeed");
        let b =
            make_ar_process(150, 2, &[0.5, -0.2], 1.0, Some(7)).expect("operation should succeed");
        assert_eq!(a, b);
    }

    #[test]
    fn test_make_ar_process_invalid_coeffs_len() {
        // order says 2 but only 1 coefficient supplied
        assert!(make_ar_process(100, 2, &[0.5], 1.0, Some(42)).is_err());
    }

    #[test]
    fn test_make_ar_process_invalid_n_samples() {
        // n_samples must exceed order
        assert!(make_ar_process(2, 2, &[0.5, -0.2], 1.0, Some(42)).is_err());
    }

    #[test]
    fn test_make_ar_process_invalid_noise_std() {
        assert!(make_ar_process(100, 1, &[0.5], -1.0, Some(42)).is_err());
    }

    #[test]
    fn test_ar1_stationary_variance() {
        // Theoretical stationary variance of AR(1) is
        // noise_std^2 / (1 - a^2) = 1.0 / (1 - 0.25) = 1.3333...
        let series =
            make_ar_process(20_000, 1, &[0.5], 1.0, Some(42)).expect("operation should succeed");
        assert!(series.iter().all(|v| v.is_finite()));
        let empirical_var = series.var(0.0);
        assert!(
            (empirical_var - 1.3333).abs() < 0.3,
            "empirical variance {empirical_var} too far from theoretical 1.3333"
        );
    }

    // ---------------------------------------------------------------
    // make_ma_process
    // ---------------------------------------------------------------

    #[test]
    fn test_make_ma_process_length() {
        let series =
            make_ma_process(200, 2, &[0.5, -0.2], 1.0, Some(42)).expect("operation should succeed");
        assert_eq!(series.len(), 200);
    }

    #[test]
    fn test_make_ma_process_deterministic_with_seed() {
        let a =
            make_ma_process(150, 2, &[0.5, -0.2], 1.0, Some(7)).expect("operation should succeed");
        let b =
            make_ma_process(150, 2, &[0.5, -0.2], 1.0, Some(7)).expect("operation should succeed");
        assert_eq!(a, b);
    }

    #[test]
    fn test_make_ma_process_invalid_coeffs_len() {
        assert!(make_ma_process(100, 2, &[0.5], 1.0, Some(42)).is_err());
    }

    #[test]
    fn test_make_ma_process_invalid_n_samples() {
        assert!(make_ma_process(2, 2, &[0.5, -0.2], 1.0, Some(42)).is_err());
    }

    #[test]
    fn test_make_ma_process_invalid_noise_std() {
        assert!(make_ma_process(100, 1, &[0.5], -1.0, Some(42)).is_err());
    }

    #[test]
    fn test_ma1_stationary_variance() {
        // Theoretical variance of MA(1) is
        // noise_std^2 * (1 + coeffs[0]^2) = 1.0 * (1 + 0.25) = 1.25
        let series =
            make_ma_process(20_000, 1, &[0.5], 1.0, Some(42)).expect("operation should succeed");
        assert!(series.iter().all(|v| v.is_finite()));
        let empirical_var = series.var(0.0);
        assert!(
            (empirical_var - 1.25).abs() < 0.3,
            "empirical variance {empirical_var} too far from theoretical 1.25"
        );
    }

    // ---------------------------------------------------------------
    // make_arma_process
    // ---------------------------------------------------------------

    #[test]
    fn test_make_arma_process_length() {
        let series = make_arma_process(300, &[0.4], &[0.3], 1.0, Some(42))
            .expect("operation should succeed");
        assert_eq!(series.len(), 300);
    }

    #[test]
    fn test_make_arma_process_deterministic_with_seed() {
        let a = make_arma_process(150, &[0.4, -0.1], &[0.3], 1.0, Some(9))
            .expect("operation should succeed");
        let b = make_arma_process(150, &[0.4, -0.1], &[0.3], 1.0, Some(9))
            .expect("operation should succeed");
        assert_eq!(a, b);
    }

    #[test]
    fn test_make_arma_process_invalid_n_samples() {
        // max_order = max(2, 1) = 2, so n_samples must exceed 2
        assert!(make_arma_process(2, &[0.4, -0.1], &[0.3], 1.0, Some(42)).is_err());
    }

    #[test]
    fn test_make_arma_process_invalid_noise_std() {
        assert!(make_arma_process(100, &[0.4], &[0.3], -1.0, Some(42)).is_err());
    }

    #[test]
    fn test_make_arma_process_degrades_to_pure_ar() {
        let series =
            make_arma_process(200, &[0.5], &[], 1.0, Some(42)).expect("operation should succeed");
        assert_eq!(series.len(), 200);
        assert!(series.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_make_arma_process_degrades_to_pure_ma() {
        let series =
            make_arma_process(200, &[], &[0.5], 1.0, Some(42)).expect("operation should succeed");
        assert_eq!(series.len(), 200);
        assert!(series.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_make_arma_process_degrades_to_white_noise() {
        let series =
            make_arma_process(200, &[], &[], 1.0, Some(42)).expect("operation should succeed");
        assert_eq!(series.len(), 200);
        assert!(series.iter().all(|v| v.is_finite()));
    }

    // ---------------------------------------------------------------
    // make_seasonal_trend
    // ---------------------------------------------------------------

    #[test]
    fn test_make_seasonal_trend_length() {
        let series = make_seasonal_trend(500, 50, 0.01, 5.0, 0.5, Some(42))
            .expect("operation should succeed");
        assert_eq!(series.len(), 500);
    }

    #[test]
    fn test_make_seasonal_trend_deterministic_with_seed() {
        let a = make_seasonal_trend(300, 20, 0.01, 5.0, 0.5, Some(3))
            .expect("operation should succeed");
        let b = make_seasonal_trend(300, 20, 0.01, 5.0, 0.5, Some(3))
            .expect("operation should succeed");
        assert_eq!(a, b);
    }

    #[test]
    fn test_make_seasonal_trend_invalid_period() {
        assert!(make_seasonal_trend(100, 0, 0.01, 5.0, 0.5, Some(42)).is_err());
    }

    #[test]
    fn test_make_seasonal_trend_invalid_n_samples() {
        // n_samples must exceed one full period
        assert!(make_seasonal_trend(10, 50, 0.01, 5.0, 0.5, Some(42)).is_err());
    }

    #[test]
    fn test_make_seasonal_trend_invalid_noise_std() {
        assert!(make_seasonal_trend(100, 10, 0.01, 5.0, -0.5, Some(42)).is_err());
    }

    #[test]
    fn test_make_seasonal_trend_periodic_with_zero_noise() {
        let period = 50;
        let series = make_seasonal_trend(300, period, 0.0, 5.0, 0.0, Some(42))
            .expect("operation should succeed");
        for t in [0usize, 10, 25, 49, 100] {
            assert!(t + period < series.len());
            let diff = (series[t] - series[t + period]).abs();
            assert!(diff < 1e-9, "series not periodic at t={t}: diff={diff}");
        }
    }

    // ---------------------------------------------------------------
    // make_random_walk
    // ---------------------------------------------------------------

    #[test]
    fn test_make_random_walk_length() {
        let series = make_random_walk(500, 1.0, 0.1, Some(42)).expect("operation should succeed");
        assert_eq!(series.len(), 500);
    }

    #[test]
    fn test_make_random_walk_deterministic_with_seed() {
        let a = make_random_walk(300, 1.0, 0.1, Some(5)).expect("operation should succeed");
        let b = make_random_walk(300, 1.0, 0.1, Some(5)).expect("operation should succeed");
        assert_eq!(a, b);
    }

    #[test]
    fn test_make_random_walk_invalid_n_samples() {
        assert!(make_random_walk(0, 1.0, 0.1, Some(42)).is_err());
    }

    #[test]
    fn test_make_random_walk_invalid_step_std() {
        assert!(make_random_walk(100, -1.0, 0.1, Some(42)).is_err());
    }

    #[test]
    fn test_make_random_walk_increment_statistics() {
        // The increments of a single random-walk path are themselves i.i.d.,
        // so the law of large numbers applies within one path: their sample
        // mean should be close to `drift` and their sample std close to
        // `step_std`.
        let step_std = 1.0;
        let drift = 0.1;
        let series =
            make_random_walk(5000, step_std, drift, Some(42)).expect("operation should succeed");

        let diffs: Vec<f64> = (1..series.len())
            .map(|t| series[t] - series[t - 1])
            .collect();
        let n = diffs.len() as f64;
        let mean: f64 = diffs.iter().sum::<f64>() / n;
        let variance: f64 = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        assert!(
            (mean - drift).abs() < 0.15,
            "increment mean {mean} too far from drift {drift}"
        );
        assert!(
            (std - step_std).abs() < 0.15,
            "increment std {std} too far from step_std {step_std}"
        );
    }
}
