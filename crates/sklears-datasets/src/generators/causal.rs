//! Causal inference and structural model generators
//!
//! Every generator in this module is built around a documented, recoverable
//! ground-truth causal effect (an average treatment effect, an instrumental-variable
//! coefficient, or a regression coefficient vector), so that consumers -- and the
//! tests in this module -- can verify that the data-generating process (DGP) is
//! correct, not merely that its output is shaped correctly.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{RngExt, StandardNormal};
use sklears_core::error::{Result, SklearsError};

/// Standard logistic sigmoid, used to convert a linear index into a propensity score in (0,1).
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Synthetic binary-treatment dataset with a known, recoverable average treatment
/// effect (ATE).
///
/// Draws `n_features` i.i.d. `N(0, 1)` confounder columns `X`; assigns a binary
/// `treatment` via a logistic propensity score `sigmoid(confounding_strength *
/// X[:, 0])`; and generates a continuous `outcome` as
/// `true_ate * treatment + f(X) + eps`, `eps ~ N(0, 1)`, where `f(X)` is the fixed,
/// deterministic decaying-weight linear combination `f(X)_i = sum_j gamma_j *
/// X[i, j]`, `gamma_j = 0.5^j`. `gamma_0` is intentionally nonzero because `X[:, 0]`
/// is also what drives the propensity score above -- that shared dependence on
/// `X[:, 0]` is exactly what makes the confounding real, creating a backdoor path
/// `X0 -> treatment` and `X0 -> outcome`.
///
/// With `confounding_strength = 0.0`, `propensity` is constant `sigmoid(0)=0.5`
/// regardless of `X`, so treatment assignment is independent of the confounders and a
/// naive difference-in-means estimator `mean(outcome | treatment=1) - mean(outcome |
/// treatment=0)` recovers `true_ate` (up to sampling noise). With
/// `confounding_strength != 0.0`, treated and control groups systematically differ in
/// `X[:,0]`, and since `outcome` also depends on `X[:,0]` through `f(X)`, the naive
/// estimator is biased -- recovering the true causal effect then requires
/// conditioning on `X` (e.g. regression adjustment or propensity-score methods).
///
/// Returns `(X, treatment, outcome)`.
///
/// # Errors
/// Returns `SklearsError::InvalidInput` if `n_samples == 0` or `n_features == 0`
/// (at least one confounder column is required).
pub fn make_treatment_effect(
    n_samples: usize,
    n_features: usize,
    true_ate: f64,
    confounding_strength: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>, Array1<f64>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }
    if n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "n_features must be at least 1 (need at least one confounder column)".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    // Fixed, deterministic decaying-weight linear combination: gamma = [1.0, 0.5, 0.25, ...].
    // gamma[0] is intentionally nonzero because X[:, 0] also drives the propensity score
    // below; that shared dependence on X[:, 0] is exactly what makes the confounding real.
    let gamma: Vec<f64> = (0..n_features).map(|j| 0.5_f64.powi(j as i32)).collect();

    let mut x = Array2::zeros((n_samples, n_features));
    let mut treatment = Vec::with_capacity(n_samples);
    let mut outcome = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.sample::<f64, _>(StandardNormal);
        }

        let propensity = sigmoid(confounding_strength * x[[i, 0]]);
        let treated = i32::from(rng.random::<f64>() < propensity);

        let mut f_x = 0.0;
        for (j, &gamma_j) in gamma.iter().enumerate() {
            f_x += gamma_j * x[[i, j]];
        }

        let eps = rng.sample::<f64, _>(StandardNormal);
        let outcome_i = true_ate * f64::from(treated) + f_x + eps;

        treatment.push(treated);
        outcome.push(outcome_i);
    }

    Ok((x, Array1::from_vec(treatment), Array1::from_vec(outcome)))
}

/// Synthetic instrumental-variable (IV) dataset with a known, exactly recoverable
/// causal effect, even under strong confounding.
///
/// Draws an instrument `Z ~ N(0, 1)` and an unobserved confounder `U ~ N(0, 1)`,
/// independent of each other by construction -- the classic IV exogeneity/exclusion
/// assumption -- and generates:
/// - `X = instrument_strength * Z + confounding_strength * U + eps_x`, `eps_x ~ N(0, 0.5)`
/// - `Y = true_effect * X + confounding_strength * U + eps_y`, `eps_y ~ N(0, 0.5)`
///
/// `Y` depends on `X` and `U` but NOT directly on `Z` -- that is the exclusion
/// restriction (`Z` affects `Y` only through `X`).
///
/// Because `Z` is independent of `U` by construction, the Wald/IV estimator
/// `Cov(Y,Z)/Cov(X,Z)` recovers `true_effect` exactly in population regardless of
/// `confounding_strength`, whereas naive OLS of `Y` on `X` is biased whenever
/// `confounding_strength != 0.0`.
///
/// Returns `(instrument Z, confounder U, treatment X, outcome Y)`.
///
/// # Errors
/// Returns `SklearsError::InvalidInput` if `n_samples == 0`. All other parameters
/// (including `instrument_strength == 0.0`, a legitimately "weak instrument" edge
/// case) are meaningful at any real value and are intentionally not validated.
pub fn make_iv_dataset(
    n_samples: usize,
    true_effect: f64,
    instrument_strength: f64,
    confounding_strength: f64,
    random_state: Option<u64>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let mut z = Vec::with_capacity(n_samples);
    let mut u = Vec::with_capacity(n_samples);
    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let z_i = rng.sample::<f64, _>(StandardNormal);
        let u_i = rng.sample::<f64, _>(StandardNormal);
        let eps_x = 0.5 * rng.sample::<f64, _>(StandardNormal);
        let eps_y = 0.5 * rng.sample::<f64, _>(StandardNormal);

        let x_i = instrument_strength * z_i + confounding_strength * u_i + eps_x;
        let y_i = true_effect * x_i + confounding_strength * u_i + eps_y;

        z.push(z_i);
        u.push(u_i);
        x.push(x_i);
        y.push(y_i);
    }

    Ok((
        Array1::from_vec(z),
        Array1::from_vec(u),
        Array1::from_vec(x),
        Array1::from_vec(y),
    ))
}

/// Synthetic linear-regression dataset with a known omitted-variable-bias (OVB)
/// mechanism affecting exactly one feature.
///
/// Draws an unobserved confounder `U ~ N(0, 1)` (never returned to the caller); a
/// first feature column `X[:, 0] = confounding_strength * U + eps0`, `eps0 ~ N(0, 1)`,
/// correlated with `U`; remaining feature columns `X[:, 1..]` drawn i.i.d. `N(0, 1)`
/// and independent of `U`; and a target `y = X . true_coef + U + eps_y`,
/// `eps_y ~ N(0, 0.1)`.
///
/// `U` entering both `X[:,0]` and `y` directly is the omitted-variable-bias
/// mechanism: since the caller never observes `U`, naive OLS of `y` on `X` will have
/// a biased estimate specifically for the coefficient of feature 0 (larger
/// `confounding_strength` implies larger bias), while the other features'
/// coefficients stay approximately unbiased.
///
/// Returns `(X, y)`.
///
/// # Errors
/// Returns `SklearsError::InvalidInput` if `n_samples == 0`, `n_features == 0`, or
/// `true_coef.len() != n_features`.
pub fn make_confounded_regression(
    n_samples: usize,
    n_features: usize,
    true_coef: &Array1<f64>,
    confounding_strength: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
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
    if true_coef.len() != n_features {
        return Err(SklearsError::InvalidInput(format!(
            "true_coef.len() ({}) must equal n_features ({})",
            true_coef.len(),
            n_features
        )));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Unobserved confounder U for this row; used below but never stored/returned.
        let u_i = rng.sample::<f64, _>(StandardNormal);
        let eps0 = rng.sample::<f64, _>(StandardNormal);
        x[[i, 0]] = confounding_strength * u_i + eps0;

        for j in 1..n_features {
            x[[i, j]] = rng.sample::<f64, _>(StandardNormal);
        }

        let mut pred = 0.0;
        for (j, &coef_j) in true_coef.iter().enumerate() {
            pred += x[[i, j]] * coef_j;
        }

        let eps_y = 0.1 * rng.sample::<f64, _>(StandardNormal);
        y.push(pred + u_i + eps_y);
    }

    Ok((x, Array1::from_vec(y)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::linalg::solve_ndarray;

    /// Sample covariance between two equal-length series:
    /// `mean((a - mean(a)) * (b - mean(b)))`.
    fn sample_cov(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        assert_eq!(a.len(), b.len(), "arrays must have equal length");
        let n = a.len() as f64;
        let mean_a = a.iter().sum::<f64>() / n;
        let mean_b = b.iter().sum::<f64>() / n;
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - mean_a) * (bi - mean_b))
            .sum::<f64>()
            / n
    }

    // ---- make_treatment_effect ----

    #[test]
    fn test_make_treatment_effect_shapes() {
        let (x, treatment, outcome) =
            make_treatment_effect(100, 4, 1.5, 0.5, Some(42)).expect("operation should succeed");
        assert_eq!(x.shape(), &[100, 4]);
        assert_eq!(treatment.len(), 100);
        assert_eq!(outcome.len(), 100);
        assert!(treatment.iter().all(|&t| t == 0 || t == 1));
    }

    #[test]
    fn test_make_treatment_effect_seed_determinism() {
        let (x1, t1, o1) =
            make_treatment_effect(50, 3, 1.0, 0.5, Some(7)).expect("operation should succeed");
        let (x2, t2, o2) =
            make_treatment_effect(50, 3, 1.0, 0.5, Some(7)).expect("operation should succeed");
        assert_eq!(x1, x2);
        assert_eq!(t1, t2);
        assert_eq!(o1, o2);
    }

    #[test]
    fn test_make_treatment_effect_invalid_n_features_zero() {
        let result = make_treatment_effect(10, 0, 1.0, 0.5, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_make_treatment_effect_diff_in_means_recovers_ate() {
        // confounding_strength = 0.0 => propensity is constant sigmoid(0) = 0.5, so
        // treatment is independent of X and the naive estimator below is unbiased.
        let (_x, treatment, outcome) =
            make_treatment_effect(20_000, 3, 2.0, 0.0, Some(1)).expect("operation should succeed");

        let mut sum_treated = 0.0;
        let mut n_treated = 0usize;
        let mut sum_control = 0.0;
        let mut n_control = 0usize;

        for (&t, &o) in treatment.iter().zip(outcome.iter()) {
            if t == 1 {
                sum_treated += o;
                n_treated += 1;
            } else {
                sum_control += o;
                n_control += 1;
            }
        }

        assert!(n_treated > 0, "expected at least one treated unit");
        assert!(n_control > 0, "expected at least one control unit");

        let ate_hat = sum_treated / n_treated as f64 - sum_control / n_control as f64;

        assert!(
            (ate_hat - 2.0).abs() < 0.15,
            "naive difference-in-means estimate {ate_hat} too far from true ATE 2.0"
        );
    }

    // ---- make_iv_dataset ----

    #[test]
    fn test_make_iv_dataset_shapes() {
        let (z, u, x, y) =
            make_iv_dataset(200, 1.0, 1.0, 1.0, Some(3)).expect("operation should succeed");
        assert_eq!(z.len(), 200);
        assert_eq!(u.len(), 200);
        assert_eq!(x.len(), 200);
        assert_eq!(y.len(), 200);
    }

    #[test]
    fn test_make_iv_dataset_seed_determinism() {
        let (z1, u1, x1, y1) =
            make_iv_dataset(150, 2.0, 1.2, 0.8, Some(9)).expect("operation should succeed");
        let (z2, u2, x2, y2) =
            make_iv_dataset(150, 2.0, 1.2, 0.8, Some(9)).expect("operation should succeed");
        assert_eq!(z1, z2);
        assert_eq!(u1, u2);
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_make_iv_dataset_wald_estimator_recovers_true_effect() {
        // Strong confounding (confounding_strength = 2.0) is chosen deliberately: naive
        // OLS of y on x, cov(y,x)/var(x), would NOT recover 3.0 here because of the
        // confounding. The Wald/IV estimator below recovers it anyway, which is the
        // whole point of a correct IV setup (Z independent of U by construction).
        let (z, _u, x, y) =
            make_iv_dataset(20_000, 3.0, 1.5, 2.0, Some(1)).expect("operation should succeed");

        let iv_estimate = sample_cov(&y, &z) / sample_cov(&x, &z);

        assert!(
            (iv_estimate - 3.0).abs() < 0.3,
            "Wald/IV estimate {iv_estimate} too far from true_effect 3.0"
        );
    }

    // ---- make_confounded_regression ----

    #[test]
    fn test_make_confounded_regression_shapes() {
        let true_coef = Array1::from_vec(vec![1.0, 2.0, -3.0]);
        let (x, y) = make_confounded_regression(100, 3, &true_coef, 0.5, Some(11))
            .expect("operation should succeed");
        assert_eq!(x.shape(), &[100, 3]);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_make_confounded_regression_seed_determinism() {
        let true_coef = Array1::from_vec(vec![1.0, -1.0]);
        let (x1, y1) = make_confounded_regression(80, 2, &true_coef, 0.3, Some(5))
            .expect("operation should succeed");
        let (x2, y2) = make_confounded_regression(80, 2, &true_coef, 0.3, Some(5))
            .expect("operation should succeed");
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_make_confounded_regression_invalid_coef_length_mismatch() {
        let true_coef = Array1::from_vec(vec![1.0, 2.0]);
        let result = make_confounded_regression(10, 3, &true_coef, 0.5, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_make_confounded_regression_ols_recovers_coefficients_without_confounding() {
        let true_coef = Array1::from_vec(vec![2.0, -1.0]);
        let (x, y) = make_confounded_regression(20_000, 2, &true_coef, 0.0, Some(1))
            .expect("operation should succeed");

        let xt_x = x.t().dot(&x);
        let xt_y = x.t().dot(&y);
        let coef_hat = solve_ndarray(&xt_x, &xt_y).expect("operation should succeed");

        assert!(
            (coef_hat[0] - 2.0).abs() < 0.2,
            "recovered coef[0] {} too far from true 2.0",
            coef_hat[0]
        );
        assert!(
            (coef_hat[1] - (-1.0)).abs() < 0.2,
            "recovered coef[1] {} too far from true -1.0",
            coef_hat[1]
        );
    }
}
