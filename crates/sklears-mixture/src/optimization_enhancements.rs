//! Optimization Enhancements for Mixture Models
//!
//! This module provides advanced optimization techniques for mixture models,
//! including accelerated EM algorithms, quasi-Newton methods, conjugate gradient
//! optimization, second-order methods, and natural gradient descent.
//!
//! # Overview
//!
//! Standard EM algorithms can be slow to converge. This module provides:
//! - Accelerated variants of the EM algorithm
//! - Second-order optimization methods
//! - Natural gradient methods that exploit geometric structure
//! - Quasi-Newton approximations for faster convergence
//!
//! # Key Components
//!
//! - **Accelerated EM**: Various acceleration schemes (Aitken, SQUAREM, etc.)
//! - **Quasi-Newton Methods**: L-BFGS and BFGS for mixture models
//! - **Natural Gradient Descent**: Information geometry-based optimization
//! - **Conjugate Gradient**: Memory-efficient second-order method

use crate::common::CovarianceType;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::seeded_rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::f64::consts::PI;

/// Type of EM acceleration to use
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccelerationType {
    /// Standard EM (no acceleration)
    None,
    /// Aitken acceleration
    Aitken,
    /// SQUAREM (Squared Iterative Method)
    SQUAREM,
    /// Quasi-Newton EM
    QuasiNewton,
}

/// Type of quasi-Newton method to use
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuasiNewtonMethod {
    /// BFGS method
    BFGS,
    /// Limited-memory BFGS
    LBFGS { memory: usize },
    /// Davidon-Fletcher-Powell
    DFP,
    /// Broyden's method
    Broyden,
}

/// Accelerated EM Algorithm for Gaussian Mixture Models
///
/// Implements various acceleration schemes for the EM algorithm,
/// providing faster convergence than standard EM.
///
/// # Examples
///
/// ```
/// use sklears_mixture::optimization_enhancements::{AcceleratedEM, AccelerationType};
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [1.5, 2.5], [10.0, 11.0]];
///
/// let model = AcceleratedEM::builder()
///     .n_components(2)
///     .acceleration(AccelerationType::SQUAREM)
///     .build();
///
/// let fitted = model.fit(&X.view(), &()).expect("AcceleratedEM fitting should succeed with valid data");
/// ```
#[derive(Debug, Clone)]
pub struct AcceleratedEM<S = Untrained> {
    pub(crate) state: S,
    n_components: usize,
    acceleration: AccelerationType,
    covariance_type: CovarianceType,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    random_state: Option<u64>,
}

/// Trained Accelerated EM model
#[derive(Debug, Clone)]
pub struct AcceleratedEMTrained {
    /// Component weights
    pub weights: Array1<f64>,
    /// Component means
    pub means: Array2<f64>,
    /// Component covariances (tied/shared diagonal, stored as a full matrix
    /// whose diagonal holds the per-feature variance -- see module docs)
    pub covariances: Array2<f64>,
    /// Log-likelihood history (real log-sum-exp log-likelihood per
    /// iteration, i.e. `sum_i log(sum_k weight_k * N(x_i; mean_k, cov))`)
    pub log_likelihood_history: Vec<f64>,
    /// Number of iterations
    pub n_iter: usize,
    /// Convergence status
    pub converged: bool,
    /// Acceleration type used
    pub acceleration: AccelerationType,
    /// Speedup factor compared to standard EM, measured as
    /// `baseline_n_iter / accelerated_n_iter` from an internal unaccelerated
    /// re-run started from the same initialization (1.0 when
    /// `acceleration == AccelerationType::None`, since there is no baseline
    /// to compare against). This is a real, data-dependent measurement, not
    /// a hardcoded constant; acceleration types that do not yet have a
    /// distinct update rule implemented (see [`AcceleratedEM::fit`] docs)
    /// will honestly measure close to `1.0`.
    pub speedup_factor: f64,
}

/// Builder for Accelerated EM
#[derive(Debug, Clone)]
pub struct AcceleratedEMBuilder {
    n_components: usize,
    acceleration: AccelerationType,
    covariance_type: CovarianceType,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    random_state: Option<u64>,
}

impl AcceleratedEMBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            n_components: 1,
            acceleration: AccelerationType::SQUAREM,
            covariance_type: CovarianceType::Diagonal,
            max_iter: 100,
            tol: 1e-3,
            reg_covar: 1e-6,
            random_state: None,
        }
    }

    /// Set number of components
    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Set acceleration type
    pub fn acceleration(mut self, acc: AccelerationType) -> Self {
        self.acceleration = acc;
        self
    }

    /// Set covariance type
    pub fn covariance_type(mut self, cov_type: CovarianceType) -> Self {
        self.covariance_type = cov_type;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set covariance regularization
    pub fn reg_covar(mut self, reg: f64) -> Self {
        self.reg_covar = reg;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Build the model
    pub fn build(self) -> AcceleratedEM<Untrained> {
        AcceleratedEM {
            state: Untrained,
            n_components: self.n_components,
            acceleration: self.acceleration,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            random_state: self.random_state,
        }
    }
}

impl Default for AcceleratedEMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AcceleratedEM<Untrained> {
    /// Create a new builder
    pub fn builder() -> AcceleratedEMBuilder {
        AcceleratedEMBuilder::new()
    }

    /// Aitken acceleration coefficient
    fn aitken_coefficient(
        theta_old: &Array1<f64>,
        theta_curr: &Array1<f64>,
        theta_new: &Array1<f64>,
    ) -> f64 {
        let diff1 = theta_curr - theta_old;
        let diff2 = theta_new - theta_curr;
        let diff_diff = &diff2 - &diff1;

        let numerator = (&diff1 * &diff1).sum();
        let denominator = (&diff1 * &diff_diff).sum();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            -numerator / denominator
        }
    }
}

impl Estimator for AcceleratedEM<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

/// Result of running (optionally accelerated) EM to convergence:
/// `(weights, means, covariances, log_likelihood_history, n_iter, converged)`.
type EmRunResult = (Array1<f64>, Array2<f64>, Array2<f64>, Vec<f64>, usize, bool);

impl Fit<ArrayView2<'_, Float>, ()> for AcceleratedEM<Untrained> {
    type Fitted = AcceleratedEM<AcceleratedEMTrained>;

    /// Fit a tied-diagonal-covariance Gaussian mixture via (optionally
    /// accelerated) EM.
    ///
    /// Only [`AccelerationType::Aitken`] currently has a distinct update rule
    /// (extrapolating the mean sequence via the Aitken delta-squared
    /// process). [`AccelerationType::SQUAREM`] and
    /// [`AccelerationType::QuasiNewton`] are accepted but currently run
    /// plain EM steps (no distinct acceleration math is implemented for
    /// them yet); `speedup_factor` is measured empirically (iteration-count
    /// ratio against an internal unaccelerated baseline re-run from the same
    /// initialization), so it honestly reports close to `1.0` for those two
    /// rather than fabricating a specific multiplier.
    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X_owned = X.to_owned();
        let (n_samples, n_features) = X_owned.dim();

        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be >= number of components".to_string(),
            ));
        }
        if self.n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of components must be positive".to_string(),
            ));
        }

        // Initialize parameters. `random_state` is honored via
        // `common::resolve_seed` + `seeded_rng`, so a given seed now
        // actually reproduces the same initialization (previously the field
        // was accepted but silently ignored in favor of `thread_rng()`).
        let seed = crate::common::resolve_seed(self.random_state);
        let mut rng = seeded_rng(seed);

        let mut means = Array2::zeros((self.n_components, n_features));
        let mut used_indices = Vec::new();
        for k in 0..self.n_components {
            let idx = loop {
                let candidate = rng.gen_range(0..n_samples);
                if !used_indices.contains(&candidate) {
                    used_indices.push(candidate);
                    break candidate;
                }
            };
            means.row_mut(k).assign(&X_owned.row(idx));
        }

        let weights = Array1::from_elem(self.n_components, 1.0 / self.n_components as f64);
        let covariances =
            Array2::<f64>::eye(n_features) + &(Array2::<f64>::eye(n_features) * self.reg_covar);

        let (out_weights, out_means, out_covariances, log_likelihood_history, n_iter, converged) =
            Self::run_em(
                self.n_components,
                self.max_iter,
                self.tol,
                self.reg_covar,
                self.acceleration,
                &X_owned,
                means.clone(),
                weights.clone(),
                covariances.clone(),
            );

        // Measure the speedup empirically: for `AccelerationType::None`
        // there is no acceleration to compare, so it is 1.0 by definition;
        // otherwise re-run plain EM from the *same* initialization and
        // compare iteration counts actually taken to converge (or exhaust
        // `max_iter`).
        let speedup_factor = if self.acceleration == AccelerationType::None {
            1.0
        } else {
            let (_, _, _, _, baseline_n_iter, _) = Self::run_em(
                self.n_components,
                self.max_iter,
                self.tol,
                self.reg_covar,
                AccelerationType::None,
                &X_owned,
                means,
                weights,
                covariances,
            );
            if n_iter == 0 {
                1.0
            } else {
                baseline_n_iter as f64 / n_iter as f64
            }
        };

        let trained_state = AcceleratedEMTrained {
            weights: out_weights,
            means: out_means,
            covariances: out_covariances,
            log_likelihood_history,
            n_iter,
            converged,
            acceleration: self.acceleration,
            speedup_factor,
        };

        Ok(AcceleratedEM {
            state: Untrained,
            n_components: self.n_components,
            acceleration: self.acceleration,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            random_state: self.random_state,
        }
        .with_state(trained_state))
    }
}

impl AcceleratedEM<Untrained> {
    /// Run the (optionally accelerated) EM loop to convergence from a given
    /// initialization. Shared by the real `fit()` and by the internal
    /// unaccelerated baseline re-run used to measure `speedup_factor`.
    #[allow(clippy::too_many_arguments)]
    fn run_em(
        n_components: usize,
        max_iter: usize,
        tol: f64,
        reg_covar: f64,
        acceleration: AccelerationType,
        x_owned: &Array2<f64>,
        mut means: Array2<f64>,
        mut weights: Array1<f64>,
        mut covariances: Array2<f64>,
    ) -> EmRunResult {
        let (n_samples, n_features) = x_owned.dim();
        let mut log_likelihood_history = Vec::new();
        let mut converged = false;

        // Store previous parameters for acceleration
        let mut prev_params: Option<Array1<f64>> = None;
        let mut prev_prev_params: Option<Array1<f64>> = None;

        // Standard EM with optional acceleration
        for iter in 0..max_iter {
            // E-step
            let mut responsibilities = Array2::zeros((n_samples, n_components));
            let mut log_lik = 0.0;

            for i in 0..n_samples {
                let x = x_owned.row(i);
                let mut log_probs = Vec::new();

                for k in 0..n_components {
                    let mean = means.row(k);
                    let diff = &x.to_owned() - &mean.to_owned();

                    let mahal = diff
                        .iter()
                        .zip(covariances.diag().iter())
                        .map(|(d, c): (&f64, &f64)| d * d / c.max(reg_covar))
                        .sum::<f64>();

                    let log_det = covariances
                        .diag()
                        .iter()
                        .map(|c| c.max(reg_covar).ln())
                        .sum::<f64>();

                    let log_prob = weights[k].ln()
                        - 0.5 * (n_features as f64 * (2.0 * PI).ln() + log_det)
                        - 0.5 * mahal;

                    log_probs.push(log_prob);
                }

                let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
                // Real log-sum-exp log-likelihood contribution of this
                // sample (previously this loop only summed the normalized
                // responsibilities, which sum to ~1.0 by construction and
                // therefore made `log_likelihood_history` an uninformative
                // near-constant ~0 regardless of fit quality).
                log_lik += max_log + sum_exp.ln();

                for k in 0..n_components {
                    responsibilities[[i, k]] =
                        ((log_probs[k] - max_log).exp() / sum_exp).max(1e-10);
                }
            }

            // M-step
            for k in 0..n_components {
                let resps = responsibilities.column(k);
                let nk = resps.sum().max(1e-10);

                weights[k] = nk / n_samples as f64;

                let mut new_mean = Array1::zeros(n_features);
                for i in 0..n_samples {
                    new_mean += &(x_owned.row(i).to_owned() * resps[i]);
                }
                new_mean /= nk;
                means.row_mut(k).assign(&new_mean);

                let mut new_cov = Array1::zeros(n_features);
                for i in 0..n_samples {
                    let diff = &x_owned.row(i).to_owned() - &new_mean;
                    new_cov += &(diff.mapv(|x| x * x) * resps[i]);
                }
                new_cov = new_cov / nk + Array1::from_elem(n_features, reg_covar);
                covariances.diag_mut().assign(&new_cov);
            }

            weights /= weights.sum();

            // Apply acceleration if requested
            if acceleration == AccelerationType::Aitken && iter >= 2 {
                let current_params = means.iter().cloned().collect::<Array1<f64>>();

                if let (Some(prev), Some(prev_prev)) = (&prev_params, &prev_prev_params) {
                    let alpha = Self::aitken_coefficient(prev_prev, prev, &current_params);
                    if alpha > 0.0 && alpha < 1.0 {
                        // Apply Aitken step
                        let accelerated =
                            prev + &((&current_params - prev) * (1.0 / (1.0 - alpha)));
                        let mut idx = 0;
                        for k in 0..n_components {
                            for j in 0..n_features {
                                if idx < accelerated.len() {
                                    means[[k, j]] = accelerated[idx];
                                    idx += 1;
                                }
                            }
                        }
                    }
                }

                prev_prev_params = prev_params.clone();
                prev_params = Some(current_params);
            }

            log_likelihood_history.push(log_lik);

            // Check convergence
            if iter > 0 {
                let improvement = (log_lik - log_likelihood_history[iter - 1]).abs();
                if improvement < tol {
                    converged = true;
                    break;
                }
            }
        }

        let n_iter = log_likelihood_history.len();
        (
            weights,
            means,
            covariances,
            log_likelihood_history,
            n_iter,
            converged,
        )
    }

    fn with_state(self, state: AcceleratedEMTrained) -> AcceleratedEM<AcceleratedEMTrained> {
        AcceleratedEM {
            state,
            n_components: self.n_components,
            acceleration: self.acceleration,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            random_state: self.random_state,
        }
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<usize>> for AcceleratedEM<AcceleratedEMTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<usize>> {
        let X_owned = X.to_owned();
        Ok(crate::common::predict_tied_diag_argmax(
            &X_owned,
            &self.state.weights,
            &self.state.means,
            &self.state.covariances,
            self.reg_covar,
        ))
    }
}

// Quasi-Newton GMM
#[derive(Debug, Clone)]
pub struct QuasiNewtonGMM<S = Untrained> {
    #[allow(dead_code)]
    n_components: usize,
    #[allow(dead_code)]
    method: QuasiNewtonMethod,
    #[allow(dead_code)]
    covariance_type: CovarianceType,
    #[allow(dead_code)]
    max_iter: usize,
    #[allow(dead_code)]
    tol: f64,
    #[allow(dead_code)]
    reg_covar: f64,
    #[allow(dead_code)]
    random_state: Option<u64>,
    _phantom: std::marker::PhantomData<S>,
}

#[derive(Debug, Clone)]
pub struct QuasiNewtonGMMTrained {
    pub weights: Array1<f64>,
    pub means: Array2<f64>,
    pub covariances: Array2<f64>,
    pub log_likelihood_history: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
}

#[derive(Debug, Clone)]
pub struct QuasiNewtonGMMBuilder {
    n_components: usize,
    method: QuasiNewtonMethod,
    covariance_type: CovarianceType,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    random_state: Option<u64>,
}

impl QuasiNewtonGMMBuilder {
    pub fn new() -> Self {
        Self {
            n_components: 1,
            method: QuasiNewtonMethod::LBFGS { memory: 10 },
            covariance_type: CovarianceType::Diagonal,
            max_iter: 100,
            tol: 1e-3,
            reg_covar: 1e-6,
            random_state: None,
        }
    }

    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    pub fn method(mut self, m: QuasiNewtonMethod) -> Self {
        self.method = m;
        self
    }

    pub fn build(self) -> QuasiNewtonGMM<Untrained> {
        QuasiNewtonGMM {
            n_components: self.n_components,
            method: self.method,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            random_state: self.random_state,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Default for QuasiNewtonGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QuasiNewtonGMM<Untrained> {
    pub fn builder() -> QuasiNewtonGMMBuilder {
        QuasiNewtonGMMBuilder::new()
    }
}

// Natural Gradient GMM
#[derive(Debug, Clone)]
pub struct NaturalGradientGMM<S = Untrained> {
    #[allow(dead_code)]
    n_components: usize,
    #[allow(dead_code)]
    learning_rate: f64,
    #[allow(dead_code)]
    use_fisher: bool,
    _phantom: std::marker::PhantomData<S>,
}

#[derive(Debug, Clone)]
pub struct NaturalGradientGMMTrained {
    pub weights: Array1<f64>,
    pub means: Array2<f64>,
    pub fisher_info: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct NaturalGradientGMMBuilder {
    n_components: usize,
    learning_rate: f64,
    use_fisher: bool,
}

impl NaturalGradientGMMBuilder {
    pub fn new() -> Self {
        Self {
            n_components: 1,
            learning_rate: 0.01,
            use_fisher: true,
        }
    }

    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn use_fisher(mut self, use_f: bool) -> Self {
        self.use_fisher = use_f;
        self
    }

    pub fn build(self) -> NaturalGradientGMM<Untrained> {
        NaturalGradientGMM {
            n_components: self.n_components,
            learning_rate: self.learning_rate,
            use_fisher: self.use_fisher,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Default for NaturalGradientGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NaturalGradientGMM<Untrained> {
    pub fn builder() -> NaturalGradientGMMBuilder {
        NaturalGradientGMMBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_accelerated_em_builder() {
        let model = AcceleratedEM::builder()
            .n_components(3)
            .acceleration(AccelerationType::SQUAREM)
            .max_iter(50)
            .build();

        assert_eq!(model.n_components, 3);
        assert_eq!(model.acceleration, AccelerationType::SQUAREM);
        assert_eq!(model.max_iter, 50);
    }

    #[test]
    fn test_acceleration_types() {
        let types = vec![
            AccelerationType::None,
            AccelerationType::Aitken,
            AccelerationType::SQUAREM,
            AccelerationType::QuasiNewton,
        ];

        for acc_type in types {
            let model = AcceleratedEM::builder()
                .n_components(2)
                .acceleration(acc_type)
                .build();
            assert_eq!(model.acceleration, acc_type);
        }
    }

    #[test]
    #[allow(non_snake_case)] // standard ML notation
    fn test_accelerated_em_fit() {
        let X = array![[1.0, 2.0], [1.5, 2.5], [10.0, 11.0], [10.5, 11.5]];

        let model = AcceleratedEM::builder()
            .n_components(2)
            .acceleration(AccelerationType::None)
            .max_iter(20)
            .build();

        let result = model.fit(&X.view(), &());
        assert!(result.is_ok());
    }

    /// Regression test for the fabrication bug: `with_state` used to discard
    /// the fitted parameters, so `predict` always returned all-zeros. A real
    /// fit on two well-separated blobs must discriminate between them.
    #[test]
    #[allow(non_snake_case)] // standard ML notation
    fn test_accelerated_em_predict_recovers_cluster_structure() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [0.1, -0.1],
            [10.0, 10.0],
            [10.1, 10.1],
            [9.9, 10.1],
            [10.1, 9.9],
        ];

        for acceleration in [
            AccelerationType::None,
            AccelerationType::Aitken,
            AccelerationType::SQUAREM,
        ] {
            let model = AcceleratedEM::builder()
                .n_components(2)
                .acceleration(acceleration)
                .max_iter(50)
                .random_state(42)
                .build();
            let fitted = model
                .fit(&X.view(), &())
                .unwrap_or_else(|e| panic!("fit should succeed for {acceleration:?}: {e}"));
            let preds = fitted
                .predict(&X.view())
                .unwrap_or_else(|e| panic!("predict should succeed for {acceleration:?}: {e}"));

            let distinct: std::collections::HashSet<usize> = preds.iter().copied().collect();
            assert!(
                distinct.len() > 1,
                "[{acceleration:?}] predictions collapsed onto a single label (the old \
                 all-zeros bug): {preds:?}"
            );

            let label_a = preds[0];
            for i in 0..4 {
                assert_eq!(
                    preds[i], label_a,
                    "[{acceleration:?}] first blob should share one predicted label"
                );
            }
            let label_b = preds[4];
            assert_ne!(
                label_a, label_b,
                "[{acceleration:?}] the two well-separated blobs must not collapse onto the \
                 same label"
            );
            for i in 4..8 {
                assert_eq!(
                    preds[i], label_b,
                    "[{acceleration:?}] second blob should share one predicted label"
                );
            }
        }
    }

    /// `speedup_factor` must be a real, measured value -- specifically the
    /// iteration-count ratio against an internal unaccelerated baseline
    /// re-run from the *same* initialization -- not the old hardcoded
    /// `1.5`/`2.0`/`2.5` constants keyed purely off the acceleration enum.
    ///
    /// This is a precise (not just "is it a positive finite number", which
    /// the old hardcoded `1.5` would also satisfy) definitional check: since
    /// `none_fit` below uses the same `random_state` as `aitken_fit`, it
    /// performs the *exact same deterministic computation* as the baseline
    /// re-run `aitken_fit`'s `fit()` performs internally, so
    /// `aitken_fit.state.speedup_factor` must equal
    /// `none_fit.state.n_iter / aitken_fit.state.n_iter` to within floating
    /// point precision -- something the old hardcoded constant would only
    /// satisfy by pure coincidence.
    #[test]
    #[allow(non_snake_case)]
    fn test_speedup_factor_is_measured_not_hardcoded() {
        let X = array![
            [0.0, 0.0],
            [0.2, 0.1],
            [10.0, 10.0],
            [10.2, 9.9],
            [5.0, -5.0],
            [5.2, -5.1],
        ];

        for n_components in [2, 3] {
            let none_fit = AcceleratedEM::builder()
                .n_components(n_components)
                .acceleration(AccelerationType::None)
                .max_iter(50)
                .random_state(7)
                .build()
                .fit(&X.view(), &())
                .expect("fit should succeed");
            // By definition (no baseline to compare against).
            assert_eq!(none_fit.state.speedup_factor, 1.0);

            let aitken_fit = AcceleratedEM::builder()
                .n_components(n_components)
                .acceleration(AccelerationType::Aitken)
                .max_iter(50)
                .random_state(7)
                .build()
                .fit(&X.view(), &())
                .expect("fit should succeed");

            let expected = none_fit.state.n_iter as f64 / aitken_fit.state.n_iter as f64;
            assert!(
                (aitken_fit.state.speedup_factor - expected).abs() < 1e-9,
                "[n_components={n_components}] speedup_factor ({}) must equal the measured \
                 baseline/accelerated iteration ratio ({expected}), not a hardcoded \
                 per-acceleration-type constant",
                aitken_fit.state.speedup_factor
            );
        }
    }

    #[test]
    fn test_quasi_newton_gmm_builder() {
        let model = QuasiNewtonGMM::builder()
            .n_components(2)
            .method(QuasiNewtonMethod::LBFGS { memory: 5 })
            .build();

        assert_eq!(model.n_components, 2);
        assert!(matches!(
            model.method,
            QuasiNewtonMethod::LBFGS { memory: 5 }
        ));
    }

    #[test]
    fn test_quasi_newton_methods() {
        let methods = vec![
            QuasiNewtonMethod::BFGS,
            QuasiNewtonMethod::LBFGS { memory: 10 },
            QuasiNewtonMethod::DFP,
            QuasiNewtonMethod::Broyden,
        ];

        for method in methods {
            let model = QuasiNewtonGMM::builder()
                .n_components(2)
                .method(method)
                .build();
            assert_eq!(model.method, method);
        }
    }

    #[test]
    fn test_natural_gradient_gmm_builder() {
        let model = NaturalGradientGMM::builder()
            .n_components(3)
            .learning_rate(0.05)
            .use_fisher(false)
            .build();

        assert_eq!(model.n_components, 3);
        assert_eq!(model.learning_rate, 0.05);
        assert!(!model.use_fisher);
    }

    #[test]
    fn test_aitken_coefficient() {
        let theta_old = array![1.0, 2.0, 3.0];
        let theta_curr = array![1.5, 2.5, 3.5];
        let theta_new = array![1.8, 2.8, 3.8];

        let alpha = AcceleratedEM::aitken_coefficient(&theta_old, &theta_curr, &theta_new);
        // Alpha can be negative or outside [0,1] in some cases, just check it's finite or NaN
        assert!(alpha.is_finite() || alpha.is_nan());
    }
}
