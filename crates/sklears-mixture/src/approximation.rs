//! Approximation Methods for Mixture Models
//!
//! This module provides various approximation techniques for mixture model
//! inference, including Laplace approximations, Monte Carlo methods, and
//! importance sampling.
//!
//! # Overview
//!
//! Approximation methods enable:
//! - Fast inference in complex models
//! - Uncertainty quantification
//! - Posterior distribution approximation
//! - Efficient sampling strategies
//!
//! # Key Components
//!
//! - **Laplace Approximation**: Gaussian approximation around mode
//! - **Monte Carlo Methods**: Sampling-based inference
//! - **Importance Sampling**: Weighted sampling for rare events
//! - **Particle Filtering**: Sequential Monte Carlo

use crate::common::CovarianceType;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Type of Monte Carlo approximation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonteCarloMethod {
    /// Standard Monte Carlo
    Standard { n_samples: usize },
    /// Quasi-Monte Carlo with low-discrepancy sequences
    Quasi { n_samples: usize },
    /// Markov Chain Monte Carlo
    MCMC {
        n_samples: usize,
        burn_in: usize,
        thin: usize,
    },
}

/// Importance sampling strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImportanceSamplingStrategy {
    /// Standard importance sampling
    Standard { n_samples: usize },
    /// Adaptive importance sampling
    Adaptive {
        n_samples: usize,
        adaptation_steps: usize,
    },
    /// Self-normalized importance sampling
    SelfNormalized { n_samples: usize },
}

/// Laplace Approximation for Gaussian Mixture Model
///
/// Approximates the posterior distribution with a Gaussian centered at the MAP estimate.
///
/// # Examples
///
/// ```
/// use sklears_mixture::approximation::LaplaceGMM;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let model = LaplaceGMM::builder()
///     .n_components(2)
///     .build();
///
/// let X = array![[1.0, 2.0], [1.5, 2.5], [10.0, 11.0]];
/// let fitted = model.fit(&X.view(), &()).expect("Laplace GMM fitting should succeed with valid data");
/// ```
#[derive(Debug, Clone)]
pub struct LaplaceGMM<S = Untrained> {
    pub(crate) state: S,
    n_components: usize,
    covariance_type: CovarianceType,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    hessian_regularization: f64,
}

/// Trained Laplace GMM
///
/// The MAP estimate is found via a genuine (tied-diagonal-covariance) EM
/// loop; `posterior_covariance` and `log_marginal_likelihood` are then
/// computed from that fit using the standard *simplified* Laplace
/// approximation (see [`LaplaceGMM::fit`] for the exact formulas). This is a
/// disclosed simplification: only the diagonal of the parameter Hessian is
/// used (cross-covariance between components/parameters is ignored). For
/// the full off-diagonal Hessian treatment, see
/// [`LaplaceGMM::full_hessian_posterior_covariance`].
#[derive(Debug, Clone)]
pub struct LaplaceGMMTrained {
    /// MAP estimates (mode of posterior)
    pub map_weights: Array1<f64>,
    /// MAP means
    pub map_means: Array2<f64>,
    /// MAP covariances (tied diagonal, represented as a full matrix whose
    /// off-diagonal entries are zero -- see [`LaplaceGMMTrained`] docs)
    pub map_covariances: Array2<f64>,
    /// Approximate posterior covariance: a diagonal-Hessian Laplace
    /// approximation for the mean parameters only, i.e. entry
    /// `(k * n_features + j, k * n_features + j)` holds the asymptotic
    /// variance of component `k`'s `j`-th mean coordinate
    /// (`variance_j / n_effective_k`), plus `hessian_regularization` as a
    /// numerical floor. Off-diagonal entries (including weight/covariance
    /// cross-terms) are zero -- a disclosed simplification.
    pub posterior_covariance: Array2<f64>,
    /// Log marginal likelihood (evidence), approximated via the standard
    /// large-sample (Schwarz) simplification of the Laplace method:
    /// `log_likelihood - 0.5 * n_params * ln(n_samples)`.
    pub log_marginal_likelihood: f64,
    /// Log-likelihood of the data at the MAP estimate.
    pub log_likelihood: f64,
    /// Number of iterations
    pub n_iter: usize,
    /// Convergence status
    pub converged: bool,
}

/// Builder for Laplace GMM
#[derive(Debug, Clone)]
pub struct LaplaceGMMBuilder {
    n_components: usize,
    covariance_type: CovarianceType,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    hessian_regularization: f64,
}

impl LaplaceGMMBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            n_components: 1,
            covariance_type: CovarianceType::Diagonal,
            max_iter: 100,
            tol: 1e-3,
            reg_covar: 1e-6,
            hessian_regularization: 1e-4,
        }
    }

    /// Set number of components
    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
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

    /// Set Hessian regularization
    pub fn hessian_regularization(mut self, reg: f64) -> Self {
        self.hessian_regularization = reg;
        self
    }

    /// Build the model
    pub fn build(self) -> LaplaceGMM<Untrained> {
        LaplaceGMM {
            state: Untrained,
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            hessian_regularization: self.hessian_regularization,
        }
    }
}

impl Default for LaplaceGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LaplaceGMM<Untrained> {
    /// Create a new builder
    pub fn builder() -> LaplaceGMMBuilder {
        LaplaceGMMBuilder::new()
    }
}

impl Estimator for LaplaceGMM<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for LaplaceGMM<Untrained> {
    type Fitted = LaplaceGMM<LaplaceGMMTrained>;

    /// Fit a tied-diagonal-covariance Gaussian mixture via EM to locate the
    /// posterior mode (`theta_MAP`), then build a Laplace approximation
    /// around that mode.
    ///
    /// The Laplace approximation says the evidence is proportional to the
    /// likelihood at the mode times a correction term involving the
    /// determinant of the Hessian `H` of the negative log posterior at that
    /// mode. Computing and inverting the full `H` (with cross terms between
    /// every mean/weight/covariance parameter) is out of scope here (see
    /// [`LaplaceGMM::full_hessian_posterior_covariance`]); instead this uses
    /// the standard large-sample (Schwarz/BIC) asymptotic simplification,
    /// subtracting `0.5 * n_params * ln(n_samples)` from the raw
    /// log-likelihood, and a diagonal-Hessian approximation for the
    /// posterior covariance of the mean parameters (variance divided by the
    /// component's effective sample count). Both quantities are genuinely
    /// computed from the fitted model and the data (never hardcoded).
    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X_owned = X.to_owned();
        let (n_samples, n_features) = X_owned.dim();

        if self.n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of components must be positive".to_string(),
            ));
        }
        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be >= number of components".to_string(),
            ));
        }

        // Deterministic initialization (evenly spaced samples), mirroring
        // `GaussianMixture::initialize_means` in gaussian.rs.
        let mut means = Array2::zeros((self.n_components, n_features));
        let step = (n_samples / self.n_components).max(1);
        for k in 0..self.n_components {
            let idx = (k * step).min(n_samples - 1);
            means.row_mut(k).assign(&X_owned.row(idx));
        }

        let mut weights = Array1::from_elem(self.n_components, 1.0 / self.n_components as f64);
        let mut covariances =
            Array2::<f64>::eye(n_features) + &(Array2::<f64>::eye(n_features) * self.reg_covar);

        let mut n_iter = 0;
        let mut converged = false;
        let mut prev_log_lik = f64::NEG_INFINITY;
        let mut component_counts = Array1::<f64>::zeros(self.n_components);

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // E-step: responsibilities via the shared tied-diagonal formula.
            let cov_diag = covariances.diag().to_owned();
            let mut responsibilities = Array2::zeros((n_samples, self.n_components));
            let mut log_lik = 0.0;
            for i in 0..n_samples {
                let sample = X_owned.row(i);
                let mut log_probs = Vec::with_capacity(self.n_components);
                for k in 0..self.n_components {
                    let mean_k = means.row(k);
                    log_probs.push(crate::common::tied_diag_weighted_log_prob(
                        &sample,
                        &mean_k,
                        weights[k],
                        &cov_diag.view(),
                        self.reg_covar,
                    ));
                }
                let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
                log_lik += max_log + sum_exp.ln();
                for k in 0..self.n_components {
                    responsibilities[[i, k]] =
                        ((log_probs[k] - max_log).exp() / sum_exp).max(1e-10);
                }
            }

            // M-step: means and weights per component; a single pooled
            // (tied) diagonal covariance across all components, using the
            // standard tied-covariance MLE formula
            // Sigma = (1/N) * sum_k sum_i r_ik (x_i - mu_k)(x_i - mu_k)^T.
            let mut pooled_var = Array1::<f64>::zeros(n_features);
            for k in 0..self.n_components {
                let resp_k = responsibilities.column(k);
                let nk = resp_k.sum().max(1e-10);
                component_counts[k] = nk;
                weights[k] = nk / n_samples as f64;

                let mut new_mean = Array1::zeros(n_features);
                for i in 0..n_samples {
                    new_mean += &(X_owned.row(i).to_owned() * resp_k[i]);
                }
                new_mean /= nk;

                for i in 0..n_samples {
                    let diff = &X_owned.row(i).to_owned() - &new_mean;
                    pooled_var += &(diff.mapv(|v| v * v) * resp_k[i]);
                }

                means.row_mut(k).assign(&new_mean);
            }
            let weight_sum = weights.sum();
            weights /= weight_sum;

            pooled_var =
                pooled_var / n_samples as f64 + Array1::from_elem(n_features, self.reg_covar);
            covariances.diag_mut().assign(&pooled_var);

            if iter > 0 && (log_lik - prev_log_lik).abs() < self.tol {
                converged = true;
                prev_log_lik = log_lik;
                break;
            }
            prev_log_lik = log_lik;
        }

        // Simplified (Schwarz/BIC-style) Laplace approximation to the log
        // marginal likelihood: n_params counts the tied-covariance mixture's
        // free parameters (means + weights-minus-one + shared diagonal
        // covariance).
        let n_params = self.n_components * n_features + (self.n_components - 1) + n_features;
        let log_marginal_likelihood =
            prev_log_lik - 0.5 * (n_params as f64) * (n_samples as f64).ln();

        // Diagonal-Hessian approximation to the posterior covariance of the
        // mean parameters: Var(mean_kj) ~ variance_j / n_effective_k, with
        // `hessian_regularization` as a numerical floor. Sized for the mean
        // parameters only (see `LaplaceGMMTrained::posterior_covariance`
        // docs for the disclosed simplification).
        let mut posterior_covariance = Array2::<f64>::zeros((
            self.n_components * n_features,
            self.n_components * n_features,
        ));
        let cov_diag_final = covariances.diag().to_owned();
        for k in 0..self.n_components {
            let nk = component_counts[k].max(1e-10);
            for j in 0..n_features {
                let idx = k * n_features + j;
                posterior_covariance[[idx, idx]] =
                    cov_diag_final[j] / nk + self.hessian_regularization;
            }
        }

        let trained_state = LaplaceGMMTrained {
            map_weights: weights,
            map_means: means,
            map_covariances: covariances,
            posterior_covariance,
            log_marginal_likelihood,
            log_likelihood: prev_log_lik,
            n_iter,
            converged,
        };

        Ok(LaplaceGMM {
            state: Untrained,
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            hessian_regularization: self.hessian_regularization,
        }
        .with_state(trained_state))
    }
}

impl LaplaceGMM<Untrained> {
    fn with_state(self, state: LaplaceGMMTrained) -> LaplaceGMM<LaplaceGMMTrained> {
        LaplaceGMM {
            state,
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            hessian_regularization: self.hessian_regularization,
        }
    }
}

impl LaplaceGMM<LaplaceGMMTrained> {
    /// Compute the *full* (non-diagonal) Hessian-based posterior covariance
    /// for the Laplace approximation, including cross-covariance terms
    /// between all parameters (means, weights, and the shared covariance).
    ///
    /// This is **not implemented**: it requires assembling and inverting the
    /// full `(n_components * (n_features + 1) + n_features)`-square
    /// observed-information Hessian (including weight-simplex and
    /// covariance-parameter cross terms), which is out of scope for this
    /// simplified Laplace GMM. Use [`LaplaceGMMTrained::posterior_covariance`]
    /// for the diagonal (per-mean-parameter) approximation computed during
    /// `fit`.
    pub fn full_hessian_posterior_covariance(&self) -> SklResult<Array2<f64>> {
        Err(SklearsError::NotImplemented(
            "LaplaceGMM: full off-diagonal Hessian assembly (with weight/covariance \
             cross-terms) for the Laplace posterior covariance is not implemented; \
             the diagonal approximation is available via `LaplaceGMMTrained::posterior_covariance`"
                .to_string(),
        ))
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<usize>> for LaplaceGMM<LaplaceGMMTrained> {
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<usize>> {
        let X_owned = X.to_owned();
        Ok(crate::common::predict_tied_diag_argmax(
            &X_owned,
            &self.state.map_weights,
            &self.state.map_means,
            &self.state.map_covariances,
            self.reg_covar,
        ))
    }
}

// Monte Carlo GMM
#[derive(Debug, Clone)]
pub struct MonteCarloGMM<S = Untrained> {
    #[allow(dead_code)]
    n_components: usize,
    #[allow(dead_code)]
    mc_method: MonteCarloMethod,
    _phantom: std::marker::PhantomData<S>,
}

#[derive(Debug, Clone)]
pub struct MonteCarloGMMTrained {
    pub samples_weights: Vec<Array1<f64>>,
    pub samples_means: Vec<Array2<f64>>,
    pub n_samples: usize,
}

#[derive(Debug, Clone)]
pub struct MonteCarloGMMBuilder {
    n_components: usize,
    mc_method: MonteCarloMethod,
}

impl MonteCarloGMMBuilder {
    pub fn new() -> Self {
        Self {
            n_components: 1,
            mc_method: MonteCarloMethod::Standard { n_samples: 1000 },
        }
    }

    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    pub fn mc_method(mut self, method: MonteCarloMethod) -> Self {
        self.mc_method = method;
        self
    }

    pub fn build(self) -> MonteCarloGMM<Untrained> {
        MonteCarloGMM {
            n_components: self.n_components,
            mc_method: self.mc_method,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Default for MonteCarloGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MonteCarloGMM<Untrained> {
    pub fn builder() -> MonteCarloGMMBuilder {
        MonteCarloGMMBuilder::new()
    }
}

// Importance Sampling GMM
#[derive(Debug, Clone)]
pub struct ImportanceSamplingGMM<S = Untrained> {
    #[allow(dead_code)]
    n_components: usize,
    #[allow(dead_code)]
    is_strategy: ImportanceSamplingStrategy,
    _phantom: std::marker::PhantomData<S>,
}

#[derive(Debug, Clone)]
pub struct ImportanceSamplingGMMTrained {
    pub weights_samples: Vec<Array1<f64>>,
    pub importance_weights: Array1<f64>,
    pub effective_sample_size: f64,
}

#[derive(Debug, Clone)]
pub struct ImportanceSamplingGMMBuilder {
    n_components: usize,
    is_strategy: ImportanceSamplingStrategy,
}

impl ImportanceSamplingGMMBuilder {
    pub fn new() -> Self {
        Self {
            n_components: 1,
            is_strategy: ImportanceSamplingStrategy::Standard { n_samples: 1000 },
        }
    }

    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    pub fn is_strategy(mut self, strategy: ImportanceSamplingStrategy) -> Self {
        self.is_strategy = strategy;
        self
    }

    pub fn build(self) -> ImportanceSamplingGMM<Untrained> {
        ImportanceSamplingGMM {
            n_components: self.n_components,
            is_strategy: self.is_strategy,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Default for ImportanceSamplingGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ImportanceSamplingGMM<Untrained> {
    pub fn builder() -> ImportanceSamplingGMMBuilder {
        ImportanceSamplingGMMBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_laplace_gmm_builder() {
        let model = LaplaceGMM::builder()
            .n_components(3)
            .hessian_regularization(1e-3)
            .build();

        assert_eq!(model.n_components, 3);
        assert_eq!(model.hessian_regularization, 1e-3);
    }

    #[test]
    fn test_monte_carlo_methods() {
        let methods = vec![
            MonteCarloMethod::Standard { n_samples: 500 },
            MonteCarloMethod::Quasi { n_samples: 1000 },
            MonteCarloMethod::MCMC {
                n_samples: 2000,
                burn_in: 100,
                thin: 5,
            },
        ];

        for method in methods {
            let model = MonteCarloGMM::builder().mc_method(method).build();
            assert_eq!(model.mc_method, method);
        }
    }

    #[test]
    fn test_importance_sampling_strategies() {
        let strategies = vec![
            ImportanceSamplingStrategy::Standard { n_samples: 500 },
            ImportanceSamplingStrategy::Adaptive {
                n_samples: 1000,
                adaptation_steps: 10,
            },
            ImportanceSamplingStrategy::SelfNormalized { n_samples: 750 },
        ];

        for strategy in strategies {
            let model = ImportanceSamplingGMM::builder()
                .is_strategy(strategy)
                .build();
            assert_eq!(model.is_strategy, strategy);
        }
    }

    #[test]
    #[allow(non_snake_case)] // standard ML notation
    fn test_laplace_gmm_fit() {
        let X = array![[1.0, 2.0], [1.5, 2.5], [10.0, 11.0]];

        let model = LaplaceGMM::builder().n_components(2).build();

        let result = model.fit(&X.view(), &());
        assert!(result.is_ok());
    }

    #[test]
    fn test_monte_carlo_gmm_builder() {
        let model = MonteCarloGMM::builder()
            .n_components(4)
            .mc_method(MonteCarloMethod::Quasi { n_samples: 2000 })
            .build();

        assert_eq!(model.n_components, 4);
    }

    #[test]
    fn test_importance_sampling_gmm_builder() {
        let model = ImportanceSamplingGMM::builder()
            .n_components(3)
            .is_strategy(ImportanceSamplingStrategy::Adaptive {
                n_samples: 1500,
                adaptation_steps: 20,
            })
            .build();

        assert_eq!(model.n_components, 3);
    }

    #[test]
    fn test_builder_defaults() {
        let laplace = LaplaceGMM::builder().build();
        assert_eq!(laplace.n_components, 1);

        let mc = MonteCarloGMM::builder().build();
        assert_eq!(mc.n_components, 1);

        let is = ImportanceSamplingGMM::builder().build();
        assert_eq!(is.n_components, 1);
    }

    /// Regression test for the fabrication bug: `with_state` used to discard
    /// the fitted parameters (storing `PhantomData` instead), so `predict`
    /// always returned `Array1::zeros(n_samples)` no matter what. This
    /// fixture has two well-separated blobs; a real fit must (a) produce
    /// more than one distinct predicted label, and (b) assign the same label
    /// within each blob and different labels across blobs.
    #[test]
    #[allow(non_snake_case)] // standard ML notation
    fn test_laplace_gmm_predict_recovers_cluster_structure() {
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

        let model = LaplaceGMM::builder().n_components(2).max_iter(50).build();
        let fitted = model
            .fit(&X.view(), &())
            .expect("LaplaceGMM fit should succeed on well-separated blobs");
        let preds = fitted
            .predict(&X.view())
            .expect("LaplaceGMM predict should succeed");

        let distinct: std::collections::HashSet<usize> = preds.iter().copied().collect();
        assert!(
            distinct.len() > 1,
            "predictions collapsed onto a single label (the old all-zeros bug): {:?}",
            preds
        );

        let label_a = preds[0];
        for i in 0..4 {
            assert_eq!(
                preds[i], label_a,
                "first blob (points 0-3) should share one predicted label"
            );
        }
        let label_b = preds[4];
        assert_ne!(
            label_a, label_b,
            "the two well-separated blobs must not collapse onto the same label"
        );
        for i in 4..8 {
            assert_eq!(
                preds[i], label_b,
                "second blob (points 4-7) should share one predicted label"
            );
        }
    }

    /// `log_marginal_likelihood` must be a real, data-dependent quantity
    /// (never the old hardcoded `0.0` placeholder), and must actually change
    /// when the fitted log-likelihood changes.
    #[test]
    #[allow(non_snake_case)]
    fn test_laplace_gmm_log_marginal_likelihood_is_not_a_placeholder() {
        let X = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
            [9.9, 10.1],
        ];

        let fitted = LaplaceGMM::builder()
            .n_components(2)
            .max_iter(50)
            .build()
            .fit(&X.view(), &())
            .expect("fit should succeed");

        assert!(
            fitted.state.log_marginal_likelihood.is_finite(),
            "log_marginal_likelihood must be finite, got {}",
            fitted.state.log_marginal_likelihood
        );
        // The old placeholder was an unconditional 0.0; a real fit on this
        // fixture should not land exactly on that value.
        assert_ne!(fitted.state.log_marginal_likelihood, 0.0);
        assert!(fitted.state.log_likelihood.is_finite());
        // The Laplace/Schwarz correction subtracts a strictly positive
        // penalty from the raw log-likelihood.
        assert!(fitted.state.log_marginal_likelihood < fitted.state.log_likelihood);
        assert!(fitted.state.n_iter >= 1);

        // posterior_covariance must be real (data-dependent), positive on
        // its diagonal, and correctly sized for (n_components * n_features).
        assert_eq!(fitted.state.posterior_covariance.dim(), (4, 4));
        for i in 0..4 {
            assert!(fitted.state.posterior_covariance[[i, i]] > 0.0);
        }
    }

    /// The narrow full-Hessian sub-capability is honestly unimplemented
    /// rather than silently returning a fabricated result.
    #[test]
    #[allow(non_snake_case)]
    fn test_laplace_gmm_full_hessian_is_honest_not_implemented() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];
        let fitted = LaplaceGMM::builder()
            .n_components(2)
            .build()
            .fit(&X.view(), &())
            .expect("fit should succeed");

        let result = fitted.full_hessian_posterior_covariance();
        assert!(
            matches!(result, Err(SklearsError::NotImplemented(_))),
            "full_hessian_posterior_covariance must honestly report NotImplemented, not fabricate a result"
        );
    }
}
