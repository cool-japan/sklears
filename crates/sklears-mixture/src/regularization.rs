//! Regularization Techniques for Mixture Models
//!
//! This module provides various regularization techniques for mixture models,
//! including L1 regularization for sparsity, L2 regularization for stability,
//! elastic net for combined sparsity and stability, and group lasso for
//! structured sparsity.
//!
//! # Overview
//!
//! Regularization is crucial for:
//! - Preventing overfitting in high-dimensional settings
//! - Promoting sparsity in parameter estimates
//! - Improving numerical stability
//! - Incorporating structural constraints
//! - Feature selection in mixture models
//!
//! # Key Components
//!
//! - **L1 Regularization**: Promotes sparsity through LASSO penalty
//! - **L2 Regularization**: Promotes stability through ridge penalty
//! - **Elastic Net**: Combines L1 and L2 penalties
//! - **Group Lasso**: Structured sparsity for grouped features

use crate::common::CovarianceType;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::seeded_rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::f64::consts::PI;

/// Type of regularization to apply
#[derive(Debug, Clone, PartialEq)]
pub enum RegularizationType {
    /// L1 regularization (LASSO)
    L1 { lambda: f64 },
    /// L2 regularization (Ridge)
    L2 { lambda: f64 },
    /// Elastic Net (combination of L1 and L2)
    ElasticNet { l1_ratio: f64, lambda: f64 },
    /// Group LASSO for structured sparsity
    GroupLasso {
        lambda: f64,
        groups: Vec<Vec<usize>>,
    },
}

/// L1 Regularized Gaussian Mixture Model
///
/// Implements sparse Gaussian mixture modeling using L1 (LASSO) regularization.
/// This promotes sparsity in the parameter estimates, which is useful for
/// feature selection and high-dimensional data.
///
/// # Examples
///
/// ```
/// use sklears_mixture::regularization::L1RegularizedGMM;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// let model = L1RegularizedGMM::builder()
///     .n_components(2)
///     .lambda(0.01)
///     .build();
///
/// let fitted = model.fit(&X.view(), &()).expect("L1-regularized GMM fitting should succeed with valid data");
/// ```
#[derive(Debug, Clone)]
pub struct L1RegularizedGMM<S = Untrained> {
    pub(crate) state: S,
    n_components: usize,
    lambda: f64,
    covariance_type: CovarianceType,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    random_state: Option<u64>,
}

/// Trained L1 Regularized GMM
#[derive(Debug, Clone)]
pub struct L1RegularizedGMMTrained {
    /// Component weights
    pub weights: Array1<f64>,
    /// Component means
    pub means: Array2<f64>,
    /// Component covariances (tied/shared diagonal, see module docs)
    pub covariances: Array2<f64>,
    /// Sparsity pattern (true = non-zero coefficient)
    pub sparsity_pattern: Vec<Vec<bool>>,
    /// Number of non-zero parameters
    pub n_nonzero: usize,
    /// Log-likelihood history: the real penalized log-likelihood
    /// (log-sum-exp data log-likelihood minus the L1 penalty on the means)
    /// at each iteration.
    pub log_likelihood_history: Vec<f64>,
    /// Number of iterations
    pub n_iter: usize,
    /// Convergence status
    pub converged: bool,
}

/// Builder for L1 Regularized GMM
#[derive(Debug, Clone)]
pub struct L1RegularizedGMMBuilder {
    n_components: usize,
    lambda: f64,
    covariance_type: CovarianceType,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    random_state: Option<u64>,
}

impl L1RegularizedGMMBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            n_components: 1,
            lambda: 0.01,
            covariance_type: CovarianceType::Diagonal,
            max_iter: 100,
            tol: 1e-3,
            reg_covar: 1e-6,
            random_state: None,
        }
    }

    /// Set number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set L1 regularization parameter
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set covariance type
    pub fn covariance_type(mut self, covariance_type: CovarianceType) -> Self {
        self.covariance_type = covariance_type;
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
    pub fn reg_covar(mut self, reg_covar: f64) -> Self {
        self.reg_covar = reg_covar;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Build the model
    pub fn build(self) -> L1RegularizedGMM<Untrained> {
        L1RegularizedGMM {
            state: Untrained,
            n_components: self.n_components,
            lambda: self.lambda,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            random_state: self.random_state,
        }
    }
}

impl Default for L1RegularizedGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl L1RegularizedGMM<Untrained> {
    /// Create a new builder
    pub fn builder() -> L1RegularizedGMMBuilder {
        L1RegularizedGMMBuilder::new()
    }

    /// Soft thresholding operator for L1 regularization
    fn soft_threshold(x: f64, lambda: f64) -> f64 {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }
}

impl Estimator for L1RegularizedGMM<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for L1RegularizedGMM<Untrained> {
    type Fitted = L1RegularizedGMM<L1RegularizedGMMTrained>;

    /// Fit a tied-diagonal-covariance Gaussian mixture via EM with an L1
    /// (LASSO) penalty applied to the means through soft-thresholding.
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

        // Initialize with simple k-means-like approach. `random_state` is
        // honored via `common::resolve_seed` + `seeded_rng` (previously the
        // field was accepted but silently ignored in favor of
        // non-reproducible `thread_rng()`).
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

        let mut weights = Array1::from_elem(self.n_components, 1.0 / self.n_components as f64);
        let mut covariances =
            Array2::<f64>::eye(n_features) + &(Array2::<f64>::eye(n_features) * self.reg_covar);

        let mut log_likelihood_history = Vec::new();
        let mut converged = false;

        // EM algorithm with L1 regularization
        for iter in 0..self.max_iter {
            // E-step
            let mut responsibilities = Array2::zeros((n_samples, self.n_components));
            let mut log_lik = 0.0;

            for i in 0..n_samples {
                let x = X_owned.row(i);
                let mut log_probs = Vec::new();

                for k in 0..self.n_components {
                    let mean = means.row(k);
                    let diff = &x.to_owned() - &mean.to_owned();

                    let mahal = diff
                        .iter()
                        .zip(covariances.diag().iter())
                        .map(|(d, c): (&f64, &f64)| d * d / c.max(self.reg_covar))
                        .sum::<f64>();

                    let log_det = covariances
                        .diag()
                        .iter()
                        .map(|c| c.max(self.reg_covar).ln())
                        .sum::<f64>();

                    let log_prob = weights[k].ln()
                        - 0.5 * (n_features as f64 * (2.0 * PI).ln() + log_det)
                        - 0.5 * mahal;

                    log_probs.push(log_prob);
                }

                let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
                // Real log-sum-exp data log-likelihood contribution
                // (previously this was thrown away and replaced below by
                // summing normalized responsibilities, which always sum to
                // ~1.0 and therefore carried no information about fit
                // quality).
                log_lik += max_log + sum_exp.ln();

                for k in 0..self.n_components {
                    responsibilities[[i, k]] =
                        ((log_probs[k] - max_log).exp() / sum_exp).max(1e-10);
                }
            }

            // M-step with L1 regularization
            for k in 0..self.n_components {
                let resps = responsibilities.column(k);
                let nk = resps.sum().max(1e-10);

                weights[k] = nk / n_samples as f64;

                // Update mean with soft thresholding
                let mut new_mean = Array1::zeros(n_features);
                for i in 0..n_samples {
                    new_mean += &(X_owned.row(i).to_owned() * resps[i]);
                }
                new_mean /= nk;

                // Apply L1 penalty via soft thresholding
                for j in 0..n_features {
                    new_mean[j] = Self::soft_threshold(new_mean[j], self.lambda);
                }
                means.row_mut(k).assign(&new_mean);

                // Update covariance
                let mut new_cov = Array1::zeros(n_features);
                for i in 0..n_samples {
                    let diff = &X_owned.row(i).to_owned() - &new_mean;
                    new_cov += &(diff.mapv(|x| x * x) * resps[i]);
                }
                new_cov = new_cov / nk + Array1::from_elem(n_features, self.reg_covar);
                covariances.diag_mut().assign(&new_cov);
            }

            weights /= weights.sum();

            // Penalized objective: real data log-likelihood minus the L1
            // penalty on the means (the standard penalized-likelihood
            // formulation for LASSO-regularized MLE).
            let l1_penalty: f64 = means.iter().map(|&m| m.abs()).sum::<f64>() * self.lambda;
            let log_lik = log_lik - l1_penalty;

            log_likelihood_history.push(log_lik);

            if iter > 0 {
                let improvement = (log_lik - log_likelihood_history[iter - 1]).abs();
                if improvement < self.tol {
                    converged = true;
                    break;
                }
            }
        }

        // Compute sparsity pattern
        let mut sparsity_pattern = Vec::new();
        let mut n_nonzero = 0;
        for k in 0..self.n_components {
            let mut pattern = Vec::new();
            for j in 0..n_features {
                let is_nonzero = means[[k, j]].abs() > 1e-10;
                pattern.push(is_nonzero);
                if is_nonzero {
                    n_nonzero += 1;
                }
            }
            sparsity_pattern.push(pattern);
        }

        let n_iter = log_likelihood_history.len();
        let trained_state = L1RegularizedGMMTrained {
            weights,
            means,
            covariances,
            sparsity_pattern,
            n_nonzero,
            log_likelihood_history,
            n_iter,
            converged,
        };

        Ok(L1RegularizedGMM {
            state: Untrained,
            n_components: self.n_components,
            lambda: self.lambda,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            random_state: self.random_state,
        }
        .with_state(trained_state))
    }
}

impl L1RegularizedGMM<Untrained> {
    fn with_state(
        self,
        state: L1RegularizedGMMTrained,
    ) -> L1RegularizedGMM<L1RegularizedGMMTrained> {
        L1RegularizedGMM {
            state,
            n_components: self.n_components,
            lambda: self.lambda,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            random_state: self.random_state,
        }
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<usize>> for L1RegularizedGMM<L1RegularizedGMMTrained> {
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

// L2 Regularized GMM (similar structure)
#[derive(Debug, Clone)]
pub struct L2RegularizedGMM<S = Untrained> {
    #[allow(dead_code)]
    n_components: usize,
    #[allow(dead_code)]
    lambda: f64,
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
pub struct L2RegularizedGMMTrained {
    pub weights: Array1<f64>,
    pub means: Array2<f64>,
    pub covariances: Array2<f64>,
    pub log_likelihood_history: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
}

#[derive(Debug, Clone)]
pub struct L2RegularizedGMMBuilder {
    n_components: usize,
    lambda: f64,
    covariance_type: CovarianceType,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    random_state: Option<u64>,
}

impl L2RegularizedGMMBuilder {
    pub fn new() -> Self {
        Self {
            n_components: 1,
            lambda: 0.01,
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

    pub fn lambda(mut self, l: f64) -> Self {
        self.lambda = l;
        self
    }

    pub fn build(self) -> L2RegularizedGMM<Untrained> {
        L2RegularizedGMM {
            n_components: self.n_components,
            lambda: self.lambda,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            random_state: self.random_state,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Default for L2RegularizedGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl L2RegularizedGMM<Untrained> {
    pub fn builder() -> L2RegularizedGMMBuilder {
        L2RegularizedGMMBuilder::new()
    }
}

// Elastic Net GMM
#[derive(Debug, Clone)]
pub struct ElasticNetGMM<S = Untrained> {
    #[allow(dead_code)]
    n_components: usize,
    #[allow(dead_code)]
    l1_ratio: f64,
    #[allow(dead_code)]
    lambda: f64,
    _phantom: std::marker::PhantomData<S>,
}

#[derive(Debug, Clone)]
pub struct ElasticNetGMMTrained {
    pub weights: Array1<f64>,
    pub means: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct ElasticNetGMMBuilder {
    n_components: usize,
    l1_ratio: f64,
    lambda: f64,
}

impl ElasticNetGMMBuilder {
    pub fn new() -> Self {
        Self {
            n_components: 1,
            l1_ratio: 0.5,
            lambda: 0.01,
        }
    }

    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    pub fn l1_ratio(mut self, r: f64) -> Self {
        self.l1_ratio = r;
        self
    }

    pub fn lambda(mut self, l: f64) -> Self {
        self.lambda = l;
        self
    }

    pub fn build(self) -> ElasticNetGMM<Untrained> {
        ElasticNetGMM {
            n_components: self.n_components,
            l1_ratio: self.l1_ratio,
            lambda: self.lambda,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Default for ElasticNetGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ElasticNetGMM<Untrained> {
    pub fn builder() -> ElasticNetGMMBuilder {
        ElasticNetGMMBuilder::new()
    }
}

// Group Lasso GMM
#[derive(Debug, Clone)]
pub struct GroupLassoGMM<S = Untrained> {
    #[allow(dead_code)]
    n_components: usize,
    #[allow(dead_code)]
    lambda: f64,
    #[allow(dead_code)]
    groups: Vec<Vec<usize>>,
    _phantom: std::marker::PhantomData<S>,
}

#[derive(Debug, Clone)]
pub struct GroupLassoGMMTrained {
    pub weights: Array1<f64>,
    pub means: Array2<f64>,
    pub active_groups: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct GroupLassoGMMBuilder {
    n_components: usize,
    lambda: f64,
    groups: Vec<Vec<usize>>,
}

impl GroupLassoGMMBuilder {
    pub fn new() -> Self {
        Self {
            n_components: 1,
            lambda: 0.01,
            groups: Vec::new(),
        }
    }

    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    pub fn lambda(mut self, l: f64) -> Self {
        self.lambda = l;
        self
    }

    pub fn add_group(mut self, group: Vec<usize>) -> Self {
        self.groups.push(group);
        self
    }

    pub fn build(self) -> GroupLassoGMM<Untrained> {
        GroupLassoGMM {
            n_components: self.n_components,
            lambda: self.lambda,
            groups: self.groups,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Default for GroupLassoGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GroupLassoGMM<Untrained> {
    pub fn builder() -> GroupLassoGMMBuilder {
        GroupLassoGMMBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_soft_threshold() {
        assert_eq!(L1RegularizedGMM::soft_threshold(2.0, 0.5), 1.5);
        assert_eq!(L1RegularizedGMM::soft_threshold(-2.0, 0.5), -1.5);
        assert_eq!(L1RegularizedGMM::soft_threshold(0.3, 0.5), 0.0);
    }

    #[test]
    fn test_l1_regularized_gmm_builder() {
        let model = L1RegularizedGMM::builder()
            .n_components(3)
            .lambda(0.05)
            .max_iter(50)
            .build();

        assert_eq!(model.n_components, 3);
        assert_eq!(model.lambda, 0.05);
        assert_eq!(model.max_iter, 50);
    }

    #[test]
    #[allow(non_snake_case)] // standard ML notation
    fn test_l1_regularized_gmm_fit() {
        let X = array![[1.0, 2.0], [1.5, 2.5], [10.0, 11.0], [10.5, 11.5]];

        let model = L1RegularizedGMM::builder()
            .n_components(2)
            .lambda(0.01)
            .max_iter(20)
            .build();

        let result = model.fit(&X.view(), &());
        assert!(result.is_ok());
    }

    /// Regression test for the fabrication bug: `with_state` used to
    /// discard the fitted parameters, so `predict` always returned
    /// all-zeros regardless of input. A real fit on two well-separated
    /// blobs must discriminate between them.
    #[test]
    #[allow(non_snake_case)] // standard ML notation
    fn test_l1_regularized_gmm_predict_recovers_cluster_structure() {
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

        let model = L1RegularizedGMM::builder()
            .n_components(2)
            .lambda(0.001)
            .max_iter(50)
            .random_state(42)
            .build();
        let fitted = model
            .fit(&X.view(), &())
            .expect("L1RegularizedGMM fit should succeed on well-separated blobs");
        let preds = fitted
            .predict(&X.view())
            .expect("L1RegularizedGMM predict should succeed");

        let distinct: std::collections::HashSet<usize> = preds.iter().copied().collect();
        assert!(
            distinct.len() > 1,
            "predictions collapsed onto a single label (the old all-zeros bug): {:?}",
            preds
        );

        let label_a = preds[0];
        for i in 0..4 {
            assert_eq!(preds[i], label_a, "first blob should share one label");
        }
        let label_b = preds[4];
        assert_ne!(
            label_a, label_b,
            "the two well-separated blobs must not collapse onto the same label"
        );
        for i in 4..8 {
            assert_eq!(preds[i], label_b, "second blob should share one label");
        }
    }

    /// The old `log_likelihood_history` summed *normalized* responsibilities
    /// (which sum to ~1.0 per sample by construction), making it an
    /// uninformative near-constant regardless of fit quality. The real
    /// log-sum-exp log-likelihood must differ substantially between a good
    /// fit (means placed at the true cluster centers) and a deliberately bad
    /// one (means forced far away from all data).
    #[test]
    #[allow(non_snake_case)]
    fn test_l1_log_likelihood_is_not_the_old_constant_bug() {
        let X = array![[0.0, 0.0], [0.2, -0.1], [10.0, 10.0], [10.1, 9.9],];

        let fitted = L1RegularizedGMM::builder()
            .n_components(2)
            .lambda(0.0)
            .max_iter(50)
            .random_state(1)
            .build()
            .fit(&X.view(), &())
            .expect("fit should succeed");

        let final_ll = *fitted
            .state
            .log_likelihood_history
            .last()
            .expect("history should be non-empty");
        assert!(final_ll.is_finite());
        // The old bug summed *normalized* responsibilities, each ~1.0 by
        // softmax construction, so `log_lik` always landed within floating
        // rounding error of `n_samples * ln(1.0) == 0.0` (with lambda == 0.0
        // here, no penalty to shift it away from that). A real per-sample
        // Gaussian log-density on tight, well-separated 2D blobs is many
        // orders of magnitude away from 0 (it can be positive when the
        // fitted variance is small, since a continuous density can exceed
        // 1), so this threshold safely distinguishes "real" from "the old
        // near-zero placeholder" without having to predict the exact sign.
        assert!(
            final_ll.abs() > 1e-3,
            "log-likelihood looks like the old near-zero placeholder bug: {final_ll}"
        );

        // Cross-check against an independent, deliberately bad set of
        // parameters (single shared mean at the global centroid, huge
        // variance): the fitted model must score the data at least as well
        // under its own (real) log-likelihood definition.
        let bad_mean = array![5.05_f64, 4.95];
        let huge_cov = Array2::<f64>::eye(2) * 1.0e6;
        let huge_cov_diag = huge_cov.diag().to_owned();
        let mut bad_ll = 0.0;
        for row in X.outer_iter() {
            bad_ll += crate::common::tied_diag_weighted_log_prob(
                &row,
                &bad_mean.view(),
                1.0,
                &huge_cov_diag.view(),
                1e-6,
            );
        }
        assert!(
            final_ll > bad_ll,
            "a real fit ({final_ll}) should score the data better than an obviously bad \
             single-blob model ({bad_ll})"
        );
    }

    #[test]
    fn test_l2_regularized_gmm_builder() {
        let model = L2RegularizedGMM::builder()
            .n_components(2)
            .lambda(0.1)
            .build();

        assert_eq!(model.n_components, 2);
        assert_eq!(model.lambda, 0.1);
    }

    #[test]
    fn test_elastic_net_gmm_builder() {
        let model = ElasticNetGMM::builder()
            .n_components(3)
            .l1_ratio(0.7)
            .lambda(0.05)
            .build();

        assert_eq!(model.n_components, 3);
        assert_eq!(model.l1_ratio, 0.7);
        assert_eq!(model.lambda, 0.05);
    }

    #[test]
    fn test_group_lasso_gmm_builder() {
        let model = GroupLassoGMM::builder()
            .n_components(2)
            .lambda(0.02)
            .add_group(vec![0, 1, 2])
            .add_group(vec![3, 4])
            .build();

        assert_eq!(model.n_components, 2);
        assert_eq!(model.lambda, 0.02);
        assert_eq!(model.groups.len(), 2);
    }

    #[test]
    fn test_regularization_type() {
        let l1 = RegularizationType::L1 { lambda: 0.1 };
        let l2 = RegularizationType::L2 { lambda: 0.2 };
        let enet = RegularizationType::ElasticNet {
            l1_ratio: 0.5,
            lambda: 0.15,
        };

        assert!(matches!(l1, RegularizationType::L1 { .. }));
        assert!(matches!(l2, RegularizationType::L2 { .. }));
        assert!(matches!(enet, RegularizationType::ElasticNet { .. }));
    }
}
