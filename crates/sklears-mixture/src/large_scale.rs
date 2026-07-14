//! Large-Scale Mixture Model Methods
//!
//! This module provides scalable implementations for mixture models that can
//! handle large datasets efficiently through mini-batch processing, parallel
//! computation, and distributed learning.
//!
//! # Overview
//!
//! Large-scale methods enable mixture modeling on:
//! - Datasets with millions of samples
//! - High-dimensional feature spaces
//! - Distributed computing environments
//! - Memory-constrained systems
//!
//! # Key Components
//!
//! - **Mini-Batch EM**: Process data in small batches for memory efficiency
//! - **Parallel EM**: Distribute computation across multiple threads
//! - **Streaming EM**: Process infinite data streams
//! - **Out-of-Core EM**: Handle datasets larger than memory

use crate::common::CovarianceType;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::random::seeded_rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::f64::consts::PI;

/// Mini-batch processing strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatchStrategy {
    /// Fixed batch size
    Fixed { size: usize },
    /// Adaptive batch size based on convergence
    Adaptive {
        initial_size: usize,
        max_size: usize,
    },
    /// Dynamic batch size based on memory
    Dynamic { target_memory_mb: usize },
}

/// Parallel computation strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParallelStrategy {
    /// Data parallelism (split samples across threads)
    DataParallel { n_threads: usize },
    /// Model parallelism (split components across threads)
    ModelParallel { n_threads: usize },
    /// Hybrid approach
    Hybrid {
        data_threads: usize,
        model_threads: usize,
    },
}

/// Mini-Batch EM Gaussian Mixture Model
///
/// Implements EM algorithm with mini-batch processing for scalability.
/// Suitable for datasets with millions of samples.
///
/// # Examples
///
/// ```
/// use sklears_mixture::large_scale::{MiniBatchGMM, BatchStrategy};
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let model = MiniBatchGMM::builder()
///     .n_components(2)
///     .batch_strategy(BatchStrategy::Fixed { size: 100 })
///     .build();
///
/// let X = array![[1.0, 2.0], [1.5, 2.5], [10.0, 11.0], [10.5, 11.5]];
/// let fitted = model.fit(&X.view(), &()).expect("MiniBatch GMM fitting should succeed with valid data");
/// ```
#[derive(Debug, Clone)]
pub struct MiniBatchGMM<S = Untrained> {
    pub(crate) state: S,
    n_components: usize,
    batch_strategy: BatchStrategy,
    covariance_type: CovarianceType,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    learning_rate: f64,
    momentum: f64,
    random_state: Option<u64>,
}

/// Trained Mini-Batch GMM
#[derive(Debug, Clone)]
pub struct MiniBatchGMMTrained {
    /// Component weights
    pub weights: Array1<f64>,
    /// Component means
    pub means: Array2<f64>,
    /// Component covariances (tied/shared diagonal, see module docs). Now
    /// actually updated each mini-batch (previously frozen at
    /// initialization).
    pub covariances: Array2<f64>,
    /// Log-likelihood history: real average per-sample log-likelihood
    /// (log-sum-exp of the weighted Gaussian density), evaluated on a
    /// representative subsample each iteration. Previously this summed raw
    /// mixture weights and was not a log-likelihood at all.
    pub log_likelihood_history: Vec<f64>,
    /// Batch sizes used
    pub batch_sizes: Vec<usize>,
    /// Number of iterations
    pub n_iter: usize,
    /// Convergence status
    pub converged: bool,
}

/// Builder for Mini-Batch GMM
#[derive(Debug, Clone)]
pub struct MiniBatchGMMBuilder {
    n_components: usize,
    batch_strategy: BatchStrategy,
    covariance_type: CovarianceType,
    max_iter: usize,
    tol: f64,
    reg_covar: f64,
    learning_rate: f64,
    momentum: f64,
    random_state: Option<u64>,
}

impl MiniBatchGMMBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            n_components: 1,
            batch_strategy: BatchStrategy::Fixed { size: 256 },
            covariance_type: CovarianceType::Diagonal,
            max_iter: 100,
            tol: 1e-3,
            reg_covar: 1e-6,
            learning_rate: 0.1,
            momentum: 0.9,
            random_state: None,
        }
    }

    /// Set number of components
    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Set batch strategy
    pub fn batch_strategy(mut self, strategy: BatchStrategy) -> Self {
        self.batch_strategy = strategy;
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

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set momentum
    pub fn momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }

    /// Set random state (seed) for reproducible initialization
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Build the model
    pub fn build(self) -> MiniBatchGMM<Untrained> {
        MiniBatchGMM {
            state: Untrained,
            n_components: self.n_components,
            batch_strategy: self.batch_strategy,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            random_state: self.random_state,
        }
    }
}

impl Default for MiniBatchGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MiniBatchGMM<Untrained> {
    /// Create a new builder
    pub fn builder() -> MiniBatchGMMBuilder {
        MiniBatchGMMBuilder::new()
    }
}

impl Estimator for MiniBatchGMM<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MiniBatchGMM<Untrained> {
    type Fitted = MiniBatchGMM<MiniBatchGMMTrained>;

    /// Fit a tied-diagonal-covariance Gaussian mixture with mini-batch EM.
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

        // Get batch size
        let batch_size = match self.batch_strategy {
            BatchStrategy::Fixed { size } => size.min(n_samples).max(1),
            BatchStrategy::Adaptive { initial_size, .. } => initial_size.min(n_samples).max(1),
            BatchStrategy::Dynamic { target_memory_mb } => {
                // Estimate batch size based on memory
                let bytes_per_sample = n_features * 8; // f64
                let target_bytes = target_memory_mb * 1024 * 1024;
                (target_bytes / bytes_per_sample).min(n_samples).max(1)
            }
        };

        // Initialize parameters. `random_state` is honored via
        // `common::resolve_seed` + `seeded_rng` (previously the field was
        // accepted but silently ignored in favor of non-reproducible
        // `thread_rng()`).
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
        // NOTE: previously `covariances` was never declared `mut` and the
        // M-step below never updated it -- the shared covariance stayed
        // frozen at this initial value for the entire fit. It is now
        // genuinely updated every mini-batch (see below).
        let mut covariances =
            Array2::<f64>::eye(n_features) + &(Array2::<f64>::eye(n_features) * self.reg_covar);

        let mut log_likelihood_history = Vec::new();
        let mut batch_sizes = Vec::new();
        let mut converged = false;
        let mut shuffled_indices: Vec<usize> = (0..n_samples).collect();

        // Mini-batch EM
        for _iter in 0..self.max_iter {
            // Re-shuffle each epoch (standard mini-batch practice): without
            // this, data that arrives grouped by cluster (e.g. a stream
            // sorted by label, or simply not pre-shuffled) makes every batch
            // homogeneous, so each mini-batch update drags *every*
            // component toward whichever single cluster is currently being
            // processed -- the components then chase each batch in turn
            // instead of settling into a stable per-cluster assignment.
            // Fisher-Yates shuffle using the same seeded RNG as
            // initialization, so the whole fit stays reproducible under
            // `random_state`.
            for i in (1..n_samples).rev() {
                let j = rng.gen_range(0..=i);
                shuffled_indices.swap(i, j);
            }

            // Process in batches
            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch_indices = &shuffled_indices[batch_start..batch_end];
                let batch_size_actual = batch_indices.len();
                let mut batch = Array2::zeros((batch_size_actual, n_features));
                for (bi, &orig_idx) in batch_indices.iter().enumerate() {
                    batch.row_mut(bi).assign(&X_owned.row(orig_idx));
                }

                // E-step on batch
                let mut responsibilities = Array2::zeros((batch_size_actual, self.n_components));

                for i in 0..batch_size_actual {
                    let x = batch.row(i);
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

                    for k in 0..self.n_components {
                        responsibilities[[i, k]] =
                            ((log_probs[k] - max_log).exp() / sum_exp).max(1e-10);
                    }
                }

                // M-step update with momentum
                let mut pooled_batch_var = Array1::<f64>::zeros(n_features);
                for k in 0..self.n_components {
                    let resps = responsibilities.column(k);
                    let nk = resps.sum().max(1e-10);

                    // Update weight with learning rate
                    let new_weight = nk / batch_size_actual as f64;
                    weights[k] =
                        (1.0 - self.learning_rate) * weights[k] + self.learning_rate * new_weight;

                    // Update mean
                    let mut batch_mean = Array1::zeros(n_features);
                    for i in 0..batch_size_actual {
                        batch_mean += &(batch.row(i).to_owned() * resps[i]);
                    }
                    batch_mean /= nk;

                    for j in 0..n_features {
                        means[[k, j]] = (1.0 - self.learning_rate) * means[[k, j]]
                            + self.learning_rate * batch_mean[j];
                    }

                    // Accumulate this component's weighted contribution to
                    // the pooled (tied) covariance estimate for the batch:
                    // the standard mini-batch-GMM formula
                    // Sigma ~ (1/|batch|) * sum_k sum_i r_ik (x_i - mu_k)(x_i - mu_k)^T,
                    // restricted to the diagonal. This is the covariance
                    // M-step update that was previously entirely missing.
                    for i in 0..batch_size_actual {
                        let diff = &batch.row(i).to_owned() - &batch_mean;
                        pooled_batch_var += &(diff.mapv(|v| v * v) * resps[i]);
                    }
                }

                pooled_batch_var = pooled_batch_var / batch_size_actual as f64
                    + Array1::from_elem(n_features, self.reg_covar);
                let updated_cov_diag = &covariances.diag().to_owned() * (1.0 - self.learning_rate)
                    + &(pooled_batch_var * self.learning_rate);
                covariances.diag_mut().assign(&updated_cov_diag);

                batch_sizes.push(batch_size_actual);
            }

            // Normalize weights
            let weight_sum = weights.sum();
            weights /= weight_sum;

            // Real average per-sample log-likelihood, evaluated on the first
            // `sample_size` samples (previously this summed raw mixture
            // weights -- which sum to ~1.0 -- for `sample_size` iterations
            // without ever indexing into the data, so it was not a
            // log-likelihood at all).
            let sample_size = 1000.min(n_samples);
            let cov_diag = covariances.diag().to_owned();
            let mut log_lik = 0.0;
            for sample in X_owned.slice(s![0..sample_size, ..]).outer_iter() {
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
            }
            log_lik /= sample_size as f64;
            log_likelihood_history.push(log_lik);

            // Check convergence
            if log_likelihood_history.len() > 1 {
                let improvement =
                    (log_lik - log_likelihood_history[log_likelihood_history.len() - 2]).abs();
                if improvement < self.tol {
                    converged = true;
                    break;
                }
            }
        }

        let n_iter = log_likelihood_history.len();
        let trained_state = MiniBatchGMMTrained {
            weights,
            means,
            covariances,
            log_likelihood_history,
            batch_sizes,
            n_iter,
            converged,
        };

        Ok(MiniBatchGMM {
            state: Untrained,
            n_components: self.n_components,
            batch_strategy: self.batch_strategy,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            random_state: self.random_state,
        }
        .with_state(trained_state))
    }
}

impl MiniBatchGMM<Untrained> {
    fn with_state(self, state: MiniBatchGMMTrained) -> MiniBatchGMM<MiniBatchGMMTrained> {
        MiniBatchGMM {
            state,
            n_components: self.n_components,
            batch_strategy: self.batch_strategy,
            covariance_type: self.covariance_type,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            random_state: self.random_state,
        }
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<usize>> for MiniBatchGMM<MiniBatchGMMTrained> {
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

// Parallel EM GMM (placeholder structure)
#[derive(Debug, Clone)]
pub struct ParallelGMM<S = Untrained> {
    #[allow(dead_code)]
    n_components: usize,
    #[allow(dead_code)]
    parallel_strategy: ParallelStrategy,
    _phantom: std::marker::PhantomData<S>,
}

#[derive(Debug, Clone)]
pub struct ParallelGMMTrained {
    pub weights: Array1<f64>,
    pub means: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct ParallelGMMBuilder {
    n_components: usize,
    parallel_strategy: ParallelStrategy,
}

impl ParallelGMMBuilder {
    pub fn new() -> Self {
        Self {
            n_components: 1,
            parallel_strategy: ParallelStrategy::DataParallel { n_threads: 4 },
        }
    }

    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    pub fn parallel_strategy(mut self, strategy: ParallelStrategy) -> Self {
        self.parallel_strategy = strategy;
        self
    }

    pub fn build(self) -> ParallelGMM<Untrained> {
        ParallelGMM {
            n_components: self.n_components,
            parallel_strategy: self.parallel_strategy,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Default for ParallelGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelGMM<Untrained> {
    pub fn builder() -> ParallelGMMBuilder {
        ParallelGMMBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_minibatch_gmm_builder() {
        let model = MiniBatchGMM::builder()
            .n_components(3)
            .batch_strategy(BatchStrategy::Fixed { size: 128 })
            .learning_rate(0.05)
            .build();

        assert_eq!(model.n_components, 3);
        assert_eq!(model.batch_strategy, BatchStrategy::Fixed { size: 128 });
        assert_eq!(model.learning_rate, 0.05);
    }

    #[test]
    fn test_batch_strategy_types() {
        let strategies = vec![
            BatchStrategy::Fixed { size: 100 },
            BatchStrategy::Adaptive {
                initial_size: 50,
                max_size: 500,
            },
            BatchStrategy::Dynamic {
                target_memory_mb: 100,
            },
        ];

        for strategy in strategies {
            let model = MiniBatchGMM::builder().batch_strategy(strategy).build();
            assert_eq!(model.batch_strategy, strategy);
        }
    }

    #[test]
    fn test_parallel_strategy_types() {
        let strategies = vec![
            ParallelStrategy::DataParallel { n_threads: 4 },
            ParallelStrategy::ModelParallel { n_threads: 2 },
            ParallelStrategy::Hybrid {
                data_threads: 2,
                model_threads: 2,
            },
        ];

        for strategy in strategies {
            let model = ParallelGMM::builder().parallel_strategy(strategy).build();
            assert_eq!(model.parallel_strategy, strategy);
        }
    }

    #[test]
    #[allow(non_snake_case)] // standard ML notation
    fn test_minibatch_gmm_fit() {
        let X = array![
            [1.0, 2.0],
            [1.5, 2.5],
            [10.0, 11.0],
            [10.5, 11.5],
            [5.0, 6.0],
            [5.5, 6.5]
        ];

        let model = MiniBatchGMM::builder()
            .n_components(2)
            .batch_strategy(BatchStrategy::Fixed { size: 3 })
            .max_iter(10)
            .build();

        let result = model.fit(&X.view(), &());
        assert!(result.is_ok());
    }

    /// Regression test for the fabrication bug: `with_state` used to
    /// discard the fitted parameters, so `predict` always returned
    /// all-zeros. Uses `scirs2_core::random::seeded_rng` (per this crate's
    /// SciRS2 Policy -- `rand`/`rand_chacha` are banned) to generate two
    /// well-separated Gaussian blobs; a real fit must recover the cluster
    /// structure.
    #[test]
    #[allow(non_snake_case)]
    fn test_minibatch_gmm_predict_recovers_cluster_structure_seeded_blobs() {
        use scirs2_core::random::RandNormal;

        let mut rng = seeded_rng(42);
        let normal = RandNormal::new(0.0, 0.5).expect("normal distribution should be valid");
        let n_per_cluster = 30;
        let centers = [[0.0_f64, 0.0], [20.0, 20.0]];
        let mut data = Vec::with_capacity(n_per_cluster * centers.len() * 2);
        for center in &centers {
            for _ in 0..n_per_cluster {
                data.push(center[0] + rng.sample(normal));
                data.push(center[1] + rng.sample(normal));
            }
        }
        let X = Array2::from_shape_vec((n_per_cluster * centers.len(), 2), data)
            .expect("data length must match the declared shape");

        let model = MiniBatchGMM::builder()
            .n_components(2)
            .batch_strategy(BatchStrategy::Fixed { size: 10 })
            .max_iter(30)
            .learning_rate(0.3)
            .random_state(7)
            .build();
        let fitted = model
            .fit(&X.view(), &())
            .expect("MiniBatchGMM fit should succeed on well-separated blobs");
        let preds = fitted
            .predict(&X.view())
            .expect("MiniBatchGMM predict should succeed");

        let distinct: std::collections::HashSet<usize> = preds.iter().copied().collect();
        assert!(
            distinct.len() > 1,
            "predictions collapsed onto a single label (the old all-zeros bug): {:?}",
            preds
        );

        let label_0 = preds[0];
        for i in 0..n_per_cluster {
            assert_eq!(
                preds[i], label_0,
                "cluster-0 point {i} should share the cluster label"
            );
        }
        let label_1 = preds[n_per_cluster];
        assert_ne!(
            label_0, label_1,
            "the two well-separated blobs must not collapse onto the same label"
        );
        for i in n_per_cluster..(2 * n_per_cluster) {
            assert_eq!(
                preds[i], label_1,
                "cluster-1 point {i} should share the cluster label"
            );
        }
    }

    /// Regression test for the missing covariance M-step update: previously
    /// `covariances` was never declared `mut` and was never touched after
    /// initialization, so it stayed frozen at `eye(n_features) *
    /// (1.0 + reg_covar)` regardless of the data's actual spread.
    #[test]
    #[allow(non_snake_case)]
    fn test_minibatch_gmm_updates_covariances_not_frozen_at_init() {
        let X = array![
            [0.0, 0.0],
            [0.05, -0.05],
            [0.1, 0.0],
            [-0.05, 0.05],
            [10.0, 10.0],
            [10.05, 9.95],
            [9.95, 10.05],
            [10.1, 10.0],
        ];
        let model = MiniBatchGMM::builder()
            .n_components(2)
            .batch_strategy(BatchStrategy::Fixed { size: 4 })
            .max_iter(20)
            .learning_rate(0.5)
            .random_state(3)
            .build();
        let fitted = model.fit(&X.view(), &()).expect("fit should succeed");

        let initial_cov_diag_value = 1.0 + 1e-6; // eye(n_features) * (1.0 + reg_covar)
        let final_diag = fitted.state.covariances.diag();
        let changed = final_diag
            .iter()
            .any(|&v| (v - initial_cov_diag_value).abs() > 1e-9);
        assert!(
            changed,
            "covariances must be updated by the M-step, not frozen at the initial \
             identity+reg_covar value: {:?}",
            final_diag
        );
        // These clusters have a tight spread (~0.05-0.1); the fitted tied
        // variance should shrink well below the initial ~1.0 seed value.
        for &v in final_diag.iter() {
            assert!(
                v < 0.5,
                "fitted variance should shrink toward the tight cluster spread, got {v}"
            );
        }
    }

    /// The old "log-likelihood" summed raw mixture weights (which sum to
    /// ~1.0) for `sample_size` iterations without ever indexing into the
    /// data. The real log-sum-exp log-likelihood must be a meaningfully
    /// different, data-dependent number.
    #[test]
    #[allow(non_snake_case)]
    fn test_minibatch_log_likelihood_is_not_the_old_weights_sum_bug() {
        let X = array![
            [0.0, 0.0],
            [0.1, -0.1],
            [10.0, 10.0],
            [10.1, 9.9],
            [5.0, -5.0],
            [5.1, -4.9],
        ];
        let fitted = MiniBatchGMM::builder()
            .n_components(3)
            .batch_strategy(BatchStrategy::Fixed { size: 6 })
            .max_iter(20)
            .learning_rate(0.5)
            .random_state(11)
            .build()
            .fit(&X.view(), &())
            .expect("fit should succeed");

        let final_ll = *fitted
            .state
            .log_likelihood_history
            .last()
            .expect("history should be non-empty");
        assert!(final_ll.is_finite());
        // The old bug computed `ln(sum_k weights[k])` each "sample"
        // (weights sum to ~1.0 after normalization), landing within
        // floating rounding of 0.0 regardless of fit quality.
        assert!(
            final_ll.abs() > 1e-3,
            "log-likelihood looks like the old ln(sum(weights)) ~ 0 placeholder bug: {final_ll}"
        );
    }

    #[test]
    fn test_builder_defaults() {
        let model = MiniBatchGMM::builder().build();
        assert_eq!(model.n_components, 1);
        assert_eq!(model.learning_rate, 0.1);
        assert_eq!(model.momentum, 0.9);
    }

    #[test]
    fn test_parallel_gmm_builder() {
        let model = ParallelGMM::builder()
            .n_components(4)
            .parallel_strategy(ParallelStrategy::DataParallel { n_threads: 8 })
            .build();

        assert_eq!(model.n_components, 4);
    }
}
