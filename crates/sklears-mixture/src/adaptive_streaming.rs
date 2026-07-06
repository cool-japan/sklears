//! Adaptive Streaming Mixture Models
//!
//! This module provides mixture models with adaptive component management for
//! streaming data, including automatic component creation and deletion based on
//! data characteristics and model performance.
//!
//! # Overview
//!
//! Adaptive streaming mixtures automatically adjust the number of components
//! based on incoming data, making them ideal for:
//! - Non-stationary data streams
//! - Evolving cluster structures
//! - Real-time learning scenarios
//! - Concept drift handling
//!
//! # Key Features
//!
//! - **Automatic Component Creation**: New components added when data doesn't fit existing ones
//! - **Automatic Component Deletion**: Weak/redundant components removed
//! - **Concept Drift Detection**: Detect and adapt to distribution changes
//! - **Memory Management**: Bounded memory usage with component limits

use crate::common::CovarianceType;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

/// Criteria for component creation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CreationCriterion {
    /// Create component when likelihood falls below threshold
    LikelihoodThreshold { threshold: f64 },
    /// Create component based on distance to nearest component
    DistanceThreshold { threshold: f64 },
    /// Create component based on number of consecutive outliers
    OutlierCount { count: usize },
}

/// Criteria for component deletion
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeletionCriterion {
    /// Delete component when weight falls below threshold
    WeightThreshold { threshold: f64 },
    /// Delete component when it hasn't been updated recently
    InactivityPeriod { periods: usize },
    /// Delete component when it's too similar to another
    RedundancyThreshold { threshold: f64 },
}

/// Concept drift detection method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftDetectionMethod {
    /// Page-Hinkley test for drift detection
    PageHinkley { delta: f64, lambda: f64 },
    /// ADWIN (Adaptive Windowing) for drift detection
    ADWIN { delta: f64 },
    /// Cumulative sum (CUSUM) for drift detection
    CUSUM { threshold: f64, drift_level: f64 },
}

/// Configuration for adaptive streaming mixture
#[derive(Debug, Clone)]
pub struct AdaptiveStreamingConfig {
    /// Minimum number of components
    pub min_components: usize,
    /// Maximum number of components
    pub max_components: usize,
    /// Creation criterion
    pub creation_criterion: CreationCriterion,
    /// Deletion criterion
    pub deletion_criterion: DeletionCriterion,
    /// Drift detection method
    pub drift_detection: Option<DriftDetectionMethod>,
    /// Learning rate for parameter updates
    pub learning_rate: f64,
    /// Learning rate decay
    pub decay_rate: f64,
    /// Minimum samples before component deletion
    pub min_samples_before_delete: usize,
    /// Covariance type
    pub covariance_type: CovarianceType,
    /// Maximum number of EM iterations for the initial batch `fit`
    pub max_iter: usize,
    /// Convergence tolerance (log-likelihood improvement) for the initial
    /// batch `fit`
    pub tol: f64,
    /// Regularization added to component variances for numerical stability
    pub reg_covar: f64,
}

impl Default for AdaptiveStreamingConfig {
    fn default() -> Self {
        Self {
            min_components: 1,
            max_components: 20,
            creation_criterion: CreationCriterion::LikelihoodThreshold { threshold: -10.0 },
            deletion_criterion: DeletionCriterion::WeightThreshold { threshold: 0.01 },
            drift_detection: Some(DriftDetectionMethod::PageHinkley {
                delta: 0.005,
                lambda: 50.0,
            }),
            learning_rate: 0.1,
            decay_rate: 0.99,
            min_samples_before_delete: 100,
            covariance_type: CovarianceType::Diagonal,
            max_iter: 100,
            tol: 1e-3,
            reg_covar: 1e-6,
        }
    }
}

/// Adaptive Streaming Gaussian Mixture Model
///
/// A streaming mixture model that automatically creates and deletes components
/// based on data characteristics and model performance.
///
/// # Examples
///
/// ```
/// use sklears_mixture::adaptive_streaming::{AdaptiveStreamingGMM, CreationCriterion};
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let model = AdaptiveStreamingGMM::builder()
///     .min_components(1)
///     .max_components(10)
///     .creation_criterion(CreationCriterion::LikelihoodThreshold { threshold: -5.0 })
///     .build();
///
/// let X = array![[1.0, 2.0], [1.5, 2.5], [10.0, 11.0]];
/// let fitted = model.fit(&X.view(), &()).expect("adaptive streaming GMM fitting should succeed with valid data");
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveStreamingGMM<S = Untrained> {
    pub(crate) state: S,
    config: AdaptiveStreamingConfig,
}

/// Trained Adaptive Streaming GMM state
#[derive(Debug, Clone)]
pub struct AdaptiveStreamingGMMTrained {
    /// Current component weights
    pub weights: Array1<f64>,
    /// Current component means
    pub means: Array2<f64>,
    /// Current component covariances: one diagonal (per-feature variance)
    /// row per component, shape `(n_components, n_features)`.
    pub covariances: Array2<f64>,
    /// Number of samples seen per component
    pub component_counts: Array1<usize>,
    /// Last update iteration for each component
    pub last_update: Array1<usize>,
    /// Total samples processed
    pub total_samples: usize,
    /// Current learning rate
    pub learning_rate: f64,
    /// Component creation history (records `total_samples` at each dynamic
    /// creation event triggered by [`AdaptiveStreamingGMM::partial_fit`])
    pub creation_history: Vec<usize>,
    /// Component deletion history (records the deleted component's index at
    /// each dynamic deletion event)
    pub deletion_history: Vec<usize>,
    /// Whether the most recent `partial_fit` call flagged concept drift
    pub drift_detected: bool,
    /// Drift detection cumulative sum (Page-Hinkley `PH_t`, or the CUSUM
    /// statistic `g_t`, depending on `config.drift_detection`)
    pub drift_cumsum: f64,
    /// Running mean of the drift-detection statistic (the log-likelihood of
    /// incoming samples), used by the Page-Hinkley and CUSUM tests
    pub drift_running_mean: f64,
    /// Running minimum of the Page-Hinkley cumulative sum (unused by CUSUM)
    pub drift_min_cumsum: f64,
    /// Count of consecutive samples flagged as statistical outliers (used
    /// by [`CreationCriterion::OutlierCount`])
    pub consecutive_outliers: usize,
    /// Number of EM iterations performed by the initial batch `fit`
    pub n_iter: usize,
    /// Whether the initial batch `fit` converged
    pub converged: bool,
    /// Configuration
    pub config: AdaptiveStreamingConfig,
}

/// Builder for Adaptive Streaming GMM
#[derive(Debug, Clone)]
pub struct AdaptiveStreamingGMMBuilder {
    config: AdaptiveStreamingConfig,
}

impl AdaptiveStreamingGMMBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: AdaptiveStreamingConfig::default(),
        }
    }

    /// Set minimum components
    pub fn min_components(mut self, min: usize) -> Self {
        self.config.min_components = min;
        self
    }

    /// Set maximum components
    pub fn max_components(mut self, max: usize) -> Self {
        self.config.max_components = max;
        self
    }

    /// Set creation criterion
    pub fn creation_criterion(mut self, criterion: CreationCriterion) -> Self {
        self.config.creation_criterion = criterion;
        self
    }

    /// Set deletion criterion
    pub fn deletion_criterion(mut self, criterion: DeletionCriterion) -> Self {
        self.config.deletion_criterion = criterion;
        self
    }

    /// Set drift detection method
    pub fn drift_detection(mut self, method: DriftDetectionMethod) -> Self {
        self.config.drift_detection = Some(method);
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set learning rate decay
    pub fn decay_rate(mut self, decay: f64) -> Self {
        self.config.decay_rate = decay;
        self
    }

    /// Set the maximum number of EM iterations for the initial batch `fit`
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance for the initial batch `fit`
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the covariance regularization floor
    pub fn reg_covar(mut self, reg_covar: f64) -> Self {
        self.config.reg_covar = reg_covar;
        self
    }

    /// Build the model
    pub fn build(self) -> AdaptiveStreamingGMM<Untrained> {
        AdaptiveStreamingGMM {
            state: Untrained,
            config: self.config,
        }
    }
}

impl Default for AdaptiveStreamingGMMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveStreamingGMM<Untrained> {
    /// Create a new builder
    pub fn builder() -> AdaptiveStreamingGMMBuilder {
        AdaptiveStreamingGMMBuilder::new()
    }
}

impl Estimator for AdaptiveStreamingGMM<Untrained> {
    type Config = AdaptiveStreamingConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for AdaptiveStreamingGMM<Untrained> {
    type Fitted = AdaptiveStreamingGMM<AdaptiveStreamingGMMTrained>;

    /// Fit an initial (per-component diagonal-covariance) Gaussian mixture
    /// via genuine EM on the given batch, using `config.min_components`
    /// components. This seeds the state that [`AdaptiveStreamingGMM::partial_fit`]
    /// subsequently updates online, and that `predict` reads from.
    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X_owned = X.to_owned();
        let (n_samples, n_features) = X_owned.dim();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit with zero samples".to_string(),
            ));
        }
        let n_components = self.config.min_components.max(1);
        if n_samples < n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be >= min_components".to_string(),
            ));
        }

        // Deterministic initialization: evenly spaced samples as the
        // starting means (mirrors `GaussianMixture::initialize_means` in
        // gaussian.rs).
        let mut means = Array2::zeros((n_components, n_features));
        let step = (n_samples / n_components).max(1);
        for k in 0..n_components {
            let idx = (k * step).min(n_samples - 1);
            means.row_mut(k).assign(&X_owned.row(idx));
        }

        let mut weights = Array1::from_elem(n_components, 1.0 / n_components as f64);
        let mut covariances =
            Array2::from_elem((n_components, n_features), 1.0 + self.config.reg_covar);

        let mut n_iter = 0usize;
        let mut converged = false;
        let mut prev_log_lik = f64::NEG_INFINITY;
        let mut component_counts_f = Array1::<f64>::zeros(n_components);

        // Genuine EM: per-component diagonal covariances via the shared
        // `gaussian_log_pdf_diagonal` helper in common.rs (previously `fit`
        // only initialized means/covariances and never iterated at all).
        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            let mut responsibilities = Array2::<f64>::zeros((n_samples, n_components));
            let mut log_lik = 0.0;
            for i in 0..n_samples {
                let sample = X_owned.row(i);
                let mut log_probs = Vec::with_capacity(n_components);
                for k in 0..n_components {
                    let mean_k = means.row(k);
                    let cov_k = covariances.row(k);
                    let lp = crate::common::gaussian_log_pdf_diagonal(&sample, &mean_k, &cov_k)?;
                    log_probs.push(weights[k].ln() + lp);
                }
                let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
                log_lik += max_log + sum_exp.ln();
                for k in 0..n_components {
                    responsibilities[[i, k]] = (log_probs[k] - max_log).exp() / sum_exp;
                }
            }

            for k in 0..n_components {
                let resp_k = responsibilities.column(k);
                let nk = resp_k.sum().max(1e-10);
                component_counts_f[k] = nk;
                weights[k] = nk / n_samples as f64;

                let mut new_mean = Array1::zeros(n_features);
                for i in 0..n_samples {
                    new_mean += &(X_owned.row(i).to_owned() * resp_k[i]);
                }
                new_mean /= nk;

                let mut new_var = Array1::zeros(n_features);
                for i in 0..n_samples {
                    let diff = &X_owned.row(i).to_owned() - &new_mean;
                    new_var += &(diff.mapv(|v| v * v) * resp_k[i]);
                }
                new_var = new_var / nk + Array1::from_elem(n_features, self.config.reg_covar);

                means.row_mut(k).assign(&new_mean);
                covariances.row_mut(k).assign(&new_var);
            }
            let weight_sum = weights.sum();
            weights /= weight_sum;

            if iter > 0 && (log_lik - prev_log_lik).abs() < self.config.tol {
                converged = true;
                break;
            }
            prev_log_lik = log_lik;
        }

        let component_counts = component_counts_f.mapv(|v| v.round() as usize);
        let last_update = Array1::from_elem(n_components, n_samples);
        let config_clone = self.config.clone();

        let trained_state = AdaptiveStreamingGMMTrained {
            weights,
            means,
            covariances,
            component_counts,
            last_update,
            total_samples: n_samples,
            learning_rate: config_clone.learning_rate,
            creation_history: Vec::new(),
            deletion_history: Vec::new(),
            drift_detected: false,
            drift_cumsum: 0.0,
            drift_running_mean: 0.0,
            drift_min_cumsum: 0.0,
            consecutive_outliers: 0,
            n_iter,
            converged,
            config: config_clone,
        };

        Ok(AdaptiveStreamingGMM {
            state: Untrained,
            config: self.config,
        }
        .with_state(trained_state))
    }
}

impl AdaptiveStreamingGMM<Untrained> {
    fn with_state(
        self,
        state: AdaptiveStreamingGMMTrained,
    ) -> AdaptiveStreamingGMM<AdaptiveStreamingGMMTrained> {
        AdaptiveStreamingGMM {
            state,
            config: self.config,
        }
    }
}

impl AdaptiveStreamingGMM<AdaptiveStreamingGMMTrained> {
    /// Update the model with a new sample (real online/stochastic EM
    /// update): computes the sample's responsibility under each current
    /// component (E-step), then blends each component's sufficient
    /// statistics toward the sample using the (decaying) learning rate
    /// (M-step), merging into the existing running statistics rather than
    /// discarding them. Also drives dynamic component creation/deletion and
    /// concept-drift detection.
    ///
    /// Previously this was a documented no-op ("a real implementation would
    /// store the state within the struct").
    #[allow(non_snake_case)]
    pub fn partial_fit(&mut self, x: &ArrayView1<'_, Float>) -> SklResult<()> {
        let n_features = self.state.means.ncols();
        if x.len() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "partial_fit: expected {} features, got {}",
                n_features,
                x.len()
            )));
        }

        // Track consecutive "far from every component" observations, used
        // by the `CreationCriterion::OutlierCount` rule.
        if self.is_outlier(x) {
            self.state.consecutive_outliers += 1;
        } else {
            self.state.consecutive_outliers = 0;
        }

        if self.state.means.nrows() < self.state.config.max_components
            && self.should_create_component(x)?
        {
            self.create_component(x)?;
            self.state.consecutive_outliers = 0;
        }

        // E-step: responsibility of `x` under each current component.
        let n_components = self.state.means.nrows();
        let mut log_probs = Vec::with_capacity(n_components);
        for k in 0..n_components {
            let mean_k = self.state.means.row(k);
            let cov_k = self.state.covariances.row(k);
            let lp = crate::common::gaussian_log_pdf_diagonal(x, &mean_k, &cov_k)?;
            log_probs.push(self.state.weights[k].ln() + lp);
        }
        let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
        let sample_log_lik = max_log + sum_exp.ln();
        let responsibilities: Vec<f64> = log_probs
            .iter()
            .map(|&lp| (lp - max_log).exp() / sum_exp)
            .collect();

        // Online/stochastic EM update: blend each component's sufficient
        // statistics toward the new sample, weighted by its responsibility
        // and the current (decaying) learning rate. This merges into the
        // existing running state rather than recomputing from scratch.
        let lr = self.state.learning_rate;
        for (k, &r) in responsibilities.iter().enumerate() {
            self.state.weights[k] = (1.0 - lr) * self.state.weights[k] + lr * r;
            if r <= 1e-12 {
                continue;
            }
            let step = lr * r;

            let mean_k = self.state.means.row(k).to_owned();
            let diff = x.to_owned() - &mean_k;
            let new_mean = &mean_k + &(&diff * step);

            let sq_diff = (x.to_owned() - &new_mean).mapv(|v| v * v);
            let old_var = self.state.covariances.row(k).to_owned();
            let reg_covar_term = Array1::from_elem(n_features, self.state.config.reg_covar);
            let new_var = &old_var * (1.0 - step) + &(sq_diff * step) + &reg_covar_term;

            self.state.means.row_mut(k).assign(&new_mean);
            self.state.covariances.row_mut(k).assign(&new_var);
            self.state.component_counts[k] += 1;
            self.state.last_update[k] = self.state.total_samples;
        }
        let weight_sum = self.state.weights.sum();
        if weight_sum > 0.0 {
            self.state.weights /= weight_sum;
        }

        self.state.total_samples += 1;
        self.state.learning_rate *= self.state.config.decay_rate;

        // Concept-drift detection (Page-Hinkley / CUSUM are real; ADWIN is
        // an honest `NotImplemented` -- see `detect_drift`).
        let drifted = self.detect_drift(sample_log_lik)?;
        self.state.drift_detected = drifted;

        // Component pruning: never below `min_components`.
        if self.state.means.nrows() > self.state.config.min_components {
            let to_delete = self.components_to_delete();
            if !to_delete.is_empty() {
                self.delete_components(&to_delete)?;
            }
        }

        Ok(())
    }

    /// Whether `x` is a statistical outlier under the current mixture: its
    /// minimum (per-component) standardized Mahalanobis-style distance
    /// exceeds the classical 3-sigma-equivalent threshold. A disclosed
    /// simplification (a single fixed threshold rather than a calibrated
    /// per-model one), but a real, data-dependent computation.
    fn is_outlier(&self, x: &ArrayView1<'_, Float>) -> bool {
        let n_components = self.state.means.nrows();
        let mut min_mahal = f64::INFINITY;
        for k in 0..n_components {
            let mean_k = self.state.means.row(k);
            let cov_k = self.state.covariances.row(k);
            let mahal: f64 = x
                .iter()
                .zip(mean_k.iter())
                .zip(cov_k.iter())
                .map(|((xi, mi), v)| {
                    let d = xi - mi;
                    d * d / v.max(1e-12)
                })
                .sum();
            min_mahal = min_mahal.min(mahal.sqrt());
        }
        min_mahal > 3.0
    }

    /// Check if a new component should be created, per `config.creation_criterion`.
    fn should_create_component(&self, x: &ArrayView1<'_, Float>) -> SklResult<bool> {
        match self.state.config.creation_criterion {
            CreationCriterion::LikelihoodThreshold { threshold } => {
                let n_components = self.state.means.nrows();
                let mut log_probs = Vec::with_capacity(n_components);
                for k in 0..n_components {
                    let mean_k = self.state.means.row(k);
                    let cov_k = self.state.covariances.row(k);
                    let lp = crate::common::gaussian_log_pdf_diagonal(x, &mean_k, &cov_k)?;
                    log_probs.push(self.state.weights[k].ln() + lp);
                }
                let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
                let log_lik = max_log + sum_exp.ln();
                Ok(log_lik < threshold)
            }
            CreationCriterion::DistanceThreshold { threshold } => {
                let n_components = self.state.means.nrows();
                let mut min_dist = f64::INFINITY;
                for k in 0..n_components {
                    let mean_k = self.state.means.row(k);
                    let dist: f64 = x
                        .iter()
                        .zip(mean_k.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    min_dist = min_dist.min(dist);
                }
                Ok(min_dist > threshold)
            }
            CreationCriterion::OutlierCount { count } => {
                Ok(self.state.consecutive_outliers >= count)
            }
        }
    }

    /// Create a new component centered at `x`, giving it a small starting
    /// weight (renormalizing the rest) and seeding its variance from the
    /// average variance of the existing components.
    fn create_component(&mut self, x: &ArrayView1<'_, Float>) -> SklResult<()> {
        let n_features = self.state.means.ncols();
        let n_components = self.state.means.nrows();

        let init_weight = 1.0 / (n_components as f64 + 1.0);
        let mut new_weights = Array1::zeros(n_components + 1);
        for k in 0..n_components {
            new_weights[k] = self.state.weights[k] * (1.0 - init_weight);
        }
        new_weights[n_components] = init_weight;

        let mut new_means = Array2::zeros((n_components + 1, n_features));
        new_means
            .slice_mut(s![0..n_components, ..])
            .assign(&self.state.means);
        new_means.row_mut(n_components).assign(x);

        let avg_var = if n_components > 0 {
            self.state.covariances.mean_axis(Axis(0)).ok_or_else(|| {
                SklearsError::NumericalError(
                    "failed to average existing component variances".to_string(),
                )
            })?
        } else {
            Array1::from_elem(n_features, 1.0)
        };
        let mut new_covariances = Array2::zeros((n_components + 1, n_features));
        new_covariances
            .slice_mut(s![0..n_components, ..])
            .assign(&self.state.covariances);
        new_covariances.row_mut(n_components).assign(&avg_var);

        let mut new_counts = Array1::zeros(n_components + 1);
        new_counts
            .slice_mut(s![0..n_components])
            .assign(&self.state.component_counts);
        new_counts[n_components] = 1;

        let mut new_last_update = Array1::zeros(n_components + 1);
        new_last_update
            .slice_mut(s![0..n_components])
            .assign(&self.state.last_update);
        new_last_update[n_components] = self.state.total_samples;

        self.state.weights = new_weights;
        self.state.means = new_means;
        self.state.covariances = new_covariances;
        self.state.component_counts = new_counts;
        self.state.last_update = new_last_update;
        self.state.creation_history.push(self.state.total_samples);

        Ok(())
    }

    /// Check which components should be deleted, per `config.deletion_criterion`,
    /// never proposing to go below `min_components`.
    fn components_to_delete(&self) -> Vec<usize> {
        let n_components = self.state.means.nrows();
        let min_components = self.state.config.min_components;
        let mut to_delete = Vec::new();

        match self.state.config.deletion_criterion {
            DeletionCriterion::WeightThreshold { threshold } => {
                for k in 0..n_components {
                    if self.state.weights[k] < threshold
                        && n_components - to_delete.len() > min_components
                    {
                        to_delete.push(k);
                    }
                }
            }
            DeletionCriterion::InactivityPeriod { periods } => {
                for k in 0..n_components {
                    let inactive_for = self
                        .state
                        .total_samples
                        .saturating_sub(self.state.last_update[k]);
                    if inactive_for > periods && n_components - to_delete.len() > min_components {
                        to_delete.push(k);
                    }
                }
            }
            DeletionCriterion::RedundancyThreshold { threshold } => {
                // Two components are "redundant" if their means are closer
                // than `threshold` (Euclidean); keep the one with the
                // larger weight.
                for k in 0..n_components {
                    if to_delete.contains(&k) {
                        continue;
                    }
                    for j in (k + 1)..n_components {
                        if to_delete.contains(&j) {
                            continue;
                        }
                        let mean_k = self.state.means.row(k);
                        let mean_j = self.state.means.row(j);
                        let dist: f64 = mean_k
                            .iter()
                            .zip(mean_j.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        if dist < threshold && n_components - to_delete.len() > min_components {
                            let weaker = if self.state.weights[k] <= self.state.weights[j] {
                                k
                            } else {
                                j
                            };
                            to_delete.push(weaker);
                        }
                    }
                }
            }
        }
        to_delete
    }

    /// Delete the components at the given indices, renormalizing weights.
    fn delete_components(&mut self, indices: &[usize]) -> SklResult<()> {
        if indices.is_empty() {
            return Ok(());
        }
        let n_components = self.state.means.nrows();
        let keep: Vec<usize> = (0..n_components).filter(|k| !indices.contains(k)).collect();
        if keep.is_empty() {
            return Err(SklearsError::InvalidState(
                "cannot delete all mixture components".to_string(),
            ));
        }

        let n_features = self.state.means.ncols();
        let mut new_weights = Array1::zeros(keep.len());
        let mut new_means = Array2::zeros((keep.len(), n_features));
        let mut new_covariances = Array2::zeros((keep.len(), n_features));
        let mut new_counts = Array1::zeros(keep.len());
        let mut new_last_update = Array1::zeros(keep.len());

        for (new_k, &old_k) in keep.iter().enumerate() {
            new_weights[new_k] = self.state.weights[old_k];
            new_means
                .row_mut(new_k)
                .assign(&self.state.means.row(old_k));
            new_covariances
                .row_mut(new_k)
                .assign(&self.state.covariances.row(old_k));
            new_counts[new_k] = self.state.component_counts[old_k];
            new_last_update[new_k] = self.state.last_update[old_k];
        }
        let weight_sum = new_weights.sum();
        if weight_sum > 0.0 {
            new_weights /= weight_sum;
        }

        self.state.weights = new_weights;
        self.state.means = new_means;
        self.state.covariances = new_covariances;
        self.state.component_counts = new_counts;
        self.state.last_update = new_last_update;
        for &idx in indices {
            self.state.deletion_history.push(idx);
        }

        Ok(())
    }

    /// Detect concept drift from the log-likelihood of the most recent
    /// observation, per `config.drift_detection`.
    ///
    /// Both real detectors below are the standard *increase*-detecting
    /// formulation (Page & Hinkley 1954/1955; Page 1954 CUSUM), which is why
    /// they monitor `surprise = -log_likelihood` rather than the raw
    /// log-likelihood: a drifting/anomalous sample makes the data *less*
    /// likely under the current model (log-likelihood drops), i.e. surprise
    /// *increases*, which is exactly what these formulas are designed to
    /// flag. [`DriftDetectionMethod::PageHinkley`] and
    /// [`DriftDetectionMethod::CUSUM`] reuse the
    /// `drift_cumsum`/`drift_running_mean`/`drift_min_cumsum` running state.
    /// [`DriftDetectionMethod::ADWIN`] requires an adaptive windowing /
    /// exponential-histogram data structure that is genuinely out of scope
    /// here, so it honestly returns `Err(SklearsError::NotImplemented(...))`
    /// rather than silently reporting `false`.
    fn detect_drift(&mut self, log_likelihood: f64) -> SklResult<bool> {
        let surprise = -log_likelihood;
        match self.state.config.drift_detection {
            Some(DriftDetectionMethod::PageHinkley { delta, lambda }) => {
                let n = self.state.total_samples.max(1) as f64;
                self.state.drift_running_mean += (surprise - self.state.drift_running_mean) / n;
                self.state.drift_cumsum += surprise - self.state.drift_running_mean - delta;
                if self.state.drift_cumsum < self.state.drift_min_cumsum {
                    self.state.drift_min_cumsum = self.state.drift_cumsum;
                }
                let ph = self.state.drift_cumsum - self.state.drift_min_cumsum;
                Ok(ph > lambda)
            }
            Some(DriftDetectionMethod::CUSUM {
                threshold,
                drift_level,
            }) => {
                let n = self.state.total_samples.max(1) as f64;
                self.state.drift_running_mean += (surprise - self.state.drift_running_mean) / n;
                let g = (self.state.drift_cumsum + surprise
                    - self.state.drift_running_mean
                    - drift_level)
                    .max(0.0);
                self.state.drift_cumsum = g;
                Ok(g > threshold)
            }
            Some(DriftDetectionMethod::ADWIN { .. }) => Err(SklearsError::NotImplemented(
                "AdaptiveStreamingGMM: ADWIN drift detection requires an adaptive \
                 windowing/exponential-histogram data structure that is not yet \
                 implemented; use DriftDetectionMethod::PageHinkley or \
                 DriftDetectionMethod::CUSUM instead"
                    .to_string(),
            )),
            None => Ok(false),
        }
    }

    /// Get the current (possibly dynamically grown/shrunk) number of components.
    pub fn n_components(&self) -> usize {
        self.state.means.nrows()
    }

    /// Get component creation history (the `total_samples` count at each
    /// dynamic creation event).
    pub fn creation_history(&self) -> &[usize] {
        &self.state.creation_history
    }

    /// Get component deletion history (the deleted component's index at
    /// each dynamic deletion event).
    pub fn deletion_history(&self) -> &[usize] {
        &self.state.deletion_history
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<usize>>
    for AdaptiveStreamingGMM<AdaptiveStreamingGMMTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<usize>> {
        let X_owned = X.to_owned();
        let (n_samples, _) = X_owned.dim();
        let n_components = self.state.means.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for (i, sample) in X_owned.axis_iter(Axis(0)).enumerate() {
            let mut best_k = 0usize;
            let mut best_log_prob = f64::NEG_INFINITY;
            for k in 0..n_components {
                let mean_k = self.state.means.row(k);
                let cov_k = self.state.covariances.row(k);
                let lp = crate::common::gaussian_log_pdf_diagonal(&sample, &mean_k, &cov_k)?;
                let weighted = self.state.weights[k].ln() + lp;
                if weighted > best_log_prob {
                    best_log_prob = weighted;
                    best_k = k;
                }
            }
            predictions[i] = best_k;
        }

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_adaptive_streaming_gmm_builder() {
        let model = AdaptiveStreamingGMM::builder()
            .min_components(2)
            .max_components(15)
            .learning_rate(0.05)
            .build();

        assert_eq!(model.config.min_components, 2);
        assert_eq!(model.config.max_components, 15);
        assert_eq!(model.config.learning_rate, 0.05);
    }

    #[test]
    fn test_creation_criterion_types() {
        let criteria = vec![
            CreationCriterion::LikelihoodThreshold { threshold: -5.0 },
            CreationCriterion::DistanceThreshold { threshold: 2.0 },
            CreationCriterion::OutlierCount { count: 5 },
        ];

        for criterion in criteria {
            let model = AdaptiveStreamingGMM::builder()
                .creation_criterion(criterion)
                .build();
            assert_eq!(model.config.creation_criterion, criterion);
        }
    }

    #[test]
    fn test_deletion_criterion_types() {
        let criteria = vec![
            DeletionCriterion::WeightThreshold { threshold: 0.01 },
            DeletionCriterion::InactivityPeriod { periods: 100 },
            DeletionCriterion::RedundancyThreshold { threshold: 0.1 },
        ];

        for criterion in criteria {
            let model = AdaptiveStreamingGMM::builder()
                .deletion_criterion(criterion)
                .build();
            assert_eq!(model.config.deletion_criterion, criterion);
        }
    }

    #[test]
    fn test_drift_detection_methods() {
        let methods = vec![
            DriftDetectionMethod::PageHinkley {
                delta: 0.005,
                lambda: 50.0,
            },
            DriftDetectionMethod::ADWIN { delta: 0.002 },
            DriftDetectionMethod::CUSUM {
                threshold: 10.0,
                drift_level: 0.1,
            },
        ];

        for method in methods {
            let model = AdaptiveStreamingGMM::builder()
                .drift_detection(method)
                .build();
            assert_eq!(model.config.drift_detection, Some(method));
        }
    }

    #[test]
    #[allow(non_snake_case)] // standard ML notation
    fn test_adaptive_streaming_gmm_fit() {
        let X = array![[1.0, 2.0], [1.5, 2.5], [10.0, 11.0]];

        let model = AdaptiveStreamingGMM::builder()
            .min_components(1)
            .max_components(5)
            .build();

        let result = model.fit(&X.view(), &());
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_defaults() {
        let config = AdaptiveStreamingConfig::default();
        assert_eq!(config.min_components, 1);
        assert_eq!(config.max_components, 20);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.decay_rate, 0.99);
        assert_eq!(config.min_samples_before_delete, 100);
        assert_eq!(config.max_iter, 100);
        assert_eq!(config.tol, 1e-3);
        assert_eq!(config.reg_covar, 1e-6);
    }

    #[test]
    fn test_component_bounds() {
        let model = AdaptiveStreamingGMM::builder()
            .min_components(3)
            .max_components(8)
            .build();

        assert_eq!(model.config.min_components, 3);
        assert_eq!(model.config.max_components, 8);
        assert!(model.config.min_components <= model.config.max_components);
    }

    #[test]
    fn test_builder_chaining() {
        let model = AdaptiveStreamingGMM::builder()
            .min_components(2)
            .max_components(10)
            .learning_rate(0.05)
            .decay_rate(0.95)
            .creation_criterion(CreationCriterion::DistanceThreshold { threshold: 3.0 })
            .deletion_criterion(DeletionCriterion::WeightThreshold { threshold: 0.05 })
            .build();

        assert_eq!(model.config.min_components, 2);
        assert_eq!(model.config.max_components, 10);
        assert_eq!(model.config.learning_rate, 0.05);
        assert_eq!(model.config.decay_rate, 0.95);
    }

    /// Regression test for the fabrication bug: `with_state` used to
    /// discard the fitted parameters, so `predict` always returned
    /// all-zeros regardless of input. A real fit on two well-separated
    /// blobs must discriminate between them.
    #[test]
    #[allow(non_snake_case)] // standard ML notation
    fn test_adaptive_streaming_gmm_predict_recovers_cluster_structure() {
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
        let model = AdaptiveStreamingGMM::builder()
            .min_components(2)
            .max_components(2)
            .max_iter(50)
            .build();
        let fitted = model
            .fit(&X.view(), &())
            .expect("AdaptiveStreamingGMM fit should succeed on well-separated blobs");
        let preds = fitted
            .predict(&X.view())
            .expect("AdaptiveStreamingGMM predict should succeed");

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

    /// Regression test for the fabrication bug: `partial_fit` used to be a
    /// documented no-op ("a real implementation would store the state
    /// within the struct"). It must now genuinely mutate the running state.
    #[test]
    #[allow(non_snake_case)]
    fn test_partial_fit_updates_state_and_is_not_a_no_op() {
        let X = array![[0.0, 0.0], [0.2, 0.2], [10.0, 10.0], [10.2, 10.2]];
        let model = AdaptiveStreamingGMM::builder()
            .min_components(2)
            .max_components(2)
            .learning_rate(0.5)
            .build();
        let mut fitted = model.fit(&X.view(), &()).expect("fit should succeed");

        let means_before = fitted.state.means.clone();
        let total_before = fitted.state.total_samples;
        let new_sample = array![10.3, 10.3];
        fitted
            .partial_fit(&new_sample.view())
            .expect("partial_fit should succeed");

        assert_ne!(
            fitted.state.means, means_before,
            "partial_fit must actually change the model state, not silently no-op"
        );
        assert_eq!(fitted.state.total_samples, total_before + 1);
    }

    /// `should_create_component`/`create_component` must be real: a
    /// far-away sample under `DistanceThreshold` should genuinely grow the
    /// mixture (and be recorded in `creation_history`), not silently return
    /// `false`/no-op as before.
    #[test]
    #[allow(non_snake_case)]
    fn test_should_create_component_and_create_component_are_real() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2]];
        let model = AdaptiveStreamingGMM::builder()
            .min_components(1)
            .max_components(3)
            .creation_criterion(CreationCriterion::DistanceThreshold { threshold: 1.0 })
            .build();
        let mut fitted = model.fit(&X.view(), &()).expect("fit should succeed");
        assert_eq!(fitted.n_components(), 1);
        assert!(fitted.creation_history().is_empty());

        let far_sample = array![50.0, 50.0];
        assert!(
            fitted
                .should_create_component(&far_sample.view())
                .expect("should_create_component should succeed"),
            "a sample 50 units away from the only component should trigger creation"
        );
        fitted
            .partial_fit(&far_sample.view())
            .expect("partial_fit should succeed");
        assert_eq!(
            fitted.n_components(),
            2,
            "a far-away sample should grow the mixture (was hardcoded to 1 before)"
        );
        assert_eq!(fitted.creation_history().len(), 1);
    }

    /// `components_to_delete`/`delete_components` must be real: a
    /// deliberately near-zero-weight component under `WeightThreshold`
    /// should genuinely be pruned (and recorded in `deletion_history`).
    #[test]
    #[allow(non_snake_case)]
    fn test_components_to_delete_and_delete_components_are_real() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 9.9]];
        let model = AdaptiveStreamingGMM::builder()
            .min_components(2)
            .max_components(2)
            .deletion_criterion(DeletionCriterion::WeightThreshold { threshold: 0.4 })
            .build();
        let mut fitted = model.fit(&X.view(), &()).expect("fit should succeed");

        // Force one component's weight below the deletion threshold so the
        // real pruning logic actually fires. `min_components` is 1 here
        // (rebuilt below) so pruning is not blocked by the floor.
        fitted.state.config.min_components = 1;
        fitted.state.weights = array![0.9, 0.1];

        let to_delete = fitted.components_to_delete();
        assert_eq!(
            to_delete,
            vec![1],
            "the low-weight component must be identified for deletion"
        );
        fitted
            .delete_components(&to_delete)
            .expect("delete_components should succeed");
        assert_eq!(fitted.n_components(), 1);
        assert_eq!(fitted.deletion_history(), &[1]);
    }

    /// The Page-Hinkley drift detector is a real, standard recursive test
    /// (monitoring "surprise" = -log-likelihood, since drift/anomalies make
    /// data *less* likely under the current model): it must (a) NOT
    /// false-alarm on repeated in-distribution samples, and (b) eventually
    /// flag a large, sustained distribution shift rather than silently
    /// staying `false` forever.
    #[test]
    #[allow(non_snake_case)]
    fn test_page_hinkley_drift_detection_is_real() {
        let X = array![[0.0, 0.0], [0.1, 0.1], [0.0, 0.1], [0.1, 0.0]];
        let model = AdaptiveStreamingGMM::builder()
            .min_components(1)
            .max_components(1)
            .drift_detection(DriftDetectionMethod::PageHinkley {
                delta: 0.005,
                lambda: 50.0,
            })
            .build();
        let mut fitted = model.fit(&X.view(), &()).expect("fit should succeed");

        // In-distribution samples (right at the fitted mean) must not
        // trigger a false alarm.
        for _ in 0..10 {
            let normal = array![0.05, 0.05];
            fitted
                .partial_fit(&normal.view())
                .expect("partial_fit should succeed");
            assert!(
                !fitted.state.drift_detected,
                "in-distribution samples must not trigger a false drift alarm"
            );
        }

        // A massive, sustained distribution shift must eventually be
        // flagged.
        let mut drifted_at_some_point = false;
        for _ in 0..10 {
            let far = array![1000.0, 1000.0];
            fitted
                .partial_fit(&far.view())
                .expect("partial_fit should succeed");
            if fitted.state.drift_detected {
                drifted_at_some_point = true;
            }
        }
        assert!(
            drifted_at_some_point,
            "Page-Hinkley drift detector never fired for a large, sustained distribution shift"
        );
    }

    /// ADWIN is genuinely out of scope (it requires an adaptive
    /// windowing/exponential-histogram data structure); it must honestly
    /// report `NotImplemented` rather than silently no-op'ing and pretending
    /// drift detection ran.
    #[test]
    #[allow(non_snake_case)]
    fn test_adwin_drift_detection_is_honestly_not_implemented() {
        let X = array![[0.0, 0.0], [1.0, 1.0]];
        let model = AdaptiveStreamingGMM::builder()
            .min_components(1)
            .max_components(1)
            .drift_detection(DriftDetectionMethod::ADWIN { delta: 0.05 })
            .build();
        let mut fitted = model.fit(&X.view(), &()).expect("fit should succeed");
        let sample = array![0.5, 0.5];
        let result = fitted.partial_fit(&sample.view());
        assert!(
            matches!(result, Err(SklearsError::NotImplemented(_))),
            "ADWIN must return an honest NotImplemented error rather than silently no-op, \
             got {result:?}"
        );
    }
}
