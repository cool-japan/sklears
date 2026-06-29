//! Temporal Gaussian mixture with smoothly time-varying mixing weights.
//!
//! A Gaussian mixture in which the `K` component *means* `mu_k` and full
//! *covariances* `Sigma_k` are global (time-invariant), but the mixing weights
//! `pi[t, k]` evolve across time and are smoothed along the time axis with a
//! Gaussian kernel of bandwidth `h` (in timesteps). The model is fit by
//! Expectation-Maximization: the E-step computes per-timestep responsibilities
//! from the current time-varying weights and global emissions, and the M-step
//! re-estimates the global means/covariances from those responsibilities while
//! the time-varying weights are obtained by temporal kernel smoothing of the
//! responsibilities.
//!
//! This captures regime structure: when the data shifts from one cluster to
//! another over time, the corresponding mixing weight `pi[t, k]` rises and falls
//! smoothly across `t` rather than staying constant as in a static mixture.

use super::{logsumexp, REG_COVAR};
use crate::common::gaussian_log_pdf;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Configuration for a [`TemporalGaussianMixture`].
#[derive(Debug, Clone)]
pub struct TemporalMixtureConfig {
    /// Number of mixture components `K`.
    pub n_components: usize,
    /// Gaussian temporal-smoothing kernel bandwidth `h`, expressed in timesteps.
    pub bandwidth: f64,
    /// Maximum number of EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the per-iteration data log-likelihood change.
    pub tol: f64,
}

impl Default for TemporalMixtureConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            bandwidth: 3.0,
            max_iter: 100,
            tol: 1e-4,
        }
    }
}

/// Error type for temporal Gaussian mixture operations.
#[derive(Debug, thiserror::Error)]
pub enum TemporalMixtureError {
    /// The requested number of components is invalid (zero, or larger than the
    /// sequence length, so the deterministic initialization cannot pick that
    /// many distinct seed observations).
    #[error("Invalid number of components: {0}")]
    InvalidComponents(usize),
    /// The observation sequence is empty, malformed, or has the wrong feature count.
    #[error("Invalid observation sequence")]
    InvalidObservation,
    /// A numerical failure occurred (e.g. a non-positive-definite covariance).
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Evaluate the (unnormalized) Gaussian temporal kernel `Kh(d) = exp(-0.5 (d/h)^2)`.
fn temporal_kernel(distance: f64, bandwidth: f64) -> f64 {
    let scaled = distance / bandwidth;
    (-0.5 * scaled * scaled).exp()
}

/// Per-timestep, per-component Gaussian emission log-probabilities
/// `log N(x_t | mu_k, Sigma_k)` for the global means and covariances.
fn emission_log_probs(
    observations: &ArrayView2<f64>,
    means: &Array2<f64>,
    covariances: &[Array2<f64>],
) -> Result<Array2<f64>, TemporalMixtureError> {
    let n_obs = observations.nrows();
    let n_components = means.nrows();
    let mut log_emit = Array2::<f64>::zeros((n_obs, n_components));
    for t in 0..n_obs {
        let x = observations.row(t);
        for k in 0..n_components {
            log_emit[[t, k]] = gaussian_log_pdf(&x, &means.row(k), &covariances[k].view())
                .map_err(|e| TemporalMixtureError::NumericalError(e.to_string()))?;
        }
    }
    Ok(log_emit)
}

/// Compute responsibilities and the accumulated data log-likelihood from
/// per-timestep emission log-probabilities and per-timestep log mixing weights.
///
/// `log_weights[t, k] = ln(pi[t, k])`. Returns `(responsibilities, log_likelihood)`
/// where each row of `responsibilities` sums to one and
/// `log_likelihood = sum_t logsumexp_k(log_weights[t, k] + log_emit[t, k])`.
fn responsibilities_from_emissions(
    log_emit: &Array2<f64>,
    log_weights: &Array2<f64>,
) -> (Array2<f64>, f64) {
    let (n_obs, n_components) = log_emit.dim();
    let mut responsibilities = Array2::<f64>::zeros((n_obs, n_components));
    let mut log_likelihood = 0.0_f64;
    let mut scratch = vec![0.0_f64; n_components];
    for t in 0..n_obs {
        for k in 0..n_components {
            scratch[k] = log_weights[[t, k]] + log_emit[[t, k]];
        }
        let norm = logsumexp(&scratch);
        log_likelihood += norm;
        if norm.is_finite() {
            for k in 0..n_components {
                responsibilities[[t, k]] = (scratch[k] - norm).exp();
            }
        } else {
            // No component places any mass on this timestep; fall back to a
            // uniform assignment so the row still sums to one.
            let uniform = 1.0 / n_components as f64;
            for k in 0..n_components {
                responsibilities[[t, k]] = uniform;
            }
        }
    }
    (responsibilities, log_likelihood)
}

/// Smooth responsibilities along the time axis with a Gaussian kernel of the
/// given bandwidth, then renormalize each row so the weights sum to one over the
/// components. The kernel support is truncated to `|t - tau| <= ceil(3 h)` for
/// efficiency, which captures essentially all of the Gaussian mass.
fn smooth_weights(responsibilities: &Array2<f64>, bandwidth: f64) -> Array2<f64> {
    let (n_obs, n_components) = responsibilities.dim();
    let mut weights = Array2::<f64>::zeros((n_obs, n_components));
    let radius = (3.0 * bandwidth).ceil() as isize;
    for t in 0..n_obs {
        let mut acc = vec![0.0_f64; n_components];
        let mut kernel_sum = 0.0_f64;
        let lo = (t as isize - radius).max(0) as usize;
        let hi = ((t as isize + radius) as usize).min(n_obs - 1);
        for tau in lo..=hi {
            let distance = t as f64 - tau as f64;
            let kernel = temporal_kernel(distance, bandwidth);
            kernel_sum += kernel;
            for k in 0..n_components {
                acc[k] += kernel * responsibilities[[tau, k]];
            }
        }
        // Smoothed, then renormalized across components so each row sums to one.
        let mut row_sum = 0.0_f64;
        for value in acc.iter_mut() {
            *value = if kernel_sum > 0.0 {
                *value / kernel_sum
            } else {
                1.0 / n_components as f64
            };
            row_sum += *value;
        }
        for k in 0..n_components {
            weights[[t, k]] = if row_sum > 0.0 {
                acc[k] / row_sum
            } else {
                1.0 / n_components as f64
            };
        }
    }
    weights
}

/// Temporal Gaussian Mixture Model (untrained).
///
/// `K` Gaussian components with global means and full covariances and mixing
/// weights `pi[t, k]` that vary smoothly across time. Fit by EM with temporal
/// kernel smoothing of the responsibilities driving the weight update.
#[derive(Debug, Clone)]
pub struct TemporalGaussianMixture {
    config: TemporalMixtureConfig,
}

impl TemporalGaussianMixture {
    /// Create a new untrained temporal Gaussian mixture from a configuration.
    pub fn new(config: TemporalMixtureConfig) -> Self {
        Self { config }
    }

    /// Access the model configuration.
    pub fn config(&self) -> &TemporalMixtureConfig {
        &self.config
    }

    /// Fit the model to a `(n_timesteps, n_features)` observation sequence via EM.
    ///
    /// Means are initialized to `K` observations spread evenly across the
    /// sequence, covariances to the global per-feature variance (plus
    /// regularization), and the time-varying weights to the uniform `1/K`.
    pub fn fit(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<TemporalGaussianMixtureTrained, TemporalMixtureError> {
        let n_components = self.config.n_components;
        let n_obs = observations.nrows();
        let n_features = observations.ncols();

        if n_components == 0 {
            return Err(TemporalMixtureError::InvalidComponents(0));
        }
        if n_obs == 0 || n_features == 0 {
            return Err(TemporalMixtureError::InvalidObservation);
        }
        if n_obs < n_components {
            return Err(TemporalMixtureError::InvalidComponents(n_components));
        }
        if !self.config.bandwidth.is_finite() || self.config.bandwidth <= 0.0 {
            return Err(TemporalMixtureError::NumericalError(
                "bandwidth must be a finite, strictly positive number".to_string(),
            ));
        }

        // --- Deterministic initialization ---
        // Means: K observations spread evenly across the sequence.
        let mut means = Array2::<f64>::zeros((n_components, n_features));
        for k in 0..n_components {
            let idx = k * (n_obs - 1) / (n_components - 1).max(1);
            means.row_mut(k).assign(&observations.row(idx));
        }
        // Covariances: the global per-feature variance shared by every component,
        // stored as a diagonal matrix with the regularization on the diagonal.
        let global_mean = observations
            .mean_axis(Axis(0))
            .ok_or(TemporalMixtureError::InvalidObservation)?;
        let mut global_var = Array1::<f64>::zeros(n_features);
        for row in observations.rows() {
            for k in 0..n_features {
                let diff = row[k] - global_mean[k];
                global_var[k] += diff * diff;
            }
        }
        for k in 0..n_features {
            global_var[k] = global_var[k] / n_obs as f64 + REG_COVAR;
        }
        let mut covariances: Vec<Array2<f64>> = (0..n_components)
            .map(|_| Array2::from_diag(&global_var))
            .collect();
        // Time-varying weights: uniform over components for every timestep.
        let mut weights = Array2::from_elem((n_obs, n_components), 1.0 / n_components as f64);

        let mut prev_ll = f64::NEG_INFINITY;
        let mut final_ll = f64::NEG_INFINITY;

        for _iter in 0..self.config.max_iter {
            // --- E-step ---
            let log_emit = emission_log_probs(observations, &means, &covariances)?;
            let log_weights = weights.mapv(|w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY });
            let (responsibilities, ll) = responsibilities_from_emissions(&log_emit, &log_weights);
            final_ll = ll;

            // --- M-step ---
            // Global means and full covariances from the responsibilities.
            for k in 0..n_components {
                let weight: f64 = (0..n_obs).map(|t| responsibilities[[t, k]]).sum();
                if weight <= 0.0 {
                    continue;
                }
                let mut mean = Array1::<f64>::zeros(n_features);
                for t in 0..n_obs {
                    let r = responsibilities[[t, k]];
                    for d in 0..n_features {
                        mean[d] += r * observations[[t, d]];
                    }
                }
                for d in 0..n_features {
                    mean[d] /= weight;
                }
                let mut cov = Array2::<f64>::zeros((n_features, n_features));
                for t in 0..n_obs {
                    let r = responsibilities[[t, k]];
                    for a in 0..n_features {
                        let da = observations[[t, a]] - mean[a];
                        for b in 0..n_features {
                            let db = observations[[t, b]] - mean[b];
                            cov[[a, b]] += r * da * db;
                        }
                    }
                }
                for a in 0..n_features {
                    for b in 0..n_features {
                        cov[[a, b]] /= weight;
                    }
                }
                for d in 0..n_features {
                    cov[[d, d]] += REG_COVAR;
                }
                means.row_mut(k).assign(&mean);
                covariances[k] = cov;
            }

            // Time-varying mixing weights via temporal kernel smoothing of the
            // responsibilities, renormalized so each row sums to one.
            weights = smooth_weights(&responsibilities, self.config.bandwidth);

            if (ll - prev_ll).abs() < self.config.tol {
                break;
            }
            prev_ll = ll;
        }

        Ok(TemporalGaussianMixtureTrained {
            means,
            covariances,
            weights,
            log_likelihood: final_ll,
            n_components,
            n_features,
            bandwidth: self.config.bandwidth,
        })
    }
}

/// A fitted temporal Gaussian mixture.
///
/// Holds the global component means and covariances, the `(T, K)` matrix of
/// time-varying mixing weights estimated during training, and the training data
/// log-likelihood.
#[derive(Debug, Clone)]
pub struct TemporalGaussianMixtureTrained {
    means: Array2<f64>,
    covariances: Vec<Array2<f64>>,
    weights: Array2<f64>,
    log_likelihood: f64,
    n_components: usize,
    n_features: usize,
    bandwidth: f64,
}

impl TemporalGaussianMixtureTrained {
    /// Global component means (`K x D`).
    pub fn means(&self) -> &Array2<f64> {
        &self.means
    }

    /// Global component covariance matrices (length `K`, each `D x D`).
    pub fn covariances(&self) -> &[Array2<f64>] {
        &self.covariances
    }

    /// Time-varying mixing weights estimated during training (`T x K`); each row
    /// sums to one.
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Number of mixture components `K`.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Number of features `D`.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Temporal-smoothing kernel bandwidth used during training.
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// Training-set data log-likelihood at the final EM iteration.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Average mixing-weight vector over the training timesteps (length `K`,
    /// sums to one). Used as the fallback weighting for inputs whose length does
    /// not match the training horizon `T`.
    fn average_weights(&self) -> Array1<f64> {
        let n_components = self.n_components;
        let n_obs = self.weights.nrows();
        if n_obs == 0 {
            return Array1::from_elem(n_components, 1.0 / n_components as f64);
        }
        let mut avg = Array1::<f64>::zeros(n_components);
        for t in 0..n_obs {
            for k in 0..n_components {
                avg[k] += self.weights[[t, k]];
            }
        }
        let mut sum = 0.0_f64;
        for k in 0..n_components {
            avg[k] /= n_obs as f64;
            sum += avg[k];
        }
        if sum > 0.0 {
            for k in 0..n_components {
                avg[k] /= sum;
            }
        } else {
            for k in 0..n_components {
                avg[k] = 1.0 / n_components as f64;
            }
        }
        avg
    }

    /// Build the per-timestep log mixing weights for an input of `n_obs`
    /// timesteps. When `n_obs == T` (the training horizon) the per-timestep
    /// training weights are used directly; otherwise every timestep is assigned
    /// the average weight vector.
    fn log_weights_for(&self, n_obs: usize) -> Array2<f64> {
        let n_components = self.n_components;
        if n_obs == self.weights.nrows() {
            self.weights
                .mapv(|w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY })
        } else {
            let avg = self.average_weights();
            let mut log_weights = Array2::<f64>::zeros((n_obs, n_components));
            for t in 0..n_obs {
                for k in 0..n_components {
                    log_weights[[t, k]] = if avg[k] > 0.0 {
                        avg[k].ln()
                    } else {
                        f64::NEG_INFINITY
                    };
                }
            }
            log_weights
        }
    }

    /// Validate that an observation matrix matches the trained feature count and
    /// is non-empty.
    fn validate(&self, observations: &ArrayView2<f64>) -> Result<(), TemporalMixtureError> {
        if observations.ncols() != self.n_features || observations.nrows() == 0 {
            return Err(TemporalMixtureError::InvalidObservation);
        }
        Ok(())
    }

    /// Posterior responsibilities `r[t, k]` for each timestep (`T x K`, rows sum
    /// to one).
    ///
    /// When the number of input timesteps equals the training horizon `T`, the
    /// per-timestep training weights are used so the regime structure learned at
    /// fit time is reflected. Otherwise the average training weight vector is
    /// applied uniformly across all timesteps.
    pub fn predict_proba(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<Array2<f64>, TemporalMixtureError> {
        self.validate(observations)?;
        let log_emit = emission_log_probs(observations, &self.means, &self.covariances)?;
        let log_weights = self.log_weights_for(observations.nrows());
        let (responsibilities, _ll) = responsibilities_from_emissions(&log_emit, &log_weights);
        Ok(responsibilities)
    }

    /// Most likely component per timestep (`argmax_k r[t, k]`).
    pub fn predict(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<Array1<usize>, TemporalMixtureError> {
        let proba = self.predict_proba(observations)?;
        let n_obs = proba.nrows();
        let mut labels = Array1::<usize>::zeros(n_obs);
        for t in 0..n_obs {
            labels[t] = argmax_row(&proba.row(t));
        }
        Ok(labels)
    }

    /// Data log-likelihood of an observation sequence under the fitted model,
    /// using the same per-timestep / average weighting rule as [`Self::predict_proba`].
    pub fn score(&self, observations: &ArrayView2<f64>) -> Result<f64, TemporalMixtureError> {
        self.validate(observations)?;
        let log_emit = emission_log_probs(observations, &self.means, &self.covariances)?;
        let log_weights = self.log_weights_for(observations.nrows());
        let (_responsibilities, ll) = responsibilities_from_emissions(&log_emit, &log_weights);
        Ok(ll)
    }
}

/// Index of the maximum entry of a responsibility row.
fn argmax_row(row: &ArrayView1<f64>) -> usize {
    let mut best = f64::NEG_INFINITY;
    let mut best_k = 0;
    for (k, &value) in row.iter().enumerate() {
        if value > best {
            best = value;
            best_k = k;
        }
    }
    best_k
}

/// Builder for [`TemporalGaussianMixture`].
#[derive(Debug, Clone)]
pub struct TemporalGaussianMixtureBuilder {
    n_components: usize,
    bandwidth: f64,
    max_iter: usize,
    tol: f64,
}

impl TemporalGaussianMixtureBuilder {
    /// Start configuring a temporal Gaussian mixture with the given number of
    /// components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            bandwidth: 3.0,
            max_iter: 100,
            tol: 1e-4,
        }
    }

    /// Set the temporal-smoothing kernel bandwidth `h` in timesteps (default 3.0).
    pub fn bandwidth(mut self, bandwidth: f64) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Set the maximum number of EM iterations (default 100).
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the data log-likelihood convergence tolerance (default 1e-4).
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Build the untrained model.
    pub fn build(self) -> TemporalGaussianMixture {
        TemporalGaussianMixture::new(TemporalMixtureConfig {
            n_components: self.n_components,
            bandwidth: self.bandwidth,
            max_iter: self.max_iter,
            tol: self.tol,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Two-regime 1-D sequence of length 40: the first 20 points sit near
    /// component A (~0.0) and the last 20 near component B (~10.0), with a small
    /// deterministic jitter so the per-component covariances stay non-degenerate.
    fn two_regime_sequence() -> Array2<f64> {
        let mut obs = Array2::<f64>::zeros((40, 1));
        for t in 0..40 {
            let base = if t < 20 { 0.0 } else { 10.0 };
            let jitter = if t % 2 == 0 { 0.1 } else { -0.1 };
            obs[[t, 0]] = base + jitter;
        }
        obs
    }

    #[test]
    fn test_temporal_weights_shift_over_time() {
        let obs = two_regime_sequence();
        let model = TemporalGaussianMixtureBuilder::new(2)
            .bandwidth(3.0)
            .max_iter(100)
            .tol(1e-6)
            .build();
        let trained = model.fit(&obs.view()).expect("temporal fit should succeed");

        // The two global means should straddle the two regimes near 0 and 10.
        let m0 = trained.means()[[0, 0]];
        let m1 = trained.means()[[1, 0]];
        let (low_k, high_k) = if m0 < m1 { (0, 1) } else { (1, 0) };
        let low_mean = trained.means()[[low_k, 0]];
        let high_mean = trained.means()[[high_k, 0]];
        assert!(low_mean < 3.0, "low-regime mean was {low_mean}");
        assert!(high_mean > 7.0, "high-regime mean was {high_mean}");

        // Time-varying weights: shape (40, 2), every row sums to one.
        let weights = trained.weights();
        assert_eq!(weights.dim(), (40, 2));
        for t in 0..40 {
            let row_sum: f64 = weights.row(t).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "weight row {t} summed to {row_sum}"
            );
        }

        // The component near 0 carries more average weight over the first half,
        // and the component near 10 dominates over the second half.
        let avg = |k: usize, lo: usize, hi: usize| -> f64 {
            let mut acc = 0.0_f64;
            for t in lo..hi {
                acc += weights[[t, k]];
            }
            acc / (hi - lo) as f64
        };
        let low_first = avg(low_k, 0, 20);
        let low_last = avg(low_k, 20, 40);
        let high_first = avg(high_k, 0, 20);
        let high_last = avg(high_k, 20, 40);
        assert!(
            low_first > low_last,
            "low component weight should fall: first={low_first}, last={low_last}"
        );
        assert!(
            high_last > high_first,
            "high component weight should rise: first={high_first}, last={high_last}"
        );

        // The time-varying weights track the regime shift, so the argmax label
        // at the start differs from the end.
        let labels = trained
            .predict(&obs.view())
            .expect("predict should succeed");
        assert_eq!(labels.len(), 40);
        assert_ne!(labels[0], labels[39]);

        // The training-set score is finite.
        let score = trained.score(&obs.view()).expect("score should succeed");
        assert!(score.is_finite());
    }

    #[test]
    fn test_temporal_rows_normalized() {
        let obs = two_regime_sequence();
        let model = TemporalGaussianMixtureBuilder::new(2)
            .bandwidth(3.0)
            .max_iter(100)
            .tol(1e-6)
            .build();
        let trained = model.fit(&obs.view()).expect("temporal fit should succeed");

        let proba = trained
            .predict_proba(&obs.view())
            .expect("predict_proba should succeed");
        assert_eq!(proba.dim(), (40, 2));
        for t in 0..40 {
            let row_sum: f64 = proba.row(t).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "proba row {t} summed to {row_sum}"
            );
        }
    }

    #[test]
    fn test_temporal_rejects_bad_input() {
        // Zero components is rejected.
        let obs = array![[0.0_f64], [1.0], [2.0]];
        let zero = TemporalGaussianMixtureBuilder::new(0).build();
        assert!(matches!(
            zero.fit(&obs.view()),
            Err(TemporalMixtureError::InvalidComponents(0))
        ));

        // An empty observation matrix is rejected.
        let empty = Array2::<f64>::zeros((0, 1));
        let model = TemporalGaussianMixtureBuilder::new(2).build();
        assert!(matches!(
            model.fit(&empty.view()),
            Err(TemporalMixtureError::InvalidObservation)
        ));
    }
}
