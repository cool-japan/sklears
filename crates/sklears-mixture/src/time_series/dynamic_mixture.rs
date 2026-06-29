//! Dynamic mixture with time-varying component means.
//!
//! This model is a finite Gaussian mixture in which the mixing weights
//! `pi_k` and the full covariances `Sigma_k` are **global** (shared across
//! the whole sequence), but the component **means evolve over time**,
//! `mu[t, k, :]`, according to a chosen [`ParameterEvolution`] prior. The
//! parameters are estimated by Expectation-Maximization over a single
//! ordered observation sequence `(n_timesteps, n_features)`.
//!
//! # Temporal smoother
//!
//! The defining feature of the model is the way the per-time component means
//! are re-estimated in the M-step. A plain weighted average would collapse to
//! a static estimate; instead, the means are produced by a **responsibility-
//! gated temporal smoother** that realizes the requested evolution prior. For
//! a fixed component `k` with per-time responsibilities `r[t, k]` and targets
//! `x_t`, the smoother performs an online state-space update of a latent level
//! `mu_{t, k}`:
//!
//! * [`ParameterEvolution::LocalLevel`] — exponential smoothing gated by the
//!   responsibility, `mu_t = mu_{t-1} + alpha * r[t, k] * (x_t - mu_{t-1})`,
//!   seeded at the global component mean `gbar_k`. A symmetric backward pass is
//!   averaged with the forward pass to remove the phase lag inherent to a
//!   causal filter (a fixed-interval-style smoother), which keeps the recovered
//!   trajectory centred on the data rather than trailing it.
//! * [`ParameterEvolution::RandomWalk`] — the same gated exponential smoother
//!   with a deliberately small effective `alpha`, encoding the heavy temporal
//!   smoothing of a random-walk state prior `mu_t ~ N(mu_{t-1}, .)`. Forward and
//!   backward passes are averaged as above.
//! * [`ParameterEvolution::AR1`] — a first-order autoregressive pull toward the
//!   global component mean, `mu_t = phi * mu_{t-1} + (1 - phi) * gbar_k`, with a
//!   responsibility-gated nudge toward the data,
//!   `+ alpha * r[t, k] * (x_t - mu_{t-1})`. The AR(1) term mean-reverts the
//!   trajectory to `gbar_k`, so a single causal pass already yields a centred,
//!   smooth path.
//!
//! In every case `gbar_k = sum_t r[t, k] x_t / sum_t r[t, k]` is the global,
//! time-averaged component mean used both as the smoother's initial level / AR
//! anchor and as the fallback static mean for scoring sequences whose length
//! does not match the training horizon.

use super::{logsumexp, REG_COVAR};
use crate::common::gaussian_log_pdf;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// First-order autoregressive coefficient used by [`ParameterEvolution::AR1`].
///
/// Controls how strongly the per-time mean mean-reverts toward the global
/// component mean `gbar_k`; a value near one yields slow, smooth drift.
const AR1_PHI: f64 = 0.9;

/// Multiplier applied to the smoothing strength under
/// [`ParameterEvolution::RandomWalk`] to obtain the heavier smoothing implied by
/// a random-walk state prior relative to the local-level process.
const RANDOM_WALK_DAMPING: f64 = 0.3;

/// Parameter evolution processes for dynamic mixtures.
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterEvolution {
    /// Component means follow a random walk (heavy temporal smoothing).
    RandomWalk,
    /// Component means follow a first-order autoregressive process that
    /// mean-reverts toward the global component mean.
    AR1,
    /// Component means follow a local-level (exponential smoothing) process.
    LocalLevel,
}

/// Configuration for a [`DynamicMixture`].
#[derive(Debug, Clone)]
pub struct DynamicMixtureConfig {
    /// Number of mixture components.
    pub n_components: usize,
    /// Temporal evolution prior governing how the component means change.
    pub evolution: ParameterEvolution,
    /// Maximum number of EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the per-iteration log-likelihood change.
    pub tol: f64,
    /// Smoothing strength of the responsibility-gated temporal smoother.
    ///
    /// Interpreted as the base exponential-smoothing factor `alpha in (0, 1]`
    /// for the local-level / random-walk processes and as the data-nudge gain
    /// for the AR(1) process. Larger values track the data more closely;
    /// smaller values impose stronger temporal smoothing.
    pub smoothing: f64,
}

impl Default for DynamicMixtureConfig {
    fn default() -> Self {
        Self {
            n_components: 1,
            evolution: ParameterEvolution::LocalLevel,
            max_iter: 100,
            tol: 1e-4,
            smoothing: 0.3,
        }
    }
}

/// Error type for dynamic mixture operations.
#[derive(Debug, thiserror::Error)]
pub enum DynamicMixtureError {
    /// The requested number of components is invalid (zero, or larger than the
    /// sequence length).
    #[error("Invalid number of components: {0}")]
    InvalidComponents(usize),
    /// The observation sequence is empty, malformed, or has the wrong feature
    /// count for the fitted model.
    #[error("Invalid observation sequence")]
    InvalidObservation,
    /// A numerical failure occurred (e.g. a non-positive-definite covariance).
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Per-timestep, per-component Gaussian emission log-probabilities using the
/// supplied time-varying means and global covariances.
///
/// `means[t]` is the `(n_components, n_features)` mean matrix at timestep `t`.
fn emission_log_probs(
    observations: &ArrayView2<f64>,
    means: &[Array2<f64>],
    covariances: &[Array2<f64>],
) -> Result<Array2<f64>, DynamicMixtureError> {
    let n_obs = observations.nrows();
    let n_components = covariances.len();
    let mut log_b = Array2::<f64>::zeros((n_obs, n_components));
    for t in 0..n_obs {
        let x = observations.row(t);
        let mean_t = &means[t];
        for k in 0..n_components {
            log_b[[t, k]] = gaussian_log_pdf(&x, &mean_t.row(k), &covariances[k].view())
                .map_err(|e| DynamicMixtureError::NumericalError(e.to_string()))?;
        }
    }
    Ok(log_b)
}

/// Emission log-probabilities against a single static `(K, D)` mean matrix.
///
/// Used when scoring a sequence whose length does not match the training
/// horizon, in which case the time-averaged component means are reused at
/// every timestep.
fn emission_log_probs_static(
    observations: &ArrayView2<f64>,
    means: &Array2<f64>,
    covariances: &[Array2<f64>],
) -> Result<Array2<f64>, DynamicMixtureError> {
    let n_obs = observations.nrows();
    let n_components = covariances.len();
    let mut log_b = Array2::<f64>::zeros((n_obs, n_components));
    for t in 0..n_obs {
        let x = observations.row(t);
        for k in 0..n_components {
            log_b[[t, k]] = gaussian_log_pdf(&x, &means.row(k), &covariances[k].view())
                .map_err(|e| DynamicMixtureError::NumericalError(e.to_string()))?;
        }
    }
    Ok(log_b)
}

/// Row-normalize emission log-probabilities plus log-weights into
/// responsibilities and return `(responsibilities, log_likelihood)`.
fn responsibilities_from_log_b(
    log_weights: &Array1<f64>,
    log_b: &Array2<f64>,
) -> (Array2<f64>, f64) {
    let (n_obs, n_components) = log_b.dim();
    let mut resp = Array2::<f64>::zeros((n_obs, n_components));
    let mut scratch = vec![0.0_f64; n_components];
    let mut ll = 0.0_f64;
    for t in 0..n_obs {
        for k in 0..n_components {
            scratch[k] = log_weights[k] + log_b[[t, k]];
        }
        let norm = logsumexp(&scratch);
        ll += norm;
        for k in 0..n_components {
            resp[[t, k]] = (scratch[k] - norm).exp();
        }
    }
    (resp, ll)
}

/// Compute the global, time-averaged mean of a single component,
/// `gbar_k = sum_t r[t, k] x_t / sum_t r[t, k]`.
///
/// Falls back to `fallback` (the previous estimate) when the component has
/// negligible total responsibility.
fn global_component_mean(
    observations: &ArrayView2<f64>,
    resp: &Array2<f64>,
    k: usize,
    fallback: &ArrayView1<f64>,
) -> Array1<f64> {
    let n_obs = observations.nrows();
    let n_features = observations.ncols();
    let weight: f64 = (0..n_obs).map(|t| resp[[t, k]]).sum();
    if weight <= f64::EPSILON {
        return fallback.to_owned();
    }
    let mut mean = Array1::<f64>::zeros(n_features);
    for t in 0..n_obs {
        let r = resp[[t, k]];
        for d in 0..n_features {
            mean[d] += r * observations[[t, d]];
        }
    }
    mean.mapv_into(|v| v / weight)
}

/// Responsibility-gated exponential (local-level) smoother for one component.
///
/// Runs a causal forward pass `level_t = level_{t-1} + alpha * r[t] *
/// (x_t - level_{t-1})` and an anti-causal backward pass with the same update,
/// then averages the two so the recovered trajectory is centred on the data
/// rather than lagging it. `anchor` seeds both passes (the global component
/// mean `gbar_k`).
fn smooth_local_level(
    observations: &ArrayView2<f64>,
    resp: &Array2<f64>,
    k: usize,
    anchor: &ArrayView1<f64>,
    alpha: f64,
    out: &mut [Array2<f64>],
) {
    let n_obs = observations.nrows();
    let n_features = observations.ncols();

    let mut forward = vec![Array1::<f64>::zeros(n_features); n_obs];
    let mut level = anchor.to_owned();
    for t in 0..n_obs {
        let gain = alpha * resp[[t, k]];
        for d in 0..n_features {
            level[d] += gain * (observations[[t, d]] - level[d]);
        }
        forward[t].assign(&level);
    }

    let mut backward = vec![Array1::<f64>::zeros(n_features); n_obs];
    let mut level_b = anchor.to_owned();
    for t in (0..n_obs).rev() {
        let gain = alpha * resp[[t, k]];
        for d in 0..n_features {
            level_b[d] += gain * (observations[[t, d]] - level_b[d]);
        }
        backward[t].assign(&level_b);
    }

    for t in 0..n_obs {
        for d in 0..n_features {
            out[t][[k, d]] = 0.5 * (forward[t][d] + backward[t][d]);
        }
    }
}

/// Responsibility-gated AR(1) smoother for one component.
///
/// A single causal pass `mu_t = phi * mu_{t-1} + (1 - phi) * gbar_k +
/// alpha * r[t] * (x_t - mu_{t-1})`. The autoregressive term mean-reverts the
/// trajectory toward the global component mean `anchor`, while the gated
/// residual nudges it toward the observed data where the component is active.
fn smooth_ar1(
    observations: &ArrayView2<f64>,
    resp: &Array2<f64>,
    k: usize,
    anchor: &ArrayView1<f64>,
    alpha: f64,
    out: &mut [Array2<f64>],
) {
    let n_obs = observations.nrows();
    let n_features = observations.ncols();
    let mut state = anchor.to_owned();
    for t in 0..n_obs {
        let gain = alpha * resp[[t, k]];
        for d in 0..n_features {
            let reverted = AR1_PHI * state[d] + (1.0 - AR1_PHI) * anchor[d];
            let nudged = reverted + gain * (observations[[t, d]] - state[d]);
            state[d] = nudged;
            out[t][[k, d]] = nudged;
        }
    }
}

/// Apply the chosen evolution prior as a temporal smoother for component `k`,
/// writing the smoothed per-time means into `out`.
fn smooth_component(
    evolution: &ParameterEvolution,
    observations: &ArrayView2<f64>,
    resp: &Array2<f64>,
    k: usize,
    anchor: &ArrayView1<f64>,
    smoothing: f64,
    out: &mut [Array2<f64>],
) {
    match evolution {
        ParameterEvolution::LocalLevel => {
            smooth_local_level(observations, resp, k, anchor, smoothing, out);
        }
        ParameterEvolution::RandomWalk => {
            let damped = (smoothing * RANDOM_WALK_DAMPING).clamp(f64::EPSILON, 1.0);
            smooth_local_level(observations, resp, k, anchor, damped, out);
        }
        ParameterEvolution::AR1 => {
            smooth_ar1(observations, resp, k, anchor, smoothing, out);
        }
    }
}

/// Re-estimate the global covariance of component `k` using the time-varying
/// means, `Sigma_k = sum_t r[t, k] (x_t - mu[t, k])(x_t - mu[t, k])^T /
/// sum_t r[t, k]`, with a diagonal regularizer added for positive-definiteness.
fn update_covariance(
    observations: &ArrayView2<f64>,
    resp: &Array2<f64>,
    means: &[Array2<f64>],
    k: usize,
    fallback: &Array2<f64>,
) -> Array2<f64> {
    let n_obs = observations.nrows();
    let n_features = observations.ncols();
    let weight: f64 = (0..n_obs).map(|t| resp[[t, k]]).sum();
    if weight <= f64::EPSILON {
        return fallback.clone();
    }
    let mut cov = Array2::<f64>::zeros((n_features, n_features));
    for t in 0..n_obs {
        let r = resp[[t, k]];
        if r <= 0.0 {
            continue;
        }
        for a in 0..n_features {
            let da = observations[[t, a]] - means[t][[k, a]];
            for b in 0..n_features {
                let db = observations[[t, b]] - means[t][[k, b]];
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
    cov
}

/// Average the per-time component means over the training horizon, producing a
/// single static `(K, D)` summary matrix reused for off-horizon scoring.
fn average_means(means: &[Array2<f64>], n_components: usize, n_features: usize) -> Array2<f64> {
    let mut summary = Array2::<f64>::zeros((n_components, n_features));
    if means.is_empty() {
        return summary;
    }
    for mean_t in means {
        summary += mean_t;
    }
    summary.mapv_into(|v| v / means.len() as f64)
}

/// Dynamic Mixture Model (untrained).
///
/// A Gaussian mixture with global weights and covariances but time-varying
/// component means that evolve under a [`ParameterEvolution`] prior, fitted by
/// Expectation-Maximization. See the [module documentation](self) for the
/// responsibility-gated temporal smoother used in the M-step.
#[derive(Debug, Clone)]
pub struct DynamicMixture {
    config: DynamicMixtureConfig,
}

impl DynamicMixture {
    /// Create a new untrained dynamic mixture from the given configuration.
    pub fn new(config: DynamicMixtureConfig) -> Self {
        Self { config }
    }

    /// Access the configuration backing this model.
    pub fn config(&self) -> &DynamicMixtureConfig {
        &self.config
    }

    /// Fit the model to a `(n_timesteps, n_features)` observation sequence via EM.
    pub fn fit(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<DynamicMixtureTrained, DynamicMixtureError> {
        let n_components = self.config.n_components;
        let n_obs = observations.nrows();
        let n_features = observations.ncols();

        if n_components == 0 {
            return Err(DynamicMixtureError::InvalidComponents(0));
        }
        if n_obs == 0 || n_features == 0 || n_obs < n_components {
            return Err(DynamicMixtureError::InvalidObservation);
        }

        let smoothing = self.config.smoothing.clamp(f64::EPSILON, 1.0);

        // --- Deterministic initialization ---
        // Component means: K observations spread evenly across the sequence,
        // held constant across time for the initial parameter set.
        let mut init_mean = Array2::<f64>::zeros((n_components, n_features));
        for k in 0..n_components {
            let idx = k * (n_obs - 1) / (n_components - 1).max(1);
            init_mean.row_mut(k).assign(&observations.row(idx));
        }
        let mut means: Vec<Array2<f64>> = vec![init_mean.clone(); n_obs];

        // Global covariances: the global diagonal variance shared by every
        // component, regularized for positive-definiteness.
        let global_mean = observations
            .mean_axis(Axis(0))
            .ok_or(DynamicMixtureError::InvalidObservation)?;
        let mut global_var = Array1::<f64>::zeros(n_features);
        for row in observations.rows() {
            for d in 0..n_features {
                let diff = row[d] - global_mean[d];
                global_var[d] += diff * diff;
            }
        }
        for d in 0..n_features {
            global_var[d] = global_var[d] / n_obs as f64 + REG_COVAR;
        }
        let mut covariances: Vec<Array2<f64>> = (0..n_components)
            .map(|_| Array2::from_diag(&global_var))
            .collect();

        // Global mixing weights: uniform.
        let mut weights = Array1::from_elem(n_components, 1.0 / n_components as f64);

        let mut prev_ll = f64::NEG_INFINITY;
        let mut final_ll = f64::NEG_INFINITY;

        for _iter in 0..self.config.max_iter {
            // --- E-step: responsibilities r[t, k] and log-likelihood. ---
            let log_weights = weights.mapv(|w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY });
            let log_b = emission_log_probs(observations, &means, &covariances)?;
            let (resp, ll) = responsibilities_from_log_b(&log_weights, &log_b);
            final_ll = ll;

            // --- M-step ---
            // Global mixing weights pi_k = (sum_t r[t, k]) / T.
            for k in 0..n_components {
                let mass: f64 = (0..n_obs).map(|t| resp[[t, k]]).sum();
                weights[k] = mass / n_obs as f64;
            }

            // Re-estimate the time-varying means with the responsibility-gated
            // temporal smoother realizing the chosen evolution prior. The
            // global component mean `gbar_k` anchors and seeds the smoother.
            let mut new_means: Vec<Array2<f64>> =
                vec![Array2::<f64>::zeros((n_components, n_features)); n_obs];
            let mut gbars: Vec<Array1<f64>> = Vec::with_capacity(n_components);
            for k in 0..n_components {
                let fallback = means[0].row(k).to_owned();
                let gbar = global_component_mean(observations, &resp, k, &fallback.view());
                smooth_component(
                    &self.config.evolution,
                    observations,
                    &resp,
                    k,
                    &gbar.view(),
                    smoothing,
                    &mut new_means,
                );
                gbars.push(gbar);
            }
            means = new_means;

            // Global covariances using the freshly smoothed time-varying means.
            covariances = (0..n_components)
                .map(|k| update_covariance(observations, &resp, &means, k, &covariances[k]))
                .collect();

            if (ll - prev_ll).abs() < self.config.tol {
                break;
            }
            prev_ll = ll;
        }

        // EM reaching the iteration cap without hitting `tol` is reported as the
        // final fitted state rather than an error (matching scikit-learn).
        let mean_summary = average_means(&means, n_components, n_features);

        Ok(DynamicMixtureTrained {
            means,
            mean_summary,
            covariances,
            weights,
            evolution: self.config.evolution.clone(),
            log_likelihood: final_ll,
            n_components,
            n_features,
        })
    }
}

/// A fitted Dynamic Mixture Model.
///
/// Holds the global mixing weights and covariances together with the full
/// time-varying component means `means[t]` (each `(n_components, n_features)`).
#[derive(Debug, Clone)]
pub struct DynamicMixtureTrained {
    means: Vec<Array2<f64>>,
    mean_summary: Array2<f64>,
    covariances: Vec<Array2<f64>>,
    weights: Array1<f64>,
    evolution: ParameterEvolution,
    log_likelihood: f64,
    n_components: usize,
    n_features: usize,
}

impl DynamicMixtureTrained {
    /// Global mixing weights `pi_k` (length `n_components`, summing to one).
    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    /// Per-component global covariance matrices (`n_components` matrices of
    /// shape `(n_features, n_features)`).
    pub fn covariances(&self) -> &[Array2<f64>] {
        &self.covariances
    }

    /// Number of mixture components.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Number of observation features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Temporal evolution prior used to fit the time-varying means.
    pub fn evolution(&self) -> &ParameterEvolution {
        &self.evolution
    }

    /// Training-set log-likelihood at the final EM iteration.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Number of timesteps in the training horizon.
    pub fn n_timesteps(&self) -> usize {
        self.means.len()
    }

    /// Full time-varying component means, one `(n_components, n_features)`
    /// matrix per training timestep.
    pub fn means(&self) -> &[Array2<f64>] {
        &self.means
    }

    /// The `(n_components, n_features)` component means at timestep `t`, or
    /// `None` if `t` is outside the training horizon.
    pub fn mean_at(&self, t: usize) -> Option<&Array2<f64>> {
        self.means.get(t)
    }

    /// Time-averaged component means (`n_components x n_features`), used as the
    /// static fallback when scoring off-horizon sequences.
    pub fn mean_summary(&self) -> &Array2<f64> {
        &self.mean_summary
    }

    /// Posterior component responsibilities `r[t, k]` (rows sum to one).
    ///
    /// When the sequence length matches the training horizon the time-varying
    /// means are used at each timestep; otherwise the time-averaged means
    /// ([`mean_summary`](Self::mean_summary)) are reused for every timestep.
    pub fn predict_proba(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<Array2<f64>, DynamicMixtureError> {
        if observations.ncols() != self.n_features || observations.nrows() == 0 {
            return Err(DynamicMixtureError::InvalidObservation);
        }
        let log_weights = self
            .weights
            .mapv(|w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY });
        let log_b = if observations.nrows() == self.means.len() {
            emission_log_probs(observations, &self.means, &self.covariances)?
        } else {
            emission_log_probs_static(observations, &self.mean_summary, &self.covariances)?
        };
        let (resp, _ll) = responsibilities_from_log_b(&log_weights, &log_b);
        Ok(resp)
    }

    /// Hard component assignment (arg-max responsibility) for each timestep.
    pub fn predict(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<Array1<usize>, DynamicMixtureError> {
        let resp = self.predict_proba(observations)?;
        let n_obs = resp.nrows();
        let mut labels = Array1::<usize>::zeros(n_obs);
        for t in 0..n_obs {
            let mut best = f64::NEG_INFINITY;
            let mut best_k = 0;
            for k in 0..self.n_components {
                let value = resp[[t, k]];
                if value > best {
                    best = value;
                    best_k = k;
                }
            }
            labels[t] = best_k;
        }
        Ok(labels)
    }

    /// Total log-likelihood of an observation sequence under the fitted model.
    ///
    /// Uses the time-varying means when the sequence length matches the
    /// training horizon, otherwise the time-averaged means.
    pub fn score(&self, observations: &ArrayView2<f64>) -> Result<f64, DynamicMixtureError> {
        if observations.ncols() != self.n_features || observations.nrows() == 0 {
            return Err(DynamicMixtureError::InvalidObservation);
        }
        let log_weights = self
            .weights
            .mapv(|w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY });
        let log_b = if observations.nrows() == self.means.len() {
            emission_log_probs(observations, &self.means, &self.covariances)?
        } else {
            emission_log_probs_static(observations, &self.mean_summary, &self.covariances)?
        };
        let (_resp, ll) = responsibilities_from_log_b(&log_weights, &log_b);
        Ok(ll)
    }
}

/// Builder for [`DynamicMixture`].
#[derive(Debug, Clone)]
pub struct DynamicMixtureBuilder {
    n_components: usize,
    evolution: ParameterEvolution,
    max_iter: usize,
    tol: f64,
    smoothing: f64,
}

impl DynamicMixtureBuilder {
    /// Start configuring a dynamic mixture with the given number of components.
    pub fn new(n_components: usize) -> Self {
        let defaults = DynamicMixtureConfig::default();
        Self {
            n_components,
            evolution: defaults.evolution,
            max_iter: defaults.max_iter,
            tol: defaults.tol,
            smoothing: defaults.smoothing,
        }
    }

    /// Set the temporal evolution prior for the component means
    /// (default [`ParameterEvolution::LocalLevel`]).
    pub fn evolution(mut self, evolution: ParameterEvolution) -> Self {
        self.evolution = evolution;
        self
    }

    /// Set the maximum number of EM iterations (default 100).
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the log-likelihood convergence tolerance (default 1e-4).
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the smoothing strength of the temporal smoother (default 0.3).
    pub fn smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing;
        self
    }

    /// Build the untrained model.
    pub fn build(self) -> DynamicMixture {
        DynamicMixture::new(DynamicMixtureConfig {
            n_components: self.n_components,
            evolution: self.evolution,
            max_iter: self.max_iter,
            tol: self.tol,
            smoothing: self.smoothing,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Single-component 1-D sequence whose mean drifts linearly from 0 to ~10.
    fn drifting_sequence() -> Array2<f64> {
        let mut obs = Array2::<f64>::zeros((40, 1));
        for t in 0..40 {
            let jitter = if t % 2 == 0 { 0.05 } else { -0.05 };
            obs[[t, 0]] = 0.25 * t as f64 + jitter;
        }
        obs
    }

    /// Two-component 1-D sequence: a low block near 0 then a high block near 8.
    fn two_component_sequence() -> Array2<f64> {
        let mut obs = Array2::<f64>::zeros((36, 1));
        for t in 0..36 {
            let base = if t < 18 { 0.0 } else { 8.0 };
            let jitter = if t % 2 == 0 { 0.1 } else { -0.1 };
            obs[[t, 0]] = base + jitter;
        }
        obs
    }

    #[test]
    fn test_dynamic_mean_tracks_drift() {
        let obs = drifting_sequence();
        let model = DynamicMixtureBuilder::new(1)
            .evolution(ParameterEvolution::LocalLevel)
            .max_iter(50)
            .tol(1e-6)
            .build();
        let trained = model
            .fit(&obs.view())
            .expect("dynamic mixture fit should succeed");

        let early = trained.mean_at(5).expect("timestep 5 within horizon")[[0, 0]];
        let late = trained.mean_at(35).expect("timestep 35 within horizon")[[0, 0]];
        assert!(
            late - early >= 3.0,
            "expected drift recovery: mean_at(35)={late} should exceed mean_at(5)={early} by >= 3.0"
        );

        let score = trained.score(&obs.view()).expect("score should succeed");
        assert!(score.is_finite(), "score should be finite, got {score}");

        let labels = trained
            .predict(&obs.view())
            .expect("predict should succeed");
        assert_eq!(labels.len(), 40);
        assert!(labels.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_dynamic_two_components_ar1_runs() {
        let obs = two_component_sequence();
        let model = DynamicMixtureBuilder::new(2)
            .evolution(ParameterEvolution::AR1)
            .max_iter(60)
            .tol(1e-6)
            .build();
        let trained = model.fit(&obs.view()).expect("AR1 fit should succeed");

        assert_eq!(trained.n_components(), 2);
        assert_eq!(*trained.evolution(), ParameterEvolution::AR1);

        let weight_sum: f64 = trained.weights().sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-9,
            "weights should sum to 1, got {weight_sum}"
        );

        let proba = trained
            .predict_proba(&obs.view())
            .expect("predict_proba should succeed");
        assert_eq!(proba.dim(), (obs.nrows(), 2));
        for t in 0..obs.nrows() {
            let row_sum: f64 = proba.row(t).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-9,
                "responsibility row {t} should sum to 1, got {row_sum}"
            );
        }
    }

    #[test]
    fn test_dynamic_rejects_bad_input() {
        let obs = array![[0.0_f64], [1.0], [2.0]];
        let zero_components = DynamicMixtureBuilder::new(0).build();
        assert!(matches!(
            zero_components.fit(&obs.view()),
            Err(DynamicMixtureError::InvalidComponents(0))
        ));

        let empty = Array2::<f64>::zeros((0, 1));
        let model = DynamicMixtureBuilder::new(1).build();
        assert!(matches!(
            model.fit(&empty.view()),
            Err(DynamicMixtureError::InvalidObservation)
        ));
    }
}
