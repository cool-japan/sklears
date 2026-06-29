//! Switching linear-Gaussian state-space model via the Interacting Multiple
//! Model (IMM) filter (Blom & Bar-Shalom, 1988).
//!
//! # Model
//!
//! A univariate (scalar-state) *local-level* model whose process-noise variance
//! switches between `K` regimes according to a first-order Markov chain. For a
//! single scalar observation stream `y_t`:
//!
//! ```text
//! state:        x_t = x_{t-1} + w_t,   w_t ~ N(0, q_k)   (regime-specific q_k)
//! observation:  y_t = x_t     + v_t,   v_t ~ N(0, r)     (shared r)
//! regime:       k_t ~ Markov(P),       P is K x K
//! ```
//!
//! The regimes differ only in their process noise `q_k` (e.g. a *calm* regime
//! with small `q` and a *volatile* regime with large `q`). The observation noise
//! `r` and the local-level dynamics are shared.
//!
//! # Multivariate observations
//!
//! When the observation matrix has more than one feature column, every feature
//! is treated as an **independent** scalar local-level stream that shares the
//! same regime sequence. Under conditional independence given the active regime,
//! the joint mode log-likelihood is the sum of the per-feature mode
//! log-likelihoods, so the regime posterior is well-defined. The single reported
//! filtered state at each timestep is the mean of the per-feature filtered
//! states. This keeps the public [`SwitchingFilterResult`] shape `T x K`
//! (regime probabilities) and length `T` (filtered states) regardless of the
//! feature count, while remaining a genuine multi-stream IMM filter.
//!
//! # IMM filter (one timestep, K modes)
//!
//! Per mode the filter maintains a filtered mean `m_k`, variance `p_k`, and mode
//! probability `mu_k`:
//!
//! 1. **Mixing.** `cbar_j = sum_i P[i,j] mu_i`,
//!    `w_{i|j} = P[i,j] mu_i / cbar_j`,
//!    `m0_j = sum_i w_{i|j} m_i`,
//!    `p0_j = sum_i w_{i|j} (p_i + (m_i - m0_j)^2)`.
//! 2. **Mode-matched Kalman step.** Predict `m_pred = m0_j`,
//!    `p_pred = p0_j + q_j`; innovation `innov = y_t - m_pred`,
//!    `s = p_pred + r`; gain `K = p_pred / s`; update
//!    `m_j = m_pred + K innov`, `p_j = (1 - K) p_pred`; mode log-likelihood
//!    `log L_j = -0.5 (ln 2pi + ln s + innov^2 / s)`.
//! 3. **Mode update.** `log mu_j = ln cbar_j + log L_j`, normalized with
//!    log-sum-exp.
//! 4. **Combination.** `m_t = sum_j mu_j m_j`,
//!    `p_t = sum_j mu_j (p_j + (m_j - m_t)^2)`.
//! 5. **Sequence log-likelihood.** accumulate `logsumexp_j(ln cbar_j + log L_j)`.
//!
//! # Fitting
//!
//! [`SwitchingStateSpaceModel::fit`] runs the IMM filter forward over the whole
//! sequence and performs a light, stable EM-style moment refinement of the noise
//! parameters across a fixed number of iterations (`config.max_iter`, capped):
//!
//! * The shared observation noise `r` is re-estimated from the mode-posterior
//!   weighted innovations using the identity `E[innov^2] = p_pred + r`, i.e.
//!   `r_hat = weighted_mean(innov^2 - p_pred)`, floored to stay positive.
//! * Each process noise `q_k` is re-estimated from the mode-`k`-posterior
//!   weighted squared state corrections `(m_k - m_pred_k)^2` (the realized
//!   innovation absorbed by the state), which is large for whichever regime is
//!   active during volatile stretches, floored to stay positive.
//!
//! Updates are damped (a convex blend of the previous and freshly estimated
//! values) for numerical stability. The transition matrix is held at its
//! high-self-persistence initialization; only the noise parameters are refined.
//! Initialization uses `q_k = var * 10^{k - (K-1)/2}` spanning small to large,
//! `r = 0.5 var`, uniform mode probabilities, and a `0.95`-diagonal transition
//! matrix.

use super::{logsumexp, safe_ln};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

/// `ln(2 pi)`, used in the Gaussian innovation log-likelihood.
const LN_2PI: f64 = 1.837_877_066_409_345_3;

/// Smallest variance a noise parameter is allowed to take during refinement.
const VARIANCE_FLOOR: f64 = 1e-8;

/// Damping factor applied to the EM moment updates (`new = (1 - d) old + d est`).
const REFINEMENT_DAMPING: f64 = 0.5;

/// Error type for switching state-space model operations.
#[derive(Debug, thiserror::Error)]
pub enum SSMError {
    /// The configuration is invalid (e.g. zero regimes or a non-scalar state dimension).
    #[error("Invalid switching state-space configuration: {0}")]
    InvalidConfig(String),
    /// The observation sequence is empty or malformed.
    #[error("Invalid observation sequence")]
    InvalidObservation,
    /// A numerical failure occurred during filtering or refinement.
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Configuration for a Switching State Space Model.
#[derive(Debug, Clone)]
pub struct SSMConfig {
    /// Number of switching regimes (`K`). Must be at least one.
    pub n_regimes: usize,
    /// Dimension of the latent state. The local-level model is scalar, so this
    /// must be `1`; it is validated and stored for forward compatibility.
    pub state_dim: usize,
    /// Maximum number of EM-style refinement iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the per-iteration log-likelihood change.
    pub tol: f64,
}

impl Default for SSMConfig {
    fn default() -> Self {
        Self {
            n_regimes: 2,
            state_dim: 1,
            max_iter: 20,
            tol: 1e-4,
        }
    }
}

/// Switching State Space Model (untrained).
///
/// Holds the configuration for a switching local-level model fitted with the
/// Interacting Multiple Model (IMM) filter. See the module documentation for the
/// full mathematical description.
#[derive(Debug, Clone)]
pub struct SwitchingStateSpaceModel {
    config: SSMConfig,
}

/// Output of running the IMM filter over an observation sequence.
#[derive(Debug, Clone)]
pub struct SwitchingFilterResult {
    /// Per-timestep regime posterior probabilities, shape `(T, K)`; each row
    /// sums to one.
    pub regime_probabilities: Array2<f64>,
    /// Per-timestep combined filtered state, length `T`.
    pub filtered_states: Array1<f64>,
    /// Sequence log-likelihood under the fitted parameters.
    pub log_likelihood: f64,
}

/// A fitted Switching State Space Model.
///
/// Stores the (fixed) transition matrix, the refined per-regime process noises
/// and shared observation noise, the initial mode probabilities, and the
/// training log-likelihood. All inference methods re-run the IMM filter with
/// these parameters, so the struct is cheap to clone and store.
#[derive(Debug, Clone)]
pub struct SwitchingStateSpaceModelTrained {
    transmat: Array2<f64>,
    process_noises: Array1<f64>,
    observation_noise: f64,
    initial_mode_probs: Array1<f64>,
    log_likelihood: f64,
    n_regimes: usize,
}

/// Builder for [`SwitchingStateSpaceModel`].
#[derive(Debug, Clone)]
pub struct SwitchingStateSpaceModelBuilder {
    n_regimes: usize,
    state_dim: usize,
    max_iter: usize,
    tol: f64,
}

/// Per-timestep diagnostics produced by one forward IMM pass, reused by the
/// refinement M-step to re-estimate the noise parameters.
struct FilterTrace {
    /// Regime posterior probabilities, shape `(T, K)`.
    regime_probabilities: Array2<f64>,
    /// Combined filtered state, length `T`.
    filtered_states: Array1<f64>,
    /// Mode-posterior weighted `sum_t innov^2` accumulated over modes and
    /// features (numerator of the `r` moment estimate).
    innov_sq_sum: f64,
    /// Mode-posterior weighted `sum_t p_pred` accumulated over modes and
    /// features (subtracted term of the `r` moment estimate).
    p_pred_sum: f64,
    /// Total posterior weight contributing to the `r` estimate.
    r_weight: f64,
    /// Per-mode posterior weighted `sum_t (m_j - m_pred_j)^2` over features.
    correction_sq_sum: Array1<f64>,
    /// Per-mode total posterior weight contributing to the `q_k` estimates.
    mode_weight: Array1<f64>,
    /// Sequence log-likelihood for this pass.
    log_likelihood: f64,
}

impl SwitchingStateSpaceModel {
    /// Create a new untrained model from the given configuration.
    pub fn new(config: SSMConfig) -> Self {
        Self { config }
    }

    /// Validate the configuration, returning a descriptive error if it is invalid.
    fn validate_config(&self) -> Result<(), SSMError> {
        if self.config.n_regimes == 0 {
            return Err(SSMError::InvalidConfig(
                "n_regimes must be at least 1".to_string(),
            ));
        }
        if self.config.state_dim != 1 {
            return Err(SSMError::InvalidConfig(format!(
                "the switching local-level model requires state_dim == 1, got {}",
                self.config.state_dim
            )));
        }
        Ok(())
    }

    /// Fit the model to a `(n_timesteps, n_features)` observation sequence.
    ///
    /// Runs the IMM filter forward and refines the shared observation noise `r`
    /// and the per-regime process noises `q_k` with damped EM-style moment
    /// updates (see the module documentation). Returns the fitted parameters and
    /// the final-pass sequence log-likelihood.
    pub fn fit(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<SwitchingStateSpaceModelTrained, SSMError> {
        self.validate_config()?;
        let n_obs = observations.nrows();
        let n_features = observations.ncols();
        if n_obs == 0 || n_features == 0 {
            return Err(SSMError::InvalidObservation);
        }

        let n_regimes = self.config.n_regimes;
        let data_var = data_variance(observations)?;

        // --- Deterministic initialization ---
        let transmat = init_transmat(n_regimes);
        let mut initial_mode_probs = Array1::from_elem(n_regimes, 1.0 / n_regimes as f64);
        let mut process_noises = init_process_noises(n_regimes, data_var);
        let mut observation_noise = (0.5 * data_var).max(VARIANCE_FLOOR);

        // --- EM-style refinement of the noise parameters ---
        let max_iter = self.config.max_iter.clamp(1, 50);
        let mut prev_ll = f64::NEG_INFINITY;
        let mut log_likelihood = f64::NEG_INFINITY;
        for _ in 0..max_iter {
            let trace = run_imm(
                observations,
                &transmat,
                &process_noises,
                observation_noise,
                &initial_mode_probs,
            )?;
            log_likelihood = trace.log_likelihood;

            // M-step: re-estimate the shared observation noise r.
            if trace.r_weight > 0.0 {
                let est_r = (trace.innov_sq_sum - trace.p_pred_sum) / trace.r_weight;
                let est_r = est_r.max(VARIANCE_FLOOR);
                observation_noise = blend(observation_noise, est_r);
            }

            // M-step: re-estimate each process noise q_k.
            for k in 0..n_regimes {
                if trace.mode_weight[k] > 0.0 {
                    let est_q =
                        (trace.correction_sq_sum[k] / trace.mode_weight[k]).max(VARIANCE_FLOOR);
                    process_noises[k] = blend(process_noises[k], est_q);
                }
            }

            // M-step: refresh the initial mode probabilities from the first
            // filtered posterior (a stable, in-sample estimate).
            for k in 0..n_regimes {
                initial_mode_probs[k] = trace.regime_probabilities[[0, k]];
            }
            normalize_in_place(&mut initial_mode_probs);

            if (log_likelihood - prev_ll).abs() < self.config.tol {
                break;
            }
            prev_ll = log_likelihood;
        }

        Ok(SwitchingStateSpaceModelTrained {
            transmat,
            process_noises,
            observation_noise,
            initial_mode_probs,
            log_likelihood,
            n_regimes,
        })
    }
}

impl SwitchingStateSpaceModelTrained {
    /// Number of switching regimes.
    pub fn n_regimes(&self) -> usize {
        self.n_regimes
    }

    /// The fitted per-regime process noise variances `q_k`.
    pub fn process_noises(&self) -> &Array1<f64> {
        &self.process_noises
    }

    /// The fitted shared observation noise variance `r`.
    pub fn observation_noise(&self) -> f64 {
        self.observation_noise
    }

    /// The (fixed) regime transition matrix `P`.
    pub fn transmat(&self) -> &Array2<f64> {
        &self.transmat
    }

    /// The training-time sequence log-likelihood.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Run the IMM filter on a new observation sequence, returning per-timestep
    /// regime posteriors, combined filtered states, and the sequence
    /// log-likelihood.
    pub fn filter(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<SwitchingFilterResult, SSMError> {
        if observations.nrows() == 0 || observations.ncols() == 0 {
            return Err(SSMError::InvalidObservation);
        }
        let trace = run_imm(
            observations,
            &self.transmat,
            &self.process_noises,
            self.observation_noise,
            &self.initial_mode_probs,
        )?;
        Ok(SwitchingFilterResult {
            regime_probabilities: trace.regime_probabilities,
            filtered_states: trace.filtered_states,
            log_likelihood: trace.log_likelihood,
        })
    }

    /// Most-likely regime per timestep (the argmax of each regime-posterior row).
    pub fn predict(&self, observations: &ArrayView2<f64>) -> Result<Array1<usize>, SSMError> {
        let result = self.filter(observations)?;
        let n_obs = result.regime_probabilities.nrows();
        let mut path = Array1::<usize>::zeros(n_obs);
        for t in 0..n_obs {
            let mut best = f64::NEG_INFINITY;
            let mut best_k = 0;
            for k in 0..self.n_regimes {
                let p = result.regime_probabilities[[t, k]];
                if p > best {
                    best = p;
                    best_k = k;
                }
            }
            path[t] = best_k;
        }
        Ok(path)
    }

    /// Sequence log-likelihood of the observations under the fitted model.
    pub fn score(&self, observations: &ArrayView2<f64>) -> Result<f64, SSMError> {
        Ok(self.filter(observations)?.log_likelihood)
    }
}

impl SwitchingStateSpaceModelBuilder {
    /// Start configuring a switching state-space model.
    ///
    /// Defaults: `max_iter = 20`, `tol = 1e-4`.
    pub fn new(n_regimes: usize, state_dim: usize) -> Self {
        Self {
            n_regimes,
            state_dim,
            max_iter: 20,
            tol: 1e-4,
        }
    }

    /// Set the maximum number of refinement iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance on the log-likelihood change.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Build the untrained model.
    pub fn build(self) -> SwitchingStateSpaceModel {
        SwitchingStateSpaceModel::new(SSMConfig {
            n_regimes: self.n_regimes,
            state_dim: self.state_dim,
            max_iter: self.max_iter,
            tol: self.tol,
        })
    }
}

/// Pooled per-feature sample variance of the observation matrix, floored to stay
/// strictly positive so it can seed the noise parameters.
fn data_variance(observations: &ArrayView2<f64>) -> Result<f64, SSMError> {
    let n_obs = observations.nrows();
    let n_features = observations.ncols();
    if n_obs == 0 || n_features == 0 {
        return Err(SSMError::InvalidObservation);
    }
    let mut total = 0.0;
    for k in 0..n_features {
        let mut mean = 0.0;
        for t in 0..n_obs {
            mean += observations[[t, k]];
        }
        mean /= n_obs as f64;
        let mut var = 0.0;
        for t in 0..n_obs {
            let diff = observations[[t, k]] - mean;
            var += diff * diff;
        }
        total += var / n_obs as f64;
    }
    let var = total / n_features as f64;
    if !var.is_finite() {
        return Err(SSMError::NumericalError(
            "non-finite data variance".to_string(),
        ));
    }
    Ok(var.max(VARIANCE_FLOOR))
}

/// Transition matrix with high self-persistence: `0.95` on the diagonal and the
/// remaining `0.05` spread uniformly over the off-diagonal entries. For a single
/// regime this is the `1 x 1` identity.
fn init_transmat(n_regimes: usize) -> Array2<f64> {
    if n_regimes == 1 {
        return Array2::from_elem((1, 1), 1.0);
    }
    let diag = 0.95;
    let off = (1.0 - diag) / (n_regimes - 1) as f64;
    let mut transmat = Array2::from_elem((n_regimes, n_regimes), off);
    for k in 0..n_regimes {
        transmat[[k, k]] = diag;
    }
    transmat
}

/// Process-noise variances spanning small to large multiples of the data
/// variance: `q_k = var * 10^{k - (K-1)/2}`, floored to stay positive.
fn init_process_noises(n_regimes: usize, data_var: f64) -> Array1<f64> {
    let center = (n_regimes as f64 - 1.0) / 2.0;
    let mut q = Array1::<f64>::zeros(n_regimes);
    for k in 0..n_regimes {
        let exponent = k as f64 - center;
        q[k] = (data_var * 10f64.powf(exponent)).max(VARIANCE_FLOOR);
    }
    q
}

/// Convex blend of a previous and a freshly estimated value, damped by
/// [`REFINEMENT_DAMPING`] for numerical stability.
fn blend(previous: f64, estimate: f64) -> f64 {
    (1.0 - REFINEMENT_DAMPING) * previous + REFINEMENT_DAMPING * estimate
}

/// Normalize a probability vector in place; falls back to a uniform vector if
/// the total mass is non-positive or non-finite.
fn normalize_in_place(probs: &mut Array1<f64>) {
    let sum: f64 = probs.iter().copied().sum();
    if sum > 0.0 && sum.is_finite() {
        probs.mapv_inplace(|p| p / sum);
    } else {
        let uniform = 1.0 / probs.len().max(1) as f64;
        probs.fill(uniform);
    }
}

/// Run the IMM filter forward over the whole sequence, returning the regime
/// posteriors, combined filtered states, sequence log-likelihood, and the
/// sufficient statistics needed by the refinement M-step.
///
/// Each feature column is filtered as an independent scalar local-level stream
/// sharing the same regime sequence; the per-feature mode log-likelihoods are
/// summed (joint likelihood under conditional independence) before the mode
/// probability update, and the combined filtered state is averaged across
/// features.
fn run_imm(
    observations: &ArrayView2<f64>,
    transmat: &Array2<f64>,
    process_noises: &Array1<f64>,
    observation_noise: f64,
    initial_mode_probs: &Array1<f64>,
) -> Result<FilterTrace, SSMError> {
    let n_obs = observations.nrows();
    let n_features = observations.ncols();
    let n_regimes = process_noises.len();
    if n_obs == 0 || n_features == 0 || n_regimes == 0 {
        return Err(SSMError::InvalidObservation);
    }
    if transmat.nrows() != n_regimes
        || transmat.ncols() != n_regimes
        || initial_mode_probs.len() != n_regimes
    {
        return Err(SSMError::NumericalError(
            "parameter shapes inconsistent with the number of regimes".to_string(),
        ));
    }
    let r = observation_noise.max(VARIANCE_FLOOR);

    // Per-mode, per-feature filtered state means and variances. Each feature is
    // seeded with the first observation and that feature's sample variance,
    // shared across all modes.
    let var_seed: Vec<f64> = (0..n_features)
        .map(|f| data_var_seed(observations, f))
        .collect();
    let mut means = Array2::<f64>::zeros((n_regimes, n_features));
    let mut variances = Array2::<f64>::zeros((n_regimes, n_features));
    for k in 0..n_regimes {
        for f in 0..n_features {
            means[[k, f]] = observations[[0, f]];
            variances[[k, f]] = var_seed[f];
        }
    }
    let mut mode_probs = initial_mode_probs.clone();
    normalize_in_place(&mut mode_probs);

    let mut regime_probabilities = Array2::<f64>::zeros((n_obs, n_regimes));
    let mut filtered_states = Array1::<f64>::zeros(n_obs);

    let mut innov_sq_sum = 0.0;
    let mut p_pred_sum = 0.0;
    let mut r_weight = 0.0;
    let mut correction_sq_sum = Array1::<f64>::zeros(n_regimes);
    let mut mode_weight = Array1::<f64>::zeros(n_regimes);
    let mut sequence_ll = 0.0;

    // Scratch buffers reused across timesteps.
    let mut cbar = vec![0.0_f64; n_regimes];
    let mut mixed_means = Array2::<f64>::zeros((n_regimes, n_features));
    let mut mixed_vars = Array2::<f64>::zeros((n_regimes, n_features));
    let mut new_means = Array2::<f64>::zeros((n_regimes, n_features));
    let mut new_vars = Array2::<f64>::zeros((n_regimes, n_features));
    let mut log_mode_lik = vec![0.0_f64; n_regimes];
    let mut log_post = vec![0.0_f64; n_regimes];
    // Per-(mode, feature) predicted variance and squared correction for this step.
    let mut p_pred_step = Array2::<f64>::zeros((n_regimes, n_features));
    let mut correction_step = Array2::<f64>::zeros((n_regimes, n_features));

    for t in 0..n_obs {
        // --- 1. Mixing ---
        for j in 0..n_regimes {
            let mut c = 0.0;
            for i in 0..n_regimes {
                c += transmat[[i, j]] * mode_probs[i];
            }
            cbar[j] = c;
        }
        for j in 0..n_regimes {
            if cbar[j] > 0.0 {
                for f in 0..n_features {
                    let mut m0 = 0.0;
                    for i in 0..n_regimes {
                        let w = transmat[[i, j]] * mode_probs[i] / cbar[j];
                        m0 += w * means[[i, f]];
                    }
                    mixed_means[[j, f]] = m0;
                    let mut p0 = 0.0;
                    for i in 0..n_regimes {
                        let w = transmat[[i, j]] * mode_probs[i] / cbar[j];
                        let diff = means[[i, f]] - m0;
                        p0 += w * (variances[[i, f]] + diff * diff);
                    }
                    mixed_vars[[j, f]] = p0;
                }
            } else {
                // Degenerate predicted mode mass: fall back to the mode's own
                // previous estimate so the filter stays well-defined.
                for f in 0..n_features {
                    mixed_means[[j, f]] = means[[j, f]];
                    mixed_vars[[j, f]] = variances[[j, f]];
                }
            }
        }

        // --- 2. Mode-matched Kalman step (summed over independent features) ---
        for j in 0..n_regimes {
            let q_j = process_noises[j].max(VARIANCE_FLOOR);
            let mut log_lik = 0.0;
            for f in 0..n_features {
                let m_pred = mixed_means[[j, f]];
                let p_pred = mixed_vars[[j, f]] + q_j;
                let innov = observations[[t, f]] - m_pred;
                let s = p_pred + r;
                if s <= 0.0 || !s.is_finite() {
                    return Err(SSMError::NumericalError(
                        "non-positive innovation variance".to_string(),
                    ));
                }
                let gain = p_pred / s;
                let m_new = m_pred + gain * innov;
                let p_new = (1.0 - gain) * p_pred;
                new_means[[j, f]] = m_new;
                new_vars[[j, f]] = p_new;
                log_lik += -0.5 * (LN_2PI + safe_ln(s) + innov * innov / s);

                p_pred_step[[j, f]] = p_pred;
                let correction = m_new - m_pred;
                correction_step[[j, f]] = correction * correction;
            }
            log_mode_lik[j] = log_lik;
        }

        // --- 3. Mode probability update (log-sum-exp normalization) ---
        for j in 0..n_regimes {
            log_post[j] = safe_ln(cbar[j]) + log_mode_lik[j];
        }
        let log_norm = logsumexp(&log_post);
        if !log_norm.is_finite() {
            return Err(SSMError::NumericalError(
                "degenerate mode posterior (all modes have zero likelihood)".to_string(),
            ));
        }
        for j in 0..n_regimes {
            mode_probs[j] = (log_post[j] - log_norm).exp();
        }
        normalize_in_place(&mut mode_probs);

        // --- 4. Combination (output) ---
        for f in 0..n_features {
            let mut m_comb = 0.0;
            for j in 0..n_regimes {
                m_comb += mode_probs[j] * new_means[[j, f]];
            }
            filtered_states[t] += m_comb;
        }
        filtered_states[t] /= n_features as f64;

        for j in 0..n_regimes {
            regime_probabilities[[t, j]] = mode_probs[j];
        }

        // --- 5. Sequence log-likelihood ---
        sequence_ll += log_norm;

        // --- Sufficient statistics for the refinement M-step ---
        for j in 0..n_regimes {
            let w = mode_probs[j];
            mode_weight[j] += w * n_features as f64;
            for f in 0..n_features {
                let innov = observations[[t, f]] - mixed_means[[j, f]];
                innov_sq_sum += w * innov * innov;
                p_pred_sum += w * p_pred_step[[j, f]];
                r_weight += w;
                correction_sq_sum[j] += w * correction_step[[j, f]];
            }
        }

        // Carry the updated per-mode estimates into the next timestep.
        means.assign(&new_means);
        variances.assign(&new_vars);
    }

    if !sequence_ll.is_finite() {
        return Err(SSMError::NumericalError(
            "non-finite sequence log-likelihood".to_string(),
        ));
    }

    Ok(FilterTrace {
        regime_probabilities,
        filtered_states,
        innov_sq_sum,
        p_pred_sum,
        r_weight,
        correction_sq_sum,
        mode_weight,
        log_likelihood: sequence_ll,
    })
}

/// Initial per-feature state variance seed for the filter: the sample variance
/// of that feature column, floored to stay positive.
fn data_var_seed(observations: &ArrayView2<f64>, feature: usize) -> f64 {
    let n_obs = observations.nrows();
    if n_obs == 0 {
        return VARIANCE_FLOOR;
    }
    let mut mean = 0.0;
    for t in 0..n_obs {
        mean += observations[[t, feature]];
    }
    mean /= n_obs as f64;
    let mut var = 0.0;
    for t in 0..n_obs {
        let diff = observations[[t, feature]] - mean;
        var += diff * diff;
    }
    (var / n_obs as f64).max(VARIANCE_FLOOR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build the deterministic calm-then-volatile scalar sequence used by the
    /// regime-detection test: 30 near-constant calm steps followed by 30
    /// large-amplitude alternating volatile steps around a drifting level.
    fn calm_then_volatile() -> Array2<f64> {
        let calm_len = 30usize;
        let volatile_len = 30usize;
        let total = calm_len + volatile_len;
        let mut data = Array2::<f64>::zeros((total, 1));

        // Calm segment: tiny deterministic jitter around zero (amplitude 0.05).
        for t in 0..calm_len {
            let jitter = if t % 2 == 0 { 0.05 } else { -0.05 };
            data[[t, 0]] = jitter;
        }

        // Volatile segment: large alternating +2 / -2 jumps around a slowly
        // drifting level, deterministic and reproducible (no RNG).
        for i in 0..volatile_len {
            let t = calm_len + i;
            let jump = if i % 2 == 0 { 2.0 } else { -2.0 };
            let drift = 0.1 * i as f64;
            data[[t, 0]] = drift + jump;
        }

        data
    }

    #[test]
    fn test_imm_detects_volatility_regime() {
        let data = calm_then_volatile();
        let model = SwitchingStateSpaceModelBuilder::new(2, 1)
            .max_iter(15)
            .build();
        let trained = model.fit(&data.view()).expect("fit should succeed");

        let result = trained.filter(&data.view()).expect("filter should succeed");
        assert_eq!(result.regime_probabilities.dim(), (60, 2));

        // Every regime-posterior row is a valid distribution.
        for t in 0..result.regime_probabilities.nrows() {
            let row_sum: f64 = (0..2).map(|k| result.regime_probabilities[[t, k]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "row {t} sums to {row_sum}, expected 1.0"
            );
        }

        // Identify the volatile regime as the one with the larger fitted q_k.
        let q = trained.process_noises();
        let volatile_regime = if q[1] > q[0] { 1 } else { 0 };
        let calm_regime = 1 - volatile_regime;
        assert!(
            q[volatile_regime] > q[calm_regime],
            "expected a strict process-noise ordering, got {q:?}"
        );

        // Segment-averaged probability of the volatile regime must be higher in
        // the volatile half than in the calm half.
        let calm_avg: f64 = (0..30)
            .map(|t| result.regime_probabilities[[t, volatile_regime]])
            .sum::<f64>()
            / 30.0;
        let volatile_avg: f64 = (30..60)
            .map(|t| result.regime_probabilities[[t, volatile_regime]])
            .sum::<f64>()
            / 30.0;
        assert!(
            volatile_avg > calm_avg,
            "volatile-regime probability should be higher over the volatile \
             segment: volatile_avg={volatile_avg}, calm_avg={calm_avg}"
        );

        // predict returns valid regimes; score is finite.
        let path = trained
            .predict(&data.view())
            .expect("predict should succeed");
        assert_eq!(path.len(), 60);
        for &k in path.iter() {
            assert!(k < 2, "predicted regime {k} out of range");
        }
        let score = trained.score(&data.view()).expect("score should succeed");
        assert!(score.is_finite(), "score must be finite, got {score}");
    }

    #[test]
    fn test_ssm_rows_normalized_and_states_track() {
        let data = calm_then_volatile();
        let model = SwitchingStateSpaceModelBuilder::new(2, 1)
            .max_iter(10)
            .build();
        let trained = model.fit(&data.view()).expect("fit should succeed");
        let result = trained.filter(&data.view()).expect("filter should succeed");

        assert_eq!(result.filtered_states.len(), 60);
        for &s in result.filtered_states.iter() {
            assert!(s.is_finite(), "filtered state must be finite, got {s}");
        }
        for t in 0..result.regime_probabilities.nrows() {
            let row_sum: f64 = (0..2).map(|k| result.regime_probabilities[[t, k]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "row {t} sums to {row_sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_ssm_rejects_empty() {
        let empty = Array2::<f64>::zeros((0, 1));
        let model = SwitchingStateSpaceModelBuilder::new(2, 1).build();
        let fit_result = model.fit(&empty.view());
        assert!(fit_result.is_err(), "empty observations must error on fit");
    }
}
