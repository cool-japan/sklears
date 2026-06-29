//! Gaussian Hidden Markov Model: Baum-Welch training, Viterbi decoding, and
//! log-space scoring.

use super::{backward_log, forward_log, posterior_gamma, posterior_xi_sum, viterbi, REG_COVAR};
use crate::common::gaussian_log_pdf;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};

/// Configuration for a Gaussian Hidden Markov Model.
#[derive(Debug, Clone)]
pub struct HMMConfig {
    /// Number of hidden states.
    pub n_states: usize,
    /// Maximum number of Baum-Welch (EM) iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the per-iteration log-likelihood change.
    pub tol: f64,
}

impl Default for HMMConfig {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 100,
            tol: 1e-4,
        }
    }
}

/// Error type for HMM operations.
#[derive(Debug, thiserror::Error)]
pub enum HMMError {
    /// The requested number of states is invalid (zero, or larger than the sequence length).
    #[error("Invalid number of states: {0}")]
    InvalidStates(usize),
    /// Baum-Welch failed to reach the tolerance within the iteration budget.
    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize),
    /// The observation sequence is empty, malformed, or has the wrong feature count.
    #[error("Invalid observation sequence")]
    InvalidObservation,
    /// A numerical failure occurred (e.g. a non-positive-definite covariance).
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Per-timestep, per-state Gaussian emission log-probabilities `log b_s(o_t)`.
fn emission_log_probs(
    observations: &ArrayView2<f64>,
    means: &Array2<f64>,
    covars: &[Array2<f64>],
) -> Result<Array2<f64>, HMMError> {
    let n_obs = observations.nrows();
    let n_states = means.nrows();
    let mut log_b = Array2::<f64>::zeros((n_obs, n_states));
    for t in 0..n_obs {
        let x = observations.row(t);
        for s in 0..n_states {
            log_b[[t, s]] = gaussian_log_pdf(&x, &means.row(s), &covars[s].view())
                .map_err(|e| HMMError::NumericalError(e.to_string()))?;
        }
    }
    Ok(log_b)
}

/// Gaussian Hidden Markov Model (untrained).
///
/// Models a single multivariate observation sequence with full-covariance
/// Gaussian emissions and a first-order Markov chain over hidden states. Trained
/// with Baum-Welch (Expectation-Maximization) using numerically stable log-space
/// forward-backward recursions (Rabiner, 1989).
#[derive(Debug, Clone)]
pub struct HiddenMarkovModel {
    config: HMMConfig,
}

impl HiddenMarkovModel {
    /// Create a new untrained HMM from the given configuration.
    pub fn new(config: HMMConfig) -> Self {
        Self { config }
    }

    /// Fit the model to a `(n_timesteps, n_features)` observation sequence via Baum-Welch.
    pub fn fit(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<HiddenMarkovModelTrained, HMMError> {
        let n_states = self.config.n_states;
        let n_obs = observations.nrows();
        let n_features = observations.ncols();

        if n_states == 0 {
            return Err(HMMError::InvalidStates(0));
        }
        if n_obs < n_states || n_features == 0 {
            return Err(HMMError::InvalidObservation);
        }

        // --- Deterministic initialization ---
        // Means: n_states observations spread evenly across the sequence.
        let mut means = Array2::<f64>::zeros((n_states, n_features));
        for s in 0..n_states {
            let idx = s * (n_obs - 1) / (n_states - 1).max(1);
            means.row_mut(s).assign(&observations.row(idx));
        }
        // Covariances: the global diagonal variance shared by every state.
        let global_mean = observations
            .mean_axis(Axis(0))
            .ok_or(HMMError::InvalidObservation)?;
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
        let mut covars: Vec<Array2<f64>> = (0..n_states)
            .map(|_| Array2::from_diag(&global_var))
            .collect();
        // Start and transition distributions: uniform.
        let mut startprob = Array1::from_elem(n_states, 1.0 / n_states as f64);
        let mut transmat = Array2::from_elem((n_states, n_states), 1.0 / n_states as f64);

        let mut prev_ll = f64::NEG_INFINITY;
        let mut converged = false;

        for _iter in 0..self.config.max_iter {
            // --- E-step ---
            let log_b = emission_log_probs(observations, &means, &covars)?;
            let (log_alpha, ll) = forward_log(&startprob, &transmat, &log_b);
            let log_beta = backward_log(&transmat, &log_b);
            let gamma = posterior_gamma(&log_alpha, &log_beta, ll);
            let xi_sum = posterior_xi_sum(&log_alpha, &log_beta, &transmat, &log_b, ll);

            // --- M-step ---
            // Initial-state distribution.
            startprob = gamma.row(0).to_owned();
            // Transition matrix.
            for i in 0..n_states {
                let denom: f64 = (0..n_obs - 1).map(|t| gamma[[t, i]]).sum();
                if denom > 0.0 {
                    for j in 0..n_states {
                        transmat[[i, j]] = xi_sum[[i, j]] / denom;
                    }
                }
            }
            // Emission means and covariances.
            for s in 0..n_states {
                let weight: f64 = (0..n_obs).map(|t| gamma[[t, s]]).sum();
                if weight <= 0.0 {
                    continue;
                }
                let mut mean = Array1::<f64>::zeros(n_features);
                for t in 0..n_obs {
                    let g = gamma[[t, s]];
                    for k in 0..n_features {
                        mean[k] += g * observations[[t, k]];
                    }
                }
                for k in 0..n_features {
                    mean[k] /= weight;
                }
                let mut cov = Array2::<f64>::zeros((n_features, n_features));
                for t in 0..n_obs {
                    let g = gamma[[t, s]];
                    for a in 0..n_features {
                        let da = observations[[t, a]] - mean[a];
                        for b in 0..n_features {
                            let db = observations[[t, b]] - mean[b];
                            cov[[a, b]] += g * da * db;
                        }
                    }
                }
                for a in 0..n_features {
                    for b in 0..n_features {
                        cov[[a, b]] /= weight;
                    }
                }
                for k in 0..n_features {
                    cov[[k, k]] += REG_COVAR;
                }
                means.row_mut(s).assign(&mean);
                covars[s] = cov;
            }

            if (ll - prev_ll).abs() < self.config.tol {
                converged = true;
                prev_ll = ll;
                break;
            }
            prev_ll = ll;
        }

        // EM running to the iteration cap without hitting `tol` is reported as
        // the final fitted state rather than an error (matching scikit-learn).
        let _ = converged;

        Ok(HiddenMarkovModelTrained {
            startprob,
            transmat,
            means,
            covars,
            log_likelihood: prev_ll,
            n_states,
            n_features,
        })
    }
}

/// A fitted Gaussian Hidden Markov Model.
#[derive(Debug, Clone)]
pub struct HiddenMarkovModelTrained {
    startprob: Array1<f64>,
    transmat: Array2<f64>,
    means: Array2<f64>,
    covars: Vec<Array2<f64>>,
    log_likelihood: f64,
    n_states: usize,
    n_features: usize,
}

impl HiddenMarkovModelTrained {
    /// Initial hidden-state distribution.
    pub fn startprob(&self) -> &Array1<f64> {
        &self.startprob
    }

    /// State transition probability matrix.
    pub fn transmat(&self) -> &Array2<f64> {
        &self.transmat
    }

    /// Per-state Gaussian emission means (`n_states x n_features`).
    pub fn means(&self) -> &Array2<f64> {
        &self.means
    }

    /// Per-state Gaussian emission covariance matrices.
    pub fn covars(&self) -> &[Array2<f64>] {
        &self.covars
    }

    /// Number of hidden states.
    pub fn n_states(&self) -> usize {
        self.n_states
    }

    /// Training-set log-likelihood at the final EM iteration.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Log-likelihood of an observation sequence under the fitted model.
    pub fn score(&self, observations: &ArrayView2<f64>) -> Result<f64, HMMError> {
        if observations.ncols() != self.n_features || observations.nrows() == 0 {
            return Err(HMMError::InvalidObservation);
        }
        let log_b = emission_log_probs(observations, &self.means, &self.covars)?;
        let (_log_alpha, ll) = forward_log(&self.startprob, &self.transmat, &log_b);
        Ok(ll)
    }

    /// Most likely hidden-state sequence via Viterbi decoding.
    pub fn predict(&self, observations: &ArrayView2<f64>) -> Result<Array1<usize>, HMMError> {
        if observations.ncols() != self.n_features || observations.nrows() == 0 {
            return Err(HMMError::InvalidObservation);
        }
        let log_b = emission_log_probs(observations, &self.means, &self.covars)?;
        Ok(viterbi(&self.startprob, &self.transmat, &log_b))
    }

    /// Posterior state probabilities `gamma[t, s]` for each timestep.
    pub fn predict_proba(&self, observations: &ArrayView2<f64>) -> Result<Array2<f64>, HMMError> {
        if observations.ncols() != self.n_features || observations.nrows() == 0 {
            return Err(HMMError::InvalidObservation);
        }
        let log_b = emission_log_probs(observations, &self.means, &self.covars)?;
        let (log_alpha, ll) = forward_log(&self.startprob, &self.transmat, &log_b);
        let log_beta = backward_log(&self.transmat, &log_b);
        Ok(posterior_gamma(&log_alpha, &log_beta, ll))
    }
}

/// Builder for [`HiddenMarkovModel`].
#[derive(Debug, Clone)]
pub struct HiddenMarkovModelBuilder {
    n_states: usize,
    max_iter: usize,
    tol: f64,
}

impl HiddenMarkovModelBuilder {
    /// Start configuring an HMM with the given number of hidden states.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            max_iter: 100,
            tol: 1e-4,
        }
    }

    /// Set the maximum number of Baum-Welch iterations (default 100).
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the log-likelihood convergence tolerance (default 1e-4).
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Build the untrained model.
    pub fn build(self) -> HiddenMarkovModel {
        HiddenMarkovModel::new(HMMConfig {
            n_states: self.n_states,
            max_iter: self.max_iter,
            tol: self.tol,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Build a two-regime 1-D sequence: a block near 0 followed by a block near 10.
    fn two_regime_sequence() -> Array2<f64> {
        let mut obs = Array2::<f64>::zeros((30, 1));
        for t in 0..30 {
            let base = if t < 15 { 0.0 } else { 10.0 };
            let jitter = if t % 2 == 0 { 0.1 } else { -0.1 };
            obs[[t, 0]] = base + jitter;
        }
        obs
    }

    #[test]
    fn test_hmm_builder_configures_model() {
        let model = HiddenMarkovModelBuilder::new(3)
            .max_iter(42)
            .tol(1e-3)
            .build();
        assert_eq!(model.config.n_states, 3);
        assert_eq!(model.config.max_iter, 42);
        assert_eq!(model.config.tol, 1e-3);
    }

    #[test]
    fn test_hmm_rejects_too_many_states() {
        let obs = array![[0.0], [1.0]];
        let model = HiddenMarkovModelBuilder::new(5).build();
        assert!(matches!(
            model.fit(&obs.view()),
            Err(HMMError::InvalidObservation)
        ));
    }

    #[test]
    fn test_hmm_fit_recovers_two_regimes() {
        let obs = two_regime_sequence();
        let model = HiddenMarkovModelBuilder::new(2)
            .max_iter(100)
            .tol(1e-6)
            .build();
        let trained = model.fit(&obs.view()).expect("HMM fit should succeed");

        let start_sum: f64 = trained.startprob().sum();
        assert!((start_sum - 1.0).abs() < 1e-6);
        for i in 0..trained.n_states() {
            let row_sum: f64 = trained.transmat().row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }

        let m0 = trained.means()[[0, 0]];
        let m1 = trained.means()[[1, 0]];
        let (lo, hi) = if m0 < m1 { (m0, m1) } else { (m1, m0) };
        assert!(lo < 3.0, "low regime mean was {lo}");
        assert!(hi > 7.0, "high regime mean was {hi}");
    }

    #[test]
    fn test_hmm_predict_and_score() {
        let obs = two_regime_sequence();
        let model = HiddenMarkovModelBuilder::new(2)
            .max_iter(100)
            .tol(1e-6)
            .build();
        let trained = model.fit(&obs.view()).expect("HMM fit should succeed");

        let path = trained
            .predict(&obs.view())
            .expect("predict should succeed");
        assert_eq!(path.len(), obs.nrows());
        assert!(path.iter().all(|&s| s < 2));
        assert_ne!(path[0], path[obs.nrows() - 1]);

        let proba = trained
            .predict_proba(&obs.view())
            .expect("predict_proba should succeed");
        assert_eq!(proba.dim(), (obs.nrows(), 2));
        for t in 0..obs.nrows() {
            let row_sum: f64 = proba.row(t).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "row {t} summed to {row_sum}");
        }

        let score = trained.score(&obs.view()).expect("score should succeed");
        assert!(score.is_finite());
        assert!((score - trained.log_likelihood()).abs() < 1e-9);
    }
}
