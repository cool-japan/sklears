//! Time Series Mixture Models
//!
//! This module provides time series mixture model implementations: Gaussian
//! Hidden Markov Models (HMM), Markov regime-switching models, switching
//! state-space models, temporal Gaussian mixtures, and dynamic mixtures with
//! evolving parameters. All implementations follow SciRS2 Policy.
//!
//! The shared, numerically stable log-space filtering primitives
//! (`forward_log`, `backward_log`, `viterbi`, …) live in this module root
//! and are reused across the HMM and regime-switching models.

use scirs2_core::ndarray::{Array1, Array2};

pub mod dynamic_mixture;
pub mod hmm;
pub mod regime_switching;
pub mod switching_state_space;
pub mod temporal_gaussian_mixture;

pub use dynamic_mixture::{
    DynamicMixture, DynamicMixtureBuilder, DynamicMixtureTrained, ParameterEvolution,
};
pub use hmm::{
    HMMConfig, HMMError, HiddenMarkovModel, HiddenMarkovModelBuilder, HiddenMarkovModelTrained,
};
pub use regime_switching::{
    RSMConfig, RegimeParameters, RegimeSwitchingModel, RegimeSwitchingModelBuilder,
    RegimeSwitchingModelTrained, RegimeType,
};
pub use switching_state_space::{
    SSMConfig, SwitchingStateSpaceModel, SwitchingStateSpaceModelBuilder,
    SwitchingStateSpaceModelTrained,
};
pub use temporal_gaussian_mixture::{
    TemporalGaussianMixture, TemporalGaussianMixtureBuilder, TemporalGaussianMixtureTrained,
};

/// Covariance regularization added to the diagonal of every state covariance
/// during an M-step to keep the matrices positive definite.
pub(crate) const REG_COVAR: f64 = 1e-6;

/// Natural logarithm that maps non-positive inputs to negative infinity.
pub(crate) fn safe_ln(value: f64) -> f64 {
    if value <= 0.0 {
        f64::NEG_INFINITY
    } else {
        value.ln()
    }
}

/// Numerically stable log-sum-exp over a slice.
pub(crate) fn logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = values.iter().map(|v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Log-space forward recursion for a Markov chain with precomputed emission
/// log-probabilities `log_b[t, s]`. Returns `(log_alpha, log_likelihood)`.
pub(crate) fn forward_log(
    startprob: &Array1<f64>,
    transmat: &Array2<f64>,
    log_b: &Array2<f64>,
) -> (Array2<f64>, f64) {
    let n_obs = log_b.nrows();
    let n_states = startprob.len();
    let log_trans = transmat.mapv(safe_ln);
    let log_start = startprob.mapv(safe_ln);

    let mut log_alpha = Array2::from_elem((n_obs, n_states), f64::NEG_INFINITY);
    for s in 0..n_states {
        log_alpha[[0, s]] = log_start[s] + log_b[[0, s]];
    }
    let mut scratch = vec![0.0_f64; n_states];
    for t in 1..n_obs {
        for s in 0..n_states {
            for i in 0..n_states {
                scratch[i] = log_alpha[[t - 1, i]] + log_trans[[i, s]];
            }
            log_alpha[[t, s]] = logsumexp(&scratch) + log_b[[t, s]];
        }
    }
    let last: Vec<f64> = (0..n_states).map(|s| log_alpha[[n_obs - 1, s]]).collect();
    (log_alpha, logsumexp(&last))
}

/// Log-space backward recursion for a Markov chain with precomputed emission
/// log-probabilities `log_b[t, s]`.
pub(crate) fn backward_log(transmat: &Array2<f64>, log_b: &Array2<f64>) -> Array2<f64> {
    let n_obs = log_b.nrows();
    let n_states = transmat.nrows();
    let log_trans = transmat.mapv(safe_ln);

    let mut log_beta = Array2::from_elem((n_obs, n_states), f64::NEG_INFINITY);
    for s in 0..n_states {
        log_beta[[n_obs - 1, s]] = 0.0;
    }
    let mut scratch = vec![0.0_f64; n_states];
    for t in (0..n_obs - 1).rev() {
        for i in 0..n_states {
            for j in 0..n_states {
                scratch[j] = log_trans[[i, j]] + log_b[[t + 1, j]] + log_beta[[t + 1, j]];
            }
            log_beta[[t, i]] = logsumexp(&scratch);
        }
    }
    log_beta
}

/// Posterior state probabilities `gamma[t, s] = P(state_t = s | observations)`.
pub(crate) fn posterior_gamma(
    log_alpha: &Array2<f64>,
    log_beta: &Array2<f64>,
    ll: f64,
) -> Array2<f64> {
    let (n_obs, n_states) = log_alpha.dim();
    let mut gamma = Array2::<f64>::zeros((n_obs, n_states));
    for t in 0..n_obs {
        for s in 0..n_states {
            gamma[[t, s]] = (log_alpha[[t, s]] + log_beta[[t, s]] - ll).exp();
        }
    }
    gamma
}

/// Sum over time of the pairwise posteriors `xi[t, i, j]`, used for
/// transition-matrix re-estimation in an M-step.
pub(crate) fn posterior_xi_sum(
    log_alpha: &Array2<f64>,
    log_beta: &Array2<f64>,
    transmat: &Array2<f64>,
    log_b: &Array2<f64>,
    ll: f64,
) -> Array2<f64> {
    let (n_obs, n_states) = log_alpha.dim();
    let log_trans = transmat.mapv(safe_ln);
    let mut xi_sum = Array2::<f64>::zeros((n_states, n_states));
    for t in 0..n_obs - 1 {
        for i in 0..n_states {
            for j in 0..n_states {
                let log_xi = log_alpha[[t, i]]
                    + log_trans[[i, j]]
                    + log_b[[t + 1, j]]
                    + log_beta[[t + 1, j]]
                    - ll;
                xi_sum[[i, j]] += log_xi.exp();
            }
        }
    }
    xi_sum
}

/// Viterbi decoding: the single most likely hidden-state path (log space).
pub(crate) fn viterbi(
    startprob: &Array1<f64>,
    transmat: &Array2<f64>,
    log_b: &Array2<f64>,
) -> Array1<usize> {
    let n_obs = log_b.nrows();
    let n_states = startprob.len();
    let log_trans = transmat.mapv(safe_ln);
    let log_start = startprob.mapv(safe_ln);

    let mut delta = Array2::from_elem((n_obs, n_states), f64::NEG_INFINITY);
    let mut psi = Array2::<usize>::zeros((n_obs, n_states));
    for s in 0..n_states {
        delta[[0, s]] = log_start[s] + log_b[[0, s]];
    }
    for t in 1..n_obs {
        for s in 0..n_states {
            let mut best = f64::NEG_INFINITY;
            let mut best_i = 0;
            for i in 0..n_states {
                let candidate = delta[[t - 1, i]] + log_trans[[i, s]];
                if candidate > best {
                    best = candidate;
                    best_i = i;
                }
            }
            delta[[t, s]] = best + log_b[[t, s]];
            psi[[t, s]] = best_i;
        }
    }

    let mut path = Array1::<usize>::zeros(n_obs);
    let mut best = f64::NEG_INFINITY;
    let mut best_s = 0;
    for s in 0..n_states {
        if delta[[n_obs - 1, s]] > best {
            best = delta[[n_obs - 1, s]];
            best_s = s;
        }
    }
    path[n_obs - 1] = best_s;
    for t in (0..n_obs - 1).rev() {
        path[t] = psi[[t + 1, path[t + 1]]];
    }
    path
}
