//! Markov regime-switching model (Hamilton, 1989).
//!
//! A constrained Gaussian-emission hidden Markov model in which `K` latent
//! regimes are governed by a first-order Markov chain. The regime-conditional
//! emission distribution is constrained according to a [`RegimeType`]:
//!
//! * [`RegimeType::MeanSwitching`] — regime-specific means with a single shared
//!   full covariance.
//! * [`RegimeType::VarianceSwitching`] — a single shared mean with
//!   regime-specific covariances.
//! * [`RegimeType::FullSwitching`] — regime-specific means *and* covariances
//!   (equivalent to a general Gaussian HMM).
//! * [`RegimeType::AutoregressiveSwitching`] — univariate regime-specific
//!   AR(1) dynamics `o_t = c_k + phi_k * o_{t-1} + eps`, `eps ~ N(0, sigma2_k)`.
//!
//! Training uses Expectation-Maximization: the Hamilton filter and Kim smoother
//! are realized through the shared, numerically stable log-space
//! forward-backward primitives (`forward_log`, `backward_log`,
//! `posterior_gamma`, `posterior_xi_sum`). Decoding uses `viterbi`.

use super::{
    backward_log, forward_log, logsumexp, posterior_gamma, posterior_xi_sum, viterbi, REG_COVAR,
};
use crate::common::gaussian_log_pdf;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};

/// The way in which regimes are allowed to differ from one another.
#[derive(Debug, Clone, PartialEq)]
pub enum RegimeType {
    /// Regimes differ only in their mean; a single full covariance is shared.
    MeanSwitching,
    /// Regimes differ only in their covariance; a single mean is shared.
    VarianceSwitching,
    /// Regimes follow regime-specific univariate AR(1) dynamics.
    AutoregressiveSwitching,
    /// Regimes differ in both mean and covariance (a general Gaussian HMM).
    FullSwitching,
}

/// Error type for regime-switching-model operations.
#[derive(Debug, thiserror::Error)]
pub enum RSMError {
    /// The requested number of regimes is invalid (zero, or larger than the
    /// observation-sequence length).
    #[error("Invalid number of regimes: {0}")]
    InvalidRegimes(usize),
    /// The observation sequence is empty, malformed, or has the wrong feature
    /// count for the fitted model (including a non-univariate input supplied to
    /// an autoregressive-switching model).
    #[error("Invalid observation sequence")]
    InvalidObservation,
    /// A numerical failure occurred (for example a non-positive-definite
    /// covariance or a singular regression design).
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Fitted regime-conditional emission parameters.
///
/// The fields are populated according to the [`RegimeType`] of the model.
///
/// * For [`RegimeType::MeanSwitching`], [`RegimeType::VarianceSwitching`] and
///   [`RegimeType::FullSwitching`], `means` holds one mean per regime
///   (`n_regimes x n_features`) and `covariances` holds one covariance matrix
///   per regime. Under the *MeanSwitching* constraint every entry of
///   `covariances` is the same shared matrix; under the *VarianceSwitching*
///   constraint every row of `means` is the same shared mean. The AR fields are
///   empty.
/// * For [`RegimeType::AutoregressiveSwitching`], `ar_intercepts`, `ar_coefs`
///   and `ar_sigmas` hold the per-regime intercept `c_k`, lag coefficient
///   `phi_k` and innovation variance `sigma2_k`. The `means`/`covariances`
///   fields are left empty.
#[derive(Debug, Clone)]
pub struct RegimeParameters {
    /// Regime-conditional emission means (`n_regimes x n_features`).
    pub means: Array2<f64>,
    /// Regime-conditional emission covariance matrices.
    pub covariances: Vec<Array2<f64>>,
    /// Per-regime AR(1) intercepts `c_k` (autoregressive switching only).
    pub ar_intercepts: Array1<f64>,
    /// Per-regime AR(1) lag coefficients `phi_k` (autoregressive switching only).
    pub ar_coefs: Array1<f64>,
    /// Per-regime AR(1) innovation variances `sigma2_k` (autoregressive switching only).
    pub ar_sigmas: Array1<f64>,
}

impl RegimeParameters {
    /// Allocate Gaussian-emission parameters for `n_regimes` regimes over
    /// `n_features` features, with empty AR fields.
    fn gaussian(n_regimes: usize, n_features: usize) -> Self {
        Self {
            means: Array2::<f64>::zeros((n_regimes, n_features)),
            covariances: (0..n_regimes)
                .map(|_| Array2::<f64>::zeros((n_features, n_features)))
                .collect(),
            ar_intercepts: Array1::<f64>::zeros(0),
            ar_coefs: Array1::<f64>::zeros(0),
            ar_sigmas: Array1::<f64>::zeros(0),
        }
    }

    /// Allocate AR(1) parameters for `n_regimes` regimes, with empty Gaussian
    /// fields.
    fn autoregressive(n_regimes: usize) -> Self {
        Self {
            means: Array2::<f64>::zeros((0, 0)),
            covariances: Vec::new(),
            ar_intercepts: Array1::<f64>::zeros(n_regimes),
            ar_coefs: Array1::<f64>::zeros(n_regimes),
            ar_sigmas: Array1::<f64>::zeros(n_regimes),
        }
    }
}

/// Configuration for a [`RegimeSwitchingModel`].
#[derive(Debug, Clone)]
pub struct RSMConfig {
    /// Number of latent regimes.
    pub n_regimes: usize,
    /// The structural constraint imposed on the regime-conditional emissions.
    pub regime_type: RegimeType,
    /// Maximum number of EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the per-iteration log-likelihood change.
    pub tol: f64,
}

impl Default for RSMConfig {
    fn default() -> Self {
        Self {
            n_regimes: 2,
            regime_type: RegimeType::FullSwitching,
            max_iter: 100,
            tol: 1e-4,
        }
    }
}

/// Univariate normal log-density `ln N(x; mean, var)`.
///
/// The variance is floored at [`REG_COVAR`] so the density stays finite even
/// when an empty/degenerate regime collapses its innovation variance to zero.
fn univariate_normal_logpdf(x: f64, mean: f64, var: f64) -> f64 {
    let safe_var = var.max(REG_COVAR);
    let diff = x - mean;
    -0.5 * ((2.0 * std::f64::consts::PI * safe_var).ln() + diff * diff / safe_var)
}

/// Per-timestep, per-regime emission log-probabilities for the Gaussian-emission
/// regime types (`MeanSwitching`, `VarianceSwitching`, `FullSwitching`).
///
/// The constraint is materialized into the `means`/`covariances` arrays during
/// the M-step, so a single full-covariance evaluation suffices here regardless
/// of the active [`RegimeType`].
fn gaussian_emission_log_probs(
    observations: &ArrayView2<f64>,
    means: &Array2<f64>,
    covariances: &[Array2<f64>],
) -> Result<Array2<f64>, RSMError> {
    let n_obs = observations.nrows();
    let n_regimes = means.nrows();
    let mut log_b = Array2::<f64>::zeros((n_obs, n_regimes));
    for t in 0..n_obs {
        let x = observations.row(t);
        for k in 0..n_regimes {
            log_b[[t, k]] = gaussian_log_pdf(&x, &means.row(k), &covariances[k].view())
                .map_err(|e| RSMError::NumericalError(e.to_string()))?;
        }
    }
    Ok(log_b)
}

/// Per-timestep, per-regime emission log-probabilities for univariate AR(1)
/// regimes.
///
/// At `t = 0` the stationary marginal of the AR(1) process is used (guarding the
/// `|phi| < 1` stationarity condition); for `t >= 1` the conditional density of
/// `o_t` given `o_{t-1}` is used.
fn ar_emission_log_probs(
    series: &[f64],
    intercepts: &Array1<f64>,
    coefs: &Array1<f64>,
    sigmas: &Array1<f64>,
) -> Array2<f64> {
    let n_obs = series.len();
    let n_regimes = intercepts.len();
    let mut log_b = Array2::<f64>::zeros((n_obs, n_regimes));
    for k in 0..n_regimes {
        let c = intercepts[k];
        let phi = coefs[k];
        let sigma2 = sigmas[k];
        // Stationary marginal for the first observation.
        let (mean0, var0) = if phi.abs() < 1.0 {
            (c / (1.0 - phi), sigma2 / (1.0 - phi * phi))
        } else {
            (c, sigma2)
        };
        log_b[[0, k]] = univariate_normal_logpdf(series[0], mean0, var0);
        for t in 1..n_obs {
            let mean = c + phi * series[t - 1];
            log_b[[t, k]] = univariate_normal_logpdf(series[t], mean, sigma2);
        }
    }
    log_b
}

/// Outer-product-weighted covariance `sum_t g_t (x_t - mu)(x_t - mu)^T`
/// accumulated over the whole sequence, divided by `weight` and ridge-
/// regularized on the diagonal.
fn weighted_covariance(
    observations: &ArrayView2<f64>,
    gammas: impl Iterator<Item = f64>,
    mean: &Array1<f64>,
    weight: f64,
) -> Array2<f64> {
    let n_features = mean.len();
    let mut cov = Array2::<f64>::zeros((n_features, n_features));
    for (t, g) in gammas.enumerate() {
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
    cov
}

/// Markov regime-switching model (untrained).
///
/// Models a single observation sequence with regime-conditional Gaussian (or
/// AR(1)) emissions whose structure is constrained by the configured
/// [`RegimeType`]. Trained with Expectation-Maximization built on numerically
/// stable log-space filtering/smoothing (Hamilton, 1989; Kim, 1994).
#[derive(Debug, Clone)]
pub struct RegimeSwitchingModel {
    config: RSMConfig,
}

impl RegimeSwitchingModel {
    /// Create a new untrained model from the given configuration.
    pub fn new(config: RSMConfig) -> Self {
        Self { config }
    }

    /// Fit the model to a `(n_timesteps, n_features)` observation sequence.
    pub fn fit(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<RegimeSwitchingModelTrained, RSMError> {
        let n_regimes = self.config.n_regimes;
        let n_obs = observations.nrows();
        let n_features = observations.ncols();

        if n_regimes == 0 {
            return Err(RSMError::InvalidRegimes(0));
        }
        if n_features == 0 {
            return Err(RSMError::InvalidObservation);
        }
        if n_obs < n_regimes {
            return Err(RSMError::InvalidRegimes(n_regimes));
        }
        if self.config.regime_type == RegimeType::AutoregressiveSwitching && n_features != 1 {
            return Err(RSMError::InvalidObservation);
        }

        match self.config.regime_type {
            RegimeType::AutoregressiveSwitching => self.fit_autoregressive(observations),
            _ => self.fit_gaussian(observations),
        }
    }

    /// EM for the Gaussian-emission regime types (`MeanSwitching`,
    /// `VarianceSwitching`, `FullSwitching`).
    fn fit_gaussian(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<RegimeSwitchingModelTrained, RSMError> {
        let n_regimes = self.config.n_regimes;
        let n_obs = observations.nrows();
        let n_features = observations.ncols();
        let regime_type = self.config.regime_type.clone();

        // --- Deterministic initialization ---
        let global_mean = observations
            .mean_axis(Axis(0))
            .ok_or(RSMError::InvalidObservation)?;
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
        let global_cov = Array2::from_diag(&global_var);

        let mut params = RegimeParameters::gaussian(n_regimes, n_features);
        for k in 0..n_regimes {
            let idx = k * (n_obs - 1) / (n_regimes - 1).max(1);
            // Under the variance-switching constraint every regime shares one
            // mean, so all rows are seeded with the global mean; otherwise the
            // means are spread over evenly-spaced observations.
            match &regime_type {
                RegimeType::VarianceSwitching => params.means.row_mut(k).assign(&global_mean),
                _ => params.means.row_mut(k).assign(&observations.row(idx)),
            }
            params.covariances[k] = global_cov.clone();
        }

        let mut startprob = Array1::from_elem(n_regimes, 1.0 / n_regimes as f64);
        let mut transmat = Array2::from_elem((n_regimes, n_regimes), 1.0 / n_regimes as f64);

        let mut prev_ll = f64::NEG_INFINITY;
        for _iter in 0..self.config.max_iter {
            // --- E-step ---
            let log_b =
                gaussian_emission_log_probs(observations, &params.means, &params.covariances)?;
            let (log_alpha, ll) = forward_log(&startprob, &transmat, &log_b);
            let log_beta = backward_log(&transmat, &log_b);
            let gamma = posterior_gamma(&log_alpha, &log_beta, ll);
            let xi_sum = posterior_xi_sum(&log_alpha, &log_beta, &transmat, &log_b, ll);

            // --- M-step: Markov-chain parameters ---
            startprob = gamma.row(0).to_owned();
            for i in 0..n_regimes {
                let denom: f64 = (0..n_obs - 1).map(|t| gamma[[t, i]]).sum();
                if denom > 0.0 {
                    for j in 0..n_regimes {
                        transmat[[i, j]] = xi_sum[[i, j]] / denom;
                    }
                }
            }

            // --- M-step: regime-conditional emissions ---
            self.update_gaussian_emissions(observations, &gamma, &global_mean, &mut params);

            if (ll - prev_ll).abs() < self.config.tol {
                prev_ll = ll;
                break;
            }
            prev_ll = ll;
        }

        Ok(RegimeSwitchingModelTrained {
            startprob,
            transmat,
            params,
            regime_type,
            log_likelihood: prev_ll,
            n_regimes,
            n_features,
        })
    }

    /// M-step update of the Gaussian regime emissions under the active
    /// [`RegimeType`] constraint.
    fn update_gaussian_emissions(
        &self,
        observations: &ArrayView2<f64>,
        gamma: &Array2<f64>,
        global_mean: &Array1<f64>,
        params: &mut RegimeParameters,
    ) {
        let n_regimes = params.means.nrows();
        let n_obs = observations.nrows();
        let n_features = observations.ncols();

        // Per-regime responsibility masses and weighted means.
        let mut weights = vec![0.0_f64; n_regimes];
        let mut regime_means: Vec<Array1<f64>> = (0..n_regimes)
            .map(|_| Array1::<f64>::zeros(n_features))
            .collect();
        for k in 0..n_regimes {
            let mut weight = 0.0;
            let mut mean = Array1::<f64>::zeros(n_features);
            for t in 0..n_obs {
                let g = gamma[[t, k]];
                weight += g;
                for f in 0..n_features {
                    mean[f] += g * observations[[t, f]];
                }
            }
            if weight > 0.0 {
                for f in 0..n_features {
                    mean[f] /= weight;
                }
            } else {
                mean.assign(global_mean);
            }
            weights[k] = weight;
            regime_means[k] = mean;
        }

        match self.config.regime_type {
            RegimeType::MeanSwitching => {
                // Regime-specific means.
                for (k, regime_mean) in regime_means.iter().enumerate() {
                    params.means.row_mut(k).assign(regime_mean);
                }
                // A single covariance shared across regimes, pooled over all
                // responsibilities relative to each regime's own mean.
                let total: f64 = weights.iter().sum();
                let mut shared = Array2::<f64>::zeros((n_features, n_features));
                for k in 0..n_regimes {
                    let mean = &regime_means[k];
                    for t in 0..n_obs {
                        let g = gamma[[t, k]];
                        for a in 0..n_features {
                            let da = observations[[t, a]] - mean[a];
                            for b in 0..n_features {
                                let db = observations[[t, b]] - mean[b];
                                shared[[a, b]] += g * da * db;
                            }
                        }
                    }
                }
                if total > 0.0 {
                    for a in 0..n_features {
                        for b in 0..n_features {
                            shared[[a, b]] /= total;
                        }
                    }
                }
                for d in 0..n_features {
                    shared[[d, d]] += REG_COVAR;
                }
                for k in 0..n_regimes {
                    params.covariances[k] = shared.clone();
                }
            }
            RegimeType::VarianceSwitching => {
                // A single mean shared across regimes. Because `sum_k gamma[t,k]
                // == 1` for every `t`, the responsibility-weighted overall mean
                // is simply the empirical mean of the observations.
                let shared_mean = observations
                    .mean_axis(Axis(0))
                    .unwrap_or_else(|| global_mean.clone());
                for k in 0..n_regimes {
                    params.means.row_mut(k).assign(&shared_mean);
                }
                // Regime-specific covariances about the shared mean.
                for k in 0..n_regimes {
                    if weights[k] > 0.0 {
                        params.covariances[k] = weighted_covariance(
                            observations,
                            (0..n_obs).map(|t| gamma[[t, k]]),
                            &shared_mean,
                            weights[k],
                        );
                    }
                }
            }
            _ => {
                // FullSwitching: standard HMM M-step (per-regime mean and
                // covariance).
                for k in 0..n_regimes {
                    if weights[k] > 0.0 {
                        params.means.row_mut(k).assign(&regime_means[k]);
                        params.covariances[k] = weighted_covariance(
                            observations,
                            (0..n_obs).map(|t| gamma[[t, k]]),
                            &regime_means[k],
                            weights[k],
                        );
                    }
                }
            }
        }
    }

    /// EM for the univariate AR(1) regime type.
    fn fit_autoregressive(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<RegimeSwitchingModelTrained, RSMError> {
        let n_regimes = self.config.n_regimes;
        let n_obs = observations.nrows();
        let series: Vec<f64> = (0..n_obs).map(|t| observations[[t, 0]]).collect();

        // --- Deterministic initialization ---
        let mean: f64 = series.iter().sum::<f64>() / n_obs as f64;
        let mut global_var: f64 =
            series.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n_obs as f64 + REG_COVAR;
        if !(global_var.is_finite() && global_var > 0.0) {
            global_var = REG_COVAR;
        }

        let mut params = RegimeParameters::autoregressive(n_regimes);
        for k in 0..n_regimes {
            let idx = k * (n_obs - 1) / (n_regimes - 1).max(1);
            params.ar_intercepts[k] = series[idx];
            params.ar_coefs[k] = 0.0;
            params.ar_sigmas[k] = global_var;
        }

        let mut startprob = Array1::from_elem(n_regimes, 1.0 / n_regimes as f64);
        let mut transmat = Array2::from_elem((n_regimes, n_regimes), 1.0 / n_regimes as f64);

        let mut prev_ll = f64::NEG_INFINITY;
        for _iter in 0..self.config.max_iter {
            // --- E-step ---
            let log_b = ar_emission_log_probs(
                &series,
                &params.ar_intercepts,
                &params.ar_coefs,
                &params.ar_sigmas,
            );
            let (log_alpha, ll) = forward_log(&startprob, &transmat, &log_b);
            let log_beta = backward_log(&transmat, &log_b);
            let gamma = posterior_gamma(&log_alpha, &log_beta, ll);
            let xi_sum = posterior_xi_sum(&log_alpha, &log_beta, &transmat, &log_b, ll);

            // --- M-step: Markov-chain parameters ---
            startprob = gamma.row(0).to_owned();
            for i in 0..n_regimes {
                let denom: f64 = (0..n_obs - 1).map(|t| gamma[[t, i]]).sum();
                if denom > 0.0 {
                    for j in 0..n_regimes {
                        transmat[[i, j]] = xi_sum[[i, j]] / denom;
                    }
                }
            }

            // --- M-step: AR(1) emissions via weighted least squares ---
            self.update_ar_emissions(&series, &gamma, global_var, &mut params);

            if (ll - prev_ll).abs() < self.config.tol {
                prev_ll = ll;
                break;
            }
            prev_ll = ll;
        }

        Ok(RegimeSwitchingModelTrained {
            startprob,
            transmat,
            params,
            regime_type: RegimeType::AutoregressiveSwitching,
            log_likelihood: prev_ll,
            n_regimes,
            n_features: 1,
        })
    }

    /// M-step update of the AR(1) regime emissions.
    ///
    /// For each regime `k` this performs a responsibility-weighted least-squares
    /// regression of `o_t` on `[1, o_{t-1}]` (`t >= 1`) to obtain `c_k, phi_k`,
    /// then re-estimates the innovation variance `sigma2_k` from the weighted
    /// residuals.
    fn update_ar_emissions(
        &self,
        series: &[f64],
        gamma: &Array2<f64>,
        global_var: f64,
        params: &mut RegimeParameters,
    ) {
        let n_obs = series.len();
        let n_regimes = params.ar_intercepts.len();
        for k in 0..n_regimes {
            // Accumulate the weighted normal-equation moments for the 2x2
            // system [[s_w, s_x], [s_x, s_xx]] [c, phi]^T = [s_y, s_xy]^T.
            let mut s_w = 0.0_f64;
            let mut s_x = 0.0_f64;
            let mut s_xx = 0.0_f64;
            let mut s_y = 0.0_f64;
            let mut s_xy = 0.0_f64;
            for t in 1..n_obs {
                let g = gamma[[t, k]];
                let x = series[t - 1];
                let y = series[t];
                s_w += g;
                s_x += g * x;
                s_xx += g * x * x;
                s_y += g * y;
                s_xy += g * x * y;
            }
            let det = s_w * s_xx - s_x * s_x;
            let (c, phi) = if s_w > 0.0 && det.abs() > 1e-12 {
                let c = (s_y * s_xx - s_xy * s_x) / det;
                let phi = (s_w * s_xy - s_x * s_y) / det;
                (c, phi)
            } else if s_w > 0.0 {
                // Degenerate design (e.g. a near-constant lag): fall back to a
                // zero-lag intercept-only fit.
                (s_y / s_w, 0.0)
            } else {
                // Empty regime: retain the previous parameters.
                (params.ar_intercepts[k], params.ar_coefs[k])
            };

            let mut sigma2 = 0.0_f64;
            if s_w > 0.0 {
                let mut acc = 0.0_f64;
                for t in 1..n_obs {
                    let g = gamma[[t, k]];
                    let resid = series[t] - c - phi * series[t - 1];
                    acc += g * resid * resid;
                }
                sigma2 = acc / s_w + REG_COVAR;
            }
            if !(sigma2.is_finite() && sigma2 > 0.0) {
                sigma2 = global_var;
            }

            params.ar_intercepts[k] = c;
            params.ar_coefs[k] = phi;
            params.ar_sigmas[k] = sigma2;
        }
    }
}

/// A fitted Markov regime-switching model.
#[derive(Debug, Clone)]
pub struct RegimeSwitchingModelTrained {
    startprob: Array1<f64>,
    transmat: Array2<f64>,
    params: RegimeParameters,
    regime_type: RegimeType,
    log_likelihood: f64,
    n_regimes: usize,
    n_features: usize,
}

impl RegimeSwitchingModelTrained {
    /// Initial regime distribution.
    pub fn startprob(&self) -> &Array1<f64> {
        &self.startprob
    }

    /// Regime transition probability matrix.
    pub fn transmat(&self) -> &Array2<f64> {
        &self.transmat
    }

    /// Fitted regime-conditional emission parameters.
    pub fn parameters(&self) -> &RegimeParameters {
        &self.params
    }

    /// Number of latent regimes.
    pub fn n_regimes(&self) -> usize {
        self.n_regimes
    }

    /// Training-set log-likelihood at the final EM iteration.
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Build the emission log-probability matrix for a validated sequence.
    fn emission_log_probs(&self, observations: &ArrayView2<f64>) -> Result<Array2<f64>, RSMError> {
        if observations.ncols() != self.n_features || observations.nrows() == 0 {
            return Err(RSMError::InvalidObservation);
        }
        match self.regime_type {
            RegimeType::AutoregressiveSwitching => {
                let series: Vec<f64> = (0..observations.nrows())
                    .map(|t| observations[[t, 0]])
                    .collect();
                Ok(ar_emission_log_probs(
                    &series,
                    &self.params.ar_intercepts,
                    &self.params.ar_coefs,
                    &self.params.ar_sigmas,
                ))
            }
            _ => gaussian_emission_log_probs(
                observations,
                &self.params.means,
                &self.params.covariances,
            ),
        }
    }

    /// Smoothed regime probabilities `gamma[t, k]` (Kim smoother). Each row sums
    /// to one.
    pub fn smoothed_probabilities(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<Array2<f64>, RSMError> {
        let log_b = self.emission_log_probs(observations)?;
        let (log_alpha, ll) = forward_log(&self.startprob, &self.transmat, &log_b);
        let log_beta = backward_log(&self.transmat, &log_b);
        Ok(posterior_gamma(&log_alpha, &log_beta, ll))
    }

    /// Filtered regime probabilities `P(regime_t = k | o_{1..t})` from the
    /// Hamilton filter. Each row sums to one.
    pub fn filtered_probabilities(
        &self,
        observations: &ArrayView2<f64>,
    ) -> Result<Array2<f64>, RSMError> {
        let log_b = self.emission_log_probs(observations)?;
        let (log_alpha, _ll) = forward_log(&self.startprob, &self.transmat, &log_b);
        let (n_obs, n_regimes) = log_alpha.dim();
        let mut filtered = Array2::<f64>::zeros((n_obs, n_regimes));
        let mut scratch = vec![0.0_f64; n_regimes];
        for t in 0..n_obs {
            for k in 0..n_regimes {
                scratch[k] = log_alpha[[t, k]];
            }
            let norm = logsumexp(&scratch);
            for k in 0..n_regimes {
                filtered[[t, k]] = (log_alpha[[t, k]] - norm).exp();
            }
        }
        Ok(filtered)
    }

    /// Most likely regime sequence via Viterbi decoding.
    pub fn predict(&self, observations: &ArrayView2<f64>) -> Result<Array1<usize>, RSMError> {
        let log_b = self.emission_log_probs(observations)?;
        Ok(viterbi(&self.startprob, &self.transmat, &log_b))
    }

    /// Log-likelihood of an observation sequence under the fitted model.
    pub fn score(&self, observations: &ArrayView2<f64>) -> Result<f64, RSMError> {
        let log_b = self.emission_log_probs(observations)?;
        let (_log_alpha, ll) = forward_log(&self.startprob, &self.transmat, &log_b);
        Ok(ll)
    }
}

/// Builder for [`RegimeSwitchingModel`].
#[derive(Debug, Clone)]
pub struct RegimeSwitchingModelBuilder {
    n_regimes: usize,
    regime_type: RegimeType,
    max_iter: usize,
    tol: f64,
}

impl RegimeSwitchingModelBuilder {
    /// Start configuring a regime-switching model with the given number of
    /// regimes. Defaults: `FullSwitching`, `max_iter = 100`, `tol = 1e-4`.
    pub fn new(n_regimes: usize) -> Self {
        Self {
            n_regimes,
            regime_type: RegimeType::FullSwitching,
            max_iter: 100,
            tol: 1e-4,
        }
    }

    /// Set the structural [`RegimeType`] constraint (default `FullSwitching`).
    pub fn regime_type(mut self, regime_type: RegimeType) -> Self {
        self.regime_type = regime_type;
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

    /// Build the untrained model.
    pub fn build(self) -> RegimeSwitchingModel {
        RegimeSwitchingModel::new(RSMConfig {
            n_regimes: self.n_regimes,
            regime_type: self.regime_type,
            max_iter: self.max_iter,
            tol: self.tol,
        })
    }
}

/// Univariate normal log-density helper exposed for the AR(1) smoke test's
/// internal consistency checks. Kept private to the module.
#[cfg(test)]
fn ar_logpdf_for_test(x: f64, mean: f64, var: f64) -> f64 {
    univariate_normal_logpdf(x, mean, var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Build a two-regime 1-D sequence: a block near 0 followed by a block near
    /// 8, with a tiny alternating jitter to keep the empirical covariance
    /// positive definite.
    fn two_regime_sequence() -> Array2<f64> {
        let mut obs = Array2::<f64>::zeros((30, 1));
        for t in 0..30 {
            let base = if t < 15 { 0.0 } else { 8.0 };
            let jitter = if t % 2 == 0 { 0.1 } else { -0.1 };
            obs[[t, 0]] = base + jitter;
        }
        obs
    }

    /// A univariate AR-switching sequence: a low-mean, near-i.i.d. block followed
    /// by a high-mean, persistent block.
    fn ar_switching_sequence() -> Array2<f64> {
        let mut obs = Array2::<f64>::zeros((40, 1));
        let mut prev = 0.0_f64;
        for t in 0..40 {
            let drift = if t < 20 { 0.0 } else { 5.0 };
            let phi = if t < 20 { 0.0 } else { 0.5 };
            let jitter = if t % 2 == 0 { 0.05 } else { -0.05 };
            let value = drift + phi * (prev - drift) + jitter;
            obs[[t, 0]] = value;
            prev = value;
        }
        obs
    }

    #[test]
    fn test_regime_switching_default_and_builder() {
        let config = RSMConfig::default();
        assert_eq!(config.n_regimes, 2);
        assert_eq!(config.regime_type, RegimeType::FullSwitching);

        let model = RegimeSwitchingModelBuilder::new(3)
            .regime_type(RegimeType::MeanSwitching)
            .max_iter(42)
            .tol(1e-3)
            .build();
        assert_eq!(model.config.n_regimes, 3);
        assert_eq!(model.config.regime_type, RegimeType::MeanSwitching);
        assert_eq!(model.config.max_iter, 42);
        assert_eq!(model.config.tol, 1e-3);
    }

    #[test]
    fn test_regime_switching_mean_recovers_two_regimes() {
        let obs = two_regime_sequence();
        let model = RegimeSwitchingModelBuilder::new(2)
            .regime_type(RegimeType::MeanSwitching)
            .max_iter(100)
            .tol(1e-6)
            .build();
        let trained = model
            .fit(&obs.view())
            .expect("mean-switching fit should succeed");

        // Markov-chain stochasticity.
        let start_sum: f64 = trained.startprob().sum();
        assert!(
            (start_sum - 1.0).abs() < 1e-6,
            "startprob summed to {start_sum}"
        );
        for i in 0..trained.n_regimes() {
            let row_sum: f64 = trained.transmat().row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "transmat row {i} summed to {row_sum}"
            );
        }

        // The two recovered means must straddle the regimes.
        let params = trained.parameters();
        let m0 = params.means[[0, 0]];
        let m1 = params.means[[1, 0]];
        let (lo, hi) = if m0 < m1 { (m0, m1) } else { (m1, m0) };
        assert!(lo < 3.0, "low regime mean was {lo}");
        assert!(hi > 5.0, "high regime mean was {hi}");

        // The shared covariance constraint: both regimes carry the same matrix.
        assert!(
            (params.covariances[0][[0, 0]] - params.covariances[1][[0, 0]]).abs() < 1e-9,
            "mean-switching covariances should be shared"
        );

        // Viterbi assigns valid, distinct regimes to the first and last steps.
        let path = trained
            .predict(&obs.view())
            .expect("predict should succeed");
        assert_eq!(path.len(), obs.nrows());
        assert!(path.iter().all(|&k| k < 2));
        assert_ne!(path[0], path[obs.nrows() - 1]);

        // Smoothed posteriors form valid distributions per timestep.
        let gamma = trained
            .smoothed_probabilities(&obs.view())
            .expect("smoothed_probabilities should succeed");
        assert_eq!(gamma.dim(), (obs.nrows(), 2));
        for t in 0..obs.nrows() {
            let row_sum: f64 = gamma.row(t).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "gamma row {t} summed to {row_sum}"
            );
        }

        // Filtered posteriors are likewise distributions per timestep.
        let filtered = trained
            .filtered_probabilities(&obs.view())
            .expect("filtered_probabilities should succeed");
        for t in 0..obs.nrows() {
            let row_sum: f64 = filtered.row(t).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "filtered row {t} summed to {row_sum}"
            );
        }

        let score = trained.score(&obs.view()).expect("score should succeed");
        assert!(score.is_finite());
        assert!((score - trained.log_likelihood()).abs() < 1e-6);
    }

    #[test]
    fn test_regime_switching_variance_runs() {
        let obs = two_regime_sequence();
        let model = RegimeSwitchingModelBuilder::new(2)
            .regime_type(RegimeType::VarianceSwitching)
            .max_iter(100)
            .tol(1e-6)
            .build();
        let trained = model
            .fit(&obs.view())
            .expect("variance-switching fit should succeed");

        // The shared-mean constraint: both regimes carry the same mean.
        let params = trained.parameters();
        assert!(
            (params.means[[0, 0]] - params.means[[1, 0]]).abs() < 1e-9,
            "variance-switching means should be shared"
        );
        let score = trained.score(&obs.view()).expect("score should succeed");
        assert!(score.is_finite());
    }

    #[test]
    fn test_regime_switching_ar_runs() {
        let obs = ar_switching_sequence();
        let model = RegimeSwitchingModelBuilder::new(2)
            .regime_type(RegimeType::AutoregressiveSwitching)
            .max_iter(100)
            .tol(1e-6)
            .build();
        let trained = model
            .fit(&obs.view())
            .expect("AR-switching fit should succeed");

        let params = trained.parameters();
        assert_eq!(params.ar_intercepts.len(), 2);
        assert_eq!(params.ar_coefs.len(), 2);
        assert_eq!(params.ar_sigmas.len(), 2);
        assert!(params.ar_sigmas.iter().all(|&s| s.is_finite() && s > 0.0));

        let score = trained.score(&obs.view()).expect("score should succeed");
        assert!(score.is_finite());

        let path = trained
            .predict(&obs.view())
            .expect("predict should succeed");
        assert_eq!(path.len(), obs.nrows());
        assert!(path.iter().all(|&k| k < 2));

        // The AR univariate log-density helper is internally consistent: a value
        // at the mean has higher density than one a standard deviation away.
        let at_mean = ar_logpdf_for_test(0.0, 0.0, 1.0);
        let off_mean = ar_logpdf_for_test(1.0, 0.0, 1.0);
        assert!(at_mean > off_mean);
    }

    #[test]
    fn test_regime_switching_rejects_bad_input() {
        // Autoregressive switching requires univariate observations.
        let obs_2d = array![[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let ar_model = RegimeSwitchingModelBuilder::new(2)
            .regime_type(RegimeType::AutoregressiveSwitching)
            .build();
        assert!(matches!(
            ar_model.fit(&obs_2d.view()),
            Err(RSMError::InvalidObservation)
        ));

        // More regimes than observations is rejected.
        let obs_short = array![[0.0], [1.0]];
        let big_model = RegimeSwitchingModelBuilder::new(5).build();
        assert!(matches!(
            big_model.fit(&obs_short.view()),
            Err(RSMError::InvalidRegimes(5))
        ));
    }
}
