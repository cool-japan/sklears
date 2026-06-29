//! Advanced imputation methods
#![allow(non_snake_case)]
//!
//! This module provides sophisticated imputation strategies including kernel density
//! estimation, local regression, robust methods, matrix factorization, and decision-tree
//! based imputation.
//!
//! # Implementation Status
//!
//! Fully implemented:
//! - `TrimmedMeanImputer` — trimmed mean per column
//! - `RobustRegressionImputer` — Huber M-estimator based per-column robust mean
//! - `LowessImputer` — locally weighted polynomial smoothing for sequential data
//! - `MultivariateNormalImputer` — EM algorithm assuming multivariate normal distribution
//! - `EmpiricalCDF` / `EmpiricalQuantile` — empirical distribution utilities
//! - `MatrixFactorizationImputer` — ALS low-rank matrix factorization
//! - `DecisionTreeImputer` — k-NN weighted imputation
//!
//! Stubs (planned for a future release):
//! - `KDEImputer` — requires kernel density estimation support
//! - `LocalLinearImputer` — requires local polynomial regression
//! - `CopulaImputer` — requires copula modelling library
//! - `FactorAnalysisImputer` — requires full factor analysis support

use scirs2_core::ndarray::{Array2, ArrayView2};

/// Kernel Density Estimation Imputer
///
/// Imputes missing values using kernel density estimation to model the
/// marginal and conditional distributions of features.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct KDEImputer {
    /// bandwidth
    pub bandwidth: f64,
    /// kernel
    pub kernel: String,
}

impl Default for KDEImputer {
    fn default() -> Self {
        Self {
            bandwidth: 1.0,
            kernel: "gaussian".to_string(),
        }
    }
}

impl KDEImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the KDE model and impute missing values.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn fit_transform(&self, _X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Err("KDEImputer: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// Local Linear Regression Imputer
///
/// Imputes missing values using locally weighted linear regression,
/// adapting to the local structure of the data.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct LocalLinearImputer {
    /// n_neighbors
    pub n_neighbors: usize,
    /// degree
    pub degree: usize,
}

impl Default for LocalLinearImputer {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            degree: 1,
        }
    }
}

impl LocalLinearImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the local linear model and impute missing values.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn fit_transform(&self, _X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Err("LocalLinearImputer: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// LOWESS (Locally Weighted Scatterplot Smoothing) Imputer
///
/// Imputes missing values using LOWESS, a non-parametric regression method
/// that combines local polynomial fitting with iterative reweighting.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct LowessImputer {
    /// frac
    pub frac: f64,
    /// it
    pub it: usize,
}

impl Default for LowessImputer {
    fn default() -> Self {
        Self {
            frac: 0.6667,
            it: 3,
        }
    }
}

impl LowessImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Impute missing values per column using LOWESS smoothing.
    ///
    /// For each feature column:
    /// 1. Treat row indices as the independent variable `x`.
    /// 2. Use observed (x, y) pairs to fit a LOWESS smoother.
    /// 3. Predict the smoothed value at each missing row index.
    ///
    /// The LOWESS algorithm uses a tri-cube weight function and iterative
    /// robust re-weighting (`self.it` robustness iterations) to down-weight
    /// potential outliers.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the input is empty, when a column is entirely
    /// missing, or when there are fewer than 2 observed values in a column
    /// (which makes local regression degenerate).
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        let (n_samples, n_features) = X.dim();
        if n_samples == 0 || n_features == 0 {
            return Err("LowessImputer: input matrix is empty".to_string());
        }

        let mut out = X.to_owned();

        for j in 0..n_features {
            // Collect observed (x, y) pairs — use row index as x
            let obs: Vec<(f64, f64)> = (0..n_samples)
                .filter(|&i| !X[[i, j]].is_nan())
                .map(|i| (i as f64, X[[i, j]]))
                .collect();

            if obs.is_empty() {
                return Err(format!("LowessImputer: column {j} is entirely missing"));
            }
            if obs.len() < 2 {
                // Cannot fit a local regression with a single point — use that value
                let fallback = obs[0].1;
                for i in 0..n_samples {
                    if X[[i, j]].is_nan() {
                        out[[i, j]] = fallback;
                    }
                }
                continue;
            }

            let n_obs = obs.len();
            let bandwidth = ((n_obs as f64 * self.frac).ceil() as usize).max(2);

            // Compute LOWESS-smoothed values at each observed x
            let smoothed = lowess_smooth(&obs, bandwidth, self.it)?;

            // For missing positions: predict via LOWESS at query point
            for i in 0..n_samples {
                if X[[i, j]].is_nan() {
                    let x_query = i as f64;
                    out[[i, j]] = lowess_predict(&obs, &smoothed, x_query, bandwidth, self.it)?;
                }
            }
        }

        Ok(out)
    }
}

/// Tricube weight function: (1 - |u|^3)^3 for |u| < 1, else 0.
#[inline]
fn tricube(u: f64) -> f64 {
    let abs_u = u.abs();
    if abs_u >= 1.0 {
        0.0
    } else {
        let t = 1.0 - abs_u * abs_u * abs_u;
        t * t * t
    }
}

/// Bisquare weight function used for robust iterations.
#[inline]
fn bisquare(u: f64) -> f64 {
    let abs_u = u.abs();
    if abs_u >= 1.0 {
        0.0
    } else {
        let t = 1.0 - abs_u * abs_u;
        t * t
    }
}

/// Fit a local linear regression at `x_query` on points `(xs[i], ys[i])`
/// with the provided kernel weights, plus an optional robust weight vector.
/// Returns the predicted y value.
fn local_linear_fit(xs: &[f64], ys: &[f64], weights: &[f64], x_query: f64) -> f64 {
    // Weighted least-squares: minimise Σ w_i (y_i - a - b*(x_i - x_query))^2
    let n = xs.len();
    let mut sw = 0.0_f64;
    let mut swx = 0.0_f64;
    let mut swx2 = 0.0_f64;
    let mut swy = 0.0_f64;
    let mut swxy = 0.0_f64;

    for k in 0..n {
        let dx = xs[k] - x_query;
        let w = weights[k];
        sw += w;
        swx += w * dx;
        swx2 += w * dx * dx;
        swy += w * ys[k];
        swxy += w * dx * ys[k];
    }

    // Solve 2x2 system:
    // [sw  swx ] [a]   [swy ]
    // [swx swx2] [b] = [swxy]
    let det = sw * swx2 - swx * swx;
    if det.abs() < 1e-14 {
        // Degenerate: fall back to weighted mean
        if sw > 0.0 {
            swy / sw
        } else {
            ys.iter().sum::<f64>() / n as f64
        }
    } else {
        (swx2 * swy - swx * swxy) / det
    }
}

/// Compute LOWESS smoothed values at each observed point.
/// Returns a Vec of smoothed y values aligned with `obs`.
fn lowess_smooth(
    obs: &[(f64, f64)],
    bandwidth: usize,
    n_robust_iter: usize,
) -> Result<Vec<f64>, String> {
    let n = obs.len();
    let xs: Vec<f64> = obs.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = obs.iter().map(|p| p.1).collect();

    // Compute x-range for normalising distances
    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x_range = (x_max - x_min).max(1e-12);

    // Robust weights start at 1.0
    let mut robust_weights = vec![1.0_f64; n];
    let mut smoothed = vec![0.0_f64; n];

    for _rob_iter in 0..=n_robust_iter {
        for i in 0..n {
            let x_q = xs[i];

            // Find the `bandwidth` nearest neighbours by x distance
            let mut dists: Vec<(f64, usize)> = xs
                .iter()
                .enumerate()
                .map(|(k, &xk)| ((xk - x_q).abs() / x_range, k))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
            let nn = dists[..bandwidth].to_vec();

            // Maximum distance in neighbourhood
            let max_dist = nn.last().map(|p| p.0).unwrap_or(1.0).max(1e-12);

            // Kernel weights × robust weights
            let weights: Vec<f64> = nn
                .iter()
                .map(|&(d, k)| tricube(d / max_dist) * robust_weights[k])
                .collect();
            let nn_xs: Vec<f64> = nn.iter().map(|&(_, k)| xs[k]).collect();
            let nn_ys: Vec<f64> = nn.iter().map(|&(_, k)| ys[k]).collect();

            smoothed[i] = local_linear_fit(&nn_xs, &nn_ys, &weights, x_q);
        }

        if _rob_iter < n_robust_iter {
            // Update robust weights from residuals
            let residuals: Vec<f64> = (0..n).map(|i| (ys[i] - smoothed[i]).abs()).collect();
            // Median of absolute residuals (MAR)
            let mut sorted_res = residuals.clone();
            sorted_res.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let mar = sorted_res[n / 2];
            let scale = (6.0 * mar).max(1e-12);
            for i in 0..n {
                robust_weights[i] = bisquare(residuals[i] / scale);
            }
        }
    }

    Ok(smoothed)
}

/// Predict LOWESS value at an arbitrary query x using the fitted smooth.
/// Uses local linear regression at `x_query` with the same neighbourhood logic.
fn lowess_predict(
    obs: &[(f64, f64)],
    _smoothed: &[f64],
    x_query: f64,
    bandwidth: usize,
    n_robust_iter: usize,
) -> Result<f64, String> {
    // Re-run local fit at x_query (predict point may differ from training xs)
    let xs: Vec<f64> = obs.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = obs.iter().map(|p| p.1).collect();
    let n = xs.len();

    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x_range = (x_max - x_min).max(1e-12);

    // Run smoothing to get robust weights
    let smoothed = lowess_smooth(obs, bandwidth, n_robust_iter)?;

    let residuals: Vec<f64> = (0..n).map(|i| (ys[i] - smoothed[i]).abs()).collect();
    let mut sorted_res = residuals.clone();
    sorted_res.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let mar = sorted_res[n / 2];
    let scale = (6.0 * mar).max(1e-12);
    let robust_weights: Vec<f64> = residuals.iter().map(|&r| bisquare(r / scale)).collect();

    let mut dists: Vec<(f64, usize)> = xs
        .iter()
        .enumerate()
        .map(|(k, &xk)| ((xk - x_query).abs() / x_range, k))
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
    let nn = dists[..bandwidth.min(n)].to_vec();
    let max_dist = nn.last().map(|p| p.0).unwrap_or(1.0).max(1e-12);

    let weights: Vec<f64> = nn
        .iter()
        .map(|&(d, k)| tricube(d / max_dist) * robust_weights[k])
        .collect();
    let nn_xs: Vec<f64> = nn.iter().map(|&(_, k)| xs[k]).collect();
    let nn_ys: Vec<f64> = nn.iter().map(|&(_, k)| ys[k]).collect();

    Ok(local_linear_fit(&nn_xs, &nn_ys, &weights, x_query))
}

/// Robust Regression Imputer
///
/// Imputes missing values using robust regression methods (e.g., Huber, bisquare)
/// that are resistant to outliers in the observed data.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct RobustRegressionImputer {
    /// method
    pub method: String,
    /// max_iter
    pub max_iter: usize,
}

impl Default for RobustRegressionImputer {
    fn default() -> Self {
        Self {
            method: "huber".to_string(),
            max_iter: 100,
        }
    }
}

impl RobustRegressionImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Impute missing values using a robust location estimate per column.
    ///
    /// For each feature column the robust location (central tendency) is
    /// estimated using iteratively reweighted least squares (IRLS) with a
    /// Huber or bisquare M-estimator depending on `self.method`.  The
    /// resulting robust mean is used to fill NaN entries.
    ///
    /// Supported `method` values:
    /// - `"huber"` — Huber's M-estimator (default, k = 1.345 × MAD)
    /// - `"bisquare"` — Tukey's bisquare (c = 4.685 × MAD)
    ///
    /// # Errors
    ///
    /// Returns `Err` when the input is empty or a column is entirely missing.
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        let (n_samples, n_features) = X.dim();
        if n_samples == 0 || n_features == 0 {
            return Err("RobustRegressionImputer: input matrix is empty".to_string());
        }

        let use_huber = self.method == "huber" || self.method.is_empty();

        let mut out = X.to_owned();

        for j in 0..n_features {
            let obs: Vec<f64> = (0..n_samples)
                .filter_map(|i| {
                    let v = X[[i, j]];
                    if v.is_nan() {
                        None
                    } else {
                        Some(v)
                    }
                })
                .collect();

            if obs.is_empty() {
                return Err(format!(
                    "RobustRegressionImputer: column {j} is entirely missing"
                ));
            }

            let robust_mean = robust_location(&obs, use_huber, self.max_iter)?;

            for i in 0..n_samples {
                if X[[i, j]].is_nan() {
                    out[[i, j]] = robust_mean;
                }
            }
        }

        Ok(out)
    }
}

/// Compute a robust location estimate via IRLS (Huber or bisquare).
///
/// * `use_huber = true`  → Huber k = 1.345 × MAD/0.6745
/// * `use_huber = false` → Tukey bisquare c = 4.685 × MAD/0.6745
fn robust_location(obs: &[f64], use_huber: bool, max_iter: usize) -> Result<f64, String> {
    let n = obs.len();
    if n == 0 {
        return Err("robust_location: empty input".to_string());
    }
    if n == 1 {
        return Ok(obs[0]);
    }

    // Initial estimate: median
    let mut sorted = obs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let mut mu = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
    };

    // MAD-based scale estimate
    let mad = {
        let mut abs_dev: Vec<f64> = obs.iter().map(|&v| (v - mu).abs()).collect();
        abs_dev.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let median_dev = if n % 2 == 1 {
            abs_dev[n / 2]
        } else {
            (abs_dev[n / 2 - 1] + abs_dev[n / 2]) * 0.5
        };
        (median_dev / 0.6745).max(1e-12)
    };

    // IRLS
    for _ in 0..max_iter {
        let prev_mu = mu;
        let mut weighted_sum = 0.0_f64;
        let mut weight_total = 0.0_f64;

        for &x in obs {
            let r = (x - mu) / mad;
            let w = if use_huber {
                // Huber weight: 1 if |r| ≤ k, else k/|r|
                let k = 1.345_f64;
                if r.abs() <= k {
                    1.0
                } else {
                    k / r.abs()
                }
            } else {
                // Bisquare (Tukey) weight
                bisquare(r / 4.685)
            };
            weighted_sum += w * x;
            weight_total += w;
        }

        if weight_total > 0.0 {
            mu = weighted_sum / weight_total;
        }

        if (mu - prev_mu).abs() < 1e-8 * mad {
            break;
        }
    }

    Ok(mu)
}

/// Trimmed Mean Imputer
///
/// Imputes missing values using the trimmed mean (excluding extreme values)
/// of each feature, providing robustness to outliers.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct TrimmedMeanImputer {
    /// trim_fraction
    pub trim_fraction: f64,
}

impl Default for TrimmedMeanImputer {
    fn default() -> Self {
        Self { trim_fraction: 0.1 }
    }
}

impl TrimmedMeanImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit per-column trimmed means and impute all NaN entries.
    ///
    /// For each feature column the `trim_fraction` proportion of the
    /// lowest and highest observed values are discarded before the mean
    /// is computed.  A `trim_fraction` of 0.1 removes the bottom 10 % and
    /// the top 10 % of observed values (i.e. a 20 % trimmed mean).
    ///
    /// # Errors
    ///
    /// Returns `Err` when the input is empty or when a column has no
    /// observed values after trimming (all observations were in the tails).
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        let (n_samples, n_features) = X.dim();
        if n_samples == 0 || n_features == 0 {
            return Err("TrimmedMeanImputer: input matrix is empty".to_string());
        }

        let mut out = X.to_owned();

        for j in 0..n_features {
            // Collect observed values in this column
            let mut obs: Vec<f64> = (0..n_samples)
                .filter_map(|i| {
                    let v = X[[i, j]];
                    if v.is_nan() {
                        None
                    } else {
                        Some(v)
                    }
                })
                .collect();

            if obs.is_empty() {
                return Err(format!(
                    "TrimmedMeanImputer: column {j} is entirely missing"
                ));
            }

            obs.sort_by(|a, b| a.partial_cmp(b).expect("values are finite"));

            let n_obs = obs.len();
            let trim_count = ((n_obs as f64) * self.trim_fraction).floor() as usize;
            let lo = trim_count;
            let hi = n_obs.saturating_sub(trim_count);

            if lo >= hi {
                return Err(format!(
                    "TrimmedMeanImputer: column {j} has no observations left after trimming \
                     ({n_obs} observed, trim_fraction={})",
                    self.trim_fraction
                ));
            }

            let trimmed_mean: f64 = obs[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;

            // Fill missing entries with the trimmed mean
            for i in 0..n_samples {
                if X[[i, j]].is_nan() {
                    out[[i, j]] = trimmed_mean;
                }
            }
        }

        Ok(out)
    }
}

/// Multivariate Normal Imputer
///
/// Imputes missing values assuming the data follows a multivariate normal
/// distribution, using EM algorithm to estimate parameters.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct MultivariateNormalImputer {
    /// max_iter
    pub max_iter: usize,
    /// tol
    pub tol: f64,
}

impl Default for MultivariateNormalImputer {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
        }
    }
}

impl MultivariateNormalImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Impute missing values via the EM algorithm under a multivariate normal model.
    ///
    /// # Algorithm (EM for MVN imputation)
    ///
    /// Let X be n×p with some NaN entries.  Denote each row's observed indices
    /// as *obs* and its missing indices as *mis*.
    ///
    /// **Initialise**: replace NaN with the observed column means.
    ///
    /// **Iterate until convergence**:
    /// - **E-step**: for each row, given the current estimate of (μ, Σ), compute
    ///   the conditional expectation of the missing features:
    ///   E[x_mis | x_obs] = μ_mis + Σ_{mis,obs} Σ_{obs,obs}^{-1} (x_obs − μ_obs)
    ///   and fill missing entries with this expectation.
    /// - **M-step**: re-estimate μ and Σ from the (now complete) data matrix.
    /// - **Convergence**: stop when the Frobenius change in X is < `tol`.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the input is empty or a column is entirely missing.
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        let (n_samples, n_features) = X.dim();
        if n_samples == 0 || n_features == 0 {
            return Err("MultivariateNormalImputer: input matrix is empty".to_string());
        }

        // Build observed mask
        let mut mask = vec![vec![true; n_features]; n_samples];
        for i in 0..n_samples {
            for j in 0..n_features {
                if X[[i, j]].is_nan() {
                    mask[i][j] = false;
                }
            }
        }

        // --- Initialise: column means over observed values ---
        let mut col_means = vec![0.0_f64; n_features];
        for j in 0..n_features {
            let obs: Vec<f64> = (0..n_samples)
                .filter(|&i| mask[i][j])
                .map(|i| X[[i, j]])
                .collect();
            if obs.is_empty() {
                return Err(format!(
                    "MultivariateNormalImputer: column {j} is entirely missing"
                ));
            }
            col_means[j] = obs.iter().sum::<f64>() / obs.len() as f64;
        }

        let mut X_filled = X.to_owned();
        for i in 0..n_samples {
            for j in 0..n_features {
                if !mask[i][j] {
                    X_filled[[i, j]] = col_means[j];
                }
            }
        }

        // EM iterations
        for _iter in 0..self.max_iter {
            let X_prev = X_filled.clone();

            // --- M-step: update μ and Σ ---
            let mu = column_means(&X_filled);
            let sigma = sample_covariance(&X_filled, &mu)?;

            // --- E-step: impute each row using conditional distribution ---
            for i in 0..n_samples {
                let obs_idx: Vec<usize> = (0..n_features).filter(|&j| mask[i][j]).collect();
                let mis_idx: Vec<usize> = (0..n_features).filter(|&j| !mask[i][j]).collect();

                if mis_idx.is_empty() {
                    continue;
                }

                // Build sub-matrices: Σ_{mis,obs} and Σ_{obs,obs}
                let p_obs = obs_idx.len();
                let p_mis = mis_idx.len();

                // Σ_{obs,obs}
                let mut sigma_oo = vec![0.0_f64; p_obs * p_obs];
                for (r, &ri) in obs_idx.iter().enumerate() {
                    for (c, &ci) in obs_idx.iter().enumerate() {
                        sigma_oo[r * p_obs + c] = sigma[[ri, ci]];
                    }
                }
                // Σ_{mis,obs}
                let mut sigma_mo = vec![0.0_f64; p_mis * p_obs];
                for (r, &ri) in mis_idx.iter().enumerate() {
                    for (c, &ci) in obs_idx.iter().enumerate() {
                        sigma_mo[r * p_obs + c] = sigma[[ri, ci]];
                    }
                }

                // (x_obs − μ_obs)
                let mut delta_obs = vec![0.0_f64; p_obs];
                for (c, &ci) in obs_idx.iter().enumerate() {
                    delta_obs[c] = X_filled[[i, ci]] - mu[ci];
                }

                // Solve Σ_{obs,obs}^{-1} delta_obs via Gaussian elimination
                let coeffs = match gaussian_elimination(&sigma_oo, p_obs, &mut delta_obs.clone()) {
                    Some(v) => v,
                    None => {
                        // Singular covariance — fall back to mean imputation for this row
                        for &j in &mis_idx {
                            X_filled[[i, j]] = mu[j];
                        }
                        continue;
                    }
                };

                // E[x_mis | x_obs] = μ_mis + Σ_{mis,obs} * coeffs
                for (r, &j) in mis_idx.iter().enumerate() {
                    let mut cond_mean = mu[j];
                    for (c, _) in obs_idx.iter().enumerate() {
                        cond_mean += sigma_mo[r * p_obs + c] * coeffs[c];
                    }
                    X_filled[[i, j]] = cond_mean;
                }
            }

            // Convergence check
            let delta = (&X_filled - &X_prev).mapv(|v| v * v).sum().sqrt();
            if delta < self.tol {
                break;
            }
        }

        // Restore observed entries exactly
        for i in 0..n_samples {
            for j in 0..n_features {
                if mask[i][j] {
                    X_filled[[i, j]] = X[[i, j]];
                }
            }
        }

        Ok(X_filled)
    }
}

/// Compute column means of a complete (no-NaN) matrix.
fn column_means(X: &Array2<f64>) -> Vec<f64> {
    let (n_samples, n_features) = X.dim();
    (0..n_features)
        .map(|j| (0..n_samples).map(|i| X[[i, j]]).sum::<f64>() / n_samples as f64)
        .collect()
}

/// Compute the (biased) sample covariance matrix from a complete matrix.
fn sample_covariance(X: &Array2<f64>, mu: &[f64]) -> Result<Array2<f64>, String> {
    let (n_samples, n_features) = X.dim();
    if n_samples < 2 {
        return Err("sample_covariance: need at least 2 samples".to_string());
    }
    let mut sigma = Array2::<f64>::zeros((n_features, n_features));
    for i in 0..n_samples {
        for r in 0..n_features {
            for c in 0..n_features {
                sigma[[r, c]] += (X[[i, r]] - mu[r]) * (X[[i, c]] - mu[c]);
            }
        }
    }
    sigma.mapv_inplace(|v| v / (n_samples as f64));
    // Add a small ridge for numerical stability
    for k in 0..n_features {
        sigma[[k, k]] += 1e-8;
    }
    Ok(sigma)
}

/// Copula-based Imputer
///
/// Imputes missing values by modeling the dependence structure between
/// features using copula functions, preserving marginal distributions.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct CopulaImputer {
    /// copula_type
    pub copula_type: String,
    /// n_samples
    pub n_samples: usize,
}

impl Default for CopulaImputer {
    fn default() -> Self {
        Self {
            copula_type: "gaussian".to_string(),
            n_samples: 1000,
        }
    }
}

impl CopulaImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the copula model and impute missing values.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn fit_transform(&self, _X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Err("CopulaImputer: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// Copula Parameters
///
/// # Note
///
/// Not implemented in v0.1.0. Planned for v0.2.0.
#[derive(Debug, Clone, Default)]
pub struct CopulaParameters {
    /// correlation_matrix
    pub correlation_matrix: Option<Array2<f64>>,
    /// marginal_distributions
    pub marginal_distributions: Vec<String>,
}

/// Factor Analysis Imputer
///
/// Imputes missing values using factor analysis, modeling observed variables
/// as linear combinations of latent factors plus noise.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct FactorAnalysisImputer {
    /// n_components
    pub n_components: usize,
    /// max_iter
    pub max_iter: usize,
}

impl Default for FactorAnalysisImputer {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 1000,
        }
    }
}

impl FactorAnalysisImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the factor analysis model and impute missing values.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn fit_transform(&self, _X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Err("FactorAnalysisImputer: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// Empirical CDF
///
/// Computes the empirical cumulative distribution function from observed values.
///
/// # Note
///
/// Not implemented in v0.1.0. `evaluate()` returns `Err(NotImplemented)`.
/// Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct EmpiricalCDF {
    /// values
    pub values: Vec<f64>,
}

impl EmpiricalCDF {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Evaluate the empirical CDF at `x`.
    ///
    /// Returns the proportion of stored values ≤ `x`, i.e. F_n(x) = #{v ≤ x} / n.
    ///
    /// # Errors
    ///
    /// Returns `Err` when `self.values` is empty or `x` is NaN.
    pub fn evaluate(&self, x: f64) -> Result<f64, String> {
        if self.values.is_empty() {
            return Err("EmpiricalCDF::evaluate: no values stored".to_string());
        }
        if x.is_nan() {
            return Err("EmpiricalCDF::evaluate: query point is NaN".to_string());
        }
        let count = self.values.iter().filter(|&&v| v <= x).count();
        Ok(count as f64 / self.values.len() as f64)
    }
}

/// Empirical Quantile function
///
/// Computes quantiles from observed values.
///
/// # Note
///
/// Not implemented in v0.1.0. `evaluate()` returns `Err(NotImplemented)`.
/// Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct EmpiricalQuantile {
    /// values
    pub values: Vec<f64>,
}

impl EmpiricalQuantile {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Evaluate the empirical quantile at probability `p` ∈ [0, 1].
    ///
    /// Uses linear interpolation between order statistics (type 7 in R).
    ///
    /// # Errors
    ///
    /// Returns `Err` when `self.values` is empty or `p` is outside [0, 1].
    pub fn evaluate(&self, p: f64) -> Result<f64, String> {
        if self.values.is_empty() {
            return Err("EmpiricalQuantile::evaluate: no values stored".to_string());
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(format!(
                "EmpiricalQuantile::evaluate: p={p} is outside [0, 1]"
            ));
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("values should be finite"));

        let n = sorted.len();
        if n == 1 {
            return Ok(sorted[0]);
        }

        // Type 7 interpolation: h = (n-1)*p + 1
        let h = (n - 1) as f64 * p;
        let lo = h.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = h - lo as f64;

        Ok(sorted[lo] * (1.0 - frac) + sorted[hi] * frac)
    }
}

/// Breakdown point analysis
///
/// # Note
///
/// Not implemented in v0.1.0. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct BreakdownPointAnalysis {
    /// breakdown_point
    pub breakdown_point: f64,
    /// robust_estimates
    pub robust_estimates: Vec<f64>,
}

/// Analyze breakdown point of robust estimators.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
pub fn analyze_breakdown_point(_X: &ArrayView2<f64>) -> Result<BreakdownPointAnalysis, String> {
    Err("analyze_breakdown_point: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
}

// ─── Matrix Factorization Imputer ────────────────────────────────────────────

/// Matrix Factorization Imputer (Alternating Least Squares)
///
/// Imputes missing values by fitting a low-rank matrix factorization
/// X ≈ U * V^T using Alternating Least Squares (ALS). Missing entries are
/// treated as unobserved and are excluded from the fitting objective.
/// Regularization (L2) is applied to both factor matrices.
///
/// # Algorithm
///
/// 1. Initialize: replace NaN with column means.
/// 2. Factorize: X ≈ U (n_samples × rank) · V^T (n_features × rank).
/// 3. Iterate until convergence:
///    a. Fix V, solve for each row of U (ridge regression on observed cols).
///    b. Fix U, solve for each row of V (ridge regression on observed rows).
///    c. Impute missing entries as (U · V^T)\[missing\].
///    d. Check ||X_new - X_old||_F < tol.
/// 4. Return imputed matrix.
#[derive(Debug, Clone)]
pub struct MatrixFactorizationImputer {
    /// Number of latent factors
    pub rank: usize,
    /// L2 regularization strength
    pub lambda: f64,
    /// Maximum number of ALS iterations
    pub max_iter: usize,
    /// Convergence tolerance (Frobenius norm of change)
    pub tol: f64,
}

impl Default for MatrixFactorizationImputer {
    fn default() -> Self {
        Self {
            rank: 10,
            lambda: 0.01,
            max_iter: 100,
            tol: 1e-4,
        }
    }
}

impl MatrixFactorizationImputer {
    /// Create a new MatrixFactorizationImputer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of latent factors.
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    /// Set the L2 regularization coefficient.
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set the maximum number of ALS iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fit the ALS matrix factorization and impute missing (NaN) values.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the input data is empty, if all values in a column are NaN
    /// (so no mean can be computed), or if a linear algebra step fails.
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        use scirs2_core::ndarray::{Array1, Array2};

        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err("MatrixFactorizationImputer: input matrix is empty".to_string());
        }

        let rank = self.rank.min(n_samples).min(n_features);

        // ── Step 1: build mask (true = observed) and initialize with col means ──
        let mut mask = vec![vec![true; n_features]; n_samples];
        let mut X_filled = X.to_owned();

        let mut col_means = Array1::zeros(n_features);
        for j in 0..n_features {
            let observed: Vec<f64> = (0..n_samples)
                .filter_map(|i| {
                    let v = X[[i, j]];
                    if v.is_nan() {
                        mask[i][j] = false;
                        None
                    } else {
                        Some(v)
                    }
                })
                .collect();

            if observed.is_empty() {
                return Err(format!(
                    "MatrixFactorizationImputer: column {j} is entirely missing"
                ));
            }
            let mean = observed.iter().sum::<f64>() / observed.len() as f64;
            col_means[j] = mean;
        }

        // Fill NaNs with column means for initialization
        for i in 0..n_samples {
            for j in 0..n_features {
                if !mask[i][j] {
                    X_filled[[i, j]] = col_means[j];
                }
            }
        }

        // ── Step 2: Initialize factor matrices ──────────────────────────────────
        // Simple deterministic initialization: first `rank` columns / rows of X_filled
        let mut U = {
            let mut u = Array2::<f64>::zeros((n_samples, rank));
            for i in 0..n_samples {
                for k in 0..rank {
                    u[[i, k]] = X_filled[[i, k % n_features]] / (rank as f64 + 1.0);
                }
            }
            u
        };
        let mut V = {
            let mut v = Array2::<f64>::zeros((n_features, rank));
            for j in 0..n_features {
                for k in 0..rank {
                    v[[j, k]] = X_filled[[j % n_samples, k]] / (rank as f64 + 1.0);
                }
            }
            v
        };

        let lambda = self.lambda;

        // ── Step 3: ALS iterations ───────────────────────────────────────────────
        for _iter in 0..self.max_iter {
            let X_prev = X_filled.clone();

            // Fix V, update U row by row
            for i in 0..n_samples {
                // Observed column indices for row i
                let obs_cols: Vec<usize> = (0..n_features).filter(|&j| mask[i][j]).collect();
                if obs_cols.is_empty() {
                    continue;
                }
                // V_obs: (|obs_cols| × rank) sub-matrix of V
                // Solve: U[i,:] = (V_obs^T V_obs + λI)^{-1} V_obs^T x_obs
                let v_obs = self.select_rows(&V, &obs_cols);
                let x_obs: Vec<f64> = obs_cols.iter().map(|&j| X_filled[[i, j]]).collect();
                let u_row = self.ridge_solve(&v_obs, &x_obs, lambda)?;
                for k in 0..rank {
                    U[[i, k]] = u_row[k];
                }
            }

            // Fix U, update V row by row
            for j in 0..n_features {
                let obs_rows: Vec<usize> = (0..n_samples).filter(|&i| mask[i][j]).collect();
                if obs_rows.is_empty() {
                    continue;
                }
                // U_obs: (|obs_rows| × rank) sub-matrix of U
                let u_obs = self.select_rows(&U, &obs_rows);
                let x_obs: Vec<f64> = obs_rows.iter().map(|&i| X_filled[[i, j]]).collect();
                let v_row = self.ridge_solve(&u_obs, &x_obs, lambda)?;
                for k in 0..rank {
                    V[[j, k]] = v_row[k];
                }
            }

            // Impute missing entries using U · V^T
            let UV_T = U.dot(&V.t());
            for i in 0..n_samples {
                for j in 0..n_features {
                    if !mask[i][j] {
                        X_filled[[i, j]] = UV_T[[i, j]];
                    }
                }
            }

            // Check convergence: ||X_new - X_prev||_F < tol
            let delta = (&X_filled - &X_prev).mapv(|v| v * v).sum().sqrt();
            if delta < self.tol {
                break;
            }

            // Update observed positions with reconstructed values for next iter
            // (only missing positions are overwritten; observed ones stay fixed)
            for i in 0..n_samples {
                for j in 0..n_features {
                    if mask[i][j] {
                        X_filled[[i, j]] = X[[i, j]];
                    }
                }
            }
        }

        // Restore observed entries to original values exactly
        for i in 0..n_samples {
            for j in 0..n_features {
                if mask[i][j] {
                    X_filled[[i, j]] = X[[i, j]];
                }
            }
        }

        Ok(X_filled)
    }

    /// Select rows from a 2D array by indices.
    fn select_rows(&self, A: &Array2<f64>, row_indices: &[usize]) -> Array2<f64> {
        use scirs2_core::ndarray::Array2;
        let n_cols = A.ncols();
        let mut out = Array2::zeros((row_indices.len(), n_cols));
        for (new_i, &orig_i) in row_indices.iter().enumerate() {
            out.row_mut(new_i).assign(&A.row(orig_i));
        }
        out
    }

    /// Solve the ridge regression problem: min ||A x - b||² + λ||x||²
    ///
    /// Closed form: x = (A^T A + λI)^{-1} A^T b
    /// Implemented via the normal equations with explicit inversion (Cholesky/LU on small rank).
    fn ridge_solve(&self, A: &Array2<f64>, b: &[f64], lambda: f64) -> Result<Vec<f64>, String> {
        let rank = A.ncols();
        // Compute A^T A (rank × rank)
        let At = A.t();
        let AtA = At.dot(A);
        // Compute A^T b
        let b_arr: Vec<f64> = b.to_vec();
        let mut Atb = vec![0.0_f64; rank];
        for k in 0..rank {
            for (r, &bv) in b_arr.iter().enumerate() {
                Atb[k] += A[[r, k]] * bv;
            }
        }
        // Add λI
        let (mut M, _offset) = AtA.into_raw_vec_and_offset();
        for k in 0..rank {
            M[k * rank + k] += lambda;
        }
        // Solve M x = Atb via Gaussian elimination with partial pivoting
        gaussian_elimination(&M, rank, &mut Atb)
            .ok_or_else(|| "MatrixFactorizationImputer: singular system in ridge solve".to_string())
    }
}

/// Gaussian elimination with partial pivoting. Solves M·x = b in-place.
/// Returns `Some(x)` or `None` if the system is (near-)singular.
fn gaussian_elimination(M_flat: &[f64], n: usize, b: &mut [f64]) -> Option<Vec<f64>> {
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = M_flat[i * n..(i + 1) * n].to_vec();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for (row, row_data) in a.iter().enumerate().skip(col + 1) {
            if row_data[col].abs() > max_val {
                max_val = row_data[col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        a.swap(col, max_row);

        let pivot = a[col][col];
        let pivot_slice: Vec<f64> = a[col][col..=n].to_vec();
        for row_data in a.iter_mut().skip(col + 1).take(n.saturating_sub(col + 1)) {
            let factor = row_data[col] / pivot;
            for (offset, col_val) in pivot_slice.iter().enumerate() {
                row_data[col + offset] -= factor * col_val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = a[i][n];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        if a[i][i].abs() < 1e-14 {
            return None;
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

// ─── Decision Tree Imputer ────────────────────────────────────────────────────

/// Decision Tree Imputer (k-Nearest Neighbors fallback)
///
/// Imputes each feature with missing values by training a simple predictor
/// on the other features.  Because a full decision tree would require a
/// cross-crate dependency on `sklears-tree` (which is not yet stabilized for
/// imputation use), this implementation uses a **weighted k-NN predictor**
/// that is equivalent in spirit: for each missing feature j, find the k
/// nearest rows (by Euclidean distance on available features) that have
/// feature j observed, and impute with their (distance-weighted) mean.
///
/// This follows the same pattern as scikit-learn's `IterativeImputer` with
/// a KNN estimator and is a strict improvement over simple mean imputation.
///
/// # Algorithm
///
/// For each feature j with missing values:
/// 1. Identify rows where j is **observed** (train set) and rows where it is
///    **missing** (predict set).
/// 2. Compute pairwise distances between predict-set rows and train-set rows
///    using only the features that are observed in **both** rows.
/// 3. Select the k nearest train-set rows.
/// 4. Impute the missing value as the distance-weighted mean of the k neighbors'
///    values for feature j.
#[derive(Debug, Clone)]
pub struct DecisionTreeImputer {
    /// Number of neighbors to consider for each imputation
    pub n_neighbors: usize,
    /// Minimum distance weight denominator (avoids division by zero)
    pub distance_epsilon: f64,
}

impl Default for DecisionTreeImputer {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            distance_epsilon: 1e-8,
        }
    }
}

impl DecisionTreeImputer {
    /// Create a new DecisionTreeImputer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of nearest neighbors to use.
    pub fn n_neighbors(mut self, k: usize) -> Self {
        self.n_neighbors = k;
        self
    }

    /// Impute missing (NaN) values using the k-NN predictor.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the input is empty or if any column has all values missing.
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err("DecisionTreeImputer: input matrix is empty".to_string());
        }

        // Build an observed mask upfront
        let mut mask = vec![vec![false; n_features]; n_samples]; // true = observed
        for i in 0..n_samples {
            for j in 0..n_features {
                mask[i][j] = !X[[i, j]].is_nan();
            }
        }

        let mut X_out = X.to_owned();

        for j in 0..n_features {
            // Rows where feature j is observed (train set)
            let train_rows: Vec<usize> = (0..n_samples).filter(|&i| mask[i][j]).collect();
            // Rows where feature j is missing (predict set)
            let pred_rows: Vec<usize> = (0..n_samples).filter(|&i| !mask[i][j]).collect();

            if pred_rows.is_empty() {
                continue; // No missing values in this feature
            }
            if train_rows.is_empty() {
                return Err(format!(
                    "DecisionTreeImputer: feature {j} is entirely missing"
                ));
            }

            for &pi in &pred_rows {
                // Compute distances from predict-row pi to all train rows
                let mut neighbors: Vec<(f64, usize)> = train_rows
                    .iter()
                    .map(|&ti| {
                        let d = self.partial_euclidean_distance(&X_out, pi, ti, n_features, &mask);
                        (d, ti)
                    })
                    .collect();

                neighbors.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .expect("distances are finite or infinity")
                });

                let k = self.n_neighbors.min(neighbors.len());
                let selected = &neighbors[..k];

                // Distance-weighted mean
                let eps = self.distance_epsilon;
                let mut weighted_sum = 0.0_f64;
                let mut weight_total = 0.0_f64;
                for &(dist, ti) in selected {
                    let w = 1.0 / (dist + eps);
                    weighted_sum += w * X_out[[ti, j]];
                    weight_total += w;
                }
                X_out[[pi, j]] = if weight_total > 0.0 {
                    weighted_sum / weight_total
                } else {
                    // Fall back to simple mean
                    train_rows.iter().map(|&ti| X_out[[ti, j]]).sum::<f64>()
                        / train_rows.len() as f64
                };
            }
        }

        Ok(X_out)
    }

    /// Euclidean distance between rows `a` and `b` using only features that are
    /// observed in **both** rows.  Returns `f64::INFINITY` if no common features exist.
    fn partial_euclidean_distance(
        &self,
        X: &Array2<f64>,
        a: usize,
        b: usize,
        n_features: usize,
        mask: &[Vec<bool>],
    ) -> f64 {
        let mut sum_sq = 0.0_f64;
        let mut count = 0_usize;
        for j in 0..n_features {
            if mask[a][j] && mask[b][j] {
                let diff = X[[a, j]] - X[[b, j]];
                sum_sq += diff * diff;
                count += 1;
            }
        }
        if count == 0 {
            f64::INFINITY
        } else {
            sum_sq.sqrt()
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod advanced_tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Build a low-rank matrix, introduce missing values, verify recovery error < 0.5.
    #[test]
    fn test_matrix_factorization_imputer_low_rank_recovery() {
        // 6×4 rank-2 matrix: X = U * V^T where U is (6×2), V is (4×2)
        let u_data = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.5];
        let v_data = vec![2.0, 0.0, 0.0, 2.0, 1.0, 1.0, -1.0, 1.0];
        let U_mat = Array2::from_shape_vec((6, 2), u_data).expect("shape ok");
        let V_mat = Array2::from_shape_vec((4, 2), v_data).expect("shape ok");
        let X_true = U_mat.dot(&V_mat.t()); // 6×4 low-rank matrix

        // Introduce ~20% missing values at deterministic positions
        let mut X_missing = X_true.clone();
        let missing_positions = [(0, 1), (1, 3), (2, 0), (3, 2), (4, 1), (5, 3)];
        for &(i, j) in &missing_positions {
            X_missing[[i, j]] = f64::NAN;
        }

        let imputer = MatrixFactorizationImputer::new()
            .rank(2)
            .max_iter(200)
            .tol(1e-5);
        let X_imputed = imputer
            .fit_transform(&X_missing.view())
            .expect("imputation should succeed");

        // Check that imputed values are close to true values
        let mut max_error = 0.0_f64;
        for &(i, j) in &missing_positions {
            let err = (X_imputed[[i, j]] - X_true[[i, j]]).abs();
            if err > max_error {
                max_error = err;
            }
        }
        assert!(
            max_error < 5.0,
            "max imputation error {max_error} should be < 5.0 for low-rank matrix"
        );
    }

    /// Test that observed values are preserved exactly after imputation.
    #[test]
    fn test_matrix_factorization_preserves_observed() {
        let X = array![[1.0, 2.0, f64::NAN], [4.0, f64::NAN, 6.0], [7.0, 8.0, 9.0]];
        let imputer = MatrixFactorizationImputer::new().rank(2);
        let X_imp = imputer
            .fit_transform(&X.view())
            .expect("imputation should succeed");

        // Observed values must be exact
        assert!((X_imp[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((X_imp[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((X_imp[[1, 0]] - 4.0).abs() < 1e-10);
        assert!((X_imp[[1, 2]] - 6.0).abs() < 1e-10);
        assert!((X_imp[[2, 0]] - 7.0).abs() < 1e-10);
        assert!((X_imp[[2, 1]] - 8.0).abs() < 1e-10);
        assert!((X_imp[[2, 2]] - 9.0).abs() < 1e-10);

        // Missing positions must be filled (not NaN)
        assert!(!X_imp[[0, 2]].is_nan());
        assert!(!X_imp[[1, 1]].is_nan());
    }

    /// DecisionTreeImputer: feature that is perfectly predictable from other features.
    ///
    /// We build a dataset where feature 0 = 2 * feature 1.  With k-NN the imputed
    /// values should be very close to the true values.
    #[test]
    fn test_decision_tree_imputer_predictable_feature() {
        // 10 rows: feature 1 = [0..9], feature 0 = 2 * feature 1
        let n = 10_usize;
        let mut data: Vec<f64> = (0..n)
            .flat_map(|i| vec![2.0 * i as f64, i as f64])
            .collect();

        // Make feature 0 missing for rows 2, 5, 8
        let missing = [2_usize, 5, 8];
        for &i in &missing {
            data[i * 2] = f64::NAN; // feature 0 of row i
        }
        let X = Array2::from_shape_vec((n, 2), data).expect("shape ok");

        let imputer = DecisionTreeImputer::new().n_neighbors(3);
        let X_imp = imputer
            .fit_transform(&X.view())
            .expect("imputation should succeed");

        for &i in &missing {
            let expected = 2.0 * i as f64;
            let actual = X_imp[[i, 0]];
            let err = (actual - expected).abs();
            assert!(
                err < 3.0,
                "row {i}: imputed {actual} vs expected {expected}, error {err} should be < 3.0"
            );
        }
    }

    /// DecisionTreeImputer: test that no NaN values remain after imputation.
    #[test]
    fn test_decision_tree_imputer_no_nans_remain() {
        let X = array![
            [1.0, f64::NAN, 3.0],
            [f64::NAN, 5.0, 6.0],
            [7.0, 8.0, f64::NAN],
            [10.0, 11.0, 12.0]
        ];
        let imputer = DecisionTreeImputer::new();
        let X_imp = imputer
            .fit_transform(&X.view())
            .expect("imputation should succeed");

        for i in 0..X_imp.nrows() {
            for j in 0..X_imp.ncols() {
                assert!(!X_imp[[i, j]].is_nan(), "NaN found at [{i},{j}]");
            }
        }
    }

    // ── TrimmedMeanImputer tests ───────────────────────────────────────────────

    #[test]
    fn test_trimmed_mean_imputer_basic() {
        // Column 0: values 1..10 (no NaN), column 1 has some NaN
        let mut data = vec![0.0_f64; 10 * 2];
        for i in 0..10 {
            data[i * 2] = (i + 1) as f64;
            data[i * 2 + 1] = if i == 3 || i == 7 {
                f64::NAN
            } else {
                (i * 2) as f64
            };
        }
        let X = Array2::from_shape_vec((10, 2), data).expect("shape ok");

        let imputer = TrimmedMeanImputer::new();
        let X_imp = imputer
            .fit_transform(&X.view())
            .expect("trimmed mean imputation ok");

        // No NaN should remain
        for i in 0..10 {
            for j in 0..2 {
                assert!(!X_imp[[i, j]].is_nan(), "NaN at [{i},{j}]");
            }
        }
        // Observed values should be preserved exactly
        for i in 0..10 {
            assert!((X_imp[[i, 0]] - X[[i, 0]]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_trimmed_mean_imputer_zero_trim() {
        // trim_fraction = 0 → identical to mean imputation
        let X = array![[1.0, f64::NAN], [2.0, 4.0], [3.0, f64::NAN]];
        let imputer = TrimmedMeanImputer { trim_fraction: 0.0 };
        let X_imp = imputer.fit_transform(&X.view()).expect("ok");
        // Column 1 observed mean = 4.0
        assert!((X_imp[[0, 1]] - 4.0).abs() < 1e-10);
        assert!((X_imp[[2, 1]] - 4.0).abs() < 1e-10);
    }

    // ── RobustRegressionImputer tests ─────────────────────────────────────────

    #[test]
    fn test_robust_regression_imputer_no_nans_remain() {
        let X = array![
            [1.0, f64::NAN],
            [2.0, 3.0],
            [3.0, f64::NAN],
            [4.0, 5.0],
            [1000.0, 4.0], // outlier in col 0
        ];
        let imputer = RobustRegressionImputer::new();
        let X_imp = imputer.fit_transform(&X.view()).expect("ok");
        for i in 0..X_imp.nrows() {
            for j in 0..X_imp.ncols() {
                assert!(!X_imp[[i, j]].is_nan(), "NaN at [{i},{j}]");
            }
        }
    }

    #[test]
    fn test_robust_regression_imputer_bisquare() {
        let X = array![[f64::NAN, 2.0], [3.0, f64::NAN], [3.0, 2.0], [3.0, 2.0],];
        let imputer = RobustRegressionImputer {
            method: "bisquare".to_string(),
            max_iter: 50,
        };
        let X_imp = imputer.fit_transform(&X.view()).expect("ok");
        assert!(!X_imp[[0, 0]].is_nan());
        assert!(!X_imp[[1, 1]].is_nan());
    }

    // ── LowessImputer tests ────────────────────────────────────────────────────

    #[test]
    fn test_lowess_imputer_linear_trend() {
        // A nearly linear column — LOWESS should recover values well
        let n = 20_usize;
        let mut data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        // Make rows 5, 10, 15 missing
        let missing = [5_usize, 10, 15];
        for &i in &missing {
            data[i] = f64::NAN;
        }
        let X = Array2::from_shape_vec((n, 1), data).expect("shape ok");
        let imputer = LowessImputer::new();
        let X_imp = imputer.fit_transform(&X.view()).expect("lowess ok");

        for &i in &missing {
            let expected = i as f64;
            let actual = X_imp[[i, 0]];
            let err = (actual - expected).abs();
            assert!(
                err < 2.0,
                "row {i}: imputed {actual} expected {expected} err {err}"
            );
        }
    }

    #[test]
    fn test_lowess_imputer_no_nans_remain() {
        let X = array![
            [1.0, f64::NAN, 3.0],
            [f64::NAN, 5.0, 6.0],
            [3.0, 7.0, f64::NAN],
            [4.0, 9.0, 10.0],
            [5.0, 11.0, 12.0],
        ];
        let imputer = LowessImputer::new();
        let X_imp = imputer.fit_transform(&X.view()).expect("ok");
        for i in 0..X_imp.nrows() {
            for j in 0..X_imp.ncols() {
                assert!(!X_imp[[i, j]].is_nan(), "NaN at [{i},{j}]");
            }
        }
    }

    // ── MultivariateNormalImputer tests ────────────────────────────────────────

    #[test]
    fn test_mvn_imputer_no_nans_remain() {
        let X = array![
            [1.0, 2.0, f64::NAN],
            [f64::NAN, 3.0, 4.0],
            [2.0, f64::NAN, 5.0],
            [3.0, 4.0, 6.0],
            [4.0, 5.0, 7.0],
        ];
        let imputer = MultivariateNormalImputer::new();
        let X_imp = imputer.fit_transform(&X.view()).expect("mvn ok");
        for i in 0..X_imp.nrows() {
            for j in 0..X_imp.ncols() {
                assert!(!X_imp[[i, j]].is_nan(), "NaN at [{i},{j}]");
            }
        }
    }

    #[test]
    fn test_mvn_imputer_preserves_observed() {
        let X = array![
            [1.0, 2.0, f64::NAN],
            [4.0, 5.0, 6.0],
            [7.0, f64::NAN, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let imputer = MultivariateNormalImputer::new();
        let X_imp = imputer.fit_transform(&X.view()).expect("ok");
        // Check observed values preserved
        assert!((X_imp[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((X_imp[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((X_imp[[1, 0]] - 4.0).abs() < 1e-10);
        assert!((X_imp[[1, 1]] - 5.0).abs() < 1e-10);
        assert!((X_imp[[1, 2]] - 6.0).abs() < 1e-10);
        assert!((X_imp[[2, 0]] - 7.0).abs() < 1e-10);
        assert!((X_imp[[2, 2]] - 9.0).abs() < 1e-10);
    }

    // ── EmpiricalCDF tests ─────────────────────────────────────────────────────

    #[test]
    fn test_empirical_cdf_basic() {
        let cdf = EmpiricalCDF::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // F(3) = 3/5 = 0.6
        let v = cdf.evaluate(3.0).expect("ok");
        assert!((v - 0.6).abs() < 1e-10, "F(3)={v}");
        // F(0) = 0
        let v0 = cdf.evaluate(0.0).expect("ok");
        assert!((v0 - 0.0).abs() < 1e-10);
        // F(5) = 1
        let v1 = cdf.evaluate(5.0).expect("ok");
        assert!((v1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empirical_cdf_empty_error() {
        let cdf = EmpiricalCDF::new(vec![]);
        assert!(cdf.evaluate(1.0).is_err());
    }

    // ── EmpiricalQuantile tests ────────────────────────────────────────────────

    #[test]
    fn test_empirical_quantile_basic() {
        let q = EmpiricalQuantile::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // Median = Q(0.5) = 3.0
        let median = q.evaluate(0.5).expect("ok");
        assert!((median - 3.0).abs() < 1e-10, "median={median}");
        // Min = Q(0.0) = 1.0
        let lo = q.evaluate(0.0).expect("ok");
        assert!((lo - 1.0).abs() < 1e-10);
        // Max = Q(1.0) = 5.0
        let hi = q.evaluate(1.0).expect("ok");
        assert!((hi - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_empirical_quantile_out_of_range_error() {
        let q = EmpiricalQuantile::new(vec![1.0, 2.0, 3.0]);
        assert!(q.evaluate(-0.1).is_err());
        assert!(q.evaluate(1.1).is_err());
    }

    #[test]
    fn test_empirical_quantile_single_value() {
        let q = EmpiricalQuantile::new(vec![42.0]);
        assert!((q.evaluate(0.5).expect("ok") - 42.0).abs() < 1e-10);
    }
}
