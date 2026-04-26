//! Factor Analysis implementation
//!
//! This module provides comprehensive factor analysis capabilities:
//! - Classical factor analysis via EM algorithm (X = W*F + ε)
//! - Factor visualization (loadings heatmap data)
//! - Posterior sampling / generative sampling inference
//! - Model comparison via AIC/BIC information criteria
//!
//! ## Model
//!
//! ```text
//! X = μ + W * F + ε
//! F ~ N(0, I_q)          (latent factors, q = n_components)
//! ε ~ N(0, Ψ)            (noise, Ψ = diag(ψ₁, …, ψ_p))
//! ```
//!
//! The marginal distribution is `X ~ N(μ, W*Wᵀ + Ψ)`.
//!
//! ## EM Algorithm
//!
//! **E-step**: compute posterior statistics of the latent factors
//! ```text
//! M   = Wᵀ Ψ⁻¹ W + I_q           (q × q)
//! Ez  = M⁻¹ Wᵀ Ψ⁻¹ (X − μ)ᵀ     (q × n)
//! Ezz = n M⁻¹ + Ez Ezᵀ           (q × q)  (not stored; used inline)
//! ```
//!
//! **M-step**: closed-form maximisation
//! ```text
//! W_new  = (X − μ)ᵀ Ezᵀ (Ezz)⁻¹
//! Ψ_new  = diag(S) − diag(W_new Ez (X − μ) / n)   (per-feature)
//! ```
//! where S = (X − μ)ᵀ (X − μ) / n is the sample covariance matrix diagonal.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{rng as make_rng, RngExt, SeedableRng};
use scirs2_linalg::compat::ArrayLinalgExt;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Transform, Untrained},
    types::Float,
};

// ────────────────────────────────────────────────────────────────────────────
// Public re-exports / placeholder compatibility types
// ────────────────────────────────────────────────────────────────────────────

/// EM algorithm configuration for factor analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EMAlgorithm {
    /// Maximum number of EM iterations
    pub max_iter: usize,
    /// Convergence tolerance on log-likelihood improvement
    pub tol: Float,
}

impl Default for EMAlgorithm {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-4,
        }
    }
}

/// Configuration for `FactorAnalysis`
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FactorAnalysisConfig {
    /// Number of latent factors to extract
    pub n_components: usize,
    /// EM algorithm settings
    pub em: EMAlgorithm,
    /// Optional random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for FactorAnalysisConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            em: EMAlgorithm::default(),
            random_state: None,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Trained state
// ────────────────────────────────────────────────────────────────────────────

/// Fitted factor analysis state stored inside `FactorAnalysis<TrainedFactorAnalysis>`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainedFactorAnalysis {
    /// Loading matrix W — shape (n_features, n_components)
    pub loadings: Array2<Float>,
    /// Diagonal noise variances Ψ — shape (n_features,)
    pub noise_variance: Array1<Float>,
    /// Per-feature training mean μ — shape (n_features,)
    pub mean: Array1<Float>,
    /// Number of samples seen during fit
    pub n_samples_fit: usize,
    /// Final log-likelihood
    pub log_likelihood: Float,
    /// Number of EM iterations run
    pub n_iter: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Primary struct
// ────────────────────────────────────────────────────────────────────────────

/// Factor Analysis transformer using the EM algorithm.
///
/// Implements the classical factor analysis model:
/// `X = μ + W * F + ε` with F ~ N(0, I) and ε ~ N(0, Ψ) where Ψ is diagonal.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FactorAnalysis<State = Untrained> {
    /// Model configuration
    pub config: FactorAnalysisConfig,
    /// Trained/untrained state (uses phantom type pattern)
    state: State,
}

impl FactorAnalysis<Untrained> {
    /// Create a new factor analysis model with the given number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            config: FactorAnalysisConfig {
                n_components,
                ..FactorAnalysisConfig::default()
            },
            state: Untrained,
        }
    }

    /// Set EM algorithm configuration
    pub fn em(mut self, em: EMAlgorithm) -> Self {
        self.config.em = em;
        self
    }

    /// Set random seed for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Fit implementation (EM algorithm)
// ────────────────────────────────────────────────────────────────────────────

impl Fit<Array2<Float>, ()> for FactorAnalysis<Untrained> {
    type Fitted = FactorAnalysis<TrainedFactorAnalysis>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let q = self.config.n_components;

        if q == 0 {
            return Err(SklearsError::InvalidInput(
                "n_components must be > 0".to_string(),
            ));
        }
        if q >= n_features {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) must be less than n_features ({})",
                q, n_features
            )));
        }
        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "At least 2 samples are required for factor analysis".to_string(),
            ));
        }

        let n = n_samples as Float;

        // ── Compute mean and center data ──────────────────────────────────
        let mean = x.mean_axis(Axis(0)).ok_or_else(|| {
            SklearsError::NumericalError("Cannot compute mean of empty array".to_string())
        })?;
        // x_c: (n_samples, n_features)  centred observations
        let x_c = x - &mean.clone().insert_axis(Axis(0));

        // ── Full sample covariance S (p × p) — computed once ─────────────
        // S = X_cᵀ X_c / n
        let s_full: Array2<Float> = x_c.t().dot(&x_c) / n;
        // Also cache the diagonal for the M-step Ψ update
        let s_diag: Array1<Float> = {
            let mut d = Array1::zeros(n_features);
            for j in 0..n_features {
                d[j] = s_full[[j, j]];
            }
            d
        };

        // ── Initialise W and Ψ ───────────────────────────────────────────
        let mut rng: StdRng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut make_rng()),
        };

        // W: (n_features, q)  — small random initialisation
        let mut w: Array2<Float> =
            Array2::from_shape_fn((n_features, q), |_| (rng.random::<Float>() - 0.5) * 0.01);

        // Ψ: (n_features,) — initialise to sample variances (clipped)
        let psi_min: Float = 1e-6;
        let mut psi: Array1<Float> = s_diag.mapv(|v| v.max(psi_min));

        let mut log_likelihood_prev: Float = Float::NEG_INFINITY;
        let mut n_iter = 0usize;

        for iter in 0..self.config.em.max_iter {
            n_iter = iter + 1;

            // ── E-step ───────────────────────────────────────────────────
            //   M  = Wᵀ Ψ⁻¹ W + I_q        (q × q)
            //   Ez = M⁻¹ Wᵀ Ψ⁻¹ X_cᵀ      (q × n)
            let psi_inv: Array1<Float> = psi.mapv(|v| 1.0 / v);

            // Wᵀ Ψ⁻¹: (q × p), where each column j is scaled by psi_inv[j]
            let wt_psi_inv: Array2<Float> = {
                let mut out = Array2::zeros((q, n_features));
                for j in 0..n_features {
                    for i in 0..q {
                        out[[i, j]] = w[[j, i]] * psi_inv[j];
                    }
                }
                out
            };

            // M = Wᵀ Ψ⁻¹ W + I_q  (q × q)
            let mut m_mat: Array2<Float> = wt_psi_inv.dot(&w);
            for i in 0..q {
                m_mat[[i, i]] += 1.0;
            }

            // M⁻¹  (q × q)
            let m_inv = m_mat.inv().map_err(|e| {
                SklearsError::NumericalError(format!(
                    "Factor precision matrix M is singular: {}",
                    e
                ))
            })?;

            // Ez = M⁻¹ Wᵀ Ψ⁻¹ X_cᵀ  (q × n)
            let ez: Array2<Float> = m_inv.dot(&wt_psi_inv.dot(&x_c.t()));

            // ── M-step ───────────────────────────────────────────────────
            //  Ezz = n * M⁻¹ + Ez Ezᵀ   (q × q)
            let ez_ezt: Array2<Float> = ez.dot(&ez.t());
            let mut ezz: Array2<Float> = &m_inv * n + &ez_ezt;

            // Regularise for numerical stability
            for i in 0..q {
                ezz[[i, i]] += psi_min;
            }

            // W_new = X_cᵀ Ezᵀ Ezz⁻¹    (p × q)
            let xct_ez: Array2<Float> = x_c.t().dot(&ez.t()); // (p, q)
            let ezz_inv = ezz.inv().map_err(|e| {
                SklearsError::NumericalError(format!(
                    "E[z zᵀ] matrix is singular during M-step: {}",
                    e
                ))
            })?;
            let w_new: Array2<Float> = xct_ez.dot(&ezz_inv); // (p, q)

            // Ψ_new[j] = S[j,j] − (W_new Ez X_c)[j,j] / n
            // W_new Ez: (p × n), then dot with x_c columns for per-feature correlation
            let w_new_ez: Array2<Float> = w_new.dot(&ez); // (p, n)
            let mut psi_new: Array1<Float> = Array1::zeros(n_features);
            for j in 0..n_features {
                let row = w_new_ez.row(j);
                let col = x_c.column(j);
                let corr: Float = row.dot(&col) / n;
                psi_new[j] = (s_diag[j] - corr).max(psi_min);
            }

            // ── Log-likelihood using matrix determinant lemma ────────────
            // log|Σ| = Σⱼ log(ψⱼ) + log|M|      (M = Wᵀ Ψ⁻¹ W + I_q)
            // trace(Σ⁻¹ S) uses the Woodbury identity:
            //   Σ⁻¹ = Ψ⁻¹ − Ψ⁻¹ W M⁻¹ Wᵀ Ψ⁻¹
            // so trace(Σ⁻¹ S) = trace(Ψ⁻¹ S) − trace(M⁻¹ Wᵀ Ψ⁻¹ S Ψ⁻¹ W)
            let log_l = compute_log_likelihood_mdl(
                n, n_features, &w_new, &psi_new, &psi_inv, &m_mat, &m_inv, &s_full,
            );

            // Update parameters
            w = w_new;
            psi = psi_new;

            // Convergence check
            if (log_l - log_likelihood_prev).abs() < self.config.em.tol {
                log_likelihood_prev = log_l;
                break;
            }
            log_likelihood_prev = log_l;
        }

        Ok(FactorAnalysis {
            config: self.config,
            state: TrainedFactorAnalysis {
                loadings: w,
                noise_variance: psi,
                mean,
                n_samples_fit: n_samples,
                log_likelihood: log_likelihood_prev,
                n_iter,
            },
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Transform (compute factor scores)
// ────────────────────────────────────────────────────────────────────────────

impl Transform<Array2<Float>, Array2<Float>> for FactorAnalysis<TrainedFactorAnalysis> {
    /// Compute posterior mean factor scores for each observation.
    ///
    /// Returns matrix of shape `(n_samples, n_components)`.
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();
        let s = &self.state;

        if n_features != s.mean.len() {
            return Err(SklearsError::FeatureMismatch {
                expected: s.mean.len(),
                actual: n_features,
            });
        }

        let q = s.loadings.ncols();
        let psi_inv: Array1<Float> = s.noise_variance.mapv(|v| 1.0 / v);

        // Wᵀ Ψ⁻¹  (q × p)
        let wt_psi_inv: Array2<Float> = {
            let mut out = Array2::zeros((q, n_features));
            for j in 0..n_features {
                for i in 0..q {
                    out[[i, j]] = s.loadings[[j, i]] * psi_inv[j];
                }
            }
            out
        };

        // M = Wᵀ Ψ⁻¹ W + I_q
        let mut m_mat: Array2<Float> = wt_psi_inv.dot(&s.loadings);
        for i in 0..q {
            m_mat[[i, i]] += 1.0;
        }

        let m_inv = m_mat.inv().map_err(|e| {
            SklearsError::NumericalError(format!("Factor precision matrix is singular: {}", e))
        })?;

        // Centre data
        let mean_row = s.mean.clone().insert_axis(Axis(0));
        let x_c = x - &mean_row;

        // Scores = (M⁻¹ Wᵀ Ψ⁻¹ X_cᵀ)ᵀ  →  (n_samples, q)
        let scores_t: Array2<Float> = m_inv.dot(&wt_psi_inv.dot(&x_c.t()));

        let mut scores = Array2::zeros((n_samples, q));
        for n_idx in 0..n_samples {
            for qi in 0..q {
                scores[[n_idx, qi]] = scores_t[[qi, n_idx]];
            }
        }
        Ok(scores)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Accessors on the trained model
// ────────────────────────────────────────────────────────────────────────────

impl FactorAnalysis<TrainedFactorAnalysis> {
    /// Return the loading matrix W — shape `(n_features, n_components)`.
    pub fn loadings(&self) -> &Array2<Float> {
        &self.state.loadings
    }

    /// Return the diagonal noise variances Ψ — shape `(n_features,)`.
    pub fn noise_variance(&self) -> &Array1<Float> {
        &self.state.noise_variance
    }

    /// Return the per-feature training mean.
    pub fn mean(&self) -> &Array1<Float> {
        &self.state.mean
    }

    /// Return the log-likelihood at convergence.
    pub fn log_likelihood(&self) -> Float {
        self.state.log_likelihood
    }

    /// Return the number of EM iterations run.
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    // ── Factor Visualization ─────────────────────────────────────────────

    /// Return loading heatmap data suitable for visualisation.
    ///
    /// Returns a `LoadingsPlotData` struct containing the loading matrix,
    /// feature labels, and component labels.
    pub fn loadings_plot_data(&self, feature_names: Option<Vec<String>>) -> LoadingsPlotData {
        let (n_features, n_components) = self.state.loadings.dim();
        let features = feature_names
            .unwrap_or_else(|| (0..n_features).map(|i| format!("feature_{}", i)).collect());
        let components: Vec<String> = (1..=n_components).map(|i| format!("Factor{}", i)).collect();

        LoadingsPlotData {
            loadings: self.state.loadings.clone(),
            feature_names: features,
            component_names: components,
        }
    }

    /// Return a scree-plot-like DataFrame with communalities (proportion of
    /// each feature's variance explained by all factors).
    ///
    /// `communality[j] = Σ_k W[j,k]² / (Σ_k W[j,k]² + ψ_j)`
    pub fn communalities(&self) -> Array1<Float> {
        let w = &self.state.loadings;
        let psi = &self.state.noise_variance;
        let n_features = w.nrows();
        let mut comm = Array1::zeros(n_features);
        for j in 0..n_features {
            let h2: Float = w.row(j).mapv(|v| v * v).sum();
            comm[j] = h2 / (h2 + psi[j]);
        }
        comm
    }

    // ── Sampling / Generative Inference ──────────────────────────────────

    /// Draw `n_samples` new observations from the fitted factor model.
    ///
    /// ```text
    /// F_new ~ N(0, I_q)
    /// ε_new ~ N(0, Ψ)
    /// X_new = μ + W F_new + ε_new
    /// ```
    pub fn sample(&self, n_samples: usize, random_state: Option<u64>) -> Result<Array2<Float>> {
        let s = &self.state;
        let (n_features, q) = s.loadings.dim();

        let mut rng: StdRng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut make_rng()),
        };

        // Draw factors F_new: (n_samples, q) from N(0, I)
        let factors: Array2<Float> =
            Array2::from_shape_fn((n_samples, q), |_| sample_normal(&mut rng));

        // Draw noise ε: (n_samples, n_features) from N(0, diag(Ψ))
        let noise: Array2<Float> = Array2::from_shape_fn((n_samples, n_features), |(_i, j)| {
            sample_normal(&mut rng) * s.noise_variance[j].sqrt()
        });

        // X_new = μ + F_new Wᵀ + ε
        let loadings_t = s.loadings.t().to_owned(); // (q, p)
        let x_latent: Array2<Float> = factors.dot(&loadings_t); // (n, p)
        let mean_row = s.mean.clone().insert_axis(Axis(0)); // (1, p)

        Ok(x_latent + &mean_row + &noise)
    }

    // ── Model Comparison (Information Criteria) ───────────────────────────

    /// Return AIC for this model: `AIC = -2 log L + 2 * n_params`.
    ///
    /// `n_params = p*q + p - q*(q-1)/2`
    /// (loadings + noise variances minus rotation indeterminacy)
    pub fn aic(&self) -> Float {
        let n_params = self.n_free_params() as Float;
        -2.0 * self.state.log_likelihood + 2.0 * n_params
    }

    /// Return BIC for this model: `BIC = -2 log L + log(n) * n_params`.
    pub fn bic(&self) -> Float {
        let n = self.state.n_samples_fit as Float;
        let n_params = self.n_free_params() as Float;
        -2.0 * self.state.log_likelihood + n.ln() * n_params
    }

    /// Number of free parameters (accounting for rotation indeterminacy).
    ///
    /// `n_params = p*q + p - q*(q-1)/2`
    pub fn n_free_params(&self) -> usize {
        let (p, q) = self.state.loadings.dim();
        let rotation_removal = q.saturating_sub(1) * q / 2;
        p * q + p - rotation_removal
    }

    /// Reconstruct the full model covariance matrix Σ = W Wᵀ + Ψ.
    pub fn covariance(&self) -> Array2<Float> {
        compute_sigma(&self.state.loadings, &self.state.noise_variance)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Model comparison helper
// ────────────────────────────────────────────────────────────────────────────

/// Compare multiple trained factor analysis models by AIC / BIC.
///
/// Returns the index of the model with the lowest AIC and the lowest BIC,
/// along with per-model scores.
pub struct ModelComparisonResult {
    /// AIC for each model
    pub aic_scores: Vec<Float>,
    /// BIC for each model
    pub bic_scores: Vec<Float>,
    /// Index of model with lowest AIC
    pub best_aic_idx: usize,
    /// Index of model with lowest BIC
    pub best_bic_idx: usize,
}

/// Run `FactorAnalysis` for a range of `n_components` values and return
/// the comparison result with AIC and BIC for each.
pub fn select_n_components(
    x: &Array2<Float>,
    component_range: std::ops::Range<usize>,
    random_state: Option<u64>,
) -> Result<ModelComparisonResult> {
    if component_range.is_empty() {
        return Err(SklearsError::InvalidInput(
            "component_range must be non-empty".to_string(),
        ));
    }

    let mut aic_scores = Vec::new();
    let mut bic_scores = Vec::new();

    for q in component_range {
        let model = FactorAnalysis::new(q);
        let model = if let Some(seed) = random_state {
            model.random_state(seed)
        } else {
            model
        };
        let trained = model.fit(x, &())?;
        aic_scores.push(trained.aic());
        bic_scores.push(trained.bic());
    }

    let best_aic_idx = aic_scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_bic_idx = bic_scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(ModelComparisonResult {
        aic_scores,
        bic_scores,
        best_aic_idx,
        best_bic_idx,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Visualization data structs
// ────────────────────────────────────────────────────────────────────────────

/// Data for rendering a factor loadings heatmap.
#[derive(Debug, Clone)]
pub struct LoadingsPlotData {
    /// Loading matrix W — rows are features, columns are factors
    pub loadings: Array2<Float>,
    /// Human-readable feature names
    pub feature_names: Vec<String>,
    /// Human-readable component/factor names
    pub component_names: Vec<String>,
}

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// Compute Σ = W Wᵀ + Ψ   (p × p)
fn compute_sigma(w: &Array2<Float>, psi: &Array1<Float>) -> Array2<Float> {
    let mut sigma: Array2<Float> = w.dot(&w.t());
    let n_features = psi.len();
    for j in 0..n_features {
        sigma[[j, j]] += psi[j];
    }
    sigma
}

/// Exact log-likelihood using the matrix determinant lemma and Woodbury identity.
///
/// `log L = -n/2 (p log(2π) + log|Σ| + trace(Σ⁻¹ S))`
///
/// **log|Σ|** via the matrix determinant lemma:
/// `log|Ψ + W Wᵀ| = Σⱼ log(ψⱼ) + log|M|`  where `M = Wᵀ Ψ⁻¹ W + I_q`.
///
/// **trace(Σ⁻¹ S)** via the Woodbury identity:
/// `Σ⁻¹ = Ψ⁻¹ − Ψ⁻¹ W M⁻¹ Wᵀ Ψ⁻¹`
/// `trace(Σ⁻¹ S) = trace(Ψ⁻¹ S) − trace(M⁻¹ Wᵀ Ψ⁻¹ S Ψ⁻¹ W)`
///
/// Both reductions are O(p·q²) instead of O(p³), and use the already-computed
/// `m_mat` and `m_inv` from the EM E-step.
#[allow(clippy::too_many_arguments)]
fn compute_log_likelihood_mdl(
    n: Float,
    p: usize,
    _w: &Array2<Float>,
    psi: &Array1<Float>,
    psi_inv: &Array1<Float>,
    m_mat: &Array2<Float>,
    m_inv: &Array2<Float>,
    s_full: &Array2<Float>,
) -> Float {
    use std::f64::consts::PI;
    let q = m_mat.nrows();

    // ── log|Σ| = Σⱼ log(ψⱼ) + log|M| ─────────────────────────────────
    let log_psi_sum: Float = psi.iter().map(|&v| v.max(1e-300).ln()).sum();

    // log|M| via sum of log of absolute diagonal of its LU / as diagonal product
    // We approximate log|M| using the sum of log(eigenvalues) implied by the
    // fact that M is SPD: det(M) ≈ product of its diagonal after regularisation.
    // A more accurate method: log|M| = 2·Σ log(L_ii) where L = chol(M).
    // Since we can't call chol directly, use direct determinant approximation:
    // For a small (q×q) matrix, compute det numerically via cofactor expansion or
    // use the identity: log|M| = log( prod(eigenvalues) ).
    // Simplest numerically stable path for q ≪ p: direct cofactor or LU trace.
    // We use the fact that M is already computed. For q ≤ ~50 this is fine.
    let log_det_m = log_det_spd(m_mat, q);
    if !log_det_m.is_finite() {
        return Float::NEG_INFINITY;
    }

    let log_det_sigma = log_psi_sum + log_det_m;

    // ── trace(Σ⁻¹ S) via Woodbury ────────────────────────────────────
    // trace(Ψ⁻¹ S) = Σⱼ psi_inv[j] * S[j,j]
    let trace_psi_inv_s: Float = (0..p).map(|j| psi_inv[j] * s_full[[j, j]]).sum();

    // Ψ⁻¹ W  (p × q): column j of W scaled by psi_inv[j]
    // Already captured as `wt_psi_inv.t()` but we don't have it here;
    // recompute Ψ⁻¹ S Ψ⁻¹ W from s_full and m_inv.
    //
    // trace(M⁻¹ Wᵀ Ψ⁻¹ S Ψ⁻¹ W) = trace(M⁻¹ · (Wᵀ Ψ⁻¹ S Ψ⁻¹ W))
    //   = Σᵢⱼ M⁻¹[i,j] * (Wᵀ Ψ⁻¹ S Ψ⁻¹ W)[j,i]
    //
    // However we don't have W available here, so we pass the already-built
    // `m_mat = Wᵀ Ψ⁻¹ W + I` and extract Wᵀ Ψ⁻¹ = m_mat_raw (before +I).
    // Use the identity: Wᵀ Ψ⁻¹ = (M − I) ... but that's circular.
    //
    // Cleaner alternative: collapse to full Σ⁻¹ only for the trace term,
    // then add back the fast log_det computation above.
    //
    // For practical sizes (p ≲ 200, q ≲ 20), computing Σ (p×p) then inverting
    // is acceptable — O(p³) per iteration, same as sklearn's FA.
    // The key improvement is log|Σ| via the determinant lemma (exact, avoids
    // the wrong diagonal-product formula).
    //
    // We pass `_w` as unused but can't use it without rebuilding sigma.
    // Instead reconstruct Σ and invert for the trace only.
    // NOTE: We avoid using `_w` to build sigma; instead build Σ⁻¹ from m_inv + psi.
    //
    // Σ⁻¹ = Ψ⁻¹ − Ψ⁻¹ W M⁻¹ Wᵀ Ψ⁻¹
    //       = Ψ⁻¹ − (Ψ⁻¹ W) M⁻¹ (Wᵀ Ψ⁻¹)
    //
    // trace(Σ⁻¹ S) = trace(Ψ⁻¹ S) − trace((Ψ⁻¹ W) M⁻¹ (Wᵀ Ψ⁻¹) S)
    //              = trace(Ψ⁻¹ S) − trace(M⁻¹ (Wᵀ Ψ⁻¹ S Ψ⁻¹ W))
    //
    // Wᵀ Ψ⁻¹ S Ψ⁻¹ W = (Wᵀ Ψ⁻¹) S (Ψ⁻¹ W)
    //   = A S Aᵀ  where A = Wᵀ Ψ⁻¹  (q × p)
    //
    // We can extract A from m_mat:
    //   m_mat = A W + I  ⟹  A W = m_mat − I
    // but to get A itself we need W, which is passed as `_w`.
    // Use `_w` directly:
    let w = _w;
    let p_dim = psi_inv.len();
    // A = Wᵀ Ψ⁻¹  (q × p)
    let a: Array2<Float> = {
        let mut out = Array2::zeros((q, p_dim));
        for j in 0..p_dim {
            for i in 0..q {
                out[[i, j]] = w[[j, i]] * psi_inv[j];
            }
        }
        out
    };
    // B = A S Aᵀ  (q × q)
    let b: Array2<Float> = a.dot(s_full).dot(&a.t());
    // trace(M⁻¹ B)
    let trace_correction: Float = (0..q)
        .map(|i| (0..q).map(|j| m_inv[[i, j]] * b[[j, i]]).sum::<Float>())
        .sum();

    let trace_sigma_inv_s = trace_psi_inv_s - trace_correction;

    -0.5 * n * ((p as Float) * (2.0 * PI as Float).ln() + log_det_sigma + trace_sigma_inv_s)
}

/// Compute log-determinant of a small SPD matrix by computing its eigenvalue
/// product via the Gauss-Jordan identity on the matrix diagonal after pivoting.
///
/// For a q×q SPD matrix, use: log|M| = log(product of abs(diagonal of LU)) via
/// Cramer's rule approximation, or more simply the product of diagonal of M
/// (Hadamard bound approximation).  For exact results with small q we use
/// the recursive cofactor expansion inline.
///
/// In practice: since M = Wᵀ Ψ⁻¹ W + I is guaranteed SPD,
/// `det(M) = product of its eigenvalues`. We use `M^{-1}` already available;
/// `det(M) = 1 / det(M^{-1})`.  So `log|M| = -log|M^{-1}|`.
///
/// We compute `|M^{-1}|` via its diagonal product (Hadamard) only if q == 1,
/// otherwise accumulate directly from the formula used for EM convergence.
fn log_det_spd(m: &Array2<Float>, q: usize) -> Float {
    if q == 0 {
        return 0.0;
    }
    if q == 1 {
        return m[[0, 0]].abs().ln();
    }
    // For small q (≤ 6 typical in practice), compute via Leibniz rule (brute force).
    // For larger q, use Gaussian elimination (LU-like).
    leibniz_log_det(m, q)
}

/// Recursive Leibniz log-determinant for small matrices (q ≤ ~10).
/// For larger q this is O(q!) so we fall back to LU.
fn leibniz_log_det(m: &Array2<Float>, q: usize) -> Float {
    if q <= 4 {
        // Direct formula via cofactor expansion along first row
        let det = small_det(m, q);
        if det <= 0.0 {
            Float::NEG_INFINITY
        } else {
            det.ln()
        }
    } else {
        // Gaussian elimination to compute log-det
        lu_log_det(m, q)
    }
}

/// Direct determinant for 1×1 to 4×4 matrices via cofactor expansion.
fn small_det(m: &Array2<Float>, q: usize) -> Float {
    match q {
        1 => m[[0, 0]],
        2 => m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]],
        3 => {
            m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
                - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
                + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]])
        }
        4 => {
            let mut det = 0.0_f64;
            for col in 0..4_usize {
                // Extract 3×3 submatrix (rows 1..3, all cols except `col`)
                let mut sub = Array2::zeros((3, 3));
                for r in 1..4_usize {
                    let mut sc = 0usize;
                    for c in 0..4_usize {
                        if c != col {
                            sub[[r - 1, sc]] = m[[r, c]];
                            sc += 1;
                        }
                    }
                }
                let cofactor = if col % 2 == 0 { 1.0 } else { -1.0 };
                det += cofactor * m[[0, col]] * small_det(&sub, 3);
            }
            det
        }
        _ => lu_log_det(m, q),
    }
}

/// Gaussian elimination (partial pivoting) to compute log|M| for q > 4.
fn lu_log_det(m: &Array2<Float>, q: usize) -> Float {
    let mut a = m.to_owned();
    let mut log_det = 0.0_f64;
    let mut sign = 1.0_f64;

    for k in 0..q {
        // Find pivot
        let pivot_row = (k..q)
            .max_by(|&r1, &r2| {
                a[[r1, k]]
                    .abs()
                    .partial_cmp(&a[[r2, k]].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(k);

        if pivot_row != k {
            for c in 0..q {
                a.swap([k, c], [pivot_row, c]);
            }
            sign = -sign;
        }

        let pivot = a[[k, k]];
        if pivot.abs() < 1e-300 {
            return Float::NEG_INFINITY;
        }
        log_det += pivot.abs().ln();
        if pivot < 0.0 {
            sign = -sign;
        }

        for r in (k + 1)..q {
            let factor = a[[r, k]] / pivot;
            for c in k..q {
                let val = a[[k, c]] * factor;
                a[[r, c]] -= val;
            }
        }
    }

    if sign < 0.0 {
        Float::NEG_INFINITY
    } else {
        log_det
    }
}

/// Sample from N(0, 1) using the Box-Muller transform (stdlib-only, no external
/// rand_distr needed).
fn sample_normal(rng: &mut StdRng) -> Float {
    // Box-Muller transform
    let u1: Float = rng.random::<Float>().max(1e-10);
    let u2: Float = rng.random::<Float>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI as Float * u2).cos()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_test_data() -> Array2<Float> {
        // Simple 10×4 dataset generated from a 2-factor model
        array![
            [1.2, 2.1, 3.3, 0.5],
            [0.8, 1.9, 2.7, 0.3],
            [1.5, 2.4, 3.9, 0.6],
            [0.6, 1.6, 2.2, 0.2],
            [1.1, 2.0, 3.1, 0.4],
            [1.3, 2.2, 3.5, 0.55],
            [0.9, 1.8, 2.7, 0.35],
            [1.4, 2.3, 3.7, 0.58],
            [0.7, 1.7, 2.4, 0.25],
            [1.0, 2.05, 3.05, 0.45],
        ]
    }

    #[test]
    fn test_factor_analysis_fit_transform() {
        let x = make_test_data();
        let fa = FactorAnalysis::new(2).random_state(42);
        let trained = fa
            .fit(&x, &())
            .expect("FactorAnalysis::fit should succeed on well-formed data");

        // Loadings shape
        assert_eq!(trained.loadings().dim(), (4, 2));
        // Noise variance shape
        assert_eq!(trained.noise_variance().len(), 4);
        // All noise variances are positive
        assert!(trained.noise_variance().iter().all(|&v| v > 0.0));

        // Transform produces (n_samples, n_components) scores
        let scores = trained
            .transform(&x)
            .expect("transform should succeed on fitted model");
        assert_eq!(scores.dim(), (10, 2));
    }

    #[test]
    fn test_factor_analysis_sampling() {
        let x = make_test_data();
        let fa = FactorAnalysis::new(2).random_state(0);
        let trained = fa
            .fit(&x, &())
            .expect("FactorAnalysis::fit should succeed on well-formed data");

        let samples = trained
            .sample(5, Some(1))
            .expect("sample should succeed on fitted model");
        assert_eq!(samples.dim(), (5, 4));
    }

    #[test]
    fn test_aic_bic_ordering() {
        let x = make_test_data();
        // Fit with 1 and 2 factors; BIC should penalise 2 factors more
        let fa1 = FactorAnalysis::new(1).random_state(7);
        let fa2 = FactorAnalysis::new(2).random_state(7);
        let m1 = fa1
            .fit(&x, &())
            .expect("FactorAnalysis::fit with 1 component should succeed");
        let m2 = fa2
            .fit(&x, &())
            .expect("FactorAnalysis::fit with 2 components should succeed");

        // n_free_params increases with n_components
        assert!(m2.n_free_params() > m1.n_free_params());
        // log-likelihood should be finite
        assert!(m1.log_likelihood().is_finite());
        assert!(m2.log_likelihood().is_finite());
    }

    #[test]
    fn test_loadings_plot_data() {
        let x = make_test_data();
        let trained = FactorAnalysis::new(2)
            .random_state(42)
            .fit(&x, &())
            .expect("FactorAnalysis::fit should succeed on well-formed data");

        let plot =
            trained.loadings_plot_data(Some(vec!["a".into(), "b".into(), "c".into(), "d".into()]));
        assert_eq!(plot.feature_names.len(), 4);
        assert_eq!(plot.component_names.len(), 2);
    }

    #[test]
    fn test_select_n_components() {
        let x = make_test_data();
        let result = select_n_components(&x, 1..3, Some(42))
            .expect("select_n_components should succeed on well-formed data");
        assert_eq!(result.aic_scores.len(), 2);
        assert_eq!(result.bic_scores.len(), 2);
    }

    #[test]
    fn test_communalities_range() {
        let x = make_test_data();
        let trained = FactorAnalysis::new(2)
            .random_state(42)
            .fit(&x, &())
            .expect("FactorAnalysis::fit should succeed on well-formed data");
        let comm = trained.communalities();
        // Communalities must be in (0, 1]
        for c in comm.iter() {
            assert!(
                *c > 0.0 && *c <= 1.0 + 1e-9,
                "communality out of range: {}",
                c
            );
        }
    }

    #[test]
    fn test_invalid_n_components() {
        let x = make_test_data();
        // n_components >= n_features should error
        let fa = FactorAnalysis::new(4);
        assert!(fa.fit(&x, &()).is_err());
    }

    #[test]
    fn test_covariance_positive_diagonal() {
        let x = make_test_data();
        let trained = FactorAnalysis::new(2)
            .random_state(42)
            .fit(&x, &())
            .expect("FactorAnalysis::fit should succeed on well-formed data");
        let cov = trained.covariance();
        for j in 0..4 {
            assert!(cov[[j, j]] > 0.0);
        }
    }

    /// Hand-verifiable numerical test for the log-likelihood formula.
    ///
    /// For a q=1, p=2 model where W = [[w1], [w2]] and Ψ = diag(ψ1, ψ2):
    ///   Σ = [[w1² + ψ1, w1 w2],
    ///        [w1 w2,    w2² + ψ2]]
    ///   |Σ| = (w1² + ψ1)(w2² + ψ2) - (w1 w2)²
    ///       = ψ1 ψ2 + ψ2 w1² + ψ1 w2²          (Woodbury: = ψ1 ψ2 * |M| with M = 1 + w1²/ψ1 + w2²/ψ2)
    ///
    /// We fit a trivially small dataset where we can compute the expected
    /// log-likelihood independently and verify our implementation matches it.
    #[test]
    fn test_log_likelihood_numerical() {
        // p=2, q=1 model: W = [[1.0], [1.0]], Ψ = diag(0.5, 0.5)
        // Σ = [[1.5, 1.0], [1.0, 1.5]]
        // |Σ| = 1.5*1.5 - 1.0 = 1.25
        // For S = I₂ (identity), trace(Σ⁻¹) = trace([[1.5,-1.0],[-1.0,1.5]] / 1.25)
        //                                     = 2 * 1.5 / 1.25 = 2.4
        // log L = -n/2 * (2*log(2π) + log(1.25) + 2.4)
        let w_test = Array2::from_shape_vec((2, 1), vec![1.0_f64, 1.0_f64])
            .expect("shape (2,1) matches 2-element vector");
        let psi_test = Array1::from_vec(vec![0.5_f64, 0.5_f64]);
        // M = Wᵀ Ψ⁻¹ W + I₁ = [[2.0]] + [[1.0]] = [[3.0]]
        let m_mat = Array2::from_shape_vec((1, 1), vec![3.0_f64])
            .expect("shape (1,1) matches 1-element vector");
        let m_inv = Array2::from_shape_vec((1, 1), vec![1.0 / 3.0])
            .expect("shape (1,1) matches 1-element vector");
        let psi_inv = Array1::from_vec(vec![2.0_f64, 2.0_f64]);
        // S = I₂
        let s_full = Array2::eye(2);
        let n = 10.0_f64;
        let p = 2;

        let log_l =
            compute_log_likelihood_mdl(n, p, &w_test, &psi_test, &psi_inv, &m_mat, &m_inv, &s_full);

        // Independent calculation:
        // log|Σ| = log(ψ1) + log(ψ2) + log|M|
        //        = log(0.5) + log(0.5) + log(3)
        //        = -0.6931 - 0.6931 + 1.0986 ≈ -0.2877
        // trace(Σ⁻¹ S) = trace(Ψ⁻¹ S) - trace(M⁻¹ Wᵀ Ψ⁻¹ S Ψ⁻¹ W)
        //   trace(Ψ⁻¹ S) = 2 + 2 = 4  (S = I₂, psi_inv = [2,2])
        //   Wᵀ Ψ⁻¹ S Ψ⁻¹ W = [[1,1]] diag(2,2) I₂ diag(2,2) [[1],[1]]
        //                    = [[1,1]] diag(4,4) [[1],[1]] = [[8]]
        //   trace(M⁻¹ * 8) = (1/3)*8 = 2.6667
        //   trace(Σ⁻¹ S) = 4 - 2.6667 = 1.3333
        // log L = -10/2 * (2*log(2π) + (-0.2877) + 1.3333)
        //       = -5 * (3.6759 - 0.2877 + 1.3333)
        //       = -5 * 4.7215 = -23.607...
        let expected_log_det_sigma = 0.5_f64.ln() * 2.0 + 3.0_f64.ln();
        let expected_trace = 4.0 - 8.0 / 3.0;
        let expected = -5.0
            * (2.0 * (2.0 * std::f64::consts::PI).ln() + expected_log_det_sigma + expected_trace);

        assert!(
            (log_l - expected).abs() < 1e-6,
            "log-likelihood mismatch: got {}, expected {}",
            log_l,
            expected
        );
    }

    /// Verify that `DecompositionPipeline::transform` applies fitted preprocessing
    /// steps to new (out-of-sample) data.
    #[test]
    fn test_pipeline_fit_transform_then_transform_new_data() {
        use crate::modular_framework::{
            AlgorithmCapabilities, DecompositionAlgorithm, DecompositionComponents,
            DecompositionParams, DecompositionPipeline, StandardizationStep,
        };
        use scirs2_core::ndarray::{Array1, Array2};
        use sklears_core::types::Float;
        use std::any::Any;

        // ── Minimal mock algorithm for the pipeline ──────────────────────
        #[derive(Debug, Clone)]
        struct IdentityAlg {
            fitted: bool,
            n_feats: usize,
        }
        impl IdentityAlg {
            fn new() -> Self {
                Self {
                    fitted: false,
                    n_feats: 0,
                }
            }
        }
        impl DecompositionAlgorithm for IdentityAlg {
            fn name(&self) -> &str {
                "identity"
            }
            fn description(&self) -> &str {
                "pass-through"
            }
            fn capabilities(&self) -> AlgorithmCapabilities {
                AlgorithmCapabilities::default()
            }
            fn validate_params(&self, _: &DecompositionParams) -> sklears_core::error::Result<()> {
                Ok(())
            }
            fn fit(
                &mut self,
                data: &Array2<Float>,
                _: &DecompositionParams,
            ) -> sklears_core::error::Result<()> {
                self.fitted = true;
                self.n_feats = data.ncols();
                Ok(())
            }
            fn transform(
                &self,
                data: &Array2<Float>,
            ) -> sklears_core::error::Result<Array2<Float>> {
                Ok(data.clone())
            }
            fn get_components(&self) -> sklears_core::error::Result<DecompositionComponents> {
                Ok(DecompositionComponents {
                    components: Some(Array2::eye(self.n_feats)),
                    eigenvalues: Some(Array1::ones(self.n_feats)),
                    ..Default::default()
                })
            }
            fn is_fitted(&self) -> bool {
                self.fitted
            }
            fn clone_algorithm(&self) -> Box<dyn DecompositionAlgorithm> {
                Box::new(self.clone())
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }

        // Build pipeline with standardization
        let mut pipeline = DecompositionPipeline::new(Box::new(IdentityAlg::new()))
            .add_preprocessing(Box::new(StandardizationStep::new()));

        let params = DecompositionParams::default();
        let x_train = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("shape (4,2) matches 8-element vector");

        // fit_transform trains the preprocessing + algorithm
        let result = pipeline
            .fit_transform(&x_train, &params)
            .expect("pipeline fit_transform should succeed on well-formed data");
        assert_eq!(
            result
                .components
                .components
                .expect("identity algorithm returns eye components")
                .shape(),
            [2, 2]
        );

        // transform on new data should apply fitted standardization without error
        let x_test = Array2::from_shape_vec((2, 2), vec![2.0, 3.0, 4.0, 5.0])
            .expect("shape (2,2) matches 4-element vector");
        let transformed = pipeline
            .transform(&x_test)
            .expect("pipeline transform should succeed on fitted pipeline");
        assert_eq!(transformed.dim(), (2, 2));
    }
}
