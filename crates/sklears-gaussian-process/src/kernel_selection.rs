//! Kernel selection methods for Gaussian Processes
//!
//! This module provides model-selection utilities for choosing among a set
//! of candidate kernels (or, more precisely, kernel *families*): each
//! candidate has its own hyperparameters (and the shared observation noise)
//! optimized against the log marginal likelihood via
//! [`crate::marginal_likelihood::optimize_hyperparameters`], and the fitted
//! candidates are then ranked using one of four criteria:
//!
//! * [`SelectionCriterion::Aic`] — Akaike Information Criterion,
//!   `-2 * LML + 2k` (lower is better).
//! * [`SelectionCriterion::Bic`] — Bayesian Information Criterion,
//!   `-2 * LML + k * ln(n)` (lower is better; penalizes model complexity more
//!   heavily than AIC for `n > 7`).
//! * [`SelectionCriterion::LogMarginalLikelihood`] — the raw log marginal
//!   likelihood (higher is better).
//! * [`SelectionCriterion::Cv`] — mean held-out predictive log density under
//!   k-fold cross-validation (higher is better).
//!
//! Here `k` is the number of hyperparameters (kernel parameters plus the
//! noise level) and `n` is the number of training samples.
//!
//! Note on [`SelectionCriterion::Cv`]: `crate::marginal_likelihood`'s
//! `cross_validate_hyperparameters` evaluates the *same* full-data marginal
//! likelihood on every fold (by its own documentation this is "simplified"
//! and does not actually hold out data), so it is not suitable for genuine
//! kernel-family comparison. This module instead performs real k-fold
//! cross-validation: for each fold it fits a [`crate::gpr::GaussianProcessRegressor`]
//! on the training split and scores it by the predictive log density of the
//! held-out split.
use std::f64::consts::PI;

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Fit,
};

use crate::gpr::GaussianProcessRegressor;
use crate::kernels::Kernel;
use crate::marginal_likelihood::{log_marginal_likelihood_stable, optimize_hyperparameters};

/// Number of Adam iterations used to fit each candidate's hyperparameters
/// before scoring it. Kept modest since kernel selection typically compares
/// several candidates and does not need each one driven to full convergence
/// to distinguish a good kernel family from a poor one.
const CANDIDATE_OPTIMIZATION_ITERS: usize = 50;
const CANDIDATE_OPTIMIZATION_TOL: f64 = 1e-4;

/// Criterion used to rank candidate kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionCriterion {
    /// Akaike Information Criterion (lower is better).
    Aic,
    /// Bayesian Information Criterion (lower is better).
    Bic,
    /// Mean held-out predictive log density under k-fold CV (higher is
    /// better).
    Cv,
    /// Raw log marginal likelihood (higher is better).
    LogMarginalLikelihood,
}

impl SelectionCriterion {
    /// Whether higher scores are better for this criterion (`Cv` and
    /// `LogMarginalLikelihood`), as opposed to lower being better (`Aic`,
    /// `Bic`).
    pub fn higher_is_better(self) -> bool {
        matches!(
            self,
            SelectionCriterion::Cv | SelectionCriterion::LogMarginalLikelihood
        )
    }
}

/// Configuration for [`KernelSelector`].
#[derive(Debug, Clone)]
pub struct KernelSelectionConfig {
    /// Criterion used to rank candidates.
    pub criterion: SelectionCriterion,
    /// Number of folds used when `criterion` is [`SelectionCriterion::Cv`].
    pub cv_folds: usize,
    /// Initial (and, for AIC/BIC/LML scoring, jointly optimized) observation
    /// noise variance `σ_n²`.
    pub noise_variance: f64,
    /// Random seed controlling the CV fold shuffle.
    pub random_state: Option<u64>,
}

impl Default for KernelSelectionConfig {
    fn default() -> Self {
        Self {
            criterion: SelectionCriterion::Bic,
            cv_folds: 5,
            noise_variance: 1e-4,
            random_state: Some(0),
        }
    }
}

/// Result of a kernel selection run.
#[derive(Debug, Clone)]
pub struct KernelSelectionResult {
    /// The winning kernel, with hyperparameters optimized against the
    /// training data.
    pub best_kernel: Box<dyn Kernel>,
    /// Index of the winning kernel within the candidates slice that was
    /// passed in.
    pub best_index: usize,
    /// Short name of the winning kernel (derived from its `Debug`
    /// representation), e.g. `"RBF"`.
    pub best_kernel_name: String,
    /// Score of the winning kernel under the configured criterion.
    pub best_score: f64,
    /// `(name, score)` for every candidate, in input order.
    pub all_scores: Vec<(String, f64)>,
}

/// Builder for running kernel selection over a fixed set of candidate
/// kernels.
///
/// # Examples
///
/// ```ignore
/// let candidates: Vec<Box<dyn Kernel>> = vec![Box::new(RBF::new(1.0)), Box::new(Linear::new(1.0, 1.0))];
/// let selector = KernelSelector::new(candidates).criterion(SelectionCriterion::Bic);
/// let result = selector.select(&X, &y).expect("selection should succeed with valid data");
/// ```
#[derive(Debug, Clone)]
pub struct KernelSelector {
    /// Candidate kernels to evaluate.
    pub candidates: Vec<Box<dyn Kernel>>,
    /// Selection configuration.
    pub config: KernelSelectionConfig,
}

impl KernelSelector {
    /// Create a new selector over the given candidate kernels, using default
    /// configuration (BIC criterion).
    pub fn new(candidates: Vec<Box<dyn Kernel>>) -> Self {
        Self {
            candidates,
            config: KernelSelectionConfig::default(),
        }
    }

    /// Set the full configuration at once.
    pub fn config(mut self, config: KernelSelectionConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the selection criterion.
    pub fn criterion(mut self, criterion: SelectionCriterion) -> Self {
        self.config.criterion = criterion;
        self
    }

    /// Set the number of cross-validation folds (used only for
    /// [`SelectionCriterion::Cv`]).
    pub fn cv_folds(mut self, cv_folds: usize) -> Self {
        self.config.cv_folds = cv_folds;
        self
    }

    /// Set the initial observation noise variance.
    pub fn noise_variance(mut self, noise_variance: f64) -> Self {
        self.config.noise_variance = noise_variance;
        self
    }

    /// Set the random state controlling the CV fold shuffle.
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Run selection against the given training data.
    #[allow(non_snake_case)]
    pub fn select(&self, X: &Array2<f64>, y: &Array1<f64>) -> SklResult<KernelSelectionResult> {
        let candidates: Vec<Box<dyn Kernel>> =
            self.candidates.iter().map(|k| k.clone_box()).collect();
        select_best_kernel_with_config(candidates, X, y, &self.config)
    }
}

/// Select the best kernel among `candidates` using the given `criterion`
/// (with all other selection settings left at their defaults; use
/// [`KernelSelector`] to customize e.g. `cv_folds`).
#[allow(non_snake_case)]
pub fn select_best_kernel(
    candidates: Vec<Box<dyn Kernel>>,
    X: &Array2<f64>,
    y: &Array1<f64>,
    criterion: SelectionCriterion,
) -> SklResult<KernelSelectionResult> {
    let config = KernelSelectionConfig {
        criterion,
        ..KernelSelectionConfig::default()
    };
    select_best_kernel_with_config(candidates, X, y, &config)
}

/// Convenience wrapper: select the best candidate by AIC.
#[allow(non_snake_case)]
pub fn select_kernel_aic(
    candidates: Vec<Box<dyn Kernel>>,
    X: &Array2<f64>,
    y: &Array1<f64>,
) -> SklResult<KernelSelectionResult> {
    select_best_kernel(candidates, X, y, SelectionCriterion::Aic)
}

/// Convenience wrapper: select the best candidate by BIC.
#[allow(non_snake_case)]
pub fn select_kernel_bic(
    candidates: Vec<Box<dyn Kernel>>,
    X: &Array2<f64>,
    y: &Array1<f64>,
) -> SklResult<KernelSelectionResult> {
    select_best_kernel(candidates, X, y, SelectionCriterion::Bic)
}

/// Convenience wrapper: select the best candidate by cross-validated
/// predictive log density.
#[allow(non_snake_case)]
pub fn select_kernel_cv(
    candidates: Vec<Box<dyn Kernel>>,
    X: &Array2<f64>,
    y: &Array1<f64>,
) -> SklResult<KernelSelectionResult> {
    select_best_kernel(candidates, X, y, SelectionCriterion::Cv)
}

#[allow(non_snake_case)]
fn select_best_kernel_with_config(
    candidates: Vec<Box<dyn Kernel>>,
    X: &Array2<f64>,
    y: &Array1<f64>,
    config: &KernelSelectionConfig,
) -> SklResult<KernelSelectionResult> {
    if candidates.is_empty() {
        return Err(SklearsError::InvalidInput(
            "At least one candidate kernel is required".to_string(),
        ));
    }
    if X.nrows() != y.len() {
        return Err(SklearsError::InvalidInput(
            "X and y must have the same number of samples".to_string(),
        ));
    }

    let higher_is_better = config.criterion.higher_is_better();

    let mut all_scores = Vec::with_capacity(candidates.len());
    let mut best: Option<(Box<dyn Kernel>, f64)> = None;
    let mut best_index = 0usize;

    for (idx, candidate) in candidates.iter().enumerate() {
        let (optimized_kernel, score) = evaluate_candidate(candidate.as_ref(), X, y, config)?;
        all_scores.push((kernel_name(optimized_kernel.as_ref()), score));

        let is_better = match &best {
            None => true,
            Some((_, best_score)) => {
                if higher_is_better {
                    score > *best_score
                } else {
                    score < *best_score
                }
            }
        };
        if is_better {
            best = Some((optimized_kernel, score));
            best_index = idx;
        }
    }

    let (best_kernel, best_score) = best.ok_or_else(|| {
        SklearsError::InvalidInput("At least one candidate kernel is required".to_string())
    })?;
    let best_kernel_name = kernel_name(best_kernel.as_ref());

    Ok(KernelSelectionResult {
        best_kernel,
        best_index,
        best_kernel_name,
        best_score,
        all_scores,
    })
}

/// Optimize one candidate's hyperparameters (and noise level) against the
/// log marginal likelihood, then score it under `config.criterion`.
#[allow(non_snake_case)]
fn evaluate_candidate(
    kernel: &dyn Kernel,
    X: &Array2<f64>,
    y: &Array1<f64>,
    config: &KernelSelectionConfig,
) -> SklResult<(Box<dyn Kernel>, f64)> {
    let n = X.nrows();
    let mut kernel_opt = kernel.clone_box();
    let sigma_n_init = config.noise_variance.max(1e-12).sqrt();

    let opt_result = optimize_hyperparameters(
        &X.view(),
        &y.view(),
        kernel_opt.as_mut(),
        sigma_n_init,
        CANDIDATE_OPTIMIZATION_ITERS,
        CANDIDATE_OPTIMIZATION_TOL,
        false,
    )?;
    let sigma_n = opt_result.optimal_params.last().copied().ok_or_else(|| {
        SklearsError::NumericalError(
            "hyperparameter optimization returned no parameters".to_string(),
        )
    })?;

    let score = match config.criterion {
        SelectionCriterion::Cv => {
            cross_validated_log_density(kernel_opt.as_ref(), X, y, sigma_n * sigma_n, config)?
        }
        SelectionCriterion::Aic
        | SelectionCriterion::Bic
        | SelectionCriterion::LogMarginalLikelihood => {
            let lml =
                log_marginal_likelihood_stable(&X.view(), &y.view(), kernel_opt.as_ref(), sigma_n)?;
            let k = kernel_opt.get_params().len() as f64 + 1.0; // +1 for the noise level
            match config.criterion {
                SelectionCriterion::Aic => -2.0 * lml + 2.0 * k,
                SelectionCriterion::Bic => -2.0 * lml + k * (n as f64).ln(),
                _ => lml, // LogMarginalLikelihood
            }
        }
    };

    Ok((kernel_opt, score))
}

/// Mean held-out predictive log density under k-fold cross-validation.
#[allow(non_snake_case)]
fn cross_validated_log_density(
    kernel: &dyn Kernel,
    X: &Array2<f64>,
    y: &Array1<f64>,
    noise_variance: f64,
    config: &KernelSelectionConfig,
) -> SklResult<f64> {
    let n = X.nrows();
    let k_folds = config.cv_folds.clamp(2, n.max(2));

    let mut indices: Vec<usize> = (0..n).collect();
    if let Some(seed) = config.random_state {
        let mut rng = seed;
        for i in (1..indices.len()).rev() {
            rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let j = (rng as usize) % (i + 1);
            indices.swap(i, j);
        }
    }

    let fold_size = n / k_folds;
    let mut total_log_density = 0.0;
    let mut total_count = 0usize;

    for fold in 0..k_folds {
        let start = fold * fold_size;
        let end = if fold == k_folds - 1 {
            n
        } else {
            (fold + 1) * fold_size
        };
        if start == end {
            continue;
        }

        let val_idx = &indices[start..end];
        let train_idx: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|i| *i < start || *i >= end)
            .collect();
        if train_idx.is_empty() {
            continue;
        }

        let X_train = select_rows(X, &train_idx);
        let y_train = select_entries(y, &train_idx);
        let X_val = select_rows(X, val_idx);
        let y_val = select_entries(y, val_idx);

        let model = GaussianProcessRegressor::new()
            .kernel(kernel.clone_box())
            .alpha(noise_variance)
            .fit(&X_train, &y_train)?;
        let (mean, std) = model.predict_with_std(&X_val)?;

        for i in 0..X_val.nrows() {
            let var = (std[i] * std[i] + noise_variance).max(1e-12);
            let residual = y_val[i] - mean[i];
            let point_log_density = -0.5 * (residual * residual / var + var.ln() + (2.0 * PI).ln());
            total_log_density += point_log_density;
            total_count += 1;
        }
    }

    if total_count == 0 {
        return Err(SklearsError::InvalidInput(
            "Cross-validation produced no held-out predictions; check cv_folds and dataset size"
                .to_string(),
        ));
    }

    Ok(total_log_density / total_count as f64)
}

#[allow(non_snake_case)]
fn select_rows(X: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((idx.len(), X.ncols()));
    for (new_i, &old_i) in idx.iter().enumerate() {
        out.row_mut(new_i).assign(&X.row(old_i));
    }
    out
}

fn select_entries(y: &Array1<f64>, idx: &[usize]) -> Array1<f64> {
    Array1::from_iter(idx.iter().map(|&i| y[i]))
}

/// Extract a short display name for a kernel from its `Debug` representation
/// (e.g. `"RBF { length_scale: 1.0 }"` -> `"RBF"`).
fn kernel_name(kernel: &dyn Kernel) -> String {
    let debug_repr = format!("{kernel:?}");
    match debug_repr.find(['{', '(']) {
        Some(idx) => debug_repr[..idx].trim().to_string(),
        None => debug_repr,
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::kernels::{Linear, RBF};
    use crate::utils::robust_cholesky;

    /// Deterministic standard-normal samples via a seeded 64-bit LCG +
    /// Box-Muller transform, mirroring the RNG-free sampling style already
    /// used elsewhere in this crate (see
    /// `regression.rs::MultiOutputGaussianProcessRegressor::uniform`).
    fn standard_normal_vec(n: usize, seed: u64) -> Array1<f64> {
        let mut state = seed.max(1);
        let mut next_uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64).clamp(1e-12, 1.0 - 1e-12)
        };
        Array1::from_iter((0..n).map(|_| {
            let u1 = next_uniform();
            let u2 = next_uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }))
    }

    /// Sample `y = L @ z` where `L` is the Cholesky factor of a known RBF
    /// Gram matrix (plus a small noise variance on the diagonal), so the
    /// data is drawn from a GP with a known, correct kernel family.
    fn rbf_generated_data(
        n: usize,
        length_scale: f64,
        noise_std: f64,
        seed: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let X = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 * 10.0 / n as f64);
        let kernel = RBF::new(length_scale);
        let mut K = kernel
            .compute_kernel_matrix(&X, None)
            .expect("kernel matrix computation should succeed");
        for i in 0..n {
            K[[i, i]] += noise_std * noise_std + 1e-8;
        }
        let L = robust_cholesky(&K).expect("Cholesky of RBF Gram matrix should succeed");
        let z = standard_normal_vec(n, seed);
        let y = L.dot(&z);
        (X, y)
    }

    #[test]
    fn test_select_kernel_aic_prefers_rbf_on_rbf_data() {
        let (X, y) = rbf_generated_data(24, 1.0, 0.05, 7);
        let candidates: Vec<Box<dyn Kernel>> =
            vec![Box::new(RBF::new(1.0)), Box::new(Linear::new(1.0, 1.0))];
        let result = select_best_kernel(candidates, &X, &y, SelectionCriterion::Aic)
            .expect("kernel selection should succeed");
        assert_eq!(result.best_kernel_name, "RBF");
        assert_eq!(result.best_index, 0);
    }

    #[test]
    fn test_select_kernel_bic_prefers_rbf_on_rbf_data() {
        let (X, y) = rbf_generated_data(24, 1.0, 0.05, 11);
        let candidates: Vec<Box<dyn Kernel>> =
            vec![Box::new(RBF::new(1.0)), Box::new(Linear::new(1.0, 1.0))];
        let result =
            select_kernel_bic(candidates, &X, &y).expect("kernel selection should succeed");
        assert_eq!(result.best_kernel_name, "RBF");
        assert_eq!(result.all_scores.len(), 2);
    }

    #[test]
    fn test_select_kernel_cv_runs_and_scores_all_candidates() {
        let (X, y) = rbf_generated_data(24, 1.0, 0.05, 13);
        let candidates: Vec<Box<dyn Kernel>> =
            vec![Box::new(RBF::new(1.0)), Box::new(Linear::new(1.0, 1.0))];
        let result =
            select_kernel_cv(candidates, &X, &y).expect("CV kernel selection should succeed");
        assert_eq!(result.all_scores.len(), 2);
        assert!(result.best_score.is_finite());
    }

    #[test]
    fn test_kernel_selector_builder() {
        let candidates: Vec<Box<dyn Kernel>> = vec![Box::new(RBF::new(1.0))];
        let selector = KernelSelector::new(candidates).criterion(SelectionCriterion::Bic);
        assert_eq!(selector.config.criterion, SelectionCriterion::Bic);
    }

    #[test]
    fn test_select_best_kernel_rejects_empty_candidates() {
        let (X, y) = rbf_generated_data(10, 1.0, 0.05, 3);
        let result = select_best_kernel(Vec::new(), &X, &y, SelectionCriterion::Aic);
        assert!(result.is_err());
    }

    #[test]
    fn test_selection_criterion_direction() {
        assert!(!SelectionCriterion::Aic.higher_is_better());
        assert!(!SelectionCriterion::Bic.higher_is_better());
        assert!(SelectionCriterion::Cv.higher_is_better());
        assert!(SelectionCriterion::LogMarginalLikelihood.higher_is_better());
    }
}
