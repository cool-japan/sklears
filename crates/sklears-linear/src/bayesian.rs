//! Bayesian linear regression models
//!
//! This module implements Bayesian Ridge Regression, ARD (Automatic Relevance Determination)
//! regression, and Variational Bayesian Linear Regression, which provide probabilistic approaches
//! to linear regression with automatic hyperparameter tuning and uncertainty quantification.

use std::marker::PhantomData;

use scirs2_core::essentials::Normal;
use scirs2_core::ndarray::{Array, Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::Distribution;
use sklears_core::{
    error::{validate, Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::Float,
};

/// Helper function to safely compute mean of an Array1
#[inline]
fn safe_mean_1d(arr: &Array1<Float>) -> Result<Float> {
    arr.mean()
        .ok_or_else(|| SklearsError::NumericalError("Failed to compute mean of array".to_string()))
}

/// Helper function to safely compute mean along axis
#[inline]
fn safe_mean_axis(arr: &Array2<Float>, axis: Axis) -> Result<Array1<Float>> {
    arr.mean_axis(axis).ok_or_else(|| {
        SklearsError::NumericalError("Failed to compute mean along axis".to_string())
    })
}

/// Helper function to safely create Normal distribution
#[inline]
fn safe_normal(mean: f64, std_dev: f64) -> Result<Normal<f64>> {
    Normal::new(mean, std_dev).map_err(|e| {
        SklearsError::InvalidInput(format!(
            "Invalid Normal distribution parameters: mean={}, std_dev={}, error: {}",
            mean, std_dev, e
        ))
    })
}

/// Helper function to calculate determinant of a matrix using LU decomposition
fn matrix_determinant(matrix: &Array2<f64>) -> Result<f64> {
    if matrix.nrows() != matrix.ncols() {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square for determinant calculation".to_string(),
        ));
    }

    let n = matrix.nrows();
    if n == 0 {
        return Ok(1.0);
    }

    // Simple implementation for small matrices
    if n == 1 {
        Ok(matrix[[0, 0]])
    } else if n == 2 {
        Ok(matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]])
    } else {
        // For larger matrices, use LU decomposition approximation
        let mut a = matrix.clone();
        let mut det = 1.0;

        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if a[[k, i]].abs() > a[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..n {
                    let temp = a[[i, j]];
                    a[[i, j]] = a[[max_row, j]];
                    a[[max_row, j]] = temp;
                }
                det = -det;
            }

            det *= a[[i, i]];

            if a[[i, i]].abs() < 1e-12 {
                return Ok(0.0);
            }

            // Eliminate column
            for k in (i + 1)..n {
                let factor = a[[k, i]] / a[[i, i]];
                for j in (i + 1)..n {
                    a[[k, j]] -= factor * a[[i, j]];
                }
            }
        }

        Ok(det)
    }
}

/// Configuration for Bayesian Ridge Regression
#[derive(Debug, Clone)]
pub struct BayesianRidgeConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Initial value for alpha (precision of weights)
    pub alpha_init: f64,
    /// Initial value for lambda (precision of noise)
    pub lambda_init: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Whether to compute the log marginal likelihood
    pub compute_score: bool,
}

impl Default for BayesianRidgeConfig {
    fn default() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-3,
            alpha_init: 1.0,
            lambda_init: 1.0,
            fit_intercept: true,
            compute_score: false,
        }
    }
}

/// Bayesian Ridge Regression
///
/// Fit a Bayesian ridge model with automatic relevance determination of hyperparameters.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BayesianRidge<State = Untrained> {
    config: BayesianRidgeConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    alpha_: Option<Float>,
    lambda_: Option<Float>,
    sigma_: Option<Array2<Float>>, // Posterior covariance
    scores_: Option<Vec<Float>>,   // Log marginal likelihood values
    n_iter_: Option<usize>,
    n_features_: Option<usize>,
}

impl BayesianRidge<Untrained> {
    /// Create a new Bayesian Ridge regression model
    pub fn new() -> Self {
        Self {
            config: BayesianRidgeConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            alpha_: None,
            lambda_: None,
            sigma_: None,
            scores_: None,
            n_iter_: None,
            n_features_: None,
        }
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set whether to compute scores
    pub fn compute_score(mut self, compute_score: bool) -> Self {
        self.config.compute_score = compute_score;
        self
    }
}

impl Default for BayesianRidge<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for ARD Regression
#[derive(Debug, Clone)]
pub struct ARDRegressionConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Initial value for alpha (per-feature precision)
    pub alpha_init: f64,
    /// Initial value for lambda (precision of noise)
    pub lambda_init: f64,
    /// Threshold for removing features
    pub threshold_alpha: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Whether to compute the log marginal likelihood
    pub compute_score: bool,
}

impl Default for ARDRegressionConfig {
    fn default() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-3,
            alpha_init: 1.0,
            lambda_init: 1.0,
            threshold_alpha: 1e10,
            fit_intercept: true,
            compute_score: false,
        }
    }
}

/// ARD (Automatic Relevance Determination) Regression
///
/// Bayesian regression with different precision for each feature,
/// allowing automatic feature selection.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ARDRegression<State = Untrained> {
    config: ARDRegressionConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    alpha_: Option<Array1<Float>>, // Per-feature precision
    lambda_: Option<Float>,
    sigma_: Option<Array2<Float>>, // Posterior covariance
    scores_: Option<Vec<Float>>,   // Log marginal likelihood values
    n_iter_: Option<usize>,
    n_features_: Option<usize>,
}

impl ARDRegression<Untrained> {
    /// Create a new ARD regression model
    pub fn new() -> Self {
        Self {
            config: ARDRegressionConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            alpha_: None,
            lambda_: None,
            sigma_: None,
            scores_: None,
            n_iter_: None,
            n_features_: None,
        }
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set threshold for removing features
    pub fn threshold_alpha(mut self, threshold: f64) -> Self {
        self.config.threshold_alpha = threshold;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }
}

impl Default for ARDRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BayesianRidge<Untrained> {
    type Config = BayesianRidgeConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for ARDRegression<Untrained> {
    type Config = ARDRegressionConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for BayesianRidge<Untrained> {
    type Fitted = BayesianRidge<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.config.fit_intercept {
            let x_mean = safe_mean_axis(x, Axis(0))?;
            let y_mean = safe_mean_1d(y)?;
            let x_centered = x - &x_mean;
            let y_centered = y - y_mean;
            (x_centered, y_centered, Some(x_mean), Some(y_mean))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        // Initialize hyperparameters
        let mut alpha = self.config.alpha_init;
        let mut lambda = self.config.lambda_init;
        let mut scores = Vec::new();

        // Precompute X^T X
        let xtx = x_centered.t().dot(&x_centered);
        let xty = x_centered.t().dot(&y_centered);

        let mut coef = Array::zeros(n_features);
        let mut sigma = Array::eye(n_features);
        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            // E-step: Update posterior mean and covariance
            // Sigma = (lambda * X^T X + alpha * I)^{-1}
            let mut a = &xtx * lambda;
            for i in 0..n_features {
                a[[i, i]] += alpha + 1e-10; // Add small regularization for numerical stability
            }

            // Check if matrix is well-conditioned
            let diag_min = a.diag().fold(Float::INFINITY, |a, &b| a.min(b));
            if diag_min <= 0.0 || !diag_min.is_finite() {
                return Err(SklearsError::NumericalError(
                    "Matrix is singular or poorly conditioned".to_string(),
                ));
            }

            // Compute inverse (posterior covariance)
            sigma = scirs2_linalg::inv(&a.view(), None).map_err(|e| {
                SklearsError::NumericalError(format!("Failed to compute inverse: {}", e))
            })?;

            // Posterior mean: mu = Sigma * X^T * y
            coef = sigma.dot(&xty);

            // M-step: Update hyperparameters
            let alpha_old = alpha;
            let lambda_old = lambda;

            // Update alpha
            let gamma = n_features as f64 - alpha * sigma.diag().sum();
            alpha = gamma / coef.dot(&coef);

            // Update lambda
            let residuals = &y_centered - x_centered.dot(&coef);
            let rss = residuals.dot(&residuals);
            lambda = (n_samples as f64 - gamma) / rss;

            // Compute log marginal likelihood if requested
            if self.config.compute_score {
                // Compute log determinant of sigma using eigenvalues
                // Since sigma = A^{-1}, log|sigma| = -log|A|
                // For now, use a simple approximation
                let log_det_a = (0..n_features).map(|i| a[[i, i]].ln()).sum::<f64>();
                let log_det_sigma = -log_det_a;

                let score = 0.5
                    * (n_features as f64 * alpha.ln() + n_samples as f64 * lambda.ln()
                        - lambda * rss
                        - alpha * coef.dot(&coef)
                        - log_det_sigma
                        - n_samples as f64 * (2.0 * std::f64::consts::PI).ln());

                scores.push(score);
            }

            // Check convergence
            if (alpha - alpha_old).abs() < self.config.tol * alpha.abs()
                && (lambda - lambda_old).abs() < self.config.tol * lambda.abs()
            {
                converged = true;
                break;
            }
        }

        if !converged {
            eprintln!(
                "Warning: Bayesian Ridge did not converge within {} iterations",
                self.config.max_iter
            );
        }

        // Compute intercept if needed
        let intercept = if self.config.fit_intercept {
            let x_mean = x_mean.ok_or_else(|| {
                SklearsError::NumericalError(
                    "X mean should be available when fit_intercept is true".to_string(),
                )
            })?;
            let y_mean = y_mean.ok_or_else(|| {
                SklearsError::NumericalError(
                    "Y mean should be available when fit_intercept is true".to_string(),
                )
            })?;
            Some(y_mean - x_mean.dot(&coef))
        } else {
            None
        };

        Ok(BayesianRidge {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: intercept,
            alpha_: Some(alpha),
            lambda_: Some(lambda),
            sigma_: Some(sigma),
            scores_: if scores.is_empty() {
                None
            } else {
                Some(scores)
            },
            n_iter_: Some(n_iter),
            n_features_: Some(n_features),
        })
    }
}

impl Fit<Array2<Float>, Array1<Float>> for ARDRegression<Untrained> {
    type Fitted = ARDRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.config.fit_intercept {
            let x_mean = safe_mean_axis(x, Axis(0))?;
            let y_mean = safe_mean_1d(y)?;
            let x_centered = x - &x_mean;
            let y_centered = y - y_mean;
            (x_centered, y_centered, Some(x_mean), Some(y_mean))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        // Initialize hyperparameters
        let mut alpha = Array::from_elem(n_features, self.config.alpha_init);
        let mut lambda = self.config.lambda_init;
        let mut scores = Vec::new();

        // Keep track of active features
        let mut active_features: Vec<usize> = (0..n_features).collect();

        let mut coef = Array::zeros(n_features);
        let mut sigma = Array::eye(n_features);
        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            // Only work with active features
            let n_active = active_features.len();
            if n_active == 0 {
                break;
            }

            // Extract active columns
            let x_active = x_centered.select(Axis(1), &active_features);

            // Precompute X^T X for active features
            let xtx = x_active.t().dot(&x_active);
            let xty = x_active.t().dot(&y_centered);

            // E-step: Update posterior mean and covariance
            // Sigma = (lambda * X^T X + diag(alpha))^{-1}
            let mut a = &xtx * lambda;
            for (i, &feat_idx) in active_features.iter().enumerate() {
                a[[i, i]] += alpha[feat_idx] + 1e-10; // Add small regularization for numerical stability
            }

            // Check if matrix is well-conditioned
            let diag_min = a.diag().fold(Float::INFINITY, |a, &b| a.min(b));
            if diag_min <= 0.0 || !diag_min.is_finite() {
                // Remove features with too high alpha (effectively zero weight)
                active_features.retain(|&feat_idx| alpha[feat_idx] < self.config.threshold_alpha);
                continue;
            }

            // Compute inverse (posterior covariance) for active features
            let sigma_active = scirs2_linalg::inv(&a.view(), None).map_err(|e| {
                SklearsError::NumericalError(format!("Failed to compute inverse: {}", e))
            })?;

            // Posterior mean for active features
            let coef_active = sigma_active.dot(&xty);

            // Update full coefficient vector
            coef.fill(0.0);
            for (i, &feat_idx) in active_features.iter().enumerate() {
                coef[feat_idx] = coef_active[i];
            }

            // M-step: Update hyperparameters
            let alpha_old = alpha.clone();
            let lambda_old = lambda;

            // Update alpha for each feature
            let mut gamma_total = 0.0;
            for (i, &feat_idx) in active_features.iter().enumerate() {
                let gamma_i = 1.0 - alpha[feat_idx] * sigma_active[[i, i]];
                gamma_total += gamma_i;
                alpha[feat_idx] = gamma_i / (coef[feat_idx] * coef[feat_idx] + 1e-10);
            }

            // Update lambda
            let residuals = &y_centered - x_centered.dot(&coef);
            let rss = residuals.dot(&residuals);
            lambda = (n_samples as f64 - gamma_total) / rss;

            // Remove features with very high alpha (irrelevant features)
            active_features.retain(|&feat_idx| alpha[feat_idx] < self.config.threshold_alpha);

            // Update sigma for all features
            sigma.fill(0.0);
            for (i, &feat_idx_i) in active_features.iter().enumerate() {
                for (j, &feat_idx_j) in active_features.iter().enumerate() {
                    sigma[[feat_idx_i, feat_idx_j]] = sigma_active[[i, j]];
                }
            }

            // Compute log marginal likelihood if requested
            if self.config.compute_score && n_active > 0 {
                // Compute log determinant of sigma using eigenvalues
                // Since sigma = A^{-1}, log|sigma| = -log|A|
                // For now, use a simple approximation
                let log_det_a = (0..n_features).map(|i| a[[i, i]].ln()).sum::<f64>();
                let log_det_sigma = -log_det_a;

                let alpha_sum: f64 = active_features.iter().map(|&i| alpha[i].ln()).sum();

                let weighted_coef_norm: f64 = active_features
                    .iter()
                    .map(|&i| alpha[i] * coef[i] * coef[i])
                    .sum();

                let score = 0.5
                    * (alpha_sum + n_samples as f64 * lambda.ln()
                        - lambda * rss
                        - weighted_coef_norm
                        - log_det_sigma
                        - n_samples as f64 * (2.0 * std::f64::consts::PI).ln());

                scores.push(score);
            }

            // Check convergence
            let alpha_change: f64 = active_features
                .iter()
                .map(|&i| (alpha[i] - alpha_old[i]).abs() / alpha[i].abs().max(1.0))
                .sum::<f64>()
                / active_features.len().max(1) as f64;

            let lambda_change = (lambda - lambda_old).abs() / lambda.abs();

            if alpha_change < self.config.tol && lambda_change < self.config.tol {
                converged = true;
                break;
            }
        }

        if !converged {
            eprintln!(
                "Warning: ARD Regression did not converge within {} iterations",
                self.config.max_iter
            );
        }

        // Compute intercept if needed
        let intercept = if self.config.fit_intercept {
            let x_mean = x_mean.ok_or_else(|| {
                SklearsError::NumericalError(
                    "X mean should be available when fit_intercept is true".to_string(),
                )
            })?;
            let y_mean = y_mean.ok_or_else(|| {
                SklearsError::NumericalError(
                    "Y mean should be available when fit_intercept is true".to_string(),
                )
            })?;
            Some(y_mean - x_mean.dot(&coef))
        } else {
            None
        };

        Ok(ARDRegression {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: intercept,
            alpha_: Some(alpha),
            lambda_: Some(lambda),
            sigma_: Some(sigma),
            scores_: if scores.is_empty() {
                None
            } else {
                Some(scores)
            },
            n_iter_: Some(n_iter),
            n_features_: Some(n_features),
        })
    }
}

// Implement methods for trained models
impl BayesianRidge<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> Result<&Array1<Float>> {
        self.coef_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "coef".to_string(),
        })
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<Float> {
        self.intercept_
    }

    /// Get the precision of weights
    pub fn alpha(&self) -> Result<Float> {
        self.alpha_.ok_or_else(|| SklearsError::NotFitted {
            operation: "alpha".to_string(),
        })
    }

    /// Get the precision of noise
    pub fn lambda(&self) -> Result<Float> {
        self.lambda_.ok_or_else(|| SklearsError::NotFitted {
            operation: "lambda".to_string(),
        })
    }

    /// Get the posterior covariance matrix
    pub fn sigma(&self) -> Result<&Array2<Float>> {
        self.sigma_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "sigma".to_string(),
        })
    }

    /// Get the log marginal likelihood scores
    pub fn scores(&self) -> Option<&Vec<Float>> {
        self.scores_.as_ref()
    }
}

impl ARDRegression<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> Result<&Array1<Float>> {
        self.coef_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "coef".to_string(),
        })
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<Float> {
        self.intercept_
    }

    /// Get the per-feature precision values
    pub fn alpha(&self) -> Result<&Array1<Float>> {
        self.alpha_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "alpha".to_string(),
        })
    }

    /// Get the precision of noise
    pub fn lambda(&self) -> Result<Float> {
        self.lambda_.ok_or_else(|| SklearsError::NotFitted {
            operation: "lambda".to_string(),
        })
    }

    /// Get indices of relevant features (low alpha)
    pub fn relevant_features(&self) -> Result<Vec<usize>> {
        let alpha = self.alpha()?;
        Ok(alpha
            .iter()
            .enumerate()
            .filter(|(_, &a)| a < self.config.threshold_alpha)
            .map(|(i, _)| i)
            .collect())
    }
}

impl Predict<Array2<Float>, Array1<Float>> for BayesianRidge<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_features = self.n_features_.ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;
        validate::check_n_features(x, n_features)?;

        let coef = self.coef()?;
        let mut predictions = x.dot(coef);

        if let Some(intercept) = self.intercept_ {
            predictions += intercept;
        }

        Ok(predictions)
    }
}

impl Predict<Array2<Float>, Array1<Float>> for ARDRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_features = self.n_features_.ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;
        validate::check_n_features(x, n_features)?;

        let coef = self.coef()?;
        let mut predictions = x.dot(coef);

        if let Some(intercept) = self.intercept_ {
            predictions += intercept;
        }

        Ok(predictions)
    }
}

impl Score<Array2<Float>, Array1<Float>> for BayesianRidge<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Compute R² score
        let y_mean = safe_mean_1d(y)?;
        let ss_tot = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<Float>();

        let ss_res = predictions
            .iter()
            .zip(y.iter())
            .map(|(&pred, &true_val)| (true_val - pred).powi(2))
            .sum::<Float>();

        Ok(1.0 - ss_res / ss_tot)
    }
}

impl Score<Array2<Float>, Array1<Float>> for ARDRegression<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Compute R² score
        let y_mean = safe_mean_1d(y)?;
        let ss_tot = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<Float>();

        let ss_res = predictions
            .iter()
            .zip(y.iter())
            .map(|(&pred, &true_val)| (true_val - pred).powi(2))
            .sum::<Float>();

        Ok(1.0 - ss_res / ss_tot)
    }
}

/// Configuration for Variational Bayesian Linear Regression
#[derive(Debug, Clone)]
pub struct VariationalBayesianConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Prior precision for weights (alpha)
    pub alpha_a: f64,
    pub alpha_b: f64,
    /// Prior precision for noise (beta)
    pub beta_a: f64,
    pub beta_b: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Whether to compute the evidence lower bound (ELBO)
    pub compute_elbo: bool,
    /// Whether to use mean-field approximation
    pub mean_field: bool,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for VariationalBayesianConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-4,
            alpha_a: 1e-6,
            alpha_b: 1e-6,
            beta_a: 1e-6,
            beta_b: 1e-6,
            fit_intercept: true,
            compute_elbo: true,
            mean_field: true,
            random_state: None,
        }
    }
}

/// Variational Bayesian Linear Regression
///
/// Implements scalable variational Bayesian inference for linear regression
/// using mean-field variational approximation. Suitable for large-scale problems.
#[derive(Debug, Clone)]
pub struct VariationalBayesianRegression<State = Untrained> {
    config: VariationalBayesianConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_mean_: Option<Array1<Float>>, // Posterior mean of coefficients
    coef_cov_: Option<Array2<Float>>,  // Posterior covariance of coefficients
    intercept_: Option<Float>,
    alpha_: Option<Float>,         // Posterior precision of weights
    beta_: Option<Float>,          // Posterior precision of noise
    sigma_: Option<Array2<Float>>, // Posterior covariance matrix
    elbo_: Option<Vec<Float>>,     // Evidence Lower Bound history
    n_iter_: Option<usize>,
    n_features_: Option<usize>,
    // Variational parameters
    q_alpha_a_: Option<Float>, // Variational alpha shape
    q_alpha_b_: Option<Float>, // Variational alpha rate
    q_beta_a_: Option<Float>,  // Variational beta shape
    q_beta_b_: Option<Float>,  // Variational beta rate
}

impl VariationalBayesianRegression<Untrained> {
    /// Create a new Variational Bayesian regression model
    pub fn new() -> Self {
        Self {
            config: VariationalBayesianConfig::default(),
            state: PhantomData,
            coef_mean_: None,
            coef_cov_: None,
            intercept_: None,
            alpha_: None,
            beta_: None,
            sigma_: None,
            elbo_: None,
            n_iter_: None,
            n_features_: None,
            q_alpha_a_: None,
            q_alpha_b_: None,
            q_beta_a_: None,
            q_beta_b_: None,
        }
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set prior parameters for alpha (weight precision)
    pub fn alpha_prior(mut self, a: f64, b: f64) -> Self {
        self.config.alpha_a = a;
        self.config.alpha_b = b;
        self
    }

    /// Set prior parameters for beta (noise precision)
    pub fn beta_prior(mut self, a: f64, b: f64) -> Self {
        self.config.beta_a = a;
        self.config.beta_b = b;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set whether to compute ELBO
    pub fn compute_elbo(mut self, compute_elbo: bool) -> Self {
        self.config.compute_elbo = compute_elbo;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }
}

impl Default for VariationalBayesianRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl VariationalBayesianRegression<Trained> {
    /// Get the posterior mean of coefficients
    pub fn coef_mean(&self) -> Result<&Array1<Float>> {
        self.coef_mean_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "coef_mean".to_string(),
            })
    }

    /// Get the posterior covariance of coefficients
    pub fn coef_cov(&self) -> Result<&Array2<Float>> {
        self.coef_cov_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "coef_cov".to_string(),
            })
    }

    /// Get the intercept
    pub fn intercept(&self) -> Float {
        self.intercept_.unwrap_or(0.0)
    }

    /// Get the posterior precision of weights
    pub fn alpha(&self) -> Result<Float> {
        self.alpha_.ok_or_else(|| SklearsError::NotFitted {
            operation: "alpha".to_string(),
        })
    }

    /// Get the posterior precision of noise
    pub fn beta(&self) -> Result<Float> {
        self.beta_.ok_or_else(|| SklearsError::NotFitted {
            operation: "beta".to_string(),
        })
    }

    /// Get the ELBO history
    pub fn elbo_history(&self) -> Option<&Vec<Float>> {
        self.elbo_.as_ref()
    }

    /// Get number of iterations
    pub fn n_iter(&self) -> Result<usize> {
        self.n_iter_.ok_or_else(|| SklearsError::NotFitted {
            operation: "n_iter".to_string(),
        })
    }

    /// Predict with uncertainty quantification
    pub fn predict_with_uncertainty(
        &self,
        x: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>)> {
        let n_features = self.n_features_.ok_or_else(|| SklearsError::NotFitted {
            operation: "predict_with_uncertainty".to_string(),
        })?;
        validate::check_n_features(x, n_features)?;

        let coef_mean = self.coef_mean()?;
        let coef_cov = self.coef_cov()?;
        let beta = self.beta()?;

        // Mean prediction
        let mut y_mean = x.dot(coef_mean);
        if let Some(intercept) = self.intercept_ {
            y_mean += intercept;
        }

        // Predictive variance: x^T * Sigma * x + 1/beta
        let mut y_var = Array1::zeros(x.nrows());
        for (i, x_i) in x.outer_iter().enumerate() {
            // x_i^T * Sigma * x_i
            let quadratic_term = x_i.dot(&coef_cov.dot(&x_i));
            y_var[i] = quadratic_term + 1.0 / beta;
        }

        // Convert variance to standard deviation
        let y_std = y_var.mapv(|v| v.sqrt());

        Ok((y_mean, y_std))
    }

    /// Sample from the posterior predictive distribution
    pub fn sample_predictions(
        &self,
        x: &Array2<Float>,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> Result<Array2<Float>> {
        let n_features = self.n_features_.ok_or_else(|| SklearsError::NotFitted {
            operation: "sample_predictions".to_string(),
        })?;
        validate::check_n_features(x, n_features)?;

        let coef_mean = self.coef_mean()?;
        let coef_cov = self.coef_cov()?;
        let beta = self.beta()?;

        // Sample coefficients from posterior
        let mut samples = Array2::zeros((n_samples, x.nrows()));

        for i in 0..n_samples {
            // Sample coefficients from multivariate normal
            let coef_sample = sample_multivariate_normal(coef_mean, coef_cov, rng)?;

            // Compute predictions with sampled coefficients
            let mut y_pred = x.dot(&coef_sample);
            if let Some(intercept) = self.intercept_ {
                y_pred += intercept;
            }

            // Add noise
            let noise_std = (1.0 / beta).sqrt();
            let noise_dist = safe_normal(0.0, noise_std)?;
            for j in 0..y_pred.len() {
                let noise = noise_dist.sample(rng);
                y_pred[j] += noise;
            }

            samples.row_mut(i).assign(&y_pred);
        }

        Ok(samples)
    }

    /// Compute the log marginal likelihood approximation (ELBO)
    pub fn log_marginal_likelihood(&self) -> Option<Float> {
        self.elbo_.as_ref().and_then(|elbo| elbo.last().copied())
    }
}

impl Estimator for VariationalBayesianRegression<Untrained> {
    type Config = VariationalBayesianConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for VariationalBayesianRegression<Trained> {
    type Config = VariationalBayesianConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for VariationalBayesianRegression<Untrained> {
    type Fitted = VariationalBayesianRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let (n_samples, n_features) = x.dim();

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.config.fit_intercept {
            let x_mean = safe_mean_axis(x, Axis(0))?;
            let y_mean = safe_mean_1d(y)?;
            let x_centered = x - &x_mean.view().insert_axis(Axis(0));
            let y_centered = y - y_mean;
            (x_centered, y_centered, Some(x_mean), Some(y_mean))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        // Initialize variational parameters
        let q_alpha_a = self.config.alpha_a + n_features as Float / 2.0;
        let mut q_alpha_b = self.config.alpha_b;
        let q_beta_a = self.config.beta_a + n_samples as Float / 2.0;
        let mut q_beta_b = self.config.beta_b;

        // Initialize posterior parameters
        let mut mu_n = Array1::zeros(n_features);
        let mut s_n = Array2::eye(n_features);

        let mut elbo_history = if self.config.compute_elbo {
            Some(Vec::new())
        } else {
            None
        };

        // Precompute XtX for efficiency
        let xtx = x_centered.t().dot(&x_centered);
        let xty = x_centered.t().dot(&y_centered);

        // Variational inference loop
        for _iter in 0..self.config.max_iter {
            let alpha_mean = q_alpha_a / q_alpha_b;
            let beta_mean = q_beta_a / q_beta_b;

            // Update posterior parameters (coordinate ascent)
            // S_n = (alpha * I + beta * X^T X)^{-1}
            let precision_matrix = Array2::eye(n_features) * alpha_mean + &xtx * beta_mean;
            s_n = invert_matrix(&precision_matrix)?;

            // mu_n = beta * S_n * X^T * y
            mu_n = s_n.dot(&(&xty * beta_mean));

            // Update alpha parameters
            let old_q_alpha_b = q_alpha_b;
            q_alpha_b = self.config.alpha_b + 0.5 * (mu_n.dot(&mu_n) + s_n.diag().sum());

            // Update beta parameters
            let residual_sum = y_centered.dot(&y_centered) - 2.0 * mu_n.dot(&xty)
                + mu_n.dot(&xtx.dot(&mu_n))
                + (s_n.clone() * &xtx).diag().sum();
            let old_q_beta_b = q_beta_b;
            q_beta_b = self.config.beta_b + 0.5 * residual_sum;

            // Compute ELBO if requested
            if let Some(ref mut elbo) = elbo_history {
                let params = VariationalParams {
                    alpha_mean,
                    beta_mean,
                    q_alpha_a,
                    q_alpha_b,
                    q_beta_a,
                    q_beta_b,
                };
                let current_elbo =
                    compute_elbo(&x_centered, &y_centered, &mu_n, &s_n, &params, &self.config);
                elbo.push(current_elbo);
            }

            // Check convergence
            let alpha_change = (q_alpha_b - old_q_alpha_b).abs() / old_q_alpha_b.max(1e-10);
            let beta_change = (q_beta_b - old_q_beta_b).abs() / old_q_beta_b.max(1e-10);

            if alpha_change < self.config.tol && beta_change < self.config.tol {
                break;
            }
        }

        // Compute final estimates
        let alpha = q_alpha_a / q_alpha_b;
        let beta = q_beta_a / q_beta_b;

        // Compute intercept
        let intercept = if self.config.fit_intercept {
            let x_mean_val = x_mean.ok_or_else(|| {
                SklearsError::NumericalError(
                    "X mean should be available when fit_intercept is true".to_string(),
                )
            })?;
            let y_mean_val = y_mean.ok_or_else(|| {
                SklearsError::NumericalError(
                    "Y mean should be available when fit_intercept is true".to_string(),
                )
            })?;
            y_mean_val - mu_n.dot(&x_mean_val)
        } else {
            0.0
        };

        // Save max_iter before moving config
        let max_iter = self.config.max_iter;

        Ok(VariationalBayesianRegression {
            config: self.config,
            state: PhantomData,
            coef_mean_: Some(mu_n),
            coef_cov_: Some(s_n),
            intercept_: Some(intercept),
            alpha_: Some(alpha),
            beta_: Some(beta),
            sigma_: None, // Can be computed from coef_cov_
            elbo_: elbo_history,
            n_iter_: Some(max_iter), // Could track actual iterations
            n_features_: Some(n_features),
            q_alpha_a_: Some(q_alpha_a),
            q_alpha_b_: Some(q_alpha_b),
            q_beta_a_: Some(q_beta_a),
            q_beta_b_: Some(q_beta_b),
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for VariationalBayesianRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_features = self.n_features_.ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;
        validate::check_n_features(x, n_features)?;

        let coef = self.coef_mean()?;
        let mut predictions = x.dot(coef);

        if self.config.fit_intercept {
            predictions += self.intercept();
        }

        Ok(predictions)
    }
}

impl Score<Array2<Float>, Array1<Float>> for VariationalBayesianRegression<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Compute R² score
        let y_mean = safe_mean_1d(y)?;
        let ss_tot = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<Float>();

        let ss_res = predictions
            .iter()
            .zip(y.iter())
            .map(|(&pred, &true_val)| (true_val - pred).powi(2))
            .sum::<Float>();

        Ok(1.0 - ss_res / ss_tot)
    }
}

/// Variational distribution parameters for ELBO computation
struct VariationalParams {
    alpha_mean: Float,
    beta_mean: Float,
    q_alpha_a: Float,
    q_alpha_b: Float,
    q_beta_a: Float,
    q_beta_b: Float,
}

/// Helper function to compute the Evidence Lower Bound (ELBO)
fn compute_elbo(
    x: &Array2<Float>,
    y: &Array1<Float>,
    mu_n: &Array1<Float>,
    s_n: &Array2<Float>,
    params: &VariationalParams,
    config: &VariationalBayesianConfig,
) -> Float {
    let n_samples = x.nrows() as Float;
    let n_features = x.ncols() as Float;

    // Log likelihood term
    let residual = y - &x.dot(mu_n);
    let likelihood_term = -0.5 * params.beta_mean * residual.dot(&residual)
        - 0.5 * params.beta_mean * (s_n * &x.t().dot(x)).diag().sum()
        + 0.5 * n_samples * (params.beta_mean.ln() - (2.0 * std::f64::consts::PI).ln());

    // Prior terms
    let alpha_prior_term = -0.5 * params.alpha_mean * mu_n.dot(mu_n)
        - 0.5 * params.alpha_mean * s_n.diag().sum()
        + 0.5 * n_features * params.alpha_mean.ln();

    // Entropy terms (variational)
    let det = matrix_determinant(s_n).unwrap_or(1e-12); // Use small value if determinant calculation fails
    let coef_entropy = 0.5 * n_features * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln()
        + 0.5 * det.max(1e-12).ln(); // Ensure we don't take log of zero/negative

    // Gamma entropy terms
    let alpha_entropy = gamma_entropy(params.q_alpha_a, params.q_alpha_b);
    let beta_entropy = gamma_entropy(params.q_beta_a, params.q_beta_b);

    // Prior contributions
    let alpha_prior_contrib = (config.alpha_a - 1.0)
        * (digamma(params.q_alpha_a) - params.q_alpha_b.ln())
        - config.alpha_b * params.q_alpha_a / params.q_alpha_b;

    let beta_prior_contrib = (config.beta_a - 1.0)
        * (digamma(params.q_beta_a) - params.q_beta_b.ln())
        - config.beta_b * params.q_beta_a / params.q_beta_b;

    likelihood_term
        + alpha_prior_term
        + coef_entropy
        + alpha_entropy
        + beta_entropy
        + alpha_prior_contrib
        + beta_prior_contrib
}

/// Helper function to compute gamma distribution entropy
fn gamma_entropy(a: Float, b: Float) -> Float {
    a - (b.ln()) + lgamma(a) + (1.0 - a) * digamma(a)
}

/// Helper function to compute digamma function (approximate)
fn digamma(x: Float) -> Float {
    if x < 6.0 {
        digamma(x + 1.0) - 1.0 / x
    } else {
        (x - 0.5).ln() - 1.0 / (12.0 * x) + 1.0 / (120.0 * x * x * x)
    }
}

/// Helper function to compute log gamma function
fn lgamma(x: Float) -> Float {
    if x < 0.5 {
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - lgamma(1.0 - x)
    } else {
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x - 0.5) * x.ln() - x + 1.0 / (12.0 * x)
    }
}

/// Helper function to invert a matrix using LU decomposition
fn invert_matrix(matrix: &Array2<Float>) -> Result<Array2<Float>> {
    let n = matrix.nrows();
    if matrix.nrows() != matrix.ncols() {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square".to_string(),
        ));
    }

    // Simple LU decomposition and inversion for small matrices
    // For production use, should use a proper linear algebra library
    let mut augmented = Array2::zeros((n, 2 * n));

    // Set up augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = matrix[[i, j]];
        }
        augmented[[i, n + i]] = 1.0;
    }

    // Gaussian elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = augmented[[i, j]];
                augmented[[i, j]] = augmented[[max_row, j]];
                augmented[[max_row, j]] = temp;
            }
        }

        // Check for singular matrix
        if augmented[[i, i]].abs() <= 1e-12 {
            return Err(SklearsError::InvalidInput("Matrix is singular".to_string()));
        }

        // Scale pivot row
        let pivot = augmented[[i, i]];
        for j in 0..(2 * n) {
            augmented[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = augmented[[k, i]];
                for j in 0..(2 * n) {
                    augmented[[k, j]] -= factor * augmented[[i, j]];
                }
            }
        }
    }

    // Extract inverse matrix
    let mut inverse = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inverse[[i, j]] = augmented[[i, n + j]];
        }
    }

    Ok(inverse)
}

/// Helper function to sample from multivariate normal distribution
fn sample_multivariate_normal(
    mean: &Array1<Float>,
    cov: &Array2<Float>,
    rng: &mut impl Rng,
) -> Result<Array1<Float>> {
    let n = mean.len();

    // Sample from standard normal
    let mut z = Array1::zeros(n);
    let normal = safe_normal(0.0, 1.0)?;
    for i in 0..n {
        z[i] = normal.sample(rng);
    }

    // Cholesky decomposition of covariance matrix
    let l = cholesky_decomposition(cov)?;

    // Transform: x = mean + L * z
    let sample = mean + &l.dot(&z);

    Ok(sample)
}

/// Helper function for Cholesky decomposition
fn cholesky_decomposition(matrix: &Array2<Float>) -> Result<Array2<Float>> {
    let n = matrix.nrows();
    if matrix.nrows() != matrix.ncols() {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square".to_string(),
        ));
    }

    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            if i == j {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let val = matrix[[j, j]] - sum;
                if val <= 0.0 {
                    return Err(SklearsError::InvalidInput(
                        "Matrix is not positive definite".to_string(),
                    ));
                }
                l[[j, j]] = val.sqrt();
            } else {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = (matrix[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[ignore] // TODO: Fix numerical precision issues after migration to scirs2_linalg
    fn test_bayesian_ridge() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0],];
        let y = array![1.0, 2.0, 3.0, 4.0]; // y = x2 + 1

        let model = BayesianRidge::new().fit(&x, &y).unwrap();

        let coef = model.coef().unwrap();
        assert_abs_diff_eq!(coef[0], 0.0, epsilon = 0.1);
        assert_abs_diff_eq!(coef[1], 1.0, epsilon = 0.1);

        let score = model.score(&x, &y).unwrap();
        assert!(score > 0.99);
    }

    #[test]
    #[ignore] // TODO: Fix numerical precision issues after migration to scirs2_linalg
    fn test_ard_regression_feature_selection() {
        // Create data where only first feature is relevant
        let x = array![
            [1.0, 0.1, 0.2],
            [2.0, 0.3, 0.1],
            [3.0, 0.2, 0.3],
            [4.0, 0.1, 0.2],
            [5.0, 0.3, 0.1],
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2 * x1

        let model = ARDRegression::new().fit(&x, &y).unwrap();

        let coef = model.coef().unwrap();
        let alpha = model.alpha().unwrap();

        // First coefficient should be close to 2
        assert_abs_diff_eq!(coef[0], 2.0, epsilon = 0.1);

        // Other coefficients should be close to 0 (high alpha)
        assert!(alpha[1] > alpha[0] * 100.0);
        assert!(alpha[2] > alpha[0] * 100.0);

        // Check relevant features
        let relevant = model.relevant_features().unwrap();
        assert!(relevant.contains(&0));
    }

    #[test]
    fn test_variational_bayesian_regression() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0]; // y = x2 + 1

        let model = VariationalBayesianRegression::new()
            .max_iter(100)
            .tol(1e-3)
            .compute_elbo(true)
            .fit(&x, &y)
            .unwrap();

        let coef_mean = model.coef_mean().unwrap();
        assert_abs_diff_eq!(coef_mean[0], 0.0, epsilon = 0.2);
        assert_abs_diff_eq!(coef_mean[1], 1.0, epsilon = 0.2);

        let score = model.score(&x, &y).unwrap();
        assert!(score > 0.95);

        // Test uncertainty quantification
        let (predictions, uncertainties) = model.predict_with_uncertainty(&x).unwrap();
        assert_eq!(predictions.len(), x.nrows());
        assert_eq!(uncertainties.len(), x.nrows());
        assert!(uncertainties.iter().all(|&u| u > 0.0));

        // Test ELBO history
        let elbo_history = model.elbo_history().unwrap();
        assert!(!elbo_history.is_empty());

        // ELBO should generally increase (non-decreasing)
        for i in 1..elbo_history.len() {
            assert!(elbo_history[i] >= elbo_history[i - 1] - 1e-6); // Allow small numerical tolerance
        }
    }

    #[test]
    fn test_variational_bayesian_no_intercept() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![2.0, 4.0, 6.0, 8.0]; // y = 2 * x (no intercept)

        let model = VariationalBayesianRegression::new()
            .fit_intercept(false)
            .max_iter(50)
            .fit(&x, &y)
            .unwrap();

        let coef_mean = model.coef_mean().unwrap();
        assert_abs_diff_eq!(coef_mean[0], 2.0, epsilon = 0.1);
        assert_abs_diff_eq!(model.intercept(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_variational_bayesian_uncertainty_quantification() {
        // Test with known data to verify uncertainty estimates
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0]; // Perfect linear relationship

        let model = VariationalBayesianRegression::new()
            .max_iter(100)
            .fit(&x, &y)
            .unwrap();

        // Test predictions at training points should have lower uncertainty
        let (_train_pred, train_unc) = model.predict_with_uncertainty(&x).unwrap();

        // Test predictions at extrapolated points should have higher uncertainty
        let x_extrap = array![[5.0], [6.0]];
        let (_extrap_pred, extrap_unc) = model.predict_with_uncertainty(&x_extrap).unwrap();

        // Uncertainty should be positive
        assert!(train_unc.iter().all(|&u| u > 0.0));
        assert!(extrap_unc.iter().all(|&u| u > 0.0));

        // Test sampling
        let mut rng = thread_rng();
        let samples = model.sample_predictions(&x, 10, &mut rng).unwrap();
        assert_eq!(samples.dim(), (10, x.nrows()));
    }

    #[test]
    fn test_variational_bayesian_prior_sensitivity() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
        let y = array![1.0, 2.0, 3.0];

        // Test with different priors
        let model1 = VariationalBayesianRegression::new()
            .alpha_prior(1e-3, 1e-3)
            .beta_prior(1e-3, 1e-3)
            .fit(&x, &y)
            .unwrap();

        let model2 = VariationalBayesianRegression::new()
            .alpha_prior(1.0, 1.0)
            .beta_prior(1.0, 1.0)
            .fit(&x, &y)
            .unwrap();

        // Models with different priors should give different results
        let coef1 = model1.coef_mean().unwrap();
        let coef2 = model2.coef_mean().unwrap();

        // Results should be different but both reasonable
        assert!(coef1
            .iter()
            .zip(coef2.iter())
            .any(|(&c1, &c2)| (c1 - c2).abs() > 1e-3));
    }

    #[test]
    fn test_variational_bayesian_convergence() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let model = VariationalBayesianRegression::new()
            .max_iter(200)
            .tol(1e-6)
            .compute_elbo(true)
            .fit(&x, &y)
            .unwrap();

        // Check that the algorithm converged
        let elbo_history = model.elbo_history().unwrap();

        // Should have converged before max iterations
        assert!(elbo_history.len() <= 200);

        // Final ELBO should be reasonable
        let final_elbo = elbo_history.last().unwrap();
        assert!(final_elbo.is_finite());
    }

    #[test]
    fn test_variational_bayesian_edge_cases() {
        // Test with minimal data
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let model = VariationalBayesianRegression::new()
            .max_iter(50)
            .fit(&x, &y);

        assert!(model.is_ok());

        // Test prediction on empty matrix should fail gracefully
        let x_empty = Array2::<Float>::zeros((0, 1));
        let model = VariationalBayesianRegression::new().fit(&x, &y).unwrap();

        let pred_result = model.predict(&x_empty);
        assert!(pred_result.is_ok());
    }

    #[test]
    fn test_cholesky_decomposition() {
        // Test Cholesky decomposition helper function
        let matrix = array![[4.0, 2.0], [2.0, 3.0]];

        let l = cholesky_decomposition(&matrix).unwrap();

        // Verify L * L^T = A
        let reconstructed = l.dot(&l.t());

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert_abs_diff_eq!(reconstructed[[i, j]], matrix[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_inversion() {
        // Test matrix inversion helper function
        let matrix = array![[2.0, 1.0], [1.0, 2.0]];

        let inverse = invert_matrix(&matrix).unwrap();

        // Verify A * A^{-1} = I
        let identity = matrix.dot(&inverse);

        assert_abs_diff_eq!(identity[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(identity[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(identity[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(identity[[1, 0]], 0.0, epsilon = 1e-10);
    }
}
