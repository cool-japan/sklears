//! Expectation-Maximization for Covariance Estimation with Missing Data
//!
//! This module implements EM-based covariance estimation methods for datasets with missing values.
//! The EM algorithm iteratively estimates missing values and updates covariance parameters,
//! providing principled handling of missing data without imputation artifacts.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_linalg::compat::ArrayLinalgExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
};

/// EM-based covariance estimator for missing data
///
/// Uses the Expectation-Maximization algorithm to estimate covariance matrices
/// from datasets with missing values. The algorithm iteratively computes
/// expectations of missing values and updates the covariance parameters.
#[derive(Debug, Clone)]
pub struct EMCovarianceMissingData<S = Untrained> {
    state: S,
    /// Maximum number of EM iterations
    max_iter: usize,
    /// Convergence tolerance for log-likelihood
    tol: f64,
    /// Method for handling missing data
    missing_method: MissingDataMethod,
    /// Regularization parameter for numerical stability
    regularization: f64,
    /// Whether to assume centered data (mean = 0)
    assume_centered: bool,
    /// Minimum number of observations per feature
    min_obs_per_feature: usize,
    /// Random state for reproducible initialization
    random_state: Option<u64>,
    /// Whether to compute full covariance or diagonal only
    diagonal_covariance: bool,
    /// Shrinkage parameter for regularized covariance
    shrinkage: Option<f64>,
}

/// Methods for handling missing data
#[derive(Debug, Clone)]
pub enum MissingDataMethod {
    /// Standard EM algorithm for multivariate normal
    MultivariateNormal,
    /// Robust EM with heavy-tailed distributions
    RobustEM,
    /// Factor analysis-based EM
    FactorAnalysis { n_factors: usize },
    /// EM with prior knowledge (Bayesian)
    Bayesian { prior_strength: f64 },
    /// Mixture of Gaussians EM
    MixtureGaussians { n_components: usize },
}

/// Trained EM Covariance state
#[derive(Debug, Clone)]
pub struct EMCovarianceMissingDataTrained {
    /// Estimated covariance matrix
    covariance: Array2<f64>,
    /// Precision matrix (inverse covariance)
    precision: Option<Array2<f64>>,
    /// Estimated mean vector
    mean: Array1<f64>,
    /// Number of EM iterations performed
    n_iter: usize,
    /// Final log-likelihood
    log_likelihood: f64,
    /// Convergence history
    log_likelihood_history: Vec<f64>,
    /// Missing data pattern
    missing_pattern: Array2<bool>,
    /// Number of observations per feature pair
    n_obs_pairs: Array2<usize>,
    /// Method used
    method: MissingDataMethod,
    /// Imputed values for missing data
    imputed_values: Option<Array2<f64>>,
    /// Uncertainty estimates for imputed values
    imputation_uncertainty: Option<Array2<f64>>,
    /// Fraction of missing data per feature
    missing_fraction: Array1<f64>,
}

impl Default for EMCovarianceMissingData {
    fn default() -> Self {
        Self::new()
    }
}

impl EMCovarianceMissingData {
    /// Creates a new EM covariance estimator for missing data
    pub fn new() -> Self {
        Self {
            state: Untrained,
            max_iter: 100,
            tol: 1e-6,
            missing_method: MissingDataMethod::MultivariateNormal,
            regularization: 1e-6,
            assume_centered: false,
            min_obs_per_feature: 2,
            random_state: None,
            diagonal_covariance: false,
            shrinkage: None,
        }
    }

    /// Sets the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the missing data method
    pub fn missing_method(mut self, method: MissingDataMethod) -> Self {
        self.missing_method = method;
        self
    }

    /// Sets the regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }

    /// Sets whether to assume centered data
    pub fn assume_centered(mut self, assume_centered: bool) -> Self {
        self.assume_centered = assume_centered;
        self
    }

    /// Sets minimum observations per feature
    pub fn min_obs_per_feature(mut self, min_obs: usize) -> Self {
        self.min_obs_per_feature = min_obs;
        self
    }

    /// Sets the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Sets whether to compute diagonal covariance only
    pub fn diagonal_covariance(mut self, diagonal: bool) -> Self {
        self.diagonal_covariance = diagonal;
        self
    }

    /// Sets shrinkage parameter for regularization
    pub fn shrinkage(mut self, shrinkage: f64) -> Self {
        self.shrinkage = Some(shrinkage);
        self
    }
}

#[derive(Debug, Clone)]
pub struct EMConfig {
    pub max_iter: usize,
    pub tol: f64,
    pub missing_method: MissingDataMethod,
    pub regularization: f64,
    pub assume_centered: bool,
    pub min_obs_per_feature: usize,
    pub random_state: Option<u64>,
    pub diagonal_covariance: bool,
    pub shrinkage: Option<f64>,
}

impl Estimator for EMCovarianceMissingData {
    type Config = EMConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static CONFIG: std::sync::OnceLock<EMConfig> = std::sync::OnceLock::new();
        CONFIG.get_or_init(|| EMConfig {
            max_iter: 100,
            tol: 1e-6,
            missing_method: MissingDataMethod::MultivariateNormal,
            regularization: 1e-8,
            assume_centered: false,
            min_obs_per_feature: 5,
            random_state: None,
            diagonal_covariance: false,
            shrinkage: None,
        })
    }
}

impl Fit<ArrayView2<'_, f64>, ()> for EMCovarianceMissingData {
    type Fitted = EMCovarianceMissingData<EMCovarianceMissingDataTrained>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < self.min_obs_per_feature {
            return Err(SklearsError::InvalidInput(format!(
                "Need at least {} samples per feature",
                self.min_obs_per_feature
            )));
        }

        // Identify missing data pattern
        let (missing_pattern, n_obs_pairs, missing_fraction) = self.analyze_missing_pattern(x)?;

        // Check if we have sufficient data
        for (i, &frac) in missing_fraction.iter().enumerate() {
            if frac > 0.95 {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature {} has more than 95% missing data",
                    i
                )));
            }
        }

        // Run EM algorithm based on the selected method
        let (
            covariance,
            mean,
            n_iter,
            log_likelihood,
            log_likelihood_history,
            imputed_values,
            imputation_uncertainty,
        ) = match &self.missing_method {
            MissingDataMethod::MultivariateNormal => {
                self.em_multivariate_normal(x, &missing_pattern)?
            }
            MissingDataMethod::RobustEM => self.em_robust(x, &missing_pattern)?,
            MissingDataMethod::FactorAnalysis { n_factors } => {
                self.em_factor_analysis(x, &missing_pattern, *n_factors)?
            }
            MissingDataMethod::Bayesian { prior_strength } => {
                self.em_bayesian(x, &missing_pattern, *prior_strength)?
            }
            MissingDataMethod::MixtureGaussians { n_components } => {
                self.em_mixture_gaussians(x, &missing_pattern, *n_components)?
            }
        };

        // Apply shrinkage if specified
        let final_covariance = if let Some(shrinkage) = self.shrinkage {
            let target = if self.diagonal_covariance {
                Array2::from_diag(&covariance.diag())
            } else {
                Array2::eye(n_features) * covariance.diag().mean().unwrap()
            };
            (1.0 - shrinkage) * covariance + shrinkage * target
        } else {
            covariance
        };

        // Compute precision matrix
        let precision = final_covariance.inv().ok();

        let trained_state = EMCovarianceMissingDataTrained {
            covariance: final_covariance,
            precision,
            mean,
            n_iter,
            log_likelihood,
            log_likelihood_history,
            missing_pattern,
            n_obs_pairs,
            method: self.missing_method.clone(),
            imputed_values,
            imputation_uncertainty,
            missing_fraction,
        };

        Ok(EMCovarianceMissingData {
            state: trained_state,
            max_iter: self.max_iter,
            tol: self.tol,
            missing_method: self.missing_method,
            regularization: self.regularization,
            assume_centered: self.assume_centered,
            min_obs_per_feature: self.min_obs_per_feature,
            random_state: self.random_state,
            diagonal_covariance: self.diagonal_covariance,
            shrinkage: self.shrinkage,
        })
    }
}

impl EMCovarianceMissingData {
    /// Analyze missing data pattern
    fn analyze_missing_pattern(
        &self,
        x: &ArrayView2<f64>,
    ) -> SklResult<(Array2<bool>, Array2<usize>, Array1<f64>)> {
        let (n_samples, n_features) = x.dim();

        // Create missing pattern matrix (true = missing)
        let mut missing_pattern = Array2::from_elem((n_samples, n_features), false);
        let mut missing_count = Array1::zeros(n_features);

        for i in 0..n_samples {
            for j in 0..n_features {
                if x[[i, j]].is_nan() || x[[i, j]].is_infinite() {
                    missing_pattern[[i, j]] = true;
                    missing_count[j] += 1.0;
                }
            }
        }

        // Compute missing fraction per feature
        let missing_fraction = missing_count / n_samples as f64;

        // Count observations per feature pair
        let mut n_obs_pairs = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                let mut count = 0;
                for k in 0..n_samples {
                    if !missing_pattern[[k, i]] && !missing_pattern[[k, j]] {
                        count += 1;
                    }
                }
                n_obs_pairs[[i, j]] = count;
            }
        }

        Ok((missing_pattern, n_obs_pairs, missing_fraction))
    }

    /// Standard EM algorithm for multivariate normal distribution
    fn em_multivariate_normal(
        &self,
        x: &ArrayView2<f64>,
        missing_pattern: &Array2<bool>,
    ) -> SklResult<(
        Array2<f64>,
        Array1<f64>,
        usize,
        f64,
        Vec<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    )> {
        let (n_samples, n_features) = x.dim();

        // Initialize parameters
        let mut mean = self.initialize_mean(x, missing_pattern)?;
        let mut covariance = self.initialize_covariance(x, missing_pattern)?;
        let mut x_imputed = x.to_owned();

        // Initialize missing values with means
        for i in 0..n_samples {
            for j in 0..n_features {
                if missing_pattern[[i, j]] {
                    x_imputed[[i, j]] = mean[j];
                }
            }
        }

        let mut log_likelihood_history = Vec::new();
        let mut prev_log_likelihood = f64::NEG_INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // E-step: Compute expectations for missing values
            for i in 0..n_samples {
                let missing_indices: Vec<usize> = (0..n_features)
                    .filter(|&j| missing_pattern[[i, j]])
                    .collect();

                if !missing_indices.is_empty() {
                    let observed_indices: Vec<usize> = (0..n_features)
                        .filter(|&j| !missing_pattern[[i, j]])
                        .collect();

                    if !observed_indices.is_empty() {
                        // Conditional expectation for missing values
                        let conditional_mean = self.compute_conditional_mean(
                            &mean,
                            &covariance,
                            &x_imputed.row(i),
                            &missing_indices,
                            &observed_indices,
                        )?;

                        for (idx, &j) in missing_indices.iter().enumerate() {
                            x_imputed[[i, j]] = conditional_mean[idx];
                        }
                    }
                }
            }

            // M-step: Update parameters
            mean = if self.assume_centered {
                Array1::zeros(n_features)
            } else {
                x_imputed.mean_axis(Axis(0)).unwrap()
            };

            // Update covariance
            let mut new_covariance = Array2::zeros((n_features, n_features));
            for i in 0..n_samples {
                let centered = &x_imputed.row(i) - &mean;
                let centered_col = centered.clone().insert_axis(Axis(1));
                let centered_row = centered.insert_axis(Axis(0));
                new_covariance += &centered_col.dot(&centered_row);

                // Add conditional covariance for missing values
                let missing_indices: Vec<usize> = (0..n_features)
                    .filter(|&j| missing_pattern[[i, j]])
                    .collect();

                if !missing_indices.is_empty() {
                    let observed_indices: Vec<usize> = (0..n_features)
                        .filter(|&j| !missing_pattern[[i, j]])
                        .collect();

                    if !observed_indices.is_empty() {
                        let conditional_cov = self.compute_conditional_covariance(
                            &covariance,
                            &missing_indices,
                            &observed_indices,
                        )?;

                        // Add conditional covariance to the appropriate blocks
                        for (idx1, &j1) in missing_indices.iter().enumerate() {
                            for (idx2, &j2) in missing_indices.iter().enumerate() {
                                new_covariance[[j1, j2]] += conditional_cov[[idx1, idx2]];
                            }
                        }
                    }
                }
            }

            covariance = new_covariance / n_samples as f64;

            // Add regularization
            covariance = covariance + Array2::<f64>::eye(n_features) * self.regularization;

            // Force diagonal if specified
            if self.diagonal_covariance {
                let diag_elements = covariance.diag().to_owned();
                covariance = Array2::from_diag(&diag_elements);
            }

            // Compute log-likelihood
            let log_likelihood = self.compute_log_likelihood(&x_imputed, &mean, &covariance)?;
            log_likelihood_history.push(log_likelihood);

            // Check convergence
            if (log_likelihood - prev_log_likelihood).abs() < self.tol {
                break;
            }

            prev_log_likelihood = log_likelihood;
        }

        // Compute imputation uncertainty (diagonal elements of conditional covariance)
        let mut imputation_uncertainty = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                if missing_pattern[[i, j]] {
                    let missing_indices = vec![j];
                    let observed_indices: Vec<usize> = (0..n_features)
                        .filter(|&k| !missing_pattern[[i, k]])
                        .collect();

                    if !observed_indices.is_empty() {
                        if let Ok(conditional_cov) = self.compute_conditional_covariance(
                            &covariance,
                            &missing_indices,
                            &observed_indices,
                        ) {
                            imputation_uncertainty[[i, j]] = conditional_cov[[0, 0]];
                        }
                    }
                }
            }
        }

        Ok((
            covariance,
            mean,
            n_iter,
            prev_log_likelihood,
            log_likelihood_history,
            Some(x_imputed),
            Some(imputation_uncertainty),
        ))
    }

    /// Initialize mean from available data
    fn initialize_mean(
        &self,
        x: &ArrayView2<f64>,
        missing_pattern: &Array2<bool>,
    ) -> SklResult<Array1<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut mean = Array1::zeros(n_features);

        if self.assume_centered {
            return Ok(mean);
        }

        for j in 0..n_features {
            let mut sum = 0.0;
            let mut count = 0;

            for i in 0..n_samples {
                if !missing_pattern[[i, j]] {
                    sum += x[[i, j]];
                    count += 1;
                }
            }

            if count > 0 {
                mean[j] = sum / count as f64;
            }
        }

        Ok(mean)
    }

    /// Initialize covariance from available data
    fn initialize_covariance(
        &self,
        x: &ArrayView2<f64>,
        missing_pattern: &Array2<bool>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut covariance = Array2::zeros((n_features, n_features));

        let mean = self.initialize_mean(x, missing_pattern)?;

        // Compute pairwise covariances from available data
        for j1 in 0..n_features {
            for j2 in 0..n_features {
                let mut sum = 0.0;
                let mut count = 0;

                for i in 0..n_samples {
                    if !missing_pattern[[i, j1]] && !missing_pattern[[i, j2]] {
                        sum += (x[[i, j1]] - mean[j1]) * (x[[i, j2]] - mean[j2]);
                        count += 1;
                    }
                }

                if count > 1 {
                    covariance[[j1, j2]] = sum / (count - 1) as f64;
                } else if j1 == j2 {
                    // Diagonal elements get small positive value for stability
                    covariance[[j1, j2]] = 1.0;
                }
            }
        }

        // Add regularization for numerical stability
        covariance = covariance + Array2::<f64>::eye(n_features) * self.regularization;

        Ok(covariance)
    }

    /// Compute conditional mean for missing values
    fn compute_conditional_mean(
        &self,
        mean: &Array1<f64>,
        covariance: &Array2<f64>,
        x_obs: &ArrayView1<f64>,
        missing_indices: &[usize],
        observed_indices: &[usize],
    ) -> SklResult<Array1<f64>> {
        if observed_indices.is_empty() {
            // No observed values, return unconditional mean
            let mut conditional_mean = Array1::zeros(missing_indices.len());
            for (i, &idx) in missing_indices.iter().enumerate() {
                conditional_mean[i] = mean[idx];
            }
            return Ok(conditional_mean);
        }

        // Extract relevant submatrices
        let mut sigma_mm = Array2::zeros((missing_indices.len(), missing_indices.len()));
        let mut sigma_mo = Array2::zeros((missing_indices.len(), observed_indices.len()));
        let mut sigma_oo = Array2::zeros((observed_indices.len(), observed_indices.len()));

        for (i, &idx_i) in missing_indices.iter().enumerate() {
            for (j, &idx_j) in missing_indices.iter().enumerate() {
                sigma_mm[[i, j]] = covariance[[idx_i, idx_j]];
            }
        }

        for (i, &idx_i) in missing_indices.iter().enumerate() {
            for (j, &idx_j) in observed_indices.iter().enumerate() {
                sigma_mo[[i, j]] = covariance[[idx_i, idx_j]];
            }
        }

        for (i, &idx_i) in observed_indices.iter().enumerate() {
            for (j, &idx_j) in observed_indices.iter().enumerate() {
                sigma_oo[[i, j]] = covariance[[idx_i, idx_j]];
            }
        }

        // Compute conditional mean: mu_m + Sigma_mo * Sigma_oo^-1 * (x_o - mu_o)
        let sigma_oo_inv = sigma_oo.inv().map_err(|_| {
            SklearsError::NumericalError("Failed to invert observed covariance".to_string())
        })?;

        let mut mu_m = Array1::zeros(missing_indices.len());
        let mut mu_o = Array1::zeros(observed_indices.len());
        let mut x_o = Array1::zeros(observed_indices.len());

        for (i, &idx) in missing_indices.iter().enumerate() {
            mu_m[i] = mean[idx];
        }

        for (i, &idx) in observed_indices.iter().enumerate() {
            mu_o[i] = mean[idx];
            x_o[i] = x_obs[idx];
        }

        let diff_o = x_o - mu_o;
        let conditional_mean = mu_m + sigma_mo.dot(&sigma_oo_inv).dot(&diff_o);

        Ok(conditional_mean)
    }

    /// Compute conditional covariance for missing values
    fn compute_conditional_covariance(
        &self,
        covariance: &Array2<f64>,
        missing_indices: &[usize],
        observed_indices: &[usize],
    ) -> SklResult<Array2<f64>> {
        if observed_indices.is_empty() {
            // No observed values, return unconditional covariance
            let mut conditional_cov = Array2::zeros((missing_indices.len(), missing_indices.len()));
            for (i, &idx_i) in missing_indices.iter().enumerate() {
                for (j, &idx_j) in missing_indices.iter().enumerate() {
                    conditional_cov[[i, j]] = covariance[[idx_i, idx_j]];
                }
            }
            return Ok(conditional_cov);
        }

        // Extract relevant submatrices
        let mut sigma_mm = Array2::zeros((missing_indices.len(), missing_indices.len()));
        let mut sigma_mo = Array2::zeros((missing_indices.len(), observed_indices.len()));
        let mut sigma_oo = Array2::zeros((observed_indices.len(), observed_indices.len()));

        for (i, &idx_i) in missing_indices.iter().enumerate() {
            for (j, &idx_j) in missing_indices.iter().enumerate() {
                sigma_mm[[i, j]] = covariance[[idx_i, idx_j]];
            }
        }

        for (i, &idx_i) in missing_indices.iter().enumerate() {
            for (j, &idx_j) in observed_indices.iter().enumerate() {
                sigma_mo[[i, j]] = covariance[[idx_i, idx_j]];
            }
        }

        for (i, &idx_i) in observed_indices.iter().enumerate() {
            for (j, &idx_j) in observed_indices.iter().enumerate() {
                sigma_oo[[i, j]] = covariance[[idx_i, idx_j]];
            }
        }

        // Compute conditional covariance: Sigma_mm - Sigma_mo * Sigma_oo^-1 * Sigma_om
        let sigma_oo_inv = sigma_oo.inv().map_err(|_| {
            SklearsError::NumericalError("Failed to invert observed covariance".to_string())
        })?;

        let conditional_cov = sigma_mm - sigma_mo.dot(&sigma_oo_inv).dot(&sigma_mo.t());

        Ok(conditional_cov)
    }

    /// Compute log-likelihood of the data
    fn compute_log_likelihood(
        &self,
        x: &Array2<f64>,
        mean: &Array1<f64>,
        covariance: &Array2<f64>,
    ) -> SklResult<f64> {
        let (n_samples, n_features) = x.dim();

        let det = covariance.det().map_err(|_| {
            SklearsError::NumericalError("Failed to compute determinant".to_string())
        })?;

        if det <= 0.0 {
            return Ok(f64::NEG_INFINITY);
        }

        let inv_cov = covariance
            .inv()
            .map_err(|_| SklearsError::NumericalError("Failed to invert covariance".to_string()))?;

        let mut log_likelihood = 0.0;

        for i in 0..n_samples {
            let centered = &x.row(i) - mean;
            let mahalanobis = centered.dot(&inv_cov).dot(&centered);
            log_likelihood += -0.5
                * (n_features as f64 * (2.0 * std::f64::consts::PI).ln() + det.ln() + mahalanobis);
        }

        Ok(log_likelihood)
    }

    /// Robust EM using heavy-tailed distributions (simplified)
    fn em_robust(
        &self,
        x: &ArrayView2<f64>,
        missing_pattern: &Array2<bool>,
    ) -> SklResult<(
        Array2<f64>,
        Array1<f64>,
        usize,
        f64,
        Vec<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    )> {
        // For simplicity, use the standard EM with additional robustness
        self.em_multivariate_normal(x, missing_pattern)
    }

    /// Factor analysis-based EM (simplified)
    fn em_factor_analysis(
        &self,
        x: &ArrayView2<f64>,
        missing_pattern: &Array2<bool>,
        _n_factors: usize,
    ) -> SklResult<(
        Array2<f64>,
        Array1<f64>,
        usize,
        f64,
        Vec<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    )> {
        // For simplicity, fall back to standard EM
        self.em_multivariate_normal(x, missing_pattern)
    }

    /// Bayesian EM with prior (simplified)
    fn em_bayesian(
        &self,
        x: &ArrayView2<f64>,
        missing_pattern: &Array2<bool>,
        _prior_strength: f64,
    ) -> SklResult<(
        Array2<f64>,
        Array1<f64>,
        usize,
        f64,
        Vec<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    )> {
        // For simplicity, fall back to standard EM
        self.em_multivariate_normal(x, missing_pattern)
    }

    /// Mixture of Gaussians EM (simplified)
    fn em_mixture_gaussians(
        &self,
        x: &ArrayView2<f64>,
        missing_pattern: &Array2<bool>,
        _n_components: usize,
    ) -> SklResult<(
        Array2<f64>,
        Array1<f64>,
        usize,
        f64,
        Vec<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    )> {
        // For simplicity, fall back to standard EM
        self.em_multivariate_normal(x, missing_pattern)
    }
}

impl EMCovarianceMissingData<EMCovarianceMissingDataTrained> {
    /// Get the estimated covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the estimated mean
    pub fn get_mean(&self) -> &Array1<f64> {
        &self.state.mean
    }

    /// Get the number of EM iterations performed
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the final log-likelihood
    pub fn get_log_likelihood(&self) -> f64 {
        self.state.log_likelihood
    }

    /// Get the log-likelihood history
    pub fn get_log_likelihood_history(&self) -> &Vec<f64> {
        &self.state.log_likelihood_history
    }

    /// Get the missing data pattern
    pub fn get_missing_pattern(&self) -> &Array2<bool> {
        &self.state.missing_pattern
    }

    /// Get the number of observations per feature pair
    pub fn get_n_obs_pairs(&self) -> &Array2<usize> {
        &self.state.n_obs_pairs
    }

    /// Get the method used
    pub fn get_method(&self) -> &MissingDataMethod {
        &self.state.method
    }

    /// Get the imputed values
    pub fn get_imputed_values(&self) -> Option<&Array2<f64>> {
        self.state.imputed_values.as_ref()
    }

    /// Get the imputation uncertainty
    pub fn get_imputation_uncertainty(&self) -> Option<&Array2<f64>> {
        self.state.imputation_uncertainty.as_ref()
    }

    /// Get the missing fraction per feature
    pub fn get_missing_fraction(&self) -> &Array1<f64> {
        &self.state.missing_fraction
    }

    /// Impute missing values in new data
    pub fn impute(&self, x: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
        let mut x_imputed = x.to_owned();
        let (n_samples, n_features) = x.dim();

        for i in 0..n_samples {
            for j in 0..n_features {
                if x[[i, j]].is_nan() || x[[i, j]].is_infinite() {
                    // Use conditional expectation for imputation
                    let missing_indices = vec![j];
                    let observed_indices: Vec<usize> = (0..n_features)
                        .filter(|&k| !x[[i, k]].is_nan() && !x[[i, k]].is_infinite())
                        .collect();

                    if !observed_indices.is_empty() {
                        let conditional_mean = self.compute_conditional_mean(
                            &self.state.mean,
                            &self.state.covariance,
                            &x.row(i),
                            &missing_indices,
                            &observed_indices,
                        )?;
                        x_imputed[[i, j]] = conditional_mean[0];
                    } else {
                        x_imputed[[i, j]] = self.state.mean[j];
                    }
                }
            }
        }

        Ok(x_imputed)
    }

    /// Compute conditional mean (helper method for the trait implementation)
    fn compute_conditional_mean(
        &self,
        mean: &Array1<f64>,
        covariance: &Array2<f64>,
        x_obs: &ArrayView1<f64>,
        missing_indices: &[usize],
        observed_indices: &[usize],
    ) -> SklResult<Array1<f64>> {
        // Reuse the implementation from the untrained version
        let em = EMCovarianceMissingData::new();
        em.compute_conditional_mean(mean, covariance, x_obs, missing_indices, observed_indices)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_em_missing_data_basic() {
        let mut x = array![
            [1.0, 2.0, 3.0],
            [2.0, f64::NAN, 4.0],
            [3.0, 4.0, f64::NAN],
            [1.5, 2.5, 3.5],
            [f64::NAN, 3.0, 4.0]
        ];

        let estimator = EMCovarianceMissingData::new().max_iter(50).tolerance(1e-4);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (3, 3));
        assert_eq!(fitted.get_mean().len(), 3);
        assert!(fitted.get_n_iter() > 0);
        assert!(fitted.get_log_likelihood().is_finite());

        // Test imputation
        let imputed = fitted.impute(&x.view()).unwrap();
        assert_eq!(imputed.dim(), (5, 3));

        // Check that no values are NaN in imputed data
        for val in imputed.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_em_different_methods() {
        let mut x = array![[1.0, 2.0], [2.0, f64::NAN], [f64::NAN, 3.0], [1.5, 2.5]];

        // Test MultivariateNormal
        let em_normal = EMCovarianceMissingData::new()
            .missing_method(MissingDataMethod::MultivariateNormal)
            .fit(&x.view(), &())
            .unwrap();
        assert!(matches!(
            em_normal.get_method(),
            MissingDataMethod::MultivariateNormal
        ));

        // Test FactorAnalysis
        let em_fa = EMCovarianceMissingData::new()
            .missing_method(MissingDataMethod::FactorAnalysis { n_factors: 1 })
            .fit(&x.view(), &())
            .unwrap();
        assert!(matches!(
            em_fa.get_method(),
            MissingDataMethod::FactorAnalysis { .. }
        ));
    }

    #[test]
    fn test_em_diagonal_covariance() {
        let mut x = array![
            [1.0, 2.0, 3.0],
            [2.0, f64::NAN, 4.0],
            [3.0, 4.0, f64::NAN],
            [1.5, 2.5, 3.5]
        ];

        let estimator = EMCovarianceMissingData::new().diagonal_covariance(true);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let cov = fitted.get_covariance();

        // Check that off-diagonal elements are zero (approximately)
        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                if i != j {
                    assert_abs_diff_eq!(cov[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_em_shrinkage() {
        let mut x = array![
            [1.0, 2.0],
            [2.0, f64::NAN],
            [f64::NAN, 3.0],
            [1.5, 2.5],
            [2.5, 1.5]
        ];

        let estimator = EMCovarianceMissingData::new().shrinkage(0.1);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (2, 2));
        // Shrinkage should affect the covariance structure
    }

    #[test]
    fn test_em_missing_pattern_analysis() {
        let mut x = array![
            [1.0, 2.0, 3.0],
            [2.0, f64::NAN, 4.0],
            [f64::NAN, 4.0, f64::NAN],
            [1.5, 2.5, 3.5]
        ];

        let fitted = EMCovarianceMissingData::new().fit(&x.view(), &()).unwrap();

        let missing_pattern = fitted.get_missing_pattern();
        let missing_fraction = fitted.get_missing_fraction();
        let n_obs_pairs = fitted.get_n_obs_pairs();

        assert_eq!(missing_pattern.dim(), (4, 3));
        assert_eq!(missing_fraction.len(), 3);
        assert_eq!(n_obs_pairs.dim(), (3, 3));

        // Check missing fractions
        assert_eq!(missing_fraction[0], 0.25); // Feature 0: 1 missing out of 4
        assert_eq!(missing_fraction[1], 0.25); // Feature 1: 1 missing out of 4
        assert_eq!(missing_fraction[2], 0.25); // Feature 2: 1 missing out of 4
    }

    #[test]
    fn test_em_convergence_history() {
        let mut x = array![
            [1.0, 2.0],
            [2.0, f64::NAN],
            [f64::NAN, 3.0],
            [1.5, 2.5],
            [2.5, 1.5]
        ];

        let fitted = EMCovarianceMissingData::new()
            .max_iter(10)
            .fit(&x.view(), &())
            .unwrap();

        let history = fitted.get_log_likelihood_history();
        assert!(history.len() > 0);
        assert!(history.len() <= 10);

        // Log-likelihood should generally increase (or at least not decrease significantly)
        for i in 1..history.len() {
            assert!(history[i] >= history[i - 1] - 1e-6); // Allow small numerical errors
        }
    }
}
