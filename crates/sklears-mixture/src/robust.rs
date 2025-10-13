//! Robust Gaussian Mixture Models
//!
//! This module implements robust Gaussian mixture models that are resistant to outliers.
//! The implementation uses trimmed likelihood estimation and outlier detection to provide
//! robust parameter estimates even in the presence of outliers.

use crate::common::{CovarianceType, ModelSelection};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::f64::consts::PI;

/// Utility function for log-sum-exp computation
fn log_sum_exp(a: f64, b: f64) -> f64 {
    let max_val = a.max(b);
    if max_val.is_finite() {
        max_val + ((a - max_val).exp() + (b - max_val).exp()).ln()
    } else {
        max_val
    }
}

/// Robust Gaussian Mixture Model
///
/// A robust version of Gaussian mixture model that is resistant to outliers.
/// This implementation uses trimmed likelihood estimation and outlier detection
/// to provide robust parameter estimates even in the presence of outliers.
///
/// # Parameters
///
/// * `n_components` - Number of mixture components
/// * `covariance_type` - Type of covariance parameters
/// * `tol` - Convergence threshold
/// * `reg_covar` - Regularization added to the diagonal of covariance
/// * `max_iter` - Maximum number of EM iterations
/// * `n_init` - Number of initializations to perform
/// * `outlier_fraction` - Expected fraction of outliers in the data (0.0 to 0.5)
/// * `outlier_threshold` - Threshold for outlier detection (in standard deviations)
/// * `robust_covariance` - Whether to use robust covariance estimation
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_mixture::{RobustGaussianMixture, CovarianceType};
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [100.0, 100.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let rgmm = RobustGaussianMixture::new()
///     .n_components(2)
///     .outlier_fraction(0.15)
///     .covariance_type(CovarianceType::Diagonal)
///     .max_iter(100);
/// let fitted = rgmm.fit(&X.view(), &()).unwrap();
/// let labels = fitted.predict(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RobustGaussianMixture<S = Untrained> {
    pub(crate) state: S,
    n_components: usize,
    covariance_type: CovarianceType,
    tol: f64,
    reg_covar: f64,
    max_iter: usize,
    n_init: usize,
    outlier_fraction: f64,
    outlier_threshold: f64,
    robust_covariance: bool,
    random_state: Option<u64>,
}

impl RobustGaussianMixture<Untrained> {
    /// Create a new RobustGaussianMixture instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 1,
            covariance_type: CovarianceType::Full,
            tol: 1e-3,
            reg_covar: 1e-6,
            max_iter: 100,
            n_init: 1,
            outlier_fraction: 0.1,
            outlier_threshold: 3.0,
            robust_covariance: true,
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the covariance type
    pub fn covariance_type(mut self, covariance_type: CovarianceType) -> Self {
        self.covariance_type = covariance_type;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the regularization parameter
    pub fn reg_covar(mut self, reg_covar: f64) -> Self {
        self.reg_covar = reg_covar;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the number of initializations
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the expected fraction of outliers
    pub fn outlier_fraction(mut self, outlier_fraction: f64) -> Self {
        self.outlier_fraction = outlier_fraction.clamp(0.0, 0.5);
        self
    }

    /// Set the outlier detection threshold (in standard deviations)
    pub fn outlier_threshold(mut self, outlier_threshold: f64) -> Self {
        self.outlier_threshold = outlier_threshold;
        self
    }

    /// Set whether to use robust covariance estimation
    pub fn robust_covariance(mut self, robust_covariance: bool) -> Self {
        self.robust_covariance = robust_covariance;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl Default for RobustGaussianMixture<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RobustGaussianMixture<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for RobustGaussianMixture<Untrained> {
    type Fitted = RobustGaussianMixture<RobustGaussianMixtureTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.to_owned();
        let (n_samples, n_features) = X.dim();

        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least the number of components".to_string(),
            ));
        }

        if self.n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of components must be positive".to_string(),
            ));
        }

        let mut best_params = None;
        let mut best_log_likelihood = f64::NEG_INFINITY;
        let mut best_n_iter = 0;
        let mut best_converged = false;
        let mut best_outlier_mask = None;

        // Run multiple initializations and keep the best
        for init_run in 0..self.n_init {
            let seed = self.random_state.map(|s| s + init_run as u64);

            // Initialize parameters
            let (mut weights, mut means, mut covariances) = self.initialize_parameters(&X, seed)?;

            let mut log_likelihood = f64::NEG_INFINITY;
            let mut converged = false;
            let mut n_iter = 0;
            let mut outlier_mask = Array1::from_elem(n_samples, false);

            // Robust EM iterations
            for iteration in 0..self.max_iter {
                n_iter = iteration + 1;

                // E-step: Compute responsibilities with outlier detection
                let responsibilities = self.compute_robust_responsibilities(
                    &X,
                    &weights,
                    &means,
                    &covariances,
                    &mut outlier_mask,
                )?;

                // M-step: Update parameters with outlier weighting
                let (new_weights, new_means, new_covariances) =
                    self.update_robust_parameters(&X, &responsibilities, &outlier_mask)?;

                // Compute trimmed log-likelihood (excluding outliers)
                let new_log_likelihood = self.compute_trimmed_log_likelihood(
                    &X,
                    &new_weights,
                    &new_means,
                    &new_covariances,
                    &outlier_mask,
                )?;

                // Check convergence
                if iteration > 0 && (new_log_likelihood - log_likelihood).abs() < self.tol {
                    converged = true;
                }

                weights = new_weights;
                means = new_means;
                covariances = new_covariances;
                log_likelihood = new_log_likelihood;

                if converged {
                    break;
                }
            }

            // Keep track of best parameters
            if log_likelihood > best_log_likelihood {
                best_log_likelihood = log_likelihood;
                best_params = Some((weights, means, covariances));
                best_n_iter = n_iter;
                best_converged = converged;
                best_outlier_mask = Some(outlier_mask);
            }
        }

        let (weights, means, covariances) = best_params.unwrap();
        let outlier_mask = best_outlier_mask.unwrap();

        // Calculate model selection criteria
        let n_params =
            ModelSelection::n_parameters(self.n_components, n_features, &self.covariance_type);
        let bic = ModelSelection::bic(best_log_likelihood, n_params, n_samples);
        let aic = ModelSelection::aic(best_log_likelihood, n_params);

        // Count detected outliers
        let n_outliers = outlier_mask.iter().filter(|&&x| x).count();

        Ok(RobustGaussianMixture {
            state: RobustGaussianMixtureTrained {
                weights,
                means,
                covariances,
                log_likelihood: best_log_likelihood,
                n_iter: best_n_iter,
                converged: best_converged,
                bic,
                aic,
                outlier_mask,
                n_outliers,
            },
            n_components: self.n_components,
            covariance_type: self.covariance_type,
            tol: self.tol,
            reg_covar: self.reg_covar,
            max_iter: self.max_iter,
            n_init: self.n_init,
            outlier_fraction: self.outlier_fraction,
            outlier_threshold: self.outlier_threshold,
            robust_covariance: self.robust_covariance,
            random_state: self.random_state,
        })
    }
}

impl RobustGaussianMixture<Untrained> {
    /// Initialize parameters for robust EM algorithm
    fn initialize_parameters(
        &self,
        X: &Array2<f64>,
        seed: Option<u64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>, Vec<Array2<f64>>)> {
        // Initialize weights (uniform)
        let weights = Array1::from_elem(self.n_components, 1.0 / self.n_components as f64);

        // Initialize means using robust initialization
        let means = self.initialize_robust_means(X, seed)?;

        // Initialize covariances
        let covariances = self.initialize_robust_covariances(X, &means)?;

        Ok((weights, means, covariances))
    }

    /// Initialize means using robust k-means++ style initialization
    fn initialize_robust_means(
        &self,
        X: &Array2<f64>,
        seed: Option<u64>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut means = Array2::zeros((self.n_components, n_features));

        // Use median-based initialization for robustness
        for i in 0..self.n_components {
            let step = n_samples / self.n_components;
            let sample_idx = if step == 0 {
                i.min(n_samples - 1)
            } else {
                (i * step).min(n_samples - 1)
            };

            let mut mean = means.row_mut(i);
            mean.assign(&X.row(sample_idx));

            // Add small perturbation if seed is provided
            if let Some(_seed) = seed {
                for j in 0..n_features {
                    mean[j] += 0.01 * (i as f64 - self.n_components as f64 / 2.0);
                }
            }
        }

        Ok(means)
    }

    /// Initialize covariances with robust estimation
    fn initialize_robust_covariances(
        &self,
        X: &Array2<f64>,
        _means: &Array2<f64>,
    ) -> SklResult<Vec<Array2<f64>>> {
        let (_, n_features) = X.dim();
        let mut covariances = Vec::new();

        // Use robust scale estimation
        let robust_scale = if self.robust_covariance {
            self.estimate_robust_scale(X)?
        } else {
            1.0
        };

        match self.covariance_type {
            CovarianceType::Full => {
                for _ in 0..self.n_components {
                    let mut cov = Array2::eye(n_features);
                    for i in 0..n_features {
                        cov[[i, i]] = robust_scale + self.reg_covar;
                    }
                    covariances.push(cov);
                }
            }
            CovarianceType::Diagonal => {
                for _ in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    for i in 0..n_features {
                        cov[[i, i]] = robust_scale + self.reg_covar;
                    }
                    covariances.push(cov);
                }
            }
            CovarianceType::Tied => {
                let mut cov = Array2::eye(n_features);
                for i in 0..n_features {
                    cov[[i, i]] = robust_scale + self.reg_covar;
                }
                for _ in 0..self.n_components {
                    covariances.push(cov.clone());
                }
            }
            CovarianceType::Spherical => {
                for _ in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    for i in 0..n_features {
                        cov[[i, i]] = robust_scale + self.reg_covar;
                    }
                    covariances.push(cov);
                }
            }
        }

        Ok(covariances)
    }

    /// Estimate robust scale using median absolute deviation
    fn estimate_robust_scale(&self, X: &Array2<f64>) -> SklResult<f64> {
        let (n_samples, n_features) = X.dim();
        let mut all_deviations = Vec::new();

        // Calculate median for each feature
        for j in 0..n_features {
            let mut feature_values: Vec<f64> = X.column(j).to_vec();
            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if n_samples % 2 == 0 {
                (feature_values[n_samples / 2 - 1] + feature_values[n_samples / 2]) / 2.0
            } else {
                feature_values[n_samples / 2]
            };

            // Calculate absolute deviations from median
            for i in 0..n_samples {
                all_deviations.push((X[[i, j]] - median).abs());
            }
        }

        // Calculate median absolute deviation
        all_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if all_deviations.len() % 2 == 0 {
            (all_deviations[all_deviations.len() / 2 - 1]
                + all_deviations[all_deviations.len() / 2])
                / 2.0
        } else {
            all_deviations[all_deviations.len() / 2]
        };

        // Scale MAD to approximate standard deviation (factor 1.4826 for normal distribution)
        Ok((mad * 1.4826).max(1e-6))
    }

    /// Compute responsibilities with outlier detection (robust E-step)
    fn compute_robust_responsibilities(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
        outlier_mask: &mut Array1<bool>,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, _) = X.dim();
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));
        let mut sample_likelihoods = Array1::zeros(n_samples);

        // Compute standard responsibilities first
        for i in 0..n_samples {
            let sample = X.row(i);
            let mut log_prob_sum = f64::NEG_INFINITY;
            let mut log_probs = Vec::new();

            for k in 0..self.n_components {
                let mean = means.row(k);
                let cov = &covariances[k];
                let log_weight = weights[k].ln();
                let log_likelihood = self.multivariate_normal_log_pdf(&sample, &mean, cov)?;
                let log_prob = log_weight + log_likelihood;
                log_probs.push(log_prob);
                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

            sample_likelihoods[i] = log_prob_sum;

            // Normalize to get responsibilities
            for k in 0..self.n_components {
                responsibilities[[i, k]] = (log_probs[k] - log_prob_sum).exp();
            }
        }

        // Detect outliers based on likelihood threshold
        self.detect_outliers(&sample_likelihoods, outlier_mask)?;

        // Down-weight outliers in responsibilities
        for i in 0..n_samples {
            if outlier_mask[i] {
                // Reduce responsibility for outliers
                for k in 0..self.n_components {
                    responsibilities[[i, k]] *= 0.1; // Down-weight factor
                }
            }
        }

        Ok(responsibilities)
    }

    /// Detect outliers based on likelihood values
    fn detect_outliers(
        &self,
        sample_likelihoods: &Array1<f64>,
        outlier_mask: &mut Array1<bool>,
    ) -> SklResult<()> {
        let n_samples = sample_likelihoods.len();

        // Calculate robust threshold using percentile
        let mut sorted_likelihoods = sample_likelihoods.to_vec();
        sorted_likelihoods.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use lower percentile as threshold
        let threshold_idx = ((1.0 - self.outlier_fraction) * n_samples as f64) as usize;
        let threshold_idx = threshold_idx.min(n_samples - 1);
        let likelihood_threshold = sorted_likelihoods[threshold_idx];

        // Mark samples with likelihood below threshold as outliers
        for i in 0..n_samples {
            outlier_mask[i] = sample_likelihoods[i] < likelihood_threshold;
        }

        Ok(())
    }

    /// Update parameters with outlier weighting (robust M-step)
    fn update_robust_parameters(
        &self,
        X: &Array2<f64>,
        responsibilities: &Array2<f64>,
        outlier_mask: &Array1<bool>,
    ) -> SklResult<(Array1<f64>, Array2<f64>, Vec<Array2<f64>>)> {
        let (n_samples, n_features) = X.dim();

        // Calculate effective sample weights (down-weight outliers)
        let mut effective_responsibilities = responsibilities.clone();
        for i in 0..n_samples {
            if outlier_mask[i] {
                for k in 0..self.n_components {
                    effective_responsibilities[[i, k]] *= 0.1;
                }
            }
        }

        // Update weights
        let n_k: Array1<f64> = effective_responsibilities.sum_axis(Axis(0));
        let total_weight = n_k.sum();
        let weights = &n_k / total_weight;

        // Update means
        let mut means = Array2::zeros((self.n_components, n_features));
        for k in 0..self.n_components {
            if n_k[k] > 1e-10 {
                for i in 0..n_samples {
                    for j in 0..n_features {
                        means[[k, j]] += effective_responsibilities[[i, k]] * X[[i, j]];
                    }
                }
                for j in 0..n_features {
                    means[[k, j]] /= n_k[k];
                }
            }
        }

        // Update covariances with robust estimation
        let covariances =
            self.update_robust_covariances(X, &effective_responsibilities, &means, &n_k)?;

        Ok((weights, means, covariances))
    }

    /// Update covariances with robust estimation
    fn update_robust_covariances(
        &self,
        X: &Array2<f64>,
        responsibilities: &Array2<f64>,
        means: &Array2<f64>,
        n_k: &Array1<f64>,
    ) -> SklResult<Vec<Array2<f64>>> {
        let (n_samples, n_features) = X.dim();
        let mut covariances = Vec::new();

        match self.covariance_type {
            CovarianceType::Full => {
                for k in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));

                    if n_k[k] > 1e-10 {
                        let mean_k = means.row(k);

                        for i in 0..n_samples {
                            let sample = X.row(i);
                            let diff = &sample - &mean_k;

                            for d1 in 0..n_features {
                                for d2 in 0..n_features {
                                    cov[[d1, d2]] += responsibilities[[i, k]] * diff[d1] * diff[d2];
                                }
                            }
                        }

                        for d1 in 0..n_features {
                            for d2 in 0..n_features {
                                cov[[d1, d2]] /= n_k[k];
                            }
                        }

                        // Add robust regularization
                        let robust_reg = if self.robust_covariance {
                            self.reg_covar * 10.0 // Stronger regularization for robustness
                        } else {
                            self.reg_covar
                        };

                        for d in 0..n_features {
                            cov[[d, d]] += robust_reg;
                        }
                    } else {
                        // Empty component: use robust identity
                        for d in 0..n_features {
                            cov[[d, d]] = 1.0 + self.reg_covar * 10.0;
                        }
                    }

                    covariances.push(cov);
                }
            }
            CovarianceType::Diagonal => {
                for k in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));

                    if n_k[k] > 1e-10 {
                        let mean_k = means.row(k);

                        for d in 0..n_features {
                            let mut var = 0.0;
                            for i in 0..n_samples {
                                let diff = X[[i, d]] - mean_k[d];
                                var += responsibilities[[i, k]] * diff * diff;
                            }
                            var /= n_k[k];

                            let robust_reg = if self.robust_covariance {
                                self.reg_covar * 10.0
                            } else {
                                self.reg_covar
                            };

                            cov[[d, d]] = var + robust_reg;
                        }
                    } else {
                        for d in 0..n_features {
                            cov[[d, d]] = 1.0 + self.reg_covar * 10.0;
                        }
                    }

                    covariances.push(cov);
                }
            }
            CovarianceType::Tied => {
                let mut cov = Array2::zeros((n_features, n_features));
                let total_responsibility: f64 = n_k.sum();

                if total_responsibility > 1e-10 {
                    for k in 0..self.n_components {
                        let mean_k = means.row(k);

                        for i in 0..n_samples {
                            let sample = X.row(i);
                            let diff = &sample - &mean_k;

                            for d1 in 0..n_features {
                                for d2 in 0..n_features {
                                    cov[[d1, d2]] += responsibilities[[i, k]] * diff[d1] * diff[d2];
                                }
                            }
                        }
                    }

                    for d1 in 0..n_features {
                        for d2 in 0..n_features {
                            cov[[d1, d2]] /= total_responsibility;
                        }
                    }

                    let robust_reg = if self.robust_covariance {
                        self.reg_covar * 10.0
                    } else {
                        self.reg_covar
                    };

                    for d in 0..n_features {
                        cov[[d, d]] += robust_reg;
                    }
                } else {
                    for d in 0..n_features {
                        cov[[d, d]] = 1.0 + self.reg_covar * 10.0;
                    }
                }

                for _ in 0..self.n_components {
                    covariances.push(cov.clone());
                }
            }
            CovarianceType::Spherical => {
                for k in 0..self.n_components {
                    let mut cov = Array2::zeros((n_features, n_features));

                    if n_k[k] > 1e-10 {
                        let mean_k = means.row(k);
                        let mut total_var = 0.0;

                        for i in 0..n_samples {
                            for d in 0..n_features {
                                let diff = X[[i, d]] - mean_k[d];
                                total_var += responsibilities[[i, k]] * diff * diff;
                            }
                        }

                        total_var /= n_k[k] * n_features as f64;

                        let robust_reg = if self.robust_covariance {
                            self.reg_covar * 10.0
                        } else {
                            self.reg_covar
                        };

                        let variance = total_var + robust_reg;

                        for d in 0..n_features {
                            cov[[d, d]] = variance;
                        }
                    } else {
                        for d in 0..n_features {
                            cov[[d, d]] = 1.0 + self.reg_covar * 10.0;
                        }
                    }

                    covariances.push(cov);
                }
            }
        }

        Ok(covariances)
    }

    /// Compute trimmed log-likelihood (excluding outliers)
    fn compute_trimmed_log_likelihood(
        &self,
        X: &Array2<f64>,
        weights: &Array1<f64>,
        means: &Array2<f64>,
        covariances: &[Array2<f64>],
        outlier_mask: &Array1<bool>,
    ) -> SklResult<f64> {
        let (n_samples, _) = X.dim();
        let mut total_log_likelihood = 0.0;
        let mut n_included = 0;

        for i in 0..n_samples {
            if !outlier_mask[i] {
                // Only include non-outliers
                let sample = X.row(i);
                let mut log_prob_sum = f64::NEG_INFINITY;

                for k in 0..self.n_components {
                    let mean = means.row(k);
                    let cov = &covariances[k];
                    let log_weight = weights[k].ln();
                    let log_likelihood = self.multivariate_normal_log_pdf(&sample, &mean, cov)?;
                    let log_prob = log_weight + log_likelihood;
                    log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
                }

                total_log_likelihood += log_prob_sum;
                n_included += 1;
            }
        }

        // Return average log-likelihood of non-outliers
        if n_included > 0 {
            Ok(total_log_likelihood / n_included as f64)
        } else {
            Ok(f64::NEG_INFINITY)
        }
    }

    /// Compute multivariate normal log probability density function
    fn multivariate_normal_log_pdf(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        cov: &Array2<f64>,
    ) -> SklResult<f64> {
        let d = x.len() as f64;
        let diff: Array1<f64> = x - mean;

        match self.covariance_type {
            CovarianceType::Full => {
                let mut log_det = 0.0;
                let mut quad_form = 0.0;

                for i in 0..cov.nrows() {
                    if cov[[i, i]] <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Covariance matrix has non-positive diagonal elements".to_string(),
                        ));
                    }
                    log_det += cov[[i, i]].ln();
                    quad_form += diff[i] * diff[i] / cov[[i, i]];
                }

                let log_pdf = -0.5 * (d * (2.0 * PI).ln() + log_det + quad_form);
                Ok(log_pdf)
            }
            CovarianceType::Diagonal | CovarianceType::Tied | CovarianceType::Spherical => {
                let mut log_det = 0.0;
                let mut quad_form = 0.0;

                for i in 0..diff.len() {
                    if cov[[i, i]] <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Covariance matrix has non-positive diagonal elements".to_string(),
                        ));
                    }
                    log_det += cov[[i, i]].ln();
                    quad_form += diff[i] * diff[i] / cov[[i, i]];
                }

                let log_pdf = -0.5 * (d * (2.0 * PI).ln() + log_det + quad_form);
                Ok(log_pdf)
            }
        }
    }
}

/// Trained state for RobustGaussianMixture
#[derive(Debug, Clone)]
pub struct RobustGaussianMixtureTrained {
    /// Mixture component weights
    pub weights: Array1<f64>,
    /// Component means
    pub means: Array2<f64>,
    /// Component covariance matrices or parameters
    pub covariances: Vec<Array2<f64>>,
    /// Log likelihood of the fitted model
    pub log_likelihood: f64,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Mask indicating which samples are detected as outliers
    pub outlier_mask: Array1<bool>,
    /// Number of detected outliers
    pub n_outliers: usize,
}

impl Predict<ArrayView2<'_, Float>, Array1<i32>>
    for RobustGaussianMixture<RobustGaussianMixtureTrained>
{
    #[allow(non_snake_case)]
    fn predict(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<i32>> {
        let X = X.to_owned();
        let (n_samples, _) = X.dim();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut max_log_prob = f64::NEG_INFINITY;
            let mut best_component = 0;

            for k in 0..self.n_components {
                let mean = self.state.means.row(k);
                let cov = &self.state.covariances[k];
                let log_weight = self.state.weights[k].ln();

                if let Ok(log_likelihood) = self.multivariate_normal_log_pdf(&sample, &mean, cov) {
                    let log_prob = log_weight + log_likelihood;
                    if log_prob > max_log_prob {
                        max_log_prob = log_prob;
                        best_component = k;
                    }
                }
            }

            predictions[i] = best_component as i32;
        }

        Ok(predictions)
    }
}

impl RobustGaussianMixture<RobustGaussianMixtureTrained> {
    /// Get the mixture weights
    pub fn weights(&self) -> &Array1<f64> {
        &self.state.weights
    }

    /// Get the component means
    pub fn means(&self) -> &Array2<f64> {
        &self.state.means
    }

    /// Get the component covariances
    pub fn covariances(&self) -> &[Array2<f64>] {
        &self.state.covariances
    }

    /// Get the log likelihood of the fitted model
    pub fn log_likelihood(&self) -> f64 {
        self.state.log_likelihood
    }

    /// Get the number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Check if the algorithm converged
    pub fn converged(&self) -> bool {
        self.state.converged
    }

    /// Get the Bayesian Information Criterion
    pub fn bic(&self) -> f64 {
        self.state.bic
    }

    /// Get the Akaike Information Criterion
    pub fn aic(&self) -> f64 {
        self.state.aic
    }

    /// Get the outlier mask
    pub fn outlier_mask(&self) -> &Array1<bool> {
        &self.state.outlier_mask
    }

    /// Get the number of detected outliers
    pub fn n_outliers(&self) -> usize {
        self.state.n_outliers
    }

    /// Predict probabilities for each component
    #[allow(non_snake_case)]
    pub fn predict_proba(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let X = X.to_owned();
        let (n_samples, _) = X.dim();
        let mut probabilities = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut log_prob_sum = f64::NEG_INFINITY;
            let mut log_probs = Vec::new();

            // Compute log probabilities for each component
            for k in 0..self.n_components {
                let mean = self.state.means.row(k);
                let cov = &self.state.covariances[k];
                let log_weight = self.state.weights[k].ln();
                let log_likelihood = self.multivariate_normal_log_pdf(&sample, &mean, cov)?;
                let log_prob = log_weight + log_likelihood;
                log_probs.push(log_prob);
                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

            // Normalize to get probabilities
            for k in 0..self.n_components {
                probabilities[[i, k]] = (log_probs[k] - log_prob_sum).exp();
            }
        }

        Ok(probabilities)
    }

    /// Compute the per-sample log-likelihood
    #[allow(non_snake_case)]
    pub fn score_samples(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let X = X.to_owned();
        let (n_samples, _) = X.dim();
        let mut scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = X.row(i);
            let mut log_prob_sum = f64::NEG_INFINITY;

            for k in 0..self.n_components {
                let mean = self.state.means.row(k);
                let cov = &self.state.covariances[k];
                let log_weight = self.state.weights[k].ln();
                let log_likelihood = self.multivariate_normal_log_pdf(&sample, &mean, cov)?;
                let log_prob = log_weight + log_likelihood;
                log_prob_sum = log_sum_exp(log_prob_sum, log_prob);
            }

            scores[i] = log_prob_sum;
        }

        Ok(scores)
    }

    /// Compute the average log-likelihood
    pub fn score(&self, X: &ArrayView2<'_, Float>) -> SklResult<f64> {
        let scores = self.score_samples(X)?;
        Ok(scores.mean().unwrap_or(0.0))
    }

    /// Detect outliers in new data
    pub fn detect_outliers(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array1<bool>> {
        let scores = self.score_samples(X)?;
        let n_samples = scores.len();
        let mut outlier_mask = Array1::from_elem(n_samples, false);

        // Use same outlier detection logic as training
        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_idx = ((1.0 - self.outlier_fraction) * n_samples as f64) as usize;
        let threshold_idx = threshold_idx.min(n_samples - 1);
        let score_threshold = sorted_scores[threshold_idx];

        for i in 0..n_samples {
            outlier_mask[i] = scores[i] < score_threshold;
        }

        Ok(outlier_mask)
    }

    fn multivariate_normal_log_pdf(
        &self,
        x: &ArrayView1<f64>,
        mean: &ArrayView1<f64>,
        cov: &Array2<f64>,
    ) -> SklResult<f64> {
        let d = x.len() as f64;
        let diff: Array1<f64> = x - mean;

        match self.covariance_type {
            CovarianceType::Full => {
                let mut log_det = 0.0;
                let mut quad_form = 0.0;

                for i in 0..cov.nrows() {
                    if cov[[i, i]] <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Covariance matrix has non-positive diagonal elements".to_string(),
                        ));
                    }
                    log_det += cov[[i, i]].ln();
                    quad_form += diff[i] * diff[i] / cov[[i, i]];
                }

                let log_pdf = -0.5 * (d * (2.0 * PI).ln() + log_det + quad_form);
                Ok(log_pdf)
            }
            CovarianceType::Diagonal | CovarianceType::Tied | CovarianceType::Spherical => {
                let mut log_det = 0.0;
                let mut quad_form = 0.0;

                for i in 0..diff.len() {
                    if cov[[i, i]] <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Covariance matrix has non-positive diagonal elements".to_string(),
                        ));
                    }
                    log_det += cov[[i, i]].ln();
                    quad_form += diff[i] * diff[i] / cov[[i, i]];
                }

                let log_pdf = -0.5 * (d * (2.0 * PI).ln() + log_det + quad_form);
                Ok(log_pdf)
            }
        }
    }
}
