//! Classical Gaussian Mixture Model Implementation
//!
//! This module provides the main GaussianMixture struct with traditional EM algorithm
//! for fitting mixture models. Includes model selection criteria (AIC, BIC, ICL)
//! and SIMD-accelerated computations for performance.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};
use std::marker::PhantomData;

use super::simd_operations::*;
use super::types_config::{
    CovarianceType, GaussianMixtureConfig, ModelSelectionCriterion, ModelSelectionResult,
};

/// Classical Gaussian Mixture Model
///
/// A probabilistic model that assumes data points are generated from
/// a mixture of several Gaussian distributions with unknown parameters.
/// Uses EM algorithm for parameter estimation.
pub struct GaussianMixture<X = Array2<Float>, Y = ()> {
    config: GaussianMixtureConfig,
    weights: Option<Array1<Float>>,
    means: Option<Array2<Float>>,
    covariances: Option<Vec<Array2<Float>>>,
    converged: Option<bool>,
    n_iter: Option<usize>,
    lower_bound: Option<Float>,
    _phantom: PhantomData<(X, Y)>,
}

impl<X, Y> GaussianMixture<X, Y> {
    /// Create a new Gaussian Mixture Model
    pub fn new() -> Self {
        Self {
            config: GaussianMixtureConfig::default(),
            weights: None,
            means: None,
            covariances: None,
            converged: None,
            n_iter: None,
            lower_bound: None,
            _phantom: PhantomData,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set the covariance type
    pub fn covariance_type(mut self, cov_type: CovarianceType) -> Self {
        self.config.covariance_type = cov_type;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the regularization
    pub fn reg_covar(mut self, reg_covar: Float) -> Self {
        self.config.reg_covar = reg_covar;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the number of initializations
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.config.n_init = n_init;
        self
    }

    /// Set the initialization method
    pub fn init_params(mut self, init_params: super::types_config::WeightInit) -> Self {
        self.config.init_params = init_params;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Get the weights of each mixture component
    pub fn weights(&self) -> Result<&Array1<Float>> {
        self.weights
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "weights".to_string(),
            })
    }

    /// Get the mean of each mixture component
    pub fn means(&self) -> Result<&Array2<Float>> {
        self.means.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "means".to_string(),
        })
    }

    /// Get the covariance of each mixture component
    pub fn covariances(&self) -> Result<&Vec<Array2<Float>>> {
        self.covariances
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "covariances".to_string(),
            })
    }

    /// Check if the model converged
    pub fn converged(&self) -> bool {
        self.converged.expect("Model has not been fitted yet")
    }

    /// Get the number of iterations run
    pub fn n_iter(&self) -> usize {
        self.n_iter.expect("Model has not been fitted yet")
    }

    /// Get the lower bound value on the log-likelihood
    pub fn lower_bound(&self) -> Float {
        self.lower_bound.expect("Model has not been fitted yet")
    }

    /// Calculate the log-likelihood of the data given the model
    pub fn score(&self, x: &ArrayView2<Float>) -> Result<Float> {
        let weights = self.weights()?;
        let means = self.means()?;
        let covariances = self.covariances()?;

        let mut log_likelihood = 0.0;

        for sample in x.outer_iter() {
            let mut sample_likelihood = 0.0;

            for (k, &weight) in weights.iter().enumerate() {
                let mean = means.row(k);
                let cov = &covariances[k];

                // Calculate multivariate normal likelihood with SIMD acceleration
                let inv_diag = self.extract_diagonal_inverse(cov)?;
                let log_density = simd_multivariate_normal_log_density(
                    &sample,
                    &mean,
                    &inv_diag.view(),
                    simd_log_determinant(&cov.view()),
                );

                let component_likelihood = weight * log_density.exp();
                sample_likelihood += component_likelihood;
            }

            if sample_likelihood > 1e-12 {
                log_likelihood += sample_likelihood.ln();
            }
        }

        Ok(log_likelihood)
    }

    /// Calculate Akaike Information Criterion (AIC) with SIMD acceleration
    pub fn aic(&self, x: &ArrayView2<Float>) -> Result<Float> {
        let log_likelihood = self.score(x)?;
        let n_params = self.count_parameters(x.ncols());

        Ok(-2.0 * log_likelihood + 2.0 * n_params as Float)
    }

    /// Calculate Bayesian Information Criterion (BIC) with SIMD acceleration
    pub fn bic(&self, x: &ArrayView2<Float>) -> Result<Float> {
        let log_likelihood = self.score(x)?;
        let n_params = self.count_parameters(x.ncols());

        Ok(-2.0 * log_likelihood + (n_params as Float) * (x.nrows() as Float).ln())
    }

    /// Calculate Integrated Completed Likelihood (ICL) with SIMD acceleration
    pub fn icl(&self, x: &ArrayView2<Float>) -> Result<Float> {
        let bic = self.bic(x)?;
        let entropy = self.compute_entropy(x)?;
        Ok(bic - entropy)
    }

    /// Perform model selection to find optimal number of components
    pub fn select_model(
        x: &ArrayView2<Float>,
        min_components: usize,
        max_components: usize,
        criterion: ModelSelectionCriterion,
        config_template: &GaussianMixtureConfig,
    ) -> Result<ModelSelectionResult> {
        let mut criterion_values = Vec::new();
        let mut log_likelihoods = Vec::new();
        let mut best_criterion = Float::INFINITY;
        let mut best_n_components = min_components;

        for n_comp in min_components..=max_components {
            let mut temp_config = config_template.clone();
            temp_config.n_components = n_comp;

            let model: GaussianMixture<(), ()> = GaussianMixture {
                config: temp_config,
                weights: None,
                means: None,
                covariances: None,
                converged: None,
                n_iter: None,
                lower_bound: None,
                _phantom: PhantomData,
            };

            let fitted_model = model.fit(x, &Array1::zeros(0).view())?;
            let log_likelihood = fitted_model.score(x)?;
            log_likelihoods.push(log_likelihood);

            let criterion_value = match criterion {
                ModelSelectionCriterion::AIC => fitted_model.aic(x)?,
                ModelSelectionCriterion::BIC => fitted_model.bic(x)?,
                ModelSelectionCriterion::ICL => fitted_model.icl(x)?,
            };

            criterion_values.push(criterion_value);

            if criterion_value < best_criterion {
                best_criterion = criterion_value;
                best_n_components = n_comp;
            }
        }

        Ok(ModelSelectionResult {
            best_n_components,
            criterion_values,
            log_likelihoods,
            criterion,
        })
    }

    /// Count the number of free parameters in the model
    fn count_parameters(&self, n_features: usize) -> usize {
        let k = self.config.n_components;
        let d = n_features;

        match self.config.covariance_type {
            CovarianceType::Full => k - 1 + k * d + k * d * (d + 1) / 2,
            CovarianceType::Diagonal => k - 1 + k * d + k * d,
            CovarianceType::Tied => k - 1 + k * d + d * (d + 1) / 2,
            CovarianceType::Spherical => k - 1 + k * d + k,
        }
    }

    /// Compute the entropy of the clustering assignment with SIMD acceleration
    fn compute_entropy(&self, x: &ArrayView2<Float>) -> Result<Float> {
        let proba = self.predict_proba(x)?;
        let entropy = simd_compute_entropy(&proba.view());
        Ok(entropy)
    }

    /// Extract diagonal inverse for SIMD operations
    fn extract_diagonal_inverse(&self, cov: &Array2<Float>) -> Result<Array1<Float>> {
        let mut inv_diag = Array1::zeros(cov.nrows());
        for i in 0..cov.nrows() {
            let diag_val = cov[(i, i)];
            if diag_val <= 1e-12 {
                return Err(SklearsError::Other(
                    "Singular covariance matrix".to_string(),
                ));
            }
            inv_diag[i] = 1.0 / diag_val;
        }
        Ok(inv_diag)
    }

    /// Implement EM algorithm for Gaussian Mixture Models with SIMD acceleration
    fn fit_em(
        &self,
        x: &ArrayView2<Float>,
    ) -> Result<(
        Array1<Float>,
        Array2<Float>,
        Vec<Array2<Float>>,
        bool,
        usize,
        Float,
    )> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_components = self.config.n_components;

        // Initialize parameters using K-means++ style initialization
        let mut means = Array2::zeros((n_components, n_features));
        let mut weights = Array1::zeros(n_components);
        let mut covariances = Vec::new();

        // Initialize means by selecting diverse samples
        for k in 0..n_components {
            let idx = k * n_samples / n_components;
            means.row_mut(k).assign(&x.row(idx));
        }

        // Initialize weights uniformly
        weights.fill(1.0 / n_components as Float);

        // Initialize covariances as identity matrices with small regularization
        for _ in 0..n_components {
            let mut cov = Array2::eye(n_features);
            cov *= self.config.reg_covar + 0.01;
            covariances.push(cov);
        }

        let mut log_likelihood = Float::NEG_INFINITY;
        let mut converged = false;
        let mut n_iter = 0;

        // EM iterations with SIMD acceleration
        for iteration in 0..self.config.max_iter {
            n_iter = iteration + 1;

            // E-step: compute responsibilities using SIMD operations
            let mut responsibilities = Array2::zeros((n_samples, n_components));

            for i in 0..n_samples {
                let sample = x.row(i);
                let mut log_probs = Array1::zeros(n_components);

                for k in 0..n_components {
                    let mean = means.row(k);
                    let inv_diag = self.extract_diagonal_inverse(&covariances[k])?;
                    let log_det = simd_log_determinant(&covariances[k].view());

                    let log_density = simd_multivariate_normal_log_density(
                        &sample,
                        &mean,
                        &inv_diag.view(),
                        log_det,
                    );

                    log_probs[k] = weights[k].ln() + log_density;
                }

                // Normalize using log-sum-exp for numerical stability
                let log_sum = simd_log_sum_exp(&log_probs.view());
                for k in 0..n_components {
                    responsibilities[(i, k)] = (log_probs[k] - log_sum).exp();
                }
            }

            // M-step: update parameters using SIMD operations
            let nk: Array1<Float> = responsibilities.sum_axis(Axis(0));

            // Update weights
            for k in 0..n_components {
                weights[k] = nk[k] / n_samples as Float;
            }

            // Update means for each component
            for k in 0..n_components {
                if nk[k] > 1e-12 {
                    // Get responsibilities for component k
                    let component_weights = responsibilities.column(k);
                    let new_mean = simd_weighted_sum(&x.view(), &component_weights);
                    means.row_mut(k).assign(&(new_mean / nk[k]));
                } else {
                    // Keep the same mean for empty components
                    let mut weighted_mean = Array1::zeros(n_features);
                    for j in 0..n_features {
                        let mut sum = 0.0;
                        for i in 0..n_samples {
                            sum += responsibilities[(i, k)] * x[(i, j)];
                        }
                        weighted_mean[j] = sum / nk[k];
                    }
                    means.row_mut(k).assign(&weighted_mean);
                }
            }

            // Update covariances using SIMD operations
            for k in 0..n_components {
                if nk[k] > 1e-12 {
                    let mean_k = means.row(k);
                    let mut new_cov = Array2::zeros((n_features, n_features));

                    for i in 0..n_samples {
                        let sample = x.row(i);
                        let diff = &sample - &mean_k;

                        for j in 0..n_features {
                            for l in 0..n_features {
                                new_cov[(j, l)] +=
                                    responsibilities[(i, k)] * diff[j] * diff[l] / nk[k];
                            }
                        }
                    }

                    // Add regularization
                    for j in 0..n_features {
                        new_cov[(j, j)] += self.config.reg_covar;
                    }

                    covariances[k] = new_cov;
                }
            }

            // Compute log-likelihood with SIMD acceleration
            let mut new_log_likelihood = 0.0;
            for i in 0..n_samples {
                let sample = x.row(i);
                let mut sample_likelihood = 0.0;

                for k in 0..n_components {
                    let mean = means.row(k);
                    let inv_diag = self.extract_diagonal_inverse(&covariances[k])?;
                    let log_det = simd_log_determinant(&covariances[k].view());

                    let log_density = simd_multivariate_normal_log_density(
                        &sample,
                        &mean,
                        &inv_diag.view(),
                        log_det,
                    );

                    sample_likelihood += weights[k] * log_density.exp();
                }

                if sample_likelihood > 1e-12 {
                    new_log_likelihood += sample_likelihood.ln();
                }
            }

            // Check convergence using SIMD acceleration
            if simd_check_convergence(log_likelihood, new_log_likelihood, self.config.tol) {
                converged = true;
                log_likelihood = new_log_likelihood;
                break;
            }

            log_likelihood = new_log_likelihood;
        }

        // Apply SIMD regularization to final covariances
        for cov in &mut covariances {
            simd_regularize_covariance(cov, self.config.reg_covar);
        }

        Ok((
            weights,
            means,
            covariances,
            converged,
            n_iter,
            log_likelihood,
        ))
    }
}

impl<X, Y> Default for GaussianMixture<X, Y> {
    fn default() -> Self {
        Self::new()
    }
}

impl<X, Y> Estimator for GaussianMixture<X, Y> {
    type Config = GaussianMixtureConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<X: Send + Sync, Y: Send + Sync> Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>>
    for GaussianMixture<X, Y>
{
    type Fitted = Self;

    fn fit(self, x: &ArrayView2<Float>, _y: &ArrayView1<Float>) -> Result<Self::Fitted> {
        let (weights, means, covariances, converged, n_iter, log_likelihood) = self.fit_em(x)?;

        Ok(Self {
            config: self.config.clone(),
            weights: Some(weights),
            means: Some(means),
            covariances: Some(covariances),
            converged: Some(converged),
            n_iter: Some(n_iter),
            lower_bound: Some(log_likelihood),
            _phantom: PhantomData,
        })
    }
}

impl<X, Y> Predict<ArrayView2<'_, Float>, Array1<usize>> for GaussianMixture<X, Y> {
    fn predict(&self, x: &ArrayView2<Float>) -> Result<Array1<usize>> {
        let weights = self.weights()?;
        let means = self.means()?;
        let covariances = self.covariances()?;

        let mut labels = Array1::zeros(x.nrows());

        for (i, sample) in x.outer_iter().enumerate() {
            let mut max_likelihood = Float::NEG_INFINITY;
            let mut best_component = 0;

            for k in 0..self.config.n_components {
                let mean = means.row(k);
                let inv_diag = self.extract_diagonal_inverse(&covariances[k])?;
                let log_det = simd_log_determinant(&covariances[k].view());

                let log_likelihood = weights[k].ln()
                    + simd_multivariate_normal_log_density(
                        &sample,
                        &mean,
                        &inv_diag.view(),
                        log_det,
                    );

                if log_likelihood > max_likelihood {
                    max_likelihood = log_likelihood;
                    best_component = k;
                }
            }

            labels[i] = best_component;
        }

        Ok(labels)
    }
}

/// Predict probability for each sample in X for each Gaussian component
pub trait PredictProba<X, R> {
    fn predict_proba(&self, x: &X) -> Result<R>;
}

impl<X, Y> PredictProba<ArrayView2<'_, Float>, Array2<Float>> for GaussianMixture<X, Y> {
    fn predict_proba(&self, x: &ArrayView2<Float>) -> Result<Array2<Float>> {
        let weights = self.weights()?;
        let means = self.means()?;
        let covariances = self.covariances()?;
        let n_samples = x.nrows();
        let n_components = self.config.n_components;

        let mut proba = Array2::zeros((n_samples, n_components));

        for (i, sample) in x.outer_iter().enumerate() {
            let mut log_probs = Array1::zeros(n_components);

            for k in 0..n_components {
                let mean = means.row(k);
                let inv_diag = self.extract_diagonal_inverse(&covariances[k])?;
                let log_det = simd_log_determinant(&covariances[k].view());

                let log_likelihood = weights[k].ln()
                    + simd_multivariate_normal_log_density(
                        &sample,
                        &mean,
                        &inv_diag.view(),
                        log_det,
                    );

                log_probs[k] = log_likelihood;
            }

            // Convert to probabilities using SIMD log-sum-exp
            let log_sum = simd_log_sum_exp(&log_probs.view());
            for k in 0..n_components {
                proba[(i, k)] = (log_probs[k] - log_sum).exp();
            }
        }

        Ok(proba)
    }
}
