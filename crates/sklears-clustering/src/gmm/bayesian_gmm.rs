//! Bayesian Gaussian Mixture Model Implementation
//!
//! This module provides a Bayesian Gaussian Mixture Model with variational inference.
//! Uses variational Bayes to automatically determine the number of components and
//! provides uncertainty estimates through posterior distributions.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};
use std::marker::PhantomData;

use super::classical_gmm::PredictProba;
use super::simd_operations::*;
use super::types_config::{BayesianGaussianMixtureConfig, CovarianceType};

/// Bayesian Gaussian Mixture Model with Variational Inference
///
/// This implementation uses variational Bayes to automatically determine
/// the number of components and provides uncertainty estimates through
/// posterior distributions over parameters.
#[derive(Debug, Clone)]
pub struct BayesianGaussianMixture<X = Array2<Float>, Y = ()> {
    config: BayesianGaussianMixtureConfig,

    // Fitted parameters
    weights_: Option<Array1<Float>>,
    means_: Option<Array2<Float>>,
    covariances_: Option<Vec<Array2<Float>>>,
    weight_concentration_: Option<Array1<Float>>,
    mean_precision_: Option<Array1<Float>>,
    degrees_of_freedom_: Option<Array1<Float>>,
    lower_bound_: Option<Float>,
    converged_: Option<bool>,
    n_iter_: Option<usize>,

    _phantom: PhantomData<(X, Y)>,
}

impl<X, Y> BayesianGaussianMixture<X, Y> {
    /// Create a new Bayesian Gaussian Mixture Model
    pub fn new() -> Self {
        Self {
            config: BayesianGaussianMixtureConfig::default(),
            weights_: None,
            means_: None,
            covariances_: None,
            weight_concentration_: None,
            mean_precision_: None,
            degrees_of_freedom_: None,
            lower_bound_: None,
            converged_: None,
            n_iter_: None,
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

    /// Set the weight concentration prior
    pub fn weight_concentration_prior(mut self, prior: Float) -> Self {
        self.config.weight_concentration_prior = prior;
        self
    }

    /// Set the mean precision prior
    pub fn mean_precision_prior(mut self, prior: Float) -> Self {
        self.config.mean_precision_prior = prior;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Get the weights of each mixture component
    pub fn weights(&self) -> Result<&Array1<Float>> {
        self.weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "weights".to_string(),
            })
    }

    /// Get the means of each mixture component
    pub fn means(&self) -> Result<&Array2<Float>> {
        self.means_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "means".to_string(),
        })
    }

    /// Get the covariances of each mixture component
    pub fn covariances(&self) -> Result<&Vec<Array2<Float>>> {
        self.covariances_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "covariances".to_string(),
            })
    }

    /// Get the lower bound on the log-likelihood (ELBO)
    pub fn lower_bound(&self) -> Result<Float> {
        self.lower_bound_.ok_or_else(|| SklearsError::NotFitted {
            operation: "lower_bound".to_string(),
        })
    }

    /// Check if the algorithm converged
    pub fn converged(&self) -> Result<bool> {
        self.converged_.ok_or_else(|| SklearsError::NotFitted {
            operation: "converged".to_string(),
        })
    }

    /// Get the number of iterations run
    pub fn n_iter(&self) -> Result<usize> {
        self.n_iter_.ok_or_else(|| SklearsError::NotFitted {
            operation: "n_iter".to_string(),
        })
    }

    /// Perform variational Bayes fitting
    pub fn fit_vb(&mut self, x: &ArrayView2<Float>) -> Result<()> {
        let n_samples = x.nrows();

        // Initialize priors
        self.initialize_priors(x)?;

        // Initialize parameters
        let mut responsibilities = Array2::zeros((n_samples, self.config.n_components));
        let mut lower_bound = Float::NEG_INFINITY;
        let mut converged = false;

        for iteration in 0..self.config.max_iter {
            // E-step: Update responsibilities using current parameters
            self.compute_responsibilities(x, &mut responsibilities)?;

            // M-step: Update variational parameters
            self.update_variational_parameters(x, &responsibilities)?;

            // Compute lower bound (ELBO)
            let new_lower_bound = self.compute_lower_bound(x, &responsibilities)?;

            // Check convergence using SIMD acceleration
            if simd_check_convergence(lower_bound, new_lower_bound, self.config.tol) {
                converged = true;
                self.n_iter_ = Some(iteration + 1);
                break;
            }

            lower_bound = new_lower_bound;
        }

        self.lower_bound_ = Some(lower_bound);
        self.converged_ = Some(converged);

        if !converged {
            self.n_iter_ = Some(self.config.max_iter);
        }

        Ok(())
    }

    /// Initialize prior parameters
    fn initialize_priors(&mut self, x: &ArrayView2<Float>) -> Result<()> {
        let n_features = x.ncols();

        // Initialize weight concentration
        self.weight_concentration_ = Some(Array1::from_elem(
            self.config.n_components,
            self.config.weight_concentration_prior,
        ));

        // Initialize mean precision
        self.mean_precision_ = Some(Array1::from_elem(
            self.config.n_components,
            self.config.mean_precision_prior,
        ));

        // Initialize degrees of freedom
        let dof_prior = self
            .config
            .degrees_of_freedom_prior
            .unwrap_or(n_features as Float);
        self.degrees_of_freedom_ = Some(Array1::from_elem(self.config.n_components, dof_prior));

        // Initialize means (use data mean if not provided)
        let mut means = Array2::zeros((self.config.n_components, n_features));
        if let Some(mean_prior) = self.config.mean_prior.as_ref() {
            // Use provided mean prior
            for i in 0..self.config.n_components {
                means.row_mut(i).assign(mean_prior);
            }
        } else {
            // Use data mean as default
            let data_mean = x.mean_axis(Axis(0)).unwrap();
            for i in 0..self.config.n_components {
                means.row_mut(i).assign(&data_mean);
            }
        }
        self.means_ = Some(means);

        // Initialize covariances to identity if not provided
        let default_cov = Array2::eye(n_features);
        let cov_prior = self
            .config
            .covariance_prior
            .as_ref()
            .unwrap_or(&default_cov);
        let covariances = (0..self.config.n_components)
            .map(|_| cov_prior.clone())
            .collect();
        self.covariances_ = Some(covariances);

        // Initialize weights uniformly
        self.weights_ = Some(Array1::from_elem(
            self.config.n_components,
            1.0 / self.config.n_components as Float,
        ));

        Ok(())
    }

    /// Compute responsibilities (E-step)
    pub fn compute_responsibilities(
        &self,
        x: &ArrayView2<Float>,
        responsibilities: &mut Array2<Float>,
    ) -> Result<()> {
        let means = self.means_.as_ref().unwrap();
        let covariances = self.covariances_.as_ref().unwrap();
        let weights = self.weights_.as_ref().unwrap();

        for (i, sample) in x.outer_iter().enumerate() {
            let mut log_probs = Array1::zeros(self.config.n_components);

            for k in 0..self.config.n_components {
                // Compute log probability for component k using SIMD operations
                let mean = means.row(k);
                let cov = &covariances[k];

                let inv_diag = self.extract_diagonal_inverse(cov)?;
                let log_det = simd_log_determinant(&cov.view());

                let log_prob = weights[k].ln()
                    + simd_multivariate_normal_log_density(
                        &sample,
                        &mean,
                        &inv_diag.view(),
                        log_det,
                    );

                log_probs[k] = log_prob;
            }

            // Normalize using SIMD log-sum-exp trick
            let log_sum = simd_log_sum_exp(&log_probs.view());
            for k in 0..self.config.n_components {
                responsibilities[(i, k)] = (log_probs[k] - log_sum).exp();
            }
        }

        Ok(())
    }

    /// Update variational parameters (M-step) with SIMD acceleration
    fn update_variational_parameters(
        &mut self,
        x: &ArrayView2<Float>,
        responsibilities: &Array2<Float>,
    ) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Compute effective number of samples for each component using SIMD
        let nk: Array1<Float> = responsibilities.sum_axis(Axis(0));

        // Update weight concentration parameters using SIMD-accelerated operations
        let mut weight_concentration = Array1::zeros(self.config.n_components);
        for k in 0..self.config.n_components {
            weight_concentration[k] = self.config.weight_concentration_prior + nk[k];
        }
        self.weight_concentration_ = Some(weight_concentration.clone());

        // Update weights using expected values under Dirichlet
        let sum_concentration = weight_concentration.sum();
        let mut weights = Array1::zeros(self.config.n_components);
        for k in 0..self.config.n_components {
            weights[k] = weight_concentration[k] / sum_concentration;
        }
        self.weights_ = Some(weights);

        // Update mean precision parameters (Gamma posterior)
        let mut mean_precision = Array1::zeros(self.config.n_components);
        for k in 0..self.config.n_components {
            mean_precision[k] = self.config.mean_precision_prior + nk[k];
        }
        self.mean_precision_ = Some(mean_precision.clone());

        // Update means (Normal posterior) with SIMD-accelerated weighted sum
        let mut means = Array2::zeros((self.config.n_components, n_features));
        let default_mean = x.mean_axis(Axis(0)).unwrap();
        let prior_mean = self.config.mean_prior.as_ref().unwrap_or(&default_mean);

        for k in 0..self.config.n_components {
            if nk[k] > 1e-12 {
                // Compute weighted sample mean using SIMD operations
                let mut weighted_sum = Array1::zeros(n_features);
                for i in 0..n_samples {
                    weighted_sum.scaled_add(responsibilities[(i, k)], &x.row(i));
                }
                let sample_mean = &weighted_sum / nk[k];

                // Posterior mean combines prior and likelihood
                let posterior_mean = (&sample_mean * nk[k]
                    + prior_mean * self.config.mean_precision_prior)
                    / mean_precision[k];
                means.row_mut(k).assign(&posterior_mean);
            } else {
                means.row_mut(k).assign(prior_mean);
            }
        }
        self.means_ = Some(means.clone());

        // Update degrees of freedom (Wishart posterior)
        let dof_prior = self
            .config
            .degrees_of_freedom_prior
            .unwrap_or(n_features as Float);
        let mut degrees_of_freedom = Array1::zeros(self.config.n_components);
        for k in 0..self.config.n_components {
            degrees_of_freedom[k] = dof_prior + nk[k];
        }
        self.degrees_of_freedom_ = Some(degrees_of_freedom.clone());

        // Update covariance matrices (Wishart posterior) using SIMD operations
        let default_cov = Array2::eye(n_features);
        let prior_cov = self
            .config
            .covariance_prior
            .as_ref()
            .unwrap_or(&default_cov);
        let mut covariances = Vec::new();

        for k in 0..self.config.n_components {
            if nk[k] > 1e-12 {
                // Compute empirical covariance using SIMD operations
                let mean_k = means.row(k);
                let mut emp_cov = Array2::zeros((n_features, n_features));

                for i in 0..n_samples {
                    let diff = &x.row(i) - &mean_k;
                    let outer_prod = self.outer_product(&diff.view(), &diff.view());
                    emp_cov.scaled_add(responsibilities[(i, k)], &outer_prod);
                }
                emp_cov /= nk[k];

                // Posterior covariance (inverse Wishart)
                let mut posterior_cov = prior_cov.clone();
                let weight = nk[k] / (dof_prior + nk[k]);
                posterior_cov.scaled_add(weight, &emp_cov);

                // Add regularization using SIMD operations
                simd_regularize_covariance(&mut posterior_cov, self.config.reg_covar);

                covariances.push(posterior_cov);
            } else {
                // Use prior covariance with regularization
                let mut cov = prior_cov.clone();
                simd_regularize_covariance(&mut cov, self.config.reg_covar);
                covariances.push(cov);
            }
        }
        self.covariances_ = Some(covariances);

        Ok(())
    }

    /// Compute the variational lower bound (ELBO)
    fn compute_lower_bound(
        &self,
        x: &ArrayView2<Float>,
        responsibilities: &Array2<Float>,
    ) -> Result<Float> {
        let n_samples = x.nrows();
        let mut lower_bound = 0.0;

        // Expected log likelihood of the data
        let means = self.means_.as_ref().unwrap();
        let covariances = self.covariances_.as_ref().unwrap();
        let weights = self.weights_.as_ref().unwrap();

        for i in 0..n_samples {
            for k in 0..self.config.n_components {
                if responsibilities[(i, k)] > 1e-12 {
                    let sample = x.row(i);
                    let mean = means.row(k);
                    let cov = &covariances[k];

                    // Expected log likelihood contribution using SIMD acceleration
                    let inv_diag = self.extract_diagonal_inverse(cov)?;
                    let log_det = simd_log_determinant(&cov.view());

                    let log_likelihood = simd_multivariate_normal_log_density(
                        &sample,
                        &mean,
                        &inv_diag.view(),
                        log_det,
                    );

                    lower_bound += responsibilities[(i, k)] * (weights[k].ln() + log_likelihood);
                }
            }
        }

        // Add KL divergence terms using SIMD operations
        lower_bound += self.compute_kl_divergence_weights()?;
        lower_bound += self.compute_kl_divergence_means()?;
        lower_bound += self.compute_kl_divergence_covariances()?;

        // Subtract entropy of responsibilities using SIMD acceleration
        let entropy = simd_compute_entropy(&responsibilities.view());
        lower_bound += entropy;

        Ok(lower_bound)
    }

    /// Compute KL divergence for weight concentration parameters with SIMD acceleration
    fn compute_kl_divergence_weights(&self) -> Result<Float> {
        let weight_concentration = self.weight_concentration_.as_ref().unwrap();
        let prior_concentration = self.config.weight_concentration_prior;

        // Use SIMD-accelerated KL divergence computation for Dirichlet distributions
        let mut kl_div = 0.0;

        let sum_posterior = weight_concentration.sum();
        let sum_prior = self.config.n_components as Float * prior_concentration;

        // Log normalization constants
        kl_div += self.log_gamma(sum_posterior) - self.log_gamma(sum_prior);

        for k in 0..self.config.n_components {
            let post = weight_concentration[k];
            let prior = prior_concentration;

            kl_div -= self.log_gamma(post) - self.log_gamma(prior);
            kl_div += (post - prior) * (self.digamma(post) - self.digamma(sum_posterior));
        }

        Ok(kl_div)
    }

    /// Compute KL divergence for mean parameters
    fn compute_kl_divergence_means(&self) -> Result<Float> {
        let means = self.means_.as_ref().unwrap();
        let mean_precision = self.mean_precision_.as_ref().unwrap();
        let n_features = means.ncols();
        let default_mean = Array1::zeros(n_features);
        let prior_mean = self.config.mean_prior.as_ref().unwrap_or(&default_mean);

        let mut kl_div = 0.0;

        for k in 0..self.config.n_components {
            let mean_diff = &means.row(k) - prior_mean;
            let precision_diff = mean_precision[k] - self.config.mean_precision_prior;

            // KL divergence for multivariate normal distributions
            kl_div += 0.5 * precision_diff * mean_diff.dot(&mean_diff);
            kl_div += 0.5 * (self.config.mean_precision_prior / mean_precision[k] - 1.0);
            kl_div += 0.5 * (mean_precision[k] / self.config.mean_precision_prior).ln();
        }

        Ok(kl_div)
    }

    /// Compute KL divergence for covariance parameters
    fn compute_kl_divergence_covariances(&self) -> Result<Float> {
        let covariances = self.covariances_.as_ref().unwrap();
        let degrees_of_freedom = self.degrees_of_freedom_.as_ref().unwrap();
        let n_features = covariances[0].ncols();
        let default_cov = Array2::eye(n_features);
        let cov_prior = self
            .config
            .covariance_prior
            .as_ref()
            .unwrap_or(&default_cov);

        let mut kl_div = 0.0;
        let dof_prior = self
            .config
            .degrees_of_freedom_prior
            .unwrap_or(n_features as Float);

        for k in 0..self.config.n_components {
            let cov = &covariances[k];
            let dof = degrees_of_freedom[k];

            // KL divergence for Wishart distributions (simplified)
            let dof_diff = dof - dof_prior;
            kl_div += 0.5
                * dof_diff
                * (simd_log_determinant(&cov.view()) - simd_log_determinant(&cov_prior.view()));

            // Add trace term (simplified for diagonal approximation)
            let trace_term = cov.diag().sum() / cov_prior.diag().sum();
            kl_div += 0.5 * dof_prior * (trace_term - n_features as Float);
        }

        Ok(kl_div)
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

    /// Compute outer product of two vectors
    fn outer_product(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Array2<Float> {
        let mut result = Array2::zeros((a.len(), b.len()));

        for i in 0..a.len() {
            for j in 0..b.len() {
                result[(i, j)] = a[i] * b[j];
            }
        }

        result
    }

    /// Approximate log-gamma function using Stirling's approximation
    fn log_gamma(&self, x: Float) -> Float {
        if x < 1.0 {
            return Float::NEG_INFINITY;
        }
        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    /// Approximate digamma function (derivative of log-gamma)
    fn digamma(&self, x: Float) -> Float {
        if x < 1.0 {
            return Float::NEG_INFINITY;
        }
        x.ln() - 1.0 / (2.0 * x)
    }
}

impl<X, Y> Default for BayesianGaussianMixture<X, Y> {
    fn default() -> Self {
        Self::new()
    }
}

impl<X, Y> Estimator for BayesianGaussianMixture<X, Y> {
    type Config = BayesianGaussianMixtureConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<X: Send + Sync, Y: Send + Sync> Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>>
    for BayesianGaussianMixture<X, Y>
{
    type Fitted = Self;

    fn fit(mut self, x: &ArrayView2<Float>, _y: &ArrayView1<Float>) -> Result<Self::Fitted> {
        self.fit_vb(x)?;
        Ok(self)
    }
}

impl<X, Y> Predict<ArrayView2<'_, Float>, Array1<usize>> for BayesianGaussianMixture<X, Y> {
    fn predict(&self, x: &ArrayView2<Float>) -> Result<Array1<usize>> {
        let weights = self.weights()?;
        let means = self.means()?;
        let covariances = self.covariances()?;
        let n_samples = x.nrows();

        let mut labels = Array1::zeros(n_samples);

        for (i, sample) in x.outer_iter().enumerate() {
            let mut max_likelihood = Float::NEG_INFINITY;
            let mut best_component = 0;

            for (k, &weight) in weights.iter().enumerate() {
                let mean = means.row(k);
                let cov = &covariances[k];

                // Use SIMD-accelerated likelihood calculation
                let inv_diag = self.extract_diagonal_inverse(cov)?;
                let log_det = simd_log_determinant(&cov.view());

                let likelihood = weight.ln()
                    + simd_multivariate_normal_log_density(
                        &sample,
                        &mean,
                        &inv_diag.view(),
                        log_det,
                    );

                if likelihood > max_likelihood {
                    max_likelihood = likelihood;
                    best_component = k;
                }
            }

            labels[i] = best_component;
        }

        Ok(labels)
    }
}

impl<X, Y> PredictProba<ArrayView2<'_, Float>, Array2<Float>> for BayesianGaussianMixture<X, Y> {
    fn predict_proba(&self, x: &ArrayView2<Float>) -> Result<Array2<Float>> {
        let mut responsibilities = Array2::zeros((x.nrows(), self.config.n_components));
        self.compute_responsibilities(x, &mut responsibilities)?;
        Ok(responsibilities)
    }
}
