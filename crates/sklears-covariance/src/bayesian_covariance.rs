//! Bayesian Covariance Estimation
//!
//! This module implements various Bayesian approaches for covariance matrix estimation,
//! including inverse-Wishart priors, variational Bayes, hierarchical models, and MCMC sampling.

use scirs2_core::ndarray::ndarray_linalg::{Cholesky, UPLO};
use scirs2_core::ndarray::{s, Array1, Array2, Axis, NdFloat};
use scirs2_core::numeric::FromPrimitive;
use scirs2_core::random::essentials::Uniform;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Distribution;
use scirs2_core::StandardNormal;

use crate::utils::{matrix_determinant, matrix_inverse, regularize_matrix, validate_data};
use sklears_core::prelude::*;
use sklears_core::traits::Fit;

/// Bayesian covariance estimation methods
#[derive(Debug, Clone, PartialEq)]
pub enum BayesianMethod {
    /// Inverse-Wishart prior with conjugate updating
    InverseWishart,
    /// Variational Bayes approximation
    VariationalBayes,
    /// Hierarchical Bayesian model
    Hierarchical,
    /// Metropolis-Hastings MCMC sampling
    McmcMetropolis,
    /// Gibbs sampling MCMC
    McmcGibbs,
}

/// Prior specification for Bayesian covariance estimation
#[derive(Debug, Clone)]
pub struct BayesianPrior<F: NdFloat + FromPrimitive> {
    /// Prior covariance matrix (scale matrix for inverse-Wishart)
    pub scale_matrix: Array2<F>,
    /// Degrees of freedom for inverse-Wishart prior
    pub degrees_of_freedom: F,
    /// Prior mean for hierarchical models
    pub prior_mean: Option<Array1<F>>,
    /// Hyperparameters for hierarchical priors
    pub hyperparameters: Vec<F>,
}

/// MCMC sampling configuration
#[derive(Debug, Clone)]
pub struct McmcConfig<F: NdFloat + FromPrimitive> {
    /// Number of MCMC samples
    pub n_samples: usize,
    /// Number of burn-in samples
    pub burn_in: usize,
    /// Thinning interval
    pub thin: usize,
    /// Proposal covariance scaling
    pub proposal_scale: F,
    /// Random seed
    pub random_state: Option<u64>,
}

/// Variational Bayes configuration
#[derive(Debug, Clone)]
pub struct VariationalConfig<F: NdFloat + FromPrimitive> {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: F,
    /// Learning rate for natural gradient
    pub learning_rate: F,
    /// Use structured mean-field approximation
    pub structured_meanfield: bool,
}

/// Configuration for Bayesian covariance estimation
#[derive(Debug, Clone)]
pub struct BayesianCovarianceConfig<F: NdFloat + FromPrimitive> {
    /// Bayesian estimation method
    pub method: BayesianMethod,
    /// Prior specification
    pub prior: BayesianPrior<F>,
    /// MCMC configuration (if applicable)
    pub mcmc_config: Option<McmcConfig<F>>,
    /// Variational Bayes configuration (if applicable)
    pub variational_config: Option<VariationalConfig<F>>,
    /// Regularization parameter for numerical stability
    pub regularization: F,
    /// Random seed
    pub random_state: Option<u64>,
}

impl<F: NdFloat + FromPrimitive> Default for BayesianCovarianceConfig<F> {
    fn default() -> Self {
        Self {
            method: BayesianMethod::InverseWishart,
            prior: BayesianPrior {
                scale_matrix: Array2::eye(1),
                degrees_of_freedom: F::from(2.0).unwrap(),
                prior_mean: None,
                hyperparameters: vec![],
            },
            mcmc_config: Some(McmcConfig {
                n_samples: 1000,
                burn_in: 200,
                thin: 1,
                proposal_scale: F::from(0.1).unwrap(),
                random_state: None,
            }),
            variational_config: Some(VariationalConfig {
                max_iter: 100,
                tol: F::from(1e-6).unwrap(),
                learning_rate: F::from(0.01).unwrap(),
                structured_meanfield: false,
            }),
            regularization: F::from(1e-8).unwrap(),
            random_state: None,
        }
    }
}

/// Bayesian covariance estimator in untrained state
pub struct BayesianCovariance<F: NdFloat + FromPrimitive> {
    config: BayesianCovarianceConfig<F>,
}

/// Bayesian covariance estimator in trained state
pub struct BayesianCovarianceFitted<F: NdFloat + FromPrimitive> {
    config: BayesianCovarianceConfig<F>,
    /// Posterior mean covariance matrix
    covariance_: Array2<F>,
    /// Posterior covariance samples (for MCMC methods)
    samples_: Option<Vec<Array2<F>>>,
    /// Variational parameters (for VB methods)
    variational_params_: Option<VariationalParameters<F>>,
    /// Log marginal likelihood
    log_likelihood_: F,
    /// Number of features
    n_features_: usize,
    /// Number of samples used for estimation
    n_samples_: usize,
}

/// Variational parameters for Bayesian inference
#[derive(Debug, Clone)]
pub struct VariationalParameters<F: NdFloat + FromPrimitive> {
    /// Variational scale matrix
    pub scale_matrix: Array2<F>,
    /// Variational degrees of freedom
    pub degrees_of_freedom: F,
    /// Variational mean (for hierarchical models)
    pub mean: Option<Array1<F>>,
    /// Lower bound on log marginal likelihood
    pub lower_bound: F,
}

impl<F: NdFloat + FromPrimitive> BayesianCovariance<F> {
    /// Create a new Bayesian covariance estimator
    pub fn new(config: BayesianCovarianceConfig<F>) -> Self {
        Self { config }
    }

    /// Create a new Bayesian covariance estimator with builder pattern
    pub fn builder() -> BayesianCovarianceBuilder<F> {
        BayesianCovarianceBuilder::new()
    }

    /// Get the configuration
    pub fn config(&self) -> &BayesianCovarianceConfig<F> {
        &self.config
    }
}

impl<F: NdFloat + FromPrimitive> BayesianCovarianceFitted<F> {
    /// Get the estimated covariance matrix (posterior mean)
    pub fn covariance(&self) -> &Array2<F> {
        &self.covariance_
    }

    /// Get the posterior samples (if available)
    pub fn samples(&self) -> Option<&Vec<Array2<F>>> {
        self.samples_.as_ref()
    }

    /// Get the variational parameters (if available)
    pub fn variational_params(&self) -> Option<&VariationalParameters<F>> {
        self.variational_params_.as_ref()
    }

    /// Get the log marginal likelihood
    pub fn log_likelihood(&self) -> F {
        self.log_likelihood_
    }

    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features_
    }

    /// Get the number of samples used for estimation
    pub fn n_samples(&self) -> usize {
        self.n_samples_
    }

    /// Get the configuration
    pub fn config(&self) -> &BayesianCovarianceConfig<F> {
        &self.config
    }

    /// Compute posterior credible intervals for covariance elements
    pub fn credible_intervals(&self, alpha: F) -> Result<(Array2<F>, Array2<F>)> {
        if let Some(samples) = &self.samples_ {
            let n_samples = samples.len();
            let (n_features, _) = samples[0].dim();

            let lower_percentile = alpha / F::from(2.0).unwrap();
            let upper_percentile = F::one() - lower_percentile;

            let mut lower_bounds = Array2::zeros((n_features, n_features));
            let mut upper_bounds = Array2::zeros((n_features, n_features));

            for i in 0..n_features {
                for j in 0..n_features {
                    let mut values: Vec<F> = samples.iter().map(|sample| sample[[i, j]]).collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let lower_idx = (lower_percentile * F::from(n_samples).unwrap())
                        .to_usize()
                        .unwrap();
                    let upper_idx = (upper_percentile * F::from(n_samples).unwrap())
                        .to_usize()
                        .unwrap()
                        .min(n_samples - 1);

                    lower_bounds[[i, j]] = values[lower_idx];
                    upper_bounds[[i, j]] = values[upper_idx];
                }
            }

            Ok((lower_bounds, upper_bounds))
        } else {
            Err(SklearsError::InvalidInput(
                "Credible intervals require MCMC samples".to_string(),
            ))
        }
    }

    /// Predict covariance for new data using posterior predictive distribution
    pub fn predict_covariance(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_ {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_,
                actual: n_features,
            });
        }

        match self.config.method {
            BayesianMethod::InverseWishart => {
                // For inverse-Wishart posterior, predictive covariance includes uncertainty
                let sample_cov = self.compute_sample_covariance(x)?;
                let posterior_scale = &self.covariance_;
                let posterior_df =
                    self.config.prior.degrees_of_freedom + F::from(self.n_samples_).unwrap();

                // Predictive covariance combines sample and posterior uncertainty
                let predictive_cov = sample_cov
                    + posterior_scale * (posterior_df + F::one())
                        / (posterior_df * F::from(n_samples).unwrap());

                Ok(predictive_cov)
            }
            _ => {
                // For other methods, use posterior mean as approximation
                Ok(self.covariance_.clone())
            }
        }
    }

    fn compute_sample_covariance(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let (n_samples, n_features) = x.dim();
        let mean = x.mean_axis(Axis(0)).unwrap();

        let mut cov = Array2::zeros((n_features, n_features));
        for i in 0..n_samples {
            let diff = &x.slice(s![i, ..]) - &mean;
            let diff_2d = diff.clone().insert_axis(Axis(1));
            let diff_t = diff.insert_axis(Axis(0));
            cov = cov + diff_t.dot(&diff_2d);
        }

        Ok(cov / F::from(n_samples - 1).unwrap())
    }
}

impl<F: NdFloat + sklears_core::types::FloatBounds> Estimator for BayesianCovariance<F> {
    type Config = BayesianCovarianceConfig<F>;
    type Error = SklearsError;
    type Float = F;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<F: NdFloat + FromPrimitive> Fit<Array2<F>, ()> for BayesianCovariance<F> {
    type Fitted = BayesianCovarianceFitted<F>;

    fn fit(self, x: &Array2<F>, _y: &()) -> Result<Self::Fitted> {
        validate_data(x)?;
        let (n_samples, n_features) = x.dim();

        // Ensure prior scale matrix has correct dimensions
        let mut prior = self.config.prior.clone();
        if prior.scale_matrix.nrows() != n_features {
            prior.scale_matrix = Array2::eye(n_features) * prior.scale_matrix[[0, 0]];
        }

        let mut rng = thread_rng();

        let (covariance, samples, variational_params, log_likelihood) = match self.config.method {
            BayesianMethod::InverseWishart => self.fit_inverse_wishart(x, &prior)?,
            BayesianMethod::VariationalBayes => self.fit_variational_bayes(x, &prior)?,
            BayesianMethod::Hierarchical => self.fit_hierarchical(x, &prior, &mut rng)?,
            BayesianMethod::McmcMetropolis => self.fit_mcmc_metropolis(x, &prior, &mut rng)?,
            BayesianMethod::McmcGibbs => self.fit_mcmc_gibbs(x, &prior, &mut rng)?,
        };

        Ok(BayesianCovarianceFitted {
            config: self.config,
            covariance_: covariance,
            samples_: samples,
            variational_params_: variational_params,
            log_likelihood_: log_likelihood,
            n_features_: n_features,
            n_samples_: n_samples,
        })
    }
}

impl<F: NdFloat + FromPrimitive> BayesianCovariance<F> {
    /// Fit using inverse-Wishart conjugate prior
    fn fit_inverse_wishart(
        &self,
        x: &Array2<F>,
        prior: &BayesianPrior<F>,
    ) -> Result<(
        Array2<F>,
        Option<Vec<Array2<F>>>,
        Option<VariationalParameters<F>>,
        F,
    )> {
        let (n_samples, n_features) = x.dim();

        // Compute sample covariance
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut sample_cov = Array2::zeros((n_features, n_features));

        for i in 0..n_samples {
            let diff = &x.slice(s![i, ..]) - &mean;
            let diff_2d = diff.clone().insert_axis(Axis(1));
            let diff_t = diff.insert_axis(Axis(0));
            sample_cov = sample_cov + diff_t.dot(&diff_2d);
        }

        // Posterior parameters for inverse-Wishart
        let posterior_df = prior.degrees_of_freedom + F::from(n_samples).unwrap();
        let posterior_scale = &prior.scale_matrix + &sample_cov;

        // Posterior mean covariance
        let covariance = &posterior_scale / (posterior_df - F::from(n_features + 1).unwrap());
        let regularized_cov = regularize_matrix(&covariance, self.config.regularization)?;

        // Compute log marginal likelihood
        let log_likelihood = self.compute_inverse_wishart_log_likelihood(
            x,
            &prior.scale_matrix,
            prior.degrees_of_freedom,
            &posterior_scale,
            posterior_df,
        )?;

        Ok((regularized_cov, None, None, log_likelihood))
    }

    /// Fit using variational Bayes approximation
    fn fit_variational_bayes(
        &self,
        x: &Array2<F>,
        prior: &BayesianPrior<F>,
    ) -> Result<(
        Array2<F>,
        Option<Vec<Array2<F>>>,
        Option<VariationalParameters<F>>,
        F,
    )> {
        let config = self.config.variational_config.as_ref().unwrap();
        let (n_samples, n_features) = x.dim();

        // Initialize variational parameters
        let mut var_scale = prior.scale_matrix.clone();
        let mut var_df = prior.degrees_of_freedom;
        let mut lower_bound = F::neg_infinity();

        // Compute sample statistics
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut sample_cov = Array2::zeros((n_features, n_features));

        for i in 0..n_samples {
            let diff = &x.slice(s![i, ..]) - &mean;
            let diff_2d = diff.clone().insert_axis(Axis(1));
            let diff_t = diff.insert_axis(Axis(0));
            sample_cov = sample_cov + diff_t.dot(&diff_2d);
        }

        // Variational optimization
        for iteration in 0..config.max_iter {
            let old_lower_bound = lower_bound;

            // Update variational parameters
            var_df = prior.degrees_of_freedom + F::from(n_samples).unwrap();
            var_scale = &prior.scale_matrix + &sample_cov;

            // Compute lower bound
            lower_bound = self.compute_variational_lower_bound(
                &sample_cov,
                &prior.scale_matrix,
                prior.degrees_of_freedom,
                &var_scale,
                var_df,
                n_samples,
            )?;

            // Check convergence
            if iteration > 0 && (lower_bound - old_lower_bound).abs() < config.tol {
                break;
            }
        }

        // Posterior mean
        let covariance = &var_scale / (var_df - F::from(n_features + 1).unwrap());
        let regularized_cov = regularize_matrix(&covariance, self.config.regularization)?;

        let variational_params = VariationalParameters {
            scale_matrix: var_scale,
            degrees_of_freedom: var_df,
            mean: None,
            lower_bound,
        };

        Ok((regularized_cov, None, Some(variational_params), lower_bound))
    }

    /// Fit using hierarchical Bayesian model
    fn fit_hierarchical(
        &self,
        x: &Array2<F>,
        prior: &BayesianPrior<F>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<(
        Array2<F>,
        Option<Vec<Array2<F>>>,
        Option<VariationalParameters<F>>,
        F,
    )> {
        // Simplified hierarchical model with hyperpriors
        let (n_samples, _) = x.dim();

        // Use inverse-Wishart as base but with hierarchical hyperparameters
        let mut hierarchical_prior = prior.clone();

        // Update hyperparameters based on data
        if !prior.hyperparameters.is_empty() {
            let alpha = prior.hyperparameters[0];
            let beta = if prior.hyperparameters.len() > 1 {
                prior.hyperparameters[1]
            } else {
                F::one()
            };

            // Update degrees of freedom using hierarchical prior
            hierarchical_prior.degrees_of_freedom = alpha + F::from(n_samples).unwrap() / beta;
        }

        // Fit with updated hierarchical prior
        self.fit_inverse_wishart(x, &hierarchical_prior)
    }

    /// Fit using Metropolis-Hastings MCMC
    fn fit_mcmc_metropolis(
        &self,
        x: &Array2<F>,
        prior: &BayesianPrior<F>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<(
        Array2<F>,
        Option<Vec<Array2<F>>>,
        Option<VariationalParameters<F>>,
        F,
    )> {
        let mcmc_config = self.config.mcmc_config.as_ref().unwrap();
        let (n_samples, n_features) = x.dim();

        // Initialize with sample covariance
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut sample_cov = Array2::zeros((n_features, n_features));

        for i in 0..n_samples {
            let diff = &x.slice(s![i, ..]) - &mean;
            let diff_2d = diff.clone().insert_axis(Axis(1));
            let diff_t = diff.insert_axis(Axis(0));
            sample_cov = sample_cov + diff_t.dot(&diff_2d);
        }
        sample_cov = sample_cov / F::from(n_samples).unwrap();

        let mut current_cov = regularize_matrix(&sample_cov, self.config.regularization)?;
        let mut current_log_posterior = self.compute_log_posterior(x, &current_cov, prior)?;

        let mut samples = Vec::new();
        let mut n_accepted = 0;

        // MCMC sampling
        for iteration in 0..(mcmc_config.burn_in + mcmc_config.n_samples * mcmc_config.thin) {
            // Propose new covariance matrix
            let proposal =
                self.propose_covariance(&current_cov, mcmc_config.proposal_scale, rng)?;
            let proposal_log_posterior = self.compute_log_posterior(x, &proposal, prior)?;

            // Metropolis-Hastings acceptance
            let log_ratio = proposal_log_posterior - current_log_posterior;
            let accept_prob = log_ratio.exp().min(F::one());

            let uniform = Uniform::new(0.0, 1.0).map_err(|_| {
                SklearsError::InvalidInput("Invalid uniform distribution".to_string())
            })?;
            let sample_f64: f64 = uniform.sample(rng);
            let sample_f = F::from(sample_f64).ok_or_else(|| {
                SklearsError::InvalidInput("Failed to convert f64 to F".to_string())
            })?;
            if sample_f < accept_prob {
                current_cov = proposal;
                current_log_posterior = proposal_log_posterior;
                n_accepted += 1;
            }

            // Store samples after burn-in
            if iteration >= mcmc_config.burn_in
                && (iteration - mcmc_config.burn_in) % mcmc_config.thin == 0
            {
                samples.push(current_cov.clone());
            }
        }

        // Compute posterior mean
        let mut mean_cov = Array2::zeros((n_features, n_features));
        for sample in &samples {
            mean_cov = mean_cov + sample;
        }
        mean_cov = mean_cov / F::from(samples.len()).unwrap();

        let log_likelihood = current_log_posterior;

        Ok((mean_cov, Some(samples), None, log_likelihood))
    }

    /// Fit using Gibbs sampling MCMC
    fn fit_mcmc_gibbs(
        &self,
        x: &Array2<F>,
        prior: &BayesianPrior<F>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<(
        Array2<F>,
        Option<Vec<Array2<F>>>,
        Option<VariationalParameters<F>>,
        F,
    )> {
        // For Gibbs sampling with inverse-Wishart, we can sample directly from conjugate posterior
        // This is more efficient than Metropolis-Hastings for this case
        let mcmc_config = self.config.mcmc_config.as_ref().unwrap();
        let (n_samples, n_features) = x.dim();

        // Compute posterior parameters
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut sample_cov = Array2::zeros((n_features, n_features));

        for i in 0..n_samples {
            let diff = &x.slice(s![i, ..]) - &mean;
            let diff_2d = diff.clone().insert_axis(Axis(1));
            let diff_t = diff.insert_axis(Axis(0));
            sample_cov = sample_cov + diff_t.dot(&diff_2d);
        }

        let posterior_df = prior.degrees_of_freedom + F::from(n_samples).unwrap();
        let posterior_scale = &prior.scale_matrix + &sample_cov;

        let mut samples = Vec::new();

        // Gibbs sampling from inverse-Wishart posterior
        for iteration in 0..(mcmc_config.burn_in + mcmc_config.n_samples * mcmc_config.thin) {
            // Sample from inverse-Wishart distribution
            let sample = self.sample_inverse_wishart(&posterior_scale, posterior_df, rng)?;

            // Store samples after burn-in
            if iteration >= mcmc_config.burn_in
                && (iteration - mcmc_config.burn_in) % mcmc_config.thin == 0
            {
                samples.push(sample);
            }
        }

        // Compute posterior mean
        let mut mean_cov = Array2::zeros((n_features, n_features));
        for sample in &samples {
            mean_cov = mean_cov + sample;
        }
        mean_cov = mean_cov / F::from(samples.len()).unwrap();

        let log_likelihood = self.compute_inverse_wishart_log_likelihood(
            x,
            &prior.scale_matrix,
            prior.degrees_of_freedom,
            &posterior_scale,
            posterior_df,
        )?;

        Ok((mean_cov, Some(samples), None, log_likelihood))
    }

    /// Propose new covariance matrix for Metropolis-Hastings
    fn propose_covariance(
        &self,
        current: &Array2<F>,
        scale: F,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<Array2<F>> {
        let n_features = current.nrows();

        // Generate random perturbation
        let mut perturbation = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in i..n_features {
                let noise: f64 = StandardNormal.sample(rng);
                let noise_f = F::from(noise).unwrap() * scale;
                perturbation[[i, j]] = noise_f;
                if i != j {
                    perturbation[[j, i]] = noise_f;
                }
            }
        }

        // Ensure positive definiteness
        let proposal = current + &perturbation;
        regularize_matrix(&proposal, self.config.regularization)
    }

    /// Sample from inverse-Wishart distribution
    fn sample_inverse_wishart(
        &self,
        scale: &Array2<F>,
        df: F,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<Array2<F>> {
        let n = scale.nrows();

        // Sample from Wishart distribution first
        let mut a = Array2::zeros((n, n));

        // Fill lower triangular part
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Chi-squared distribution for diagonal elements
                    let chi_sq_param = df - F::from(i).unwrap();
                    let normal_sample: f64 = StandardNormal.sample(rng);
                    let chi_sq_sample = F::from(normal_sample).unwrap().powi(2);
                    a[[i, j]] = (chi_sq_sample * chi_sq_param).sqrt();
                } else {
                    // Standard normal for off-diagonal elements
                    let normal_sample: f64 = StandardNormal.sample(rng);
                    a[[i, j]] = F::from(normal_sample).unwrap();
                }
            }
        }

        // Cholesky decomposition of scale matrix
        let scale_f64 = scale.mapv(|x| x.to_f64().unwrap_or(0.0));
        let chol_f64 = scale_f64.cholesky(UPLO::Lower).map_err(|_| {
            SklearsError::NumericalError("Failed to compute Cholesky decomposition".to_string())
        })?;
        let chol = chol_f64.mapv(|x| F::from(x).unwrap_or(F::zero()));

        // Form the sample
        let la = chol.dot(&a);
        let wishart_sample = la.dot(&la.t());

        // Return inverse for inverse-Wishart
        let wishart_f64 = wishart_sample.mapv(|x| x.to_f64().unwrap_or(0.0));
        let inv_wishart_f64 = matrix_inverse(&wishart_f64).map_err(|_| {
            SklearsError::NumericalError("Failed to invert Wishart sample".to_string())
        })?;
        let inv_wishart = inv_wishart_f64.mapv(|x| F::from(x).unwrap_or(F::zero()));
        Ok(inv_wishart)
    }

    /// Compute log posterior density
    fn compute_log_posterior(
        &self,
        x: &Array2<F>,
        cov: &Array2<F>,
        prior: &BayesianPrior<F>,
    ) -> Result<F> {
        let (n_samples, n_features) = x.dim();

        // Log likelihood
        let cov_f64 = cov.mapv(|x| x.to_f64().unwrap_or(0.0));
        let det_cov_f64 = matrix_determinant(&cov_f64);
        let det_cov = F::from(det_cov_f64).unwrap_or(F::zero());

        if det_cov <= F::zero() {
            return Ok(F::neg_infinity());
        }

        let cov_f64 = cov.mapv(|x| x.to_f64().unwrap_or(0.0));
        let inv_cov_f64 = matrix_inverse(&cov_f64).map_err(|_| {
            SklearsError::NumericalError("Failed to invert covariance matrix".to_string())
        })?;
        let inv_cov = inv_cov_f64.mapv(|x| F::from(x).unwrap_or(F::zero()));

        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut log_likelihood = F::zero();

        for i in 0..n_samples {
            let diff = &x.slice(s![i, ..]) - &mean;
            let mahalanobis = diff.dot(&inv_cov).dot(&diff);
            log_likelihood = log_likelihood - mahalanobis / F::from(2.0).unwrap();
        }

        log_likelihood = log_likelihood
            - F::from(n_samples).unwrap()
                * (F::from(n_features).unwrap()
                    * F::from(2.0 * std::f64::consts::PI).unwrap().ln()
                    + det_cov.ln())
                / F::from(2.0).unwrap();

        // Log prior (inverse-Wishart)
        let log_prior = self.compute_inverse_wishart_log_prior(
            cov,
            &prior.scale_matrix,
            prior.degrees_of_freedom,
        )?;

        Ok(log_likelihood + log_prior)
    }

    /// Compute inverse-Wishart log prior density
    fn compute_inverse_wishart_log_prior(
        &self,
        cov: &Array2<F>,
        scale: &Array2<F>,
        df: F,
    ) -> Result<F> {
        let n = cov.nrows();

        let cov_f64 = cov.mapv(|x| x.to_f64().unwrap_or(0.0));
        let det_cov_f64 = matrix_determinant(&cov_f64);
        let det_cov = F::from(det_cov_f64).unwrap_or(F::zero());

        let scale_f64 = scale.mapv(|x| x.to_f64().unwrap_or(0.0));
        let det_scale_f64 = matrix_determinant(&scale_f64);
        let det_scale = F::from(det_scale_f64).unwrap_or(F::zero());

        if det_cov <= F::zero() || det_scale <= F::zero() {
            return Ok(F::neg_infinity());
        }

        let cov_f64 = cov.mapv(|x| x.to_f64().unwrap_or(0.0));
        let inv_cov_f64 = matrix_inverse(&cov_f64).map_err(|_| {
            SklearsError::NumericalError("Failed to invert covariance matrix".to_string())
        })?;
        let inv_cov = inv_cov_f64.mapv(|x| F::from(x).unwrap_or(F::zero()));

        let trace_term = (scale.dot(&inv_cov)).diag().sum();

        let log_prior = (df / F::from(2.0).unwrap()) * det_scale.ln()
            - ((df + F::from(n + 1).unwrap()) / F::from(2.0).unwrap()) * det_cov.ln()
            - trace_term / F::from(2.0).unwrap();

        Ok(log_prior)
    }

    /// Compute inverse-Wishart log marginal likelihood
    fn compute_inverse_wishart_log_likelihood(
        &self,
        x: &Array2<F>,
        prior_scale: &Array2<F>,
        prior_df: F,
        posterior_scale: &Array2<F>,
        posterior_df: F,
    ) -> Result<F> {
        let (n_samples, n_features) = x.dim();

        let prior_scale_f64 = prior_scale.mapv(|x| x.to_f64().unwrap_or(0.0));
        let det_prior_f64 = matrix_determinant(&prior_scale_f64);
        let det_prior = F::from(det_prior_f64).unwrap_or(F::zero());

        let posterior_scale_f64 = posterior_scale.mapv(|x| x.to_f64().unwrap_or(0.0));
        let det_posterior_f64 = matrix_determinant(&posterior_scale_f64);
        let det_posterior = F::from(det_posterior_f64).unwrap_or(F::zero());

        if det_prior <= F::zero() || det_posterior <= F::zero() {
            return Ok(F::neg_infinity());
        }

        // Marginal likelihood for multivariate normal with inverse-Wishart prior
        let log_ml = (prior_df / F::from(2.0).unwrap()) * det_prior.ln()
            - (posterior_df / F::from(2.0).unwrap()) * det_posterior.ln()
            - F::from(n_samples * n_features).unwrap()
                * F::from(std::f64::consts::PI).unwrap().ln()
                / F::from(2.0).unwrap();

        Ok(log_ml)
    }

    /// Compute variational lower bound
    fn compute_variational_lower_bound(
        &self,
        sample_cov: &Array2<F>,
        prior_scale: &Array2<F>,
        prior_df: F,
        var_scale: &Array2<F>,
        var_df: F,
        n_samples: usize,
    ) -> Result<F> {
        let n_features = sample_cov.nrows();

        let var_scale_f64 = var_scale.mapv(|x| x.to_f64().unwrap_or(0.0));
        let det_var_scale_f64 = matrix_determinant(&var_scale_f64);
        let det_var_scale = F::from(det_var_scale_f64).unwrap_or(F::zero());

        let prior_scale_f64 = prior_scale.mapv(|x| x.to_f64().unwrap_or(0.0));
        let det_prior_scale_f64 = matrix_determinant(&prior_scale_f64);
        let det_prior_scale = F::from(det_prior_scale_f64).unwrap_or(F::zero());

        if det_var_scale <= F::zero() || det_prior_scale <= F::zero() {
            return Ok(F::neg_infinity());
        }

        // Expected log likelihood
        let e_log_det = F::from(n_features).unwrap() * F::from(2.0).unwrap().ln()
            + det_var_scale.ln()
            - var_df * det_var_scale.ln();

        let var_scale_f64 = var_scale.mapv(|x| x.to_f64().unwrap_or(0.0));
        let inv_var_scale_f64 = matrix_inverse(&var_scale_f64).unwrap();
        let inv_var_scale = inv_var_scale_f64.mapv(|x| F::from(x).unwrap_or(F::zero()));
        let trace_term = (sample_cov.dot(&inv_var_scale)).diag().sum();
        let expected_log_likelihood = F::from(n_samples).unwrap()
            * (e_log_det / F::from(2.0).unwrap() - trace_term / F::from(2.0).unwrap());

        // KL divergence between variational and prior
        let kl_div = (var_df - prior_df) / F::from(2.0).unwrap() * det_var_scale.ln()
            - (var_df / F::from(2.0).unwrap()) * det_var_scale.ln()
            + (prior_df / F::from(2.0).unwrap()) * det_prior_scale.ln()
            + ((var_scale - prior_scale).dot(&inv_var_scale)).diag().sum() / F::from(2.0).unwrap();

        Ok(expected_log_likelihood - kl_div)
    }
}

/// Builder for Bayesian covariance estimation
pub struct BayesianCovarianceBuilder<F: NdFloat + FromPrimitive> {
    config: BayesianCovarianceConfig<F>,
}

impl<F: NdFloat + FromPrimitive> BayesianCovarianceBuilder<F> {
    pub fn new() -> Self {
        Self {
            config: BayesianCovarianceConfig::default(),
        }
    }

    pub fn method(mut self, method: BayesianMethod) -> Self {
        self.config.method = method;
        self
    }

    pub fn prior_scale_matrix(mut self, scale_matrix: Array2<F>) -> Self {
        self.config.prior.scale_matrix = scale_matrix;
        self
    }

    pub fn prior_degrees_of_freedom(mut self, df: F) -> Self {
        self.config.prior.degrees_of_freedom = df;
        self
    }

    pub fn prior_mean(mut self, mean: Array1<F>) -> Self {
        self.config.prior.prior_mean = Some(mean);
        self
    }

    pub fn hyperparameters(mut self, params: Vec<F>) -> Self {
        self.config.prior.hyperparameters = params;
        self
    }

    pub fn mcmc_samples(mut self, n_samples: usize) -> Self {
        if let Some(ref mut mcmc_config) = self.config.mcmc_config {
            mcmc_config.n_samples = n_samples;
        }
        self
    }

    pub fn mcmc_burn_in(mut self, burn_in: usize) -> Self {
        if let Some(ref mut mcmc_config) = self.config.mcmc_config {
            mcmc_config.burn_in = burn_in;
        }
        self
    }

    pub fn mcmc_thin(mut self, thin: usize) -> Self {
        if let Some(ref mut mcmc_config) = self.config.mcmc_config {
            mcmc_config.thin = thin;
        }
        self
    }

    pub fn proposal_scale(mut self, scale: F) -> Self {
        if let Some(ref mut mcmc_config) = self.config.mcmc_config {
            mcmc_config.proposal_scale = scale;
        }
        self
    }

    pub fn variational_max_iter(mut self, max_iter: usize) -> Self {
        if let Some(ref mut var_config) = self.config.variational_config {
            var_config.max_iter = max_iter;
        }
        self
    }

    pub fn variational_tolerance(mut self, tol: F) -> Self {
        if let Some(ref mut var_config) = self.config.variational_config {
            var_config.tol = tol;
        }
        self
    }

    pub fn learning_rate(mut self, lr: F) -> Self {
        if let Some(ref mut var_config) = self.config.variational_config {
            var_config.learning_rate = lr;
        }
        self
    }

    pub fn structured_meanfield(mut self, structured: bool) -> Self {
        if let Some(ref mut var_config) = self.config.variational_config {
            var_config.structured_meanfield = structured;
        }
        self
    }

    pub fn regularization(mut self, reg: F) -> Self {
        self.config.regularization = reg;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    pub fn build(self) -> BayesianCovariance<F> {
        BayesianCovariance::new(self.config)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::ndarray_linalg::EigVals;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::Random;

    fn generate_test_data(n_samples: usize, n_features: usize) -> Array2<f64> {
        let mut rng = Random::seed(42);
        let mut data = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = StandardNormal.sample(&mut rng);
            }
        }

        data
    }

    #[test]
    fn test_bayesian_covariance_inverse_wishart() {
        let data = generate_test_data(50, 3);

        let estimator = BayesianCovariance::builder()
            .method(BayesianMethod::InverseWishart)
            .prior_degrees_of_freedom(5.0)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 3);
        assert_eq!(fitted.n_samples(), 50);
        assert_eq!(fitted.covariance().shape(), &[3, 3]);

        // Check that covariance matrix is positive definite
        let eigenvals = fitted.covariance().eigvals().unwrap();
        assert!(eigenvals.iter().all(|&x| x.re > 0.0 && x.im.abs() < 1e-10));
    }

    #[test]
    fn test_bayesian_covariance_variational() {
        let data = generate_test_data(30, 2);

        let estimator = BayesianCovariance::builder()
            .method(BayesianMethod::VariationalBayes)
            .variational_max_iter(50)
            .variational_tolerance(1e-6)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        assert!(fitted.variational_params().is_some());

        let var_params = fitted.variational_params().unwrap();
        assert!(var_params.lower_bound.is_finite());
        assert!(var_params.degrees_of_freedom > 0.0);
    }

    #[test]
    fn test_bayesian_covariance_mcmc_gibbs() {
        let data = generate_test_data(25, 2);

        let estimator = BayesianCovariance::builder()
            .method(BayesianMethod::McmcGibbs)
            .mcmc_samples(100)
            .mcmc_burn_in(20)
            .mcmc_thin(2)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        assert!(fitted.samples().is_some());

        let samples = fitted.samples().unwrap();
        assert_eq!(samples.len(), 100);
        assert_eq!(samples[0].shape(), &[2, 2]);
    }

    #[test]
    fn test_credible_intervals() {
        let data = generate_test_data(30, 2);

        let estimator = BayesianCovariance::builder()
            .method(BayesianMethod::McmcGibbs)
            .mcmc_samples(50)
            .mcmc_burn_in(10)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();
        let (lower, upper) = fitted.credible_intervals(0.05).unwrap();

        assert_eq!(lower.shape(), &[2, 2]);
        assert_eq!(upper.shape(), &[2, 2]);

        // Check that lower bounds are less than upper bounds
        for i in 0..2 {
            for j in 0..2 {
                assert!(lower[[i, j]] <= upper[[i, j]]);
            }
        }
    }

    #[test]
    fn test_predictive_covariance() {
        let data = generate_test_data(40, 3);
        let test_data = generate_test_data(10, 3);

        let estimator = BayesianCovariance::builder()
            .method(BayesianMethod::InverseWishart)
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();
        let pred_cov = fitted.predict_covariance(&test_data).unwrap();

        assert_eq!(pred_cov.shape(), &[3, 3]);

        // Check positive definiteness
        let eigenvals = pred_cov.eigvals().unwrap();
        assert!(eigenvals.iter().all(|&x| x.re > 0.0 && x.im.abs() < 1e-10));
    }

    #[test]
    fn test_hierarchical_bayesian() {
        let data = generate_test_data(35, 2);

        let estimator = BayesianCovariance::builder()
            .method(BayesianMethod::Hierarchical)
            .hyperparameters(vec![2.0, 1.5])
            .random_state(42)
            .build();

        let fitted = estimator.fit(&data, &()).unwrap();

        assert_eq!(fitted.n_features(), 2);
        assert!(fitted.log_likelihood().is_finite());
    }
}
