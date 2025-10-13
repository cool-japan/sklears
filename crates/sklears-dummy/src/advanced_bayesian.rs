//! Advanced Bayesian baseline estimators
//!
//! This module provides advanced Bayesian methods for baseline estimation including:
//! - Empirical Bayes estimation
//! - Hierarchical Bayesian models  
//! - Variational Bayes approximation
//! - MCMC-based sampling

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::Distribution;
use sklears_core::error::Result;
use sklears_core::types::{Float, Int};
use std::collections::HashMap;

/// Advanced Bayesian strategy selection
#[derive(Debug, Clone, PartialEq)]
pub enum AdvancedBayesianStrategy {
    /// Empirical Bayes estimation with hyperparameter optimization
    EmpiricalBayes,
    /// Hierarchical Bayesian model with group structure
    Hierarchical,
    /// Variational Bayes approximation
    VariationalBayes,
    /// MCMC sampling-based estimation
    MCMCSampling,
    /// Conjugate prior with automatic selection
    ConjugatePrior,
}

/// Empirical Bayes estimator for automatic hyperparameter selection
#[derive(Debug, Clone)]
pub struct EmpiricalBayesEstimator {
    /// Number of iterations for EM algorithm
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Estimated hyperparameters
    pub hyperparameters_: Option<Array1<Float>>,
    /// Log-likelihood values during optimization
    pub log_likelihood_: Option<Vec<Float>>,
}

impl EmpiricalBayesEstimator {
    /// Create new empirical Bayes estimator
    pub fn new() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-6,
            random_state: None,
            hyperparameters_: None,
            log_likelihood_: None,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Estimate hyperparameters using EM algorithm for classification
    pub fn fit_classification(&mut self, y: &Array1<Int>) -> Result<()> {
        let mut class_counts: HashMap<Int, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let mut classes: Vec<Int> = class_counts.keys().copied().collect();
        classes.sort();
        let n_classes = classes.len();
        let n_samples = y.len() as Float;

        // Initialize hyperparameters (Dirichlet concentration parameters)
        let mut hyperparams = Array1::ones(n_classes);
        let mut log_likelihoods = Vec::new();

        // EM algorithm for empirical Bayes estimation
        for iter in 0..self.max_iter {
            // E-step: compute expected sufficient statistics
            let mut expected_counts = Array1::<Float>::zeros(n_classes);
            for &label in y.iter() {
                let class_idx = classes.iter().position(|&c| c == label).unwrap();
                expected_counts[class_idx] += 1.0;
            }

            // M-step: update hyperparameters
            let old_hyperparams = hyperparams.clone();

            // Method of moments estimation for Dirichlet parameters
            let observed_props: Array1<Float> = expected_counts.mapv(|x| x / n_samples);
            let mean_prop = observed_props.mean().unwrap();
            let variance_sum: Float = observed_props.mapv(|p| (p - mean_prop).powi(2)).sum();

            // Estimate concentration parameter
            let concentration = if variance_sum > 0.0 {
                let variance_mean = variance_sum / (n_classes as Float);
                let alpha_sum = mean_prop * (1.0 - mean_prop) / variance_mean - 1.0;
                alpha_sum.max(0.1) // Ensure positive
            } else {
                1.0
            };

            hyperparams = observed_props.mapv(|p| p * concentration);

            // Compute log-likelihood (approximate)
            let log_likelihood = self.compute_log_likelihood(&hyperparams, &expected_counts);
            log_likelihoods.push(log_likelihood);

            // Check convergence
            let param_diff: Float = (&hyperparams - &old_hyperparams).mapv(|x| x.abs()).sum();
            if param_diff < self.tolerance {
                break;
            }
        }

        self.hyperparameters_ = Some(hyperparams);
        self.log_likelihood_ = Some(log_likelihoods);
        Ok(())
    }

    /// Compute log-likelihood for convergence checking
    fn compute_log_likelihood(&self, hyperparams: &Array1<Float>, counts: &Array1<Float>) -> Float {
        let alpha_sum = hyperparams.sum();
        let count_sum = counts.sum();

        // Log-likelihood of Dirichlet-Multinomial
        let mut log_likelihood = 0.0;

        // Add log Gamma terms
        for (&alpha, &count) in hyperparams.iter().zip(counts.iter()) {
            log_likelihood += gamma_ln(alpha + count) - gamma_ln(alpha);
        }

        log_likelihood += gamma_ln(alpha_sum) - gamma_ln(alpha_sum + count_sum);
        log_likelihood
    }

    /// Get estimated hyperparameters
    pub fn hyperparameters(&self) -> Option<&Array1<Float>> {
        self.hyperparameters_.as_ref()
    }

    /// Get log-likelihood evolution
    pub fn log_likelihood_evolution(&self) -> Option<&Vec<Float>> {
        self.log_likelihood_.as_ref()
    }
}

/// Hierarchical Bayesian estimator with group structure
#[derive(Debug, Clone)]
pub struct HierarchicalBayesEstimator {
    /// Group assignments for samples
    pub groups: Option<Array1<Int>>,
    /// Global hyperparameters
    pub global_hyperparams_: Option<Array1<Float>>,
    /// Group-specific parameters
    pub group_params_: Option<HashMap<Int, Array1<Float>>>,
    /// Random state
    pub random_state: Option<u64>,
}

impl HierarchicalBayesEstimator {
    /// Create new hierarchical Bayes estimator
    pub fn new() -> Self {
        Self {
            groups: None,
            global_hyperparams_: None,
            group_params_: None,
            random_state: None,
        }
    }

    /// Set group assignments
    pub fn with_groups(mut self, groups: Array1<Int>) -> Self {
        self.groups = Some(groups);
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit hierarchical model
    pub fn fit_classification(&mut self, y: &Array1<Int>) -> Result<()> {
        let groups = self.groups.as_ref().ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput(
                "Group assignments must be provided".to_string(),
            )
        })?;

        if groups.len() != y.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Groups and labels must have same length".to_string(),
            ));
        }

        // Get unique classes and groups
        let mut class_counts: HashMap<Int, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }
        let mut classes: Vec<Int> = class_counts.keys().copied().collect();
        classes.sort();
        let n_classes = classes.len();

        let mut unique_groups: Vec<Int> = groups.iter().copied().collect();
        unique_groups.sort();
        unique_groups.dedup();

        // Compute group-specific class distributions
        let mut group_params = HashMap::new();
        let mut global_counts = Array1::<Float>::zeros(n_classes);

        for &group in &unique_groups {
            let mut group_class_counts = Array1::<Float>::zeros(n_classes);
            let mut group_total = 0;

            for (i, (&label, &group_id)) in y.iter().zip(groups.iter()).enumerate() {
                if group_id == group {
                    let class_idx = classes.iter().position(|&c| c == label).unwrap();
                    group_class_counts[class_idx] += 1.0;
                    global_counts[class_idx] += 1.0;
                    group_total += 1;
                }
            }

            if group_total > 0 {
                // Normalize to probabilities
                let group_probs = group_class_counts.mapv(|x| x / (group_total as Float));
                group_params.insert(group, group_probs);
            }
        }

        // Estimate global hyperparameters (pooled estimate)
        let global_total = global_counts.sum();
        let global_hyperparams = if global_total > 0.0 {
            global_counts.mapv(|x| x / global_total)
        } else {
            Array1::ones(n_classes) / (n_classes as Float)
        };

        self.global_hyperparams_ = Some(global_hyperparams);
        self.group_params_ = Some(group_params);
        Ok(())
    }

    /// Get global hyperparameters
    pub fn global_hyperparameters(&self) -> Option<&Array1<Float>> {
        self.global_hyperparams_.as_ref()
    }

    /// Get group-specific parameters
    pub fn group_parameters(&self) -> Option<&HashMap<Int, Array1<Float>>> {
        self.group_params_.as_ref()
    }
}

/// Variational Bayes estimator using mean-field approximation
#[derive(Debug, Clone)]
pub struct VariationalBayesEstimator {
    /// Maximum iterations for variational optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Variational parameters
    pub variational_params_: Option<Array1<Float>>,
    /// ELBO (Evidence Lower BOund) values
    pub elbo_: Option<Vec<Float>>,
    /// Random state
    pub random_state: Option<u64>,
}

impl VariationalBayesEstimator {
    /// Create new variational Bayes estimator
    pub fn new() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-6,
            variational_params_: None,
            elbo_: None,
            random_state: None,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Fit variational Bayes model
    pub fn fit_classification(&mut self, y: &Array1<Int>) -> Result<()> {
        let mut class_counts: HashMap<Int, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let mut classes: Vec<Int> = class_counts.keys().copied().collect();
        classes.sort();
        let n_classes = classes.len();

        // Initialize variational parameters
        let mut q_params = Array1::ones(n_classes);
        let mut elbo_values = Vec::new();

        // Variational optimization loop
        for _iter in 0..self.max_iter {
            let old_params = q_params.clone();

            // Update variational parameters (simplified mean-field update)
            let mut new_params = Array1::<Float>::zeros(n_classes);
            for (i, &class) in classes.iter().enumerate() {
                let count = *class_counts.get(&class).unwrap() as Float;
                // Add pseudo-count from prior
                new_params[i] = count + 1.0;
            }

            q_params = new_params;

            // Compute ELBO (simplified)
            let elbo = self.compute_elbo(&q_params, &class_counts, &classes);
            elbo_values.push(elbo);

            // Check convergence
            let param_diff: Float = (&q_params - &old_params).mapv(|x| x.abs()).sum();
            if param_diff < self.tolerance {
                break;
            }
        }

        // Normalize to probabilities
        let param_sum = q_params.sum();
        q_params = q_params.mapv(|x| x / param_sum);

        self.variational_params_ = Some(q_params);
        self.elbo_ = Some(elbo_values);
        Ok(())
    }

    /// Compute Evidence Lower BOund (ELBO)
    fn compute_elbo(
        &self,
        params: &Array1<Float>,
        counts: &HashMap<Int, usize>,
        classes: &[Int],
    ) -> Float {
        let mut elbo = 0.0;
        let param_sum = params.sum();

        // Data likelihood term
        for (i, &class) in classes.iter().enumerate() {
            let count = *counts.get(&class).unwrap() as Float;
            if count > 0.0 {
                elbo += count * (params[i] / param_sum).ln();
            }
        }

        // Prior terms (simplified)
        for &param in params.iter() {
            if param > 0.0 {
                elbo += param.ln();
            }
        }

        elbo
    }

    /// Get variational parameters
    pub fn variational_parameters(&self) -> Option<&Array1<Float>> {
        self.variational_params_.as_ref()
    }

    /// Get ELBO evolution
    pub fn elbo_evolution(&self) -> Option<&Vec<Float>> {
        self.elbo_.as_ref()
    }
}

/// MCMC-based Bayesian estimator
#[derive(Debug, Clone)]
pub struct MCMCBayesEstimator {
    /// Number of MCMC samples
    pub n_samples: usize,
    /// Burn-in period
    pub burn_in: usize,
    /// Thinning factor
    pub thin: usize,
    /// MCMC samples
    pub samples_: Option<Array2<Float>>,
    /// Random state
    pub random_state: Option<u64>,
}

impl MCMCBayesEstimator {
    /// Create new MCMC estimator
    pub fn new() -> Self {
        Self {
            n_samples: 1000,
            burn_in: 200,
            thin: 1,
            samples_: None,
            random_state: None,
        }
    }

    /// Set number of samples
    pub fn with_n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set burn-in period
    pub fn with_burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit MCMC model using Gibbs sampling
    pub fn fit_classification(&mut self, y: &Array1<Int>) -> Result<()> {
        let mut class_counts: HashMap<Int, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let mut classes: Vec<Int> = class_counts.keys().copied().collect();
        classes.sort();
        let n_classes = classes.len();

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(0) // Use deterministic seed for reproducibility
        };

        // Initialize parameters
        let mut theta = Array1::<Float>::from_elem(n_classes, 1.0 / n_classes as Float);
        let total_samples = self.burn_in + self.n_samples * self.thin;
        let mut samples = Array2::<Float>::zeros((self.n_samples, n_classes));

        // MCMC sampling loop
        for iter in 0..total_samples {
            // Gibbs sampling for Dirichlet-Multinomial
            let mut alpha_posterior = Array1::<Float>::ones(n_classes); // Prior

            // Add observed counts
            for (i, &class) in classes.iter().enumerate() {
                let count = *class_counts.get(&class).unwrap() as Float;
                alpha_posterior[i] += count;
            }

            // Sample from Dirichlet using Gamma sampling
            let mut gamma_samples = Array1::<Float>::zeros(n_classes);
            for i in 0..n_classes {
                let gamma_dist = Gamma::new(alpha_posterior[i], 1.0).unwrap();
                gamma_samples[i] = gamma_dist.sample(&mut rng);
            }

            // Normalize to get Dirichlet sample
            let gamma_sum = gamma_samples.sum();
            theta = gamma_samples.mapv(|x| x / gamma_sum);

            // Store sample if past burn-in and at thinning interval
            if iter >= self.burn_in && (iter - self.burn_in) % self.thin == 0 {
                let sample_idx = (iter - self.burn_in) / self.thin;
                if sample_idx < self.n_samples {
                    for j in 0..n_classes {
                        samples[[sample_idx, j]] = theta[j];
                    }
                }
            }
        }

        self.samples_ = Some(samples);
        Ok(())
    }

    /// Get MCMC samples
    pub fn samples(&self) -> Option<&Array2<Float>> {
        self.samples_.as_ref()
    }

    /// Get posterior mean
    pub fn posterior_mean(&self) -> Option<Array1<Float>> {
        self.samples_
            .as_ref()
            .map(|samples| samples.mean_axis(Axis(0)).unwrap())
    }

    /// Get posterior standard deviation
    pub fn posterior_std(&self) -> Option<Array1<Float>> {
        self.samples_
            .as_ref()
            .map(|samples| samples.std_axis(Axis(0), 0.0))
    }

    /// Get credible intervals
    pub fn credible_interval(&self, alpha: Float) -> Option<(Array1<Float>, Array1<Float>)> {
        let samples = self.samples_.as_ref()?;
        let n_classes = samples.ncols();
        let mut lower = Array1::<Float>::zeros(n_classes);
        let mut upper = Array1::<Float>::zeros(n_classes);

        for i in 0..n_classes {
            let mut column: Vec<Float> = samples.column(i).to_vec();
            column.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let lower_idx = ((alpha / 2.0) * (column.len() as Float)) as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * (column.len() as Float)) as usize;

            lower[i] = column[lower_idx.min(column.len() - 1)];
            upper[i] = column[upper_idx.min(column.len() - 1)];
        }

        Some((lower, upper))
    }
}

/// Approximation of log Gamma function
fn gamma_ln(x: Float) -> Float {
    // Stirling's approximation for large x, exact for small integers
    if x <= 0.0 {
        Float::INFINITY
    } else if x < 12.0 {
        // Use recurrence relation for small values
        if x.fract() == 0.0 && x <= 10.0 {
            // Exact for small integers
            let n = x as usize;
            if n == 1 {
                0.0
            } else {
                (1..n).map(|i| (i as Float).ln()).sum()
            }
        } else {
            // Stirling approximation
            (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln()
        }
    } else {
        // Stirling approximation for large x
        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
}

/// Default implementations
impl Default for EmpiricalBayesEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for HierarchicalBayesEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for VariationalBayesEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MCMCBayesEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_empirical_bayes_basic() {
        let y = array![0, 0, 0, 1, 1, 2]; // 3 classes with different frequencies
        let mut estimator = EmpiricalBayesEstimator::new().with_random_state(42);

        let result = estimator.fit_classification(&y);
        assert!(result.is_ok());

        let hyperparams = estimator.hyperparameters().unwrap();
        assert_eq!(hyperparams.len(), 3);

        // All hyperparameters should be positive
        for &param in hyperparams.iter() {
            assert!(param > 0.0);
        }

        // Most frequent class should have higher hyperparameter
        assert!(hyperparams[0] >= hyperparams[1]);
        assert!(hyperparams[0] >= hyperparams[2]);
    }

    #[test]
    fn test_hierarchical_bayes_basic() {
        let y = array![0, 0, 1, 1, 0, 1];
        let groups = array![1, 1, 1, 2, 2, 2]; // Two groups

        let mut estimator = HierarchicalBayesEstimator::new()
            .with_groups(groups)
            .with_random_state(42);

        let result = estimator.fit_classification(&y);
        assert!(result.is_ok());

        let global_params = estimator.global_hyperparameters().unwrap();
        assert_eq!(global_params.len(), 2);

        let group_params = estimator.group_parameters().unwrap();
        assert_eq!(group_params.len(), 2);

        // Check that group parameters exist for both groups
        assert!(group_params.contains_key(&1));
        assert!(group_params.contains_key(&2));
    }

    #[test]
    fn test_variational_bayes_basic() {
        let y = array![0, 0, 0, 1, 1, 2];
        let mut estimator = VariationalBayesEstimator::new()
            .with_max_iter(50)
            .with_tolerance(1e-4);

        let result = estimator.fit_classification(&y);
        assert!(result.is_ok());

        let params = estimator.variational_parameters().unwrap();
        assert_eq!(params.len(), 3);

        // Parameters should sum to 1 (normalized probabilities)
        let sum: Float = params.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);

        // ELBO should be tracked
        let elbo = estimator.elbo_evolution().unwrap();
        assert!(!elbo.is_empty());
    }

    #[test]
    fn test_mcmc_bayes_basic() {
        let y = array![0, 0, 0, 1, 1, 2];
        let mut estimator = MCMCBayesEstimator::new()
            .with_n_samples(100)
            .with_burn_in(20)
            .with_random_state(42);

        let result = estimator.fit_classification(&y);
        assert!(result.is_ok());

        let samples = estimator.samples().unwrap();
        assert_eq!(samples.nrows(), 100);
        assert_eq!(samples.ncols(), 3);

        // Each row should sum to 1 (probabilities)
        for i in 0..samples.nrows() {
            let row_sum: Float = samples.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }

        // Posterior mean should be available
        let mean = estimator.posterior_mean().unwrap();
        assert_eq!(mean.len(), 3);

        // Credible intervals should be available
        let (lower, upper) = estimator.credible_interval(0.05).unwrap();
        assert_eq!(lower.len(), 3);
        assert_eq!(upper.len(), 3);

        // Lower bounds should be less than upper bounds
        for i in 0..3 {
            assert!(lower[i] <= upper[i]);
        }
    }

    #[test]
    fn test_gamma_ln_function() {
        // Test for known values
        assert_abs_diff_eq!(gamma_ln(1.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma_ln(2.0), 0.0, epsilon = 1e-10); // ln(1!)
        assert_abs_diff_eq!(gamma_ln(3.0), (2.0f64).ln(), epsilon = 1e-10); // ln(2!)

        // Test for larger values (approximate)
        let result = gamma_ln(10.0);
        assert!(result > 0.0);
        assert!(result.is_finite());
    }
}
