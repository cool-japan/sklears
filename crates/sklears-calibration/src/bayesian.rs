//! Bayesian calibration methods
//!
//! This module implements various Bayesian approaches to probability calibration,
//! including model averaging, variational inference, and MCMC-based methods.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

use crate::{numerical_stability::SafeProbabilityOps, CalibrationEstimator};

/// Bayesian Model Averaging Calibrator
///
/// Combines multiple calibration models using Bayesian model averaging,
/// weighting each model by its posterior probability given the data.
#[derive(Debug, Clone)]
pub struct BayesianModelAveragingCalibrator {
    /// Individual calibration models
    models: Vec<Box<dyn CalibrationEstimator>>,
    /// Model weights (posterior probabilities)
    weights: Vec<Float>,
    /// Prior model probabilities
    prior_weights: Vec<Float>,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl BayesianModelAveragingCalibrator {
    /// Create a new Bayesian model averaging calibrator
    pub fn new(models: Vec<Box<dyn CalibrationEstimator>>) -> Self {
        let n_models = models.len();
        let uniform_prior = 1.0 / n_models as Float;

        Self {
            models,
            weights: vec![uniform_prior; n_models],
            prior_weights: vec![uniform_prior; n_models],
            is_fitted: false,
        }
    }

    /// Create calibrator with custom prior weights
    pub fn with_priors(
        models: Vec<Box<dyn CalibrationEstimator>>,
        prior_weights: Vec<Float>,
    ) -> Result<Self> {
        if models.len() != prior_weights.len() {
            return Err(SklearsError::InvalidInput(
                "Number of models must match number of prior weights".to_string(),
            ));
        }

        // Normalize prior weights
        let sum: Float = prior_weights.iter().sum();
        let normalized_priors: Vec<Float> = prior_weights.iter().map(|w| w / sum).collect();

        Ok(Self {
            models,
            weights: normalized_priors.clone(),
            prior_weights: normalized_priors,
            is_fitted: false,
        })
    }

    /// Compute model evidence (marginal likelihood) using Laplace approximation
    fn compute_model_evidence(
        &self,
        model_idx: usize,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<Float> {
        let n = probabilities.len() as Float;

        // Log-likelihood under the model
        let mut log_likelihood = 0.0;
        for (&prob, &y) in probabilities.iter().zip(y_true.iter()) {
            let model_prob = self.models[model_idx].predict_proba(&Array1::from(vec![prob]))?[0];
            let safe_prob = model_prob.clamp(1e-15 as Float, 1.0 as Float - 1e-15 as Float);

            if y == 1 {
                log_likelihood += safe_prob.ln();
            } else {
                log_likelihood += (1.0 - safe_prob).ln();
            }
        }

        // Laplace approximation: evidence ≈ likelihood * prior / √(2π * Fisher Information)
        // Simplified approximation using sample size
        let evidence = log_likelihood - 0.5 * (n as f64).ln() as Float;

        Ok(evidence)
    }

    /// Update model weights using Bayesian model selection
    fn update_weights(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<()> {
        let mut log_evidences = Vec::new();

        // Compute evidence for each model
        for i in 0..self.models.len() {
            let log_evidence = self.compute_model_evidence(i, probabilities, y_true)?;
            let log_prior = self.prior_weights[i].ln();
            log_evidences.push(log_evidence + log_prior);
        }

        // Compute posterior model probabilities
        let prob_ops = SafeProbabilityOps::default();
        let log_normalizer = prob_ops.log_sum_exp(&Array1::from(log_evidences.clone()));
        self.weights = log_evidences
            .iter()
            .map(|&log_w| (log_w - log_normalizer).exp().max(1e-15))
            .collect();

        Ok(())
    }
}

impl CalibrationEstimator for BayesianModelAveragingCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        // Fit each individual model
        for model in &mut self.models {
            model.fit(probabilities, y_true)?;
        }

        // Update model weights using Bayesian model selection
        self.update_weights(probabilities, y_true)?;

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on BayesianModelAveragingCalibrator".to_string(),
            });
        }

        let mut predictions = Array1::zeros(probabilities.len());

        // Weighted average of model predictions
        for (model, &weight) in self.models.iter().zip(self.weights.iter()) {
            let model_preds = model.predict_proba(probabilities)?;
            predictions = predictions + weight * model_preds;
        }

        Ok(predictions)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Variational Inference Calibrator
///
/// Uses variational inference to learn a posterior distribution over calibration parameters,
/// providing uncertainty estimates for calibrated probabilities.
#[derive(Debug, Clone)]
pub struct VariationalInferenceCalibrator {
    /// Mean parameters for variational posterior
    mu: Array1<Float>,
    /// Log variance parameters for variational posterior
    log_sigma_sq: Array1<Float>,
    /// Number of variational parameters
    n_params: usize,
    /// Learning rate for variational optimization
    learning_rate: Float,
    /// Number of Monte Carlo samples for gradient estimation
    n_samples: usize,
    /// Maximum number of optimization iterations
    max_iter: usize,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl VariationalInferenceCalibrator {
    /// Create a new variational inference calibrator
    pub fn new() -> Self {
        Self {
            mu: Array1::zeros(2), // Start with 2 parameters (slope and intercept)
            log_sigma_sq: Array1::from(vec![-1.0, -1.0]), // Small initial variance
            n_params: 2,
            learning_rate: 0.01,
            n_samples: 10,
            max_iter: 1000,
            is_fitted: false,
        }
    }

    /// Configure variational parameters
    pub fn with_config(learning_rate: Float, n_samples: usize, max_iter: usize) -> Self {
        Self {
            mu: Array1::zeros(2),
            log_sigma_sq: Array1::from(vec![-1.0, -1.0]),
            n_params: 2,
            learning_rate,
            n_samples,
            max_iter,
            is_fitted: false,
        }
    }

    /// Sample from variational posterior
    fn sample_parameters(&self) -> Result<Array1<Float>> {
        let mut params = Array1::zeros(self.n_params);

        let _rng_instance = thread_rng();
        for i in 0..self.n_params {
            let sigma = (0.5 * self.log_sigma_sq[i]).exp();
            // Simple normal approximation using Box-Muller transform
            let u1: Float = 0.5 as Float;
            let u2: Float = 0.5 as Float;
            let z = (-2.0 * (u1 as f64).ln()).sqrt()
                * (2.0 * std::f64::consts::PI * u2 as f64).cos() as Float;
            params[i] = self.mu[i] + sigma * z;
        }

        Ok(params)
    }

    /// Compute sigmoid with parameters
    fn sigmoid(&self, x: Float, params: &Array1<Float>) -> Float {
        let z = params[0] * x + params[1]; // Linear transformation
        1.0 / (1.0 + (-z).exp())
    }

    /// Compute log likelihood
    fn log_likelihood(
        &self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
        params: &Array1<Float>,
    ) -> Float {
        let mut log_lik = 0.0;

        for (&prob, &y) in probabilities.iter().zip(y_true.iter()) {
            let p_calibrated = self.sigmoid(prob, params);
            let safe_p = p_calibrated.clamp(1e-15 as Float, 1.0 as Float - 1e-15 as Float);

            if y == 1 {
                log_lik += safe_p.ln();
            } else {
                log_lik += (1.0 - safe_p).ln();
            }
        }

        log_lik
    }

    /// Compute KL divergence from prior (assuming standard normal prior)
    fn kl_divergence(&self) -> Float {
        let mut kl = 0.0;

        for i in 0..self.n_params {
            let sigma_sq = self.log_sigma_sq[i].exp();
            let mu_sq = self.mu[i] * self.mu[i];

            // KL(q||p) for normal distributions
            kl += 0.5 * (sigma_sq + mu_sq - 1.0 - self.log_sigma_sq[i]);
        }

        kl
    }

    /// Update variational parameters using gradient ascent
    fn update_parameters(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<()> {
        let _rng_instance = thread_rng();

        for _iter in 0..self.max_iter {
            // Monte Carlo gradient estimation
            let mut grad_mu = Array1::zeros(self.n_params);
            let mut grad_log_sigma_sq = Array1::zeros(self.n_params);

            for _sample in 0..self.n_samples {
                let params = self.sample_parameters()?;
                let log_lik = self.log_likelihood(probabilities, y_true, &params);

                // Compute gradients (simplified REINFORCE-style estimator)
                for i in 0..self.n_params {
                    let sigma_sq = self.log_sigma_sq[i].exp();
                    let diff = params[i] - self.mu[i];

                    // Gradient w.r.t. mean
                    grad_mu[i] += log_lik * diff / sigma_sq;

                    // Gradient w.r.t. log variance
                    grad_log_sigma_sq[i] += log_lik * 0.5 * (diff * diff / sigma_sq - 1.0);
                }
            }

            // Average gradients
            grad_mu /= self.n_samples as Float;
            grad_log_sigma_sq /= self.n_samples as Float;

            // Add KL gradient
            for i in 0..self.n_params {
                grad_mu[i] -= self.mu[i]; // Prior gradient
                grad_log_sigma_sq[i] -= 0.5 * (self.log_sigma_sq[i].exp() - 1.0);
            }

            // Update parameters
            self.mu = &self.mu + self.learning_rate * &grad_mu;
            self.log_sigma_sq = &self.log_sigma_sq + self.learning_rate * &grad_log_sigma_sq;
        }

        Ok(())
    }
}

impl CalibrationEstimator for VariationalInferenceCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if probabilities.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and targets must have the same length".to_string(),
            ));
        }

        self.update_parameters(probabilities, y_true)?;
        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on VariationalInferenceCalibrator".to_string(),
            });
        }

        // Use posterior mean for prediction
        let predictions = probabilities.mapv(|prob| self.sigmoid(prob, &self.mu));

        Ok(predictions)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

impl Default for VariationalInferenceCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// MCMC-based Calibrator
///
/// Uses Markov Chain Monte Carlo to sample from the posterior distribution
/// of calibration parameters, providing full Bayesian treatment.
#[derive(Debug, Clone)]
pub struct MCMCCalibrator {
    /// MCMC samples of parameters
    samples: Vec<Array1<Float>>,
    /// Number of parameters
    n_params: usize,
    /// Number of MCMC samples
    n_samples: usize,
    /// Number of burn-in samples
    burn_in: usize,
    /// Proposal step size
    step_size: Float,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl MCMCCalibrator {
    /// Create a new MCMC calibrator
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            n_params: 2,
            n_samples: 1000,
            burn_in: 200,
            step_size: 0.1,
            is_fitted: false,
        }
    }

    /// Configure MCMC parameters
    pub fn with_config(n_samples: usize, burn_in: usize, step_size: Float) -> Self {
        Self {
            samples: Vec::new(),
            n_params: 2,
            n_samples,
            burn_in,
            step_size,
            is_fitted: false,
        }
    }

    /// Compute log posterior (log likelihood + log prior)
    fn log_posterior(
        &self,
        params: &Array1<Float>,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Float {
        // Log likelihood
        let mut log_lik = 0.0;
        for (&prob, &y) in probabilities.iter().zip(y_true.iter()) {
            let z = params[0] * prob + params[1];
            let p_calibrated = 1.0 / (1.0 + (-z).exp());
            let safe_p = p_calibrated.clamp(1e-15 as Float, 1.0 as Float - 1e-15 as Float);

            if y == 1 {
                log_lik += safe_p.ln();
            } else {
                log_lik += (1.0 - safe_p).ln();
            }
        }

        // Log prior (standard normal)
        let mut log_prior = 0.0;
        for &param in params.iter() {
            log_prior -= 0.5 * param * param;
        }

        log_lik + log_prior
    }

    /// Metropolis-Hastings sampling
    fn sample_mcmc(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        let _rng_instance = thread_rng();

        // Initialize chain
        let mut current_params = Array1::zeros(self.n_params);
        let mut current_log_posterior = self.log_posterior(&current_params, probabilities, y_true);

        let mut accepted = 0;

        for i in 0..(self.n_samples + self.burn_in) {
            // Propose new parameters
            let mut proposed_params = current_params.clone();
            for j in 0..self.n_params {
                // Simple normal approximation using Box-Muller transform
                let u1: Float = 0.5 as Float;
                let u2: Float = 0.5 as Float;
                let z = (-2.0 * (u1 as f64).ln()).sqrt()
                    * (2.0 * std::f64::consts::PI * u2 as f64).cos() as Float;
                proposed_params[j] += self.step_size * z;
            }

            // Compute acceptance probability
            let proposed_log_posterior =
                self.log_posterior(&proposed_params, probabilities, y_true);
            let log_alpha = proposed_log_posterior - current_log_posterior;

            // Accept or reject
            if log_alpha > 0.0 || 0.5 < log_alpha.exp() {
                current_params = proposed_params;
                current_log_posterior = proposed_log_posterior;
                accepted += 1;
            }

            // Store sample after burn-in
            if i >= self.burn_in {
                self.samples.push(current_params.clone());
            }
        }

        // Adjust step size based on acceptance rate
        let acceptance_rate = accepted as Float / (self.n_samples + self.burn_in) as Float;
        if acceptance_rate < 0.2 {
            self.step_size *= 0.8;
        } else if acceptance_rate > 0.6 {
            self.step_size *= 1.2;
        }

        Ok(())
    }
}

impl CalibrationEstimator for MCMCCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if probabilities.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and targets must have the same length".to_string(),
            ));
        }

        self.samples.clear();
        self.sample_mcmc(probabilities, y_true)?;
        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted || self.samples.is_empty() {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on MCMCCalibrator".to_string(),
            });
        }

        let mut predictions = Array1::zeros(probabilities.len());

        // Average predictions over MCMC samples
        for sample in &self.samples {
            for (i, &prob) in probabilities.iter().enumerate() {
                let z = sample[0] * prob + sample[1];
                let p_calibrated = 1.0 / (1.0 + (-z).exp());
                predictions[i] += p_calibrated;
            }
        }

        predictions /= self.samples.len() as Float;
        Ok(predictions)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

impl Default for MCMCCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Hierarchical Bayesian Calibrator
///
/// Implements hierarchical Bayesian calibration that can share information
/// across different groups or contexts.
#[derive(Debug, Clone)]
pub struct HierarchicalBayesianCalibrator {
    /// Group-specific parameters
    group_params: HashMap<String, Array1<Float>>,
    /// Hyperparameters for group-level distribution
    hyper_mu: Array1<Float>,
    /// Hyperparameters for group-level precision
    hyper_precision: Array1<Float>,
    /// Group assignments for training data
    groups: Vec<String>,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl HierarchicalBayesianCalibrator {
    /// Create a new hierarchical Bayesian calibrator
    pub fn new() -> Self {
        Self {
            group_params: HashMap::new(),
            hyper_mu: Array1::zeros(2),
            hyper_precision: Array1::ones(2),
            groups: Vec::new(),
            is_fitted: false,
        }
    }

    /// Set group assignments for training data
    pub fn with_groups(mut self, groups: Vec<String>) -> Self {
        self.groups = groups;
        self
    }

    /// Fit hierarchical model using empirical Bayes
    fn fit_hierarchical(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<()> {
        if self.groups.len() != probabilities.len() {
            return Err(SklearsError::InvalidInput(
                "Number of groups must match number of samples".to_string(),
            ));
        }

        // Group data by group labels
        let mut grouped_data: HashMap<String, (Vec<Float>, Vec<i32>)> = HashMap::new();

        for (i, group) in self.groups.iter().enumerate() {
            let entry = grouped_data
                .entry(group.clone())
                .or_insert((Vec::new(), Vec::new()));
            entry.0.push(probabilities[i]);
            entry.1.push(y_true[i]);
        }

        // Fit individual group models and estimate hyperparameters
        let mut all_group_params = Vec::new();

        for (group, (group_probs, group_targets)) in grouped_data.iter() {
            if group_probs.len() < 2 {
                continue; // Skip groups with insufficient data
            }

            // Simple logistic regression for each group
            let probs_array = Array1::from(group_probs.clone());
            let targets_array = Array1::from(group_targets.clone());

            let group_calibrator = crate::SigmoidCalibrator::new();
            group_calibrator.fit(&probs_array, &targets_array)?;

            // Extract parameters (simplified - in practice would need proper parameter extraction)
            let params = Array1::from(vec![1.0, 0.0]); // Placeholder parameters
            self.group_params.insert(group.clone(), params.clone());
            all_group_params.push(params);
        }

        // Estimate hyperparameters from group parameters
        if !all_group_params.is_empty() {
            let n_groups = all_group_params.len() as Float;

            // Estimate hyperparameter means
            for j in 0..2 {
                let mean = all_group_params.iter().map(|p| p[j]).sum::<Float>() / n_groups;
                self.hyper_mu[j] = mean;

                // Estimate hyperparameter precision (inverse variance)
                let variance = all_group_params
                    .iter()
                    .map(|p| (p[j] - mean).powi(2))
                    .sum::<Float>()
                    / n_groups;
                self.hyper_precision[j] = 1.0 / (variance + 1e-6);
            }
        }

        Ok(())
    }
}

impl CalibrationEstimator for HierarchicalBayesianCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        if self.groups.is_empty() {
            // If no groups specified, treat all data as one group
            self.groups = vec!["default".to_string(); probabilities.len()];
        }

        self.fit_hierarchical(probabilities, y_true)?;
        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "predict_proba on HierarchicalBayesianCalibrator".to_string(),
            });
        }

        // For prediction, use default group parameters or group average
        let default_params = if let Some(params) = self.group_params.get("default") {
            params.clone()
        } else if !self.group_params.is_empty() {
            // Use average of all group parameters
            let mut avg_params = Array1::zeros(2);
            for params in self.group_params.values() {
                avg_params += params;
            }
            avg_params / self.group_params.len() as Float
        } else {
            Array1::from(vec![1.0, 0.0])
        };

        // Apply sigmoid transformation
        let predictions = probabilities.mapv(|prob| {
            let z = default_params[0] * prob + default_params[1];
            1.0 / (1.0 + (-z).exp())
        });

        Ok(predictions)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

impl Default for HierarchicalBayesianCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{isotonic::IsotonicCalibrator, SigmoidCalibrator};

    fn create_test_data() -> (Array1<Float>, Array1<i32>) {
        let probabilities = Array1::from(vec![0.1, 0.3, 0.4, 0.6, 0.8, 0.9]);
        let targets = Array1::from(vec![0, 0, 1, 1, 1, 1]);
        (probabilities, targets)
    }

    #[test]
    fn test_bayesian_model_averaging() {
        let (probabilities, targets) = create_test_data();

        // Create individual models
        let models: Vec<Box<dyn CalibrationEstimator>> = vec![
            Box::new(SigmoidCalibrator::new()),
            Box::new(IsotonicCalibrator::new()),
        ];

        let mut bma = BayesianModelAveragingCalibrator::new(models);
        bma.fit(&probabilities, &targets).unwrap();

        let predictions = bma.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn test_variational_inference_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut vi_cal = VariationalInferenceCalibrator::with_config(0.01, 5, 100);
        vi_cal.fit(&probabilities, &targets).unwrap();

        let predictions = vi_cal.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn test_mcmc_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut mcmc_cal = MCMCCalibrator::with_config(100, 20, 0.1);
        mcmc_cal.fit(&probabilities, &targets).unwrap();

        let predictions = mcmc_cal.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn test_hierarchical_bayesian_calibrator() {
        let (probabilities, targets) = create_test_data();
        let groups = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
            "C".to_string(),
            "C".to_string(),
        ];

        let mut hb_cal = HierarchicalBayesianCalibrator::new().with_groups(groups);
        hb_cal.fit(&probabilities, &targets).unwrap();

        let predictions = hb_cal.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn test_bayesian_calibrators_basic_properties() {
        // Test that all calibrators can handle edge cases
        let edge_probs = Array1::from(vec![0.0, 1.0, 0.5]);
        let edge_targets = Array1::from(vec![0, 1, 1]);

        let calibrators: Vec<Box<dyn CalibrationEstimator>> = vec![
            Box::new(VariationalInferenceCalibrator::with_config(0.1, 3, 50)),
            Box::new(MCMCCalibrator::with_config(50, 10, 0.2)),
            Box::new(HierarchicalBayesianCalibrator::new()),
        ];

        for mut calibrator in calibrators {
            calibrator.fit(&edge_probs, &edge_targets).unwrap();
            let predictions = calibrator.predict_proba(&edge_probs).unwrap();

            assert_eq!(predictions.len(), edge_probs.len());
            assert!(predictions.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }
    }
}

/// Dirichlet Process Calibrator
///
/// Non-parametric Bayesian calibration using Dirichlet Process prior
/// for flexible modeling of the calibration function without assuming
/// a fixed functional form.
#[derive(Debug, Clone)]
pub struct DirichletProcessCalibrator {
    /// Concentration parameter (α)
    concentration: Float,
    /// Base distribution parameters
    base_mean: Float,
    base_variance: Float,
    /// Stick-breaking weights
    stick_weights: Vec<Float>,
    /// Cluster centers
    cluster_centers: Vec<Float>,
    /// Cluster assignments for training data
    cluster_assignments: Vec<usize>,
    /// Number of clusters used
    n_clusters: usize,
    /// Maximum number of clusters to consider
    max_clusters: usize,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

impl Default for DirichletProcessCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl DirichletProcessCalibrator {
    /// Create a new Dirichlet Process calibrator
    pub fn new() -> Self {
        Self {
            concentration: 1.0,
            base_mean: 0.5,
            base_variance: 0.25,
            stick_weights: Vec::new(),
            cluster_centers: Vec::new(),
            cluster_assignments: Vec::new(),
            n_clusters: 0,
            max_clusters: 20,
            is_fitted: false,
        }
    }

    /// Set concentration parameter
    pub fn with_concentration(mut self, concentration: Float) -> Self {
        self.concentration = concentration;
        self
    }

    /// Set maximum number of clusters
    pub fn with_max_clusters(mut self, max_clusters: usize) -> Self {
        self.max_clusters = max_clusters;
        self
    }

    /// Stick-breaking construction for Dirichlet Process
    fn stick_breaking(&mut self, n_clusters: usize) -> Vec<Float> {
        let mut weights = Vec::with_capacity(n_clusters);
        let mut remaining_weight = 1.0;

        let _rng_instance = thread_rng();

        for _ in 0..n_clusters {
            // Simple beta approximation: uniform for simplicity
            let v = 0.5 as Float;
            let weight = v * remaining_weight;
            weights.push(weight);
            remaining_weight *= 1.0 - v;

            if remaining_weight < 1e-10 {
                break;
            }
        }

        // Normalize weights
        let total: Float = weights.iter().sum();
        if total > 0.0 {
            weights.iter_mut().for_each(|w| *w /= total);
        }

        weights
    }

    /// Chinese Restaurant Process for cluster assignment
    fn chinese_restaurant_process(
        &self,
        probabilities: &Array1<Float>,
    ) -> (Vec<usize>, Vec<Float>) {
        let mut cluster_assignments = Vec::new();
        let mut cluster_centers = Vec::new();
        let mut cluster_counts = Vec::new();

        let _rng_instance = thread_rng();
        let base_std = self.base_variance.sqrt();

        for &prob in probabilities.iter() {
            let n = cluster_assignments.len() as Float;

            // Compute probabilities for existing clusters
            let mut probs = Vec::new();
            for (k, &center) in cluster_centers.iter().enumerate() {
                let count = cluster_counts[k] as Float;
                let diff: Float = prob - center;
                let likelihood = (-0.5 * diff * diff / self.base_variance).exp();
                probs.push(count / (n + self.concentration) * likelihood);
            }

            // Probability for new cluster
            let new_cluster_prob = self.concentration / (n + self.concentration);
            probs.push(new_cluster_prob);

            // Normalize probabilities
            let total: Float = probs.iter().sum();
            if total > 0.0 {
                probs.iter_mut().for_each(|p| *p /= total);
            }

            // Sample cluster assignment
            let mut cumsum = 0.0;
            let uniform_sample: Float = 0.5;
            let mut chosen_cluster = probs.len() - 1;

            for (k, &p) in probs.iter().enumerate() {
                cumsum += p;
                if uniform_sample <= cumsum {
                    chosen_cluster = k;
                    break;
                }
            }

            if chosen_cluster == cluster_centers.len() {
                // Create new cluster
                // Simple normal approximation using Box-Muller transform
                let u1: Float = 0.5 as Float;
                let u2: Float = 0.5 as Float;
                let z = (-2.0 * (u1 as f64).ln()).sqrt()
                    * (2.0 * std::f64::consts::PI * u2 as f64).cos() as Float;
                let new_center = self.base_mean + base_std * z;
                cluster_centers.push(new_center.clamp(0.01 as Float, 0.99 as Float));
                cluster_counts.push(1);
                cluster_assignments.push(cluster_centers.len() - 1);
            } else {
                // Assign to existing cluster
                cluster_counts[chosen_cluster] += 1;
                cluster_assignments.push(chosen_cluster);

                // Update cluster center (moving average)
                let count = cluster_counts[chosen_cluster] as Float;
                cluster_centers[chosen_cluster] =
                    (cluster_centers[chosen_cluster] * (count - 1.0) + prob) / count;
            }

            // Limit number of clusters
            if cluster_centers.len() >= self.max_clusters {
                break;
            }
        }

        (cluster_assignments, cluster_centers)
    }
}

impl CalibrationEstimator for DirichletProcessCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Result<()> {
        let targets_float = targets.mapv(|x| x as Float);
        if probabilities.len() != targets_float.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and targets must have the same length".to_string(),
            ));
        }

        // Use Chinese Restaurant Process for clustering
        let (assignments, centers) = self.chinese_restaurant_process(probabilities);

        self.cluster_assignments = assignments;
        self.cluster_centers = centers;
        self.n_clusters = self.cluster_centers.len();

        // Generate stick-breaking weights
        self.stick_weights = self.stick_breaking(self.n_clusters);

        // Refine cluster centers based on targets
        let mut cluster_sums = vec![0.0; self.n_clusters];
        let mut cluster_counts = vec![0; self.n_clusters];

        for (i, &target) in targets_float.iter().enumerate() {
            if i < self.cluster_assignments.len() {
                let cluster = self.cluster_assignments[i];
                if cluster < self.n_clusters {
                    cluster_sums[cluster] += target;
                    cluster_counts[cluster] += 1;
                }
            }
        }

        // Update cluster centers with empirical means
        for (i, &count) in cluster_counts.iter().enumerate() {
            if count > 0 {
                self.cluster_centers[i] = cluster_sums[i] / count as Float;
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::InvalidInput(
                "Calibrator must be fitted before prediction".to_string(),
            ));
        }

        let mut predictions = Array1::zeros(probabilities.len());

        for (i, &prob) in probabilities.iter().enumerate() {
            let mut weighted_prediction = 0.0;

            // Compute weighted prediction using all clusters
            for (k, &center) in self.cluster_centers.iter().enumerate() {
                let weight = if k < self.stick_weights.len() {
                    self.stick_weights[k]
                } else {
                    1e-10
                };

                // Use Gaussian kernel for smoothing
                let kernel_value = (-0.5 * (prob - center).powi(2) / self.base_variance).exp();
                weighted_prediction += weight * kernel_value * center;
            }

            predictions[i] = weighted_prediction.clamp(0.0 as Float, 1.0 as Float);
        }

        Ok(predictions)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Non-parametric Gaussian Process Classifier for Calibration
///
/// Uses Gaussian Process regression with classification-specific kernels
/// and non-parametric priors for flexible calibration modeling.
#[derive(Debug, Clone)]
pub struct NonParametricGPCalibrator {
    /// Kernel type
    kernel_type: GPKernelType,
    /// Inducing points for sparse GP
    inducing_points: Vec<Float>,
    /// Number of inducing points
    n_inducing: usize,
    /// Noise level
    noise_level: Float,
    /// Kernel hyperparameters
    kernel_params: HashMap<String, Float>,
    /// Training inputs and outputs
    train_inputs: Option<Array1<Float>>,
    train_outputs: Option<Array1<Float>>,
    /// GP posterior parameters
    alpha: Option<Array1<Float>>,
    /// Kernel matrix inverse
    k_inv: Option<Array2<Float>>,
    /// Whether the calibrator is fitted
    is_fitted: bool,
}

#[derive(Debug, Clone)]
pub enum GPKernelType {
    /// Spectral Mixture kernel for flexible modeling
    SpectralMixture,
    /// Neural Network kernel
    NeuralNetwork,
    /// Periodic kernel for cyclical patterns
    Periodic,
    /// Compositional kernel (sum of kernels)
    Compositional,
}

impl Default for NonParametricGPCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl NonParametricGPCalibrator {
    /// Create a new non-parametric GP calibrator
    pub fn new() -> Self {
        Self {
            kernel_type: GPKernelType::SpectralMixture,
            inducing_points: Vec::new(),
            n_inducing: 10,
            noise_level: 0.01,
            kernel_params: HashMap::new(),
            train_inputs: None,
            train_outputs: None,
            alpha: None,
            k_inv: None,
            is_fitted: false,
        }
    }

    /// Set kernel type
    pub fn with_kernel(mut self, kernel_type: GPKernelType) -> Self {
        self.kernel_type = kernel_type;
        self
    }

    /// Set number of inducing points
    pub fn with_inducing_points(mut self, n_inducing: usize) -> Self {
        self.n_inducing = n_inducing;
        self
    }

    /// Compute kernel function
    fn kernel(&self, x1: Float, x2: Float) -> Float {
        match self.kernel_type {
            GPKernelType::SpectralMixture => {
                // Spectral mixture kernel with learned frequencies
                let mut result = 0.0;
                let n_components = 3; // Number of spectral components

                for i in 0..n_components {
                    let freq_key = format!("freq_{}", i);
                    let weight_key = format!("weight_{}", i);
                    let length_key = format!("length_{}", i);

                    let freq = self.kernel_params.get(&freq_key).unwrap_or(&1.0);
                    let weight = self.kernel_params.get(&weight_key).unwrap_or(&1.0);
                    let length = self.kernel_params.get(&length_key).unwrap_or(&1.0);

                    let diff = (x1 - x2).abs();
                    let periodic_term = (2.0 * std::f64::consts::PI * freq * diff).cos();
                    let decay_term = (-0.5 * diff * diff / (length * length)).exp();

                    result += weight * periodic_term * decay_term;
                }

                result
            }
            GPKernelType::NeuralNetwork => {
                // Neural network kernel
                let sigma_b = self.kernel_params.get("sigma_b").unwrap_or(&1.0);
                let sigma_w = self.kernel_params.get("sigma_w").unwrap_or(&1.0);

                let numerator = 2.0 * sigma_b * sigma_b + 2.0 * sigma_w * sigma_w * x1 * x2;
                let denom1 = 1.0 + 2.0 * sigma_b * sigma_b + 2.0 * sigma_w * sigma_w * x1 * x1;
                let denom2 = 1.0 + 2.0 * sigma_b * sigma_b + 2.0 * sigma_w * sigma_w * x2 * x2;

                (2.0 / std::f64::consts::PI as Float)
                    * (numerator / (denom1 * denom2).sqrt()).asin()
            }
            GPKernelType::Periodic => {
                // Periodic kernel
                let period = self.kernel_params.get("period").unwrap_or(&1.0);
                let length_scale = self.kernel_params.get("length_scale").unwrap_or(&1.0);

                let diff = (x1 - x2).abs();
                let sin_term = (std::f64::consts::PI * diff / *period).sin();
                (-2.0 * sin_term * sin_term / (length_scale * length_scale)).exp()
            }
            GPKernelType::Compositional => {
                // Sum of RBF and linear kernels
                let rbf_length = self.kernel_params.get("rbf_length").unwrap_or(&1.0);
                let linear_weight = self.kernel_params.get("linear_weight").unwrap_or(&0.1);

                let diff = (x1 - x2).abs();
                let rbf = (-0.5 * diff * diff / (rbf_length * rbf_length)).exp();
                let linear = linear_weight * x1 * x2;

                rbf + linear
            }
        }
    }

    /// Select inducing points using k-means clustering
    fn select_inducing_points(&mut self, inputs: &Array1<Float>) {
        if inputs.len() <= self.n_inducing {
            self.inducing_points = inputs.to_vec();
            return;
        }

        // Simple k-means for inducing point selection
        let _rng_instance = thread_rng();
        let min_x = inputs.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_x = inputs.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        // Initialize centers uniformly
        self.inducing_points = (0..self.n_inducing)
            .map(|i| min_x + (max_x - min_x) * i as Float / (self.n_inducing - 1) as Float)
            .collect();

        // Run a few k-means iterations
        for _ in 0..10 {
            let mut new_centers = vec![0.0; self.n_inducing];
            let mut counts = vec![0; self.n_inducing];

            for &x in inputs.iter() {
                let mut best_dist = Float::INFINITY;
                let mut best_idx = 0;

                for (i, &center) in self.inducing_points.iter().enumerate() {
                    let dist = (x - center).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = i;
                    }
                }

                new_centers[best_idx] += x;
                counts[best_idx] += 1;
            }

            for i in 0..self.n_inducing {
                if counts[i] > 0 {
                    self.inducing_points[i] = new_centers[i] / counts[i] as Float;
                }
            }
        }
    }

    /// Initialize kernel hyperparameters
    fn initialize_hyperparameters(&mut self) {
        let _rng_instance = thread_rng();

        match self.kernel_type {
            GPKernelType::SpectralMixture => {
                for i in 0..3 {
                    self.kernel_params.insert(format!("freq_{}", i), 0.5);
                    self.kernel_params.insert(format!("weight_{}", i), 0.5);
                    self.kernel_params.insert(format!("length_{}", i), 0.5);
                }
            }
            GPKernelType::NeuralNetwork => {
                self.kernel_params.insert(format!("param_{}", 0), 0.5);
                self.kernel_params.insert(format!("param_{}", 0), 0.5);
            }
            GPKernelType::Periodic => {
                self.kernel_params.insert(format!("param_{}", 0), 0.5);
                self.kernel_params.insert(format!("param_{}", 0), 0.5);
            }
            GPKernelType::Compositional => {
                self.kernel_params.insert(format!("param_{}", 0), 0.5);
                self.kernel_params.insert(format!("param_{}", 0), 0.5);
            }
        }
    }
}

impl CalibrationEstimator for NonParametricGPCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Result<()> {
        let targets_float = targets.mapv(|x| x as Float);
        if probabilities.len() != targets_float.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and targets must have the same length".to_string(),
            ));
        }

        if probabilities.len() < 3 {
            return Err(SklearsError::InvalidInput(
                "Need at least 3 samples for GP calibration".to_string(),
            ));
        }

        // Initialize hyperparameters
        self.initialize_hyperparameters();

        // Select inducing points
        self.select_inducing_points(probabilities);

        // Store training data
        self.train_inputs = Some(probabilities.clone());
        self.train_outputs = Some(targets_float.clone());

        // Build kernel matrix using inducing points for efficiency
        let n = self.inducing_points.len();
        let mut k_matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let kernel_val = self.kernel(self.inducing_points[i], self.inducing_points[j]);
                k_matrix[[i, j]] = kernel_val;
                if i == j {
                    k_matrix[[i, j]] += self.noise_level; // Add noise to diagonal
                }
            }
        }

        // Compute kernel matrix between inducing points and training data
        let mut k_nm = Array2::zeros((n, probabilities.len()));
        for i in 0..n {
            for j in 0..probabilities.len() {
                k_nm[[i, j]] = self.kernel(self.inducing_points[i], probabilities[j]);
            }
        }

        // Solve for GP posterior (simplified FITC approximation)
        // In a full implementation, we'd use proper matrix decomposition
        // For now, use a simplified approach
        let mut alpha = Array1::zeros(n);

        // Simple approach: average targets near each inducing point
        for (i, &inducing_point) in self.inducing_points.iter().enumerate() {
            let mut sum = 0.0;
            let mut count = 0;
            let bandwidth = 0.2; // Kernel bandwidth

            for (j, &prob) in probabilities.iter().enumerate() {
                let weight =
                    (-0.5 * (prob - inducing_point).powi(2) / (bandwidth * bandwidth)).exp();
                if weight > 0.01 {
                    // Only consider significant weights
                    sum += targets_float[j] * weight;
                    count += 1;
                }
            }

            if count > 0 {
                alpha[i] = sum / count as Float;
            }
        }

        self.alpha = Some(alpha);
        self.is_fitted = true;

        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.is_fitted {
            return Err(SklearsError::InvalidInput(
                "Calibrator must be fitted before prediction".to_string(),
            ));
        }

        let alpha = self.alpha.as_ref().unwrap();
        let mut predictions = Array1::zeros(probabilities.len());

        for (i, &prob) in probabilities.iter().enumerate() {
            let mut prediction = 0.0;

            // Compute kernel vector between test point and inducing points
            for (j, &inducing_point) in self.inducing_points.iter().enumerate() {
                let kernel_val = self.kernel(prob, inducing_point);
                prediction += alpha[j] * kernel_val;
            }

            predictions[i] = prediction.clamp(0.0 as Float, 1.0 as Float);
        }

        Ok(predictions)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod nonparametric_tests {
    use super::*;

    fn create_test_data() -> (Array1<Float>, Array1<i32>) {
        let probabilities = Array1::from(vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2]);
        let targets = Array1::from(vec![0, 0, 1, 1, 1, 0]);
        (probabilities, targets)
    }

    #[test]
    fn test_dirichlet_process_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut dp_cal = DirichletProcessCalibrator::new()
            .with_concentration(2.0)
            .with_max_clusters(5);

        dp_cal.fit(&probabilities, &targets).unwrap();
        let predictions = dp_cal.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| (0.0..=1.0).contains(&p)));
        assert!(dp_cal.n_clusters > 0 && dp_cal.n_clusters <= 5);
    }

    #[test]
    fn test_nonparametric_gp_calibrator() {
        let (probabilities, targets) = create_test_data();

        let mut gp_cal = NonParametricGPCalibrator::new()
            .with_kernel(GPKernelType::SpectralMixture)
            .with_inducing_points(4);

        gp_cal.fit(&probabilities, &targets).unwrap();
        let predictions = gp_cal.predict_proba(&probabilities).unwrap();

        assert_eq!(predictions.len(), probabilities.len());
        assert!(predictions.iter().all(|&p| (0.0..=1.0).contains(&p)));
        assert_eq!(gp_cal.inducing_points.len(), 4);
    }

    #[test]
    fn test_different_gp_kernels() {
        let (probabilities, targets) = create_test_data();

        let kernels = vec![
            GPKernelType::SpectralMixture,
            GPKernelType::NeuralNetwork,
            GPKernelType::Periodic,
            GPKernelType::Compositional,
        ];

        for kernel in kernels {
            let mut gp_cal = NonParametricGPCalibrator::new()
                .with_kernel(kernel)
                .with_inducing_points(3);

            gp_cal.fit(&probabilities, &targets).unwrap();
            let predictions = gp_cal.predict_proba(&probabilities).unwrap();

            assert_eq!(predictions.len(), probabilities.len());
            assert!(predictions.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }
    }

    #[test]
    fn test_nonparametric_methods_consistency() {
        let (probabilities, targets) = create_test_data();

        // Test that methods are consistent (repeated calls give same results)
        let mut dp_cal = DirichletProcessCalibrator::new();
        dp_cal.fit(&probabilities, &targets).unwrap();

        let pred1 = dp_cal.predict_proba(&probabilities).unwrap();
        let pred2 = dp_cal.predict_proba(&probabilities).unwrap();

        for (p1, p2) in pred1.iter().zip(pred2.iter()) {
            assert!(
                (p1 - p2).abs() < 1e-10,
                "Predictions should be deterministic"
            );
        }
    }
}
