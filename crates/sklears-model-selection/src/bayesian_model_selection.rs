//! Bayesian Model Selection with Evidence Estimation
//!
//! This module implements Bayesian model selection techniques that use the marginal
//! likelihood (evidence) to compare models. It includes various methods for estimating
//! the evidence, including:
//!
//! - Laplace approximation
//! - Bayesian Information Criterion (BIC) approximation
//! - Harmonic mean estimator
//! - Thermodynamic integration
//! - Nested sampling approximation
//! - Model averaging with Bayesian weights

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float as FloatTrait, ToPrimitive};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
use sklears_core::error::{Result, SklearsError};
use std::fmt::Debug;

/// Result of Bayesian model selection
#[derive(Debug, Clone)]
pub struct BayesianModelSelectionResult {
    /// Model identifiers
    pub model_names: Vec<String>,
    /// Log evidence for each model
    pub log_evidence: Vec<f64>,
    /// Model probabilities (Bayesian weights)
    pub model_probabilities: Vec<f64>,
    /// Bayes factors relative to the best model
    pub bayes_factors: Vec<f64>,
    /// Best model index
    pub best_model_index: usize,
    /// Evidence estimation method used
    pub method: EvidenceEstimationMethod,
}

impl BayesianModelSelectionResult {
    /// Get the best model name
    pub fn best_model(&self) -> &str {
        &self.model_names[self.best_model_index]
    }

    /// Get ranking of models by evidence
    pub fn model_ranking(&self) -> Vec<(usize, &str, f64)> {
        let mut ranking: Vec<(usize, &str, f64)> = self
            .model_names
            .iter()
            .enumerate()
            .map(|(i, name)| (i, name.as_str(), self.log_evidence[i]))
            .collect();

        ranking.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        ranking
    }

    /// Interpret the strength of evidence using Jeffreys' scale
    pub fn evidence_interpretation(&self, model1_idx: usize, model2_idx: usize) -> String {
        let log_bf = self.log_evidence[model1_idx] - self.log_evidence[model2_idx];
        let _bf = log_bf.exp();

        match log_bf {
            x if x < 1.0 => "Weak evidence".to_string(),
            x if x < 2.5 => "Positive evidence".to_string(),
            x if x <= 5.0 => "Strong evidence".to_string(),
            _ => "Very strong evidence".to_string(),
        }
    }
}

/// Methods for estimating the evidence (marginal likelihood)
#[derive(Debug, Clone)]
pub enum EvidenceEstimationMethod {
    /// Laplace approximation (Gaussian approximation around MAP)
    LaplaceApproximation,
    /// BIC approximation (asymptotic approximation)
    BIC,
    /// AIC with correction for finite sample size
    AICc,
    /// Harmonic mean estimator
    HarmonicMean,
    /// Thermodynamic integration
    ThermodynamicIntegration { n_temperatures: usize },
    /// Nested sampling approximation
    NestedSampling { n_live_points: usize },
    /// Cross-validation based evidence approximation
    CrossValidationEvidence { n_folds: usize },
}

/// Bayesian model selector
pub struct BayesianModelSelector {
    /// Evidence estimation method
    method: EvidenceEstimationMethod,
    /// Prior model probabilities (if not uniform)
    prior_probabilities: Option<Vec<f64>>,
    /// Random number generator
    rng: StdRng,
}

impl BayesianModelSelector {
    /// Create a new Bayesian model selector
    pub fn new(method: EvidenceEstimationMethod, random_state: Option<u64>) -> Self {
        let rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
        };

        Self {
            method,
            prior_probabilities: None,
            rng,
        }
    }

    /// Set prior probabilities for models (must sum to 1)
    pub fn with_prior_probabilities(mut self, priors: Vec<f64>) -> Result<Self> {
        let sum: f64 = priors.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(SklearsError::InvalidInput(
                "Prior probabilities must sum to 1".to_string(),
            ));
        }
        self.prior_probabilities = Some(priors);
        Ok(self)
    }

    /// Compare models using Bayesian evidence
    pub fn compare_models<F>(
        &mut self,
        model_results: &[(String, ModelEvidenceData)],
    ) -> Result<BayesianModelSelectionResult>
    where
        F: FloatTrait + ToPrimitive,
    {
        if model_results.is_empty() {
            return Err(SklearsError::InvalidInput("No models provided".to_string()));
        }

        let model_names: Vec<String> = model_results.iter().map(|(name, _)| name.clone()).collect();
        let _n_models = model_names.len();

        // Estimate log evidence for each model
        let mut log_evidence = Vec::new();
        for (_, data) in model_results {
            let log_ev = self.estimate_evidence(data)?;
            log_evidence.push(log_ev);
        }

        // Calculate model probabilities
        let model_probabilities = self.calculate_model_probabilities(&log_evidence)?;

        // Calculate Bayes factors relative to best model
        let best_log_evidence = log_evidence
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let bayes_factors: Vec<f64> = log_evidence
            .iter()
            .map(|&log_ev| (log_ev - best_log_evidence).exp())
            .collect();

        let best_model_index = log_evidence
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        Ok(BayesianModelSelectionResult {
            model_names,
            log_evidence,
            model_probabilities,
            bayes_factors,
            best_model_index,
            method: self.method.clone(),
        })
    }

    /// Estimate evidence for a single model
    fn estimate_evidence(&mut self, data: &ModelEvidenceData) -> Result<f64> {
        match &self.method {
            EvidenceEstimationMethod::LaplaceApproximation => self.laplace_approximation(data),
            EvidenceEstimationMethod::BIC => self.bic_approximation(data),
            EvidenceEstimationMethod::AICc => self.aicc_approximation(data),
            EvidenceEstimationMethod::HarmonicMean => self.harmonic_mean_estimator(data),
            EvidenceEstimationMethod::ThermodynamicIntegration { n_temperatures } => {
                self.thermodynamic_integration(data, *n_temperatures)
            }
            EvidenceEstimationMethod::NestedSampling { n_live_points } => {
                self.nested_sampling_approximation(data, *n_live_points)
            }
            EvidenceEstimationMethod::CrossValidationEvidence { n_folds } => {
                self.cross_validation_evidence(data, *n_folds)
            }
        }
    }

    /// Laplace approximation to the evidence
    fn laplace_approximation(&self, data: &ModelEvidenceData) -> Result<f64> {
        let n_params = data.n_parameters as f64;
        let n_data = data.n_data_points as f64;

        // Log likelihood at MAP estimate
        let log_likelihood_map = data.max_log_likelihood;

        // Approximate Hessian determinant (assuming it's available or estimated)
        let log_det_hessian = data.hessian_log_determinant.unwrap_or_else(|| {
            // Rough approximation if Hessian not available
            n_params * (2.0 * std::f64::consts::PI).ln() + n_params * n_data.ln()
        });

        // Prior contribution (assuming flat priors for simplicity)
        let log_prior = data.log_prior.unwrap_or(0.0);

        // Laplace approximation: log Z ≈ log L(θ_MAP) + log π(θ_MAP) + (k/2) log(2π) - (1/2) log|H|
        let log_evidence =
            log_likelihood_map + log_prior + (n_params / 2.0) * (2.0 * std::f64::consts::PI).ln()
                - 0.5 * log_det_hessian;

        Ok(log_evidence)
    }

    /// BIC approximation to the evidence
    fn bic_approximation(&self, data: &ModelEvidenceData) -> Result<f64> {
        let n_params = data.n_parameters as f64;
        let n_data = data.n_data_points as f64;

        // BIC = -2 * log L + k * log(n)
        // log Z ≈ log L - (k/2) * log(n) - (k/2) * log(2π)
        let log_evidence = data.max_log_likelihood
            - (n_params / 2.0) * n_data.ln()
            - (n_params / 2.0) * (2.0 * std::f64::consts::PI).ln();

        Ok(log_evidence)
    }

    /// AICc approximation to the evidence
    fn aicc_approximation(&self, data: &ModelEvidenceData) -> Result<f64> {
        let k = data.n_parameters as f64;
        let n = data.n_data_points as f64;

        if n <= k + 1.0 {
            return Err(SklearsError::InvalidInput(
                "AICc requires n > k + 1".to_string(),
            ));
        }

        // AICc = AIC + 2k(k+1)/(n-k-1)
        let aicc_correction = 2.0 * k * (k + 1.0) / (n - k - 1.0);
        let log_evidence = data.max_log_likelihood - k - aicc_correction / 2.0;

        Ok(log_evidence)
    }

    /// Harmonic mean estimator
    fn harmonic_mean_estimator(&self, data: &ModelEvidenceData) -> Result<f64> {
        if data.posterior_samples.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Harmonic mean estimator requires posterior samples".to_string(),
            ));
        }

        // Harmonic mean: 1/Z = (1/N) Σ 1/L(θ_i)
        // This is known to be unreliable, but included for completeness
        let n_samples = data.posterior_samples.len() as f64;
        let harmonic_mean: f64 = data
            .posterior_samples
            .iter()
            .map(|&log_likelihood| (-log_likelihood).exp())
            .sum::<f64>()
            / n_samples;

        let log_evidence = -harmonic_mean.ln();

        // Add warning about reliability
        eprintln!("Warning: Harmonic mean estimator is known to be unreliable");

        Ok(log_evidence)
    }

    /// Thermodynamic integration
    fn thermodynamic_integration(
        &mut self,
        data: &ModelEvidenceData,
        n_temperatures: usize,
    ) -> Result<f64> {
        if data.posterior_samples.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Thermodynamic integration requires posterior samples".to_string(),
            ));
        }

        // Create temperature ladder from 0 to 1
        let temperatures: Vec<f64> = (0..=n_temperatures)
            .map(|i| i as f64 / n_temperatures as f64)
            .collect();

        // Estimate mean log likelihood at each temperature
        let mut mean_log_likelihoods = Vec::new();

        for &temp in &temperatures {
            if temp == 0.0 {
                // At temperature 0, all weight is on the prior
                mean_log_likelihoods.push(0.0);
            } else {
                // Approximate using available samples (simplified)
                let mean_ll = data.posterior_samples.iter().sum::<f64>()
                    / data.posterior_samples.len() as f64;
                mean_log_likelihoods.push(temp * mean_ll);
            }
        }

        // Integrate using trapezoidal rule
        let mut integral = 0.0;
        for i in 1..temperatures.len() {
            let dt = temperatures[i] - temperatures[i - 1];
            integral += 0.5 * dt * (mean_log_likelihoods[i] + mean_log_likelihoods[i - 1]);
        }

        // Add prior contribution
        let log_prior = data.log_prior.unwrap_or(0.0);
        let log_evidence = log_prior + integral;

        Ok(log_evidence)
    }

    /// Nested sampling approximation
    fn nested_sampling_approximation(
        &mut self,
        data: &ModelEvidenceData,
        n_live_points: usize,
    ) -> Result<f64> {
        // Simplified nested sampling approximation
        // In practice, this would require implementing the full nested sampling algorithm

        if data.posterior_samples.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Nested sampling requires posterior samples".to_string(),
            ));
        }

        let n_samples = data.posterior_samples.len();
        let max_iterations = n_samples.min(1000); // Limit iterations

        // Sort samples by likelihood
        let mut sorted_samples = data.posterior_samples.clone();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Approximate evidence using nested sampling logic
        let mut log_evidence = f64::NEG_INFINITY;
        let mut log_width = -(1.0 / n_live_points as f64).ln();

        for (i, &log_likelihood) in sorted_samples.iter().enumerate() {
            if i >= max_iterations {
                break;
            }

            let log_weight = log_width + log_likelihood;
            log_evidence = log_sum_exp(log_evidence, log_weight);

            // Update width (shrinkage)
            log_width -= (n_live_points as f64).ln();
        }

        Ok(log_evidence)
    }

    /// Cross-validation based evidence approximation
    fn cross_validation_evidence(&self, data: &ModelEvidenceData, n_folds: usize) -> Result<f64> {
        if data.cv_log_likelihoods.is_none() {
            return Err(SklearsError::InvalidInput(
                "Cross-validation evidence requires CV log-likelihoods".to_string(),
            ));
        }

        let cv_log_likes = data.cv_log_likelihoods.as_ref().unwrap();
        if cv_log_likes.len() != n_folds {
            return Err(SklearsError::InvalidInput(
                "Number of CV scores must match number of folds".to_string(),
            ));
        }

        // Approximate evidence using cross-validation
        let mean_cv_log_likelihood = cv_log_likes.iter().sum::<f64>() / cv_log_likes.len() as f64;

        // Apply correction for finite sample effects
        let n_data = data.n_data_points as f64;
        let correction = (n_data / (n_data - 1.0)).ln();

        let log_evidence = mean_cv_log_likelihood + correction;

        Ok(log_evidence)
    }

    /// Calculate model probabilities from log evidence
    fn calculate_model_probabilities(&self, log_evidence: &[f64]) -> Result<Vec<f64>> {
        let n_models = log_evidence.len();

        // Get prior probabilities
        let log_priors = if let Some(ref priors) = self.prior_probabilities {
            if priors.len() != n_models {
                return Err(SklearsError::InvalidInput(
                    "Number of prior probabilities must match number of models".to_string(),
                ));
            }
            priors.iter().map(|&p| p.ln()).collect()
        } else {
            // Uniform priors
            vec![-(n_models as f64).ln(); n_models]
        };

        // Calculate log posterior probabilities
        let log_posteriors: Vec<f64> = log_evidence
            .iter()
            .zip(log_priors.iter())
            .map(|(&log_ev, &log_prior)| log_ev + log_prior)
            .collect();

        // Normalize using log-sum-exp trick
        let log_normalizer = log_sum_exp_vec(&log_posteriors);
        let probabilities: Vec<f64> = log_posteriors
            .iter()
            .map(|&log_p| (log_p - log_normalizer).exp())
            .collect();

        Ok(probabilities)
    }
}

/// Data required for evidence estimation
#[derive(Debug, Clone)]
pub struct ModelEvidenceData {
    /// Maximum log-likelihood achieved
    pub max_log_likelihood: f64,
    /// Number of model parameters
    pub n_parameters: usize,
    /// Number of data points
    pub n_data_points: usize,
    /// Log determinant of Hessian at MAP (for Laplace approximation)
    pub hessian_log_determinant: Option<f64>,
    /// Log prior probability at MAP
    pub log_prior: Option<f64>,
    /// Posterior samples (log-likelihoods)
    pub posterior_samples: Vec<f64>,
    /// Cross-validation log-likelihoods
    pub cv_log_likelihoods: Option<Vec<f64>>,
}

impl ModelEvidenceData {
    /// Create new evidence data with required fields
    pub fn new(max_log_likelihood: f64, n_parameters: usize, n_data_points: usize) -> Self {
        Self {
            max_log_likelihood,
            n_parameters,
            n_data_points,
            hessian_log_determinant: None,
            log_prior: None,
            posterior_samples: Vec::new(),
            cv_log_likelihoods: None,
        }
    }

    /// Add Hessian information for Laplace approximation
    pub fn with_hessian_log_determinant(mut self, log_det: f64) -> Self {
        self.hessian_log_determinant = Some(log_det);
        self
    }

    /// Add prior information
    pub fn with_log_prior(mut self, log_prior: f64) -> Self {
        self.log_prior = Some(log_prior);
        self
    }

    /// Add posterior samples
    pub fn with_posterior_samples(mut self, samples: Vec<f64>) -> Self {
        self.posterior_samples = samples;
        self
    }

    /// Add cross-validation log-likelihoods
    pub fn with_cv_log_likelihoods(mut self, cv_scores: Vec<f64>) -> Self {
        self.cv_log_likelihoods = Some(cv_scores);
        self
    }
}

/// Model averaging using Bayesian weights
pub struct BayesianModelAverager {
    /// Model selection result containing weights
    selection_result: BayesianModelSelectionResult,
}

impl BayesianModelAverager {
    /// Create new model averager from selection result
    pub fn new(selection_result: BayesianModelSelectionResult) -> Self {
        Self { selection_result }
    }

    /// Make prediction using Bayesian model averaging
    pub fn predict(&self, model_predictions: &[Array1<f64>]) -> Result<Array1<f64>> {
        if model_predictions.len() != self.selection_result.model_names.len() {
            return Err(SklearsError::InvalidInput(
                "Number of predictions must match number of models".to_string(),
            ));
        }

        if model_predictions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No predictions provided".to_string(),
            ));
        }

        let n_samples = model_predictions[0].len();

        // Check all predictions have same length
        for pred in model_predictions {
            if pred.len() != n_samples {
                return Err(SklearsError::InvalidInput(
                    "All predictions must have the same length".to_string(),
                ));
            }
        }

        // Weighted average of predictions
        let mut averaged_prediction = Array1::zeros(n_samples);

        for (i, pred) in model_predictions.iter().enumerate() {
            let weight = self.selection_result.model_probabilities[i];
            averaged_prediction = averaged_prediction + pred * weight;
        }

        Ok(averaged_prediction)
    }

    /// Get model weights
    pub fn get_weights(&self) -> &[f64] {
        &self.selection_result.model_probabilities
    }

    /// Get effective number of models (entropy-based measure)
    pub fn effective_number_of_models(&self) -> f64 {
        let entropy: f64 = self
            .selection_result
            .model_probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        entropy.exp()
    }
}

// Utility functions
fn log_sum_exp(a: f64, b: f64) -> f64 {
    let max_val = a.max(b);
    max_val + ((a - max_val).exp() + (b - max_val).exp()).ln()
}

fn log_sum_exp_vec(values: &[f64]) -> f64 {
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f64 = values.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_bic_approximation() {
        let data = ModelEvidenceData::new(-100.0, 5, 100);
        let mut selector = BayesianModelSelector::new(EvidenceEstimationMethod::BIC, Some(42));

        let log_evidence = selector.estimate_evidence(&data).unwrap();
        assert!(log_evidence < 0.0); // Should be negative
    }

    #[test]
    fn test_model_comparison() {
        let data1 = ModelEvidenceData::new(-95.0, 3, 100);
        let data2 = ModelEvidenceData::new(-105.0, 5, 100);

        let models = vec![("Model1".to_string(), data1), ("Model2".to_string(), data2)];

        let mut selector = BayesianModelSelector::new(EvidenceEstimationMethod::BIC, Some(42));

        let result: BayesianModelSelectionResult = selector.compare_models::<f64>(&models).unwrap();

        assert_eq!(result.model_names.len(), 2);
        assert_eq!(result.best_model(), "Model1"); // Better likelihood, fewer parameters
        assert!((result.model_probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bayesian_model_averaging() {
        let data1 = ModelEvidenceData::new(-95.0, 3, 100);
        let data2 = ModelEvidenceData::new(-105.0, 5, 100);

        let models = vec![("Model1".to_string(), data1), ("Model2".to_string(), data2)];

        let mut selector = BayesianModelSelector::new(EvidenceEstimationMethod::BIC, Some(42));

        let selection_result = selector.compare_models::<f64>(&models).unwrap();
        let averager = BayesianModelAverager::new(selection_result);

        let pred1 = array![1.0, 2.0, 3.0];
        let pred2 = array![1.1, 2.1, 3.1];
        let predictions = vec![pred1, pred2];

        let averaged = averager.predict(&predictions).unwrap();
        assert_eq!(averaged.len(), 3);

        // Check effective number of models
        let effective_n = averager.effective_number_of_models();
        assert!(effective_n >= 1.0 && effective_n <= 2.0);
    }

    #[test]
    fn test_evidence_interpretation() {
        let log_evidence = vec![-95.0, -100.0];
        let model_probabilities = vec![0.8, 0.2];
        let bayes_factors = vec![1.0, 0.007]; // exp(-100 - (-95))

        let result = BayesianModelSelectionResult {
            model_names: vec!["Model1".to_string(), "Model2".to_string()],
            log_evidence,
            model_probabilities,
            bayes_factors,
            best_model_index: 0,
            method: EvidenceEstimationMethod::BIC,
        };

        let interpretation = result.evidence_interpretation(0, 1);
        assert!(interpretation.contains("Strong"));
    }

    #[test]
    fn test_model_ranking() {
        let result = BayesianModelSelectionResult {
            model_names: vec![
                "ModelA".to_string(),
                "ModelB".to_string(),
                "ModelC".to_string(),
            ],
            log_evidence: vec![-100.0, -95.0, -98.0],
            model_probabilities: vec![0.1, 0.7, 0.2],
            bayes_factors: vec![0.007, 1.0, 0.05],
            best_model_index: 1,
            method: EvidenceEstimationMethod::BIC,
        };

        let ranking = result.model_ranking();
        assert_eq!(ranking[0].1, "ModelB"); // Best model
        assert_eq!(ranking[1].1, "ModelC"); // Second best
        assert_eq!(ranking[2].1, "ModelA"); // Worst model
    }
}
