//! Bayesian Model Averaging (BMA) implementation
//!
//! This module provides Bayesian Model Averaging functionality for combining
//! multiple models' predictions weighted by their posterior probabilities.

use scirs2_core::ndarray::{Array1, ArrayView1};
use sklears_core::prelude::*;
use std::collections::HashMap;

fn bma_error(msg: &str) -> SklearsError {
    SklearsError::InvalidInput(msg.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PriorType {
    /// Uniform
    Uniform,
    /// Jeffreys
    Jeffreys,
    /// Exponential
    Exponential(f64),
    /// Custom
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvidenceMethod {
    /// MarginalLikelihood
    MarginalLikelihood,
    /// BIC
    BIC,
    /// AIC
    AIC,
    /// AICc
    AICc,
    /// DIC
    DIC,
    /// WAIC
    WAIC,
    /// CrossValidation
    CrossValidation,
    /// BootstrapEstimate
    BootstrapEstimate,
}

#[derive(Debug, Clone)]
pub struct BMAConfig {
    pub prior_type: PriorType,
    pub evidence_method: EvidenceMethod,
    pub min_weight_threshold: f64,
    pub normalize_weights: bool,
    pub use_log_space: bool,
    pub regularization_lambda: f64,
    pub bootstrap_samples: usize,
    pub cv_folds: usize,
}

impl Default for BMAConfig {
    fn default() -> Self {
        Self {
            prior_type: PriorType::Uniform,
            evidence_method: EvidenceMethod::CrossValidation,
            min_weight_threshold: 1e-6,
            normalize_weights: true,
            use_log_space: true,
            regularization_lambda: 1e-3,
            bootstrap_samples: 100,
            cv_folds: 5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_id: String,
    pub complexity: usize,
    pub training_accuracy: f64,
    pub validation_accuracy: f64,
    pub log_likelihood: f64,
    pub n_parameters: usize,
    pub predictions: Array1<f64>,
    pub prediction_variance: Option<Array1<f64>>,
}

#[derive(Debug, Clone)]
pub struct BMAResult {
    pub averaged_predictions: Array1<f64>,
    pub prediction_variance: Array1<f64>,
    pub model_weights: HashMap<String, f64>,
    pub effective_model_count: f64,
    pub total_evidence: f64,
    pub model_posterior_probabilities: HashMap<String, f64>,
    pub ensemble_accuracy: f64,
}

pub struct BayesianModelAverager {
    config: BMAConfig,
    models: Vec<ModelInfo>,
    prior_weights: Option<HashMap<String, f64>>,
    evidence_cache: HashMap<String, f64>,
}

impl BayesianModelAverager {
    pub fn new(config: BMAConfig) -> Self {
        Self {
            config,
            models: Vec::new(),
            prior_weights: None,
            evidence_cache: HashMap::new(),
        }
    }

    pub fn with_prior_weights(mut self, weights: HashMap<String, f64>) -> Result<Self> {
        for weight in weights.values() {
            if *weight < 0.0 {
                return Err(bma_error("Prior weights cannot be negative"));
            }
        }
        self.prior_weights = Some(weights);
        Ok(self)
    }

    pub fn add_model(&mut self, model: ModelInfo) -> Result<()> {
        if !self.models.is_empty() {
            let expected_len = self.models[0].predictions.len();
            if model.predictions.len() != expected_len {
                return Err(bma_error(&format!(
                    "Inconsistent prediction dimensions: expected {}, got {}",
                    expected_len,
                    model.predictions.len()
                )));
            }
        }
        self.models.push(model);
        Ok(())
    }

    pub fn add_models(&mut self, models: Vec<ModelInfo>) -> Result<()> {
        for model in models {
            self.add_model(model)?;
        }
        Ok(())
    }

    pub fn compute_average(&mut self, y_true: Option<&ArrayView1<f64>>) -> Result<BMAResult> {
        if self.models.is_empty() {
            return Err(bma_error("No models provided"));
        }

        let posterior_weights = self.compute_posterior_weights(y_true)?;
        let averaged_predictions = self.compute_weighted_predictions(&posterior_weights)?;
        let prediction_variance =
            self.compute_prediction_variance(&posterior_weights, &averaged_predictions)?;
        let effective_model_count = self.compute_effective_model_count(&posterior_weights);
        let total_evidence = self.compute_total_evidence(y_true)?;

        let ensemble_accuracy = if let Some(y_true) = y_true {
            self.compute_ensemble_accuracy(&averaged_predictions, y_true)
        } else {
            0.0
        };

        let model_posterior_probabilities: HashMap<String, f64> = self
            .models
            .iter()
            .zip(posterior_weights.iter())
            .map(|(model, &weight)| (model.model_id.clone(), weight))
            .collect();

        let model_weights = model_posterior_probabilities.clone();

        Ok(BMAResult {
            averaged_predictions,
            prediction_variance,
            model_weights,
            effective_model_count,
            total_evidence,
            model_posterior_probabilities,
            ensemble_accuracy,
        })
    }

    fn compute_posterior_weights(&mut self, y_true: Option<&ArrayView1<f64>>) -> Result<Vec<f64>> {
        let n_models = self.models.len();
        let mut log_posteriors = vec![0.0; n_models];

        // Collect model references to avoid borrow checker issues
        let models: Vec<_> = self.models.to_vec();

        for (i, model) in models.iter().enumerate() {
            let log_prior = self.compute_log_prior(model)?;
            let log_evidence = self.compute_log_evidence(model, y_true)?;

            log_posteriors[i] = log_prior + log_evidence;
        }

        if self.config.use_log_space {
            self.normalize_log_weights(&mut log_posteriors)
        } else {
            let posteriors: Vec<f64> = log_posteriors.iter().map(|&lp| lp.exp()).collect();
            self.normalize_weights(&posteriors)
        }
    }

    fn compute_log_prior(&self, model: &ModelInfo) -> Result<f64> {
        match self.config.prior_type {
            PriorType::Uniform => Ok(-(self.models.len() as f64).ln()),
            PriorType::Jeffreys => {
                let complexity = model.complexity as f64;
                Ok(-0.5 * complexity.ln())
            }
            PriorType::Exponential(lambda) => {
                let complexity = model.complexity as f64;
                Ok(lambda.ln() - lambda * complexity)
            }
            PriorType::Custom => {
                if let Some(ref prior_weights) = self.prior_weights {
                    if let Some(&weight) = prior_weights.get(&model.model_id) {
                        Ok(weight.ln())
                    } else {
                        Ok(-(self.models.len() as f64).ln())
                    }
                } else {
                    Err(bma_error("Invalid prior specification"))
                }
            }
        }
    }

    fn compute_log_evidence(
        &mut self,
        model: &ModelInfo,
        y_true: Option<&ArrayView1<f64>>,
    ) -> Result<f64> {
        if let Some(cached_evidence) = self.evidence_cache.get(&model.model_id) {
            return Ok(*cached_evidence);
        }

        let log_evidence = match self.config.evidence_method {
            EvidenceMethod::MarginalLikelihood => {
                if let Some(y_true) = y_true {
                    self.compute_marginal_likelihood(model, y_true)?
                } else {
                    model.log_likelihood
                }
            }
            EvidenceMethod::BIC => {
                let n = model.predictions.len() as f64;
                let k = model.n_parameters as f64;
                model.log_likelihood - 0.5 * k * n.ln()
            }
            EvidenceMethod::AIC => {
                let k = model.n_parameters as f64;
                model.log_likelihood - k
            }
            EvidenceMethod::AICc => {
                let n = model.predictions.len() as f64;
                let k = model.n_parameters as f64;
                let aic = model.log_likelihood - k;
                let correction = (2.0 * k * (k + 1.0)) / (n - k - 1.0);
                aic - correction
            }
            EvidenceMethod::DIC => {
                let deviance = -2.0 * model.log_likelihood;
                let p_dic = 2.0 * (model.training_accuracy - model.validation_accuracy).abs();
                -(deviance + p_dic)
            }
            EvidenceMethod::WAIC => {
                if let Some(ref var) = model.prediction_variance {
                    let lppd = model.log_likelihood;
                    let p_waic = var.sum();
                    lppd - p_waic
                } else {
                    model.log_likelihood
                }
            }
            EvidenceMethod::CrossValidation => -self.compute_cv_error(model)?,
            EvidenceMethod::BootstrapEstimate => -self.compute_bootstrap_error(model)?,
        };

        self.evidence_cache
            .insert(model.model_id.clone(), log_evidence);
        Ok(log_evidence)
    }

    fn compute_marginal_likelihood(
        &self,
        model: &ModelInfo,
        y_true: &ArrayView1<f64>,
    ) -> Result<f64> {
        let mut log_likelihood = 0.0;
        let n = y_true.len();

        for i in 0..n {
            let residual = y_true[i] - model.predictions[i];
            let variance = model
                .prediction_variance
                .as_ref()
                .map(|v| v[i])
                .unwrap_or(1.0);

            if variance <= 0.0 {
                return Err(bma_error("Numerical instability in posterior computation"));
            }

            log_likelihood += -0.5
                * (residual.powi(2) / variance + variance.ln() + (2.0 * std::f64::consts::PI).ln());
        }

        let regularization = -0.5 * self.config.regularization_lambda * model.n_parameters as f64;
        Ok(log_likelihood + regularization)
    }

    fn compute_cv_error(&self, model: &ModelInfo) -> Result<f64> {
        let validation_error = 1.0 - model.validation_accuracy;
        Ok(validation_error.max(1e-10).ln())
    }

    fn compute_bootstrap_error(&self, model: &ModelInfo) -> Result<f64> {
        let training_error = 1.0 - model.training_accuracy;
        let validation_error = 1.0 - model.validation_accuracy;
        let bootstrap_error = (training_error + validation_error) / 2.0;
        Ok(bootstrap_error.max(1e-10).ln())
    }

    fn normalize_log_weights(&self, log_weights: &mut [f64]) -> Result<Vec<f64>> {
        let max_log_weight = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if max_log_weight.is_infinite() {
            return Err(bma_error("Numerical instability in posterior computation"));
        }

        for w in log_weights.iter_mut() {
            *w -= max_log_weight;
        }

        let weights: Vec<f64> = log_weights.iter().map(|&lw| lw.exp()).collect();
        self.normalize_weights(&weights)
    }

    fn normalize_weights(&self, weights: &[f64]) -> Result<Vec<f64>> {
        let sum: f64 = weights.iter().sum();

        if sum == 0.0 || !sum.is_finite() {
            return Err(bma_error("Numerical instability in posterior computation"));
        }

        let normalized: Vec<f64> = weights
            .iter()
            .map(|&w| w / sum)
            .map(|w| {
                if w < self.config.min_weight_threshold {
                    0.0
                } else {
                    w
                }
            })
            .collect();

        let final_sum: f64 = normalized.iter().sum();
        if final_sum == 0.0 {
            return Err(bma_error("Numerical instability in posterior computation"));
        }

        Ok(normalized.iter().map(|&w| w / final_sum).collect())
    }

    fn compute_weighted_predictions(&self, weights: &[f64]) -> Result<Array1<f64>> {
        if self.models.is_empty() {
            return Err(bma_error("No models provided"));
        }

        let n_predictions = self.models[0].predictions.len();
        let mut averaged = Array1::zeros(n_predictions);

        for (weight, model) in weights.iter().zip(self.models.iter()) {
            averaged = averaged + *weight * &model.predictions;
        }

        Ok(averaged)
    }

    fn compute_prediction_variance(
        &self,
        weights: &[f64],
        averaged_predictions: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n_predictions = averaged_predictions.len();
        let mut variance = Array1::zeros(n_predictions);

        for i in 0..n_predictions {
            let mut prediction_var = 0.0;
            let mut model_var = 0.0;

            for (weight, model) in weights.iter().zip(self.models.iter()) {
                let diff = model.predictions[i] - averaged_predictions[i];
                prediction_var += weight * diff.powi(2);

                if let Some(ref var) = model.prediction_variance {
                    model_var += weight * var[i];
                }
            }

            variance[i] = prediction_var + model_var;
        }

        Ok(variance)
    }

    fn compute_effective_model_count(&self, weights: &[f64]) -> f64 {
        let sum_squares: f64 = weights.iter().map(|w| w.powi(2)).sum();
        if sum_squares > 0.0 {
            1.0 / sum_squares
        } else {
            0.0
        }
    }

    fn compute_total_evidence(&self, y_true: Option<&ArrayView1<f64>>) -> Result<f64> {
        let mut total_evidence = 0.0;

        for model in &self.models {
            let evidence = self
                .evidence_cache
                .get(&model.model_id)
                .copied()
                .unwrap_or(model.log_likelihood);
            total_evidence += evidence.exp();
        }

        Ok(total_evidence.ln())
    }

    fn compute_ensemble_accuracy(
        &self,
        predictions: &Array1<f64>,
        y_true: &ArrayView1<f64>,
    ) -> f64 {
        let mse: f64 = predictions
            .iter()
            .zip(y_true.iter())
            .map(|(pred, true_val)| (pred - true_val).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;

        (-mse).exp()
    }

    pub fn get_model_rankings(&self, result: &BMAResult) -> Vec<(String, f64)> {
        let mut rankings: Vec<_> = result
            .model_weights
            .iter()
            .map(|(id, &weight)| (id.clone(), weight))
            .collect();
        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rankings
    }

    pub fn prune_models(&mut self, min_weight: f64) -> usize {
        let weights_result = self.compute_posterior_weights(None);
        if let Ok(weights) = weights_result {
            let indices_to_keep: Vec<usize> = weights
                .iter()
                .enumerate()
                .filter(|(_, &w)| w >= min_weight)
                .map(|(i, _)| i)
                .collect();

            let mut new_models = Vec::new();
            for &idx in &indices_to_keep {
                new_models.push(self.models[idx].clone());
            }

            let pruned_count = self.models.len() - new_models.len();
            self.models = new_models;
            self.evidence_cache.clear();

            pruned_count
        } else {
            0
        }
    }
}

pub fn bayesian_model_average(
    models: Vec<ModelInfo>,
    y_true: Option<&ArrayView1<f64>>,
    config: Option<BMAConfig>,
) -> Result<BMAResult> {
    let config = config.unwrap_or_default();
    let mut averager = BayesianModelAverager::new(config);
    averager.add_models(models)?;
    averager.compute_average(y_true)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    fn create_test_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                model_id: "model1".to_string(),
                complexity: 10,
                training_accuracy: 0.85,
                validation_accuracy: 0.80,
                log_likelihood: -100.0,
                n_parameters: 10,
                predictions: arr1(&[0.8, 0.6, 0.9, 0.7, 0.5]),
                prediction_variance: Some(arr1(&[0.01, 0.02, 0.01, 0.03, 0.02])),
            },
            ModelInfo {
                model_id: "model2".to_string(),
                complexity: 15,
                training_accuracy: 0.90,
                validation_accuracy: 0.82,
                log_likelihood: -95.0,
                n_parameters: 15,
                predictions: arr1(&[0.9, 0.7, 0.8, 0.8, 0.6]),
                prediction_variance: Some(arr1(&[0.02, 0.01, 0.02, 0.01, 0.03])),
            },
            ModelInfo {
                model_id: "model3".to_string(),
                complexity: 5,
                training_accuracy: 0.75,
                validation_accuracy: 0.78,
                log_likelihood: -110.0,
                n_parameters: 5,
                predictions: arr1(&[0.7, 0.8, 0.7, 0.6, 0.7]),
                prediction_variance: Some(arr1(&[0.03, 0.02, 0.03, 0.02, 0.01])),
            },
        ]
    }

    #[test]
    fn test_basic_bma() {
        let models = create_test_models();
        let config = BMAConfig::default();
        let result = bayesian_model_average(models, None, Some(config)).unwrap();

        assert_eq!(result.averaged_predictions.len(), 5);
        assert_eq!(result.prediction_variance.len(), 5);
        assert_eq!(result.model_weights.len(), 3);
        assert!(result.effective_model_count > 0.0);
        assert!(result.total_evidence.is_finite());
    }

    #[test]
    fn test_bma_with_ground_truth() {
        let models = create_test_models();
        let y_true = arr1(&[0.8, 0.7, 0.8, 0.7, 0.6]);
        let config = BMAConfig::default();

        let result = bayesian_model_average(models, Some(&y_true.view()), Some(config)).unwrap();

        assert_eq!(result.averaged_predictions.len(), 5);
        assert!(result.ensemble_accuracy > 0.0);
        assert!(result.ensemble_accuracy <= 1.0);
    }

    #[test]
    fn test_uniform_prior() {
        let models = create_test_models();
        let config = BMAConfig {
            prior_type: PriorType::Uniform,
            evidence_method: EvidenceMethod::BIC,
            ..Default::default()
        };

        let result = bayesian_model_average(models, None, Some(config)).unwrap();
        assert!(result.model_weights.values().all(|&w| w > 0.0));
    }

    #[test]
    fn test_jeffreys_prior() {
        let models = create_test_models();
        let config = BMAConfig {
            prior_type: PriorType::Jeffreys,
            evidence_method: EvidenceMethod::AIC,
            ..Default::default()
        };

        let result = bayesian_model_average(models, None, Some(config)).unwrap();
        assert!(result.model_weights.values().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_model_pruning() {
        let models = create_test_models();
        let config = BMAConfig::default();
        let mut averager = BayesianModelAverager::new(config);
        averager.add_models(models).unwrap();

        let initial_count = averager.models.len();
        let pruned = averager.prune_models(0.1);

        assert!(pruned <= initial_count);
        assert!(averager.models.len() <= initial_count);
    }

    #[test]
    fn test_inconsistent_dimensions() {
        let mut models = create_test_models();
        models[1].predictions = arr1(&[0.9, 0.7, 0.8]);

        let result = bayesian_model_average(models, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_models() {
        let models = Vec::new();
        let result = bayesian_model_average(models, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_prior() {
        let models = create_test_models();
        let mut prior_weights = HashMap::new();
        prior_weights.insert("model1".to_string(), 0.5);
        prior_weights.insert("model2".to_string(), 0.3);
        prior_weights.insert("model3".to_string(), 0.2);

        let config = BMAConfig {
            prior_type: PriorType::Custom,
            ..Default::default()
        };

        let mut averager = BayesianModelAverager::new(config);
        averager = averager.with_prior_weights(prior_weights).unwrap();
        averager.add_models(models).unwrap();

        let result = averager.compute_average(None).unwrap();
        assert!(result.model_weights.values().all(|&w| w >= 0.0));
    }
}
