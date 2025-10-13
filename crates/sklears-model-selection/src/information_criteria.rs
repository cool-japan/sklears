//! Information Criteria for Model Comparison
//!
//! This module implements various information criteria for model selection and comparison.
//! Information criteria balance model fit (likelihood) with model complexity (number of parameters)
//! to prevent overfitting and enable fair comparison between models.
//!
//! Implemented criteria:
//! - AIC (Akaike Information Criterion)
//! - AICc (Corrected AIC for finite samples)
//! - BIC (Bayesian Information Criterion)
//! - DIC (Deviance Information Criterion)
//! - WAIC (Watanabe-Akaike Information Criterion)
//! - LOOIC (Leave-One-Out Information Criterion)
//! - TIC (Takeuchi Information Criterion)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float as FloatTrait;
use sklears_core::error::{Result, SklearsError};
use std::fmt::Debug;

/// Result of information criterion calculation
#[derive(Debug, Clone)]
pub struct InformationCriterionResult {
    /// Name of the criterion
    pub criterion_name: String,
    /// Value of the information criterion
    pub value: f64,
    /// Log-likelihood of the model
    pub log_likelihood: f64,
    /// Number of parameters
    pub n_parameters: usize,
    /// Number of data points
    pub n_data_points: usize,
    /// Effective number of parameters (for DIC, WAIC)
    pub effective_parameters: Option<f64>,
    /// Standard error of the criterion (if available)
    pub standard_error: Option<f64>,
    /// Model weight relative to other models
    pub weight: Option<f64>,
}

impl InformationCriterionResult {
    /// Create new IC result
    pub fn new(
        criterion_name: String,
        value: f64,
        log_likelihood: f64,
        n_parameters: usize,
        n_data_points: usize,
    ) -> Self {
        Self {
            criterion_name,
            value,
            log_likelihood,
            n_parameters,
            n_data_points,
            effective_parameters: None,
            standard_error: None,
            weight: None,
        }
    }

    /// Set effective number of parameters
    pub fn with_effective_parameters(mut self, p_eff: f64) -> Self {
        self.effective_parameters = Some(p_eff);
        self
    }

    /// Set standard error
    pub fn with_standard_error(mut self, se: f64) -> Self {
        self.standard_error = Some(se);
        self
    }

    /// Set model weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = Some(weight);
        self
    }
}

/// Comparison result for multiple models
#[derive(Debug, Clone)]
pub struct ModelComparisonResult {
    /// Model names
    pub model_names: Vec<String>,
    /// Information criterion results for each model
    pub results: Vec<InformationCriterionResult>,
    /// Delta values (difference from best model)
    pub delta_values: Vec<f64>,
    /// Model weights (Akaike weights)
    pub weights: Vec<f64>,
    /// Index of best model
    pub best_model_index: usize,
    /// Evidence ratio for best vs second best
    pub evidence_ratio: f64,
}

impl ModelComparisonResult {
    /// Get best model name
    pub fn best_model(&self) -> &str {
        &self.model_names[self.best_model_index]
    }

    /// Get model ranking by IC value
    pub fn model_ranking(&self) -> Vec<(usize, &str, f64)> {
        let mut ranking: Vec<(usize, &str, f64)> = self
            .model_names
            .iter()
            .enumerate()
            .map(|(i, name)| (i, name.as_str(), self.results[i].value))
            .collect();

        ranking.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        ranking
    }

    /// Interpret model strength using Burnham & Anderson guidelines
    pub fn model_strength_interpretation(&self, model_idx: usize) -> String {
        let delta = self.delta_values[model_idx];
        match delta {
            d if d <= 2.0 => "Substantial support".to_string(),
            d if d <= 4.0 => "Considerably less support".to_string(),
            d if d <= 7.0 => "Little support".to_string(),
            _ => "No support".to_string(),
        }
    }
}

/// Information criterion calculator
pub struct InformationCriterionCalculator {
    /// Whether to use bias correction for finite samples
    pub use_bias_correction: bool,
    /// Whether to calculate model weights
    pub calculate_weights: bool,
}

impl Default for InformationCriterionCalculator {
    fn default() -> Self {
        Self {
            use_bias_correction: true,
            calculate_weights: true,
        }
    }
}

impl InformationCriterionCalculator {
    /// Create new calculator
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate AIC (Akaike Information Criterion)
    /// AIC = 2k - 2ln(L)
    pub fn aic(
        &self,
        log_likelihood: f64,
        n_parameters: usize,
        n_data_points: usize,
    ) -> InformationCriterionResult {
        let k = n_parameters as f64;
        let aic_value = 2.0 * k - 2.0 * log_likelihood;

        InformationCriterionResult::new(
            "AIC".to_string(),
            aic_value,
            log_likelihood,
            n_parameters,
            n_data_points,
        )
    }

    /// Calculate AICc (Corrected AIC for finite samples)
    /// AICc = AIC + 2k(k+1)/(n-k-1)
    pub fn aicc(
        &self,
        log_likelihood: f64,
        n_parameters: usize,
        n_data_points: usize,
    ) -> Result<InformationCriterionResult> {
        let k = n_parameters as f64;
        let n = n_data_points as f64;

        if n <= k + 1.0 {
            return Err(SklearsError::InvalidInput(
                "AICc requires n > k + 1".to_string(),
            ));
        }

        let aic_value = 2.0 * k - 2.0 * log_likelihood;
        let correction = 2.0 * k * (k + 1.0) / (n - k - 1.0);
        let aicc_value = aic_value + correction;

        Ok(InformationCriterionResult::new(
            "AICc".to_string(),
            aicc_value,
            log_likelihood,
            n_parameters,
            n_data_points,
        ))
    }

    /// Calculate BIC (Bayesian Information Criterion)
    /// BIC = k*ln(n) - 2ln(L)
    pub fn bic(
        &self,
        log_likelihood: f64,
        n_parameters: usize,
        n_data_points: usize,
    ) -> InformationCriterionResult {
        let k = n_parameters as f64;
        let n = n_data_points as f64;
        let bic_value = k * n.ln() - 2.0 * log_likelihood;

        InformationCriterionResult::new(
            "BIC".to_string(),
            bic_value,
            log_likelihood,
            n_parameters,
            n_data_points,
        )
    }

    /// Calculate DIC (Deviance Information Criterion)
    /// DIC = D(θ̄) + 2p_D, where p_D is effective number of parameters
    pub fn dic(
        &self,
        log_likelihood_mean: f64,
        log_likelihood_samples: &[f64],
        n_data_points: usize,
    ) -> Result<InformationCriterionResult> {
        if log_likelihood_samples.is_empty() {
            return Err(SklearsError::InvalidInput(
                "DIC requires posterior samples".to_string(),
            ));
        }

        // Deviance at posterior mean
        let deviance_mean = -2.0 * log_likelihood_mean;

        // Mean deviance
        let mean_deviance =
            -2.0 * log_likelihood_samples.iter().sum::<f64>() / log_likelihood_samples.len() as f64;

        // Effective number of parameters
        let p_d = mean_deviance - deviance_mean;

        // DIC value
        let dic_value = deviance_mean + p_d;

        Ok(InformationCriterionResult::new(
            "DIC".to_string(),
            dic_value,
            log_likelihood_mean,
            0, // Not applicable for DIC
            n_data_points,
        )
        .with_effective_parameters(p_d))
    }

    /// Calculate WAIC (Watanabe-Akaike Information Criterion)
    /// WAIC = -2 * (lppd - p_WAIC)
    pub fn waic(
        &self,
        pointwise_log_likelihoods: &Array2<f64>, // Rows: samples, Columns: data points
    ) -> Result<InformationCriterionResult> {
        let (n_samples, n_data) = pointwise_log_likelihoods.dim();

        if n_samples == 0 || n_data == 0 {
            return Err(SklearsError::InvalidInput(
                "WAIC requires non-empty likelihood matrix".to_string(),
            ));
        }

        // Log pointwise predictive density (lppd)
        let mut lppd = 0.0;
        let mut p_waic = 0.0;

        for j in 0..n_data {
            let column = pointwise_log_likelihoods.column(j);

            // Log of mean of likelihoods for data point j
            let column_data: Vec<f64> = column.iter().copied().collect();
            let log_mean_likelihood = log_mean_exp(&column_data);
            lppd += log_mean_likelihood;

            // Variance of log-likelihoods for data point j
            let mean_log_likelihood = column.mean().unwrap();
            let variance = column
                .iter()
                .map(|&x| (x - mean_log_likelihood).powi(2))
                .sum::<f64>()
                / (n_samples - 1) as f64;
            p_waic += variance;
        }

        let waic_value = -2.0 * (lppd - p_waic);

        // Calculate total log likelihood (approximate)
        let total_log_likelihood = pointwise_log_likelihoods.sum();

        Ok(InformationCriterionResult::new(
            "WAIC".to_string(),
            waic_value,
            total_log_likelihood,
            0, // Not directly applicable
            n_data,
        )
        .with_effective_parameters(p_waic))
    }

    /// Calculate LOOIC (Leave-One-Out Information Criterion) using Pareto smoothed importance sampling
    pub fn looic(
        &self,
        pointwise_log_likelihoods: &Array2<f64>,
        pareto_k_diagnostics: Option<&Array1<f64>>,
    ) -> Result<InformationCriterionResult> {
        let (n_samples, n_data) = pointwise_log_likelihoods.dim();

        if n_samples == 0 || n_data == 0 {
            return Err(SklearsError::InvalidInput(
                "LOOIC requires non-empty likelihood matrix".to_string(),
            ));
        }

        let mut elpd_loo = 0.0; // Expected log pointwise predictive density
        let mut p_loo = 0.0; // Effective number of parameters

        for j in 0..n_data {
            let column = pointwise_log_likelihoods.column(j);
            let log_likes = column.as_slice().unwrap();

            // Check Pareto k diagnostic if available
            if let Some(k_values) = pareto_k_diagnostics {
                if k_values[j] > 0.7 {
                    eprintln!(
                        "Warning: High Pareto k ({:.3}) for observation {}",
                        k_values[j], j
                    );
                }
            }

            // Importance sampling LOO estimate
            let max_log_like = log_likes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let rel_log_likes: Vec<f64> = log_likes.iter().map(|&x| x - max_log_like).collect();

            // Calculate importance weights (simplified PSIS)
            let weights: Vec<f64> = rel_log_likes.iter().map(|&x| x.exp()).collect();

            let sum_weights: f64 = weights.iter().sum();
            if sum_weights == 0.0 {
                return Err(SklearsError::InvalidInput(
                    "Zero importance weights".to_string(),
                ));
            }

            let normalized_weights: Vec<f64> = weights.iter().map(|&w| w / sum_weights).collect();

            // LOO predictive density
            let loo_lpd = (sum_weights / n_samples as f64).ln() + max_log_like;
            elpd_loo += loo_lpd;

            // Effective number of parameters contribution
            let mean_log_like = log_likes.iter().sum::<f64>() / n_samples as f64;
            p_loo += mean_log_like - loo_lpd;
        }

        let looic_value = -2.0 * elpd_loo;

        Ok(InformationCriterionResult::new(
            "LOOIC".to_string(),
            looic_value,
            0.0, // Not directly applicable
            0,   // Not directly applicable
            n_data,
        )
        .with_effective_parameters(p_loo))
    }

    /// Calculate TIC (Takeuchi Information Criterion)
    /// TIC = -2ln(L) + 2tr(J^{-1}K), where J is Fisher information and K is outer product of scores
    pub fn tic(
        &self,
        log_likelihood: f64,
        fisher_information_trace: f64,
        n_data_points: usize,
    ) -> Result<InformationCriterionResult> {
        if fisher_information_trace <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "Fisher information trace must be positive".to_string(),
            ));
        }

        let tic_value = -2.0 * log_likelihood + 2.0 * fisher_information_trace;

        Ok(InformationCriterionResult::new(
            "TIC".to_string(),
            tic_value,
            log_likelihood,
            0, // Parameters counted via Fisher information
            n_data_points,
        )
        .with_effective_parameters(fisher_information_trace))
    }

    /// Compare multiple models using specified criterion
    pub fn compare_models(
        &self,
        models: &[(String, f64, usize, usize)], // (name, log_likelihood, n_params, n_data)
        criterion: InformationCriterion,
    ) -> Result<ModelComparisonResult> {
        if models.is_empty() {
            return Err(SklearsError::InvalidInput("No models provided".to_string()));
        }

        let mut results = Vec::new();
        let model_names: Vec<String> = models.iter().map(|(name, _, _, _)| name.clone()).collect();

        // Calculate IC for each model
        for (name, log_likelihood, n_params, n_data) in models {
            let result = match criterion {
                InformationCriterion::AIC => self.aic(*log_likelihood, *n_params, *n_data),
                InformationCriterion::AICc => self.aicc(*log_likelihood, *n_params, *n_data)?,
                InformationCriterion::BIC => self.bic(*log_likelihood, *n_params, *n_data),
            };
            results.push(result);
        }

        // Find best model (lowest IC value)
        let best_idx = results
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.value.partial_cmp(&b.value).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let best_value = results[best_idx].value;

        // Calculate delta values
        let delta_values: Vec<f64> = results.iter().map(|r| r.value - best_value).collect();

        // Calculate Akaike weights
        let weights = if self.calculate_weights {
            self.calculate_akaike_weights(&delta_values)
        } else {
            vec![0.0; results.len()]
        };

        // Evidence ratio (best vs second best)
        let mut sorted_deltas = delta_values.clone();
        sorted_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let evidence_ratio = if sorted_deltas.len() > 1 {
            (-0.5 * sorted_deltas[1]).exp()
        } else {
            1.0
        };

        Ok(ModelComparisonResult {
            model_names,
            results,
            delta_values,
            weights,
            best_model_index: best_idx,
            evidence_ratio,
        })
    }

    /// Calculate Akaike weights from delta values
    fn calculate_akaike_weights(&self, delta_values: &[f64]) -> Vec<f64> {
        // Akaike weights: w_i = exp(-Δ_i/2) / Σ exp(-Δ_j/2)
        let weights: Vec<f64> = delta_values
            .iter()
            .map(|&delta| (-0.5 * delta).exp())
            .collect();

        let sum_weights: f64 = weights.iter().sum();
        if sum_weights == 0.0 {
            return vec![1.0 / weights.len() as f64; weights.len()];
        }

        weights.iter().map(|&w| w / sum_weights).collect()
    }

    /// Calculate model-averaged prediction using IC weights
    pub fn model_averaged_prediction(
        &self,
        predictions: &[Array1<f64>],
        weights: &[f64],
    ) -> Result<Array1<f64>> {
        if predictions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No predictions provided".to_string(),
            ));
        }

        if predictions.len() != weights.len() {
            return Err(SklearsError::InvalidInput(
                "Number of predictions must match number of weights".to_string(),
            ));
        }

        let n_samples = predictions[0].len();
        for pred in predictions {
            if pred.len() != n_samples {
                return Err(SklearsError::InvalidInput(
                    "All predictions must have the same length".to_string(),
                ));
            }
        }

        let mut averaged = Array1::zeros(n_samples);
        for (pred, &weight) in predictions.iter().zip(weights.iter()) {
            averaged = averaged + pred * weight;
        }

        Ok(averaged)
    }
}

/// Types of information criteria
#[derive(Debug, Clone, Copy)]
pub enum InformationCriterion {
    /// AIC
    AIC,
    /// AICc
    AICc,
    /// BIC
    BIC,
}

// Utility functions
fn log_mean_exp(values: &[f64]) -> f64 {
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f64 = values.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + (sum / values.len() as f64).ln()
}

/// Model selection using information criteria with cross-validation
pub struct CrossValidatedIC {
    criterion: InformationCriterion,
    n_folds: usize,
}

impl CrossValidatedIC {
    /// Create new cross-validated IC selector
    pub fn new(criterion: InformationCriterion, n_folds: usize) -> Self {
        Self { criterion, n_folds }
    }

    /// Select best model using cross-validated IC
    pub fn select_model(
        &self,
        cv_results: &[(String, Vec<f64>, Vec<usize>, Vec<usize>)], // (name, cv_log_likes, cv_n_params, cv_n_data)
    ) -> Result<ModelComparisonResult> {
        let calculator = InformationCriterionCalculator::new();
        let mut aggregated_models = Vec::new();

        for (name, cv_log_likes, cv_n_params, cv_n_data) in cv_results {
            if cv_log_likes.len() != self.n_folds {
                return Err(SklearsError::InvalidInput(
                    "CV results must match number of folds".to_string(),
                ));
            }

            // Aggregate across folds
            let total_log_likelihood: f64 = cv_log_likes.iter().sum();
            let avg_n_params = cv_n_params.iter().sum::<usize>() / cv_n_params.len();
            let total_n_data = cv_n_data.iter().sum::<usize>();

            aggregated_models.push((
                name.clone(),
                total_log_likelihood,
                avg_n_params,
                total_n_data,
            ));
        }

        calculator.compare_models(&aggregated_models, self.criterion)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_aic_calculation() {
        let calculator = InformationCriterionCalculator::new();
        let result = calculator.aic(-100.0, 5, 200);

        assert_eq!(result.criterion_name, "AIC");
        assert_eq!(result.value, 210.0); // 2*5 - 2*(-100)
        assert_eq!(result.n_parameters, 5);
        assert_eq!(result.n_data_points, 200);
    }

    #[test]
    fn test_aicc_calculation() {
        let calculator = InformationCriterionCalculator::new();
        let result = calculator.aicc(-100.0, 5, 20).unwrap();

        assert_eq!(result.criterion_name, "AICc");
        assert!(result.value > 210.0); // Should be higher than AIC due to correction
    }

    #[test]
    fn test_bic_calculation() {
        let calculator = InformationCriterionCalculator::new();
        let result = calculator.bic(-100.0, 5, 200);

        assert_eq!(result.criterion_name, "BIC");
        assert!(result.value > 210.0); // BIC penalizes complexity more than AIC
    }

    #[test]
    fn test_model_comparison() {
        let calculator = InformationCriterionCalculator::new();
        let models = vec![
            ("Model1".to_string(), -95.0, 3, 100),
            ("Model2".to_string(), -100.0, 5, 100),
            ("Model3".to_string(), -98.0, 4, 100),
        ];

        let result = calculator
            .compare_models(&models, InformationCriterion::AIC)
            .unwrap();

        assert_eq!(result.model_names.len(), 3);
        assert_eq!(result.best_model(), "Model1"); // Best log-likelihood with fewest parameters
        assert!((result.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_akaike_weights() {
        let calculator = InformationCriterionCalculator::new();
        let delta_values = vec![0.0, 2.0, 4.0]; // Best model, 2 AIC units worse, 4 AIC units worse
        let weights = calculator.calculate_akaike_weights(&delta_values);

        assert!(weights[0] > weights[1]); // Best model should have highest weight
        assert!(weights[1] > weights[2]); // Second best should be better than worst
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-6); // Weights sum to 1
    }

    #[test]
    fn test_waic_calculation() {
        let calculator = InformationCriterionCalculator::new();

        // Mock pointwise log-likelihoods: 3 samples, 5 data points
        let pointwise_ll = array![
            [-1.0, -1.2, -0.9, -1.1, -1.0],
            [-1.1, -1.0, -1.0, -1.0, -0.9],
            [-0.9, -1.1, -1.1, -0.9, -1.1]
        ];

        let result = calculator.waic(&pointwise_ll).unwrap();
        assert_eq!(result.criterion_name, "WAIC");
        assert!(result.effective_parameters.is_some());
    }

    #[test]
    fn test_model_ranking() {
        let models = vec![
            ("ModelA".to_string(), -100.0, 5, 100),
            ("ModelB".to_string(), -95.0, 3, 100),
            ("ModelC".to_string(), -98.0, 4, 100),
        ];

        let calculator = InformationCriterionCalculator::new();
        let result = calculator
            .compare_models(&models, InformationCriterion::AIC)
            .unwrap();
        let ranking = result.model_ranking();

        assert_eq!(ranking[0].1, "ModelB"); // Best model
        assert_eq!(ranking[2].1, "ModelA"); // Worst model
    }

    #[test]
    fn test_model_averaged_prediction() {
        let calculator = InformationCriterionCalculator::new();

        let pred1 = array![1.0, 2.0, 3.0];
        let pred2 = array![1.1, 2.1, 3.1];
        let pred3 = array![0.9, 1.9, 2.9];

        let predictions = vec![pred1, pred2, pred3];
        let weights = vec![0.5, 0.3, 0.2];

        let averaged = calculator
            .model_averaged_prediction(&predictions, &weights)
            .unwrap();
        assert_eq!(averaged.len(), 3);

        // Check weighted average
        let expected_0 = 1.0 * 0.5 + 1.1 * 0.3 + 0.9 * 0.2;
        assert!((averaged[0] - expected_0).abs() < 1e-10);
    }
}
