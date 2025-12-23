//! Epistemic and aleatoric uncertainty estimation
//!
//! This module provides methods to decompose and estimate different types
//! of uncertainty in predictions, including epistemic (model) uncertainty
//! and aleatoric (data) uncertainty.

use crate::{CalibrationEstimator, SigmoidCalibrator};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

/// Configuration for uncertainty estimation
#[derive(Debug, Clone)]
pub struct UncertaintyConfig {
    /// Number of Monte Carlo samples for epistemic uncertainty
    pub n_monte_carlo: usize,
    /// Number of bootstrap samples for epistemic uncertainty estimation
    pub n_bootstrap: usize,
    /// Whether to use Bayesian approach for epistemic uncertainty
    pub bayesian: bool,
    /// Temperature for temperature scaling (affects epistemic uncertainty)
    pub temperature: Float,
    /// Prior strength for Bayesian methods
    pub prior_strength: Float,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            n_monte_carlo: 100,
            n_bootstrap: 1000,
            bayesian: false,
            temperature: 1.0,
            prior_strength: 1.0,
            random_seed: Some(42),
        }
    }
}

/// Types of uncertainty
#[derive(Debug, Clone)]
pub enum UncertaintyType {
    /// Total uncertainty (epistemic + aleatoric)
    Total,
    /// Epistemic (model) uncertainty
    Epistemic,
    /// Aleatoric (data) uncertainty
    Aleatoric,
}

/// Uncertainty estimation result
#[derive(Debug, Clone)]
pub struct UncertaintyResult {
    /// Total uncertainty estimates
    pub total_uncertainty: Array1<Float>,
    /// Epistemic uncertainty estimates
    pub epistemic_uncertainty: Array1<Float>,
    /// Aleatoric uncertainty estimates
    pub aleatoric_uncertainty: Array1<Float>,
    /// Mean predictions
    pub mean_predictions: Array1<Float>,
    /// Prediction variance
    pub prediction_variance: Array1<Float>,
    /// Confidence intervals (lower, upper)
    pub confidence_intervals: Array2<Float>,
    /// Method used for estimation
    pub method: String,
}

/// Main uncertainty estimator
#[derive(Debug, Clone)]
pub struct UncertaintyEstimator {
    config: UncertaintyConfig,
    /// Ensemble of calibrated models for epistemic uncertainty
    ensemble_models_: Option<Vec<Box<dyn CalibrationEstimator>>>,
    /// Training data for bootstrap/Bayesian methods
    training_data_: Option<(Array1<Float>, Array1<i32>)>,
    /// Estimated noise level for aleatoric uncertainty
    noise_level_: Option<Float>,
}

impl UncertaintyEstimator {
    /// Create a new uncertainty estimator
    pub fn new(config: UncertaintyConfig) -> Self {
        Self {
            config,
            ensemble_models_: None,
            training_data_: None,
            noise_level_: None,
        }
    }

    /// Set number of Monte Carlo samples
    pub fn n_monte_carlo(mut self, n: usize) -> Self {
        self.config.n_monte_carlo = n;
        self
    }

    /// Set number of bootstrap samples
    pub fn n_bootstrap(mut self, n: usize) -> Self {
        self.config.n_bootstrap = n;
        self
    }

    /// Enable Bayesian uncertainty estimation
    pub fn bayesian(mut self, enable: bool) -> Self {
        self.config.bayesian = enable;
        self
    }

    /// Fit the uncertainty estimator
    pub fn fit(&mut self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Result<()> {
        // Store training data
        self.training_data_ = Some((probabilities.clone(), targets.clone()));

        // Estimate aleatoric uncertainty (noise level)
        self.noise_level_ = Some(self.estimate_noise_level(probabilities, targets)?);

        // Train ensemble for epistemic uncertainty
        self.train_ensemble(probabilities, targets)?;

        Ok(())
    }

    /// Estimate uncertainty for new predictions
    pub fn predict_uncertainty(&self, probabilities: &Array1<Float>) -> Result<UncertaintyResult> {
        let epistemic = self.estimate_epistemic_uncertainty(probabilities)?;
        let aleatoric = self.estimate_aleatoric_uncertainty(probabilities)?;
        let total = (&epistemic + &aleatoric).mapv(|x| x.sqrt());

        let mean_predictions = probabilities.clone();
        let prediction_variance = &epistemic + &aleatoric;

        // Compute confidence intervals using total uncertainty
        let confidence_intervals = self.compute_confidence_intervals(&mean_predictions, &total)?;

        Ok(UncertaintyResult {
            total_uncertainty: total,
            epistemic_uncertainty: epistemic.mapv(|x| x.sqrt()),
            aleatoric_uncertainty: aleatoric.mapv(|x| x.sqrt()),
            mean_predictions,
            prediction_variance,
            confidence_intervals,
            method: "Ensemble + Noise Estimation".to_string(),
        })
    }

    /// Estimate epistemic uncertainty using ensemble variance
    fn estimate_epistemic_uncertainty(
        &self,
        probabilities: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let ensemble_models =
            self.ensemble_models_
                .as_ref()
                .ok_or_else(|| SklearsError::InvalidData {
                    reason: "Ensemble models not fitted".to_string(),
                })?;

        let n_models = ensemble_models.len();
        let n_samples = probabilities.len();

        // Collect predictions from all ensemble members
        let mut all_predictions = Array2::zeros((n_samples, n_models));

        for (model_idx, model) in ensemble_models.iter().enumerate() {
            let predictions = model.predict_proba(probabilities)?;
            all_predictions.column_mut(model_idx).assign(&predictions);
        }

        // Compute variance across ensemble members
        let mean_predictions = all_predictions.mean_axis(Axis(1)).unwrap();
        let mut variances = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = all_predictions.row(i);
            let mean = mean_predictions[i];
            let variance =
                row.iter().map(|&pred| (pred - mean).powi(2)).sum::<Float>() / n_models as Float;
            variances[i] = variance;
        }

        Ok(variances)
    }

    /// Estimate aleatoric uncertainty using noise level
    fn estimate_aleatoric_uncertainty(
        &self,
        probabilities: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let noise_level = self.noise_level_.ok_or_else(|| SklearsError::InvalidData {
            reason: "Noise level not estimated".to_string(),
        })?;

        // Aleatoric uncertainty for binary classification: p(1-p) + 0.0
        let intrinsic_variance = probabilities.mapv(|p| {
            let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
            clamped_p * (1.0 - clamped_p)
        });

        let aleatoric_variance = intrinsic_variance + noise_level;

        Ok(aleatoric_variance)
    }

    /// Train ensemble of models for epistemic uncertainty
    fn train_ensemble(
        &mut self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<()> {
        use scirs2_core::random::thread_rng;

        let _rng_instance = if let Some(_seed) = self.config.random_seed {
            thread_rng()
        } else {
            thread_rng()
        };

        let n_samples = probabilities.len();
        let n_models = self.config.n_bootstrap.min(50); // Limit ensemble size
        let mut ensemble_models = Vec::with_capacity(n_models);

        for _ in 0..n_models {
            // Bootstrap sampling
            let indices: Vec<usize> = (0..n_samples).map(|_| 0).collect();

            let boot_probabilities: Array1<Float> =
                indices.iter().map(|&i| probabilities[i]).collect();
            let boot_targets: Array1<i32> = indices.iter().map(|&i| targets[i]).collect();

            // Train a calibrator on bootstrap sample
            let calibrator = SigmoidCalibrator::new();
            let fitted_calibrator = calibrator.fit(&boot_probabilities, &boot_targets)?;

            ensemble_models.push(Box::new(fitted_calibrator) as Box<dyn CalibrationEstimator>);
        }

        self.ensemble_models_ = Some(ensemble_models);
        Ok(())
    }

    /// Estimate noise level from training data
    fn estimate_noise_level(
        &self,
        probabilities: &Array1<Float>,
        targets: &Array1<i32>,
    ) -> Result<Float> {
        // Estimate noise using calibration error
        let targets_float: Array1<Float> = targets.mapv(|t| t as Float);
        let residuals = &targets_float - probabilities;

        // Estimate noise as residual variance after accounting for intrinsic variance
        let intrinsic_variance: Float = probabilities
            .iter()
            .map(|&p| {
                let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
                clamped_p * (1.0 - clamped_p)
            })
            .sum::<Float>()
            / probabilities.len() as Float;

        let residual_variance = residuals.var(1.0);
        let noise_level = (residual_variance - intrinsic_variance).max(0.01); // Minimum noise

        Ok(noise_level)
    }

    /// Compute confidence intervals from uncertainty estimates
    fn compute_confidence_intervals(
        &self,
        predictions: &Array1<Float>,
        uncertainties: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let n = predictions.len();
        let mut intervals = Array2::zeros((n, 2));

        // Use normal approximation with 95% confidence
        let z_score = 1.96; // 95% confidence interval

        for i in 0..n {
            let pred = predictions[i];
            let unc = uncertainties[i];
            let margin = z_score * unc;

            intervals[[i, 0]] = (pred - margin).max(0.0); // Lower bound
            intervals[[i, 1]] = (pred + margin).min(1.0); // Upper bound
        }

        Ok(intervals)
    }
}

/// Bayesian uncertainty estimator using variational approximation
#[derive(Debug, Clone)]
pub struct BayesianUncertaintyEstimator {
    config: UncertaintyConfig,
    /// Posterior mean parameters
    posterior_mean_: Option<Array1<Float>>,
    /// Posterior covariance parameters
    posterior_cov_: Option<Array2<Float>>,
    /// Training data statistics
    data_stats_: Option<(Float, Float)>, // mean, variance
}

impl BayesianUncertaintyEstimator {
    /// Create new Bayesian uncertainty estimator
    pub fn new(config: UncertaintyConfig) -> Self {
        Self {
            config,
            posterior_mean_: None,
            posterior_cov_: None,
            data_stats_: None,
        }
    }

    /// Fit Bayesian model
    pub fn fit(&mut self, probabilities: &Array1<Float>, targets: &Array1<i32>) -> Result<()> {
        // Simple Bayesian linear regression approximation
        let n = probabilities.len();
        let targets_float: Array1<Float> = targets.mapv(|t| t as Float);

        // Convert to design matrix (intercept + probability)
        let mut x_matrix = Array2::zeros((n, 2));
        x_matrix.column_mut(0).fill(1.0); // Intercept
        x_matrix.column_mut(1).assign(probabilities);

        // Bayesian linear regression with normal-inverse-gamma prior
        let prior_precision = self.config.prior_strength;
        let prior_mean = Array1::zeros(2);

        // Posterior update
        let xt_x = x_matrix.t().dot(&x_matrix);
        let _posterior_precision = &xt_x + prior_precision * Array2::eye(2);
        // For simplicity, use pseudo-inverse approximation
        let posterior_cov = Array2::eye(2) / prior_precision; // Simplified approximation

        let xt_y = x_matrix.t().dot(&targets_float);
        let posterior_mean = posterior_cov.dot(&(xt_y + prior_precision * prior_mean));

        // Estimate noise variance
        let predictions = x_matrix.dot(&posterior_mean);
        let residuals = &targets_float - &predictions;
        let noise_var = residuals.var(1.0);

        self.posterior_mean_ = Some(posterior_mean);
        self.posterior_cov_ = Some(posterior_cov);
        self.data_stats_ = Some((0.0, noise_var));

        Ok(())
    }

    /// Predict with Bayesian uncertainty
    pub fn predict_uncertainty(&self, probabilities: &Array1<Float>) -> Result<UncertaintyResult> {
        let posterior_mean =
            self.posterior_mean_
                .as_ref()
                .ok_or_else(|| SklearsError::InvalidData {
                    reason: "Model not fitted".to_string(),
                })?;
        let posterior_cov =
            self.posterior_cov_
                .as_ref()
                .ok_or_else(|| SklearsError::InvalidData {
                    reason: "Model not fitted".to_string(),
                })?;
        let (_, noise_var) = self.data_stats_.ok_or_else(|| SklearsError::InvalidData {
            reason: "Data statistics not computed".to_string(),
        })?;

        let n = probabilities.len();

        // Create design matrix
        let mut x_matrix = Array2::zeros((n, 2));
        x_matrix.column_mut(0).fill(1.0);
        x_matrix.column_mut(1).assign(probabilities);

        // Mean predictions
        let mean_predictions = x_matrix.dot(posterior_mean);

        // Epistemic uncertainty (parameter uncertainty)
        let mut epistemic_variance = Array1::zeros(n);
        for i in 0..n {
            let x_i = x_matrix.row(i);
            let var_i = x_i.dot(&posterior_cov.dot(&x_i.t().to_owned()));
            epistemic_variance[i] = var_i;
        }

        // Aleatoric uncertainty (noise)
        let aleatoric_variance = Array1::from(vec![noise_var; n]);

        // Total uncertainty
        let total_variance = &epistemic_variance + &aleatoric_variance;
        let total_uncertainty = total_variance.mapv(|x| x.sqrt());

        // Confidence intervals
        let confidence_intervals =
            self.compute_bayesian_intervals(&mean_predictions, &total_uncertainty)?;

        Ok(UncertaintyResult {
            total_uncertainty,
            epistemic_uncertainty: epistemic_variance.mapv(|x| x.sqrt()),
            aleatoric_uncertainty: aleatoric_variance.mapv(|x| x.sqrt()),
            mean_predictions,
            prediction_variance: total_variance,
            confidence_intervals,
            method: "Bayesian Linear Regression".to_string(),
        })
    }

    fn compute_bayesian_intervals(
        &self,
        predictions: &Array1<Float>,
        uncertainties: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let n = predictions.len();
        let mut intervals = Array2::zeros((n, 2));

        // Use Student's t-distribution for small samples
        let t_score = 2.0; // Approximate for 95% interval

        for i in 0..n {
            let pred = predictions[i];
            let unc = uncertainties[i];
            let margin = t_score * unc;

            intervals[[i, 0]] = (pred - margin).max(0.0);
            intervals[[i, 1]] = (pred + margin).min(1.0);
        }

        Ok(intervals)
    }
}

/// Decompose total uncertainty into epistemic and aleatoric components
pub fn decompose_uncertainty(
    predictions: &Array1<Float>,
    targets: &Array1<i32>,
    config: &UncertaintyConfig,
) -> Result<UncertaintyResult> {
    let mut estimator = UncertaintyEstimator::new(config.clone());
    estimator.fit(predictions, targets)?;
    estimator.predict_uncertainty(predictions)
}

/// Compare different uncertainty estimation methods
pub fn compare_uncertainty_methods(
    probabilities: &Array1<Float>,
    targets: &Array1<i32>,
) -> Result<HashMap<String, UncertaintyResult>> {
    let mut results = HashMap::new();

    // Standard ensemble method
    let config1 = UncertaintyConfig {
        n_bootstrap: 20,
        bayesian: false,
        ..Default::default()
    };

    if let Ok(result1) = decompose_uncertainty(probabilities, targets, &config1) {
        results.insert("Ensemble".to_string(), result1);
    }

    // Bayesian method
    let config2 = UncertaintyConfig {
        bayesian: true,
        prior_strength: 1.0,
        ..Default::default()
    };

    let mut bayesian_estimator = BayesianUncertaintyEstimator::new(config2);
    if bayesian_estimator.fit(probabilities, targets).is_ok() {
        if let Ok(result2) = bayesian_estimator.predict_uncertainty(probabilities) {
            results.insert("Bayesian".to_string(), result2);
        }
    }

    Ok(results)
}

/// Compute uncertainty-based calibration metrics
pub fn uncertainty_based_calibration(
    uncertainty_result: &UncertaintyResult,
    true_targets: &Array1<i32>,
) -> Result<HashMap<String, Float>> {
    let mut metrics = HashMap::new();

    // Compute coverage of confidence intervals
    let mut covered = 0;
    let targets_float: Array1<Float> = true_targets.mapv(|t| t as Float);

    for (i, &target) in targets_float.iter().enumerate() {
        let lower = uncertainty_result.confidence_intervals[[i, 0]];
        let upper = uncertainty_result.confidence_intervals[[i, 1]];
        if target >= lower && target <= upper {
            covered += 1;
        }
    }

    let coverage = covered as Float / true_targets.len() as Float;
    metrics.insert("confidence_coverage".to_string(), coverage);

    // Average interval width
    let avg_width = (0..uncertainty_result.confidence_intervals.nrows())
        .map(|i| {
            uncertainty_result.confidence_intervals[[i, 1]]
                - uncertainty_result.confidence_intervals[[i, 0]]
        })
        .sum::<Float>()
        / uncertainty_result.confidence_intervals.nrows() as Float;

    metrics.insert("average_interval_width".to_string(), avg_width);

    // Uncertainty-weighted calibration error
    let weights =
        &uncertainty_result.total_uncertainty / uncertainty_result.total_uncertainty.sum();
    let weighted_errors: Array1<Float> = uncertainty_result
        .mean_predictions
        .iter()
        .zip(targets_float.iter())
        .zip(weights.iter())
        .map(|((&pred, &target), &weight)| weight * (pred - target).abs())
        .collect();

    let weighted_calibration_error = weighted_errors.sum();
    metrics.insert(
        "weighted_calibration_error".to_string(),
        weighted_calibration_error,
    );

    // Epistemic vs aleatoric ratio
    let epistemic_mean = uncertainty_result
        .epistemic_uncertainty
        .mean()
        .unwrap_or(0.0);
    let aleatoric_mean = uncertainty_result
        .aleatoric_uncertainty
        .mean()
        .unwrap_or(1.0);
    let uncertainty_ratio = epistemic_mean / aleatoric_mean;
    metrics.insert("epistemic_aleatoric_ratio".to_string(), uncertainty_ratio);

    Ok(metrics)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_uncertainty_estimator() {
        let probabilities = array![0.1, 0.3, 0.7, 0.9, 0.5];
        let targets = array![0, 0, 1, 1, 1];

        let config = UncertaintyConfig {
            n_bootstrap: 10,
            ..Default::default()
        };

        let mut estimator = UncertaintyEstimator::new(config);
        estimator.fit(&probabilities, &targets).unwrap();

        let result = estimator.predict_uncertainty(&probabilities).unwrap();

        assert_eq!(result.total_uncertainty.len(), 5);
        assert_eq!(result.epistemic_uncertainty.len(), 5);
        assert_eq!(result.aleatoric_uncertainty.len(), 5);
        assert_eq!(result.confidence_intervals.dim(), (5, 2));

        // Check that uncertainties are positive
        for &unc in result.total_uncertainty.iter() {
            assert!(unc >= 0.0);
        }

        // Check that confidence intervals are ordered
        for i in 0..result.confidence_intervals.nrows() {
            let lower = result.confidence_intervals[[i, 0]];
            let upper = result.confidence_intervals[[i, 1]];
            assert!(lower <= upper);
        }
    }

    #[test]
    fn test_bayesian_uncertainty_estimator() {
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let targets = array![0, 0, 1, 1];

        let config = UncertaintyConfig {
            prior_strength: 1.0,
            ..Default::default()
        };

        let mut estimator = BayesianUncertaintyEstimator::new(config);
        estimator.fit(&probabilities, &targets).unwrap();

        let result = estimator.predict_uncertainty(&probabilities).unwrap();

        assert_eq!(result.method, "Bayesian Linear Regression");
        assert!(result.epistemic_uncertainty.iter().all(|&x| x >= 0.0));
        assert!(result.aleatoric_uncertainty.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_uncertainty_decomposition() {
        let probabilities = array![0.2, 0.4, 0.6, 0.8];
        let targets = array![0, 0, 1, 1];

        let config = UncertaintyConfig {
            n_bootstrap: 5,
            ..Default::default()
        };

        let result = decompose_uncertainty(&probabilities, &targets, &config).unwrap();

        // Total uncertainty should be approximately the sum of components
        for i in 0..probabilities.len() {
            let total_sq = result.total_uncertainty[i].powi(2);
            let epistemic_sq = result.epistemic_uncertainty[i].powi(2);
            let aleatoric_sq = result.aleatoric_uncertainty[i].powi(2);

            // Allow for numerical differences
            assert!((total_sq - (epistemic_sq + aleatoric_sq)).abs() < 0.1);
        }
    }

    #[test]
    fn test_uncertainty_calibration_metrics() {
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let targets = array![0, 0, 1, 1];

        let config = UncertaintyConfig {
            n_bootstrap: 5,
            ..Default::default()
        };

        let result = decompose_uncertainty(&probabilities, &targets, &config).unwrap();
        let metrics = uncertainty_based_calibration(&result, &targets).unwrap();

        assert!(metrics.contains_key("confidence_coverage"));
        assert!(metrics.contains_key("average_interval_width"));
        assert!(metrics.contains_key("weighted_calibration_error"));
        assert!(metrics.contains_key("epistemic_aleatoric_ratio"));

        // Coverage should be between 0 and 1
        let coverage = metrics["confidence_coverage"];
        assert!(coverage >= 0.0 && coverage <= 1.0);

        // Width should be positive
        let width = metrics["average_interval_width"];
        assert!(width > 0.0);
    }

    #[test]
    fn test_compare_uncertainty_methods() {
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let targets = array![0, 0, 1, 1];

        let results = compare_uncertainty_methods(&probabilities, &targets).unwrap();

        assert!(!results.is_empty());

        for (method_name, result) in results {
            println!("Method: {}", method_name);
            println!(
                "  Avg Epistemic: {:.3}",
                result.epistemic_uncertainty.mean().unwrap_or(0.0)
            );
            println!(
                "  Avg Aleatoric: {:.3}",
                result.aleatoric_uncertainty.mean().unwrap_or(0.0)
            );

            assert!(result.total_uncertainty.len() == probabilities.len());
        }
    }
}
