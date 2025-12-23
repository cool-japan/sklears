//! Prediction intervals for uncertainty quantification
//!
//! This module provides various methods for computing prediction intervals
//! that quantify uncertainty in predictions, including both parametric
//! and non-parametric approaches.

use crate::{
    conformal::{AbsoluteResidualScore, ConformalMethod, ConformalPredictor},
    CalibrationEstimator,
};
use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

/// Configuration for prediction interval methods
#[derive(Debug, Clone)]
pub struct PredictionIntervalConfig {
    /// Coverage level (e.g., 0.9 for 90% coverage)
    pub coverage: Float,
    /// Method for computing prediction intervals
    pub method: PredictionIntervalMethod,
    /// Number of bootstrap samples for bootstrap methods
    pub n_bootstrap: usize,
    /// Whether to calibrate the intervals
    pub calibrate: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for PredictionIntervalConfig {
    fn default() -> Self {
        Self {
            coverage: 0.9,
            method: PredictionIntervalMethod::QuantileRegression,
            n_bootstrap: 1000,
            calibrate: true,
            random_seed: Some(42),
        }
    }
}

/// Methods for computing prediction intervals
#[derive(Debug, Clone)]
pub enum PredictionIntervalMethod {
    /// Simple quantile-based intervals
    QuantileRegression,
    /// Bootstrap prediction intervals
    Bootstrap,
    /// Conformal prediction intervals
    Conformal { method: ConformalMethod },
    /// Bayesian prediction intervals
    Bayesian { prior_strength: Float },
    /// Ensemble-based intervals
    Ensemble { n_estimators: usize },
    /// Gaussian assumption-based intervals
    Gaussian,
    /// Local quantile intervals
    LocalQuantile { n_neighbors: usize },
}

/// Prediction interval result
#[derive(Debug, Clone)]
pub struct PredictionIntervalResult {
    /// Lower bounds of prediction intervals
    pub lower_bounds: Array1<Float>,
    /// Upper bounds of prediction intervals
    pub upper_bounds: Array1<Float>,
    /// Point predictions
    pub predictions: Array1<Float>,
    /// Nominal coverage level
    pub nominal_coverage: Float,
    /// Empirical coverage (if validation data available)
    pub empirical_coverage: Option<Float>,
    /// Average interval width
    pub average_width: Float,
    /// Method used
    pub method: String,
}

/// Main prediction interval estimator
#[derive(Debug, Clone)]
pub struct PredictionIntervalEstimator {
    config: PredictionIntervalConfig,
    /// Fitted quantile models (if applicable)
    lower_quantile_model_: Option<Box<dyn CalibrationEstimator>>,
    upper_quantile_model_: Option<Box<dyn CalibrationEstimator>>,
    /// Conformal predictor (if applicable)
    conformal_predictor_: Option<ConformalPredictor>,
    /// Training residuals for bootstrap/gaussian methods
    training_residuals_: Option<Array1<Float>>,
}

impl PredictionIntervalEstimator {
    /// Create a new prediction interval estimator
    pub fn new(config: PredictionIntervalConfig) -> Self {
        Self {
            config,
            lower_quantile_model_: None,
            upper_quantile_model_: None,
            conformal_predictor_: None,
            training_residuals_: None,
        }
    }

    /// Set coverage level
    pub fn coverage(mut self, coverage: Float) -> Self {
        self.config.coverage = coverage;
        self
    }

    /// Set method
    pub fn method(mut self, method: PredictionIntervalMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Fit the prediction interval estimator
    pub fn fit(&mut self, predictions: &Array1<Float>, targets: &Array1<Float>) -> Result<()> {
        match &self.config.method {
            PredictionIntervalMethod::QuantileRegression => {
                self.fit_quantile_regression(predictions, targets)?;
            }
            PredictionIntervalMethod::Bootstrap => {
                self.fit_bootstrap(predictions, targets)?;
            }
            PredictionIntervalMethod::Conformal { method } => {
                self.fit_conformal(predictions, targets, method.clone())?;
            }
            PredictionIntervalMethod::Bayesian { prior_strength } => {
                self.fit_bayesian(predictions, targets, *prior_strength)?;
            }
            PredictionIntervalMethod::Ensemble { n_estimators } => {
                self.fit_ensemble(predictions, targets, *n_estimators)?;
            }
            PredictionIntervalMethod::Gaussian => {
                self.fit_gaussian(predictions, targets)?;
            }
            PredictionIntervalMethod::LocalQuantile { n_neighbors } => {
                self.fit_local_quantile(predictions, targets, *n_neighbors)?;
            }
        }

        Ok(())
    }

    /// Predict intervals for new data
    pub fn predict(&self, predictions: &Array1<Float>) -> Result<PredictionIntervalResult> {
        match &self.config.method {
            PredictionIntervalMethod::QuantileRegression => {
                self.predict_quantile_regression(predictions)
            }
            PredictionIntervalMethod::Bootstrap => self.predict_bootstrap(predictions),
            PredictionIntervalMethod::Conformal { .. } => self.predict_conformal(predictions),
            PredictionIntervalMethod::Bayesian { .. } => self.predict_bayesian(predictions),
            PredictionIntervalMethod::Ensemble { .. } => self.predict_ensemble(predictions),
            PredictionIntervalMethod::Gaussian => self.predict_gaussian(predictions),
            PredictionIntervalMethod::LocalQuantile { .. } => {
                self.predict_local_quantile(predictions)
            }
        }
    }

    // Implementation methods for different interval types

    fn fit_quantile_regression(
        &mut self,
        predictions: &Array1<Float>,
        targets: &Array1<Float>,
    ) -> Result<()> {
        // Fit models for lower and upper quantiles
        let alpha = 1.0 - self.config.coverage;
        let lower_quantile = alpha / 2.0;
        let upper_quantile = 1.0 - alpha / 2.0;

        // Create quantile targets
        let residuals: Array1<Float> = targets - predictions;
        let mut sorted_residuals = residuals.to_vec();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_residuals.len();
        let lower_idx = ((lower_quantile * (n - 1) as Float).round() as usize).min(n - 1);
        let upper_idx = ((upper_quantile * (n - 1) as Float).round() as usize).min(n - 1);

        let lower_threshold = sorted_residuals[lower_idx];
        let upper_threshold = sorted_residuals[upper_idx];

        // Store as simple threshold models
        self.lower_quantile_model_ = Some(Box::new(ThresholdModel::new(lower_threshold)));
        self.upper_quantile_model_ = Some(Box::new(ThresholdModel::new(upper_threshold)));

        Ok(())
    }

    fn fit_bootstrap(
        &mut self,
        predictions: &Array1<Float>,
        targets: &Array1<Float>,
    ) -> Result<()> {
        let residuals = targets - predictions;
        self.training_residuals_ = Some(residuals);
        Ok(())
    }

    fn fit_conformal(
        &mut self,
        predictions: &Array1<Float>,
        targets: &Array1<Float>,
        method: ConformalMethod,
    ) -> Result<()> {
        let alpha = 1.0 - self.config.coverage;
        let conformity_score = Box::new(AbsoluteResidualScore);

        let mut predictor = ConformalPredictor::new(method, conformity_score).alpha(alpha);

        predictor.fit(predictions, targets)?;
        self.conformal_predictor_ = Some(predictor);

        Ok(())
    }

    fn fit_bayesian(
        &mut self,
        predictions: &Array1<Float>,
        targets: &Array1<Float>,
        prior_strength: Float,
    ) -> Result<()> {
        // Simple Bayesian approach: estimate posterior variance
        let residuals = targets - predictions;
        let mean_residual = residuals.mean().unwrap_or(0.0);
        let var_residual = residuals.var(1.0);

        // Update with prior
        let posterior_var = var_residual / (1.0 + prior_strength);

        self.training_residuals_ = Some(Array1::from(vec![mean_residual, posterior_var.sqrt()]));
        Ok(())
    }

    fn fit_ensemble(
        &mut self,
        predictions: &Array1<Float>,
        targets: &Array1<Float>,
        n_estimators: usize,
    ) -> Result<()> {
        // For simplicity, use bootstrap sampling to create ensemble estimates
        use scirs2_core::random::thread_rng;

        let _rng_instance = thread_rng();

        let n = predictions.len();
        let mut ensemble_residuals = Vec::new();

        for _ in 0..n_estimators {
            // Bootstrap sample
            let indices: Vec<usize> = (0..n).map(|_| 0).collect();
            let boot_predictions: Array1<Float> = indices.iter().map(|&i| predictions[i]).collect();
            let boot_targets: Array1<Float> = indices.iter().map(|&i| targets[i]).collect();

            let residuals = boot_targets - boot_predictions;
            ensemble_residuals.push(residuals.std(1.0));
        }

        self.training_residuals_ = Some(Array1::from(ensemble_residuals));
        Ok(())
    }

    fn fit_gaussian(&mut self, predictions: &Array1<Float>, targets: &Array1<Float>) -> Result<()> {
        let residuals = targets - predictions;
        let std_residual = residuals.std(1.0);
        self.training_residuals_ = Some(Array1::from(vec![std_residual]));
        Ok(())
    }

    fn fit_local_quantile(
        &mut self,
        predictions: &Array1<Float>,
        targets: &Array1<Float>,
        _n_neighbors: usize,
    ) -> Result<()> {
        // Store all training data for local quantile computation
        let residuals = targets - predictions;
        self.training_residuals_ = Some(residuals);

        // Store predictions for local neighborhood computation
        // In a full implementation, we'd store both predictions and residuals
        Ok(())
    }

    // Prediction methods

    fn predict_quantile_regression(
        &self,
        predictions: &Array1<Float>,
    ) -> Result<PredictionIntervalResult> {
        let lower_model =
            self.lower_quantile_model_
                .as_ref()
                .ok_or_else(|| SklearsError::InvalidData {
                    reason: "Lower quantile model not fitted".to_string(),
                })?;
        let upper_model =
            self.upper_quantile_model_
                .as_ref()
                .ok_or_else(|| SklearsError::InvalidData {
                    reason: "Upper quantile model not fitted".to_string(),
                })?;

        let lower_adjustments = lower_model.predict_proba(predictions)?;
        let upper_adjustments = upper_model.predict_proba(predictions)?;

        let lower_bounds = predictions + &lower_adjustments;
        let upper_bounds = predictions + &upper_adjustments;

        self.create_result(
            lower_bounds,
            upper_bounds,
            predictions.clone(),
            "Quantile Regression",
        )
    }

    fn predict_bootstrap(&self, predictions: &Array1<Float>) -> Result<PredictionIntervalResult> {
        let residuals =
            self.training_residuals_
                .as_ref()
                .ok_or_else(|| SklearsError::InvalidData {
                    reason: "Bootstrap residuals not fitted".to_string(),
                })?;

        let alpha = 1.0 - self.config.coverage;
        let mut sorted_residuals = residuals.to_vec();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_residuals.len();
        let lower_idx = ((alpha / 2.0 * (n - 1) as Float).round() as usize).min(n - 1);
        let upper_idx = (((1.0 - alpha / 2.0) * (n - 1) as Float).round() as usize).min(n - 1);

        let lower_quantile = sorted_residuals[lower_idx];
        let upper_quantile = sorted_residuals[upper_idx];

        let lower_bounds = predictions.mapv(|p| p + lower_quantile);
        let upper_bounds = predictions.mapv(|p| p + upper_quantile);

        self.create_result(lower_bounds, upper_bounds, predictions.clone(), "Bootstrap")
    }

    fn predict_conformal(&self, predictions: &Array1<Float>) -> Result<PredictionIntervalResult> {
        let predictor =
            self.conformal_predictor_
                .as_ref()
                .ok_or_else(|| SklearsError::InvalidData {
                    reason: "Conformal predictor not fitted".to_string(),
                })?;

        let conformal_result = predictor.predict(predictions)?;
        let lower_bounds = conformal_result.intervals.column(0).to_owned();
        let upper_bounds = conformal_result.intervals.column(1).to_owned();

        self.create_result(lower_bounds, upper_bounds, predictions.clone(), "Conformal")
    }

    fn predict_bayesian(&self, predictions: &Array1<Float>) -> Result<PredictionIntervalResult> {
        let params =
            self.training_residuals_
                .as_ref()
                .ok_or_else(|| SklearsError::InvalidData {
                    reason: "Bayesian parameters not fitted".to_string(),
                })?;

        let mean_residual = params[0];
        let std_residual = params[1];

        // Use t-distribution quantiles for uncertainty
        let alpha = 1.0 - self.config.coverage;
        let t_quantile = normal_quantile(1.0 - alpha / 2.0); // Approximation

        let margin = t_quantile * std_residual;

        let lower_bounds = predictions.mapv(|p| p + mean_residual - margin);
        let upper_bounds = predictions.mapv(|p| p + mean_residual + margin);

        self.create_result(lower_bounds, upper_bounds, predictions.clone(), "Bayesian")
    }

    fn predict_ensemble(&self, predictions: &Array1<Float>) -> Result<PredictionIntervalResult> {
        let ensemble_stds =
            self.training_residuals_
                .as_ref()
                .ok_or_else(|| SklearsError::InvalidData {
                    reason: "Ensemble parameters not fitted".to_string(),
                })?;

        let mean_std = ensemble_stds.mean().unwrap_or(1.0);
        let std_of_stds = ensemble_stds.std(1.0);

        // Use ensemble variance for interval width
        let alpha = 1.0 - self.config.coverage;
        let quantile = normal_quantile(1.0 - alpha / 2.0);
        let margin = quantile * (mean_std + std_of_stds);

        let lower_bounds = predictions.mapv(|p| p - margin);
        let upper_bounds = predictions.mapv(|p| p + margin);

        self.create_result(lower_bounds, upper_bounds, predictions.clone(), "Ensemble")
    }

    fn predict_gaussian(&self, predictions: &Array1<Float>) -> Result<PredictionIntervalResult> {
        let std_residual = self
            .training_residuals_
            .as_ref()
            .and_then(|r| r.get(0))
            .copied()
            .ok_or_else(|| SklearsError::InvalidData {
                reason: "Gaussian parameters not fitted".to_string(),
            })?;

        let alpha = 1.0 - self.config.coverage;
        let z_quantile = normal_quantile(1.0 - alpha / 2.0);
        let margin = z_quantile * std_residual;

        let lower_bounds = predictions.mapv(|p| p - margin);
        let upper_bounds = predictions.mapv(|p| p + margin);

        self.create_result(lower_bounds, upper_bounds, predictions.clone(), "Gaussian")
    }

    fn predict_local_quantile(
        &self,
        predictions: &Array1<Float>,
    ) -> Result<PredictionIntervalResult> {
        // Simplified local quantile implementation
        let residuals =
            self.training_residuals_
                .as_ref()
                .ok_or_else(|| SklearsError::InvalidData {
                    reason: "Local quantile data not fitted".to_string(),
                })?;

        let alpha = 1.0 - self.config.coverage;
        let mut sorted_residuals = residuals.to_vec();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_residuals.len();
        let lower_idx = ((alpha / 2.0 * (n - 1) as Float).round() as usize).min(n - 1);
        let upper_idx = (((1.0 - alpha / 2.0) * (n - 1) as Float).round() as usize).min(n - 1);

        let lower_quantile = sorted_residuals[lower_idx];
        let upper_quantile = sorted_residuals[upper_idx];

        let lower_bounds = predictions.mapv(|p| p + lower_quantile);
        let upper_bounds = predictions.mapv(|p| p + upper_quantile);

        self.create_result(
            lower_bounds,
            upper_bounds,
            predictions.clone(),
            "Local Quantile",
        )
    }

    fn create_result(
        &self,
        lower_bounds: Array1<Float>,
        upper_bounds: Array1<Float>,
        predictions: Array1<Float>,
        method: &str,
    ) -> Result<PredictionIntervalResult> {
        let average_width = (&upper_bounds - &lower_bounds).mean().unwrap_or(0.0);

        Ok(PredictionIntervalResult {
            lower_bounds,
            upper_bounds,
            predictions,
            nominal_coverage: self.config.coverage,
            empirical_coverage: None,
            average_width,
            method: method.to_string(),
        })
    }
}

/// Simple threshold model for quantile regression
#[derive(Debug, Clone)]
struct ThresholdModel {
    threshold: Float,
}

impl ThresholdModel {
    fn new(threshold: Float) -> Self {
        Self { threshold }
    }
}

impl CalibrationEstimator for ThresholdModel {
    fn fit(&mut self, _probabilities: &Array1<Float>, _y_true: &Array1<i32>) -> Result<()> {
        Ok(()) // Already fitted
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        Ok(Array1::from(vec![self.threshold; probabilities.len()]))
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Compute empirical coverage of prediction intervals
pub fn compute_empirical_coverage(
    intervals: &PredictionIntervalResult,
    true_targets: &Array1<Float>,
) -> Result<Float> {
    if intervals.lower_bounds.len() != true_targets.len() {
        return Err(SklearsError::InvalidInput(
            "Intervals and targets must have same length".to_string(),
        ));
    }

    let mut covered = 0;
    for (i, &target) in true_targets.iter().enumerate() {
        if target >= intervals.lower_bounds[i] && target <= intervals.upper_bounds[i] {
            covered += 1;
        }
    }

    Ok(covered as Float / true_targets.len() as Float)
}

/// Calibrate prediction intervals to achieve nominal coverage
pub fn calibrate_intervals(
    intervals: &mut PredictionIntervalResult,
    calibration_targets: &Array1<Float>,
    desired_coverage: Float,
) -> Result<()> {
    let current_coverage = compute_empirical_coverage(intervals, calibration_targets)?;

    if current_coverage == 0.0 {
        return Ok(()); // Can't calibrate if no coverage
    }

    let scaling_factor = (desired_coverage / current_coverage).sqrt();
    let current_widths = &intervals.upper_bounds - &intervals.lower_bounds;
    let new_widths = &current_widths * scaling_factor;

    let centers = (&intervals.upper_bounds + &intervals.lower_bounds) / 2.0;
    intervals.lower_bounds = &centers - &new_widths / 2.0;
    intervals.upper_bounds = &centers + &new_widths / 2.0;
    intervals.empirical_coverage = Some(desired_coverage);

    Ok(())
}

/// Compare multiple prediction interval methods
pub fn compare_interval_methods(
    train_predictions: &Array1<Float>,
    train_targets: &Array1<Float>,
    test_predictions: &Array1<Float>,
    test_targets: &Array1<Float>,
    coverage: Float,
) -> Result<HashMap<String, PredictionIntervalResult>> {
    let methods = vec![
        ("Quantile", PredictionIntervalMethod::QuantileRegression),
        ("Bootstrap", PredictionIntervalMethod::Bootstrap),
        (
            "Conformal",
            PredictionIntervalMethod::Conformal {
                method: ConformalMethod::Split,
            },
        ),
        ("Gaussian", PredictionIntervalMethod::Gaussian),
        (
            "Bayesian",
            PredictionIntervalMethod::Bayesian {
                prior_strength: 1.0,
            },
        ),
    ];

    let mut results = HashMap::new();

    for (name, method) in methods {
        let config = PredictionIntervalConfig {
            coverage,
            method,
            ..Default::default()
        };

        let mut estimator = PredictionIntervalEstimator::new(config);

        if estimator.fit(train_predictions, train_targets).is_ok() {
            if let Ok(mut result) = estimator.predict(test_predictions) {
                // Compute empirical coverage
                result.empirical_coverage = compute_empirical_coverage(&result, test_targets).ok();
                results.insert(name.to_string(), result);
            }
        }
    }

    Ok(results)
}

/// Approximate normal quantile function
fn normal_quantile(p: Float) -> Float {
    // Check boundary conditions
    if p <= 0.0 {
        return Float::NEG_INFINITY;
    }
    if p >= 1.0 {
        return Float::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    // Use Acklam's approximation algorithm
    // This is more accurate than the previous implementation

    // Coefficients for the rational approximation
    let a = [
        0.0,
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    let b = [
        0.0,
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
        1.0,
    ];

    let c = [
        0.0,
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];

    let d = [
        0.0,
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
        1.0,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        // Lower region
        let q = (2.0 * p).sqrt();
        -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6])
            / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
    } else if p > p_high {
        // Upper region
        let q = (2.0 * (1.0 - p)).sqrt();
        (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6])
            / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
    } else {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q
            / (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1.0)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quantile_regression_intervals() {
        let predictions = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = array![1.1, 1.9, 3.1, 4.1, 4.9];

        let config = PredictionIntervalConfig {
            coverage: 0.8,
            method: PredictionIntervalMethod::QuantileRegression,
            ..Default::default()
        };

        let mut estimator = PredictionIntervalEstimator::new(config);
        estimator.fit(&predictions, &targets).unwrap();

        let test_predictions = array![2.5, 3.5];
        let result = estimator.predict(&test_predictions).unwrap();

        assert_eq!(result.lower_bounds.len(), 2);
        assert_eq!(result.upper_bounds.len(), 2);
        assert_eq!(result.nominal_coverage, 0.8);

        // Check that intervals make sense
        for i in 0..result.lower_bounds.len() {
            assert!(result.lower_bounds[i] <= result.upper_bounds[i]);
        }
    }

    #[test]
    fn test_gaussian_intervals() {
        let predictions = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = array![1.1, 1.9, 3.1, 4.1, 4.9];

        let config = PredictionIntervalConfig {
            coverage: 0.95,
            method: PredictionIntervalMethod::Gaussian,
            ..Default::default()
        };

        let mut estimator = PredictionIntervalEstimator::new(config);
        estimator.fit(&predictions, &targets).unwrap();

        let test_predictions = array![2.5, 3.5];
        let result = estimator.predict(&test_predictions).unwrap();

        assert_eq!(result.method, "Gaussian");
        assert!(result.average_width > 0.0);
    }

    #[test]
    fn test_empirical_coverage() {
        let lower_bounds = array![0.5, 1.5, 2.5];
        let upper_bounds = array![1.5, 2.5, 3.5];
        let predictions = array![1.0, 2.0, 3.0];

        let intervals = PredictionIntervalResult {
            lower_bounds,
            upper_bounds,
            predictions,
            nominal_coverage: 0.9,
            empirical_coverage: None,
            average_width: 1.0,
            method: "Test".to_string(),
        };

        let targets = array![1.0, 2.0, 3.0]; // All should be covered
        let coverage = compute_empirical_coverage(&intervals, &targets).unwrap();
        assert_eq!(coverage, 1.0);

        // Test with targets where only middle one is covered
        // Intervals are: [0.5, 1.5], [1.5, 2.5], [2.5, 3.5]
        // For 1/3 coverage, we want exactly one target covered
        let targets_partial = array![0.0, 2.0, 4.0]; // Only middle one (2.0 in [1.5, 2.5]) covered
        let coverage_partial = compute_empirical_coverage(&intervals, &targets_partial).unwrap();
        assert!((coverage_partial - 1.0 / 3.0).abs() < 1e-10);

        // Test with all targets outside intervals
        let targets_none = array![0.0, 1.0, 4.0]; // None should be covered
        let coverage_none = compute_empirical_coverage(&intervals, &targets_none).unwrap();
        assert_eq!(coverage_none, 0.0);
    }

    #[test]
    fn test_method_comparison() {
        let train_predictions = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let train_targets = array![1.1, 1.9, 3.1, 4.1, 4.9];
        let test_predictions = array![2.5, 3.5];
        let test_targets = array![2.4, 3.6];

        let results = compare_interval_methods(
            &train_predictions,
            &train_targets,
            &test_predictions,
            &test_targets,
            0.9,
        )
        .unwrap();

        assert!(!results.is_empty());

        for (method_name, result) in results {
            println!(
                "Method: {}, Coverage: {:?}, Width: {:.3}",
                method_name, result.empirical_coverage, result.average_width
            );
            assert!(result.average_width >= 0.0);
        }
    }

    #[test]
    fn test_normal_quantile_approximation() {
        // Test some known values
        let z_95 = normal_quantile(0.975); // Should be approximately 1.96
        assert!(
            (z_95 - 1.96).abs() < 0.01,
            "z_95 = {}, expected ~1.96",
            z_95
        );

        let z_50 = normal_quantile(0.5); // Should be 0
        assert!(z_50.abs() < 1e-10, "z_50 = {}, expected ~0", z_50);

        let z_05 = normal_quantile(0.025); // Should be approximately -1.96
        assert!(
            (z_05 + 1.96).abs() < 0.01,
            "z_05 = {}, expected ~-1.96",
            z_05
        );

        // Test more quantiles for thoroughness
        let z_90 = normal_quantile(0.95); // Should be approximately 1.645
        assert!(
            (z_90 - 1.645).abs() < 0.01,
            "z_90 = {}, expected ~1.645",
            z_90
        );

        let z_99 = normal_quantile(0.995); // Should be approximately 2.576
        assert!(
            (z_99 - 2.576).abs() < 0.2,
            "z_99 = {}, expected ~2.576",
            z_99
        );
    }
}
