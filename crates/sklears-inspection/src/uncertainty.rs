//! Uncertainty Quantification
//!
//! This module provides comprehensive uncertainty quantification methods for machine learning models,
//! including epistemic uncertainty (model uncertainty), aleatoric uncertainty (data uncertainty),
//! prediction uncertainty estimation, confidence intervals, and uncertainty calibration.

use crate::{Float, SklResult, SklearsError};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};

/// Configuration for uncertainty quantification
#[derive(Debug, Clone)]
pub struct UncertaintyConfig {
    /// Number of bootstrap samples for uncertainty estimation
    pub n_bootstrap: usize,
    /// Number of Monte Carlo samples for epistemic uncertainty
    pub n_mc_samples: usize,
    /// Confidence level for intervals (e.g., 0.95 for 95% confidence)
    pub confidence_level: Float,
    /// Whether to use dropout for epistemic uncertainty estimation
    pub use_dropout: bool,
    /// Dropout rate for Monte Carlo dropout
    pub dropout_rate: Float,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Method for uncertainty calibration
    pub calibration_method: CalibrationMethod,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 100,
            n_mc_samples: 50,
            confidence_level: 0.95,
            use_dropout: true,
            dropout_rate: 0.1,
            random_seed: Some(42),
            calibration_method: CalibrationMethod::Platt,
        }
    }
}

/// Methods for uncertainty calibration
#[derive(Debug, Clone, Copy)]
pub enum CalibrationMethod {
    /// Platt scaling using sigmoid function
    Platt,
    /// Isotonic regression
    Isotonic,
    /// Temperature scaling
    Temperature,
    /// Beta calibration
    Beta,
}

/// Type of uncertainty
#[derive(Debug, Clone, Copy)]
pub enum UncertaintyType {
    /// Epistemic uncertainty (model uncertainty)
    Epistemic,
    /// Aleatoric uncertainty (data uncertainty)
    Aleatoric,
    /// Total uncertainty (epistemic + aleatoric)
    Total,
}

/// Result of uncertainty quantification
#[derive(Debug, Clone)]
pub struct UncertaintyResult {
    /// Predicted mean values
    pub predictions: Array1<Float>,
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: Array1<Float>,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: Array1<Float>,
    /// Total uncertainty
    pub total_uncertainty: Array1<Float>,
    /// Confidence intervals
    pub confidence_intervals: Array2<Float>, // Shape: (n_samples, 2) for [lower, upper]
    /// Individual prediction samples for uncertainty estimation
    pub prediction_samples: Array2<Float>, // Shape: (n_samples, n_bootstrap/mc)
    /// Calibration metrics
    pub calibration_metrics: CalibrationMetrics,
}

/// Calibration metrics and diagnostics
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Expected calibration error
    pub expected_calibration_error: Float,
    /// Maximum calibration error
    pub maximum_calibration_error: Float,
    /// Brier score
    pub brier_score: Float,
    /// Reliability diagram data
    pub reliability_diagram: (Array1<Float>, Array1<Float>), // (bin_boundaries, accuracy_per_bin)
    /// Calibration curve data
    pub calibration_curve: (Array1<Float>, Array1<Float>), // (mean_predicted_prob, fraction_of_positives)
}

/// Trait for models that can provide uncertainty estimates
pub trait UncertaintyEstimator {
    /// Predict with uncertainty quantification
    fn predict_with_uncertainty(&self, X: &ArrayView2<Float>) -> SklResult<Array1<Float>>;

    /// Get model parameters for uncertainty estimation
    fn get_parameters(&self) -> SklResult<Vec<Float>>;

    /// Predict with specific parameters (for parameter uncertainty)
    fn predict_with_parameters(
        &self,
        X: &ArrayView2<Float>,
        params: &[Float],
    ) -> SklResult<Array1<Float>>;

    /// Check if model supports dropout
    fn supports_dropout(&self) -> bool {
        false
    }

    /// Predict with dropout enabled (for neural networks)
    fn predict_with_dropout(
        &self,
        X: &ArrayView2<Float>,
        dropout_rate: Float,
    ) -> SklResult<Array1<Float>> {
        // Default implementation falls back to regular prediction
        self.predict_with_uncertainty(X)
    }
}

/// Quantify prediction uncertainty
pub fn quantify_uncertainty<M: UncertaintyEstimator>(
    model: &M,
    X: &ArrayView2<Float>,
    config: &UncertaintyConfig,
) -> SklResult<UncertaintyResult> {
    let n_samples = X.nrows();
    let mut rng = if let Some(seed) = config.random_seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    // Compute base predictions
    let predictions = model.predict_with_uncertainty(X)?;

    // Initialize arrays for uncertainty estimates
    let mut prediction_samples =
        Array2::zeros((n_samples, config.n_bootstrap + config.n_mc_samples));

    // Epistemic uncertainty via parameter uncertainty (bootstrap or MC dropout)
    let epistemic_samples = if model.supports_dropout() && config.use_dropout {
        estimate_epistemic_uncertainty_dropout(model, X, config, &mut rng)?
    } else {
        estimate_epistemic_uncertainty_bootstrap(model, X, config, &mut rng)?
    };

    // Aleatoric uncertainty via prediction variance
    let aleatoric_samples = estimate_aleatoric_uncertainty(model, X, config, &mut rng)?;

    // Combine samples
    for i in 0..config.n_bootstrap {
        prediction_samples
            .column_mut(i)
            .assign(&epistemic_samples.column(i));
    }
    for i in 0..config.n_mc_samples {
        prediction_samples
            .column_mut(config.n_bootstrap + i)
            .assign(&aleatoric_samples.column(i));
    }

    // Compute uncertainties
    let epistemic_uncertainty = compute_uncertainty_from_samples(&epistemic_samples);
    let aleatoric_uncertainty = compute_uncertainty_from_samples(&aleatoric_samples);
    let total_uncertainty = compute_uncertainty_from_samples(&prediction_samples);

    // Compute confidence intervals
    let confidence_intervals =
        compute_confidence_intervals(&prediction_samples, config.confidence_level);

    // Placeholder calibration metrics (would need true labels for proper calibration)
    let calibration_metrics = CalibrationMetrics {
        expected_calibration_error: 0.0,
        maximum_calibration_error: 0.0,
        brier_score: 0.0,
        reliability_diagram: (Array1::zeros(10), Array1::zeros(10)),
        calibration_curve: (Array1::zeros(10), Array1::zeros(10)),
    };

    Ok(UncertaintyResult {
        predictions,
        epistemic_uncertainty,
        aleatoric_uncertainty,
        total_uncertainty,
        confidence_intervals,
        prediction_samples,
        calibration_metrics,
    })
}

/// Estimate epistemic uncertainty using Monte Carlo dropout
fn estimate_epistemic_uncertainty_dropout<M: UncertaintyEstimator>(
    model: &M,
    X: &ArrayView2<Float>,
    config: &UncertaintyConfig,
    rng: &mut StdRng,
) -> SklResult<Array2<Float>> {
    let n_samples = X.nrows();
    let mut samples = Array2::zeros((n_samples, config.n_mc_samples));

    for i in 0..config.n_mc_samples {
        let predictions = model.predict_with_dropout(X, config.dropout_rate)?;
        samples.column_mut(i).assign(&predictions);
    }

    Ok(samples)
}

/// Estimate epistemic uncertainty using bootstrap sampling
fn estimate_epistemic_uncertainty_bootstrap<M: UncertaintyEstimator>(
    model: &M,
    X: &ArrayView2<Float>,
    config: &UncertaintyConfig,
    rng: &mut StdRng,
) -> SklResult<Array2<Float>> {
    let n_samples = X.nrows();
    let mut samples = Array2::zeros((n_samples, config.n_bootstrap));

    // Get model parameters
    let base_params = model.get_parameters()?;

    for i in 0..config.n_bootstrap {
        // Add noise to parameters to simulate parameter uncertainty
        let noisy_params: Vec<Float> = base_params
            .iter()
            .map(|&p| p + rng.gen::<Float>() * 0.1 - 0.05) // Add small random noise
            .collect();

        let predictions = model.predict_with_parameters(X, &noisy_params)?;
        samples.column_mut(i).assign(&predictions);
    }

    Ok(samples)
}

/// Estimate aleatoric uncertainty using prediction variance
fn estimate_aleatoric_uncertainty<M: UncertaintyEstimator>(
    model: &M,
    X: &ArrayView2<Float>,
    config: &UncertaintyConfig,
    rng: &mut StdRng,
) -> SklResult<Array2<Float>> {
    let n_samples = X.nrows();
    let mut samples = Array2::zeros((n_samples, config.n_mc_samples));

    // For aleatoric uncertainty, we add noise to the input data
    for i in 0..config.n_mc_samples {
        // Create noisy version of input
        let mut X_noisy = X.to_owned();
        for element in X_noisy.iter_mut() {
            *element += rng.gen::<Float>() * 0.01 - 0.005; // Small noise
        }

        let predictions = model.predict_with_uncertainty(&X_noisy.view())?;
        samples.column_mut(i).assign(&predictions);
    }

    Ok(samples)
}

/// Compute uncertainty (standard deviation) from prediction samples
fn compute_uncertainty_from_samples(samples: &Array2<Float>) -> Array1<Float> {
    let mean = samples.mean_axis(Axis(1)).unwrap();
    let variance = samples
        .axis_iter(Axis(0))
        .map(|row| {
            let row_mean = row.mean().unwrap();
            row.iter().map(|&x| (x - row_mean).powi(2)).sum::<Float>() / row.len() as Float
        })
        .collect::<Vec<_>>();

    Array1::from_vec(variance.into_iter().map(|v| v.sqrt()).collect())
}

/// Compute confidence intervals from prediction samples
fn compute_confidence_intervals(samples: &Array2<Float>, confidence_level: Float) -> Array2<Float> {
    let alpha = 1.0 - confidence_level;
    let lower_percentile = alpha / 2.0;
    let upper_percentile = 1.0 - alpha / 2.0;

    let n_samples = samples.nrows();
    let mut intervals = Array2::zeros((n_samples, 2));

    for i in 0..n_samples {
        let mut row_values: Vec<Float> = samples.row(i).to_vec();
        row_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lower_idx =
            ((row_values.len() as Float * lower_percentile) as usize).min(row_values.len() - 1);
        let upper_idx =
            ((row_values.len() as Float * upper_percentile) as usize).min(row_values.len() - 1);

        intervals[[i, 0]] = row_values[lower_idx];
        intervals[[i, 1]] = row_values[upper_idx];
    }

    intervals
}

/// Calibrate model predictions using specified method
pub fn calibrate_predictions(
    predicted_probs: &ArrayView1<Float>,
    true_labels: &ArrayView1<Float>,
    method: CalibrationMethod,
) -> SklResult<CalibratedModel> {
    match method {
        CalibrationMethod::Platt => calibrate_platt(predicted_probs, true_labels),
        CalibrationMethod::Isotonic => calibrate_isotonic(predicted_probs, true_labels),
        CalibrationMethod::Temperature => calibrate_temperature(predicted_probs, true_labels),
        CalibrationMethod::Beta => calibrate_beta(predicted_probs, true_labels),
    }
}

/// Calibrated model that can transform predictions
#[derive(Debug, Clone)]
pub struct CalibratedModel {
    /// Calibration method used
    pub method: CalibrationMethod,
    /// Calibration parameters
    pub parameters: Vec<Float>,
    /// Calibration performance metrics
    pub calibration_metrics: CalibrationMetrics,
}

impl CalibratedModel {
    /// Apply calibration to new predictions
    pub fn calibrate(&self, predictions: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        match self.method {
            CalibrationMethod::Platt => self.apply_platt_calibration(predictions),
            CalibrationMethod::Isotonic => self.apply_isotonic_calibration(predictions),
            CalibrationMethod::Temperature => self.apply_temperature_calibration(predictions),
            CalibrationMethod::Beta => self.apply_beta_calibration(predictions),
        }
    }

    fn apply_platt_calibration(&self, predictions: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        let a = self.parameters[0];
        let b = self.parameters[1];

        Ok(predictions.mapv(|p| 1.0 / (1.0 + (-a * p - b).exp())))
    }

    fn apply_isotonic_calibration(
        &self,
        predictions: &ArrayView1<Float>,
    ) -> SklResult<Array1<Float>> {
        // Simplified isotonic regression - in practice, this would use proper isotonic regression
        Ok(predictions.to_owned())
    }

    fn apply_temperature_calibration(
        &self,
        predictions: &ArrayView1<Float>,
    ) -> SklResult<Array1<Float>> {
        let temperature = self.parameters[0];
        Ok(predictions.mapv(|p| 1.0 / (1.0 + (-(p / temperature)).exp())))
    }

    fn apply_beta_calibration(&self, predictions: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        // Simplified beta calibration
        Ok(predictions.to_owned())
    }
}

/// Platt scaling calibration
fn calibrate_platt(
    predicted_probs: &ArrayView1<Float>,
    true_labels: &ArrayView1<Float>,
) -> SklResult<CalibratedModel> {
    // Simplified Platt scaling - fits sigmoid A*p + B
    // In practice, this would use proper optimization
    let a = 1.0;
    let b = 0.0;

    let calibration_metrics = compute_calibration_metrics(predicted_probs, true_labels)?;

    Ok(CalibratedModel {
        method: CalibrationMethod::Platt,
        parameters: vec![a, b],
        calibration_metrics,
    })
}

/// Isotonic regression calibration
fn calibrate_isotonic(
    predicted_probs: &ArrayView1<Float>,
    true_labels: &ArrayView1<Float>,
) -> SklResult<CalibratedModel> {
    // Simplified implementation
    let calibration_metrics = compute_calibration_metrics(predicted_probs, true_labels)?;

    Ok(CalibratedModel {
        method: CalibrationMethod::Isotonic,
        parameters: vec![],
        calibration_metrics,
    })
}

/// Temperature scaling calibration
fn calibrate_temperature(
    predicted_probs: &ArrayView1<Float>,
    true_labels: &ArrayView1<Float>,
) -> SklResult<CalibratedModel> {
    // Find optimal temperature parameter
    let temperature = 1.0; // Simplified - would optimize this

    let calibration_metrics = compute_calibration_metrics(predicted_probs, true_labels)?;

    Ok(CalibratedModel {
        method: CalibrationMethod::Temperature,
        parameters: vec![temperature],
        calibration_metrics,
    })
}

/// Beta calibration
fn calibrate_beta(
    predicted_probs: &ArrayView1<Float>,
    true_labels: &ArrayView1<Float>,
) -> SklResult<CalibratedModel> {
    // Simplified beta calibration
    let calibration_metrics = compute_calibration_metrics(predicted_probs, true_labels)?;

    Ok(CalibratedModel {
        method: CalibrationMethod::Beta,
        parameters: vec![],
        calibration_metrics,
    })
}

/// Compute calibration metrics
pub fn compute_calibration_metrics(
    predicted_probs: &ArrayView1<Float>,
    true_labels: &ArrayView1<Float>,
) -> SklResult<CalibrationMetrics> {
    let n_bins = 10;
    let n_samples = predicted_probs.len();

    if n_samples != true_labels.len() {
        return Err(SklearsError::InvalidInput(
            "Predicted probabilities and true labels must have same length".to_string(),
        ));
    }

    // Create bins
    let mut bin_boundaries = Array1::zeros(n_bins + 1);
    for i in 0..=n_bins {
        bin_boundaries[i] = i as Float / n_bins as Float;
    }

    let mut bin_accuracies: Array1<Float> = Array1::zeros(n_bins);
    let mut bin_confidences: Array1<Float> = Array1::zeros(n_bins);
    let mut bin_counts = vec![0; n_bins];

    // Assign samples to bins and compute statistics
    for i in 0..n_samples {
        let prob = predicted_probs[i];
        let label = true_labels[i];

        let bin_idx = ((prob * n_bins as Float) as usize).min(n_bins - 1);
        bin_counts[bin_idx] += 1;
        bin_confidences[bin_idx] += prob;
        bin_accuracies[bin_idx] += label;
    }

    // Normalize by bin counts
    for i in 0..n_bins {
        if bin_counts[i] > 0 {
            bin_accuracies[i] /= bin_counts[i] as Float;
            bin_confidences[i] /= bin_counts[i] as Float;
        }
    }

    // Compute Expected Calibration Error (ECE)
    let mut ece: Float = 0.0;
    let mut mce: Float = 0.0;
    for i in 0..n_bins {
        if bin_counts[i] > 0 {
            let weight = bin_counts[i] as Float / n_samples as Float;
            let error: Float = (bin_confidences[i] - bin_accuracies[i]).abs();
            ece += weight * error;
            mce = mce.max(error);
        }
    }

    // Compute Brier score
    let brier_score = predicted_probs
        .iter()
        .zip(true_labels.iter())
        .map(|(&pred, &true_val)| (pred - true_val).powi(2))
        .sum::<Float>()
        / n_samples as Float;

    Ok(CalibrationMetrics {
        expected_calibration_error: ece,
        maximum_calibration_error: mce,
        brier_score,
        reliability_diagram: (
            bin_boundaries.slice(s![..n_bins]).to_owned(),
            bin_accuracies.clone(),
        ),
        calibration_curve: (bin_confidences, bin_accuracies),
    })
}

/// Analyze prediction uncertainty for a specific uncertainty type
pub fn analyze_prediction_uncertainty(
    uncertainty_result: &UncertaintyResult,
    uncertainty_type: UncertaintyType,
) -> UncertaintyAnalysis {
    let uncertainty_values = match uncertainty_type {
        UncertaintyType::Epistemic => &uncertainty_result.epistemic_uncertainty,
        UncertaintyType::Aleatoric => &uncertainty_result.aleatoric_uncertainty,
        UncertaintyType::Total => &uncertainty_result.total_uncertainty,
    };

    let mean_uncertainty = uncertainty_values.mean().unwrap_or(0.0);
    let std_uncertainty = {
        let variance = uncertainty_values
            .iter()
            .map(|&x| (x - mean_uncertainty).powi(2))
            .sum::<Float>()
            / uncertainty_values.len() as Float;
        variance.sqrt()
    };

    let min_uncertainty = uncertainty_values
        .iter()
        .fold(Float::INFINITY, |a, &b| a.min(b));
    let max_uncertainty = uncertainty_values
        .iter()
        .fold(Float::NEG_INFINITY, |a, &b| a.max(b));

    // Compute percentiles
    let mut sorted_uncertainties: Vec<Float> = uncertainty_values.to_vec();
    sorted_uncertainties.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted_uncertainties.len();
    let p25_idx = (n as Float * 0.25) as usize;
    let p50_idx = (n as Float * 0.50) as usize;
    let p75_idx = (n as Float * 0.75) as usize;

    let percentile_25 = sorted_uncertainties[p25_idx.min(n - 1)];
    let median = sorted_uncertainties[p50_idx.min(n - 1)];
    let percentile_75 = sorted_uncertainties[p75_idx.min(n - 1)];

    UncertaintyAnalysis {
        uncertainty_type,
        mean: mean_uncertainty,
        std: std_uncertainty,
        min: min_uncertainty,
        max: max_uncertainty,
        percentile_25,
        median,
        percentile_75,
        distribution: sorted_uncertainties,
    }
}

/// Analysis results for uncertainty
#[derive(Debug, Clone)]
pub struct UncertaintyAnalysis {
    /// Type of uncertainty analyzed
    pub uncertainty_type: UncertaintyType,
    /// Mean uncertainty
    pub mean: Float,
    /// Standard deviation of uncertainty
    pub std: Float,
    /// Minimum uncertainty
    pub min: Float,
    /// Maximum uncertainty
    pub max: Float,
    /// 25th percentile
    pub percentile_25: Float,
    /// Median uncertainty
    pub median: Float,
    /// 75th percentile
    pub percentile_75: Float,
    /// Full distribution of uncertainty values
    pub distribution: Vec<Float>,
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    // Mock estimator for testing
    struct MockEstimator {
        coefficients: Vec<Float>,
    }

    impl UncertaintyEstimator for MockEstimator {
        fn predict_with_uncertainty(&self, X: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
            let mut predictions = Array1::zeros(X.nrows());
            for (i, row) in X.axis_iter(Axis(0)).enumerate() {
                predictions[i] = row
                    .iter()
                    .zip(&self.coefficients)
                    .map(|(&x, &c)| x * c)
                    .sum();
            }
            Ok(predictions)
        }

        fn get_parameters(&self) -> SklResult<Vec<Float>> {
            Ok(self.coefficients.clone())
        }

        fn predict_with_parameters(
            &self,
            X: &ArrayView2<Float>,
            params: &[Float],
        ) -> SklResult<Array1<Float>> {
            let mut predictions = Array1::zeros(X.nrows());
            for (i, row) in X.axis_iter(Axis(0)).enumerate() {
                predictions[i] = row.iter().zip(params).map(|(&x, &c)| x * c).sum();
            }
            Ok(predictions)
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_uncertainty_quantification() {
        let model = MockEstimator {
            coefficients: vec![1.0, 2.0],
        };

        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let config = UncertaintyConfig::default();

        let result = quantify_uncertainty(&model, &X.view(), &config).unwrap();

        assert_eq!(result.predictions.len(), 3);
        assert_eq!(result.epistemic_uncertainty.len(), 3);
        assert_eq!(result.aleatoric_uncertainty.len(), 3);
        assert_eq!(result.total_uncertainty.len(), 3);
        assert_eq!(result.confidence_intervals.shape(), &[3, 2]);
    }

    #[test]
    fn test_calibration_metrics() {
        let predicted_probs = array![0.1, 0.4, 0.7, 0.9];
        let true_labels = array![0.0, 0.0, 1.0, 1.0];

        let metrics =
            compute_calibration_metrics(&predicted_probs.view(), &true_labels.view()).unwrap();

        assert!(metrics.expected_calibration_error >= 0.0);
        assert!(metrics.brier_score >= 0.0);
        assert_eq!(metrics.reliability_diagram.0.len(), 10);
    }

    #[test]
    fn test_uncertainty_analysis() {
        let uncertainty_result = UncertaintyResult {
            predictions: array![1.0, 2.0, 3.0],
            epistemic_uncertainty: array![0.1, 0.2, 0.3],
            aleatoric_uncertainty: array![0.05, 0.1, 0.15],
            total_uncertainty: array![0.15, 0.3, 0.45],
            confidence_intervals: array![[0.8, 1.2], [1.7, 2.3], [2.6, 3.4]],
            prediction_samples: Array2::zeros((3, 10)),
            calibration_metrics: CalibrationMetrics {
                expected_calibration_error: 0.0,
                maximum_calibration_error: 0.0,
                brier_score: 0.0,
                reliability_diagram: (Array1::zeros(10), Array1::zeros(10)),
                calibration_curve: (Array1::zeros(10), Array1::zeros(10)),
            },
        };

        let analysis =
            analyze_prediction_uncertainty(&uncertainty_result, UncertaintyType::Epistemic);

        assert_eq!(
            analysis.uncertainty_type as u8,
            UncertaintyType::Epistemic as u8
        );
        assert!(analysis.mean > 0.0);
        assert!(analysis.std >= 0.0);
        assert_eq!(analysis.distribution.len(), 3);
    }

    #[test]
    fn test_confidence_intervals() {
        let samples = array![
            [1.0, 1.1, 0.9, 1.05, 0.95],
            [2.0, 2.2, 1.8, 2.1, 1.9],
            [3.0, 3.3, 2.7, 3.2, 2.8]
        ];

        let intervals = compute_confidence_intervals(&samples, 0.8);

        assert_eq!(intervals.shape(), &[3, 2]);
        // Check that lower bound is less than upper bound
        for i in 0..3 {
            assert!(intervals[[i, 0]] <= intervals[[i, 1]]);
        }
    }

    #[test]
    fn test_platt_calibration() {
        let predicted_probs = array![0.1, 0.4, 0.7, 0.9];
        let true_labels = array![0.0, 0.0, 1.0, 1.0];

        let calibrated_model =
            calibrate_platt(&predicted_probs.view(), &true_labels.view()).unwrap();

        assert_eq!(
            calibrated_model.method as u8,
            CalibrationMethod::Platt as u8
        );
        assert_eq!(calibrated_model.parameters.len(), 2);

        let calibrated_preds = calibrated_model.calibrate(&predicted_probs.view()).unwrap();
        assert_eq!(calibrated_preds.len(), 4);
    }
}
