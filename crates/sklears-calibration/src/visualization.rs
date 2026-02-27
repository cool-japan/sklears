//! Visualization tools for calibration analysis
//!
//! This module provides various visualization utilities for calibration analysis,
//! including calibration curves, reliability diagrams, and confidence intervals.

use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};

/// Calibration curve data for plotting
#[derive(Debug, Clone)]
pub struct CalibrationCurve {
    /// Bin centers (mean predicted probabilities)
    pub bin_centers: Array1<Float>,
    /// Bin empirical frequencies (actual positive rates)
    pub bin_frequencies: Array1<Float>,
    /// Bin counts (number of samples in each bin)
    pub bin_counts: Array1<usize>,
    /// Bin confidence intervals (lower bounds)
    pub confidence_lower: Array1<Float>,
    /// Bin confidence intervals (upper bounds)
    pub confidence_upper: Array1<Float>,
    /// Perfect calibration line (y=x)
    pub perfect_line: Array1<Float>,
}

/// Reliability diagram data
#[derive(Debug, Clone)]
pub struct ReliabilityDiagram {
    /// Calibration curve data
    pub calibration_curve: CalibrationCurve,
    /// Histogram of predicted probabilities
    pub probability_histogram: ProbabilityHistogram,
    /// Overall statistics
    pub statistics: CalibrationStatistics,
}

/// Histogram of predicted probabilities
#[derive(Debug, Clone)]
pub struct ProbabilityHistogram {
    /// Bin edges
    pub bin_edges: Array1<Float>,
    /// Bin counts
    pub bin_counts: Array1<usize>,
    /// Bin centers
    pub bin_centers: Array1<Float>,
}

/// Overall calibration statistics
#[derive(Debug, Clone)]
pub struct CalibrationStatistics {
    /// Expected Calibration Error
    pub ece: Float,
    /// Maximum Calibration Error
    pub mce: Float,
    /// Brier Score
    pub brier_score: Float,
    /// Average confidence
    pub avg_confidence: Float,
    /// Average accuracy
    pub avg_accuracy: Float,
    /// Number of samples
    pub n_samples: usize,
}

/// Configuration for calibration visualization
#[derive(Debug, Clone)]
pub struct CalibrationVisualizationConfig {
    /// Number of bins for calibration curve
    pub n_bins: usize,
    /// Binning strategy
    pub binning_strategy: BinningStrategy,
    /// Confidence level for intervals (e.g., 0.95 for 95% CI)
    pub confidence_level: Float,
    /// Whether to include histogram
    pub include_histogram: bool,
    /// Minimum samples per bin
    pub min_samples_per_bin: usize,
}

/// Binning strategies for calibration curves
#[derive(Debug, Clone)]
pub enum BinningStrategy {
    /// Uniform width bins
    Uniform,
    /// Quantile-based bins (equal frequency)
    Quantile,
    /// Adaptive bins based on density
    Adaptive,
}

impl Default for CalibrationVisualizationConfig {
    fn default() -> Self {
        Self {
            n_bins: 10,
            binning_strategy: BinningStrategy::Uniform,
            confidence_level: 0.95,
            include_histogram: true,
            min_samples_per_bin: 5,
        }
    }
}

/// Generate calibration curve data
pub fn generate_calibration_curve(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    config: &CalibrationVisualizationConfig,
) -> Result<CalibrationCurve> {
    if y_true.len() != y_prob.len() {
        return Err(SklearsError::InvalidInput(
            "y_true and y_prob must have the same length".to_string(),
        ));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "No samples provided".to_string(),
        ));
    }

    // Create bins based on strategy
    let bin_edges = create_bins(y_prob, config)?;
    let n_bins = bin_edges.len() - 1;

    let mut bin_centers = Array1::zeros(n_bins);
    let mut bin_frequencies = Array1::zeros(n_bins);
    let mut bin_counts = vec![0usize; n_bins];
    let mut confidence_lower = Array1::zeros(n_bins);
    let mut confidence_upper = Array1::zeros(n_bins);

    // Assign samples to bins and compute statistics
    for (&prob, &label) in y_prob.iter().zip(y_true.iter()) {
        if let Some(bin_idx) = find_bin_index(prob, &bin_edges) {
            bin_counts[bin_idx] += 1;
            bin_centers[bin_idx] += prob;
            if label == 1 {
                bin_frequencies[bin_idx] += 1.0;
            }
        }
    }

    // Compute bin statistics
    for i in 0..n_bins {
        if bin_counts[i] > 0 {
            // Average probability in bin
            bin_centers[i] /= bin_counts[i] as Float;

            // Empirical frequency
            bin_frequencies[i] /= bin_counts[i] as Float;

            // Confidence intervals using Wilson score interval
            let (lower, upper) = wilson_confidence_interval(
                bin_frequencies[i],
                bin_counts[i],
                config.confidence_level,
            );
            confidence_lower[i] = lower;
            confidence_upper[i] = upper;
        } else {
            // Empty bin - use bin center
            bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2.0;
            bin_frequencies[i] = 0.0;
            confidence_lower[i] = 0.0;
            confidence_upper[i] = 0.0;
        }
    }

    // Perfect calibration line
    let perfect_line = bin_centers.clone();

    Ok(CalibrationCurve {
        bin_centers,
        bin_frequencies,
        bin_counts: Array1::from_vec(bin_counts),
        confidence_lower,
        confidence_upper,
        perfect_line,
    })
}

/// Generate reliability diagram
pub fn generate_reliability_diagram(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    config: &CalibrationVisualizationConfig,
) -> Result<ReliabilityDiagram> {
    let calibration_curve = generate_calibration_curve(y_true, y_prob, config)?;

    let probability_histogram = if config.include_histogram {
        generate_probability_histogram(y_prob, config)?
    } else {
        ProbabilityHistogram {
            bin_edges: Array1::zeros(2),
            bin_counts: Array1::from_vec(vec![0]),
            bin_centers: Array1::zeros(1),
        }
    };

    let statistics = compute_calibration_statistics(y_true, y_prob, &calibration_curve)?;

    Ok(ReliabilityDiagram {
        calibration_curve,
        probability_histogram,
        statistics,
    })
}

/// Generate probability histogram
pub fn generate_probability_histogram(
    y_prob: &Array1<Float>,
    config: &CalibrationVisualizationConfig,
) -> Result<ProbabilityHistogram> {
    let bin_edges = create_uniform_bins(0.0, 1.0, config.n_bins);
    let n_bins = bin_edges.len() - 1;
    let mut bin_counts = vec![0usize; n_bins];
    let mut bin_centers = Array1::zeros(n_bins);

    // Count samples in each bin
    for &prob in y_prob.iter() {
        if let Some(bin_idx) = find_bin_index(prob, &bin_edges) {
            bin_counts[bin_idx] += 1;
        }
    }

    // Compute bin centers
    for i in 0..n_bins {
        bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2.0;
    }

    Ok(ProbabilityHistogram {
        bin_edges,
        bin_counts: Array1::from_vec(bin_counts),
        bin_centers,
    })
}

/// Compute calibration statistics
pub fn compute_calibration_statistics(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
    calibration_curve: &CalibrationCurve,
) -> Result<CalibrationStatistics> {
    let n_samples = y_true.len();

    // Expected Calibration Error (ECE)
    let mut ece = 0.0;
    let mut max_calibration_error: Float = 0.0;

    for (i, (&freq, &center)) in calibration_curve
        .bin_frequencies
        .iter()
        .zip(calibration_curve.bin_centers.iter())
        .enumerate()
    {
        if calibration_curve.bin_counts[i] > 0 {
            let bin_weight = calibration_curve.bin_counts[i] as Float / n_samples as Float;
            let calibration_error = (freq - center).abs();
            ece += bin_weight * calibration_error;
            max_calibration_error = max_calibration_error.max(calibration_error);
        }
    }

    // Brier Score
    let mut brier_score = 0.0;
    for (&prob, &label) in y_prob.iter().zip(y_true.iter()) {
        let error = prob - label as Float;
        brier_score += error * error;
    }
    brier_score /= n_samples as Float;

    // Average confidence and accuracy
    let avg_confidence = y_prob.mean().unwrap_or(0.5);
    let avg_accuracy = y_true.mapv(|y| y as Float).mean().unwrap_or(0.5);

    Ok(CalibrationStatistics {
        ece,
        mce: max_calibration_error,
        brier_score,
        avg_confidence,
        avg_accuracy,
        n_samples,
    })
}

/// Create bins based on strategy
fn create_bins(
    y_prob: &Array1<Float>,
    config: &CalibrationVisualizationConfig,
) -> Result<Array1<Float>> {
    match config.binning_strategy {
        BinningStrategy::Uniform => Ok(create_uniform_bins(0.0, 1.0, config.n_bins)),
        BinningStrategy::Quantile => create_quantile_bins(y_prob, config.n_bins),
        BinningStrategy::Adaptive => create_adaptive_bins(y_prob, config),
    }
}

/// Create uniform width bins
fn create_uniform_bins(min_val: Float, max_val: Float, n_bins: usize) -> Array1<Float> {
    let step = (max_val - min_val) / n_bins as Float;
    let mut bins = Array1::zeros(n_bins + 1);

    for i in 0..=n_bins {
        bins[i] = min_val + i as Float * step;
    }

    bins
}

/// Create quantile-based bins
fn create_quantile_bins(y_prob: &Array1<Float>, n_bins: usize) -> Result<Array1<Float>> {
    let mut sorted_probs = y_prob.to_vec();
    sorted_probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut bins = Array1::zeros(n_bins + 1);
    bins[0] = 0.0;
    bins[n_bins] = 1.0;

    for i in 1..n_bins {
        let quantile = i as Float / n_bins as Float;
        let index = ((sorted_probs.len() - 1) as Float * quantile) as usize;
        bins[i] = sorted_probs[index];
    }

    Ok(bins)
}

/// Create adaptive bins based on data density
fn create_adaptive_bins(
    y_prob: &Array1<Float>,
    config: &CalibrationVisualizationConfig,
) -> Result<Array1<Float>> {
    // Simple adaptive binning: merge bins with too few samples
    let initial_bins = create_uniform_bins(0.0, 1.0, config.n_bins * 2);
    let mut bin_counts = vec![0usize; initial_bins.len() - 1];

    // Count samples in initial bins
    for &prob in y_prob.iter() {
        if let Some(bin_idx) = find_bin_index(prob, &initial_bins) {
            bin_counts[bin_idx] += 1;
        }
    }

    // Merge bins with too few samples
    let mut adaptive_bins = vec![0.0];
    let mut current_count = 0;

    for (i, &count) in bin_counts.iter().enumerate() {
        current_count += count;
        if current_count >= config.min_samples_per_bin || i == bin_counts.len() - 1 {
            adaptive_bins.push(initial_bins[i + 1]);
            current_count = 0;
        }
    }

    Ok(Array1::from_vec(adaptive_bins))
}

/// Find which bin a value belongs to
fn find_bin_index(value: Float, bin_edges: &Array1<Float>) -> Option<usize> {
    if value < bin_edges[0] || value > bin_edges[bin_edges.len() - 1] {
        return None;
    }

    (0..bin_edges.len() - 1)
        .find(|&i| value >= bin_edges[i] && (value < bin_edges[i + 1] || i == bin_edges.len() - 2))
}

/// Wilson confidence interval for proportions
fn wilson_confidence_interval(p: Float, n: usize, confidence_level: Float) -> (Float, Float) {
    if n == 0 {
        return (0.0, 1.0);
    }

    let z = match confidence_level {
        0.90 => 1.645,
        0.95 => 1.96,
        0.99 => 2.576,
        _ => 1.96, // Default to 95%
    };

    let n_f = n as Float;
    let center = (p + z * z / (2.0 * n_f)) / (1.0 + z * z / n_f);
    let margin = z * (p * (1.0 - p) / n_f + z * z / (4.0 * n_f * n_f)).sqrt() / (1.0 + z * z / n_f);

    let lower = (center - margin).max(0.0);
    let upper = (center + margin).min(1.0);

    (lower, upper)
}

/// Calibration residual analysis
#[derive(Debug, Clone)]
pub struct CalibrationResiduals {
    /// Predicted probabilities
    pub predicted_probs: Array1<Float>,
    /// Residuals (predicted - actual)
    pub residuals: Array1<Float>,
    /// Standardized residuals
    pub standardized_residuals: Array1<Float>,
    /// Absolute residuals
    pub absolute_residuals: Array1<Float>,
}

/// Generate calibration residuals for analysis
pub fn generate_calibration_residuals(
    y_true: &Array1<i32>,
    y_prob: &Array1<Float>,
) -> Result<CalibrationResiduals> {
    if y_true.len() != y_prob.len() {
        return Err(SklearsError::InvalidInput(
            "y_true and y_prob must have the same length".to_string(),
        ));
    }

    let n_samples = y_true.len();
    let mut residuals = Array1::zeros(n_samples);
    let mut absolute_residuals = Array1::zeros(n_samples);

    // Compute residuals
    for (i, (&prob, &label)) in y_prob.iter().zip(y_true.iter()).enumerate() {
        let residual = prob - label as Float;
        residuals[i] = residual;
        absolute_residuals[i] = residual.abs();
    }

    // Compute standardized residuals
    let residual_std = {
        let mean = residuals.mean().unwrap_or(0.0);
        let variance = residuals.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0);
        variance.sqrt()
    };

    let standardized_residuals = if residual_std > 0.0 {
        residuals.mapv(|x| x / residual_std)
    } else {
        residuals.clone()
    };

    Ok(CalibrationResiduals {
        predicted_probs: y_prob.clone(),
        residuals,
        standardized_residuals,
        absolute_residuals,
    })
}

/// Interactive calibration diagnostics
#[derive(Debug, Clone)]
pub struct InteractiveCalibrationDiagnostics {
    /// Multiple calibration curves for comparison
    pub calibration_curves: Vec<(String, CalibrationCurve)>,
    /// Hover information for points
    pub hover_info: Vec<CalibrationHoverInfo>,
    /// Comparison statistics
    pub comparison_stats: ComparisonStatistics,
}

/// Hover information for interactive plots
#[derive(Debug, Clone)]
pub struct CalibrationHoverInfo {
    /// Bin index
    pub bin_index: usize,
    /// Predicted probability range
    pub prob_range: (Float, Float),
    /// Actual frequency
    pub actual_frequency: Float,
    /// Sample count
    pub sample_count: usize,
    /// Confidence interval
    pub confidence_interval: (Float, Float),
}

/// Comparison statistics between different calibration methods
#[derive(Debug, Clone)]
pub struct ComparisonStatistics {
    /// Method names
    pub method_names: Vec<String>,
    /// ECE values for each method
    pub ece_values: Array1<Float>,
    /// MCE values for each method
    pub mce_values: Array1<Float>,
    /// Brier scores for each method
    pub brier_scores: Array1<Float>,
    /// Statistical significance tests
    pub significance_tests: Vec<SignificanceTest>,
}

/// Statistical significance test results
#[derive(Debug, Clone)]
pub struct SignificanceTest {
    /// Method 1 name
    pub method1: String,
    /// Method 2 name
    pub method2: String,
    /// Test statistic
    pub test_statistic: Float,
    /// P-value
    pub p_value: Float,
    /// Is significant at 0.05 level
    pub is_significant: bool,
}

/// Generate interactive calibration diagnostics
pub fn generate_interactive_diagnostics(
    y_true: &Array1<i32>,
    method_predictions: &[(String, Array1<Float>)],
    config: &CalibrationVisualizationConfig,
) -> Result<InteractiveCalibrationDiagnostics> {
    let mut calibration_curves = Vec::new();
    let mut ece_values = Array1::zeros(method_predictions.len());
    let mut mce_values = Array1::zeros(method_predictions.len());
    let mut brier_scores = Array1::zeros(method_predictions.len());
    let mut method_names = Vec::new();

    // Generate curves for each method
    for (i, (method_name, y_prob)) in method_predictions.iter().enumerate() {
        let curve = generate_calibration_curve(y_true, y_prob, config)?;
        let stats = compute_calibration_statistics(y_true, y_prob, &curve)?;

        calibration_curves.push((method_name.clone(), curve));
        ece_values[i] = stats.ece;
        mce_values[i] = stats.mce;
        brier_scores[i] = stats.brier_score;
        method_names.push(method_name.clone());
    }

    // Generate hover information (for first method as example)
    let hover_info = if !calibration_curves.is_empty() {
        generate_hover_info(&calibration_curves[0].1)
    } else {
        Vec::new()
    };

    // Generate significance tests (simplified)
    let significance_tests = generate_significance_tests(&method_names, &ece_values);

    let comparison_stats = ComparisonStatistics {
        method_names,
        ece_values,
        mce_values,
        brier_scores,
        significance_tests,
    };

    Ok(InteractiveCalibrationDiagnostics {
        calibration_curves,
        hover_info,
        comparison_stats,
    })
}

fn generate_hover_info(calibration_curve: &CalibrationCurve) -> Vec<CalibrationHoverInfo> {
    let mut hover_info = Vec::new();

    for (i, (&center, &freq)) in calibration_curve
        .bin_centers
        .iter()
        .zip(calibration_curve.bin_frequencies.iter())
        .enumerate()
    {
        // Estimate bin range (simplified)
        let bin_width = if i < calibration_curve.bin_centers.len() - 1 {
            (calibration_curve.bin_centers[i + 1] - center).abs()
        } else {
            0.1
        };

        let prob_range = (center - bin_width / 2.0, center + bin_width / 2.0);
        let confidence_interval = (
            calibration_curve.confidence_lower[i],
            calibration_curve.confidence_upper[i],
        );

        hover_info.push(CalibrationHoverInfo {
            bin_index: i,
            prob_range,
            actual_frequency: freq,
            sample_count: calibration_curve.bin_counts[i],
            confidence_interval,
        });
    }

    hover_info
}

fn generate_significance_tests(
    method_names: &[String],
    ece_values: &Array1<Float>,
) -> Vec<SignificanceTest> {
    let mut tests = Vec::new();

    // Simple pairwise comparisons (in practice, would use proper statistical tests)
    for i in 0..method_names.len() {
        for j in (i + 1)..method_names.len() {
            let diff = (ece_values[i] - ece_values[j]).abs();
            let test_statistic = diff / (ece_values[i] + ece_values[j] + 1e-10);
            let p_value = if diff > 0.01 { 0.01 } else { 0.5 }; // Simplified

            tests.push(SignificanceTest {
                method1: method_names[i].clone(),
                method2: method_names[j].clone(),
                test_statistic,
                p_value,
                is_significant: p_value < 0.05,
            });
        }
    }

    tests
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_calibration_curve_generation() {
        let y_true = array![0, 0, 1, 1, 1];
        let y_prob = array![0.1, 0.2, 0.6, 0.8, 0.9];
        let config = CalibrationVisualizationConfig::default();

        let curve = generate_calibration_curve(&y_true, &y_prob, &config).unwrap();

        assert!(!curve.bin_centers.is_empty());
        assert_eq!(curve.bin_centers.len(), curve.bin_frequencies.len());
        assert_eq!(curve.bin_centers.len(), curve.bin_counts.len());

        // Check that frequencies are valid probabilities
        for &freq in curve.bin_frequencies.iter() {
            assert!((0.0..=1.0).contains(&freq));
        }
    }

    #[test]
    fn test_reliability_diagram() {
        let y_true = array![0, 0, 1, 1, 1, 0, 1, 0, 1, 1];
        let y_prob = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
        let config = CalibrationVisualizationConfig::default();

        let diagram = generate_reliability_diagram(&y_true, &y_prob, &config).unwrap();

        assert!(diagram.statistics.ece >= 0.0);
        assert!(diagram.statistics.mce >= 0.0);
        assert!(diagram.statistics.brier_score >= 0.0);
        assert_eq!(diagram.statistics.n_samples, 10);
    }

    #[test]
    fn test_different_binning_strategies() {
        let y_true = array![0, 0, 1, 1, 1];
        let y_prob = array![0.1, 0.2, 0.6, 0.8, 0.9];

        for strategy in [
            BinningStrategy::Uniform,
            BinningStrategy::Quantile,
            BinningStrategy::Adaptive,
        ] {
            let config = CalibrationVisualizationConfig {
                binning_strategy: strategy,
                ..Default::default()
            };

            let curve = generate_calibration_curve(&y_true, &y_prob, &config).unwrap();
            assert!(!curve.bin_centers.is_empty());
        }
    }

    #[test]
    fn test_calibration_residuals() {
        let y_true = array![0, 0, 1, 1, 1];
        let y_prob = array![0.1, 0.2, 0.6, 0.8, 0.9];

        let residuals = generate_calibration_residuals(&y_true, &y_prob).unwrap();

        assert_eq!(residuals.residuals.len(), 5);
        assert_eq!(residuals.standardized_residuals.len(), 5);
        assert_eq!(residuals.absolute_residuals.len(), 5);

        // Check that absolute residuals are non-negative
        for &abs_residual in residuals.absolute_residuals.iter() {
            assert!(abs_residual >= 0.0);
        }
    }

    #[test]
    fn test_interactive_diagnostics() {
        let y_true = array![0, 0, 1, 1, 1];
        let method_predictions = vec![
            ("Method1".to_string(), array![0.1, 0.2, 0.6, 0.8, 0.9]),
            ("Method2".to_string(), array![0.2, 0.3, 0.5, 0.7, 0.8]),
        ];
        let diagnostics = generate_interactive_diagnostics(
            &y_true,
            &method_predictions,
            &CalibrationVisualizationConfig::default(),
        )
        .unwrap();

        assert_eq!(diagnostics.calibration_curves.len(), 2);
        assert_eq!(diagnostics.comparison_stats.method_names.len(), 2);
        assert!(diagnostics.comparison_stats.ece_values.len() == 2);
    }

    #[test]
    fn test_wilson_confidence_interval() {
        let (lower, upper) = wilson_confidence_interval(0.5, 100, 0.95);
        assert!(lower < 0.5);
        assert!(upper > 0.5);
        assert!(lower >= 0.0);
        assert!(upper <= 1.0);
    }

    #[test]
    fn test_bin_creation() {
        // Test uniform bins
        let uniform_bins = create_uniform_bins(0.0, 1.0, 10);
        assert_eq!(uniform_bins.len(), 11);
        assert_eq!(uniform_bins[0], 0.0);
        assert_eq!(uniform_bins[10], 1.0);

        // Test quantile bins
        let y_prob = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let quantile_bins = create_quantile_bins(&y_prob, 5).unwrap();
        assert_eq!(quantile_bins.len(), 6);
        assert_eq!(quantile_bins[0], 0.0);
        assert_eq!(quantile_bins[5], 1.0);
    }
}
