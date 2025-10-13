//! Helper functions for perturbation analysis
//!
//! This module contains utility functions used throughout the perturbation
//! analysis system for calculating statistics and metrics.

use super::core::{PerturbationStats, RobustnessMetrics};
use crate::{Float, SklResult, SklearsError};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

/// Calculate perturbation statistics
///
/// # Arguments
///
/// * `original` - Original input data
/// * `perturbed_data` - Vector of perturbed data samples
///
/// # Returns
///
/// Result containing perturbation statistics
pub fn calculate_perturbation_stats(
    original: &ArrayView2<Float>,
    perturbed_data: &[Array2<Float>],
) -> SklResult<PerturbationStats> {
    let n_features = original.ncols();
    let n_samples = perturbed_data.len();

    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "No perturbed data provided".to_string(),
        ));
    }

    let mut mean_magnitude = Array1::zeros(n_features);
    let mut std_magnitude = Array1::zeros(n_features);
    let mut max_magnitude = Array1::zeros(n_features);

    // Calculate per-feature perturbation statistics
    for j in 0..n_features {
        let mut magnitudes = Vec::new();

        for perturbed in perturbed_data {
            for i in 0..original.nrows() {
                let magnitude = (perturbed[[i, j]] - original[[i, j]]).abs();
                magnitudes.push(magnitude);
            }
        }

        mean_magnitude[j] = magnitudes.iter().sum::<Float>() / magnitudes.len() as Float;
        max_magnitude[j] = magnitudes.iter().fold(0.0 as Float, |a, &b| a.max(b));

        let variance = magnitudes
            .iter()
            .map(|&x| (x - mean_magnitude[j]).powi(2))
            .sum::<Float>()
            / magnitudes.len() as Float;
        std_magnitude[j] = variance.sqrt();
    }

    // Calculate correlation matrix (simplified)
    let correlation_matrix = Array2::eye(n_features);

    Ok(PerturbationStats {
        mean_magnitude,
        std_magnitude,
        max_magnitude,
        correlation_matrix,
    })
}

/// Calculate robustness metrics
///
/// # Arguments
///
/// * `original_predictions` - Original model predictions
/// * `perturbed_predictions` - Predictions for perturbed data
///
/// # Returns
///
/// Robustness metrics
pub fn calculate_robustness_metrics(
    original_predictions: &[Float],
    perturbed_predictions: &[Vec<Float>],
) -> RobustnessMetrics {
    let n_samples = original_predictions.len();
    let n_perturbations = perturbed_predictions.len();

    if n_samples == 0 || n_perturbations == 0 {
        return RobustnessMetrics {
            prediction_stability: Float::INFINITY,
            prediction_variance: Float::INFINITY,
            max_prediction_change: Float::INFINITY,
            significant_change_fraction: 1.0,
            local_lipschitz: Float::INFINITY,
        };
    }

    let mut all_changes = Vec::new();
    let mut max_change: Float = 0.0;
    let mut significant_changes = 0;
    let significance_threshold = 0.1; // 10% change threshold

    for (i, &orig_pred) in original_predictions.iter().enumerate() {
        for perturbed_preds in perturbed_predictions {
            if i < perturbed_preds.len() {
                let change = (perturbed_preds[i] - orig_pred).abs();
                all_changes.push(change);
                max_change = max_change.max(change);

                if change / orig_pred.abs().max(1e-10) > significance_threshold {
                    significant_changes += 1;
                }
            }
        }
    }

    let mean_change = all_changes.iter().sum::<Float>() / all_changes.len() as Float;
    let variance = all_changes
        .iter()
        .map(|&x| (x - mean_change).powi(2))
        .sum::<Float>()
        / all_changes.len() as Float;

    let significant_fraction = significant_changes as Float / all_changes.len() as Float;

    // Simple local Lipschitz estimate
    let local_lipschitz = all_changes.iter().fold(0.0 as Float, |a, &b| a.max(b));

    RobustnessMetrics {
        prediction_stability: mean_change,
        prediction_variance: variance.sqrt(),
        max_prediction_change: max_change,
        significant_change_fraction: significant_fraction,
        local_lipschitz,
    }
}

/// Calculate feature importance based on perturbation sensitivity
///
/// # Arguments
///
/// * `original_predictions` - Original model predictions
/// * `feature_perturbations` - Predictions after perturbing each feature
///
/// # Returns
///
/// Vector of feature importance scores
pub fn calculate_feature_importance(
    original_predictions: &[Float],
    feature_perturbations: &[Vec<Float>],
) -> Vec<Float> {
    let n_features = feature_perturbations.len();
    let mut importance_scores = Vec::with_capacity(n_features);

    for perturbed_preds in feature_perturbations {
        let mut total_change = 0.0;
        let mut valid_samples = 0;

        for (orig, &perturbed) in original_predictions.iter().zip(perturbed_preds.iter()) {
            let change = (perturbed - orig).abs();
            total_change += change;
            valid_samples += 1;
        }

        let avg_change = if valid_samples > 0 {
            total_change / valid_samples as Float
        } else {
            0.0
        };

        importance_scores.push(avg_change);
    }

    importance_scores
}

/// Calculate prediction confidence intervals based on perturbations
///
/// # Arguments
///
/// * `perturbed_predictions` - All perturbed predictions for each sample
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95% confidence)
///
/// # Returns
///
/// Tuple of (lower_bounds, upper_bounds) for each sample
pub fn calculate_confidence_intervals(
    perturbed_predictions: &[Vec<Float>],
    confidence_level: Float,
) -> (Vec<Float>, Vec<Float>) {
    let alpha = 1.0 - confidence_level;
    let lower_quantile = alpha / 2.0;
    let upper_quantile = 1.0 - alpha / 2.0;

    let n_samples = if perturbed_predictions.is_empty() {
        0
    } else {
        perturbed_predictions[0].len()
    };

    let mut lower_bounds = Vec::with_capacity(n_samples);
    let mut upper_bounds = Vec::with_capacity(n_samples);

    for sample_idx in 0..n_samples {
        let mut sample_predictions = Vec::new();

        // Collect all predictions for this sample across perturbations
        for perturbed_preds in perturbed_predictions {
            if sample_idx < perturbed_preds.len() {
                sample_predictions.push(perturbed_preds[sample_idx]);
            }
        }

        if !sample_predictions.is_empty() {
            // Sort predictions to calculate quantiles
            sample_predictions.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n_preds = sample_predictions.len();
            let lower_idx = ((n_preds - 1) as Float * lower_quantile) as usize;
            let upper_idx = ((n_preds - 1) as Float * upper_quantile) as usize;

            lower_bounds.push(sample_predictions[lower_idx]);
            upper_bounds.push(sample_predictions[upper_idx]);
        } else {
            lower_bounds.push(0.0);
            upper_bounds.push(0.0);
        }
    }

    (lower_bounds, upper_bounds)
}

/// Calculate perturbation diversity metrics
///
/// # Arguments
///
/// * `perturbed_data` - Vector of perturbed data samples
///
/// # Returns
///
/// Diversity metrics including coverage and uniqueness
pub fn calculate_diversity_metrics(perturbed_data: &[Array2<Float>]) -> DiversityMetrics {
    if perturbed_data.is_empty() {
        return DiversityMetrics {
            coverage_score: 0.0,
            uniqueness_score: 0.0,
            variance_score: 0.0,
        };
    }

    // Calculate coverage as the range of values covered
    let mut all_values = Vec::new();
    for perturbed in perturbed_data {
        all_values.extend(perturbed.iter().cloned());
    }

    let min_val = all_values.iter().fold(Float::INFINITY, |a, &b| a.min(b));
    let max_val = all_values
        .iter()
        .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
    let coverage_score = (max_val - min_val).abs();

    // Calculate uniqueness as average pairwise distance
    let mut pairwise_distances = Vec::new();
    for i in 0..perturbed_data.len() {
        for j in (i + 1)..perturbed_data.len() {
            let distance = calculate_euclidean_distance(&perturbed_data[i], &perturbed_data[j]);
            pairwise_distances.push(distance);
        }
    }

    let uniqueness_score = if !pairwise_distances.is_empty() {
        pairwise_distances.iter().sum::<Float>() / pairwise_distances.len() as Float
    } else {
        0.0
    };

    // Calculate variance across all values
    let mean_val = all_values.iter().sum::<Float>() / all_values.len() as Float;
    let variance_score = all_values
        .iter()
        .map(|&x| (x - mean_val).powi(2))
        .sum::<Float>()
        / all_values.len() as Float;

    DiversityMetrics {
        coverage_score,
        uniqueness_score,
        variance_score,
    }
}

/// Diversity metrics for perturbations
#[derive(Debug, Clone)]
pub struct DiversityMetrics {
    /// Coverage of the feature space
    pub coverage_score: Float,
    /// Uniqueness of perturbations (average pairwise distance)
    pub uniqueness_score: Float,
    /// Variance across all perturbed values
    pub variance_score: Float,
}

/// Calculate Euclidean distance between two arrays
fn calculate_euclidean_distance(a: &Array2<Float>, b: &Array2<Float>) -> Float {
    if a.dim() != b.dim() {
        return Float::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<Float>()
        .sqrt()
}

/// Calculate outlier detection metrics for perturbations
///
/// # Arguments
///
/// * `original_predictions` - Original model predictions
/// * `perturbed_predictions` - All perturbed predictions
/// * `outlier_threshold` - Threshold for outlier detection (in standard deviations)
///
/// # Returns
///
/// Outlier analysis results
pub fn detect_prediction_outliers(
    original_predictions: &[Float],
    perturbed_predictions: &[Vec<Float>],
    outlier_threshold: Float,
) -> OutlierAnalysis {
    let mut all_changes = Vec::new();
    let mut outlier_indices = Vec::new();
    let mut sample_outlier_counts = vec![0; original_predictions.len()];

    // Collect all prediction changes
    for (sample_idx, &orig_pred) in original_predictions.iter().enumerate() {
        for perturbed_preds in perturbed_predictions {
            if sample_idx < perturbed_preds.len() {
                let change = (perturbed_preds[sample_idx] - orig_pred).abs();
                all_changes.push(change);
            }
        }
    }

    if all_changes.is_empty() {
        return OutlierAnalysis {
            outlier_fraction: 0.0,
            outlier_threshold_used: outlier_threshold,
            sample_outlier_counts,
            mean_change: 0.0,
            std_change: 0.0,
        };
    }

    // Calculate mean and standard deviation of changes
    let mean_change = all_changes.iter().sum::<Float>() / all_changes.len() as Float;
    let variance = all_changes
        .iter()
        .map(|&x| (x - mean_change).powi(2))
        .sum::<Float>()
        / all_changes.len() as Float;
    let std_change = variance.sqrt();

    let threshold = mean_change + outlier_threshold * std_change;

    // Identify outliers
    let mut change_idx = 0;
    for (sample_idx, &orig_pred) in original_predictions.iter().enumerate() {
        for perturbed_preds in perturbed_predictions {
            if sample_idx < perturbed_preds.len() {
                let change = (perturbed_preds[sample_idx] - orig_pred).abs();
                if change > threshold {
                    outlier_indices.push((sample_idx, change_idx));
                    sample_outlier_counts[sample_idx] += 1;
                }
                change_idx += 1;
            }
        }
    }

    let outlier_fraction = outlier_indices.len() as Float / all_changes.len() as Float;

    OutlierAnalysis {
        outlier_fraction,
        outlier_threshold_used: threshold,
        sample_outlier_counts,
        mean_change,
        std_change,
    }
}

/// Result of outlier analysis
#[derive(Debug, Clone)]
pub struct OutlierAnalysis {
    /// Fraction of predictions that are outliers
    pub outlier_fraction: Float,
    /// Threshold used for outlier detection
    pub outlier_threshold_used: Float,
    /// Number of outlier predictions per sample
    pub sample_outlier_counts: Vec<usize>,
    /// Mean prediction change
    pub mean_change: Float,
    /// Standard deviation of prediction changes
    pub std_change: Float,
}

/// Calculate model consistency across different perturbation runs
///
/// # Arguments
///
/// * `perturbation_results` - Multiple sets of perturbation results
///
/// # Returns
///
/// Consistency metrics
pub fn calculate_consistency_metrics(
    perturbation_results: &[Vec<Vec<Float>>],
) -> ConsistencyMetrics {
    if perturbation_results.len() < 2 {
        return ConsistencyMetrics {
            inter_run_correlation: 1.0,
            consistency_score: 1.0,
            variance_across_runs: 0.0,
        };
    }

    let n_samples = perturbation_results[0][0].len();
    let mut sample_variances = Vec::new();

    // Calculate variance for each sample across runs
    for sample_idx in 0..n_samples {
        let mut sample_predictions_across_runs = Vec::new();

        for run_results in perturbation_results {
            for perturbation_preds in run_results {
                if sample_idx < perturbation_preds.len() {
                    sample_predictions_across_runs.push(perturbation_preds[sample_idx]);
                }
            }
        }

        if !sample_predictions_across_runs.is_empty() {
            let mean = sample_predictions_across_runs.iter().sum::<Float>()
                / sample_predictions_across_runs.len() as Float;
            let variance = sample_predictions_across_runs
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<Float>()
                / sample_predictions_across_runs.len() as Float;
            sample_variances.push(variance);
        }
    }

    let variance_across_runs = if !sample_variances.is_empty() {
        sample_variances.iter().sum::<Float>() / sample_variances.len() as Float
    } else {
        0.0
    };

    let consistency_score = 1.0 / (1.0 + variance_across_runs); // Higher is more consistent
    let inter_run_correlation = (1.0 - variance_across_runs).max(0.0).min(1.0);

    ConsistencyMetrics {
        inter_run_correlation,
        consistency_score,
        variance_across_runs,
    }
}

/// Consistency metrics across multiple perturbation runs
#[derive(Debug, Clone)]
pub struct ConsistencyMetrics {
    /// Correlation between different runs
    pub inter_run_correlation: Float,
    /// Overall consistency score (0-1, higher is better)
    pub consistency_score: Float,
    /// Variance of predictions across runs
    pub variance_across_runs: Float,
}
