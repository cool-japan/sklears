//! Main perturbation analysis functions
//!
//! This module provides the core analysis functions for perturbation-based model
//! inspection and robustness analysis.

use super::core::{PerturbationConfig, PerturbationResult, PerturbationStats, RobustnessMetrics};
use super::helpers::{calculate_perturbation_stats, calculate_robustness_metrics};
use super::strategies::generate_perturbations;
use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{ArrayView1, ArrayView2};

///
/// let X = array![[0.5, 0.7], [0.3, 0.9]];
/// let config = PerturbationConfig {
///     strategy: PerturbationStrategy::Gaussian,
///     magnitude: 0.1,
///     n_samples: 10,
///     ..Default::default()
/// };
///
/// // Mock model function
/// let model_fn = |x: &scirs2_core::ndarray::ArrayView2<f64>| -> Vec<f64> {
///     x.rows().into_iter().map(|row| row[0] * 2.0 + row[1] * 0.5).collect()
/// };
///
/// let result = analyze_robustness(&model_fn, &X.view(), &config).unwrap();
/// ```
pub fn analyze_robustness<F>(
    model_fn: &F,
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<PerturbationResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    // Generate perturbations
    let perturbed_data = generate_perturbations(X, config)?;

    // Get original predictions
    let original_predictions = model_fn(X);

    // Get perturbed predictions
    let mut perturbed_predictions = Vec::new();
    for perturbed_x in &perturbed_data {
        let preds = model_fn(&perturbed_x.view());
        perturbed_predictions.push(preds);
    }

    // Calculate perturbation statistics
    let perturbation_stats = calculate_perturbation_stats(X, &perturbed_data)?;

    // Calculate robustness metrics
    let robustness_metrics =
        calculate_robustness_metrics(&original_predictions, &perturbed_predictions);

    Ok(PerturbationResult {
        original_data: X.to_owned(),
        perturbed_data,
        original_predictions,
        perturbed_predictions,
        perturbation_stats,
        robustness_metrics,
    })
}

/// Generate counterfactual examples using perturbations
///
/// # Arguments
///
/// * `model_fn` - Function that takes input features and returns predictions
/// * `X` - Original input data
/// * `target_predictions` - Desired target predictions
/// * `config` - Configuration for perturbation generation
/// * `max_iterations` - Maximum number of iterations for optimization
/// * `tolerance` - Convergence tolerance for target achievement
///
/// # Returns
///
/// Result containing counterfactual examples and their statistics
pub fn generate_counterfactuals<F>(
    model_fn: &F,
    X: &ArrayView2<Float>,
    target_predictions: &ArrayView1<Float>,
    config: &PerturbationConfig,
    max_iterations: usize,
    tolerance: Float,
) -> SklResult<CounterfactualResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let mut counterfactuals = Vec::new();
    let mut distances = Vec::new();
    let mut achieved_targets = Vec::new();

    for (i, &target) in target_predictions.iter().enumerate() {
        let original_sample = X.row(i);
        let original_pred =
            model_fn(&original_sample.insert_axis(scirs2_core::ndarray::Axis(0)))[0];

        let mut best_counterfactual = original_sample.to_owned();
        let mut best_distance = Float::INFINITY;
        let mut best_pred = original_pred;

        // Iterative perturbation search
        for _ in 0..max_iterations {
            // Generate perturbations around current best
            let current_data = best_counterfactual
                .clone()
                .insert_axis(scirs2_core::ndarray::Axis(0));
            let perturbed_samples = generate_perturbations(&current_data.view(), config)?;

            for perturbed in perturbed_samples {
                let perturbed_sample = perturbed.row(0);
                let pred = model_fn(&perturbed.view())[0];

                // Calculate distance to target
                let pred_distance = (pred - target).abs();
                let feature_distance = original_sample
                    .iter()
                    .zip(perturbed_sample.iter())
                    .map(|(&orig, &pert)| (orig - pert).powi(2))
                    .sum::<Float>()
                    .sqrt();

                // Update best if closer to target
                if pred_distance < (best_pred - target).abs() {
                    best_counterfactual = perturbed_sample.to_owned();
                    best_distance = feature_distance;
                    best_pred = pred;
                }
            }

            // Check convergence
            if (best_pred - target).abs() < tolerance {
                break;
            }
        }

        counterfactuals.push(best_counterfactual);
        distances.push(best_distance);
        achieved_targets.push(best_pred);
    }

    // Calculate success rate
    let successful_targets = achieved_targets
        .iter()
        .zip(target_predictions.iter())
        .filter(|(&achieved, &target)| (achieved - target).abs() < tolerance)
        .count();

    let success_rate = successful_targets as Float / target_predictions.len() as Float;

    // Calculate average distance
    let avg_distance = distances.iter().sum::<Float>() / distances.len() as Float;

    Ok(CounterfactualResult {
        counterfactuals,
        distances,
        achieved_predictions: achieved_targets,
        target_predictions: target_predictions.to_owned(),
        success_rate,
        avg_distance,
        tolerance,
    })
}

/// Result of counterfactual generation
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// Generated counterfactual examples
    pub counterfactuals: Vec<scirs2_core::ndarray::Array1<Float>>,
    /// Distance from original examples to counterfactuals
    pub distances: Vec<Float>,
    /// Achieved predictions for counterfactuals
    pub achieved_predictions: Vec<Float>,
    /// Target predictions
    pub target_predictions: scirs2_core::ndarray::Array1<Float>,
    /// Success rate (fraction of targets achieved within tolerance)
    pub success_rate: Float,
    /// Average distance to counterfactuals
    pub avg_distance: Float,
    /// Tolerance used for success measurement
    pub tolerance: Float,
}

/// Perform sensitivity analysis using perturbations
///
/// # Arguments
///
/// * `model_fn` - Function that takes input features and returns predictions
/// * `X` - Original input data
/// * `config` - Configuration for perturbation generation
///
/// # Returns
///
/// Result containing sensitivity analysis results
pub fn sensitivity_analysis<F>(
    model_fn: &F,
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<SensitivityResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = X.ncols();
    let mut feature_sensitivities = Vec::with_capacity(n_features);

    // Get baseline predictions
    let baseline_predictions = model_fn(X);

    // Test each feature individually
    for feature_idx in 0..n_features {
        let mut feature_changes = Vec::new();

        // Generate perturbations for this feature only
        for sample_idx in 0..X.nrows() {
            let mut perturbed_sample = X.row(sample_idx).to_owned();

            // Apply multiple perturbations to this feature
            let mut perturbations = Vec::new();
            let step_size = config.magnitude / 10.0; // 10 steps

            for step in -5..=5 {
                let perturbation = step as Float * step_size;
                perturbed_sample[feature_idx] = X[[sample_idx, feature_idx]] + perturbation;

                let perturbed_data = perturbed_sample
                    .clone()
                    .insert_axis(scirs2_core::ndarray::Axis(0));
                let pred = model_fn(&perturbed_data.view())[0];

                let change = (pred - baseline_predictions[sample_idx]).abs();
                perturbations.push(change);
            }

            // Calculate average change for this sample and feature
            let avg_change = perturbations.iter().sum::<Float>() / perturbations.len() as Float;
            feature_changes.push(avg_change);
        }

        // Calculate statistics for this feature
        let mean_sensitivity =
            feature_changes.iter().sum::<Float>() / feature_changes.len() as Float;
        let variance = feature_changes
            .iter()
            .map(|&x| (x - mean_sensitivity).powi(2))
            .sum::<Float>()
            / feature_changes.len() as Float;
        let std_sensitivity = variance.sqrt();

        feature_sensitivities.push(FeatureSensitivity {
            feature_index: feature_idx,
            mean_sensitivity,
            std_sensitivity,
            max_sensitivity: feature_changes.iter().cloned().fold(0.0, Float::max),
            min_sensitivity: feature_changes
                .iter()
                .cloned()
                .fold(Float::INFINITY, Float::min),
        });
    }

    // Sort by mean sensitivity (descending)
    feature_sensitivities
        .sort_by(|a, b| b.mean_sensitivity.partial_cmp(&a.mean_sensitivity).unwrap());

    let overall_sensitivity = feature_sensitivities
        .iter()
        .map(|fs| fs.mean_sensitivity)
        .sum::<Float>()
        / n_features as Float;

    Ok(SensitivityResult {
        feature_sensitivities,
        overall_sensitivity,
    })
}

/// Result of sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityResult {
    /// Sensitivity analysis for each feature
    pub feature_sensitivities: Vec<FeatureSensitivity>,
    /// Overall model sensitivity
    pub overall_sensitivity: Float,
}

/// Sensitivity information for a single feature
#[derive(Debug, Clone)]
pub struct FeatureSensitivity {
    /// Feature index
    pub feature_index: usize,
    /// Mean sensitivity across all samples
    pub mean_sensitivity: Float,
    /// Standard deviation of sensitivity
    pub std_sensitivity: Float,
    /// Maximum sensitivity observed
    pub max_sensitivity: Float,
    /// Minimum sensitivity observed
    pub min_sensitivity: Float,
}

/// Test model stability across different perturbation levels
///
/// # Arguments
///
/// * `model_fn` - Function that takes input features and returns predictions
/// * `X` - Original input data
/// * `base_config` - Base configuration for perturbation generation
/// * `magnitude_levels` - Different magnitude levels to test
///
/// # Returns
///
/// Result containing stability analysis across magnitude levels
pub fn stability_analysis<F>(
    model_fn: &F,
    X: &ArrayView2<Float>,
    base_config: &PerturbationConfig,
    magnitude_levels: &[Float],
) -> SklResult<StabilityResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let mut stability_results = Vec::new();

    for &magnitude in magnitude_levels {
        let mut config = base_config.clone();
        config.magnitude = magnitude;

        // Perform robustness analysis at this magnitude level
        let robustness_result = analyze_robustness(model_fn, X, &config)?;

        stability_results.push(MagnitudeStabilityResult {
            magnitude,
            robustness_metrics: robustness_result.robustness_metrics,
            perturbation_stats: robustness_result.perturbation_stats,
        });
    }

    Ok(StabilityResult {
        magnitude_results: stability_results,
        magnitude_levels: magnitude_levels.to_vec(),
    })
}

/// Result of stability analysis across magnitude levels
#[derive(Debug, Clone)]
pub struct StabilityResult {
    /// Results for each magnitude level
    pub magnitude_results: Vec<MagnitudeStabilityResult>,
    /// Magnitude levels tested
    pub magnitude_levels: Vec<Float>,
}

/// Stability result for a specific magnitude level
#[derive(Debug, Clone)]
pub struct MagnitudeStabilityResult {
    /// Magnitude level tested
    pub magnitude: Float,
    /// Robustness metrics at this magnitude
    pub robustness_metrics: RobustnessMetrics,
    /// Perturbation statistics at this magnitude
    pub perturbation_stats: PerturbationStats,
}
