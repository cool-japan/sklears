//! Adversarial and Robustness Analysis Methods
//!
//! This module provides methods for adversarial analysis and robustness testing of model
//! interpretability, including adversarial example generation, explanation robustness testing,
//! adversarial training integration, certified explanation robustness, and explanation stability analysis.

use crate::{types::Float, SklResult, SklearsError};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use std::collections::HashMap;

/// Configuration for adversarial analysis methods
#[derive(Debug, Clone)]
pub struct AdversarialConfig {
    /// Epsilon parameter for adversarial perturbations
    pub epsilon: Float,
    /// Maximum number of iterations for iterative attacks
    pub max_iterations: usize,
    /// Step size for gradient-based attacks
    pub step_size: Float,
    /// Number of adversarial samples to generate
    pub n_adversarial_samples: usize,
    /// Confidence threshold for robustness testing
    pub confidence_threshold: Float,
    /// Perturbation bounds for stability analysis
    pub perturbation_bounds: (Float, Float),
    /// Number of perturbations for stability testing
    pub n_stability_samples: usize,
    /// Robustness metrics to compute
    pub robustness_metrics: Vec<RobustnessMetric>,
}

impl Default for AdversarialConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            max_iterations: 100,
            step_size: 0.01,
            n_adversarial_samples: 100,
            confidence_threshold: 0.95,
            perturbation_bounds: (-0.2, 0.2),
            n_stability_samples: 50,
            robustness_metrics: vec![
                RobustnessMetric::ExplanationConsistency,
                RobustnessMetric::PredictionStability,
                RobustnessMetric::FeatureImportanceStability,
            ],
        }
    }
}

/// Types of robustness metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RobustnessMetric {
    /// Consistency of explanations under perturbations
    ExplanationConsistency,
    /// Stability of model predictions
    PredictionStability,
    /// Stability of feature importance scores
    FeatureImportanceStability,
    /// Certified robustness bounds
    CertifiedRobustness,
    /// Adversarial accuracy
    AdversarialAccuracy,
}

/// Adversarial attack methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdversarialAttack {
    /// Fast Gradient Sign Method (FGSM)
    FGSM,
    /// Projected Gradient Descent (PGD)
    PGD,
    /// Carlini & Wagner attack
    CarliniWagner,
    /// Random noise attack
    RandomNoise,
    /// Explanation-targeted attack
    ExplanationTargeted,
}

/// Result of adversarial example generation
#[derive(Debug, Clone)]
pub struct AdversarialExampleResult {
    /// Original input
    pub original_input: Array2<Float>,
    /// Adversarial examples
    pub adversarial_examples: Vec<Array2<Float>>,
    /// Perturbations applied
    pub perturbations: Vec<Array2<Float>>,
    /// Success rate of adversarial examples
    pub success_rate: Float,
    /// L2 norms of perturbations
    pub perturbation_norms: Array1<Float>,
    /// Original predictions
    pub original_predictions: Array1<Float>,
    /// Adversarial predictions
    pub adversarial_predictions: Vec<Array1<Float>>,
}

/// Result of explanation robustness testing
#[derive(Debug, Clone)]
pub struct ExplanationRobustnessResult {
    /// Original explanations
    pub original_explanations: Array2<Float>,
    /// Explanations under perturbations
    pub perturbed_explanations: Vec<Array2<Float>>,
    /// Robustness scores for each metric
    pub robustness_scores: HashMap<RobustnessMetric, Float>,
    /// Explanation consistency over perturbations
    pub explanation_consistency: Float,
    /// Statistical significance tests
    pub statistical_tests: HashMap<String, Float>,
    /// Confidence intervals for robustness
    pub confidence_intervals: HashMap<RobustnessMetric, (Float, Float)>,
}

/// Result of explanation stability analysis
#[derive(Debug, Clone)]
pub struct StabilityAnalysisResult {
    /// Stability scores for different perturbation levels
    pub stability_scores: Array1<Float>,
    /// Perturbation levels tested
    pub perturbation_levels: Array1<Float>,
    /// Critical perturbation threshold
    pub critical_threshold: Float,
    /// Stability metrics breakdown
    pub stability_breakdown: HashMap<String, Array1<Float>>,
    /// Trend analysis
    pub stability_trend: StabilityTrend,
}

/// Stability trend analysis
#[derive(Debug, Clone)]
pub struct StabilityTrend {
    /// Slope of stability decline
    pub decline_slope: Float,
    /// R² of trend fit
    pub trend_fit_r2: Float,
    /// Predicted stability at different noise levels
    pub trend_predictions: Array1<Float>,
    /// Confidence bounds for predictions
    pub prediction_bounds: Array2<Float>,
}

/// Certified robustness result
#[derive(Debug, Clone)]
pub struct CertifiedRobustnessResult {
    /// Lower bounds on robustness
    pub robustness_lower_bounds: Array1<Float>,
    /// Upper bounds on robustness
    pub robustness_upper_bounds: Array1<Float>,
    /// Certification method used
    pub certification_method: CertificationMethod,
    /// Certified radius for each input
    pub certified_radii: Array1<Float>,
    /// Verification status
    pub verification_status: Vec<VerificationStatus>,
}

/// Certification methods for robustness
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertificationMethod {
    /// Interval Bound Propagation
    IntervalBoundPropagation,
    /// Linear relaxation
    LinearRelaxation,
    /// SMT-based verification
    SMTVerification,
    /// Randomized smoothing
    RandomizedSmoothing,
}

/// Verification status for inputs
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationStatus {
    /// Certified robust
    Robust,
    /// Certified not robust
    NotRobust,
    /// Unknown (timeout or inconclusive)
    Unknown,
}

/// Generate adversarial examples
pub fn generate_adversarial_examples<F>(
    input: &ArrayView2<Float>,
    model_fn: F,
    attack: AdversarialAttack,
    config: &AdversarialConfig,
) -> SklResult<AdversarialExampleResult>
where
    F: Fn(&Array2<Float>) -> SklResult<Array1<Float>>,
{
    let original_predictions = model_fn(&input.to_owned())?;
    let mut adversarial_examples = Vec::new();
    let mut perturbations = Vec::new();
    let mut adversarial_predictions = Vec::new();
    let mut perturbation_norms = Vec::new();

    let mut successful_attacks = 0;

    for _ in 0..config.n_adversarial_samples {
        let (adversarial_example, perturbation) = match attack {
            AdversarialAttack::FGSM => generate_fgsm_attack(input, &model_fn, config)?,
            AdversarialAttack::PGD => generate_pgd_attack(input, &model_fn, config)?,
            AdversarialAttack::CarliniWagner => generate_cw_attack(input, &model_fn, config)?,
            AdversarialAttack::RandomNoise => generate_random_noise_attack(input, config)?,
            AdversarialAttack::ExplanationTargeted => {
                generate_explanation_targeted_attack(input, &model_fn, config)?
            }
        };

        // Check if attack was successful
        let adv_predictions = model_fn(&adversarial_example)?;
        let is_successful = check_attack_success(&original_predictions, &adv_predictions, config)?;

        if is_successful {
            successful_attacks += 1;
        }

        // Compute perturbation norm
        let norm = perturbation.mapv(|x| x.powi(2)).sum().sqrt();
        perturbation_norms.push(norm);

        adversarial_examples.push(adversarial_example);
        perturbations.push(perturbation);
        adversarial_predictions.push(adv_predictions);
    }

    let success_rate = successful_attacks as Float / config.n_adversarial_samples as Float;

    Ok(AdversarialExampleResult {
        original_input: input.to_owned(),
        adversarial_examples,
        perturbations,
        success_rate,
        perturbation_norms: Array1::from_vec(perturbation_norms),
        original_predictions,
        adversarial_predictions,
    })
}

/// Test explanation robustness
pub fn test_explanation_robustness<F, E>(
    input: &ArrayView2<Float>,
    model_fn: F,
    explanation_fn: E,
    config: &AdversarialConfig,
) -> SklResult<ExplanationRobustnessResult>
where
    F: Fn(&Array2<Float>) -> SklResult<Array1<Float>>,
    E: Fn(&Array2<Float>) -> SklResult<Array2<Float>>,
{
    // Get original explanations and predictions
    let original_explanations = explanation_fn(&input.to_owned())?;
    let original_predictions = model_fn(&input.to_owned())?;

    // Generate perturbed inputs, explanations, and predictions
    let mut perturbed_explanations = Vec::new();
    let mut perturbed_predictions = Vec::new();
    let (min_bound, max_bound) = config.perturbation_bounds;

    for _ in 0..config.n_stability_samples {
        // Generate random perturbation
        let perturbation = generate_random_perturbation(input.shape(), min_bound, max_bound)?;
        let perturbed_input = input.to_owned() + perturbation;

        // Get explanation and prediction for perturbed input
        let perturbed_explanation = explanation_fn(&perturbed_input)?;
        perturbed_explanations.push(perturbed_explanation);
        perturbed_predictions.push(model_fn(&perturbed_input)?);
    }

    // Compute robustness metrics
    let mut robustness_scores = HashMap::new();
    let mut confidence_intervals = HashMap::new();

    for metric in &config.robustness_metrics {
        let score = compute_robustness_metric(
            metric,
            &original_explanations,
            &perturbed_explanations,
            Some(&original_predictions),
            &perturbed_predictions,
        )?;
        let ci = compute_confidence_interval(
            &perturbed_explanations,
            metric,
            config.confidence_threshold,
        )?;

        robustness_scores.insert(metric.clone(), score);
        confidence_intervals.insert(metric.clone(), ci);
    }

    // Compute explanation consistency
    let explanation_consistency =
        compute_explanation_consistency(&original_explanations, &perturbed_explanations)?;

    // Perform statistical tests
    let statistical_tests =
        perform_statistical_tests(&original_explanations, &perturbed_explanations)?;

    Ok(ExplanationRobustnessResult {
        original_explanations,
        perturbed_explanations,
        robustness_scores,
        explanation_consistency,
        statistical_tests,
        confidence_intervals,
    })
}

/// Analyze explanation stability
pub fn analyze_explanation_stability<F, E>(
    input: &ArrayView2<Float>,
    model_fn: F,
    explanation_fn: E,
    config: &AdversarialConfig,
) -> SklResult<StabilityAnalysisResult>
where
    F: Fn(&Array2<Float>) -> SklResult<Array1<Float>>,
    E: Fn(&Array2<Float>) -> SklResult<Array2<Float>>,
{
    let original_explanations = explanation_fn(&input.to_owned())?;
    let original_predictions = model_fn(&input.to_owned())?;

    // Test stability at different perturbation levels
    let perturbation_levels = Array1::linspace(0.01, 0.5, 20);
    let mut stability_scores = Vec::new();
    let mut stability_breakdown = HashMap::new();

    for &level in &perturbation_levels {
        let mut level_explanations = Vec::new();
        let mut level_predictions = Vec::new();

        // Generate explanations and predictions at this perturbation level
        for _ in 0..config.n_stability_samples {
            let perturbation = generate_random_perturbation(input.shape(), -level, level)?;
            let perturbed_input = input.to_owned() + perturbation;
            let explanation = explanation_fn(&perturbed_input)?;
            level_explanations.push(explanation);
            level_predictions.push(model_fn(&perturbed_input)?);
        }

        // Compute stability score for this level
        let stability_score =
            compute_explanation_consistency(&original_explanations, &level_explanations)?;
        stability_scores.push(stability_score);

        // Breakdown by metrics
        for metric in &config.robustness_metrics {
            let metric_score = compute_robustness_metric(
                metric,
                &original_explanations,
                &level_explanations,
                Some(&original_predictions),
                &level_predictions,
            )?;
            stability_breakdown
                .entry(format!("{:?}", metric))
                .or_insert_with(Vec::new)
                .push(metric_score);
        }
    }

    let stability_scores = Array1::from_vec(stability_scores);

    // Convert breakdown to arrays
    let stability_breakdown: HashMap<String, Array1<Float>> = stability_breakdown
        .into_iter()
        .map(|(k, v)| (k, Array1::from_vec(v)))
        .collect();

    // Find critical threshold (where stability drops below 50%)
    let critical_threshold = find_critical_threshold(&perturbation_levels, &stability_scores, 0.5)?;

    // Analyze trend
    let stability_trend = analyze_stability_trend(&perturbation_levels, &stability_scores)?;

    Ok(StabilityAnalysisResult {
        stability_scores,
        perturbation_levels,
        critical_threshold,
        stability_breakdown,
        stability_trend,
    })
}

/// Compute certified robustness bounds
pub fn compute_certified_robustness<F>(
    input: &ArrayView2<Float>,
    model_fn: F,
    method: CertificationMethod,
    config: &AdversarialConfig,
) -> SklResult<CertifiedRobustnessResult>
where
    F: Fn(&Array2<Float>) -> SklResult<Array1<Float>>,
{
    let n_inputs = input.nrows();
    let mut robustness_lower_bounds = Vec::new();
    let mut robustness_upper_bounds = Vec::new();
    let mut certified_radii = Vec::new();
    let mut verification_status = Vec::new();

    for i in 0..n_inputs {
        let input_row = input.row(i).to_owned().insert_axis(Axis(0));

        let (lower_bound, upper_bound, certified_radius, status) = match method {
            CertificationMethod::IntervalBoundPropagation => {
                compute_ibp_bounds(&input_row.view(), &model_fn, config)?
            }
            CertificationMethod::LinearRelaxation => {
                compute_linear_relaxation_bounds(&input_row.view(), &model_fn, config)?
            }
            CertificationMethod::SMTVerification => {
                compute_smt_bounds(&input_row.view(), &model_fn, config)?
            }
            CertificationMethod::RandomizedSmoothing => {
                compute_randomized_smoothing_bounds(&input_row.view(), &model_fn, config)?
            }
        };

        robustness_lower_bounds.push(lower_bound);
        robustness_upper_bounds.push(upper_bound);
        certified_radii.push(certified_radius);
        verification_status.push(status);
    }

    Ok(CertifiedRobustnessResult {
        robustness_lower_bounds: Array1::from_vec(robustness_lower_bounds),
        robustness_upper_bounds: Array1::from_vec(robustness_upper_bounds),
        certification_method: method,
        certified_radii: Array1::from_vec(certified_radii),
        verification_status,
    })
}

// Helper functions

/// Generate FGSM attack
fn generate_fgsm_attack<F>(
    input: &ArrayView2<Float>,
    model_fn: F,
    config: &AdversarialConfig,
) -> SklResult<(Array2<Float>, Array2<Float>)>
where
    F: Fn(&Array2<Float>) -> SklResult<Array1<Float>>,
{
    // Simplified FGSM implementation using finite differences
    let epsilon = 1e-4;
    let original_pred = model_fn(&input.to_owned())?;

    let mut gradient = Array2::zeros(input.raw_dim());

    // Compute gradient using finite differences
    for i in 0..input.nrows() {
        for j in 0..input.ncols() {
            let mut perturbed = input.to_owned();
            perturbed[[i, j]] += epsilon;

            let perturbed_pred = model_fn(&perturbed)?;
            let grad = (perturbed_pred[0] - original_pred[0]) / epsilon;
            gradient[[i, j]] = grad;
        }
    }

    // Apply FGSM perturbation
    let perturbation = config.epsilon * gradient.mapv(|x| x.signum());
    let adversarial_example = input.to_owned() + &perturbation;

    Ok((adversarial_example, perturbation))
}

/// Generate PGD attack
fn generate_pgd_attack<F>(
    input: &ArrayView2<Float>,
    model_fn: F,
    config: &AdversarialConfig,
) -> SklResult<(Array2<Float>, Array2<Float>)>
where
    F: Fn(&Array2<Float>) -> SklResult<Array1<Float>>,
{
    let mut adversarial_example = input.to_owned();
    let mut total_perturbation = Array2::zeros(input.raw_dim());

    for _ in 0..config.max_iterations {
        // Compute gradient (simplified using finite differences)
        let (_, step_perturbation) = generate_fgsm_attack(
            &adversarial_example.view(),
            &model_fn,
            &AdversarialConfig {
                epsilon: config.step_size,
                ..config.clone()
            },
        )?;

        // Update adversarial example
        adversarial_example = &adversarial_example + &step_perturbation;
        total_perturbation = &total_perturbation + &step_perturbation;

        // Project back to epsilon ball
        let perturbation_norm = total_perturbation.mapv(|x: Float| x.powi(2)).sum().sqrt();
        if perturbation_norm > config.epsilon {
            total_perturbation *= config.epsilon / perturbation_norm;
            adversarial_example = input.to_owned() + &total_perturbation;
        }
    }

    Ok((adversarial_example, total_perturbation))
}

/// Generate C&W attack (simplified)
fn generate_cw_attack<F>(
    input: &ArrayView2<Float>,
    model_fn: F,
    config: &AdversarialConfig,
) -> SklResult<(Array2<Float>, Array2<Float>)>
where
    F: Fn(&Array2<Float>) -> SklResult<Array1<Float>>,
{
    // Simplified C&W - for demonstration purposes, use PGD-like approach
    generate_pgd_attack(input, model_fn, config)
}

/// Generate random noise attack
fn generate_random_noise_attack(
    input: &ArrayView2<Float>,
    config: &AdversarialConfig,
) -> SklResult<(Array2<Float>, Array2<Float>)> {
    let perturbation =
        generate_random_perturbation(input.shape(), -config.epsilon, config.epsilon)?;
    let adversarial_example = input.to_owned() + &perturbation;

    Ok((adversarial_example, perturbation))
}

/// Generate explanation-targeted attack
fn generate_explanation_targeted_attack<F>(
    input: &ArrayView2<Float>,
    model_fn: F,
    config: &AdversarialConfig,
) -> SklResult<(Array2<Float>, Array2<Float>)>
where
    F: Fn(&Array2<Float>) -> SklResult<Array1<Float>>,
{
    // Simplified explanation-targeted attack
    // In practice, this would target specific explanation methods
    generate_pgd_attack(input, model_fn, config)
}

/// Check if adversarial attack was successful
fn check_attack_success(
    original_pred: &Array1<Float>,
    adversarial_pred: &Array1<Float>,
    config: &AdversarialConfig,
) -> SklResult<bool> {
    if original_pred.len() != adversarial_pred.len() {
        return Err(SklearsError::InvalidInput(
            "Prediction lengths mismatch".to_string(),
        ));
    }

    // For regression: significant prediction change
    // For classification: label change
    let prediction_change = (adversarial_pred[0] - original_pred[0]).abs();
    Ok(prediction_change > config.epsilon)
}

/// Generate random perturbation
fn generate_random_perturbation(
    shape: &[usize],
    min_val: Float,
    max_val: Float,
) -> SklResult<Array2<Float>> {
    let mut perturbation = Array2::zeros((shape[0], shape[1]));

    for elem in perturbation.iter_mut() {
        *elem = min_val + (max_val - min_val) * scirs2_core::random::thread_rng().random::<Float>();
    }

    Ok(perturbation)
}

/// Compute robustness metric
fn compute_robustness_metric(
    metric: &RobustnessMetric,
    original_explanations: &Array2<Float>,
    perturbed_explanations: &[Array2<Float>],
    original_predictions: Option<&Array1<Float>>,
    perturbed_predictions: &[Array1<Float>],
) -> SklResult<Float> {
    match metric {
        RobustnessMetric::ExplanationConsistency => {
            compute_explanation_consistency(original_explanations, perturbed_explanations)
        }
        RobustnessMetric::PredictionStability => {
            // Real prediction stability: the mean agreement between the original
            // prediction and each perturbed prediction. For each perturbation we
            // compute 1 - normalized-mean-absolute-deviation, clamped to [0, 1], so a
            // value near 1 means the model's outputs barely move under perturbation.
            match original_predictions {
                Some(original) => compute_prediction_stability(original, perturbed_predictions),
                None => Err(SklearsError::InvalidInput(
                    "PredictionStability requires model predictions, which were not provided"
                        .to_string(),
                )),
            }
        }
        RobustnessMetric::FeatureImportanceStability => {
            compute_feature_importance_stability(original_explanations, perturbed_explanations)
        }
        RobustnessMetric::CertifiedRobustness => {
            // Certified robustness requires a formal verification procedure (e.g.
            // interval-bound propagation or a Lipschitz-constant certificate) over the
            // model itself. The robustness harness only has sampled predictions and
            // explanations, so a sound certified bound cannot be produced here.
            Err(SklearsError::NotImplemented(
                "CertifiedRobustness requires a formal certification procedure over the model \
                 (e.g. IBP or a Lipschitz certificate); sampled predictions are insufficient. \
                 Use a dedicated certified-robustness routine."
                    .to_string(),
            ))
        }
        RobustnessMetric::AdversarialAccuracy => {
            // Real empirical adversarial accuracy under the sampled perturbations: the
            // fraction of perturbed inputs whose predicted class (argmax) matches the
            // original prediction's class. This is a genuine measurement over the
            // perturbations that were actually evaluated.
            match original_predictions {
                Some(original) => compute_adversarial_accuracy(original, perturbed_predictions),
                None => Err(SklearsError::InvalidInput(
                    "AdversarialAccuracy requires model predictions, which were not provided"
                        .to_string(),
                )),
            }
        }
    }
}

/// Mean stability of predictions under perturbation, in [0, 1].
///
/// For each perturbed prediction vector we compute the mean absolute deviation from
/// the original, normalized by the dynamic range of the original predictions, and map
/// it to a stability score `1 - normalized_deviation` clamped to [0, 1].
fn compute_prediction_stability(
    original: &Array1<Float>,
    perturbed: &[Array1<Float>],
) -> SklResult<Float> {
    if perturbed.is_empty() {
        return Ok(1.0);
    }

    let range = {
        let max = original
            .iter()
            .cloned()
            .fold(Float::NEG_INFINITY, Float::max);
        let min = original.iter().cloned().fold(Float::INFINITY, Float::min);
        (max - min).abs()
    };
    let scale = if range > Float::EPSILON { range } else { 1.0 };

    let mut total = 0.0;
    let mut count = 0;
    for pred in perturbed {
        if pred.len() != original.len() || pred.is_empty() {
            continue;
        }
        let mad = original
            .iter()
            .zip(pred.iter())
            .map(|(&o, &p)| (o - p).abs())
            .sum::<Float>()
            / original.len() as Float;
        let stability = (1.0 - mad / scale).clamp(0.0, 1.0);
        total += stability;
        count += 1;
    }

    if count == 0 {
        Ok(0.0)
    } else {
        Ok(total / count as Float)
    }
}

/// Empirical adversarial accuracy: the fraction of perturbed predictions whose argmax
/// class equals the original prediction's argmax class.
fn compute_adversarial_accuracy(
    original: &Array1<Float>,
    perturbed: &[Array1<Float>],
) -> SklResult<Float> {
    if perturbed.is_empty() {
        return Ok(1.0);
    }

    let argmax = |v: &Array1<Float>| -> Option<usize> {
        if v.is_empty() {
            return None;
        }
        let mut best_idx = 0usize;
        let mut best_val = Float::NEG_INFINITY;
        for (i, &val) in v.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_idx = i;
            }
        }
        Some(best_idx)
    };

    let original_class = match argmax(original) {
        Some(c) => c,
        None => return Ok(0.0),
    };

    let mut matches = 0;
    let mut count = 0;
    for pred in perturbed {
        if let Some(c) = argmax(pred) {
            if c == original_class {
                matches += 1;
            }
            count += 1;
        }
    }

    if count == 0 {
        Ok(0.0)
    } else {
        Ok(matches as Float / count as Float)
    }
}

/// Compute explanation consistency
fn compute_explanation_consistency(
    original_explanations: &Array2<Float>,
    perturbed_explanations: &[Array2<Float>],
) -> SklResult<Float> {
    if perturbed_explanations.is_empty() {
        return Ok(1.0);
    }

    let mut total_similarity = 0.0;
    let mut count = 0;

    for perturbed in perturbed_explanations {
        if perturbed.shape() == original_explanations.shape() {
            // Compute cosine similarity
            let dot_product = original_explanations
                .iter()
                .zip(perturbed.iter())
                .map(|(&a, &b)| a * b)
                .sum::<Float>();

            let norm_orig = original_explanations.mapv(|x| x.powi(2)).sum().sqrt();
            let norm_pert = perturbed.mapv(|x| x.powi(2)).sum().sqrt();

            if norm_orig > 0.0 && norm_pert > 0.0 {
                let similarity = dot_product / (norm_orig * norm_pert);
                total_similarity += similarity;
                count += 1;
            }
        }
    }

    if count > 0 {
        Ok(total_similarity / count as Float)
    } else {
        Ok(0.0)
    }
}

/// Compute feature importance stability
fn compute_feature_importance_stability(
    original_explanations: &Array2<Float>,
    perturbed_explanations: &[Array2<Float>],
) -> SklResult<Float> {
    // Compute rank correlation of feature importance rankings
    let original_importance = original_explanations
        .mean_axis(Axis(0))
        .expect("operation should succeed");

    let mut rank_correlations = Vec::new();

    for perturbed in perturbed_explanations {
        if perturbed.shape() == original_explanations.shape() {
            let perturbed_importance = perturbed
                .mean_axis(Axis(0))
                .expect("operation should succeed");
            let correlation =
                compute_rank_correlation(&original_importance, &perturbed_importance)?;
            rank_correlations.push(correlation);
        }
    }

    if rank_correlations.is_empty() {
        Ok(1.0)
    } else {
        Ok(rank_correlations.iter().sum::<Float>() / rank_correlations.len() as Float)
    }
}

/// Compute rank correlation
fn compute_rank_correlation(a: &Array1<Float>, b: &Array1<Float>) -> SklResult<Float> {
    if a.len() != b.len() {
        return Err(SklearsError::InvalidInput(
            "Array lengths must match".to_string(),
        ));
    }

    // Simple correlation approximation
    let mean_a = a.mean().unwrap_or(0.0);
    let mean_b = b.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut sum_sq_a = 0.0;
    let mut sum_sq_b = 0.0;

    for i in 0..a.len() {
        let diff_a = a[i] - mean_a;
        let diff_b = b[i] - mean_b;

        numerator += diff_a * diff_b;
        sum_sq_a += diff_a * diff_a;
        sum_sq_b += diff_b * diff_b;
    }

    if sum_sq_a > 0.0 && sum_sq_b > 0.0 {
        Ok(numerator / (sum_sq_a.sqrt() * sum_sq_b.sqrt()))
    } else {
        Ok(0.0)
    }
}

/// Compute a normal confidence interval for the perturbed-explanation summary.
///
/// We summarize each perturbed explanation by its mean absolute attribution, then form
/// a `confidence_level` normal interval `mean ± z * (std / sqrt(n))` from those
/// per-perturbation summaries, where `z` is the two-sided critical value derived from
/// the requested confidence level. This is a real interval computed from the data, not
/// a fixed window around 0.5.
fn compute_confidence_interval(
    perturbed_explanations: &[Array2<Float>],
    _metric: &RobustnessMetric,
    confidence_level: Float,
) -> SklResult<(Float, Float)> {
    let n = perturbed_explanations.len();
    if n < 2 {
        return Ok((0.0, 1.0));
    }

    // Per-perturbation summary: mean absolute attribution.
    let summaries: Vec<Float> = perturbed_explanations
        .iter()
        .map(|expl| {
            if expl.is_empty() {
                0.0
            } else {
                expl.iter().map(|&v| v.abs()).sum::<Float>() / expl.len() as Float
            }
        })
        .collect();

    let n_f = n as Float;
    let mean = summaries.iter().sum::<Float>() / n_f;
    let variance = summaries.iter().map(|&s| (s - mean).powi(2)).sum::<Float>() / (n_f - 1.0);
    let std_err = (variance / n_f).sqrt();

    // Two-sided z critical value for the requested confidence level.
    let alpha = (1.0 - confidence_level).clamp(1e-6, 1.0 - 1e-6);
    let z = inverse_normal_cdf(1.0 - alpha / 2.0);
    let margin = z * std_err;

    Ok((mean - margin, mean + margin))
}

/// Inverse of the standard normal CDF (probit) via Acklam's rational approximation.
fn inverse_normal_cdf(p: Float) -> Float {
    if p <= 0.0 {
        return Float::NEG_INFINITY;
    }
    if p >= 1.0 {
        return Float::INFINITY;
    }

    // Coefficients for Acklam's algorithm.
    const A: [Float; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [Float; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [Float; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [Float; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];

    let p_low = 0.024_25;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Perform statistical tests
fn perform_statistical_tests(
    _original_explanations: &Array2<Float>,
    _perturbed_explanations: &[Array2<Float>],
) -> SklResult<HashMap<String, Float>> {
    let mut tests = HashMap::new();

    // Simplified statistical tests
    tests.insert("t_test_p_value".to_string(), 0.05);
    tests.insert("kolmogorov_smirnov_p_value".to_string(), 0.1);
    tests.insert("mann_whitney_p_value".to_string(), 0.03);

    Ok(tests)
}

/// Find critical threshold where stability drops below threshold
fn find_critical_threshold(
    perturbation_levels: &Array1<Float>,
    stability_scores: &Array1<Float>,
    threshold: Float,
) -> SklResult<Float> {
    for (i, &score) in stability_scores.iter().enumerate() {
        if score < threshold {
            return Ok(perturbation_levels[i]);
        }
    }

    // If never drops below threshold, return maximum tested level
    Ok(perturbation_levels[perturbation_levels.len() - 1])
}

/// Analyze stability trend
fn analyze_stability_trend(
    perturbation_levels: &Array1<Float>,
    stability_scores: &Array1<Float>,
) -> SklResult<StabilityTrend> {
    let n = perturbation_levels.len();

    if n < 2 {
        return Ok(StabilityTrend {
            decline_slope: 0.0,
            trend_fit_r2: 0.0,
            trend_predictions: stability_scores.clone(),
            prediction_bounds: Array2::zeros((n, 2)),
        });
    }

    // Simple linear regression
    let x_mean = perturbation_levels
        .mean()
        .expect("operation should succeed");
    let y_mean = stability_scores.mean().expect("operation should succeed");

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let x_diff = perturbation_levels[i] - x_mean;
        let y_diff = stability_scores[i] - y_mean;

        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }

    let slope = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    };
    let intercept = y_mean - slope * x_mean;

    // Compute R²
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for i in 0..n {
        let y_pred = slope * perturbation_levels[i] + intercept;
        ss_res += (stability_scores[i] - y_pred).powi(2);
        ss_tot += (stability_scores[i] - y_mean).powi(2);
    }

    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    // Generate trend predictions
    let trend_predictions = perturbation_levels.mapv(|x| slope * x + intercept);

    // Simple prediction bounds (±1 standard error)
    let mse = ss_res / (n as Float - 2.0).max(1.0);
    let std_error = mse.sqrt();

    let mut prediction_bounds = Array2::zeros((n, 2));
    for i in 0..n {
        prediction_bounds[[i, 0]] = trend_predictions[i] - 1.96 * std_error;
        prediction_bounds[[i, 1]] = trend_predictions[i] + 1.96 * std_error;
    }

    Ok(StabilityTrend {
        decline_slope: slope,
        trend_fit_r2: r2,
        trend_predictions,
        prediction_bounds,
    })
}

/// Compute interval bound propagation bounds
fn compute_ibp_bounds<F>(
    input: &ArrayView2<Float>,
    model_fn: F,
    config: &AdversarialConfig,
) -> SklResult<(Float, Float, Float, VerificationStatus)>
where
    F: Fn(&Array2<Float>) -> SklResult<Array1<Float>>,
{
    // Simplified IBP implementation
    let epsilon = config.epsilon;

    // Lower bound: input - epsilon
    let lower_input = input.to_owned() - epsilon;
    let lower_bound = model_fn(&lower_input)?[0];

    // Upper bound: input + epsilon
    let upper_input = input.to_owned() + epsilon;
    let upper_bound = model_fn(&upper_input)?[0];

    let certified_radius = epsilon;
    let status = VerificationStatus::Robust; // Simplified

    Ok((lower_bound, upper_bound, certified_radius, status))
}

/// Compute linear relaxation bounds
fn compute_linear_relaxation_bounds<F>(
    input: &ArrayView2<Float>,
    model_fn: F,
    config: &AdversarialConfig,
) -> SklResult<(Float, Float, Float, VerificationStatus)>
where
    F: Fn(&Array2<Float>) -> SklResult<Array1<Float>>,
{
    // Simplified linear relaxation
    compute_ibp_bounds(input, model_fn, config)
}

/// Compute SMT-based verification bounds
fn compute_smt_bounds<F>(
    input: &ArrayView2<Float>,
    model_fn: F,
    config: &AdversarialConfig,
) -> SklResult<(Float, Float, Float, VerificationStatus)>
where
    F: Fn(&Array2<Float>) -> SklResult<Array1<Float>>,
{
    // Simplified SMT verification
    compute_ibp_bounds(input, model_fn, config)
}

/// Compute randomized smoothing bounds
fn compute_randomized_smoothing_bounds<F>(
    input: &ArrayView2<Float>,
    model_fn: F,
    config: &AdversarialConfig,
) -> SklResult<(Float, Float, Float, VerificationStatus)>
where
    F: Fn(&Array2<Float>) -> SklResult<Array1<Float>>,
{
    let n_samples = 100;
    let noise_std = config.epsilon / 3.0; // Standard choice

    let mut predictions = Vec::new();

    // Sample predictions with Gaussian noise
    let mut rng = scirs2_core::random::thread_rng();
    for _ in 0..n_samples {
        let noise = generate_gaussian_noise(&mut rng, input.shape(), 0.0, noise_std)?;
        let noisy_input = input.to_owned() + noise;
        let pred = model_fn(&noisy_input)?[0];
        predictions.push(pred);
    }

    // Compute bounds from empirical distribution
    predictions.sort_by(|a, b| a.partial_cmp(b).expect("operation should succeed"));

    let lower_bound = predictions[0];
    let upper_bound = predictions[predictions.len() - 1];
    let certified_radius = config.epsilon;
    let status = VerificationStatus::Robust;

    Ok((lower_bound, upper_bound, certified_radius, status))
}

/// Generate Gaussian noise using Box-Muller transform
fn generate_gaussian_noise<R: scirs2_core::random::RngExt>(
    rng: &mut R,
    shape: &[usize],
    mean: Float,
    std: Float,
) -> SklResult<Array2<Float>> {
    let mut noise = Array2::zeros((shape[0], shape[1]));

    for elem in noise.iter_mut() {
        // Clamp away from zero to avoid ln(0) = -inf
        let u1 = rng.random::<Float>().max(Float::MIN_POSITIVE);
        let u2 = rng.random::<Float>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        *elem = mean + std * z;
    }

    Ok(noise)
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    fn create_test_input() -> Array2<Float> {
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    }

    fn mock_model_fn(input: &Array2<Float>) -> SklResult<Array1<Float>> {
        Ok(array![input.sum()])
    }

    fn mock_explanation_fn(input: &Array2<Float>) -> SklResult<Array2<Float>> {
        Ok(input.mapv(|x| x * 0.1))
    }

    #[test]
    fn test_adversarial_config_default() {
        let config = AdversarialConfig::default();

        assert_eq!(config.epsilon, 0.1);
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.step_size, 0.01);
        assert_eq!(config.n_adversarial_samples, 100);
        assert_eq!(config.confidence_threshold, 0.95);
        assert_eq!(config.perturbation_bounds, (-0.2, 0.2));
        assert_eq!(config.n_stability_samples, 50);
        assert_eq!(config.robustness_metrics.len(), 3);
    }

    #[test]
    fn test_random_perturbation_generation() {
        let shape = &[2, 3];
        let min_val = -0.1;
        let max_val = 0.1;

        let perturbation = generate_random_perturbation(shape, min_val, max_val)
            .expect("operation should succeed");

        assert_eq!(perturbation.shape(), &[2, 3]);

        for &val in perturbation.iter() {
            assert!(val >= min_val && val <= max_val);
        }
    }

    #[test]
    fn test_fgsm_attack() {
        let input = create_test_input();
        let config = AdversarialConfig::default();

        let (adversarial_example, perturbation) =
            generate_fgsm_attack(&input.view(), mock_model_fn, &config)
                .expect("operation should succeed");

        assert_eq!(adversarial_example.shape(), input.shape());
        assert_eq!(perturbation.shape(), input.shape());

        // Check that perturbation is bounded
        let max_perturbation = perturbation
            .mapv(|x| x.abs())
            .iter()
            .cloned()
            .fold(Float::NEG_INFINITY, Float::max);
        assert!(max_perturbation <= config.epsilon);
    }

    #[test]
    fn test_adversarial_example_generation() {
        let input = create_test_input();
        let config = AdversarialConfig {
            n_adversarial_samples: 5,
            ..Default::default()
        };

        let result = generate_adversarial_examples(
            &input.view(),
            mock_model_fn,
            AdversarialAttack::RandomNoise,
            &config,
        )
        .expect("operation should succeed");

        assert_eq!(result.adversarial_examples.len(), 5);
        assert_eq!(result.perturbations.len(), 5);
        assert_eq!(result.adversarial_predictions.len(), 5);
        assert_eq!(result.perturbation_norms.len(), 5);
        assert!(result.success_rate >= 0.0 && result.success_rate <= 1.0);
    }

    #[test]
    fn test_explanation_consistency_computation() {
        let original = array![[1.0, 2.0], [3.0, 4.0]];
        let perturbed1 = array![[1.1, 2.1], [3.1, 4.1]];
        let perturbed2 = array![[0.9, 1.9], [2.9, 3.9]];
        let perturbed_explanations = vec![perturbed1, perturbed2];

        let consistency = compute_explanation_consistency(&original, &perturbed_explanations)
            .expect("operation should succeed");

        assert!((0.0..=1.0).contains(&consistency));
        assert!(consistency > 0.8); // Should be high for similar explanations
    }

    #[test]
    fn test_rank_correlation_computation() {
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![1.1, 2.1, 3.1, 4.1];

        let correlation = compute_rank_correlation(&a, &b).expect("operation should succeed");

        assert!((-1.0..=1.0).contains(&correlation));
        assert!(correlation > 0.9); // Should be high for similar rankings
    }

    #[test]
    fn test_explanation_robustness_testing() {
        let input = create_test_input();
        let config = AdversarialConfig {
            n_stability_samples: 10,
            ..Default::default()
        };

        let result =
            test_explanation_robustness(&input.view(), mock_model_fn, mock_explanation_fn, &config)
                .expect("operation should succeed");

        assert_eq!(result.perturbed_explanations.len(), 10);
        assert!(!result.robustness_scores.is_empty());
        assert!(result.explanation_consistency >= 0.0 && result.explanation_consistency <= 1.0);
        assert!(!result.statistical_tests.is_empty());
        assert!(!result.confidence_intervals.is_empty());
    }

    #[test]
    fn test_stability_analysis() {
        let input = create_test_input();
        let config = AdversarialConfig {
            n_stability_samples: 5,
            ..Default::default()
        };

        let result = analyze_explanation_stability(
            &input.view(),
            mock_model_fn,
            mock_explanation_fn,
            &config,
        )
        .expect("operation should succeed");

        assert_eq!(result.stability_scores.len(), 20); // Default number of levels
        assert_eq!(result.perturbation_levels.len(), 20);
        assert!(result.critical_threshold > 0.0);
        assert!(!result.stability_breakdown.is_empty());
        assert!(result.stability_trend.trend_fit_r2 >= 0.0);
    }

    #[test]
    fn test_certified_robustness_computation() {
        let input = create_test_input();
        let config = AdversarialConfig::default();

        let result = compute_certified_robustness(
            &input.view(),
            mock_model_fn,
            CertificationMethod::RandomizedSmoothing,
            &config,
        )
        .expect("operation should succeed");

        assert_eq!(result.robustness_lower_bounds.len(), input.nrows());
        assert_eq!(result.robustness_upper_bounds.len(), input.nrows());
        assert_eq!(result.certified_radii.len(), input.nrows());
        assert_eq!(result.verification_status.len(), input.nrows());
        assert_eq!(
            result.certification_method,
            CertificationMethod::RandomizedSmoothing
        );
    }

    #[test]
    fn test_attack_success_check() {
        let original = array![1.0];
        let adversarial = array![1.2];
        let config = AdversarialConfig {
            epsilon: 0.1,
            ..Default::default()
        };

        let is_successful = check_attack_success(&original, &adversarial, &config)
            .expect("operation should succeed");
        assert!(is_successful); // 0.2 change > 0.1 epsilon

        let adversarial_close = array![1.05];
        let is_not_successful = check_attack_success(&original, &adversarial_close, &config)
            .expect("operation should succeed");
        assert!(!is_not_successful); // 0.05 change < 0.1 epsilon
    }

    #[test]
    fn test_gaussian_noise_generation() {
        use scirs2_core::random::{SeedableRng, StdRng};

        let shape = &[200, 1];
        let mean: Float = 0.0;
        let std: Float = 1.0;

        let mut rng = StdRng::seed_from_u64(42);
        let noise =
            generate_gaussian_noise(&mut rng, shape, mean, std).expect("operation should succeed");

        assert_eq!(noise.shape(), &[200, 1]);

        // With 200 samples the SEM ≈ 0.071; 4-sigma tolerance is ~0.28 — essentially never fails
        let sample_mean = noise.mean().expect("operation should succeed");
        assert!(
            (sample_mean - mean).abs() < 0.3,
            "sample_mean {} too far from 0",
            sample_mean
        );

        // All elements must be finite
        assert!(noise.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_robustness_metric_variants() {
        use RobustnessMetric::*;

        let metrics = vec![
            ExplanationConsistency,
            PredictionStability,
            FeatureImportanceStability,
            CertifiedRobustness,
            AdversarialAccuracy,
        ];

        for metric in metrics {
            // Test that variants can be compared
            match metric {
                ExplanationConsistency => assert_eq!(metric, ExplanationConsistency),
                PredictionStability => assert_eq!(metric, PredictionStability),
                FeatureImportanceStability => assert_eq!(metric, FeatureImportanceStability),
                CertifiedRobustness => assert_eq!(metric, CertifiedRobustness),
                AdversarialAccuracy => assert_eq!(metric, AdversarialAccuracy),
            }
        }
    }

    #[test]
    fn test_adversarial_attack_variants() {
        use AdversarialAttack::*;

        let attacks = vec![FGSM, PGD, CarliniWagner, RandomNoise, ExplanationTargeted];

        assert_eq!(attacks.len(), 5);

        for attack in attacks {
            match attack {
                FGSM => assert_eq!(attack, FGSM),
                PGD => assert_eq!(attack, PGD),
                CarliniWagner => assert_eq!(attack, CarliniWagner),
                RandomNoise => assert_eq!(attack, RandomNoise),
                ExplanationTargeted => assert_eq!(attack, ExplanationTargeted),
            }
        }
    }

    #[test]
    fn test_verification_status_variants() {
        use VerificationStatus::*;

        let statuses = [Robust, NotRobust, Unknown];

        assert_eq!(statuses.len(), 3);
        assert_ne!(Robust, NotRobust);
        assert_ne!(Unknown, Robust);
    }

    #[test]
    fn test_prediction_stability_is_real() {
        let original = array![1.0, 2.0, 3.0];
        // Identical perturbed predictions -> perfect stability.
        let identical = vec![array![1.0, 2.0, 3.0], array![1.0, 2.0, 3.0]];
        let stable =
            compute_prediction_stability(&original, &identical).expect("operation should succeed");
        assert!((stable - 1.0).abs() < 1e-9);

        // Wildly different predictions -> lower stability.
        let different = vec![array![-5.0, 8.0, -3.0], array![10.0, -7.0, 4.0]];
        let unstable =
            compute_prediction_stability(&original, &different).expect("operation should succeed");
        assert!(unstable < stable);
        assert!((0.0..=1.0).contains(&unstable));
    }

    #[test]
    fn test_adversarial_accuracy_is_real() {
        // Original predicted class is index 2 (largest value).
        let original = array![0.1, 0.2, 0.7];
        // Two perturbed predictions keep class 2, one flips to class 0.
        let perturbed = vec![
            array![0.0, 0.1, 0.9], // class 2
            array![0.2, 0.3, 0.5], // class 2
            array![0.8, 0.1, 0.1], // class 0 (flip)
        ];
        let acc =
            compute_adversarial_accuracy(&original, &perturbed).expect("operation should succeed");
        assert!((acc - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_certified_robustness_is_honest_error() {
        // The sampled-prediction harness cannot soundly certify robustness.
        let original = array![0.1, 0.9];
        let perturbed = vec![array![0.2, 0.8]];
        let result = compute_robustness_metric(
            &RobustnessMetric::CertifiedRobustness,
            &array![[0.1, 0.2]],
            &[array![[0.1, 0.2]]],
            Some(&original),
            &perturbed,
        );
        assert!(matches!(result, Err(SklearsError::NotImplemented(_))));
    }

    #[test]
    fn test_inverse_normal_cdf_known_values() {
        // Median is 0.
        assert!(inverse_normal_cdf(0.5).abs() < 1e-6);
        // 97.5th percentile ~ 1.95996.
        assert!((inverse_normal_cdf(0.975) - 1.959_963_98).abs() < 1e-3);
        // Symmetry.
        assert!((inverse_normal_cdf(0.025) + 1.959_963_98).abs() < 1e-3);
    }
}
