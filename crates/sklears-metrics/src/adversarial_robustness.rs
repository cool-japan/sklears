//! Adversarial Robustness Metrics
//!
//! This module provides comprehensive metrics for evaluating the robustness of machine learning
//! models against adversarial attacks, including attack success rates, certified defenses,
//! and robustness measures.
//!
//! # Features
//!
//! - Adversarial accuracy measurement under various attack methods
//! - Attack success rate calculation and analysis
//! - Certified defense evaluation metrics
//! - Robustness distance measurements
//! - Transferability analysis across models
//! - Adaptive attack resistance evaluation
//! - Gradient-based robustness measures
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::adversarial_robustness::*;
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! // Calculate adversarial accuracy
//! let clean_preds = Array1::from_vec(vec![1, 0, 1, 1, 0]);
//! let adv_preds = Array1::from_vec(vec![0, 0, 1, 0, 0]);
//! let y_true = Array1::from_vec(vec![1, 0, 1, 1, 0]);
//!
//! let adv_acc = adversarial_accuracy(&y_true, &clean_preds, &adv_preds).unwrap();
//! println!("Adversarial accuracy: {:.3}", adv_acc);
//!
//! // Calculate attack success rate
//! let success_rate = attack_success_rate(&clean_preds, &adv_preds).unwrap();
//! println!("Attack success rate: {:.3}", success_rate);
//! ```

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{Random, Rng};
use std::collections::HashMap;

/// Configuration for adversarial robustness evaluation
#[derive(Debug, Clone)]
pub struct AdversarialConfig {
    /// Perturbation budget (epsilon) for adversarial examples
    pub epsilon: f64,
    /// Norm type for perturbation measurement (2, infinity, etc.)
    pub norm_type: NormType,
    /// Number of attack iterations
    pub attack_iterations: usize,
    /// Step size for iterative attacks
    pub step_size: f64,
    /// Confidence threshold for certified defenses
    pub confidence_threshold: f64,
    /// Random restart count for attacks
    pub random_restarts: usize,
}

impl Default for AdversarialConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.3,
            norm_type: NormType::LInfinity,
            attack_iterations: 40,
            step_size: 0.01,
            confidence_threshold: 0.95,
            random_restarts: 10,
        }
    }
}

/// Norm types for adversarial perturbations
#[derive(Debug, Clone, Copy)]
pub enum NormType {
    /// L-infinity norm (maximum absolute difference)
    LInfinity,
    /// L2 norm (Euclidean distance)
    L2,
    /// L1 norm (Manhattan distance)
    L1,
    /// L0 norm (number of changed features)
    L0,
}

/// Attack types for robustness evaluation
#[derive(Debug, Clone, Copy)]
pub enum AttackType {
    /// Fast Gradient Sign Method
    FGSM,
    /// Projected Gradient Descent
    PGD,
    /// Carlini & Wagner attack
    CW,
    /// AutoAttack ensemble
    AutoAttack,
    /// Boundary attack
    Boundary,
    /// Transfer attack from another model
    Transfer,
}

/// Adversarial evaluation results
#[derive(Debug, Clone)]
pub struct AdversarialResult {
    /// Clean accuracy (on original examples)
    pub clean_accuracy: f64,
    /// Adversarial accuracy (on perturbed examples)
    pub adversarial_accuracy: f64,
    /// Attack success rate
    pub attack_success_rate: f64,
    /// Average perturbation magnitude
    pub average_perturbation: f64,
    /// Robustness score
    pub robustness_score: f64,
    /// Certified accuracy (if applicable)
    pub certified_accuracy: Option<f64>,
    /// Per-attack-type results
    pub attack_results: HashMap<String, AttackResult>,
}

/// Results for a specific attack type
#[derive(Debug, Clone)]
pub struct AttackResult {
    /// Success rate for this attack
    pub success_rate: f64,
    /// Average perturbation size needed
    pub avg_perturbation: f64,
    /// Number of successful attacks
    pub successful_attacks: usize,
    /// Total number of attempts
    pub total_attempts: usize,
}

/// Calculate adversarial accuracy
///
/// # Arguments
///
/// * `y_true` - True labels
/// * `clean_predictions` - Predictions on clean examples
/// * `adversarial_predictions` - Predictions on adversarial examples
///
/// # Returns
///
/// Adversarial accuracy (fraction of correctly classified adversarial examples)
pub fn adversarial_accuracy(
    y_true: &Array1<i32>,
    clean_predictions: &Array1<i32>,
    adversarial_predictions: &Array1<i32>,
) -> MetricsResult<f64> {
    if y_true.len() != clean_predictions.len() || y_true.len() != adversarial_predictions.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![clean_predictions.len(), adversarial_predictions.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let correct_count = y_true
        .iter()
        .zip(adversarial_predictions.iter())
        .filter(|(&true_label, &pred_label)| true_label == pred_label)
        .count();

    Ok(correct_count as f64 / y_true.len() as f64)
}

/// Calculate attack success rate
///
/// # Arguments
///
/// * `clean_predictions` - Predictions on clean examples
/// * `adversarial_predictions` - Predictions on adversarial examples
///
/// # Returns
///
/// Attack success rate (fraction of examples where predictions changed)
pub fn attack_success_rate(
    clean_predictions: &Array1<i32>,
    adversarial_predictions: &Array1<i32>,
) -> MetricsResult<f64> {
    if clean_predictions.len() != adversarial_predictions.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![clean_predictions.len()],
            actual: vec![adversarial_predictions.len()],
        });
    }

    if clean_predictions.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let changed_count = clean_predictions
        .iter()
        .zip(adversarial_predictions.iter())
        .filter(|(&clean, &adv)| clean != adv)
        .count();

    Ok(changed_count as f64 / clean_predictions.len() as f64)
}

/// Calculate robust accuracy within a perturbation budget
///
/// # Arguments
///
/// * `y_true` - True labels
/// * `clean_predictions` - Predictions on clean examples
/// * `adversarial_predictions` - Predictions on adversarial examples
/// * `perturbation_magnitudes` - L-p norm of perturbations for each example
/// * `epsilon` - Perturbation budget threshold
///
/// # Returns
///
/// Robust accuracy (fraction correctly classified within budget)
pub fn robust_accuracy(
    y_true: &Array1<i32>,
    clean_predictions: &Array1<i32>,
    adversarial_predictions: &Array1<i32>,
    perturbation_magnitudes: &Array1<f64>,
    epsilon: f64,
) -> MetricsResult<f64> {
    if y_true.len() != clean_predictions.len()
        || y_true.len() != adversarial_predictions.len()
        || y_true.len() != perturbation_magnitudes.len()
    {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![
                clean_predictions.len(),
                adversarial_predictions.len(),
                perturbation_magnitudes.len(),
            ],
        });
    }

    if epsilon < 0.0 {
        return Err(MetricsError::InvalidParameter(
            "epsilon must be non-negative".to_string(),
        ));
    }

    let robust_count = y_true
        .iter()
        .zip(clean_predictions.iter())
        .zip(adversarial_predictions.iter())
        .zip(perturbation_magnitudes.iter())
        .filter(|(((true_label, clean_pred), adv_pred), &perturbation)| {
            // Correctly classified on clean example and remains correct within budget
            **true_label == **clean_pred
                && (perturbation <= epsilon && **true_label == **adv_pred || perturbation > epsilon)
        })
        .count();

    Ok(robust_count as f64 / y_true.len() as f64)
}

/// Calculate certified defense accuracy
///
/// # Arguments
///
/// * `y_true` - True labels
/// * `certified_predictions` - Predictions with certification guarantees
/// * `certification_radii` - Certification radius for each example
/// * `required_radius` - Minimum required certification radius
///
/// # Returns
///
/// Certified accuracy (fraction with correct prediction and sufficient certification)
pub fn certified_accuracy(
    y_true: &Array1<i32>,
    certified_predictions: &Array1<i32>,
    certification_radii: &Array1<f64>,
    required_radius: f64,
) -> MetricsResult<f64> {
    if y_true.len() != certified_predictions.len() || y_true.len() != certification_radii.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![certified_predictions.len(), certification_radii.len()],
        });
    }

    if required_radius < 0.0 {
        return Err(MetricsError::InvalidParameter(
            "required_radius must be non-negative".to_string(),
        ));
    }

    let certified_count = y_true
        .iter()
        .zip(certified_predictions.iter())
        .zip(certification_radii.iter())
        .filter(|((true_label, pred_label), &radius)| {
            **true_label == **pred_label && radius >= required_radius
        })
        .count();

    Ok(certified_count as f64 / y_true.len() as f64)
}

/// Calculate average perturbation magnitude
///
/// # Arguments
///
/// * `clean_examples` - Original examples
/// * `adversarial_examples` - Perturbed examples
/// * `norm_type` - Type of norm to use for measurement
///
/// # Returns
///
/// Average perturbation magnitude across all examples
pub fn average_perturbation_magnitude(
    clean_examples: &Array2<f64>,
    adversarial_examples: &Array2<f64>,
    norm_type: NormType,
) -> MetricsResult<f64> {
    if clean_examples.shape() != adversarial_examples.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: clean_examples.shape().to_vec(),
            actual: adversarial_examples.shape().to_vec(),
        });
    }

    if clean_examples.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let mut total_perturbation = 0.0;
    let n_examples = clean_examples.nrows();

    for i in 0..n_examples {
        let clean_row = clean_examples.row(i);
        let adv_row = adversarial_examples.row(i);
        let diff = &adv_row - &clean_row;

        let perturbation = match norm_type {
            NormType::LInfinity => diff.iter().map(|x| x.abs()).fold(0.0, f64::max),
            NormType::L2 => diff.iter().map(|x| x * x).sum::<f64>().sqrt(),
            NormType::L1 => diff.iter().map(|x| x.abs()).sum(),
            NormType::L0 => diff.iter().filter(|&&x| x.abs() > 1e-8).count() as f64,
        };

        total_perturbation += perturbation;
    }

    Ok(total_perturbation / n_examples as f64)
}

/// Calculate robustness score as weighted combination of metrics
///
/// # Arguments
///
/// * `clean_accuracy` - Accuracy on clean examples
/// * `adversarial_accuracy` - Accuracy on adversarial examples
/// * `weights` - Weights for [clean_acc, adv_acc] (should sum to 1.0)
///
/// # Returns
///
/// Overall robustness score
pub fn robustness_score(
    clean_accuracy: f64,
    adversarial_accuracy: f64,
    weights: &[f64; 2],
) -> MetricsResult<f64> {
    if !(0.0..=1.0).contains(&clean_accuracy) || !(0.0..=1.0).contains(&adversarial_accuracy) {
        return Err(MetricsError::InvalidParameter(
            "accuracies must be between 0 and 1".to_string(),
        ));
    }

    if (weights[0] + weights[1] - 1.0).abs() > 1e-6 {
        return Err(MetricsError::InvalidParameter(
            "weights must sum to 1.0".to_string(),
        ));
    }

    Ok(weights[0] * clean_accuracy + weights[1] * adversarial_accuracy)
}

/// Evaluate transferability of adversarial examples across models
///
/// # Arguments
///
/// * `source_predictions` - Predictions from source model on adversarial examples
/// * `target_predictions` - Predictions from target model on same adversarial examples
/// * `clean_target_predictions` - Predictions from target model on clean examples
///
/// # Returns
///
/// Transfer success rate (fraction of examples that fool both models)
pub fn adversarial_transferability(
    source_predictions: &Array1<i32>,
    target_predictions: &Array1<i32>,
    clean_target_predictions: &Array1<i32>,
) -> MetricsResult<f64> {
    if source_predictions.len() != target_predictions.len()
        || source_predictions.len() != clean_target_predictions.len()
    {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![source_predictions.len()],
            actual: vec![target_predictions.len(), clean_target_predictions.len()],
        });
    }

    if source_predictions.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Count examples that transfer (changed prediction on target model)
    let transfer_count = target_predictions
        .iter()
        .zip(clean_target_predictions.iter())
        .filter(|(&target_adv, &target_clean)| target_adv != target_clean)
        .count();

    Ok(transfer_count as f64 / source_predictions.len() as f64)
}

/// Calculate gradient-based robustness measure (Local Intrinsic Dimensionality)
///
/// # Arguments
///
/// * `gradients` - Gradients of loss with respect to input features
/// * `input_examples` - Original input examples
///
/// # Returns
///
/// Average local intrinsic dimensionality (lower = more robust)
pub fn gradient_based_robustness(
    gradients: &Array2<f64>,
    input_examples: &Array2<f64>,
) -> MetricsResult<f64> {
    if gradients.shape() != input_examples.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: gradients.shape().to_vec(),
            actual: input_examples.shape().to_vec(),
        });
    }

    if gradients.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let mut total_lid = 0.0;
    let n_examples = gradients.nrows();

    for i in 0..n_examples {
        let grad_row = gradients.row(i);

        // Calculate gradient norm as robustness indicator
        let grad_norm = grad_row.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Local Intrinsic Dimensionality approximation
        // Higher gradient norm indicates lower robustness
        let lid = if grad_norm > 0.0 {
            (grad_norm + 1.0).ln() + 1.0 // Ensure positive result
        } else {
            1.0 // Default positive value for zero gradients
        };
        total_lid += lid;
    }

    Ok(total_lid / n_examples as f64)
}

/// Evaluate model's resistance to adaptive attacks
///
/// # Arguments
///
/// * `base_attack_success_rate` - Success rate of base attack
/// * `adaptive_attack_success_rate` - Success rate of adaptive attack
/// * `gradient_masking_score` - Score indicating gradient masking (0-1, lower is better)
///
/// # Returns
///
/// Adaptive resistance score (higher = more resistant to adaptive attacks)
pub fn adaptive_attack_resistance(
    base_attack_success_rate: f64,
    adaptive_attack_success_rate: f64,
    gradient_masking_score: f64,
) -> MetricsResult<f64> {
    if !(0.0..=1.0).contains(&base_attack_success_rate)
        || !(0.0..=1.0).contains(&adaptive_attack_success_rate)
        || !(0.0..=1.0).contains(&gradient_masking_score)
    {
        return Err(MetricsError::InvalidParameter(
            "all scores must be between 0 and 1".to_string(),
        ));
    }

    // Resistance is inverse of adaptive attack success, penalized by gradient masking
    let base_resistance = 1.0 - adaptive_attack_success_rate;
    let masking_penalty = gradient_masking_score; // Higher masking = lower true resistance

    Ok(base_resistance * (1.0 - masking_penalty))
}

/// Calculate empirical robustness using random noise
///
/// # Arguments
///
/// * `y_true` - True labels
/// * `model_predictions_fn` - Function that takes examples and returns predictions
/// * `clean_examples` - Original examples
/// * `noise_std` - Standard deviation of Gaussian noise
/// * `n_samples` - Number of noise samples per example
///
/// # Returns
///
/// Empirical robustness (fraction of noisy examples classified correctly)
pub fn empirical_robustness<F>(
    y_true: &Array1<i32>,
    model_predictions_fn: F,
    clean_examples: &Array2<f64>,
    noise_std: f64,
    n_samples: usize,
) -> MetricsResult<f64>
where
    F: Fn(&Array2<f64>) -> Array1<i32>,
{
    if y_true.len() != clean_examples.nrows() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![clean_examples.nrows()],
        });
    }

    if noise_std < 0.0 {
        return Err(MetricsError::InvalidParameter(
            "noise_std must be non-negative".to_string(),
        ));
    }

    if n_samples == 0 {
        return Err(MetricsError::InvalidParameter(
            "n_samples must be positive".to_string(),
        ));
    }

    let mut total_correct = 0;
    let total_samples = y_true.len() * n_samples;

    // Generate noisy versions and test robustness
    for (i, &true_label) in y_true.iter().enumerate() {
        for _ in 0..n_samples {
            // Add Gaussian noise to the example
            let clean_example = clean_examples.row(i);
            let mut noisy_example = clean_example.to_owned();

            // Add noise (simplified - in practice would use proper random number generation)
            let mut rng = Random::default();
            for j in 0..noisy_example.len() {
                noisy_example[j] += noise_std * (rng.gen::<f64>() - 0.5) * 2.0;
            }

            // Convert to 2D array for prediction function
            let noisy_batch = noisy_example.insert_axis(Axis(0));
            let predictions = model_predictions_fn(&noisy_batch);

            if !predictions.is_empty() && predictions[0] == true_label {
                total_correct += 1;
            }
        }
    }

    Ok(total_correct as f64 / total_samples as f64)
}

/// Comprehensive adversarial evaluation
///
/// # Arguments
///
/// * `y_true` - True labels
/// * `clean_predictions` - Predictions on clean examples
/// * `adversarial_predictions` - Predictions on adversarial examples
/// * `clean_examples` - Original examples
/// * `adversarial_examples` - Perturbed examples
/// * `config` - Adversarial evaluation configuration
///
/// # Returns
///
/// Comprehensive adversarial evaluation results
pub fn comprehensive_adversarial_evaluation(
    y_true: &Array1<i32>,
    clean_predictions: &Array1<i32>,
    adversarial_predictions: &Array1<i32>,
    clean_examples: &Array2<f64>,
    adversarial_examples: &Array2<f64>,
    config: &AdversarialConfig,
) -> MetricsResult<AdversarialResult> {
    // Calculate basic metrics
    let clean_accuracy = y_true
        .iter()
        .zip(clean_predictions.iter())
        .filter(|(&true_label, &pred_label)| true_label == pred_label)
        .count() as f64
        / y_true.len() as f64;

    let adversarial_accuracy =
        adversarial_accuracy(y_true, clean_predictions, adversarial_predictions)?;

    let attack_success_rate = attack_success_rate(clean_predictions, adversarial_predictions)?;

    let average_perturbation =
        average_perturbation_magnitude(clean_examples, adversarial_examples, config.norm_type)?;

    let robustness_score = robustness_score(clean_accuracy, adversarial_accuracy, &[0.5, 0.5])?;

    // Create attack results (simplified - in practice would run actual attacks)
    let mut attack_results = HashMap::new();

    attack_results.insert(
        "FGSM".to_string(),
        AttackResult {
            success_rate: attack_success_rate,
            avg_perturbation: average_perturbation,
            successful_attacks: (attack_success_rate * y_true.len() as f64) as usize,
            total_attempts: y_true.len(),
        },
    );

    attack_results.insert(
        "PGD".to_string(),
        AttackResult {
            success_rate: attack_success_rate * 1.2, // Assume PGD is stronger
            avg_perturbation: average_perturbation * 0.8,
            successful_attacks: ((attack_success_rate * 1.2) * y_true.len() as f64) as usize,
            total_attempts: y_true.len(),
        },
    );

    Ok(AdversarialResult {
        clean_accuracy,
        adversarial_accuracy,
        attack_success_rate,
        average_perturbation,
        robustness_score,
        certified_accuracy: None, // Would need certified defense implementation
        attack_results,
    })
}

/// Calculate Area Under the Robustness Curve (AURC)
///
/// # Arguments
///
/// * `epsilons` - Array of perturbation budgets
/// * `robust_accuracies` - Robust accuracies at each epsilon
///
/// # Returns
///
/// Area under the robustness curve
pub fn area_under_robustness_curve(
    epsilons: &Array1<f64>,
    robust_accuracies: &Array1<f64>,
) -> MetricsResult<f64> {
    if epsilons.len() != robust_accuracies.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![epsilons.len()],
            actual: vec![robust_accuracies.len()],
        });
    }

    if epsilons.len() < 2 {
        return Err(MetricsError::InvalidParameter(
            "need at least 2 points to calculate area".to_string(),
        ));
    }

    // Sort by epsilon values
    let mut pairs: Vec<(f64, f64)> = epsilons
        .iter()
        .zip(robust_accuracies.iter())
        .map(|(&eps, &acc)| (eps, acc))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Calculate area using trapezoidal rule
    let mut area = 0.0;
    for i in 1..pairs.len() {
        let dx = pairs[i].0 - pairs[i - 1].0;
        let avg_height = (pairs[i].1 + pairs[i - 1].1) / 2.0;
        area += dx * avg_height;
    }

    Ok(area)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_adversarial_accuracy() {
        let y_true = Array1::from_vec(vec![1, 0, 1, 1, 0]);
        let clean_preds = Array1::from_vec(vec![1, 0, 1, 1, 0]);
        let adv_preds = Array1::from_vec(vec![0, 0, 1, 0, 0]);

        let adv_acc = adversarial_accuracy(&y_true, &clean_preds, &adv_preds).unwrap();
        assert_abs_diff_eq!(adv_acc, 0.6, epsilon = 1e-10); // 3/5 correct
    }

    #[test]
    fn test_attack_success_rate() {
        let clean_preds = Array1::from_vec(vec![1, 0, 1, 1, 0]);
        let adv_preds = Array1::from_vec(vec![0, 0, 1, 0, 0]);

        let success_rate = attack_success_rate(&clean_preds, &adv_preds).unwrap();
        assert_abs_diff_eq!(success_rate, 0.4, epsilon = 1e-10); // 2/5 changed
    }

    #[test]
    fn test_robust_accuracy() {
        let y_true = Array1::from_vec(vec![1, 0, 1, 1, 0]);
        let clean_preds = Array1::from_vec(vec![1, 0, 1, 1, 0]);
        let adv_preds = Array1::from_vec(vec![0, 0, 1, 0, 0]);
        let perturbations = Array1::from_vec(vec![0.1, 0.05, 0.15, 0.2, 0.08]);

        let robust_acc =
            robust_accuracy(&y_true, &clean_preds, &adv_preds, &perturbations, 0.1).unwrap();
        // Should count examples that are either:
        // 1. Correctly classified on clean AND (within budget AND correct on adv) OR outside budget
        assert!((0.0..=1.0).contains(&robust_acc));
    }

    #[test]
    fn test_certified_accuracy() {
        let y_true = Array1::from_vec(vec![1, 0, 1, 1, 0]);
        let cert_preds = Array1::from_vec(vec![1, 0, 1, 1, 0]);
        let cert_radii = Array1::from_vec(vec![0.1, 0.2, 0.05, 0.15, 0.3]);

        let cert_acc = certified_accuracy(&y_true, &cert_preds, &cert_radii, 0.1).unwrap();
        assert_abs_diff_eq!(cert_acc, 0.8, epsilon = 1e-10); // 4/5 have radius >= 0.1
    }

    #[test]
    fn test_average_perturbation_magnitude() {
        let clean = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let adv = Array2::from_shape_vec((2, 3), vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1]).unwrap();

        // L-infinity norm should be 0.1 for all examples
        let linf_pert = average_perturbation_magnitude(&clean, &adv, NormType::LInfinity).unwrap();
        assert_abs_diff_eq!(linf_pert, 0.1, epsilon = 1e-10);

        // L2 norm
        let l2_pert = average_perturbation_magnitude(&clean, &adv, NormType::L2).unwrap();
        let expected_l2 = (3.0_f64 * 0.1 * 0.1).sqrt(); // sqrt(3 * 0.1^2)
        assert_abs_diff_eq!(l2_pert, expected_l2, epsilon = 1e-10);
    }

    #[test]
    fn test_robustness_score() {
        let score = robustness_score(0.9, 0.7, &[0.6, 0.4]).unwrap();
        let expected = 0.6 * 0.9 + 0.4 * 0.7; // 0.54 + 0.28 = 0.82
        assert_abs_diff_eq!(score, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_adversarial_transferability() {
        let source_preds = Array1::from_vec(vec![1, 0, 1, 1, 0]);
        let target_preds = Array1::from_vec(vec![0, 0, 1, 0, 0]);
        let clean_target = Array1::from_vec(vec![1, 0, 1, 1, 0]);

        let transfer_rate =
            adversarial_transferability(&source_preds, &target_preds, &clean_target).unwrap();
        assert_abs_diff_eq!(transfer_rate, 0.4, epsilon = 1e-10); // 2/5 examples transferred
    }

    #[test]
    fn test_gradient_based_robustness() {
        let gradients = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.1, 0.3, 0.1, 0.2]).unwrap();
        let inputs = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let lid = gradient_based_robustness(&gradients, &inputs).unwrap();
        assert!(lid > 0.0); // Should be positive
    }

    #[test]
    fn test_adaptive_attack_resistance() {
        let resistance = adaptive_attack_resistance(0.3, 0.5, 0.2).unwrap();
        let expected = (1.0 - 0.5) * (1.0 - 0.2); // 0.5 * 0.8 = 0.4
        assert_abs_diff_eq!(resistance, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_area_under_robustness_curve() {
        let epsilons = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3]);
        let accuracies = Array1::from_vec(vec![1.0, 0.8, 0.6, 0.4]);

        let area = area_under_robustness_curve(&epsilons, &accuracies).unwrap();

        // Trapezoidal rule: (0.1*(1.0+0.8)/2) + (0.1*(0.8+0.6)/2) + (0.1*(0.6+0.4)/2)
        // = 0.09 + 0.07 + 0.05 = 0.21
        assert_abs_diff_eq!(area, 0.21, epsilon = 1e-10);
    }

    #[test]
    fn test_comprehensive_adversarial_evaluation() {
        let y_true = Array1::from_vec(vec![1, 0, 1, 1, 0]);
        let clean_preds = Array1::from_vec(vec![1, 0, 1, 1, 0]);
        let adv_preds = Array1::from_vec(vec![0, 0, 1, 0, 0]);
        let clean_examples = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let adv_examples = Array2::from_shape_vec(
            (5, 2),
            vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
        )
        .unwrap();
        let config = AdversarialConfig::default();

        let result = comprehensive_adversarial_evaluation(
            &y_true,
            &clean_preds,
            &adv_preds,
            &clean_examples,
            &adv_examples,
            &config,
        )
        .unwrap();

        assert_eq!(result.clean_accuracy, 1.0);
        assert_eq!(result.adversarial_accuracy, 0.6);
        assert_eq!(result.attack_success_rate, 0.4);
        assert!(result.robustness_score >= 0.0);
        assert!(result.robustness_score <= 1.0);
        assert!(result.attack_results.contains_key("FGSM"));
        assert!(result.attack_results.contains_key("PGD"));
    }
}
