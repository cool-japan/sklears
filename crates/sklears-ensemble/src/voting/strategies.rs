//! Voting strategies and algorithms for ensemble methods

use crate::voting::simd_ops::{simd_mean_f32, simd_variance_f32};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, types::Float};

/// Calculate mean of array (scalar implementation)
pub fn mean_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f32 = data.iter().sum();
    sum / data.len() as f32
}

/// Calculate variance of array (scalar implementation)
pub fn variance_f32(data: &[f32], mean: f32) -> f32 {
    if data.len() <= 1 {
        return 0.0;
    }

    let sum_sq_diff: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum();
    sum_sq_diff / (data.len() - 1) as f32
}

/// Calculate entropy of probability distribution (scalar implementation)
pub fn entropy_f32(probabilities: &[f32]) -> f32 {
    let mut entropy = 0.0;
    for &prob in probabilities {
        if prob > 1e-8 {
            entropy -= prob * prob.ln();
        }
    }
    entropy
}

/// Weighted average of predictions (scalar implementation)
pub fn weighted_average_f32(values: &[f32], weights: &[f32]) -> f32 {
    if values.len() != weights.len() || values.is_empty() {
        return 0.0;
    }

    let weighted_sum: f32 = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| v * w)
        .sum();

    let weight_sum: f32 = weights.iter().sum();

    if weight_sum > 0.0 {
        weighted_sum / weight_sum
    } else {
        0.0
    }
}

/// Consensus-based voting strategy
pub fn consensus_voting(
    all_predictions: &[Array1<Float>],
    consensus_threshold: f32,
) -> Result<Array1<Float>> {
    if all_predictions.is_empty() {
        return Err(sklears_core::error::SklearsError::InvalidParameter {
            name: "predictions".to_string(),
            reason: "Empty predictions".to_string(),
        });
    }

    let n_samples = all_predictions[0].len();
    let n_estimators = all_predictions.len();
    let mut result = Array1::zeros(n_samples);

    for sample_idx in 0..n_samples {
        // Count votes for each possible prediction
        let mut vote_counts = std::collections::HashMap::new();

        for estimator_idx in 0..n_estimators {
            let prediction = all_predictions[estimator_idx][sample_idx];
            *vote_counts.entry(prediction as i32).or_insert(0) += 1;
        }

        // Find prediction with highest consensus
        let mut best_prediction = 0.0;
        let mut max_votes = 0;
        let mut consensus_ratio = 0.0;

        for (&prediction, &votes) in vote_counts.iter() {
            if votes > max_votes {
                max_votes = votes;
                best_prediction = prediction as Float;
                consensus_ratio = votes as f32 / n_estimators as f32;
            }
        }

        // Only use consensus prediction if threshold is met
        if consensus_ratio >= consensus_threshold {
            result[sample_idx] = best_prediction;
        } else {
            // Fall back to simple majority voting
            result[sample_idx] = best_prediction;
        }
    }

    Ok(result)
}

/// Meta-voting using learned combination weights
pub fn meta_voting(
    all_predictions: &[Array1<Float>],
    meta_weights: &Array2<Float>,
) -> Result<Array1<Float>> {
    if all_predictions.is_empty() || meta_weights.is_empty() {
        return Err(sklears_core::error::SklearsError::InvalidParameter {
            name: "input_data".to_string(),
            reason: "Empty predictions or weights".to_string(),
        });
    }

    let n_samples = all_predictions[0].len();
    let n_estimators = all_predictions.len().min(meta_weights.ncols());

    if meta_weights.nrows() != n_samples {
        return Err(sklears_core::error::SklearsError::ShapeMismatch {
            expected: format!("{}x{}", n_samples, n_estimators),
            actual: format!("{}x{}", meta_weights.nrows(), meta_weights.ncols()),
        });
    }

    let mut result = Array1::zeros(n_samples);

    for sample_idx in 0..n_samples {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for estimator_idx in 0..n_estimators {
            let prediction = all_predictions[estimator_idx][sample_idx];
            let weight = meta_weights[[sample_idx, estimator_idx]];

            weighted_sum += prediction * weight;
            weight_sum += weight;
        }

        if weight_sum > 1e-8 {
            result[sample_idx] = weighted_sum / weight_sum;
        } else {
            // Fall back to simple average
            let sum: Float = all_predictions[..n_estimators]
                .iter()
                .map(|pred| pred[sample_idx])
                .sum();
            result[sample_idx] = sum / n_estimators as Float;
        }
    }

    Ok(result)
}

/// Dynamic weight adjustment based on recent performance
pub fn dynamic_weight_adjustment(
    current_weights: &[Float],
    recent_performances: &[Float],
    learning_rate: f32,
) -> Result<Vec<Float>> {
    if current_weights.len() != recent_performances.len() {
        return Err(sklears_core::error::SklearsError::ShapeMismatch {
            expected: format!("{}", current_weights.len()),
            actual: format!("{}", recent_performances.len()),
        });
    }

    let n_estimators = current_weights.len();
    let mut new_weights = Vec::with_capacity(n_estimators);

    // Convert to f32 for SIMD operations
    let performances_f32: Vec<f32> = recent_performances.iter().map(|&x| x as f32).collect();
    let weights_f32: Vec<f32> = current_weights.iter().map(|&x| x as f32).collect();

    // Calculate performance statistics
    let mean_performance = simd_mean_f32(&performances_f32);
    let performance_variance = simd_variance_f32(&performances_f32, mean_performance);

    // Update weights based on relative performance
    for i in 0..n_estimators {
        let current_weight = weights_f32[i];
        let performance = performances_f32[i];

        // Calculate relative performance score
        let relative_performance = if performance_variance > 1e-8 {
            (performance - mean_performance) / performance_variance.sqrt()
        } else {
            0.0
        };

        // Update weight: increase for better performers, decrease for worse
        let weight_adjustment = learning_rate * relative_performance;
        let new_weight = (current_weight + weight_adjustment).max(0.01); // Minimum weight

        new_weights.push(new_weight as Float);
    }

    // Normalize weights to sum to 1.0
    let weight_sum: Float = new_weights.iter().sum();
    if weight_sum > 1e-8 {
        for weight in new_weights.iter_mut() {
            *weight /= weight_sum;
        }
    }

    Ok(new_weights)
}

/// Temperature-scaled soft voting
pub fn temperature_scaled_voting(
    all_probabilities: &[Array2<Float>],
    temperature: f32,
) -> Result<Array2<Float>> {
    if all_probabilities.is_empty() {
        return Err(sklears_core::error::SklearsError::InvalidParameter {
            name: "probabilities".to_string(),
            reason: "Empty probabilities".to_string(),
        });
    }

    let n_samples = all_probabilities[0].nrows();
    let n_classes = all_probabilities[0].ncols();
    let n_estimators = all_probabilities.len();

    let mut result = Array2::zeros((n_samples, n_classes));

    for sample_idx in 0..n_samples {
        for class_idx in 0..n_classes {
            let mut scaled_prob_sum = 0.0;

            for estimator_idx in 0..n_estimators {
                let prob = all_probabilities[estimator_idx][[sample_idx, class_idx]] as f32;

                // Apply temperature scaling
                let scaled_prob = if temperature > 1e-6 {
                    (prob / temperature).exp()
                } else if prob > 0.5 {
                    f32::INFINITY
                } else {
                    0.0
                };

                scaled_prob_sum += scaled_prob;
            }

            result[[sample_idx, class_idx]] = (scaled_prob_sum / n_estimators as f32) as Float;
        }

        // Normalize probabilities for this sample
        let row_sum: Float = result.row(sample_idx).sum();
        if row_sum > 1e-8 {
            for class_idx in 0..n_classes {
                result[[sample_idx, class_idx]] /= row_sum;
            }
        }
    }

    Ok(result)
}

/// Rank-based voting using ordinal rankings
pub fn rank_based_voting(all_probabilities: &[Array2<Float>]) -> Result<Array1<Float>> {
    if all_probabilities.is_empty() {
        return Err(sklears_core::error::SklearsError::InvalidParameter {
            name: "probabilities".to_string(),
            reason: "Empty probabilities".to_string(),
        });
    }

    let n_samples = all_probabilities[0].nrows();
    let n_classes = all_probabilities[0].ncols();
    let n_estimators = all_probabilities.len();

    let mut result = Array1::zeros(n_samples);

    for sample_idx in 0..n_samples {
        // Accumulate ranks for each class across all estimators
        let mut class_rank_sums = vec![0.0f32; n_classes];

        for estimator_idx in 0..n_estimators {
            // Get probabilities for this sample from this estimator
            let sample_probs: Vec<f32> = (0..n_classes)
                .map(|class_idx| all_probabilities[estimator_idx][[sample_idx, class_idx]] as f32)
                .collect();

            // Calculate ranks (higher probability gets lower rank number)
            let mut indexed_probs: Vec<(usize, f32)> = sample_probs
                .iter()
                .enumerate()
                .map(|(idx, &prob)| (idx, prob))
                .collect();

            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Assign ranks
            for (rank, (class_idx, _)) in indexed_probs.iter().enumerate() {
                class_rank_sums[*class_idx] += (rank + 1) as f32;
            }
        }

        // Find class with lowest average rank (best ranking)
        let mut best_class = 0;
        let mut lowest_rank_sum = class_rank_sums[0];

        for (class_idx, &rank_sum) in class_rank_sums.iter().enumerate().skip(1) {
            if rank_sum < lowest_rank_sum {
                lowest_rank_sum = rank_sum;
                best_class = class_idx;
            }
        }

        result[sample_idx] = best_class as Float;
    }

    Ok(result)
}

/// Bootstrap aggregation with uncertainty estimation
pub fn bootstrap_aggregation_with_uncertainty(
    bootstrap_predictions: &[Vec<Array1<Float>>],
) -> Result<(Array1<Float>, Array1<Float>)> {
    if bootstrap_predictions.is_empty() {
        return Err(sklears_core::error::SklearsError::InvalidParameter {
            name: "bootstrap_predictions".to_string(),
            reason: "Empty bootstrap predictions".to_string(),
        });
    }

    let n_samples = bootstrap_predictions[0][0].len();
    let mut final_predictions = Array1::zeros(n_samples);
    let mut prediction_uncertainty = Array1::zeros(n_samples);

    for sample_idx in 0..n_samples {
        // Collect all bootstrap predictions for this sample
        let mut sample_predictions = Vec::new();

        for bootstrap_set in bootstrap_predictions {
            for prediction_array in bootstrap_set {
                sample_predictions.push(prediction_array[sample_idx] as f32);
            }
        }

        // Calculate mean and variance as uncertainty measure
        let mean_prediction = simd_mean_f32(&sample_predictions);
        let prediction_variance = simd_variance_f32(&sample_predictions, mean_prediction);

        final_predictions[sample_idx] = mean_prediction as Float;
        prediction_uncertainty[sample_idx] = prediction_variance.sqrt() as Float;
    }

    Ok((final_predictions, prediction_uncertainty))
}

/// Uncertainty-aware voting with confidence weighting
pub fn uncertainty_aware_voting(
    all_predictions: &[Array1<Float>],
    prediction_uncertainties: &[Array1<Float>],
    uncertainty_threshold: f32,
) -> Result<Array1<Float>> {
    if all_predictions.len() != prediction_uncertainties.len() {
        return Err(sklears_core::error::SklearsError::ShapeMismatch {
            expected: format!("{}", all_predictions.len()),
            actual: format!("{}", prediction_uncertainties.len()),
        });
    }

    let n_samples = all_predictions[0].len();
    let n_estimators = all_predictions.len();
    let mut result = Array1::zeros(n_samples);

    for sample_idx in 0..n_samples {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for estimator_idx in 0..n_estimators {
            let prediction = all_predictions[estimator_idx][sample_idx];
            let uncertainty = prediction_uncertainties[estimator_idx][sample_idx] as f32;

            // Calculate confidence weight (inverse of uncertainty)
            let confidence_weight = if uncertainty > uncertainty_threshold {
                1.0 / (1.0 + uncertainty)
            } else {
                1.0
            };

            weighted_sum += prediction * confidence_weight as Float;
            weight_sum += confidence_weight as Float;
        }

        if weight_sum > 1e-8 {
            result[sample_idx] = weighted_sum / weight_sum;
        } else {
            // Fall back to simple average
            let sum: Float = all_predictions.iter().map(|pred| pred[sample_idx]).sum();
            result[sample_idx] = sum / n_estimators as Float;
        }
    }

    Ok(result)
}

/// Adaptive ensemble voting with dynamic strategy selection
pub fn adaptive_ensemble_voting(
    all_predictions: &[Array1<Float>],
    all_probabilities: Option<&[Array2<Float>]>,
    ensemble_diversity: f32,
    performance_history: &[f32],
) -> Result<Array1<Float>> {
    if all_predictions.is_empty() {
        return Err(sklears_core::error::SklearsError::InvalidParameter {
            name: "predictions".to_string(),
            reason: "Empty predictions".to_string(),
        });
    }

    let n_samples = all_predictions[0].len();

    // Choose voting strategy based on ensemble characteristics
    if ensemble_diversity > 0.7 && performance_history.len() > 10 {
        // High diversity: use consensus voting
        consensus_voting(all_predictions, 0.6)
    } else if let Some(probabilities) = all_probabilities {
        // Use entropy-weighted voting if probabilities available
        let entropy_predictions =
            crate::voting::simd_ops::simd_entropy_weighted_voting(probabilities, 1.0);

        // Convert probabilities to class predictions
        let mut result = Array1::zeros(n_samples);
        for sample_idx in 0..n_samples {
            let sample_probs = entropy_predictions.row(sample_idx);
            let mut max_prob = sample_probs[0];
            let mut best_class = 0;

            for (class_idx, &prob) in sample_probs.iter().enumerate().skip(1) {
                if prob > max_prob {
                    max_prob = prob;
                    best_class = class_idx;
                }
            }

            result[sample_idx] = best_class as Float;
        }

        Ok(result)
    } else {
        // Default to weighted voting based on performance
        let mean_performance = simd_mean_f32(performance_history);
        let weights: Vec<Float> = all_predictions
            .iter()
            .enumerate()
            .map(|(idx, _)| {
                if idx < performance_history.len() {
                    performance_history[idx] as Float
                } else {
                    mean_performance as Float
                }
            })
            .collect();

        let mut result = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let predictions: Vec<f32> = all_predictions
                .iter()
                .map(|pred| pred[sample_idx] as f32)
                .collect();

            let weights_f32: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
            result[sample_idx] = weighted_average_f32(&predictions, &weights_f32) as Float;
        }

        Ok(result)
    }
}
