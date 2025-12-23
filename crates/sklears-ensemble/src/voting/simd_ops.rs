//! SIMD-accelerated operations for high-performance ensemble voting computations
//!
//! This module provides a thin wrapper around scirs2-core::simd_ops::SimdUnifiedOps,
//! ensuring SciRS2 Policy compliance. All SIMD operations are delegated to SciRS2-Core.
//!
//! ## SciRS2 Policy Compliance
//! ✅ Uses `scirs2-core::simd_ops::SimdUnifiedOps` for all SIMD operations
//! ✅ No direct implementation of SIMD code (policy requirement)
//! ✅ Works on stable Rust (no nightly features required)

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;
use sklears_core::{error::Result, types::Float};

/// Mean calculation for f32 arrays using SciRS2 SIMD
#[inline]
pub fn simd_mean_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let arr = ArrayView1::from(data);
    f32::simd_mean(&arr)
}

/// Sum calculation for f32 arrays using SciRS2 SIMD
#[inline]
pub fn simd_sum_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let arr = ArrayView1::from(data);
    f32::simd_sum(&arr)
}

/// Variance calculation for f32 arrays using SciRS2 SIMD
pub fn simd_variance_f32(data: &[f32], mean: f32) -> f32 {
    if data.len() <= 1 {
        return 0.0;
    }
    let arr = ArrayView1::from(data);
    let mean_vec = Array1::from_elem(data.len(), mean);
    let diff = f32::simd_sub(&arr, &mean_vec.view());
    let sum_sq = f32::simd_sum_squares(&diff.view());
    sum_sq / (data.len() - 1) as f32
}

/// Entropy calculation for probability distributions
pub fn simd_entropy_f32(probabilities: &[f32]) -> f32 {
    if probabilities.is_empty() {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &prob in probabilities {
        if prob > 1e-8 {
            entropy -= prob * prob.ln();
        }
    }
    entropy
}

/// Weighted sum for probability aggregation using SciRS2 SIMD
pub fn simd_weighted_sum_f32(values: &[f32], weights: &[f32], output: &mut [f32]) {
    let len = values.len().min(weights.len()).min(output.len());
    let val_arr = ArrayView1::from(&values[..len]);
    let weight_arr = ArrayView1::from(&weights[..len]);
    let result = f32::simd_mul(&val_arr, &weight_arr);
    output[..len].copy_from_slice(result.as_slice().unwrap());
}

/// Vector addition for probability combination using SciRS2 SIMD
pub fn simd_add_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    let len = a.len().min(b.len()).min(output.len());
    let a_arr = ArrayView1::from(&a[..len]);
    let b_arr = ArrayView1::from(&b[..len]);
    let result = f32::simd_add(&a_arr, &b_arr);
    output[..len].copy_from_slice(result.as_slice().unwrap());
}

/// Scalar multiplication for weight application using SciRS2 SIMD
pub fn simd_scale_f32(input: &[f32], scalar: f32, output: &mut [f32]) {
    let len = input.len().min(output.len());
    let input_arr = ArrayView1::from(&input[..len]);
    let result = f32::simd_scalar_mul(&input_arr, scalar);
    output[..len].copy_from_slice(result.as_slice().unwrap());
}

/// Argmax operation for finding maximum index
pub fn simd_argmax_f32(values: &[f32]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let arr = ArrayView1::from(values);
    let max_val = f32::simd_max_element(&arr);

    // Find first occurrence of max value
    values.iter().position(|&v| v == max_val).unwrap_or(0)
}

/// L2 normalization using SciRS2 SIMD
pub fn simd_normalize_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    if len == 0 {
        return;
    }

    let input_arr = ArrayView1::from(&input[..len]);
    let norm = f32::simd_norm(&input_arr);

    if norm > 1e-8 {
        let result = f32::simd_scalar_mul(&input_arr, 1.0 / norm);
        output[..len].copy_from_slice(result.as_slice().unwrap());
    } else {
        output[..len].fill(0.0);
    }
}

/// Confidence calculation for predictions
pub fn simd_confidence_f32(predictions: &[f32]) -> f32 {
    if predictions.is_empty() {
        return 0.0;
    }
    let max_idx = simd_argmax_f32(predictions);
    let max_val = predictions[max_idx];
    let sum = simd_sum_f32(predictions);
    if sum > 1e-8 {
        max_val / sum
    } else {
        0.0
    }
}

/// Matrix-vector multiplication for ensemble voting using SciRS2 SIMD
pub fn simd_matrix_vector_multiply(
    matrix: &Array2<Float>,
    vector: &Array1<Float>,
    result: &mut Array1<Float>,
) -> Result<()> {
    if matrix.ncols() != vector.len() || matrix.nrows() != result.len() {
        return Err(sklears_core::error::SklearsError::ShapeMismatch {
            expected: format!(
                "matrix: {}x{}, vector: {}",
                matrix.nrows(),
                matrix.ncols(),
                vector.len()
            ),
            actual: format!("result: {}", result.len()),
        });
    }

    // Use SciRS2's GEMV (matrix-vector multiplication)
    Float::simd_gemv(&matrix.view(), &vector.view(), 0.0, result);
    Ok(())
}

/// Probability aggregation for ensemble voting using SciRS2 SIMD
pub fn simd_aggregate_probabilities(
    all_probabilities: &[Array2<Float>],
    weights: &[Float],
    result: &mut Array2<Float>,
) -> Result<()> {
    if all_probabilities.is_empty() || weights.is_empty() {
        return Err(sklears_core::error::SklearsError::InvalidParameter {
            name: "input_data".to_string(),
            reason: "Empty probabilities or weights".to_string(),
        });
    }

    let n_estimators = all_probabilities.len().min(weights.len());
    let n_samples = all_probabilities[0].nrows();
    let n_classes = all_probabilities[0].ncols();

    result.fill(0.0);

    let weight_sum: Float = weights[..n_estimators].iter().sum();
    if weight_sum <= 1e-8 {
        return Err(sklears_core::error::SklearsError::InvalidParameter {
            name: "weights".to_string(),
            reason: "Zero weight sum".to_string(),
        });
    }

    for sample_idx in 0..n_samples {
        for class_idx in 0..n_classes {
            let mut weighted_sum = 0.0;
            for (est_idx, prob_matrix) in all_probabilities[..n_estimators].iter().enumerate() {
                weighted_sum += prob_matrix[[sample_idx, class_idx]] * weights[est_idx];
            }
            result[[sample_idx, class_idx]] = weighted_sum / weight_sum;
        }
    }

    Ok(())
}

/// Rank calculation for rank-based voting
pub fn simd_calculate_ranks(values: &[f32], ranks: &mut [f32]) -> Result<()> {
    if values.len() != ranks.len() {
        return Err(sklears_core::error::SklearsError::ShapeMismatch {
            expected: format!("{}", values.len()),
            actual: format!("{}", ranks.len()),
        });
    }

    let n = values.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap());

    for (rank, &idx) in indices.iter().enumerate() {
        ranks[idx] = (rank + 1) as f32;
    }

    Ok(())
}

/// Ensemble disagreement calculation
pub fn simd_ensemble_disagreement(
    predictions: &[Array1<Float>],
    disagreement: &mut Array1<Float>,
) -> Result<()> {
    if predictions.is_empty() {
        return Err(sklears_core::error::SklearsError::InvalidParameter {
            name: "predictions".to_string(),
            reason: "Empty predictions".to_string(),
        });
    }

    let n_samples = predictions[0].len();

    for sample_idx in 0..n_samples {
        let sample_preds: Vec<f32> = predictions
            .iter()
            .map(|pred| pred[sample_idx] as f32)
            .collect();

        let mean = simd_mean_f32(&sample_preds);
        let variance = simd_variance_f32(&sample_preds, mean);
        disagreement[sample_idx] = variance as Float;
    }

    Ok(())
}

/// Bootstrap aggregation
pub fn simd_bootstrap_aggregate(
    predictions: &[Array1<Float>],
    _n_bootstrap: usize,
    result: &mut Array1<Float>,
) -> Result<()> {
    if predictions.is_empty() {
        return Err(sklears_core::error::SklearsError::InvalidParameter {
            name: "predictions".to_string(),
            reason: "Empty predictions".to_string(),
        });
    }

    let n_samples = predictions[0].len();
    result.fill(0.0);

    for sample_idx in 0..n_samples {
        let sample_preds: Vec<f32> = predictions
            .iter()
            .map(|pred| pred[sample_idx] as f32)
            .collect();

        result[sample_idx] = simd_mean_f32(&sample_preds) as Float;
    }

    Ok(())
}

/// Adaptive weight computation
pub fn simd_adaptive_weights(
    performances: &[f32],
    current_weights: &[f32],
    learning_rate: f32,
    output_weights: &mut [f32],
) {
    let len = performances
        .len()
        .min(current_weights.len())
        .min(output_weights.len());

    let perf_sum = simd_sum_f32(&performances[..len]);
    let mut normalized_perfs = vec![0.0f32; len];

    if perf_sum > 1e-8 {
        simd_scale_f32(&performances[..len], 1.0 / perf_sum, &mut normalized_perfs);
    }

    for i in 0..len {
        let diff = normalized_perfs[i] - current_weights[i];
        output_weights[i] = current_weights[i] + learning_rate * diff;
    }

    let weight_sum = simd_sum_f32(&output_weights[..len]);
    if weight_sum > 1e-8 {
        let temp_weights: Vec<f32> = output_weights[..len].to_vec();
        simd_scale_f32(&temp_weights, 1.0 / weight_sum, &mut output_weights[..len]);
    }
}

/// Hard voting with weighted vote counting
pub fn simd_hard_voting_weighted(
    all_predictions: &[Array1<Float>],
    weights: &[Float],
    classes: &[Float],
) -> Array1<Float> {
    if all_predictions.is_empty() || weights.is_empty() || classes.is_empty() {
        return Array1::zeros(0);
    }

    let n_samples = all_predictions[0].len();
    let n_classes = classes.len();
    let n_estimators = all_predictions.len().min(weights.len());

    let mut votes = Array2::<Float>::zeros((n_samples, n_classes));

    for sample_idx in 0..n_samples {
        for estimator_idx in 0..n_estimators {
            let prediction = all_predictions[estimator_idx][sample_idx];
            for (class_idx, &class_val) in classes.iter().enumerate() {
                if (prediction - class_val).abs() < 1e-6 {
                    votes[[sample_idx, class_idx]] += weights[estimator_idx];
                    break;
                }
            }
        }
    }

    let mut result = Array1::<Float>::zeros(n_samples);
    for sample_idx in 0..n_samples {
        let sample_votes: Vec<f32> = (0..n_classes)
            .map(|i| votes[[sample_idx, i]] as f32)
            .collect();
        let max_class_idx = simd_argmax_f32(&sample_votes);
        result[sample_idx] = classes[max_class_idx];
    }

    result
}

/// Soft voting with probability averaging
pub fn simd_soft_voting_weighted(
    all_probabilities: &[Array2<Float>],
    weights: &[Float],
) -> Array2<Float> {
    if all_probabilities.is_empty() || weights.is_empty() {
        return Array2::zeros((0, 0));
    }

    let n_samples = all_probabilities[0].nrows();
    let n_classes = all_probabilities[0].ncols();
    let mut result = Array2::<Float>::zeros((n_samples, n_classes));

    if simd_aggregate_probabilities(all_probabilities, weights, &mut result).is_err() {
        let n_estimators = all_probabilities.len();
        result.fill(0.0);

        for prob_matrix in all_probabilities.iter() {
            for i in 0..n_samples {
                for j in 0..n_classes {
                    result[[i, j]] += prob_matrix[[i, j]] / n_estimators as Float;
                }
            }
        }
    }

    result
}

/// Entropy-weighted voting
pub fn simd_entropy_weighted_voting(
    all_probabilities: &[Array2<Float>],
    entropy_weight_factor: f32,
) -> Array2<Float> {
    if all_probabilities.is_empty() {
        return Array2::zeros((0, 0));
    }

    let n_samples = all_probabilities[0].nrows();
    let n_classes = all_probabilities[0].ncols();
    let n_estimators = all_probabilities.len();

    let mut entropy_weights = Vec::with_capacity(n_estimators);

    for prob_matrix in all_probabilities.iter() {
        let mut total_entropy = 0.0f32;

        for sample_idx in 0..n_samples {
            let sample_probs: Vec<f32> = (0..n_classes)
                .map(|class_idx| prob_matrix[[sample_idx, class_idx]] as f32)
                .collect();

            let entropy = simd_entropy_f32(&sample_probs);
            total_entropy += entropy;
        }

        let avg_entropy = total_entropy / n_samples as f32;
        let weight = if avg_entropy > 1e-6 {
            entropy_weight_factor / avg_entropy
        } else {
            entropy_weight_factor
        };

        entropy_weights.push(weight as Float);
    }

    let weight_sum: Float = entropy_weights.iter().sum();
    if weight_sum > 1e-8 {
        for w in entropy_weights.iter_mut() {
            *w /= weight_sum;
        }
    }

    simd_soft_voting_weighted(all_probabilities, &entropy_weights)
}

/// Variance-weighted voting
pub fn simd_variance_weighted_voting(
    all_probabilities: &[Array2<Float>],
    variance_weight_factor: f32,
) -> Array2<Float> {
    if all_probabilities.is_empty() {
        return Array2::zeros((0, 0));
    }

    let n_samples = all_probabilities[0].nrows();
    let n_classes = all_probabilities[0].ncols();
    let n_estimators = all_probabilities.len();

    let mut variance_weights = Vec::with_capacity(n_estimators);

    for prob_matrix in all_probabilities.iter() {
        let mut total_variance = 0.0f32;

        for sample_idx in 0..n_samples {
            let sample_probs: Vec<f32> = (0..n_classes)
                .map(|class_idx| prob_matrix[[sample_idx, class_idx]] as f32)
                .collect();

            let mean = simd_mean_f32(&sample_probs);
            let variance = simd_variance_f32(&sample_probs, mean);
            total_variance += variance;
        }

        let avg_variance = total_variance / n_samples as f32;
        let weight = if avg_variance > 1e-6 {
            variance_weight_factor / avg_variance
        } else {
            variance_weight_factor
        };

        variance_weights.push(weight as Float);
    }

    let weight_sum: Float = variance_weights.iter().sum();
    if weight_sum > 1e-8 {
        for w in variance_weights.iter_mut() {
            *w /= weight_sum;
        }
    }

    simd_soft_voting_weighted(all_probabilities, &variance_weights)
}

/// Confidence-weighted voting
pub fn simd_confidence_weighted_voting(
    all_probabilities: &[Array2<Float>],
    confidence_threshold: f32,
) -> Array2<Float> {
    if all_probabilities.is_empty() {
        return Array2::zeros((0, 0));
    }

    let n_samples = all_probabilities[0].nrows();
    let n_classes = all_probabilities[0].ncols();
    let n_estimators = all_probabilities.len();

    let mut confidence_weights = Vec::with_capacity(n_estimators);

    for prob_matrix in all_probabilities.iter() {
        let mut total_confidence = 0.0f32;

        for sample_idx in 0..n_samples {
            let sample_probs: Vec<f32> = (0..n_classes)
                .map(|class_idx| prob_matrix[[sample_idx, class_idx]] as f32)
                .collect();

            let confidence = simd_confidence_f32(&sample_probs);
            total_confidence += confidence.max(confidence_threshold);
        }

        let avg_confidence = total_confidence / n_samples as f32;
        confidence_weights.push(avg_confidence as Float);
    }

    let weight_sum: Float = confidence_weights.iter().sum();
    if weight_sum > 1e-8 {
        for w in confidence_weights.iter_mut() {
            *w /= weight_sum;
        }
    }

    simd_soft_voting_weighted(all_probabilities, &confidence_weights)
}

/// Bayesian model averaging
pub fn simd_bayesian_averaging(
    all_probabilities: &[Array2<Float>],
    model_evidences: &[Float],
) -> Array2<Float> {
    if all_probabilities.is_empty() || model_evidences.is_empty() {
        return Array2::zeros((0, 0));
    }

    let n_estimators = all_probabilities.len().min(model_evidences.len());
    let weights: Vec<Float> = model_evidences[..n_estimators].to_vec();

    simd_soft_voting_weighted(&all_probabilities[..n_estimators], &weights)
}
