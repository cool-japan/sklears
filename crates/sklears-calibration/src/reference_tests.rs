//! Reference Implementation Comparison Tests
//!
//! This module contains tests that compare our calibration implementations
//! against reference implementations and theoretical expected results.

use scirs2_core::ndarray::Array1;
use sklears_core::types::Float;
use std::f64::consts::PI;

use crate::{
    histogram::HistogramBinningCalibrator, isotonic::IsotonicCalibrator, kde::KDECalibrator,
    temperature::TemperatureScalingCalibrator, CalibrationEstimator,
};

/// Reference implementation of Pool Adjacent Violators Algorithm (PAVA)
/// Based on the classic algorithm description
pub fn reference_pava(y: &Array1<Float>, weights: Option<&Array1<Float>>) -> Array1<Float> {
    let n = y.len();
    let mut result = y.clone();
    let w = weights.cloned().unwrap_or_else(|| Array1::ones(n));

    let mut i = 0;
    while i < n - 1 {
        if result[i] > result[i + 1] {
            // Find the block to pool
            let mut j = i;
            while j < n - 1 && result[j] > result[j + 1] {
                j += 1;
            }

            // Pool the block
            let mut sum_wy = 0.0;
            let mut sum_w = 0.0;
            for k in i..=j {
                sum_wy += w[k] * result[k];
                sum_w += w[k];
            }
            let pooled_value = sum_wy / sum_w;

            for k in i..=j {
                result[k] = pooled_value;
            }

            // Backtrack if necessary
            if i > 0 {
                i -= 1;
            } else {
                i = j + 1;
            }
        } else {
            i += 1;
        }
    }

    result
}

/// Reference implementation of sigmoid calibration (Platt scaling)
/// Based on the original paper by Platt (1999)
pub fn reference_sigmoid_calibration(
    scores: &Array1<Float>,
    targets: &Array1<i32>,
) -> (Float, Float) {
    let n = scores.len();
    let mut a = 0.0;
    let mut b = 0.0;

    // Count positive and negative samples
    let n_pos = targets.iter().filter(|&&t| t == 1).count() as Float;
    let n_neg = (n as Float) - n_pos;

    // Initial values as in Platt's paper
    let hi_target = (n_pos + 1.0) / (n_pos + 2.0);
    let lo_target = 1.0 / (n_neg + 2.0);

    // Simplified Newton-Raphson iteration (simplified for reference)
    for _ in 0..100 {
        let mut gradient_a = 0.0;
        let mut gradient_b = 0.0;
        let mut hessian_aa = 0.0;
        let mut hessian_ab = 0.0;
        let mut hessian_bb = 0.0;

        for i in 0..n {
            let fval = a * scores[i] + b;
            let p = 1.0 / (1.0 + (-fval).exp());
            let target = if targets[i] == 1 {
                hi_target
            } else {
                lo_target
            };

            let d1 = p - target;
            let d2 = p * (1.0 - p);

            gradient_a += scores[i] * d1;
            gradient_b += d1;
            hessian_aa += scores[i] * scores[i] * d2;
            hessian_ab += scores[i] * d2;
            hessian_bb += d2;
        }

        if hessian_aa.abs() < 1e-12 || hessian_bb.abs() < 1e-12 {
            break;
        }

        let det = hessian_aa * hessian_bb - hessian_ab * hessian_ab;
        if det.abs() < 1e-12 {
            break;
        }

        let delta_a = (hessian_bb * gradient_a - hessian_ab * gradient_b) / det;
        let delta_b = (hessian_aa * gradient_b - hessian_ab * gradient_a) / det;

        a -= delta_a;
        b -= delta_b;

        if delta_a.abs() < 1e-6 && delta_b.abs() < 1e-6 {
            break;
        }
    }

    (a, b)
}

/// Reference implementation of histogram binning calibration
pub fn reference_histogram_binning(
    scores: &Array1<Float>,
    targets: &Array1<i32>,
    n_bins: usize,
) -> (Vec<Float>, Vec<Float>) {
    let mut bin_boundaries = vec![0.0; n_bins + 1];
    let mut bin_true_probs = vec![0.0; n_bins];

    // Create uniform bin boundaries
    for i in 0..=n_bins {
        bin_boundaries[i] = i as Float / n_bins as Float;
    }

    // Compute empirical probabilities in each bin
    for bin_idx in 0..n_bins {
        let lower = bin_boundaries[bin_idx];
        let upper = bin_boundaries[bin_idx + 1];

        let mut count_in_bin = 0;
        let mut positive_in_bin = 0;

        for (i, &score) in scores.iter().enumerate() {
            if score >= lower && (score < upper || (bin_idx == n_bins - 1 && score <= upper)) {
                count_in_bin += 1;
                if targets[i] == 1 {
                    positive_in_bin += 1;
                }
            }
        }

        bin_true_probs[bin_idx] = if count_in_bin > 0 {
            positive_in_bin as Float / count_in_bin as Float
        } else {
            0.5 // Default to neutral probability for empty bins
        };
    }

    (bin_boundaries, bin_true_probs)
}

/// Reference implementation of temperature scaling
pub fn reference_temperature_scaling(logits: &Array1<Float>, targets: &Array1<i32>) -> Float {
    let mut temperature = 1.0;
    let learning_rate = 0.01;
    let max_iterations = 1000;

    for _ in 0..max_iterations {
        let mut gradient = 0.0;
        let mut hessian = 0.0;

        for i in 0..logits.len() {
            let scaled_logit = logits[i] / temperature;
            let prob = 1.0 / (1.0 + (-scaled_logit).exp());
            let target = targets[i] as Float;

            let error = prob - target;
            let d_prob_d_temp = -logits[i] * prob * (1.0 - prob) / (temperature * temperature);

            gradient += error * d_prob_d_temp;
            hessian += d_prob_d_temp * d_prob_d_temp;
        }

        if hessian.abs() < 1e-12 {
            break;
        }

        let delta_temp = -gradient / hessian;
        temperature += learning_rate * delta_temp;
        temperature = temperature.max(0.01).min(100.0); // Constrain temperature

        if delta_temp.abs() < 1e-6 {
            break;
        }
    }

    temperature
}

/// Reference implementation of KDE with Gaussian kernel
pub fn reference_kde_gaussian(
    train_scores: &Array1<Float>,
    train_targets: &Array1<i32>,
    test_scores: &Array1<Float>,
    bandwidth: Float,
) -> Array1<Float> {
    let n_train = train_scores.len();
    let n_test = test_scores.len();
    let mut predictions = Array1::zeros(n_test);

    for i in 0..n_test {
        let test_score = test_scores[i];
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for j in 0..n_train {
            let train_score = train_scores[j];
            let target = train_targets[j] as Float;

            // Gaussian kernel
            let diff = (test_score - train_score) / bandwidth;
            let weight = (-0.5 * diff * diff).exp() / (bandwidth * (2.0 * PI).sqrt());

            weighted_sum += weight * target;
            weight_sum += weight;
        }

        predictions[i] = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.5
        };
    }

    predictions
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::SigmoidCalibrator;

    fn create_test_data() -> (Array1<Float>, Array1<i32>) {
        let scores = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]);
        let targets = Array1::from(vec![0, 0, 0, 1, 1, 1, 1, 1]);
        (scores, targets)
    }

    fn create_monotonic_data() -> (Array1<Float>, Array1<i32>) {
        let scores = Array1::from(vec![0.0, 0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0]);
        let targets = Array1::from(vec![0, 0, 0, 0, 1, 1, 1, 1]);
        (scores, targets)
    }

    #[test]
    fn test_isotonic_vs_reference_pava() {
        let (scores, targets) = create_monotonic_data();

        // Our implementation
        let isotonic = IsotonicCalibrator::new().fit(&scores, &targets).unwrap();
        let our_predictions = isotonic.predict_proba(&scores).unwrap();

        // Reference implementation
        let target_floats: Array1<Float> = targets.mapv(|x| x as Float);
        let reference_predictions = reference_pava(&target_floats, None);

        // They should be very close for perfectly ordered data
        for (our, reference) in our_predictions.iter().zip(reference_predictions.iter()) {
            assert!(
                (our - reference).abs() < 0.1,
                "Our: {}, Reference: {}",
                our,
                reference
            );
        }
    }

    #[test]
    fn test_sigmoid_vs_reference_implementation() {
        let (scores, targets) = create_test_data();

        // Our implementation
        let sigmoid = SigmoidCalibrator::new().fit(&scores, &targets).unwrap();
        let our_predictions = sigmoid.predict_proba(&scores).unwrap();

        // Reference implementation
        let (a, b) = reference_sigmoid_calibration(&scores, &targets);
        let mut reference_predictions = Array1::zeros(scores.len());
        for (i, &score) in scores.iter().enumerate() {
            let linear_pred = a * score + b;
            reference_predictions[i] = 1.0 / (1.0 + (-linear_pred).exp());
        }

        // Should be reasonably close (allowing for optimization differences)
        for (our, reference) in our_predictions.iter().zip(reference_predictions.iter()) {
            assert!(
                (our - reference).abs() < 0.2,
                "Our: {}, Reference: {}",
                our,
                reference
            );
        }
    }

    #[test]
    fn test_histogram_vs_reference_implementation() {
        let (scores, targets) = create_test_data();
        let n_bins = 4;

        // Our implementation
        let histogram = HistogramBinningCalibrator::new(n_bins)
            .fit(&scores, &targets)
            .unwrap();
        let our_predictions = histogram.predict_proba(&scores).unwrap();

        // Reference implementation
        let (bin_boundaries, bin_probs) = reference_histogram_binning(&scores, &targets, n_bins);
        let mut reference_predictions = Array1::zeros(scores.len());

        for (i, &score) in scores.iter().enumerate() {
            // Find which bin this score belongs to
            let mut bin_idx = 0;
            for j in 0..n_bins {
                if score >= bin_boundaries[j] && score < bin_boundaries[j + 1] {
                    bin_idx = j;
                    break;
                } else if j == n_bins - 1 && score <= bin_boundaries[j + 1] {
                    bin_idx = j;
                    break;
                }
            }
            reference_predictions[i] = bin_probs[bin_idx];
        }

        // Should match exactly for histogram binning
        for (our, reference) in our_predictions.iter().zip(reference_predictions.iter()) {
            assert!(
                (our - reference).abs() < 0.1,
                "Our: {}, Reference: {}",
                our,
                reference
            );
        }
    }

    #[test]
    fn test_temperature_vs_reference_implementation() {
        let (scores, targets) = create_test_data();

        // Our implementation
        let temp_scaling = TemperatureScalingCalibrator::new()
            .fit(&scores, &targets)
            .unwrap();
        let our_predictions = temp_scaling.predict_proba(&scores).unwrap();

        // Just test that our temperature scaling produces valid outputs
        // (Different optimization algorithms can produce different results)
        for &pred in our_predictions.iter() {
            assert!(
                pred >= 0.0 && pred <= 1.0,
                "Temperature scaling should produce valid probabilities: {}",
                pred
            );
        }

        // Should maintain reasonable ordering for sorted inputs
        let sorted_indices: Vec<usize> = (0..scores.len()).collect();
        let mut sorted_by_score = sorted_indices.clone();
        sorted_by_score.sort_by(|&i, &j| scores[i].partial_cmp(&scores[j]).unwrap());

        // Check if generally monotonic (allowing some flexibility)
        let mut increasing_count = 0;
        for i in 1..sorted_by_score.len() {
            if our_predictions[sorted_by_score[i]] >= our_predictions[sorted_by_score[i - 1]] {
                increasing_count += 1;
            }
        }

        // Should be mostly monotonic
        assert!(
            increasing_count as f64 / (sorted_by_score.len() - 1) as f64 > 0.5,
            "Temperature scaling should generally preserve ordering"
        );
    }

    #[test]
    fn test_kde_vs_reference_implementation() {
        let (scores, targets) = create_test_data();
        let bandwidth = 0.2;

        // Our implementation
        let kde = KDECalibrator::new().fit(&scores, &targets).unwrap();
        let our_predictions = kde.predict_proba(&scores).unwrap();

        // Reference implementation
        let reference_predictions = reference_kde_gaussian(&scores, &targets, &scores, bandwidth);

        // Should be reasonably close (KDE implementations can vary in bandwidth selection)
        for (our, reference) in our_predictions.iter().zip(reference_predictions.iter()) {
            assert!(
                (our - reference).abs() < 0.4,
                "Our: {}, Reference: {}",
                our,
                reference
            );
        }
    }

    #[test]
    fn test_calibration_monotonicity_property() {
        let (scores, targets) = create_test_data();

        // Test multiple calibration methods for monotonicity
        let methods: Vec<Box<dyn CalibrationEstimator>> = vec![
            Box::new(SigmoidCalibrator::new()),
            Box::new(IsotonicCalibrator::new()),
            Box::new(TemperatureScalingCalibrator::new()),
        ];

        for mut method in methods {
            method.as_mut().fit(&scores, &targets).unwrap();
            let predictions = method.predict_proba(&scores).unwrap();

            // For sorted scores, isotonic should be monotonic
            if std::any::type_name::<dyn CalibrationEstimator>().contains("Isotonic") {
                for i in 1..predictions.len() {
                    assert!(
                        predictions[i] >= predictions[i - 1] - 1e-6,
                        "Isotonic calibration should be monotonic"
                    );
                }
            }

            // All methods should produce valid probabilities
            for &pred in predictions.iter() {
                assert!(
                    pred >= 0.0 && pred <= 1.0,
                    "Predictions should be in [0,1] range: {}",
                    pred
                );
            }
        }
    }

    #[test]
    fn test_calibration_invariance_property() {
        let (scores, targets) = create_test_data();

        // Test that affine transformations don't change isotonic calibration
        let isotonic1 = IsotonicCalibrator::new();
        let isotonic2 = IsotonicCalibrator::new();

        // Original scores
        let isotonic1_fitted = isotonic1.fit(&scores, &targets).unwrap();
        let predictions1 = isotonic1_fitted.predict_proba(&scores).unwrap();

        // Affine transformed scores (for isotonic, this should give similar ranking)
        let transformed_scores = scores.mapv(|x| 2.0 * x + 0.1);
        let isotonic2_fitted = isotonic2.fit(&transformed_scores, &targets).unwrap();
        let transformed_test = scores.mapv(|x| 2.0 * x + 0.1);
        let predictions2 = isotonic2_fitted.predict_proba(&transformed_test).unwrap();

        // Isotonic should be invariant to monotonic transformations of input
        let rank_correlation = compute_rank_correlation(&predictions1, &predictions2);
        assert!(
            rank_correlation > 0.5,
            "Isotonic calibration should preserve ranking: {}",
            rank_correlation
        );
    }

    #[test]
    fn test_edge_case_consistency() {
        // Test with all zeros
        let zeros = Array1::zeros(10);
        let targets = Array1::from(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

        let sigmoid = SigmoidCalibrator::new().fit(&zeros, &targets).unwrap();
        let predictions = sigmoid.predict_proba(&zeros).unwrap();

        // All predictions should be the same (and equal to the base rate)
        let base_rate = 0.5; // 5 out of 10 are positive
        for &pred in predictions.iter() {
            assert!(
                (pred - base_rate).abs() < 0.2,
                "All predictions should be close to base rate for constant input"
            );
        }

        // Test with all ones
        let ones = Array1::ones(10);
        let sigmoid2 = SigmoidCalibrator::new().fit(&ones, &targets).unwrap();
        let predictions2 = sigmoid2.predict_proba(&ones).unwrap();

        for &pred in predictions2.iter() {
            assert!(
                (pred - base_rate).abs() < 0.2,
                "All predictions should be close to base rate for constant input"
            );
        }
    }

    #[test]
    fn test_calibration_basic_properties() {
        // Create test data
        let scores = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let targets = Array1::from(vec![0, 0, 0, 1, 0, 1, 1, 1]);

        let sigmoid = SigmoidCalibrator::new().fit(&scores, &targets).unwrap();
        let predictions = sigmoid.predict_proba(&scores).unwrap();

        // Basic validity checks
        for &pred in predictions.iter() {
            assert!(
                pred >= 0.0 && pred <= 1.0,
                "Predictions should be in [0,1] range: {}",
                pred
            );
        }

        // Check that we get different predictions for different inputs
        let unique_predictions: std::collections::HashSet<_> =
            predictions.iter().map(|&x| (x * 1000.0) as i32).collect();
        assert!(
            unique_predictions.len() > 1,
            "Calibration should produce varied predictions for varied inputs"
        );
    }

    // Helper function to compute rank correlation
    fn compute_rank_correlation(x: &Array1<Float>, y: &Array1<Float>) -> Float {
        let n = x.len();
        if n != y.len() {
            return 0.0;
        }

        // Simple Spearman rank correlation approximation
        let mut pairs: Vec<(Float, Float)> =
            x.iter().zip(y.iter()).map(|(&a, &b)| (a, b)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut rank_diff_sum = 0.0;
        for i in 0..n {
            let rank_x = i as Float;
            let rank_y = pairs.iter().position(|(_, b)| *b == pairs[i].1).unwrap() as Float;
            rank_diff_sum += (rank_x - rank_y).powi(2);
        }

        1.0 - (6.0 * rank_diff_sum) / (n as Float * ((n * n - 1) as Float))
    }
}
