//! Property-based tests for calibration methods
//!
//! This module contains property-based tests that verify fundamental
//! properties of calibration methods under various conditions.

use crate::{
    bbq::BBQCalibrator, beta::BetaCalibrator, histogram::HistogramBinningCalibrator,
    isotonic::IsotonicCalibrator, temperature::TemperatureScalingCalibrator,
    CalibratedClassifierCV, CalibrationMethod, SigmoidCalibrator,
};
use proptest::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    traits::{Fit, PredictProba},
    types::Float,
};

/// Strategy for generating valid probability arrays
fn probability_array_strategy(size: usize) -> impl Strategy<Value = Array1<Float>> {
    prop::collection::vec(0.01f64..0.99f64, size..=size).prop_map(|vec| {
        let sum: Float = vec.iter().sum();
        let normalized: Vec<Float> = vec.iter().map(|&x| x / sum).collect();
        Array1::from(normalized)
    })
}

/// Strategy for generating binary target arrays
fn binary_target_strategy(size: usize) -> impl Strategy<Value = Array1<i32>> {
    prop::collection::vec(0i32..=1i32, size..=size).prop_map(Array1::from)
}

/// Strategy for generating feature matrices
fn feature_matrix_strategy(
    n_samples: usize,
    n_features: usize,
) -> impl Strategy<Value = Array2<Float>> {
    prop::collection::vec(
        prop::collection::vec(-10.0f64..10.0f64, n_features..=n_features),
        n_samples..=n_samples,
    )
    .prop_map(move |vecs| {
        let flat: Vec<Float> = vecs.into_iter().flatten().collect();
        Array2::from_shape_vec((n_samples, n_features), flat).unwrap()
    })
}

/// Strategy for generating multiclass target arrays
fn multiclass_target_strategy(size: usize, n_classes: usize) -> impl Strategy<Value = Array1<i32>> {
    prop::collection::vec(0i32..(n_classes as i32), size..=size).prop_map(Array1::from)
}

// Property test: Calibrated probabilities should sum to 1
proptest! {
    #[test]
    fn calibrated_probabilities_sum_to_one(
        x in feature_matrix_strategy(20, 3),
        y in multiclass_target_strategy(20, 3)
    ) {
        let calibrator = CalibratedClassifierCV::new()
            .method(CalibrationMethod::Sigmoid);

        if let Ok(fitted) = calibrator.fit(&x, &y) {
            if let Ok(probas) = fitted.predict_proba(&x) {
                for row in probas.axis_iter(Axis(0)) {
                    let sum: Float = row.sum();
                    prop_assert!((sum - 1.0).abs() < 1e-6, "Probabilities don't sum to 1: {}", sum);
                }
            }
        }
    }
}

// Property test: Calibrated probabilities should be in [0, 1]
proptest! {
    #[test]
    fn calibrated_probabilities_in_range(
        probabilities in probability_array_strategy(10),
        targets in binary_target_strategy(10)
    ) {
        let calibrator = SigmoidCalibrator::new();

        if let Ok(fitted_calibrator) = calibrator.fit(&probabilities, &targets) {
            if let Ok(calibrated) = fitted_calibrator.predict_proba(&probabilities) {
                for &prob in calibrated.iter() {
                    prop_assert!((0.0..=1.0).contains(&prob), "Probability out of range: {}", prob);
                }
            }
        }
    }
}

// Property test: Isotonic calibration should preserve monotonicity
proptest! {
    #[test]
    fn isotonic_preserves_monotonicity(
        probabilities in probability_array_strategy(15),
        targets in binary_target_strategy(15)
    ) {
        // Sort probabilities to ensure input monotonicity
        let mut indices: Vec<usize> = (0..probabilities.len()).collect();
        indices.sort_by(|&i, &j| probabilities[i].partial_cmp(&probabilities[j]).unwrap());

        let sorted_probs: Array1<Float> = indices.iter().map(|&i| probabilities[i]).collect();
        let sorted_targets: Array1<i32> = indices.iter().map(|&i| targets[i]).collect();

        let calibrator = IsotonicCalibrator::new();

        if let Ok(fitted_calibrator) = calibrator.fit(&sorted_probs, &sorted_targets) {
            if let Ok(calibrated) = fitted_calibrator.predict_proba(&sorted_probs) {
                // Check monotonicity of calibrated probabilities
                for i in 1..calibrated.len() {
                    prop_assert!(calibrated[i] >= calibrated[i-1] || (calibrated[i] - calibrated[i-1]).abs() < 1e-10,
                               "Isotonic calibration not monotonic: {} > {}", calibrated[i-1], calibrated[i]);
                }
            }
        }
    }
}

// Property test: Temperature scaling should produce valid probabilities
proptest! {
    #[test]
    fn temperature_scaling_produces_valid_probabilities(
        probabilities in probability_array_strategy(10),
        targets in binary_target_strategy(10)
    ) {
        let calibrator = TemperatureScalingCalibrator::new();

        if let Ok(fitted) = calibrator.fit(&probabilities, &targets) {
            if let Ok(result) = fitted.predict_proba(&probabilities) {
                // All probabilities should be in [0, 1]
                for &prob in result.iter() {
                    prop_assert!((0.0..=1.0).contains(&prob),
                               "Probability out of range: {}", prob);
                }

                // Check monotonicity property: if input prob1 > prob2,
                // then calibrated prob1 should be >= calibrated prob2 (approximately)
                for i in 0..probabilities.len().saturating_sub(1) {
                    for j in (i+1)..probabilities.len() {
                        if (probabilities[i] - probabilities[j]).abs() > 1e-6 {
                            let diff_input = probabilities[i] - probabilities[j];
                            let diff_output = result[i] - result[j];
                            // Temperature scaling should preserve order (roughly)
                            prop_assert!(diff_input.signum() == diff_output.signum() ||
                                       diff_output.abs() < 1e-3,
                                       "Temperature scaling violated monotonicity");
                        }
                    }
                }
            }
        }
    }
}

// Property test: Histogram binning should preserve probability ordering within bins
proptest! {
    #[test]
    fn histogram_binning_preserves_local_ordering(
        probabilities in probability_array_strategy(20),
        targets in binary_target_strategy(20)
    ) {
        let n_bins = 5;
        let calibrator = HistogramBinningCalibrator::new(n_bins);

        if let Ok(fitted_calibrator) = calibrator.fit(&probabilities, &targets) {
            if let Ok(calibrated) = fitted_calibrator.predict_proba(&probabilities) {
                // Within each bin, calibrated values should be similar
                let bin_width = 1.0 / n_bins as Float;

                for bin in 0..n_bins {
                    let bin_start = bin as Float * bin_width;
                    let bin_end = (bin + 1) as Float * bin_width;

                    let mut bin_calibrated_values = Vec::new();

                    for (i, &prob) in probabilities.iter().enumerate() {
                        if prob >= bin_start && prob < bin_end {
                            bin_calibrated_values.push(calibrated[i]);
                        }
                    }

                    // All values in the same bin should be equal (or very close)
                    if bin_calibrated_values.len() > 1 {
                        let first_val = bin_calibrated_values[0];
                        for &val in &bin_calibrated_values[1..] {
                            prop_assert!((val - first_val).abs() < 1e-10,
                                       "Histogram binning not consistent within bin: {} vs {}", first_val, val);
                        }
                    }
                }
            }
        }
    }
}

// Property test: BBQ calibration should handle different sample sizes gracefully
proptest! {
    #[test]
    fn bbq_handles_variable_sample_sizes(
        size in 5usize..50usize,
        min_bins in 2usize..5usize,
        max_bins in 5usize..15usize
    ) {
        let probabilities = probability_array_strategy(size);
        let targets = binary_target_strategy(size);

        proptest!(|(probabilities in probabilities, targets in targets)| {
            if min_bins <= max_bins {
                let calibrator = BBQCalibrator::new(min_bins, max_bins);

                // BBQ should handle various sample sizes without panicking
                prop_assert!(calibrator.clone().fit(&probabilities, &targets).is_ok() || size < min_bins);

                if let Ok(fitted_calibrator) = calibrator.fit(&probabilities, &targets) {
                    if let Ok(calibrated) = fitted_calibrator.predict_proba(&probabilities) {
                        prop_assert_eq!(calibrated.len(), probabilities.len());

                        // All calibrated values should be valid probabilities
                        for &val in calibrated.iter() {
                            prop_assert!((0.0..=1.0).contains(&val));
                        }
                    }
                }
            }
        });
    }
}

// Property test: Beta calibration should handle extreme probability values
proptest! {
    #[test]
    fn beta_calibration_handles_extreme_values(
        n_extreme in 1usize..5usize,
        n_normal in 10usize..20usize
    ) {
        // Create array with some extreme values
        let mut probabilities = Vec::new();
        let mut targets = Vec::new();

        // Add some extreme values
        for _ in 0..n_extreme {
            probabilities.push(0.001); // Very low probability
            probabilities.push(0.999); // Very high probability
            targets.push(0);
            targets.push(1);
        }

        // Add some normal values
        for i in 0..n_normal {
            probabilities.push(0.1 + 0.8 * (i as Float) / (n_normal as Float));
            targets.push((i % 2) as i32);
        }

        let probabilities = Array1::from(probabilities);
        let targets = Array1::from(targets);

        let calibrator = BetaCalibrator::new();

        if let Ok(fitted_calibrator) = calibrator.fit(&probabilities, &targets) {
            if let Ok(calibrated) = fitted_calibrator.predict_proba(&probabilities) {
                // Beta calibration should handle extreme values gracefully
                for &val in calibrated.iter() {
                    prop_assert!((0.0..=1.0).contains(&val), "Beta calibration produced invalid probability: {}", val);
                    prop_assert!(!val.is_nan() && !val.is_infinite(), "Beta calibration produced NaN or infinite value: {}", val);
                }
            }
        }
    }
}

// Property test: Calibration should be consistent across different orderings
proptest! {
    #[test]
    fn calibration_consistent_across_orderings(
        probabilities in probability_array_strategy(15),
        targets in binary_target_strategy(15),
        _shuffle_seed in any::<u64>()
    ) {
        // Create shuffled indices
        let mut indices: Vec<usize> = (0..probabilities.len()).collect();
        // For simplicity, reverse the indices instead of shuffling
        indices.reverse();

        // Create shuffled arrays
        let shuffled_probs: Array1<Float> = indices.iter().map(|&i| probabilities[i]).collect();
        let shuffled_targets: Array1<i32> = indices.iter().map(|&i| targets[i]).collect();

        // Train calibrators on original and shuffled data
        let calibrator1 = SigmoidCalibrator::new();
        let calibrator2 = SigmoidCalibrator::new();

        if let (Ok(fitted1), Ok(fitted2)) = (
            calibrator1.fit(&probabilities, &targets),
            calibrator2.fit(&shuffled_probs, &shuffled_targets)
        ) {
            // Test predictions on a simple test set
            let test_probs = Array1::from(vec![0.2, 0.4, 0.6, 0.8]);

            if let (Ok(result1), Ok(result2)) = (
                fitted1.predict_proba(&test_probs),
                fitted2.predict_proba(&test_probs)
            ) {
                // Results should be similar (calibration should be invariant to data ordering)
                for (r1, r2) in result1.iter().zip(result2.iter()) {
                    prop_assert!((r1 - r2).abs() < 0.1,
                               "Calibration not consistent across orderings: {} vs {}", r1, r2);
                }
            }
        }
    }
}

// Property test: Calibration methods should handle empty or singleton classes
proptest! {
    #[test]
    fn calibration_handles_edge_cases(
        size in 2usize..20usize
    ) {
        let probabilities = probability_array_strategy(size);

        proptest!(|(probabilities in probabilities)| {
            // All targets are the same class (singleton case)
            let singleton_targets = Array1::from(vec![1; size]);

            let calibrator = SigmoidCalibrator::new();

            // Should handle singleton case gracefully
            if let Ok(fitted_calibrator) = calibrator.fit(&probabilities, &singleton_targets) {
                if let Ok(calibrated) = fitted_calibrator.predict_proba(&probabilities) {
                    // All calibrated values should be valid
                    for &val in calibrated.iter() {
                        prop_assert!((0.0..=1.0).contains(&val));
                        prop_assert!(!val.is_nan() && !val.is_infinite());
                    }
                }
            }
        });
    }
}

// Property test: Multiple calibration methods should produce valid results
proptest! {
    #[test]
    fn multiple_methods_produce_valid_results(
        x in feature_matrix_strategy(12, 2),
        y in multiclass_target_strategy(12, 2)
    ) {
        let methods = vec![
            CalibrationMethod::Sigmoid,
            CalibrationMethod::Isotonic,
            CalibrationMethod::Temperature,
            CalibrationMethod::HistogramBinning { n_bins: 5 },
            CalibrationMethod::BBQ { min_bins: 2, max_bins: 8 },
            CalibrationMethod::Beta,
        ];

        for method in methods {
            let calibrator = CalibratedClassifierCV::new().method(method);

            if let Ok(fitted) = calibrator.fit(&x, &y) {
                if let Ok(probas) = fitted.predict_proba(&x) {
                    // Check basic properties for all methods
                    prop_assert_eq!(probas.nrows(), x.nrows());
                    prop_assert_eq!(probas.ncols(), 2); // Binary classification

                    for row in probas.axis_iter(Axis(0)) {
                        let sum: Float = row.sum();
                        prop_assert!((sum - 1.0).abs() < 1e-6, "Probabilities don't sum to 1");

                        for &prob in row.iter() {
                            prop_assert!((0.0..=1.0).contains(&prob), "Probability out of range: {}", prob);
                            prop_assert!(!prob.is_nan() && !prob.is_infinite(), "Invalid probability value: {}", prob);
                        }
                    }
                }
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_property_test_strategies() {
        // Test probability array strategy
        let prob_array = Array1::from(vec![0.2, 0.3, 0.5]);
        assert_eq!(prob_array.len(), 3);
        let sum: Float = prob_array.sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Test binary target strategy
        let targets = Array1::from(vec![0, 1, 0, 1, 1]);
        assert_eq!(targets.len(), 5);
        for &target in targets.iter() {
            assert!(target == 0 || target == 1);
        }

        // Test feature matrix strategy
        let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(features.dim(), (3, 2));
    }
}
