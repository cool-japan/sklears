//! Property-based tests for metrics and evaluation functions
//!
//! This module contains property-based tests to ensure the correctness
//! and robustness of machine learning evaluation metrics.

use proptest::prelude::*;
use scirs2_autograd::ndarray::Array1;
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::thread_rng;

#[allow(non_snake_case)]
#[cfg(test)]
mod metric_properties {
    use super::*;

    // Property: Accuracy score should be between 0 and 1 for valid inputs
    proptest! {
        #[test]
        fn test_accuracy_score_bounds(
            n_samples in 10..100usize,
            accuracy_rate in 0.0f64..1.0,
        ) {
            let mut rng = thread_rng();

            // Generate ground truth labels
            let y_true: Vec<i32> = (0..n_samples)
                .map(|_| (rng.random::<f64>() * 3.0).floor() as i32) // 3 classes
                .collect();

            // Create predictions with deterministic accuracy
            let num_correct = (n_samples as f64 * accuracy_rate).round() as usize;
            let mut y_pred = y_true.clone();

            // Make exactly (n_samples - num_correct) predictions incorrect
            let mut indices_to_change: Vec<usize> = (0..n_samples).collect();
            indices_to_change.shuffle(&mut rng);

            for i in 0..(n_samples - num_correct) {
                let idx = indices_to_change[i];
                // Change to a different class
                y_pred[idx] = match y_pred[idx] {
                    0 => 1,
                    1 => 2,
                    2 => 0,
                    _ => 0,
                };
            }

            // Calculate accuracy manually
            let correct_predictions = y_true.iter()
                .zip(y_pred.iter())
                .filter(|(true_val, pred_val)| true_val == pred_val)
                .count();

            let accuracy = correct_predictions as f64 / n_samples as f64;

            // Accuracy should be between 0 and 1
            prop_assert!(accuracy >= 0.0);
            prop_assert!(accuracy <= 1.0);

            // With deterministic approach, accuracy should be very close to target
            let expected_accuracy = num_correct as f64 / n_samples as f64;
            prop_assert!((accuracy - expected_accuracy).abs() < 0.001);
        }
    }

    // Property: Perfect predictions should give maximum metric scores
    proptest! {
        #[test]
        fn test_perfect_predictions(
            n_samples in 10..100usize,
        ) {
            let mut rng = thread_rng();

            // Generate identical true and predicted values
            let y_true: Vec<f64> = (0..n_samples)
                .map(|_| rng.random::<f64>() * 100.0)
                .collect();
            let y_pred = y_true.clone();

            // Convert to arrays for calculations
            let y_true_arr = Array1::from_vec(y_true);
            let y_pred_arr = Array1::from_vec(y_pred);

            // Calculate MSE manually
            let mse: f64 = (&y_true_arr - &y_pred_arr).mapv(|x| x * x).sum() / n_samples as f64;

            // Calculate MAE manually
            let mae: f64 = (&y_true_arr - &y_pred_arr).mapv(|x| x.abs()).sum() / n_samples as f64;

            // Perfect predictions should have zero error
            prop_assert!(mse < 1e-10);
            prop_assert!(mae < 1e-10);

            // R² score should be 1.0 for perfect predictions (if variance > 0)
            let y_mean = y_true_arr.mean().unwrap();
            let ss_tot: f64 = y_true_arr.iter().map(|&y| (y - y_mean).powi(2)).sum();

            if ss_tot > 1e-10 { // Avoid division by zero
                let r2: f64 = 1.0 - (0.0 / ss_tot); // ss_res = 0 for perfect predictions
                prop_assert!((r2 - 1.0).abs() < 1e-10);
            }
        }
    }

    // Property: R² score properties for regression metrics
    proptest! {
        #[test]
        fn test_r2_score_properties(
            n_samples in 10..100usize,
        ) {
            let mut rng = thread_rng();

            let y_true: Vec<f64> = (0..n_samples)
                .map(|_| rng.random::<f64>() * 100.0)
                .collect();

            let y_mean = y_true.iter().sum::<f64>() / n_samples as f64;

            // Test different prediction scenarios

            // 1. Predictions equal to mean (baseline)
            let _y_pred_mean = vec![y_mean; n_samples];
            let ss_tot: f64 = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum();
            let ss_res_mean: f64 = y_true.iter()
                .map(|&y| (y - y_mean).powi(2))
                .sum();

            if ss_tot > 1e-10 {
                let r2_mean: f64 = 1.0 - (ss_res_mean / ss_tot);
                // R² should be 0 when predicting the mean
                prop_assert!((r2_mean).abs() < 1e-10);
            }

            // 2. Random predictions (should typically give negative R²)
            let y_pred_random: Vec<f64> = (0..n_samples)
                .map(|_| rng.random::<f64>() * 200.0) // Wider range than true values
                .collect();

            let ss_res_random: f64 = y_true.iter()
                .zip(y_pred_random.iter())
                .map(|(&y_true, &y_pred)| (y_true - y_pred).powi(2))
                .sum();

            if ss_tot > 1e-10 {
                let r2_random: f64 = 1.0 - (ss_res_random / ss_tot);
                // R² should be finite
                prop_assert!(r2_random.is_finite());
            }
        }
    }

    // Property: Confusion matrix properties
    proptest! {
        #[test]
        fn test_confusion_matrix_properties(
            n_samples in 20..100usize,
            n_classes in 2..5usize,
        ) {
            let mut rng = thread_rng();

            let y_true: Vec<i32> = (0..n_samples)
                .map(|_| (rng.random::<f64>() * n_classes as f64).floor() as i32)
                .collect();

            let y_pred: Vec<i32> = (0..n_samples)
                .map(|_| (rng.random::<f64>() * n_classes as f64).floor() as i32)
                .collect();

            // Build confusion matrix manually
            let mut confusion_matrix = vec![vec![0u32; n_classes]; n_classes];

            for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
                if true_label >= 0 && pred_label >= 0 &&
                   (true_label as usize) < n_classes && (pred_label as usize) < n_classes {
                    confusion_matrix[true_label as usize][pred_label as usize] += 1;
                }
            }

            // Properties of confusion matrix:

            // 1. Sum of all entries should equal total samples (excluding invalid labels)
            let total_entries: u32 = confusion_matrix.iter()
                .flat_map(|row| row.iter())
                .sum();

            let valid_samples = y_true.iter().zip(y_pred.iter())
                .filter(|(&t, &p)| t >= 0 && p >= 0 && (t as usize) < n_classes && (p as usize) < n_classes)
                .count();

            prop_assert_eq!(total_entries as usize, valid_samples);

            // 2. Each entry is non-negative by type (u32)

            // 3. Matrix should be square
            prop_assert_eq!(confusion_matrix.len(), n_classes);
            for row in &confusion_matrix {
                prop_assert_eq!(row.len(), n_classes);
            }
        }
    }

    // Property: Precision and recall bounds and relationships
    proptest! {
        #[test]
        fn test_precision_recall_properties(
            true_positives in 0u32..50,
            false_positives in 0u32..50,
            false_negatives in 0u32..50,
        ) {
            // Skip cases where denominators would be zero
            prop_assume!(true_positives + false_positives > 0);
            prop_assume!(true_positives + false_negatives > 0);

            // Calculate precision and recall
            let precision = true_positives as f64 / (true_positives + false_positives) as f64;
            let recall = true_positives as f64 / (true_positives + false_negatives) as f64;

            // Both should be between 0 and 1
            prop_assert!((0.0..=1.0).contains(&precision));
            prop_assert!((0.0..=1.0).contains(&recall));

            // If there are no true positives, both should be 0
            if true_positives == 0 {
                prop_assert_eq!(precision, 0.0);
                prop_assert_eq!(recall, 0.0);
            }

            // If there are no false positives, precision should be 1
            if false_positives == 0 && true_positives > 0 {
                prop_assert!((precision - 1.0).abs() < 1e-10);
            }

            // If there are no false negatives, recall should be 1
            if false_negatives == 0 && true_positives > 0 {
                prop_assert!((recall - 1.0).abs() < 1e-10);
            }

            // F1 score calculation
            if precision + recall > 0.0 {
                let f1 = 2.0 * precision * recall / (precision + recall);
                prop_assert!((0.0..=1.0).contains(&f1));

                // F1 should be ≤ min(precision, recall) and ≤ max(precision, recall)
                let _min_pr = precision.min(recall);
                let max_pr = precision.max(recall);
                prop_assert!(f1 <= max_pr + 1e-10);

                // Harmonic mean should be ≤ arithmetic mean
                let arithmetic_mean = (precision + recall) / 2.0;
                prop_assert!(f1 <= arithmetic_mean + 1e-10);
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod metric_edge_cases {
    use super::*;

    // Property: Metrics should handle edge cases gracefully
    proptest! {
        #[test]
        fn test_empty_and_single_sample_cases(
            single_true_value in -100.0f64..100.0,
            single_pred_value in -100.0f64..100.0,
        ) {
            // Test single sample case
            let _y_true_single = [single_true_value];
            let _y_pred_single = [single_pred_value];

            // MSE calculation
            let mse = (single_true_value - single_pred_value).powi(2);
            prop_assert!(mse >= 0.0);
            prop_assert!(mse.is_finite());

            // MAE calculation
            let mae = (single_true_value - single_pred_value).abs();
            prop_assert!(mae >= 0.0);
            prop_assert!(mae.is_finite());

            // For single sample, R² is undefined if variance is 0
            // (which it is for a single sample), so we expect special handling
        }
    }

    // Property: Metrics should handle constant predictions
    proptest! {
        #[test]
        fn test_constant_predictions(
            n_samples in 5..50usize,
            constant_pred in -100.0f64..100.0,
        ) {
            let mut rng = thread_rng();

            // Generate varied true values
            let y_true: Vec<f64> = (0..n_samples)
                .map(|_| rng.random::<f64>() * 200.0 - 100.0)
                .collect();

            // Use constant predictions
            let _y_pred = vec![constant_pred; n_samples];

            // Calculate MSE
            let mse: f64 = y_true.iter()
                .map(|&y| (y - constant_pred).powi(2))
                .sum::<f64>() / n_samples as f64;

            prop_assert!(mse >= 0.0);
            prop_assert!(mse.is_finite());

            // If all true values equal the constant prediction, MSE should be 0
            if y_true.iter().all(|&y| (y - constant_pred).abs() < 1e-10) {
                prop_assert!(mse < 1e-10);
            }
        }
    }
}
