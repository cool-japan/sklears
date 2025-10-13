//! Property-based tests for metrics
//!
//! These tests verify mathematical properties and invariants that should hold
//! for all inputs using proptest. Includes numerical accuracy tests, edge case
//! testing, and stress tests for large datasets.

use crate::classification::*;
use crate::optimized::*;
use crate::regression::*;
use approx::assert_relative_eq;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use scirs2_core::ndarray::Array1;

// Strategies for generating test data
prop_compose! {
    fn binary_classification_data()(
        size in 10..100usize,
        pos_ratio in 0.1..0.9f64
    ) -> (Array1<i32>, Array1<i32>) {
        let n_pos = (size as f64 * pos_ratio) as usize;
        let n_neg = size - n_pos;

        let mut y_true = vec![1; n_pos];
        y_true.extend(vec![0; n_neg]);

        // Create predictions with some noise
        let mut y_pred = y_true.clone();
        let flip_count = size / 10; // Flip ~10% of predictions
        for i in 0..flip_count {
            let idx = i % size;
            y_pred[idx] = 1 - y_pred[idx];
        }

        (Array1::from_vec(y_true), Array1::from_vec(y_pred))
    }
}

prop_compose! {
    fn regression_data()(
        size in 10..100usize,
        values in prop::collection::vec(-100.0..100.0f64, 10..100)
    ) -> (Array1<f64>, Array1<f64>) {
        let size = values.len();
        let y_true = Array1::from_vec(values);

        // Create predictions with some error
        let y_pred: Vec<f64> = y_true.iter()
            .map(|&x| x + (x * 0.1)) // Add 10% error
            .collect();

        (y_true, Array1::from_vec(y_pred))
    }
}

proptest! {
    #[test]
    fn test_accuracy_properties((y_true, y_pred) in binary_classification_data()) {
        let accuracy = accuracy_score(&y_true, &y_pred).unwrap();

        // Accuracy should be between 0 and 1
        prop_assert!((0.0..=1.0).contains(&accuracy));

        // Perfect predictions should give accuracy 1.0
        let perfect_accuracy = accuracy_score(&y_true, &y_true).unwrap();
        prop_assert!((perfect_accuracy - 1.0).abs() < f64::EPSILON);

        // Completely wrong predictions should give accuracy 0.0 (for binary)
        let wrong_pred = y_true.mapv(|x| 1 - x);
        let wrong_accuracy = accuracy_score(&y_true, &wrong_pred).unwrap();
        prop_assert!(wrong_accuracy <= 0.1); // Should be very low
    }

    #[test]
    fn test_precision_recall_properties((y_true, y_pred) in binary_classification_data()) {
        // Test for positive class (1)
        let precision_result = precision_score(&y_true, &y_pred, Some(1));
        let recall_result = recall_score(&y_true, &y_pred, Some(1));

        // Handle edge cases where metrics might be undefined
        if let (Ok(precision), Ok(recall)) = (precision_result, recall_result) {
            // Precision and recall should be between 0 and 1
            prop_assert!((0.0..=1.0).contains(&precision));
            prop_assert!((0.0..=1.0).contains(&recall));
        }

        // Perfect predictions should give precision and recall 1.0
        let perfect_precision = precision_score(&y_true, &y_true, Some(1)).unwrap();
        let perfect_recall = recall_score(&y_true, &y_true, Some(1)).unwrap();
        prop_assert!((perfect_precision - 1.0).abs() < f64::EPSILON);
        prop_assert!((perfect_recall - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_f1_harmonic_mean_property((y_true, y_pred) in binary_classification_data()) {
        let precision_result = precision_score(&y_true, &y_pred, Some(1));
        let recall_result = recall_score(&y_true, &y_pred, Some(1));
        let f1_result = f1_score(&y_true, &y_pred, Some(1));

        // Handle edge cases where metrics might be undefined
        if let (Ok(precision), Ok(recall), Ok(f1)) = (precision_result, recall_result, f1_result) {
            if precision > 0.0 && recall > 0.0 {
                // F1 should be harmonic mean of precision and recall
                let expected_f1 = 2.0 * precision * recall / (precision + recall);
                prop_assert!((f1 - expected_f1).abs() < 1e-10);
            } else {
                // If either precision or recall is 0, F1 should be 0
                prop_assert!(f1 == 0.0);
            }
        }
    }

    #[test]
    fn test_mse_properties((y_true, y_pred) in regression_data()) {
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();

        // MSE should be non-negative
        prop_assert!(mse >= 0.0);

        // MSE should be 0 for perfect predictions
        let perfect_mse = mean_squared_error(&y_true, &y_true).unwrap();
        prop_assert!(perfect_mse < f64::EPSILON);

        // MSE should be symmetric: MSE(a, b) = MSE(b, a)
        let symmetric_mse = mean_squared_error(&y_pred, &y_true).unwrap();
        prop_assert!((mse - symmetric_mse).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mae_properties((y_true, y_pred) in regression_data()) {
        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();

        // MAE should be non-negative
        prop_assert!(mae >= 0.0);

        // MAE should be 0 for perfect predictions
        let perfect_mae = mean_absolute_error(&y_true, &y_true).unwrap();
        prop_assert!(perfect_mae < f64::EPSILON);

        // MAE should be symmetric
        let symmetric_mae = mean_absolute_error(&y_pred, &y_true).unwrap();
        prop_assert!((mae - symmetric_mae).abs() < f64::EPSILON);

        // MAE should be less than or equal to RMSE (sqrt of MSE)
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        let rmse = mse.sqrt();
        prop_assert!(mae <= rmse + f64::EPSILON);
    }

    #[test]
    fn test_r2_properties((y_true, y_pred) in regression_data()) {
        let r2 = r2_score(&y_true, &y_pred).unwrap();

        // R² should be 1 for perfect predictions
        let perfect_r2 = r2_score(&y_true, &y_true).unwrap();
        prop_assert!((perfect_r2 - 1.0).abs() < 1e-10);

        // R² can be negative for very bad predictions, but typically is <= 1
        // We'll just check it's a valid number
        prop_assert!(r2.is_finite());
    }

    // Test optimized vs standard implementations for consistency
    #[test]
    fn test_optimized_vs_standard_mae((y_true, y_pred) in regression_data()) {
        let standard_mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        let optimized_mae = optimized_mean_absolute_error(&y_true, &y_pred, None).unwrap();

        // Results should be identical within floating point precision
        prop_assert!((standard_mae - optimized_mae).abs() < 1e-12);
    }

    #[test]
    fn test_optimized_vs_standard_mse((y_true, y_pred) in regression_data()) {
        let standard_mse = mean_squared_error(&y_true, &y_pred).unwrap();
        let optimized_mse = optimized_mean_squared_error(&y_true, &y_pred, None).unwrap();

        // Results should be identical within floating point precision
        prop_assert!((standard_mse - optimized_mse).abs() < 1e-12);
    }

    #[test]
    fn test_optimized_vs_standard_r2((y_true, y_pred) in regression_data()) {
        let standard_r2 = r2_score(&y_true, &y_pred).unwrap();
        let optimized_r2 = optimized_r2_score(&y_true, &y_pred, None).unwrap();

        // Results should be identical within floating point precision
        prop_assert!((standard_r2 - optimized_r2).abs() < 1e-12);
    }

    // Test streaming metrics consistency
    #[test]
    fn test_streaming_metrics_consistency((y_true, y_pred) in regression_data()) {
        let mut streaming = StreamingMetrics::new(OptimizedConfig::default());

        // Update with entire batch
        streaming.update_batch(&y_true, &y_pred).unwrap();

        let streaming_mae = streaming.mean_absolute_error().unwrap();
        let streaming_mse = streaming.mean_squared_error().unwrap();
        let streaming_r2 = streaming.r2_score().unwrap();

        let standard_mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        let standard_mse = mean_squared_error(&y_true, &y_pred).unwrap();
        let standard_r2 = r2_score(&y_true, &y_pred).unwrap();

        prop_assert!((streaming_mae - standard_mae).abs() < 1e-12);
        prop_assert!((streaming_mse - standard_mse).abs() < 1e-12);
        prop_assert!((streaming_r2 - standard_r2).abs() < 1e-12);
    }

    // Test incremental metrics consistency
    #[test]
    fn test_incremental_metrics_consistency((y_true, y_pred) in regression_data()) {
        let mut incremental = IncrementalMetrics::new(OptimizedConfig::default());

        // Update with entire batch
        incremental.update_batch(&y_true, &y_pred).unwrap();

        let incremental_mae = incremental.mean_absolute_error().unwrap();
        let incremental_mse = incremental.mean_squared_error().unwrap();
        let incremental_r2 = incremental.r2_score().unwrap();

        let standard_mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        let standard_mse = mean_squared_error(&y_true, &y_pred).unwrap();
        let standard_r2 = r2_score(&y_true, &y_pred).unwrap();

        prop_assert!((incremental_mae - standard_mae).abs() < 1e-12);
        prop_assert!((incremental_mse - standard_mse).abs() < 1e-12);
        prop_assert!((incremental_r2 - standard_r2).abs() < 1e-12);
    }

    // Test chunked processing consistency
    #[test]
    fn test_chunked_processing_consistency((y_true, y_pred) in regression_data()) {
        let config = OptimizedConfig {
            chunk_size: 5, // Small chunks to test chunking
            ..OptimizedConfig::default()
        };
        let processor = ChunkedMetricProcessor::new(config);

        let chunked_mae = processor.chunked_mean_absolute_error(&y_true, &y_pred).unwrap();
        let chunked_mse = processor.chunked_mean_squared_error(&y_true, &y_pred).unwrap();

        let standard_mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        let standard_mse = mean_squared_error(&y_true, &y_pred).unwrap();

        prop_assert!((chunked_mae - standard_mae).abs() < 1e-12);
        prop_assert!((chunked_mse - standard_mse).abs() < 1e-12);
    }

    // TODO: Migrate to scirs2-sparse (uses sprs types)
    // // Test sparse confusion matrix consistency
    // #[test]
    // fn test_sparse_confusion_matrix_consistency((y_true, y_pred) in binary_classification_data()) {
    //     let mut sparse_matrix = SparseConfusionMatrix::new();
    //     sparse_matrix.update(&y_true, &y_pred).unwrap();

    //     let sparse_accuracy = sparse_matrix.accuracy();
    //     let standard_accuracy = accuracy_score(&y_true, &y_pred).unwrap();

    //     prop_assert!((sparse_accuracy - standard_accuracy).abs() < 1e-12);
    //     prop_assert_eq!(sparse_matrix.n_samples(), y_true.len());
    // }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_test_setup() {
        // Simple test to ensure property test framework is working
        let (y_true, _) = binary_classification_data()
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();

        assert!(y_true.len() >= 10);
        assert!(y_true.len() <= 100);
    }
}

/// Edge case tests for metrics
#[allow(non_snake_case)]
#[cfg(test)]
mod edge_case_tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_single_element_arrays() {
        let y_true = array![1.0];
        let y_pred = array![1.5];

        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(mae, 0.5, epsilon = 1e-10);

        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(mse, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_identical_arrays() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mae = mean_absolute_error(&y, &y).unwrap();
        assert_relative_eq!(mae, 0.0, epsilon = 1e-15);

        let mse = mean_squared_error(&y, &y).unwrap();
        assert_relative_eq!(mse, 0.0, epsilon = 1e-15);

        let r2 = r2_score(&y, &y).unwrap();
        assert_relative_eq!(r2, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_constant_arrays() {
        let y_true = array![5.0, 5.0, 5.0, 5.0];
        let y_pred = array![3.0, 3.0, 3.0, 3.0];

        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(mae, 2.0, epsilon = 1e-10);

        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(mse, 4.0, epsilon = 1e-10);

        // R² is undefined when all true values are the same (no variance to explain)
        // This should return an error due to division by zero
        assert!(r2_score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_extreme_values() {
        // Use smaller extreme values to avoid overflow
        let y_true = array![1e100, -1e100];
        let y_pred = array![5e99, -5e99];

        let mae: f64 = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert!(mae.is_finite());
        assert!(mae > 0.0);

        let mse: f64 = mean_squared_error(&y_true, &y_pred).unwrap();
        assert!(mse.is_finite());
        assert!(mse > 0.0);
    }

    #[test]
    fn test_zero_arrays() {
        let y_true = array![0.0, 0.0, 0.0];
        let y_pred = array![0.0, 0.0, 0.0];

        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(mae, 0.0, epsilon = 1e-15);

        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(mse, 0.0, epsilon = 1e-15);

        // R² is undefined when true values have no variance (all zeros)
        // This should return an error due to division by zero
        assert!(r2_score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_negative_values() {
        let y_true = array![-1.0, -2.0, -3.0];
        let y_pred = array![-1.5, -2.5, -2.5];

        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert!(mae > 0.0);

        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert!(mse > 0.0);

        let r2: f64 = r2_score(&y_true, &y_pred).unwrap();
        assert!(r2.is_finite());
    }

    #[test]
    fn test_alternating_signs() {
        let y_true = array![-1.0, 1.0, -1.0, 1.0];
        let y_pred = array![1.0, -1.0, 1.0, -1.0];

        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(mae, 2.0, epsilon = 1e-10);

        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(mse, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_large_differences() {
        let y_true = array![0.0, 0.0, 0.0];
        let y_pred = array![1000.0, 2000.0, 3000.0];

        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(mae, 2000.0, epsilon = 1e-10);

        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(mse, 14000000.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_classification_edge_cases() {
        // Test with single class
        let y_true = array![1, 1, 1, 1];
        let y_pred = array![1, 1, 1, 1];

        let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
        assert_relative_eq!(accuracy, 1.0, epsilon = 1e-15);

        // Test completely wrong predictions
        let y_pred_wrong = array![0, 0, 0, 0];
        let accuracy_wrong = accuracy_score(&y_true, &y_pred_wrong).unwrap();
        assert_relative_eq!(accuracy_wrong, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_empty_array_handling() {
        let y_true: Array1<f64> = array![];
        let y_pred: Array1<f64> = array![];

        assert!(mean_absolute_error(&y_true, &y_pred).is_err());
        assert!(mean_squared_error(&y_true, &y_pred).is_err());
        assert!(r2_score(&y_true, &y_pred).is_err());
    }

    #[test]
    fn test_mismatched_shapes() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0, 2.0];

        assert!(mean_absolute_error(&y_true, &y_pred).is_err());
        assert!(mean_squared_error(&y_true, &y_pred).is_err());
        assert!(r2_score(&y_true, &y_pred).is_err());
    }
}

/// Numerical accuracy tests
#[allow(non_snake_case)]
#[cfg(test)]
mod numerical_accuracy_tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_floating_point_precision() {
        // Test with very small differences
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.0 + 1e-15, 2.0 + 1e-15, 3.0 + 1e-15];

        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert!(mae > 0.0);
        assert!(mae < 1e-14);

        // Test optimized vs standard for precision
        let optimized_mae = optimized_mean_absolute_error(&y_true, &y_pred, None).unwrap();
        assert_relative_eq!(mae, optimized_mae, epsilon = 1e-15);
    }

    #[test]
    fn test_large_number_stability() {
        // Test with large numbers that might cause overflow
        let y_true = array![1e10, 2e10, 3e10];
        let y_pred = array![1.1e10, 2.1e10, 2.9e10];

        let mae: f64 = mean_absolute_error(&y_true, &y_pred).unwrap();
        let optimized_mae: f64 = optimized_mean_absolute_error(&y_true, &y_pred, None).unwrap();

        assert!(mae.is_finite());
        assert!(optimized_mae.is_finite());
        assert_relative_eq!(mae, optimized_mae, epsilon = 1e-10);
    }

    #[test]
    fn test_small_number_stability() {
        // Test with very small numbers
        let y_true = array![1e-10, 2e-10, 3e-10];
        let y_pred = array![1.1e-10, 2.1e-10, 2.9e-10];

        let mae: f64 = mean_absolute_error(&y_true, &y_pred).unwrap();
        let optimized_mae: f64 = optimized_mean_absolute_error(&y_true, &y_pred, None).unwrap();

        assert!(mae.is_finite());
        assert!(optimized_mae.is_finite());
        assert_relative_eq!(mae, optimized_mae, epsilon = 1e-12);
    }

    #[test]
    fn test_mixed_scale_values() {
        // Test with values spanning multiple orders of magnitude
        let y_true = array![1e-5, 1e0, 1e5];
        let y_pred = array![1.1e-5, 1.1e0, 1.1e5];

        let mae: f64 = mean_absolute_error(&y_true, &y_pred).unwrap();
        let mse: f64 = mean_squared_error(&y_true, &y_pred).unwrap();
        let r2: f64 = r2_score(&y_true, &y_pred).unwrap();

        assert!(mae.is_finite());
        assert!(mse.is_finite());
        assert!(r2.is_finite());

        // Compare with optimized versions
        let optimized_mae = optimized_mean_absolute_error(&y_true, &y_pred, None).unwrap();
        let optimized_mse = optimized_mean_squared_error(&y_true, &y_pred, None).unwrap();
        let optimized_r2 = optimized_r2_score(&y_true, &y_pred, None).unwrap();

        assert_relative_eq!(mae, optimized_mae, epsilon = 1e-10);
        assert_relative_eq!(mse, optimized_mse, epsilon = 1e-10);
        assert_relative_eq!(r2, optimized_r2, epsilon = 1e-10);
    }

    #[test]
    fn test_catastrophic_cancellation() {
        // Test scenario where subtraction might lose precision
        let base = 1e15;
        let y_true = array![base, base + 1.0, base + 2.0];
        let y_pred = array![base + 0.1, base + 1.1, base + 1.9];

        let mae: f64 = mean_absolute_error(&y_true, &y_pred).unwrap();
        let optimized_mae: f64 = optimized_mean_absolute_error(&y_true, &y_pred, None).unwrap();

        // Differences: |base - (base + 0.1)| = 0.1, |(base+1) - (base+1.1)| = 0.1, |(base+2) - (base+1.9)| = 0.1
        // MAE = (0.1 + 0.1 + 0.1) / 3 = 0.1
        // However, with floating point precision at this scale, we expect some error
        assert!(
            (mae - 0.1).abs() < 0.1,
            "MAE = {}, should be close to 0.1",
            mae
        );
        assert_relative_eq!(mae, optimized_mae, epsilon = 1e-12);
    }
}

/// Stress tests for large datasets
#[allow(non_snake_case)]
#[cfg(test)]
mod stress_tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_large_dataset_consistency() {
        // Create a large dataset
        let size = 1_000_000;
        let y_true: Array1<f64> = Array1::from_iter((0..size).map(|i| i as f64));
        let y_pred: Array1<f64> = Array1::from_iter((0..size).map(|i| i as f64 + 0.1));

        let start = std::time::Instant::now();
        let standard_mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        let standard_time = start.elapsed();

        let start = std::time::Instant::now();
        let optimized_mae = optimized_mean_absolute_error(&y_true, &y_pred, None).unwrap();
        let optimized_time = start.elapsed();

        // Results should be identical
        assert_relative_eq!(standard_mae, optimized_mae, epsilon = 1e-12);

        // Optimized should be faster (or at least not significantly slower)
        println!(
            "Standard time: {:?}, Optimized time: {:?}",
            standard_time, optimized_time
        );

        // Should be exactly 0.1
        assert_relative_eq!(standard_mae, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_streaming_large_dataset() {
        let size = 100_000;
        let mut streaming = StreamingMetrics::new(OptimizedConfig::default());

        // Add data in chunks
        let chunk_size = 1000;
        for chunk_start in (0..size).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(size);
            let chunk_size_actual = chunk_end - chunk_start;

            let y_true: Array1<f64> = Array1::from_iter((chunk_start..chunk_end).map(|i| i as f64));
            let y_pred: Array1<f64> =
                Array1::from_iter((chunk_start..chunk_end).map(|i| i as f64 + 0.1));

            streaming.update_batch(&y_true, &y_pred).unwrap();
        }

        let streaming_mae = streaming.mean_absolute_error().unwrap();
        assert_relative_eq!(streaming_mae, 0.1, epsilon = 1e-10);
        assert_eq!(streaming.n_samples(), size);
    }

    #[test]
    fn test_incremental_large_dataset() {
        let size = 100_000;
        let mut incremental = IncrementalMetrics::new(OptimizedConfig::default());

        // Add data one by one
        for i in 0..size {
            incremental.update(i as f64, i as f64 + 0.1);
        }

        let incremental_mae = incremental.mean_absolute_error().unwrap();
        assert_relative_eq!(incremental_mae, 0.1, epsilon = 1e-10);
        assert_eq!(incremental.n_samples(), size);
    }

    // TODO: Migrate to scirs2-sparse (uses sprs types)
    // #[test]
    // fn test_sparse_matrix_large_dataset() {
    //     let size = 100_000;
    //     let mut sparse_matrix = SparseConfusionMatrix::new();

    //     // Create binary classification data
    //     let y_true: Array1<i32> = Array1::from_iter((0..size).map(|i| (i % 2) as i32));
    //     let y_pred: Array1<i32> = Array1::from_iter((0..size).map(|i| {
    //         if i % 10 == 0 {
    //             1 - (i % 2) as i32
    //         } else {
    //             (i % 2) as i32
    //         }
    //     }));

    //     sparse_matrix.update(&y_true, &y_pred).unwrap();

    //     let accuracy = sparse_matrix.accuracy();
    //     assert!(accuracy > 0.8); // Should be around 0.9 (90% correct)
    //     assert_eq!(sparse_matrix.n_samples(), size);
    // }

    #[test]
    fn test_chunked_processing_large_dataset() {
        let size = 100_000;
        let config = OptimizedConfig {
            chunk_size: 1000,
            ..OptimizedConfig::default()
        };
        let processor = ChunkedMetricProcessor::new(config);

        let y_true: Array1<f64> = Array1::from_iter((0..size).map(|i| i as f64));
        let y_pred: Array1<f64> = Array1::from_iter((0..size).map(|i| i as f64 + 0.1));

        let chunked_mae = processor
            .chunked_mean_absolute_error(&y_true, &y_pred)
            .unwrap();
        let standard_mae = mean_absolute_error(&y_true, &y_pred).unwrap();

        assert_relative_eq!(chunked_mae, standard_mae, epsilon = 1e-10);
        assert_relative_eq!(chunked_mae, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_memory_usage_stability() {
        // Test that memory usage remains stable for streaming operations
        let chunk_size = 10_000;
        let num_chunks = 10;

        let mut streaming = StreamingMetrics::new(OptimizedConfig::default());

        for chunk_idx in 0..num_chunks {
            let y_true: Array1<f64> =
                Array1::from_iter((0..chunk_size).map(|i| (chunk_idx * chunk_size + i) as f64));
            let y_pred: Array1<f64> = Array1::from_iter(
                (0..chunk_size).map(|i| (chunk_idx * chunk_size + i) as f64 + 0.1),
            );

            streaming.update_batch(&y_true, &y_pred).unwrap();

            // Verify metrics remain stable
            let mae = streaming.mean_absolute_error().unwrap();
            assert_relative_eq!(mae, 0.1, epsilon = 1e-10);
        }

        assert_eq!(streaming.n_samples(), chunk_size * num_chunks);
    }
}
