//! Comparative tests against reference implementations
//!
//! These tests compare our metric implementations against known reference values
//! and other widely-used implementations to ensure correctness.

use crate::classification::*;
use crate::optimized::*;
use crate::regression::*;
use approx::assert_relative_eq;
use scirs2_core::ndarray::{array, Array1};

#[allow(non_snake_case)]
#[cfg(test)]
mod reference_values_tests {
    use super::*;

    #[test]
    fn test_mae_reference_values() {
        // Known reference values computed manually and verified
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        // Manual calculation: |3-2.5| + |-0.5-0| + |2-2| + |7-8| = 0.5 + 0.5 + 0 + 1 = 2.0
        // Average: 2.0 / 4 = 0.5
        let expected_mae = 0.5;

        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(mae, expected_mae, epsilon = 1e-15);

        // Test optimized version
        let optimized_mae = optimized_mean_absolute_error(&y_true, &y_pred, None).unwrap();
        assert_relative_eq!(optimized_mae, expected_mae, epsilon = 1e-15);
    }

    #[test]
    fn test_mse_reference_values() {
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];

        // Manual calculation: (3-2.5)² + (-0.5-0)² + (2-2)² + (7-8)² = 0.25 + 0.25 + 0 + 1 = 1.5
        // Average: 1.5 / 4 = 0.375
        let expected_mse = 0.375;

        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(mse, expected_mse, epsilon = 1e-15);

        // Test optimized version
        let optimized_mse = optimized_mean_squared_error(&y_true, &y_pred, None).unwrap();
        assert_relative_eq!(optimized_mse, expected_mse, epsilon = 1e-15);
    }

    #[test]
    fn test_r2_reference_values() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.1, 1.9, 3.1, 3.9, 5.1];

        // Manual calculation of R²
        // mean_true = (1+2+3+4+5)/5 = 3.0
        // SS_tot = Σ(y_true - mean_true)² = (1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)² = 4+1+0+1+4 = 10
        // SS_res = Σ(y_true - y_pred)² = (1-1.1)² + (2-1.9)² + (3-3.1)² + (4-3.9)² + (5-5.1)²
        //        = 0.01 + 0.01 + 0.01 + 0.01 + 0.01 = 0.05
        // R² = 1 - SS_res/SS_tot = 1 - 0.05/10 = 1 - 0.005 = 0.995
        let expected_r2 = 0.995;

        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert_relative_eq!(r2, expected_r2, epsilon = 1e-15);

        // Test optimized version
        let optimized_r2 = optimized_r2_score(&y_true, &y_pred, None).unwrap();
        assert_relative_eq!(optimized_r2, expected_r2, epsilon = 1e-15);
    }

    #[test]
    fn test_accuracy_reference_values() {
        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 2, 1, 0, 0, 1];

        // Manual calculation: correct predictions at indices 0, 3 = 2 out of 6
        // Accuracy = 2/6 = 1/3 ≈ 0.333333...
        let expected_accuracy = 1.0 / 3.0;

        let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
        assert_relative_eq!(accuracy, expected_accuracy, epsilon = 1e-15);
    }

    #[test]
    fn test_precision_recall_reference_values() {
        // Binary classification case
        let y_true = array![1, 1, 0, 1, 0, 0];
        let y_pred = array![1, 0, 0, 1, 1, 0];

        // For class 1:
        // TP = 2 (indices 0, 3), FP = 1 (index 4), FN = 1 (index 1)
        // Precision = TP/(TP+FP) = 2/(2+1) = 2/3
        // Recall = TP/(TP+FN) = 2/(2+1) = 2/3
        let expected_precision = 2.0 / 3.0;
        let expected_recall = 2.0 / 3.0;

        let precision = precision_score(&y_true, &y_pred, Some(1)).unwrap();
        let recall = recall_score(&y_true, &y_pred, Some(1)).unwrap();

        assert_relative_eq!(precision, expected_precision, epsilon = 1e-15);
        assert_relative_eq!(recall, expected_recall, epsilon = 1e-15);

        // F1 = 2 * precision * recall / (precision + recall) = 2 * (2/3) * (2/3) / (2/3 + 2/3) = 2/3
        let expected_f1 = 2.0 / 3.0;
        let f1 = f1_score(&y_true, &y_pred, Some(1)).unwrap();
        assert_relative_eq!(f1, expected_f1, epsilon = 1e-15);
    }

    #[test]
    fn test_perfect_predictions() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // Perfect predictions should give MAE=0, MSE=0, R²=1
        let mae = mean_absolute_error(&y, &y).unwrap();
        let mse = mean_squared_error(&y, &y).unwrap();
        let r2 = r2_score(&y, &y).unwrap();

        assert_relative_eq!(mae, 0.0, epsilon = 1e-15);
        assert_relative_eq!(mse, 0.0, epsilon = 1e-15);
        assert_relative_eq!(r2, 1.0, epsilon = 1e-15);

        // Test classification perfect predictions
        let y_class = array![0, 1, 2, 0, 1, 2];
        let accuracy = accuracy_score(&y_class, &y_class).unwrap();
        assert_relative_eq!(accuracy, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_worst_case_predictions() {
        // For binary classification, worst case is flipping all predictions
        let y_true = array![0, 0, 1, 1];
        let y_pred = array![1, 1, 0, 0];

        let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
        assert_relative_eq!(accuracy, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_constant_prediction() {
        // Predicting the mean should give R² = 0
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let y_pred = array![mean, mean, mean, mean, mean];

        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert_relative_eq!(r2, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_negative_r2() {
        // Predictions worse than mean should give negative R²
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![5.0, 1.0, 5.0, 1.0, 5.0]; // Very bad predictions

        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert!(
            r2 < 0.0,
            "R² should be negative for very bad predictions, got {}",
            r2
        );
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod sklearn_compatibility_tests {
    use super::*;

    /// These tests use values computed from scikit-learn to ensure compatibility
    /// Results were generated using:
    /// ```python
    /// from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
    /// import numpy as np
    /// ```

    #[test]
    fn test_sklearn_mae_compatibility() {
        // From sklearn: mean_absolute_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];
        let sklearn_result = 0.5;

        let our_result = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(our_result, sklearn_result, epsilon = 1e-15);
    }

    #[test]
    fn test_sklearn_mse_compatibility() {
        // From sklearn: mean_squared_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
        let y_true = array![3.0, -0.5, 2.0, 7.0];
        let y_pred = array![2.5, 0.0, 2.0, 8.0];
        let sklearn_result = 0.375;

        let our_result = mean_squared_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(our_result, sklearn_result, epsilon = 1e-15);
    }

    #[test]
    fn test_sklearn_r2_compatibility() {
        // From sklearn: r2_score([1, 2, 3, 4, 5], [1.1, 1.9, 3.1, 3.9, 5.1])
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.1, 1.9, 3.1, 3.9, 5.1];
        let sklearn_result = 0.995;

        let our_result = r2_score(&y_true, &y_pred).unwrap();
        assert_relative_eq!(our_result, sklearn_result, epsilon = 1e-15);
    }

    #[test]
    fn test_sklearn_accuracy_compatibility() {
        // From sklearn: accuracy_score([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1])
        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 2, 1, 0, 0, 1];
        let sklearn_result = 0.3333333333333333;

        let our_result = accuracy_score(&y_true, &y_pred).unwrap();
        assert_relative_eq!(our_result, sklearn_result, epsilon = 1e-15);
    }

    #[test]
    fn test_sklearn_larger_dataset() {
        // Test with a larger dataset to ensure scaling behavior matches
        let size = 1000;
        let y_true: Array1<f64> = Array1::from_iter((0..size).map(|i| i as f64));
        let y_pred: Array1<f64> = Array1::from_iter((0..size).map(|i| i as f64 + 0.5));

        // Expected MAE should be exactly 0.5
        let expected_mae = 0.5;
        let our_mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(our_mae, expected_mae, epsilon = 1e-15);

        // Expected MSE should be exactly 0.25
        let expected_mse = 0.25;
        let our_mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(our_mse, expected_mse, epsilon = 1e-15);
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod mathematical_properties_tests {
    use super::*;

    #[test]
    fn test_mae_triangle_inequality() {
        // MAE should satisfy triangle inequality: MAE(a,c) <= MAE(a,b) + MAE(b,c)
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.5, 2.5, 2.5];
        let c = array![2.0, 3.0, 2.0];

        let mae_ac = mean_absolute_error(&a, &c).unwrap();
        let mae_ab = mean_absolute_error(&a, &b).unwrap();
        let mae_bc = mean_absolute_error(&b, &c).unwrap();

        assert!(mae_ac <= mae_ab + mae_bc + 1e-15);
    }

    #[test]
    fn test_mse_convexity() {
        // MSE is convex: MSE(λa + (1-λ)b, c) <= λ*MSE(a,c) + (1-λ)*MSE(b,c)
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 3.0, 1.0];
        let c = array![1.5, 2.5, 2.5];
        let lambda = 0.3;

        let interpolated = &a * lambda + &b * (1.0 - lambda);

        let mse_interp = mean_squared_error(&interpolated, &c).unwrap();
        let mse_a = mean_squared_error(&a, &c).unwrap();
        let mse_b = mean_squared_error(&b, &c).unwrap();
        let convex_bound = lambda * mse_a + (1.0 - lambda) * mse_b;

        assert!(mse_interp <= convex_bound + 1e-15);
    }

    #[test]
    fn test_mae_scale_invariance() {
        // MAE should scale linearly: MAE(ka, kb) = k * MAE(a, b) for k > 0
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.1, 2.1, 2.9];
        let k = 5.0;

        let mae_original = mean_absolute_error(&a, &b).unwrap();
        let mae_scaled = mean_absolute_error(&(&a * k), &(&b * k)).unwrap();

        assert_relative_eq!(mae_scaled, k * mae_original, epsilon = 1e-15);
    }

    #[test]
    fn test_mse_quadratic_scaling() {
        // MSE should scale quadratically: MSE(ka, kb) = k² * MSE(a, b) for k > 0
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.1, 2.1, 2.9];
        let k = 3.0;

        let mse_original = mean_squared_error(&a, &b).unwrap();
        let mse_scaled = mean_squared_error(&(&a * k), &(&b * k)).unwrap();

        assert_relative_eq!(mse_scaled, k * k * mse_original, epsilon = 1e-15);
    }

    #[test]
    fn test_r2_translation_invariance() {
        // R² should be invariant under translation: R²(a+c, b+c) = R²(a, b)
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.1, 2.1, 2.9];
        let c = 100.0;

        let r2_original = r2_score(&a, &b).unwrap();
        let r2_translated = r2_score(&(&a + c), &(&b + c)).unwrap();

        // Use a more relaxed tolerance due to floating point precision with large numbers
        assert_relative_eq!(r2_translated, r2_original, epsilon = 1e-12);
    }

    // TODO: Migrate to scirs2-sparse (uses sprs types)
    // #[test]
    // fn test_confusion_matrix_properties() {
    //     let y_true = array![0, 1, 0, 1, 0, 1];
    //     let y_pred = array![0, 1, 1, 0, 0, 1];

    //     let mut matrix = SparseConfusionMatrix::new();
    //     matrix.update(&y_true, &y_pred).unwrap();

    //     // Sum of all entries should equal total samples
    //     let total_from_matrix = matrix
    //         .labels()
    //         .iter()
    //         .map(|&true_label| {
    //             matrix
    //                 .labels()
    //                 .iter()
    //                 .map(|&pred_label| matrix.get(true_label, pred_label))
    //                 .sum::<usize>()
    //         })
    //         .sum::<usize>();

    //     assert_eq!(total_from_matrix, matrix.n_samples());

    //     // Accuracy should be sum of diagonal divided by total
    //     let diagonal_sum: usize = matrix
    //         .labels()
    //         .iter()
    //         .map(|&label| matrix.get(label, label))
    //         .sum();

    //     let expected_accuracy = diagonal_sum as f64 / matrix.n_samples() as f64;
    //     assert_relative_eq!(matrix.accuracy(), expected_accuracy, epsilon = 1e-15);
    // }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod consistency_across_implementations {
    use super::*;

    #[test]
    fn test_all_implementations_consistency() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y_pred = array![1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9];

        // Standard implementations
        let std_mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        let std_mse = mean_squared_error(&y_true, &y_pred).unwrap();
        let std_r2 = r2_score(&y_true, &y_pred).unwrap();

        // Optimized implementations
        let opt_mae = optimized_mean_absolute_error(&y_true, &y_pred, None).unwrap();
        let opt_mse = optimized_mean_squared_error(&y_true, &y_pred, None).unwrap();
        let opt_r2 = optimized_r2_score(&y_true, &y_pred, None).unwrap();

        // Streaming implementation
        let mut streaming = StreamingMetrics::new(OptimizedConfig::default());
        streaming.update_batch(&y_true, &y_pred).unwrap();
        let stream_mae = streaming.mean_absolute_error().unwrap();
        let stream_mse = streaming.mean_squared_error().unwrap();
        let stream_r2 = streaming.r2_score().unwrap();

        // Incremental implementation
        let mut incremental = IncrementalMetrics::new(OptimizedConfig::default());
        incremental.update_batch(&y_true, &y_pred).unwrap();
        let inc_mae = incremental.mean_absolute_error().unwrap();
        let inc_mse = incremental.mean_squared_error().unwrap();
        let inc_r2 = incremental.r2_score().unwrap();

        // Chunked implementation
        let processor = ChunkedMetricProcessor::new(OptimizedConfig {
            chunk_size: 3,
            ..OptimizedConfig::default()
        });
        let chunk_mae = processor
            .chunked_mean_absolute_error(&y_true, &y_pred)
            .unwrap();
        let chunk_mse = processor
            .chunked_mean_squared_error(&y_true, &y_pred)
            .unwrap();

        // All should be identical
        let epsilon = 1e-15;

        // MAE consistency
        assert_relative_eq!(opt_mae, std_mae, epsilon = epsilon);
        assert_relative_eq!(stream_mae, std_mae, epsilon = epsilon);
        assert_relative_eq!(inc_mae, std_mae, epsilon = epsilon);
        assert_relative_eq!(chunk_mae, std_mae, epsilon = epsilon);

        // MSE consistency
        assert_relative_eq!(opt_mse, std_mse, epsilon = epsilon);
        assert_relative_eq!(stream_mse, std_mse, epsilon = epsilon);
        assert_relative_eq!(inc_mse, std_mse, epsilon = epsilon);
        assert_relative_eq!(chunk_mse, std_mse, epsilon = epsilon);

        // R² consistency
        assert_relative_eq!(opt_r2, std_r2, epsilon = epsilon);
        assert_relative_eq!(stream_r2, std_r2, epsilon = epsilon);
        assert_relative_eq!(inc_r2, std_r2, epsilon = epsilon);
    }
}
