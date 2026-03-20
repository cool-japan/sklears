//! Property-based tests for sklears
//! 
//! This module uses proptest to verify mathematical properties
//! and invariants that should hold for all valid inputs.

use proptest::prelude::*;
use ndarray::{Array1, Array2};
use sklears_core::traits::{Transform};
use sklears_metrics::{accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score};
use sklears_preprocessing::{StandardScaler, MinMaxScaler, RobustScaler, Normalizer};
use sklears_utils::validation::{check_consistent_length, check_X_y};
use sklears_utils::array_utils::{unique_labels, label_counts};
use approx::assert_abs_diff_eq;

// Custom strategies for generating test data
fn small_array_2d() -> impl Strategy<Value = Array2<f64>> {
    prop::collection::vec(
        prop::collection::vec(
            any::<f64>().prop_filter("Must be finite", |x| x.is_finite()),
            2..=5
        ),
        2..=10
    ).prop_map(|rows| {
        let ncols = rows[0].len();
        let data: Vec<f64> = rows.into_iter().flatten().collect();
        Array2::from_shape_vec((data.len() / ncols, ncols), data).unwrap()
    })
}

fn small_array_1d() -> impl Strategy<Value = Array1<f64>> {
    prop::collection::vec(
        any::<f64>().prop_filter("Must be finite", |x| x.is_finite()),
        2..=10
    ).prop_map(Array1::from_vec)
}

fn classification_labels() -> impl Strategy<Value = Array1<i32>> {
    prop::collection::vec(0i32..3, 2..=10).prop_map(Array1::from_vec)
}

fn regression_targets() -> impl Strategy<Value = Array1<f64>> {
    prop::collection::vec(
        any::<f64>().prop_filter("Must be finite", |x| x.is_finite()),
        2..=10
    ).prop_map(Array1::from_vec)
}

proptest! {
    #[test]
    fn test_standard_scaler_properties(x in small_array_2d()) {
        // Skip if data has zero variance in any column
        let mut has_zero_variance = false;
        for col_idx in 0..x.ncols() {
            let column = x.column(col_idx);
            let mean = column.mean().unwrap_or(0.0);
            let variance = column.iter()
                .map(|&val| (val - mean).powi(2))
                .sum::<f64>() / (column.len() as f64);
            if variance < 1e-10 {
                has_zero_variance = true;
                break;
            }
        }
        
        if !has_zero_variance && x.nrows() > 1 {
            let scaler = StandardScaler::new().fit(&x, &()).unwrap();
            let x_scaled = scaler.transform(&x).unwrap();
            
            // Property 1: Same shape after transformation
            prop_assert_eq!(x_scaled.shape(), x.shape());
            
            // Property 2: Mean should be approximately zero
            let means = x_scaled.mean_axis(ndarray::Axis(0)).unwrap();
            for &mean in means.iter() {
                prop_assert!(mean.abs() < 1e-10, "Mean should be close to zero, got {}", mean);
            }
            
            // Property 3: Standard deviation should be approximately 1
            for col_idx in 0..x_scaled.ncols() {
                let column = x_scaled.column(col_idx);
                let mean = column.mean().unwrap_or(0.0);
                let std = (column.iter()
                    .map(|&val| (val - mean).powi(2))
                    .sum::<f64>() / (column.len() as f64 - 1.0)).sqrt();
                
                prop_assert!((std - 1.0).abs() < 1e-10, 
                           "Standard deviation should be close to 1, got {}", std);
            }
        }
    }

    #[test]
    fn test_minmax_scaler_properties(x in small_array_2d()) {
        if x.nrows() > 0 && x.ncols() > 0 {
            let scaler = MinMaxScaler::new().fit(&x, &()).unwrap();
            let x_scaled = scaler.transform(&x).unwrap();
            
            // Property 1: Same shape
            prop_assert_eq!(x_scaled.shape(), x.shape());
            
            // Property 2: Values should be in [0, 1] range
            for &value in x_scaled.iter() {
                prop_assert!(value >= -1e-10 && value <= 1.0 + 1e-10, 
                           "Scaled value should be in [0,1], got {}", value);
            }
            
            // Property 3: Min and max values should be 0 and 1 (unless constant column)
            for col_idx in 0..x_scaled.ncols() {
                let column = x_scaled.column(col_idx);
                let min_val = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                
                // Check if original column was constant
                let orig_column = x.column(col_idx);
                let orig_min = orig_column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let orig_max = orig_column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                
                if (orig_max - orig_min).abs() > 1e-10 {
                    prop_assert!(min_val < 1e-10, "Min should be close to 0, got {}", min_val);
                    prop_assert!((max_val - 1.0).abs() < 1e-10, "Max should be close to 1, got {}", max_val);
                }
            }
        }
    }

    #[test]
    fn test_normalizer_properties(x in small_array_2d()) {
        if x.nrows() > 0 && x.ncols() > 0 {
            let normalizer = Normalizer::new().norm("l2").fit(&x, &()).unwrap();
            let x_normalized = normalizer.transform(&x).unwrap();
            
            // Property 1: Same shape
            prop_assert_eq!(x_normalized.shape(), x.shape());
            
            // Property 2: Each row should have unit norm (L2)
            for row_idx in 0..x_normalized.nrows() {
                let row = x_normalized.row(row_idx);
                let norm = row.iter().map(|&x| x * x).sum::<f64>().sqrt();
                
                // Check if original row was zero vector
                let orig_row = x.row(row_idx);
                let orig_norm = orig_row.iter().map(|&x| x * x).sum::<f64>().sqrt();
                
                if orig_norm > 1e-10 {
                    prop_assert!((norm - 1.0).abs() < 1e-10, 
                               "Row {} should have unit norm, got {}", row_idx, norm);
                }
            }
        }
    }

    #[test]
    fn test_accuracy_score_properties(
        y_true in classification_labels(),
        y_pred in classification_labels()
    ) {
        if y_true.len() == y_pred.len() && !y_true.is_empty() {
            let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
            
            // Property 1: Accuracy is between 0 and 1
            prop_assert!(accuracy >= 0.0 && accuracy <= 1.0,
                        "Accuracy should be in [0,1], got {}", accuracy);
            
            // Property 2: Perfect predictions give accuracy 1
            let perfect_accuracy = accuracy_score(&y_true, &y_true).unwrap();
            prop_assert!((perfect_accuracy - 1.0).abs() < 1e-10,
                        "Perfect predictions should give accuracy 1, got {}", perfect_accuracy);
            
            // Property 3: Accuracy is symmetric for binary classification
            if y_true.iter().all(|&x| x == 0 || x == 1) && y_pred.iter().all(|&x| x == 0 || x == 1) {
                let flipped_true = y_true.mapv(|x| if x == 0 { 1 } else { 0 });
                let flipped_pred = y_pred.mapv(|x| if x == 0 { 1 } else { 0 });
                let flipped_accuracy = accuracy_score(&flipped_true, &flipped_pred).unwrap();
                
                prop_assert!((accuracy - flipped_accuracy).abs() < 1e-10,
                           "Accuracy should be symmetric for binary classification");
            }
        }
    }

    #[test]
    fn test_mse_properties(
        y_true in regression_targets(),
        y_pred in regression_targets()
    ) {
        if y_true.len() == y_pred.len() && !y_true.is_empty() {
            let mse = mean_squared_error(&y_true, &y_pred).unwrap();
            
            // Property 1: MSE is non-negative
            prop_assert!(mse >= 0.0, "MSE should be non-negative, got {}", mse);
            
            // Property 2: Perfect predictions give MSE 0
            let perfect_mse = mean_squared_error(&y_true, &y_true).unwrap();
            prop_assert!(perfect_mse < 1e-10, "Perfect predictions should give MSE 0, got {}", perfect_mse);
            
            // Property 3: MSE is symmetric
            let swapped_mse = mean_squared_error(&y_pred, &y_true).unwrap();
            prop_assert!((mse - swapped_mse).abs() < 1e-10,
                        "MSE should be symmetric, got {} vs {}", mse, swapped_mse);
        }
    }

    #[test]
    fn test_r2_score_properties(
        y_true in regression_targets(),
        y_pred in regression_targets()
    ) {
        if y_true.len() == y_pred.len() && y_true.len() > 1 {
            // Check if y_true has variance
            let mean_true = y_true.mean().unwrap_or(0.0);
            let variance = y_true.iter()
                .map(|&y| (y - mean_true).powi(2))
                .sum::<f64>() / y_true.len() as f64;
            
            if variance > 1e-10 {
                let r2 = r2_score(&y_true, &y_pred).unwrap();
                
                // Property 1: R² ≤ 1 for any predictions
                prop_assert!(r2 <= 1.0 + 1e-10, "R² should be ≤ 1, got {}", r2);
                
                // Property 2: Perfect predictions give R² = 1
                let perfect_r2 = r2_score(&y_true, &y_true).unwrap();
                prop_assert!((perfect_r2 - 1.0).abs() < 1e-10,
                           "Perfect predictions should give R² = 1, got {}", perfect_r2);
                
                // Property 3: Constant predictions give R² = 0
                let mean_predictions = Array1::from_elem(y_true.len(), mean_true);
                let constant_r2 = r2_score(&y_true, &mean_predictions).unwrap();
                prop_assert!(constant_r2.abs() < 1e-10,
                           "Constant predictions should give R² ≈ 0, got {}", constant_r2);
            }
        }
    }

    #[test]
    fn test_precision_recall_f1_relationship(
        y_true in classification_labels(),
        y_pred in classification_labels()
    ) {
        if y_true.len() == y_pred.len() && !y_true.is_empty() {
            // Test for macro averaging
            if let (Ok(precision), Ok(recall), Ok(f1)) = (
                precision_score(&y_true, &y_pred, "macro"),
                recall_score(&y_true, &y_pred, "macro"), 
                f1_score(&y_true, &y_pred, "macro")
            ) {
                // Property 1: All metrics are in [0, 1]
                prop_assert!(precision >= 0.0 && precision <= 1.0,
                           "Precision should be in [0,1], got {}", precision);
                prop_assert!(recall >= 0.0 && recall <= 1.0,
                           "Recall should be in [0,1], got {}", recall);
                prop_assert!(f1 >= 0.0 && f1 <= 1.0,
                           "F1 should be in [0,1], got {}", f1);
                
                // Property 2: F1 is harmonic mean of precision and recall
                if precision > 0.0 && recall > 0.0 {
                    let expected_f1 = 2.0 * (precision * recall) / (precision + recall);
                    prop_assert!((f1 - expected_f1).abs() < 1e-6,
                               "F1 should be harmonic mean of precision and recall, got {} vs expected {}",
                               f1, expected_f1);
                }
                
                // Property 3: Perfect predictions give all metrics = 1
                let perfect_precision = precision_score(&y_true, &y_true, "macro").unwrap();
                let perfect_recall = recall_score(&y_true, &y_true, "macro").unwrap();
                let perfect_f1 = f1_score(&y_true, &y_true, "macro").unwrap();
                
                prop_assert!((perfect_precision - 1.0).abs() < 1e-10,
                           "Perfect precision should be 1, got {}", perfect_precision);
                prop_assert!((perfect_recall - 1.0).abs() < 1e-10,
                           "Perfect recall should be 1, got {}", perfect_recall);
                prop_assert!((perfect_f1 - 1.0).abs() < 1e-10,
                           "Perfect F1 should be 1, got {}", perfect_f1);
            }
        }
    }

    #[test]
    fn test_unique_labels_properties(y in classification_labels()) {
        if !y.is_empty() {
            let unique = unique_labels(&y);
            
            // Property 1: All unique labels should be present in original array
            for &label in unique.iter() {
                prop_assert!(y.iter().any(|&x| x == label),
                           "Unique label {} not found in original array", label);
            }
            
            // Property 2: Unique labels should be sorted
            for i in 1..unique.len() {
                prop_assert!(unique[i-1] < unique[i],
                           "Unique labels should be sorted, but {} >= {}", unique[i-1], unique[i]);
            }
            
            // Property 3: No duplicates in unique labels
            let mut sorted_unique = unique.clone();
            sorted_unique.sort();
            sorted_unique.dedup();
            prop_assert_eq!(unique.len(), sorted_unique.len(),
                          "Unique labels should contain no duplicates");
        }
    }

    #[test]
    fn test_label_counts_properties(y in classification_labels()) {
        if !y.is_empty() {
            let counts = label_counts(&y);
            
            // Property 1: Sum of counts should equal array length
            let total_count: usize = counts.values().sum();
            prop_assert_eq!(total_count, y.len(),
                          "Sum of counts should equal array length");
            
            // Property 2: All counts should be positive
            for (&label, &count) in counts.iter() {
                prop_assert!(count > 0, "Count for label {} should be positive, got {}", label, count);
            }
            
            // Property 3: Each label should appear correct number of times
            for (&label, &expected_count) in counts.iter() {
                let actual_count = y.iter().filter(|&&x| x == label).count();
                prop_assert_eq!(actual_count, expected_count,
                              "Label {} should appear {} times, but appears {} times", 
                              label, expected_count, actual_count);
            }
        }
    }

    #[test]
    fn test_check_consistent_length_properties(
        x in small_array_2d(),
        y in classification_labels()
    ) {
        // Property 1: Arrays with same length should pass
        if x.nrows() == y.len() {
            prop_assert!(check_consistent_length(&[x.nrows(), y.len()]).is_ok(),
                        "Arrays with same length should pass consistency check");
        }
        
        // Property 2: Arrays with different lengths should fail
        if x.nrows() != y.len() {
            prop_assert!(check_consistent_length(&[x.nrows(), y.len()]).is_err(),
                        "Arrays with different lengths should fail consistency check");
        }
        
        // Property 3: Single array should always pass
        prop_assert!(check_consistent_length(&[x.nrows()]).is_ok(),
                    "Single array should always pass consistency check");
    }

    #[test]
    fn test_transformer_inverse_property(x in small_array_2d()) {
        if x.nrows() > 1 && x.ncols() > 0 {
            // Test if we can invert scaling transformations
            let mut has_variance = true;
            for col_idx in 0..x.ncols() {
                let column = x.column(col_idx);
                let mean = column.mean().unwrap_or(0.0);
                let variance = column.iter()
                    .map(|&val| (val - mean).powi(2))
                    .sum::<f64>() / column.len() as f64;
                if variance < 1e-10 {
                    has_variance = false;
                    break;
                }
            }
            
            if has_variance {
                // Test StandardScaler inverse
                let scaler = StandardScaler::new().fit(&x, &()).unwrap();
                let x_scaled = scaler.transform(&x).unwrap();
                let x_inverse = scaler.inverse_transform(&x_scaled).unwrap();
                
                // Property: inverse transform should recover original data
                for i in 0..x.nrows() {
                    for j in 0..x.ncols() {
                        prop_assert!((x[[i, j]] - x_inverse[[i, j]]).abs() < 1e-10,
                                   "Inverse transform failed at ({}, {}): {} vs {}", 
                                   i, j, x[[i, j]], x_inverse[[i, j]]);
                    }
                }
            }
        }
    }

    #[test] 
    fn test_scaler_fit_transform_equivalence(x in small_array_2d()) {
        if x.nrows() > 0 && x.ncols() > 0 {
            // Property: fit_transform should be equivalent to fit then transform
            let scaler1 = StandardScaler::new();
            let x_fit_transform = scaler1.fit_transform(&x, &()).unwrap();
            
            let scaler2 = StandardScaler::new().fit(&x, &()).unwrap();
            let x_transform = scaler2.transform(&x).unwrap();
            
            prop_assert_eq!(x_fit_transform.shape(), x_transform.shape(),
                          "fit_transform and fit+transform should have same shape");
            
            for i in 0..x.nrows() {
                for j in 0..x.ncols() {
                    prop_assert!((x_fit_transform[[i, j]] - x_transform[[i, j]]).abs() < 1e-10,
                               "fit_transform and fit+transform should be equivalent at ({}, {}): {} vs {}", 
                               i, j, x_fit_transform[[i, j]], x_transform[[i, j]]);
                }
            }
        }
    }
}