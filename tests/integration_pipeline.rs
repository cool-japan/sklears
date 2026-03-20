//! Integration tests for complete ML pipelines
//!
//! These tests demonstrate end-to-end functionality across multiple crates
//! without relying on BLAS operations to avoid linking issues.

use ndarray::{Array1, Array2};
use sklears_core::traits::{Transform};
use sklears_utils::data_generation::{make_classification, make_regression, make_blobs};
use sklears_utils::random::train_test_split_indices;
use sklears_metrics::classification::{accuracy_score, precision_score, recall_score, f1_score};
use sklears_metrics::regression::{mean_squared_error, mean_absolute_error, r2_score};

#[test]
fn test_classification_pipeline_simple() {
    // Generate synthetic classification data
    let (x, y) = make_classification(100, 4, 2, None, None, 0.0, 1.0, Some(42))
        .expect("Failed to generate classification data");

    // Split data
    let (train_indices, test_indices) = train_test_split_indices(100, 0.3, true, Some(42))
        .expect("Failed to split data");

    // Create train/test splits
    let x_train = x.select(ndarray::Axis(0), &train_indices);
    let x_test = x.select(ndarray::Axis(0), &test_indices);
    let y_train = y.select(ndarray::Axis(0), &train_indices);
    let y_test = y.select(ndarray::Axis(0), &test_indices);

    // Verify data integrity
    assert_eq!(x_train.nrows(), train_indices.len());
    assert_eq!(x_test.nrows(), test_indices.len());
    assert_eq!(y_train.len(), train_indices.len());
    assert_eq!(y_test.len(), test_indices.len());

    // Check that all labels are valid (0 or 1 for binary classification)
    for &label in y_train.iter() {
        assert!(label == 0 || label == 1);
    }
    for &label in y_test.iter() {
        assert!(label == 0 || label == 1);
    }

    println!("Classification pipeline test completed successfully");
}

#[test]
fn test_regression_pipeline_simple() {
    // Generate synthetic regression data
    let (x, y) = make_regression(80, 3, Some(3), 0.1, 0.0, Some(42))
        .expect("Failed to generate regression data");

    // Split data
    let (train_indices, test_indices) = train_test_split_indices(80, 0.25, true, Some(42))
        .expect("Failed to split data");

    // Create train/test splits
    let x_train = x.select(ndarray::Axis(0), &train_indices);
    let x_test = x.select(ndarray::Axis(0), &test_indices);
    let y_train = y.select(ndarray::Axis(0), &train_indices);
    let y_test = y.select(ndarray::Axis(0), &test_indices);

    // Verify data integrity
    assert_eq!(x_train.nrows(), train_indices.len());
    assert_eq!(x_test.nrows(), test_indices.len());
    assert_eq!(y_train.len(), train_indices.len());
    assert_eq!(y_test.len(), test_indices.len());

    // Check that all values are finite
    for &val in x_train.iter() {
        assert!(val.is_finite());
    }
    for &val in y_train.iter() {
        assert!(val.is_finite());
    }

    println!("Regression pipeline test completed successfully");
}

#[test]
fn test_clustering_pipeline_simple() {
    // Generate synthetic clustering data
    let (x, y_true) = make_blobs(60, 2, Some(3), 1.0, (-5.0, 5.0), Some(42))
        .expect("Failed to generate clustering data");

    // Verify data integrity
    assert_eq!(x.shape(), &[60, 2]);
    assert_eq!(y_true.len(), 60);

    // Check that we have 3 clusters as requested
    let unique_labels: std::collections::HashSet<i32> = y_true.iter().copied().collect();
    assert_eq!(unique_labels.len(), 3);

    // Check that all cluster labels are in expected range (0, 1, 2)
    for &label in y_true.iter() {
        assert!(label >= 0 && label < 3);
    }

    // Check that all feature values are finite
    for &val in x.iter() {
        assert!(val.is_finite());
    }

    println!("Clustering pipeline test completed successfully");
}

#[test]
fn test_data_preprocessing_pipeline() {
    // Create simple test data
    let mut x = Array2::<f64>::zeros((20, 3));
    for i in 0..20 {
        for j in 0..3 {
            x[[i, j]] = (i as f64 * 0.1) + (j as f64 * 0.5) + 1.0;
        }
    }

    // Test basic preprocessing operations
    use sklears_preprocessing::scaling::{StandardScaler, MinMaxScaler};

    // Standard scaling
    let standard_scaler = StandardScaler::new();
    let fitted_scaler = standard_scaler.fit(&x, &()).expect("StandardScaler fit failed");
    let x_scaled = fitted_scaler.transform(&x).expect("StandardScaler transform failed");

    // Verify shapes are preserved
    assert_eq!(x_scaled.shape(), x.shape());

    // Verify all values are finite
    for &val in x_scaled.iter() {
        assert!(val.is_finite());
    }

    // MinMax scaling  
    let minmax_scaler = MinMaxScaler::new(0.0, 1.0);
    let fitted_minmax = minmax_scaler.fit(&x, &()).expect("MinMaxScaler fit failed");
    let x_minmax = fitted_minmax.transform(&x).expect("MinMaxScaler transform failed");

    // Verify shapes are preserved
    assert_eq!(x_minmax.shape(), x.shape());

    // Verify all values are in [0, 1] range
    for &val in x_minmax.iter() {
        assert!(val >= 0.0 && val <= 1.0);
        assert!(val.is_finite());
    }

    println!("Preprocessing pipeline test completed successfully");
}

#[test]
fn test_metrics_computation() {
    // Test classification metrics
    let y_true = Array1::from_vec(vec![0, 1, 0, 1, 1, 0, 1, 0]);
    let y_pred = Array1::from_vec(vec![0, 1, 1, 1, 0, 0, 1, 0]);

    let accuracy = accuracy_score(&y_true, &y_pred).expect("Accuracy computation failed");
    let precision = precision_score(&y_true, &y_pred, None).expect("Precision computation failed");
    let recall = recall_score(&y_true, &y_pred, None).expect("Recall computation failed");
    let f1 = f1_score(&y_true, &y_pred, None).expect("F1 computation failed");

    // Basic sanity checks
    assert!(accuracy >= 0.0 && accuracy <= 1.0);
    assert!(precision >= 0.0 && precision <= 1.0);
    assert!(recall >= 0.0 && recall <= 1.0);
    assert!(f1 >= 0.0 && f1 <= 1.0);

    // Test regression metrics
    let y_true_reg = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_pred_reg = Array1::from_vec(vec![1.1, 2.1, 2.9, 3.8, 5.2]);

    let mse = mean_squared_error(&y_true_reg, &y_pred_reg).expect("MSE computation failed");
    let mae = mean_absolute_error(&y_true_reg, &y_pred_reg).expect("MAE computation failed");
    let r2 = r2_score(&y_true_reg, &y_pred_reg).expect("R2 computation failed");

    // Basic sanity checks
    assert!(mse >= 0.0);
    assert!(mae >= 0.0);
    assert!(r2.is_finite());

    println!("Metrics computation test completed successfully");
}

#[test]
fn test_cross_validation_utilities() {
    use sklears_utils::random::{k_fold_indices, stratified_split_indices};

    let n_samples = 50;
    let n_splits = 5;

    // Test K-fold cross validation
    let folds = k_fold_indices(n_samples, n_splits, true, Some(42))
        .expect("K-fold indices generation failed");

    assert_eq!(folds.len(), n_splits);

    // Verify all samples are used exactly once
    let mut all_test_indices = Vec::new();
    for (train_indices, test_indices) in &folds {
        all_test_indices.extend(test_indices.iter().copied());
        
        // Check that train and test don't overlap
        let train_set: std::collections::HashSet<usize> = train_indices.iter().copied().collect();
        let test_set: std::collections::HashSet<usize> = test_indices.iter().copied().collect();
        assert!(train_set.intersection(&test_set).count() == 0);
    }

    all_test_indices.sort_unstable();
    let expected: Vec<usize> = (0..n_samples).collect();
    assert_eq!(all_test_indices, expected);

    // Test stratified split
    let labels: Vec<i32> = (0..n_samples).map(|i| (i % 3) as i32).collect();
    let (train_indices, test_indices) = stratified_split_indices(&labels, 0.3, Some(42))
        .expect("Stratified split failed");

    assert_eq!(train_indices.len() + test_indices.len(), n_samples);

    // Check that all indices are valid and unique
    let all_indices: std::collections::HashSet<usize> = train_indices.iter()
        .chain(test_indices.iter())
        .copied()
        .collect();
    assert_eq!(all_indices.len(), n_samples);

    println!("Cross-validation utilities test completed successfully");
}

#[test]
fn test_data_validation_utilities() {
    use sklears_utils::validation::{check_array_2d, check_consistent_length, check_finite};

    // Test array validation
    let valid_x = Array2::<f64>::ones((10, 5));
    let valid_y = Array1::<i32>::zeros(10);

    assert!(check_array_2d(&valid_x).is_ok());
    assert!(check_consistent_length(&valid_x, &valid_y).is_ok());
    assert!(check_finite(&valid_x).is_ok());

    // Test inconsistent length detection
    let wrong_y = Array1::<i32>::zeros(15);
    assert!(check_consistent_length(&valid_x, &wrong_y).is_err());

    // Test finite value checking
    let mut invalid_x = Array2::<f64>::ones((5, 3));
    invalid_x[[2, 1]] = f64::NAN;
    assert!(check_finite(&invalid_x).is_err());

    println!("Data validation utilities test completed successfully");
}

#[test]
fn test_multiclass_utilities() {
    use sklears_utils::multiclass::type_of_target;
    use sklears_utils::array_utils::{unique_labels, label_counts};

    // Test binary classification target
    let binary_target = Array1::from_vec(vec![0, 1, 0, 1, 1, 0]);
    let target_type = type_of_target(&binary_target).expect("Target type detection failed");
    assert_eq!(target_type, "binary");

    // Test multiclass target
    let multiclass_target = Array1::from_vec(vec![0, 1, 2, 0, 1, 2, 1]);
    let target_type = type_of_target(&multiclass_target).expect("Target type detection failed");
    assert_eq!(target_type, "multiclass");

    // Test unique labels extraction
    let labels = Array1::from_vec(vec![2, 1, 3, 1, 2, 3, 1]);
    let unique = unique_labels(&labels);
    let mut unique_sorted = unique.clone();
    unique_sorted.sort_unstable();
    assert_eq!(unique_sorted, vec![1, 2, 3]);

    // Test label counting
    let counts = label_counts(&labels);
    assert_eq!(counts.get(&1), Some(&3));
    assert_eq!(counts.get(&2), Some(&2));
    assert_eq!(counts.get(&3), Some(&2));

    println!("Multiclass utilities test completed successfully");
}

#[test]
fn test_random_utilities() {
    use sklears_utils::random::{bootstrap_indices, shuffle_indices};

    let n_samples = 30;

    // Test bootstrap sampling
    let bootstrap_sample = bootstrap_indices(n_samples, Some(42));
    assert_eq!(bootstrap_sample.len(), n_samples);

    // All indices should be valid
    for &idx in &bootstrap_sample {
        assert!(idx < n_samples);
    }

    // Test shuffling
    let shuffled = shuffle_indices(n_samples, Some(42));
    assert_eq!(shuffled.len(), n_samples);

    // Should contain all indices exactly once
    let mut sorted_shuffled = shuffled.clone();
    sorted_shuffled.sort_unstable();
    let expected: Vec<usize> = (0..n_samples).collect();
    assert_eq!(sorted_shuffled, expected);

    // Should be different from original order (with high probability)
    let original: Vec<usize> = (0..n_samples).collect();
    assert_ne!(shuffled, original);

    println!("Random utilities test completed successfully");
}

#[test]
fn test_comprehensive_pipeline() {
    // Generate comprehensive test data
    let (x, y) = make_classification(100, 6, 3, None, None, 0.05, 1.5, Some(42))
        .expect("Failed to generate test data");

    // Data splitting
    let (train_indices, test_indices) = train_test_split_indices(100, 0.2, true, Some(42))
        .expect("Failed to split data");

    let x_train = x.select(ndarray::Axis(0), &train_indices);
    let x_test = x.select(ndarray::Axis(0), &test_indices);
    let y_train = y.select(ndarray::Axis(0), &train_indices);
    let y_test = y.select(ndarray::Axis(0), &test_indices);

    // Preprocessing
    use sklears_preprocessing::scaling::StandardScaler;
    let scaler = StandardScaler::new();
    let fitted_scaler = scaler.fit(&x_train, &()).expect("Scaler fit failed");
    let x_train_scaled = fitted_scaler.transform(&x_train).expect("Scaler transform failed");
    let x_test_scaled = fitted_scaler.transform(&x_test).expect("Scaler transform failed");

    // Validation checks
    use sklears_utils::validation::{check_array_2d, check_consistent_length, check_finite};
    assert!(check_array_2d(&x_train_scaled).is_ok());
    assert!(check_array_2d(&x_test_scaled).is_ok());
    assert!(check_consistent_length(&x_train_scaled, &y_train).is_ok());
    assert!(check_consistent_length(&x_test_scaled, &y_test).is_ok());
    assert!(check_finite(&x_train_scaled).is_ok());
    assert!(check_finite(&x_test_scaled).is_ok());

    // Target analysis
    use sklears_utils::multiclass::type_of_target;
    use sklears_utils::array_utils::{unique_labels, label_counts};

    let target_type = type_of_target(&y_train).expect("Target type detection failed");
    assert_eq!(target_type, "multiclass");

    let unique_classes = unique_labels(&y_train);
    assert_eq!(unique_classes.len(), 3);

    let class_counts = label_counts(&y_train);
    assert_eq!(class_counts.len(), 3);

    // Verify class distribution
    let total_train_samples: usize = class_counts.values().sum();
    assert_eq!(total_train_samples, y_train.len());

    println!("Comprehensive pipeline test completed successfully");
    println!("Train samples: {}, Test samples: {}", x_train.nrows(), x_test.nrows());
    println!("Features: {}, Classes: {}", x_train.ncols(), unique_classes.len());
}